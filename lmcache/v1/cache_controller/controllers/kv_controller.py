# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict

# First Party
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
)
from lmcache.v1.token_database import ChunkedTokenDatabase


class KVController:
    def __init__(self) -> None:
        # Mapping from `(instance_id, worker_id)` -> [location -> set[chunk_hash]]
        self.kv_pool: dict[tuple[str, int], dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()

    def post_init(self, reg_controller, cluster_executor):
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor

    async def admit(self, msg: KVAdmitMsg) -> None:
        """
        Admit a new kv chunk.
        """
        report_id = (msg.instance_id, msg.worker_id)
        self.kv_pool[report_id][msg.location].add(msg.key)

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        report_id = (msg.instance_id, msg.worker_id)
        location = msg.location
        key = msg.key

        if (
            report_id not in self.kv_pool
            or location not in self.kv_pool[report_id]
            or key not in self.kv_pool[report_id][location]
        ):
            return

        self.kv_pool[report_id][location].remove(key)
        if not self.kv_pool[report_id][location]:
            del self.kv_pool[report_id][location]
        if not self.kv_pool[report_id]:
            del self.kv_pool[report_id]

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        return await self.cluster_executor.execute("check_finish", msg)

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        """
        Deregister all kv chunks of an instance-worker.
        """
        report_id = (instance_id, worker_id)
        if report_id in self.kv_pool:
            del self.kv_pool[report_id]

    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        chunk_infos = list(self.token_database.process_tokens(tokens, make_key=False))
        num_hit_tokens = 0
        matched_instance_id = ""
        for (instance_id, worker_id), location_kvs in self.kv_pool.items():
            tmp_hit_tokens = 0
            for start, end, key in chunk_infos:
                contains = False
                for location, kvs in location_kvs.items():
                    if key in kvs:
                        contains = True
                        break
                if not contains:
                    break
                tmp_hit_tokens = end
            if tmp_hit_tokens > num_hit_tokens:
                num_hit_tokens = tmp_hit_tokens
                matched_instance_id = instance_id
        if num_hit_tokens > 0:
            # TODO(Jiayi): The current implementation does not consider
            # the location of the kv chunks. It simply returns the
            # `instance_id` with longest prefix.
            layout_info[matched_instance_id] = ("", num_hit_tokens)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        """
        Perform batched P2P lookup for multiple keys.

        :param BatchedP2PLookupMsg msg: The batched P2P lookup message containing keys.

        :return: A BatchedP2PLookupRetMsg containing the lookup results.
        """

        query_instance_id = msg.instance_id
        num_hit_chunks = 0
        matched_instance_id = ""
        matched_location = ""
        peer_init_url = ""
        # TODO(Jiayi): The KV Cache could be from different
        # instances. We need to handle this case as well.
        for (instance_id, worker_id), location_kvs in self.kv_pool.items():
            if instance_id == query_instance_id:
                continue
            for location, kvs in location_kvs.items():
                tmp_hit_chunks = 0
                for key in msg.hashes:
                    if key not in kvs:
                        break
                    tmp_hit_chunks += 1
                if tmp_hit_chunks > num_hit_chunks:
                    num_hit_chunks = tmp_hit_chunks
                    matched_instance_id = instance_id
                    matched_location = location
                    peer_init_url = self.reg_controller.get_distributed_url(
                        instance_id, worker_id
                    )
                    assert peer_init_url is not None
        return BatchedP2PLookupRetMsg(
            layout_info=[
                (matched_instance_id, matched_location, num_hit_chunks, peer_init_url),
            ]
        )
