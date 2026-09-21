# SPDX-License-Identifier: Apache-2.0
"""External SGLang connector used to validate sgl-project/sglang#40534.

This adapter currently supports full-attention CUDA models and kv_both only.
The transfer implementation comes from LMCache#4828 and SGLang#38652.
"""

# Third Party
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_kv_connector import BaseKVConnector
from sglang.srt.mem_cache.kv_transfer_config import KVTransferConfig
from sglang.srt.mem_cache.registry import TreeCacheBuildContext
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# First Party
from lmcache.integration.sglang.lmcache_unified_radix_cache import (
    LMCacheUnifiedRadixCache,
)


class LMCacheUnifiedConnector(LMCacheUnifiedRadixCache, BaseKVConnector):
    """Load the LMCache MP implementation through SGLang's public factory."""

    @classmethod
    def validate_config(
        cls, context: TreeCacheBuildContext, config: KVTransferConfig
    ) -> None:
        """Reject configurations not covered by this validation adapter."""
        args = context.server_args
        if config.kv_role != "kv_both":
            raise ValueError("This LMCache validation adapter supports kv_both only")
        if context.is_hybrid_swa or context.is_hybrid_ssm or context.is_dsa:
            raise ValueError("This LMCache validation adapter supports full attention")
        if SpeculativeAlgorithm.from_string(
            args.speculative_algorithm
        ).is_speculative():
            raise ValueError("LMCache adapter does not support speculative decoding")
        if args.enable_dp_attention or args.dcp_size > 1 or args.pp_size > 1:
            raise ValueError("LMCache validation adapter supports TP without DP/DCP/PP")
        allowed = {"config_file", "lmcache.mp.host", "lmcache.mp.port"}
        unknown = set(config.kv_connector_extra_config) - allowed
        if unknown:
            raise ValueError(f"Unknown LMCache connector options: {sorted(unknown)}")

    def __init__(
        self, context: TreeCacheBuildContext, config: KVTransferConfig
    ) -> None:
        context.params.tree_components = (ComponentType.FULL,)
        extra = config.kv_connector_extra_config
        overrides = {}
        if "lmcache.mp.host" in extra:
            overrides["mp_host"] = extra["lmcache.mp.host"]
        if "lmcache.mp.port" in extra:
            overrides["mp_port"] = int(extra["lmcache.mp.port"])
        super().__init__(
            context.params,
            model_config=context.model_config,
            tp_size=context.tp_size,
            tp_rank=context.tp_rank,
            lmcache_config_file=extra.get("config_file"),
            config_overrides=overrides,
            forward_stream=context.tp_worker.model_runner.forward_stream,
        )

    def prefetch_request(self, req: Req) -> None:
        """Look up the cacheable prompt prefix before scheduler admission."""
        if req.positional_embed_overrides is not None:
            return
        tokens = req.full_untruncated_fill_ids
        match_end = max(len(tokens) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            match_end = min(match_end, req.logprob_start_len)
        matched_len = len(req.prefix_indices) + req.host_hit_length
        req.storage_prefetch_last_match_len = matched_len
        self.prefetch_from_storage(
            req.cache_request_handle,
            req.last_host_node,
            tokens[matched_len:match_end],
            matched_prefix_tokens=tokens[:matched_len],
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
        )
