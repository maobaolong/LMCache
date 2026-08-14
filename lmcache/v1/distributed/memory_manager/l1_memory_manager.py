# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-DRAM L1 memory manager."""

# Standard
from collections import defaultdict
from multiprocessing import shared_memory

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def _unlink_stale_shm(shm_name: str) -> None:
    """Remove a stale LMCache shm segment if it exists."""
    normalized = shm_name.lstrip("/")
    if "/" in normalized or "\\" in normalized:
        logger.warning("Refusing to unlink invalid shm name %s", shm_name)
        return
    if not normalized.startswith("lmcache_l1_pool_"):
        return
    try:
        shm = shared_memory.SharedMemory(name=normalized, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Failed to remove stale shm segment %s", normalized, exc_info=True
        )


def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        shm_name = config.shm_name
        if shm_name:
            # Keep the lmcache_l1_pool_ prefix in normalized SHM names so
            # stale-segment cleanup can recognize and unlink user-provided names.
            bare = shm_name.lstrip("/")
            if not bare.startswith("lmcache_l1_pool_"):
                shm_name = f"lmcache_l1_pool_{bare}"
            _unlink_stale_shm(shm_name)
            return MixedMemoryAllocator(
                config.size_in_bytes,
                align_bytes=config.align_bytes,
                shm_name=shm_name,
            )
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
        )


def _free_by_parent_allocator(
    local_allocator: MemoryAllocatorInterface,
    mem_objs: list[MemoryObj],
) -> None:
    """Free objects through the allocator that owns each object.

    ``L1MemoryManager`` primarily manages pinned-DRAM allocations from its
    local CPU allocator. Some callers, however, may temporarily store
    externally-owned ``MemoryObj`` instances in L1 (for example, a future
    device-backed read path that swaps in a DMA buffer after reserving a CPU
    placeholder). Those objects must be returned to the allocator that created
    them rather than to the local DRAM allocator.

    Args:
        local_allocator: The allocator owned by this L1 memory manager.
        mem_objs: Objects to free.
    """
    local_objs: list[MemoryObj] = []
    foreign_batches: dict[MemoryAllocatorInterface, list[MemoryObj]] = defaultdict(list)

    for memory_obj in mem_objs:
        parent = memory_obj.parent()
        if parent is None or parent is local_allocator:
            local_objs.append(memory_obj)
            continue
        foreign_batches[parent].append(memory_obj)

    if local_objs:
        local_allocator.batched_free(local_objs)

    for allocator, objs in foreign_batches.items():
        allocator.batched_free(objs)


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free. Objects
                allocated by a different parent allocator are dispatched back to
                that allocator instead of being returned to the local DRAM pool.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        _free_by_parent_allocator(self._allocator, mem_objs)
        return L1Error.SUCCESS

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``.

        Args:
            memory_obj: An object allocated by this manager.

        Returns:
            ``L1BackendType.DRAM`` — the CPU tier is pinned DRAM only.
        """
        return L1BackendType.DRAM

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """

        if hasattr(self._allocator, "get_memory_usage"):
            return self._allocator.get_memory_usage()

        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            if isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            raise NotImplementedError(
                "get_memory_usage is not implemented for this allocator type."
            )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            L1MemoryDesc: Pointer, size, and alignment of the L1 buffer.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
