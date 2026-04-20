# SPDX-License-Identifier: Apache-2.0

"""
ABO decompress manager (CUDA stream + launch_host_func mode).

Manages async decompression in the RETRIEVE path:
1. Batch-submit decompress tasks via cupy launch_host_func on _decompress_stream
2. Each decompress callback runs sync decompress (codec.decompress with numpy buffers)
3. After decompress, record event for GPU-side sync (current_stream.wait_event)
4. After H2D, release staging buffer via launch_host_func callback

Key constraint:
  cudaLaunchHostFunc callbacks need GIL. Never call stream.synchronize()
  or event.synchronize() from the same thread that holds GIL while
  host callbacks are pending on that stream.

Timing design:
  _decompress_stream:
    launch_host_func(decomp0) → decomp_event0 → launch_host_func(decomp1) → ...
  high_priority stream (H2D):
    wait(decomp_event0) → H2D chunk0 → launch_host_func(release_staging0) → ...
    wait(decomp_event1) → H2D chunk1 → launch_host_func(release_staging1) → ...
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj

if TYPE_CHECKING:
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class ABODecompressManager:
    """Manages async decompression in the RETRIEVE path (stream mode).

    Uses a dedicated CUDA stream + cupy launch_host_func to schedule
    sync decompress as host callbacks. The callback executes in the
    CUDA driver's callback thread (which acquires GIL automatically).

    H2D loop uses .tensor property's wait_event for GPU-side sync.
    After H2D, staging release is also done via launch_host_func.
    """

    def __init__(
        self,
        codec,
        device: Optional[torch.device] = None,
    ):
        """Initialize decompress manager.

        Args:
            codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            device: CUDA device (None = current device).
        """
        self._codec = codec

        # Cached num_readers from prefetch path, used when retrieve
        # path needs to acquire staging (has_staging=False fallback)
        self._num_readers: int = 1

        # Lazily created CUDA stream + cupy wrapper
        self._decompress_stream: Optional[torch.cuda.Stream] = None
        self._cupy_decompress_stream: Optional[cupy.cuda.ExternalStream] = None
        self._device = device

    def _get_decompress_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or create the dedicated decompress CUDA stream + cupy wrapper."""
        if self._decompress_stream is None:
            device = self._device or torch.cuda.current_device()
            self._decompress_stream = torch.cuda.Stream(device=device)
            self._cupy_decompress_stream = cupy.cuda.ExternalStream(
                self._decompress_stream.cuda_stream
            )
        return self._decompress_stream, self._cupy_decompress_stream

    def prepare_decompress_batch(
        self,
        memory_objs: list["MemoryObj"],
        is_retrieve: bool = False,
        num_readers: int = 1,
    ) -> None:
        """Batch-submit async decompress tasks for a list of MemoryObj.

        For CompressedMemoryObj without staging buffer (data in raw_data),
        acquire staging buffer, submit sync decompress via launch_host_func
        on _decompress_stream, record decompress_event.

        For non-compressed MemoryObj or those already having staging, do nothing.

        Args:
            memory_objs: MemoryObj list from read_prefetched_results.
            is_retrieve: If True, this is the retrieve (final) path;
                acquire failure marks decompress_failed on the obj.
                If False, this is the prefetch (early) path;
                acquire failure silently skips the obj.
            num_readers: Number of ref counts to add when acquiring
                or bumping staging ref. For prefetch path this should
                be tp_size (1 + extra_count); for retrieve path this
                should be 0 (prefetch already holds the refs).
                Note: when retrieve path hits has_staging=False (prefetch
                didn't acquire staging in time), try_acquire_staging uses
                the cached _num_readers from the last prefetch call.
        """
        decompress_stream, cupy_decompress_stream = self._get_decompress_stream()

        # Prefetch path: cache num_readers for later retrieve fallback
        if not is_retrieve:
            self._num_readers = num_readers

        for obj in memory_objs:
            if not isinstance(obj, CompressedMemoryObj):
                continue

            # Already has staging (data still in staging, no decompress needed)
            # add 0 for retrieve path, extra_count+1 for prefetch path
            if obj.has_staging:
                obj.staging_ref_acquire(num_readers)
                continue

            # Acquire staging buffer via obj's class-level pool
            if not obj.try_acquire_staging(num_readers=self._num_readers):
                if is_retrieve:
                    logger.warning(
                        "StagingPool exhausted during retrieve, "
                        "fallback and mark decompress failed for obj id=%s",
                        id(obj),
                    )
                    # Mark decompress as failed so .tensor returns None,
                    # triggering error in H2D -> retrieve failure path
                    obj.mark_decompress_failed()
                    continue
                else:
                    logger.warning(
                        "StagingPool has no free buffer, "
                        "delayed decompress to retrieve stage"
                    )
                continue

            # Capture references for the closure
            _obj = obj
            _codec = self._codec
            _staging = obj._staging_tensor

            def _decompress_callback(_arg, _o=_obj, _c=_codec, _s=_staging):
                """Host callback: sync decompress raw_data → staging.

                Runs in CUDA driver's callback thread (GIL acquired by cupy).
                Args:
                    _arg: User data from cupy launch_host_func (unused).
                """
                try:
                    # codec.decompress(dst_np, dst_size, src_np, src_size)
                    # Pass full raw_data; codec self-describes block boundaries
                    dst_np = _s.numpy()
                    src_np = _o.raw_data.numpy()
                    result = _c.decompress(dst_np, dst_np.nbytes, src_np, src_np.nbytes)
                    if not result.success:
                        logger.warning("ABO decompress failed: %s", result.error_msg)
                        _o.mark_decompress_failed()
                    else:
                        _o.mark_decompress_done()
                except Exception as e:
                    logger.warning("Decompress callback failed: %s", e)
                    _o.mark_decompress_failed()

            # Switch to decompress_stream's device context to ensure
            # launch_host_func and event.record run in the correct CUDA context
            # (MLA: any TP worker's thread may call this, but stream is fixed)
            with torch.cuda.device(decompress_stream.device):
                # Register host callback on decompress_stream
                cupy_decompress_stream.launch_host_func(_decompress_callback, None)

                # Record decompress done event on decompress_stream
                decompress_event = torch.cuda.Event()
                decompress_event.record(decompress_stream)
            obj.setup_decompress(decompress_event)

    def close(self) -> None:
        """Clean up resources.

        NOTE: Do NOT call decompress_stream.synchronize() here — it would
        deadlock if any host callbacks are pending.
        """
        pass
