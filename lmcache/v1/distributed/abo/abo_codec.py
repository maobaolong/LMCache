# SPDX-License-Identifier: Apache-2.0

"""
ABO codec basics.

Aligned with tx-lmcache's abo_basics.py pattern:
- ABOConfig: ABO configuration (user-facing + codec)
- ABOCodecFactory: creates abokvpress.HuffmanCodec directly
- estimate_compressed_bytes: standalone utility function
- resolve_abo_dtype: convert torch.dtype to ABO dtype string

The codec returned by ABOCodecFactory.create_codec() is a raw
abokvpress.HuffmanCodec instance. Callers invoke codec.compress()
and codec.decompress() directly with numpy buffers:
    result = codec.compress(dst_np, dst_size, src_np, src_size, dtype="bf16")
    result = codec.decompress(dst_np, dst_size, src_np, src_size)
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional
import argparse

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import get_size_bytes

logger = init_logger(__name__)

# PyTorch dtype -> ABO dtype string
PYTORCH_DTYPE_TO_ABO: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.uint8: "fp8e4m3",
    torch.float8_e4m3fn: "fp8e4m3",
}

# Default compression ratio per ABO dtype
_DEFAULT_RATIO: dict[str, int] = {
    "bf16": 22,
    "fp8e4m3": 28,
}

# Alignment granularity (bytes)
_ALIGN_BYTES = 4096

@dataclass
class ABOConfig:
    """ABO KV compression configuration.

    When enabled, L1 stores compressed KV data with StagingPool providing
    GPU<->CPU transfer intermediary, reducing L1 memory usage while preserving
    D2H/H2D and compress/decompress overlap.
    """

    enable: bool = field(default=False)
    """ Whether to enable ABO KV compression. """

    staging_size_gb: float = field(default=16.0)
    """ Pinned memory size (GB) used by StagingPool. """

    ratio: Optional[int] = field(default=None)
    """ Compression ratio (20-32). None means auto-detect by dtype. """

    codec: str = field(default="huffman")
    """ Codec method, default is huffman. """

    num_threads: int = field(default=32)
    """ Number of threads for compress/decompress. """

    # Internal field: ABO dtype string (set by caller)
    _dtype: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.enable:
            if self.staging_size_gb <= 0:
                raise ValueError(
                    f"abo_staging_size_gb must be positive, got: {self.staging_size_gb}"
                )
            if self.ratio is not None and not (20 <= self.ratio <= 32):
                raise ValueError(f"abo_ratio must be in range 20-32, got: {self.ratio}")
            if self.num_threads <= 0:
                raise ValueError(
                    "abo_num_threads must be a positive integer, "
                    f"got: {self.num_threads}"
                )
            try:
                import abokvpress  # type: ignore # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "Enabling ABO compression (enable_abo=True) requires the "
                    "abokvpress library. Please run: pip install abokvpress"
                ) from e

def resolve_abo_dtype(dtype: torch.dtype) -> str:
    """Convert torch.dtype to ABO dtype string.

    Args:
        dtype: PyTorch dtype.

    Returns:
        ABO dtype string (e.g. "bf16", "fp8e4m3").

    Raises:
        ValueError: If dtype is not supported by ABO.
    """
    abo_dtype = PYTORCH_DTYPE_TO_ABO.get(dtype)
    if abo_dtype is None:
        raise ValueError(
            f"Unsupported dtype for ABO: {dtype}. "
            f"Supported: {list(PYTORCH_DTYPE_TO_ABO.keys())}"
        )
    return abo_dtype


class ABOCodecFactory:
    """Factory for creating ABO codec instances.

    Returns raw abokvpress.HuffmanCodec — no Python wrapper layer.
    dtype and ratio are NOT passed to the constructor; they are
    specified per compress() call.
    """

    @staticmethod
    def create_codec(method: str, config: ABOConfig):
        """Create a codec instance based on the specified method.

        Args:
            method: Codec method name (e.g. "huffman").
            config: ABOConfig with num_threads, etc.

        Returns:
            abokvpress.HuffmanCodec instance.

        Raises:
            ValueError: If method is unsupported.
        """
        method = method.lower()

        if method == "huffman":
            try:
                from abokvpress import HuffmanCodec
            except ImportError as e:
                raise ImportError(
                    "abokvpress is not installed. Please install it with: pip install abokvpress"
                ) from e

            codec = HuffmanCodec(use_avx512=True)
            codec.set_num_threads(config.num_threads)

            logger.info(
                "ABO HuffmanCodec created: num_threads=%d",
                config.num_threads,
            )
            return codec
        else:
            raise ValueError(
                f"Unsupported ABO codec method: {method}. Supported: huffman"
            )


def estimate_compressed_bytes(
    original_shapes: list[torch.Size],
    original_dtypes: list[torch.dtype],
    ratio: Optional[int] = None,
) -> int:
    """Estimate compressed size in bytes.

    Formula: original_bytes * ratio / 32, aligned up to _ALIGN_BYTES.

    Args:
        shapes: List of tensor shapes.
        dtypes: List of tensor dtypes.
        ratio: Compression ratio (20-32). None = auto-select based on dtype.

    Returns:
        Estimated compressed bytes (aligned).
    """
    if ratio is None:
        abo_dtype = PYTORCH_DTYPE_TO_ABO.get(original_dtypes[-1])
        ratio = _DEFAULT_RATIO[abo_dtype]

    original_chunk_bytes = get_size_bytes(original_shapes, original_dtypes)
    compressed_bytes = int(original_chunk_bytes * ratio / 32)
    aligned_bytes = (compressed_bytes + _ALIGN_BYTES - 1) // _ALIGN_BYTES * _ALIGN_BYTES

    return aligned_bytes


def add_abo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ABO compression arguments to an existing parser."""
    abo_group = parser.add_argument_group(
        "ABO Compression", "KV cache ABO compression configuration"
    )
    abo_group.add_argument(
        "--enable-abo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable ABO KV compression. Default is False.",
    )
    abo_group.add_argument(
        "--abo-staging-size-gb",
        type=float,
        default=16.0,
        help="StagingPool pinned memory size in GB. Default is 16.0.",
    )
    abo_group.add_argument(
        "--abo-ratio",
        type=int,
        default=None,
        help="Compression ratio (20-32). Default is auto-detect by dtype.",
    )
    abo_group.add_argument(
        "--abo-codec",
        type=str,
        default="huffman",
        help='Codec method for ABO compression. Default is "huffman".',
    )
    abo_group.add_argument(
        "--abo-num-threads",
        type=int,
        default=32,
        help="Number of threads for ABO compress/decompress. Default is 32.",
    )
    return parser


def parse_args_to_abo_config(args: argparse.Namespace) -> ABOConfig:
    """Build ABOConfig from parsed command-line arguments."""
    return ABOConfig(
        enable=args.enable_abo,
        staging_size_gb=args.abo_staging_size_gb,
        ratio=args.abo_ratio,
        codec=args.abo_codec,
        num_threads=args.abo_num_threads,
    )
