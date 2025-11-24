# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Callable, Dict

# First Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.nixl_channel import NixlChannel
from lmcache.v1.transfer_channel.socket_channel import SocketChannel


def _create_nixl_channel(
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """Factory function for creating NixlChannel"""
    assert "backends" in kwargs, (
        "`backends` must be provided to create nixl transfer channel."
    )
    return NixlChannel(
        async_mode=async_mode,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )


def _create_socket_channel(
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """Factory function for creating SocketChannel"""
    return SocketChannel(
        async_mode=async_mode,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )


CHANNEL_REGISTRY: Dict[str, Callable[..., BaseTransferChannel]] = {
    "nixl": _create_nixl_channel,
    "socket": _create_socket_channel,
}


def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """
    Create a transfer channel based on the specified channel type.
    Supports multiple channel types: "nixl", "socket".

    :param channel_type: Type of the transfer channel (e.g., "nixl", "socket").
    :param async_mode: Whether to operate in asynchronous mode.
    :param role: Role of the channel (e.g., "both", "sender" or "receiver").
    :param buffer_ptr: Pointer to the pre-allocated buffer.
    :param buffer_size: Size of the pre-allocated buffer in bytes.
    :param align_bytes: Alignment requirement in bytes.
    :param tp_rank: Tensor parallel rank of the current process.
    :param peer_init_url: Initialization URL for the peer.
    :kwargs: Additional keyword arguments specific to the channel type.

    :return: An instance of the specified transfer channel.
    """

    if channel_type not in CHANNEL_REGISTRY:
        raise ValueError(
            f"Unsupported channel type: {channel_type}. "
            f"Available types: {list(CHANNEL_REGISTRY.keys())}"
        )

    factory = CHANNEL_REGISTRY[channel_type]
    return factory(
        async_mode=async_mode,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )
