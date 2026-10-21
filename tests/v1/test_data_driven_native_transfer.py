# SPDX-License-Identifier: Apache-2.0
"""Data-driven object pointer, layout, and completion-event plans."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform import current_device_spec
import lmcache.lmcache_native as lmcache_native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]


def test_device_pointer_layer_kv_layout_round_trip() -> None:
    """A plan can transfer more than four padded layer-major objects."""
    cuda_ops = pytest.importorskip(
        "lmcache.cuda_ops", reason="Requires the CUDA extension"
    )
    nl, nb, bs, nh, hs = 3, 64, 4, 2, 8
    chunk = 8
    num_objects = 5
    blocks_per_object = chunk // bs
    total_blocks = num_objects * blocks_per_object
    device = torch.device(torch_device_type, torch_dev.current_device())

    torch.manual_seed(0)
    paged = [
        torch.randn(2, nb, bs, nh, hs, dtype=torch.bfloat16, device=device)
        for _ in range(nl)
    ]
    paged_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in paged], dtype=torch.int64, device=device
    )
    block_ids = torch.arange(total_blocks, dtype=torch.int64, device=device)

    shape_desc = lmcache_native.PageBufferShapeDesc()
    shape_desc.kv_size = 2
    shape_desc.nl = nl
    shape_desc.nb = nb
    shape_desc.bs = bs
    shape_desc.nh = nh
    shape_desc.hs = hs
    shape_desc.element_size = 2
    shape_desc.block_stride_elems = 0

    tight_layer_elems = 2 * chunk * nh * hs
    layer_stride_elems = tight_layer_elems + 16
    layer_object_elems = (nl - 1) * layer_stride_elems + tight_layer_elems

    default_objects = [
        torch.zeros(2, nl, chunk, nh, hs, dtype=torch.bfloat16, device=device)
        for _ in range(num_objects)
    ]
    layer_objects = [
        torch.zeros(layer_object_elems, dtype=torch.bfloat16, device=device)
        for _ in range(num_objects)
    ]
    default_ptrs = torch.tensor(
        [obj.data_ptr() for obj in default_objects],
        dtype=torch.int64,
        device=device,
    )
    layer_ptrs = torch.tensor(
        [obj.data_ptr() for obj in layer_objects],
        dtype=torch.int64,
        device=device,
    )

    def spec(
        paged_pointer_array: torch.Tensor,
        object_pointer_array: torch.Tensor,
        layout: lmcache_native.MemoryObjectLayout,
        layer_stride: int = 0,
    ) -> lmcache_native.KernelGroupSpec:
        return lmcache_native.KernelGroupSpec(
            paged_pointer_array.data_ptr(),
            [],
            shape_desc,
            chunk,
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            block_ids.data_ptr(),
            block_ids.numel(),
            lmcache_native.MemoryObjectPointerSource.DEVICE_ARRAY,
            object_pointer_array.data_ptr(),
            object_pointer_array.numel(),
            layout,
            layer_stride,
        )

    launch = cuda_ops.LaunchVar(0, 0, total_blocks, num_objects, 0)
    d2h = lmcache_native.TransferDirection.D2H
    cuda_ops.execute_object_group_transfer(
        d2h,
        device,
        1 << 20,
        [
            spec(
                paged_ptrs,
                default_ptrs,
                lmcache_native.MemoryObjectLayout.KV_LAYER_TOKEN,
            )
        ],
        [cuda_ops.BatchStep([], [launch])],
    )

    completion: Any = current_device_spec.create_stream_event(device)
    current_device_spec.record_stream_event(
        completion, current_device_spec.current_stream(device)
    )
    cuda_ops.execute_object_group_transfer(
        d2h,
        device,
        1 << 20,
        [
            spec(
                paged_ptrs,
                layer_ptrs,
                lmcache_native.MemoryObjectLayout.LAYER_KV_TOKEN,
                layer_stride_elems,
            )
        ],
        [cuda_ops.BatchStep([], [launch], completion.cuda_event)],
    )
    completion.synchronize()

    for default, layer_major in zip(default_objects, layer_objects, strict=True):
        for layer_idx in range(nl):
            start = layer_idx * layer_stride_elems
            got = layer_major[start : start + tight_layer_elems].view(2, chunk, nh, hs)
            assert torch.equal(got, default[:, layer_idx])
            if layer_idx + 1 < nl:
                padding = layer_major[
                    start + tight_layer_elems : (layer_idx + 1) * layer_stride_elems
                ]
                assert torch.count_nonzero(padding).item() == 0

    restored = [torch.zeros_like(tensor) for tensor in paged]
    restored_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in restored], dtype=torch.int64, device=device
    )
    cuda_ops.execute_object_group_transfer(
        lmcache_native.TransferDirection.H2D,
        device,
        1 << 20,
        [
            spec(
                restored_ptrs,
                layer_ptrs,
                lmcache_native.MemoryObjectLayout.LAYER_KV_TOKEN,
                layer_stride_elems,
            )
        ],
        [cuda_ops.BatchStep([], [launch])],
    )
    torch_dev.synchronize(device)
    for got, expected in zip(restored, paged, strict=True):
        assert torch.equal(got[:, block_ids], expected[:, block_ids])
