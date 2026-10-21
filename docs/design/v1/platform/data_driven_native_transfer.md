# Data-driven native transfer plans

## Purpose

The CUDA object-group executor accepts all transfer-specific choices as plan
data. A Python server-module plug-in can therefore use a different LMCache
object layout, a device-resident object-pointer table, and per-step completion
events without compiling against LMCache's C++ implementation.

The boundary deliberately remains the existing pybind API. LMCache does not
install its internal CUDA headers or expose C++ standard-library objects through
a cross-extension ABI.

## Kernel-group fields

`KernelGroupSpec` retains the existing host-vector pointer path and
`[KV, layer, token, hidden]` object layout as defaults. The additional fields
are explicit so a plug-in does not depend on a sentinel pointer:

| Field | Meaning |
| --- | --- |
| `object_pointer_source` | `HOST_VECTOR` passes at most four pointers by value; `DEVICE_ARRAY` reads pointers from a device `int64` array. |
| `lmcache_objects_device_base` | Address of the device pointer array. The caller owns it and keeps it alive until execution returns. |
| `lmcache_objects_device_capacity` | Number of pointer entries available at the device address. |
| `object_layout` | `KV_LAYER_TOKEN` (`[KV, L, T, D]`) or `LAYER_KV_TOKEN` (`[L, KV, T, D]`). |
| `object_layer_stride_elems` | Source-dtype elements between layers for `LAYER_KV_TOKEN`; zero means tightly packed. |

The executor validates the pointer capacity, block-ID capacity, layout, and
stride before launching a kernel. Layout selection happens on the host and
chooses a specialized kernel instantiation; it does not add a per-element
device branch.

The caller creates a device pointer array with its normal tensor library and
retains that tensor with the plan:

```python
object_ptrs = torch.tensor(
    [buffer.data_ptr() for buffer in buffers],
    dtype=torch.int64,
    device=device,
)
spec = device_ops.KernelGroupSpec(
    paged_ptrs.data_ptr(),
    [],
    shape_desc,
    chunk_size,
    engine_kv_format,
    block_ids.data_ptr(),
    block_ids.numel(),
    lmcache_native.MemoryObjectPointerSource.DEVICE_ARRAY,
    object_ptrs.data_ptr(),
    object_ptrs.numel(),
    lmcache_native.MemoryObjectLayout.LAYER_KV_TOKEN,
    layer_stride_elems,
)
```

## Step completion

`BatchStep(..., completion_event=event_ptr)` records the supplied CUDA/HIP
event after that step's staging copies and kernel launches have been enqueued.
Zero preserves the old behavior. A plug-in can flatten several logical layer
batches into one executor call, put the event only on the final step of each
batch, and publish those events after the call returns. Stream ordering still
lets the consumer begin work on an early batch while later batches transfer.

The event is borrowed. Its owner must keep it alive until the stream has passed
the record operation.

## Compatibility

Existing call sites omit every new argument and keep the original behavior:

- at most four object pointers are passed by value;
- objects use `[KV, layer, token, hidden]` layout;
- no completion event is recorded.

This interface is source-level Python compatibility, not a stable C++ ABI.
Plug-ins should declare an LMCache version range matching the plan fields they
use.
