# CUDA-compatible toolchain capabilities

Runtime API identity and compiler capabilities are separate. CoreX exposes
`torch.cuda`, so it uses `CudaDeviceSpec` and `lmcache.cuda_ops`. No runtime
vendor-name detection or new device registration is needed for this build path.

```text
BUILD_WITH_ILUVATAR=1
  -> BuildPolicy explicit selection
  -> IluvatarProfile -> inherited CudaProfile.build()
       cuda_defines(): LMCACHE_DISABLE_STREAMING_IO=1
       nvcc_flags(): installed PyTorch ABI, no NVIDIA --threads
       requirements_file(): None; extras_requirements(): {}
  -> one CUDAExtension / one source list
  -> streaming_io.cuh -> ordinary uint4 loads/stores
```

`CudaProfile.cuda_defines()` supplies capability macros to both C++ and device
compilation through `define_macros`. `nvcc_flags()` owns compiler-specific
flags and NVIDIA build parallelism. The shared builder owns the source list,
module name and include directories. CoreX overrides `default_cxx_flags()` so
common extensions and CUDA sources match its installed PyTorch ABI. It inherits
the compiler-presence check, which is not a vendor detector; callers must select
the profile explicitly. The first-match policy and other profiles are unchanged.

The MP copy kernel delegates vector access to `load_streaming` and
`store_streaming`. NVIDIA retains `ld/st.global.cs.v4.u32`; toolchains defining
`LMCACHE_DISABLE_STREAMING_IO` use ordinary `uint4` dereferences. The existing
MetaX macro remains valid. Cache streaming is a performance hint, not a fence or
an atomic operation; this change adds no ordering guarantees. Existing launch,
stream and lifetime rules still apply. `mem_kernels.cu` does not use FP8 symbols,
so its unused FP8 includes are removed instead of adding a vendor exception.

Build-profile tests check explicit/headless selection, unchanged CUDA automatic
selection, explicit conflicts, shared source lists, capability macros, ABI
propagation and NVIDIA thread flags. The standalone GPU smoke test exercises
all four vector lanes with a partial last block, in both compiler modes on
NVIDIA and portable mode on CoreX. It does not substitute for layout-specific
KV roundtrips, MP/inference tests or a bandwidth benchmark on BI-V150.

This is a source-build adapter. CoreX SDK/PyTorch compatibility, common package
dependencies, optional CUDA driver APIs and the full extension must be checked
in a vendor environment. A successful NVIDIA fallback test does not establish
that the CoreX compiler accepts every source in current `dev`.
