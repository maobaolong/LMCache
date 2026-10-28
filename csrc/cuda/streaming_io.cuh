// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector_types.h>

namespace lmc {

// Streaming cache hints reduce pollution during KV transfers. Toolchains
// without PTX support use the same vector copy with ordinary cache policy.
__device__ inline uint4 load_streaming(const uint4* addr) {
#if defined(__CUDA_ARCH__) && !defined(LMCACHE_DISABLE_STREAMING_IO)
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
#else
  return *addr;
#endif
}

__device__ inline void store_streaming(uint4* addr, uint4 val) {
#if defined(__CUDA_ARCH__) && !defined(LMCACHE_DISABLE_STREAMING_IO)
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  *addr = val;
#endif
}

}  // namespace lmc
