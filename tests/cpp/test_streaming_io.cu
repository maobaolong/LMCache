// SPDX-License-Identifier: Apache-2.0

// Standalone GPU smoke test: compile once normally and once with
// -DLMCACHE_DISABLE_STREAMING_IO=1. See the CoreX build documentation.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda/streaming_io.cuh"

#define CHECK_CUDA(call)                                       \
  do {                                                         \
    const cudaError_t error = (call);                          \
    if (error != cudaSuccess) {                                \
      std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); \
      return 1;                                                \
    }                                                          \
  } while (0)

__global__ void copy_vectors(uint4* dst, const uint4* src, size_t count) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    lmc::store_streaming(dst + i, lmc::load_streaming(src + i));
  }
}

int main() {
  // A partial last block catches boundary mistakes; all four lanes differ.
  const size_t count = 259;
  std::vector<uint4> expected(count);
  std::vector<uint4> actual(count);
  for (size_t i = 0; i < count; ++i) {
    const unsigned int value = static_cast<unsigned int>(i);
    expected[i] = make_uint4(value, ~value, value * 17, 0x80000000U + value);
  }
  uint4* src;
  uint4* dst;
  CHECK_CUDA(cudaMalloc(&src, count * sizeof(uint4)));
  CHECK_CUDA(cudaMalloc(&dst, count * sizeof(uint4)));
  CHECK_CUDA(cudaMemcpy(src, expected.data(), count * sizeof(uint4),
                        cudaMemcpyHostToDevice));
  copy_vectors<<<(count + 127) / 128, 128>>>(dst, src, count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(actual.data(), dst, count * sizeof(uint4),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(dst));
  CHECK_CUDA(cudaFree(src));
  for (size_t i = 0; i < count; ++i) {
    if (actual[i].x != expected[i].x || actual[i].y != expected[i].y ||
        actual[i].z != expected[i].z || actual[i].w != expected[i].w) {
      std::fprintf(stderr, "vector mismatch at %zu\n", i);
      return 1;
    }
  }
  std::puts("streaming I/O roundtrip passed (259 vectors)");
  return 0;
}
