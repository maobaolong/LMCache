// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-agnostic transfer descriptors, free of any vendor runtime headers so
// every backend (CUDA, ROCm, MUSA, SYCL/XPU, ...) shares one definition.

#include "engine_kv_format.h"  // EngineKVFormat + its classification predicates

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

// Where a transfer kernel reads the base address of each LMCache object.
// HOST_VECTOR preserves the existing fast path, which passes up to four
// pointers in the kernel arguments. DEVICE_ARRAY reads an arbitrary number of
// pointers from a device-resident int64 array owned by the caller.
enum class MemoryObjectPointerSource : int {
  HOST_VECTOR = 0,
  DEVICE_ARRAY = 1,
};

// Logical order of the KV and layer axes inside one LMCache object. The token
// and hidden dimensions are always the two innermost dimensions.
enum class MemoryObjectLayout : int {
  KV_LAYER_TOKEN = 0,
  LAYER_KV_TOKEN = 1,
};
