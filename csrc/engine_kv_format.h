// SPDX-License-Identifier: Apache-2.0

#pragma once

// Physical KV-cache memory layout an engine hands to LMCache, plus the
// classification predicates over it. Vendor-header-free, so every backend and
// the Python facade (lmc_ops) share one definition. Detection (raw layout ->
// format) lives in lmcache/v1/gpu_connector/kv_format.

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
CS: content size (per-head content width; 2 * head size when K/V are fused)
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)
*/
// One X-macro row per format: (NAME, cross, kv, layer, mla, hnd, fused, two,
// pbs). The enum below assigns each an implicit index (0, 1, 2, ...) in row
// order, and FORMAT_FACTS is built from the same rows, so the two can never
// drift and a new format is added in exactly one place. Exactly one structural
// shape (cross / kv / layer) is true per format; the rest are modifiers. The
// facts mirror the Python KVFormatSpec.
//
// clang-format off
#define LMC_KV_FORMATS(X)                                                    \
  X(NB_NL_TWO_BS_NH_HS,     true,  false, false, false, false, false, false, false) \
  X(NL_X_TWO_NB_BS_NH_HS,   false, false, true,  false, false, false, true,  false) \
  X(NL_X_NB_TWO_BS_NH_HS,   false, false, true,  false, false, false, false, false) \
  X(NL_X_NB_BS_HS,          false, false, true,  true,  false, false, false, false) \
  X(TWO_X_NL_X_NBBS_NH_HS,  false, true,  false, false, false, false, false, false) \
  X(NL_X_NBBS_ONE_HS,       false, false, true,  true,  false, false, false, true)  \
  X(NL_X_TWO_NB_NH_BS_HS,   false, false, true,  false, true,  false, true,  false) \
  X(NL_X_NB_TWO_NH_BS_HS,   false, false, true,  false, true,  false, false, false) \
  X(NB_NL_TWO_NH_BS_HS,     true,  false, false, false, true,  false, false, false) \
  X(TWO_X_NL_X_NB_BS_NH_HS, false, true,  false, false, false, false, false, false) \
  X(NL_X_NB_NH_BS_TWO_HS,   false, false, true,  false, true,  true,  false, false) \
  X(NL_X_NB_BS_NH_TWO_HS,   false, false, true,  false, false, true,  false, false) \
  X(NL_X_NB_NH_BS_CS,       false, false, true,  false, true,  true,  false, false) \
  X(NL_X_NB_BS_NH_CS,       false, false, true,  false, false, true,  false, false) \
  X(NL_X_NB_BSV_BSS,        false, false, true,  true,  false, false, false, false)
// clang-format on

enum class EngineKVFormat : int {
#define LMC_KV_FORMAT_ENUM(name, cross, kv, layer, mla, hnd, fused, two, pbs) \
  name,
  LMC_KV_FORMATS(LMC_KV_FORMAT_ENUM)
#undef LMC_KV_FORMAT_ENUM
};

// __host__ __device__ under CUDA/HIP so the kernels can call these; the guard
// keeps the header vendor-runtime-free.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_KV_FORMAT_HD __host__ __device__
#else
  #define LMC_KV_FORMAT_HD
#endif

// Static layout facts for each format, indexed by the enum value above. The
// predicates below are one-line lookups.
struct FormatFacts {
  bool is_cross_layer;   // all layers in one fused tensor
  bool is_kv_list;       // keys and values in two top-level lists
  bool is_layer_list;    // one list entry per layer
  bool is_mla;           // MLA: single latent KV head (no separate K/V)
  bool is_hnd;           // heads before block tokens (HND layout)
  bool is_fused_packed;  // K/V packed in trailing dim (kv_size == 1)
  bool is_two_major;     // size-2 K/V axis precedes the block axis
  bool is_pbs_fused;     // paged buffer size fused into one axis
};

LMC_KV_FORMAT_HD constexpr FormatFacts FORMAT_FACTS[] = {
#define LMC_KV_FORMAT_FACTS(name, cross, kv, layer, mla, hnd, fused, two, pbs) \
  {cross, kv, layer, mla, hnd, fused, two, pbs},
    LMC_KV_FORMATS(LMC_KV_FORMAT_FACTS)
#undef LMC_KV_FORMAT_FACTS
};

LMC_KV_FORMAT_HD constexpr const FormatFacts& format_facts(EngineKVFormat f) {
  return FORMAT_FACTS[static_cast<int>(f)];
}

// All layers in one fused tensor.
LMC_KV_FORMAT_HD constexpr bool is_cross_layer(EngineKVFormat f) {
  return format_facts(f).is_cross_layer;
}

// Keys and values in two separate top-level lists: [key_layers, value_layers].
LMC_KV_FORMAT_HD constexpr bool is_kv_list(EngineKVFormat f) {
  return format_facts(f).is_kv_list;
}

// One list entry per layer: kv_caches[layer_idx] is that layer's tensor.
LMC_KV_FORMAT_HD constexpr bool is_layer_list(EngineKVFormat f) {
  return format_facts(f).is_layer_list;
}

// Multi-head Latent Attention: a single latent KV head (no separate K/V split).
// The blocked-scale indexer cache transfers like MLA (single plane,
// kv_size == 1); only its paged addressing differs.
LMC_KV_FORMAT_HD constexpr bool is_mla(EngineKVFormat f) {
  return format_facts(f).is_mla;
}

// vLLM fused K/V: K and V packed in the trailing dim (2 * head_size), no
// separate K/V axis — transferred as one k_or_v == 0 pass (like MLA).
LMC_KV_FORMAT_HD constexpr bool is_fused_packed(EngineKVFormat f) {
  return format_facts(f).is_fused_packed;
}
