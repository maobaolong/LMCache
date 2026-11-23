# SPDX-License-Identifier: Apache-2.0
"""Generate one token with the CI's tiny MLA profile, without LMCache."""
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    model = LLM(
        model="/work/deepseek-model",
        dtype="bfloat16",
        load_format="dummy",
        hf_overrides={
            "num_hidden_layers": 2,
            "n_routed_experts": 4,
            "first_k_dense_replace": 0,
            "num_experts_per_tok": 2,
        },
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=4,
        kv_cache_memory_bytes=1073741824,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True,
    )
    print(model.generate(["Hello"], SamplingParams(max_tokens=5)), flush=True)
