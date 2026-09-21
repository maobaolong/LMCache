# External SGLang KV connector: H200 E2E evidence

**Passed: 36 checks at TP=1 and TP=2.** The final run compared 26 responses against no-cache baselines: all 832 output token IDs and all generated text matched, with a maximum absolute output-token log-probability difference of **0.0** (asserted tolerance: 0.001). An independent, standard-library-only verifier also checked the saved responses against completed transfers on every TP rank and the backend cleanup state.

This validates [SGLang PR #40534](https://github.com/sgl-project/sglang/pull/40534) with a real, separately installed LMCache adapter. SGLang loads `lmcache.integration.sglang.external_kv_connector.LMCacheUnifiedConnector` through `--kv-transfer-config`. The adapter ports the implementation from [SGLang #38652](https://github.com/sgl-project/sglang/pull/38652), at `9effa29925d73508fb43e526bb436c3fcefc6bb7`, into LMCache. The transfer implementation comes from [LMCache #4828](https://github.com/LMCache/LMCache/pull/4828). This is a validation adapter on a separate branch; it is not the existing vLLM connector or a released LMCache integration.

## Results

The 36 entries include six no-cache baseline requests and two shutdown checks. Every completed non-baseline generation produces 32 greedy tokens and is compared with the appropriate baseline at the same TP size.

| Scenario | TP=1 | TP=2 |
|---|---|---|
| Cold 2049-token prompt | 0 cached tokens | 0 cached tokens |
| Immediate local reuse | device=2048, storage=0 | device=2048, storage=0 |
| Successful local flush, then external reuse | device=0, storage=2048 | device=0, storage=2048 |
| Partial hit: 2305-token prompt | 2048 restored, 257 prefilled | 2048 restored, 257 prefilled |
| 129-token prompt, below the LMCache chunk size | 0 cached tokens | 0 cached tokens |
| Same prompt with another cache salt | 0 cached tokens | 0 cached tokens |
| Four concurrent requests after local flush | all four: storage=2048 | all four: storage=2048 |
| Cancel streaming request | explicit abort event; resources drained | explicit abort event; resources drained |
| Request after cancellation and local flush | device=0, storage=2048 | device=0, storage=2048 |
| Restart SGLang; keep LMCache running | device=0, storage=2048 | device=0, storage=2048 |
| Clear external storage, then request | 0 cached tokens | 0 cached tokens |
| Graceful shutdown | CUDA IPC registrations=0 | CUDA IPC registrations=0 |

Cold-store completions cover the entire 2048-token prefix on all three participating ranks (one TP=1 rank and two TP=2 ranks). The verifier matches **24 successful per-rank retrieve completions** to the externally cached, completed responses. Cancellation also has actual retrieve events in the raw logs.

After cancellation/drain, active sessions, prefetch jobs, temporary objects, and read/write locks are all zero. Before each restart and after final shutdown, registered GPU instances are also zero. All six SGLang frontend processes exit with code 0; no process needs a forced kill. The final GPU snapshot shows 4 MiB used on each card, matching the pre-test idle state.

## Exact environment

| Item | Value |
|---|---|
| Date | 2026-09-21 |
| Hardware | 2 × NVIDIA H200, 143771 MiB each |
| NVIDIA driver | 580.178.04 |
| Image | `lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1` |
| Python / PyTorch / CUDA | 3.12.3 / 2.13.0+cu130 / 13.0 |
| SGLang source | [`38efeeca38f28ee6b30464905596e89a68947430`](https://github.com/maobaolong/sglang/commit/38efeeca38f28ee6b30464905596e89a68947430) |
| LMCache source | [`a4d8e82c4e396c6b56178844918da82aadd9c22a`](https://github.com/maobaolong/LMCache/commit/a4d8e82c4e396c6b56178844918da82aadd9c22a) |
| LMCache adapter base | `8d25da3a32bf05a68d9f7eb9d1b89dfd8733e7d5` |
| Model | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` |
| FlashInfer | Python 0.6.18, cubin 0.6.18, JIT cache 0.6.18+cu130 |
| SGLang kernel / Transformers | 0.4.7 / 5.12.1 |

Both source checkouts were clean when the run started. The manifest records the actual source commits. Editable-install version banners can retain earlier install-time metadata; the source commits, rather than those banners, identify the tested implementation. The package inventory excludes the two editable projects and lists their runtime dependencies separately.

The run uses BF16, full attention, `kv_both`, colocated serving, the Python serving path, normal overlap scheduling, and prefill/decode CUDA Graphs. SGLang uses 16-token pages, 512-token chunked prefill, a 16384-token pool, and an 8192-token context limit. LMCache runs as a separate process with CUDA IPC, 256-token chunks, and a 4 GiB CPU cache. [commands.json](evidence/commands.json) contains every complete server command.

## Evidence and offline verification

- [Independent verification result](evidence/verification.json)
- [All 36 results](evidence/verdict.json), [source/environment manifest](evidence/manifest.json), [process exit records](evidence/process-exits.json)
- [Transfer and factory-selection log excerpts](evidence/transfer-events.log), including source-log filenames and line numbers
- [Complete evidence archive](evidence/h200-tp1-tp2-evidence.tar.gz): 131 evidence files plus a checksum manifest, containing raw server logs, requests, responses, flush/clear results, metrics, backend status snapshots, cancellation events, environment setup logs, and the final GPU snapshot
- [Archive SHA256](evidence/SHA256SUMS); the archive also contains per-file checksums

The following only needs Python 3.12 and standard command-line tools, without CUDA or either package installed. Run from this directory:

```bash
(cd evidence && sha256sum -c SHA256SUMS)
mkdir decoded-evidence
tar -xzf evidence/h200-tp1-tp2-evidence.tar.gz -C decoded-evidence
(cd decoded-evidence && sha256sum -c SHA256SUMS)
python3.12 verify_evidence.py decoded-evidence/run
```

The LMCache process returns `-15` after the requested SIGTERM. Its log records `MPCacheServer closed` and successful application shutdown. Uvicorn 0.52.4 restores and re-raises the original signal after lifespan cleanup; the verifier checks both the exit status and these cleanup records. This is distinct from a forced kill or a scheduler failure.

## Reproduce on GPUs

Prerequisites: Linux x86_64, Git, Docker with NVIDIA Container Toolkit, two idle H200 GPUs, a CUDA 13-compatible driver, internet access to GitHub/PyPI/FlashInfer/Hugging Face, and sufficient disk for the serving image, model, sources, and compiler caches. The HTTP/RPC ports 13085, 15585, and 18085 must be free. Select two idle GPU indices after checking `nvidia-smi`.

```bash
git clone --branch validation/sglang-external-kv-40534 \
  https://github.com/maobaolong/LMCache.git lmcache-connector-validation
GPU_DEVICES=0,1 E2E_WORK_DIR="$PWD/kv-connector-e2e" \
  bash lmcache-connector-validation/examples/sglang_external_connector/reproduce.sh
```

[reproduce.sh](reproduce.sh) checks out the exact two source commits and pins the image. [setup.sh](setup.sh) builds LMCache for SM90 with `MAX_JOBS=1`, installs SGLang's declared runtime dependencies, aligns all FlashInfer artifacts, and downloads the pinned model. [run.sh](run.sh) checks clean source trees and invokes the [GPU harness in the SGLang PR](https://github.com/maobaolong/sglang/blob/38efeeca38f28ee6b30464905596e89a68947430/examples/runtime/kv_connector/validate_lmcache.py). The container mount `/workspace` is defined by the reproduction script; all paths under it in the evidence refer to that mount.

The reproduction retains its container and writes logs under `kv-connector-e2e/evidence`. Run the orchestration in a persistent terminal session when connecting over SSH. You can follow progress with `tail -f kv-connector-e2e/evidence/run/runner.log` after setup completes. After inspection, stop the retained container with `docker stop sglang-external-kv-40534`.

## Scope

This is correctness validation, not a latency benchmark. Durations are diagnostic only. It does not establish hybrid/SWA/Mamba, MLA/DSA, speculative decoding, DP/DCP/PP, prefill/decode disaggregation, transport-failure recovery, persistence across an LMCache restart, other GPU architectures, or production load capacity. The adapter rejects non-`kv_both` roles, hybrid/SWA/SSM and DSA contexts, speculative decoding, DP attention, DCP, and PP. SGLang's connector interface still needs maintainer review; these results apply to the explicit configuration above.
