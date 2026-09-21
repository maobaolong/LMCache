#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/sglang/python:/workspace/LMCache
export UV_BREAK_SYSTEM_PACKAGES=1
mkdir -p evidence
exec > evidence/setup.log 2>&1
git config --global --add safe.directory /workspace/sglang
git config --global --add safe.directory /workspace/LMCache
python --version
nvcc --version
nvidia-smi
uv pip install --system -r LMCache/requirements/build.txt setuptools-rust
# Build against the image's torch/CUDA, then install the checkout's runtime pins.
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1
export TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"
uv pip install --system --no-build-isolation -e LMCache
SGLANG_BUILD_RUST_EXTS=none uv pip install --system --no-build-isolation -e sglang/python
uv pip install --system flashinfer-cubin==0.6.18 --index-url https://flashinfer.ai/whl
uv pip install --system flashinfer-jit-cache==0.6.18 --index-url https://flashinfer.ai/whl/cu130
python - <<'PY'
import torch, sglang, lmcache, lmcache.cuda_ops
from lmcache.integration.sglang.external_kv_connector import LMCacheUnifiedConnector
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('sglang', sglang.__file__, 'lmcache', lmcache.__file__)
print('abstract_methods', LMCacheUnifiedConnector.__abstractmethods__)
PY
python -m pip list --format=freeze --exclude-editable > evidence/runtime-packages.txt
lmcache server --help > evidence/lmcache-server-help.txt
python - <<'PY'
import json
from huggingface_hub import snapshot_download
revision = 'c1899de289a04d12100db370d81485cdf75e47ca'
model = snapshot_download('Qwen/Qwen3-0.6B', revision=revision)
with open('evidence/model.json', 'w') as f:
    json.dump({'id': 'Qwen/Qwen3-0.6B', 'revision': revision, 'snapshot': model}, f, indent=2)
print('MODEL', model, revision)
PY
