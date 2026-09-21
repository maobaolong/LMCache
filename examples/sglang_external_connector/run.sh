#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/sglang/python:/workspace/LMCache
mkdir -p evidence/run
exec > evidence/run/runner.log 2>&1
python - <<'PYTHON'
import json, subprocess, torch, importlib.metadata
from pathlib import Path
manifest = {
    "sglang_commit": subprocess.check_output(["git", "-C", "sglang", "rev-parse", "HEAD"], text=True).strip(),
    "lmcache_commit": subprocess.check_output(["git", "-C", "LMCache", "rev-parse", "HEAD"], text=True).strip(),
    "image": "lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1",
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(),
    "model": json.loads(Path("evidence/model.json").read_text()),
}
for repo in ("sglang", "LMCache"):
    status = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True)
    manifest[repo + "_git_status"] = status
    assert not status, (repo, status)
Path("evidence/run/manifest.json").write_text(json.dumps(manifest, indent=2))
PYTHON
revision=$(python -c 'import json; print(json.load(open("evidence/model.json"))["revision"])')
python sglang/examples/runtime/kv_connector/validate_lmcache.py --revision "$revision" --evidence-dir evidence/run
