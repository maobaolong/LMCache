#!/usr/bin/env bash
# Unit test entrypoint for K8s pods.
# Installs LMCache + test deps, optionally qualifying a real vLLM nightly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Per-build scratch dir ────────────────────────────────────
# /scratch is a shared hostPath mount (see pipeline.yml) backed by ext4/xfs
# on local NVMe — the only place GDS (cuFile) works in this pod (overlayfs
# /tmp fails with CU_FILE_IO_NOT_SUPPORTED). Give this build its own
# subdirectory so concurrent pods don't collide, and clean it up on exit.
# Direct subdir rather than K8s subPathExpr since the latter's bind mount
# breaks cuFile's fs-type detection.
#
# LMCACHE_TEST_TMPDIR is consumed by the handful of test files that need
# GDS-capable scratch (test_gds_backend, test_cache_engine, the xpu
# benchmarks). Leaving TMPDIR unset means /tmp stays pod-internal overlay
# for pip/uv/build caches — those don't need direct I/O.
BUILD_TAG="${BUILDKITE_BUILD_ID:-manual-$$}"
export LMCACHE_TEST_TMPDIR="/scratch/bk-${BUILD_TAG}"
mkdir -p "${LMCACHE_TEST_TMPDIR}"
trap 'rm -rf "${LMCACHE_TEST_TMPDIR}" 2>/dev/null || true' EXIT

# ── Environment setup ────────────────────────────────────────
if [[ "${VLLM_NIGHTLY_UNIT_TESTS:-0}" == "1" ]]; then
    source .buildkite/k3_harness/setup-env.sh
else
    source .buildkite/k3_harness/setup-lmcache-only-env.sh
fi
uv pip install -r requirements/test.txt
# Generated gRPC bindings are intentionally not tracked. Generate them only
# after the test dependencies have installed grpcio-tools.
python lmcache/v1/transport/grpc/_proto_gen/_generate.py

if [[ "${VLLM_NIGHTLY_UNIT_TESTS:-0}" == "1" ]]; then
    python - <<'PY'
import importlib.metadata
import os

import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

expected = os.environ["PINNED_VLLM_VERSION"]
if importlib.metadata.version("vllm") != expected or vllm.__version__ != expected:
    raise SystemExit("Nightly qualification requires the exact selected runtime")
if not hasattr(KVConnectorBase_V1, "on_kv_cache_ready"):
    raise SystemExit("Selected runtime lacks the startup-ready callback")
print("Nightly full units use", vllm.__version__, "with startup-ready callback")
PY
fi

# ── Run unit tests with coverage ─────────────────────────────
# PR builds stop at the first failure for fast feedback. The weekly full build
# runs every test so one regression cannot hide the others, and keeps a JUnit
# report of each failure and skip.
pytest_mode_args=(--maxfail=1)
if [[ "${CI_WEEKLY_FULL:-0}" == "1" || "${VLLM_NIGHTLY_UNIT_TESTS:-0}" == "1" ]]; then
    pytest_mode_args=(-rfEs --junitxml=junit-unit.xml)
fi
RUN_DEVDAX_L1_INTEGRATION=1 RUN_DAX_L2_INTEGRATION=1 \
LMCACHE_TEST_DEVDAX_L1_PATHS="" LMCACHE_TRACK_USAGE="false" \
pytest "${pytest_mode_args[@]}" --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/skipped

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF
