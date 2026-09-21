#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${GPU_DEVICES:?Set GPU_DEVICES to two idle NVIDIA GPU indices, e.g. 0,1}"
work_dir=${E2E_WORK_DIR:-"$PWD/external-kv-40534"}
container_name=${E2E_CONTAINER_NAME:-sglang-external-kv-40534}
sglang_revision=38efeeca38f28ee6b30464905596e89a68947430
lmcache_revision=a4d8e82c4e396c6b56178844918da82aadd9c22a
image=lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work_dir"
work_dir=$(cd "$work_dir" && pwd)
git clone --filter=blob:none https://github.com/maobaolong/sglang.git "$work_dir/sglang"
git -C "$work_dir/sglang" checkout "$sglang_revision"
git clone --filter=blob:none https://github.com/maobaolong/LMCache.git "$work_dir/LMCache"
git -C "$work_dir/LMCache" checkout "$lmcache_revision"
cp "$script_dir/setup.sh" "$work_dir/setup.sh"
cp "$script_dir/run.sh" "$work_dir/run.sh"
docker run -d --name "$container_name" --gpus "\"device=$GPU_DEVICES\"" \
    --ipc=host --network=host --entrypoint bash \
    -v "$work_dir:/workspace" -e HF_HOME=/workspace/hf-cache \
    "$image" -lc 'sleep infinity'
docker exec "$container_name" bash /workspace/setup.sh
docker exec "$container_name" bash /workspace/run.sh
printf 'Evidence: %s/evidence\nContainer retained for inspection: %s\n' "$work_dir" "$container_name"
