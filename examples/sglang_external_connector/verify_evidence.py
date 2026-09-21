# SPDX-License-Identifier: Apache-2.0
"""Independently check saved GPU evidence, without importing SGLang or CUDA."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import re


def read_json(root: Path, name: str) -> Any:
    """Read a JSON artifact relative to the evidence run directory."""
    return json.loads((root / name).read_text())


def verify(root: Path) -> dict[str, Any]:
    """Cross-check responses, transfer completions, cleanup, and process exits."""
    verdict = read_json(root, "verdict.json")
    assert verdict["passed"] is True, verdict
    rows = verdict["results"]
    assert len(rows) == 36 and all(row["passed"] for row in rows)
    manifest = read_json(root, "manifest.json")
    assert manifest["sglang_git_status"] == manifest["LMCache_git_status"] == ""
    assert manifest["gpu_count"] == 2
    exits = read_json(root, "process-exits.json")
    assert len(exits) == 7, exits
    assert all(not item["forced_kill"] for item in exits)
    sglang_exits = [item for item in exits if item["name"] != "lmcache"]
    assert len(sglang_exits) == 6
    assert all(item["returncode"] == 0 for item in sglang_exits)
    # Uvicorn restores and re-raises SIGTERM after completing lifespan cleanup.
    lmcache_exit = next(item for item in exits if item["name"] == "lmcache")
    assert lmcache_exit["returncode"] in (0, -15), lmcache_exit
    lmcache_log = (root / "lmcache.log").read_text()
    assert "MPCacheServer closed" in lmcache_log
    assert "Application shutdown complete." in lmcache_log

    compared = 0
    max_logprob_diff = 0.0
    completed_transfers = []
    for tp in (1, 2):
        prefix = f"tp{tp}"
        logs = "\n".join(
            path.read_text() for path in sorted(root.glob(f"{prefix}-external*.log"))
        )
        assert "impl=LMCacheUnifiedConnector" in logs
        assert "source=external_kv_connector" in logs
        assert "success=False" not in logs
        for rank in range(tp):
            stores = re.findall(
                rf"External KV store completed: rid={prefix}-cold rank={rank} "
                r"start=(\d+) end=(\d+) success=True",
                logs,
            )
            covered = 0
            for start, end in sorted((int(start), int(end)) for start, end in stores):
                assert start <= covered < end, (prefix, rank, stores)
                covered = end
            assert covered >= 2048, (prefix, rank, stores)
        for row in rows:
            name = row["case"]
            if not name.startswith(prefix) or name in (
                f"{prefix}-cancel",
                f"{prefix}-shutdown",
            ):
                continue
            result = read_json(root, f"{name}-response.json")
            assert len(result["output_ids"]) == 32
            meta = result["meta_info"]
            details = meta.get("cached_tokens_details") or {}
            if name.endswith(
                (
                    "-baseline",
                    "-cold",
                    "-short-prompt",
                    "-salt-isolation",
                    "-after-clear",
                )
            ):
                assert meta["cached_tokens"] == details.get("storage", 0) == 0
            if name.endswith("-local-warm"):
                assert details["device"] == 2048 and details["storage"] == 0
            if "baseline" not in name:
                kind = "extended-" if "partial-hit" in name else ""
                if "short-prompt" in name:
                    kind = "short-"
                baseline = read_json(root, f"{prefix}-{kind}baseline-response.json")
                assert result["output_ids"] == baseline["output_ids"], name
                assert result["text"] == baseline["text"], name
                diff = max(
                    abs(x[0] - y[0])
                    for x, y in zip(
                        result["meta_info"]["output_token_logprobs"],
                        baseline["meta_info"]["output_token_logprobs"],
                        strict=True,
                    )
                )
                assert diff <= 1e-3, (name, diff)
                max_logprob_diff = max(max_logprob_diff, diff)
                compared += 1
            if row["storage"] > 0:
                details = result["meta_info"]["cached_tokens_details"]
                assert details["device"] == 0 and details["storage"] == 2048
                for rank in range(tp):
                    pattern = (
                        rf"External KV retrieve completed: rid={re.escape(name)} "
                        rf"rank={rank} start=0 end=2048 success=True"
                    )
                    assert re.search(pattern, logs), (name, rank)
                    completed_transfers.append({"case": name, "rank": rank})
        for path in sorted(root.glob(f"{prefix}-*-backend-status.json")):
            status = json.loads(path.read_text())
            l1 = status["storage_manager"]["l1_manager"]
            assert status["active_sessions"] == status["active_prefetch_jobs"] == 0
            assert l1["write_locked_count"] == l1["read_locked_count"] == 0
            assert l1["temporary_count"] == 0
            if "before-restart" in path.name or "shutdown" in path.name:
                assert status["registered_gpu_ids"] == []
            elif "final" in path.name:
                assert len(status["registered_gpu_ids"]) == tp
        assert len(list(root.glob(f"{prefix}-*-backend-status.json"))) == 4
        tail = (root / f"{prefix}-cancel-stream-tail.txt").read_text().splitlines()
        events = [json.loads(line[6:]) for line in tail if line.startswith("data: {")]
        assert events[-1]["meta_info"]["finish_reason"]["type"] == "abort"

    assert compared == 26, compared
    return {
        "passed": True,
        "case_count": len(rows),
        "responses_compared_to_no_cache_baselines": compared,
        "matching_output_token_ids": compared * 32,
        "max_output_logprob_abs_diff": max_logprob_diff,
        "verified_rank_retrieves": completed_transfers,
        "cold_store_prefix_verified_on_all_three_ranks": True,
        "all_six_sglang_processes_exited_zero": True,
        "lmcache_lifespan_cleanup_completed": True,
        "forced_kills": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    print(json.dumps(verify(parser.parse_args().run_directory), indent=2))
