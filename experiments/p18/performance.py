"""P18 performance decomposition (CPU, local; no GPU/Qwen)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import LocalBGEDenseIndex, configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p18"


def main() -> None:
    result = {"schema": "agentrecx.p18.performance.v1", "mode": "CPU_MCP", "gpu_used": False,
              "qwen_used": False, "notes": [
                  "component timings are in one local CPU process; MCP protocol startup is measured separately by demos/client",
                  "BGE index and model are reused in-process; no index/rebuild was performed"]}
    started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load()
    result["catalogue_load_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    started = time.perf_counter()
    config = configure_catalogue(catalogue)
    result["index_and_search_source_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    result["search_backend"] = config.get("status")
    started = time.perf_counter()
    task = next(json.loads(line) for line in
                (REPO / "data/p14e2/benchmark/dev.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and "p14e2_lite_0049" in line)
    preferences, history = _memory_read(catalogue, task)
    result["memory_fixture_load_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    result["memory_task"] = task["task_id"]
    # Distinguish encoder load from actual query+search where possible.
    dense = getattr(catalogue.search_source, "_dense", None)
    if dense is not None:
        started = time.perf_counter()
        dense._load_encoder()
        result["bge_encoder_load_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    started = time.perf_counter()
    first = catalogue.search_source.search("wireless earbuds for working out", limit=10)
    result["first_search_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    result["first_search_mode"] = first.score_kind
    started = time.perf_counter()
    second = catalogue.search_source.search("wireless keyboard for a desk", limit=10)
    result["second_search_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    result["second_search_mode"] = second.score_kind
    result["catalogue_size"] = first.catalogue_size
    (OUT / "performance_components.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
