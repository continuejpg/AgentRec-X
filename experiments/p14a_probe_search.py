"""P14-A search/catalogue resource probe (limited catalogue).

Measures, on the real projected catalogue:

* catalogue load time and projected record count;
* BM25 index build time (the accepted ``CatalogSearchSource`` build);
* end-to-end ``product_search`` latency for a deterministic sample of real train
  task queries (question text only; no ground truth is read);
* peak process RSS on Linux/WSL (``resource.getrusage``).

Writes one JSON artifact under ``runs/p14a/logs/``.  This is a resource/perf
observation for the P14-A report, not a retrieval-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATE_PREFIX = re.compile(r"^Current Date:[^\n]*\n", re.IGNORECASE)


def _load_questions(tasks_path: Path, sample_count: int) -> list[str]:
    questions: list[str] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question = DATE_PREFIX.sub("", str(row.get("question") or "")).strip()
            if question:
                questions.append(question)
    if not questions:
        raise SystemExit(f"no questions found in {tasks_path}")
    step = max(1, len(questions) // sample_count)
    sampled = questions[::step][:sample_count]
    return sampled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=REPO_ROOT / "data/p14a/catalogue/docs.jsonl")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/p14a/catalogue/manifest.json")
    parser.add_argument("--tasks", type=Path, default=REPO_ROOT / "data/p14a/dataset/train_tasks.jsonl")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs/p14a/logs/search_probe.json")
    args = parser.parse_args(argv)

    from recommendation.shopping_companion.catalogue import ShoppingCompanionCatalogue
    from recommendation.shopping_companion.search import ShoppingCompanionSearchTool

    started = time.perf_counter()
    catalogue = ShoppingCompanionCatalogue.load(
        args.catalogue,
        manifest_path=args.manifest if args.manifest.is_file() else None,
    )
    load_seconds = time.perf_counter() - started

    index_started = time.perf_counter()
    source = catalogue.search_source  # builds the accepted BM25 index once
    index_seconds = time.perf_counter() - index_started

    questions = _load_questions(args.tasks, args.queries)
    tool = ShoppingCompanionSearchTool(catalogue)
    latencies: list[float] = []
    hits_per_query: list[int] = []
    samples: list[dict] = []
    for question in questions:
        result = tool.search(question, limit=args.limit)
        latencies.append(result.latency_ms)
        hits_per_query.append(len(result.hits))
        samples.append(
            {
                "query": question,
                "latency_ms": round(result.latency_ms, 3),
                "hits": len(result.hits),
                "top_product_ids": list(result.product_ids[:5]),
                "matched": result.matched,
            }
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else {}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload = {
        "schema": "agentrecx.p14a.search_probe.v1",
        "catalogue": {
            "docs_path": str(args.catalogue),
            "catalogue_size": catalogue.catalogue_size,
            "limited_catalogue": catalogue.limited,
            "docs_sha256": manifest.get("docs", {}).get("sha256") or manifest.get("docs_sha256"),
            "load_seconds": round(load_seconds, 3),
            "bm25_index_build_seconds": round(index_seconds, 3),
            "searchable_fields": list(source.fields),
            "score_kind": source.score_kind,
        },
        "queries": {
            "count": len(samples),
            "limit": args.limit,
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
                "median": round(statistics.median(latencies), 3) if latencies else None,
                "max": round(max(latencies), 3) if latencies else None,
                "all": [round(value, 3) for value in latencies],
            },
            "hits_mean": round(statistics.fmean(hits_per_query), 2) if hits_per_query else None,
            "samples": samples,
        },
        "resources": {
            "peak_rss_mb": round(usage.ru_maxrss / 1024.0, 1),
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "environment": "WSL Ubuntu-22.04, CPU-only (no GPU used)",
        "note": (
            "limited-catalogue resource probe; latency is for the real BM25 product_search tool "
            "over the loaded sample, not a full-catalogue claim"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "catalogue_size": payload["catalogue"]["catalogue_size"],
        "limited": payload["catalogue"]["limited_catalogue"],
        "load_seconds": payload["catalogue"]["load_seconds"],
        "index_build_seconds": payload["catalogue"]["bm25_index_build_seconds"],
        "latency_ms_median": payload["queries"]["latency_ms"]["median"],
        "latency_ms_max": payload["queries"]["latency_ms"]["max"],
        "peak_rss_mb": payload["resources"]["peak_rss_mb"],
        "out": str(args.out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
