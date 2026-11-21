"""M3-C Baseline/Candidate retrieval experiment.

One experiment variable: the candidate searches a **typed category scope** with the same
effective BM25 query and the same BM25 parameters.  The baseline searches the full catalogue
with the current production recommendation query.  No production retrieval path is changed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_eval.retrieval.labeling import LABEL_POLICY, annotate_pool
from agent_eval.retrieval.queries import QuerySet, load_queries, queries_sha256
from agent_eval.retrieval.scope import CATEGORY_TERMS, CategoryScope
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.catalog.enrichment import MetadataSidecar
from recommendation.control.catalog_search import BM25_B, BM25_K1
from recommendation.preference.task_requirements import (
    augment_query,
    extract_task_requirements,
)

__all__ = ["RetrievalExperiment", "ExperimentConfig", "main", "REPORT_SCHEMA"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data/p13r/electronics/catalog_meta.jsonl"
SIDECAR_PATH = REPO_ROOT / "data/enrichment/electronics_metadata_sidecar.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "retrieval_experiments"
REPORT_SCHEMA = "agentrecx.retrieval_eval.report.v1"
TOP_K = 5
REPEATS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False,
            capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median_ms(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def _top_rows(catalogue: ElectronicsCatalogue, hits: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for hit in hits:
        item_id = str(hit.parent_asin).split(":")[-1]
        product = catalogue.get(item_id)
        rows.append({
            "item_id": item_id,
            "title": str(getattr(product, "title", "") or ""),
            "score": round(float(hit.score), 6),
        })
    return rows


@dataclasses.dataclass
class ExperimentConfig:
    queries_path: Path = REPO_ROOT / "agent_eval/retrieval/queries_v1.json"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_id: str | None = None
    top_k: int = TOP_K
    repeats: int = REPEATS


class RetrievalExperiment:
    def __init__(self, query_set: QuerySet, config: ExperimentConfig) -> None:
        self.query_set = query_set
        self.config = config

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = self.config.run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + (_git("rev-parse", "--short", "HEAD") or "nogit")
        )
        out_dir = self.config.output_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        index_started = time.perf_counter()
        catalogue = ElectronicsCatalogue.load(price_mode="strict")
        baseline_index_ms = round((time.perf_counter() - index_started) * 1000.0, 2)
        sidecar_started = time.perf_counter()
        sidecar = MetadataSidecar.load(SIDECAR_PATH)
        sidecar_load_ms = round((time.perf_counter() - sidecar_started) * 1000.0, 2)
        scope = CategoryScope(catalogue=catalogue, sidecar=sidecar, top_k=self.config.top_k)
        # Build the full baseline source before timing, exactly like the production process
        # builds its BM25 index once.
        baseline_source = catalogue.search_source
        baseline_index_build_ms = baseline_index_ms

        scope_sizes = scope.sizes()
        scope_build_ms: dict[str, float] = {}
        per_query: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for query in self.query_set.queries:
            requirements = extract_task_requirements(query.query)
            effective_query = augment_query(query.query, requirements)
            target = requirements.category
            if target not in CATEGORY_TERMS:
                target = None
            baseline_hits = baseline_source.search(
                effective_query, limit=self.config.top_k
            ).hits
            baseline_ms: list[float] = []
            for _ in range(max(1, self.config.repeats)):
                t0 = time.perf_counter()
                baseline_source.search(effective_query, limit=self.config.top_k)
                baseline_ms.append((time.perf_counter() - t0) * 1000.0)
            candidate_hits = []
            candidate_ms: list[float] = []
            if target is not None:
                if target not in scope_build_ms:
                    t0 = time.perf_counter()
                    scope.source(target)
                    scope_build_ms[target] = round((time.perf_counter() - t0) * 1000.0, 2)
                candidate_hits = scope.search(
                    target, effective_query, limit=self.config.top_k
                ).hits
                for _ in range(max(1, self.config.repeats)):
                    t0 = time.perf_counter()
                    scope.search(target, effective_query, limit=self.config.top_k)
                    candidate_ms.append((time.perf_counter() - t0) * 1000.0)
            base_rows = _top_rows(catalogue, list(baseline_hits))
            cand_rows = _top_rows(catalogue, list(candidate_hits))
            union_ids = [row["item_id"] for row in base_rows + cand_rows]
            annotation = annotate_pool(
                query_id=query.query_id,
                target=target or query.expected_category,
                union_ids=union_ids,
                catalogue=catalogue,
                use_cases=list(requirements.use_cases),
                features=list(query.expected_features),
            )
            annotations.append(annotation)
            labels = annotation["judged_ids"]

            def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
                relevant = sum(1 for row in rows if labels.get(row["item_id"], 0) == 1)
                unjudged = [row["item_id"] for row in rows if row["item_id"] not in labels]
                return {
                    "relevant_count": relevant,
                    "denominator": self.config.top_k,
                    "precision_at_5": round(relevant / self.config.top_k, 4),
                    "error_count": self.config.top_k - relevant,
                    "error_share": round(
                        (self.config.top_k - relevant) / self.config.top_k, 4
                    ),
                    "unjudged_ids": unjudged,
                    "rows": [
                        {
                            **row,
                            "relevant": labels.get(row["item_id"], 0),
                            "label_reason": annotation["rows"].get(
                                row["item_id"], {}
                            ).get("reason"),
                        }
                        for row in rows
                    ],
                }

            baseline_score = score(base_rows)
            candidate_score = score(cand_rows)
            delta = round(
                candidate_score["precision_at_5"] - baseline_score["precision_at_5"], 4
            )
            if target is None:
                status = "not_applicable"
            elif delta > 0:
                status = "improved"
            elif delta < 0:
                status = "degraded"
            else:
                status = "unchanged"
            per_query.append({
                "query_id": query.query_id,
                "query": query.query,
                "expected_category": query.expected_category,
                "extracted_category": requirements.category,
                "expected_use_cases": list(query.expected_use_cases),
                "extracted_use_cases": list(requirements.use_cases),
                "expected_features": list(query.expected_features),
                "effective_query": effective_query,
                "target_scope": target,
                "scope_size": scope_sizes.get(target) if target else None,
                "baseline": baseline_score,
                "candidate": candidate_score,
                "delta_precision_at_5": delta,
                "status": status,
                "baseline_latency_ms": {
                    "samples": [round(value, 4) for value in baseline_ms],
                    "median": _median_ms(baseline_ms),
                },
                "candidate_latency_ms": {
                    "samples": [round(value, 4) for value in candidate_ms],
                    "median": _median_ms(candidate_ms),
                },
                "pool_size": annotation["pool_size"],
                "relevant_union_count": annotation["relevant_union_count"],
            })

        def aggregate(key: str) -> dict[str, Any]:
            numerator = sum(row[key]["relevant_count"] for row in per_query)
            denominator = sum(row["baseline"]["denominator"] for row in per_query)
            errors = sum(row[key]["error_count"] for row in per_query)
            return {
                "numerator": numerator,
                "denominator": denominator,
                "precision_at_5": round(numerator / denominator, 4)
                if denominator else None,
                "error_count": errors,
                "error_share": round(errors / denominator, 4) if denominator else None,
            }

        baseline = aggregate("baseline")
        candidate = aggregate("candidate")
        improved = [row["query_id"] for row in per_query if row["status"] == "improved"]
        degraded = [row["query_id"] for row in per_query if row["status"] == "degraded"]
        same = [row["query_id"] for row in per_query if row["status"] == "unchanged"]
        metadata = {
            "run_id": run_id,
            "generated_at": _now(),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "query_dataset_version": self.query_set.dataset_version,
            "query_dataset_sha256": queries_sha256(self.config.queries_path),
            "query_count": len(self.query_set.queries),
            "catalog_sha256": _sha256(CATALOG_PATH),
            "catalog_size": catalogue.catalogue_size,
            "sidecar_sha256": _sha256(SIDECAR_PATH),
            "sidecar_records": len(sidecar),
            "price_mode": "strict",
            "ranking_price_dependency": "none; M3-C does not use price for retrieval",
            "bm25": {
                "mode": "bm25",
                "k1": BM25_K1,
                "b": BM25_B,
                "top_k": self.config.top_k,
                "repeats": self.config.repeats,
            },
            "test_environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "baseline_index_build_ms": baseline_index_build_ms,
            "sidecar_load_ms": sidecar_load_ms,
            "candidate_scope_build_ms": dict(sorted(scope_build_ms.items())),
            "candidate_scope_sizes": scope_sizes,
            "run_wall_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "label_policy": LABEL_POLICY,
            "label_source": "machine-assisted by Harness; pending human review",
            "no_production_change": True,
        }
        metrics = {
            "baseline": baseline,
            "candidate": candidate,
            "delta_precision_at_5": round(
                (candidate["precision_at_5"] or 0.0)
                - (baseline["precision_at_5"] or 0.0),
                4,
            ),
            "improved_queries": improved,
            "degraded_queries": degraded,
            "unchanged_queries": same,
            "latency_ms": {
                "baseline_median_sum": round(
                    sum(row["baseline_latency_ms"]["median"] for row in per_query), 3
                ),
                "candidate_median_sum": round(
                    sum(row["candidate_latency_ms"]["median"] for row in per_query), 3
                ),
                "note": "search-only medians of 3 repetitions; index build times are separate",
            },
        }
        report = {
            "report_schema": REPORT_SCHEMA,
            "metadata": metadata,
            "queries": self.query_set.as_dict(),
            "annotations": annotations,
            "per_query": per_query,
            "metrics": metrics,
            "conclusion": self._conclusion(baseline, candidate, improved, degraded),
        }
        _write_reports(out_dir, report)
        return {"report": report, "output_dir": out_dir}

    @staticmethod
    def _conclusion(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        improved: list[str],
        degraded: list[str],
    ) -> dict[str, Any]:
        delta = (candidate.get("precision_at_5") or 0.0) - (
            baseline.get("precision_at_5") or 0.0
        )
        if delta <= 0:
            verdict = "no adopt"
            rationale = (
                "The candidate did not improve pooled Precision@5; production retrieval "
                "stays unchanged."
            )
        elif degraded:
            verdict = "not adopted in this milestone"
            rationale = (
                "The candidate improved the aggregate but at least one query degraded; the "
                "labels are machine-assisted and still need human review. Production stays "
                "unchanged."
            )
        else:
            verdict = "promising; not integrated"
            rationale = (
                "The candidate improved pooled Precision@5 with no observed per-query "
                "regression on this 10-query fixture, but the labels are machine-assisted "
                "and pending human review. Production stays unchanged in M3-C."
            )
        return {
            "verdict": verdict,
            "aggregate_delta_precision_at_5": round(delta, 4),
            "improved_queries": improved,
            "degraded_queries": degraded,
            "rationale": rationale,
            "production_change": False,
        }


def _write_reports(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "annotations.json").write_text(
        json.dumps(
            {
                "schema": "agentrecx.retrieval_eval.annotations.v1",
                "label_policy": report["metadata"]["label_policy"],
                "label_source": report["metadata"]["label_source"],
                "review_status": "pending_human_review",
                "queries": report["annotations"],
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in report["per_query"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_annotations_md(out_dir / "annotations.md", report["annotations"])
    _write_report_md(out_dir / "report.md", report)


def _write_annotations_md(path: Path, annotations: list[dict[str, Any]]) -> None:
    lines = [
        "# M3-C pooled relevance annotations",
        "",
        "> **Machine-assisted labels by the Harness; pending human review.**",
        "> Labels use product titles only; BM25 scores/ranks were not used.",
        "",
        "| Query | Target | Item | Title | Grade | Reason |",
        "|---|---|---|---|---:|---|",
    ]
    for annotation in annotations:
        for item_id, row in annotation["rows"].items():
            title = str(row.get("title", "")).replace("|", "\\|")
            lines.append(
                f"| `{annotation['query_id']}` | {annotation['target_category']} | "
                f"`{item_id}` | {title} | {row['grade']} | {row['reason']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_md(path: Path, report: dict[str, Any]) -> None:
    meta = report["metadata"]
    metrics = report["metrics"]
    base = metrics["baseline"]
    cand = metrics["candidate"]
    lines = [
        "# M3-C BM25 Retrieval Experiment",
        "",
        "> Baseline and candidate use the same catalogue, query set, BM25 parameters, top_k "
        "and price mode. The only experiment variable is candidate retrieval scope.",
        "> Relevance labels are **machine-assisted by the Harness and pending human review**; "
        "they are not independently human-verified.",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Run id | `{meta['run_id']}` |",
        f"| Git commit | `{meta['git_sha']}` |",
        f"| Git dirty | {meta['git_dirty']} |",
        f"| Query dataset | `{meta['query_dataset_version']}` / `{meta['query_dataset_sha256']}` |",
        f"| Query count | {meta['query_count']} |",
        f"| Catalogue SHA256 | `{meta['catalog_sha256']}` |",
        f"| Catalogue records | {meta['catalog_size']} |",
        f"| Sidecar SHA256 | `{meta['sidecar_sha256']}` |",
        f"| BM25 | k1={meta['bm25']['k1']}, b={meta['bm25']['b']}, top_k={meta['bm25']['top_k']} |",
        f"| Price mode | `{meta['price_mode']}` (not used by retrieval ranking) |",
        f"| Label source | {meta['label_source']} |",
        "",
        "**Baseline:** current production effective recommendation query "
        "(`augment_query(raw_query, extracted_requirements)`) over the full catalogue BM25 "
        "index.",
        "",
        "**Candidate:** the same effective query and BM25 parameters, but BM25 is run over a "
        "category-scoped index inferred from the query category and existing title/sidecar "
        "category metadata. No query wording changed.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Precision@5 | {base['precision_at_5']:.1%} ({base['numerator']}/{base['denominator']}) | "
        f"{cand['precision_at_5']:.1%} ({cand['numerator']}/{cand['denominator']}) | "
        f"{metrics['delta_precision_at_5']:+.1%} |",
        f"| Obvious error/accessory count | {base['error_count']}/{base['denominator']} | "
        f"{cand['error_count']}/{cand['denominator']} | "
        f"{cand['error_count'] - base['error_count']:+d} |",
        f"| Obvious error/accessory share | {base['error_share']:.1%} | "
        f"{cand['error_share']:.1%} | "
        f"{cand['error_share'] - base['error_share']:+.1%} |",
        "",
        f"- Improved queries: {', '.join(metrics['improved_queries']) or 'none'}",
        f"- Degraded queries: {', '.join(metrics['degraded_queries']) or 'none'}",
        f"- Unchanged queries: {', '.join(metrics['unchanged_queries']) or 'none'}",
        f"- Search-only median latency sum: baseline {metrics['latency_ms']['baseline_median_sum']} ms; "
        f"candidate {metrics['latency_ms']['candidate_median_sum']} ms",
        "",
        "## Per-query results",
        "",
        "| Query | Target | Baseline P@5 | Candidate P@5 | Delta | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["per_query"]:
        lines.append(
            f"| `{row['query_id']}` | {row['target_scope'] or 'n/a'} | "
            f"{row['baseline']['precision_at_5']:.1%} ({row['baseline']['relevant_count']}/5) | "
            f"{row['candidate']['precision_at_5']:.1%} ({row['candidate']['relevant_count']}/5) | "
            f"{row['delta_precision_at_5']:+.1%} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"**{report['conclusion']['verdict']}.** {report['conclusion']['rationale']}",
        "",
        "No production retrieval order, budget Gate, requirement merge or Finalize/Replay "
        "semantics were modified by this experiment. The existing M3-B downstream result "
        "(13/15 tasks, 0/12 budget violations) therefore remains the current end-to-end "
        "recommendation result.",
        "",
        "**Interpretation limits.** This is a 10-query engineering fixture, not a "
        "statistically significant benchmark. Labels are machine-assisted and pending human "
        "review; no recall/MRR/nDCG is reported because the judged pool is only the union of "
        "the two top-5 lists, not the full corpus. Candidate scope is a heuristic product-type "
        "filter, not a new ranking model.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python -m agent_eval.retrieval --output-dir runs/retrieval_experiments",
        "```",
        "",
        "The complete per-query rows and machine-assisted pooled labels are in "
        "`per_query.jsonl` and `annotations.json`/`annotations.md`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the M3-C full-catalogue BM25 vs category-scoped BM25 experiment."
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "agent_eval/retrieval/queries_v1.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    try:
        query_set = load_queries(args.queries)
    except Exception as exc:  # noqa: BLE001
        print(f"query set validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    config = ExperimentConfig(
        queries_path=args.queries,
        output_dir=args.output_dir,
        run_id=args.run_id,
        top_k=args.top_k,
        repeats=args.repeats,
    )
    started = time.perf_counter()
    try:
        outcome = RetrievalExperiment(query_set, config).run()
    except Exception as exc:  # noqa: BLE001
        print(f"retrieval experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = outcome["report"]
    print(f"report: {outcome['output_dir'] / 'report.md'}")
    print(json.dumps({
        "queries": report["metadata"]["query_count"],
        "baseline_precision_at_5": report["metrics"]["baseline"]["precision_at_5"],
        "candidate_precision_at_5": report["metrics"]["candidate"]["precision_at_5"],
        "delta": report["metrics"]["delta_precision_at_5"],
        "improved": report["metrics"]["improved_queries"],
        "degraded": report["metrics"]["degraded_queries"],
        "verdict": report["conclusion"]["verdict"],
    }, indent=2, ensure_ascii=False))
    print(f"wall_s={time.perf_counter() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
