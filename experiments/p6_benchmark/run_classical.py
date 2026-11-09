"""P6 classical full-catalogue evaluation runner (CPU only, read-only).

What this measures
------------------
The frozen P6 classical rows (``data/p6_benchmark_v0/classical_{dev,holdout}.jsonl``) are
scored by the accepted SASRec and Two-Tower inference engines through the adapters in
:mod:`experiments.p6_benchmark.full_ranking`.  The runner then hands each **full
catalogue score vector** to ``recommendation.evaluation.metrics.sorted_ranking``, which
owns seen-item masking, target re-admission and the canonical tie rule, and reports the
P6 metric definitions from :mod:`experiments.p6_benchmark.metrics`.

Two history variants are evaluated and reported separately:

``p6_shared_history``
    ``history = items[:-2]`` - exactly the P6 benchmark v0 artifact as delivered.
``eval_protocol_v1_test_history``
    ``history = items[:-2] + validation target`` - the repository-wide
    ``agentrecx.eval_protocol.v1`` test mode, in which the validation interaction is an
    earlier observed event.  This is the variant comparable with the formal metrics
    recorded in ``runs/sasrec_canonical_2026/run.json``.

Why the metrics are exact under a top-50 truncation
---------------------------------------------------
Every P6 classical cutoff is ``<= 50`` (Recall@10, Recall@50, NDCG@10, MRR@10,
CatalogueCoverage@10).  ``sorted_ranking`` produces the complete catalogue ordering; the
runner keeps the first 50 for metric input.  A target ranked 51 or lower contributes
exactly zero to every P6 cutoff, so the truncated list and the full list give identical
metric values.  The truncated list is never treated as a retrieval head: it is read from
a full-catalogue ranking.

What is deliberately not evaluated
----------------------------------
``FusionInferenceEngine`` exposes only a fused top-k head, not a full-catalogue score
vector, so Serving V1 Fusion has no defined strict full-ranking here.  It is not
simulated by inventing a tail order; the report records it as not evaluated.  This
runner also does not train, does not touch Serving V1 and does not modify Gold/Evaluator
or the frozen P6 tasks.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.p6_benchmark.full_ranking import (  # noqa: E402
    CANONICAL_VARIANT,
    P6_VARIANT,
    VARIANTS,
    build_scorer,
    history_for_row,
    load_mapping_num_items,
    scalar_identity_checksum,
)
from experiments.p6_benchmark.metrics import (  # noqa: E402
    aggregate_classical,
    classical_case_metrics,
)
from recommendation.evaluation.metrics import sorted_ranking  # noqa: E402
from recommendation.training.checkpoint import sha256_file  # noqa: E402

P6_DIR = REPO / "data" / "p6_benchmark_v0"
DEFAULT_MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
DEFAULT_SASREC_CHECKPOINT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
DEFAULT_SASREC_MANIFEST = REPO / "runs" / "sasrec_canonical_2026" / "run.json"
DEFAULT_TWOTOWER_CHECKPOINT = REPO / "runs" / "twotower_public_2026" / "checkpoint.pt"
DEFAULT_TWOTOWER_CATEGORIES = REPO / "runs" / "twotower_public_2026" / "metadata_categories.json"
DEFAULT_OUT = REPO / "runs" / "p6_classical_eval"

#: Every P6 classical cutoff is at most 50; see the module docstring.
TOP_K_FOR_METRICS = 50

#: Which classical arms this runner supports.  Fusion is intentionally absent.
MODEL_NAMES: tuple[str, ...] = ("sasrec", "two_tower")

VARIANT_ALIASES: dict[str, str] = {
    "p6": P6_VARIANT,
    "canonical": CANONICAL_VARIANT,
    P6_VARIANT: P6_VARIANT,
    CANONICAL_VARIANT: CANONICAL_VARIANT,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_state() -> dict[str, Any]:
    """Best-effort provenance; never makes the evaluation fail if git is unavailable."""
    state: dict[str, Any] = {"head": None, "branch": None, "dirty_entry_count": None}
    try:
        state["head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        state["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        state["dirty_entry_count"] = len([line for line in status.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def _resolve_variants(names: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        try:
            variant = VARIANT_ALIASES[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown variant {name!r}; known aliases: {sorted(VARIANT_ALIASES)}"
            ) from exc
        if variant not in resolved:
            resolved.append(variant)
    if not resolved:
        raise ValueError("at least one history variant is required")
    return resolved


def _load_rows(split: str) -> list[dict[str, Any]]:
    path = P6_DIR / f"classical_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"P6 classical split not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if str(row.get("split")) != split:
            raise ValueError(
                f"row {row.get('case_id')!r} declares split {row.get('split')!r}, expected {split!r}"
            )
    if not rows:
        raise ValueError(f"P6 classical split {split!r} is empty")
    return rows


def _unique_dir(base: Path) -> Path:
    """Never overwrite an existing result directory."""
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free result directory next to {base}")


def evaluate_rows(
    *,
    scorer: Any,
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    num_items: int,
) -> dict[str, Any]:
    """Evaluate one scorer over one split under one history variant.

    The scorer sees history item ids only.  The evaluation target is used by this
    function (and by the ranking kernel) to measure the result; it is never passed to
    the model.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown history variant {variant!r}")
    cases: list[tuple[Sequence[str], str]] = []
    per_case: list[dict[str, Any]] = []
    target_in_model_history_cases = 0
    started = time.perf_counter()
    for row in rows:
        case_started = time.perf_counter()
        history = history_for_row(row, variant)
        target_item_id = int(row["test_item_id"])
        target_parent_asin = str(row["test_parent_asin"])
        if not history:
            raise ValueError(f"case {row.get('case_id')!r} produced an empty model history")
        if target_item_id in set(history):
            # Legitimate only in the repeated-interaction case: the validation interaction
            # equals the test target and happened before it.  Recorded, never hidden.
            target_in_model_history_cases += 1

        scores = scorer.score_catalog(history)
        if len(scores) != num_items + 1:
            raise ValueError(
                f"case {row.get('case_id')!r}: scorer returned {len(scores)} scores, "
                f"expected num_items + 1 = {num_items + 1}"
            )
        # One full-catalogue ranking: the evaluator kernel owns the seen mask, keeps a
        # repeated target eligible, and applies the canonical tie rule.
        ranking = sorted_ranking(
            scores,
            history=history,
            target_item_id=target_item_id,
            num_items=num_items,
        )
        try:
            exact_rank = ranking.index(target_item_id) + 1
        except ValueError as exc:  # pragma: no cover - the kernel guarantees eligibility
            raise RuntimeError(
                f"case {row.get('case_id')!r}: target {target_item_id} missing from the "
                "candidate ranking"
            ) from exc
        top50 = ranking[:TOP_K_FOR_METRICS]
        top10 = ranking[:10]
        if (exact_rank <= TOP_K_FOR_METRICS) != (target_item_id in top50):
            raise RuntimeError(
                f"case {row.get('case_id')!r}: target rank {exact_rank} disagrees with "
                "top-50 membership"
            )

        top50_parent_asins = [scorer.item_id_to_parent_asin(item_id) for item_id in top50]
        top10_parent_asins = [scorer.item_id_to_parent_asin(item_id) for item_id in top10]
        metrics = classical_case_metrics(top50_parent_asins, target_parent_asin)
        cases.append((top50_parent_asins, target_parent_asin))
        per_case.append(
            {
                "case_id": str(row["case_id"]),
                "user_id": str(row["user_id"]),
                "history_len": len(history),
                "exact_rank": exact_rank,
                "in_top10": exact_rank <= 10,
                "in_top50": exact_rank <= TOP_K_FOR_METRICS,
                "target_item_id": target_item_id,
                "target_parent_asin": target_parent_asin,
                "top10_parent_asins": top10_parent_asins,
                "metrics": metrics,
                "seconds": round(time.perf_counter() - case_started, 6),
            }
        )

    summary = aggregate_classical(cases, catalogue_size=num_items)
    ranks = [row["exact_rank"] for row in per_case]
    history_lengths = [row["history_len"] for row in per_case]
    diagnostics = {
        "exact_rank_mean": round(sum(ranks) / len(ranks), 3),
        "exact_rank_min": min(ranks),
        "exact_rank_max": max(ranks),
        "target_in_top10": sum(row["in_top10"] for row in per_case),
        "target_in_top50": sum(row["in_top50"] for row in per_case),
        "history_len_min": min(history_lengths),
        "history_len_mean": round(sum(history_lengths) / len(history_lengths), 3),
        "history_len_max": max(history_lengths),
        # Provenance only: the P6 variant never feeds the test target to the model in this
        # artifact.  The canonical variant does so only when the repeated interaction makes
        # validation_target == test_target (the protocol re-admits the target from the mask).
        "test_target_in_model_history_cases": target_in_model_history_cases,
    }
    return {
        "variant": variant,
        "variant_description": VARIANTS[variant],
        "cases": len(per_case),
        "metrics": summary,
        "diagnostics": diagnostics,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "per_case": per_case,
    }


def determinism_check(
    *,
    scorer: Any,
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    num_items: int,
    count: int,
) -> dict[str, Any]:
    """Score the first ``count`` cases twice and require identical full rankings."""
    checked = list(rows[:count])
    mismatches: list[dict[str, Any]] = []
    for row in checked:
        history = history_for_row(row, variant)
        target_item_id = int(row["test_item_id"])
        first = sorted_ranking(
            scorer.score_catalog(history),
            history=history,
            target_item_id=target_item_id,
            num_items=num_items,
        )
        second = sorted_ranking(
            scorer.score_catalog(history),
            history=history,
            target_item_id=target_item_id,
            num_items=num_items,
        )
        if first != second:
            mismatches.append({"case_id": row.get("case_id"), "identical": False})
    return {
        "variant": variant,
        "cases_checked": len(checked),
        "deterministic": not mismatches,
        "mismatches": mismatches,
    }


def _format_metric_line(model: str, variant: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"{model:<10s} {variant:<30s} "
        f"R@10={metrics['Recall@10']:.4f} R@50={metrics['Recall@50']:.4f} "
        f"NDCG@10={metrics['NDCG@10']:.4f} MRR@10={metrics['MRR@10']:.4f} "
        f"Cov@10={metrics['CatalogueCoverage@10']:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P6 classical full-catalogue evaluation (CPU, read-only)"
    )
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["p6", "canonical"],
        help="history variants: p6 / canonical (or full variant names)",
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases")
    parser.add_argument(
        "--check-determinism",
        type=int,
        default=0,
        metavar="N",
        help="score the first N cases twice per model/variant and require identical rankings",
    )
    parser.add_argument("--out", type=Path, default=None, help="result directory (never overwritten)")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--sasrec-checkpoint", type=Path, default=DEFAULT_SASREC_CHECKPOINT)
    parser.add_argument("--sasrec-manifest", type=Path, default=DEFAULT_SASREC_MANIFEST)
    parser.add_argument("--twotower-checkpoint", type=Path, default=DEFAULT_TWOTOWER_CHECKPOINT)
    parser.add_argument("--twotower-categories", type=Path, default=DEFAULT_TWOTOWER_CATEGORIES)
    args = parser.parse_args(argv)

    variants = _resolve_variants(list(args.variants))
    rows = _load_rows(args.split)
    rows_all = rows
    if args.limit:
        if args.limit < 1:
            parser.error("--limit must be >= 1 when supplied")
        rows = rows[: args.limit]

    num_items = load_mapping_num_items(args.mappings)
    leakage = json.loads((P6_DIR / "leakage_report.json").read_text(encoding="utf-8"))
    if int(leakage["num_items"]) != num_items:
        parser.error(
            f"num_items disagreement: mappings={num_items}, leakage_report={leakage['num_items']}"
        )

    expected_sasrec_sha = None
    if args.sasrec_manifest.exists():
        expected_sasrec_sha = (
            json.loads(args.sasrec_manifest.read_text(encoding="utf-8"))
            .get("checkpoints", {})
            .get("best", {})
            .get("sha256")
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = args.out or (DEFAULT_OUT / f"{args.split}_{timestamp}")
    out = _unique_dir(Path(base_out))
    out.mkdir(parents=True, exist_ok=False)

    split_path = P6_DIR / f"classical_{args.split}.jsonl"
    summary: dict[str, Any] = {
        "schema_version": "agentrecx.p6.classical_eval.v0",
        "created_utc": _utc_now(),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "split": args.split,
        "limit": args.limit or None,
        "results_dir": str(out),
        "data": {
            "split_file": str(split_path.relative_to(REPO)),
            "split_sha256": sha256_file(split_path),
            "mappings_file": str(args.mappings.relative_to(REPO))
            if args.mappings.is_relative_to(REPO)
            else str(args.mappings),
            "mappings_sha256": sha256_file(args.mappings),
            "leakage_report_sha256": sha256_file(P6_DIR / "leakage_report.json"),
            "protocol_sha256": sha256_file(P6_DIR / "protocol.json"),
            "cases_evaluated": len(rows),
            "cases_in_split": len(rows_all),
            "num_items": num_items,
            "case_identity_checksum": scalar_identity_checksum(rows),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _torch_version(),
            "device": args.device,
        },
        "git": _git_state(),
        "protocol": {
            "benchmark_schema": "agentrecx.p6.benchmark_protocol.v0",
            "candidate_set": (
                "full catalogue 1..num_items excluding seen history; target re-admitted"
            ),
            "ranking_kernel": "recommendation.evaluation.metrics.sorted_ranking",
            "tie_rule": "higher score first, then lower item id",
            "mask_rule": "excluded_seen(history, target) = set(history) - {target}",
            "metric_source": "experiments.p6_benchmark.metrics.aggregate_classical",
            "metrics": ["Recall@10", "Recall@50", "NDCG@10", "MRR@10", "CatalogueCoverage@10"],
            "top_k_for_metrics": TOP_K_FOR_METRICS,
            "top_k_note": (
                "full-catalogue ranking is computed; the first 50 ids are passed to the "
                "metric functions because every P6 cutoff is <= 50"
            ),
            "history_variants": {variant: VARIANTS[variant] for variant in variants},
            "fusion": (
                "NOT evaluated: FusionInferenceEngine exposes a head-fusion top-k, not a "
                "full-catalogue score vector; strict full-ranking Fusion is undefined"
            ),
            "holdout_policy": (
                "dev is evaluated first; holdout is evaluated once after the script is frozen "
                "and is never used for prompt/model selection or tuning"
            ),
        },
        "models": {},
        "determinism": {},
        "wall_seconds": 0.0,
    }
    run_started = time.perf_counter()

    try:
        for model in args.models:
            model_started = time.perf_counter()
            scorer = build_scorer(
                model,
                sasrec_checkpoint=args.sasrec_checkpoint,
                twotower_checkpoint=args.twotower_checkpoint,
                twotower_categories=args.twotower_categories,
                mappings_path=args.mappings,
                sasrec_manifest=args.sasrec_manifest if model == "sasrec" else None,
                expected_sasrec_sha256=expected_sasrec_sha if model == "sasrec" else None,
                device=args.device,
            )
            load_seconds = round(time.perf_counter() - model_started, 3)
            model_result: dict[str, Any] = {
                "scorer": {
                    "name": scorer.name,
                    "score_kind": scorer.score_kind,
                    "num_items": scorer.num_items,
                    "load_seconds": load_seconds,
                    "metadata": scorer.model_metadata(),
                },
                "variants": {},
            }
            if scorer.num_items != num_items:
                raise ValueError(
                    f"{model} engine num_items {scorer.num_items} != mappings {num_items}"
                )
            for variant in variants:
                result = evaluate_rows(
                    scorer=scorer, rows=rows, variant=variant, num_items=num_items
                )
                model_result["variants"][variant] = {
                    key: value for key, value in result.items() if key != "per_case"
                }
                with (out / "per_case.jsonl").open("a", encoding="utf-8") as handle:
                    for case in result["per_case"]:
                        handle.write(
                            json.dumps(
                                {
                                    "model": model,
                                    "split": args.split,
                                    "variant": variant,
                                    **case,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                if args.check_determinism:
                    determinism = determinism_check(
                        scorer=scorer,
                        rows=rows,
                        variant=variant,
                        num_items=num_items,
                        count=min(args.check_determinism, len(rows)),
                    )
                    summary["determinism"].setdefault(model, {})[variant] = determinism
                print(
                    _format_metric_line(model, variant, result["metrics"]),
                    flush=True,
                )
            summary["models"][model] = model_result
            gc.collect()

        summary["wall_seconds"] = round(time.perf_counter() - run_started, 3)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "results_dir": str(out),
            "split": args.split,
            "cases": len(rows),
            "wall_seconds": summary["wall_seconds"],
        }, indent=2), flush=True)
        return 0
    finally:
        summary["wall_seconds"] = round(time.perf_counter() - run_started, 3)
        # Always leave a summary, even on failure, so a partial run is never mistaken for
        # a complete one: a missing per-model entry is visible.
        if (out / "summary.json").exists() is False:
            (out / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # noqa: BLE001 - provenance only
        return None


if __name__ == "__main__":
    raise SystemExit(main())
