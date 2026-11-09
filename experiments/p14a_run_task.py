"""P14-A end-to-end CLI: run one real Shopping Companion train task.

Usage (from the repository root):

    PYTHONPATH=. .venv/bin/python experiments/p14a_run_task.py \
        --catalogue data/p14a/catalogue/docs.jsonl \
        --catalogue-manifest data/p14a/catalogue/manifest.json \
        --tasks data/p14a/dataset/train_tasks.jsonl \
        --eval data/p14a/dataset/train_eval.jsonl \
        --task-index 0 --mode real_limited_catalogue \
        --out-dir runs/p14a/trajectories

The CLI never reads ground truth before the trajectory is complete: the eval file is
only loaded after the run, for the post-run evaluator.  The dataset task file is the
agent-visible file produced by ``experiments/p14a_prepare_dataset.py`` (oracle block
already removed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from recommendation.shopping_companion.catalogue import ShoppingCompanionCatalogue
from recommendation.shopping_companion.evaluator import evaluate_trajectory
from recommendation.shopping_companion.product_info import ShoppingCompanionProductInfoTool
from recommendation.shopping_companion.runtime import (
    EXECUTION_MODES,
    BlockedMemoryTool,
    RunConfig,
    ShoppingCompanionRuntime,
    TaskInput,
)
from recommendation.shopping_companion.search import ShoppingCompanionSearchTool

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_tasks(path: Path) -> list[TaskInput]:
    rows = _read_jsonl(path)
    tasks = [TaskInput.from_mapping(row) for row in rows]
    tasks = [task for task in tasks if task.task_id and task.question]
    # deterministic: sorted by task id, so --task-index is reproducible
    return sorted(tasks, key=lambda task: task.task_id)


def _load_memory():
    """Use the memory workstream's adapter when present; otherwise report blocked.

    The released benchmark has no conversation-history corpus; ``BlockedMemoryTool``
    is the honest representation of that state and is never a fabricated history.
    """
    try:
        from recommendation.shopping_companion.memory import ShoppingCompanionMemory  # type: ignore

        return ShoppingCompanionMemory()
    except Exception:
        return BlockedMemoryTool()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=REPO_ROOT / "data/p14a/catalogue/docs.jsonl")
    parser.add_argument(
        "--catalogue-manifest", type=Path, default=REPO_ROOT / "data/p14a/catalogue/manifest.json"
    )
    parser.add_argument("--tasks", type=Path, default=REPO_ROOT / "data/p14a/dataset/train_tasks.jsonl")
    parser.add_argument("--eval", type=Path, default=REPO_ROOT / "data/p14a/dataset/train_eval.jsonl")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--mode", choices=EXECUTION_MODES, default="real_limited_catalogue")
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--inspect-top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/p14a/trajectories")
    parser.add_argument(
        "--diagnostic-stage2-without-preferences",
        action="store_true",
        help=(
            "explicitly run stage_2 as a labelled diagnostic continuation even though stage_1 "
            "is awaiting user clarification (never counted as a completed task)"
        ),
    )
    parser.add_argument("--no-eval", action="store_true", help="skip the post-run evaluator")
    args = parser.parse_args(argv)

    if not args.catalogue.is_file():
        print(f"catalogue docs not found: {args.catalogue}", file=sys.stderr)
        return 2
    if not args.tasks.is_file():
        print(f"train tasks not found: {args.tasks}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    catalogue = ShoppingCompanionCatalogue.load(
        args.catalogue,
        manifest_path=args.catalogue_manifest if args.catalogue_manifest.is_file() else None,
    )
    load_seconds = time.perf_counter() - started
    print(
        f"catalogue: {catalogue.catalogue_size} products "
        f"(limited={catalogue.limited}, load={load_seconds:.2f}s, docs={args.catalogue})"
    )

    tasks = _load_tasks(args.tasks)
    if args.task_id is not None:
        selected = [task for task in tasks if task.task_id == args.task_id]
        if not selected:
            print(f"task id not found: {args.task_id}", file=sys.stderr)
            return 2
        task = selected[0]
    else:
        if not 0 <= args.task_index < len(tasks):
            print(f"task index out of range: {args.task_index} (n={len(tasks)})", file=sys.stderr)
            return 2
        task = tasks[args.task_index]
    print(f"task: id={task.task_id} type={task.question_type} question={task.question[:120]!r}")

    search_tool = ShoppingCompanionSearchTool(catalogue)
    info_tool = ShoppingCompanionProductInfoTool(catalogue)
    memory = _load_memory()
    config = RunConfig(
        mode=args.mode,
        search_limit=args.search_limit,
        inspect_top_k=args.inspect_top_k,
        catalogue_version=str((catalogue.manifest or {}).get("docs_sha256") or ""),
        stage2_diagnostic_without_preferences=bool(args.diagnostic_stage2_without_preferences),
    )
    runtime = ShoppingCompanionRuntime(
        catalogue=catalogue,
        search_tool=search_tool,
        info_tool=info_tool,
        memory=memory,
        config=config,
    )
    trajectory = runtime.run_task(task)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_{task.task_id}"
    if args.diagnostic_stage2_without_preferences:
        stem += "_diagnostic_stage2"
    trajectory_path = args.out_dir / f"{stem}.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
    print(f"trajectory: {trajectory_path}")

    if not args.no_eval:
        if args.eval.is_file():
            eval_rows = {row.get("task_id"): row for row in _read_jsonl(args.eval)}
            eval_row = eval_rows.get(task.task_id)
            if eval_row is None:
                print(f"eval row not found for {task.task_id}; skipping evaluator", file=sys.stderr)
            else:
                evaluation = evaluate_trajectory(trajectory, eval_row)
                eval_path = args.out_dir / f"{stem}.eval.json"
                eval_path.write_text(
                    json.dumps(evaluation, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
                )
                print(f"evaluation: {eval_path}")
                print(
                    "eval: terminal={} target_ids={} target_in_search={} target_in_rec={} top1={}".format(
                        evaluation["terminal"].get("status"),
                        evaluation["target_ids"],
                        evaluation["target_in_search_hits"],
                        evaluation["target_in_recommendation"],
                        evaluation["recommendation_top1"],
                    )
                )
        else:
            print(f"eval file not found: {args.eval}; skipping evaluator", file=sys.stderr)

    print(
        "summary: mode={} termination={} succeeded={} tool_calls={} ledger_entries={} recommendations={} total={:.2f}s".format(
            trajectory["mode"],
            trajectory["terminal"]["termination"],
            trajectory["terminal"]["succeeded"],
            trajectory["tool_call_count"],
            len((trajectory.get("ledger") or {}).get("entries") or ()),
            len(trajectory.get("recommendation") or ()),
            trajectory["timing"]["total_seconds"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
