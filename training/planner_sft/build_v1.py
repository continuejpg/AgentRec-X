"""Build Planner Action SFT v1 from real successful Qwen traces plus v0 rule data.

The label boundary is deliberately narrow:

* Qwen controls the Planner Action only (ASK/RETRIEVE/RERANK/FINALIZE).
* Qwen does not select the product item; the trusted Decision Gate selects after FINALIZE.
* Real-trace labels are existing Qwen actions from tasks whose end-to-end Agent outcome was
  successful.  They are *verified legal*, not claimed to be the unique optimal action.
* The failed task is excluded rather than relabelled.
* Eval100 tasks are held out entirely; no test task appears in train/val.

v0 synthetic rule-derived rows are reused unchanged.  Their user requests/templates are
independent of the real P14 task IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import ElectronicsPlannerExecutor
from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerObservation,
    allowed_planner_actions,
    run_planner,
)
from training.agent_sft.planner_prompt import planner_messages

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "runs/p20_agent_eval/task_suite.json"
TRACE = REPO / "runs/advanced_retrieval_v2/task_arm_original_bge.json"
V0 = REPO / "data/planner_sft_v0"
OUT = REPO / "data/planner_sft_v1"
CONTRACT_VERSION = "agentrecx.planner_action.v1"
VAL_TASK_IDS = ("p14e2_lite_0046", "p14e2_lite_0054", "p14e2_lite_0057", "p14e2_lite_0059")


class ReplayPlanner:
    """Replay successful Qwen actions while capturing the full runtime observation."""

    def __init__(self, trace: dict[str, Any], split: str) -> None:
        self.trace = trace
        self.split = split
        self.actions = list(trace.get("actions") or [])
        self.steps = list(trace.get("steps") or [])
        self.samples: list[dict[str, Any]] = []
        self.index = 0

    def propose(self, observation: PlannerObservation) -> ActionProposal:
        if self.index >= len(self.actions):
            raise RuntimeError("trace has fewer actions than replay steps")
        step = self.steps[self.index]
        action = str(step.get("action") or "")
        arguments = dict(step.get("action_arguments") or {})
        rationale = (self.trace.get("planner", {}).get("rationales") or [])
        rationale = str(rationale[self.index]) if self.index < len(rationale) else ""
        request = {
            "contract_version": CONTRACT_VERSION,
            "step": observation.step,
            "query": observation.query,
            "candidate_ids": list(observation.candidate_ids),
            "candidate_count": len(observation.candidate_ids),
            "candidate_evidence": list(observation.candidate_evidence),
            "last_retrieve_query": observation.last_retrieve_query,
            "retrieval_feedback": observation.retrieval_feedback,
            "role_target": observation.role_target,
            "eligibility": observation.eligibility,
            "source_policy": observation.source_policy,
            "source_feedback": observation.source_feedback,
            "state_version": observation.state_version,
            "m5_state": observation.m5_state,
            "allowed_actions": list(observation.allowed_actions),
            "last_tool_observation": observation.last_tool_observation,
            "remaining_steps": observation.remaining_steps,
            "needs_user_input": observation.needs_user_input,
            "reranked": observation.reranked,
        }
        target = {"action": action, "arguments": arguments, "rationale": rationale}
        self.samples.append({
            "schema_version": "agentrecx.planner_sft.v1",
            "sample_id": f"planner-sft-v1-real-{self.trace['task_id']}-{self.index + 1:02d}",
            "split": self.split,
            "source": {
                "request": "real_successful_qwen_trace",
                "catalogue": "amazon_reviews_2023:electronics_c4",
                "label_source": "actual_qwen_action_task_success_legal_verified",
                "qwen_output_used": True,
                "task_success": True,
            },
            "group_id": self.trace["task_id"],
            "trace_id": self.trace["task_id"],
            "user_request": self.trace["current_query"],
            "observation": request,
            "target": target,
            "label_basis": {
                "action_allowed_at_runtime": action in observation.allowed_actions,
                "task_success": True,
                "candidate_count": len(observation.candidate_ids),
                "item_selection_by_model": False,
                "tested_on_heldout_eval100": False,
            },
        })
        self.index += 1
        return ActionProposal(PlannerAction(action), arguments, rationale)

    def done(self) -> bool:
        return self.index >= len(self.actions)


def _chat_row(sample: dict[str, Any]) -> dict[str, Any]:
    messages = planner_messages(sample["observation"])
    target = sample["target"]
    messages.append({
        "role": "assistant",
        "content": json.dumps(
            {"action": target["action"], "arguments": target["arguments"], "rationale": target["rationale"]},
            ensure_ascii=False, separators=(",", ":"),
        ),
    })
    return {
        "schema_version": "agentrecx.planner_sft.chat.v1",
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "messages": messages,
        "target_action": target["action"],
        "label_source": sample["source"]["label_source"],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v0() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [json.loads(line) for line in (V0 / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    val = [json.loads(line) for line in (V0 / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "val"
    return train, val


def main() -> int:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    traces = {record["task_id"]: record for record in json.loads(TRACE.read_text(encoding="utf-8"))["tasks"]}
    dev_ids = sorted(task_id for task_id, task in tasks.items() if task["split"] == "dev")
    eval_ids = sorted(task_id for task_id, task in tasks.items() if task["split"] == "eval100")
    if set(VAL_TASK_IDS) - set(dev_ids):
        raise RuntimeError("validation task IDs are not all dev tasks")
    if set(VAL_TASK_IDS) & set(eval_ids):
        raise RuntimeError("validation leak from eval100")

    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    configure_catalogue(
        catalogue, mode="bge",
        index_path=REPO / "runs/p14d2/bge_index.npz",
        model_path=REPO / "models/bge-base-en-v1.5",
    )

    real_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task_id in dev_ids:
        record = traces.get(task_id)
        if record is None:
            skipped.append({"task_id": task_id, "reason": "missing_trace"})
            continue
        if not record.get("success"):
            skipped.append({"task_id": task_id, "reason": "task_failed_excluded"})
            continue
        task = tasks[task_id]
        preferences, _history = _memory_read(catalogue, task)
        executor = ElectronicsPlannerExecutor(
            catalogue=catalogue, current_query=task["current_query"],
            preferences=preferences, hard=list(task.get("hard") or []),
            corrective_rag=False,
        )
        observation = PlannerObservation(
            step=0, query=task["current_query"],
            m5_state={"phase": "product_retrieve", "next_query": task["current_query"],
                      "preferences": preferences},
        )
        split = "val" if task_id in VAL_TASK_IDS else "train"
        planner = ReplayPlanner(record, split)
        try:
            run_planner(planner=planner, executor=executor, initial_observation=observation, max_steps=4)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"task_id": task_id, "reason": f"replay_failed:{type(exc).__name__}: {exc}"})
            continue
        if not planner.done():
            skipped.append({"task_id": task_id, "reason": "replay_ended_before_all_actions"})
            continue
        real_rows.extend(planner.samples)

    v0_train, v0_val = _load_v0()
    train = [row for row in v0_train] + [row for row in real_rows if row["split"] == "train"]
    val = [row for row in v0_val] + [row for row in real_rows if row["split"] == "val"]
    if len(train) < 200 or len(train) > 500:
        raise RuntimeError(f"expected 200-500 train rows, got {len(train)}")
    if {row["trace_id"] for row in train + val} & set(eval_ids):
        raise RuntimeError("eval100 task leaked into train/val")

    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "train.jsonl"
    val_path = OUT / "val.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train), encoding="utf-8")
    val_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in val), encoding="utf-8")
    chat_train = OUT / "chat_train.jsonl"
    chat_val = OUT / "chat_val.jsonl"
    chat_train.write_text("".join(json.dumps(_chat_row(row), ensure_ascii=False, sort_keys=True) + "\n" for row in train), encoding="utf-8")
    chat_val.write_text("".join(json.dumps(_chat_row(row), ensure_ascii=False, sort_keys=True) + "\n" for row in val), encoding="utf-8")

    action_counts = Counter(row["target"]["action"] for row in train + val)
    real_count = len([row for row in train + val if row["source"]["request"] == "real_successful_qwen_trace"])
    synthetic_count = len(train + val) - real_count
    train_ids = sorted({row["trace_id"] for row in train if row["source"]["request"] == "real_successful_qwen_trace"})
    val_ids = sorted({row["trace_id"] for row in val if row["source"]["request"] == "real_successful_qwen_trace"})
    manifest = {
        "schema_version": "agentrecx.planner_sft.v1",
        "contract_version": CONTRACT_VERSION,
        "dataset_version": "planner-sft-v1",
        "sources": {
            "v0_rule_derived": "training/planner_sft/build_v0.py; no Qwen output",
            "real_qwen_traces": "runs/advanced_retrieval_v2/task_arm_original_bge.json; successful tasks only",
            "catalogue": "amazon_reviews_2023:electronics_c4",
        },
        "counts": {"train": len(train), "val": len(val), "real": real_count, "synthetic": synthetic_count},
        "actions": dict(sorted(action_counts.items())),
        "real_task_ids": {"train": train_ids, "val": val_ids},
        "heldout_eval100_task_ids": eval_ids,
        "excluded": skipped,
        "label_status": "provisional_automated_verified_not_human_gold",
        "label_policy": [
            "Qwen Action SFT only; product item_id selection is Gate-controlled and is not a training target",
            "real labels are actual Qwen actions from successful tasks, verified legal against the runtime observation",
            "failed real task excluded, not relabelled",
            "synthetic labels are rule/tool semantics and do not use Qwen output",
            "eval100 tasks are held out from train/val",
            "no P18 Reserved task used",
        ],
        "sha256": {
            "train": _sha(train_path), "val": _sha(val_path),
            "chat_train": _sha(chat_train), "chat_val": _sha(chat_val),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
