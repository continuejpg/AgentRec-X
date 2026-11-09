"""Build legal Train-only Planner-action SFT samples from Rule traces with BGE retrieval.

Only agent-visible PlannerObservation state is stored as input. The assistant target is the
validated PlannerAction that the executor actually accepted; no evaluator labels, expected
decisions, or target IDs are included.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor, prepare_hard, prepare_preferences
from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerAction, PlannerObservation, run_planner, validate_action
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner
from training.agent_sft.planner_prompt import PLANNER_SYSTEM_PROMPT, planner_messages

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p14c1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RecordingPlanner:
    def __init__(self) -> None:
        self.inner = P14ARulePlanner()
        self.records: list[dict[str, Any]] = []

    def propose(self, observation: PlannerObservation) -> PlannerAction:
        action = self.inner.propose(observation)
        self.records.append({"observation": asdict(observation), "action": action})
        return action


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    search_config = configure_catalogue(catalogue)
    if search_config["status"] != "active":
        raise RuntimeError(f"BGE backend not active: {search_config}")
    tasks = load_jsonl(DATA / "train.agent.jsonl")
    samples = []
    trace_rows = []
    for task in tasks:
        preferences, _history = _memory_read(catalogue, task)
        executor = ElectronicsPlannerExecutor(
            catalogue=catalogue, current_query=task["current_query"],
            preferences=preferences, hard=prepare_hard(task["current_query"]))
        observation = PlannerObservation(
            step=0, query=task["current_query"],
            m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                      "preferences": preferences},
        )
        recorder = RecordingPlanner()
        error = None
        try:
            run = run_planner(planner=recorder, executor=executor, initial_observation=observation,
                              max_steps=MAX_STEPS)
            terminal = executor.terminal_status
        except Exception as exc:  # preserve truly legal samples only; record error
            error = f"{type(exc).__name__}: {exc}"
            terminal = "error"
        trace_rows.append({"task_id": task["task_id"], "status": terminal, "error": error,
                           "samples": len(recorder.records)})
        for index, record in enumerate(recorder.records, start=1):
            obs = record["observation"]
            action = record["action"]
            request = {key: obs.get(key) for key in (
                "contract_version", "step", "query", "candidate_ids", "candidate_count",
                "candidate_evidence", "last_retrieve_query", "retrieval_feedback", "role_target",
                "eligibility", "source_policy", "source_feedback", "state_version", "m5_state",
                "allowed_actions", "last_tool_observation", "remaining_steps", "needs_user_input",
                "reranked") if key in obs}
            request["step"] = int(obs.get("step", 0))
            messages = planner_messages(request)
            completion = json.dumps({"action": action.action.value, "arguments": dict(action.arguments),
                                     "rationale": action.rationale}, ensure_ascii=False, sort_keys=True)
            # Validate the target against the recorded allowed actions/state.
            parsed = json.loads(completion)
            allowed = list(obs.get("allowed_actions") or ())
            if parsed["action"] not in allowed:
                continue
            samples.append({
                "sample_id": f"p14c1_{task['task_id']}_{index:02d}",
                "task_id": task["task_id"], "step": index,
                "messages": messages,
                "completion": completion,
                "allowed_actions": allowed,
                "observation": {key: obs.get(key) for key in (
                    "step", "candidate_ids", "candidate_count", "allowed_actions", "remaining_steps",
                    "needs_user_input", "reranked", "last_tool_observation", "m5_state",
                    "candidate_evidence", "role_target", "eligibility") if key in obs},
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sft_train.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    (OUT / "rule_train_traces.json").write_text(json.dumps(trace_rows, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    action_counts: dict[str, int] = {}
    for sample in samples:
        action = json.loads(sample["completion"])["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    manifest = {
        "schema": "agentrecx.p14c1.sft_data.v1",
        "split": "train only (40 tasks); no dev/test",
        "search_backend": search_config,
        "tasks": len(tasks), "tasks_with_samples": len({sample['task_id'] for sample in samples}),
        "samples": len(samples), "action_counts": action_counts,
        "evaluator_fields_in_input": False,
        "target_source": "validated Rule PlannerAction accepted by the executor; not evaluator labels",
        "gate_recovery_samples": sum(1 for sample in samples
                                     if sample["observation"].get("last_tool_observation", {}).get("gate")),
        "notes": ["train inputs contain only PlannerObservation state; no expected decision/target IDs",
                  "invalid/unaccepted actions are excluded", "no duplicated trajectory inflation"],
    }
    (OUT / "sft_data_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
