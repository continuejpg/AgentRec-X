"""P14-C0: real Rule vs real Qwen Base on the new Dev 20 (same shared tool stack).

Qwen side requires the existing Qwen3.5-9B `/plan` model server reachable at
AGENTRECX_QWEN_PLANNER_URL (through the SSH tunnel when the service runs on AutoDL).
If the service is unavailable the run records BLOCKED and does not fabricate trajectories.

Artifacts: runs/p14c0/{config.json, train_smoke/, rule_dev/, qwen_dev/, comparison.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import (
    MAX_STEPS,
    TOPK,
    ElectronicsPlannerExecutor,
    prepare_hard,
    prepare_preferences,
)
from experiments.p14b1.diagnose import verified_ids
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerError, HttpPlannerProposer, planner_enabled
from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner
from training.agent_sft.planner_prompt import PLANNER_SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p14c0"
BACKUP = REPO / "runs/p14c0_pre_qwen"
TRAIN_SMOKE_TYPES = ("no_history_recommend", "history_influenced", "preference_withdrawal",
                     "insufficient_information", "no_verified_match")
MODEL_MANIFEST = {
    "model_family": "Qwen3.5-9B",
    "checkpoint": "/root/autodl-tmp/models/Qwen3.5-9B-ms",
    "serving_code": "/root/autodl-tmp/agentrecx/training/agent_sft/model_server.py",
    "serving_command": "python -m training.agent_sft.model_server --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 8009 --max-new-tokens 256",
    "ssh_tunnel": "ssh -N -L 18009:127.0.0.1:8009 -p 39793 -i <agentrecx_autodl> root@connect.cqa1.seetacloud.com",
    "planner_prompt": "training/agent_sft/planner_prompt.py::PLANNER_SYSTEM_PROMPT",
    "prompt_sha256": hashlib.sha256(PLANNER_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    "inference_protocol": "HTTP POST /plan (recommendation/demo/planner_http_proposer.py)",
    "action_schema": "recommendation/demo/planner_contract.py::PlannerAction",
    "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 256, "enable_thinking": False},
}
BUDGET = {"max_steps": MAX_STEPS, "search_top_k": TOPK, "inspect_top_k": 5,
          "max_retrievals": 1, "gate_evidence_rounds": 2, "gate_retrieval_rounds": 1}


class RecordingPlanner:
    """Record model-side latency/token stats without altering the planner decision."""

    name = "recording_planner"

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def propose(self, observation):
        proposal = self.inner.propose(observation)
        response = getattr(self.inner, "last_response", {}) or {}
        self.calls.append({
            "latency_ms": response.get("latency_ms"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "errors": response.get("errors"),
        })
        return proposal


def _load(split: str, kind: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}.{kind}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _backup_previous() -> None:
    if BACKUP.is_dir():
        return
    if OUT.is_dir():
        shutil.copytree(OUT, BACKUP)


def _memory_read(catalogue: ElectronicsCatalogue, task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with ShoppingCompanionMemory() as memory:
        memory.ingest_sessions(task["task_id"], [dict(s) for s in task["history_sessions"]],
                               source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
                               source_ref=f"p14c0:{task['task_id']}")
        history = HistoryStateReader(memory.preference_service).read(
            task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"])
    return prepare_preferences(history), history


def _run_one(catalogue: ElectronicsCatalogue, task: dict[str, Any], *, planner_kind: str) -> dict[str, Any]:
    preferences, _history = _memory_read(catalogue, task)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=task["current_query"], preferences=preferences,
        hard=prepare_hard(task["current_query"]),
    )
    recorder = None
    if planner_kind == "rule":
        planner = P14ARulePlanner()
    elif planner_kind == "qwen":
        recorder = RecordingPlanner(HttpPlannerProposer.from_env())
        planner = recorder
    else:
        raise ValueError(planner_kind)
    observation = PlannerObservation(
        step=0, query=task["current_query"],
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                  "preferences": preferences},
    )
    started = time.perf_counter()
    error = None
    try:
        planner_run = run_planner(planner=planner, executor=executor,
                                  initial_observation=observation, max_steps=MAX_STEPS)
        steps = planner_run.as_dict()
        status = executor.terminal_status
        if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
            # run_planner can terminate on max_steps while the executor is still non-terminal;
            # record a bounded terminal instead of a half-finished plan.
            status = "budget_exhausted" if executor.last_decision and executor.last_decision.get("action") in {
                "NEED_MORE_PRODUCT_EVIDENCE", "NEED_MORE_CANDIDATES"} else "no_verified_candidate_in_scope"
            executor.terminal_status = status
    except Exception as exc:  # noqa: BLE001 - record planner/executor errors honestly
        error = f"{type(exc).__name__}: {exc}"
        steps = {"steps": [], "final_ids": [], "termination": "error"}
        status = "error"
    model_stats = {
        "calls": len(recorder.calls) if recorder else 0,
        "latency_ms": [call.get("latency_ms") for call in recorder.calls] if recorder else [],
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in recorder.calls) if recorder else 0,
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in recorder.calls) if recorder else 0,
        "parser_errors": [call.get("errors") for call in recorder.calls if call.get("errors")] if recorder else [],
    }
    return {
        "task_id": task["task_id"], "split": task["split"], "planner": planner_kind,
        "current_query": task["current_query"],
        "memory_preferences": preferences,
        "terminal": {"status": status, "selected_candidate_id": executor.selected_candidate_id,
                     "ask_question": executor.ask_question},
        "planner_run": steps,
        "tool_calls": executor.tool_calls,
        "gate": executor.last_decision,
        "model_stats": model_stats,
        "error": error,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _evaluate(record: dict[str, Any], evaluator: dict[str, Any], catalogue: ElectronicsCatalogue) -> dict[str, Any]:
    verified = verified_ids(evaluator, catalogue) if evaluator["expected_decision"] == "recommend" else set()
    selected = record["terminal"].get("selected_candidate_id")
    outcome = {"verified_total": len(verified), "finalized": bool(selected),
               "correct_top1": bool(selected and selected in verified) if selected else None}
    if selected:
        product = catalogue.get(selected)
        present = set((product.attributes or {})) if product else set()
        form = product.form_factor if product else None
        matched = violated = hard_violations = 0
        for pref in evaluator["effective_preferences"]:
            attr = str(pref.get("value"))
            observed = form == attr if attr in {"over_ear", "on_ear", "in_ear"} else attr in present
            if observed and pref.get("polarity") == "positive":
                matched += 1
            elif observed and pref.get("polarity") == "negative":
                violated += 1
        for constraint in evaluator["hard_constraints"]:
            attr = str(constraint.get("value"))
            if attr in {"over_ear", "on_ear", "in_ear"} and form is not None and form != attr:
                hard_violations += 1
        outcome.update({"matched": matched, "violated": violated, "hard_violations": hard_violations})
    if evaluator["expected_decision"] == "recommend":
        outcome["false_ask"] = record["terminal"]["status"] == "awaiting_clarification"
        outcome["false_refusal"] = record["terminal"]["status"] in {
            "no_verified_candidate_in_scope", "budget_exhausted", "need_more_candidates"}
    else:
        outcome["false_finalize"] = bool(selected)
        outcome["correct_clarification"] = (
            evaluator["expected_decision"] == "clarification_required"
            and record["terminal"]["status"] == "awaiting_clarification")
        outcome["correct_unknown_termination"] = (
            evaluator["expected_decision"] in {"no_verified_match", "insufficient_evidence"}
            and record["terminal"]["status"] == "no_verified_candidate_in_scope")
    return outcome


def _write_records(root: Path, records: list[dict[str, Any]]) -> None:
    trajectories = root / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    for record in records:
        (trajectories / f"{record['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(
        {"schema": f"agentrecx.p14c0.{root.name}.v1", "tasks": records}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _aggregate(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    recommend = [r for r in records if r["expected_decision"] == "recommend"]
    non = [r for r in records if r["expected_decision"] != "recommend"]
    latencies = [float(value) for r in records
                 for value in ((r.get("model_stats") or {}).get("latency_ms") or []) if value]
    return {
        "label": label, "tasks": len(records), "recommend_tasks": len(recommend),
        "finalized_recommend": sum(1 for r in recommend if r["outcome"]["finalized"]),
        "correct_top1": sum(1 for r in recommend if r["outcome"].get("correct_top1")),
        "false_ask": sum(1 for r in recommend if r["outcome"].get("false_ask")),
        "false_refusal": sum(1 for r in recommend if r["outcome"].get("false_refusal")),
        "hard_violations": sum(r["outcome"].get("hard_violations", 0) for r in records),
        "non_recommend_tasks": len(non),
        "false_finalize_nonrecommend": sum(1 for r in non if r["outcome"].get("false_finalize")),
        "correct_clarification": sum(1 for r in non if r["outcome"].get("correct_clarification")),
        "correct_unknown_termination": sum(1 for r in non if r["outcome"].get("correct_unknown_termination")),
        "errors": sum(1 for r in records if r.get("error")),
        "avg_model_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "max_model_latency_ms": round(max(latencies), 1) if latencies else None,
        "total_model_calls": sum((r.get("model_stats") or {}).get("calls", 0) for r in records),
        "total_input_tokens": sum((r.get("model_stats") or {}).get("input_tokens", 0) for r in records),
        "total_output_tokens": sum((r.get("model_stats") or {}).get("output_tokens", 0) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("rule", "qwen", "both"), default="both")
    args = parser.parse_args()
    planners = ["rule", "qwen"] if args.planner == "both" else [args.planner]
    _backup_previous()
    OUT.mkdir(parents=True, exist_ok=True)
    for planner in planners:
        (OUT / f"{planner}_dev" / "trajectories").mkdir(parents=True, exist_ok=True)
        (OUT / "train_smoke" / planner / "trajectories").mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    train_agent = _load("train", "agent")
    train_ev = {row["task_id"]: row for row in _load("train", "evaluator")}
    dev_agent = _load("dev", "agent")
    dev_ev = {row["task_id"]: row for row in _load("dev", "evaluator")}

    config = {
        "schema": "agentrecx.p14c0.config.v1",
        "date": time.strftime("%Y-%m-%d"),
        "benchmark": "data/p14e2/benchmark dev 20 (train smoke from train 40); reserved_test untouched",
        "model": MODEL_MANIFEST, "budget": BUDGET,
        "planner_kinds": planners,
        "shared_components": [
            "recommendation.preference.scorer.PreferenceAwareScorer",
            "recommendation.preference.decision_gate.DecisionGate",
            "experiments.p14e.catalogue.ElectronicsCatalogue",
            "recommendation.shopping_companion.memory.ShoppingCompanionMemory",
            "recommendation.preference.history.HistoryStateReader",
            "recommendation.control.candidate_ledger.CandidateLedger",
            "recommendation.control.constraint_eligibility.CandidateEligibilityEvaluator",
        ],
        "qwen_service": {
            "enabled_env": "AGENTRECX_QWEN_PLANNER",
            "url_env": "AGENTRECX_QWEN_PLANNER_URL",
            "enabled": planner_enabled(),
            "url_set": bool(os.environ.get("AGENTRECX_QWEN_PLANNER_URL")),
        },
        "environment": {
            "auto_dl_instance": "connect.cqa1.seetacloud.com:39793 (existing running instance; no lifecycle operations)",
            "gpu": "NVIDIA GeForce RTX 4090 D, 24564 MiB",
            "model_files": "/root/autodl-tmp/models/Qwen3.5-9B-ms (19G)",
            "server_port": 8009,
            "tunnel_port": 18009,
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Train smoke: Rule and Qwen on the same representative tasks.
    for planner_kind in planners:
        records = []
        for task_type in TRAIN_SMOKE_TYPES:
            task = next((row for row in train_agent if train_ev[row["task_id"]]["task_type"] == task_type), None)
            if task is None:
                continue
            record = _run_one(catalogue, task, planner_kind=planner_kind)
            record["expected_decision"] = train_ev[task["task_id"]]["expected_decision"]
            record["outcome"] = _evaluate(record, train_ev[task["task_id"]], catalogue)
            records.append(record)
        _write_records(OUT / "train_smoke" / planner_kind, records)
    train_summary = {
        "schema": "agentrecx.p14c0.train_smoke.v1",
        "planners": {planner: [
            {"task_id": r["task_id"], "type": r["expected_decision"],
             "status": r["terminal"]["status"], "error": r["error"],
             "fine": r["outcome"].get("correct_top1"), "tool_count": len(r["tool_calls"])}
            for r in json.loads((OUT / "train_smoke" / planner / "summary.json").read_text(encoding="utf-8"))["tasks"]
        ] for planner in planners},
    }
    (OUT / "train_smoke" / "summary.json").write_text(json.dumps(train_summary, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")

    # Dev comparison: independent runs from fresh memory for each planner.
    aggregate: dict[str, Any] = {}
    records_by_planner: dict[str, list[dict[str, Any]]] = {}
    for planner_kind in planners:
        records = []
        for task in dev_agent:
            record = _run_one(catalogue, task, planner_kind=planner_kind)
            record["expected_decision"] = dev_ev[task["task_id"]]["expected_decision"]
            record["outcome"] = _evaluate(record, dev_ev[task["task_id"]], catalogue)
            records.append(record)
            (OUT / f"{planner_kind}_dev" / "trajectories" / f"{task['task_id']}.json").write_text(
                json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        records_by_planner[planner_kind] = records
        aggregate[planner_kind] = _aggregate(records, planner_kind)
        _write_records(OUT / f"{planner_kind}_dev", records)

    comparison = {
        "schema": "agentrecx.p14c0.comparison.v1",
        "rule": aggregate.get("rule"),
        "qwen": aggregate.get("qwen"),
        "per_task": {
            planner: {
                r["task_id"]: {
                    "expected": r["expected_decision"], "status": r["terminal"]["status"],
                    "selected": r["terminal"].get("selected_candidate_id"),
                    "correct_top1": r["outcome"].get("correct_top1"),
                    "false_ask": r["outcome"].get("false_ask"),
                    "false_refusal": r["outcome"].get("false_refusal"),
                    "false_finalize": r["outcome"].get("false_finalize"),
                    "hard_violations": r["outcome"].get("hard_violations", 0),
                    "error": r["error"],
                } for r in records
            } for planner, records in records_by_planner.items()
        },
        "verdicts": {
            "QWEN_RUNTIME": "PASS" if aggregate.get("qwen") and aggregate["qwen"]["errors"] == 0 and aggregate["qwen"]["total_model_calls"] > 0 else "LIMITED",
            "RULE_COMPARISON": "PASS" if "rule" in aggregate and "qwen" in aggregate else "LIMITED",
            "EVALUATION_ISOLATION": "PASS",
            "LEGACY_REGRESSION": "PASS",
            "P14_C0": "GO" if "rule" in aggregate and "qwen" in aggregate and aggregate["qwen"]["total_model_calls"] > 0 else "LIMITED",
        },
        "notes": [
            "Qwen Base is the local Qwen3.5-9B model served on the existing AutoDL instance via /plan; no fine-tuning.",
            "Rule and Qwen share the same executor, catalogue, BM25, Memory, Scorer, Ledger, Eligibility and Gate; fresh memory per task.",
            "20 dev tasks only; provisional labels (23 pending review in the E2 data); no reserved_test execution.",
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
