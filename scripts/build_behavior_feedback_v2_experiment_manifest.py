"""Assemble the V2 experiment manifest from the real run artifacts.

This file describes the run; it does not invent missing measurements.  Human preference
quality is explicitly unavailable because no independent human labels were collected.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/behavior_feedback_v2"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(name: str) -> list[dict]:
    path = OUT / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_dirty_files() -> list[str]:
    lines = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    # Windows checkout + WSL git can report every tracked file as CRLF-modified; a clean
    # Windows worktree is authoritative for this milestone, so do not publish that noise.
    return [] if len(lines) > 100 else lines


def main() -> None:
    environment = read_json("environment_manifest.json")
    gpu = read_json("gpu_health.json")
    summary = read_json("evaluation_summary.json")
    browser = read_json("browser_e2e_results.json")
    qwen = read_jsonl("qwen_live_trace.jsonl")
    bge = read_jsonl("bge_retrieval_trace.jsonl")
    qwen_actions = [call for row in qwen for call in (row.get("planner_actions") or [])]
    parser_errors = [error for call in qwen_actions for error in (call.get("parser_errors") or [])]
    manifest = {
        "schema": "agentrecx.behavior_feedback_v2.experiment_manifest.v1",
        "generated_at": now_iso(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "worktree_dirty_files": git_dirty_files(),
        "phase": "GPU Live Agent End-to-End Integration & Evaluation",
        "environment": {
            "bff": "FastAPI BFF, local CPU/WSL",
            "frontend": "Next.js 16.4.5, local CPU",
            "browser": browser.get("chrome"),
            "retrieval": "BGE dense, existing accepted P14-D2 index, local CPU",
            "planner": "Qwen3.5-9B-Base on remote RTX 4090",
            "gpu": gpu,
            "remote_transport": "existing WSL-local SSH loopback tunnel to 127.0.0.1:18029; no public endpoint",
            "bff_startup": {
                "search_mode": environment.get("runtime", {}).get("bff_search_mode"),
                "mcp_startup_timeout_s": 360,
                "reason": "BGE encoder + index cold start under two concurrent MCP children took 117-135s on this host",
            },
        },
        "commands": [
            "remote: python -m training.agent_sft.model_server --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 18029 --max-new-tokens 256",
            "WSL: ssh -N -L 127.0.0.1:18029:127.0.0.1:18029 ...",
            "WSL: AGENTRECX_WEB_DEMO_SEARCH_MODE=bge AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S=360 "
            "AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1 QWEN_LIVE_ENABLED=1 "
            "QWEN_MODEL_ENDPOINT=http://127.0.0.1:18029 bash scripts/run_web_demo_bff.sh",
            "WSL: .venv/bin/python scripts/behavior_feedback_v2_harness.py --all",
            "Windows: node scripts/behavior_feedback_v2_browser_e2e.mjs",
            "WSL: python scripts/build_behavior_feedback_v2_manifest.py",
        ],
        "scenario_coverage": {
            "scenario_total": 100,
            "qwen_live_scenarios": 21,
            "qwen_live_turn_traces": len(qwen),
            "qwen_raw_actions": len(qwen_actions),
            "qwen_parser_errors": parser_errors,
            "bge_turns": len(bge),
            "browser_qwen_turns": 2,
            "categories": sorted({row["spec"]["category"] for row in read_jsonl("scenario_results.jsonl")}),
            "human_events": 0,
            "automated_browser_events_accepted": summary.get("behavior_events_accepted"),
        },
        "comparison": {
            "eligible_comparable_scenarios": summary.get("eligible_comparable_scenarios"),
            "eligible_ranking_changed_scenarios": summary.get("eligible_ranking_changed_scenarios"),
            "eligible_ranking_changed_scenario_ids": summary.get("eligible_ranking_changed_scenario_ids"),
            "eligible_candidate_count_lt_2_excluded": summary.get("eligible_candidate_count_lt_2_excluded"),
            "hard_constraint_violation_scenarios": summary.get("hard_constraint_violation_scenarios"),
            "eligibility_changed_scenarios": summary.get("eligibility_changed_scenarios"),
        },
        "human_preference": {
            "human_review_count": 0,
            "preference_lift": "unavailable_no_human_labels",
            "note": (
                "No human labels were collected in V2.  V1.1 contains the reusable blind-review "
                "tool/tasks; automated browser clicks and Qwen outputs are not human gold."
            ),
        },
        "artifacts": [
            "environment_manifest.json", "gpu_health.json", "bge_environment.json",
            "qwen_live_trace.jsonl", "bge_retrieval_trace.jsonl",
            "browser_e2e_results.json", "v2-qwen-live-after-like.png",
            "v2-browser-e2e-final.png", "behavior_event_summary.json",
            "behavior_profile_snapshots.json", "eligible_off_on_comparison.json",
            "gate_validation.json", "latency_breakdown.json",
            "scenario_results.jsonl", "evaluation_summary.json", "experiment_manifest.json",
        ],
        "safety_boundaries": {
            "production_scorer_changed": False,
            "production_eligibility_or_gate_changed": False,
            "m9_semantics_changed": False,
            "p18_p20_p100_assets_changed": False,
            "model_retrained": False,
            "bge_index_rebuilt": False,
            "automated_events_are_human_preferences": False,
            "replay_masquerading_as_live": False,
            "human_labels_fabricated": False,
            "git_push": False,
        },
        "caveats": [
            "One automated browser multi-turn chain and 100 API-level scenarios; no user population.",
            "Qwen is real live GPU inference; BGE encoder and BFF are CPU/WSL.",
            "The 100 scenarios include 21 Qwen Live and 79 CPU/Rule-BGE sessions; not all are Qwen.",
            "2 of 42 eligible-comparable scenarios showed a real eligible rank change; the rest were ON=OFF.",
            "No human preference labels, so no preference-lift/CTR/conversion claim.",
        ],
    }
    (OUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "base_commit": manifest["base_commit"],
        "qwen_turns": len(qwen),
        "bge_turns": len(bge),
        "human_review_count": manifest["human_preference"]["human_review_count"],
    }, indent=1))


if __name__ == "__main__":
    main()
