"""Minimal real Rule-Agent execution on a stratified P11 subset.

No Qwen server is required.  This checks that constructed tasks enter the existing Agent,
use the real user history, return catalogue identities, and can be scored against the
metadata-verified pool without confusing retrieval failure with a correct final answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data/p11_benchmark/tasks.jsonl"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
INDEX = Path("/root/autodl-tmp/bge_build/bge_base_dense.npz")
MODEL = Path(
    "/root/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/"
    "snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
)
OUT = ROOT / "runs/p11/agent"

os.environ["AGENTRECX_CATALOG_SEARCH_MODE"] = "hybrid"
os.environ["AGENTRECX_DENSE_INDEX"] = str(INDEX)
os.environ["AGENTRECX_BGE_MODEL_PATH"] = str(MODEL)
os.environ["AGENTRECX_MAPPINGS_PATH"] = str(MAPPINGS)
os.environ["AGENTRECX_CATALOG_METADATA_PATH"] = str(PRODUCTS)
os.environ["AGENTRECX_SERVING_FUSION"] = "rrf3"
os.environ["AGENTRECX_STATE_MODE"] = "off"

from recommendation.api.app import ServiceSettings, build_engine
from recommendation.catalog import MetadataIndex
from recommendation.demo.agent_schemas import AgentRecommendRequest
from recommendation.demo.profiles import DemoProfile
from recommendation.demo.runtime import build_demo_runtime
from recommendation.demo.product_role import assess_candidate, infer_target


def load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]


def price(text: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return float(match.group(0)) if match else None


def choose_subset(tasks: list[dict], limit: int = 12) -> list[dict]:
    wanted = [
        "historical_personalization",
        "explicit_intent_with_history",
        "multi_turn_state_change",
        "clarification_required",
        "unsatisfiable_or_unknown",
    ]
    chosen = []
    for task_type in wanted:
        candidates = [task for task in tasks if task["task_type"] == task_type]
        # Deterministic, spread selection; do not filter by success.
        step = max(1, len(candidates) // max(1, limit // len(wanted)))
        for task in candidates[::step][: max(1, limit // len(wanted))]:
            chosen.append(task)
    return chosen[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    os.environ["AGENTRECX_DENSE_INDEX"] = str(args.index)
    os.environ["AGENTRECX_BGE_MODEL_PATH"] = str(args.model)
    os.environ["AGENTRECX_MAPPINGS_PATH"] = str(args.mappings)
    os.environ["AGENTRECX_CATALOG_METADATA_PATH"] = str(args.products)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    subset = choose_subset(tasks, args.limit)
    settings = ServiceSettings.from_env()
    engine, error = build_engine(settings)
    if engine is None:
        raise SystemExit(f"engine unavailable: {error}")
    metadata = MetadataIndex.load(args.products)
    profiles = {}
    for task in subset:
        profile_id = f"p11-{task['user_id']}"
        profiles[profile_id] = DemoProfile(
            profile_id=profile_id,
            display_name=profile_id,
            trusted_user_history=tuple(task["visible_user_history"]),
            source_user_int_id=0,
            source_length=len(task["visible_user_history"]) + 1,
        )
    runtime = build_demo_runtime(
        settings=settings,
        engine=engine,
        profiles=profiles,
        memory_db=args.out / "memory.sqlite3",
    )
    service = runtime.agent_service()
    results = []
    for index, task in enumerate(subset, start=1):
        solution = runtime.sessions.create(f"p11-{task['user_id']}")
        payload = AgentRecommendRequest(
            message=task["conversation_turns"][0],
            history=list(task["visible_user_history"]),
            k=3,
            sources=("history", "catalog_search"),
            user_key=solution.user_key,
        )
        try:
            messages = (
                list(task["conversation_turns"])
                if task["task_type"] == "multi_turn_state_change"
                else [task["conversation_turns"][0]]
            )
            response = None
            for turn_index, message in enumerate(messages, start=1):
                response = service.run_shopping_turn(
                    AgentRecommendRequest(
                        message=message,
                        history=list(task["visible_user_history"]),
                        k=3,
                        sources=("history", "catalog_search"),
                        user_key=solution.user_key,
                    ),
                    session_id=solution.user_key,
                    task_id=solution.session_id,
                    turn_id=f"p11:{task['task_id']}:{turn_index}",
                )
            assert response is not None
            final_ids = [item.parent_asin for item in response.recommendations]
            terminal = response.terminal.model_dump() if response.terminal else {}
            source_calls = response.observation.source_calls if response.observation else {}
            role = infer_target(task["conversation_turns"][0])
            price_max = task["constraints"].get("price_max") if isinstance(task["constraints"], dict) else None
            final_roles = {}
            final_prices = {}
            for identity in final_ids:
                record = metadata.records.get(identity)
                if record is None:
                    final_roles[identity] = "UNKNOWN_RECORD"
                    final_prices[identity] = None
                    continue
                final_roles[identity] = assess_candidate(identity, record, role).status.value
                final_prices[identity] = price(getattr(record, "price_text", ""))
            pool = {str(item["parent_asin"]) for item in task.get("verified_candidate_pool") or ()}
            expected = task.get("expected_action_when_verifiable")
            all_in_pool = bool(final_ids) and all(identity in pool for identity in final_ids)
            catalog_valid = bool(final_ids) and all(
                final_roles.get(identity) == "MATCH"
                and (
                    price_max is None
                    or (
                        final_prices.get(identity) is not None
                        and final_prices[identity] <= float(price_max)
                    )
                )
                for identity in final_ids
            )
            if pool and expected in {"RECOMMEND", "FINALIZE"}:
                strict_verified_success = bool(final_ids) and all_in_pool
                catalog_success = catalog_valid
                invalid_finalize = (not final_ids) or (not catalog_valid)
            elif pool:
                strict_verified_success = not final_ids
                catalog_success = not final_ids
                invalid_finalize = bool(final_ids)
            else:
                strict_verified_success = False
                catalog_success = False
                invalid_finalize = bool(final_ids)
            hard_violation = bool(final_ids) and any(
                final_roles.get(identity) != "MATCH"
                or (
                    price_max is not None
                    and (
                        final_prices.get(identity) is None
                        or final_prices[identity] > float(price_max)
                    )
                )
                for identity in final_ids
            )
            results.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "user_id": task["user_id"],
                    "query": task["conversation_turns"][0],
                    "final_ids": final_ids,
                    "grounded_count": response.observation.grounded_count if response.observation else None,
                    "source_calls": source_calls,
                    "terminal": terminal,
                    "expected_pool_size": len(pool),
                    "final_ids_in_pool": [identity for identity in final_ids if identity in pool],
                    "final_roles": final_roles,
                    "final_prices": final_prices,
                    "expected_action": expected,
                    "verified_success": strict_verified_success,
                    "catalog_validity_success": catalog_success,
                    "all_final_ids_in_bounded_pool": all_in_pool,
                    "invalid_finalize": invalid_finalize,
                    "hard_constraint_violation": hard_violation,
                    "verification_note": (
                        "success requires final IDs inside the metadata-verified pool; "
                        "an empty/unknown task must not be finalized"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - executable benchmark must retain failure
            results.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "user_id": task["user_id"],
                    "query": task["conversation_turns"][0],
                    "error": f"{type(exc).__name__}: {exc}",
                    "verified_success": False,
                    "invalid_finalize": None,
                    "hard_constraint_violation": None,
                }
            )
        print(f"{index}/{len(subset)} {task['task_id']}", flush=True)
    summary = {
        "schema_version": "agentrecx.p11.agent_execution.v1",
        "tasks": len(results),
        "task_type_counts": {},
        "verified_success": sum(1 for row in results if row.get("verified_success")),
        "catalog_validity_success": sum(1 for row in results if row.get("catalog_validity_success")),
        "invalid_finalize": sum(1 for row in results if row.get("invalid_finalize")),
        "hard_constraint_violation": sum(1 for row in results if row.get("hard_constraint_violation")),
        "errors": sum(1 for row in results if "error" in row),
        "note": "Rule Agent, real hybrid catalogue + history sources, no Qwen planner",
    }
    for row in results:
        summary["task_type_counts"][row["task_type"]] = summary["task_type_counts"].get(row["task_type"], 0) + 1
    (args.out / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
