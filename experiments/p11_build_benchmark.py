"""P11 grounded shopping-task benchmark builder.

Evidence policy:
- Level A: requires review text.  The current local artifacts contain no review text,
  so this builder never emits Level A.
- Level B: repeated prior-history interaction with a product type/category.  This is
  behavioural affinity only, not a confirmed preference.
- Level C: constraints stated inside the current task.
- UNKNOWN: no prior-history support.

Gold is a bounded catalogue-metadata-verified candidate pool.  BM25/BGE rankings are not
used to define it.  The pool is explicitly not a full-catalogue oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recommendation.catalog import MetadataIndex
from recommendation.demo.product_role import assess_candidate, infer_target

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "runs/p8_d/multiuser/cases.jsonl"
SEQUENCES = ROOT / "data/processed/Sports_and_Outdoors_sequences.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
OUT = ROOT / "data/p11_benchmark"
POOL_LIMIT = 100
EXCLUDED_TERMS = {"", "sports & outdoors", "sports", "outdoors", "sport"}

QUERY_SPECS = [
    {
        "id": "water_filter",
        "query": "Recommend a water filter for camping under $50.",
        "terms": ("water filter", "water filtration", "filtration"),
        "price_max": 50.0,
    },
    {
        "id": "cookware",
        "query": "Recommend camping cookware for a weekend trip under $60.",
        "terms": ("cookware", "cook set", "camp kitchen", "pots", "pans", "griddles"),
        "price_max": 60.0,
    },
    {
        "id": "stove",
        "query": "Recommend a camping stove for backpacking under $100.",
        "terms": ("camping stove", "backpacking stove", "stove", "camp stove"),
        "price_max": 100.0,
    },
    {
        "id": "sleeping_pad",
        "query": "Recommend a sleeping pad for camping under $80.",
        "terms": ("sleeping pad", "sleeping mat", "camp bed", "air mattress"),
        "price_max": 80.0,
    },
    {
        "id": "filtered_bottle",
        "query": "Recommend a water bottle for hiking under $40.",
        "terms": ("water bottle", "filtered bottle", "hydration bottle"),
        "price_max": 40.0,
    },
]

PRICE_RE = re.compile(r"\d+(?:\.\d+)?")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_price(text: Any) -> float | None:
    match = PRICE_RE.search(str(text or ""))
    return float(match.group(0)) if match else None


def _record_text(record: Any) -> str:
    parts = [
        str(getattr(record, "title", "") or ""),
        " ".join(str(value) for value in (getattr(record, "categories", ()) or ())),
        " ".join(str(value) for value in (getattr(record, "features", ()) or ())),
        " ".join(str(value) for value in (getattr(record, "description", ()) or ())),
    ]
    return " ".join(parts).casefold()


def _candidate_pool(
    metadata: MetadataIndex,
    *,
    terms: tuple[str, ...],
    target_role: Any,
    price_max: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic catalogue-metadata-verified pool, independent of retrieval."""
    passes: list[dict[str, Any]] = []
    role_unknown = 0
    role_mismatch = 0
    price_missing = 0
    price_over = 0
    term_matches = 0
    for parent_asin, record in metadata.records.items():
        text = _record_text(record)
        if not any(term.casefold() in text for term in terms):
            continue
        term_matches += 1
        role_status = assess_candidate(parent_asin, record, target_role).status.value
        if role_status == "UNKNOWN":
            role_unknown += 1
            continue
        if role_status != "MATCH":
            role_mismatch += 1
            continue
        price = parse_price(getattr(record, "price_text", ""))
        if price_max is not None:
            if price is None:
                price_missing += 1
                continue
            if price > price_max:
                price_over += 1
                continue
        passes.append(
            {
                "parent_asin": parent_asin,
                "title": str(getattr(record, "title", "") or ""),
                "categories": list(getattr(record, "categories", ()) or ()),
                "price_text": str(getattr(record, "price_text", "") or "") or None,
                "price_value": price,
                "role_status": role_status,
                "verification_fields": ["title", "categories", "features", "description", "price_text"],
            }
        )
    passes.sort(key=lambda row: row["parent_asin"])
    truncated = len(passes) > POOL_LIMIT
    pool = passes[:POOL_LIMIT]
    stats = {
        "pool_limit": POOL_LIMIT,
        "pass_count": len(passes),
        "pool_count": len(pool),
        "truncated": truncated,
        "term_matches": term_matches,
        "role_unknown": role_unknown,
        "role_mismatch": role_mismatch,
        "price_missing": price_missing,
        "price_over_cap": price_over,
        "gold_scope": "bounded verified candidate pool, NOT full-catalogue oracle",
        "selection_method": "catalogue metadata predicate + role/price verification; no BM25/BGE rank",
    }
    return pool, stats


def _history_evidence(
    history: list[str],
    *,
    metadata: MetadataIndex,
    sequences_by_user: dict[str, dict[str, Any]],
    user_id: str,
    terms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int]:
    sequence = sequences_by_user.get(user_id, {})
    times = sequence.get("unix_ms", [])
    ratings = sequence.get("ratings", [])
    asins = sequence.get("parent_asins", [])
    index_by_asin = {str(asin): index for index, asin in enumerate(asins)}
    refs: list[dict[str, Any]] = []
    for parent_asin in history:
        record = metadata.records.get(parent_asin)
        if record is None:
            continue
        text = _record_text(record)
        if not any(term.casefold() in text for term in terms):
            continue
        position = index_by_asin.get(parent_asin)
        refs.append(
            {
                "parent_asin": parent_asin,
                "title": str(getattr(record, "title", "") or ""),
                "categories": list(getattr(record, "categories", ()) or ()),
                "rating": (ratings[position] if position is not None and position < len(ratings) else None),
                "timestamp_ms": (times[position] if position is not None and position < len(times) else None),
                "source": "train_history_only",
            }
        )
    return refs, len(refs)


def _history_categories(history: list[str], metadata: MetadataIndex) -> list[str]:
    values = []
    for parent_asin in history:
        record = metadata.records.get(parent_asin)
        if record is None:
            continue
        values.extend(str(value) for value in (getattr(record, "categories", ()) or ()))
    return values


def _task_common(
    *,
    task_id: str,
    user: dict[str, Any],
    user_evidence: list[dict[str, Any]],
    query: str,
    task_type: str,
    role: Any,
    price_max: float | None,
    pool: list[dict[str, Any]],
    pool_stats: dict[str, Any],
    preference_level: str,
    task_declared: dict[str, Any],
    adjudication: str,
    split: str = "dev",
) -> dict[str, Any]:
    state = {
        "goal_type": role.type,
        "role": role.role.value,
        "price_max": price_max,
        "preferences": [],
    }
    return {
        "schema_version": "agentrecx.p11.shopping_task.v1",
        "task_id": task_id,
        "user_id": user["user_id"],
        "source_dataset": "amazon_reviews_2023:sports_and_outdoors",
        "task_type": task_type,
        "query_source": "benchmark_template_over_real_history_and_catalog",
        "user_history_source": "train_history_only",
        "visibility_split": split,
        "conversation_turns": [query],
        "visible_user_history": list(user["history"]),
        "history_evidence_refs": user_evidence,
        "preference_evidence_level": preference_level,
        "preference_evidence_scope": (
            "Level B is repeated prior interaction / behavioural affinity only, not a "
            "confirmed stated preference; Level C is task-declared."
        ),
        "effective_state_per_turn": [state],
        "required_product_role": {"type": role.type, "role": role.role.value},
        "constraints": task_declared,
        "verified_candidate_pool": pool,
        "candidate_pool_stats": pool_stats,
        "expected_action_when_verifiable": "RECOMMEND" if pool else "ASK_OR_UNKNOWN",
        "adjudication_status": adjudication,
        "split": split,
        "provenance": {
            "source_cases": "runs/p8_d/multiuser/cases.jsonl",
            "source_sequences": str(SEQUENCES.relative_to(ROOT)),
            "source_products": str(PRODUCTS.relative_to(ROOT)),
            "source_mappings": str(MAPPINGS.relative_to(ROOT)),
            "history_scope": "train_history only; no validation/test target used",
            "candidate_pool_method": "catalogue metadata + role/price; no BM25/BGE rank",
            "gold_scope": "verified candidate pool, not complete catalogue oracle",
            "builder_version": "agentrecx.p11.benchmark_builder.v1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = MetadataIndex.load(PRODUCTS)
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequences_payload = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    sequences_by_user = {
        str(record["user_id"]): record for record in sequences_payload["sequences"]
    }
    # Query pools are deterministic and reusable across task instances.
    spec_data: dict[str, Any] = {}
    for spec in QUERY_SPECS:
        role = infer_target(spec["query"])
        if role.type is None:
            raise SystemExit(f"query spec {spec['id']} does not resolve a product role")
        pool, stats = _candidate_pool(
            metadata,
            terms=spec["terms"],
            target_role=role,
            price_max=spec["price_max"],
        )
        if not pool and spec["id"] != "unsatisfiable":
            raise SystemExit(f"query spec {spec['id']} has no verified candidates")
        spec_data[spec["id"]] = {
            **spec,
            "role": role,
            "pool": pool,
            "pool_stats": stats,
        }

    # Users with repeated matching history evidence (Level B) or no matching evidence.
    evidence_users: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_users: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        user = {"user_id": case["user_id"], "history": list(case["history"]), "case_id": case["case_id"]}
        for spec in QUERY_SPECS:
            refs, count = _history_evidence(
                user["history"],
                metadata=metadata,
                sequences_by_user=sequences_by_user,
                user_id=user["user_id"],
                terms=spec["terms"],
            )
            categories = [
                category
                for ref in refs
                for category in ref.get("categories", ())
                if str(category).casefold() not in EXCLUDED_TERMS
            ]
            profile = Counter(categories).most_common(1)[0][0] if categories else None
            item = {**user, "evidence": refs, "evidence_count": count, "history_profile": profile}
            if count >= 2:
                evidence_users[spec["id"]].append(item)
            elif count == 0:
                unknown_users[spec["id"]].append(item)

    tasks: list[dict[str, Any]] = []
    task_user_ids: set[str] = set()
    pair_counter = 0
    # D.1 same query, evidence vs no-evidence user and (where possible) two users
    # with different repeated-history profiles.
    for spec_id, spec in spec_data.items():
        evidence = sorted(evidence_users[spec_id], key=lambda item: item["user_id"])
        unknown = sorted(unknown_users[spec_id], key=lambda item: item["user_id"])
        # Evidence vs no-evidence pairs, one per evidence user up to the available unknown pool.
        for index, evidence_user in enumerate(evidence):
            if not unknown:
                break
            no_evidence_user = unknown[(pair_counter + index) % len(unknown)]
            pair_counter += 1
            pair_id = f"p11-pair-{pair_counter:03d}"
            for label, user in (("evidence", evidence_user), ("unknown", no_evidence_user)):
                level = "B" if label == "evidence" else "UNKNOWN"
                task = _task_common(
                    task_id=f"{pair_id}-{label}",
                    user=user,
                    user_evidence=user["evidence"],
                    query=spec["query"],
                    task_type="historical_personalization",
                    role=spec["role"],
                    price_max=spec["price_max"],
                    pool=spec["pool"],
                    pool_stats=spec["pool_stats"],
                    preference_level=level,
                    task_declared={
                        "goal_type": spec["role"].type,
                        "role": spec["role"].role.value,
                        "price_max": spec["price_max"],
                        "source": "Level C task-declared constraints",
                    },
                    adjudication=("needs_human_for_preference_claim" if level == "B"
                                  else "machine_verified_catalog_constraints"),
                )
                task["history_profile"] = evidence_user.get("history_profile") if label == "evidence" else None
                task["pair_id"] = pair_id
                task["pair_role"] = label
                tasks.append(task)
                task_user_ids.add(user["user_id"])
        # Evidence-vs-evidence pair when their repeated category profiles differ.
        paired_profiles: set[tuple[str, str]] = set()
        for left in evidence:
            for right in evidence:
                if left["user_id"] == right["user_id"]:
                    continue
                left_profile = left.get("history_profile")
                right_profile = right.get("history_profile")
                if not left_profile or not right_profile or left_profile == right_profile:
                    continue
                key = tuple(sorted((left_profile, right_profile)))
                if key in paired_profiles:
                    continue
                paired_profiles.add(key)
                pair_counter += 1
                pair_id = f"p11-pair-{pair_counter:03d}"
                for label, user, profile in (("evidence-a", left, left_profile),
                                             ("evidence-b", right, right_profile)):
                    task = _task_common(
                        task_id=f"{pair_id}-{label}",
                        user=user,
                        user_evidence=user["evidence"],
                        query=spec["query"],
                        task_type="historical_personalization",
                        role=spec["role"],
                        price_max=spec["price_max"],
                        pool=spec["pool"],
                        pool_stats=spec["pool_stats"],
                        preference_level="B",
                        task_declared={
                            "goal_type": spec["role"].type,
                            "role": spec["role"].role.value,
                            "price_max": spec["price_max"],
                            "source": "Level C task-declared constraints",
                        },
                        adjudication="needs_human_for_preference_claim",
                    )
                    task["history_profile"] = profile
                    task["pair_id"] = pair_id
                    task["pair_role"] = label
                    task["pair_contrast"] = "different_history_profiles"
                    tasks.append(task)
                    task_user_ids.add(user["user_id"])
                if len(paired_profiles) >= 2:
                    break
    # D.2 explicit intent + history: bounded evidence users, explicit hard constraints.
    for spec_id, spec in spec_data.items():
        evidence = sorted(evidence_users[spec_id], key=lambda item: item["user_id"])[:3]
        for index, user in enumerate(evidence):
            task = _task_common(
                task_id=f"p11-explicit-{spec_id}-{index:02d}",
                user=user,
                user_evidence=user["evidence"],
                query=spec["query"],
                task_type="explicit_intent_with_history",
                role=spec["role"],
                price_max=spec["price_max"],
                pool=spec["pool"],
                pool_stats=spec["pool_stats"],
                preference_level="B",
                task_declared={
                    "goal_type": spec["role"].type,
                    "role": spec["role"].role.value,
                    "price_max": spec["price_max"],
                    "hard_constraint": True,
                    "source": "Level C current request",
                },
                adjudication="needs_human_for_preference_claim",
            )
            task["history_evidence_refs"] = user["evidence"]
            tasks.append(task)
            task_user_ids.add(user["user_id"])

    # D.3 multi-turn state changes; every transition is an existing M5-compatible state.
    multiturn_specs = [
        ("p11-multiturn-budget", "water_filter", "Find me a water filter under $40.",
         [("Find me a water filter under $40.", 40.0),
          ("Actually, my budget is now $60.", 60.0),
          ("Forget about the budget.", None)]),
        ("p11-multiturn-goal-switch", "cookware", "I need a water filter under $50.",
         [("I need a water filter under $50.", 50.0),
          ("Actually, show me camping cookware under $60 instead.", 60.0)]),
        ("p11-multiturn-price-change", "sleeping_pad", "Show me a sleeping pad under $50.",
         [("Show me a sleeping pad under $50.", 50.0),
          ("I can go up to $80.", 80.0)]),
    ]
    for task_id, spec_id, first_message, turns in multiturn_specs:
        spec = spec_data[spec_id]
        evidence = sorted(evidence_users[spec_id], key=lambda item: item["user_id"])
        if not evidence:
            continue
        user = evidence[0]
        states = []
        for index, (message, price_max) in enumerate(turns):
            states.append(
                {
                    "turn": index + 1,
                    "message": message,
                    "effective_state": {
                        "goal_type": spec["role"].type,
                        "role": spec["role"].role.value,
                        "price_max": price_max,
                        "preferences": [],
                    },
                }
            )
        task = _task_common(
            task_id=task_id,
            user=user,
            user_evidence=user["evidence"],
            query=first_message,
            task_type="multi_turn_state_change",
            role=spec["role"],
            price_max=spec["price_max"],
            pool=spec["pool"],
            pool_stats=spec["pool_stats"],
            preference_level="B",
            task_declared={
                "hard_constraint": True,
                "price_max_final": spec["price_max"],
                "source": "Level C turn messages",
            },
            adjudication="needs_human_for_preference_claim",
        )
        task["conversation_turns"] = [message for message, _ in turns]
        task["effective_state_per_turn"] = states
        task["expected_action_when_verifiable"] = "FINALIZE"
        tasks.append(task)
        task_user_ids.add(user["user_id"])

    # D.4 clarification required: no verified answer because a required semantic
    # attribute/compatibility is not present in the trusted catalogue.
    clarify_specs = [
        ("p11-clarify-compatibility", "water_filter",
         "Find a water filter that fits my hydration bladder.", "compatibility"),
        ("p11-clarify-cartridge", "water_filter",
         "I need a replacement filter cartridge for my water filter.", "compatibility"),
        ("p11-clarify-lightweight", "cookware",
         "Show me the lightest camping cookware for backpacking.", "lightweight_attribute"),
        ("p11-clarify-saltwater", "water_filter",
         "Find a water filter that also works for saltwater.", "saltwater_compatibility"),
    ]
    for task_id, spec_id, query, unknown_reason in clarify_specs:
        spec = spec_data[spec_id]
        evidence = sorted(evidence_users[spec_id], key=lambda item: item["user_id"])
        user = evidence[0] if evidence else {"user_id": "p11-no-user", "history": [], "evidence": [], "case_id": None}
        task = _task_common(
            task_id=task_id,
            user={"user_id": user.get("user_id", "p11-no-user"), "history": user.get("history", [])},
            user_evidence=user.get("evidence", []),
            query=query,
            task_type="clarification_required",
            role=spec["role"],
            price_max=None,
            pool=[],
            pool_stats={
                "pool_count": 0,
                "gold_scope": "UNKNOWN: required semantic attribute/compatibility not in trusted metadata",
                "unverified_reason": unknown_reason,
            },
            preference_level="UNKNOWN",
            task_declared={"goal_type": spec["role"].type, "source": "Level C current request"},
            adjudication="needs_human",
        )
        task["effective_state_per_turn"] = [
            {"turn": 1, "message": query,
             "effective_state": {"goal_type": spec["role"].type,
                                 "role": spec["role"].role.value,
                                 "compatibility": "UNKNOWN"}}
        ]
        task["expected_action_when_verifiable"] = "ASK"
        tasks.append(task)

    # D.5 unsatisfiable / insufficient evidence.
    for task_id, spec_id, query, reason in [
        ("p11-unsat-price", "water_filter", "I need a water filter under $1.", "no_candidate_under_hard_price"),
        ("p11-unsat-role", "water_filter", "I need a water filter accessory under $1.", "no_verified_role_and_price_candidate"),
        ("p11-unknown-role", "water_filter", "Do you have something for my camping trip?", "product_role_missing"),
        ("p11-missing-attribute", "cookware", "Find camping cookware with a verified nonstick coating under $1.", "attribute_and_price_unknown"),
    ]:
        spec = spec_data[spec_id]
        user = {"user_id": "p11-no-user", "history": []}
        if spec_id != "cookware" or not tasks:
            evidence = sorted(evidence_users[spec_id], key=lambda item: item["user_id"])
            if evidence:
                user = {"user_id": evidence[0]["user_id"], "history": evidence[0]["history"]}
        task = _task_common(
            task_id=task_id,
            user=user,
            user_evidence=[],
            query=query,
            task_type="unsatisfiable_or_unknown",
            role=spec["role"],
            price_max=1.0 if "price" in reason or "unsat" in task_id else None,
            pool=[],
            pool_stats={
                "pool_count": 0,
                "gold_scope": "UNKNOWN: no verified candidate under the declared condition",
                "unverified_reason": reason,
            },
            preference_level="UNKNOWN",
            task_declared={"goal_type": spec["role"].type, "source": "Level C current request"},
            adjudication="needs_human",
        )
        task["effective_state_per_turn"] = [
            {"turn": 1, "message": query,
             "effective_state": {"goal_type": spec["role"].type,
                                 "role": spec["role"].role.value,
                                 "price_max": 1.0 if "price" in reason else None}}
        ]
        task["expected_action_when_verifiable"] = "UNKNOWN"
        tasks.append(task)

    # Provenance and quality counts.
    product_hash = sha256_file(PRODUCTS)
    sequence_hash = sha256_file(SEQUENCES)
    mapping_hash = sha256_file(MAPPINGS)
    level_counts = Counter(task["preference_evidence_level"] for task in tasks)
    type_counts = Counter(task["task_type"] for task in tasks)
    adjudication_counts = Counter(task["adjudication_status"] for task in tasks)
    pool_tasks = [task for task in tasks if task["verified_candidate_pool"]]
    unknown_tasks = [task for task in tasks if not task["verified_candidate_pool"]]
    manifest = {
        "schema_version": "agentrecx.p11.benchmark_manifest.v1",
        "created_utc": "2026-11-08",
        "task_count": len(tasks),
        "task_type_counts": dict(type_counts),
        "preference_evidence_level_counts": dict(level_counts),
        "adjudication_status_counts": dict(adjudication_counts),
        "tasks_with_verified_pool": len(pool_tasks),
        "tasks_without_verified_pool": len(unknown_tasks),
        "unique_users": len({task["user_id"] for task in tasks if task["user_id"] != "p11-no-user"}),
        "paired_same_query_tasks": len([
            task for task in tasks if task["task_type"] == "historical_personalization"
            and task.get("pair_id")
        ]),
        "paired_same_query_ids": len({
            task["pair_id"] for task in tasks if task.get("pair_id")
        }),
        "split": "dev",
        "reserved_eval_created": False,
        "holdout_used": False,
        "review_text_available": False,
        "level_a_available": False,
        "data_sources": {
            "cases": str(CASES.relative_to(ROOT)),
            "sequences": {"path": str(SEQUENCES.relative_to(ROOT)), "sha256": sequence_hash},
            "products": {"path": str(PRODUCTS.relative_to(ROOT)), "sha256": product_hash},
            "mappings": {"path": str(MAPPINGS.relative_to(ROOT)), "sha256": mapping_hash},
        },
        "candidate_pool_scope": (
            "bounded verified candidate pool from catalogue metadata + role/price; "
            "not a full-catalogue oracle"
        ),
        "benchmark_is_not_real_search_log": True,
        "known_limits": [
            "no review text in local artifacts; Level A evidence = 0",
            "Level B is repeated behaviour only, not stated preference",
            "candidate pools are bounded, not exhaustive",
            "no reserved evaluation split in this phase",
        ],
        "builder": "experiments/p11_build_benchmark.py",
    }
    schema = {
        "schema_version": "agentrecx.p11.shopping_task.v1",
        "required_fields": [
            "task_id", "user_id", "source_dataset", "task_type", "conversation_turns",
            "visible_user_history", "history_evidence_refs", "preference_evidence_level",
            "effective_state_per_turn", "required_product_role", "constraints",
            "verified_candidate_pool", "expected_action_when_verifiable",
            "adjudication_status", "split", "provenance",
        ],
        "field_descriptions": {
            "preference_evidence_level": "A=stated review evidence (not available here); B=repeated history behaviour; C=task-declared; UNKNOWN=no evidence",
            "history_evidence_refs": "train-history items only, with ASIN/title/category/rating/timestamp",
            "visible_user_history": "history visible to the Agent; no validation/test target",
            "verified_candidate_pool": "bounded catalogue-metadata/role/price verified positives; not a unique Gold",
            "adjudication_status": "machine_verified_catalog_constraints, needs_human_for_preference_claim, or needs_human",
            "expected_action_when_verifiable": "RECOMMEND/FINALIZE for candidate tasks; ASK or UNKNOWN for unverifiable tasks",
        },
    }
    (args.out / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "query_pools.json").write_text(
        json.dumps(
            {
                spec_id: {
                    "query": spec["query"],
                    "role": {"type": spec["role"].type, "role": spec["role"].role.value},
                    "pool_stats": spec["pool_stats"],
                    "pool": spec["pool"],
                }
                for spec_id, spec in spec_data.items()
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
