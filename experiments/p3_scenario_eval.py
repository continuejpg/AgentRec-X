"""Phase 3: small, deterministic shopping-scenario evaluation of the Agent path.

What it measures
----------------
Forty-eight scenarios built from ten real histories and the served catalogue's verifiable
fields (price, category, brand).  Each scenario is one Agent request through the existing
``DemoAgentService.run`` path (plan ``history + catalog_search``, ``k = 5``); no new agent
architecture, no LLM in the loop.

  hard price / category / brand / two-condition constraints
  four impossible or unverifiable constraint types
  soft preferences expressed in the message (ranking aim, never a guarantee)

Every returned candidate is verified with an **independent** predicate over the catalogue
record (not by re-reading the system's own eligibility view): price from ``price_text``,
brand from the metadata detail bag, category from ``categories``/``main_category``.  The
rendered text is checked separately: which product ids it names, whether those ids equal the
final recommendation ids, and whether each quoted catalogue fact matches the record.

    .venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval

Outputs (never written into an existing directory):
    config.json, cases.json, samples.jsonl, summary.json, SHA256SUMS.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config
from recommendation.control.grounded_reasoning import first_attribute, parse_price

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_mappings.json"
DEFAULT_OUT = REPO / "runs" / "p3_scenario_eval"

#: Candidate histories for the scenario matrix; the first eight with usable metadata are kept.
HISTORY_CANDIDATES: tuple[int, ...] = tuple(range(12000, 12040))
HISTORIES_NEEDED = 8
K = 5
HARD_MESSAGE = "I need some sports gear"
SOFT_MESSAGE = "I'd prefer {value} if possible; I need some sports gear"
BRAND_KEYS = ("Brand Name", "Brand", "Manufacturer")
ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")
ITEM_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(B[0-9A-Z]{9})\b")
FACT_LINE_RE = re.compile(r"^\s+(Main category|Category|Title|Store|Feature|Detail):\s+(.*)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# independent catalogue predicates
# --------------------------------------------------------------------------- #


def _price(record: Any) -> float | None:
    return parse_price(getattr(record, "price_text", None))


def _brand(record: Any) -> str | None:
    return first_attribute(record, BRAND_KEYS)


def _categories(record: Any) -> tuple[str, ...]:
    values = list(getattr(record, "categories", ()) or ())
    main = getattr(record, "main_category", None)
    if main:
        values.append(main)
    return tuple(str(value) for value in values)


def check_constraint(record: Any, kind: str, value: str) -> str:
    """``satisfied`` / ``violated`` / ``unknown`` from the catalogue record alone."""
    if record is None:
        return "unknown"
    if kind == "price_max":
        observed = _price(record)
        limit = parse_price(value)
        if observed is None or limit is None:
            return "unknown"
        return "satisfied" if observed <= limit else "violated"
    if kind == "brand":
        observed = _brand(record)
        if not observed:
            return "unknown"
        return "satisfied" if value.casefold() in observed.casefold() else "violated"
    if kind == "category":
        haystack = _categories(record)
        if not haystack:
            return "unknown"
        needle = value.casefold()
        return "satisfied" if any(needle in item.casefold() for item in haystack) else "violated"
    # Unverifiable by design (e.g. colour/material): the catalogue cannot decide it.
    return "unknown"


def feasible(record: Any, constraints: Sequence[Mapping[str, str]]) -> bool:
    return all(check_constraint(record, item["kind"], item["value"]) == "satisfied"
               for item in constraints)


# --------------------------------------------------------------------------- #
# explanation checks
# --------------------------------------------------------------------------- #


def _record_field(record: Any, field: str) -> list[str]:
    value = getattr(record, field, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(entry) for entry in value]


def verify_fact(record: Any, label: str, text: str) -> bool:
    """True when ``text`` is verbatim (allowing the renderer's ellipsis) from the record."""
    if record is None:
        return False
    field = {
        "Main category": "main_category",
        "Category": "categories",
        "Title": "title",
        "Store": "store",
        "Feature": "features",
        "Detail": "details",
    }.get(label)
    if field is None:
        return False
    candidates = _record_field(record, field)
    if field == "details":
        flattened = []
        for pair in getattr(record, "details", ()) or ():
            flattened.append(f"{pair[0]}: {pair[1]}")
            flattened.append(str(pair[1]))
        candidates = flattened
    needle = text.rstrip("\u2026").strip()
    return any(needle and (needle in candidate or candidate.startswith(needle))
               for candidate in candidates)


def _empty_reason_kind(message: Any, constraints: Sequence[Mapping[str, str]]) -> str | None:
    """Classify the empty-result text: proved no match, unverified, or nothing assessed."""
    if not constraints:
        return None
    text = str(message or "").casefold()
    if "no candidate could be verified" in text:
        return "unverified"
    if "no candidate satisfies" in text:
        return "confirmed_no_match"
    if "no candidate was available" in text:
        return "no_candidate_available"
    return "other"


def parse_explanation(message: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Product ids named by the text, fact claims, and their verification results."""
    lines = message.splitlines()
    current: str | None = None
    ids: list[str] = []
    claims: list[dict[str, Any]] = []
    for line in lines:
        item = ITEM_LINE_RE.match(line)
        if item:
            current = item.group(2)
            if current not in ids:
                ids.append(current)
            continue
        fact = FACT_LINE_RE.match(line)
        if fact and current:
            label, text = fact.group(1), fact.group(2).strip()
            record = metadata.get(current)
            claims.append({
                "parent_asin": current,
                "field": label,
                "text": text,
                "verified": verify_fact(record, label, text),
            })
    return {"ids": ids, "claims": claims}


# --------------------------------------------------------------------------- #
# scenario construction
# --------------------------------------------------------------------------- #


def scenario_matrix(
    eligible: Sequence[tuple[int, list[str], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Build 48 scenarios: 4 hard + 1 soft per history, plus 8 impossible/unverifiable."""
    scenarios: list[dict[str, Any]] = []
    impossible_kinds = (
        {"kind": "price_max", "value": "0.01"},
        {"kind": "brand", "value": "zzzz-no-such-brand"},
        # ``free_form_constraint`` has no deterministic catalogue dimension: it maps to UNVERIFIABLE, so every
        # candidate stays unresolved and none may be presented (fail closed).
        {"kind": "free_form_constraint", "value": "extra-large"},
        {"kind": "category", "value": "zzzz-no-such-category"},
    )
    for position, (index, pool, features) in enumerate(eligible):
        priced = sorted(item["price"] for item in features if item["price"] is not None)
        brands = sorted({item["brand"] for item in features if item["brand"]})
        categories = []
        for item in features:
            for category in item["categories"]:
                if category and category not in categories:
                    categories.append(category)
        threshold = round(priced[len(priced) // 2], 2) if priced else None
        brand = brands[0] if brands else None
        category = categories[-1] if categories else None

        if threshold is not None:
            scenarios.append({
                "id": f"h{index}-price", "history_index": index, "type": "hard_price",
                "message": HARD_MESSAGE,
                "constraints": [{"kind": "price_max", "value": f"{threshold:.2f}"}],
                "soft": None,
            })
        if category:
            scenarios.append({
                "id": f"h{index}-category", "history_index": index, "type": "hard_category",
                "message": HARD_MESSAGE,
                "constraints": [{"kind": "category", "value": category}],
                "soft": None,
            })
        if brand:
            scenarios.append({
                "id": f"h{index}-brand", "history_index": index, "type": "hard_brand",
                "message": HARD_MESSAGE,
                "constraints": [{"kind": "brand", "value": brand}],
                "soft": None,
            })
        if threshold is not None and category:
            scenarios.append({
                "id": f"h{index}-combo", "history_index": index, "type": "hard_combo",
                "message": HARD_MESSAGE,
                "constraints": [
                    {"kind": "price_max", "value": f"{threshold:.2f}"},
                    {"kind": "category", "value": category},
                ],
                "soft": None,
            })
        soft_kind = "brand" if brand else ("category" if category else None)
        soft_value = brand or category
        if soft_kind and soft_value:
            scenarios.append({
                "id": f"h{index}-soft", "history_index": index, "type": "soft",
                "message": SOFT_MESSAGE.format(value=soft_value),
                "constraints": [],
                "soft": {"kind": soft_kind, "value": soft_value},
            })
        scenarios.append({
            "id": f"h{index}-impossible",
            "history_index": index,
            "type": "impossible",
            "message": HARD_MESSAGE,
            "constraints": [impossible_kinds[position % len(impossible_kinds)]],
            "soft": None,
        })
    return scenarios


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    from recommendation.api.app import ServiceSettings
    from recommendation.demo import build_demo_runtime
    from recommendation.demo.agent_schemas import AgentConstraintItem, AgentRecommendRequest

    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    by_index = {index: sequence for index, sequence in enumerate(document["sequences"])}
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    item2id = {str(key): int(value) for key, value in mappings["item2id"].items()}
    histories = {
        index: by_index[index]["parent_asins"][:-1]
        for index in HISTORY_CANDIDATES
        if index in by_index
    }

    started = time.time()
    runtime = build_demo_runtime(settings=ServiceSettings.from_env())
    service = runtime.agent_service()
    load_seconds = time.time() - started
    metadata = runtime.metadata.records

    def run(message: str, history: Sequence[str], constraints: Sequence[Mapping[str, str]]) -> Any:
        began = time.perf_counter()
        response = service.run(AgentRecommendRequest(
            message=message,
            history=list(history),
            k=K,
            sources=("history", "catalog_search"),
            constraints=tuple(AgentConstraintItem(**item) for item in constraints),
        ))
        return response, (time.perf_counter() - began) * 1000.0

    # -- control runs: identify histories and the pools constraints will act on ------- #
    eligible: list[tuple[int, list[str], list[dict[str, Any]]]] = []
    control_runs: dict[int, dict[str, Any]] = {}
    for index, history in histories.items():
        response, elapsed = run(HARD_MESSAGE, history, ())
        pool = [view.parent_asin for view in response.grounded.candidates]
        features = []
        for asin in pool:
            record = metadata.get(asin)
            if record is None:
                continue
            features.append({
                "parent_asin": asin,
                "price": _price(record),
                "brand": _brand(record),
                "categories": _categories(record),
            })
        control_runs[index] = {
            "pool": pool,
            "features": features,
            "final": [item.parent_asin for item in response.recommendations],
            "ms": round(elapsed, 2),
        }
        has_price = any(item["price"] is not None for item in features)
        has_brand = any(item["brand"] for item in features)
        has_category = any(item["categories"] for item in features)
        if has_price and has_brand and has_category and len(eligible) < HISTORIES_NEEDED:
            eligible.append((index, pool, features))
        if len(eligible) >= HISTORIES_NEEDED:
            break
    if len(eligible) < HISTORIES_NEEDED:
        raise SystemExit(f"only {len(eligible)} histories with usable metadata")

    scenarios = scenario_matrix(eligible)

    # -- run and verify every scenario ------------------------------------------------ #
    samples: list[dict[str, Any]] = []
    failures = 0
    for scenario in scenarios:
        history = histories[scenario["history_index"]]
        constraints = [dict(item) for item in scenario["constraints"]]
        try:
            response, elapsed = run(scenario["message"], history, constraints)
        except Exception as exc:  # noqa: BLE001 - a failed scenario is recorded, never dropped
            failures += 1
            samples.append({
                **{key: scenario[key] for key in ("id", "history_index", "type", "message")},
                "constraints": constraints,
                "soft": scenario["soft"],
                "status": f"error:{type(exc).__name__}",
                "error": str(exc),
                "ms": None,
            })
            continue

        final = [item.parent_asin for item in response.recommendations]
        identity_ok: list[bool] = []
        violations: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for item in response.recommendations:
            record = metadata.get(item.parent_asin)
            expected_id = item2id.get(item.parent_asin)
            identity_ok.append(
                record is not None and expected_id is not None and int(item.item_id) == expected_id
            )
            for constraint in constraints:
                verdict = check_constraint(record, constraint["kind"], constraint["value"])
                entry = {
                    "parent_asin": item.parent_asin,
                    "kind": constraint["kind"],
                    "value": constraint["value"],
                    "verdict": verdict,
                }
                if verdict == "violated":
                    violations.append(entry)
                elif verdict == "unknown":
                    unresolved.append(entry)
        duplicates = len(final) - len(set(final))

        explanations = parse_explanation(response.message, metadata)
        message_ids = explanations["ids"]
        claims = explanations["claims"]
        verified_claims = sum(1 for claim in claims if claim["verified"])

        soft = scenario["soft"]
        soft_matches = 0
        if soft:
            for asin in final:
                verdict = check_constraint(metadata.get(asin), soft["kind"], soft["value"])
                soft_matches += int(verdict == "satisfied")

        # Is a compliant candidate expected?  The control pool is the same pool the
        # constrained run retrieves (same message/plan/history), so this is exact.
        control_pool = control_runs[scenario["history_index"]]["features"]
        expect_nonempty = any(
            feasible(metadata.get(item["parent_asin"]), constraints) for item in control_pool
        ) if constraints else True

        samples.append({
            "id": scenario["id"],
            "history_index": scenario["history_index"],
            "type": scenario["type"],
            "message": scenario["message"],
            "constraints": constraints,
            "soft": soft,
            "status": "ok",
            "route": response.route,
            "terminal": response.terminal.status,
            "termination_reason": response.terminal.termination_reason,
            "presented": final,
            "presented_count": len(final),
            "identity_ok": all(identity_ok),
            "duplicates": duplicates,
            "violations": violations,
            "unresolved_presented": unresolved,
            "expect_nonempty": bool(expect_nonempty),
            "expectation_met": bool(expect_nonempty) == bool(final),
            "message_ids": message_ids,
            "message_matches_final_order": message_ids == final,
            "message_matches_final_set": set(message_ids) == set(final),
            "message_coverage": (
                round(len(set(message_ids) & set(final)) / len(final), 4) if final else None
            ),
            "fact_claims": len(claims),
            "fact_verified": verified_claims,
            "fact_verification_rate": (
                round(verified_claims / len(claims), 4) if claims else None
            ),
            "fact_details": claims,
            "empty_reason_mentions_constraint": (
                bool(constraints) and any(
                    token in response.message.casefold()
                    for token in ("constraint", "price", "brand", "categor", "budget")
                )
            ),
            "empty_reason_kind": _empty_reason_kind(response.message, constraints),
            "soft_matches": soft_matches,
            "soft_share": round(soft_matches / len(final), 4) if soft and final else None,
            "source_calls": dict(response.observation.source_calls),
            "source_outcomes": dict(response.observation.source_outcomes),
            "ms": round(elapsed, 2),
        })

    # -- aggregates ------------------------------------------------------------------- #
    hard = [row for row in samples if row["status"] == "ok" and row["constraints"]]
    soft = [row for row in samples if row["status"] == "ok" and row["soft"]]
    explained = [row for row in samples if row["status"] == "ok" and row["presented"]]
    items_presented = sum(row["presented_count"] for row in hard)
    violations = sum(len(row["violations"]) for row in hard)
    unresolved = sum(len(row["unresolved_presented"]) for row in hard)
    latencies = sorted(row["ms"] for row in samples if row["ms"] is not None)
    summary = {
        "scenarios": len(samples),
        "ok": sum(1 for row in samples if row["status"] == "ok"),
        "failures": failures,
        "hard_scenarios": len(hard),
        "hard_items_presented": items_presented,
        "hard_violations": violations,
        "hard_violation_rate": round(violations / items_presented, 6) if items_presented else None,
        "hard_unresolved_presented": unresolved,
        "hard_unresolved_rate": round(unresolved / items_presented, 6) if items_presented else None,
        "identity_legality_rate": (
            round(sum(1 for row in samples if row["status"] == "ok" and row["identity_ok"])
                  / max(1, sum(1 for row in samples if row["status"] == "ok")), 6)
        ),
        "duplicate_items": sum(row.get("duplicates", 0) for row in samples),
        "expectation_met_rate": (
            round(sum(1 for row in hard if row["expectation_met"]) / len(hard), 6) if hard else None
        ),
        "expectation_misses": [
            {"id": row["id"], "expect_nonempty": row["expect_nonempty"],
             "presented": row["presented_count"]}
            for row in hard if not row["expectation_met"]
        ],
        "message_order_consistency": (
            round(sum(1 for row in explained if row["message_matches_final_order"]) / len(explained), 6)
            if explained else None
        ),
        "message_set_consistency": (
            round(sum(1 for row in explained if row["message_matches_final_set"]) / len(explained), 6)
            if explained else None
        ),
        "message_coverage_mean": (
            round(sum(row["message_coverage"] for row in explained) / len(explained), 4)
            if explained else None
        ),
        "fact_claims": sum(row["fact_claims"] for row in samples if row["status"] == "ok"),
        "fact_verified": sum(row["fact_verified"] for row in samples if row["status"] == "ok"),
        "fact_verification_rate": (
            round(sum(row["fact_verified"] for row in samples if row["status"] == "ok")
                  / max(1, sum(row["fact_claims"] for row in samples if row["status"] == "ok")), 6)
        ),
        "empty_with_constraints": sum(
            1 for row in hard if row["presented_count"] == 0
        ),
        "empty_with_constraints_reason_ok": sum(
            1 for row in hard if row["presented_count"] == 0
            and row["empty_reason_mentions_constraint"]
        ),
        "empty_reason_kinds": {
            kind: sum(
                1 for row in hard
                if row["presented_count"] == 0 and row["empty_reason_kind"] == kind
            )
            for kind in ("confirmed_no_match", "unverified", "no_candidate_available", "other")
        },
        "soft_scenarios": len(soft),
        "soft_share_mean": (
            round(sum(row["soft_share"] for row in soft if row["soft_share"] is not None)
                  / max(1, len([row for row in soft if row["soft_share"] is not None])), 4)
            if soft else None
        ),
        "empty_results": sum(1 for row in samples if row["status"] == "ok" and not row["presented"]),
        "p50_ms": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
    }
    config = {
        "scenario_set": "p3_shopping_scenarios_v1",
        "plan": ["history", "catalog_search"],
        "k": K,
        "history_candidates": list(HISTORY_CANDIDATES),
        "histories_used": [index for index, _, _ in eligible],
        "hard_message": HARD_MESSAGE,
        "soft_message_template": SOFT_MESSAGE,
        "constraint_kinds": ["price_max", "category", "brand", "color (unverifiable)"],
        "verification": "independent metadata predicates; the system's own eligibility view is not "
                        "used to grade results",
        "load_seconds": round(load_seconds, 2),
        "sequence_artifact": str(SEQUENCES.relative_to(REPO)),
        "git_head": _git_head(),
        "driver_sha256": _sha256(Path(__file__)),
        "python": sys.version.split()[0],
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "cases.json").write_text(
        json.dumps([
            {key: row[key] for key in ("id", "history_index", "type", "message",
                                       "constraints", "soft")}
            for row in samples
        ], indent=2), encoding="utf-8")
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in samples), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {}
    for name in ("config.json", "cases.json", "samples.jsonl", "summary.json"):
        payload = (args.out / name).read_bytes()
        manifest[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _git_head() -> str | None:
    import subprocess
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                                   capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001 - missing git must not fail the evaluation
        return None
    return completed.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
