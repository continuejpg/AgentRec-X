"""Phase 4: 36 multi-turn shopping scenarios (2-3 turns each), engineering validation only.

Every scenario drives the public ``DemoAgentService.run_shopping_turn`` path: the same
preference store, the same controller, the same candidate plane.  Labels are written by hand
from the scenario script (not by a model) and verified against the catalogue metadata.

    .venv/bin/python -m experiments.p4_multi_turn_eval --out runs/p4_multi_turn_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config
from recommendation.control.grounded_reasoning import first_attribute, parse_price

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
DEFAULT_OUT = REPO / "runs" / "p4_multi_turn_eval"
BRAND_KEYS = ("Brand Name", "Brand", "Manufacturer")
K = 5
SOURCES = ("history", "catalog_search")

PRICE_CLAIM = re.compile(r"catalogue price \$([0-9]+(?:\.[0-9]+)?)")
PRICE_MIN_CLAIM = re.compile(r"minimum \(catalogue price \$([0-9]+(?:\.[0-9]+)?)\)")
CATEGORY_CLAIM = re.compile(r"(?:required category|interest in) '([^']+)'")
BRAND_CLAIM = re.compile(r"(?:required brand|preferred brand) '([^']+)'")
FACT_PRICE = re.compile(r"price \$([0-9]+(?:\.[0-9]+)?)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- independent catalogue predicates -------------------------------------- #


def _categories(record: Any) -> tuple[str, ...]:
    values = list(getattr(record, "categories", ()) or ())
    main = getattr(record, "main_category", None)
    if main:
        values.append(main)
    return tuple(str(value) for value in values)


def check_constraint(record: Any, kind: str, value: str) -> str:
    if record is None:
        return "unknown"
    if kind == "price_max":
        price, limit = parse_price(getattr(record, "price_text", None)), parse_price(value)
        if price is None or limit is None:
            return "unknown"
        return "satisfied" if price <= limit else "violated"
    if kind == "price_min":
        price, limit = parse_price(getattr(record, "price_text", None)), parse_price(value)
        if price is None or limit is None:
            return "unknown"
        return "satisfied" if price >= limit else "violated"
    if kind == "brand":
        brand = first_attribute(record, BRAND_KEYS)
        if not brand:
            return "unknown"
        return "satisfied" if value.casefold() in brand.casefold() else "violated"
    if kind == "category":
        values = _categories(record)
        if not values:
            return "unknown"
        return "satisfied" if any(value.casefold() in c.casefold() for c in values) else "violated"
    return "unknown"


def check_reason(record: Any, reason: str) -> tuple[int, int]:
    """Return (claims, verified) for one deterministic reason string."""
    if record is None or not reason:
        return 0, 0
    claims = verified = 0
    price = parse_price(getattr(record, "price_text", None))
    for match in PRICE_CLAIM.finditer(reason):
        claims += 1
        verified += int(price is not None and abs(price - float(match.group(1))) < 0.005)
    for match in PRICE_MIN_CLAIM.finditer(reason):
        claims += 1
        verified += int(price is not None and abs(price - float(match.group(1))) < 0.005)
    for match in CATEGORY_CLAIM.finditer(reason):
        claims += 1
        verified += int(any(match.group(1).casefold() in c.casefold() for c in _categories(record)))
    for match in BRAND_CLAIM.finditer(reason):
        claims += 1
        brand = first_attribute(record, BRAND_KEYS) or ""
        verified += int(match.group(1).casefold() in brand.casefold())
    if "catalogue facts:" in reason:
        fact_text = reason.split("catalogue facts:", 1)[1]
        for match in FACT_PRICE.finditer(fact_text):
            claims += 1
            verified += int(price is not None and abs(price - float(match.group(1))) < 0.005)
        for part in fact_text.split(","):
            part = part.strip().rstrip(".")
            if part.startswith("brand "):
                claims += 1
                brand = first_attribute(record, BRAND_KEYS) or ""
                verified += int(part[6:].strip().casefold() in brand.casefold())
            elif part.startswith("category "):
                claims += 1
                value = part[9:].strip()
                verified += int(any(value.casefold() in c.casefold() for c in _categories(record)))
    return claims, verified


# -- scenario set ---------------------------------------------------------- #


def scenarios() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for offset in range(12):
        history = 12000 + offset
        out.append({
            "id": f"price-{history}",
            "history": history,
            "kind": "price_replace",
            "turns": [
                {"message": "I need a yoga mat, budget $30",
                 "hard": {"price_max": "30"}, "soft": {("category", "yoga mat")},
                 "recs": True, "clarify": None, "stale": []},
                {"message": "make it under $50 instead",
                 "hard": {"price_max": "50"}, "soft": {("category", "yoga mat")},
                 "recs": True, "clarify": None, "stale": ["30"]},
            ],
        })
    for offset in range(8):
        history = 12000 + offset
        out.append({
            "id": f"brand-{history}",
            "history": history,
            "kind": "brand_remove",
            "turns": [
                {"message": "I prefer the brand Acme",
                 "hard": {}, "soft": {("brand", "Acme")},
                 "recs": True, "clarify": None, "stale": []},
                {"message": "forget the brand",
                 "hard": {}, "soft": set(),
                 "recs": True, "clarify": None, "stale": ["Acme"]},
            ],
        })
    for offset in range(8):
        history = 12000 + offset
        out.append({
            "id": f"category-{history}",
            "history": history,
            "kind": "category_remove",
            "turns": [
                {"message": "I'm looking for hiking",
                 "hard": {}, "soft": {("category", "hiking")},
                 "recs": True, "clarify": None, "stale": []},
                {"message": "forget the hiking",
                 "hard": {}, "soft": set(),
                 "recs": True, "clarify": None, "stale": ["hiking"]},
            ],
        })
    for offset in range(4):
        history = 12004 + offset
        out.append({
            "id": f"conflict-{history}",
            "history": history,
            "kind": "conflict",
            "turns": [
                {"message": "I need a yoga mat, budget $50",
                 "hard": {"price_max": "50"}, "soft": {("category", "yoga mat")},
                 "recs": True, "clarify": None, "stale": []},
                {"message": "my budget is at least $80",
                 "hard": {"price_max": "50", "price_min": "80"}, "soft": {("category", "yoga mat")},
                 "recs": False, "clarify": "conflict", "stale": []},
                {"message": "make it under $200 instead",
                 "hard": {"price_max": "200", "price_min": "80"}, "soft": {("category", "yoga mat")},
                 "recs": True, "clarify": None, "stale": ["50"]},
            ],
        })
    for offset in range(4):
        history = 12008 + offset
        out.append({
            "id": f"unverified-{history}",
            "history": history,
            "kind": "unverified",
            "turns": [
                {"message": "I need a yoga mat",
                 "constraints": [{"kind": "free_form_constraint", "value": "extra-large"}],
                 "hard": {"free_form_constraint": "extra-large"}, "soft": {("category", "yoga mat")},
                 "recs": False, "clarify": "unverified", "stale": []},
                {"message": "make it under $30 instead", "constraints": [],
                 "hard": {"price_max": "30"}, "soft": {("category", "yoga mat")},
                 "recs": True, "clarify": None, "stale": ["extra-large"]},
            ],
        })
    return out


# -- harness --------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline", action="store_true", help="single-turn system, no intent or clarification")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    from recommendation.api.app import ServiceSettings
    from recommendation.demo import build_demo_runtime
    from recommendation.demo.agent_schemas import AgentConstraintItem, AgentRecommendRequest

    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    by_index = {index: sequence for index, sequence in enumerate(document["sequences"])}
    runtime = build_demo_runtime(settings=ServiceSettings.from_env())
    service = runtime.agent_service()
    metadata = runtime.metadata.records
    run_nonce = uuid.uuid4().hex[:8]

    samples: list[dict[str, Any]] = []
    for scenario in scenarios():
        clarification_id = None
        scenario_ok = True
        # one fresh session per scenario: same id for every turn
        session = f"p4eval-{run_nonce}-{scenario['id']}"
        for turn_index, spec in enumerate(scenario["turns"]):
            began = time.perf_counter()
            if args.baseline:
                response = service.run(
                    AgentRecommendRequest(
                        message=spec["message"],
                        history=list(by_index[scenario["history"]]["parent_asins"][:-1]),
                        k=K,
                        sources=SOURCES,
                        constraints=(),
                        user_key=session,
                    )
                )
            else:
                response = service.run_shopping_turn(
                    AgentRecommendRequest(
                        message=spec["message"],
                        history=list(by_index[scenario["history"]]["parent_asins"][:-1]),
                        k=K,
                        sources=SOURCES,
                        constraints=tuple(
                            AgentConstraintItem(**item) for item in spec.get("constraints", [])
                        ),
                        clarification_id=clarification_id,
                    ),
                    session_id=session,
                )
            elapsed = (time.perf_counter() - began) * 1000.0
            actual_hard = {
                (entry.kind, entry.value.casefold()) for entry in (response.intent.hard if response.intent else ())
            }
            expected_hard = {(kind, value.casefold()) for kind, value in spec["hard"].items()}
            actual_soft = {
                (entry.kind, entry.value.casefold(), entry.polarity)
                for entry in (response.intent.soft if response.intent else ())
            }
            expected_soft = {(kind, value.casefold(), "prefer") for kind, value in spec["soft"]}
            intent_ok = actual_hard == expected_hard and actual_soft == expected_soft
            stale_leaks = [
                value for value in spec["stale"]
                if any(value.casefold() in str(entry.value).casefold() for entry in (response.intent.hard if response.intent else ()))
                or any(value.casefold() in str(entry.value).casefold() for entry in (response.intent.soft if response.intent else ()))
            ]
            clarify_ok = (
                (response.clarification is not None and response.clarification.kind == spec["clarify"]
                 and response.terminal.termination_reason == "awaiting_user")
                if spec["clarify"] else response.clarification is None
            )
            recs_ok = bool(response.recommendations) == bool(spec["recs"])

            violations: list[str] = []
            unresolved_presented: list[str] = []
            reasons_claims = reasons_verified = 0
            for item in response.recommendations:
                record = metadata.get(item.parent_asin)
                for kind, value in spec["hard"].items():
                    verdict = check_constraint(record, kind, value)
                    if verdict == "violated":
                        violations.append(item.parent_asin)
                    elif verdict == "unknown":
                        unresolved_presented.append(item.parent_asin)
                claims, verified = check_reason(record, item.reason or "")
                reasons_claims += claims
                reasons_verified += verified

            targets = [
                value for kind, value in spec["soft"]
            ] + [value for kind, value in spec["hard"].items() if kind in ("category", "brand")]
            adapted = None
            if spec["recs"] and targets:
                adapted = any(
                    any(
                        target.casefold() in " ".join(
                            [str(getattr(item, "title", "") or "")]
                            + [str(getattr(metadata.get(item.parent_asin), "main_category", "") or "")]
                        ).casefold()
                        for target in targets
                    )
                    for item in response.recommendations
                )

            row = {
                "scenario": scenario["id"],
                "kind": scenario["kind"],
                "turn": turn_index,
                "message": spec["message"],
                "intent_ok": intent_ok,
                "expected_hard": sorted(expected_hard),
                "actual_hard": sorted(actual_hard),
                "expected_soft": sorted(expected_soft),
                "actual_soft": sorted(actual_soft),
                "stale_leaks": stale_leaks,
                "stale_expected": bool(spec["stale"]),
                "clarify_ok": clarify_ok,
                "clarification_kind": response.clarification.kind if response.clarification else None,
                "recs_ok": recs_ok,
                "presented": [item.parent_asin for item in response.recommendations],
                "violations": violations,
                "unresolved_presented": unresolved_presented,
                "adapted": adapted,
                "reason_claims": reasons_claims,
                "reason_verified": reasons_verified,
                "terminal": response.terminal.termination_reason,
                "status": response.terminal.status,
                "ms": round(elapsed, 2),
            }
            samples.append(row)
            scenario_ok = scenario_ok and intent_ok and clarify_ok and recs_ok and not stale_leaks
            if spec["clarify"]:
                clarification_id = response.clarification.clarification_id if response.clarification else None
            else:
                clarification_id = None

        samples.append({
            "scenario": scenario["id"],
            "kind": scenario["kind"],
            "turn": "scenario",
            "scenario_ok": scenario_ok,
            "ms": None,
        })

    turns = [row for row in samples if row["turn"] != "scenario"]
    scenario_rows = [row for row in samples if row["turn"] == "scenario"]
    presented = sum(len(row["presented"]) for row in turns)
    latencies = sorted(row["ms"] for row in turns)
    summary = {
        "scenarios": len(scenario_rows),
        "turns": len(turns),
        "intent_accuracy": round(sum(1 for row in turns if row["intent_ok"]) / len(turns), 6),
        "stale_expected_turns": sum(1 for row in turns if row.get("stale_expected")),
        "stale_leak_turns": sum(1 for row in turns if row["stale_leaks"]),
        "hard_items": presented,
        "hard_violations": sum(len(row["violations"]) for row in turns),
        "hard_violation_rate": (
            round(sum(len(row["violations"]) for row in turns) / presented, 6) if presented else None
        ),
        "unresolved_presented": sum(len(row["unresolved_presented"]) for row in turns),
        "conflict_turns": sum(1 for row in turns if row["clarification_kind"] == "conflict"),
        "conflict_handled": sum(
            1 for row in turns if row["clarification_kind"] == "conflict" and row["clarify_ok"]
        ),
        "clarification_turns": sum(1 for row in turns if row["clarification_kind"]),
        "clarification_scenarios": sum(1 for row in scenario_rows if row["kind"] in ("conflict", "unverified")),
        "clarification_loop_success": sum(
            1 for row in scenario_rows
            if row["kind"] in ("conflict", "unverified") and row["scenario_ok"]
        ),
        "adaptation_turns": sum(1 for row in turns if row["adapted"] is not None),
        "adapted_turns": sum(1 for row in turns if row["adapted"] is True),
        "reason_claims": sum(row["reason_claims"] for row in turns),
        "reason_verified": sum(row["reason_verified"] for row in turns),
        "reason_traceability": (
            round(sum(row["reason_verified"] for row in turns) / max(1, sum(row["reason_claims"] for row in turns)), 6)
        ),
        "task_completion": round(sum(1 for row in scenario_rows if row["scenario_ok"]) / len(scenario_rows), 6),
        "p50_ms": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
    }
    config = {
        "scenario_set": "p4_multi_turn_v1",
        "scenario_count": len(scenario_rows),
        "run_nonce": run_nonce,
        "k": K,
        "sources": list(SOURCES),
        "verification": "independent catalogue predicates; labels hand-written per turn",
        "git_head": _git_head(),
        "driver_sha256": _sha256(Path(__file__)),
        "python": sys.version.split()[0],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in samples), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checksums = {}
    for name in ("config.json", "samples.jsonl", "summary.json"):
        payload = (args.out / name).read_bytes()
        checksums[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (args.out / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _git_head() -> str | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True, timeout=10
        )
    except Exception:  # noqa: BLE001
        return None
    return completed.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
