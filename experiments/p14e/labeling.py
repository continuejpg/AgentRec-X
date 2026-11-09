"""P14-E independent evaluation logic (labels), separated from task construction.

The labeler reads only the frozen construction assets (persona/history/query/hard
constraints) and the real catalogue.  It never chooses a target product first; labels
are produced by deterministic evidence checks over the whole product-type pool.

Preference semantics (documented in P14E_BENCHMARK_SPEC.md):
* ``satisfied`` = catalogue text contains direct evidence of the attribute;
* ``violated``  = a negative preference whose attribute has direct evidence;
* ``unknown``   = no lexical evidence either way (never counted as a negative);
* hard constraints are only *verified* when direct text evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

CONFLICT_PAIRS: tuple[tuple[str, str], ...] = (
    ("wireless", "wired"),
    ("over_ear", "in_ear"),
    ("on_ear", "in_ear"),
)
MIN_RECOMMENDABLE = 3
MAX_EXAMPLES = 8


def _pref_key(pref: Mapping[str, Any]) -> tuple[str, str]:
    if pref.get("kind") == "form_factor":
        return ("form_factor", str(pref.get("value")))
    return ("attribute", str(pref.get("name")))


def active_prefs(events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay construction events into ``(effective, withdrawn)`` preference lists."""
    active: dict[tuple[str, str], dict[str, Any]] = {}
    withdrawn: list[dict[str, Any]] = []
    for event in events:
        for item in event.get("preference_candidates") or ():
            key = _pref_key(item)
            status = str(item.get("status") or "assert")
            if status == "withdraw":
                entry = active.pop(key, None)
                withdrawn.append(
                    {
                        "kind": key[0],
                        "value": key[1],
                        "polarity": (entry or item).get("polarity"),
                        "withdrawn_by_event": event["event_id"],
                        "previous_event": (entry or {}).get("source_event_id"),
                    }
                )
                continue
            if status == "supersede":
                previous = active.pop(key, None)
                if previous and previous not in withdrawn:
                    withdrawn.append({**previous, "status": "superseded", "superseded_by_event": event["event_id"]})
                continue
            active[key] = {
                "kind": item.get("kind", "attribute"),
                "value": key[1],
                "polarity": str(item.get("polarity") or "positive"),
                "source_event_id": event["event_id"],
                "source_session_id": event.get("session_id"),
                "source_text": event.get("raw_text"),
                "source_type": event.get("source_type"),
            }
    return list(active.values()), withdrawn


def detect_conflict(prefs: Iterable[Mapping[str, Any]]) -> str | None:
    positives = {
        str(pref.get("value"))
        for pref in prefs
        if str(pref.get("polarity")) == "positive"
    }
    for left, right in CONFLICT_PAIRS:
        if left in positives and right in positives:
            return f"positive preferences for {left!r} and {right!r} are mutually exclusive"
    return None


def pref_state(product, pref: Mapping[str, Any]) -> str:
    """Return satisfied / violated / unknown for one product against one preference."""
    value = str(pref.get("value"))
    polarity = str(pref.get("polarity"))
    if pref.get("kind") == "form_factor":
        observed = product.form_factor
        if observed is None:
            return "unknown"
        if polarity == "positive":
            return "satisfied" if observed == value else "violated"
        return "violated" if observed == value else "unknown"
    present = value in (product.attributes or {})
    if present:
        return "satisfied" if polarity == "positive" else "violated"
    return "unknown"


def hard_state(product, constraint: Mapping[str, Any]) -> str:
    """Return verified / violated / unknown for one hard constraint against one product."""
    if constraint.get("kind") == "unverifiable":
        return "unknown"
    value = str(constraint.get("value"))
    if constraint.get("kind") == "form_factor":
        observed = product.form_factor
        if observed is None:
            return "unknown"
        return "verified" if observed == value else "violated"
    return "verified" if value in (product.attributes or {}) else "unknown"


@dataclass
class CandidateAssessment:
    item_id: str
    title: str
    hard_states: dict[str, str]
    soft_states: dict[str, str]
    acceptable: bool
    reason: str


def assess_candidate(
    product,
    *,
    hard_constraints: Sequence[Mapping[str, Any]],
    preferences: Sequence[Mapping[str, Any]],
) -> CandidateAssessment:
    hard_states = {str(c.get("constraint_id")): hard_state(product, c) for c in hard_constraints}
    soft_states = {f"{p['kind']}:{p['value']}": pref_state(product, p) for p in preferences}
    hard_ok = all(state == "verified" for state in hard_states.values())
    hard_violated = any(state == "violated" for state in hard_states.values())
    violated_soft = [name for name, state in soft_states.items() if state == "violated"]
    satisfied_soft = [name for name, state in soft_states.items() if state == "satisfied"]
    positive_prefs = [p for p in preferences if str(p.get("polarity")) == "positive"]
    needed_soft = 1 if positive_prefs else 0
    acceptable = bool(hard_ok and not hard_violated and not violated_soft and len(satisfied_soft) >= needed_soft)
    if acceptable:
        reason = "hard constraints verified; no preference violation; positive evidence present"
    elif not hard_ok:
        reason = "hard constraint violated" if hard_violated else "hard constraint evidence unknown"
    elif violated_soft:
        reason = "negative preference violated"
    else:
        reason = "no positive preference evidence"
    return CandidateAssessment(
        item_id=product.item_id,
        title=product.title,
        hard_states=hard_states,
        soft_states=soft_states,
        acceptable=acceptable,
        reason=reason,
    )


def label_task(task: Mapping[str, Any], evaluator: Mapping[str, Any], pool: Sequence[Any]) -> dict[str, Any]:
    """Independently evaluate one frozen task over its product-type pool."""
    prefs = list(evaluator.get("effective_preferences") or ())
    hard = list(evaluator.get("hard_constraints") or ())
    conflict = detect_conflict(prefs)
    assessments: list[dict[str, Any]] = []
    acceptable_ids: list[str] = []
    hard_verified_count = 0
    hard_unknown_count = 0
    hard_violated_count = 0
    for product in pool:
        assessment = assess_candidate(product, hard_constraints=hard, preferences=prefs)
        if assessment.acceptable:
            acceptable_ids.append(product.item_id)
        if all(state == "verified" for state in assessment.hard_states.values()):
            hard_verified_count += 1
        elif any(state == "violated" for state in assessment.hard_states.values()):
            hard_violated_count += 1
        else:
            hard_unknown_count += 1
        assessments.append(
            {
                "item_id": assessment.item_id,
                "title": assessment.title,
                "hard_states": assessment.hard_states,
                "soft_states": assessment.soft_states,
                "acceptable": assessment.acceptable,
                "reason": assessment.reason,
            }
        )

    unverifiable = [c for c in hard if c.get("kind") == "unverifiable"]
    if conflict:
        expected = "clarification_required"
        rationale = f"conflicting history preference: {conflict}"
    elif unverifiable:
        expected = "insufficient_evidence"
        rationale = "a stated hard constraint has no verifiable catalogue field: " + ", ".join(
            str(c.get("name")) for c in unverifiable
        )
    elif not prefs and not hard:
        expected = "clarification_required"
        rationale = "no recoverable preference and no stated hard constraint in the request"
    elif acceptable_ids:
        expected = "recommend"
        rationale = f"{len(acceptable_ids)} catalogue products have verified evidence for all hard constraints and no violated preference"
    elif hard_verified_count == 0 and hard_violated_count == 0 and hard_unknown_count > 0:
        expected = "insufficient_evidence"
        rationale = "no product has verifiable evidence for the stated hard constraints"
    else:
        expected = "no_verified_match"
        rationale = "no catalogue product satisfies every stated hard constraint with direct evidence"

    acceptable_set = set(acceptable_ids)
    examples = [a for a in assessments if a["acceptable"]][:MAX_EXAMPLES]
    near_misses = [
        a for a in assessments
        if not a["acceptable"] and all(state != "violated" for state in a["hard_states"].values())
    ][:MAX_EXAMPLES]
    return {
        "expected_decision": expected,
        "decision_rationale": rationale,
        "conflict": conflict,
        "verified_candidate_count": len(acceptable_ids),
        "verified_candidates_sample": acceptable_ids[:20],
        "hard_verified_pool_count": hard_verified_count,
        "hard_unknown_pool_count": hard_unknown_count,
        "hard_violated_pool_count": hard_violated_count,
        "pool_size": len(pool),
        "preference_states": {
            "satisfied": sum(
                1 for a in assessments if any(state == "satisfied" for state in a["soft_states"].values())
            ),
            "violated": sum(
                1 for a in assessments if any(state == "violated" for state in a["soft_states"].values())
            ),
            "unknown": sum(
                1 for a in assessments if a["soft_states"] and all(state == "unknown" for state in a["soft_states"].values())
            ),
        },
        "candidate_examples": examples,
        "near_miss_examples": near_misses,
        "acceptable_id_count": len(acceptable_set),
        "hard_constraint_verifiable_counts": {
            str(c.get("constraint_id")): sum(
                1 for a in assessments if a["hard_states"].get(str(c.get("constraint_id"))) == "verified"
            )
            for c in hard
        },
    }
