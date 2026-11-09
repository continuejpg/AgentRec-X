"""M5.1 shopping-loop unified decision state smoke (synthetic, offline, deterministic).

This is a runnable end-to-end demonstration of the M5.1 integration; it is **not** an
evaluation and makes **no recommendation-quality claim**.  Everything is synthetic and
local: the fixture engine/catalogue from :mod:`tests.demo_fixture` (no checkpoint), no
dataset, no network, no LLM, no provider key and no server process.  The only artefacts
are a temporary directory and, when asked for, a JSON summary.

What it does
------------
It creates a demo app and session, then runs the **same fixed shopping request** (which
carries one trusted hard constraint, so the SHADOW state records an explicit user
constraint too) under the M5.1 modes and prints the observable consequence of each:

``OFF`` (unset)
    the accepted behaviour: the trusted constraint narrows the candidates exactly as
    before, and no decision-state database is touched;
``SHADOW``
    the same user-visible response, plus the M5 events / state digest / projections and
    the bounded diagnostics record;
``APPLY_CONSTRAINTS`` (experimental)
    an additional explicit user-authorized constraint seeded into the task state *before*
    the turn narrows eligibility further through the **existing** evaluator, and is
    merged into the existing constraint inputs;
``APPLY_CONSTRAINTS`` with an inferred hypothesis only
    the no-filter verdict: a soft hypothesis with the same constraint kind is recorded
    but can never exclude a candidate.

It also verifies replay-after-reopen digest equality (extract, restart, reconstruct)
and prints the three read-only projections as bounded canonical JSON.

Reproducibility
---------------
The request, catalogue, candidate order, constraint effects and verdicts are fixed.  The
demo session identity is normally a server-owned random UUID; to make repeated runs print
*identical* event ids and state digests, this experiment installs a deterministic UUID
stream (``uuid.uuid4``) for the duration of the run.  That is a synthetic-experiment
reproducibility control (the equivalent of fixing a random seed) and changes no product
code.

Usage::

    python -m experiments.m5_1_shopping_smoke
    python -m experiments.m5_1_shopping_smoke --json /tmp/m5_1_smoke.json

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Third-party notice from Starlette's TestClient (imported by the offline harness).  The
# repository's pytest configuration already ignores it; a module run would print it to
# stderr, which adds nothing to this smoke's evidence.
warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
    category=UserWarning,
)

from recommendation.state.event_store import SQLiteEventStore  # noqa: E402
from recommendation.state.schemas import PreferenceScope  # noqa: E402
from recommendation.state.state_manager import StateManager  # noqa: E402
from recommendation.state.state_projector import (  # noqa: E402
    for_agent,
    for_constraints,
    for_ranker,
)
from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402
from tests.state_fixtures import constraint_event, hypothesis_event  # noqa: E402

#: Documented M5.1 configuration surface.
MODE_ENV = "AGENTRECX_STATE_MODE"
DB_ENV = "AGENTRECX_STATE_DB"

SHOPPING_ROUTE = "/v1/demo/sessions/{session}/agent/shopping"
MESSAGE = "Recommend some hiking gear. I prefer black."
K = 4
SOURCES = ("history", "catalog_search")

#: The trusted hard constraint carried by the fixed request in **every** mode, so the
#: request is identical across OFF/SHADOW/APPLY and the SHADOW state also records an
#: explicit user constraint.  Only cand-red ($25), cand-blue ($40) and cand-black ($15)
#: satisfy price_max=50, so the OFF baseline is three candidates.
TRUSTED_CONSTRAINT = {"kind": "price_max", "value": "50"}
REQUEST_CONSTRAINTS = (TRUSTED_CONSTRAINT,)

#: The seeded explicit constraint: on top of the trusted price_max=50 the merged inputs
#: are tightest at price_max=30, so only cand-red ($25) and cand-black ($15) remain.
SEEDED_CONSTRAINT = ("price_max", "30")
#: The seeded inferred hypothesis: a hard price_max=1 would exclude every candidate
#: ($15..$60), so "nothing is filtered" is a strong observation.
SEEDED_HYPOTHESIS = ("price_max", "1")

#: Event types the M5.1 integration is allowed to emit under the **M5.2 superset** (the
#: same explicit facts plus the two trusted snapshot references a presented turn records;
#: no hypothesis / knowledge events are emitted by this scenario).
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
        "candidate_pool_referenced",
        "ranking_snapshot_referenced",
    }
)

_SESSION_TURN_ID = re.compile(r"^demo-session:[0-9a-f-]{36}:[0-9a-f]{1,32}$")
_MASKED_KEYS = frozenset({"timing", "run_id", "run_ref"})
_JSON_LIMIT = 640


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


class Checks:
    """Collect named boolean checks so the exit code reflects real evidence."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f" ({detail})"
        print(line)
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result[1] for result in self.results)

    def as_list(self) -> list[dict[str, Any]]:
        return [
            {"check": name, "passed": passed, "detail": detail}
            for name, passed, detail in self.results
        ]


def _set_env(mode: str | None, database: Path) -> None:
    """Point the demo at one state mode and one state database."""
    if mode is None:
        os.environ.pop(MODE_ENV, None)
    else:
        os.environ[MODE_ENV] = mode
    os.environ[DB_ENV] = str(database)


def _create_session(harness: Any) -> str:
    response = harness.client.post("/v1/demo/sessions", json={"profile_id": PROFILE_A})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _shopping_turn(harness: Any, session_id: str, constraints: Any = REQUEST_CONSTRAINTS) -> Any:
    """POST the fixed shopping request (the same body in every mode)."""
    body: dict[str, Any] = {
        "message": MESSAGE,
        "k": K,
        "sources": list(SOURCES),
        "constraints": list(constraints),
    }
    return harness.client.post(SHOPPING_ROUTE.format(session=session_id), json=body)


def _seed_state(database: Path, user_key: str, task_id: str, events: Any) -> None:
    """Append validated fixture events through a real M5 ``StateManager``."""
    store = SQLiteEventStore(database)
    try:
        manager = StateManager(store)
        manager.load_or_create(user_key, session_id=task_id, task_id=task_id)
        for event in events:
            manager.apply_event(event)
    finally:
        store.close()


def _read_state(database: Path, user_key: str, task_id: str) -> dict[str, Any]:
    """Replay one task from a fresh store (i.e. after a process restart)."""
    store = SQLiteEventStore(database)
    try:
        manager = StateManager(store)
        events = store.get_events(task_id, user_key)
        state = manager.replay_state(user_key, task_id)
        return {
            "events": events,
            "state": state,
            "digest": state.state_digest(),
            "agent": for_agent(state),
            "ranker": for_ranker(state),
            "constraints": for_constraints(state),
            "count": store.count(user_key),
        }
    finally:
        store.close()


def _event_type_counts(events: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = event.event_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalise(value: Any) -> Any:
    """Mask wall-clock and controller-generated identity fields for the parity verdict."""
    if isinstance(value, dict):
        return {
            key: "<masked>" if key in _MASKED_KEYS else _normalise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, str) and _SESSION_TURN_ID.match(value):
        return "<turn-id>"
    return value


def _normalised_response(body: dict[str, Any]) -> dict[str, Any]:
    """Drop ``timing`` and mask server-owned run/turn identities only."""
    data = copy.deepcopy(body)
    data.pop("timing", None)
    return _normalise(data)


def _canonical_bounded(payload: Any, limit: int = _JSON_LIMIT) -> str:
    """Canonical JSON of a payload, visibly truncated when it exceeds ``limit`` chars."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit} chars)"


def _diagnostics_summary(diagnostics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Deterministic subset of the bounded diagnostics record (no server-owned task id)."""
    if diagnostics is None:
        return None
    keep = (
        "mode",
        "events",
        "state_version",
        "state_digest",
        "event_count",
        "user_turn_count",
        "preference_count",
        "soft_hypothesis_count",
        "hard_constraint_count",
        "constraint_kinds",
        "suppressed_long_term_count",
        "agent_projection_version",
        "ranker_signal_version",
        "ranker_feature_count",
        "constraints_version",
        "excluded_hypothesis_count",
        "trusted_constraint_count",
        "constraint_match",
        "error",
    )
    return {key: diagnostics[key] for key in keep if key in diagnostics}


def _recommendation_identities(body: dict[str, Any]) -> list[str]:
    return [str(item["parent_asin"]) for item in body["recommendations"]]


def _deterministic_uuid4():
    """Return a deterministic ``uuid4`` replacement (sequential UUID integers)."""
    counter = itertools.count(1)

    def _uuid4() -> uuid.UUID:
        return uuid.UUID(int=next(counter))

    return _uuid4


# --------------------------------------------------------------------------- #
# the scenario
# --------------------------------------------------------------------------- #


def run(root: Path) -> tuple[bool, dict[str, Any]]:
    """Run the four-mode scenario; return ``(passed, machine-readable result)``."""
    checks = Checks()
    result: dict[str, Any] = {
        "milestone": "M5.1",
        "kind": "synthetic shopping-loop smoke",
        "quality_claim": "none",
        "synthetic": True,
        "request": {
            "message": MESSAGE,
            "k": K,
            "sources": list(SOURCES),
            "body_constraints": [dict(item) for item in REQUEST_CONSTRAINTS],
        },
        "modes": {},
    }

    # -- OFF (mode unset): accepted behaviour, no decision state ----------------- #
    print("\n-- OFF (AGENTRECX_STATE_MODE unset) " + "-" * 42)
    off_root = root / "off"
    off_root.mkdir(parents=True)
    off_db = off_root / "decision_state.sqlite3"
    _set_env(None, off_db)
    off_harness = build_harness(off_root)
    try:
        off_session = _create_session(off_harness)
        off_response = _shopping_turn(off_harness, off_session)
        off_service = off_harness.runtime.agent_service()
        checks.check(
            "OFF turn status is 200", off_response.status_code == 200, off_response.text[:120]
        )
        checks.check(
            "OFF composes mode 'off'", off_service.state_mode == "off", str(off_service.state_mode)
        )
        checks.check(
            "OFF records no state diagnostics",
            off_service.state_diagnostics(off_session) is None,
        )
        checks.check("OFF opens no decision-state database", not off_db.exists())
        off_body = off_response.json()
    finally:
        close_harness(off_harness)
    off_recommendations = _recommendation_identities(off_body)
    print(f"  candidate order      : {off_recommendations}")
    print(f"  terminal             : {off_body['terminal']['status']}/{off_body['terminal']['termination_reason']}")
    result["modes"]["off"] = {
        "mode": "off",
        "recommendations": off_recommendations,
        "state_db_created": off_db.exists(),
    }

    # -- SHADOW: same response, decision state recorded -------------------------- #
    print("\n-- SHADOW " + "-" * 62)
    shadow_root = root / "shadow"
    shadow_root.mkdir(parents=True)
    shadow_db = shadow_root / "decision_state.sqlite3"
    _set_env("shadow", shadow_db)
    shadow_harness = build_harness(shadow_root)
    try:
        shadow_session = _create_session(shadow_harness)
        shadow_response = _shopping_turn(shadow_harness, shadow_session)
        shadow_service = shadow_harness.runtime.agent_service()
        shadow_user_key = shadow_harness.runtime.sessions.get(shadow_session).user_key
        shadow_diagnostics = shadow_service.state_diagnostics(shadow_session)
    finally:
        close_harness(shadow_harness)
    shadow_body = shadow_response.json()
    parity = _normalised_response(off_body) == _normalised_response(shadow_body)
    checks.check("SHADOW turn status is 200", shadow_response.status_code == 200)
    checks.check(
        "SHADOW composes mode 'shadow'",
        shadow_service.state_mode == "shadow",
        str(shadow_service.state_mode),
    )
    checks.check("OFF-vs-SHADOW user-visible parity (timing/run ids masked)", parity)
    checks.check("SHADOW created the decision-state database", shadow_db.exists())

    shadow_state = _read_state(shadow_db, shadow_user_key, shadow_session)
    shadow_events = shadow_state["events"]
    shadow_types = _event_type_counts(shadow_events)
    shadow_digest = shadow_state["digest"]
    print(f"  state events by type  : {shadow_types}")
    print(f"  event count           : {len(shadow_events)}")
    print(f"  state digest          : {shadow_digest}")
    checks.check(
        "SHADOW emitted only the supported M5.2-superset event types",
        set(shadow_types) <= SUPPORTED_EVENT_TYPES and shadow_types,
        str(sorted(shadow_types)),
    )
    checks.check(
        "SHADOW event vocabulary is context + explicit preference + explicit constraint "
        "+ the two trusted snapshot references",
        shadow_types
        == {
            "task_context_updated": 1,
            "user_preference_stated": 1,
            "user_constraint_stated": 1,
            "candidate_pool_referenced": 1,
            "ranking_snapshot_referenced": 1,
        },
        str(shadow_types),
    )
    checks.check(
        "all recorded events are TASK-scoped (never durable long-term)",
        all(
            event.scope is None or event.scope is PreferenceScope.TASK
            for event in shadow_events
        )
        and not any(event.applies_long_term for event in shadow_events),
    )
    checks.check(
        "SHADOW diagnostics carry mode/events/event_count/state_digest",
        isinstance(shadow_diagnostics, dict)
        and shadow_diagnostics.get("mode") == "shadow"
        and shadow_diagnostics.get("events") == len(shadow_events)
        and shadow_diagnostics.get("event_count") == len(shadow_events)
        and shadow_diagnostics.get("state_digest") == shadow_digest,
    )
    checks.check(
        "SHADOW diagnostics carry the three projection versions and a digest",
        isinstance(shadow_diagnostics, dict)
        and shadow_diagnostics.get("agent_projection_version") == 1
        and shadow_diagnostics.get("ranker_signal_version") == 1
        and shadow_diagnostics.get("constraints_version") == 1
        and re.fullmatch(r"[0-9a-f]{64}", str(shadow_diagnostics.get("state_digest"))),
    )
    shadow_agent = shadow_state["agent"].model_dump(mode="json")
    shadow_ranker = shadow_state["ranker"].model_dump(mode="json")
    shadow_constraints = shadow_state["constraints"].model_dump(mode="json")
    checks.check(
        "for_agent projects the explicit colour preference",
        [(item["kind"], item["value"]) for item in shadow_agent["preferences"]]
        == [("color", "black")],
    )
    checks.check(
        "for_ranker projects one non-soft feature with ordinal strength",
        len(shadow_ranker["features"]) == 1
        and shadow_ranker["features"][0]["strength_rank"] == 3
        and shadow_ranker["features"][0]["is_soft"] is False,
    )
    checks.check(
        "for_constraints projects the trusted price_max constraint",
        [(item["kind"], item["value"]) for item in shadow_constraints["constraints"]]
        == [("price_max", TRUSTED_CONSTRAINT["value"])],
    )
    print(f"  for_agent             : {_canonical_bounded(shadow_agent)}")
    print(f"  for_ranker            : {_canonical_bounded(shadow_ranker)}")
    print(f"  for_constraints       : {_canonical_bounded(shadow_constraints)}")
    print(f"  diagnostics           : {_canonical_bounded(_diagnostics_summary(shadow_diagnostics))}")
    result["modes"]["shadow"] = {
        "mode": "shadow",
        "events_by_type": shadow_types,
        "event_count": len(shadow_events),
        "state_digest": shadow_digest,
        "diagnostics": _diagnostics_summary(shadow_diagnostics),
        "projections": {
            "for_agent": shadow_agent,
            "for_ranker": shadow_ranker,
            "for_constraints": shadow_constraints,
        },
        "user_visible_parity_with_off": parity,
    }

    # -- APPLY_CONSTRAINTS: a seeded explicit constraint narrows eligibility ----- #
    print("\n-- APPLY_CONSTRAINTS (seeded explicit constraint) " + "-" * 25)
    apply_root = root / "apply"
    apply_root.mkdir(parents=True)
    apply_db = apply_root / "decision_state.sqlite3"
    _set_env("apply_constraints", apply_db)
    apply_harness = build_harness(apply_root)
    try:
        apply_session = _create_session(apply_harness)
        apply_user_key = apply_harness.runtime.sessions.get(apply_session).user_key
        _seed_state(
            apply_db,
            apply_user_key,
            apply_session,
            [
                constraint_event(
                    user_key=apply_user_key,
                    task_id=apply_session,
                    session_id=apply_session,
                    kind=SEEDED_CONSTRAINT[0],
                    value=SEEDED_CONSTRAINT[1],
                )
            ],
        )
        apply_response = _shopping_turn(apply_harness, apply_session)
        apply_service = apply_harness.runtime.agent_service()
        apply_diagnostics = apply_service.state_diagnostics(apply_session)
    finally:
        close_harness(apply_harness)
    apply_body = apply_response.json()
    apply_recommendations = _recommendation_identities(apply_body)
    apply_requirements = list(apply_body["eligibility"]["requirements"])
    checks.check("APPLY turn status is 200", apply_response.status_code == 200)
    checks.check(
        "APPLY composes mode 'apply_constraints'",
        apply_service.state_mode == "apply_constraints",
        str(apply_service.state_mode),
    )
    checks.check(
        "APPLY narrowed the presented candidates using the seeded constraint",
        apply_recommendations == ["cand-red", "cand-black"],
        str(apply_recommendations),
    )
    checks.check(
        "APPLY performed the narrowing through the existing eligibility evaluator",
        apply_body["eligibility"]["evaluated"] is True
        and {
            (item["kind"], item["expected"]) for item in apply_requirements
        }
        == {("price_max", TRUSTED_CONSTRAINT["value"]), SEEDED_CONSTRAINT},
        str(apply_requirements),
    )
    eligibility = apply_body["eligibility"]
    checks.check(
        "APPLY eligibility three-state sum equals the number of assessments",
        eligibility["verified_eligible_count"]
        + eligibility["ineligible_count"]
        + eligibility["unresolved_count"]
        == len(eligibility["assessments"]),
    )
    apply_state = _read_state(apply_db, apply_user_key, apply_session)
    apply_types = _event_type_counts(apply_state["events"])
    apply_constraint_pairs = {
        (view.kind.value, view.value) for view in apply_state["constraints"].constraints
    }
    checks.check(
        "APPLY state contains the seeded and the trusted authorized hard constraints",
        apply_constraint_pairs
        == {SEEDED_CONSTRAINT, ("price_max", TRUSTED_CONSTRAINT["value"])},
        str(sorted(apply_constraint_pairs)),
    )
    checks.check(
        "APPLY diagnostics state_digest equals an independent replay",
        apply_diagnostics.get("state_digest") == apply_state["digest"],
    )
    print(f"  recommendations before (OFF)   : {off_recommendations}")
    print(f"  recommendations after (APPLY)  : {apply_recommendations}")
    print(f"  eligibility requirements       : {_canonical_bounded(apply_requirements)}")
    print(f"  state events by type           : {apply_types}")
    print(f"  state digest                   : {apply_state['digest']}")
    print(f"  diagnostics                    : {_canonical_bounded(_diagnostics_summary(apply_diagnostics))}")
    result["modes"]["apply_constraints"] = {
        "mode": "apply_constraints",
        "seeded_constraint": list(SEEDED_CONSTRAINT),
        "recommendations_before": off_recommendations,
        "recommendations_after": apply_recommendations,
        "eligibility_evaluated": apply_body["eligibility"]["evaluated"],
        "eligibility_requirements": apply_requirements,
        "events_by_type": apply_types,
        "event_count": len(apply_state["events"]),
        "state_digest": apply_state["digest"],
        "diagnostics": _diagnostics_summary(apply_diagnostics),
    }

    # -- inferred hypothesis only: no candidate may be filtered ------------------ #
    print("\n-- APPLY_CONSTRAINTS (inferred hypothesis only) " + "-" * 28)
    hypothesis_root = root / "hypothesis"
    hypothesis_root.mkdir(parents=True)
    hypothesis_db = hypothesis_root / "decision_state.sqlite3"
    _set_env("apply_constraints", hypothesis_db)
    hypothesis_harness = build_harness(hypothesis_root)
    try:
        hypothesis_session = _create_session(hypothesis_harness)
        hypothesis_user_key = hypothesis_harness.runtime.sessions.get(
            hypothesis_session
        ).user_key
        _seed_state(
            hypothesis_db,
            hypothesis_user_key,
            hypothesis_session,
            [
                hypothesis_event(
                    user_key=hypothesis_user_key,
                    task_id=hypothesis_session,
                    session_id=hypothesis_session,
                    kind=SEEDED_HYPOTHESIS[0],
                    value=SEEDED_HYPOTHESIS[1],
                )
            ],
        )
        hypothesis_response = _shopping_turn(hypothesis_harness, hypothesis_session)
        hypothesis_service = hypothesis_harness.runtime.agent_service()
        hypothesis_diagnostics = hypothesis_service.state_diagnostics(hypothesis_session)
    finally:
        close_harness(hypothesis_harness)
    hypothesis_body = hypothesis_response.json()
    hypothesis_recommendations = _recommendation_identities(hypothesis_body)
    hypothesis_state = _read_state(hypothesis_db, hypothesis_user_key, hypothesis_session)
    hypothesis_requirements = list(hypothesis_body["eligibility"]["requirements"])
    hypothesis_authorized = {
        (view.kind.value, view.value) for view in hypothesis_state["constraints"].constraints
    }
    hypothesis_no_filter = (
        hypothesis_response.status_code == 200
        # The fixed request's own trusted constraint still narrows eligibility to the same
        # three candidates as OFF; the inference adds no fourth requirement and no filter.
        and hypothesis_recommendations == off_recommendations
        and hypothesis_requirements
        == [{"kind": "price_max", "expected": TRUSTED_CONSTRAINT["value"]}]
        and SEEDED_HYPOTHESIS not in hypothesis_authorized
        and len(hypothesis_state["state"].inferred_hypotheses) == 1
        and _normalised_response(off_body) == _normalised_response(hypothesis_body)
    )
    checks.check(
        "an inferred hypothesis never filters candidates (no hard constraint applied)",
        hypothesis_no_filter,
        f"recs={hypothesis_recommendations}, requirements={hypothesis_requirements}, "
        f"authorized={sorted(hypothesis_authorized)}",
    )
    checks.check(
        "the inferred hypothesis is recorded as soft evidence, never in for_constraints",
        len(hypothesis_state["state"].inferred_hypotheses) == 1
        and [
            (item.kind.value, item.value)
            for item in hypothesis_state["state"].inferred_hypotheses
        ]
        == [SEEDED_HYPOTHESIS]
        and SEEDED_HYPOTHESIS not in hypothesis_authorized
        and hypothesis_state["constraints"].excluded_hypothesis_count >= 1,
    )
    print(f"  seeded inferred hypothesis     : {list(SEEDED_HYPOTHESIS)}")
    print(f"  recommendations                : {hypothesis_recommendations}")
    print(f"  eligibility requirements       : {_canonical_bounded(hypothesis_requirements)}")
    print(f"  no-filter verdict              : {'PASS' if hypothesis_no_filter else 'FAIL'}")
    result["modes"]["inferred_hypothesis"] = {
        "seeded_hypothesis": list(SEEDED_HYPOTHESIS),
        "recommendations": hypothesis_recommendations,
        "eligibility_requirements": hypothesis_requirements,
        "inferred_hypothesis_count": len(hypothesis_state["state"].inferred_hypotheses),
        "excluded_hypothesis_count": hypothesis_state["constraints"].excluded_hypothesis_count,
        "no_filter_verdict": hypothesis_no_filter,
    }

    # -- replay after reopen (the "restart" check) ------------------------------- #
    print("\n-- replay after reopen " + "-" * 50)
    reopened = _read_state(shadow_db, shadow_user_key, shadow_session)
    reopened_again = _read_state(shadow_db, shadow_user_key, shadow_session)
    replay_matches = (
        reopened["digest"]
        == reopened_again["digest"]
        == shadow_digest
        == shadow_diagnostics.get("state_digest")
    )
    checks.check(
        "replaying the stored events after reopen reproduces the diagnostics digest",
        replay_matches,
        f"{reopened['digest'][:16]}...",
    )
    checks.check(
        "replayed event count equals the recorded event count",
        len(reopened["events"]) == shadow_diagnostics.get("event_count"),
    )
    print(f"  digest (incremental/diagnostics) : {shadow_digest}")
    print(f"  digest (reopen, replay)          : {reopened['digest']}")
    result["replay_after_reopen"] = {
        "diagnostics_digest": shadow_diagnostics.get("state_digest"),
        "replay_digest": reopened["digest"],
        "replay_digest_again": reopened_again["digest"],
        "matches": replay_matches,
    }

    # -- summary ----------------------------------------------------------------- #
    result["checks"] = checks.as_list()
    result["passed"] = checks.passed
    print("\n" + "=" * 78)
    print("M5.1 shopping-loop unified decision state smoke")
    print("Synthetic fixture; no model, dataset, network, LLM or quality claim.")
    print(f"checks: {sum(1 for _, ok, _ in checks.results if ok)}/{len(checks.results)} passed")
    print(f"SMOKE: {'PASS' if checks.passed else 'FAIL'}")
    return checks.passed, result


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Run the smoke; returns 0 only when every check passes."""
    parser = argparse.ArgumentParser(
        description=(
            "M5.1 synthetic shopping-loop smoke: OFF / SHADOW / APPLY_CONSTRAINTS "
            "and inferred-hypothesis no-filter verdict"
        )
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the complete machine-readable result to this path",
    )
    args = parser.parse_args(argv)

    print("=" * 78)
    print("M5.1 shopping-loop unified decision state smoke")
    print("Offline and synthetic: no model checkpoint, dataset, network or LLM.")
    print("No recommendation-quality claim is made.")
    print("=" * 78)

    root = Path(tempfile.mkdtemp(prefix="agentrecx_m5_1_smoke_"))
    print(f"workspace: {root} (temporary; removed on exit)")
    original_uuid4 = uuid.uuid4
    # Deterministic identity stream (see the module docstring): makes session/task ids,
    # event ids and state digests identical on every run of this synthetic experiment.
    uuid.uuid4 = _deterministic_uuid4()  # type: ignore[assignment]
    try:
        try:
            passed, result = run(root)
        except Exception:  # pragma: no cover - defensive smoke boundary
            traceback.print_exc()
            print("\nSMOKE: FAIL (unexpected error)")
            return 1

        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {args.json}")
        return 0 if passed else 1
    finally:
        uuid.uuid4 = original_uuid4  # type: ignore[assignment]
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
