"""M5.2 interaction-feedback-loop smoke (synthetic, offline, deterministic).

This is a runnable end-to-end demonstration of the M5.2 feedback loop on the multi-turn
shopping route; it is **not** an evaluation and makes **no recommendation-quality claim**.
Everything is synthetic and local: the fixture engine/catalogue from
:mod:`tests.demo_fixture` (no checkpoint), no dataset, no network, no LLM, no provider key
and no server process.  The only artefacts are a temporary directory and, when asked for,
a JSON summary.

Six-turn scenario
-----------------
1. ``"I want a hiking backpack under $150."`` with ``preference_discovery=True`` — the
   trusted budget/category facts are recorded as M5 events and the agent asks a
   candidate-aware preference question.
2. The agent's question is pending; the state must NOT contain a fabricated preference for
   the asked dimension.
3. The user answers with the real structured option ``pref:weight:lower``; M5 records one
   explicit TASK preference ``feature=weight`` with ``direction=minimize`` and
   diagnostics ``feedback_kind=clarification_option``.
4. ``"Actually, I also prefer lightweight gear."`` — a second explicit preference is
   recorded; the earlier confirmed weight preference is not deleted.
5. ``"Forget my lightweight preference."`` — the accepted M9 REMOVE is resolved against
   the task state and becomes ``PREFERENCE_RETRACTED``; the belief is RETRACTED/REMOVED
   and no projection contains it.
6. Another recommendation request — the next turn sees the effective updated state
   (``for_agent`` / ``for_ranker`` contain the active beliefs and exclude the retracted
   one) and the presented answer records trusted candidate/ranking snapshot references.

Checks performed
----------------
* the event stream by type (bounded, TASK-scoped, attributed);
* state digest: incremental diagnostics digest == replay after reopening SQLite (twice);
* no cross-task leakage between two sessions of the same run;
* duplicate-retry idempotency with the same client ``idempotency_key`` (same event ids,
  count and digest; no duplicate row);
* candidate/ranking snapshot provenance after presented turns (opaque refs, counts,
  digests, ``snapshot_schema_version=1``, trusted actor, no product identity or score, and
  the state references match);
* APPLY_CONSTRAINTS view consistency (the enforced M5 constraint appears in
  ``intent.hard`` with ``origin="decision_state"`` and agrees with
  ``eligibility.requirements``);
* default OFF-mode parity for the same first-turn request (volatile fields normalised)
  and OFF creating no decision-state database.

Reproducibility
---------------
The messages, catalogue, candidate order and policy decisions are fixed, and the demo
session identities (normally random UUIDs) come from a deterministic ``uuid.uuid4`` stream
for the duration of the run (a synthetic-experiment reproducibility control, equivalent to
fixing a random seed; no product code is changed).  Run-to-run event ids and state digests
are therefore identical.

Usage::

    python -m experiments.m5_2_feedback_loop_smoke
    python -m experiments.m5_2_feedback_loop_smoke --json /tmp/m5_2_smoke.json

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

# Third-party notice from Starlette's TestClient (imported by the offline harness); a
# module run would otherwise print it to stderr, which adds nothing to this smoke's
# evidence.
warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
    category=UserWarning,
)

from recommendation.state import (  # noqa: E402
    PreferenceDirection,
    PreferenceLifecycle,
    PreferenceScope,
    PreferenceStatus,
    SQLiteEventStore,
    StateManager,
    for_agent,
    for_ranker,
)
from tests.demo_fixture import PROFILE_A, PROFILE_B, build_harness, close_harness  # noqa: E402

#: Documented M5.2 configuration surface.
MODE_ENV = "AGENTRECX_STATE_MODE"
DB_ENV = "AGENTRECX_STATE_DB"

SHOPPING_ROUTE = "/v1/demo/sessions/{session}/agent/shopping"
SESSIONS_ROUTE = "/v1/demo/sessions"

#: The six-turn messages.
T1_MESSAGE = "I want a hiking backpack under $150."
T4_MESSAGE = "Actually, I also prefer lightweight gear."
T5_MESSAGE = "Forget my lightweight preference."
T6_MESSAGE = "Recommend some hiking backpacks."

K = 4
SOURCES = ("history", "catalog_search")

#: M5.2 event vocabulary on the shopping route (M5.1 fact events + retraction + trusted
#: snapshot references).
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
        "preference_retracted",
        "candidate_pool_referenced",
        "ranking_snapshot_referenced",
    }
)

#: The expected event stream of the deterministic six-turn scenario, by type.
EXPECTED_EVENT_STREAM = {
    "task_context_updated": 5,       # one per served turn (T1, T3, T4, T5, T6)
    "user_preference_stated": 4,     # T1 category + T1 price_max + T3 weight + T4 lightweight
    "user_constraint_stated": 4,     # T1/T4/T5/T6 authorized price_max=150 statement
    "preference_retracted": 1,       # T5 lightweight removal
    "candidate_pool_referenced": 3,  # presented turns T4, T5, T6
    "ranking_snapshot_referenced": 3,
}

#: Session-derived / controller-generated volatile identities normalised for parity.
_SESSION_TURN_ID = re.compile(r"^demo-session:[0-9a-f-]{36}:[0-9a-f0-9]{1,32}$")
_MASKED_KEYS = frozenset(
    {
        "timing",
        "run_id",
        "run_ref",
        "source_turn_id",
        "turn_ids",
        # A pending clarification id is a server-generated capability token, not behaviour.
        "clarification_id",
    }
)
_JSON_LIMIT = 720


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


class Checks:
    """Collect named boolean checks so the exit code reflects real evidence."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: Any, detail: str = "") -> bool:
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


def _deterministic_uuid4():
    """Return a deterministic ``uuid4`` replacement (sequential UUID integers)."""
    counter = itertools.count(1)

    def _uuid4() -> uuid.UUID:
        return uuid.UUID(int=next(counter))

    return _uuid4


def _create_session(harness: Any, profile: str = PROFILE_A) -> str:
    response = harness.client.post(SESSIONS_ROUTE, json={"profile_id": profile})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _post(harness: Any, session_id: str, **body: Any) -> Any:
    """POST one shopping turn (default: a plain request with no constraints)."""
    payload: dict[str, Any] = {
        "message": "",
        "k": K,
        "sources": list(SOURCES),
        "constraints": [],
    }
    payload.update(body)
    return harness.client.post(
        SHOPPING_ROUTE.format(session=session_id), json=payload
    )


def _identity(harness: Any, session_id: str) -> tuple[str, str]:
    session = harness.runtime.sessions.get(session_id)
    return session.user_key, session.session_id


def _read_state(database: Path, user_key: str, task_id: str) -> dict[str, Any]:
    """Replay one task from a fresh store (i.e. after a reopen / restart)."""
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
            "event_count": state.interaction.event_count,
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
    """Mask wall-clock and server-generated identity fields for the parity verdict."""
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
    """Drop ``timing`` and mask server/controller-generated identities only."""
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
    """Deterministic subset of the bounded diagnostics record (no server-owned identity)."""
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
        "turn_identity_source",
        "feedback_kind",
        "retraction_match",
        "legacy_m5",
        "snapshot",
        "error",
    )
    return {key: diagnostics[key] for key in keep if key in diagnostics}


def _recommendation_identities(body: dict[str, Any]) -> list[str]:
    return [str(item["parent_asin"]) for item in body["recommendations"]]


def _belief_identities(state: Any) -> set[tuple[str, str]]:
    return {(belief.kind.value, belief.value) for belief in state.active_preference_beliefs}


def _store_events(database: Path, user_key: str, task_id: str) -> list[Any]:
    store = SQLiteEventStore(database)
    try:
        return list(store.get_events(task_id, user_key))
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# the scenario
# --------------------------------------------------------------------------- #


def run(root: Path) -> tuple[bool, dict[str, Any]]:
    """Run the six-turn M5.2 scenario; return ``(passed, machine-readable result)``."""
    checks = Checks()
    result: dict[str, Any] = {
        "milestone": "M5.2",
        "kind": "synthetic interaction-feedback-loop smoke",
        "quality_claim": "none",
        "synthetic": True,
        "scenario": {
            "turns": [T1_MESSAGE, "<agent preference question>", "option pref:weight:lower",
                      T4_MESSAGE, T5_MESSAGE, T6_MESSAGE],
            "k": K,
            "sources": list(SOURCES),
        },
        "turns": {},
    }

    # -- T1 reference: default OFF behaves as before and writes no state ------------- #
    print("\n-- OFF (default): first-turn parity reference " + "-" * 28)
    off_root = root / "off"
    off_root.mkdir(parents=True)
    off_db = off_root / "decision_state.sqlite3"
    _set_env(None, off_db)
    off_harness = build_harness(off_root)
    try:
        off_session = _create_session(off_harness)
        off_response = _post(
            off_harness, off_session, message=T1_MESSAGE, preference_discovery=True
        )
        assert off_response.status_code == 200, off_response.text
        off_body = off_response.json()
        off_service = off_harness.runtime.agent_service()
        checks.check("OFF first turn status is 200", off_response.status_code == 200)
        checks.check(
            "OFF composes mode 'off'", off_service.state_mode == "off", str(off_service.state_mode)
        )
        checks.check(
            "OFF records no state diagnostics",
            off_service.state_diagnostics(off_session) is None,
        )
        checks.check("OFF opens no decision-state database", not off_db.exists())
        checks.check(
            "OFF asks the same trusted preference question",
            (off_body.get("clarification") or {}).get("kind") == "preference",
        )
    finally:
        close_harness(off_harness)

    # -- SHADOW: the six-turn scenario ------------------------------------------------ #
    print("\n-- SHADOW: six-turn interaction feedback loop " + "-" * 30)
    shadow_root = root / "shadow"
    shadow_root.mkdir(parents=True)
    shadow_db = shadow_root / "decision_state.sqlite3"
    _set_env("shadow", shadow_db)
    shadow_harness = build_harness(shadow_root)
    shadow_session = ""
    shadow_user_key = ""
    shadow_task_id = ""
    try:
        shadow_session = _create_session(shadow_harness)
        shadow_user_key, shadow_task_id = _identity(shadow_harness, shadow_session)
        service = shadow_harness.runtime.agent_service()

        def diagnostics() -> dict[str, Any] | None:
            return service.state_diagnostics(shadow_task_id)

        # ---- Turn 1: budget request + preference discovery --------------------- #
        turn1 = _post(
            shadow_harness,
            shadow_session,
            message=T1_MESSAGE,
            preference_discovery=True,
        )
        assert turn1.status_code == 200, turn1.text
        turn1_body = turn1.json()
        clarification = turn1_body.get("clarification") or {}
        checks.check(
            "T1 SHADOW composes mode 'shadow'", service.state_mode == "shadow"
        )
        checks.check(
            "T1 request asks a candidate-aware preference question",
            clarification.get("kind") == "preference",
            str(clarification.get("kind")),
        )
        checks.check(
            "T1 question is about a trusted catalogue dimension",
            (clarification.get("preference") or {}).get("attribute") in {"price", "weight", "color"},
            str((clarification.get("preference") or {}).get("attribute")),
        )
        checks.check(
            "T1 OFF-vs-SHADOW user-visible parity (volatile ids/timing masked)",
            _normalised_response(off_body) == _normalised_response(turn1_body),
        )

        state1 = _read_state(shadow_db, shadow_user_key, shadow_task_id)
        types1 = _event_type_counts(state1["events"])
        print(f"  T1 events by type       : {types1}")
        print(f"  T1 active beliefs       : {sorted(_belief_identities(state1['state']))}")
        print(f"  T1 constraints          : {[(c.kind.value, c.value) for c in state1['state'].authorized_hard_constraints]}")
        checks.check(
            "T1 recorded the trusted category + budget facts and the budget constraint",
            _belief_identities(state1["state"])
            >= {("category", "hiking backpack under"), ("price_max", "150")}
            and [
                (c.kind.value, c.value)
                for c in state1["state"].authorized_hard_constraints
            ]
            == [("price_max", "150")],
            str(sorted(_belief_identities(state1["state"]))),
        )
        checks.check(
            "T1 events are the four fact events (no snapshot: the answer is a question)",
            types1
            == {
                "task_context_updated": 1,
                "user_preference_stated": 2,
                "user_constraint_stated": 1,
            },
            str(types1),
        )
        checks.check(
            "T1 diagnostics record explicit_statement feedback",
            (diagnostics() or {}).get("feedback_kind") == "explicit_statement",
            str((diagnostics() or {}).get("feedback_kind")),
        )
        checks.check(
            "T1 recorded no product identity, score or snapshot reference",
            state1["state"].candidate_pool_ref is None
            and state1["state"].ranking_snapshot_ref is None,
        )

        # ---- Turn 2: the clarification is pending; nothing was fabricated ----- #
        print("\n-- Turn 2: pending preference clarification " + "-" * 33)
        options = list((clarification.get("preference") or {}).get("options") or ())
        lower = next(
            (option for option in options if option.get("action") == "lower"), None
        )
        checks.check(
            "T2 a pending structured clarification with executable options is recorded",
            bool(clarification.get("clarification_id"))
            and clarification.get("blocks") == "preference"
            and lower is not None
            and lower.get("option_id", "").startswith("pref:"),
            str((clarification.get("preference") or {}).get("attribute")),
        )
        checks.check(
            "T2 no fabricated preference fact exists before the user answers",
            ("feature", str((clarification.get("preference") or {}).get("attribute")))
            not in _belief_identities(state1["state"]),
            str(sorted(_belief_identities(state1["state"]))),
        )
        result["turns"]["turn1_question"] = {
            "message": T1_MESSAGE,
            "status": turn1_body["terminal"]["status"],
            "reason": turn1_body["terminal"].get("termination_reason"),
            "clarification_kind": clarification.get("kind"),
            "dimension": (clarification.get("preference") or {}).get("attribute"),
            "options": [option.get("option_id") for option in options],
            "events_by_type": types1,
            "diagnostics": _diagnostics_summary(diagnostics()),
            "off_vs_shadow_parity": _normalised_response(off_body)
            == _normalised_response(turn1_body),
        }
        assert lower is not None

        # ---- Turn 3: answer with the trusted structured option ----------------- #
        print("\n-- Turn 3: structured option answer " + "-" * 37)
        turn3 = _post(
            shadow_harness,
            shadow_session,
            message=str(lower.get("label") or ""),
            clarification_id=clarification.get("clarification_id"),
            clarification_option=lower.get("option_id"),
            preference_discovery=True,
        )
        assert turn3.status_code == 200, turn3.text
        turn3_body = turn3.json()
        diagnostics3 = diagnostics() or {}
        state3 = _read_state(shadow_db, shadow_user_key, shadow_task_id)
        weight_beliefs = [
            belief
            for belief in state3["state"].active_preference_beliefs
            if belief.kind.value == "feature" and belief.value == "weight"
        ]
        checks.check(
            "T3 the structured answer records exactly one explicit TASK weight belief",
            len(weight_beliefs) == 1
            and weight_beliefs[0].direction is PreferenceDirection.MINIMIZE
            and weight_beliefs[0].scope is PreferenceScope.TASK
            and weight_beliefs[0].lifecycle is PreferenceLifecycle.EXPLICIT,
            str(
                [
                    (belief.kind.value, belief.value)
                    for belief in state3["state"].active_preference_beliefs
                ]
            ),
        )
        checks.check(
            "T3 diagnostics record clarification_option feedback",
            diagnostics3.get("feedback_kind") == "clarification_option",
            str(diagnostics3.get("feedback_kind")),
        )
        ranker3 = {
            (feature.kind.value, feature.value): feature
            for feature in state3["ranker"].features
        }
        checks.check(
            "T3 for_ranker exposes the trusted minimize direction",
            ("feature", "weight") in ranker3
            and ranker3[("feature", "weight")].direction is PreferenceDirection.MINIMIZE
            and ranker3[("feature", "weight")].is_soft is False,
        )
        checks.check(
            "T3 no snapshot is fabricated for a turn that asked another question",
            state3["state"].candidate_pool_ref is None
            or not turn3_body.get("recommendations"),
        )
        result["turns"]["turn3_answer"] = {
            "option_id": lower.get("option_id"),
            "status": turn3_body["terminal"]["status"],
            "next_clarification": (turn3_body.get("clarification") or {}).get("kind"),
            "active_beliefs": sorted(_belief_identities(state3["state"])),
            "diagnostics": _diagnostics_summary(diagnostics3),
        }
        weight_belief_id = weight_beliefs[0].belief_id if weight_beliefs else None

        # ---- Turn 4: a new explicit preference, earlier one retained ----------- #
        print("\n-- Turn 4: add lightweight without deleting weight " + "-" * 25)
        turn4 = _post(shadow_harness, shadow_session, message=T4_MESSAGE)
        assert turn4.status_code == 200, turn4.text
        turn4_body = turn4.json()
        state4 = _read_state(shadow_db, shadow_user_key, shadow_task_id)
        identities4 = _belief_identities(state4["state"])
        lightweight = [
            belief
            for belief in state4["state"].active_preference_beliefs
            if belief.value == "lightweight"
        ]
        weight_after_t4 = [
            belief
            for belief in state4["state"].active_preference_beliefs
            if belief.kind.value == "feature" and belief.value == "weight"
        ]
        checks.check(
            "T4 the new explicit lightweight preference is recorded",
            len(lightweight) == 1
            and lightweight[0].lifecycle is PreferenceLifecycle.EXPLICIT
            and lightweight[0].status is PreferenceStatus.ACTIVE,
            str(sorted(identities4)),
        )
        checks.check(
            "T4 the earlier confirmed weight preference is NOT deleted",
            len(weight_after_t4) == 1 and weight_after_t4[0].belief_id == weight_belief_id,
        )
        checks.check(
            "T4 presented answer recorded trusted candidate/ranking snapshot references",
            state4["state"].candidate_pool_ref is not None
            and state4["state"].ranking_snapshot_ref is not None
            and state4["state"].candidate_pool_ref.snapshot_schema_version == 1
            and state4["state"].ranking_snapshot_ref.snapshot_schema_version == 1,
        )
        checks.check(
            "T4 snapshot references are read-only and produced by trusted components",
            state4["state"].candidate_pool_ref.read_only is True
            and state4["state"].ranking_snapshot_ref.read_only is True
            and state4["state"].candidate_pool_ref.source == "shopping_candidate_pool"
            and state4["state"].ranking_snapshot_ref.source == "shopping_presented_order",
        )
        result["turns"]["turn4_add"] = {
            "message": T4_MESSAGE,
            "status": turn4_body["terminal"]["status"],
            "active_beliefs": sorted(identities4),
            "snapshot": _diagnostics_summary(diagnostics()).get("snapshot")
            if diagnostics()
            else None,
        }

        # ---- Turn 5: accepted M9 REMOVE -> PREFERENCE_RETRACTED ---------------- #
        print("\n-- Turn 5: remove the lightweight preference " + "-" * 28)
        turn5 = _post(shadow_harness, shadow_session, message=T5_MESSAGE)
        assert turn5.status_code == 200, turn5.text
        turn5_body = turn5.json()
        diagnostics5 = diagnostics() or {}
        state5 = _read_state(shadow_db, shadow_user_key, shadow_task_id)
        retracted = [
            belief
            for belief in state5["state"].preference_beliefs
            if belief.value == "lightweight"
        ]
        retraction_events = [
            event
            for event in state5["events"]
            if event.event_type.value == "preference_retracted"
        ]
        checks.check(
            "T5 the accepted removal produced exactly one PREFERENCE_RETRACTED event",
            len(retraction_events) == 1
            and retraction_events[0].scope is PreferenceScope.TASK
            and retraction_events[0].actor.value == "user"
            and retraction_events[0].payload.value is not None
            and retraction_events[0].payload.value.casefold() == "lightweight",
        )
        checks.check(
            "T5 the retracted belief is RETRACTED/REMOVED and no longer active",
            len(retracted) == 1
            and retracted[0].lifecycle is PreferenceLifecycle.RETRACTED
            and retracted[0].status is PreferenceStatus.REMOVED
            and retracted[0].is_active is False,
            str([(b.value, b.status.value) for b in state5["state"].preference_beliefs]),
        )
        checks.check(
            "T5 the value-only selector was resolved uniquely against the task state",
            diagnostics5.get("retraction_match", {}).get("requested") == 1
            and diagnostics5.get("retraction_match", {}).get("resolved") == 1,
            str(diagnostics5.get("retraction_match")),
        )
        checks.check(
            "T5 diagnostics label the turn as a retraction",
            diagnostics5.get("feedback_kind") == "retraction",
            str(diagnostics5.get("feedback_kind")),
        )
        checks.check(
            "T5 the weight preference survives the retraction",
            ("feature", "weight") in _belief_identities(state5["state"]),
            str(sorted(_belief_identities(state5["state"]))),
        )
        checks.check(
            "T5 no projection contains the retracted value",
            all(
                item.value.casefold() != "lightweight"
                for item in state5["agent"].preferences
            )
            and all(
                feature.value.casefold() != "lightweight"
                for feature in state5["ranker"].features
            ),
        )
        result["turns"]["turn5_remove"] = {
            "message": T5_MESSAGE,
            "status": turn5_body["terminal"]["status"],
            "active_beliefs": sorted(_belief_identities(state5["state"])),
            "retraction_match": diagnostics5.get("retraction_match"),
            "diagnostics": _diagnostics_summary(diagnostics5),
        }

        # ---- Turn 6: the next turn sees the effective updated state ------------ #
        print("\n-- Turn 6: next recommendation sees the updated state " + "-" * 19)
        turn6 = _post(shadow_harness, shadow_session, message=T6_MESSAGE)
        assert turn6.status_code == 200, turn6.text
        turn6_body = turn6.json()
        diagnostics6 = diagnostics() or {}
        state6 = _read_state(shadow_db, shadow_user_key, shadow_task_id)
        identities6 = _belief_identities(state6["state"])
        presented6 = _recommendation_identities(turn6_body)
        checks.check(
            "T6 the request is served as a finished recommendation",
            turn6_body["terminal"]["status"] == "finished" and bool(presented6),
            f"{turn6_body['terminal']['status']}/{presented6}",
        )
        checks.check(
            "T6 the next turn sees the active beliefs (weight + budget) and not the retracted one",
            ("feature", "weight") in identities6
            and not any(value.casefold() == "lightweight" for _kind, value in identities6),
            str(sorted(identities6)),
        )
        ranker6 = {
            (feature.kind.value, feature.value): feature
            for feature in state6["ranker"].features
        }
        checks.check(
            "T6 for_ranker reflects the active weight direction and excludes the retracted value",
            ("feature", "weight") in ranker6
            and ranker6[("feature", "weight")].direction is PreferenceDirection.MINIMIZE
            and all(
                feature.value.casefold() != "lightweight"
                for feature in state6["ranker"].features
            ),
        )
        checks.check(
            "T6 for_agent excludes the retracted value",
            all(
                item.value.casefold() != "lightweight"
                for item in state6["agent"].preferences
            ),
        )

        # Snapshot provenance of the presented turn.
        snapshot_events = [
            event
            for event in state6["events"]
            if event.event_type.value
            in {"candidate_pool_referenced", "ranking_snapshot_referenced"}
        ]
        snapshot_by_type: dict[str, Any] = {
            event.event_type.value: event for event in snapshot_events
        }
        pool_event = snapshot_by_type.get("candidate_pool_referenced")
        ranking_event = snapshot_by_type.get("ranking_snapshot_referenced")
        checks.check(
            "T6 presented turns recorded both candidate and ranking snapshot references",
            pool_event is not None
            and ranking_event is not None
            and pool_event.actor.value == "trusted_component"
            and ranking_event.actor.value == "trusted_component",
        )
        if pool_event is not None and ranking_event is not None:
            snapshot_payloads = json.dumps(
                [
                    pool_event.payload.model_dump(mode="json"),
                    ranking_event.payload.model_dump(mode="json"),
                ],
                sort_keys=True,
            )
            checks.check(
                "T6 snapshot payloads are opaque (ref/count/digest/version only)",
                pool_event.payload.ref.startswith("cp_")
                and ranking_event.payload.ref.startswith("rs_")
                and len(pool_event.payload.digest) == 64
                and len(ranking_event.payload.digest) == 64
                and pool_event.payload.snapshot_schema_version == 1
                and ranking_event.payload.snapshot_schema_version == 1
                and all(
                    identity not in snapshot_payloads
                    for identity in _recommendation_identities(turn6_body)
                ),
            )
            checks.check(
                "T6 the state references match the trusted snapshot events",
                state6["state"].candidate_pool_ref is not None
                and state6["state"].candidate_pool_ref.ref == pool_event.payload.ref
                and state6["state"].candidate_pool_ref.candidate_count
                == pool_event.payload.candidate_count
                and state6["state"].candidate_pool_ref.digest == pool_event.payload.digest
                and state6["state"].ranking_snapshot_ref is not None
                and state6["state"].ranking_snapshot_ref.ref
                == ranking_event.payload.ref
                and state6["state"].ranking_snapshot_ref.candidate_count
                == ranking_event.payload.candidate_count
                and state6["state"].ranking_snapshot_ref.referenced_by_event_id
                == ranking_event.event_id,
            )
        checks.check(
            "T6 diagnostics digest equals the independently replayed state digest",
            diagnostics6.get("state_digest") == state6["digest"],
        )

        types6 = _event_type_counts(state6["events"])
        print(f"  event stream by type    : {types6}")
        print(f"  incremental state digest: {state6['digest']}")
        print(f"  event count             : {state6['event_count']}")
        print(f"  diagnostics             : {_canonical_bounded(_diagnostics_summary(diagnostics6))}")
        checks.check(
            "T6 the six-turn event stream matches the expected deterministic scenario",
            types6 == EXPECTED_EVENT_STREAM,
            f"{types6} != {EXPECTED_EVENT_STREAM}",
        )
        checks.check(
            "T6 every event is TASK-scoped and attributed",
            all(
                event.scope is None or event.scope is PreferenceScope.TASK
                for event in state6["events"]
            )
            and all(
                event.event_type.value in SUPPORTED_EVENT_TYPES
                for event in state6["events"]
            ),
        )
        checks.check(
            "T6 the recorded events are deterministic (ids unique, store-ordered)",
            len({event.event_id for event in state6["events"]})
            == len(state6["events"])
            and [event.sequence for event in state6["events"]]
            == sorted(event.sequence for event in state6["events"]),
        )

        final_digest = state6["digest"]
        final_event_count = state6["event_count"]
        result["turns"]["turn6_next"] = {
            "message": T6_MESSAGE,
            "status": turn6_body["terminal"]["status"],
            "recommendations": presented6,
            "active_beliefs": sorted(identities6),
            "events_by_type": types6,
            "state_digest": final_digest,
            "event_count": final_event_count,
            "snapshot": _diagnostics_summary(diagnostics6).get("snapshot")
            if diagnostics6
            else None,
            "diagnostics": _diagnostics_summary(diagnostics6),
        }
    finally:
        close_harness(shadow_harness)

    # -- replay after reopen ------------------------------------------------------- #
    print("\n-- replay after reopen " + "-" * 56)
    replayed = _read_state(shadow_db, shadow_user_key, shadow_task_id)
    replayed_again = _read_state(shadow_db, shadow_user_key, shadow_task_id)
    checks.check(
        "reopening SQLite reproduces the incremental diagnostics digest",
        replayed["digest"] == replayed_again["digest"] == final_digest,
        f"{replayed['digest'][:16]}...",
    )
    checks.check(
        "replayed event count equals the incremental event count",
        replayed["event_count"] == replayed_again["event_count"] == final_event_count,
        str(final_event_count),
    )
    checks.check(
        "replayed state exposes the same trusted snapshot references",
        replayed["state"].candidate_pool_ref is not None
        and replayed["state"].ranking_snapshot_ref is not None
        and replayed["state"].candidate_pool_ref.snapshot_schema_version == 1,
    )
    print(f"  digest (incremental/diagnostics) : {final_digest}")
    print(f"  digest (reopen, replay)          : {replayed['digest']}")
    result["replay_after_reopen"] = {
        "diagnostics_digest": final_digest,
        "replay_digest": replayed["digest"],
        "replay_digest_again": replayed_again["digest"],
        "event_count": replayed["event_count"],
        "matches": replayed["digest"] == replayed_again["digest"] == final_digest,
    }

    # -- idempotency and cross-task isolation ------------------------------------- #
    print("\n-- idempotency + cross-task isolation " + "-" * 37)
    isolation_root = root / "isolation"
    isolation_root.mkdir(parents=True)
    isolation_db = isolation_root / "decision_state.sqlite3"
    _set_env("shadow", isolation_db)
    isolation_harness = build_harness(isolation_root)
    try:
        first_session = _create_session(isolation_harness, PROFILE_A)
        second_session = _create_session(isolation_harness, PROFILE_B)
        first_key, first_task = _identity(isolation_harness, first_session)
        second_key, second_task = _identity(isolation_harness, second_session)
        checks.check(
            "I1 two sessions are distinct users and tasks",
            first_key != second_key and first_task != second_task,
        )
        retry_body = dict(
            message="Recommend some hiking gear.",
            constraints=[{"kind": "price_max", "value": "60"}],
            idempotency_key="m5-2-smoke-retry",
        )
        retry_service = isolation_harness.runtime.agent_service()
        first_try = _post(isolation_harness, first_session, **retry_body)
        assert first_try.status_code == 200, first_try.text
        first_events = _store_events(isolation_db, first_key, first_task)
        first_ids = {event.event_id for event in first_events}
        first_digest = retry_service.state_diagnostics(first_task)["state_digest"]
        first_count = retry_service.state_diagnostics(first_task)["event_count"]
        checks.check(
            "I2 the idempotency-keyed turn records server-provided retry identity",
            retry_service.state_diagnostics(first_task)["turn_identity_source"]
            == "idempotency_key",
            str(retry_service.state_diagnostics(first_task)["turn_identity_source"]),
        )
        second_try = _post(isolation_harness, first_session, **retry_body)
        assert second_try.status_code == 200, second_try.text
        second_events = _store_events(isolation_db, first_key, first_task)
        second_diagnostics = retry_service.state_diagnostics(first_task)
        checks.check(
            "I3 a retried logical turn re-emits the same event ids and appends nothing",
            {event.event_id for event in second_events} == first_ids
            and len(second_events) == len(first_events)
            and second_diagnostics["event_count"] == first_count
            and second_diagnostics["state_digest"] == first_digest,
            f"{len(first_events)} events / {first_digest[:16]}...",
        )

        other_response = _post(
            isolation_harness,
            second_session,
            message="Recommend some hiking gear.",
            constraints=[{"kind": "price_max", "value": "60"}],
        )
        assert other_response.status_code == 200, other_response.text
        second_key_state = _read_state(isolation_db, second_key, second_task)
        first_key_state = _read_state(isolation_db, first_key, first_task)
        checks.check(
            "I4 cross-task isolation: neither task's events appear in the other's stream",
            not set(first_ids) & {event.event_id for event in second_key_state["events"]}
            and all(event.task_id == first_task for event in first_events)
            and all(event.task_id == second_task for event in second_key_state["events"]),
        )
        checks.check(
            "I5 the two state projections do not share a state",
            first_key_state["digest"] != second_key_state["digest"]
            and first_key_state["state"].user_key != second_key_state["state"].user_key,
        )
        result["idempotency_and_isolation"] = {
            "retry_event_ids": sorted(first_ids),
            "retry_event_count": len(first_events),
            "retry_state_digest": first_digest,
            "turn_identity_source": second_diagnostics["turn_identity_source"],
            "second_turn_event_count": len(second_events),
        }
    finally:
        close_harness(isolation_harness)

    # -- APPLY_CONSTRAINTS effective-constraint view consistency ------------------ #
    print("\n-- APPLY_CONSTRAINTS effective-constraint view " + "-" * 30)
    apply_root = root / "apply"
    apply_root.mkdir(parents=True)
    apply_db = apply_root / "decision_state.sqlite3"
    _set_env("apply_constraints", apply_db)
    apply_harness = build_harness(apply_root)
    try:
        apply_session = _create_session(apply_harness, PROFILE_A)
        apply_key, apply_task = _identity(apply_harness, apply_session)
        first_turn = _post(
            apply_harness,
            apply_session,
            message="Recommend some hiking gear.",
            constraints=[{"kind": "price_max", "value": "30"}],
        )
        assert first_turn.status_code == 200, first_turn.text
        first_body = first_turn.json()
        first_hard = {
            (entry["kind"], entry["value"]): entry
            for entry in (first_body.get("intent") or {}).get("hard", ())
        }
        checks.check(
            "A1 the current-turn constraint is enforced and reported as user_stated",
            first_hard.get(("price_max", "30"), {}).get("origin") == "user_stated"
            and {"kind": "price_max", "expected": "30"}
            in (first_body.get("eligibility") or {}).get("requirements", ()),
            _canonical_bounded(list(first_hard)),
        )
        first_recs = _recommendation_identities(first_body)
        checks.check(
            "A1 APPLY narrowed the candidates through the existing eligibility evaluator",
            first_recs == ["cand-red", "cand-black"]
            and first_body["eligibility"]["ineligible_count"] == 2,
            str(first_recs),
        )

        second_turn = _post(
            apply_harness, apply_session, message="Recommend some hiking gear."
        )
        assert second_turn.status_code == 200, second_turn.text
        second_body = second_turn.json()
        second_hard = {
            (entry["kind"], entry["value"]): entry
            for entry in (second_body.get("intent") or {}).get("hard", ())
        }
        requirements = [
            (item.get("kind"), item.get("expected"))
            for item in (second_body.get("eligibility") or {}).get("requirements", ())
        ]
        eligibility2 = second_body.get("eligibility") or {}
        checks.check(
            "A2 the authorized M5 constraint is reported with origin decision_state",
            second_hard.get(("price_max", "30"), {}).get("origin") == "decision_state",
            _canonical_bounded(list(second_hard)),
        )
        checks.check(
            "A3 intent.hard and eligibility.requirements agree on the enforced constraint",
            ("price_max", "30") in requirements
            and second_hard.get(("price_max", "30"), {}).get("value") == "30",
            _canonical_bounded(requirements),
        )
        checks.check(
            "A4 the narrowed candidates and the three-state eligibility sum agree",
            _recommendation_identities(second_body) == ["cand-red", "cand-black"]
            and eligibility2["verified_eligible_count"]
            + eligibility2["ineligible_count"]
            + eligibility2["unresolved_count"]
            == len(eligibility2["assessments"]),
            str(_recommendation_identities(second_body)),
        )
        apply_state = _read_state(apply_db, apply_key, apply_task)
        checks.check(
            "A5 the state and its authorized projection hold exactly price_max=30",
            [
                (constraint.kind.value, constraint.value)
                for constraint in apply_state["state"].authorized_hard_constraints
            ]
            == [("price_max", "30")],
        )
        apply_diagnostics = apply_harness.runtime.agent_service().state_diagnostics(apply_task)
        checks.check(
            "A6 APPLY diagnostics are clean and describe the replayed state",
            apply_diagnostics.get("error") is None
            and apply_diagnostics.get("state_digest") == apply_state["digest"]
            and apply_diagnostics.get("turn_identity_source") == "server_turn_id",
        )
        result["apply_view_consistency"] = {
            "first_recs": first_recs,
            "first_constraint_origin": first_hard.get(("price_max", "30"), {}).get("origin"),
            "second_recs": _recommendation_identities(second_body),
            "second_constraint_origin": second_hard.get(("price_max", "30"), {}).get("origin"),
            "eligibility_requirements": requirements,
            "state_digest": apply_state["digest"],
            "diagnostics": _diagnostics_summary(apply_diagnostics),
        }
    finally:
        close_harness(apply_harness)

    # -- summary ------------------------------------------------------------------ #
    result["checks"] = checks.as_list()
    result["passed"] = checks.passed
    print("\n" + "=" * 78)
    print("M5.2 interaction feedback loop smoke")
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
            "M5.2 synthetic six-turn interaction feedback loop smoke: budget -> "
            "clarification -> structured answer -> add preference -> remove preference -> "
            "updated-state recommendation, plus idempotency, snapshots and APPLY consistency"
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
    print("M5.2 interaction feedback loop smoke")
    print("Offline and synthetic: no model checkpoint, dataset, network or LLM.")
    print("No recommendation-quality claim is made.")
    print("=" * 78)

    root = Path(tempfile.mkdtemp(prefix="agentrecx_m5_2_smoke_"))
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
