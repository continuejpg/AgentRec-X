"""M7 鈥?candidate-grounded preference construction (offline lab).

These tests cover the smallest executable contract:

* the deterministic four-action policy (ASK / COMPARE / EXPLAIN / FINALIZE) on the
  accepted synthetic catalogue;
* the independent experiment switch is **default off**, so the accepted serving/preference
  behaviour is unchanged when it is unset;
* an ASK -> answer -> FINALIZE loop, a COMPARE loop and an EXPLAIN loop through the real
  multi-turn HTTP shopping route;
* the answer updates the M5 decision state and ``for_ranker``;
* the experimental M6.1 refresh actually ran and preserves the candidate set;
* the interaction budget terminates the loop;
* no comparison/explanation value is invented: every displayed value is copied from the
  trusted catalogue.

No smoke script; these are focused tests.  The fixture catalogue is the accepted
synthetic one (cand-red/cand-blue/cand-black/cand-green).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.grounded_actions import (  # noqa: E402
    ENV_M7_ACTIONS,
    ENV_M7_BUDGET,
    GroundedAction,
    choose_grounded_action,
    grounded_actions_enabled,
    grounded_budget,
    refresh_candidate_order,
)
from recommendation.demo.preference_construction import PreferenceState  # noqa: E402
from recommendation.state import SQLiteEventStore, StateManager  # noqa: E402
from recommendation.state.state_projector import for_ranker  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS, build_index  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402

BASE = "/v1/demo"
CREATE = BASE + "/sessions"
SHOPPING = BASE + "/sessions/{session_id}/agent/shopping"
ENV_STATE_MODE = "AGENTRECX_STATE_MODE"
ENV_STATE_DB = "AGENTRECX_STATE_DB"


@contextmanager
def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    m7: str | None = "on",
    budget: int = 2,
    state_mode: str = "shadow",
    rows: Any = CANDIDATE_ROWS,
) -> Iterator[tuple[Any, Path]]:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    database = root / "decision_state.sqlite3"
    if m7 is None:
        monkeypatch.delenv(ENV_M7_ACTIONS, raising=False)
    else:
        monkeypatch.setenv(ENV_M7_ACTIONS, m7)
    monkeypatch.setenv(ENV_M7_BUDGET, str(budget))
    monkeypatch.setenv(ENV_STATE_MODE, state_mode)
    monkeypatch.setenv(ENV_STATE_DB, str(database))
    harness = build_harness(root, rows=rows, memory_db=root / "memory.sqlite3")
    try:
        yield harness, database
    finally:
        close_harness(harness)


def _create_session(harness: Any) -> str:
    response = harness.client.post(CREATE, json={"profile_id": PROFILE_A})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _shopping(harness: Any, session_id: str, message: str, **overrides: Any) -> Any:
    body: dict[str, Any] = {
        "message": message,
        "k": 3,
        "sources": ["history"],
        "preference_discovery": True,
    }
    body.update(overrides)
    return harness.client.post(SHOPPING.format(session_id=session_id), json=body)


def _task_id(harness: Any, session_id: str) -> tuple[str, str]:
    session = harness.runtime.sessions.get(session_id)
    return session.user_key, session.session_id


def _answer(harness: Any, session_id: str, body: dict[str, Any], option_id: str) -> Any:
    return _shopping(
        harness,
        session_id,
        "choose",
        clarification_id=body["clarification"]["clarification_id"],
        clarification_option=option_id,
    )


def _records() -> dict[str, Any]:
    return dict(build_index().records)


def _records_with_category(
    category: tuple[str, ...] = ("Sports & Outdoors", "Camping & Hiking", "Test Gear"),
) -> dict[str, Any]:
    """Fixture records with the same trusted category path for numeric comparisons."""
    records = _records()
    for identity, record in list(records.items()):
        model_copy = getattr(record, "model_copy", None)
        if callable(model_copy):
            records[identity] = model_copy(update={"categories": category})
    return records


def _category_for(
    records: dict[str, Any], identity: str
) -> tuple[str, ...]:
    record = records[identity]
    return tuple(record.categories)


def _ranker_state(database: Path, user_key: str, task_id: str) -> Any:
    store = SQLiteEventStore(database)
    try:
        state = StateManager(store).replay_state(user_key, task_id)
        return state, for_ranker(state)
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# 1. deterministic policy
# --------------------------------------------------------------------------- #


def test_policy_chooses_ask_compare_explain_and_finalize() -> None:
    records = _records()
    ids = ("cand-red", "cand-blue", "cand-black")
    ask = choose_grounded_action(
        ids, records, message="recommend gear in different colors", top_k=3
    )
    assert ask.action is GroundedAction.ASK
    assert ask.prompt is not None and len(ask.prompt.options) >= 3
    assert ask.user_requested is False and ask.proactive is False

    # Proactive EXPLAIN: no explicit explanation request, but a technical attribute with
    # grounded values and a material impact.
    proactive_explain = choose_grounded_action(
        ids, records, message="recommend hiking gear", top_k=3
    )
    assert proactive_explain.action is GroundedAction.EXPLAIN
    assert proactive_explain.proactive is True
    assert proactive_explain.user_requested is False
    assert proactive_explain.explanation
    assert "0.8" in proactive_explain.explanation
    assert "does not assert that you understand" in proactive_explain.explanation

    # Proactive COMPARE: no explicit comparison request, but a material numeric contrast.
    proactive_compare = choose_grounded_action(
        ids, records, message="recommend gear under a tight budget", top_k=3
    )
    assert proactive_compare.action is GroundedAction.COMPARE
    assert proactive_compare.proactive is True
    assert proactive_compare.user_requested is False
    assert proactive_compare.comparison_rows

    compare = choose_grounded_action(
        ids, records, message="compare price differences before you recommend", top_k=3
    )
    assert compare.action is GroundedAction.COMPARE
    assert compare.user_requested is True
    assert compare.comparison_rows
    assert {row["identity"] for row in compare.comparison_rows} == set(ids)
    for row in compare.comparison_rows:
        assert row["attribute"] == compare.prompt.attribute
        assert row["value"] is not None

    uncertain = choose_grounded_action(
        ids, records, message="I don't know which type of gear I need.", top_k=3
    )
    assert uncertain.action in {GroundedAction.COMPARE, GroundedAction.EXPLAIN}
    assert uncertain.uncertain is True
    assert uncertain.user_requested is False
    assert uncertain.prompt is not None
    assert uncertain.comparison_rows or uncertain.explanation

    explain = choose_grounded_action(
        ids,
        records,
        message="explain what weight means before you recommend gear",
        top_k=3,
    )
    assert explain.action is GroundedAction.EXPLAIN
    assert explain.user_requested is True
    assert explain.explanation
    assert "0.8" in explain.explanation  # the real cand-red weight, not a template guess

    exhausted = choose_grounded_action(
        ids, records, message="recommend hiking gear", top_k=3, budget_remaining=0
    )
    assert exhausted.action is GroundedAction.FINALIZE
    assert choose_grounded_action((), records).action is GroundedAction.FINALIZE


def test_question_follows_explicit_user_need_before_easy_attribute() -> None:
    records = _records()
    ids = ("cand-red", "cand-blue", "cand-black")

    lightweight = choose_grounded_action(
        ids, records, message="lightweight water filter", top_k=3
    )
    assert lightweight.prompt is not None
    assert lightweight.prompt.attribute == "weight"

    budget = choose_grounded_action(
        ids, records, message="camping cookware under a budget", top_k=3
    )
    assert budget.prompt is not None
    assert budget.prompt.attribute == "price"

    durable = choose_grounded_action(
        ids, records, message="durable hiking backpack", top_k=3
    )
    # Durability is not a trusted catalogue preference attribute: do not pretend a
    # generic material/price/weight difference answers it.
    assert durable.action is GroundedAction.FINALIZE


def test_partial_price_evidence_compares_trusted_subset() -> None:
    records = _records()
    records["cand-blue"] = records["cand-blue"].model_copy(
        update={"price_text": None}
    )
    ids = ("cand-red", "cand-blue", "cand-black")
    decision = choose_grounded_action(
        ids, records, message="camping cookware under a budget", top_k=3
    )
    assert decision.prompt is not None
    assert decision.prompt.attribute == "price"
    assert decision.prompt.products == ("cand-red", "cand-black")
    assert "2 of 3 candidates" in decision.prompt.reason
    assert "UNKNOWN" in decision.prompt.reason


def test_switch_and_budget_are_independent_and_default_off() -> None:
    assert grounded_actions_enabled(None) is False
    assert grounded_actions_enabled("off") is False
    assert grounded_actions_enabled("on") is True
    assert grounded_budget("1") == 1
    assert grounded_budget("99") == 5
    assert grounded_budget("bogus") == 2


# --------------------------------------------------------------------------- #
# 2. default off parity
# --------------------------------------------------------------------------- #


def test_default_off_does_not_use_m7_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset and explicit-off produce the accepted behaviour, not the M7 policy."""
    request = "recommend hiking gear"
    with _harness(
        tmp_path, monkeypatch, name="off-default", m7=None, state_mode="off"
    ) as (default_harness, _):
        default_session = _create_session(default_harness)
        default_body = _shopping(
            default_harness, default_session, request, preference_discovery=False
        ).json()
    with _harness(
        tmp_path, monkeypatch, name="off-explicit", m7="off", state_mode="off"
    ) as (off_harness, _):
        off_session = _create_session(off_harness)
        off_body = _shopping(
            off_harness, off_session, request, preference_discovery=False
        ).json()
    assert default_body.get("preference") is None
    assert off_body.get("preference") is None
    assert [item["parent_asin"] for item in default_body["recommendations"]] == [
        item["parent_asin"] for item in off_body["recommendations"]
    ]


# --------------------------------------------------------------------------- #
# 3. API loops (ASK / COMPARE / EXPLAIN / FINALIZE)
# --------------------------------------------------------------------------- #


def test_proactive_explain_loop_updates_state_and_refreshes_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="ask-loop", budget=1) as (harness, database):
        session_id = _create_session(harness)
        user_key, task_id = _task_id(harness, session_id)
        first = _shopping(harness, session_id, "recommend hiking gear")
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["terminal"]["status"] == "waiting_for_user"
        # Proactive EXPLAIN: no explicit explanation request in the message.
        assert body["preference"]["grounded_action"] == "EXPLAIN"
        store = SQLiteEventStore(database)
        try:
            event_types = [
                event.event_type.value for event in store.get_events(task_id, user_key)
            ]
        finally:
            store.close()
        assert "feature_explanation_shown" in event_types
        assert "feature_explanation_requested" not in event_types
        option = next(
            item for item in body["preference"]["options"] if item["action"] == "lower"
        )
        answer = _answer(harness, session_id, body, option["option_id"])
        assert answer.status_code == 200, answer.text
        final = answer.json()
        assert final["terminal"]["status"] == "finished"
        assert final["recommendations"]
        assert final.get("preference") is None
        # The accepted M5 state was updated and its for_ranker projection contains the
        # TASK-scoped directional preference.
        state, projection = _ranker_state(database, user_key, task_id)
        selected_attribute = body["preference"]["attribute"]
        features = [
            feature
            for feature in projection.features
            if feature.value == selected_attribute
        ]
        assert features, [
            (feature.kind.value, feature.value, feature.direction)
            for feature in projection.features
        ]
        assert any(
            feature.direction is not None and feature.scope.value == "task"
            for feature in features
        )
        assert any(
            belief.scope.value == "task"
            and selected_attribute in belief.value.casefold()
            for belief in state.preference_beliefs
        )
        # The experimental M6.1 refresh actually ran and kept the same candidate set.
        refresh = harness.runtime.agent_service()._m7_refresh.get(task_id) or {}
        assert refresh.get("ran") is True, refresh
        assert set(item["parent_asin"] for item in final["recommendations"]) == {
            "cand-red",
            "cand-blue",
            "cand-black",
        }


def test_compare_action_shows_only_real_values_and_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="compare-loop", budget=1) as (harness, database):
        records = harness.runtime.metadata.records
        category = ("Sports & Outdoors", "Camping & Hiking", "Test Gear")
        for identity, record in list(records.items()):
            records[identity] = record.model_copy(update={"categories": category})
        session_id = _create_session(harness)
        first = _shopping(
            harness, session_id, "compare price differences before you recommend"
        ).json()
        assert first["preference"]["grounded_action"] == "COMPARE"
        assert first["preference"]["interaction"] == "COMPARE"
        # Merely displaying the comparison is not user feedback.
        store = SQLiteEventStore(database)
        try:
            session = harness.runtime.sessions.get(session_id)
            before = [
                event.event_type.value
                for event in store.get_events(session.session_id, session.user_key)
            ]
        finally:
            store.close()
        assert "product_comparison_feedback" not in before
        values = first["preference"]["values"]
        records = harness.runtime.metadata.records
        for identity, value in values.items():
            price = records[identity].price_text
            assert price is not None
            assert float(price.replace("$", "").replace(",", "")) == float(value)
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "lower"
        )
        answer = _answer(harness, session_id, first, option["option_id"])
        assert answer.status_code == 200, answer.text
        final = answer.json()
        assert final["terminal"]["status"] == "finished"
        store = SQLiteEventStore(database)
        try:
            session = harness.runtime.sessions.get(session_id)
            after = [
                event.event_type.value
                for event in store.get_events(session.session_id, session.user_key)
            ]
        finally:
            store.close()
        assert "product_comparison_feedback" in after
        # The M7 prompt counterfactual and the M6.1 category-aware experimental refresh
        # have different comparison semantics; the diagnostic says so instead of claiming
        # a failed or forced-verified ordering.
        outcome = final["preference_outcome"]
        assert outcome["comparison_status"] == "not_comparable"
        prices = [records[item["parent_asin"]].price_text for item in final["recommendations"]]
        assert prices == sorted(
            prices,
            key=lambda text: float(text.replace("$", "").replace(",", "")),
        )


def test_explain_action_uses_grounded_catalogue_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="explain-loop", budget=1) as (harness, database):
        session_id = _create_session(harness)
        first = _shopping(
            harness,
            session_id,
            "explain what weight means before you recommend hiking gear",
        ).json()
        assert first["preference"]["grounded_action"] == "EXPLAIN"
        explanation = first["preference"]["explanation"] or ""
        assert "weight" in explanation.casefold()
        assert "does not assert that you understand" in explanation
        # Every candidate value in the explanation is a real catalogue fact.
        records = harness.runtime.metadata.records
        for identity in ("cand-red", "cand-blue", "cand-black"):
            assert identity in explanation
            details = dict(records[identity].details)
            fact = details.get("Item Weight")
            assert fact and fact.split()[0] in explanation
        # An explicit explanation request records both the request and the display.
        store = SQLiteEventStore(database)
        try:
            session = harness.runtime.sessions.get(session_id)
            event_types = [
                event.event_type.value
                for event in store.get_events(session.session_id, session.user_key)
            ]
        finally:
            store.close()
        assert "feature_explanation_requested" in event_types
        assert "feature_explanation_shown" in event_types
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "higher"
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"


def test_budget_exhaustion_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="budget", budget=1) as (harness, _):
        session_id = _create_session(harness)
        first = _shopping(harness, session_id, "recommend hiking gear").json()
        assert first["preference"]["grounded_action"] == "EXPLAIN"
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "lower"
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"
        assert final["clarification"] is None


def test_proactive_compare_without_explicit_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A material price contrast is shown proactively, without a compare request."""
    with _harness(tmp_path, monkeypatch, name="proactive-compare", budget=1) as (
        harness,
        _,
    ):
        session_id = _create_session(harness)
        first = _shopping(
            harness, session_id, "recommend gear under a tight budget"
        ).json()
        assert first["preference"]["grounded_action"] == "COMPARE"
        assert first["preference"]["interaction"] == "COMPARE"
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "lower"
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"


def test_ask_loop_for_categorical_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="ask-categorical", budget=1) as (
        harness,
        _,
    ):
        session_id = _create_session(harness)
        first = _shopping(
            harness, session_id, "recommend gear in different colors"
        ).json()
        assert first["preference"]["grounded_action"] == "ASK"
        option = next(
            item
            for item in first["preference"]["options"]
            if item["action"] not in {"none", "skip"}
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"


def test_m6_1_refresh_failure_is_explicit_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="refresh-failure", budget=1) as (
        harness,
        _,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _task_id(harness, session_id)
        first = _shopping(harness, session_id, "recommend hiking gear").json()
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "lower"
        )

        def _boom(**_kwargs: Any) -> Any:
            raise RuntimeError("simulated M6.1 refresh failure")

        monkeypatch.setattr(
            "recommendation.demo.agent_service.refresh_candidate_order", _boom
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"
        assert final["recommendations"]
        diagnostics = harness.runtime.agent_service().m7_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["ran"] is False
        assert diagnostics["fallback"] == "accepted_ordering"
        assert diagnostics["error"]


def test_candidate_shortage_triggers_one_re_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _harness(
        tmp_path,
        monkeypatch,
        name="short",
        budget=1,
        rows=CANDIDATE_ROWS[:2],
    ) as (harness, _):
        session_id = _create_session(harness)
        service = harness.runtime.agent_service()
        calls: list[Any] = []
        original = service.run

        def spy(payload: Any, **kwargs: Any) -> Any:
            calls.append(payload)
            return original(payload, **kwargs)

        monkeypatch.setattr(service, "run", spy)
        first = _shopping(harness, session_id, "recommend hiking gear").json()
        assert first["terminal"]["status"] == "waiting_for_user"
        option = next(
            item
            for item in first["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
        final = _answer(harness, session_id, first, option["option_id"]).json()
        assert final["terminal"]["status"] == "finished"
        assert calls, "the protected candidate-shortage re-retrieval path did not run"


def test_pending_preference_question_does_not_swallow_an_explicit_withdrawal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A withdrawal is a fresh user act, not an answer chosen from stale options."""
    from recommendation.demo.agent_service import _pending_message_starts_fresh_turn

    assert _pending_message_starts_fresh_turn("forget the weight preference")
    assert _pending_message_starts_fresh_turn("Actually I need a lantern instead.")
    assert not _pending_message_starts_fresh_turn("I prefer the lighter option.")
    assert not _pending_message_starts_fresh_turn("No additional preference.")
    assert not _pending_message_starts_fresh_turn("skip this question")
    with _harness(tmp_path, monkeypatch, name="pending-withdraw", budget=1) as (
        harness,
        _database,
    ):
        session_id = _create_session(harness)
        first = _shopping(harness, session_id, "recommend hiking gear").json()
        assert first["terminal"]["status"] == "waiting_for_user"
        clarification_id = first["clarification"]["clarification_id"]
        answer = _shopping(
            harness,
            session_id,
            "forget the weight preference",
            clarification_id=clarification_id,
        )
        assert answer.status_code == 200, answer.text
        assert answer.json()["terminal"]["status"] == "finished"
        # The stale pending question is consumed rather than replayed as an answer.
        service = harness.runtime.agent_service()
        assert clarification_id not in service._pending_clarifications



def test_grounded_interaction_events_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recommendation.demo.state_integration import (
        ShoppingStateIntegration,
        StateMode,
    )

    database = tmp_path / "events.sqlite3"
    integration = ShoppingStateIntegration(
        mode=StateMode.SHADOW, database=database
    )
    try:
        kwargs = dict(
            user_key="user",
            session_id="task",
            task_id="task",
            turn_key="task:turn-1",
            compared_refs=("cand-red", "cand-blue"),
            comparison_feedback="Lower price",
        )
        first_ids = integration.record_grounded_interaction(**kwargs)
        second_ids = integration.record_grounded_interaction(**kwargs)
        assert first_ids and second_ids
        assert first_ids == second_ids
        explained_one = integration.record_grounded_interaction(
            user_key="user",
            session_id="task",
            task_id="task",
            turn_key="task:turn-2",
            explanation_requested="weight",
            question_text="What does weight mean?",
            explanation_shown="weight",
            explanation_ref="m7exp_demo",
        )
        explained_two = integration.record_grounded_interaction(
            user_key="user",
            session_id="task",
            task_id="task",
            turn_key="task:turn-2",
            explanation_requested="weight",
            question_text="What does weight mean?",
            explanation_shown="weight",
            explanation_ref="m7exp_demo",
        )
        assert explained_one == explained_two
        store = SQLiteEventStore(database)
        try:
            event_types = [
                event.event_type.value for event in store.get_events("task", "user")
            ]
        finally:
            store.close()
        assert event_types.count("product_comparison_feedback") == 1
        assert event_types.count("feature_explanation_requested") == 1
        assert event_types.count("feature_explanation_shown") == 1
    finally:
        integration.close()


# --------------------------------------------------------------------------- #
# 4. experimental M6.1 refresh: same set, never scored by the model
# --------------------------------------------------------------------------- #


def test_m6_1_refresh_uses_explicit_same_category_numeric_evidence() -> None:
    """A numeric task answer stays explicit and compares grounded same-category values."""
    from recommendation.state.schemas import (
        PreferenceDirection,
        PreferenceKind,
        PreferenceLifecycle,
        PreferencePolarity,
        PreferenceScope,
    )
    from recommendation.state.state_projector import (
        RankerFeature,
        RankerStateProjection,
    )
    from recommendation.demo.preference_construction import TaskPreference

    records = _records_with_category()
    recommendations = [
        type(
            "Presented",
            (),
            {"parent_asin": identity, "item_id": index, "score": float(index), "score_kind": "fused"},
        )()
        for index, identity in enumerate(("cand-black", "cand-blue", "cand-red"), start=1)
    ]
    base = RankerStateProjection(
        task_id="task",
        state_version=1,
        features=(
            RankerFeature(
                preference_id="event:weight:minimize",
                kind=PreferenceKind.FEATURE,
                value="weight",
                polarity=PreferencePolarity.PREFER,
                direction=PreferenceDirection.MINIMIZE,
                scope=PreferenceScope.TASK,
                lifecycle=PreferenceLifecycle.EXPLICIT,
                strength_rank=2,
                is_soft=False,
            ),
        ),
    )
    preference = TaskPreference(
        attribute="weight",
        action="lower",
        value=None,
        option_id="pref:weight:lower",
        label="Lower weight",
    )
    refresh = refresh_candidate_order(
        recommendations=recommendations,
        records=records,
        base_projection=base,
        preferences=(preference,),
        user_key="user",
        task_id="task",
        turn_key="turn",
    )
    assert refresh.error is None, refresh.error
    assert refresh.ran is True
    # Grounded catalogue weights: red 0.8 < blue 1.1 < black 3.4.
    assert refresh.order == ("cand-red", "cand-blue", "cand-black")


def test_m6_1_refresh_preserves_the_candidate_set_and_can_reorder() -> None:
    from recommendation.state.schemas import (
        PreferenceDirection,
        PreferenceKind,
        PreferenceLifecycle,
        PreferencePolarity,
        PreferenceScope,
    )
    from recommendation.state.state_projector import (
        RankerFeature,
        RankerStateProjection,
    )
    from recommendation.demo.preference_construction import TaskPreference

    records = _records_with_category()
    recommendations = [
        type(
            "Presented",
            (),
            {"parent_asin": identity, "item_id": index, "score": float(index), "score_kind": "fused"},
        )()
        for index, identity in enumerate(("cand-red", "cand-blue", "cand-black"), start=1)
    ]
    projection = RankerStateProjection(
        task_id="task", state_version=0, features=()
    )
    preference = TaskPreference(
        attribute="price",
        action="lower",
        value=None,
        option_id="pref:price:lower",
        label="Lower price",
    )
    refresh = refresh_candidate_order(
        recommendations=recommendations,
        records=records,
        base_projection=projection,
        preferences=(preference,),
        user_key="user",
        task_id="task",
        turn_key="turn",
    )
    assert refresh.error is None, refresh.error
    assert refresh.ran is True
    assert set(refresh.order) == {"cand-red", "cand-blue", "cand-black"}
    # Grounded prices: black 15 < red 25 < blue 40.
    assert refresh.order == ("cand-black", "cand-red", "cand-blue")


def test_m6_1_numeric_explicit_does_not_cross_categories() -> None:
    """Different product-type categories must keep the original order."""
    from recommendation.state.state_projector import RankerStateProjection
    from recommendation.demo.preference_construction import TaskPreference

    records = _records()
    for index, identity in enumerate(("cand-red", "cand-blue", "cand-black")):
        record = records[identity]
        records[identity] = record.model_copy(
            update={"categories": ("Sports & Outdoors", "Different Type", identity)}
        )
    recommendations = [
        type(
            "Presented",
            (),
            {"parent_asin": identity, "item_id": index, "score": float(index), "score_kind": "fused"},
        )()
        for index, identity in enumerate(("cand-red", "cand-blue", "cand-black"), start=1)
    ]
    preference = TaskPreference(
        attribute="weight",
        action="lower",
        value=None,
        option_id="pref:weight:lower",
        label="Lower weight",
    )
    refresh = refresh_candidate_order(
        recommendations=recommendations,
        records=records,
        base_projection=RankerStateProjection(
            task_id="task", state_version=0, features=()
        ),
        preferences=(preference,),
        user_key="user",
        task_id="task",
        turn_key="turn",
    )
    assert refresh.error is None, refresh.error
    assert refresh.ran is True
    assert refresh.order == ("cand-red", "cand-blue", "cand-black")


def _interleaved_snapshot(
    order: tuple[str, ...],
    category_by_id: dict[str, tuple[str, ...] | None],
    *,
    unknown_id: str = "cand-unknown",
) -> tuple[Any, Any]:
    from recommendation.ranking.schemas import CandidateRecord, CandidateSnapshot

    records = _records()
    for identity in records:
        category = category_by_id.get(identity)
        if category is not None:
            records[identity] = records[identity].model_copy(
                update={"categories": category}
            )
    records[unknown_id] = None
    snapshot = CandidateSnapshot(
        snapshot_id="s",
        user_key="u",
        task_id="t",
        turn_key="x",
        candidates=tuple(
            CandidateRecord(
                candidate_id=identity,
                item_id=index,
                original_rank=index,
                score=float(index),
                metadata_status="missing" if records[identity] is None else "found",
                metadata=records[identity],
            )
            for index, identity in enumerate(order, start=1)
        ),
    )
    return snapshot, None


def test_m6_1_numeric_overlay_preserves_interleaved_categories_and_unknown() -> None:
    from recommendation.ranking.features import BeliefAwareFeatureBuilder
    from recommendation.ranking.policies import rank_all_arms
    from recommendation.state.schemas import (
        PreferenceDirection,
        PreferenceKind,
        PreferenceLifecycle,
        PreferencePolarity,
        PreferenceScope,
    )
    from recommendation.state.state_projector import RankerFeature, RankerStateProjection

    order = ("cand-black", "cand-blue", "cand-unknown", "cand-red", "cand-green")
    snapshot, _ = _interleaved_snapshot(
        order,
        {
            "cand-black": ("A",),
            "cand-blue": ("B",),
            "cand-red": ("A",),
            "cand-green": ("B",),
        },
    )
    projection = RankerStateProjection(
        task_id="t",
        state_version=1,
        features=(
            RankerFeature(
                preference_id="event:weight:minimize",
                kind=PreferenceKind.FEATURE,
                value="weight",
                polarity=PreferencePolarity.PREFER,
                direction=PreferenceDirection.MINIMIZE,
                scope=PreferenceScope.TASK,
                lifecycle=PreferenceLifecycle.EXPLICIT,
                strength_rank=2,
                is_soft=False,
            ),
        ),
    )
    features = BeliefAwareFeatureBuilder().build(
        snapshot=snapshot, ranker_projection=projection
    )
    arms = rank_all_arms(features, snapshot)
    expected = ("cand-red", "cand-blue", "cand-unknown", "cand-black", "cand-green")
    for arm in arms[1:]:
        assert arm.candidate_ids == expected, (arm.arm, arm.candidate_ids)
    # Category slots are preserved: A,B,unknown,A,B -> A,B,unknown,A,B. The unknown
    # candidate keeps its position and receives no reward/penalty.
    categories = []
    for candidate_id in arms[1].candidate_ids:
        record = next(
            item.metadata
            for item in snapshot.candidates
            if item.candidate_id == candidate_id
        )
        categories.append(tuple(record.categories) if record is not None else None)
    assert categories == [("A",), ("B",), None, ("A",), ("B",)]


def test_m6_1_numeric_overlay_is_independent_of_category_size() -> None:
    from recommendation.ranking.features import BeliefAwareFeatureBuilder
    from recommendation.ranking.policies import rank_all_arms
    from recommendation.state.schemas import (
        PreferenceDirection,
        PreferenceKind,
        PreferenceLifecycle,
        PreferencePolarity,
        PreferenceScope,
    )
    from recommendation.state.state_projector import RankerFeature, RankerStateProjection

    order = ("cand-black", "cand-blue", "cand-red", "cand-green")
    snapshot, _ = _interleaved_snapshot(
        order,
        {
            "cand-black": ("A",),
            "cand-blue": ("A",),
            "cand-red": ("A",),
            "cand-green": ("B",),
        },
    )
    projection = RankerStateProjection(
        task_id="t",
        state_version=1,
        features=(
            RankerFeature(
                preference_id="event:weight:minimize",
                kind=PreferenceKind.FEATURE,
                value="weight",
                polarity=PreferencePolarity.PREFER,
                direction=PreferenceDirection.MINIMIZE,
                scope=PreferenceScope.TASK,
                lifecycle=PreferenceLifecycle.EXPLICIT,
                strength_rank=2,
                is_soft=False,
            ),
        ),
    )
    features = BeliefAwareFeatureBuilder().build(
        snapshot=snapshot, ranker_projection=projection
    )
    arms = rank_all_arms(features, snapshot)
    # Category A has three candidates and reorders internally; category B (one
    # candidate) stays in its original slot.
    assert arms[1].candidate_ids == (
        "cand-red",
        "cand-blue",
        "cand-black",
        "cand-green",
    )
    assert arms[2].candidate_ids == arms[1].candidate_ids

