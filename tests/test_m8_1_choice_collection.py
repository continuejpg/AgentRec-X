"""M8.1 — real demo choice collection tests (offline lab only).

Covers the HTTP choice loop, duplicate/conflict/invalid-input behaviour, persistence and
reload after a new store instance, satisfaction/confidence storage (never correctness),
and the evaluator's separate target-label boundary.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.decision_choice_store import (  # noqa: E402
    DecisionChoiceStore,
    choice_collection_enabled,
    load_collected_trajectories,
)
from recommendation.evaluation.decision import (  # noqa: E402
    DecisionTaskLabels,
    compute_decision_metrics,
)
from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402

CREATE = "/v1/demo/sessions"
SHOPPING = "/v1/demo/sessions/{session}/agent/shopping"
CHOICE = "/v1/demo/sessions/{session}/agent/choice"
CHOICES = "/v1/demo/sessions/{session}/agent/choices"
ENV_STATE_MODE = "AGENTRECX_STATE_MODE"
ENV_STATE_DB = "AGENTRECX_STATE_DB"
ENV_M7 = "AGENTRECX_M7_GROUNDED_ACTIONS"
ENV_M8 = "AGENTRECX_M8_TRAJECTORY"
ENV_M8_COLLECTION = "AGENTRECX_M8_CHOICE_COLLECTION"
ENV_M8_DB = "AGENTRECX_M8_CHOICE_DB"


@contextmanager
def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    collection: bool = True,
) -> Iterator[tuple[Any, Path, Path]]:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    memory_db = root / "memory.sqlite3"
    state_db = root / "state.sqlite3"
    choice_db = root / "choices.sqlite3"
    monkeypatch.setenv(ENV_M7, "on")
    monkeypatch.setenv("AGENTRECX_M7_BUDGET", "1")
    monkeypatch.setenv(ENV_STATE_MODE, "shadow")
    monkeypatch.setenv(ENV_STATE_DB, str(state_db))
    monkeypatch.setenv(ENV_M8, "on")
    if collection:
        monkeypatch.setenv(ENV_M8_COLLECTION, "on")
        monkeypatch.setenv(ENV_M8_DB, str(choice_db))
    else:
        monkeypatch.delenv(ENV_M8_COLLECTION, raising=False)
        monkeypatch.delenv(ENV_M8_DB, raising=False)
    harness = build_harness(root, memory_db=memory_db)
    try:
        yield harness, choice_db, root
    finally:
        close_harness(harness)


def _create(harness: Any) -> str:
    response = harness.client.post(CREATE, json={"profile_id": PROFILE_A})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _recommend(harness: Any, session_id: str) -> dict[str, Any]:
    response = harness.client.post(
        SHOPPING.format(session=session_id),
        json={
            "message": "recommend hiking gear",
            "k": 3,
            "sources": ["history"],
            "preference_discovery": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preference"] is not None
    option = next(
        item for item in body["preference"]["options"] if item["action"] == "lower"
    )
    final = harness.client.post(
        SHOPPING.format(session=session_id),
        json={
            "message": "choose",
            "k": 3,
            "sources": ["history"],
            "preference_discovery": True,
            "clarification_id": body["clarification"]["clarification_id"],
            "clarification_option": option["option_id"],
        },
    )
    assert final.status_code == 200, final.text
    return final.json()


def test_choice_loop_persists_and_prevents_duplicates(tmp_path: Path, monkeypatch) -> None:
    with _harness(tmp_path, monkeypatch, name="choice-loop") as (
        harness,
        choice_db,
        _,
    ):
        session_id = _create(harness)
        final = _recommend(harness, session_id)
        selected = final["recommendations"][0]["parent_asin"]
        response = harness.client.post(
            CHOICE.format(session=session_id),
            json={
                "selected_parent_asin": selected,
                "satisfaction": 5,
                "confidence": 4,
                "comment": "useful",
            },
        )
        assert response.status_code == 200, response.text
        record = response.json()
        assert record["selected_parent_asin"] == selected
        assert record["satisfaction"] == 5
        assert record["confidence"] == 4
        assert record["duplicate"] is False
        assert record["final_top_k"] == [
            item["parent_asin"] for item in final["recommendations"]
        ]
        assert record["trajectory_digest"]

        duplicate = harness.client.post(
            CHOICE.format(session=session_id),
            json={
                "selected_parent_asin": selected,
                "satisfaction": 5,
                "confidence": 4,
                "comment": "useful",
            },
        )
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["duplicate"] is True
        assert duplicate.json()["choice_id"] == record["choice_id"]

        conflict = harness.client.post(
            CHOICE.format(session=session_id),
            json={"selected_parent_asin": final["recommendations"][1]["parent_asin"]},
        )
        assert conflict.status_code == 409, conflict.text

        invalid = harness.client.post(
            CHOICE.format(session=session_id),
            json={"selected_parent_asin": "not-presented"},
        )
        assert invalid.status_code == 422, invalid.text

        history = harness.client.get(CHOICES.format(session=session_id))
        assert history.status_code == 200, history.text
        assert len(history.json()) == 1

        store = DecisionChoiceStore(choice_db)
        try:
            records = store.choices_for_session(session_id)
            assert len(records) == 1
            exported = tmp_path / "export.jsonl"
            assert store.export_jsonl(exported) == 1
        finally:
            store.close()
        lines = exported.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        encoded = lines[0]
        assert "target" not in encoded.casefold()
        assert "correct" not in encoded.casefold()


def test_collected_trajectories_reload_after_restart_and_metrics_use_real_choice(
    tmp_path: Path, monkeypatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="reload") as (harness, choice_db, _):
        session_id = _create(harness)
        final = _recommend(harness, session_id)
        selected = final["recommendations"][0]["parent_asin"]
        target = final["recommendations"][2]["parent_asin"]
        response = harness.client.post(
            CHOICE.format(session=session_id),
            json={"selected_parent_asin": selected, "satisfaction": 2},
        )
        assert response.status_code == 200, response.text
    # New store instance: the persisted row survives the harness/process lifetime.
    store = DecisionChoiceStore(choice_db)
    try:
        records = store.real_choices()
        assert len(records) == 1
        stored = records[0]
        assert stored.interaction_source == "demo_http"
        assert stored.is_real_user_interaction
    finally:
        store.close()
    trajectories = load_collected_trajectories(choice_db)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory.choice.source == "user"
    assert trajectory.choice.selected_candidate_id == selected
    # The evaluator receives the target label separately; the user's choice is not used
    # as the target.
    labels = {
        trajectory.task_id: DecisionTaskLabels(
            trajectory.task_id, (target,), label_source="real_offline_label"
        )
    }
    metrics = compute_decision_metrics(trajectories, labels, k=3)
    assert metrics.recall_at_k.status == "MEASURED"
    assert metrics.team_accuracy.status == "MEASURED"
    assert metrics.team_accuracy.value == 0.0
    assert metrics.conditional_selection_accuracy.status == "MEASURED"


def test_choice_history_is_scoped_to_one_live_session(
    tmp_path: Path, monkeypatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="scoped") as (harness, _choice_db, _):
        session_a = _create(harness)
        session_b = _create(harness)
        final = _recommend(harness, session_a)
        response = harness.client.post(
            CHOICE.format(session=session_a),
            json={"selected_parent_asin": final["recommendations"][0]["parent_asin"]},
        )
        assert response.status_code == 200, response.text
        only_a = harness.client.get(CHOICES.format(session=session_a))
        assert only_a.status_code == 200, only_a.text
        assert len(only_a.json()) == 1
        only_b = harness.client.get(CHOICES.format(session=session_b))
        assert only_b.status_code == 200, only_b.text
        assert only_b.json() == []
        missing = harness.client.get(CHOICES.format(session="does-not-exist"))
        assert missing.status_code == 404, missing.text
        assert missing.json()["error"] == "session_not_found"


def test_no_suitable_item_is_recorded_without_becoming_correctness(
    tmp_path: Path, monkeypatch
) -> None:
    with _harness(tmp_path, monkeypatch, name="no-suitable") as (
        harness,
        choice_db,
        _,
    ):
        session_id = _create(harness)
        _recommend(harness, session_id)
        response = harness.client.post(
            CHOICE.format(session=session_id),
            json={"no_suitable_item": True, "satisfaction": 1},
        )
        assert response.status_code == 200, response.text
        record = response.json()
        assert record["no_suitable_item"] is True
        assert record["selected_parent_asin"] is None
    trajectories = load_collected_trajectories(choice_db)
    assert len(trajectories) == 1
    assert trajectories[0].choice.source == "user"
    assert trajectories[0].choice.no_suitable_item is True
    labels = {
        trajectories[0].task_id: DecisionTaskLabels(
            trajectories[0].task_id, ("cand-red",), label_source="real_offline_label"
        )
    }
    metrics = compute_decision_metrics(trajectories, labels, k=3)
    assert metrics.recall_at_k.status == "MEASURED"
    assert metrics.team_accuracy.status == "NOT_EXECUTED"


def test_collection_disabled_does_not_serve_choice(tmp_path: Path, monkeypatch) -> None:
    with _harness(
        tmp_path, monkeypatch, name="disabled", collection=False
    ) as (harness, _choice_db, _):
        session_id = _create(harness)
        final = _recommend(harness, session_id)
        response = harness.client.post(
            CHOICE.format(session=session_id),
            json={"selected_parent_asin": final["recommendations"][0]["parent_asin"]},
        )
        assert response.status_code == 503, response.text
        assert choice_collection_enabled() is False
