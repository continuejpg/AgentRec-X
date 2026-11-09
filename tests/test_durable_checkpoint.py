"""Restart/replay tests for the opt-in durable demo checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402


ENDPOINT = "/v1/demo/sessions/{session}/agent/shopping"


def _preference_request(session: str) -> dict:
    return {
        "message": "recommend gear under a tight budget",
        "k": 3,
        "sources": ["history"],
        "preference_discovery": True,
    }


def test_session_and_pending_preference_survive_runtime_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
        question = first.client.post(
            ENDPOINT.format(session=session), json=_preference_request(session)
        ).json()
        assert question["terminal"]["status"] == "waiting_for_user"
        clarification_id = question["clarification"]["clarification_id"]
        option = next(
            item for item in question["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
    finally:
        close_harness(first)

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        state = second.client.get(f"/v1/demo/sessions/{session}")
        assert state.status_code == 200, state.text
        assert state.json()["session_id"] == session

        answer = second.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": option["label"],
                "k": 3,
                "sources": ["history"],
                "clarification_id": clarification_id,
                "clarification_option": option["option_id"],
            },
        )
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert body["terminal"]["status"] in {"waiting_for_user", "finished"}
        # The consumed id must not be replayable after the restart answer.
        replay = second.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": option["label"],
                "k": 3,
                "sources": ["history"],
                "clarification_id": clarification_id,
                "clarification_option": option["option_id"],
            },
        )
        assert replay.status_code == 404
    finally:
        close_harness(second)


def test_task_preferences_survive_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
        question = first.client.post(
            ENDPOINT.format(session=session), json=_preference_request(session)
        ).json()
        option = next(
            item for item in question["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
        answered = first.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": option["label"],
                "k": 3,
                "sources": ["history"],
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": option["option_id"],
            },
        )
        assert answered.status_code == 200, answered.text
    finally:
        close_harness(first)

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        # A new turn on the restored task must see the confirmed task preference and
        # its round count; it must not start the preference questionnaire again.
        body = second.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "recommend gear under a tight budget",
                "k": 3,
                "sources": ["history"],
                "preference_discovery": True,
            },
        ).json()
        assert body["clarification"] is None or (
            body["clarification"].get("kind") != "preference"
        ), body["clarification"]
    finally:
        close_harness(second)


def test_research_pending_survives_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    request = {
        "message": "compare the first two by price, color, and material",
        "k": 3,
        "sources": ["history", "catalog_search"],
    }
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
        question = first.client.post(ENDPOINT.format(session=session), json=request).json()
        assert question["clarification"] is not None, question
        clarification_id = question["clarification"]["clarification_id"]
        option_id = "compare:price"
    finally:
        close_harness(first)

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        answer = second.client.post(
            ENDPOINT.format(session=session),
            json={
                **request,
                "message": "price",
                "clarification_id": clarification_id,
                "clarification_option": option_id,
            },
        )
        assert answer.status_code == 200, answer.text
        assert answer.json()["terminal"]["status"] in {"waiting_for_user", "finished"}
    finally:
        close_harness(second)
