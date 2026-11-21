"""Pure recommendation snapshot construction (B1c-1a-2)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.sessions import Session, build_recommendation_record  # noqa: E402


def _payload() -> dict:
    return {
        "recommendation_id": "rec-snapshot1",
        "request_id": "web-s-t1",
        "query": "wireless headphones",
        "search_mode": "bm25",
        "items": [
            {"item_id": "B0TEST", "title": "Headphones", "score": 0.9},
            {"item_id": "B0OTHER", "title": "Earbuds", "score": 0.8},
        ],
        "gate": {"action": "finalize", "allow_finalize": True},
    }


def _stable(snapshot: dict) -> dict:
    return {k: v for k, v in snapshot.items() if k not in {'created_at', 'created_ts'}}


def test_pure_builder_matches_recorded_snapshot() -> None:
    session = Session(session_id="web-s", profile_id="profile_0057")
    recorded = session.record_recommendation(
        copy.deepcopy(_payload()), turn_id="web-s-t1", ranking_source="cpu_live:test"
    )
    built = build_recommendation_record(
        _payload(), turn_id="web-s-t1", ranking_source="cpu_live:test"
    )
    assert _stable(built) == _stable(recorded)
    assert built["recommendation_id"] == "rec-snapshot1"
    assert built["request_id"] == "web-s-t1"
    assert built["turn_id"] == "web-s-t1"
    assert built["ranking_source"] == "cpu_live:test"


def test_pure_builder_does_not_mutate_input_or_session() -> None:
    payload = _payload()
    before = copy.deepcopy(payload)
    session = Session(session_id="web-s", profile_id="profile_0057")

    built = build_recommendation_record(
        payload, turn_id="web-s-t1", ranking_source="cpu_live:test"
    )

    assert payload == before
    assert built is not payload
    assert session.recommendations == {}
    assert session.latest_recommendation_id is None
    assert session.turns == []


def test_pure_builder_is_deterministic_and_json_roundtrips() -> None:
    first = build_recommendation_record(
        _payload(), turn_id="web-s-t1", ranking_source="cpu_live:test"
    )
    second = build_recommendation_record(
        _payload(), turn_id="web-s-t1", ranking_source="cpu_live:test"
    )
    assert _stable(first) == _stable(second)
    assert json.loads(json.dumps(first)) == first


def test_record_recommendation_still_updates_session_cache() -> None:
    session = Session(session_id="web-s", profile_id="profile_0057")
    snapshot = session.record_recommendation(
        copy.deepcopy(_payload()), turn_id="web-s-t1", ranking_source="cpu_live:test"
    )
    assert session.latest_recommendation_id == "rec-snapshot1"
    assert session.recommendation("rec-snapshot1") == snapshot
