"""Reusable offline fixtures for the Milestone 11 web-demo tests.

Everything is deterministic and dependency-free: no accepted checkpoint, no catalogue
artifact, no network, no provider API and no browser.  The objects the API manipulates
are nevertheless the **real** accepted types:

* candidates come from the accepted Recommendation Tool over a fixed engine double;
* metadata is a real :class:`~recommendation.catalog.MetadataIndex`;
* evidence is produced by the real Milestone 10A matcher;
* order comes from the real Milestone 10B reranker;
* preference memory is the real Milestone 9 service over a real SQLite store.

The doubles exist only where a 349 MB checkpoint would otherwise be required, and they
sit behind the same structural seams the accepted components already use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    INITIAL_ORDER,
    QUERY,
    build_index,
)
from recommendation.agent import AgentDecision  # noqa: E402
from recommendation.api.app import ServiceSettings, create_app  # noqa: E402
from recommendation.demo import DemoProfile, build_demo_runtime  # noqa: E402
from recommendation.inference import Recommendation, RecommendationResult  # noqa: E402
from recommendation.memory import SQLitePreferenceStore  # noqa: E402

__all__ = [
    "CANDIDATE_ROWS",
    "HISTORY",
    "INITIAL_ORDER",
    "PROFILE_A",
    "PROFILE_B",
    "QUERY",
    "DemoEngine",
    "DemoHarness",
    "build_harness",
    "demo_profiles",
    "make_settings",
]

#: Trusted history used by the synthetic demo profiles.  Deliberately distinct from
#: every candidate identity, so a leak of one into the other is visible.
HISTORY: tuple[str, ...] = ("B000000001", "B000000002", "B000000003")

PROFILE_A = "demo-user-1"
PROFILE_B = "demo-user-2"


class DemoEngine:
    """A fixed-candidate engine implementing the accepted ``RecommendationEngine`` seam.

    It returns a real :class:`~recommendation.inference.RecommendationResult`, so it is a
    truthful stand-in for :class:`~recommendation.inference.SASRecInferenceEngine`: both
    the accepted Tool and the accepted Milestone 6 ``/v1/recommend`` endpoint consume it
    unchanged.  It exists only so the demo tests do not need the 349 MB checkpoint.
    """

    device = "cpu"
    num_items = 1000
    max_seq_len = 50

    def __init__(self, rows: Any = CANDIDATE_ROWS, *, error: Exception | None = None) -> None:
        self._rows = tuple(rows)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """How many times the engine was invoked."""
        return len(self.calls)

    @property
    def last_history(self) -> list[str] | None:
        """The exact history list the engine received on the last call."""
        return self.calls[-1]["history"] if self.calls else None

    def is_ready(self) -> bool:
        """The double is always ready."""
        return True

    def model_metadata(self) -> dict[str, Any]:
        """Metadata shaped exactly like the accepted ``/v1/model`` response."""
        return {
            "model_type": "SASRec",
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "hidden_size": 64,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "device": self.device,
            "checkpoint_sha256": "0" * 64,
            "parameter_count": 1234,
            "model_parameters_frozen": True,
            "provenance": {"serving_note": "synthetic demo fixture", "formal_run_git": {}},
        }

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationResult:
        """Record the call and return the fixed candidate list, truncated to ``k``."""
        history = list(history_parent_asins)
        self.calls.append({"history": history, "k": k})
        if self.error is not None:
            raise self.error
        rows = self._rows[:k]
        return RecommendationResult(
            recommendations=[
                Recommendation(rank=rank, item_id=row[1], parent_asin=row[0], score=row[2])
                for rank, row in enumerate(rows, start=1)
            ],
            requested_k=k,
            history_length=len(history),
            effective_history_length=min(len(history), self.max_seq_len),
            history_truncated=len(history) > self.max_seq_len,
            eligible_candidates=max(self.num_items - len(history), 0),
            timings_ms={"scoring": 0.5, "ranking": 0.1},
        )


def demo_profiles(
    *, names: tuple[str, ...] = (PROFILE_A, PROFILE_B)
) -> dict[str, DemoProfile]:
    """A small, fixed set of synthetic demo profiles.

    Each profile gets a *different* history so a cross-profile leak is observable, and
    no profile is derived from a real user record.
    """
    profiles: dict[str, DemoProfile] = {}
    for index, name in enumerate(names, start=1):
        history = tuple(f"B{index:09d}{position}" for position in range(1, 4))
        profiles[name] = DemoProfile(
            profile_id=name,
            display_name=f"Test shopper {index}",
            trusted_user_history=history,
            source_user_int_id=index,
            source_length=len(history) + 2,
        )
    return profiles


def make_settings(tmp_path: Path) -> ServiceSettings:
    """Service settings pointing at absent paths; the engine is injected instead."""
    return ServiceSettings(
        checkpoint_path=tmp_path / "absent.pt",
        mappings_path=tmp_path / "absent_mappings.json",
        manifest_path=None,
        device="cpu",
    )


class DemoHarness:
    """A built demo app plus the collaborators a test needs to inspect."""

    def __init__(
        self,
        *,
        app: Any,
        runtime: Any,
        engine: DemoEngine,
        client: Any,
        store: SQLitePreferenceStore,
        profiles: dict[str, DemoProfile],
    ) -> None:
        self.app = app
        self.runtime = runtime
        self.engine = engine
        self.client = client
        self.store = store
        self.profiles = profiles

    # -- convenience ------------------------------------------------------- #

    @property
    def manager(self) -> Any:
        """The session manager."""
        return self.runtime.sessions

    def create_session(self, profile_id: str = PROFILE_A) -> dict[str, Any]:
        """Create a session over HTTP and return the parsed body."""
        response = self.client.post("/v1/demo/sessions", json={"profile_id": profile_id})
        assert response.status_code == 201, response.text
        return response.json()

    def chat(self, session_id: str, message: str, k: int = 5) -> dict[str, Any]:
        """Send one chat turn over HTTP and return the parsed body."""
        response = self.client.post(
            f"/v1/demo/sessions/{session_id}/chat", json={"message": message, "k": k}
        )
        assert response.status_code == 200, response.text
        return response.json()

    def state(self, session_id: str) -> dict[str, Any]:
        """Read a session's state over HTTP."""
        response = self.client.get(f"/v1/demo/sessions/{session_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def active_values(self, session_id: str) -> set[tuple[str, str, str]]:
        """Active preferences of a session as ``(kind, polarity, value)`` triples."""
        body = self.state(session_id)
        return {
            (item["kind"], item["polarity"], item["value"])
            for item in body["active_preferences"]
        }


def build_harness(
    tmp_path: Path,
    *,
    rows: Any = CANDIDATE_ROWS,
    profiles: dict[str, DemoProfile] | None = None,
    max_sessions: int = 8,
    ttl_seconds: float | None = None,
    clock: Any = None,
    memory_db: Path | None = None,
    demo: bool = True,
) -> DemoHarness:
    """Build a full M11 app over synthetic artifacts and a real SQLite memory store."""
    from fastapi.testclient import TestClient

    profiles = profiles or demo_profiles()
    engine = DemoEngine(rows)
    store = SQLitePreferenceStore(memory_db or (tmp_path / "demo_memory.sqlite3"))

    kwargs: dict[str, Any] = {
        "engine": engine,
        "metadata": build_index(row[0] for row in rows),
        "store": store,
        "profiles": profiles,
        "max_sessions": max_sessions,
    }
    if ttl_seconds is not None:
        kwargs["ttl_seconds"] = ttl_seconds
    if clock is not None:
        kwargs["clock"] = clock
    runtime = build_demo_runtime(**kwargs)

    app = create_app(
        make_settings(tmp_path),
        engine=engine,
        load_on_startup=False,
        demo=runtime if demo else None,
        enable_demo=demo,
    )
    client = TestClient(app)
    client.__enter__()
    return DemoHarness(
        app=app, runtime=runtime, engine=engine, client=client, store=store, profiles=profiles
    )


def close_harness(harness: DemoHarness) -> None:
    """Exit the test client context and close the store."""
    try:
        harness.client.__exit__(None, None, None)
    finally:
        harness.runtime.close()


#: Re-exported so a test can build a decision payload directly when needed.
RECOMMEND_DECISION = AgentDecision(action="recommend", k=5)
