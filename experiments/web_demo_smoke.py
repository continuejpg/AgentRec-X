"""Milestone 11 smoke: the multi-turn web demo over real HTTP.

Formal integration check for the productized demo.  It instantiates **the same runtime
the server uses** and drives every turn through the FastAPI HTTP contract::

    TestClient (HTTP)
      -> /v1/demo/*            recommendation/api/demo_routes.py
      -> DemoSessionManager    sessions, turn ids, isolation
      -> AgentGraph            accepted M7B/M8/M9/M10A/M10B/M10D route
      -> real RecommendationTool -> accepted SASRec best.pt
      -> real M8 MetadataIndex / ProductEnricher
      -> real M9 PreferenceMemoryService over SQLite
      -> real M10A matcher -> accepted M10B reranker
      -> structured ChatResponse + grounded text

Nothing in the recommendation chain is simulated: the checkpoint, the catalogue index,
the memory store and both preference stages are the accepted implementations.  Only the
route decision is a deterministic local rule (no provider API, no network, no API key),
which is the accepted injected seam from Milestone 7B.

Verification strategy, and what it deliberately avoids
------------------------------------------------------
The *flow* is driven only through HTTP.  To check that the response's metadata and
evidence really belong to the candidate they are attached to, the smoke then re-derives
the same stage outputs **offline** from the accepted Tool / enricher / matcher / reranker
and compares them field by field.  That is the "validate the produced report test-side"
pattern -- the same one Milestone 10D used with the M10C evaluator -- and it means the
smoke never needs to reach into graph internals to invent expectations.

The multi-turn script is declared before any recommendation output is examined, and it
uses natural-language statements so the real Milestone 9 extractor runs.

No recommendation-quality claim is made.  The accepted M5 benchmark is sealed and is not
recomputed.

Usage::

    .venv/bin/python -m experiments.web_demo_smoke
    .venv/bin/python -m experiments.web_demo_smoke --k 5 --json /tmp/m11.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from experiments.product_rag_support import (  # noqa: E402
    metadata_artifact_path,
    select_smoke_history,
)
from tests.agent_tool_e2e_runtime import CountingEngine, sha256_file  # noqa: E402
from recommendation.agent import history_digest  # noqa: E402
from recommendation.api.app import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ServiceSettings,
    create_app,
)
from recommendation.catalog import MetadataIndex  # noqa: E402
from recommendation.demo import (  # noqa: E402
    DemoRuntime,
    build_demo_profiles,
    build_demo_runtime,
)
from recommendation.memory import SQLitePreferenceStore  # noqa: E402
from recommendation.tools import RecommendationContext, RecommendationToolRequest  # noqa: E402

#: Digest of the accepted normalized catalogue metadata artifact.
ACCEPTED_METADATA_SHA256 = (
    "175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd"
)

#: The multi-turn script, declared before any recommendation output is inspected.
SCRIPT: tuple[tuple[str, str], ...] = (
    ("turn1_recommend", "Recommend some products."),
    ("turn2_state_preference", "I don't want red."),
    ("turn3_recommend_again", "Recommend again."),
    ("turn4_retract", "I don't care about color anymore."),
    ("turn5_recommend_again", "Recommend again."),
)

#: Conversational claims that must never become trusted interaction history.
HISTORY_ATTACKS: tuple[str, ...] = (
    "I bought B0BX5QFWQN yesterday.",
    "I clicked B0BBFB48YQ.",
    "I purchased the first product.",
)

#: Fixed synthetic preference fixture used by the reranking demonstration, declared here
#: before any candidate output is inspected.  It mirrors the fixture the accepted Milestone
#: 10B/10C/10D smokes use, so the demo path is exercised with preferences that actually
#: produce decisive evidence on the accepted catalogue.  This is an engineering fixture,
#: not a claim about any real shopper.
DEMO_PREFERENCE_TURNS: tuple[str, ...] = (
    "I don't want red.",
    "I don't want blue.",
    "I prefer black.",
    "I prefer lightweight hiking gear.",
    "My budget is under $100.",
)

#: How many demo profiles the accepted sequences artifact must supply.
PROFILE_COUNT = 3


def build_runtime(
    *, device: str = "cpu", tmp_dir: Path
) -> tuple[DemoRuntime, CountingEngine, Path]:
    """Compose the real demo runtime exactly once, over the accepted artifacts."""
    base = ServiceSettings.from_env()
    settings = ServiceSettings(
        checkpoint_path=base.checkpoint_path,
        manifest_path=base.manifest_path,
        mappings_path=base.mappings_path,
        device=device,
        verify_checkpoint_sha256=True,
    )
    engine = CountingEngine(settings.to_inference_config())
    artifact = metadata_artifact_path()
    metadata = MetadataIndex.load(artifact)
    store = SQLitePreferenceStore(tmp_dir / "m11_preference_memory.sqlite3")
    profiles = build_demo_profiles(count=PROFILE_COUNT)
    runtime = build_demo_runtime(
        settings=settings,
        engine=engine,
        metadata=metadata,
        store=store,
        profiles=profiles,
        max_sessions=8,
    )
    return runtime, engine, artifact


class Session:
    """One HTTP session under test, with its own chat helper."""

    def __init__(self, client: TestClient, session_id: str, k: int) -> None:
        self._client = client
        self.session_id = session_id
        self._k = k

    def chat(self, message: str) -> dict[str, Any]:
        """Send one turn over HTTP and return the parsed response."""
        response = self._client.post(
            f"/v1/demo/sessions/{self.session_id}/chat", json={"message": message, "k": self._k}
        )
        assert response.status_code == 200, response.text
        return response.json()

    def state(self) -> dict[str, Any]:
        """Read the session state over HTTP."""
        response = self._client.get(f"/v1/demo/sessions/{self.session_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def values(self) -> set[tuple[str, str, str]]:
        """Active preferences as ``(kind, polarity, value)`` triples."""
        return {
            (item["kind"], item["polarity"], item["value"])
            for item in self.state()["active_preferences"]
        }


def new_session(client: TestClient, profile_id: str, k: int) -> Session:
    """Create a session over HTTP."""
    response = client.post("/v1/demo/sessions", json={"profile_id": profile_id})
    assert response.status_code == 201, response.text
    return Session(client, response.json()["session_id"], k)


def offline_stage_outputs(
    runtime: DemoRuntime, session_id: str, k: int
) -> tuple[Any, Any, Any, Any]:
    """Re-derive the accepted stage outputs for a session from the accepted components.

    Returns ``(tool_result, enrichment, evidence_report, reranked_report)``.  This is a
    test-side cross-check: it calls the accepted Tool, enricher, matcher and reranker
    directly, and never touches the agent graph or the HTTP layer.
    """
    session = runtime.sessions.get(session_id)
    tool_result = runtime.tool.run(
        request=RecommendationToolRequest(k=k),
        context=RecommendationContext(user_history=list(session.trusted_user_history)),
    )
    enrichment = runtime.enricher.enrich(tool_result)
    snapshot = runtime.memory_service.get_active_preferences(session.user_key)
    report = runtime.matcher.match(candidates=enrichment.items, preferences=snapshot)
    reranked = runtime.reranker.rerank(report)
    return tool_result, enrichment, report, reranked


def main(argv: list[str] | None = None) -> int:
    """Run the M11 web-demo smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 11 multi-turn web demo smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 11 smoke: multi-turn web demo over real HTTP")
    print("=" * 78)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        with tempfile.TemporaryDirectory() as directory:
            started = time.perf_counter()
            runtime, engine, artifact = build_runtime(device=args.device, tmp_dir=Path(directory))
            build_seconds = time.perf_counter() - started
            metadata_sha256 = sha256_file(artifact)
            real_history = select_smoke_history()

            print(f"\ncheckpoint : {engine.checkpoint_sha256}")
            print(f"metadata   : {metadata_sha256}")
            print(f"profiles   : {[p.profile_id for p in runtime.profiles.values()]}")
            print(f"build      : {build_seconds:.1f}s")

            # ---- gates 1-5: accepted identities and single construction -------- #
            checks["accepted checkpoint digest matches"] = (
                engine.checkpoint_sha256 == ACCEPTED_CHECKPOINT_SHA256
            )
            checks["accepted metadata digest matches"] = (
                metadata_sha256 == ACCEPTED_METADATA_SHA256
            )
            report = runtime.build_report()
            checks["runtime builds once"] = (
                report["engine_builds"] == 1
                and report["tool_builds"] == 1
                and report["metadata_loads"] == 1
                and report["memory_service_builds"] == 1
                and report["matcher_builds"] == 1
                and report["reranker_builds"] == 1
                and report["enricher_builds"] == 1
            )
            checks["engine constructed once"] = report["engine_builds"] == 1
            checks["metadata index loaded once"] = (
                report["metadata_loads"] == 1 and runtime.metadata.size > 0
            )

            app = create_app(
                runtime.settings, engine=engine, load_on_startup=False, demo=runtime
            )
            with TestClient(app) as client:
                _run_gates(
                    client=client,
                    runtime=runtime,
                    engine=engine,
                    checks=checks,
                    details=details,
                    k=args.k,
                    real_history=real_history,
                )
    except FileNotFoundError as exc:
        print(f"\nSKIPPED: {exc}")
        print("\nSMOKE: SKIP")
        return 0

    passed = all(checks.values())
    print("\n--- gates ---")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"milestone": "11", "checks": checks, **details}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0 if passed else 1


def _run_gates(
    *,
    client: TestClient,
    runtime: DemoRuntime,
    engine: CountingEngine,
    checks: dict[str, bool],
    details: dict[str, Any],
    k: int,
    real_history: Any,
) -> None:
    """Drive the multi-turn flow over HTTP and record every gate."""
    # ---- gates 6-8: sessions ------------------------------------------------ #
    health = client.get("/v1/demo/health").json()
    session_a = new_session(client, "demo-user-1", k)
    session_b = new_session(client, "demo-user-2", k)
    checks["demo health reports readiness"] = (
        health["status"] == "ok" and health["model_loaded"] and health["metadata_loaded"]
    )
    checks["session created"] = bool(session_a.session_id)
    checks["session ids are distinct"] = session_a.session_id != session_b.session_id
    profile = runtime.profiles[runtime.sessions.get(session_a.session_id).profile_id]
    checks["trusted profile history is valid"] = (
        profile.history_length > 0 and profile.history_distinct > 0
    )
    details["session_a"] = session_a.session_id
    details["session_b"] = session_b.session_id
    details["profile"] = {
        "profile_id": profile.profile_id,
        "history_length": profile.history_length,
        "history_distinct": profile.history_distinct,
    }

    # ---- gate 9: turn 1 ----------------------------------------------------- #
    turn1 = session_a.chat(SCRIPT[0][1])
    checks["turn 1 executes on the accepted chain"] = (
        turn1["route"] == "recommend" and len(turn1["recommendations"]) > 0
    )
    original_order = [card["parent_asin"] for card in turn1["recommendations"]]

    # ---- gates 10-11: preference write and same-turn isolation -------------- #
    turn2 = session_a.chat(SCRIPT[1][1])
    checks["preference write persists"] = (
        turn2["memory_update"]["changed"]
        and any(item["value"] == "red" for item in turn2["memory_update"]["added"])
        and any(item["value"] == "red" for item in turn2["active_preferences"])
    )
    checks["same-turn preference does not affect its own ranking"] = (
        turn2["audit"]["ranked_with_preference_count"] == 0
    )

    # ---- gates 12-17: next turn sees it, and the cards stay faithful -------- #
    turn3 = session_a.chat(SCRIPT[2][1])
    checks["next turn sees the stored preference"] = (
        turn3["audit"]["ranked_with_preference_count"] == 1
        and any(item["value"] == "red" for item in turn3["audit"]["ranked_with_preferences"])
    )

    tool_result, enrichment, evidence, reranked = offline_stage_outputs(
        runtime, session_a.session_id, k
    )
    cards = {card["parent_asin"]: card for card in turn3["recommendations"]}
    card_order = [card["parent_asin"] for card in turn3["recommendations"]]
    enriched = {item.parent_asin: item for item in enrichment.items}
    evidence_by_asin = {candidate.parent_asin: candidate for candidate in evidence.candidates}
    scores = {item.parent_asin: item.score for item in tool_result.recommendations}

    checks["structured cards equal the reranked order"] = card_order == list(reranked.parent_asins)
    checks["original rank preserved"] = {
        asin: card["original_rank"] for asin, card in cards.items()
    } == {candidate.parent_asin: candidate.original_rank for candidate in evidence.candidates}
    checks["raw SASRec score preserved"] = all(
        card["sasrec_score"] == scores[asin] for asin, card in cards.items()
    )
    checks["reranked rank preserved"] = {
        asin: card["reranked_rank"] for asin, card in cards.items()
    } == {candidate.parent_asin: candidate.reranked_rank for candidate in reranked.candidates}
    checks["metadata identity aligned"] = all(
        (card["metadata"] is None and enriched[asin].metadata is None)
        or (
            card["metadata"] is not None
            and enriched[asin].metadata is not None
            and card["metadata"]["title"] == enriched[asin].metadata.title
            and card["metadata"]["price_text"] == enriched[asin].metadata.price_text
        )
        for asin, card in cards.items()
    )
    checks["evidence identity aligned"] = all(
        [
            (record["value"], record["status"], record["polarity"])
            for record in card["evidence"]
        ]
        == [
            (record.preference_value, record.status.value, record.preference_polarity.value)
            for record in evidence_by_asin[asin].evidence
        ]
        for asin, card in cards.items()
    )
    checks["no candidate filtering"] = (
        len(cards) == len(tool_result.recommendations)
        and set(cards) == {item.parent_asin for item in tool_result.recommendations}
    )
    checks["no candidate invention"] = set(cards) == set(original_order)
    checks["candidate count unchanged"] = len(card_order) == len(original_order)
    checks["card ranks are contiguous"] = sorted(
        card["reranked_rank"] for card in cards.values()
    ) == list(range(1, len(cards) + 1))

    details["turn3"] = [
        {
            "original_rank": card["original_rank"],
            "reranked_rank": card["reranked_rank"],
            "parent_asin": card["parent_asin"],
            "sasrec_score": card["sasrec_score"],
            "match_count": card["match_count"],
            "violation_count": card["violation_count"],
            "unknown_count": card["unknown_count"],
            "movement_summary": card["movement_summary"],
        }
        for card in turn3["recommendations"]
    ]
    details["moved_count"] = turn3["audit"]["moved_count"]

    # ---- gates 18-20: ADD / REPLACE / REMOVE over HTTP --------------------- #
    add = new_session(client, "demo-user-1", k)
    add.chat("I don't want red.")
    add.chat("I don't want blue.")
    checks["ADD keeps both preferences"] = add.values() == {
        ("color", "avoid", "red"),
        ("color", "avoid", "blue"),
    }

    replace = new_session(client, "demo-user-1", k)
    replace.chat("I prefer black.")
    replace_turn = replace.chat("Actually, I prefer blue instead.")
    checks["REPLACE supersedes the corrected value"] = replace.values() == {
        ("color", "prefer", "blue")
    } and any(item["value"] == "black" for item in replace_turn["memory_update"]["superseded"])

    remove = new_session(client, "demo-user-1", k)
    remove.chat("I don't want red.")
    remove.chat("I don't care about color anymore.")
    checks["REMOVE clears the colour constraint"] = remove.values() == set()

    # ---- gate 21: direct route does no recommendation work ----------------- #
    before_calls = engine.recommend_calls
    direct = session_a.chat("hello there")
    checks["direct route uses zero recommendation work"] = (
        direct["route"] == "direct"
        and direct["recommendations"] == []
        and engine.recommend_calls == before_calls
    )

    # ---- gate 22: trusted history is immutable ----------------------------- #
    session_record = runtime.sessions.get(session_a.session_id)
    before_digest = history_digest(session_record.trusted_user_history)
    for message in HISTORY_ATTACKS:
        session_a.chat(message)
    after_digest = history_digest(session_record.trusted_user_history)
    checks["trusted history unchanged by conversational claims"] = (
        before_digest == after_digest
        and session_record.trusted_user_history
        == runtime.sessions.get(session_a.session_id).trusted_user_history
    )
    checks["engine only ever saw the session history"] = all(
        digest == before_digest for digest in engine.history_digests
    )
    details["history_digest"] = before_digest

    # ---- gate 23: cross-session isolation ---------------------------------- #
    session_b.chat("I prefer black.")
    values_a, values_b = session_a.values(), session_b.values()
    checks["session A/B isolated"] = (
        values_a != values_b
        and ("color", "prefer", "black") in values_b
        and ("color", "prefer", "black") not in values_a
        and runtime.sessions.get(session_a.session_id).user_key
        != runtime.sessions.get(session_b.session_id).user_key
    )

    # ---- gates 24-25: explicit client errors ------------------------------- #
    missing = "00000000-0000-4000-8000-000000000000"
    unknown = client.get(f"/v1/demo/sessions/{missing}")
    checks["unknown session explicit error"] = (
        unknown.status_code == 404 and unknown.json()["error"] == "session_not_found"
    )
    invalid = client.post(
        f"/v1/demo/sessions/{session_a.session_id}/chat",
        json={"message": "Recommend products.", "trusted_user_history": ["x"]},
    )
    checks["invalid request explicit error"] = (
        invalid.status_code == 422 and invalid.json()["error"] == "invalid_request"
    )

    # ---- gate 26: session-scoped reset ------------------------------------- #
    before_b = session_b.state()["active_preferences"]
    reset = client.delete(f"/v1/demo/sessions/{session_a.session_id}")
    checks["reset affects only the target session"] = (
        reset.status_code == 200
        and client.get(f"/v1/demo/sessions/{session_a.session_id}").status_code == 404
        and session_b.state()["active_preferences"] == before_b
    )

    # ---- gate 29: determinism across fresh sessions ------------------------ #
    checks["repeated deterministic sequence is stable"] = _signature(
        client, k
    ) == _signature(client, k)

    # ---- reranking demonstration on the real chain ------------------------- #
    # The mandatory script above uses a single avoidance, which the accepted M10A
    # conservatism leaves UNKNOWN for this profile's candidates.  This block installs the
    # fixed synthetic fixture so the demo path is observed actually *reordering* real
    # candidates, and reports the movement.
    demo = new_session(client, "demo-user-1", k)
    for message in DEMO_PREFERENCE_TURNS:
        demo.chat(message)
    demo_turn = demo.chat("Recommend again.")
    demo_original = list(demo_turn["audit"]["original_order"])
    demo_final = [card["parent_asin"] for card in demo_turn["recommendations"]]
    checks["preference fixture reaches the ranking"] = (
        demo_turn["audit"]["ranked_with_preference_count"] == len(DEMO_PREFERENCE_TURNS)
    )
    checks["preference-driven reranking is observable"] = demo_turn["audit"]["moved_count"] > 0
    checks["reordering preserves the candidate set"] = (
        len(demo_final) == len(demo_original) and sorted(demo_final) == sorted(demo_original)
    )
    checks["reordered cards still equal the reranked order"] = demo_final == list(
        demo_turn["audit"]["reranked_order"]
    )
    details["reranking_demonstration"] = {
        "original_order": demo_original,
        "reranked_order": demo_final,
        "moved_count": demo_turn["audit"]["moved_count"],
        "ranked_with_preferences": demo_turn["audit"]["ranked_with_preferences"],
        "candidates": [
            {
                "original_rank": card["original_rank"],
                "reranked_rank": card["reranked_rank"],
                "parent_asin": card["parent_asin"],
                "sasrec_score": card["sasrec_score"],
                "match_count": card["match_count"],
                "violation_count": card["violation_count"],
                "unknown_count": card["unknown_count"],
                "movement_summary": card["movement_summary"],
            }
            for card in demo_turn["recommendations"]
        ],
    }

    # ---- gate 30: M6 endpoints still operational ---------------------------- #
    m6_health = client.get("/health")
    m6_model = client.get("/v1/model")
    m6_recommend = client.post(
        "/v1/recommend", json={"history": list(real_history.parent_asins), "k": k}
    )
    checks["M6 endpoints remain operational"] = (
        m6_health.status_code == 200
        and m6_health.json()["model_loaded"] is True
        and m6_model.status_code == 200
        and m6_recommend.status_code == 200
        and len(m6_recommend.json()["recommendations"]) > 0
    )

    details["engine_recommend_calls"] = engine.recommend_calls
    details["compiled_graph_count"] = runtime.compiled_graph_count
    print("\n--- turn 3 candidates (original -> reranked) ---")
    for row in details["turn3"]:
        print(
            f"  {row['original_rank']:>2} -> {row['reranked_rank']:>2}  {row['parent_asin']:<14} "
            f"{row['sasrec_score']:>+11.6f}  M={row['match_count']} V={row['violation_count']} "
            f"U={row['unknown_count']}"
        )
    print(f"  original order : {original_order}")
    print(f"  reranked order : {card_order}")
    print(f"  moved_count    : {turn3['audit']['moved_count']}")
    print(f"  engine calls   : {engine.recommend_calls}")
    print(f"  audit only     : {turn3['audit']['ranked_with_preferences']}")


def _signature(client: TestClient, k: int) -> str:
    """Run the script in a fresh session and return a session-independent signature.

    Session ids, turn ids and timings are excluded, so the comparison is about
    recommendation *semantics*: order, ranks, evidence counts, rendered text and the
    preference state transitions.
    """
    session = new_session(client, "demo-user-1", k)
    payload: list[Any] = []
    for _label, message in SCRIPT:
        body = session.chat(message)
        payload.append(
            {
                "route": body["route"],
                "cards": [
                    {
                        "original_rank": card["original_rank"],
                        "reranked_rank": card["reranked_rank"],
                        "parent_asin": card["parent_asin"],
                        "match_count": card["match_count"],
                        "violation_count": card["violation_count"],
                        "unknown_count": card["unknown_count"],
                        "movement_summary": card["movement_summary"],
                    }
                    for card in body["recommendations"]
                ],
                "active_preferences": [
                    (item["kind"], item["polarity"], item["value"])
                    for item in body["active_preferences"]
                ],
                "memory_update": body["memory_update"],
                "message": body["message"],
            }
        )
    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
