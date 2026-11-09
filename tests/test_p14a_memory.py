"""Contract tests for the P14-A Shopping Companion memory adapter (M5/M9 reuse).

These tests pin the *honesty* contract of
:class:`recommendation.shopping_companion.memory.ShoppingCompanionMemory`:

* a fresh instance reports ``history_unavailable`` (the released dataset ships only
  train/test parquet and ``products.jsonl``; the conversation-history corpus is NOT
  released) and never fabricates a preference;
* the synthetic fixture is used ONLY as a contract test - it is explicitly marked
  synthetic in the fixture file and is never claimed to be released history;
* ingestion through the accepted M9 service extracts preferences with traceable
  per-turn evidence, every evidence item tagged with its ``source_kind``;
* oracle-diagnostic ingestion requires the explicit ``released_oracle_diagnostic`` tag,
  is served only by an instance constructed with ``oracle_diagnostic=True``, and is
  refused (never silently consumed) in normal mode;
* the M5 path is real: ingested user turns become accepted ``StateEvent`` batches
  replayed through ``StateManager`` and returned by ``state_projector.for_agent``;
* the module has no pyarrow / network / model dependency and its payloads are
  JSON-serialisable for the trajectory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from recommendation.shopping_companion.memory import (
    ORACLE_REFUSED_REASON,
    RELEASED_HISTORY_REASON,
    SOURCE_KIND_ORACLE_DIAGNOSTIC,
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    STATUS_HISTORY_UNAVAILABLE,
    STATUS_ORACLE_DIAGNOSTIC,
    STATUS_SYNTHETIC_FIXTURE,
    MemoryIngestError,
    OracleDiagnosticRefused,
    ShoppingCompanionMemory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "p14a" / "memory_fixture.jsonl"
FIXED_CLOCK = 1_700_000_000.0

OUTDOOR_CONV = "synthetic_conv_outdoor_001"
HOME_CONV = "synthetic_conv_home_002"

#: Dialogue used ONLY to exercise the oracle-diagnostic gate (never normal success).
ORACLE_SESSIONS: tuple[dict[str, Any], ...] = (
    {
        "session_id": "oracle_session_1",
        "date": "2023-11-20",
        "turns": [
            {"role": "user", "content": "I prefer the brand Acme."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "My budget is under $80."},
        ],
    },
)


def _memory(**kwargs: Any) -> ShoppingCompanionMemory:
    """A memory instance with a deterministic clock."""
    return ShoppingCompanionMemory(clock=lambda: FIXED_CLOCK, **kwargs)


def _fixture_rows() -> list[dict[str, Any]]:
    """Load the committed synthetic fixture and assert it is clearly marked synthetic."""
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "the memory fixture must not be empty"
    for row in rows:
        assert row["synthetic"] is True
        assert "synthetic" in row["notice"].lower()
        assert row["schema"] == "agentrecx.p14a.memory_fixture.v1"
    return rows


def _ingest_fixture(memory: ShoppingCompanionMemory) -> list[dict[str, Any]]:
    """Ingest every fixture conversation as synthetic contract-test evidence."""
    rows = _fixture_rows()
    for row in rows:
        memory.ingest_sessions(
            row["conversation_id"],
            row["sessions"],
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref=row["source_ref"],
        )
    return rows


# --------------------------------------------------------------------------- #
# honest empty state
# --------------------------------------------------------------------------- #


def test_fresh_memory_reports_history_unavailable_without_fabricating() -> None:
    """No ingestion => history_unavailable, zero preferences, and the real reason."""
    memory = _memory()
    result = memory.search("what hiking boots do I like?")

    assert result["available"] is False
    assert result["status"] == STATUS_HISTORY_UNAVAILABLE
    assert result["preferences"] == []
    assert result["evidence"] == []
    assert result["conflicts"] == []
    assert "not released" in result["reason"].lower()
    assert "conversation-history" in result["reason"].lower()
    assert RELEASED_HISTORY_REASON in result["reason"]
    json.dumps(result)  # the runtime stores this payload in a JSON trajectory

    status = memory.history_status()
    assert status["available"] is False
    assert status["history_available"] is False
    assert status["status"] == STATUS_HISTORY_UNAVAILABLE
    assert status["conversations"] == 0
    assert status["active_preferences"] == 0
    assert status["released_history_present"] is False
    json.dumps(status)

    snapshot = memory.snapshot()
    assert snapshot["conversations"] == []
    assert snapshot["preferences"] == []
    assert snapshot["evidence"] == []
    assert snapshot["history"]["status"] == STATUS_HISTORY_UNAVAILABLE
    json.dumps(snapshot)

    # A scope P14-A cannot honestly serve is unavailable, not invented.
    scoped = memory.search("anything", scope="session")
    assert scoped["available"] is False
    assert scoped["status"] == STATUS_HISTORY_UNAVAILABLE
    assert "no live task/session" in scoped["reason"]

    with pytest.raises(ValueError):
        memory.search("anything", scope="not_a_scope")
    with pytest.raises(ValueError):
        memory.search("anything", k=0)
    with pytest.raises(ValueError):
        memory.search(123)  # type: ignore[arg-type]


def test_fixture_dialogues_are_explicitly_marked_synthetic() -> None:
    """The only fixture shipped here is synthetic contract data, never history."""
    rows = _fixture_rows()
    assert len(rows) == 2
    assert {row["conversation_id"] for row in rows} == {OUTDOOR_CONV, HOME_CONV}


# --------------------------------------------------------------------------- #
# synthetic ingestion + search + provenance
# --------------------------------------------------------------------------- #


def test_synthetic_ingestion_extracts_preferences_with_traceable_evidence() -> None:
    """Ingestion writes through M9 and search returns traceable, tagged evidence."""
    memory = _memory()
    _ingest_fixture(memory)

    status = memory.history_status()
    assert status["available"] is True
    assert status["status"] == STATUS_SYNTHETIC_FIXTURE
    assert status["conversations"] == 2
    assert status["sessions"] == 3
    assert status["user_turns"] == 6
    assert status["evidence_turns"] == 9
    assert status["active_preferences"] == 6
    assert status["oracle_diagnostic_evidence_present"] is False
    assert status["synthetic_evidence_present"] is True

    result = memory.search("hiking")
    assert result["available"] is True
    assert result["status"] == STATUS_SYNTHETIC_FIXTURE
    assert result["returned_preference_count"] == 1
    preference = result["preferences"][0]
    assert (
        preference["kind"],
        preference["polarity"],
        preference["value"],
    ) == ("category", "prefer", "hiking")
    assert preference["source"] == SOURCE_KIND_SYNTHETIC_FIXTURE
    assert preference["source_kind"] == SOURCE_KIND_SYNTHETIC_FIXTURE
    assert preference["status"] == "active"
    assert preference["memory_id"]
    assert preference["source_turn_id"] == "s0t0"

    assert len(result["evidence"]) == 1
    evidence = result["evidence"][0]
    assert evidence["conversation_id"] == OUTDOOR_CONV
    assert evidence["session_index"] == 0
    assert evidence["turn_index"] == 0
    assert evidence["date"] == "2024-01-05"
    assert evidence["role"] == "user"
    assert "hiking" in evidence["text"]
    assert evidence["source_kind"] == SOURCE_KIND_SYNTHETIC_FIXTURE
    assert evidence["source_ref"].endswith(OUTDOOR_CONV)
    assert evidence["extracted_preferences"] == [
        {"kind": "category", "polarity": "prefer", "value": "hiking"}
    ]
    json.dumps(result)

    # The accepted extractor's other kinds are reachable too.
    budget = memory.search("budget")
    assert [(p["kind"], p["value"]) for p in budget["preferences"]] == [
        ("price_max", "120")
    ]
    avoid = memory.search("leather")
    assert [(p["kind"], p["polarity"], p["value"]) for p in avoid["preferences"]] == [
        ("material", "avoid", "leather")
    ]
    brand = memory.search("acme")
    assert [(p["kind"], p["value"]) for p in brand["preferences"]] == [("brand", "Acme")]
    assert brand["preferences"][0]["conversation_id"] == HOME_CONV

    # Every evidence item from a multi-turn search is tagged with its source kind.
    water = memory.search("water filter")
    assert water["preferences"][0]["value"] == "water filter"
    assert all(
        item["source_kind"] == SOURCE_KIND_SYNTHETIC_FIXTURE for item in water["evidence"]
    )

    snapshot = memory.snapshot()
    assert len(snapshot["conversations"]) == 2
    assert len(snapshot["preferences"]) == 6
    assert len(snapshot["evidence"]) == 9
    assert all(
        item["source_kind"] == SOURCE_KIND_SYNTHETIC_FIXTURE
        for item in snapshot["evidence"]
    )
    assert snapshot["m5"]["event_count"] > 0
    json.dumps(snapshot)


def test_search_returns_empty_available_preferences_when_nothing_matches() -> None:
    """A query with no lexical match returns nothing - it never invents a preference."""
    memory = _memory()
    _ingest_fixture(memory)

    result = memory.search("quantum flux capacitor")
    assert result["available"] is True
    assert result["status"] == STATUS_SYNTHETIC_FIXTURE
    assert result["preferences"] == []
    assert result["evidence"] == []
    assert result["conflicts"] == []
    assert result["active_preference_count"] == 6
    assert "none is invented" in result["reason"]


def test_search_reports_conflicts_for_contradictory_memories() -> None:
    """A genuine same-kind/value contradiction is reported, not silently merged."""
    memory = _memory()
    memory.ingest_sessions(
        "synthetic_conv_conflict_003",
        [
            {
                "date": "2024-04-01",
                "turns": [
                    {"role": "user", "content": "I prefer red."},
                    {"role": "user", "content": "I don't want red."},
                ],
            }
        ],
        source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
        source_ref="inline:#synthetic_conv_conflict_003",
    )

    result = memory.search("color")
    assert {p["polarity"] for p in result["preferences"]} == {"prefer", "avoid"}
    assert len(result["conflicts"]) == 1
    conflict = result["conflicts"][0]
    assert (conflict["kind"], conflict["value"]) == ("color", "red")
    assert conflict["polarities"] == ["avoid", "prefer"]
    assert len(conflict["memory_ids"]) == 2

    limited = memory.search("color", k=1)
    assert len(limited["preferences"]) == 1
    assert len(limited["conflicts"]) == 0  # a conflict over one returned entry cannot exist


def test_search_supports_conversation_filter_and_k() -> None:
    """Search can be confined to one conversation; unknown conversations are refused."""
    memory = _memory()
    _ingest_fixture(memory)

    all_colour = memory.search("color")
    assert {p["conversation_id"] for p in all_colour["preferences"]} == {OUTDOOR_CONV}

    confined = memory.search("color", conversation_id=HOME_CONV)
    assert confined["available"] is True
    assert confined["preferences"] == []
    assert confined["conversation_id"] == HOME_CONV

    unknown = memory.search("color", conversation_id="not-ingested-conversation")
    assert unknown["available"] is False
    assert unknown["status"] == STATUS_HISTORY_UNAVAILABLE
    assert "was not ingested" in unknown["reason"]


# --------------------------------------------------------------------------- #
# ingestion validation
# --------------------------------------------------------------------------- #


def test_blank_turns_are_skipped_and_malformed_ingestion_is_refused() -> None:
    """Blank evidence is skipped and counted; structural garbage is refused."""
    memory = _memory()
    summary = memory.ingest_sessions(
        "synthetic_conv_blank_004",
        [
            {
                "date": "2024-04-01",
                "turns": [
                    {"role": "user", "content": "   "},
                    {"role": "user", "content": "I prefer blue."},
                ],
            }
        ],
        source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
        source_ref="inline:#synthetic_conv_blank_004",
    )
    assert summary["turns_skipped_blank"] == 1
    assert summary["evidence_turns"] == 1
    assert summary["preferences_added"] == 1
    assert memory.search("blue")["preferences"][0]["value"] == "blue"

    valid_session = [
        {"date": "2024-01-01", "turns": [{"role": "user", "content": "I prefer blue."}]}
    ]
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "blank_sessions", [], source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref="x"
        )
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "empty_turns",
            [{"date": "2024-01-01", "turns": []}],
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref="x",
        )
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "missing_date",
            [{"turns": [{"role": "user", "content": "I prefer blue."}]}],
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref="x",
        )
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "all_blank",
            [{"date": "2024-01-01", "turns": [{"role": "user", "content": "   "}]}],
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref="x",
        )
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "", valid_session, source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref="x"
        )
    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            "blank_source_ref",
            valid_session,
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref="  ",
        )
    # Nothing above was stored: refusals never leave partial evidence behind.
    assert memory.history_status()["conversations"] == 1


def test_history_source_kind_is_refused_by_name() -> None:
    """A fabricated 'history' label is refused before anything is ingested."""
    memory = _memory()
    sessions = _fixture_rows()[0]["sessions"]
    with pytest.raises(MemoryIngestError) as refused:
        memory.ingest_sessions(
            "fabricated_history", sessions, source_kind="history", source_ref="nowhere"
        )
    message = str(refused.value).lower()
    assert "refused" in message
    assert "conversation-history" in message
    assert memory.history_status()["conversations"] == 0

    for illegal in ("", "synthetic", None):
        with pytest.raises(MemoryIngestError):
            memory.ingest_sessions(
                "illegal_tag",
                sessions,
                source_kind=illegal,  # type: ignore[arg-type]
                source_ref="nowhere",
            )
    assert memory.history_status()["conversations"] == 0


# --------------------------------------------------------------------------- #
# oracle-diagnostic gate
# --------------------------------------------------------------------------- #


def test_oracle_ingestion_requires_explicit_tag_and_is_refused_in_normal_mode() -> None:
    """Oracle evidence is stored but never served (or M5-read) by a strict instance."""
    strict = _memory()
    summary = strict.ingest_sessions(
        "oracle_conv_001",
        list(ORACLE_SESSIONS),
        source_kind=SOURCE_KIND_ORACLE_DIAGNOSTIC,
        source_ref="data/p14a/dataset/train_eval.jsonl#task-1",
    )
    assert summary["oracle_diagnostic"] is True
    assert summary["status"] == STATUS_ORACLE_DIAGNOSTIC

    status = strict.history_status()
    assert status["available"] is False
    assert status["status"] == STATUS_HISTORY_UNAVAILABLE
    assert status["oracle_diagnostic_evidence_present"] is True
    assert ORACLE_REFUSED_REASON in status["reason"]

    result = strict.search("brand")
    assert result["available"] is False
    assert result["status"] == STATUS_HISTORY_UNAVAILABLE
    assert result["preferences"] == []
    assert result["evidence"] == []
    assert result["oracle_diagnostic_evidence_present"] is True
    assert ORACLE_REFUSED_REASON in result["reason"]
    json.dumps(result)

    with pytest.raises(OracleDiagnosticRefused):
        strict.m5_state("oracle_conv_001")
    with pytest.raises(OracleDiagnosticRefused):
        strict.m5_agent_projection("oracle_conv_001")
    with pytest.raises(OracleDiagnosticRefused):
        strict.m5_long_term_projection("oracle_conv_001")

    # The snapshot audit keeps the evidence, always tagged.
    snapshot = strict.snapshot()
    assert len(snapshot["evidence"]) == 3
    assert all(
        item["source_kind"] == SOURCE_KIND_ORACLE_DIAGNOSTIC
        for item in snapshot["evidence"]
    )


def test_oracle_mode_returns_tagged_evidence_only_when_explicit() -> None:
    """Only an explicitly-labelled oracle diagnostic may receive oracle evidence."""
    memory = _memory(oracle_diagnostic=True)
    memory.ingest_sessions(
        "oracle_conv_001",
        list(ORACLE_SESSIONS),
        source_kind=SOURCE_KIND_ORACLE_DIAGNOSTIC,
        source_ref="data/p14a/dataset/train_eval.jsonl#task-1",
    )

    status = memory.history_status()
    assert status["available"] is True
    assert status["status"] == STATUS_ORACLE_DIAGNOSTIC
    assert "oracle-diagnostic" in status["reason"].lower()

    result = memory.search("budget")
    assert result["available"] is True
    assert result["status"] == STATUS_ORACLE_DIAGNOSTIC
    assert [(p["kind"], p["value"]) for p in result["preferences"]] == [
        ("price_max", "80")
    ]
    assert result["preferences"][0]["source"] == SOURCE_KIND_ORACLE_DIAGNOSTIC
    assert result["preferences"][0]["source_kind"] == SOURCE_KIND_ORACLE_DIAGNOSTIC
    assert all(
        item["source_kind"] == SOURCE_KIND_ORACLE_DIAGNOSTIC for item in result["evidence"]
    )

    projection = memory.m5_agent_projection("oracle_conv_001")
    assert projection.task_id == "p14a_memory_released_oracle_diagnostic"
    assert ("price_max", "80") in {
        (p.kind.value, p.value) for p in projection.preferences
    }

    long_term = memory.m5_long_term_projection("oracle_conv_001")
    assert ("price_max", "80") in {
        (p.kind.value, p.value) for p in long_term.preferences
    }
    assert all(p.scope.value == "long_term" for p in long_term.preferences)


def test_normal_mode_never_returns_oracle_evidence_even_when_mixed() -> None:
    """Even with oracle evidence ingested, a strict instance serves only synthetic data."""
    memory = _memory()
    _ingest_fixture(memory)
    memory.ingest_sessions(
        "oracle_conv_002",
        list(ORACLE_SESSIONS),
        source_kind=SOURCE_KIND_ORACLE_DIAGNOSTIC,
        source_ref="data/p14a/dataset/train_eval.jsonl#task-2",
    )

    for query in ("hiking", "brand", "budget", "acme"):
        result = memory.search(query)
        assert all(
            item["source_kind"] == SOURCE_KIND_SYNTHETIC_FIXTURE
            for item in result["evidence"]
        )
        assert all(
            preference["source"] == SOURCE_KIND_SYNTHETIC_FIXTURE
            for preference in result["preferences"]
        )

    status = memory.history_status()
    assert status["available"] is True
    assert status["status"] == STATUS_SYNTHETIC_FIXTURE
    assert status["oracle_diagnostic_evidence_present"] is True

    with pytest.raises(OracleDiagnosticRefused):
        memory.m5_state("oracle_conv_002")


# --------------------------------------------------------------------------- #
# M5 integration (real read path via the accepted projector)
# --------------------------------------------------------------------------- #


def test_m5_integration_projects_ingested_preferences_for_agent() -> None:
    """Ingested user turns become replayed M5 events and reach ``for_agent``."""
    memory = _memory()
    _ingest_fixture(memory)

    task_id = memory.m5_task_id(OUTDOOR_CONV)
    assert task_id == "p14a_memory_synthetic_fixture"

    state = memory.m5_state(OUTDOOR_CONV)
    assert state.user_key == OUTDOOR_CONV
    assert state.task_id == task_id
    assert state.interaction.user_turn_count == 4
    assert state.interaction.event_count > 0

    projection = memory.m5_agent_projection(OUTDOOR_CONV)
    assert projection.task_id == task_id
    projected = {(p.kind.value, p.polarity.value, p.value) for p in projection.preferences}
    assert {
        ("category", "prefer", "hiking"),
        ("price_max", "prefer", "120"),
        ("color", "prefer", "blue"),
        ("material", "avoid", "leather"),
    } <= projected
    # The accepted M5.1 event producer emits task-scope preferences; the adapter does
    # not relabel them (the long-term read path is the M9-backed ``search``).
    assert all(p.scope.value == "task" for p in projection.preferences)

    # The accepted M9 -> M5 long-term bridge is a second, pure-read M5 projection: the
    # same ACTIVE M9 entries appear as LONG_TERM / EXPLICIT / LEGACY_MEMORY beliefs.
    long_term_view = memory.m5_long_term_projection(OUTDOOR_CONV)
    assert long_term_view.task_id == task_id
    long_term = {(p.kind.value, p.polarity.value, p.value) for p in long_term_view.preferences}
    assert long_term == projected
    assert all(p.scope.value == "long_term" for p in long_term_view.preferences)

    snapshot = memory.snapshot()
    record = next(
        row for row in snapshot["conversations"] if row["conversation_id"] == OUTDOOR_CONV
    )
    assert record["m5_event_count"] == state.interaction.event_count
    assert record["m5_task_id"] == task_id
    assert record["m5_unmapped_retractions"] == 0

    with pytest.raises(KeyError):
        memory.m5_state("conversation-that-was-not-ingested")


def test_reingesting_identical_sessions_is_idempotent() -> None:
    """Re-processing the same turn produces no duplicate evidence or M5 events."""
    memory = _memory()
    row = _fixture_rows()[0]
    first = memory.ingest_sessions(
        row["conversation_id"],
        row["sessions"],
        source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
        source_ref=row["source_ref"],
    )
    second = memory.ingest_sessions(
        row["conversation_id"],
        row["sessions"],
        source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
        source_ref=row["source_ref"],
    )
    assert second["idempotent"] is True
    assert second["preferences_added"] == 0
    assert second["m5"]["event_count"] == first["m5"]["event_count"]
    assert len(memory.search("hiking")["preferences"]) == 1
    assert len(memory.snapshot()["evidence"]) == first["evidence_turns"]

    with pytest.raises(MemoryIngestError):
        memory.ingest_sessions(
            row["conversation_id"],
            list(ORACLE_SESSIONS),
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref=row["source_ref"],
        )


# --------------------------------------------------------------------------- #
# runtime-protocol and dependency checks
# --------------------------------------------------------------------------- #


def test_memory_satisfies_the_runtime_memory_protocol() -> None:
    """The payload shapes ``planner_bridge``/``runtime`` consume are present and JSON-safe."""
    memory = _memory()
    _ingest_fixture(memory)

    payload = memory.search("acme", scope="long_term", k=10)
    assert {"available", "status", "preferences", "evidence", "conflicts", "reason"} <= set(
        payload
    )
    for evidence in payload["evidence"]:
        assert "source_kind" in evidence  # the bridge's oracle-leak check reads this
    for preference in payload["preferences"]:
        assert {"kind", "polarity", "value", "status", "source"} <= set(preference)
    json.dumps(payload)

    status = memory.history_status()
    assert "status" in status and "available" in status
    json.dumps(status)


def test_memory_module_has_no_pyarrow_network_or_model_dependency() -> None:
    """Importing the adapter must not pull pyarrow, a network client or a model stack."""
    forbidden = (
        "pyarrow",
        "torch",
        "requests",
        "transformers",
        "openai",
        "huggingface_hub",
    )
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import recommendation.shopping_companion.memory\n"
        f"print(json.dumps({{name: name in sys.modules for name in {forbidden!r}}}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == {name: False for name in forbidden}

    source = (REPO_ROOT / "recommendation" / "shopping_companion" / "memory.py").read_text(
        encoding="utf-8"
    )
    for marker in ("pyarrow", "requests", "ground_truth", "target_product"):
        assert marker not in source
