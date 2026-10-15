"""V2 M-02A: the M9 preference snapshot must be able to shape the lexical search query.

These tests exercise the real adapter and the real policy helper.  They need no model, no service
and no provider: the adapter is a pure function and the policy helper is the exact function the
controller calls when it builds a ``SEARCH_CATALOG`` proposal.
"""

from __future__ import annotations

import pytest

from recommendation.memory.query_adapter import MAX_ADDED_TERMS, adapt_terms
from recommendation.memory.schemas import (
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
    PreferencePolarity,
    PreferenceStatus,
)


def entry(memory_id: str, kind: str, value: str, *, polarity: str = "prefer",
          status: str = "active", seq: int = 1, user_key: str = "user-a") -> PreferenceMemoryEntry:
    return PreferenceMemoryEntry(
        memory_id=memory_id, user_key=user_key, kind=kind, value=value, polarity=polarity,
        source_text="the user said so", source_turn_id="turn-1", logical_seq=seq,
        created_at=1.0, status=status,
    )


def snapshot(*entries: PreferenceMemoryEntry, user_key: str = "user-a") -> PreferenceMemorySnapshot:
    return PreferenceMemorySnapshot(user_key=user_key, entries=entries)


# ---------------------------------------------------------------- A. baseline


def test_without_memory_the_query_is_unchanged() -> None:
    base = ("hiking", "gear")
    result = adapt_terms(base, None)
    assert result.terms == base
    assert result.added == () and result.memory_ids == () and result.changed is False


def test_an_empty_snapshot_is_the_baseline() -> None:
    assert adapt_terms(("hiking",), snapshot()).terms == ("hiking",)


def test_the_adapter_never_removes_or_rewrites_the_request_terms() -> None:
    base = ("hiking", "gear")
    result = adapt_terms(base, snapshot(entry("m1", "brand", "Salomon")))
    assert result.terms[: len(base)] == base, "the current request keeps priority and its spelling"


# ---------------------------------------------------------------- B. positive memory


def test_a_prefer_preference_changes_the_query() -> None:
    result = adapt_terms(("hiking",), snapshot(entry("m1", "brand", "Salomon")))
    assert result.terms == ("hiking", "Salomon")
    assert result.added == ("Salomon",) and result.memory_ids == ("m1",) and result.changed is True


def test_adaptation_is_deterministic_and_ordered_by_logical_seq() -> None:
    snap = snapshot(entry("m2", "brand", "Salomon", seq=2),
                    entry("m1", "category", "trail running shoes", seq=1))
    first = adapt_terms((), snap)
    second = adapt_terms((), snap)
    assert first.terms == second.terms == ("trail running shoes", "Salomon")


def test_revoked_and_superseded_entries_never_participate() -> None:
    snap = snapshot(entry("m1", "brand", "Salomon", status="removed"),
                    entry("m2", "brand", "La Sportiva", status="superseded"))
    assert adapt_terms(("x",), snap).terms == ("x",)


def test_duplicate_values_are_dropped_case_insensitively() -> None:
    # The schema already forbids blank values (``NonEmptyText``), so only the dedup rule is tested.
    snap = snapshot(entry("m1", "brand", "Salomon", seq=1),
                    entry("m2", "brand", "salomon", seq=2))
    result = adapt_terms(("Salomon",), snap)
    assert result.terms == ("Salomon",) and result.changed is False


# ---------------------------------------------------------------- C. negative memory


def test_avoid_only_memory_leaves_the_query_alone() -> None:
    snap = snapshot(entry("m1", "category", "red", polarity="avoid"))
    result = adapt_terms(("running shoes",), snap)
    assert result.terms == ("running shoes",) and result.changed is False


def test_mixed_memory_appends_only_the_positive_side() -> None:
    snap = snapshot(entry("m1", "color", "red", polarity="avoid"),
                    entry("m2", "feature", "waterproof"))
    result = adapt_terms(("jacket",), snap)
    assert result.terms == ("jacket", "waterproof")
    assert "red" not in result.added, "an AVOID entry is never appended as a positive keyword"


def test_a_negated_user_message_is_left_verbatim_and_is_not_a_failure() -> None:
    """The message may itself contain a negation; that is the message's own text."""
    base = ("no red shoes",)
    snap = snapshot(entry("m1", "color", "red", polarity="avoid"))
    assert adapt_terms(base, snap).terms == base


# ---------------------------------------------------------------- C2. kinds


def test_numeric_and_free_form_kinds_are_not_appended() -> None:
    snap = snapshot(entry("m1", "price_max", "100", seq=1),
                    entry("m2", "price_min", "20", seq=2),
                    entry("m3", "free_form_constraint", "must ship before friday", seq=3))
    assert adapt_terms(("tent",), snap).terms == ("tent",)


def test_the_number_of_appended_terms_is_bounded() -> None:
    snap = snapshot(*[entry(f"m{i}", "brand", f"Brand{i}", seq=i) for i in range(1, 8)])
    assert len(adapt_terms((), snap).added) == MAX_ADDED_TERMS


# ---------------------------------------------------------------- D. isolation


def test_another_users_snapshot_cannot_leak_in() -> None:
    mine = snapshot(entry("m1", "brand", "Salomon", user_key="user-a"), user_key="user-a")
    theirs = snapshot(entry("m9", "brand", "Salomon", user_key="user-b"), user_key="user-b")
    assert adapt_terms((), mine).memory_ids == ("m1",)
    assert adapt_terms((), theirs).memory_ids == ("m9",)
    assert "m9" not in adapt_terms((), mine).memory_ids


# ---------------------------------------------------------------- E/F. real policy helper


def test_the_policy_helper_returns_the_original_object_without_usable_memory() -> None:
    from recommendation.control import SearchCatalogArguments

    from recommendation.demo.agent_policy import _adapt_search_arguments

    arguments = SearchCatalogArguments(terms=("hiking",))
    assert _adapt_search_arguments(arguments, None) is arguments
    assert _adapt_search_arguments(arguments, snapshot()).terms == ("hiking",)


def test_the_policy_helper_folds_memory_into_the_search_arguments() -> None:
    from recommendation.control import SearchCatalogArguments

    from recommendation.demo.agent_policy import _adapt_search_arguments

    adapted = _adapt_search_arguments(
        SearchCatalogArguments(terms=("hiking",)),
        snapshot(entry("m1", "category", "trail running shoes")),
    )
    assert adapted.terms == ("hiking", "trail running shoes"), (
        "the lexical search tool must receive the memory-conditioned query"
    )


def test_the_adapter_is_wired_at_the_search_proposal() -> None:
    """Wiring check: the SEARCH_CATALOG proposal builds its terms through the adapter.

    The behavioural guarantee lives in the two ``_adapt_search_arguments`` tests above; this one
    only confirms the controller-reachable call site exists (the raw builder is the adapter's
    argument, so the baseline path is preserved by construction) - and that the policy reads the
    bounded **projection** the controller hands it rather than the M9 snapshot.
    """
    import pathlib as _pathlib

    source = (_pathlib.Path(__file__).resolve().parents[1] / "recommendation" / "demo"
              / "agent_policy.py").read_text(encoding="utf-8")
    assert "arguments=_adapt_search_arguments(" in source
    assert "getattr(context, \"active_preferences\", None)" in source
    assert "preference_snapshot" not in source, (
        "a policy must not reach the snapshot: the controller hands it a value-level projection"
    )


def test_the_projection_keeps_the_feature_and_carries_nothing_else() -> None:
    """What the controller hands a policy: the ACTIVE values, and not one field more.

    This is the boundary the M-02A feature has to respect.  The feature is preserved exactly (the
    ACTIVE PREFER value still reaches the query terms), while the entry's owner key, its verbatim
    source text, its turn id, its timestamps and every withdrawn entry do not cross.
    """
    from recommendation.memory.query_adapter import (
        ActivePreference,
        ActivePreferenceProjection,
        project_active_preferences,
    )

    snap = snapshot(
        entry("m-active", "brand", "Salomon", seq=2),
        entry("m-removed", "color", "red", status="removed", seq=1),
    )
    projection = project_active_preferences(snap)

    assert isinstance(projection, ActivePreferenceProjection)
    assert [item.value for item in projection.entries] == ["Salomon"], (
        "a withdrawn preference must not cross the boundary at all"
    )
    assert projection.count == 1
    assert isinstance(projection.entries[0], ActivePreference)
    assert set(vars(projection.entries[0])) == {
        "memory_id", "kind", "polarity", "value", "status", "logical_seq"}
    assert projection.entries[0].status == PreferenceStatus.ACTIVE

    # The feature still works through the projection, value for value.
    assert adapt_terms(("hiking",), projection).terms == ("hiking", "Salomon")
    assert adapt_terms(("hiking",), projection).memory_ids == ("m-active",)

    # And nothing sensitive is reachable from what the policy received.
    rendered = repr(projection)
    for leaked in ("user-a", "the user said so", "turn-1"):
        assert leaked not in rendered, leaked
    assert not hasattr(projection, "user_key")
    assert not hasattr(projection.entries[0], "user_key")
    assert not hasattr(projection.entries[0], "source_text")
    assert not hasattr(projection.entries[0], "source_turn_id")
    assert not hasattr(projection.entries[0], "created_at")

    # No snapshot, no memory service: the projection is empty and the query is the baseline.
    assert project_active_preferences(None).entries == ()
    assert project_active_preferences(None).count == 0
    assert adapt_terms(("hiking",), project_active_preferences(None)).terms == ("hiking",)
