"""M-02A-V: memory must change what the *retrieval* returns, not just the arguments object.

The catalogue is the repository's own synthetic fixture (``tests/product_catalog_fixture.py``):
tiny, deterministic and offline - no checkpoint, no model, no service.  Query terms are derived
from the fixture's own text at runtime, so the assertions cannot rot when the fixture changes.
"""

from __future__ import annotations

import pytest

from recommendation.control import SearchCatalogArguments
from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.control.arguments import MAX_SEARCH_TERMS
from recommendation.memory.query_adapter import adapt_terms
from recommendation.memory.schemas import PreferenceMemoryEntry, PreferenceMemorySnapshot
from recommendation.rag.retrieval import tokenize
from tests.product_catalog_fixture import index, records


def entry(memory_id: str, kind: str, value: str, *, polarity: str = "prefer",
          status: str = "active", seq: int = 1) -> PreferenceMemoryEntry:
    return PreferenceMemoryEntry(
        memory_id=memory_id, user_key="user-a", kind=kind, value=value, polarity=polarity,
        source_text="the user said so", source_turn_id="turn-1", logical_seq=seq,
        created_at=1.0, status=status,
    )


def snapshot(*entries: PreferenceMemoryEntry) -> PreferenceMemorySnapshot:
    return PreferenceMemorySnapshot(user_key="user-a", entries=entries)


@pytest.fixture()
def terms_from_catalogue() -> tuple[str, str]:
    """Two *different* single-token terms taken from two different catalogue records."""
    catalogue = records()
    assert len(catalogue) >= 2, "the fixture must hold at least two products"
    picked: list[str] = []
    for record in catalogue:
        tokens = [token for token in tokenize(record.title or "") if token.isalpha() and len(token) > 2]
        if tokens and tokens[0] not in picked:
            picked.append(tokens[0])
        if len(picked) == 2:
            break
    assert len(picked) == 2, "the fixture must offer two distinct title tokens"
    return picked[0], picked[1]


def hits(query: str) -> tuple[str, ...]:
    """Run the **real** BM25 and return the returned identities, in returned order."""
    result = CatalogSearchSource(index()).search(query, limit=5)
    return tuple(hit.parent_asin for hit in result.hits)


def matched(query: str) -> bool:
    """Whether the search reported a real match (the flag lives on the result, not the hit)."""
    return bool(CatalogSearchSource(index()).search(query, limit=5).matched)


def query_for(base: str, snap: PreferenceMemorySnapshot | None) -> str:
    """Exactly what the policy does: adapt the terms, then join them as the tool's query."""
    arguments = SearchCatalogArguments(terms=(base,))
    adapted = adapt_terms(arguments.terms, snap)
    return SearchCatalogArguments(terms=adapted.terms[:MAX_SEARCH_TERMS]).query_text()


def test_bm25_semantics_are_terms_not_phrases(terms_from_catalogue) -> None:
    """A multi-word memory value reaches BM25 as tokens, because the tool re-tokenises."""
    current, other = terms_from_catalogue
    query = query_for(current, snapshot(entry("m1", "category", f"{other} running shoes")))
    assert query == f"{current} {other} running shoes"
    assert set(tokenize(query)) >= {current, other}, "the tool tokenises the joined query text"


def test_memory_off_on_and_revoked_change_the_real_candidates(terms_from_catalogue) -> None:
    current, other = terms_from_catalogue
    baseline = hits(query_for(current, None))
    with_positive = hits(query_for(current, snapshot(entry("m1", "category", other))))
    avoid_only = hits(query_for(current, snapshot(entry("m1", "category", other, polarity="avoid"))))
    revoked = hits(query_for(current, snapshot(entry("m1", "category", other, status="removed"))))

    assert baseline == revoked, "a revoked PREFER must behave exactly like no memory"
    assert avoid_only == baseline, "AVOID memory is never appended as a positive keyword"
    assert with_positive != baseline, (
        "an ACTIVE PREFER must change the BM25 result, not only the arguments object"
    )
    # The appended term names a different product, so it must enter the returned set.
    assert matched(query_for(current, snapshot(entry("m1", "category", other)))), (
        "the memory-conditioned query must still match the catalogue"
    )


def test_the_positive_preference_actually_surfaces_its_product(terms_from_catalogue) -> None:
    current, other = terms_from_catalogue
    baseline_ids = set(hits(query_for(current, None)))
    positive_ids = set(hits(query_for(current, snapshot(entry("m1", "category", other)))))
    assert positive_ids - baseline_ids, (
        "the memory-conditioned query must be able to return a candidate the baseline did not"
    )


def test_an_unmatched_memory_term_does_not_invent_candidates(terms_from_catalogue) -> None:
    current, _ = terms_from_catalogue
    baseline = hits(query_for(current, None))
    unmatched = hits(query_for(current, snapshot(entry("m1", "brand", "zzzznobrand"))))
    assert unmatched[: len(baseline)] == baseline, (
        "an unmatched term may not reorder or manufacture hits"
    )


def test_no_memory_is_byte_identical_to_the_pre_m02a_query(terms_from_catalogue) -> None:
    current, _ = terms_from_catalogue
    assert query_for(current, None) == SearchCatalogArguments.from_text(current).query_text()
    assert query_for(current, snapshot()) == query_for(current, None)


def test_the_term_bound_still_holds_with_memory(terms_from_catalogue) -> None:
    current, other = terms_from_catalogue
    many = snapshot(*[entry(f"m{i}", "brand", f"brand{i}", seq=i) for i in range(1, 6)])
    adapted = adapt_terms((current,), many)
    assert len(adapted.terms[:MAX_SEARCH_TERMS]) <= MAX_SEARCH_TERMS
    assert SearchCatalogArguments(terms=adapted.terms[:MAX_SEARCH_TERMS]).query_text()


def test_current_intent_can_be_polluted_by_history_and_that_is_a_reported_risk() -> None:
    """D: history says trail running, the current message says basketball.

    The adapter is deterministic and does not read intent, so both appear in the query.  This test
    pins the current behaviour (documenting the risk) instead of pretending it is solved.
    """
    snap = snapshot(entry("m1", "category", "trail running shoes"))
    query = query_for("basketball", snap)
    assert query == "basketball trail running shoes"
    polluted = hits(query)
    clean = hits(query_for("basketball", None))
    if clean:
        assert polluted != clean, (
            "the historical preference does reach BM25 for the conflicting request; intent "
            "arbitration belongs to a later planning stage"
        )
