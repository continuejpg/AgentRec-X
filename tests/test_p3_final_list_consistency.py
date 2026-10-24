"""Phase 3.1: the explanation names exactly the final recommendation list.

These tests pin the rendering seam itself: the text is built from the list the response is
about to publish, in that order, with facts keyed by ``parent_asin`` (never by position).
The end-to-end 48-scenario run exercises the same seam through ``DemoAgentService``.
"""

from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.agent_schemas import AgentRecommendationView
from recommendation.demo.agent_service import _consistent_message


class _Metadata:
    def __init__(self, records):
        self.records = records


def _record(title, main="Sports & Outdoors", category="Camping", store="Acme"):
    return SimpleNamespace(
        title=title,
        main_category=main,
        categories=(main, category),
        store=store,
        details=(),
        price_text=None,
    )


def _recommendation(rank, parent_asin, title=None):
    return AgentRecommendationView(
        rank=rank,
        item_id=rank,
        parent_asin=parent_asin,
        score=float(rank),
        score_kind="sasrec_logit",
        title=title,
        sources=("history", "catalog_search") if rank % 2 else ("catalog_search",),
    )


def _ids(text):
    import re

    return re.findall(r"\bB[0-9A-Z]{9}\b", text)


def test_a_mismatched_text_is_replaced_by_the_final_list() -> None:
    final = [_recommendation(1, "B000000002"), _recommendation(2, "B000000001")]
    metadata = _Metadata({
        "B000000001": _record("Alpha"),
        "B000000002": _record("Beta"),
    })
    stale = "Top 1 candidate(s) from the sequential recommender:\n\n1. B000000003"
    text = _consistent_message(stale, final, metadata)
    assert _ids(text) == ["B000000002", "B000000001"]
    assert "B000000003" not in text


def test_a_text_that_already_matches_is_kept_unchanged() -> None:
    final = [_recommendation(1, "B000000001"), _recommendation(2, "B000000002")]
    original = "Top 2 candidate(s):\n\n1. B000000001\n2. B000000002"
    assert _consistent_message(original, final, _Metadata({})) is original


def test_facts_follow_identity_not_position() -> None:
    final = [_recommendation(1, "B000000002"), _recommendation(2, "B000000001")]
    metadata = _Metadata({
        "B000000001": _record("Alpha", category="Hiking", store="Acme"),
        "B000000002": _record("Beta", category="Camping", store="Zeta"),
    })
    text = _consistent_message("stale: 1. B000000009", final, metadata)
    beta_index = text.index("1. B000000002")
    alpha_index = text.index("2. B000000001")
    assert "Title: Beta" in text[beta_index:alpha_index]
    assert "Category: Camping" in text[beta_index:alpha_index]
    assert "Store: Zeta" in text[beta_index:alpha_index]
    assert "Title: Alpha" in text[alpha_index:]
    assert "Category: Hiking" in text[alpha_index:]
    assert "Store: Acme" in text[alpha_index:]


def test_a_reordered_final_list_reorders_the_explanation() -> None:
    """The preference re-rank case: the text follows the final order, not the old one."""
    metadata = _Metadata({
        "B000000001": _record("Alpha"),
        "B000000002": _record("Beta"),
    })
    first = _consistent_message(
        "stale: 1. B000000001 2. B000000002",
        [_recommendation(1, "B000000001"), _recommendation(2, "B000000002")],
        metadata,
    )
    reranked = _consistent_message(
        first,
        [_recommendation(1, "B000000002"), _recommendation(2, "B000000001")],
        metadata,
    )
    assert _ids(reranked) == ["B000000002", "B000000001"]


def test_an_empty_final_list_keeps_the_existing_constraint_message() -> None:
    constraint_text = (
        "No candidate satisfies the active constraints (price_max=20).\n\n"
        "Candidates assessed: 3; verified eligible: 0; ineligible: 3; unresolved: 0."
    )
    assert _consistent_message(constraint_text, [], _Metadata({})) is constraint_text


def test_missing_catalogue_fields_are_omitted_not_invented() -> None:
    final = [_recommendation(1, "B000000001")]
    metadata = _Metadata({"B000000001": SimpleNamespace(
        title=None, main_category=None, categories=(), store=None, details=(), price_text=None,
    )})
    text = _consistent_message("stale: 1. B000000009", final, metadata)
    assert "B000000001" in text
    for label in ("Title:", "Main category:", "Category:", "Store:"):
        assert label not in text
