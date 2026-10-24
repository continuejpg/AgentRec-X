"""Phase 2B.3: the per-case proxy query and the four-branch attribution.

The functions are imported from the driver itself, so the test exercises the same code the
experiment runs rather than a copy of it.
"""

from __future__ import annotations

from experiments.p2b_agent_source_limit_ablation import (
    PROXY_TERMS_PER_TITLE,
    attach_proxy_queries,
    build_proxy_query,
    classify_target_branch,
    title_terms,
)


def test_title_terms_are_lowercased_alphanumeric_runs() -> None:
    assert title_terms("Funko POP! Marvel #12 - 3.75in") == (
        "funko",
        "pop",
        "marvel",
        "12",
        "3",
        "75in",
    )


def test_title_terms_are_bounded_by_the_per_title_cap() -> None:
    terms = title_terms("one two three four five six seven eight", limit=3)
    assert terms == ("one", "two", "three")


def test_a_missing_title_contributes_no_terms() -> None:
    assert title_terms(None) == ()
    assert title_terms("   ") == ()
    assert title_terms(42) == ()


def test_the_proxy_query_reads_only_the_seed_items() -> None:
    asked: list[str] = []

    def title_of(parent_asin: str) -> str | None:
        asked.append(parent_asin)
        return {"a": "red hiking boots", "b": "waterproof trail shoes"}.get(parent_asin)

    query, detail = build_proxy_query(("a", "b"), title_of)
    assert asked == ["a", "b"]
    assert query == "red hiking boots waterproof trail shoes"
    assert detail["terms"] == query.split()
    assert [item["parent_asin"] for item in detail["seed_items"]] == ["a", "b"]


def test_the_proxy_query_is_deterministic() -> None:
    titles = {"a": "alpine trekking poles", "b": "insulated water bottle"}
    first = build_proxy_query(("a", "b"), titles.get)
    second = build_proxy_query(("a", "b"), titles.get)
    assert first == second


def test_duplicate_terms_are_removed_preserving_first_occurrence() -> None:
    titles = {"a": "trail shoes for trail running", "b": "trail socks"}
    query, _ = build_proxy_query(("a", "b"), titles.get)
    assert query == "trail shoes for running socks"


def test_a_missing_title_on_one_seed_falls_back_to_the_other() -> None:
    titles = {"a": None, "b": "camp stove"}
    query, detail = build_proxy_query(("a", "b"), titles.get)
    assert query == "camp stove"
    assert detail["seed_items"][0]["terms"] == []


def test_no_usable_title_yields_no_query_instead_of_a_fabricated_one() -> None:
    query, detail = build_proxy_query(("a", "b"), lambda _: None)
    assert query is None
    assert detail["terms"] == []


def test_the_query_stays_within_the_search_argument_bound() -> None:
    # 2 x PROXY_TERMS_PER_TITLE must stay <= MAX_SEARCH_TERMS (16).
    query, _ = build_proxy_query(("a", "b"), lambda _: " ".join(["word"] * 40))
    assert query is not None
    assert len(query.split()) <= PROXY_TERMS_PER_TITLE * 2 <= 16


def test_attach_proxy_queries_uses_the_last_two_history_items_only() -> None:
    case = {
        "case_index": 7,
        "user_id": "u",
        "history": ["h1", "h2", "h3", "h4"],
        "target": "held-out-target",
    }
    askable = {"h3": "alpha beta", "h4": "gamma delta"}
    asked: list[str] = []

    def title_of(parent_asin: str) -> str | None:
        asked.append(parent_asin)
        return askable.get(parent_asin)

    attach_proxy_queries([case], title_of)
    assert asked == ["h3", "h4"]
    assert case["proxy_seed_asins"] == ["h3", "h4"]
    assert case["proxy_query"] == "alpha beta gamma delta"
    assert case["query_source"] == "history-title-proxy"
    assert "held-out-target" not in case["proxy_query"]


def test_attach_proxy_queries_marks_an_unusable_case_with_a_null_query() -> None:
    case = {"case_index": 1, "user_id": "u", "history": ["h1", "h2"], "target": "t"}
    attach_proxy_queries([case], lambda _: None)
    assert case["proxy_query"] is None
    assert case["proxy_terms"] == []


def test_the_four_branches_are_exclusive_and_exhaustive() -> None:
    assert classify_target_branch(appeared=False, in_pool=False, final_rank=None) == (
        "never_returned"
    )
    assert classify_target_branch(appeared=True, in_pool=False, final_rank=None) == "filtered"
    assert classify_target_branch(appeared=True, in_pool=True, final_rank=None) == (
        "pooled_not_top_k"
    )
    assert classify_target_branch(appeared=True, in_pool=True, final_rank=3) == "top_k"


def test_a_final_rank_always_wins_the_classification() -> None:
    # A target that was presented is a top-k case even if a source-level flag disagreed.
    assert classify_target_branch(appeared=False, in_pool=False, final_rank=1) == "top_k"
