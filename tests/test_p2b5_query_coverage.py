"""Phase 2B.5: the balanced title-term query construction.

The function lives in the experiment driver, so the test exercises the same code the
experiment runs.  These tests pin the properties the 2B.5 brief requires: deterministic,
history-only (no target, no future information), bounded, and fair across titles.
"""

from __future__ import annotations

import pytest

from experiments.m4_query_rules import deterministic_terms
from experiments.p2b5_query_coverage import (
    BALANCED_MAX_TERMS,
    BALANCED_TERMS_PER_TITLE,
    BALANCED_TITLES,
    balanced_title_terms,
    build_query,
)


def _describe(mapping: dict[str, str]):
    def describe(identity: str) -> str | None:
        return mapping.get(identity)

    return describe


def test_round_robin_gives_every_title_a_turn_before_any_second_term() -> None:
    history = ["old", "mid", "new"]
    titles = {
        "new": "gamma three green",
        "mid": "beta two blue",
        "old": "alpha one red",
    }
    terms = balanced_title_terms(history, describe=_describe(titles))
    # newest first; one term per title per round; eight-term cap.
    assert terms == ("gamma", "beta", "alpha", "three", "two", "one", "green", "blue")


def test_a_verbose_newest_title_cannot_starve_the_others() -> None:
    history = ["old", "mid", "new"]
    titles = {
        "new": "exfoil 1000lb black paracord 250 spools rope parachute cord",
        "mid": "ecosmart sup electric scooter",
        "old": "serenelife inflatable stand up paddle board",
    }
    terms = balanced_title_terms(history, describe=_describe(titles))
    # the second and third titles still contribute even though the newest one is long.
    assert "ecosmart" in terms
    assert "serenelife" in terms
    assert len(terms) <= BALANCED_MAX_TERMS


def test_the_frozen_rule_starves_the_same_fixture() -> None:
    """Contrast case: the current (frozen) rule spends its budget on the newest title."""
    history = ["old", "mid", "new"]
    titles = {
        "new": "exfoil 1000lb black paracord 250 spools rope parachute cord",
        "mid": "ecosmart sup electric scooter",
        "old": "serenelife inflatable stand up paddle board",
    }
    current = deterministic_terms(history, describe=_describe(titles))
    balanced = balanced_title_terms(history, describe=_describe(titles))
    assert "ecosmart" not in current
    assert "ecosmart" in balanced


def test_the_construction_is_bounded_by_the_total_and_per_title_budget() -> None:
    history = ["aaa", "bbb", "ccc"]
    titles = {name: " ".join(f"{name}{i}" for i in range(10)) for name in history}
    terms = balanced_title_terms(history, describe=_describe(titles))
    assert len(terms) <= BALANCED_MAX_TERMS
    for name in history:
        assert sum(1 for term in terms if term.startswith(name)) <= BALANCED_TERMS_PER_TITLE


def test_short_tokens_are_dropped() -> None:
    terms = balanced_title_terms(["a"], describe=_describe({"a": "go up 12 red"}))
    assert terms == ("red",)


def test_duplicate_terms_are_removed_preserving_first_occurrence() -> None:
    titles = {
        "new": "trail shoes",
        "mid": "trail socks",
        "old": "trail poles",
    }
    terms = balanced_title_terms(["old", "mid", "new"], describe=_describe(titles))
    assert terms == ("trail", "shoes", "socks", "poles")


def test_a_missing_title_contributes_nothing() -> None:
    titles = {"new": "camp stove", "old": "trail poles"}
    terms = balanced_title_terms(["old", "mid", "new"], describe=_describe(titles))
    assert terms == ("camp", "trail", "stove", "poles")


def test_no_titles_yields_no_terms() -> None:
    assert balanced_title_terms(["a", "b"], describe=lambda _: None) == ()
    assert balanced_title_terms([], describe=lambda _: "anything") == ()


def test_only_the_newest_titles_are_read() -> None:
    history = [f"h{index}" for index in range(10)]
    asked: list[str] = []

    def describe(identity: str) -> str | None:
        asked.append(identity)
        return f"title {identity}"

    balanced_title_terms(history, describe=describe)
    assert asked == history[-BALANCED_TITLES:][::-1]


def test_the_target_is_never_read() -> None:
    history = ["h1", "h2", "h3"]
    target = "held-out-target"
    asked: list[str] = []

    def describe(identity: str) -> str | None:
        asked.append(identity)
        return "title"

    balanced_title_terms(history, describe=describe)
    assert target not in asked
    assert set(asked) <= set(history)


def test_the_construction_is_deterministic() -> None:
    history = ["a", "b", "c"]
    titles = {"a": "one two three", "b": "four five six", "c": "seven eight nine"}
    assert balanced_title_terms(history, describe=_describe(titles)) == balanced_title_terms(
        history, describe=_describe(titles)
    )


def test_build_query_dispatches_and_refuses_unknown_modes() -> None:
    history = ["a", "b"]
    titles = {"a": "alpha beta", "b": "gamma delta"}
    describe = _describe(titles)
    assert build_query(history, describe=describe, mode="current") == deterministic_terms(
        history, describe=describe
    )
    assert build_query(history, describe=describe, mode="balanced") == balanced_title_terms(
        history, describe=describe
    )
    with pytest.raises(ValueError):
        build_query(history, describe=describe, mode="something-else")


def test_the_balanced_query_stays_within_the_search_argument_bound() -> None:
    # The trusted SEARCH_CATALOG argument accepts at most MAX_SEARCH_TERMS (16) terms.
    assert BALANCED_MAX_TERMS <= 16
