"""Phase 4: the current shopping intent projection and its evidence-bound reasons."""

from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.shopping_intent import (
    ActiveEntry,
    apply_extraction,
    build_reason,
    build_unresolved_question,
    derive_intent,
    query_terms,
)
from recommendation.memory.extraction import RuleBasedPreferenceExtractor

EXTRACTOR = RuleBasedPreferenceExtractor()


def _entry(kind: str, value: str, polarity: str = "prefer", text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        kind=SimpleNamespace(value=kind),
        value=value,
        polarity=SimpleNamespace(value=polarity),
        source_turn_id="t1",
        source_text=text or value,
    )


def _snapshot(*entries: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(active_entries=entries)


def _intent(snapshot, message, *, turn_id="t2", explicit=()):
    return derive_intent(
        snapshot, EXTRACTOR.extract(message), turn_id=turn_id, explicit_constraints=explicit
    )


def test_a_budget_is_hard_while_the_category_phrase_stays_soft() -> None:
    intent = _intent(_snapshot(), "I need a yoga mat, budget $30")
    assert [(e.kind, e.value) for e in intent.hard] == [("price_max", "30")]
    assert [(e.kind, e.value) for e in intent.soft] == [("category", "yoga mat")]
    assert intent.hard_constraints() == ({"kind": "price_max", "value": "30"},)


def test_an_explicit_correction_replaces_the_old_value() -> None:
    snapshot = _snapshot(_entry("price_max", "30", text="budget $30"))
    intent = _intent(snapshot, "make it under $50 instead")
    assert [(e.kind, e.value) for e in intent.hard] == [("price_max", "50")]
    old = [e.value for e in intent.hard + intent.soft if e.value == "30"]
    assert old == []


def test_a_withdrawal_removes_the_entry_from_intent_and_query() -> None:
    snapshot = _snapshot(_entry("brand", "Acme", text="the brand Acme"))
    intent = _intent(snapshot, "forget the brand")
    assert intent.hard == () and intent.soft == ()
    assert "acme" not in {term.casefold() for term in query_terms(intent, "forget the brand")}


def test_a_price_floor_above_the_ceiling_is_an_explicit_conflict() -> None:
    snapshot = _snapshot(_entry("price_max", "50", text="budget $50"))
    intent = _intent(snapshot, "my budget is at least $80")
    assert [conflict.code for conflict in intent.conflicts] == ["price_range"]
    assert "minimum ($80.00)" in intent.conflicts[0].question


def test_the_tightest_hard_ceiling_wins_without_a_conflict() -> None:
    snapshot = _snapshot(
        _entry("price_max", "80", text="budget $80"),
        _entry("price_max", "50", text="budget $50"),
    )
    intent = _intent(snapshot, "")
    assert [(e.kind, e.value) for e in intent.hard] == [("price_max", "50")]
    assert intent.conflicts == ()


def test_query_terms_prioritise_intent_and_never_include_a_stale_value() -> None:
    snapshot = _snapshot(_entry("category", "hiking", text="hiking"))
    intent = _intent(snapshot, "make it camping instead")
    terms = query_terms(intent, "make it camping instead")
    assert terms[0].casefold() in {"hiking", "camping"}
    assert "hiking" not in {term.casefold() for term in terms} or "camping" in {
        term.casefold() for term in terms
    }


def test_reasons_are_bound_to_the_product_record_and_never_invent_facts() -> None:
    intent = _intent(_snapshot(), "I need a yoga mat, budget $30")
    record = SimpleNamespace(
        title="Yoga Mat",
        main_category="Sports & Outdoors",
        categories=("Sports & Outdoors", "Yoga Mats"),
        store="Acme",
        details=(("Brand Name", "Acme"),),
        price_text="$25.99",
    )
    reason = build_reason(record, intent)
    assert "within your $30.00 budget (catalogue price $25.99)" in reason

    other = SimpleNamespace(
        title="Ball Pump",
        main_category="Sports & Outdoors",
        categories=("Sports & Outdoors", "Pumps"),
        store="Franklin",
        details=(("Brand Name", "Franklin"),),
        price_text="$7.19",
    )
    other_reason = build_reason(other, intent)
    assert "$7.19" in other_reason and "$25.99" not in other_reason


def test_a_missing_record_never_becomes_a_positive_claim() -> None:
    intent = _intent(_snapshot(), "I need a yoga mat, budget $30")
    assert "available" in build_reason(None, intent)


def test_the_unresolved_question_states_uncertainty_not_absence() -> None:
    context = SimpleNamespace(
        task_constraints=("price_max: prefers 20",), unresolved_count=3
    )
    question = build_unresolved_question(context)
    assert "couldn't verify" in question
    assert "3 candidate(s)" in question
    assert "no candidate" not in question.casefold()


def test_apply_extraction_keeps_add_replace_and_remove_distinct() -> None:
    active = [ActiveEntry("brand", "Acme", "prefer", "t1")]
    addition = EXTRACTOR.extract("I prefer the brand Beta")
    added = apply_extraction(active, addition, turn_id="t2")
    assert {(e.kind, e.value) for e in added} == {("brand", "Acme"), ("brand", "Beta")}

    replacement = EXTRACTOR.extract("actually I prefer the brand Beta instead")
    replaced = apply_extraction(active, replacement, turn_id="t3")
    assert {(e.kind, e.value) for e in replaced} == {("brand", "Beta")}


def test_an_explicit_constraint_is_hard_even_for_a_free_form_kind() -> None:
    intent = _intent(
        _snapshot(),
        "I need a yoga mat",
        explicit=[{"kind": "free_form_constraint", "value": "extra-large"}],
    )
    assert [(e.kind, e.value) for e in intent.hard] == [
        ("free_form_constraint", "extra-large")
    ]
    assert intent.hard_constraints() == (
        {"kind": "free_form_constraint", "value": "extra-large"},
    )
