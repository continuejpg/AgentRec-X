"""Phase 4: the current shopping intent projection and its evidence-bound reasons."""

from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.shopping_intent import (
    ActiveEntry,
    apply_extraction,
    build_reason,
    build_unresolved_question,
    derive_intent,
    promote_explicit_type_matches,
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


def test_same_scope_price_corrections_keep_only_the_latest_hard_value() -> None:
    """30 -> 10 -> 50 must not keep old ceilings in the effective intent."""
    snapshot = _snapshot(
        _entry("price_max", "30", text="I need a water filter under 30 dollars."),
        _entry("price_max", "10", text="Can you find one under 10 dollars?"),
    )
    intent = _intent(snapshot, "Actually make the budget under 50 dollars.")
    assert [(e.kind, e.value) for e in intent.hard] == [("price_max", "50")]
    assert "30" not in {e.value for e in intent.hard + intent.soft}
    assert "10" not in {e.value for e in intent.hard + intent.soft}
    assert intent.hard[0].source_text == "Actually make the budget under 50 dollars."


def test_superseded_price_clause_is_removed_from_r1_query_terms() -> None:
    from recommendation.demo.agent_service import _sanitize_price_query_terms

    assert _sanitize_price_query_terms(
        ("water filter under 30 dollars", "Show", "more", "options.")
    ) == ("water filter", "Show", "more", "options.")


def test_role_target_change_detector_is_general_and_preserves_withdrawal_fallback() -> None:
    from types import SimpleNamespace
    from recommendation.demo.agent_service import DemoAgentService

    service = SimpleNamespace(
        _role_current_targets={},
        _role_main_targets={},
    )
    update = DemoAgentService._update_role_target
    assert update(service, "task", "I need a complete water filter") is True
    assert update(service, "task", "I need a replacement mouthpiece for a water filter") is True
    # A pure withdrawal of the temporary accessory falls back to the last main target.
    assert update(service, "task", "forget the mouthpiece requirement") is True
    assert service._role_current_targets["task"].type == "water_filter"
    assert service._role_current_targets["task"].role.value == "main"


def test_a_withdrawal_removes_the_entry_from_intent_and_query() -> None:
    snapshot = _snapshot(_entry("brand", "Acme", text="the brand Acme"))
    intent = _intent(snapshot, "forget the brand")
    assert intent.hard == () and intent.soft == ()
    assert "acme" not in {term.casefold() for term in query_terms(intent, "forget the brand")}


def test_a_withdrawal_can_name_the_leading_phrase_of_a_remembered_long_value() -> None:
    snapshot = _snapshot(
        _entry(
            "feature",
            "replacement mouthpiece for a water filter",
            text="actually I want a replacement mouthpiece for a water filter",
        )
    )
    intent = _intent(snapshot, "forget the replacement mouthpiece requirement")
    assert intent.hard == () and intent.soft == ()


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


def test_explicit_product_type_promotes_only_trusted_type_matches() -> None:
    def item(asin: str, sources: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(parent_asin=asin, sources=sources, rank=1)

    def record(*categories: str, title: str = "") -> SimpleNamespace:
        return SimpleNamespace(categories=categories, title=title)

    records = {
        "water-1": record("Sports & Outdoors", "Camping & Hiking", "Water Filters"),
        "water-2": record("Sports & Outdoors", "Camping & Hiking", "Water Filters"),
        "pad": record("Sports & Outdoors", "Camping & Hiking", "Sleeping Pads"),
    }
    candidates = (
        item("pad", ("history",)),
        item("water-1", ("catalog_search",)),
        item("unknown", ()),
        item("water-2", ("catalog_search",)),
    )
    promoted = promote_explicit_type_matches(
        "lightweight water filter", candidates, records
    )
    # Type matches first, UNKNOWN stays neutral, the trusted different type is removed.
    assert [item.parent_asin for item in promoted] == [
        "water-1",
        "water-2",
        "unknown",
    ]
    # A broad request has no category/term consensus and keeps the original order.
    broad = promote_explicit_type_matches("hiking gear for a weekend trip", candidates, records)
    assert [item.parent_asin for item in broad] == [
        "pad",
        "water-1",
        "unknown",
        "water-2",
    ]


def test_main_product_queryexcludes_trusted_replacement_part() -> None:
    def item(asin: str, sources: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(parent_asin=asin, sources=sources, rank=1)

    def record(title: str) -> SimpleNamespace:
        return SimpleNamespace(
            categories=("Sports & Outdoors", "Hydration & Filtration", "Water Filters"),
            title=title,
        )

    records = {
        "filter-main": record("LifeStraw Personal Water Filter"),
        "filter-part": record("LifeStraw Replacement Mouthpiece for Water Filter"),
        "filtered-bottle": record("Clear2GO Water Bottle with Filter"),
    }
    candidates = (
        item("filter-main", ("catalog_search",)),
        item("filtered-bottle", ("catalog_search",)),
        item("filter-part", ("catalog_search",)),
    )
    main = promote_explicit_type_matches("complete water filter", candidates, records)
    assert [entry.parent_asin for entry in main] == ["filter-main"]

    parts = promote_explicit_type_matches(
        "replacement mouthpiece for water filter", candidates, records
    )
    assert {entry.parent_asin for entry in parts} == {"filter-part"}

    compatible = {
        "filter-main": record("LifeStraw Water Filter"),
        "filter-compatible": record("Compatible with LifeStraw Water Filter"),
    }
    compatible_candidates = (
        item("filter-main", ("catalog_search",)),
        item("filter-compatible", ("catalog_search",)),
    )
    kept = promote_explicit_type_matches(
        "water filter", compatible_candidates, compatible
    )
    # "compatible with" alone is too ambiguous to declare the item a replacement part.
    assert {entry.parent_asin for entry in kept} == {"filter-main", "filter-compatible"}


def test_explicit_type_removes_trusted_unrelated_history_across_split_leaves() -> None:
    def item(asin: str, sources: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(parent_asin=asin, sources=sources, rank=1)

    def record(title: str, leaf: str) -> SimpleNamespace:
        return SimpleNamespace(
            categories=("Sports & Outdoors", "Camping & Hiking", leaf),
            title=title,
        )

    records = {
        "purifier": record("General Ecology Water Filter", "Water Purifiers"),
        "canteen": record("Collapsible Water Filter Canteen", "Sports Water Bottles"),
        "pad": record("Sleeping Pad for Camping", "Self-Inflating Sleeping Mats"),
        "lantern": record("LED Camping Lantern", "Electric Lanterns"),
    }
    candidates = (
        item("purifier", ("catalog_search",)),
        item("pad", ()),
        item("canteen", ("catalog_search",)),
        item("lantern", ()),
    )
    promoted = promote_explicit_type_matches(
        "I need a complete water filter", candidates, records
    )
    assert {entry.parent_asin for entry in promoted} == {"purifier", "canteen"}


def test_bottle_bite_valve_is_not_presented_as_a_water_filter_replacement() -> None:
    def item(asin: str) -> SimpleNamespace:
        return SimpleNamespace(parent_asin=asin, sources=("catalog_search",), rank=1)

    def record(title: str) -> SimpleNamespace:
        return SimpleNamespace(
            categories=("Sports & Outdoors", "Hydration & Filtration", "Water Filters"),
            title=title,
        )

    records = {
        "filter-part": record(
            "Survivor Filter Replacement Mouthpiece Fits Water Filter Straw"
        ),
        "bottle-part": record(
            "AIEVE Water Bottle Mouthpiece Replacement Compatible with Brita Water Filter Bottle"
        ),
    }
    candidates = (item("filter-part"), item("bottle-part"))
    promoted = promote_explicit_type_matches(
        "replacement mouthpiece for a water filter", candidates, records
    )
    assert [entry.parent_asin for entry in promoted] == ["filter-part"]
