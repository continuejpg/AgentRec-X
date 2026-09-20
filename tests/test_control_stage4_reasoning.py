"""Stage 4 tests: grounded reasoning over catalogue facts.

The principle under test is the stage's one-line rule:

> Reasoning may change a decision; reasoning may not change a fact.

Every test here checks that a conclusion traces back to a catalogue attribute, and that an
absent attribute yields ``UNKNOWN`` rather than a plausible value.  Families:

**A. Facts and deterministic parsing** - unit conversion is arithmetic, not inference, and an
unreadable string yields ``None`` rather than a guessed unit.

**B. Hard constraints are three-state** - satisfied / violated / **unknown**, where unknown is
neither a pass nor a failure.  This is what lets `CompletionGuard` refuse a run whose
constraints were never actually verified.

**C. Trade-offs and comparison exclude what the catalogue does not say** - a product without
the fact is never silently placed in an ordering.

**D. Compatibility is grounded or UNKNOWN** - this catalogue carries no interface or fitment
semantics, so such a requirement is ``UNKNOWN`` and the reasoner never manufactures a
compatible verdict for a check it did not perform.

**E. Bundles report their own gaps** - an aggregate is emitted only when every member carries
the fact; otherwise the blocking members are named.  No cart or transaction concept exists.

Everything is offline: the catalogue is synthetic and no model is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    UNKNOWN,
    CompatibilityRequirement,
    CompatibilityVerdict,
    ConstraintKind,
    ConstraintVerdict,
    GroundedReasoner,
    parse_dimensions_cm,
    parse_price,
    parse_weight_kg,
)

# --------------------------------------------------------------------------- #
# fixture catalogue
# --------------------------------------------------------------------------- #


class _Record:
    """A normalized-record stand-in with the attributes the reasoner reads."""

    def __init__(
        self,
        parent_asin: str,
        *,
        title: str | None = None,
        price_text: str | None = None,
        main_category: str | None = None,
        categories: tuple[str, ...] = (),
        features: tuple[str, ...] = (),
        average_rating: float | None = None,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.parent_asin = parent_asin
        self.title = title
        self.price_text = price_text
        self.main_category = main_category
        self.categories = categories
        self.features = features
        self.average_rating = average_rating
        self.rating_number = 10
        self.details = details


class _Catalog:
    def __init__(self, records: list[_Record]) -> None:
        self.records = {record.parent_asin: record for record in records}

    def __contains__(self, parent_asin: object) -> bool:
        return isinstance(parent_asin, str) and parent_asin in self.records


def build_reasoner() -> GroundedReasoner:
    """A catalogue with one fully specified, one incomplete and one sparse product."""
    return GroundedReasoner(
        _Catalog(
            [
                _Record(
                    "LIGHT",
                    title="Ultralight Tent",
                    price_text="120.0",
                    main_category="Camping",
                    categories=("Sports & Outdoors", "Camping", "Tents"),
                    features=("2-person", "waterproof rainfly"),
                    average_rating=4.6,
                    details=(
                        ("Item Weight", "1.2 Kilograms"),
                        ("Brand Name", "Outfit"),
                        ("Color", "Green"),
                        ("Material", "Nylon"),
                        ("Item Package Dimensions L x W x H", "40 x 20 x 20 cm"),
                    ),
                ),
                _Record(
                    "HEAVY",
                    title="Heavy Tent",
                    price_text="90.0",
                    main_category="Camping",
                    categories=("Sports & Outdoors", "Camping", "Tents"),
                    features=("4-person",),
                    average_rating=4.1,
                    details=(
                        ("Item Weight", "4.5 Kilograms"),
                        ("Brand Name", "Outfit"),
                        ("Color", "Blue"),
                        ("Material", "Cotton"),
                    ),
                ),
                _Record(
                    "NOSPEC",
                    title="Mystery Tent",
                    price_text="150.0",
                    main_category="Camping",
                    categories=("Sports & Outdoors", "Camping", "Tents"),
                    details=(("Brand Name", "Outfit"),),  # deliberately no weight
                ),
                _Record(
                    "NOPRICE",
                    title="Price Unknown Tent",
                    categories=("Sports & Outdoors", "Camping"),
                    details=(("Brand Name", "Other"), ("Item Weight", "2 Kilograms")),
                ),
            ]
        )
    )


# =========================================================================== #
# A. Facts and parsing
# =========================================================================== #


def test_weight_parsing_is_deterministic_conversion() -> None:
    """Parsing converts units arithmetically; it never guesses one."""
    assert parse_weight_kg("1.2 Kilograms") == pytest.approx(1.2)
    assert parse_weight_kg("2 pounds") == pytest.approx(0.90718474, rel=1e-6)
    assert parse_weight_kg("16 ounces") == pytest.approx(0.45359237, rel=1e-6)
    assert parse_weight_kg("500 grams") == pytest.approx(0.5)


def test_unreadable_values_yield_none_not_a_guess() -> None:
    """A missing unit, a bad number or a non-string is unreadable, not zero."""
    for bad in ("heavy", "", "1.2", "1.2 stone", None, 42):
        assert parse_weight_kg(bad) is None
    for bad in ("x by y", "40 x 20", None):
        assert parse_dimensions_cm(bad) is None
    for bad in ("cheap", "", "10-20", None):
        assert parse_price(bad) is None


def test_price_and_dimension_parsing() -> None:
    """Price tolerates a currency symbol; dimensions convert all three axes."""
    assert parse_price("12.99") == pytest.approx(12.99)
    assert parse_price("$8.50") == pytest.approx(8.50)
    dims = parse_dimensions_cm("4.49 x 4.49 x 1.5 inches")
    assert dims is not None
    assert dims[0] == pytest.approx(11.4046, rel=1e-4)
    assert parse_dimensions_cm("40 x 20 x 20 cm") == pytest.approx((40.0, 20.0, 20.0))


def test_absent_attribute_is_unknown_and_never_defaulted() -> None:
    """A product without a weight reports UNKNOWN, not zero and not an estimate."""
    reasoner = build_reasoner()
    profile = reasoner.facts("NOSPEC")
    assert profile is not None
    assert profile.weight_kg is None
    assert profile.weight_text() == UNKNOWN
    assert profile.price is not None  # it does have a price


def test_ungrounded_identity_has_no_fact_profile() -> None:
    """An identity outside the catalogue returns None, distinct from "no attributes"."""
    reasoner = build_reasoner()
    assert reasoner.facts("NOT-A-PRODUCT") is None
    # The sparse-but-real product is different: it exists and simply lacks facts.
    assert reasoner.facts("NOSPEC") is not None


def test_facts_are_verbatim_from_the_catalogue() -> None:
    """Reported facts equal the catalogue's own values, unmodified."""
    reasoner = build_reasoner()
    profile = reasoner.facts("LIGHT")
    assert profile is not None
    assert profile.brand == "Outfit"
    assert profile.color == "Green"
    assert profile.material == "Nylon"
    assert profile.title == "Ultralight Tent"
    assert profile.categories == ("Sports & Outdoors", "Camping", "Tents")


# =========================================================================== #
# B. Hard constraints
# =========================================================================== #


def test_a_satisfied_constraint_is_verified_not_assumed() -> None:
    """A constraint with the fact present gets a deterministic verdict."""
    reasoner = build_reasoner()
    report = reasoner.check_constraint(
        "LIGHT", kind=ConstraintKind.WEIGHT_MAX, expected="2 Kilograms"
    )
    assert report.verdict is ConstraintVerdict.SATISFIED
    assert report.is_violation is False
    assert report.observed == "1.200"


def test_a_violated_constraint_is_a_verified_violation() -> None:
    """A constraint the product fails is violated, with the observed value recorded."""
    reasoner = build_reasoner()
    report = reasoner.check_constraint(
        "HEAVY", kind=ConstraintKind.WEIGHT_MAX, expected="2 Kilograms"
    )
    assert report.verdict is ConstraintVerdict.VIOLATED
    assert report.is_violation is True
    assert report.observed == "4.500"


def test_an_uncheckable_constraint_is_unknown_not_a_pass() -> None:
    """The central rule: a missing fact is neither compliance nor violation."""
    reasoner = build_reasoner()
    report = reasoner.check_constraint(
        "NOSPEC", kind=ConstraintKind.WEIGHT_MAX, expected="2 Kilograms"
    )
    assert report.verdict is ConstraintVerdict.UNKNOWN
    assert report.is_violation is False
    assert report.observed is None
    assert reasoner.violations((report,)) == ()
    assert reasoner.unresolved((report,)) == (report,)


def test_price_constraints_are_checked_deterministically() -> None:
    """Budget constraints use the parsed numeric price, or UNKNOWN without one."""
    reasoner = build_reasoner()
    assert (
        reasoner.check_constraint(
            "HEAVY", kind=ConstraintKind.PRICE_MAX, expected="100"
        ).verdict
        is ConstraintVerdict.SATISFIED
    )
    assert (
        reasoner.check_constraint(
            "LIGHT", kind=ConstraintKind.PRICE_MAX, expected="100"
        ).verdict
        is ConstraintVerdict.VIOLATED
    )
    assert (
        reasoner.check_constraint(
            "NOPRICE", kind=ConstraintKind.PRICE_MAX, expected="100"
        ).verdict
        is ConstraintVerdict.UNKNOWN
    )


def test_category_and_brand_constraints_are_grounded_or_unknown() -> None:
    """Text constraints match the catalogue's own wording, or report UNKNOWN."""
    reasoner = build_reasoner()
    assert (
        reasoner.check_constraint(
            "LIGHT", kind=ConstraintKind.CATEGORY, expected="Tents"
        ).verdict
        is ConstraintVerdict.SATISFIED
    )
    assert (
        reasoner.check_constraint(
            "LIGHT", kind=ConstraintKind.BRAND, expected="Outfit"
        ).verdict
        is ConstraintVerdict.SATISFIED
    )
    assert (
        reasoner.check_constraint(
            "NOPRICE", kind=ConstraintKind.BRAND, expected="Outfit"
        ).verdict
        is ConstraintVerdict.VIOLATED
    )
    # A product with no colour yields UNKNOWN, never a violation.
    assert (
        reasoner.check_constraint(
            "NOSPEC", kind=ConstraintKind.COLOR, expected="Green"
        ).verdict
        is ConstraintVerdict.UNKNOWN
    )


def test_constraint_verdicts_are_deterministic_and_repeatable() -> None:
    """The same constraint and product produce the same verdict every time."""
    reasoner = build_reasoner()
    first = reasoner.check_constraint(
        "LIGHT", kind=ConstraintKind.WEIGHT_MAX, expected="2 Kilograms"
    )
    second = reasoner.check_constraint(
        "LIGHT", kind=ConstraintKind.WEIGHT_MAX, expected="2 Kilograms"
    )
    assert first.as_dict() == second.as_dict()


# =========================================================================== #
# C. Comparison and trade-off
# =========================================================================== #


def test_comparison_reports_absent_attributes_explicitly() -> None:
    """A comparison row whose values are all UNKNOWN is still emitted."""
    reasoner = build_reasoner()
    result = reasoner.compare(("LIGHT", "HEAVY"), attributes=("weight_text", "color"))
    rows = {row["attribute"]: row for row in result.rows}
    assert rows["weight_text"]["values"]["LIGHT"] == "1.200 kg"
    assert rows["weight_text"]["all_unknown"] is False

    # ``NOSPEC`` has no weight while ``LIGHT`` does: the attribute is partially known, so the
    # row is comparable and the absence is visible per product.
    partial = reasoner.compare(("LIGHT", "NOSPEC"), attributes=("weight_text",))
    assert partial.rows[0]["all_unknown"] is False
    assert partial.rows[0]["values"]["LIGHT"] == "1.200 kg"
    assert partial.rows[0]["values"]["NOSPEC"] == UNKNOWN
    assert partial.rows[0]["distinct_known_values"] == 1

    # Compare two products that both lack the fact: now the row is wholly unknown, and it is
    # still emitted rather than omitted - the absence of a fact is itself a finding.
    sparse = reasoner.compare(("NOSPEC", "NOPRICE"), attributes=("color",))
    assert sparse.rows[0]["all_unknown"] is True
    assert "color" not in sparse.comparable_attributes


def test_comparison_flags_an_unsupported_attribute_request() -> None:
    """Asking to compare an attribute the catalogue does not track is reported, not ignored."""
    reasoner = build_reasoner()
    result = reasoner.compare(("LIGHT", "HEAVY"), attributes=("weight_text", "voltage"))
    assert "voltage" in result.unknown_attributes


def test_trade_off_orders_only_products_that_carry_the_fact() -> None:
    """A product without the fact is excluded from the ordering, not guessed into it."""
    reasoner = build_reasoner()
    result = reasoner.trade_off(("HEAVY", "LIGHT", "NOSPEC"), priority="lighter")
    assert result["supported"] is True
    assert result["attribute"] == "weight_kg"
    assert result["order"] == ("LIGHT", "HEAVY")
    assert result["unknown"] == ("NOSPEC",)


def test_trade_off_reports_an_unsupported_priority() -> None:
    """A priority the catalogue cannot ground yields no ordering at all."""
    reasoner = build_reasoner()
    result = reasoner.trade_off(("LIGHT", "HEAVY"), priority="most stylish")
    assert result["supported"] is False
    assert result["order"] == ()
    assert result["attribute"] is None
    assert set(result["unknown"]) == {"LIGHT", "HEAVY"}


def test_trade_off_by_price_ascending_and_rating_descending() -> None:
    """Ordering direction depends on the attribute's own semantics."""
    reasoner = build_reasoner()
    cheaper = reasoner.trade_off(("LIGHT", "HEAVY"), priority="cheaper")
    assert cheaper["order"] == ("HEAVY", "LIGHT")
    rated = reasoner.trade_off(("HEAVY", "LIGHT"), priority="rating")
    assert rated["order"] == ("LIGHT", "HEAVY")


# =========================================================================== #
# D. Compatibility
# =========================================================================== #


def test_grounded_compatibility_is_compatible_when_every_member_matches() -> None:
    """A requirement the catalogue can check is answered from its own attributes."""
    reasoner = build_reasoner()
    verdict = reasoner.check_compatibility(
        ("LIGHT", "HEAVY"), requirement=CompatibilityRequirement("brand", "Outfit")
    )
    assert verdict is CompatibilityVerdict.COMPATIBLE


def test_grounded_compatibility_is_incompatible_on_a_mismatch() -> None:
    """One mismatching member is enough for an incompatible verdict."""
    reasoner = build_reasoner()
    verdict = reasoner.check_compatibility(
        ("LIGHT", "NOPRICE"), requirement=CompatibilityRequirement("brand", "Outfit")
    )
    assert verdict is CompatibilityVerdict.INCOMPATIBLE


def test_an_ungrounded_requirement_is_unknown_not_compatible() -> None:
    """This catalogue has no interface/fitment semantics, so such a check is UNKNOWN.

    This is the check that stops the reasoner from manufacturing a generic
    "these probably fit" verdict it cannot support.
    """
    reasoner = build_reasoner()
    for attribute in ("interface", "fitment", "voltage", "model", "compatible_with"):
        verdict = reasoner.check_compatibility(
            ("LIGHT", "HEAVY"),
            requirement=CompatibilityRequirement(attribute, "USB-C"),
        )
        assert verdict is CompatibilityVerdict.UNKNOWN, attribute


def test_compatibility_is_unknown_when_a_member_lacks_the_fact() -> None:
    """A member whose attribute is absent makes the whole check unresolved."""
    reasoner = build_reasoner()
    verdict = reasoner.check_compatibility(
        ("LIGHT", "NOSPEC"), requirement=CompatibilityRequirement("color", "Green")
    )
    assert verdict is CompatibilityVerdict.UNKNOWN


def test_compatibility_for_an_empty_set_is_unknown() -> None:
    """Nothing to check means nothing was verified."""
    reasoner = build_reasoner()
    verdict = reasoner.check_compatibility(
        (), requirement=CompatibilityRequirement("brand", "Outfit")
    )
    assert verdict is CompatibilityVerdict.UNKNOWN


# =========================================================================== #
# E. Bundles
# =========================================================================== #


def test_bundle_aggregates_only_fully_grounded_totals() -> None:
    """A total is emitted only when every member carries the fact."""
    reasoner = build_reasoner()
    bundle = reasoner.bundle(("LIGHT", "HEAVY"))
    assert bundle["total_price"]["status"] == "grounded"
    assert bundle["total_price"]["value"] == pytest.approx(210.0)
    assert bundle["total_weight_kg"]["status"] == "grounded"
    assert bundle["total_weight_kg"]["value"] == pytest.approx(5.7)


def test_bundle_names_the_members_that_block_an_aggregate() -> None:
    """A partial sum is never presented as the total; the blockers are named instead."""
    reasoner = build_reasoner()
    bundle = reasoner.bundle(("LIGHT", "HEAVY", "NOSPEC"))
    weight = bundle["total_weight_kg"]
    assert weight["status"] == "unknown"
    assert weight["value"] is None
    assert weight["blocking"] == ["NOSPEC"]
    assert "does not carry" in weight["reason"]


def test_bundle_reports_shared_categories_and_member_counts() -> None:
    """Compositional evidence is reported factually."""
    reasoner = build_reasoner()
    bundle = reasoner.bundle(("LIGHT", "HEAVY", "NOSPEC"))
    assert bundle["shared_categories"] == (
        "Camping",
        "Sports & Outdoors",
        "Tents",
    )
    assert bundle["member_count"] == 3
    assert bundle["requested_member_count"] == 3


def test_bundle_contains_no_transaction_concept() -> None:
    """A bundle is a recommendation set, not a cart or an order."""
    reasoner = build_reasoner()
    payload = str(reasoner.bundle(("LIGHT", "HEAVY")))
    for forbidden in ("cart", "checkout", "order", "payment", "purchase", "total_due"):
        assert forbidden not in payload.lower()


# =========================================================================== #
# F. Reasoner authority
# =========================================================================== #


def test_reasoner_is_read_only_and_cannot_widen_the_candidate_set() -> None:
    """The reasoner exposes no way to add, reorder or approve anything."""
    reasoner = build_reasoner()
    for forbidden in ("add", "append", "record", "commit", "approve", "rerank", "finish"):
        assert not hasattr(reasoner, forbidden)
    # Repeated reads do not change the catalogue.
    before = dict(reasoner.catalog.records)
    reasoner.compare(("LIGHT", "HEAVY"))
    reasoner.bundle(("LIGHT", "HEAVY"))
    assert dict(reasoner.catalog.records) == before
