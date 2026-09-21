"""Grounded facts, comparisons and constraint checking (AgentRec-X 2.0-alpha Stage 4).

> Reasoning may change a decision; reasoning may not change a fact.

Everything in this module is a **pure function of catalogue records**.  It reads attributes
that the normalized catalogue actually contains, and when an attribute is absent it returns
:data:`UNKNOWN` rather than a plausible value.  There is no model call, no heuristic guess and
no default: a missing fact stays missing, because the one thing worse than "I don't know" is a
confident invention.

What the real catalogue supports (measured on 20 000 records of the shipped artifact)
----------------------------------------------------------------------------------------
======================  =======  ==================================================
``Item Weight``          96%      deterministic aggregate checks (portability)
``Item Package Dims``    67%      deterministic aggregate checks
``price_text``           77%      deterministic budget checks
``Color``                74%      matching against an explicit colour preference
``Brand Name``           69%      matching against an explicit brand preference
``Material``             64%      comparison only
categories / features    96%+     category constraints and comparison
======================  =======  ==================================================

Compatibility is the interesting exception: this catalogue carries **no interface, fitment or
model-compatibility semantics**.  `CHECK_COMPATIBILITY` therefore grounds what it can - shared
category, identical part number, verified brand match - and returns ``UNKNOWN`` for a
requirement it cannot ground.  It never manufactures a generic "these probably fit" verdict,
because that would be a fabricated fact.

Units
-----
Weight and dimension values arrive as loose human strings (`"0.68 Kilograms"`,
`"4.49 x 4.49 x 1.5 inches"`).  Parsing them is deterministic unit conversion, not reasoning:
:average: each parser returns ``None`` when it cannot read the value, and `None` propagates as
``UNKNOWN``.  No value is ever inferred from a different unit or from a similar product.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

__all__ = [
    "GROUNDED_FACTS_VERSION",
    "ComparisonResult",
    "CompatibilityRequirement",
    "CompatibilityVerdict",
    "ConstraintKind",
    "ConstraintReport",
    "ConstraintVerdict",
    "GroundedFacts",
    "GroundedReasoner",
    "UNKNOWN",
    "attribute_value",
    "parse_dimensions_cm",
    "parse_price",
    "parse_weight_kg",
]

#: The single sentinel for "the catalogue does not say".  A distinct object rather than
#: ``None`` so that "absent" cannot be confused with a legitimately empty string value.
UNKNOWN: str = "UNKNOWN"

#: Version of the grounded-reasoning contract.
GROUNDED_FACTS_VERSION = 1

# Unit conversion factors to a canonical unit.  Deterministic arithmetic only.
_WEIGHT_TO_KG: dict[str, float] = {
    "kilogram": 1.0,
    "kilograms": 1.0,
    "kg": 1.0,
    "gram": 0.001,
    "grams": 0.001,
    "g": 0.001,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "ounce": 0.028349523125,
    "ounces": 0.028349523125,
    "oz": 0.028349523125,
    "milligram": 1e-6,
    "milligrams": 1e-6,
}

_LENGTH_TO_CM: dict[str, float] = {
    "centimeter": 1.0,
    "centimeters": 1.0,
    "cm": 1.0,
    "millimeter": 0.1,
    "millimeters": 0.1,
    "mm": 0.1,
    "meter": 100.0,
    "meters": 100.0,
    "m": 100.0,
    "inch": 2.54,
    "inches": 2.54,
    "in": 2.54,
    "foot": 30.48,
    "feet": 30.48,
    "ft": 30.48,
}

#: Detail keys that carry a weight, in preference order.
_WEIGHT_KEYS: tuple[str, ...] = ("Item Weight", "Package Weight")
#: Detail keys that carry dimensions, in preference order.
_DIMENSION_KEYS: tuple[str, ...] = (
    "Item Package Dimensions L x W x H",
    "Item Dimensions  LxWxH",
    "Product Dimensions",
)
_BRAND_KEYS: tuple[str, ...] = ("Brand Name", "Brand", "Manufacturer")


# --------------------------------------------------------------------------- #
# Deterministic parsing (unit conversion, never inference)
# --------------------------------------------------------------------------- #


def parse_weight_kg(text: Any) -> float | None:
    """Parse a human weight string into kilograms, or ``None`` when unreadable.

    Reads the first number and the unit that follows it.  An unknown unit, a missing unit or
    a non-numeric value returns ``None`` - the caller reports ``UNKNOWN`` rather than assuming
    a unit.
    """
    if not isinstance(text, str):
        return None
    tokens = text.replace(",", "").split()
    for index, token in enumerate(tokens):
        try:
            number = float(token)
        except ValueError:
            continue
        unit = tokens[index + 1].strip(".,").lower() if index + 1 < len(tokens) else ""
        factor = _WEIGHT_TO_KG.get(unit)
        if factor is None:
            return None
        return number * factor
    return None


def parse_dimensions_cm(text: Any) -> tuple[float, float, float] | None:
    """Parse ``"L x W x H <unit>"`` into centimetres, or ``None`` when unreadable.

    All three components must share one readable unit; a partially parsed triple is treated
    as unreadable rather than completed with a guess.
    """
    if not isinstance(text, str):
        return None
    cleaned = text.replace(",", " ").replace("x", " ").replace("X", " ").replace("×", " ")
    tokens = cleaned.split()
    numbers: list[float] = []
    unit = ""
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            candidate = token.strip(".,").lower()
            if candidate in _LENGTH_TO_CM:
                unit = candidate
    if len(numbers) < 3 or not unit:
        return None
    factor = _LENGTH_TO_CM[unit]
    return (numbers[0] * factor, numbers[1] * factor, numbers[2] * factor)


def parse_price(text: Any) -> float | None:
    """Parse a price string into a plain number, or ``None`` when unreadable.

    The catalogue's ``price_text`` is a bare numeric string; a currency symbol, a range or
    any non-numeric content is treated as unreadable rather than guessed.
    """
    if text is None:
        return None
    raw = str(text).strip().replace(",", "")
    for symbol in ("$", "£", "€", "¥"):
        raw = raw.replace(symbol, "")
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        return None


def attribute_value(record: Any, key: str) -> str | None:
    """Return one ``details`` attribute verbatim, or ``None`` when absent.

    Matching is case-insensitive on the key because the source uses inconsistent capitalisation
    (``Brand Name`` vs ``Brand``); the *value* is returned exactly as stored.
    """
    details = getattr(record, "details", ()) or ()
    lowered = key.casefold()
    for name, value in details:
        if str(name).casefold() == lowered:
            text = str(value).strip()
            return text or None
    return None


def first_attribute(record: Any, keys: Sequence[str]) -> str | None:
    """Return the first present attribute among ``keys``."""
    for key in keys:
        value = attribute_value(record, key)
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


class GroundedFacts:
    """The factual profile of one catalogue record, with explicit absences.

    Every accessor that can be absent returns ``None`` (or :data:`UNKNOWN` for the string
    form).  A caller can therefore distinguish "the catalogue says X" from "the catalogue does
    not say", which is the distinction the whole stage rests on.
    """

    __slots__ = ("parent_asin", "record")

    def __init__(self, parent_asin: str, record: Any) -> None:
        self.parent_asin = parent_asin
        self.record = record

    # -- individual facts -------------------------------------------------- #

    @property
    def title(self) -> str | None:
        """Product title, verbatim."""
        value = getattr(self.record, "title", None)
        return value if isinstance(value, str) and value.strip() else None

    @property
    def brand(self) -> str | None:
        """Brand, from ``Brand Name`` / ``Brand`` / ``Manufacturer``."""
        return first_attribute(self.record, _BRAND_KEYS)

    @property
    def color(self) -> str | None:
        """Colour, verbatim."""
        return attribute_value(self.record, "Color")

    @property
    def material(self) -> str | None:
        """Material, verbatim."""
        return attribute_value(self.record, "Material")

    @property
    def categories(self) -> tuple[str, ...]:
        """Category path, verbatim."""
        return tuple(str(c) for c in (getattr(self.record, "categories", ()) or ()))

    @property
    def main_category(self) -> str | None:
        """Top-level category, verbatim."""
        value = getattr(self.record, "main_category", None)
        return value if isinstance(value, str) and value.strip() else None

    @property
    def features(self) -> tuple[str, ...]:
        """Feature bullets, verbatim."""
        return tuple(str(f) for f in (getattr(self.record, "features", ()) or ()))

    @property
    def price(self) -> float | None:
        """Price as a number, or ``None`` when absent or unreadable."""
        return parse_price(getattr(self.record, "price_text", None))

    @property
    def weight_kg(self) -> float | None:
        """Weight in kilograms, or ``None`` when absent or unreadable."""
        return parse_weight_kg(first_attribute(self.record, _WEIGHT_KEYS))

    @property
    def dimensions_cm(self) -> tuple[float, float, float] | None:
        """``(L, W, H)`` in centimetres, or ``None``."""
        return parse_dimensions_cm(first_attribute(self.record, _DIMENSION_KEYS))

    @property
    def average_rating(self) -> float | None:
        """Average rating, or ``None``."""
        value = getattr(self.record, "average_rating", None)
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def rating_number(self) -> int | None:
        """Number of ratings, or ``None``."""
        value = getattr(self.record, "rating_number", None)
        return int(value) if isinstance(value, int) else None

    # -- friendliness ------------------------------------------------------ #

    def weight_text(self) -> str:
        """Weight as a display string, or ``UNKNOWN``."""
        weight = self.weight_kg
        return UNKNOWN if weight is None else f"{weight:.3f} kg"

    def price_text(self) -> str:
        """Price as a display string, or ``UNKNOWN``."""
        price = self.price
        return UNKNOWN if price is None else f"{price:.2f}"

    def summary(self) -> dict[str, Any]:
        """Return the fact profile as a JSON-serialisable mapping.

        Absent facts are present as ``None``/``UNKNOWN`` rather than omitted, so a consumer
        cannot mistake a missing key for a false value.
        """
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
            "material": self.material,
            "main_category": self.main_category,
            "categories": list(self.categories),
            "price": self.price,
            "weight_kg": self.weight_kg,
            "dimensions_cm": list(self.dimensions_cm) if self.dimensions_cm else None,
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
        }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


class ComparisonResult:
    """A side-by-side comparison over grounded attributes.

    ``rows`` holds one entry per compared attribute.  A row whose values are all ``UNKNOWN``
    is still emitted: the *absence* of a shared fact is a finding, and hiding it would let a
    reader assume the attribute matched.
    """

    __slots__ = ("parent_asins", "rows", "unknown_attributes")

    def __init__(
        self,
        *,
        parent_asins: tuple[str, ...],
        rows: tuple[dict[str, Any], ...],
        unknown_attributes: tuple[str, ...],
    ) -> None:
        self.parent_asins = parent_asins
        self.rows = rows
        self.unknown_attributes = unknown_attributes

    @property
    def comparable_attributes(self) -> tuple[str, ...]:
        """Attributes where at least one product carried a value."""
        return tuple(
            row["attribute"] for row in self.rows if not row["all_unknown"]
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "parent_asins": list(self.parent_asins),
            "attributes": list(self.rows),
            "unknown_attributes": list(self.unknown_attributes),
        }


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #


class ConstraintKind(str, Enum):
    """The constraint kinds the catalogue can check deterministically.

    :attr:`UNVERIFIABLE` is the explicit "no deterministic check exists for this dimension"
    member.  It exists because a hard constraint the user really stated (``waterproof``,
    ``leather``, an interface or fitment requirement) must not be silently dropped just
    because no catalogue attribute can decide it: dropping it would let a candidate be
    presented as compliant with a constraint that was never tested.  Every check against it
    is ``UNKNOWN``, which keeps such a candidate out of the verified feasible set while
    making the reason for that visible.
    """

    PRICE_MAX = "price_max"
    PRICE_MIN = "price_min"
    CATEGORY = "category"
    BRAND = "brand"
    COLOR = "color"
    #: Weight ceiling, used for portability constraints.
    WEIGHT_MAX = "weight_max"
    #: Product material, e.g. ``leather``.  The base catalogue projection does not decide this,
    #: so it is resolved from **acquired trusted evidence** when evidence exists and is
    #: ``UNKNOWN`` otherwise.  Phase 4 added it precisely because a stated material constraint
    #: used to be permanently unanswerable.
    MATERIAL = "material"
    #: A stated product feature, e.g. ``waterproof``.  Same rule as :attr:`MATERIAL`: evidence
    #: decides it, absence of evidence leaves it ``UNKNOWN``.
    FEATURE = "feature"
    #: A stated constraint the trusted sources cannot decide at all - no catalogue attribute and
    #: no evidence attribute models it.  Never satisfiable and never a violation: it resolves to
    #: ``UNKNOWN`` by construction.  Preserved for genuinely unsupported constraints, which must
    #: stay visible rather than being dropped or guessed.
    UNVERIFIABLE = "unverifiable"


class ConstraintVerdict(str, Enum):
    """Three-state verdict, mirroring the accepted M10A evidence semantics.

    ``UNKNOWN`` is deliberately not a pass and not a failure: a constraint whose required
    fact is absent must not silently pass (that would fabricate compliance) and must not
    automatically fail (that would fabricate a violation).
    """

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class ConstraintReport:
    """The deterministic verdict for one constraint against one candidate."""

    __slots__ = ("parent_asin", "kind", "expected", "verdict", "observed")

    def __init__(
        self,
        *,
        parent_asin: str,
        kind: ConstraintKind,
        expected: str,
        verdict: ConstraintVerdict,
        observed: str | None,
    ) -> None:
        self.parent_asin = parent_asin
        self.kind = kind
        self.expected = expected
        self.verdict = verdict
        self.observed = observed

    @property
    def is_violation(self) -> bool:
        """True only for a *verified* violation, never for an unknown."""
        return self.verdict is ConstraintVerdict.VIOLATED

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "parent_asin": self.parent_asin,
            "kind": self.kind.value,
            "expected": self.expected,
            "verdict": self.verdict.value,
            "observed": self.observed,
        }


#: Which acquired-evidence attribute decides which constraint dimension.  A closed map, so a new
#: constraint kind cannot silently become "evidence-backed" without an attribute behind it.
_EVIDENCE_KINDS: dict[ConstraintKind, tuple[str, ...]] = {
    ConstraintKind.MATERIAL: ("material",),
    # ``feature`` reads both the generic feature statement and a stated waterproof rating: they
    # are two facets of the same question, and a source may carry either.
    ConstraintKind.FEATURE: ("feature", "waterproof"),
}


def _evidence_matches(stated: str, expected: str) -> bool:
    """True when a stated evidence value satisfies an expected constraint value.

    Deterministic and deliberately narrow: normalised substring membership in either direction, so
    ``expected="leather"`` is satisfied by a stated ``"Full-grain leather upper"`` and
    ``expected="waterproof"`` by ``"waterproof membrane"``.  No stemming, no synonyms, no
    inference - a near-miss is a violation of the *check*, not an invitation to guess.
    """
    needle = " ".join(str(expected).split()).casefold()
    haystack = " ".join(str(stated).split()).casefold()
    if not needle:
        return False
    return needle in haystack or haystack in needle


class CompatibilityVerdict(str, Enum):
    """The verdict a compatibility check can reach from grounded facts only."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class CompatibilityRequirement:
    """A requirement to test products against.

    ``attribute`` names a catalogue attribute; ``value`` is what the user (or a stored
    preference) requires.  Nothing here interprets the requirement semantically - the
    reasoner only checks whether the catalogue grounds it.
    """

    __slots__ = ("attribute", "value")

    def __init__(self, attribute: str, value: str) -> None:
        self.attribute = attribute
        self.value = value

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable view."""
        return {"attribute": self.attribute, "value": self.value}


@runtime_checkable
class CatalogLike(Protocol):
    """The trusted catalogue: membership plus per-identity records."""

    def __contains__(self, parent_asin: object) -> bool:
        """True when the catalogue holds a record for this identity."""
        ...

    @property
    def records(self) -> dict[str, Any]:
        """Identity -> normalized record."""
        ...


# --------------------------------------------------------------------------- #
# The reasoner
# --------------------------------------------------------------------------- #


class GroundedReasoner:
    """Read facts, compare products and check constraints - all from the catalogue.

    The reasoner is **read-only**.  It cannot add a candidate, cannot reorder the ledger and
    cannot approve a completion; it only produces evidence that a policy and the
    ``CompletionGuard`` can act on.
    """

    def __init__(self, catalog: CatalogLike, *, evidence: Any = None) -> None:
        if not isinstance(catalog, CatalogLike):
            raise TypeError("catalog must expose '__contains__' and 'records'")
        self._catalog = catalog
        self._cache: dict[str, GroundedFacts] = {}
        #: Phase 4: an acquired-evidence overlay, or ``None`` for a deployment with no evidence
        #: source.  It is consulted *before* the catalogue for the evidence-decidable dimensions
        #: and is never consulted for a verdict - see ``check_constraint``.  Absent means exactly
        #: the pre-Phase-4 behaviour, which is why every existing test still passes unchanged.
        self._evidence = evidence

    @property
    def evidence(self) -> Any:
        """The acquired-evidence overlay in use, or ``None``."""
        return self._evidence

    # -- metadata ---------------------------------------------------------- #

    @property
    def catalog(self) -> CatalogLike:
        """The trusted catalogue in use."""
        return self._catalog

    @property
    def version(self) -> int:
        """Grounded-reasoning contract version."""
        return GROUNDED_FACTS_VERSION

    # -- facts ------------------------------------------------------------- #

    def facts(self, parent_asin: str) -> GroundedFacts | None:
        """Return the fact profile for ``parent_asin``, or ``None`` when not in the catalogue.

        ``None`` means the identity is not grounded.  A caller must not substitute an empty
        profile: "not in the catalogue" and "in the catalogue with no attributes" are
        different facts.
        """
        if not isinstance(parent_asin, str) or parent_asin not in self._catalog:
            return None
        cached = self._cache.get(parent_asin)
        if cached is not None:
            return cached
        profile = GroundedFacts(parent_asin, self._catalog.records[parent_asin])
        self._cache[parent_asin] = profile
        return profile

    def facts_many(self, parent_asins: Iterable[str]) -> tuple[GroundedFacts, ...]:
        """Return fact profiles for the identities that are grounded, in order.

        Ungrounded identities are simply absent from the result; grounding is the caller's
        separate, authoritative check and is not repeated here.
        """
        profiles: list[GroundedFacts] = []
        for parent_asin in parent_asins:
            profile = self.facts(parent_asin)
            if profile is not None:
                profiles.append(profile)
        return tuple(profiles)

    # -- comparison -------------------------------------------------------- #

    _COMPARISON_ATTRIBUTES: tuple[str, ...] = (
        "title",
        "brand",
        "color",
        "material",
        "main_category",
        "price_text",
        "weight_text",
        "average_rating",
    )

    def compare(
        self,
        parent_asins: Sequence[str],
        *,
        attributes: Sequence[str] = (),
    ) -> ComparisonResult:
        """Compare grounded products attribute by attribute.

        ``attributes`` selects a subset by name; an unknown name is reported as an unknown
        attribute rather than silently ignored, because a caller asking for something the
        catalogue does not track needs to know that.
        """
        profiles = self.facts_many(parent_asins)
        selected = tuple(attributes) if attributes else self._COMPARISON_ATTRIBUTES
        known = set(self._COMPARISON_ATTRIBUTES)

        rows: list[dict[str, Any]] = []
        unknown_attributes: list[str] = []
        for attribute in selected:
            if attribute not in known:
                unknown_attributes.append(attribute)
                continue
            values = {
                profile.parent_asin: self._read(profile, attribute)
                for profile in profiles
            }
            all_unknown = all(value is UNKNOWN for value in values.values())
            rows.append(
                {
                    "attribute": attribute,
                    "values": values,
                    "all_unknown": all_unknown,
                    "distinct_known_values": len(
                        {v for v in values.values() if v is not UNKNOWN}
                    ),
                }
            )
        return ComparisonResult(
            parent_asins=tuple(profile.parent_asin for profile in profiles),
            rows=tuple(rows),
            unknown_attributes=tuple(unknown_attributes),
        )

    @staticmethod
    def _read(profile: GroundedFacts, attribute: str) -> str:
        """Read one comparison attribute as a display string, or ``UNKNOWN``."""
        if attribute == "price_text":
            return profile.price_text()
        if attribute == "weight_text":
            return profile.weight_text()
        if attribute == "average_rating":
            rating = profile.average_rating
            return UNKNOWN if rating is None else f"{rating:.2f}"
        value = getattr(profile, attribute, None)
        if value is None:
            return UNKNOWN
        if isinstance(value, (tuple, list)):
            return " | ".join(str(v) for v in value) if value else UNKNOWN
        return str(value)

    # -- trade-off --------------------------------------------------------- #

    #: Named priorities and the attribute each one reads.  A priority the catalogue cannot
    #: ground yields an ``UNKNOWN`` ordering rather than an arbitrary one.
    _PRIORITY_ATTRIBUTES: dict[str, str] = {
        "lighter": "weight_kg",
        "lightest": "weight_kg",
        "weight": "weight_kg",
        "portability": "weight_kg",
        "portable": "weight_kg",
        "cheaper": "price",
        "cheapest": "price",
        "price": "price",
        "budget": "price",
        "rating": "average_rating",
        "rated": "average_rating",
    }

    def priority_attribute(self, priority: str) -> str | None:
        """Map a stated priority onto a catalogue attribute, or ``None`` when unsupported."""
        return self._PRIORITY_ATTRIBUTES.get(str(priority).strip().casefold())

    def trade_off(
        self,
        parent_asins: Sequence[str],
        *,
        priority: str,
    ) -> dict[str, Any]:
        """Order grounded products by a stated priority, reporting every absence.

        Returns a mapping with the resolved attribute, the grounded observations, an ordering
        of the products that *have* the fact, and the products excluded because the catalogue
        does not say.  Products without the fact are never placed in the ordering: putting
        them somewhere would assert a value the catalogue does not have.
        """
        attribute = self.priority_attribute(priority)
        profiles = self.facts_many(parent_asins)
        if attribute is None:
            return {
                "priority": priority,
                "attribute": None,
                "supported": False,
                "reason": "the catalogue does not carry a fact for this priority",
                "observations": {},
                "order": (),
                "unknown": tuple(p.parent_asin for p in profiles),
            }

        observations: dict[str, float] = {}
        unknown: list[str] = []
        for profile in profiles:
            value = getattr(profile, attribute, None)
            if value is None:
                unknown.append(profile.parent_asin)
            else:
                observations[profile.parent_asin] = float(value)

        # Ascending for cost/weight (less is better), descending for rating.
        reverse = attribute == "average_rating"
        order = tuple(
            parent_asin
            for parent_asin, _ in sorted(
                observations.items(), key=lambda pair: ((-pair[1]) if reverse else pair[1], pair[0])
            )
        )
        return {
            "priority": priority,
            "attribute": attribute,
            "supported": True,
            "reason": None,
            "observations": observations,
            "order": order,
            "unknown": tuple(unknown),
        }

    # -- constraints ------------------------------------------------------- #

    def check_constraint(
        self,
        parent_asin: str,
        *,
        kind: ConstraintKind,
        expected: str,
    ) -> ConstraintReport:
        """Check one hard constraint against one grounded product.

        A constraint whose required fact is absent is ``UNKNOWN`` - neither satisfied nor
        violated.  That three-state result is what lets the completion guard refuse to certify
        a run whose constraints were never actually verified.
        """
        if kind in _EVIDENCE_KINDS:
            # These dimensions are decided by acquired trusted evidence, never by the base
            # catalogue projection: the whole point of Phase 4 is that reading the same facts
            # again cannot answer them.  No evidence means UNKNOWN - never a guess, and never a
            # violation inferred from silence.
            report = self._evidence_constraint(parent_asin, kind=kind, expected=expected)
            if report is not None:
                return report
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=ConstraintVerdict.UNKNOWN,
                observed=None,
            )

        if kind is ConstraintKind.UNVERIFIABLE:
            # Not a missing *value* but a missing *dimension*: neither the catalogue nor any
            # configured evidence source models it, so no amount of looking will resolve it.
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=ConstraintVerdict.UNKNOWN,
                observed=None,
            )
        profile = self.facts(parent_asin)
        if profile is None:
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=ConstraintVerdict.UNKNOWN,
                observed=None,
            )

        if kind in (ConstraintKind.PRICE_MAX, ConstraintKind.PRICE_MIN):
            observed = profile.price
            limit = parse_price(expected)
            if observed is None or limit is None:
                return ConstraintReport(
                    parent_asin=parent_asin,
                    kind=kind,
                    expected=expected,
                    verdict=ConstraintVerdict.UNKNOWN,
                    observed=None if observed is None else f"{observed:.2f}",
                )
            satisfied = observed <= limit if kind is ConstraintKind.PRICE_MAX else observed >= limit
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=(
                    ConstraintVerdict.SATISFIED if satisfied else ConstraintVerdict.VIOLATED
                ),
                observed=f"{observed:.2f}",
            )

        if kind is ConstraintKind.WEIGHT_MAX:
            observed = profile.weight_kg
            limit = parse_weight_kg(expected)
            if observed is None or limit is None:
                return ConstraintReport(
                    parent_asin=parent_asin,
                    kind=kind,
                    expected=expected,
                    verdict=ConstraintVerdict.UNKNOWN,
                    observed=None if observed is None else f"{observed:.3f}",
                )
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=(
                    ConstraintVerdict.SATISFIED
                    if observed <= limit
                    else ConstraintVerdict.VIOLATED
                ),
                observed=f"{observed:.3f}",
            )

        # Text attributes: category / brand / colour, matched case-insensitively against the
        # catalogue's own wording.  A product with the attribute absent is UNKNOWN; the match
        # itself is deterministic substring/subset membership, not fuzzy inference.
        if kind is ConstraintKind.CATEGORY:
            haystack = list(profile.categories)
            if profile.main_category:
                haystack.append(profile.main_category)
        elif kind is ConstraintKind.BRAND:
            haystack = [profile.brand] if profile.brand else []
        else:  # COLOR
            haystack = [profile.color] if profile.color else []

        if not haystack:
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=ConstraintVerdict.UNKNOWN,
                observed=None,
            )
        needle = str(expected).strip().casefold()
        matched = any(needle in str(value).casefold() for value in haystack)
        return ConstraintReport(
            parent_asin=parent_asin,
            kind=kind,
            expected=expected,
            verdict=(
                ConstraintVerdict.SATISFIED if matched else ConstraintVerdict.VIOLATED
            ),
            observed=" | ".join(str(value) for value in haystack),
        )

    def _evidence_constraint(
        self, parent_asin: str, *, kind: ConstraintKind, expected: str
    ) -> ConstraintReport | None:
        """Derive a verdict from acquired evidence, or ``None`` when evidence cannot decide it.

        The *only* place an evidence-backed verdict is produced, and it is still this class -
        ``EvidenceSource`` and the policy never compute one.

        Three outcomes, in the order they are decided:

        * **conflicting trusted sources** - the fact is undecided, so ``UNKNOWN``.  Both
          provenance records are kept; picking one would be fabricated certainty.
        * **evidence present** - deterministic, case-insensitive comparison against the stated
          value, with the observed value kept for audit.  ``SATISFIED`` or ``VIOLATED``.
        * **no evidence** - ``None``, so the caller reports ``UNKNOWN``.  Absence is never a
          verdict.
        """
        if self._evidence is None:
            return None
        attributes = _EVIDENCE_KINDS[kind]
        # A conflict on *any* facet the dimension reads makes the dimension undecided.
        conflicts = tuple(
            item
            for attribute in attributes
            for item in self._evidence.conflicts_for(parent_asin, attribute)
        )
        if conflicts:
            return ConstraintReport(
                parent_asin=parent_asin,
                kind=kind,
                expected=expected,
                verdict=ConstraintVerdict.UNKNOWN,
                observed=" | ".join(sorted({item.value for item in conflicts})),
            )
        stated = [
            value
            for attribute in attributes
            if (value := self._evidence.value_for(parent_asin, attribute)) is not None
        ]
        if not stated:
            return None
        # A dimension may read more than one facet (``feature`` reads both the generic feature
        # statement and a stated waterproof rating).  The constraint is satisfied when any facet
        # supports it, because a source stating "waterproof membrane" and another stating a
        # generic feature set are describing the same product, not contradicting each other.
        combined = " | ".join(stated)
        satisfied = any(_evidence_matches(value, expected) for value in stated)
        return ConstraintReport(
            parent_asin=parent_asin,
            kind=kind,
            expected=expected,
            verdict=(ConstraintVerdict.SATISFIED if satisfied else ConstraintVerdict.VIOLATED),
            observed=combined,
        )

    def check_constraints(
        self,
        parent_asins: Sequence[str],
        *,
        constraints: Sequence[tuple[ConstraintKind, str]],
    ) -> tuple[ConstraintReport, ...]:
        """Check several constraints against several products, in a stable order."""
        return tuple(
            self.check_constraint(parent_asin, kind=kind, expected=expected)
            for parent_asin in parent_asins
            for kind, expected in constraints
        )

    def violations(
        self, reports: Sequence[ConstraintReport]
    ) -> tuple[ConstraintReport, ...]:
        """Return only the verified violations - never the unknowns."""
        return tuple(report for report in reports if report.is_violation)

    def unresolved(
        self, reports: Sequence[ConstraintReport]
    ) -> tuple[ConstraintReport, ...]:
        """Return the constraints the catalogue could not resolve."""
        return tuple(
            report for report in reports if report.verdict is ConstraintVerdict.UNKNOWN
        )

    # -- compatibility ----------------------------------------------------- #

    def check_compatibility(
        self,
        parent_asins: Sequence[str],
        *,
        requirement: CompatibilityRequirement,
    ) -> CompatibilityVerdict:
        """Return a three-state compatibility verdict from grounded facts only.

        The catalogue carries **no interface, fitment or model-compatibility semantics**, so a
        requirement naming such an attribute is ``UNKNOWN`` no matter how many products are
        supplied.  What can be grounded is checked deterministically:

        * a ``brand`` requirement is met when every product's brand matches;
        * a ``category`` requirement is met when every product shares a category;
        * a ``color`` requirement is met when every product's colour matches.

        A requirement whose attribute is not declared here is ``UNKNOWN``.  The method never
        returns ``COMPATIBLE`` for a requirement it did not actually check.
        """
        profiles = self.facts_many(parent_asins)
        if not profiles:
            return CompatibilityVerdict.UNKNOWN

        attribute = requirement.attribute.strip().casefold()
        kind_by_attribute = {
            "brand": ConstraintKind.BRAND,
            "category": ConstraintKind.CATEGORY,
            "categories": ConstraintKind.CATEGORY,
            "color": ConstraintKind.COLOR,
            "colour": ConstraintKind.COLOR,
        }
        kind = kind_by_attribute.get(attribute)
        if kind is None:
            # Interfaces, fitment, model/version, voltage, capacity: not in this catalogue.
            return CompatibilityVerdict.UNKNOWN

        verdicts = [
            self.check_constraint(
                profile.parent_asin, kind=kind, expected=requirement.value
            ).verdict
            for profile in profiles
        ]
        if any(verdict is ConstraintVerdict.VIOLATED for verdict in verdicts):
            return CompatibilityVerdict.INCOMPATIBLE
        if all(verdict is ConstraintVerdict.SATISFIED for verdict in verdicts):
            return CompatibilityVerdict.COMPATIBLE
        return CompatibilityVerdict.UNKNOWN

    # -- bundles ----------------------------------------------------------- #

    def bundle(self, parent_asins: Sequence[str]) -> dict[str, Any]:
        """Report what a set of grounded products supports as a bundle.

        A bundle is not merely a list of individually good items, so this reports the
        individual facts, the aggregate facts that are **fully** grounded, and which products
        block each aggregate.  An aggregate is emitted only when every member carries the
        needed fact; otherwise the blocking products are named and the aggregate is
        ``UNKNOWN``.  Summing a partial set and calling it the total would be a fabricated
        fact.

        No cart, order or transaction concept exists here or anywhere in the result: a bundle
        is a *recommendation set* under aggregate constraints.
        """
        profiles = self.facts_many(parent_asins)
        members = [
            {
                "parent_asin": profile.parent_asin,
                "title": profile.title,
                "price": profile.price,
                "weight_kg": profile.weight_kg,
                "main_category": profile.main_category,
            }
            for profile in profiles
        ]

        prices = {p.parent_asin: p.price for p in profiles}
        weights = {p.parent_asin: p.weight_kg for p in profiles}

        def aggregate(values: dict[str, Any]) -> dict[str, Any]:
            blocking = sorted(k for k, v in values.items() if v is None)
            if blocking:
                return {
                    "value": None,
                    "status": "unknown",
                    "blocking": blocking,
                    "reason": "the catalogue does not carry this fact for every member",
                }
            return {
                "value": round(sum(values.values()), 4),
                "status": "grounded",
                "blocking": [],
                "reason": None,
            }

        categories = [set(profile.categories) for profile in profiles]
        shared = set.intersection(*categories) if categories else set()
        return {
            "members": members,
            "member_count": len(members),
            "total_price": aggregate(prices),
            "total_weight_kg": aggregate(weights),
            "shared_categories": tuple(sorted(shared)),
            "grounded_member_count": len(profiles),
            "requested_member_count": len(tuple(parent_asins)),
        }
