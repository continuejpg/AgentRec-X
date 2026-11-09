"""Grounded product type/role assessment for final candidate eligibility.

This is deliberately a small, explainable layer over the trusted catalogue title and
category.  It does not use the benchmark Gold identity.  It answers two separate
questions:

* product **type** -- for example a complete water filter versus a filtered drinking
  bottle versus cookware;
* product **role** -- main product, replacement part/accessory, or unknown.

A candidate is MATCH only when trusted title/category evidence supports the current
explicit target type and role.  MISMATCH is used only for positive contradictory
evidence.  UNKNOWN means the catalogue does not provide enough evidence; it is never
silently promoted to MATCH.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    "ProductRole",
    "RoleAssessment",
    "RoleStatus",
    "TargetRole",
    "assess_candidate",
    "infer_target",
    "select_grounded_recommendations",
]


class RoleStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class ProductRole(str, Enum):
    MAIN = "main"
    ACCESSORY = "accessory"
    CONSUMABLE = "consumable"
    UNKNOWN = "unknown"


class TargetRole(str, Enum):
    MAIN = "main"
    ACCESSORY = "accessory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProductTarget:
    """The explicit type/role target inferred from the user's message."""

    type: str | None
    role: TargetRole
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleAssessment:
    """One candidate's explainable role/type assessment."""

    parent_asin: str
    status: RoleStatus
    product_type: str | None
    product_role: ProductRole
    reasons: tuple[str, ...] = ()
    title: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "status": self.status.value,
            "product_type": self.product_type,
            "product_role": self.product_role.value,
            "reasons": list(self.reasons),
            "title": self.title,
        }


@dataclass(frozen=True)
class RoleSelection:
    """Result of applying grounded role eligibility to final candidates."""

    target: ProductTarget
    candidates: tuple[Any, ...]
    assessments: tuple[RoleAssessment, ...]
    rejection_reasons: tuple[str, ...] = ()

    @property
    def matched_count(self) -> int:
        return sum(item.status is RoleStatus.MATCH for item in self.assessments)

    @property
    def mismatch_count(self) -> int:
        return sum(item.status is RoleStatus.MISMATCH for item in self.assessments)

    @property
    def unknown_count(self) -> int:
        return sum(item.status is RoleStatus.UNKNOWN for item in self.assessments)


_ACCESSORY_REQUEST = re.compile(
    r"\b(replacement|replace|spare|part|parts|cartridge|mouthpiece|accessory|accessories)\b",
    re.IGNORECASE,
)
_FILTER_DEVICE = re.compile(
    r"\b(water filter|water purifier|water filtration|filtration system|filter pump|"
    r"filter straw|filter housing|microfilter|water filter straw)\b",
    re.IGNORECASE,
)
_ACCESSORY_TITLE = re.compile(
    r"\b(replacement|spare|part|cartridge|mouthpiece|accessory|accessories|"
    r"bite valve|extra filters|filter set)\b",
    re.IGNORECASE,
)
_BOTTLE_TITLE = re.compile(r"\b(water bottle|filtered bottle|filter bottle|bottled water)\b", re.IGNORECASE)
_TREATMENT_TITLE = re.compile(r"\b(purification tablets?|water treatment|water tablets?)\b", re.IGNORECASE)
_COOKWARE_TITLE = re.compile(r"\b(cookware|cook set|mess kit|pots? and pans?|pot and pan)\b", re.IGNORECASE)
_STOVE_TITLE = re.compile(r"\b(stove|burner)\b", re.IGNORECASE)
_LANTERN_TITLE = re.compile(r"\b(lantern)\b", re.IGNORECASE)
_SLEEPING_TITLE = re.compile(r"\b(sleeping pad|sleeping mat|air mattress)\b", re.IGNORECASE)

_TARGET_WORDS = {
    "bottle": "filtered_bottle",
    "tablet": "water_treatment",
    "filter": "water_filter",
    "cookware": "cookware",
    "stove": "stove",
    "lantern": "lantern",
    "sleeping": "sleeping_pad",
}


def _text(record: Any) -> str:
    return " ".join(str(getattr(record, "title", "") or "").split())


def _categories(record: Any) -> tuple[str, ...]:
    return tuple(
        str(value) for value in (getattr(record, "categories", ()) or ()) if str(value).strip()
    )


def _category_text(record: Any) -> str:
    return " ".join(_categories(record)).casefold()


def infer_target(message: str) -> ProductTarget:
    """Infer a conservative explicit type/role target from the user's own wording."""
    text = " ".join(str(message or "").casefold().split())
    if not text:
        return ProductTarget(None, TargetRole.UNKNOWN)
    role = TargetRole.ACCESSORY if _ACCESSORY_REQUEST.search(text) else TargetRole.MAIN
    if "water bottle" in text or "bottle with filter" in text or "filtered bottle" in text:
        product_type = "filtered_bottle"
    elif any(term in text for term in ("purification tablet", "water tablet", "water treatment")):
        product_type = "water_treatment"
    elif "sleeping pad" in text or "sleeping mat" in text or "air mattress" in text:
        product_type = "sleeping_pad"
    elif "lantern" in text:
        product_type = "lantern"
    elif any(term in text for term in ("camp stove", "camping stove", "stove", "burner")):
        product_type = "stove"
    elif any(term in text for term in ("cookware", "cook set", "mess kit", "pots and pans", "pot and pan")):
        product_type = "cookware"
    elif any(
        term in text
        for term in (
            "complete water filter",
            "water filter",
            "water purifier",
            "water filtration",
            "filtration system",
            "filter pump",
            "filter straw",
            "replacement filter",
        )
    ):
        product_type = "water_filter"
    else:
        product_type = None
    terms = tuple(
        sorted(
            {
                word
                for word in re.findall(r"[a-z0-9]+", text)
                if word not in {"i", "a", "an", "the", "for", "with", "and", "or", "need", "want", "complete", "water"}
            }
        )
    )
    return ProductTarget(product_type, role, terms)


def _product_types(record: Any) -> frozenset[str]:
    title = _text(record)
    category = _category_text(record)
    combined = f"{title.casefold()} {category}"
    types: set[str] = set()
    if _BOTTLE_TITLE.search(title) or "water bottle" in category or "bottles" in category:
        types.add("filtered_bottle")
    if _TREATMENT_TITLE.search(title) or "water treaters" in category or "water treatment" in category:
        types.add("water_treatment")
    if (
        "water filters" in category
        or "water purifiers" in category
        or _FILTER_DEVICE.search(title)
    ):
        types.add("water_filter")
    if _COOKWARE_TITLE.search(title) or "cookware" in category or "open fire cookware" in category or "pots, pans" in category:
        types.add("cookware")
    if _STOVE_TITLE.search(title) or "stove" in category:
        types.add("stove")
    if _LANTERN_TITLE.search(title) or "lantern" in category:
        types.add("lantern")
    if _SLEEPING_TITLE.search(title) or "sleeping mat" in category:
        types.add("sleeping_pad")
    return frozenset(types)


def _product_role(record: Any, *, title: str, category: str) -> ProductRole:
    if _ACCESSORY_TITLE.search(title) or "accessories" in category:
        return ProductRole.ACCESSORY
    if "extra filters" in title.casefold() or "replacement" in title.casefold():
        return ProductRole.ACCESSORY
    return ProductRole.MAIN


def _compatible_accessory(target: ProductTarget, title: str) -> bool:
    lowered = title.casefold()
    if target.type == "filtered_bottle":
        return "bottle" in lowered and ("filter" in lowered or "replacement" in lowered)
    if target.type == "water_filter":
        # A generic bottle bite valve is not evidence of a filter device.
        if "water bottle" in lowered or "bite valve" in lowered:
            return False
        return bool(_FILTER_DEVICE.search(title)) or "filter" in lowered
    return True


def assess_candidate(
    parent_asin: str,
    record: Any,
    target: ProductTarget,
) -> RoleAssessment:
    """Return MATCH / MISMATCH / UNKNOWN with trusted title/category evidence."""
    title = _text(record)
    category = _category_text(record)
    types = _product_types(record) if record is not None else frozenset()
    product_role = _product_role(record, title=title, category=category) if record is not None else ProductRole.UNKNOWN
    if target.type is None and target.role is not TargetRole.ACCESSORY:
        return RoleAssessment(
            parent_asin,
            RoleStatus.UNKNOWN,
            None,
            product_role,
            ("no explicit product type in the user request",),
            title,
        )
    if target.type is None:
        return RoleAssessment(
            parent_asin,
            RoleStatus.UNKNOWN,
            None,
            product_role,
            ("accessory target without a supported product type",),
            title,
        )
    if not types:
        return RoleAssessment(
            parent_asin,
            RoleStatus.UNKNOWN,
            None,
            product_role,
            ("trusted title/category does not identify a supported product type",),
            title,
        )
    if target.role is TargetRole.ACCESSORY:
        if target.type not in types:
            if types:
                return RoleAssessment(
                    parent_asin,
                    RoleStatus.MISMATCH,
                    next(iter(types)) if len(types) == 1 else "|".join(sorted(types)),
                    product_role,
                    ("candidate is an accessory for a different product type",),
                    title,
                )
            return RoleAssessment(
                parent_asin,
                RoleStatus.UNKNOWN,
                None,
                product_role,
                (f"accessory compatibility for {target.type} is not proven by trusted evidence",),
                title,
            )
        if product_role is not ProductRole.ACCESSORY:
            return RoleAssessment(
                parent_asin,
                RoleStatus.MISMATCH,
                target.type,
                product_role,
                ("user requested an accessory, candidate is a main product",),
                title,
            )
        if not _compatible_accessory(target, title):
            return RoleAssessment(
                parent_asin,
                RoleStatus.MISMATCH,
                target.type,
                product_role,
                ("accessory title does not prove compatibility with the requested filter device",),
                title,
            )
        return RoleAssessment(
            parent_asin,
            RoleStatus.MATCH,
            target.type,
            product_role,
            ("accessory role and filter-device compatibility are grounded in the title",),
            title,
        )
    if target.type not in types:
        return RoleAssessment(
            parent_asin,
            RoleStatus.MISMATCH,
            next(iter(types)) if len(types) == 1 else "|".join(sorted(types)),
            product_role,
            ("trusted type differs from the explicit user target",),
            title,
        )
    if product_role in {ProductRole.ACCESSORY, ProductRole.CONSUMABLE}:
        return RoleAssessment(
            parent_asin,
            RoleStatus.MISMATCH,
            target.type,
            product_role,
            ("user requested a main product, candidate is an accessory/consumable",),
            title,
        )
    return RoleAssessment(
        parent_asin,
        RoleStatus.MATCH,
        target.type,
        product_role,
        ("trusted title/category type and main-product role match the request",),
        title,
    )


def select_grounded_recommendations(
    message: str,
    recommendations: Sequence[Any],
    records: Mapping[str, Any],
    *,
    target: ProductTarget | None = None,
) -> RoleSelection:
    """Filter/reorder final candidates without filling K from MISMATCH evidence.

    When ``target`` is supplied, the caller's task-scoped effective goal is used instead
    of the current follow-up text ("show more options").  MATCH candidates are kept first.
    UNKNOWN candidates are retained after MATCH candidates because UNKNOWN is not a
    violation; MISMATCH candidates are omitted when the request has an explicit target.  If
    no MATCH exists, UNKNOWN evidence is retained rather than fabricated as a match.  The
    base candidate order is preserved within each evidence class.
    """
    target = target or infer_target(message)
    candidates = tuple(recommendations)
    if target.type is None and target.role is not TargetRole.ACCESSORY:
        return RoleSelection(target, candidates, (), ())
    assessments = tuple(
        assess_candidate(
            str(getattr(item, "parent_asin", "")),
            records.get(str(getattr(item, "parent_asin", ""))),
            target,
        )
        for item in candidates
    )
    matched = [item for item, assessment in zip(candidates, assessments) if assessment.status is RoleStatus.MATCH]
    unknown = [item for item, assessment in zip(candidates, assessments) if assessment.status is RoleStatus.UNKNOWN]
    mismatched = [
        (item, assessment)
        for item, assessment in zip(candidates, assessments)
        if assessment.status is RoleStatus.MISMATCH
    ]
    # If there is no trusted MATCH, do not present UNKNOWN items as if they satisfied the
    # explicit target; this is the "candidate not covered" path, not a K-filling path.
    selected = [*matched, *unknown] if matched else []
    reasons = tuple(
        f"{assessment.parent_asin}: " + "; ".join(assessment.reasons)
        for _, assessment in mismatched
    )
    return RoleSelection(target, tuple(selected), assessments, reasons)
