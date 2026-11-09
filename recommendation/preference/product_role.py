"""Minimal Query-to-Product Type/Role contract for the Electronics shopping agent.

It is deliberately narrow and evidence-based:

* the user request explicitly names a main product type (or a type-specific accessory);
* a candidate is MATCH only when trusted catalogue title evidence supports the same
  product type and main/accessory role;
* a different main type is SUBSTITUTE evidence (MISMATCH for an explicit target);
* an accessory (stand/case/cable/...) is ACCESSORY (MISMATCH when the user asked for the
  main product, MATCH when the user explicitly asked for that accessory);
* no trusted evidence -> UNKNOWN, never promoted to MATCH;
* no explicit product type in the request -> NOT_REQUIRED (no fabricated constraint).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "CandidateRole",
    "ProductTarget",
    "RoleAssessment",
    "RoleStatus",
    "assess_candidate_role",
    "infer_product_target",
]

TYPE_PATTERNS: dict[str, str] = {
    "headphones": r"\b(headphones?|headset)\b",
    "earbuds": r"\b(earbuds?|earphones?)\b",
    "keyboard": r"\bkeyboards?\b",
    "mouse": r"\b(mouse|mice|trackball)\b",
    "speaker": r"\b(speakers?|soundbar)\b",
}
ACCESSORY_PATTERNS = {
    "accessory": r"\b(stand|stands|mount|mounts|case|cases|cover|covers|sleeve|sleeves|cable|cables|adapter|adapters|dock|docks|replacement|skin|skins|bag|bags|keycaps?|carrying case)\b",
    "complement": r"\b(mouse pad|wrist rest|cleaning kit|batteries|battery pack|charger)\b",
}
ACCESSORY_REQUEST = re.compile(
    r"\b(stand|mount|case|cover|sleeve|cable|adapter|dock|replacement|skin|bag|keycaps?|accessor(?:y|ies)|parts?)\b",
    re.IGNORECASE,
)
FOR_TYPE = re.compile(
    r"\b(?:for|compatible\s+with)\s+(?:the\s+|my\s+|a\s+)?"
    r"(headphones?|headset|earbuds?|earphones?|keyboards?|mice|mouse|trackball|speakers?|soundbar)\b",
    re.IGNORECASE,
)
TYPE_THEN_ACCESSORY = re.compile(
    r"\b(headphones?|headset|earbuds?|earphones?|keyboards?|mouse|mice|speakers?|soundbar)\b"
    r"[^.]{0,40}?\b(stand|stands|mount|mounts|case|cases|cover|covers|sleeve|sleeves|cable|cables|"
    r"adapter|adapters|dock|docks|replacement|skin|skins|bag|bags|keycaps?|mouse pad|wrist rest)\b",
    re.IGNORECASE,
)
WITH_ACCESSORY = re.compile(r"\bwith\b[^.]{0,30}\b(case|cover|stand|sleeve|bag|cable|adapter|dock)\b", re.IGNORECASE)


class RoleStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    NOT_REQUIRED = "NOT_REQUIRED"


class CandidateRole(str, Enum):
    MAIN = "main"
    ACCESSORY = "accessory"
    COMPLEMENT = "complement"
    SUBSTITUTE = "substitute"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProductTarget:
    product_type: str | None
    role: CandidateRole
    terms: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"product_type": self.product_type, "role": self.role.value, "terms": list(self.terms)}


@dataclass(frozen=True)
class RoleAssessment:
    status: RoleStatus
    expected_product_type: str | None
    expected_role: str
    candidate_product_type: str | None
    candidate_role: CandidateRole
    evidence: tuple[str, ...] = ()
    source: str = "catalog_meta:title"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role_status": self.status.value,
            "expected_product_type": self.expected_product_type,
            "expected_role": self.expected_role,
            "candidate_product_type": self.candidate_product_type,
            "candidate_role": self.candidate_role.value,
            "role_evidence": list(self.evidence),
            "role_source": self.source,
        }


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _type_in(text: str) -> str | None:
    lowered = text.casefold()
    for name, pattern in TYPE_PATTERNS.items():
        if re.search(pattern, lowered):
            return name
    return None


def infer_product_target(query: str) -> ProductTarget:
    """Infer the explicit product type/role from the agent-visible query only."""
    text = _text(query)
    if not text:
        return ProductTarget(None, CandidateRole.UNKNOWN)
    product_type = _type_in(text)
    if product_type is None:
        return ProductTarget(None, CandidateRole.UNKNOWN)
    role = CandidateRole.ACCESSORY if ACCESSORY_REQUEST.search(text) else CandidateRole.MAIN
    if role is CandidateRole.ACCESSORY:
        # "charging case" etc. usually describe a main product bundle; only a request for
        # a type-specific add-on (e.g. "keyboard stand") flips the target role.
        details = [match.group(0) for match in ACCESSORY_REQUEST.finditer(text)]
        if not details:
            role = CandidateRole.MAIN
    return ProductTarget(product_type, role, tuple(sorted(set(re.findall(r"[a-z0-9]+", text.casefold())))))


def _first_accessory_position(lowered: str) -> int | None:
    positions = []
    for pattern in ACCESSORY_PATTERNS.values():
        match = re.search(pattern, lowered)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def _first_type_position(lowered: str) -> int | None:
    positions = [match.start() for name, pattern in TYPE_PATTERNS.items()
                 for match in [re.search(pattern, lowered)] if match]
    return min(positions) if positions else None


def _candidate_role(title: str, product_type: str | None) -> tuple[CandidateRole, tuple[str, ...]]:
    lowered = title.casefold()
    with_accessory = WITH_ACCESSORY.search(lowered) is not None
    type_then = TYPE_THEN_ACCESSORY.search(lowered)
    for_type = FOR_TYPE.search(lowered)
    complement = re.search(ACCESSORY_PATTERNS["complement"], lowered)
    accessory = re.search(ACCESSORY_PATTERNS["accessory"], lowered)
    type_in_title = _type_in(title)
    accessory_pos = _first_accessory_position(lowered)
    type_pos = _first_type_position(lowered)

    # An accessory/complement head noun before the first main-product type wins
    # (e.g. "Monitor Stand Riser ... Keyboard storage" is not a keyboard).
    if accessory_pos is not None and (type_pos is None or accessory_pos < type_pos):
        role = CandidateRole.COMPLEMENT if complement and complement.start() == accessory_pos else CandidateRole.ACCESSORY
        return role, ("accessory/complement head noun precedes the main product type",)
    if with_accessory and type_in_title and for_type is None:
        return CandidateRole.MAIN, ("main product title with a bundled accessory",)
    if type_then or for_type is not None:
        return CandidateRole.ACCESSORY, ("title names a type-specific accessory",)
    if accessory and not type_in_title:
        return CandidateRole.ACCESSORY, ("accessory noun without a main-product type",)
    if product_type in TYPE_PATTERNS or type_in_title:
        return CandidateRole.MAIN, ("main product type in title",)
    if accessory:
        return CandidateRole.ACCESSORY, ("accessory noun in title",)
    return CandidateRole.UNKNOWN, ()


def assess_candidate_role(
    *,
    product_type: str | None,
    title: str,
    target: ProductTarget,
) -> RoleAssessment:
    """Return the candidate's Match/Mismatch/Unknown status for the query target."""
    if target.product_type is None:
        return RoleAssessment(
            RoleStatus.NOT_REQUIRED, None, target.role.value, product_type,
            CandidateRole.UNKNOWN, ("no explicit product type in the request",),
        )
    candidate_type = product_type or _type_in(title)
    candidate_role, evidence = _candidate_role(title, candidate_type)
    if candidate_role is CandidateRole.UNKNOWN:
        return RoleAssessment(
            RoleStatus.UNKNOWN, target.product_type, target.role.value, candidate_type,
            candidate_role, ("trusted title does not identify a supported type/role",),
        )
    if target.role is CandidateRole.ACCESSORY:
        if candidate_role not in {CandidateRole.ACCESSORY, CandidateRole.COMPLEMENT}:
            return RoleAssessment(
                RoleStatus.MISMATCH, target.product_type, target.role.value, candidate_type,
                candidate_role, ("user requested an accessory, candidate is a main product",),
            )
        if candidate_type is not None and candidate_type != target.product_type:
            return RoleAssessment(
                RoleStatus.MISMATCH, target.product_type, target.role.value, candidate_type,
                candidate_role, ("accessory is for a different product type",),
            )
        return RoleAssessment(
            RoleStatus.MATCH, target.product_type, target.role.value, candidate_type,
            candidate_role, ("accessory role and type match the request",),
        )
    # target is a main product
    if candidate_role in {CandidateRole.ACCESSORY, CandidateRole.COMPLEMENT}:
        return RoleAssessment(
            RoleStatus.MISMATCH, target.product_type, target.role.value, candidate_type,
            candidate_role, ("user requested a main product, candidate is an accessory/complement",),
        )
    if candidate_role is CandidateRole.SUBSTITUTE or (
        candidate_type is not None and candidate_type != target.product_type
    ):
        return RoleAssessment(
            RoleStatus.MISMATCH, target.product_type, target.role.value, candidate_type,
            CandidateRole.SUBSTITUTE, ("trusted product type differs from the explicit request",),
        )
    if candidate_type == target.product_type and candidate_role is CandidateRole.MAIN:
        return RoleAssessment(
            RoleStatus.MATCH, target.product_type, target.role.value, candidate_type,
            candidate_role, ("trusted main-product type matches the request",),
        )
    return RoleAssessment(
        RoleStatus.UNKNOWN, target.product_type, target.role.value, candidate_type,
        candidate_role, ("type/role evidence is inconclusive",),
    )
