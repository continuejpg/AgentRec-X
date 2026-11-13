"""Minimal evidence-sufficiency and corrective-query logic.

This module is deliberately small and deterministic.  It does not retrieve, rank or
decide: it only classifies a Gate failure into an actionable evidence gap and proposes
one bounded corrective search query.  The existing Executor, CandidateLedger, Scorer,
Eligibility and Decision Gate remain authoritative.

The design follows the CRAG/Self-RAG *pattern* (grade the retrieved evidence, then take a
bounded corrective action) but does not copy code from those projects.  See
``docs/CORRECTIVE_RAG.md`` for the open-source decision record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Facts the fixed catalogue/search source cannot provide.  A query rewrite cannot make
# a historical price live or invent stock/warranty evidence.
UNVERIFIABLE_FACTS = frozenset({
    "price", "stock", "availability", "warranty", "shipping", "live_price",
})

RETRIEVAL_GAP = "RETRIEVAL_GAP"
EVIDENCE_GAP = "EVIDENCE_GAP"
REQUIREMENT_AMBIGUITY = "REQUIREMENT_AMBIGUITY"
NO_GAP = "NO_GAP"


@dataclass(frozen=True)
class EvidenceCheck:
    kind: str
    recoverable: bool
    reason: str
    missing_facts: tuple[str, ...] = ()
    suggested_query: str | None = None
    metadata_mode: str = "original"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "recoverable": self.recoverable,
            "reason": self.reason,
            "missing_facts": list(self.missing_facts),
            "suggested_query": self.suggested_query,
            "metadata_mode": self.metadata_mode,
        }


def _missing_facts(hard_requirements: Sequence[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in hard_requirements:
        if isinstance(item, dict):
            value = item.get("value") or item.get("attribute")
        else:
            value = getattr(item, "value", None) or getattr(item, "attribute", None)
        text = str(value or "").strip().lower()
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _normalise(query: str) -> str:
    return " ".join(str(query or "").lower().split())


def build_corrective_query(
    *,
    original_query: str,
    missing_facts: Sequence[str],
    product_type: str | None,
    search_history: Iterable[str] = (),
) -> str | None:
    """Build a focused query that adds the missing facts without dropping constraints.

    The user/task query is not modified.  This string is used only for one additional
    retrieval.  A duplicate of any already executed query is rejected.
    """
    missing = [str(item).strip() for item in missing_facts if str(item).strip()]
    if not missing:
        return None
    base = " ".join(
        part for part in (str(product_type or "").strip(), " ".join(missing)) if part
    ).strip()
    candidates = [base]
    original = " ".join(str(original_query or "").split())
    if original:
        candidates.append(" ".join([original, *missing]))
        candidates.append(" ".join([original, str(product_type or "").strip(), *missing]))
    history = {_normalise(item) for item in search_history if str(item).strip()}
    for candidate in candidates:
        candidate = " ".join(candidate.split())[:240].strip()
        if candidate and _normalise(candidate) not in history:
            return candidate
    return None


def classify_evidence_gap(
    *,
    gate_action: str | None,
    hard_requirements: Sequence[Any],
    query: str,
    product_type: str | None = None,
    search_history: Iterable[str] = (),
    metadata_mode: str = "original",
) -> EvidenceCheck:
    """Classify a failed Gate decision into a recovery action.

    Only a failed Gate should reach this function.  A normal successful recommendation
    returns ``NO_GAP`` and never causes an extra search.
    """
    action = str(gate_action or "").upper()
    missing = _missing_facts(hard_requirements)
    if action == "NEED_USER_CLARIFICATION":
        return EvidenceCheck(
            kind=REQUIREMENT_AMBIGUITY,
            recoverable=False,
            reason="the Gate requires user clarification; query rewriting must not replace ASK",
            missing_facts=missing,
            metadata_mode=metadata_mode,
        )
    if action not in {"NEED_MORE_CANDIDATES", "NEED_MORE_PRODUCT_EVIDENCE"}:
        return EvidenceCheck(
            kind=NO_GAP, recoverable=False, reason="Gate did not report a recoverable gap",
            missing_facts=missing, metadata_mode=metadata_mode,
        )
    if missing and all(value in UNVERIFIABLE_FACTS for value in missing):
        return EvidenceCheck(
            kind=EVIDENCE_GAP,
            recoverable=False,
            reason=(
                "the missing facts are not present in the catalogue/search source; "
                "another search cannot create live price or stock evidence"
            ),
            missing_facts=missing,
            metadata_mode=metadata_mode,
        )
    suggested = build_corrective_query(
        original_query=query, missing_facts=missing, product_type=product_type,
        search_history=search_history,
    )
    if suggested is None:
        return EvidenceCheck(
            kind=EVIDENCE_GAP,
            recoverable=False,
            reason="no non-duplicate corrective query can be formed from the missing facts",
            missing_facts=missing,
            metadata_mode=metadata_mode,
        )
    # NEED_MORE_PRODUCT_EVIDENCE often means the current candidate pool has no verified
    # hard fact.  A focused query may find a different product with direct evidence; it is
    # therefore a bounded retrieval gap, not permission to treat UNKNOWN as verified.
    return EvidenceCheck(
        kind=RETRIEVAL_GAP,
        recoverable=True,
        reason=(
            "candidate pool has no verified hard-constraint candidate; "
            "one bounded focused retrieval is justified"
            if action == "NEED_MORE_PRODUCT_EVIDENCE"
            else "the Gate requested a wider candidate search"
        ),
        missing_facts=missing,
        suggested_query=suggested,
        metadata_mode=metadata_mode,
    )
