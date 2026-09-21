"""Trusted evidence acquisition (Phase 4).

Why this exists
---------------
Phase 3.1 measured a capability ceiling rather than a reasoning one: some stated hard constraints
map to :attr:`~recommendation.control.grounded_reasoning.ConstraintKind.UNVERIFIABLE`, so
``check_constraint`` returns ``UNKNOWN`` *by construction* and no amount of re-reading the base
catalogue projection can change it.  The agent's only recourse was to read the same facts again.

This module adds a second, genuinely separate **trusted** source of product facts, and a store the
verdict authority consults.  The shape of the authority is the point:

    EvidenceSource.read(...)      trusted code reads a trusted artifact
            |
            v
    EvidenceItem                  attribute + value + provenance, never a verdict
            |
            v
    EvidenceStore                 an overlay keyed by (parent_asin, attribute)
            |
            v
    GroundedReasoner.check_constraint      the ONE verdict authority, unchanged
            |
            v
    CandidateEligibilityEvaluator          unchanged

What the model may and may not do
---------------------------------
The model selects **which grounded candidate needs more evidence**.  It cannot state what the
evidence says, cannot name an attribute the trusted source does not carry, and cannot produce a
verdict: an :class:`EvidenceItem` has no verdict field at all, and nothing in this module computes
one.  A model-authored claim has no path into the store.

Three-state honesty
-------------------
Acquisition has its own three states - ``FOUND`` / ``NOT_FOUND`` / ``UNSUPPORTED`` - and they
matter because "we looked and the source has nothing" must not read as "we did not look":

* ``FOUND`` - the trusted source states a value; it enters the store with provenance;
* ``NOT_FOUND`` - the source was consulted and carries nothing for that product;
* ``UNSUPPORTED`` - the source cannot answer that question at all (an attribute it does not
  model, or a product it does not cover).

Absence never becomes a verdict.  A candidate with no evidence stays ``UNKNOWN``, which keeps it
out of the verified feasible set without ever fabricating a violation.

Conflicting sources
-------------------
If two trusted sources disagree about a fact, both provenance records are kept and the fact is
treated as **undecided**: :meth:`EvidenceStore.value_for` returns ``None`` and the conflict is
reportable.  Silently preferring one source would be a fabrication of certainty.

A separate artifact, on purpose
-------------------------------
The first :class:`EvidenceSource` reads a **reproducible local artifact** that is deliberately not
the base catalogue projection.  Evidence therefore has to be acquired to be known, which is what
makes the ``UNKNOWN -> acquire -> SATISFIED/VIOLATED`` trajectory a real test rather than a
re-reading of facts the run already had.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

__all__ = [
    "EVIDENCE_ARTIFACT_VERSION",
    "EvidenceDocument",
    "EvidenceItem",
    "EvidenceSource",
    "EvidenceStore",
    "EvidenceStatus",
    "LocalEvidenceArtifact",
    "EVIDENCE_ATTRIBUTES",
]

#: Version of the evidence contract.  Bumped when the item or document shape changes, so a
#: stored artifact can be read against the semantics that produced it.
EVIDENCE_ARTIFACT_VERSION = 1

#: The attributes a trusted evidence source may state.  A closed vocabulary, so an evidence source
#: cannot introduce a dimension the reasoner would silently ignore, and a policy cannot ask about
#: one that does not exist.  ``material`` and ``feature`` are the two the base catalogue projection
#: does not decide; the rest are named so a future source can corroborate the catalogue and be
#: checked for conflicts.
EVIDENCE_ATTRIBUTES: tuple[str, ...] = (
    "material",
    "feature",
    "waterproof",
    "brand",
    "color",
    "category",
)


class EvidenceStatus(str, Enum):
    """How an acquisition attempt ended.  Never a verdict about a constraint."""

    #: The trusted source stated at least one fact for the requested product.
    FOUND = "found"
    #: The source was consulted and holds nothing for this product.
    NOT_FOUND = "not_found"
    #: The source cannot answer this request at all (unknown product, or not its subject area).
    UNSUPPORTED = "unsupported"


class EvidenceItem:
    """One fact stated by one trusted source, with provenance.

    It has no verdict field, and that absence is the design: evidence is an *input* to constraint
    evaluation, never a conclusion.  Only
    :class:`~recommendation.control.grounded_reasoning.GroundedReasoner` produces a verdict.
    """

    __slots__ = ("parent_asin", "attribute", "value", "source", "provenance")

    def __init__(
        self,
        *,
        parent_asin: str,
        attribute: str,
        value: str,
        source: str,
        provenance: str = "",
    ) -> None:
        if attribute not in EVIDENCE_ATTRIBUTES:
            raise ValueError(
                f"unknown evidence attribute {attribute!r}; known: {', '.join(EVIDENCE_ATTRIBUTES)}"
            )
        if not str(value).strip():
            # An empty statement is not evidence.  Refusing it here keeps "the source said
            # nothing" from being recorded as a value that could decide a constraint.
            raise ValueError("an evidence value must be non-empty")
        self.parent_asin = str(parent_asin)
        self.attribute = str(attribute)
        self.value = str(value).strip()
        #: Which trusted source stated it, e.g. ``"local_evidence_artifact"``.
        self.source = str(source)
        #: Where inside that source, for audit: a line number, a record key, a document id.
        self.provenance = str(provenance)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "parent_asin": self.parent_asin,
            "attribute": self.attribute,
            "value": self.value,
            "source": self.source,
            "provenance": self.provenance,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"EvidenceItem({self.parent_asin!r}, {self.attribute!r}, {self.value!r})"


class EvidenceDocument:
    """What one acquisition attempt returned for one product.

    ``status`` distinguishes "found something" from "looked and there was nothing" from "this
    source cannot answer that".  All three are legitimate outcomes and the policy is told which.
    """

    __slots__ = ("parent_asin", "status", "items", "source", "note")

    def __init__(
        self,
        *,
        parent_asin: str,
        status: EvidenceStatus,
        source: str,
        items: Sequence[EvidenceItem] = (),
        note: str | None = None,
    ) -> None:
        self.parent_asin = str(parent_asin)
        self.status = status
        self.source = str(source)
        self.items: tuple[EvidenceItem, ...] = tuple(items)
        self.note = note

    @property
    def attributes(self) -> tuple[str, ...]:
        """The attributes this document stated, in order."""
        return tuple(item.attribute for item in self.items)

    def as_dict(self) -> dict[str, Any]:
        """Return a compact view: counts and attribute names, not the values.

        Deliberately value-free.  A trajectory records *that* evidence was acquired and *which*
        attributes it covered; the values belong to the constraint report, where they are attached
        to a verdict a reader can act on.
        """
        return {
            "parent_asin": self.parent_asin,
            "status": self.status.value,
            "source": self.source,
            "attribute_count": len(self.items),
            "attributes": list(self.attributes),
            "note": self.note,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"EvidenceDocument({self.parent_asin!r}, {self.status.value}, {len(self.items)})"


@runtime_checkable
class EvidenceSource(Protocol):
    """A trusted source of product facts that is not the base catalogue projection.

    Implementations read a reproducible artifact.  They do **not** compute verdicts, do not touch
    the ledger, and do not accept free-form queries: a request is a product identity and the
    source answers with what it holds.
    """

    @property
    def name(self) -> str:
        """Stable source identity, recorded in provenance."""
        ...

    def read(self, parent_asin: str) -> EvidenceDocument:
        """Return what this source holds for ``parent_asin``.

        Must distinguish ``NOT_FOUND`` (consulted, nothing held) from ``UNSUPPORTED`` (cannot
        answer).  Must never invent a value.
        """
        ...


class EvidenceStore:
    """The overlay the verdict authority consults: ``(parent_asin, attribute) -> EvidenceItem``.

    Multiple sources may state the same fact.  Rather than overwriting, every statement is kept
    and the fact becomes **undecided** when two trusted sources disagree - a conservative failure
    that preserves both provenance records instead of silently choosing one.

    The store never computes a verdict.  It answers "what do we know", and
    :class:`~recommendation.control.grounded_reasoning.GroundedReasoner` decides what that means.
    """

    def __init__(self) -> None:
        #: ``(parent_asin, attribute) -> [EvidenceItem, ...]`` in the order absorbed.
        self._items: dict[tuple[str, str], list[EvidenceItem]] = {}
        self._documents: list[EvidenceDocument] = []

    def __len__(self) -> int:
        """How many distinct facts have been acquired."""
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and len(key) == 2 and key in self._items

    # -- ingest ------------------------------------------------------------ #

    def absorb(self, document: EvidenceDocument) -> tuple[str, ...]:
        """Record a document's facts and return the attributes that were **newly** acquired.

        The return value is how the policy learns that progress was made: an attribute already
        held is not new, so re-asking the same source changes nothing and the run can tell.
        """
        self._documents.append(document)
        acquired: list[str] = []
        for item in document.items:
            key = (item.parent_asin, item.attribute)
            existing = self._items.setdefault(key, [])
            if not existing:
                acquired.append(item.attribute)
            # Every statement is retained, including a repeat: a source that changes its answer is
            # a finding, and keeping both is what makes the conflict visible.
            existing.append(item)
        return tuple(acquired)

    @property
    def documents(self) -> tuple[EvidenceDocument, ...]:
        """Every acquisition attempt, in order, for the trajectory."""
        return tuple(self._documents)

    # -- query ------------------------------------------------------------- #

    def items_for(self, parent_asin: str, attribute: str) -> tuple[EvidenceItem, ...]:
        """Every statement about one attribute, in the order acquired."""
        return tuple(self._items.get((str(parent_asin), str(attribute)), ()))

    def attributes_for(self, parent_asin: str) -> tuple[str, ...]:
        """Which attributes are held for a product, in ``EVIDENCE_ATTRIBUTES`` order."""
        held = {attribute for (identity, attribute) in self._items if identity == str(parent_asin)}
        return tuple(attribute for attribute in EVIDENCE_ATTRIBUTES if attribute in held)

    def conflicts_for(self, parent_asin: str, attribute: str) -> tuple[EvidenceItem, ...]:
        """The statements involved when trusted sources disagree, or an empty tuple.

        Disagreement is decided on the **normalised** value, because ``"Leather"`` and
        ``" leather "`` are the same claim.  Values are compared case- and whitespace-insensitively
        and never interpreted semantically.
        """
        items = self.items_for(parent_asin, attribute)
        if len({_normalise(item.value) for item in items}) <= 1:
            return ()
        return items

    def value_for(self, parent_asin: str, attribute: str) -> str | None:
        """The agreed value for a fact, or ``None`` when nothing is held or sources disagree.

        ``None`` makes the fact undecided, which the reasoner reports as ``UNKNOWN``.  That is the
        conservative reading: a conflict means "we do not know", never "we pick the convenient one".
        """
        items = self.items_for(parent_asin, attribute)
        if not items:
            return None
        normalised = {_normalise(item.value) for item in items}
        if len(normalised) > 1:
            return None
        return items[0].value

    def as_dict(self) -> dict[str, Any]:
        """Return a payload-light, JSON-serialisable view for a trajectory record.

        Counts, attribute names and provenance identities - no fact values.  A trajectory shows
        that evidence exists and where it came from; the values appear in the constraint report,
        attached to the verdict they produced.
        """
        conflicts = [
            {
                "parent_asin": identity,
                "attribute": attribute,
                "values": sorted({item.value for item in self.items_for(identity, attribute)}),
                "sources": sorted(
                    {item.source for item in self.items_for(identity, attribute)}
                ),
            }
            for identity, attribute in sorted(self._items)
            if self.conflicts_for(identity, attribute)
        ]
        return {
            "version": EVIDENCE_ARTIFACT_VERSION,
            "fact_count": len(self._items),
            "products": sorted({identity for identity, _ in self._items}),
            "attributes": sorted({attribute for _, attribute in self._items}),
            "provenance": sorted(
                {item.source for items in self._items.values() for item in items}
            ),
            "conflicts": conflicts,
        }


def _normalise(value: str) -> str:
    """Normalise a stated value for conflict detection: case and surrounding whitespace only."""
    return " ".join(str(value).split()).casefold()


class LocalEvidenceArtifact:
    """A trusted evidence source backed by a reproducible local JSONL artifact.

    Each line is one product's evidence::

        {"parent_asin": "cand-red", "attributes": {"material": "leather"},
         "source_note": "vendor spec sheet, 2024-01"}

    The artifact is deliberately **not** the base catalogue projection.  That is what makes
    acquisition meaningful: a fact in this file is not known to the run until an evidence action
    reads it, so the ``UNKNOWN -> acquire -> SATISFIED`` trajectory proves the capability rather
    than re-reading what the run already had.

    The file is a repository **fixture** with an explicit provenance note per record, and it is
    validated on load: an unknown attribute, a blank value or a duplicate product line is an error
    rather than something to skip, because a silently skipped fact would look like absence.
    """

    def __init__(self, path: Path | str, *, name: str = "local_evidence_artifact") -> None:
        self._path = Path(path)
        self._name = str(name)
        self._by_product: dict[str, dict[str, str]] | None = None
        self._notes: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Stable source identity, recorded in every item's provenance."""
        return self._name

    @property
    def path(self) -> Path:
        """The artifact this source reads."""
        return self._path

    @property
    def attributes(self) -> tuple[str, ...]:
        """The attributes this artifact actually states, across all its records.

        Declared from the file rather than from the vocabulary, so a policy is offered only the
        questions this evidence source can really answer.  Reading it requires the file, so a
        missing artifact surfaces as an error here rather than as a silent empty menu.
        """
        declared: list[str] = []
        for attributes in self._load().values():
            for attribute in attributes:
                if attribute not in declared:
                    declared.append(str(attribute))
        return tuple(attribute for attribute in EVIDENCE_ATTRIBUTES if attribute in declared)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the artifact, so a result can be tied to the exact evidence revision."""
        return hashlib.sha256(self._path.read_bytes()).hexdigest()

    def _load(self) -> dict[str, dict[str, str]]:
        """Read and validate the artifact once."""
        if self._by_product is not None:
            return self._by_product
        records: dict[str, dict[str, str]] = {}
        if not self._path.exists():
            raise FileNotFoundError(f"evidence artifact not found: {self._path}")
        for line_number, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self._path.name} line {line_number} is not JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{self._path.name} line {line_number} is not an object")
            identity = str(record.get("parent_asin") or "").strip()
            if not identity:
                raise ValueError(f"{self._path.name} line {line_number} has no parent_asin")
            if identity in records:
                # A duplicate line would silently make one of the two authoritative.
                raise ValueError(f"{self._path.name} repeats parent_asin {identity!r}")
            attributes = record.get("attributes")
            if not isinstance(attributes, dict) or not attributes:
                raise ValueError(
                    f"{self._path.name} line {line_number} carries no attributes object"
                )
            cleaned: dict[str, str] = {}
            for attribute, value in attributes.items():
                if attribute not in EVIDENCE_ATTRIBUTES:
                    raise ValueError(
                        f"{self._path.name} line {line_number} states unknown attribute "
                        f"{attribute!r}; known: {', '.join(EVIDENCE_ATTRIBUTES)}"
                    )
                text = str(value).strip()
                if not text:
                    raise ValueError(
                        f"{self._path.name} line {line_number} states an empty {attribute!r}"
                    )
                cleaned[str(attribute)] = text
            records[identity] = cleaned
            self._notes[identity] = str(record.get("source_note") or "")
        self._by_product = records
        return records

    def covers(self, parent_asin: str) -> bool:
        """True when this artifact holds anything for the product."""
        return str(parent_asin) in self._load()

    def read(self, parent_asin: str) -> EvidenceDocument:
        """Return this artifact's facts for ``parent_asin``.

        A product the artifact does not cover is ``NOT_FOUND``, not ``UNSUPPORTED``: the source was
        consulted and holds nothing, which is exactly the honest outcome an unresolved constraint
        should keep reporting.
        """
        records = self._load()
        identity = str(parent_asin)
        attributes = records.get(identity)
        if not attributes:
            return EvidenceDocument(
                parent_asin=identity,
                status=EvidenceStatus.NOT_FOUND,
                source=self._name,
                note=f"no record for {identity} in {self._path.name}",
            )
        note = self._notes.get(identity) or ""
        items = tuple(
            EvidenceItem(
                parent_asin=identity,
                attribute=attribute,
                value=value,
                source=self._name,
                provenance=f"{self._path.name}#{identity}"
                + (f" ({note})" if note else ""),
            )
            for attribute, value in attributes.items()
        )
        return EvidenceDocument(
            parent_asin=identity,
            status=EvidenceStatus.FOUND,
            source=self._name,
            items=items,
            note=note or None,
        )
