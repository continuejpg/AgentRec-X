"""Product knowledge retrieval for products that were already recommended (V2 E-02B).

What this layer is, and what it is not
--------------------------------------
It answers one question, for **one already-selected product**: *what does the trusted catalogue
actually say about this product, and which of those facts speak to what the shopper asked for?*
It cannot answer "what should be recommended": it holds no model, no candidate source and no way to
reach a product that is not in the list it was handed.  Its input is the run's final presented
candidates - after fusion, after memory-aware ranking, after the presentation limit - so retrieval
can neither add, drop nor reorder a recommendation.

Where the facts come from
-------------------------
The accepted Milestone 8 catalogue layer, unchanged: records are looked up **by identity**
(``parent_asin``) in the same :class:`~recommendation.catalog.MetadataIndex` the enricher uses, and
the fragments come from the accepted candidate-local retriever
(:func:`~recommendation.rag.retrieval.retrieve_evidence`).  A fragment is copied verbatim from
normalized metadata, keeps the field it came from, and keeps the catalogue's own provenance label.
Nothing here reads the user's history, the memory store or another product's record; nothing is
paraphrased, summarised or generated; a product with no record, or with a record but no retrievable
text, is reported as such instead of being filled in.

The knowledge unit, and why it has a stable id
----------------------------------------------
One unit is one retrievable fact: a title, a store, a category, a feature bullet, a description
line, or one source attribute from the attribute bag.  Its ``evidence_id`` is derived from the
identity and the **field path** (``"<parent_asin>:details.Color"``), so an explanation that cites it
can always be traced back to one product and one real field, and the same fact keeps the same id
across turns.  When a value was shortened to fit the bounded context the unit says so rather than
pretending the catalogue text ended there.

Pairing with preference evidence
--------------------------------
These units are *product knowledge*: "the catalogue says the colour is black".  They are not
preference evidence: "the colour black matches the shopper's stated preference" is the accepted
Milestone 10A verdict, computed elsewhere over the same catalogue field.  The two are reported in
separate fields and are only ever linked by comparing the field path and value they agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from recommendation.catalog.metadata import MissingMetadata
from recommendation.rag.retrieval import MAX_EVIDENCE_PER_CANDIDATE, MAX_EVIDENCE_PER_FIELD
from recommendation.rag.retrieval import retrieve_evidence

__all__ = [
    "MAX_TOTAL_UNITS",
    "MAX_UNIT_CHARS",
    "MAX_UNITS_PER_PRODUCT",
    "KnowledgePack",
    "KnowledgeUnit",
    "knowledge_query",
    "ProductKnowledge",
    "retrieve_product_knowledge",
]

#: How many facts one product may contribute.  Equal to the accepted retriever's own bound, so
#: this layer cannot widen the evidence window the catalogue layer already decided.
MAX_UNITS_PER_PRODUCT = MAX_EVIDENCE_PER_CANDIDATE

#: A unit's verbatim value is shortened to this many characters.  Truncation is recorded on the
#: unit, never disguised as the catalogue's own text ending.
MAX_UNIT_CHARS = 240

#: Hard bound on the whole pack, so a wide ``k`` cannot grow the model context without limit.
MAX_TOTAL_UNITS = 24

#: The source label every unit carries: the fact came from the trusted catalogue record.
KNOWLEDGE_SOURCE = "catalog_metadata"

#: Retrieval outcomes, explicit and never silent.
KNOWLEDGE_STATUSES = ("evidence", "no_metadata", "no_evidence")


@dataclass(frozen=True)
class KnowledgeUnit:
    """One retrievable, attributable catalogue fact about one product."""

    evidence_id: str
    parent_asin: str
    item_id: int
    #: The normalized field the fragment came from (``title``, ``store``, ``details``, ...).
    field: str
    #: The field path a citation can point at, e.g. ``title`` or ``details.Color``.
    field_path: str
    #: Verbatim catalogue text (possibly shortened; see :attr:`truncated`).
    value: str
    source: str
    #: The catalogue's own provenance label for this fragment.
    provenance: str
    #: Lexical relevance of the fragment to the current request, within this product only.
    retrieval_score: float
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "parent_asin": self.parent_asin,
            "item_id": self.item_id,
            "field_path": self.field_path,
            "value": self.value,
            "source": self.source,
            "provenance": self.provenance,
            "retrieval_score": self.retrieval_score,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ProductKnowledge:
    """One product's evidence pack, or the explicit reason it has none."""

    parent_asin: str
    item_id: int
    position: int
    metadata_status: str
    status: str
    units: tuple[KnowledgeUnit, ...] = ()
    fallback_reason: str | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.units)

    def unit(self, evidence_id: str) -> KnowledgeUnit | None:
        for unit in self.units:
            if unit.evidence_id == evidence_id:
                return unit
        return None

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(unit.evidence_id for unit in self.units)


@dataclass(frozen=True)
class KnowledgePack:
    """Every final recommendation's evidence pack, in the presented order."""

    query: str
    products: tuple[ProductKnowledge, ...] = ()
    dropped_units: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def for_identity(self, parent_asin: str) -> ProductKnowledge | None:
        for product in self.products:
            if product.parent_asin == str(parent_asin):
                return product
        return None

    @property
    def products_with_evidence(self) -> int:
        return sum(1 for product in self.products if product.has_evidence)

    @property
    def unit_count(self) -> int:
        return sum(len(product.units) for product in self.products)

    @property
    def presented(self) -> tuple[str, ...]:
        return tuple(product.parent_asin for product in self.products)


def _shorten(value: str, limit: int) -> tuple[str, bool]:
    """Shorten one verbatim value, reporting that it happened."""
    text = str(value)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _field_path(field_name: str, detail_key: str | None) -> str:
    """The citable path of a fragment: ``details.Color`` for an attribute, else the field."""
    if field_name == "details" and detail_key:
        return f"details.{detail_key}"
    return field_name


def knowledge_query(query: str, related_terms: Sequence[str] = ()) -> str:
    """The text a product's own facts are ranked against.

    It is the shopper's current request plus the **values** of the preferences the ranking used.
    Adding them is safe here in a way it is not for candidate retrieval: the product set is already
    decided, so this can only change *which facts of those products* are surfaced - it cannot add,
    drop or reorder a recommendation, and it never reaches a candidate source.  Values only: no
    store id, no ``source_text``, no turn id.
    """
    parts = [str(query)]
    seen = {str(query).strip().lower()}
    for term in related_terms:
        text = str(term).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            parts.append(text)
    return " ".join(part for part in parts if part)


def retrieve_product_knowledge(
    *,
    items: Sequence[Any],
    metadata: Any,
    query: str,
    related_terms: Sequence[str] = (),
    max_units_per_product: int = MAX_UNITS_PER_PRODUCT,
    max_unit_chars: int = MAX_UNIT_CHARS,
    max_total_units: int = MAX_TOTAL_UNITS,
) -> KnowledgePack:
    """Retrieve catalogue evidence for exactly the products in ``items``.

    ``items`` are the run's **final presented** candidates, in the order the response shows them;
    each must carry ``parent_asin`` and ``item_id``.  Only those identities are looked up - the
    catalogue index is never searched, so a product outside the list is unreachable by
    construction - and each returned unit is traceable to one real field of one real record.

    ``query`` is the shopper's own current request and ``related_terms`` are the preference values
    the ranking used (see :func:`knowledge_query`); together they rank *within* a product's own
    facts.  A product whose record has nothing matching still gets its documented deterministic
    fallback from the accepted retriever, and a product with no record at all is reported as
    ``no_metadata`` rather than being given invented facts.
    """
    ordered = list(items)
    query_used = knowledge_query(query, related_terms)
    identities = [str(item.parent_asin) for item in ordered]
    records = metadata.lookup_many(identities) if identities else ()
    diagnostics: dict[str, object] = {}
    evidence_per_product = (
        retrieve_evidence(
            query_used,
            list(records),
            max_per_candidate=max_units_per_product,
            max_per_field=MAX_EVIDENCE_PER_FIELD,
            diagnostics=diagnostics,
        )
        if records
        else ()
    )
    fallback_used = bool(diagnostics.get("fallback_used"))

    products: list[ProductKnowledge] = []
    budget = max(int(max_total_units), 0)
    dropped = 0
    for position, (item, record, units) in enumerate(
        zip(ordered, records, evidence_per_product)
    ):
        parent_asin = str(item.parent_asin)
        item_id = int(getattr(item, "item_id", 0) or 0)
        missing = isinstance(record, MissingMetadata)
        built: list[KnowledgeUnit] = []
        seen: set[str] = set()
        for evidence in units:
            if budget - len(built) <= 0:
                dropped += 1
                continue
            field_path = _field_path(str(evidence.field), evidence.detail_key)
            evidence_id = f"{parent_asin}:{field_path}"
            if evidence_id in seen:
                # The same fact reached retrieval twice (for example a repeated attribute): one
                # unit, one citation, so an explanation can never cite a duplicate.
                continue
            seen.add(evidence_id)
            value, truncated = _shorten(evidence.text, max_unit_chars)
            built.append(
                KnowledgeUnit(
                    evidence_id=evidence_id,
                    parent_asin=parent_asin,
                    item_id=item_id,
                    field=str(evidence.field),
                    field_path=field_path,
                    value=value,
                    source=KNOWLEDGE_SOURCE,
                    provenance=str(evidence.provenance),
                    retrieval_score=float(evidence.retrieval_score),
                    truncated=truncated,
                )
            )
        if built:
            status = "evidence"
        elif missing:
            status = "no_metadata"
        else:
            status = "no_evidence"
        # The accepted catalogue layer's own rule, mirrored rather than reinvented: a missing
        # record, a record with no searchable text at all, and a scope where nothing matched the
        # request (the retriever's documented fallback) are three different facts.  A product that
        # simply did not match while another one did has no fallback reason to report.
        if missing:
            fallback_reason = "no_metadata"
        elif not getattr(record, "has_searchable_text", True):
            fallback_reason = "no_searchable_text"
        elif fallback_used:
            fallback_reason = "no_lexical_match"
        else:
            fallback_reason = None
        products.append(
            ProductKnowledge(
                parent_asin=parent_asin,
                item_id=item_id,
                position=position,
                metadata_status="missing" if missing else "found",
                status=status,
                units=tuple(built),
                fallback_reason=fallback_reason,
            )
        )
    return KnowledgePack(
        query=str(query),
        products=tuple(products),
        dropped_units=dropped,
        notes={
            "source": KNOWLEDGE_SOURCE,
            "query_used": query_used,
            "max_units_per_product": int(max_units_per_product),
            "max_unit_chars": int(max_unit_chars),
            "max_total_units": int(max_total_units),
        },
    )
