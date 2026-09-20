"""The trusted CandidateLedger (AgentRec-X 2.0-alpha Stage 2).

The ledger is the run's single record of **which products are candidates**, and of **where
each one came from**.  It exists because Stage 2 introduced more than one trusted source of
candidates, and with more than one source three questions stop having obvious answers:

1. *Is this product a candidate at all?*  A product is a candidate because a trusted tool
   produced it and the ledger accepted it - never because a policy named it.
2. *Why is it a candidate?*  A product found by both the history recommender and catalogue
   search has two independent grounds for being present; discarding one loses information
   that evaluation needs.
3. *How should candidates from different sources be ordered against each other?*  Raw
   scores from a sequential recommender and a lexical search are not comparable quantities.

The design decisions that follow from those questions:

**Deduplicate identity, not evidence.**  One :class:`LedgerEntry` per ``parent_asin``.  A
second sighting does not replace the first; it appends a :class:`CandidateProvenance`
record.  Every source that proposed the product stays recoverable, with its own rank and its
own score.

**Never fuse heterogeneous raw scores.**  ``HISTORY`` scores are SASRec logits; ``CATALOG_SEARCH``
scores are BM25 values; ``SIMILAR_ITEM`` scores are similarity values.  Summing or averaging
them would be a fabricated quantity with no meaning.  The ledger instead keeps every raw
score in its own provenance record and exposes a **rank-based** fusion
(:func:`reciprocal_rank_fusion`), which is well-defined precisely because it uses only the
*order* each source produced - the one thing all sources can be trusted to mean.

**Identity is validated, provenance is retained.**  An entry is ``GROUNDED`` only when its
identity was confirmed against the trusted item mapping (and, optionally, the catalogue).
An identity that a source returned but the mapping does not know is ``UNGROUNDED`` and is
retained as an *audit record* rather than silently dropped: a source that invents products
must be visible in evaluation, not invisible.

**The ledger is not writable by a policy.**  There is no public ``add`` that takes a bare
identity from a caller: entries are added through :meth:`CandidateLedger.record_from_source`,
which requires a declaration of which trusted source produced the candidates and at what
ranks.  A policy is nowhere in that call path.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .arguments import CANDIDATE_PRODUCING_SOURCES, CandidateSource

__all__ = [
    "LEDGER_VERSION",
    "CandidateProvenance",
    "CandidateVerificationStatus",
    "CandidateLedger",
    "LedgerEntry",
    "LedgerSourceError",
    "reciprocal_rank_fusion",
]

#: Ledger contract version, bumped when the entry or provenance shape changes.
LEDGER_VERSION = 1

#: RRF's smoothing constant.  The published default from the original reciprocal-rank
#: fusion work; kept explicit and named so the fusion is reproducible and auditable.
RRF_K = 60


class LedgerSourceError(Exception):
    """A caller tried to add candidates from a source the ledger does not trust."""


class CandidateVerificationStatus(str, Enum):
    """How far an entry's identity has been verified.

    * :attr:`GROUNDED` - the identity exists in the trusted item mapping (and, when a
      catalogue is available, in the catalogue).  It may be shown and reasoned about.
    * :attr:`UNGROUNDED` - a source returned this identity but the trusted mapping does not
      know it.  **Retained for audit only**; it is never presented as a candidate, and its
      presence is a finding about the source rather than about the user.
    """

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"


class CandidateProvenance(BaseModel):
    """One source's sighting of one product.

    A product seen by two sources has two of these.  Nothing merges them, because the rank
    a source assigned and the score it reported are facts about *that source's* view, and a
    fused value would be a quantity no source ever produced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: CandidateSource
    #: 1-based rank within that source's own result list.  Always present: rank is the one
    #: cross-source quantity that is meaningful, and it is what rank-based fusion uses.
    source_rank: int = Field(..., ge=1)
    #: The source's own raw score, if it reports one.  Preserved verbatim and **never**
    #: compared against another source's score.
    source_score: float | None = None
    #: The score's meaning, so a consumer cannot mistake one scale for another.
    score_kind: str = Field(
        default="unspecified",
        description="e.g. 'sasrec_logit', 'bm25', 'item_item_similarity'.",
    )
    #: The query or seed that produced this sighting, when there was one.  Untrusted text;
    #: recorded for provenance and evaluation, never re-executed.
    source_query: str | None = Field(default=None, max_length=200)
    #: The item id the source associated with this identity, when it had one.  Recorded so
    #: an identity conflict (same product, different item id) is detectable.
    item_id: int | None = Field(default=None, ge=1)
    #: The loop step that produced this sighting, for trajectory reconstruction.
    step_index: int = Field(default=0, ge=0)

    @property
    def is_history_source(self) -> bool:
        """True when this sighting came from the trusted sequential recommender."""
        return self.source is CandidateSource.HISTORY


class LedgerEntry(BaseModel):
    """One candidate product, with every source that proposed it.

    ``provenance`` is ordered by when the sighting was recorded, so the first source to
    propose the product is ``provenance[0]``.  Ordering the provenance is not the same as
    ordering the candidates: the ledger's own order is defined by
    :meth:`CandidateLedger.ranked_parent_asins`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str = Field(..., min_length=1)
    item_id: int | None = Field(default=None, ge=1)
    verification_status: CandidateVerificationStatus = CandidateVerificationStatus.GROUNDED
    provenance: tuple[CandidateProvenance, ...] = ()

    @property
    def sources(self) -> tuple[CandidateSource, ...]:
        """Every source that proposed this product, in first-seen order."""
        return tuple(record.source for record in self.provenance)

    @property
    def source_count(self) -> int:
        """How many independent trusted sources proposed this product."""
        return len(self.provenance)

    @property
    def is_multi_source(self) -> bool:
        """True when more than one trusted source proposed this product.

        This is the ledger's most useful *evidence* signal, and it is a count of
        independent grounds - not a fused score.
        """
        return len(self.provenance) > 1

    @property
    def first_seen_rank(self) -> int | None:
        """The rank the first source assigned, or ``None`` for an entry with no provenance."""
        return self.provenance[0].source_rank if self.provenance else None

    def provenance_for(self, source: CandidateSource) -> tuple[CandidateProvenance, ...]:
        """Return the sightings contributed by one source (usually zero or one)."""
        return tuple(record for record in self.provenance if record.source is source)

    def has_source(self, source: CandidateSource) -> bool:
        """True when ``source`` proposed this product."""
        return any(record.source is source for record in self.provenance)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse several ranked identity lists by reciprocal rank.

    ``score(d) = sum over lists of 1 / (k + rank(d))``

    Why rank-based fusion and not score fusion: the sources in this repository score on
    incomparable scales (SASRec logits, BM25, item-item similarity).  A weighted sum of
    those numbers would be a fabricated quantity.  Reciprocal rank fusion uses only each
    source's *ordering*, which is the one thing every retriever can be trusted to mean, and
    it is deterministic and parameter-light.

    Ties are broken by the identity string, so the result is a total order and is
    reproducible across runs.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    totals: dict[str, float] = {}
    for ranking in rankings:
        for position, identity in enumerate(ranking, start=1):
            totals[identity] = totals.get(identity, 0.0) + 1.0 / (k + position)
    return sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))


class CandidateLedger:
    """The run's trusted record of candidate identity and provenance.

    Construct one per run.  The ledger has no dependency on a policy, on a tool, or on the
    catalogue: it accepts candidates only from a declared trusted source, and grounding is
    the caller's job because only the caller holds the trusted mapping.
    """

    def __init__(self, *, version: int = LEDGER_VERSION) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._version = version

    # -- metadata ---------------------------------------------------------- #

    @property
    def version(self) -> int:
        """Ledger contract version."""
        return self._version

    def __len__(self) -> int:
        """Number of **grounded** entries."""
        return len(self.grounded_entries)

    def __contains__(self, parent_asin: object) -> bool:
        """True when a grounded entry exists for ``parent_asin``."""
        entry = self._entries.get(parent_asin) if isinstance(parent_asin, str) else None
        return entry is not None and entry.verification_status is CandidateVerificationStatus.GROUNDED

    # -- views ------------------------------------------------------------- #

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Every entry, grounded and ungrounded, in insertion order."""
        return tuple(self._entries.values())

    @property
    def grounded_entries(self) -> tuple[LedgerEntry, ...]:
        """Only the entries whose identity was verified."""
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.verification_status is CandidateVerificationStatus.GROUNDED
        )

    @property
    def ungrounded_entries(self) -> tuple[LedgerEntry, ...]:
        """Entries kept purely as evidence that a source returned an unknown identity."""
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.verification_status is CandidateVerificationStatus.UNGROUNDED
        )

    def get(self, parent_asin: str) -> LedgerEntry | None:
        """Return the entry for ``parent_asin``, or ``None``."""
        return self._entries.get(parent_asin)

    def grounded_parent_asins(self) -> tuple[str, ...]:
        """Grounded identities in ledger order (see :meth:`ranked_parent_asins`)."""
        return tuple(entry.parent_asin for entry in self.ranked_entries())

    def sources_present(self) -> tuple[CandidateSource, ...]:
        """The distinct sources that contributed at least one grounded entry."""
        seen: list[CandidateSource] = []
        for entry in self.grounded_entries:
            for record in entry.provenance:
                if record.source not in seen:
                    seen.append(record.source)
        return tuple(seen)

    def multi_source_parent_asins(self) -> tuple[str, ...]:
        """Grounded identities proposed by more than one source.

        The strongest evidence signal the ledger holds: independent agreement between
        sources, with no fused score involved.
        """
        return tuple(
            entry.parent_asin for entry in self.grounded_entries if entry.is_multi_source
        )

    def ranked_entries(self) -> tuple[LedgerEntry, ...]:
        """Grounded entries in the ledger's canonical order.

        The order is the rank-based fusion of every source's ranking.  Products no source
        ranked are impossible (every entry has provenance), and products only one source
        ranked still appear, so fusion never drops a candidate.
        """
        grounded = self.grounded_entries
        if not grounded:
            return ()
        by_source: dict[CandidateSource, list[tuple[int, str]]] = {}
        for entry in grounded:
            for record in entry.provenance:
                by_source.setdefault(record.source, []).append(
                    (record.source_rank, entry.parent_asin)
                )
        rankings = [
            [asin for _, asin in sorted(pairs)]
            for _, pairs in sorted(by_source.items(), key=lambda pair: pair[0].value)
        ]
        order = [asin for asin, _ in reciprocal_rank_fusion(rankings)]
        position = {asin: index for index, asin in enumerate(order)}
        return tuple(
            sorted(grounded, key=lambda entry: position.get(entry.parent_asin, len(order)))
        )

    def ranked_parent_asins(self) -> tuple[str, ...]:
        """Grounded identities in the ledger's canonical (rank-fused) order."""
        return tuple(entry.parent_asin for entry in self.ranked_entries())

    def fusion_scores(self) -> dict[str, float]:
        """The reciprocal-rank fusion score of each grounded identity.

        Exposed for audit.  These are **rank-based agreement scores**, not relevance,
        probability or quality, and they are not comparable to any source's raw score.
        """
        grounded = self.grounded_entries
        by_source: dict[CandidateSource, list[tuple[int, str]]] = {}
        for entry in grounded:
            for record in entry.provenance:
                by_source.setdefault(record.source, []).append(
                    (record.source_rank, entry.parent_asin)
                )
        rankings = [
            [asin for _, asin in sorted(pairs)]
            for _, pairs in sorted(by_source.items(), key=lambda pair: pair[0].value)
        ]
        return dict(reciprocal_rank_fusion(rankings))

    # -- mutation ---------------------------------------------------------- #

    def record_from_source(
        self,
        *,
        source: CandidateSource,
        candidates: list[tuple[str, int, float | None]],
        score_kind: str = "unspecified",
        item_ids: dict[str, int] | None = None,
        source_query: str | None = None,
        step_index: int = 0,
        grounded: frozenset[str] | None = None,
        absorb_ungrounded: bool = True,
    ) -> tuple[str, ...]:
        """Record one trusted source's result and return the affected identities.

        Parameters
        ----------
        source:
            Which trusted source produced these candidates.  A source that is not declared
            candidate-producing raises :class:`LedgerSourceError`: the ledger will not
            accept candidates from a component it does not know is allowed to produce them.
        candidates:
            ``(parent_asin, source_rank, source_score)`` triples in the source's own order.
            The rank is the source's rank, not a position: it is preserved verbatim.
        score_kind:
            What the scores mean, so they are never compared across sources by accident.
        item_ids:
            Optional identity -> item id map from the trusted item mapping.
        grounded:
            The set of identities the trusted mapping knows.  ``None`` means "the caller
            verified nothing", in which case entries are recorded as ungrounded rather than
            being silently trusted - grounding fails closed.
        absorb_ungrounded:
            When true, an identity outside ``grounded`` is retained as an UNGROUNDED audit
            entry instead of being discarded.  Discarding would hide a misbehaving source.

        Returns
        -------
        tuple[str, ...]
            The identities whose ledger entry changed, in the order supplied.
        """
        if not isinstance(source, CandidateSource) or source not in CANDIDATE_PRODUCING_SOURCES:
            name = source.value if isinstance(source, CandidateSource) else type(source).__name__
            raise LedgerSourceError(
                f"'{name}' is not a trusted candidate-producing source; "
                f"allowed: {', '.join(s.value for s in CANDIDATE_PRODUCING_SOURCES)}"
            )
        verified = grounded or frozenset()
        touched: list[str] = []

        for parent_asin, source_rank, source_score in candidates:
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                continue
            identity = parent_asin.strip()
            is_grounded = identity in verified
            if not is_grounded and not absorb_ungrounded:
                continue

            status = (
                CandidateVerificationStatus.GROUNDED
                if is_grounded
                else CandidateVerificationStatus.UNGROUNDED
            )
            provenance = CandidateProvenance(
                source=source,
                source_rank=int(source_rank),
                source_score=source_score,
                score_kind=score_kind,
                source_query=source_query,
                item_id=(item_ids or {}).get(identity),
                step_index=step_index,
            )

            existing = self._entries.get(identity)
            if existing is None:
                self._entries[identity] = LedgerEntry(
                    parent_asin=identity,
                    item_id=provenance.item_id,
                    verification_status=status,
                    provenance=(provenance,),
                )
                touched.append(identity)
                continue

            # Identity already present: append this source's sighting, and never let a
            # later ungrounded sighting demote an entry that was already grounded.
            merged_status = (
                CandidateVerificationStatus.GROUNDED
                if existing.verification_status is CandidateVerificationStatus.GROUNDED
                or status is CandidateVerificationStatus.GROUNDED
                else CandidateVerificationStatus.UNGROUNDED
            )
            self._entries[identity] = existing.model_copy(
                update={
                    "verification_status": merged_status,
                    "item_id": existing.item_id or provenance.item_id,
                    "provenance": (*existing.provenance, provenance),
                }
            )
            touched.append(identity)
        return tuple(touched)

    def record_ungrounded_identities(
        self,
        *,
        source: CandidateSource,
        identities: list[str],
        step_index: int = 0,
    ) -> tuple[str, ...]:
        """Record identities a trusted source returned that the mapping does not know.

        Kept separately so a hallucinating or drifting source is *visible* in evaluation.
        These entries are never presented as candidates.
        """
        if source not in CANDIDATE_PRODUCING_SOURCES:
            raise LedgerSourceError(
                f"'{source.value}' is not a trusted candidate-producing source"
            )
        touched: list[str] = []
        for rank, identity in enumerate(identities, start=1):
            if not isinstance(identity, str) or not identity.strip():
                continue
            key = identity.strip()
            provenance = CandidateProvenance(
                source=source,
                source_rank=rank,
                score_kind="unknown_identity",
                step_index=step_index,
            )
            existing = self._entries.get(key)
            if existing is None:
                self._entries[key] = LedgerEntry(
                    parent_asin=key,
                    verification_status=CandidateVerificationStatus.UNGROUNDED,
                    provenance=(provenance,),
                )
            else:
                self._entries[key] = existing.model_copy(
                    update={"provenance": (*existing.provenance, provenance)}
                )
            touched.append(key)
        return tuple(touched)

    # -- export ------------------------------------------------------------ #

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary for the trajectory.

        Counts and provenance structure only: safe to record, and enough to attribute a
        candidate set back to the sources that produced it.
        """
        return {
            "version": self._version,
            "grounded": len(self.grounded_entries),
            "ungrounded": len(self.ungrounded_entries),
            "sources": [s.value for s in self.sources_present()],
            "multi_source_count": len(self.multi_source_parent_asins()),
            "entries": [
                {
                    "parent_asin": entry.parent_asin,
                    "grounded": entry.verification_status is CandidateVerificationStatus.GROUNDED,
                    "provenance": [
                        {
                            "source": record.source.value,
                            "rank": record.source_rank,
                            "score": record.source_score,
                            "score_kind": record.score_kind,
                        }
                        for record in entry.provenance
                    ],
                }
                for entry in self._entries.values()
            ],
        }
