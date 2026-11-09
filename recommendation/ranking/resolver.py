"""Trusted candidate snapshot resolver for the M6.1 offline ranking lab.

What this module does
---------------------
It is the **join point** between the trusted recommendation pipeline and the offline
belief-aware ranking foundation.  A caller that already owns trusted candidate rows
(retrieval/fusion happened elsewhere) hands them over as
:class:`TrustedCandidateInput` records; this resolver:

1. validates candidate identity against the trusted ``parent_asin <-> item_id`` mapping
   when one is supplied (PAD, blank ids, duplicates, rank collisions and foreign
   identities are rejected);
2. attaches catalogue metadata through the injected read-only ``MetadataIndex``; a
   missing record becomes an explicit ``metadata_status="missing"`` -- no attribute is
   ever fabricated or imputed;
3. builds an immutable :class:`~recommendation.ranking.schemas.CandidateSnapshot`;
4. optionally attaches M5-compatible candidate-pool / ranking references and digests
   computed with the accepted
   :func:`recommendation.state.shopping_adapter.candidate_pool_snapshot` /
   :func:`recommendation.state.shopping_adapter.ranking_snapshot` algorithms;
5. verifies an existing snapshot against a :class:`SnapshotExpectation` (ownership,
   freshness, count, references, digests, metadata identity, candidate validity);
6. applies **one** trusted eligibility projection to the whole set (the same shared
   projection every ranking arm must see).

What this module deliberately does not do
-----------------------------------------
* It never retrieves, scores, ranks, filters, reranks, repairs or completes candidates.
  There is no retriever, no model and no ranking policy in this module.
* It never reconstructs a candidate row from a digest.  A digest is *verification
  material*: the trusted in-memory rows are supplied by the caller and the digest only
  proves that they are the rows the M5 state referenced.  See the explicit comment in
  :meth:`TrustedCandidateFeatureResolver.build_snapshot`.
* It never invents metadata.  Absent is ``missing``; unusable is still absent.
* It does not decide eligibility.  Eligibility decisions come from the trusted
  constraint evaluator and are only projected onto the snapshot.

Every rejection is a :class:`~recommendation.ranking.errors.CandidateSnapshotError`
with a short, bounded message: raw candidate payloads are never echoed back.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.catalog.schemas import MissingMetadata, ProductMetadata
from recommendation.state.shopping_adapter import (
    candidate_pool_snapshot,
    ranking_snapshot,
)

from .errors import CandidateSnapshotError
from .schemas import (
    CandidateRecord,
    CandidateSnapshot,
    EligibilityState,
    SnapshotExpectation,
    SourceRank,
)

__all__ = [
    "IDENTITY_MAP_METHODS",
    "TrustedCandidateFeatureResolver",
    "TrustedCandidateInput",
]

#: The three methods a trusted identity mapping must expose.  The mapping remains the
#: authority for ``parent_asin <-> item_id``; this module only reads it.
IDENTITY_MAP_METHODS: tuple[str, ...] = (
    "parent_asin_to_item_id",
    "item_id_to_parent_asin",
    "has_parent_asin",
)

#: Bound on the human-readable eligibility detail copied into a candidate record.
_MAX_ELIGIBILITY_DETAIL = 300


class TrustedCandidateInput(BaseModel):
    """One candidate row as supplied by the trusted recommendation pipeline.

    This is an *input* type: it is not part of the frozen snapshot contract.  The
    resolver copies these fields into a :class:`CandidateRecord`, joins catalogue
    metadata and proves identity; it never changes the candidate's value semantics.

    ``score`` is copied exactly as the trusted pipeline produced it (for the frozen
    M6.1 protocol this is a fused RRF score; ``score_kind`` names what it is).  It is
    provenance only -- no ranking arm and no feature may read it numerically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., min_length=1, max_length=200)
    item_id: int = Field(
        ..., ge=1, description="Trusted integer item id; PAD=0 is never a candidate."
    )
    original_rank: int = Field(..., ge=1, description="Trusted 1-based upstream order.")
    score: float = Field(..., description="Copy of the trusted ordering score.")
    score_kind: str = Field(default="rrf_rank_fusion", max_length=80)
    source_ranks: tuple[SourceRank, ...] = Field(
        default=(),
        description="Per-source rank provenance, copied into the snapshot unchanged.",
    )
    rrf_score: float | None = Field(
        default=None, description="Fused score provenance when the pipeline produced one."
    )
    eligibility: EligibilityState = EligibilityState.NOT_ASSESSED
    eligibility_detail: str | None = Field(default=None, max_length=_MAX_ELIGIBILITY_DETAIL)

    @field_validator("candidate_id")
    @classmethod
    def _validate_candidate_id(cls, value: object) -> str:
        text = " ".join(str(value).split())
        if not text:
            raise ValueError("candidate_id must not be blank")
        return text

    @field_validator("score")
    @classmethod
    def _validate_score(cls, value: object) -> float:
        numeric = float(value)  # type: ignore[arg-type]
        if not math.isfinite(numeric):
            raise ValueError("score must be a finite number")
        return numeric

    @field_validator("rrf_score")
    @classmethod
    def _validate_rrf_score(cls, value: object) -> float | None:
        if value is None:
            return None
        numeric = float(value)  # type: ignore[arg-type]
        if not math.isfinite(numeric):
            raise ValueError("rrf_score must be a finite number or null")
        return numeric


class TrustedCandidateFeatureResolver:
    """Trusted join of candidate identity, catalogue metadata and eligibility.

    The resolver is stateless apart from its two injected trusted dependencies (the
    read-only metadata index and the optional identity mapping), so it is reusable and
    deterministic: given the same inputs it produces the same frozen snapshot.
    """

    def __init__(self, *, metadata_index: Any, identity_map: Any | None = None) -> None:
        lookup = getattr(metadata_index, "lookup", None)
        if not callable(lookup):
            raise TypeError(
                "metadata_index must expose a callable lookup(parent_asin) "
                "(the accepted MetadataIndex protocol)"
            )
        if identity_map is not None:
            missing = [
                method
                for method in IDENTITY_MAP_METHODS
                if not callable(getattr(identity_map, method, None))
            ]
            if missing:
                raise TypeError(
                    "identity_map must expose " + ", ".join(IDENTITY_MAP_METHODS)
                )
        self._metadata_index = metadata_index
        self._identity_map = identity_map

    # ------------------------------------------------------------------ #
    # Trust boundaries
    # ------------------------------------------------------------------ #

    #: The resolver only *joins* trusted inputs.  It never retrieves, scores, ranks or
    #: repairs candidates, and it never manufactures a metadata attribute.
    TRUST_BOUNDARY = (
        "join trusted rows only: no retrieval, no scoring, no ranking, no repair, "
        "no metadata invention"
    )

    # ------------------------------------------------------------------ #
    # Metadata join
    # ------------------------------------------------------------------ #

    def _lookup_metadata(self, candidate_id: str) -> tuple[str, ProductMetadata | None]:
        """Return ``(metadata_status, metadata)`` for one trusted candidate identity.

        ``MissingMetadata`` and ``None`` both mean "the catalogue holds no record" and
        become the explicit ``missing`` status.  A catalogue record whose own
        ``parent_asin`` disagrees with the candidate is rejected: silently joining the
        wrong product would corrupt every downstream feature.
        """
        record = self._metadata_index.lookup(candidate_id)
        if record is None or isinstance(record, MissingMetadata):
            return "missing", None
        if not isinstance(record, ProductMetadata):
            raise CandidateSnapshotError(
                "the metadata index returned an unsupported record type for a candidate"
            )
        if record.parent_asin != candidate_id:
            raise CandidateSnapshotError(
                "catalogue metadata belongs to a different parent_asin than the candidate"
            )
        return "found", record

    def _validate_trusted_identity(self, candidate: TrustedCandidateInput) -> None:
        """Reject a candidate that the trusted identity mapping does not own."""
        if self._identity_map is None:
            return
        parent_asin = candidate.candidate_id
        if not bool(self._identity_map.has_parent_asin(parent_asin)):
            raise CandidateSnapshotError(
                "candidate identity is not present in the trusted identity mapping"
            )
        mapped_item_id = self._identity_map.parent_asin_to_item_id(parent_asin)
        if mapped_item_id != candidate.item_id:
            raise CandidateSnapshotError(
                "candidate item_id does not match the trusted identity mapping"
            )
        mapped_parent_asin = self._identity_map.item_id_to_parent_asin(candidate.item_id)
        if mapped_parent_asin != parent_asin:
            raise CandidateSnapshotError(
                "candidate parent_asin does not match the trusted item_id mapping"
            )

    # ------------------------------------------------------------------ #
    # Snapshot construction
    # ------------------------------------------------------------------ #

    def build_snapshot(
        self,
        *,
        snapshot_id: str,
        user_key: str,
        task_id: str,
        turn_key: str,
        candidates: Sequence[TrustedCandidateInput],
        source: str = "trusted_candidate_snapshot",
        candidate_pool_ref: str | None = None,
        ranking_snapshot_ref: str | None = None,
    ) -> CandidateSnapshot:
        """Build one immutable trusted snapshot from trusted candidate rows.

        Every candidate is validated, its trusted identity is checked against the
        identity mapping (when supplied) and its catalogue metadata is joined.  When
        ``candidate_pool_ref`` / ``ranking_snapshot_ref`` are supplied they must equal
        the references the accepted M5 digest algorithm derives from this exact
        identity set/order; the matching digests are attached to the snapshot.

        **Digests are verification material only.**  No row, field or rank in the
        returned snapshot was reconstructed from a digest: the candidate rows came from
        the caller and the digest merely proves that this set is the one the M5 state
        recorded.  The same rule applies to the frozen protocol: a digest can reject a
        snapshot, it can never recreate one.
        """
        scope = self._validate_scope(
            snapshot_id=snapshot_id,
            user_key=user_key,
            task_id=task_id,
            turn_key=turn_key,
            source=source,
        )
        records = self._build_records(candidates)

        pool_digest: str | None = None
        ranking_digest: str | None = None
        if candidate_pool_ref is not None:
            pool = candidate_pool_snapshot(_identity_sequence(records))
            if candidate_pool_ref != pool.ref:
                raise CandidateSnapshotError(
                    "candidate_pool_ref does not match the trusted candidate identity set"
                )
            pool_digest = pool.digest
        if ranking_snapshot_ref is not None:
            ranking = ranking_snapshot(_identity_sequence(records))
            if ranking_snapshot_ref != ranking.ref:
                raise CandidateSnapshotError(
                    "ranking_snapshot_ref does not match the trusted candidate order"
                )
            ranking_digest = ranking.digest

        try:
            snapshot = CandidateSnapshot(
                snapshot_id=scope["snapshot_id"],
                user_key=scope["user_key"],
                task_id=scope["task_id"],
                turn_key=scope["turn_key"],
                candidates=tuple(records),
                source=scope["source"],
                candidate_pool_ref=candidate_pool_ref,
                ranking_snapshot_ref=ranking_snapshot_ref,
                candidate_pool_digest=pool_digest,
                ranking_snapshot_digest=ranking_digest,
            )
        except Exception as error:  # validated above; translate to the typed rejection
            raise CandidateSnapshotError(
                "candidate snapshot could not be constructed from the trusted inputs"
            ) from error
        return snapshot

    def _validate_scope(self, **scope: object) -> dict[str, str]:
        """Validate the identity strings a snapshot is built for."""
        result: dict[str, str] = {}
        for name in ("snapshot_id", "user_key", "task_id", "turn_key", "source"):
            raw = scope[name]
            text = " ".join(str(raw).split()) if raw is not None else ""
            if not text:
                raise CandidateSnapshotError(f"{name} must be a non-blank string")
            result[name] = text
        return result

    def _build_records(
        self, candidates: Sequence[TrustedCandidateInput]
    ) -> list[CandidateRecord]:
        """Validate trusted candidate rows and copy them into frozen records."""
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise CandidateSnapshotError(
                "candidates must be a sequence of trusted candidate inputs"
            )

        records: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        seen_item_ids: set[int] = set()
        seen_ranks: set[int] = set()

        for position, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, TrustedCandidateInput):
                raise CandidateSnapshotError(
                    "candidates must be TrustedCandidateInput instances"
                )
            candidate_id = " ".join(str(candidate.candidate_id).split())
            if not candidate_id:
                raise CandidateSnapshotError("a trusted candidate has a blank candidate_id")
            if candidate_id in seen_ids:
                raise CandidateSnapshotError("candidate ids must be unique within a snapshot")
            seen_ids.add(candidate_id)

            item_id = candidate.item_id
            if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
                raise CandidateSnapshotError(
                    "a candidate item_id must be a positive integer; PAD=0 is never valid"
                )
            if item_id in seen_item_ids:
                raise CandidateSnapshotError(
                    "candidate item ids must be unique within a snapshot"
                )
            seen_item_ids.add(item_id)

            rank = candidate.original_rank
            if not isinstance(rank, int) or rank < 1:
                raise CandidateSnapshotError(
                    "candidate original_rank must be a positive integer"
                )
            if rank in seen_ranks:
                raise CandidateSnapshotError(
                    "candidate original ranks must not collide"
                )
            seen_ranks.add(rank)
            if rank != position:
                raise CandidateSnapshotError(
                    "candidate original ranks must be contiguous 1..n in snapshot order"
                )

            score = candidate.score
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise CandidateSnapshotError("candidate score must be numeric")
            if not math.isfinite(float(score)):
                raise CandidateSnapshotError("candidate score must be a finite number")

            self._validate_trusted_identity(candidate)
            metadata_status, metadata = self._lookup_metadata(candidate_id)

            try:
                record = CandidateRecord(
                    candidate_id=candidate_id,
                    item_id=item_id,
                    original_rank=rank,
                    score=float(score),
                    score_kind=candidate.score_kind,
                    metadata_status=metadata_status,  # type: ignore[arg-type]
                    metadata=metadata,
                    source_ranks=tuple(candidate.source_ranks),
                    rrf_score=candidate.rrf_score,
                    eligibility=candidate.eligibility,
                    eligibility_detail=candidate.eligibility_detail,
                )
            except Exception as error:  # defensive: inputs were validated above
                raise CandidateSnapshotError(
                    "a trusted candidate could not be copied into the snapshot"
                ) from error
            records.append(record)
        return records

    # ------------------------------------------------------------------ #
    # Snapshot verification
    # ------------------------------------------------------------------ #

    def verify_snapshot(
        self,
        snapshot: CandidateSnapshot,
        expectation: SnapshotExpectation | None = None,
    ) -> CandidateSnapshot:
        """Verify a snapshot's internal consistency and, when given, its ownership.

        The expectation is **authoritative** for user/task/turn identity: a snapshot
        that belongs to another user, another task or an earlier turn (a stale snapshot)
        is rejected.  Count, reference and digest mismatches are rejected whenever
        either side carries the value.  The declared digest fields must also equal the
        digests recomputed from the row identities, so a snapshot cannot carry a
        verified-looking digest it does not actually satisfy.

        Returns the validated snapshot unchanged (it is frozen; verification never
        mutates or repairs it).
        """
        if not isinstance(snapshot, CandidateSnapshot):
            raise CandidateSnapshotError("verify_snapshot requires a CandidateSnapshot")
        if expectation is not None and not isinstance(expectation, SnapshotExpectation):
            raise CandidateSnapshotError(
                "expectation must be a SnapshotExpectation when supplied"
            )

        self._verify_candidates(snapshot)

        pool_digest = snapshot.candidate_pool_digest_value()
        ranking_digest = snapshot.ranking_digest_value()
        pool_ref = candidate_pool_snapshot(_identity_sequence(snapshot.candidates)).ref
        ranking_ref = ranking_snapshot(_identity_sequence(snapshot.candidates)).ref

        if snapshot.candidate_pool_ref is not None and snapshot.candidate_pool_ref != pool_ref:
            raise CandidateSnapshotError(
                "snapshot candidate_pool_ref does not match its candidate identity set"
            )
        if snapshot.ranking_snapshot_ref is not None and snapshot.ranking_snapshot_ref != ranking_ref:
            raise CandidateSnapshotError(
                "snapshot ranking_snapshot_ref does not match its candidate order"
            )
        if (
            snapshot.candidate_pool_digest is not None
            and snapshot.candidate_pool_digest != pool_digest
        ):
            raise CandidateSnapshotError(
                "snapshot candidate_pool_digest does not equal the recomputed M5 digest"
            )
        if (
            snapshot.ranking_snapshot_digest is not None
            and snapshot.ranking_snapshot_digest != ranking_digest
        ):
            raise CandidateSnapshotError(
                "snapshot ranking_snapshot_digest does not equal the recomputed M5 digest"
            )

        if expectation is not None:
            self._verify_expectation(
                snapshot=snapshot,
                expectation=expectation,
                pool_digest=pool_digest,
                ranking_digest=ranking_digest,
                pool_ref=pool_ref,
                ranking_ref=ranking_ref,
            )
        return snapshot

    def _verify_candidates(self, snapshot: CandidateSnapshot) -> None:
        """Re-validate candidate identity, order and metadata consistency defensively."""
        seen_ids: set[str] = set()
        seen_item_ids: set[int] = set()
        for position, candidate in enumerate(snapshot.candidates, start=1):
            candidate_id = " ".join(str(candidate.candidate_id).split())
            if not candidate_id:
                raise CandidateSnapshotError("a snapshot candidate has a blank candidate_id")
            if candidate_id in seen_ids:
                raise CandidateSnapshotError(
                    "a candidate snapshot may not contain duplicate identities"
                )
            seen_ids.add(candidate_id)

            item_id = candidate.item_id
            if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
                raise CandidateSnapshotError(
                    "a snapshot candidate item_id must be positive; PAD=0 is never valid"
                )
            if item_id in seen_item_ids:
                raise CandidateSnapshotError(
                    "a candidate snapshot may not contain duplicate item ids"
                )
            seen_item_ids.add(item_id)

            if candidate.original_rank != position:
                raise CandidateSnapshotError(
                    "candidate original ranks must be contiguous 1..n in snapshot order"
                )
            if not math.isfinite(float(candidate.score)):
                raise CandidateSnapshotError("a snapshot candidate score must be finite")

            if candidate.metadata_status == "found":
                if candidate.metadata is None:
                    raise CandidateSnapshotError(
                        "a candidate declares found metadata but carries none"
                    )
                if candidate.metadata.parent_asin != candidate_id:
                    raise CandidateSnapshotError(
                        "catalogue metadata belongs to a different parent_asin than the candidate"
                    )
            elif candidate.metadata is not None:
                raise CandidateSnapshotError(
                    "a candidate carries metadata but does not declare it as found"
                )

            if self._identity_map is not None:
                self._validate_trusted_identity(
                    TrustedCandidateInput(
                        candidate_id=candidate_id,
                        item_id=item_id,
                        original_rank=candidate.original_rank,
                        score=candidate.score,
                        score_kind=candidate.score_kind,
                        source_ranks=candidate.source_ranks,
                        rrf_score=candidate.rrf_score,
                        eligibility=candidate.eligibility,
                        eligibility_detail=candidate.eligibility_detail,
                    )
                )

    def _verify_expectation(
        self,
        *,
        snapshot: CandidateSnapshot,
        expectation: SnapshotExpectation,
        pool_digest: str,
        ranking_digest: str,
        pool_ref: str,
        ranking_ref: str,
    ) -> None:
        """Check the authoritative ownership/freshness/ref expectations."""
        if snapshot.user_key != expectation.user_key:
            raise CandidateSnapshotError(
                "snapshot belongs to a different user than the trusted expectation"
            )
        if snapshot.task_id != expectation.task_id:
            raise CandidateSnapshotError(
                "snapshot belongs to a different task than the trusted expectation"
            )
        if snapshot.turn_key != expectation.turn_key:
            raise CandidateSnapshotError(
                "snapshot is stale: its turn key differs from the trusted expectation"
            )
        if (
            expectation.candidate_count is not None
            and snapshot.candidate_count != expectation.candidate_count
        ):
            raise CandidateSnapshotError(
                "snapshot candidate count does not match the trusted expectation"
            )
        if (
            expectation.candidate_pool_ref is not None
            and expectation.candidate_pool_ref != pool_ref
        ):
            raise CandidateSnapshotError(
                "snapshot does not match the expected trusted candidate pool reference"
            )
        if (
            expectation.ranking_snapshot_ref is not None
            and expectation.ranking_snapshot_ref != ranking_ref
        ):
            raise CandidateSnapshotError(
                "snapshot does not match the expected trusted ranking reference"
            )
        if (
            expectation.candidate_pool_digest is not None
            and expectation.candidate_pool_digest != pool_digest
        ):
            raise CandidateSnapshotError(
                "snapshot does not match the expected candidate-pool digest"
            )
        if (
            expectation.ranking_snapshot_digest is not None
            and expectation.ranking_snapshot_digest != ranking_digest
        ):
            raise CandidateSnapshotError(
                "snapshot does not match the expected ranking digest"
            )

    # ------------------------------------------------------------------ #
    # Eligibility projection
    # ------------------------------------------------------------------ #

    def apply_eligibility(
        self,
        snapshot: CandidateSnapshot,
        decisions: Mapping[
            str, EligibilityState | str | tuple[EligibilityState, str | None]
        ],
    ) -> CandidateSnapshot:
        """Project one trusted eligibility decision map onto the whole snapshot.

        The decisions come from the trusted constraint evaluator (this resolver does
        not assess constraints itself).  A decision naming a candidate outside the
        snapshot is a protocol error: an eligibility projection may never introduce a
        candidate.  Conflicting duplicate decisions are impossible by construction
        because the projection is a mapping (one entry per candidate id).

        Returns a new frozen snapshot with ``eligibility``/``eligibility_detail``
        updated; candidate identity, order, provenance and digests are untouched.
        """
        if not isinstance(snapshot, CandidateSnapshot):
            raise CandidateSnapshotError("apply_eligibility requires a CandidateSnapshot")
        if not isinstance(decisions, Mapping):
            raise CandidateSnapshotError("eligibility decisions must be a mapping")

        known = set(snapshot.candidate_ids)
        foreign = [candidate_id for candidate_id in decisions if candidate_id not in known]
        if foreign:
            raise CandidateSnapshotError(
                "eligibility decisions name candidate(s) outside the snapshot"
            )

        updated: list[CandidateRecord] = []
        for candidate in snapshot.candidates:
            if candidate.candidate_id not in decisions:
                updated.append(candidate)
                continue
            state, detail = _coerce_eligibility_decision(decisions[candidate.candidate_id])
            updated.append(
                candidate.model_copy(
                    update={"eligibility": state, "eligibility_detail": detail}
                )
            )
        return snapshot.model_copy(update={"candidates": tuple(updated)})


def _identity_sequence(candidates: Sequence[CandidateRecord]) -> tuple[str, ...]:
    """Return candidate identities in trusted order."""
    return tuple(candidate.candidate_id for candidate in candidates)


def _coerce_eligibility_decision(
    decision: EligibilityState | str | tuple[EligibilityState, str | None],
) -> tuple[EligibilityState, str | None]:
    """Normalise one projected eligibility decision to ``(state, detail)``."""
    if isinstance(decision, EligibilityState):
        return decision, None
    if isinstance(decision, str):
        try:
            return EligibilityState(decision), None
        except ValueError as error:
            raise CandidateSnapshotError(
                "an eligibility decision carries an unknown eligibility state"
            ) from error
    if isinstance(decision, tuple) and len(decision) == 2:
        state = _coerce_eligibility_decision(decision[0])[0]
        detail = decision[1]
        if detail is not None:
            if not isinstance(detail, str):
                raise CandidateSnapshotError(
                    "an eligibility decision detail must be a string or null"
                )
            detail = " ".join(detail.split())
            if len(detail) > _MAX_ELIGIBILITY_DETAIL:
                raise CandidateSnapshotError(
                    "an eligibility decision detail exceeds the allowed length"
                )
        return state, detail or None
    raise CandidateSnapshotError(
        "an eligibility decision must be an EligibilityState, a state string or "
        "(state, detail)"
    )
