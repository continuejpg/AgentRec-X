"""Belief-aware typed feature builder for the M6.1 offline ranking lab.

The builder turns two already-trusted inputs into one deterministic
:class:`~recommendation.ranking.schemas.CandidateFeatureSet`:

* a frozen :class:`~recommendation.ranking.schemas.CandidateSnapshot` (identity,
  original order, catalogue metadata, eligibility) -- built by
  :class:`~recommendation.ranking.resolver.TrustedCandidateFeatureResolver`;
* the M5 :class:`~recommendation.state.state_projector.RankerStateProjection`
  (``for_ranker``) of the unified decision state.

No candidate is added, dropped, reordered or repaired here.  The builder never reads a
label: its only inputs are the snapshot and the projection, so ground-truth labels cannot
leak into feature construction (the evaluator keeps them on the evaluation task).

Evidence semantics
------------------
* **Explicit evidence** is produced by the accepted M10A
  :class:`~recommendation.preference_matching.PreferenceCandidateMatcher` over the
  candidates' real catalogue metadata and the explicit/confirmed projection features
  projected into M9-shaped :class:`~recommendation.memory.schemas.PreferenceMemoryEntry`
  records.  The resulting :class:`~recommendation.preference_matching.schemas.PreferenceEvidence`
  records are copied into the feature set unchanged (three-state semantics included).
* **Inferred, non-directional soft evidence** is produced by the *same real matcher*
  over the soft feature beliefs projected into synthetic M9-shaped entries, then
  translated into :class:`~recommendation.ranking.schemas.InferredPreferenceEvidence`.
  ``UNKNOWN``/absence semantics carry through unchanged; no violation is ever invented.
* **Inferred, directional soft evidence** (``direction`` is ``minimize``/``maximize``)
  is grounded only from a catalogue numeric field: an attribute whose name contains
  ``price`` is read from ``metadata.price_text`` and one whose name contains ``weight``
  from the first ``details`` key containing ``weight``.  Any other attribute is
  *unsupported* (reported in coverage; contributes no signal).  Missing/unparseable
  values are ``UNKNOWN`` and never a violation.  Known values additionally carry
  set-relative ordinal counts: how many other candidates in the **same frozen set**
  have a strictly better / worse grounded value.  Those counts are ordinal comparisons,
  not probabilities and not cross-model scores.

Ordinal support/penalty mapping (matches the frozen A2 key
``(penalty ASC, -support ASC)``): ``better_than_count`` is the number of peers that beat
this candidate and is summed into ``inferred_ordinal_penalty``; ``worse_than_count`` is
the number of peers this candidate beats and is summed into
``inferred_ordinal_support``.  A candidate with UNKNOWN evidence contributes ``0/0``.

No raw retrieval or model score is ever mixed into a feature value.  ``score``,
``source_ranks`` and ``rrf_score`` are copied as provenance only; every ordering-relevant
field is a count or an ordinal rank.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from recommendation.catalog.schemas import ProductMetadata
from recommendation.memory.schemas import (
    PreferenceKind as MemoryPreferenceKind,
    PreferenceMemoryEntry,
    PreferencePolarity as MemoryPreferencePolarity,
    PreferenceStatus,
)
from recommendation.preference_matching.matcher import PreferenceCandidateMatcher
from recommendation.preference_matching.schemas import (
    EvidenceStatus,
    MatchingSupport,
    PreferenceEvidence,
    PreferenceEvidenceReport,
    ReasonCode,
    matching_support_for,
)
from recommendation.state.schemas import PreferenceLifecycle
from recommendation.state.state_projector import RankerFeature, RankerStateProjection

from .errors import FeatureBuildError
from .schemas import (
    CandidateFeatureSet,
    CandidateRankingFeatures,
    CandidateRecord,
    CandidateSnapshot,
    CatalogueCoverage,
    CatalogueFields,
    EvidenceState,
    InferredPreferenceEvidence,
    PreferenceStrengthMetadata,
)

__all__ = [
    "CATALOGUE_FIELD_ORDER",
    "DIRECTIONAL_DIRECTIONS",
    "BeliefAwareFeatureBuilder",
]

#: Catalogue attributes whose availability is reported per candidate, in fixed order.
CATALOGUE_FIELD_ORDER: tuple[str, ...] = (
    "title",
    "price",
    "weight",
    "color",
    "material",
    "brand",
    "features",
    "categories",
)

#: Directions a soft belief must carry to be grounded as an ordinal numeric signal.
DIRECTIONAL_DIRECTIONS: frozenset[str] = frozenset({"minimize", "maximize"})

#: Lifecycles that are authoritative explicit evidence (never soft evidence).
_EXPLICIT_LIFECYCLES: frozenset[PreferenceLifecycle] = frozenset(
    {PreferenceLifecycle.EXPLICIT, PreferenceLifecycle.CONFIRMED}
)

#: The accepted soft-evidence lifecycle.
_INFERRED_LIFECYCLE = PreferenceLifecycle.INFERRED_HYPOTHESIS

#: Unit factors for the *catalogue-coverage readability check only*.  Directional
#: evidence is parsed by the accepted ``recommendation.control.grounded_reasoning``
#: helpers, imported lazily so this module stays importable without the model stack.
_WEIGHT_TO_KG: dict[str, float] = {
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
    "ounces": 0.028349523125,
}

#: Maximum metadata provenance text length accepted by the inferred schema.
_MAX_METADATA_FIELD = 120
_MAX_METADATA_VALUE = 200
_MAX_DETAIL = 300


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #


def _bounded(value: str | None, limit: int) -> str | None:
    """Return ``value`` truncated to ``limit`` characters (``None`` stays ``None``)."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text[:limit]


def _readable_price(text: str | None) -> float | None:
    """Return a price as a finite float when the source token is readable.

    Coverage-only readability check: directional grounding uses the accepted
    ``parse_price`` helper.  Neither path converts currency or infers a value.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip().replace(",", "")
    for symbol in ("$", "\u00a3", "\u20ac", "\u00a5"):
        raw = raw.replace(symbol, "")
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _find_weight_detail(record: ProductMetadata) -> tuple[str, str] | None:
    """Return the first ``details`` pair whose key contains ``weight``."""
    for key, value in record.details:
        if "weight" in str(key).casefold():
            return str(key), str(value)
    return None


def _local_weight_kg(text: str) -> float | None:
    """Coverage-only weight readability check mirroring the accepted parser's rule."""
    tokens = str(text).replace(",", " ").split()
    for index, token in enumerate(tokens):
        try:
            number = float(token)
        except ValueError:
            continue
        unit = tokens[index + 1].strip(".,").lower() if index + 1 < len(tokens) else ""
        factor = _WEIGHT_TO_KG.get(unit)
        return None if factor is None else number * factor
    return None


def _detail_has_key(record: ProductMetadata, keys: frozenset[str]) -> bool:
    """True when ``details`` carries a non-empty value under one of ``keys``."""
    for key, value in record.details:
        if str(key).casefold() in keys and str(value).strip():
            return True
    return False


def _catalogue_fields(record: ProductMetadata | None) -> CatalogueFields:
    """Report which catalogue attributes are readable for one candidate."""
    if record is None:
        return CatalogueFields(present=(), missing=CATALOGUE_FIELD_ORDER)
    availability: dict[str, bool] = {
        "title": bool(record.title),
        "price": _readable_price(record.price_text) is not None,
        "weight": (lambda location: location is not None and _local_weight_kg(location[1]) is not None)(
            _find_weight_detail(record)
        ),
        "color": _detail_has_key(record, frozenset({"color", "colour"})),
        "material": _detail_has_key(record, frozenset({"material"})),
        "brand": bool(record.store)
        or _detail_has_key(record, frozenset({"brand name", "brand", "manufacturer"})),
        "features": bool(record.features),
        "categories": bool(record.categories),
    }
    present = tuple(field for field in CATALOGUE_FIELD_ORDER if availability[field])
    missing = tuple(field for field in CATALOGUE_FIELD_ORDER if not availability[field])
    return CatalogueFields(present=present, missing=missing)


def _projection_feature_split(
    projection: RankerStateProjection,
) -> tuple[tuple[RankerFeature, ...], tuple[RankerFeature, ...]]:
    """Split projection features into explicit evidence features and soft beliefs."""
    explicit: list[RankerFeature] = []
    soft: list[RankerFeature] = []
    for feature in projection.features:
        is_explicit = (not feature.is_soft) and feature.lifecycle in _EXPLICIT_LIFECYCLES
        is_soft = feature.is_soft and feature.lifecycle is _INFERRED_LIFECYCLE
        if not (is_explicit or is_soft):
            raise FeatureBuildError(
                "a ranker projection feature has inconsistent is_soft/lifecycle metadata"
            )
        if is_explicit:
            explicit.append(feature)
        else:
            soft.append(feature)
    return tuple(explicit), tuple(soft)


def _memory_entry(
    feature: RankerFeature,
    *,
    memory_id: str,
    source_text: str,
    logical_seq: int,
    extractor: str,
) -> PreferenceMemoryEntry:
    """Project one ranker feature into an M9-shaped preference memory entry.

    The M5 projection deliberately carries no user key and no raw source text, so the
    projected entry uses an explicit synthetic user key and an attribution label.  No
    user-authored text is invented: ``source_text`` names the decision-state derivation.
    """
    try:
        return PreferenceMemoryEntry(
            memory_id=memory_id,
            user_key="state_projection",
            kind=MemoryPreferenceKind(feature.kind.value),
            value=feature.value,
            polarity=MemoryPreferencePolarity(feature.polarity.value),
            source_text=source_text,
            source_turn_id="decision_state",
            extractor=extractor,
            status=PreferenceStatus.ACTIVE,
            logical_seq=logical_seq,
            created_at=0.0,
        )
    except Exception as error:  # bounded translation of a projection-shape violation
        raise FeatureBuildError(
            "a ranker projection feature could not be projected into a preference entry"
        ) from error


def _explicit_entries(
    features: Sequence[RankerFeature],
) -> tuple[PreferenceMemoryEntry, ...]:
    """Project explicit/confirmed features into M9-shaped entries, deterministically.

    ``logical_seq`` is the 1-based position in the projection order (the M9 store's
    stable sort key); :func:`_assign_logical_sequence` re-validates and renumbers it.
    """
    entries: list[PreferenceMemoryEntry] = []
    for position, feature in enumerate(features, start=1):
        entries.append(
            _memory_entry(
                feature,
                memory_id=feature.preference_id,
                source_text=f"decision-state:{feature.preference_id}",
                logical_seq=position,
                extractor="m5_ranker_projection",
            )
        )
    return _assign_logical_sequence(entries)


def _soft_entries(
    features: Sequence[RankerFeature],
) -> tuple[PreferenceMemoryEntry, ...]:
    """Project soft hypothesis features into synthetic M9-shaped entries."""
    entries: list[PreferenceMemoryEntry] = []
    for position, feature in enumerate(features, start=1):
        entries.append(
            _memory_entry(
                feature,
                memory_id=f"inferred:{feature.preference_id}",
                source_text=f"decision-state-hypothesis:{feature.preference_id}",
                logical_seq=position,
                extractor="m5_ranker_projection_hypothesis",
            )
        )
    return _assign_logical_sequence(entries)


def _assign_logical_sequence(
    entries: Sequence[PreferenceMemoryEntry],
) -> tuple[PreferenceMemoryEntry, ...]:
    """Sort by ``(logical_seq, memory_id)`` and renumber ``logical_seq`` 1..n.

    ``PreferenceMemoryEntry.logical_seq`` is a required ``>= 1`` field, so projection
    order is carried as the initial 1-based position and then normalised here.  Sorting
    by the sequence (then id) makes the result independent of any incidental iteration
    order while preserving the projection order.
    """
    ordered = sorted(entries, key=lambda entry: (entry.logical_seq, entry.memory_id))
    result: list[PreferenceMemoryEntry] = []
    for sequence, entry in enumerate(ordered, start=1):
        data = entry.model_dump()
        data["logical_seq"] = sequence
        result.append(PreferenceMemoryEntry.model_validate(data))
    return tuple(result)


def _enriched_candidates(candidates: Sequence[CandidateRecord]) -> list[Any]:
    """Build the accepted M8 enrichment items the real M10A matcher consumes.

    The accepted candidate/evidence types are imported lazily because their package
    initialisers pull the model stack in this repository; this module itself stays
    importable without it.  No candidate is changed: rank, identity, item id and score
    are copied from the frozen snapshot, metadata is attached as-is (absent stays
    absent) and nothing is retrieved.
    """
    from recommendation.rag.schemas import EnrichedRecommendation
    from recommendation.tools.schemas import ToolRecommendation

    enriched: list[EnrichedRecommendation] = []
    for position, candidate in enumerate(candidates):
        enriched.append(
            EnrichedRecommendation(
                recommendation=ToolRecommendation(
                    rank=candidate.original_rank,
                    parent_asin=candidate.candidate_id,
                    item_id=candidate.item_id,
                    score=candidate.score,
                    score_kind=candidate.score_kind,
                ),
                metadata_status=candidate.metadata_status,
                metadata=candidate.metadata,
                evidence=(),
                fallback_reason=None if candidate.metadata is not None else "no_metadata",
                position=position,
            )
        )
    return enriched


def _evidence_by_candidate(
    report: PreferenceEvidenceReport,
    enriched: Sequence[Any],
) -> dict[str, tuple[Any, ...]]:
    """Map the real matcher report back onto candidate identities, in input order."""
    if len(report.candidates) != len(enriched):
        raise FeatureBuildError(
            "the preference matcher did not return exactly the supplied candidates"
        )
    by_candidate: dict[str, tuple[Any, ...]] = {}
    for expected, actual in zip(enriched, report.candidates):
        if actual.parent_asin != expected.parent_asin:
            raise FeatureBuildError(
                "the preference matcher changed the candidate identity or order"
            )
        by_candidate[actual.parent_asin] = tuple(actual.evidence)
    return by_candidate


def _inferred_from_matcher(
    feature: RankerFeature,
    evidence: Any,
) -> InferredPreferenceEvidence:
    """Translate one accepted M10A evidence record into inferred-belief evidence."""
    return InferredPreferenceEvidence(
        belief_id=feature.preference_id,
        kind=feature.kind.value,
        value=feature.value,
        polarity=feature.polarity.value,
        direction=feature.direction.value if feature.direction is not None else None,
        scope=feature.scope.value,
        lifecycle=feature.lifecycle.value,
        strength_rank=feature.strength_rank,
        status=EvidenceState(evidence.status.value),
        reason_code=evidence.reason_code.value,
        metadata_field=_bounded(evidence.metadata_field, _MAX_METADATA_FIELD),
        metadata_value=_bounded(evidence.metadata_value, _MAX_METADATA_VALUE),
        ordinal_value=None,
        better_than_count=0,
        worse_than_count=0,
        detail=_bounded(evidence.detail, _MAX_DETAIL),
    )


def _is_directional(feature: RankerFeature) -> bool:
    """True when a soft belief asks for a grounded ordinal numeric comparison."""
    return (
        feature.direction is not None
        and feature.direction.value in DIRECTIONAL_DIRECTIONS
    )


def _grounding_field(feature: RankerFeature) -> str | None:
    """Return the grounded numeric attribute a directional belief refers to."""
    attribute = feature.value.casefold()
    if "price" in attribute:
        return "price"
    if "weight" in attribute:
        return "weight"
    return None


def _load_grounded_parsers() -> tuple[
    Callable[[Any], float | None], Callable[[Any], float | None]
]:
    """Import the accepted trusted parsers lazily (the control package needs the stack)."""
    from recommendation.control.grounded_reasoning import parse_price, parse_weight_kg

    return parse_price, parse_weight_kg


def _directional_evidence(
    feature: RankerFeature,
    candidates: Sequence[CandidateRecord],
) -> tuple[InferredPreferenceEvidence, ...]:
    """Ground one directional soft belief and compute set-relative ordinal counts."""
    grounding = _grounding_field(feature)
    if grounding is None:
        detail = "no grounded numeric catalogue field exists for this inferred attribute"
        return tuple(
            _directional_record(
                feature,
                status=EvidenceState.UNKNOWN,
                reason_code="unsupported_feature",
                metadata_field=None,
                metadata_value=None,
                ordinal_value=None,
                better_than_count=0,
                worse_than_count=0,
                detail=detail,
            )
            for _ in candidates
        )

    parse_price, parse_weight_kg = _load_grounded_parsers()
    values: list[float | None] = []
    sources: list[tuple[str, str] | None] = []
    for candidate in candidates:
        record = candidate.metadata
        if grounding == "price":
            raw = record.price_text if record is not None else None
            field = "price_text"
            value = parse_price(raw) if raw is not None else None
        else:
            location = _find_weight_detail(record) if record is not None else None
            if location is not None:
                detail_key, raw = location
                field = f"details.{detail_key}"
                value = parse_weight_kg(raw)
            else:
                raw = None
                field = "details.*weight*"
                value = None
        values.append(value if value is not None and math.isfinite(value) else None)
        sources.append(None if value is None else (field, raw))

    known = [(index, value) for index, value in enumerate(values) if value is not None]
    direction = feature.direction.value if feature.direction is not None else "minimize"
    better_counts: dict[int, int] = {}
    worse_counts: dict[int, int] = {}
    for index, value in known:
        better = 0
        worse = 0
        for other_index, other_value in known:
            if other_index == index:
                continue
            if (other_value < value) if direction == "minimize" else (other_value > value):
                better += 1
            elif (other_value > value) if direction == "minimize" else (other_value < value):
                worse += 1
        better_counts[index] = better
        worse_counts[index] = worse

    records: list[InferredPreferenceEvidence] = []
    for index, candidate in enumerate(candidates):
        value = values[index]
        source = sources[index]
        if value is not None and source is not None:
            records.append(
                _directional_record(
                    feature,
                    status=EvidenceState.MATCH,
                    reason_code="grounded_directional_value",
                    metadata_field=source[0],
                    metadata_value=source[1],
                    ordinal_value=value,
                    better_than_count=better_counts.get(index, 0),
                    worse_than_count=worse_counts.get(index, 0),
                    detail=f"grounded {direction} value from the candidate catalogue",
                )
            )
            continue
        if candidate.metadata is None:
            detail = "the candidate has no catalogue metadata record"
        elif source is not None:
            detail = "the catalogue value cannot be parsed as a comparable number"
        elif grounding == "price":
            detail = "the candidate metadata carries no price"
        else:
            detail = "no catalogue detail key carries a weight"
        records.append(
            _directional_record(
                feature,
                status=EvidenceState.UNKNOWN,
                reason_code="insufficient_metadata",
                metadata_field=source[0] if source is not None else (None if grounding == "weight" else "price_text"),
                metadata_value=source[1] if source is not None else None,
                ordinal_value=None,
                better_than_count=0,
                worse_than_count=0,
                detail=detail,
            )
        )
    return tuple(records)


def _explicit_numeric_record(
    feature: RankerFeature,
    candidate: CandidateRecord,
    *,
    logical_seq: int,
    status: EvidenceStatus,
    reason_code: ReasonCode,
    metadata_field: str | None,
    metadata_value: str | None,
    support: MatchingSupport,
    detail: str,
) -> PreferenceEvidence:
    """Build one explicit numeric evidence record with full provenance."""
    return PreferenceEvidence(
        preference_id=feature.preference_id,
        preference_kind=MemoryPreferenceKind(feature.kind.value),
        preference_polarity=MemoryPreferencePolarity(feature.polarity.value),
        preference_value=feature.value,
        preference_source_text=_bounded(
            f"decision-state:{feature.preference_id}", _MAX_DETAIL
        )
        or "decision-state",
        preference_source_turn_id="decision_state",
        preference_logical_seq=logical_seq,
        status=status,
        reason_code=reason_code,
        support=support,
        metadata_field=_bounded(metadata_field, _MAX_METADATA_FIELD),
        metadata_value=_bounded(metadata_value, _MAX_METADATA_VALUE),
        metadata_present=candidate.metadata is not None,
        detail=_bounded(detail, _MAX_DETAIL),
    )


def _comparable_category(record: ProductMetadata | None) -> tuple[str, ...] | None:
    """Return the trusted full category path, or ``None`` when it is not recorded.

    Only the source's own ordered ``categories`` tuple is used.  A broad fallback such as
    ``main_category`` is deliberately NOT used: it cannot distinguish a water filter from
    a sleeping pad, and guessing a product type is forbidden.
    """
    if record is None:
        return None
    path = tuple(str(item).strip() for item in record.categories if str(item).strip())
    return path or None


def _numeric_explicit_evidence(
    feature: RankerFeature,
    candidates: Sequence[CandidateRecord],
    logical_seq: int,
) -> tuple[tuple[PreferenceEvidence, ...], ...]:
    """Ground an EXPLICIT numeric direction and compare only same-category candidates.

    Returns one evidence tuple per candidate (in snapshot order).  A candidate is only
    compared with candidates that share its exact trusted category path; with no trusted
    category, no same-category peer, or no readable value, the evidence is ``UNKNOWN``
    (neutral), never a fabricated match/violation.  Pairwise ordinal degree is represented
    as repeated MATCH/VIOLATION records so the accepted M10B explicit key preserves the
    full same-category order.
    """
    grounding = _grounding_field(feature)
    direction = (
        feature.direction.value if feature.direction is not None else "minimize"
    )
    if grounding is None:
        unknown = tuple(
            _explicit_numeric_record(
                feature,
                candidate,
                logical_seq=logical_seq,
                status=EvidenceStatus.UNKNOWN,
                reason_code=ReasonCode.UNSUPPORTED_PREFERENCE_KIND,
                metadata_field=None,
                metadata_value=None,
                support=MatchingSupport.UNSUPPORTED_FOR_MATCHING,
                detail="no trusted numeric catalogue field can ground this explicit attribute",
            )
            for candidate in candidates
        )
        return tuple((record,) for record in unknown)
    parse_price, parse_weight_kg = _load_grounded_parsers()
    values: list[float | None] = []
    sources: list[tuple[str, str] | None] = []
    categories: list[tuple[str, ...] | None] = []
    for candidate in candidates:
        record = candidate.metadata
        if grounding == "price":
            raw = record.price_text if record is not None else None
            field = "price_text"
            value = parse_price(raw) if raw is not None else None
        else:
            location = _find_weight_detail(record) if record is not None else None
            if location is not None:
                detail_key, raw = location
                field = f"details.{detail_key}"
                value = parse_weight_kg(raw)
            else:
                raw = None
                field = "details.*weight*"
                value = None
        values.append(value if value is not None and math.isfinite(value) else None)
        sources.append(None if value is None else (field, raw))
        categories.append(_comparable_category(record))

    grouped: dict[tuple[str, ...], list[tuple[int, float]]] = {}
    for index, (value, category) in enumerate(zip(values, categories)):
        if value is not None and category is not None:
            grouped.setdefault(category, []).append((index, value))

    by_candidate: list[list[PreferenceEvidence]] = [[] for _ in candidates]
    for index, candidate in enumerate(candidates):
        value = values[index]
        source = sources[index]
        category = categories[index]
        field = source[0] if source is not None else (
            "price_text" if grounding == "price" else "details.*weight*"
        )
        if value is None:
            reason = (
                ReasonCode.METADATA_MISSING
                if candidate.metadata is None
                else ReasonCode.INSUFFICIENT_METADATA
            )
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.UNKNOWN,
                    reason_code=reason,
                    metadata_field=field,
                    metadata_value=None,
                    support=MatchingSupport.SUPPORTED,
                    detail=f"no readable trusted {grounding} value; UNKNOWN is neutral",
                )
            )
            continue
        if category is None:
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.UNKNOWN,
                    reason_code=ReasonCode.NUMERIC_NOT_COMPARABLE,
                    metadata_field=field,
                    metadata_value=source[1] if source is not None else None,
                    support=MatchingSupport.SUPPORTED,
                    detail="no trusted category path; numeric preference is not compared",
                )
            )
            continue
        peers = [
            peer_value
            for peer_index, peer_value in grouped.get(category, ())
            if peer_index != index
        ]
        if not peers:
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.UNKNOWN,
                    reason_code=ReasonCode.NUMERIC_NOT_COMPARABLE,
                    metadata_field=field,
                    metadata_value=source[1] if source is not None else None,
                    support=MatchingSupport.SUPPORTED,
                    detail="no second grounded candidate in the same trusted category",
                )
            )
            continue
        if direction == "minimize":
            better = sum(1 for peer in peers if peer < value)
            worse = sum(1 for peer in peers if peer > value)
        else:
            better = sum(1 for peer in peers if peer > value)
            worse = sum(1 for peer in peers if peer < value)
        if better == 0 and worse == 0:
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.UNKNOWN,
                    reason_code=ReasonCode.NUMERIC_NOT_COMPARABLE,
                    metadata_field=field,
                    metadata_value=source[1] if source is not None else None,
                    support=MatchingSupport.SUPPORTED,
                    detail="all same-category grounded values are tied; neutral",
                )
            )
            continue
        for _ in range(better):
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.VIOLATION,
                    reason_code=ReasonCode.NUMERIC_WORSE_DIRECTION,
                    metadata_field=field,
                    metadata_value=source[1] if source is not None else None,
                    support=MatchingSupport.SUPPORTED,
                    detail=(
                        "a same-category candidate has a strictly better grounded "
                        f"{grounding} value under {direction}"
                    ),
                )
            )
        for _ in range(worse):
            by_candidate[index].append(
                _explicit_numeric_record(
                    feature,
                    candidate,
                    logical_seq=logical_seq,
                    status=EvidenceStatus.MATCH,
                    reason_code=ReasonCode.NUMERIC_PREFERRED_DIRECTION,
                    metadata_field=field,
                    metadata_value=source[1] if source is not None else None,
                    support=MatchingSupport.SUPPORTED,
                    detail=(
                        "this candidate has a strictly worse same-category grounded "
                        f"{grounding} value under {direction}"
                    ),
                )
            )
    return tuple(tuple(records) for records in by_candidate)


def _directional_record(
    feature: RankerFeature,
    *,
    status: EvidenceState,
    reason_code: str,
    metadata_field: str | None,
    metadata_value: str | None,
    ordinal_value: float | None,
    better_than_count: int,
    worse_than_count: int,
    detail: str,
) -> InferredPreferenceEvidence:
    """Assemble one directional inferred-evidence record."""
    return InferredPreferenceEvidence(
        belief_id=feature.preference_id,
        kind=feature.kind.value,
        value=feature.value,
        polarity=feature.polarity.value,
        direction=feature.direction.value if feature.direction is not None else None,
        scope=feature.scope.value,
        lifecycle=feature.lifecycle.value,
        strength_rank=feature.strength_rank,
        status=status,
        reason_code=reason_code,
        metadata_field=_bounded(metadata_field, _MAX_METADATA_FIELD),
        metadata_value=_bounded(metadata_value, _MAX_METADATA_VALUE),
        ordinal_value=ordinal_value,
        better_than_count=better_than_count,
        worse_than_count=worse_than_count,
        detail=_bounded(detail, _MAX_DETAIL),
    )


def _inferred_feature_coverage(
    soft_features: Sequence[RankerFeature],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify inferred features as catalogue-supported or unsupported."""
    supported: list[str] = []
    unsupported: list[str] = []
    for feature in soft_features:
        label = f"{feature.kind.value}:{feature.value}"
        if _is_directional(feature):
            direction = feature.direction.value if feature.direction is not None else ""
            label = f"{label} ({direction})"
            (supported if _grounding_field(feature) is not None else unsupported).append(label)
            continue
        label = f"{label} (text)"
        support = matching_support_for(MemoryPreferenceKind(feature.kind.value))
        (supported if support is not MatchingSupport.UNSUPPORTED_FOR_MATCHING else unsupported).append(
            label
        )
    return tuple(supported), tuple(unsupported)


def _eligibility_note(candidates: Sequence[CandidateRecord]) -> str:
    """Deterministic summary of the shared eligibility projection."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.eligibility.value] = counts.get(candidate.eligibility.value, 0) + 1
    order = ("verified_eligible", "ineligible", "unresolved", "not_assessed")
    parts = [f"{counts[name]} {name}" for name in order if counts.get(name)]
    return "eligibility projection: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


class BeliefAwareFeatureBuilder:
    """Stateless, deterministic builder of typed belief-aware candidate features.

    The builder holds only the accepted stateless M10A matcher, so one instance can be
    reused; repeated builds of the same snapshot/projection are equal and produce an
    equal canonical JSON digest.
    """

    #: Version of the feature-building semantics (bumped with a real behaviour change).
    #: 2: explicit directional numeric preferences are grounded against same-category
    #: catalogue values instead of being text-matched (or downgraded to inferred).
    FEATURE_BUILD_VERSION = 2

    def __init__(self) -> None:
        self._matcher = PreferenceCandidateMatcher()

    # -- public API ------------------------------------------------------- #

    def build(
        self,
        *,
        snapshot: CandidateSnapshot,
        ranker_projection: RankerStateProjection,
    ) -> CandidateFeatureSet:
        """Build the complete feature set for one frozen snapshot/projection pair.

        Labels are deliberately not a parameter: ground-truth labels live on the
        evaluation task only, so they cannot leak into feature construction.
        """
        self._validate_inputs(snapshot, ranker_projection)
        candidates = snapshot.candidates
        explicit_features, soft_features = _projection_feature_split(ranker_projection)
        numeric_explicit_features = tuple(
            feature for feature in explicit_features if _is_directional(feature)
        )
        textual_explicit_features = tuple(
            feature for feature in explicit_features if not _is_directional(feature)
        )

        explicit_entries = _explicit_entries(textual_explicit_features)
        non_directional_features = tuple(
            feature for feature in soft_features if not _is_directional(feature)
        )
        soft_entries = _soft_entries(non_directional_features)

        enriched = _enriched_candidates(candidates)
        explicit_by_candidate = _evidence_by_candidate(
            self._matcher.match(candidates=enriched, preferences=explicit_entries),
            enriched,
        )
        # Numeric explicit directions never go through textual M10A matching.  They are
        # grounded from the trusted catalogue value and compared only with candidates in
        # the same trusted category; the resulting counts feed the accepted M10B key.
        for position, feature in enumerate(numeric_explicit_features, start=1):
            numeric_evidence = _numeric_explicit_evidence(feature, candidates, position)
            for candidate, evidence_tuple in zip(candidates, numeric_evidence):
                explicit_by_candidate[candidate.candidate_id] = (
                    *explicit_by_candidate.get(candidate.candidate_id, ()),
                    *evidence_tuple,
                )
        non_directional_by_feature = self._non_directional_inferred(
            enriched, non_directional_features, soft_entries
        )

        inferred_by_candidate: dict[str, list[InferredPreferenceEvidence]] = {
            candidate.candidate_id: [] for candidate in candidates
        }
        for feature in soft_features:
            if _is_directional(feature):
                for candidate, record in zip(
                    candidates, _directional_evidence(feature, candidates)
                ):
                    inferred_by_candidate[candidate.candidate_id].append(record)
                continue
            by_candidate = non_directional_by_feature.get(feature.preference_id)
            if by_candidate is None:
                raise FeatureBuildError(
                    "a soft inference feature is missing from the matcher projection"
                )
            for candidate in candidates:
                evidence = by_candidate[candidate.candidate_id]
                if evidence is None:
                    raise FeatureBuildError(
                        "the preference matcher produced no inferred evidence for a candidate"
                    )
                inferred_by_candidate[candidate.candidate_id].append(
                    _inferred_from_matcher(feature, evidence)
                )

        strength_metadata = tuple(
            PreferenceStrengthMetadata(
                belief_id=feature.preference_id,
                kind=feature.kind.value,
                value=feature.value,
                polarity=feature.polarity.value,
                direction=feature.direction.value if feature.direction is not None else None,
                scope=feature.scope.value,
                lifecycle=feature.lifecycle.value,
                strength_rank=feature.strength_rank,
                is_soft=feature.is_soft,
            )
            for feature in ranker_projection.features
        )

        features: list[CandidateRankingFeatures] = []
        for candidate in candidates:
            explicit_evidence = explicit_by_candidate.get(candidate.candidate_id, ())
            inferred_evidence = tuple(
                inferred_by_candidate.get(candidate.candidate_id, ())
            )
            features.append(
                CandidateRankingFeatures(
                    candidate_id=candidate.candidate_id,
                    item_id=candidate.item_id,
                    original_rank=candidate.original_rank,
                    source_ranks=tuple(candidate.source_ranks),
                    rrf_score=candidate.rrf_score,
                    rrf_score_kind=candidate.rrf_score_kind,
                    explicit_matches=_count_status(
                        explicit_evidence, EvidenceStatus.MATCH
                    ),
                    explicit_violations=_count_status(
                        explicit_evidence, EvidenceStatus.VIOLATION
                    ),
                    explicit_unknown=_count_status(
                        explicit_evidence, EvidenceStatus.UNKNOWN
                    ),
                    inferred_matches=_count_evidence_state(
                        inferred_evidence, EvidenceState.MATCH
                    ),
                    inferred_violations=_count_evidence_state(
                        inferred_evidence, EvidenceState.VIOLATION
                    ),
                    inferred_unknown=_count_evidence_state(
                        inferred_evidence, EvidenceState.UNKNOWN
                    ),
                    # support = peers this candidate beats, penalty = peers that beat it;
                    # A2 sorts (penalty ASC, -support ASC), so a better grounded value
                    # ranks earlier and UNKNOWN (0/0) is exactly neutral.
                    inferred_ordinal_support=sum(
                        record.worse_than_count for record in inferred_evidence
                    ),
                    inferred_ordinal_penalty=sum(
                        record.better_than_count for record in inferred_evidence
                    ),
                    explicit_evidence=tuple(explicit_evidence),
                    inferred_evidence=inferred_evidence,
                    preference_strength_metadata=strength_metadata,
                    eligibility=candidate.eligibility,
                    eligibility_detail=candidate.eligibility_detail,
                    catalogue_fields=_catalogue_fields(candidate.metadata),
                )
            )

        supported, unsupported = _inferred_feature_coverage(soft_features)
        coverage = CatalogueCoverage(
            candidate_count=len(candidates),
            metadata_found_count=sum(
                1 for candidate in candidates if candidate.metadata_status == "found"
            ),
            metadata_missing_count=sum(
                1 for candidate in candidates if candidate.metadata_status != "found"
            ),
            readable_fields=_readable_field_counts(candidates),
            unsupported_inferred_features=unsupported,
            supported_inferred_features=supported,
            notes=self._coverage_notes(
                candidates=candidates,
                soft_features=soft_features,
                unsupported=unsupported,
            ),
        )

        return CandidateFeatureSet(
            task_id=snapshot.task_id,
            user_key=snapshot.user_key,
            turn_key=snapshot.turn_key,
            snapshot_id=snapshot.snapshot_id,
            snapshot_candidate_count=snapshot.candidate_count,
            candidate_pool_digest=snapshot.candidate_pool_digest_value(),
            ranking_digest=snapshot.ranking_digest_value(),
            features=tuple(features),
            coverage=coverage,
            ranker_projection=ranker_projection,
        )

    # -- internals -------------------------------------------------------- #

    def _validate_inputs(
        self,
        snapshot: CandidateSnapshot,
        projection: RankerStateProjection,
    ) -> None:
        """Reject inconsistent trusted inputs before any evidence is produced."""
        if not isinstance(snapshot, CandidateSnapshot):
            raise FeatureBuildError("build requires a CandidateSnapshot")
        if not isinstance(projection, RankerStateProjection):
            raise FeatureBuildError("build requires a RankerStateProjection")
        if projection.task_id != snapshot.task_id:
            raise FeatureBuildError(
                "the ranker projection belongs to a different task than the candidate snapshot"
            )
        if (
            snapshot.candidate_pool_ref is not None
            and projection.candidate_pool_ref is not None
            and snapshot.candidate_pool_ref != projection.candidate_pool_ref
        ):
            raise FeatureBuildError(
                "the ranker projection refers to a different candidate pool than the snapshot"
            )
        if (
            snapshot.ranking_snapshot_ref is not None
            and projection.ranking_snapshot_ref is not None
            and snapshot.ranking_snapshot_ref != projection.ranking_snapshot_ref
        ):
            raise FeatureBuildError(
                "the ranker projection refers to a different ranking snapshot than the snapshot"
            )
        seen_preference_ids: set[str] = set()
        for feature in projection.features:
            if feature.preference_id in seen_preference_ids:
                raise FeatureBuildError(
                    "the ranker projection contains duplicate preference ids"
                )
            seen_preference_ids.add(feature.preference_id)
        for candidate in snapshot.candidates:
            if candidate.metadata is not None and candidate.metadata.parent_asin != candidate.candidate_id:
                raise FeatureBuildError(
                    "candidate metadata belongs to a different parent_asin than the candidate"
                )

    def _non_directional_inferred(
        self,
        enriched: Sequence[Any],
        soft_features: Sequence[RankerFeature],
        soft_entries: Sequence[PreferenceMemoryEntry],
    ) -> dict[str, dict[str, Any]]:
        """Run the real matcher for non-directional soft beliefs, keyed per feature.

        Returns ``{preference_id: {candidate_id: PreferenceEvidence}}`` preserving the
        accepted M10A evidence unchanged.  No violation is invented here: the matcher's
        UNKNOWN/absence semantics are simply carried through.
        """
        result: dict[str, dict[str, Any]] = {}
        if not soft_entries:
            return result
        if len(soft_entries) != len(soft_features):
            raise FeatureBuildError(
                "the soft belief projection is inconsistent with its matcher entries"
            )
        report = self._matcher.match(candidates=enriched, preferences=soft_entries)
        if len(report.candidates) != len(enriched):
            raise FeatureBuildError(
                "the preference matcher did not return exactly the supplied candidates"
            )
        feature_by_memory_id = {
            entry.memory_id: feature.preference_id
            for entry, feature in zip(soft_entries, soft_features)
        }
        for expected, actual in zip(enriched, report.candidates):
            if actual.parent_asin != expected.parent_asin:
                raise FeatureBuildError(
                    "the preference matcher changed the candidate identity or order"
                )
            if len(actual.evidence) != len(soft_entries):
                raise FeatureBuildError(
                    "the preference matcher did not return one inferred evidence record per belief"
                )
            for evidence in actual.evidence:
                preference_id = feature_by_memory_id.get(evidence.preference_id)
                if preference_id is None:
                    raise FeatureBuildError(
                        "the preference matcher returned evidence for an unknown belief"
                    )
                result.setdefault(preference_id, {})[actual.parent_asin] = evidence
        return result

    def _coverage_notes(
        self,
        *,
        candidates: Sequence[CandidateRecord],
        soft_features: Sequence[RankerFeature],
        unsupported: Sequence[str],
    ) -> tuple[str, ...]:
        """Deterministic coverage notes (diagnostics, never quality evidence)."""
        notes: list[str] = []
        missing = sum(1 for candidate in candidates if candidate.metadata_status != "found")
        if missing:
            notes.append(
                f"{missing} candidate(s) have no catalogue metadata record; their "
                "attributes are UNKNOWN and never become violations"
            )
        if any(_is_directional(feature) for feature in soft_features):
            notes.append(
                "directional inferred evidence is a set-relative ordinal comparison, "
                "not a probability and not a cross-model score"
            )
            notes.append(
                "catalogue weight readability uses a structural check; grounded "
                "directional evidence uses the accepted parse_price/parse_weight_kg helpers"
            )
        if unsupported:
            notes.append(
                f"{len(unsupported)} inferred feature(s) have no grounded catalogue "
                "field and contribute no ranking signal"
            )
        notes.append(_eligibility_note(candidates))
        notes.append(
            "ordinal belief strength is a 1-3 ordering rank, never a calibrated probability"
        )
        return tuple(notes)


def _count_status(evidence: Sequence[Any], status: EvidenceStatus) -> int:
    """Count accepted M10A evidence records by status."""
    return sum(1 for record in evidence if record.status is status)


def _count_evidence_state(
    evidence: Sequence[InferredPreferenceEvidence], state: EvidenceState
) -> int:
    """Count inferred evidence records by status."""
    return sum(1 for record in evidence if record.status is state)


def _readable_field_counts(
    candidates: Sequence[CandidateRecord],
) -> dict[str, int]:
    """Count candidates with a readable value for each catalogue attribute."""
    counts = {field: 0 for field in CATALOGUE_FIELD_ORDER}
    for candidate in candidates:
        fields = _catalogue_fields(candidate.metadata)
        for field in fields.present:
            counts[field] += 1
    return counts
