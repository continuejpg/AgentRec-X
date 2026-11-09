"""Deterministic synthetic M6.1 ranking tasks (offline lab fixture).

The scenario is the milestone's hiking-backpack task: a hard budget of at most **$150**,
an explicit ``comfort`` preference, an inferred ``lightweight`` hypothesis, candidates
with real catalogue metadata and candidates whose attributes are UNKNOWN (missing price
token, missing weight, or no catalogue record at all).

Everything here is deterministic and dependency-free: fixed ids, fixed ranks, fixed
event timestamps, no wall-clock, no random numbers, no network, no database.  The
objects flowing through the pieces are the **real** accepted types:

* candidates are real :class:`~recommendation.ranking.schemas.CandidateSnapshot` records
  built by the real :class:`~recommendation.ranking.resolver.TrustedCandidateFeatureResolver`
  over a real :class:`recommendation.catalog.MetadataIndex` of
  ``normalize_product_record`` output;
* the ``for_ranker`` projection is a real
  :class:`~recommendation.state.state_projector.RankerStateProjection` produced by
  replaying real M5 events through the real
  :class:`~recommendation.state.BeliefUpdater`;
* explicit/inferred evidence is produced by the real M10A
  :class:`~recommendation.preference_matching.PreferenceCandidateMatcher` (inside the
  feature builder), not by this fixture.

Variants
--------
``build_task(variant=...)`` builds one frozen task per scenario:

``default``
    Explicit ``comfort`` FEATURE (task scope), inferred ``lightweight`` hypothesis
    (soft, weak, task scope), a long-term explicit ``TrailForge`` brand and long-term
    ``green`` colour preference, and a task-scoped ``charcoal`` colour statement.  The
    two colour beliefs exercise scope semantics: the task statement does not silently
    override the long-term one.
``retracted``
    ``default`` plus ``PREFERENCE_RETRACTED`` for the inferred ``lightweight``
    hypothesis.  The projection then contains no soft feature at all, so A2's inferred
    tie-break is empty and A2 must equal A1.
``corrected``
    ``default`` plus an explicit correction statement that supersedes ``comfort`` and
    adds ``waterproof``.  The old value must be inactive and the new value active.
``directional``
    ``default`` plus projection-level **soft directional** inferred beliefs: ``weight``
    minimize (grounded from the catalogue weight detail) and ``durability`` minimize
    (no grounded catalogue field, so it must be reported as unsupported).  M5's
    ``PreferenceHypothesisProposed`` payload has no ``direction`` field, so a directional
    *inferred* belief cannot currently be expressed through the event vocabulary; the
    fixture composes it onto the real projection and documents that gap explicitly.  This
    is the variant that exercises the ordinal inferred tie-break: candidates #2 and #5 tie
    on explicit evidence, #5 is lighter (grounded weight prefers it) and #5 is later in the
    trusted/A1 order, so A2 must reorder that pair while UNKNOWN-weight candidates stay
    neutral (``0/0``).

No task carries labels: M6.1 has no preference-conditioned relevance labels, so quality
metrics must stay ``NOT_EXECUTED_NO_LABELS``.

Eligibility decisions
---------------------
The eligibility projection attached to the snapshot is the kind of map the trusted
constraint evaluator produces for the budget ``price_max<=150``: candidates priced above
the budget are ``INELIGIBLE``, candidates whose price cannot be read (unknown token or
no metadata record) are ``UNRESOLVED``, and readable in-budget candidates are
``VERIFIED_ELIGIBLE``.  The fixture does not import the evaluator (that package pulls the
model stack); it only supplies the projection, exactly as the trusted layer would.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import (  # noqa: E402
    MetadataIndex,
    normalize_product_record,
)
from recommendation.catalog.schemas import ProductMetadata  # noqa: E402
from recommendation.ranking.resolver import (  # noqa: E402
    TrustedCandidateFeatureResolver,
    TrustedCandidateInput,
)
from recommendation.ranking.schemas import (  # noqa: E402
    CandidateSnapshot,
    EligibilityState,
    FrozenRankingTask,
    SourceRank,
)
from recommendation.state.belief_updater import BeliefUpdater  # noqa: E402
from recommendation.state.decision_state import (  # noqa: E402
    UnifiedDecisionState,
    empty_state,
)
from recommendation.state.events import EventActor, new_event  # noqa: E402
from recommendation.state.schemas import (  # noqa: E402
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    candidate_pool_snapshot,
    ranking_snapshot,
)
from recommendation.state.state_projector import (  # noqa: E402
    RankerFeature,
    RankerStateProjection,
    for_ranker,
)

__all__ = [
    "CANDIDATE_IDS",
    "CANDIDATE_ROWS",
    "DIRECTIONAL_BELIEF_VALUES",
    "PRICE_LIMIT",
    "SESSION_ID",
    "SNAPSHOT_SCHEMA_VERSION",
    "TASK_ID",
    "TURN_KEY",
    "USER_KEY",
    "VARIANT_NAMES",
    "FixtureIdentityMap",
    "build_candidate_inputs",
    "build_eligibility_decisions",
    "build_identity_map",
    "build_metadata_index",
    "build_snapshot",
    "build_state",
    "build_task",
    "build_tasks",
    "build_variants",
    "replay_projection",
    "replay_task",
    "variant_session_id",
    "variant_snapshot_id",
    "variant_task_id",
    "variant_turn_key",
]


# --------------------------------------------------------------------------- #
# Fixed identities and scenario constants
# --------------------------------------------------------------------------- #

#: Owning synthetic user/session/task identities.  All deterministically derived later.
USER_KEY = "m6-1-fixture-user"
SESSION_ID = "m6-1-fixture-session"
TASK_ID = "m6-1-hiking-backpack"
TURN_KEY = "m6-1-hiking-backpack:turn-1"
SNAPSHOT_SCHEMA_VERSION = 1

#: The user's hard budget.  Applied once, before every arm.
PRICE_LIMIT = 150

#: Projection-level soft directional beliefs used by the ``directional`` variant.
DIRECTIONAL_BELIEF_VALUES: tuple[str, ...] = ("weight", "durability")

VARIANT_NAMES: tuple[str, ...] = ("default", "retracted", "corrected", "directional")


#: Candidate rows, in trusted original order (the frozen snapshot order / RRF order).
#:
#: ``metadata`` is the raw source-shaped record handed to ``normalize_product_record``
#: (``None`` means "the catalogue holds no record for this candidate").  Eligibility is
#: the trusted budget projection: over-budget -> INELIGIBLE, unreadable price -> UNRESOLVED.
CANDIDATE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "parent_asin": "B0M6HIKE01",
        "item_id": 101,
        "rrf_score": 0.0321,
        "sasrec_rank": 2,
        "sasrec_score": 5.5,
        "catalog_rank": 5,
        "catalog_score": 12.4,
        "eligibility": EligibilityState.VERIFIED_ELIGIBLE,
        "eligibility_detail": "price 129.99 is within the $150 budget",
        "metadata": {
            "title": "Ultralight Comfort Hiking Backpack 40L",
            "store": "TrailForge",
            "price": 129.99,
            "features": ("comfort harness", "waterproof coating", "lightweight frame"),
            "categories": ("Sports & Outdoors", "Camping & Hiking"),
            "details": {"Color": "charcoal", "Material": "nylon", "Item Weight": "1.2 Kilograms"},
        },
    },
    {
        "parent_asin": "B0M6HIKE02",
        "item_id": 102,
        "rrf_score": 0.0298,
        "sasrec_rank": 4,
        "sasrec_score": 5.1,
        "catalog_rank": 2,
        "catalog_score": 13.7,
        "eligibility": EligibilityState.VERIFIED_ELIGIBLE,
        "eligibility_detail": "price 89.0 is within the $150 budget",
        "metadata": {
            "title": "Trail Comfort Backpack 35L",
            "store": "SummitGear",
            "price": 89.0,
            "features": ("comfort back panel", "water resistant"),
            "categories": ("Sports & Outdoors", "Camping & Hiking"),
            "details": {"Color": "green", "Material": "polyester", "Item Weight": "2.6 Kilograms"},
        },
    },
    {
        "parent_asin": "B0M6HIKE03",
        "item_id": 103,
        "rrf_score": 0.0274,
        "sasrec_rank": 1,
        "sasrec_score": 5.8,
        "catalog_rank": 9,
        "catalog_score": 9.8,
        "eligibility": EligibilityState.INELIGIBLE,
        "eligibility_detail": "price 210.0 exceeds the $150 budget",
        "metadata": {
            "title": "Featherlight Summit Pack 50L",
            "store": "PeakLine",
            "price": 210.0,
            "features": ("lightweight frame", "waterproof coating"),
            "categories": ("Sports & Outdoors", "Camping & Hiking"),
            "details": {"Color": "black", "Material": "nylon", "Item Weight": "1.0 Kilograms"},
        },
    },
    {
        "parent_asin": "B0M6HIKE04",
        "item_id": 104,
        "rrf_score": 0.0251,
        "sasrec_rank": 7,
        "sasrec_score": 4.4,
        "catalog_rank": 1,
        "catalog_score": 14.9,
        "eligibility": EligibilityState.UNRESOLVED,
        "eligibility_detail": "price is unknown, so the budget cannot be verified",
        "metadata": {
            "title": "Mystery Ridge Pack",
            "store": "AlpineWorks",
            "price": "\u2014",
            "features": ("aluminium frame",),
            "categories": ("Sports & Outdoors",),
            "details": {"Color": "blue", "Material": "nylon"},
        },
    },
    {
        "parent_asin": "B0M6HIKE05",
        "item_id": 105,
        "rrf_score": 0.0229,
        "sasrec_rank": 9,
        "sasrec_score": 3.9,
        "catalog_rank": 4,
        "catalog_score": 12.9,
        "eligibility": EligibilityState.VERIFIED_ELIGIBLE,
        "eligibility_detail": "price 149.0 is within the $150 budget",
        "metadata": {
            "title": "Weekend Hiker Backpack 30L",
            "store": "TrailForge",
            "price": 149.0,
            "features": ("comfort padding", "lightweight frame", "padded straps"),
            "categories": ("Sports & Outdoors", "Camping & Hiking"),
            "details": {"Color": "red", "Item Weight": "1.6 Kilograms"},
        },
    },
    {
        "parent_asin": "B0M6HIKE06",
        "item_id": 106,
        "rrf_score": 0.0207,
        "sasrec_rank": 3,
        "sasrec_score": 5.3,
        "catalog_rank": 12,
        "catalog_score": 8.6,
        "eligibility": EligibilityState.UNRESOLVED,
        "eligibility_detail": "no catalogue metadata: price and weight are unknown",
        "metadata": None,
    },
    {
        "parent_asin": "B0M6HIKE07",
        "item_id": 107,
        "rrf_score": 0.0185,
        "sasrec_rank": 5,
        "sasrec_score": 4.9,
        "catalog_rank": 6,
        "catalog_score": 11.2,
        "eligibility": EligibilityState.INELIGIBLE,
        "eligibility_detail": "price 159.99 exceeds the $150 budget",
        "metadata": {
            "title": "Heavy Hauler Expedition Backpack 55L",
            "store": "CampCo",
            "price": 159.99,
            "features": ("comfort padding", "reinforced base"),
            "categories": ("Sports & Outdoors", "Camping & Hiking"),
            "details": {"Color": "green", "Material": "canvas", "Item Weight": "3.8 Kilograms"},
        },
    },
)

#: Candidate identities in trusted original order.
CANDIDATE_IDS: tuple[str, ...] = tuple(row["parent_asin"] for row in CANDIDATE_ROWS)


# --------------------------------------------------------------------------- #
# Variant identity helpers (all deterministic)
# --------------------------------------------------------------------------- #


def variant_task_id(variant: str) -> str:
    """Task id of one fixture variant (each variant is a separate frozen task)."""
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown fixture variant: {variant!r}")
    return f"{TASK_ID}-{variant}"


def variant_session_id(variant: str) -> str:
    """Session id of one fixture variant."""
    return f"{SESSION_ID}-{variant}"


def variant_turn_key(variant: str) -> str:
    """Turn key of one fixture variant (the same logical turn for every variant)."""
    return f"{variant_task_id(variant)}:turn-1"


def variant_snapshot_id(variant: str) -> str:
    """Snapshot id of one fixture variant."""
    return f"m6-1-snapshot-hiking-backpack-{variant}"


# --------------------------------------------------------------------------- #
# Trusted dependencies: metadata index and identity mapping
# --------------------------------------------------------------------------- #


def _normalised_metadata(row: Mapping[str, Any]) -> ProductMetadata | None:
    """Normalise one row's raw metadata, or return ``None`` when there is none."""
    raw = row["metadata"]
    if raw is None:
        return None
    return normalize_product_record({**raw, "parent_asin": row["parent_asin"]})


def build_metadata_index() -> MetadataIndex:
    """Build the real catalogue index over the fixture's metadata-bearing rows."""
    records = [
        record for record in (_normalised_metadata(row) for row in CANDIDATE_ROWS) if record is not None
    ]
    return MetadataIndex.from_records(records)


class FixtureIdentityMap:
    """Minimal trusted ``parent_asin <-> item_id`` mapping for the fixture.

    Exposes exactly the three methods the resolver requires, mirroring the trusted
    identity mapping the accepted pipeline owns.  It never guesses an identity.
    """

    def __init__(self, pairs: Mapping[str, int]) -> None:
        self._parent_asin_to_item_id = dict(pairs)
        self._item_id_to_parent_asin = {item_id: asin for asin, item_id in pairs.items()}

    def parent_asin_to_item_id(self, parent_asin: str) -> int | None:
        """Return the trusted item id for a ``parent_asin``, or ``None``."""
        return self._parent_asin_to_item_id.get(parent_asin)

    def item_id_to_parent_asin(self, item_id: int) -> str | None:
        """Return the trusted ``parent_asin`` for an item id, or ``None``."""
        return self._item_id_to_parent_asin.get(item_id)

    def has_parent_asin(self, parent_asin: str) -> bool:
        """True when the mapping owns the given ``parent_asin``."""
        return parent_asin in self._parent_asin_to_item_id


def build_identity_map() -> FixtureIdentityMap:
    """Build the trusted identity mapping over every fixture candidate."""
    return FixtureIdentityMap({row["parent_asin"]: row["item_id"] for row in CANDIDATE_ROWS})


def build_trusted_resolver() -> TrustedCandidateFeatureResolver:
    """Build the resolver wired to the fixture's trusted metadata/identity sources."""
    return TrustedCandidateFeatureResolver(
        metadata_index=build_metadata_index(),
        identity_map=build_identity_map(),
    )


# --------------------------------------------------------------------------- #
# Trusted candidate inputs, eligibility projection and snapshot
# --------------------------------------------------------------------------- #


def _source_ranks(row: Mapping[str, Any]) -> tuple[SourceRank, ...]:
    """Synthetic but plausible per-source rank provenance for one candidate."""
    return (
        SourceRank(
            source="history_sasrec",
            rank=int(row["sasrec_rank"]),
            score=float(row["sasrec_score"]),
            score_kind="sasrec_logit",
        ),
        SourceRank(
            source="catalog_search",
            rank=int(row["catalog_rank"]),
            score=float(row["catalog_score"]),
            score_kind="bm25",
        ),
    )


def build_eligibility_decisions() -> dict[str, tuple[EligibilityState, str | None]]:
    """The trusted ``price_max<=150`` eligibility projection for every candidate.

    This is the map the trusted constraint evaluator produces; the fixture only
    supplies it (importing the evaluator package pulls the model stack).  Over-budget
    candidates are INELIGIBLE, unreadable-price candidates UNRESOLVED, the rest
    VERIFIED_ELIGIBLE.
    """
    return {
        row["parent_asin"]: (row["eligibility"], row["eligibility_detail"])
        for row in CANDIDATE_ROWS
    }


def build_candidate_inputs(
    *,
    decisions: Mapping[str, tuple[EligibilityState, str | None]] | None = None,
) -> tuple[TrustedCandidateInput, ...]:
    """Build the trusted candidate rows handed to the resolver, in original order."""
    eligibility = dict(decisions) if decisions is not None else build_eligibility_decisions()
    inputs: list[TrustedCandidateInput] = []
    for position, row in enumerate(CANDIDATE_ROWS, start=1):
        state, detail = eligibility[row["parent_asin"]]
        inputs.append(
            TrustedCandidateInput(
                candidate_id=row["parent_asin"],
                item_id=int(row["item_id"]),
                original_rank=position,
                score=float(row["rrf_score"]),
                score_kind="rrf_rank_fusion",
                source_ranks=_source_ranks(row),
                rrf_score=float(row["rrf_score"]),
                eligibility=state,
                eligibility_detail=detail,
            )
        )
    return tuple(inputs)


def _trusted_refs() -> tuple[Any, Any]:
    """The M5-compatible candidate-pool and ranking references for the fixture set."""
    return (
        candidate_pool_snapshot(CANDIDATE_IDS),
        ranking_snapshot(CANDIDATE_IDS),
    )


def build_snapshot(*, variant: str = "default") -> CandidateSnapshot:
    """Build the frozen trusted snapshot for one variant."""
    pool, ranking = _trusted_refs()
    return build_trusted_resolver().build_snapshot(
        snapshot_id=variant_snapshot_id(variant),
        user_key=USER_KEY,
        task_id=variant_task_id(variant),
        turn_key=variant_turn_key(variant),
        candidates=build_candidate_inputs(),
        candidate_pool_ref=pool.ref,
        ranking_snapshot_ref=ranking.ref,
    )


# --------------------------------------------------------------------------- #
# M5 event stream and replays
# --------------------------------------------------------------------------- #

#: Fixed event time: replay determinism must not depend on a wall clock.
_EVENT_TS = 1_700_100_000.0
_EVENT_SOURCE = "m6_1_fixture"


def _event_id(variant: str, number: int) -> str:
    """Deterministic event id for one variant's event stream."""
    return f"m6-1-{variant}-evt-{number:02d}"


def _event(
    *,
    variant: str,
    number: int,
    event_type: str,
    actor: EventActor,
    payload: Mapping[str, Any],
) -> Any:
    """Build one validated real M5 event scoped to the variant's task."""
    task_id = variant_task_id(variant)
    return new_event(
        event_type=event_type,
        event_id=_event_id(variant, number),
        user_key=USER_KEY,
        task_id=task_id,
        session_id=variant_session_id(variant),
        actor=actor,
        source=_EVENT_SOURCE,
        timestamp=_EVENT_TS,
        payload=dict(payload),
    )


def _scenario_events(*, variant: str) -> list[Any]:
    """The deterministic event stream shared by every variant.

    The stream states the task context, the hard budget, the explicit preferences, one
    inferred hypothesis and the trusted candidate/ranking snapshot references.  Variant
    additions are appended by :func:`build_state`.
    """
    pool, ranking = _trusted_refs()
    events: list[Any] = [
        _event(
            variant=variant,
            number=1,
            event_type="task_context_updated",
            actor=EventActor.AGENT,
            payload={
                "intent": "recommend",
                "request_text": (
                    "I need a lightweight hiking backpack for weekend trips; comfort "
                    "matters and my budget is at most $150."
                ),
            },
        ),
        _event(
            variant=variant,
            number=2,
            event_type="user_constraint_stated",
            actor=EventActor.USER,
            payload={
                "kind": "price_max",
                "value": "150",
                "polarity": "prefer",
                "scope": "task",
                "source_text": "My budget is at most $150.",
            },
        ),
        _event(
            variant=variant,
            number=3,
            event_type="user_preference_stated",
            actor=EventActor.USER,
            payload={
                "kind": "feature",
                "value": "comfort",
                "polarity": "prefer",
                "scope": "task",
                "source_text": "I want it to be comfortable.",
            },
        ),
        _event(
            variant=variant,
            number=4,
            event_type="preference_hypothesis_proposed",
            actor=EventActor.AGENT,
            payload={
                "kind": "feature",
                "value": "lightweight",
                "polarity": "prefer",
                "scope": "task",
                "strength": "weak",
                "basis": "browsing_history",
                "reason": "recent browsing skewed toward lightweight packs",
                "evidence_refs": ("history-snapshot-1",),
            },
        ),
        _event(
            variant=variant,
            number=5,
            event_type="user_preference_stated",
            actor=EventActor.USER,
            payload={
                "kind": "brand",
                "value": "TrailForge",
                "polarity": "prefer",
                "scope": "long_term",
                "source_text": "I usually like TrailForge gear.",
            },
        ),
        _event(
            variant=variant,
            number=6,
            event_type="user_preference_stated",
            actor=EventActor.USER,
            payload={
                "kind": "color",
                "value": "green",
                "polarity": "prefer",
                "scope": "long_term",
                "source_text": "I tend to prefer green.",
            },
        ),
        _event(
            variant=variant,
            number=7,
            event_type="user_preference_stated",
            actor=EventActor.USER,
            payload={
                "kind": "color",
                "value": "charcoal",
                "polarity": "prefer",
                "scope": "task",
                "replace_same_kind": True,
                "source_text": "For this trip I'd prefer charcoal.",
            },
        ),
        _event(
            variant=variant,
            number=8,
            event_type="preference_priority_stated",
            actor=EventActor.USER,
            payload={
                "higher": "comfort",
                "lower": "weight",
                "kind": "feature",
                "scope": "task",
                "source_text": "Comfort matters more to me than weight.",
            },
        ),
        _event(
            variant=variant,
            number=9,
            event_type="candidate_pool_referenced",
            actor=EventActor.TRUSTED_COMPONENT,
            payload={
                "ref": pool.ref,
                "source": "recommendation_tool",
                "candidate_count": len(CANDIDATE_IDS),
                "digest": pool.digest,
            },
        ),
        _event(
            variant=variant,
            number=10,
            event_type="ranking_snapshot_referenced",
            actor=EventActor.TRUSTED_COMPONENT,
            payload={
                "ref": ranking.ref,
                "source": "serving_v1",
                "method": "rrf_rank_fusion",
                "candidate_count": len(CANDIDATE_IDS),
                "digest": ranking.digest,
            },
        ),
    ]
    return events


def build_state(*, variant: str = "default") -> UnifiedDecisionState:
    """Replay the M5 event stream into the unified decision state for one variant."""
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown fixture variant: {variant!r}")
    task_id = variant_task_id(variant)
    state = empty_state(
        user_key=USER_KEY,
        task_id=task_id,
        session_id=variant_session_id(variant),
    )
    events = _scenario_events(variant=variant)
    if variant == "retracted":
        events.append(
            _event(
                variant=variant,
                number=11,
                event_type="preference_retracted",
                actor=EventActor.USER,
                payload={
                    "kind": "feature",
                    "value": "lightweight",
                    "scope": "task",
                    "source_text": "Actually, forget lightweight; I do not care about that.",
                },
            )
        )
    elif variant == "corrected":
        events.append(
            _event(
                variant=variant,
                number=11,
                event_type="user_preference_stated",
                actor=EventActor.USER,
                payload={
                    "kind": "feature",
                    "value": "waterproof",
                    "polarity": "prefer",
                    "scope": "task",
                    "replaces_value": "comfort",
                    "source_text": "Actually, comfort is not the priority; I need it waterproof.",
                },
            )
        )
    return BeliefUpdater().apply_all(state, events)


def _directional_soft_features() -> tuple[RankerFeature, ...]:
    """Projection-level soft directional inferred beliefs for the ``directional`` variant.

    M5's inferred-hypothesis payload carries no ``direction`` field, so a directional
    *hypothesis* cannot yet be expressed by the event vocabulary; this fixture composes
    the features onto the real projection and documents that gap.  ``durability`` has no
    grounded numeric catalogue field and must be reported as unsupported.
    """
    return (
        RankerFeature(
            preference_id="m6-fixture:inferred:weight-minimize",
            kind=PreferenceKind.FEATURE,
            value="weight",
            polarity=PreferencePolarity.PREFER,
            direction=PreferenceDirection.MINIMIZE,
            scope=PreferenceScope.TASK,
            lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
            strength_rank=1,
            is_soft=True,
        ),
        RankerFeature(
            preference_id="m6-fixture:inferred:durability-minimize",
            kind=PreferenceKind.FEATURE,
            value="durability",
            polarity=PreferencePolarity.PREFER,
            direction=PreferenceDirection.MINIMIZE,
            scope=PreferenceScope.TASK,
            lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
            strength_rank=1,
            is_soft=True,
        ),
    )


def replay_projection(*, variant: str = "default") -> RankerStateProjection:
    """Rebuild the ``for_ranker`` projection from a replayed M5 event stream."""
    projection = for_ranker(build_state(variant=variant))
    if variant == "directional":
        projection = projection.model_copy(
            update={"features": (*projection.features, *_directional_soft_features())}
        )
    return projection


def build_task(*, variant: str = "default") -> FrozenRankingTask:
    """Build one frozen M6.1 ranking task for the requested variant."""
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown fixture variant: {variant!r}")
    return FrozenRankingTask(
        task_id=variant_task_id(variant),
        user_key=USER_KEY,
        turn_key=variant_turn_key(variant),
        snapshot=build_snapshot(variant=variant),
        ranker_projection=replay_projection(variant=variant),
        labels=(),
        label_protocol=None,
        dataset_source="synthetic:m6_1_fixture",
    )


def replay_task(*, variant: str = "default") -> FrozenRankingTask:
    """Rebuild one frozen task from the replayed events (replay-determinism helper)."""
    return build_task(variant=variant)


def build_variants() -> dict[str, FrozenRankingTask]:
    """Build every fixture variant, keyed by variant name (deterministic order)."""
    return {variant: build_task(variant=variant) for variant in VARIANT_NAMES}


def build_tasks() -> tuple[FrozenRankingTask, ...]:
    """Build the frozen task set for a paired evaluation run, in deterministic order."""
    return tuple(build_variants().values())
