"""Focused tests for the M6.1 belief-aware feature builder.

The builder must join the frozen snapshot to the M5 ``for_ranker`` projection without
inventing evidence: explicit evidence comes from the real M10A matcher, non-directional
soft evidence reuses that same matcher, and directional soft beliefs are grounded only
from catalogue numeric fields with set-relative ordinal counts.  Nothing here
re-implements the builder; the tests assert the contract over the real synthetic fixture.
"""

from __future__ import annotations

import hashlib
import itertools
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.memory.schemas import (  # noqa: E402
    PreferenceKind as MemoryPreferenceKind,
    PreferenceMemoryEntry,
    PreferencePolarity as MemoryPreferencePolarity,
    PreferenceStatus,
)
from recommendation.preference_matching.matcher import PreferenceCandidateMatcher  # noqa: E402
from recommendation.preference_matching.schemas import EvidenceStatus  # noqa: E402
from recommendation.ranking import features as features_module  # noqa: E402
from recommendation.ranking.errors import FeatureBuildError  # noqa: E402
from recommendation.ranking.features import (  # noqa: E402
    CATALOGUE_FIELD_ORDER,
    BeliefAwareFeatureBuilder,
)
from recommendation.ranking.schemas import (  # noqa: E402
    CandidateSnapshot,
    EligibilityState,
    EvidenceState,
    FrozenRankingTask,
)
from recommendation.state.schemas import (  # noqa: E402
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    canonical_json,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    candidate_pool_snapshot,
    ranking_snapshot,
)
from recommendation.state.state_projector import (  # noqa: E402
    RankerFeature,
    RankerStateProjection,
)
from tests import m6_1_fixture as fx  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def task_default() -> FrozenRankingTask:
    return fx.build_task(variant="default")


@pytest.fixture(scope="module")
def task_retracted() -> FrozenRankingTask:
    return fx.build_task(variant="retracted")


@pytest.fixture(scope="module")
def task_corrected() -> FrozenRankingTask:
    return fx.build_task(variant="corrected")


@pytest.fixture(scope="module")
def task_directional() -> FrozenRankingTask:
    return fx.build_task(variant="directional")


@pytest.fixture()
def builder() -> BeliefAwareFeatureBuilder:
    """The real builder.

    A feature build imports the accepted M8 candidate types (their package initialisers
    pull the model stack), so this fixture requires the model-stack venv; the module
    import itself is torch-free and is tested below without it.
    """
    pytest.importorskip("torch", reason="the accepted candidate types need the model stack")
    return BeliefAwareFeatureBuilder()


def _build(
    builder: BeliefAwareFeatureBuilder, task: FrozenRankingTask
):
    """Build features for one frozen task through the label-free public signature."""
    return builder.build(
        snapshot=task.snapshot, ranker_projection=task.ranker_projection
    )


def _by_id(feature_set) -> dict[str, object]:
    return {feature.candidate_id: feature for feature in feature_set.features}


def _feature_digest(feature_set) -> str:
    payload = feature_set.model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _weight_value(candidate) -> float | None:
    """Read a candidate's grounded weight in kilograms from its own metadata."""
    if candidate.metadata is None:
        return None
    for key, value in candidate.metadata.details:
        if "weight" in key.casefold():
            tokens = value.replace(",", " ").split()
            for index, token in enumerate(tokens):
                try:
                    number = float(token)
                except ValueError:
                    continue
                unit = tokens[index + 1].strip(".,").lower() if index + 1 < len(tokens) else ""
                return number if unit in ("kg", "kilogram", "kilograms") else None
    return None


def _price_value(candidate) -> float | None:
    """Read a candidate's price as a plain float, or ``None`` when unreadable."""
    if candidate.metadata is None or candidate.metadata.price_text is None:
        return None
    raw = candidate.metadata.price_text.strip().replace(",", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _expected_ordinal_counts(
    candidates, value_of
) -> dict[str, tuple[int, int]]:
    """Independently compute (better_than, worse_than) for a minimize direction."""
    known = {
        candidate.candidate_id: value_of(candidate)
        for candidate in candidates
        if value_of(candidate) is not None
    }
    counts: dict[str, tuple[int, int]] = {}
    for candidate_id, value in known.items():
        better = sum(1 for other, other_value in known.items() if other != candidate_id and other_value < value)
        worse = sum(1 for other, other_value in known.items() if other != candidate_id and other_value > value)
        counts[candidate_id] = (better, worse)
    return counts


def _explicit_evidence_for(feature):
    """Return the explicit evidence of ``feature`` whose belief value is ``value``."""
    return tuple(
        evidence
        for evidence in feature.explicit_evidence
        if evidence.preference_value == "comfort"
    )


def _inferred_for_value(feature, value: str):
    """Return the inferred evidence records for one belief value."""
    return tuple(record for record in feature.inferred_evidence if record.value == value)


def _a1_key(feature) -> tuple[int, ...]:
    return (*feature.explicit_key, feature.original_rank, feature.item_id)


def _a2_key(feature) -> tuple[int, ...]:
    return (*feature.tie_break_key, feature.original_rank, feature.item_id)


# --------------------------------------------------------------------------- #
# Import boundary and label-free signature
# --------------------------------------------------------------------------- #


def test_module_import_does_not_import_torch() -> None:
    """The feature module must be importable without the model stack at import time."""
    code = (
        "import sys; import recommendation.ranking.features; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert features_module.__name__ == "recommendation.ranking.features"


def test_builder_signature_accepts_only_snapshot_and_projection() -> None:
    signature = inspect.signature(BeliefAwareFeatureBuilder.build)
    assert list(signature.parameters) == ["self", "snapshot", "ranker_projection"]
    for name in ("snapshot", "ranker_projection"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    # The builder cannot accept a task (which would carry labels) or any label field.
    assert not hasattr(features_module, "FrozenRankingTask")
    for forbidden in ("labels", "task", "label_protocol"):
        assert forbidden not in signature.parameters


def test_fixture_tasks_carry_no_labels() -> None:
    for task in fx.build_tasks():
        assert task.labels == ()
        assert task.label_protocol is None
        assert task.dataset_source.startswith("synthetic")


# --------------------------------------------------------------------------- #
# Core determinism, identity and provenance
# --------------------------------------------------------------------------- #


def test_build_is_deterministic(builder, task_default) -> None:
    first = _build(builder, task_default)
    second = _build(builder, task_default)
    replay = _build(builder, fx.replay_task(variant="default"))
    assert first == second == replay
    assert _feature_digest(first) == _feature_digest(second) == _feature_digest(replay)
    assert first.feature_schema_version == 1


def test_feature_set_copies_snapshot_identity_and_digests(builder, task_default) -> None:
    feature_set = _build(builder, task_default)
    snapshot = task_default.snapshot
    assert feature_set.task_id == snapshot.task_id == task_default.task_id
    assert feature_set.user_key == snapshot.user_key
    assert feature_set.turn_key == snapshot.turn_key
    assert feature_set.snapshot_id == snapshot.snapshot_id
    assert feature_set.snapshot_candidate_count == snapshot.candidate_count == 7
    assert feature_set.candidate_ids == snapshot.candidate_ids
    assert feature_set.candidate_pool_digest == candidate_pool_snapshot(fx.CANDIDATE_IDS).digest
    assert feature_set.ranking_digest == ranking_snapshot(fx.CANDIDATE_IDS).digest
    assert snapshot.candidate_pool_digest == feature_set.candidate_pool_digest
    assert snapshot.ranking_snapshot_digest == feature_set.ranking_digest
    assert feature_set.coverage.candidate_count == 7
    assert feature_set.ranker_projection == task_default.ranker_projection


def test_provenance_and_eligibility_are_copied_unchanged(builder, task_default) -> None:
    feature_set = _build(builder, task_default)
    snapshot_by_id = {
        candidate.candidate_id: candidate for candidate in task_default.snapshot.candidates
    }
    for feature in feature_set.features:
        candidate = snapshot_by_id[feature.candidate_id]
        assert feature.original_rank == candidate.original_rank
        assert feature.item_id == candidate.item_id
        assert feature.source_ranks == candidate.source_ranks
        assert feature.rrf_score == candidate.rrf_score
        assert feature.rrf_score_kind == candidate.rrf_score_kind
        assert feature.eligibility is candidate.eligibility
        assert feature.eligibility_detail == candidate.eligibility_detail
    assert snapshot_by_id["B0M6HIKE03"].eligibility is EligibilityState.INELIGIBLE
    assert feature_set.coverage.metadata_found_count == 6
    assert feature_set.coverage.metadata_missing_count == 1


def test_raw_scores_are_provenance_only(builder, task_default) -> None:
    """Changing only the raw scores must not change any evidence/ordinal feature."""
    resolver = fx.build_trusted_resolver()
    shifted = tuple(
        candidate.model_copy(
            update={
                "score": candidate.score + 1000.0,
                "rrf_score": (candidate.rrf_score or 0.0) + 1000.0,
            }
        )
        for candidate in fx.build_candidate_inputs()
    )
    pool = candidate_pool_snapshot(fx.CANDIDATE_IDS)
    ranking = ranking_snapshot(fx.CANDIDATE_IDS)
    shifted_snapshot = resolver.build_snapshot(
        snapshot_id=fx.variant_snapshot_id("default"),
        user_key=fx.USER_KEY,
        task_id=fx.variant_task_id("default"),
        turn_key=fx.variant_turn_key("default"),
        candidates=shifted,
        candidate_pool_ref=pool.ref,
        ranking_snapshot_ref=ranking.ref,
    )
    shifted_features = builder.build(
        snapshot=shifted_snapshot, ranker_projection=task_default.ranker_projection
    )
    original_features = _build(builder, task_default)
    for original, modified in zip(original_features.features, shifted_features.features):
        assert modified.rrf_score == (original.rrf_score or 0.0) + 1000.0
        original_dump = original.model_dump(mode="json")
        modified_dump = modified.model_dump(mode="json")
        original_dump.pop("rrf_score")
        modified_dump.pop("rrf_score")
        assert original_dump == modified_dump


# --------------------------------------------------------------------------- #
# Explicit evidence: the real M10A matcher, unchanged
# --------------------------------------------------------------------------- #


def test_explicit_evidence_matches_real_matcher_output(builder, task_default) -> None:
    # The accepted candidate types are imported lazily (their package initialisers pull
    # the model stack, so importing them at test-module import time would break the
    # torch-free light subset).
    from recommendation.rag.schemas import EnrichedRecommendation
    from recommendation.tools.schemas import ToolRecommendation

    feature_set = _build(builder, task_default)
    projection = task_default.ranker_projection
    explicit_features = tuple(
        feature
        for feature in projection.features
        if not feature.is_soft and feature.lifecycle in {PreferenceLifecycle.EXPLICIT, PreferenceLifecycle.CONFIRMED}
    )
    entries = tuple(
        PreferenceMemoryEntry(
            memory_id=feature.preference_id,
            user_key="state_projection",
            kind=MemoryPreferenceKind(feature.kind.value),
            value=feature.value,
            polarity=MemoryPreferencePolarity(feature.polarity.value),
            source_text=f"decision-state:{feature.preference_id}",
            source_turn_id="decision_state",
            extractor="m5_ranker_projection",
            status=PreferenceStatus.ACTIVE,
            logical_seq=position,
            created_at=0.0,
        )
        for position, feature in enumerate(explicit_features, start=1)
    )
    enriched = [
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
        for position, candidate in enumerate(task_default.snapshot.candidates)
    ]
    report = PreferenceCandidateMatcher().match(
        candidates=enriched, preferences=entries
    )
    for feature, expected in zip(feature_set.features, report.candidates):
        assert feature.explicit_evidence == expected.evidence
        assert feature.explicit_matches == expected.count(EvidenceStatus.MATCH)
        assert feature.explicit_violations == expected.count(EvidenceStatus.VIOLATION)
        assert feature.explicit_unknown == expected.count(EvidenceStatus.UNKNOWN)


def test_explicit_comfort_evidence_is_grounded(builder, task_default) -> None:
    features = _by_id(_build(builder, task_default))
    comfort = {
        candidate_id: _explicit_evidence_for(feature)
        for candidate_id, feature in features.items()
    }
    assert len(comfort["B0M6HIKE01"]) == 1
    top = comfort["B0M6HIKE01"][0]
    assert top.status is EvidenceStatus.MATCH
    assert top.metadata_present is True
    assert top.metadata_field in {"features", "title"}
    assert "comfort" in (top.metadata_value or "").casefold()
    assert features["B0M6HIKE01"].explicit_matches == 3
    assert features["B0M6HIKE02"].explicit_matches == 2
    assert features["B0M6HIKE05"].explicit_matches == 2
    assert features["B0M6HIKE03"].explicit_matches == 0


def test_no_violations_are_invented_for_positive_preferences(builder) -> None:
    for variant in ("default", "corrected", "directional"):
        feature_set = _build(builder, fx.build_task(variant=variant))
        for feature in feature_set.features:
            assert feature.explicit_violations == 0
            assert feature.inferred_violations == 0
            assert all(
                record.status is not EvidenceStatus.VIOLATION
                for record in feature.explicit_evidence
            )


def test_missing_metadata_is_unknown_never_violation(builder, task_default) -> None:
    features = _by_id(_build(builder, task_default))
    no_metadata = features["B0M6HIKE06"]
    # Four explicit beliefs are active in this projection; all of them are UNKNOWN.
    assert no_metadata.explicit_unknown == 4
    assert no_metadata.explicit_matches == 0
    assert no_metadata.explicit_violations == 0
    assert no_metadata.inferred_violations == 0
    assert no_metadata.catalogue_fields.present == ()
    assert set(no_metadata.catalogue_fields.missing) == set(CATALOGUE_FIELD_ORDER)
    lightweight = _inferred_for_value(no_metadata, "lightweight")
    assert len(lightweight) == 1
    assert lightweight[0].status.value == "unknown"
    assert lightweight[0].reason_code == "metadata_missing"


# --------------------------------------------------------------------------- #
# Inferred non-directional evidence: the same real matcher semantics
# --------------------------------------------------------------------------- #


def test_inferred_non_directional_text_evidence(builder, task_default) -> None:
    features = _by_id(_build(builder, task_default))
    matched = {
        candidate_id
        for candidate_id, feature in features.items()
        if _inferred_for_value(feature, "lightweight")
        and _inferred_for_value(feature, "lightweight")[0].status.value == "match"
    }
    assert matched == {"B0M6HIKE01", "B0M6HIKE03", "B0M6HIKE05"}
    evidence = _inferred_for_value(features["B0M6HIKE01"], "lightweight")[0]
    assert evidence.kind == PreferenceKind.FEATURE.value
    assert evidence.scope == PreferenceScope.TASK.value
    assert evidence.lifecycle == PreferenceLifecycle.INFERRED_HYPOTHESIS.value
    assert evidence.strength_rank == 1
    assert evidence.better_than_count == 0
    assert evidence.worse_than_count == 0
    unknown = _inferred_for_value(features["B0M6HIKE02"], "lightweight")[0]
    assert unknown.status.value == "unknown"
    assert unknown.reason_code == "insufficient_metadata"


def test_inferred_unknown_contributes_zero_support_and_penalty(builder) -> None:
    retracted = _build(builder, fx.build_task(variant="retracted"))
    for feature in retracted.features:
        assert feature.inferred_evidence == ()
        assert feature.inferred_matches == 0
        assert feature.inferred_unknown == 0
        assert feature.inferred_key == (0, 0)
    directional = _build(builder, fx.build_task(variant="directional"))
    features = _by_id(directional)
    for candidate_id in ("B0M6HIKE04", "B0M6HIKE06"):
        feature = features[candidate_id]
        weight = _inferred_for_value(feature, "weight")[0]
        assert weight.status.value == "unknown"
        assert weight.better_than_count == 0
        assert weight.worse_than_count == 0
        assert feature.inferred_ordinal_support == 0
        assert feature.inferred_ordinal_penalty == 0


# --------------------------------------------------------------------------- #
# Directional grounded evidence and ordinal tie-break
# --------------------------------------------------------------------------- #


def test_directional_ordinal_counts_are_set_relative(builder, task_directional) -> None:
    feature_set = _build(builder, task_directional)
    expected = _expected_ordinal_counts(
        task_directional.snapshot.candidates, _weight_value
    )
    assert set(expected) == {"B0M6HIKE01", "B0M6HIKE02", "B0M6HIKE03", "B0M6HIKE05", "B0M6HIKE07"}
    for candidate_id, (better, worse) in expected.items():
        feature = _by_id(feature_set)[candidate_id]
        weight = _inferred_for_value(feature, "weight")[0]
        assert weight.better_than_count == better
        assert weight.worse_than_count == worse
        assert weight.status.value == "match"
        assert weight.reason_code == "grounded_directional_value"
        assert weight.ordinal_value == _weight_value(
            next(
                candidate
                for candidate in task_directional.snapshot.candidates
                if candidate.candidate_id == candidate_id
            )
        )
        # Only the weight belief is directional here, so the candidate totals equal it.
        assert feature.inferred_ordinal_penalty == better
        assert feature.inferred_ordinal_support == worse
    # The best grounded weight candidate has zero penalty and maximum support.
    values = {
        candidate.candidate_id: _weight_value(candidate)
        for candidate in task_directional.snapshot.candidates
        if _weight_value(candidate) is not None
    }
    best_candidate = min(values, key=lambda candidate_id: values[candidate_id])
    best_feature = _by_id(feature_set)[best_candidate]
    assert best_feature.inferred_ordinal_penalty == 0
    assert best_feature.inferred_ordinal_support == len(values) - 1


def test_directional_unknown_attribute_is_unsupported_not_violation(builder, task_directional) -> None:
    feature_set = _build(builder, task_directional)
    assert "feature:durability (minimize)" in feature_set.coverage.unsupported_inferred_features
    assert "feature:weight (minimize)" in feature_set.coverage.supported_inferred_features
    for feature in feature_set.features:
        durability = _inferred_for_value(feature, "durability")[0]
        assert durability.status.value == "unknown"
        assert durability.reason_code == "unsupported_feature"
        assert durability.better_than_count == 0
        assert durability.worse_than_count == 0
    # The unknown-weight candidates still carry a weight record, explicitly UNKNOWN.
    for candidate_id in ("B0M6HIKE04", "B0M6HIKE06"):
        weight = _inferred_for_value(_by_id(feature_set)[candidate_id], "weight")[0]
        assert weight.status.value == "unknown"
        assert weight.reason_code == "insufficient_metadata"


def test_directional_a2_breaks_an_explicit_tie_property(builder, task_directional) -> None:
    """A2 must reorder an explicit tie by grounded inferred evidence (property check).

    The assertion is property-based rather than a hard-coded output list: among the
    eligible pool, at least one pair with equal explicit evidence differs in inferred
    ordinal evidence, the A2-preferred member was later in the trusted/A1 order, and the
    grounded value favours it for the minimize direction.
    """
    feature_set = _build(builder, task_directional)
    pool = [
        feature
        for feature in feature_set.features
        if feature.eligibility is EligibilityState.VERIFIED_ELIGIBLE
    ]
    assert len(pool) == 3

    a1_order = sorted(pool, key=_a1_key)
    a2_order = sorted(pool, key=_a2_key)
    assert [feature.candidate_id for feature in a1_order] != [
        feature.candidate_id for feature in a2_order
    ]

    weight_values = {
        candidate.candidate_id: _weight_value(candidate)
        for candidate in task_directional.snapshot.candidates
    }
    conflicts = []
    for left, right in itertools.combinations(pool, 2):
        if left.explicit_key != right.explicit_key:
            continue
        if left.inferred_key == right.inferred_key:
            continue
        a1_preferred = min((left, right), key=_a1_key)
        a2_preferred = min((left, right), key=_a2_key)
        if a1_preferred.candidate_id != a2_preferred.candidate_id:
            conflicts.append((a1_preferred, a2_preferred))
    assert conflicts, "no explicit-tie pair was reordered by grounded inferred evidence"
    for a1_preferred, a2_preferred in conflicts:
        assert a2_preferred.original_rank > a1_preferred.original_rank
        assert weight_values[a2_preferred.candidate_id] is not None
        assert weight_values[a1_preferred.candidate_id] is not None
        # minimize: the A2-preferred candidate is lighter and pays a smaller penalty.
        assert weight_values[a2_preferred.candidate_id] < weight_values[a1_preferred.candidate_id]
        assert a2_preferred.inferred_ordinal_penalty < a1_preferred.inferred_ordinal_penalty
        assert a2_preferred.inferred_ordinal_support > a1_preferred.inferred_ordinal_support


def test_directional_price_grounding_from_price_text(builder, task_default) -> None:
    price_feature = RankerFeature(
        preference_id="test:inferred:price-minimize",
        kind=PreferenceKind.FEATURE,
        value="price",
        polarity=PreferencePolarity.PREFER,
        direction=PreferenceDirection.MINIMIZE,
        scope=PreferenceScope.TASK,
        lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
        strength_rank=1,
        is_soft=True,
    )
    projection = task_default.ranker_projection.model_copy(
        update={
            "features": (*task_default.ranker_projection.features, price_feature)
        }
    )
    feature_set = builder.build(
        snapshot=task_default.snapshot, ranker_projection=projection
    )
    expected = _expected_ordinal_counts(
        task_default.snapshot.candidates, _price_value
    )
    assert len(expected) == 5
    assert feature_set.coverage.readable_fields["price"] == len(expected)
    for candidate_id, (better, worse) in expected.items():
        feature = _by_id(feature_set)[candidate_id]
        evidence = _inferred_for_value(feature, "price")[0]
        assert evidence.metadata_field == "price_text"
        assert evidence.ordinal_value == _price_value(
            next(
                candidate
                for candidate in task_default.snapshot.candidates
                if candidate.candidate_id == candidate_id
            )
        )
        assert evidence.better_than_count == better
        assert evidence.worse_than_count == worse
    unreadable = _by_id(feature_set)["B0M6HIKE04"]
    price_unknown = _inferred_for_value(unreadable, "price")[0]
    assert price_unknown.status.value == "unknown"
    assert price_unknown.metadata_field == "price_text"
    assert price_unknown.better_than_count == 0
    assert price_unknown.worse_than_count == 0


# --------------------------------------------------------------------------- #
# Retraction / correction / scope semantics
# --------------------------------------------------------------------------- #


def test_retracted_hypothesis_removes_all_inferred_signal(builder, task_retracted) -> None:
    feature_set = _build(builder, task_retracted)
    assert feature_set.coverage.supported_inferred_features == ()
    assert feature_set.coverage.unsupported_inferred_features == ()
    for feature in feature_set.features:
        assert feature.inferred_evidence == ()
        assert feature.inferred_matches == 0
        assert feature.inferred_violations == 0
        assert feature.inferred_unknown == 0
        assert feature.inferred_key == (0, 0)
    # A2's inferred key is identical for every candidate, so A2 degenerates to A1.
    assert {feature.inferred_key for feature in feature_set.features} == {(0, 0)}


def test_corrected_belief_old_value_inactive_new_value_active(builder, task_corrected) -> None:
    feature_set = _build(builder, task_corrected)
    values = {entry.value for entry in feature_set.features[0].preference_strength_metadata}
    assert "comfort" not in values
    assert "waterproof" in values
    explicit_values = {
        evidence.preference_value
        for feature in feature_set.features
        for evidence in feature.explicit_evidence
    }
    assert "comfort" not in explicit_values
    assert "waterproof" in explicit_values
    waterproof_matches = sum(
        1
        for feature in feature_set.features
        for evidence in feature.explicit_evidence
        if evidence.preference_value == "waterproof" and evidence.status is EvidenceStatus.MATCH
    )
    assert waterproof_matches >= 1


def test_task_and_long_term_scopes_are_preserved_without_silent_override(builder, task_default) -> None:
    feature_set = _build(builder, task_default)
    metadata = {
        (entry.kind, entry.value): entry
        for entry in feature_set.features[0].preference_strength_metadata
    }
    assert metadata[("brand", "TrailForge")].scope == PreferenceScope.LONG_TERM.value
    assert metadata[("color", "green")].scope == PreferenceScope.LONG_TERM.value
    assert metadata[("color", "charcoal")].scope == PreferenceScope.TASK.value
    # Both colour beliefs stay active: the task statement does not silently override the
    # long-term preference (scope, not value, distinguishes them).
    assert metadata[("color", "green")].lifecycle == PreferenceLifecycle.EXPLICIT.value
    assert metadata[("color", "charcoal")].lifecycle == PreferenceLifecycle.EXPLICIT.value
    assert metadata[("feature", "lightweight")].is_soft is True
    assert metadata[("feature", "lightweight")].strength_rank == 1
    assert metadata[("feature", "comfort")].is_soft is False
    assert metadata[("feature", "comfort")].strength_rank == 3


def test_preference_strength_is_ordinal_never_a_probability(builder, task_directional) -> None:
    feature_set = _build(builder, task_directional)
    expected_fields = {
        "belief_id",
        "kind",
        "value",
        "polarity",
        "direction",
        "scope",
        "lifecycle",
        "strength_rank",
        "is_soft",
    }
    for feature in feature_set.features:
        assert not any(
            "probability" in name or "confidence" in name
            for name in feature.model_dump()
        )
        for entry in feature.preference_strength_metadata:
            assert set(entry.model_dump()) == expected_fields
            assert entry.strength_rank in (1, 2, 3)
            assert isinstance(entry.strength_rank, int)
            assert not any(
                isinstance(value, float) for value in entry.model_dump().values()
            )
    weights = [feature for feature in feature_set.features if feature.inferred_evidence]
    assert weights
    # The directional evidence carries ordinal counts, never a score-like float.
    for feature in weights:
        for record in feature.inferred_evidence:
            assert record.ordinal_value is None or isinstance(record.ordinal_value, float)
            assert isinstance(record.better_than_count, int)
            assert isinstance(record.worse_than_count, int)


# --------------------------------------------------------------------------- #
# Catalogue coverage and missingness
# --------------------------------------------------------------------------- #


def test_coverage_reports_catalogue_availability(builder, task_default) -> None:
    coverage = _build(builder, task_default).coverage
    assert coverage.candidate_count == 7
    assert coverage.metadata_found_count == 6
    assert coverage.metadata_missing_count == 1
    assert coverage.readable_fields == {
        "title": 6,
        "price": 5,
        "weight": 5,
        "color": 6,
        "material": 5,
        "brand": 6,
        "features": 6,
        "categories": 6,
    }
    assert coverage.supported_inferred_features == ("feature:lightweight (text)",)
    assert coverage.unsupported_inferred_features == ()
    assert any("UNKNOWN" in note for note in coverage.notes)
    assert any("eligibility projection" in note for note in coverage.notes)


def test_coverage_weight_readability_agrees_with_grounded_evidence(
    builder, task_directional
) -> None:
    """The coverage readability check and the accepted grounded parser must agree."""
    feature_set = _build(builder, task_directional)
    grounded = sum(
        1
        for feature in feature_set.features
        for record in feature.inferred_evidence
        if record.value == "weight" and record.status is EvidenceState.MATCH
    )
    assert grounded == 5
    assert feature_set.coverage.readable_fields["weight"] == grounded


def test_catalogue_fields_report_per_candidate_missingness(builder, task_default) -> None:
    features = _by_id(_build(builder, task_default))
    assert set(features["B0M6HIKE01"].catalogue_fields.present) == set(CATALOGUE_FIELD_ORDER)
    assert features["B0M6HIKE01"].catalogue_fields.missing == ()

    assert set(features["B0M6HIKE04"].catalogue_fields.missing) == {"price", "weight"}
    assert "color" in features["B0M6HIKE04"].catalogue_fields.present

    assert set(features["B0M6HIKE05"].catalogue_fields.missing) == {"material"}
    assert "weight" in features["B0M6HIKE05"].catalogue_fields.present
    assert features["B0M6HIKE05"].catalogue_fields.present.index("weight") == 2


def test_directional_variant_coverage_lists_supported_and_unsupported(builder, task_directional) -> None:
    coverage = _build(builder, task_directional).coverage
    assert "feature:weight (minimize)" in coverage.supported_inferred_features
    assert "feature:lightweight (text)" in coverage.supported_inferred_features
    assert coverage.unsupported_inferred_features == ("feature:durability (minimize)",)
    assert any("no grounded catalogue field" in note for note in coverage.notes)
    assert any("ordinal" in note for note in coverage.notes)


# --------------------------------------------------------------------------- #
# Rejections and edge cases
# --------------------------------------------------------------------------- #


def test_builder_rejects_inconsistent_projection(builder, task_default) -> None:
    projection = task_default.ranker_projection
    with pytest.raises(FeatureBuildError, match="inconsistent"):
        builder.build(
            snapshot=task_default.snapshot,
            ranker_projection=projection.model_copy(
                update={
                    "features": (
                        *projection.features,
                        RankerFeature(
                            preference_id="test:inconsistent",
                            kind=PreferenceKind.FEATURE,
                            value="something",
                            polarity=PreferencePolarity.PREFER,
                            scope=PreferenceScope.TASK,
                            lifecycle=PreferenceLifecycle.EXPLICIT,
                            strength_rank=3,
                            is_soft=True,
                        ),
                    )
                }
            ),
        )


def test_builder_rejects_foreign_task_projection(builder, task_default) -> None:
    projection = task_default.ranker_projection.model_copy(update={"task_id": "some-other-task"})
    with pytest.raises(FeatureBuildError, match="different task"):
        builder.build(snapshot=task_default.snapshot, ranker_projection=projection)


def test_builder_rejects_duplicate_preference_ids(builder, task_default) -> None:
    projection = task_default.ranker_projection
    duplicated = (*projection.features, projection.features[0])
    with pytest.raises(FeatureBuildError, match="duplicate preference ids"):
        builder.build(
            snapshot=task_default.snapshot,
            ranker_projection=projection.model_copy(update={"features": duplicated}),
        )


def test_builder_rejects_wrong_input_types(builder, task_default) -> None:
    with pytest.raises(FeatureBuildError, match="CandidateSnapshot"):
        builder.build(snapshot=object(), ranker_projection=task_default.ranker_projection)
    with pytest.raises(FeatureBuildError, match="RankerStateProjection"):
        builder.build(snapshot=task_default.snapshot, ranker_projection=object())


def test_empty_snapshot_builds_an_empty_feature_set(builder) -> None:
    snapshot = CandidateSnapshot(
        snapshot_id="empty-snapshot",
        user_key="empty-user",
        task_id="empty-task",
        turn_key="empty-turn",
        candidates=(),
    )
    projection = RankerStateProjection(task_id="empty-task", state_version=1)
    feature_set = builder.build(snapshot=snapshot, ranker_projection=projection)
    assert feature_set.features == ()
    assert feature_set.snapshot_candidate_count == 0
    assert feature_set.coverage.candidate_count == 0
    assert feature_set.candidate_pool_digest == candidate_pool_snapshot(()).digest
    assert feature_set.ranking_digest == ranking_snapshot(()).digest
