"""M6.1 offline ranking smoke example (synthetic hiking-backpack fixture).

Runs A0 / A1 / A2 on the **same** frozen candidates from ``tests/m6_1_fixture.py`` and
prints the candidate table required by the milestone brief:

    candidate_id, original_rank, A0_rank, A1_rank, A2_rank,
    explicit_match_status, inferred_match_status, eligibility_status

Assertions are derived from the data, never from hardcoded result strings:

* A0 preserves the trusted snapshot order (restricted to the shared eligible pool);
* A1 equals a *direct* call to the real, unmodified M10B ``rerank_candidates`` over the
  same M10A evidence, built independently in this example;
* A2 equals an independently computed implementation of the exact preregistered six-key
  policy, preserves explicit-preference priority, and only uses inferred evidence inside
  explicit ties;
* UNKNOWN explicit/inferred evidence is neutral (never a violation / never a penalty);
* the retracted-hypothesis variant makes A2 order identical to A1 (no inferred signal);
* candidate identities are identical across arms and eligibility-excluded candidates are
  absent from every arm;
* rebuilding features and re-ranking is byte-identical, and rebuilding a task from the
  replayed M5 event stream reproduces the same feature set;
* the directional variant exercises the ordinal inferred tie-break: A2 differs from A1,
  and every reordering is between candidates tied on explicit priority.

Everything is offline and deterministic: no checkpoint, no network, no LLM, no labels.
This is a lab example only; Serving V1 is not activated and no quality claim is made.

Usage::

    python -m experiments.m6_1_ranking_smoke [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.preference_matching.schemas import (  # noqa: E402
    CandidatePreferenceEvidence,
    EvidenceStatus,
    PreferenceEvidenceReport,
)
from recommendation.ranking.policies import (  # noqa: E402
    rank_all_arms,
    shared_eligibility_projection,
)
from recommendation.ranking.schemas import (  # noqa: E402
    CandidateFeatureSet,
    FrozenRankingTask,
    RankingArm,
)
from recommendation.reranking.reranker import rerank_candidates  # noqa: E402

DISCLAIMER = "offline lab only; Serving V1 not activated; no quality claim"

_CHECKS: list[str] = []
_FAILURES: list[str] = []


class SmokeError(RuntimeError):
    """A smoke assertion failed."""


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion; a failure is reported with its bounded detail."""
    _CHECKS.append(name)
    if not condition:
        _FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------------------- #
# fixture loading (tests/m6_1_fixture.py, owned by the ranking-features task)
# --------------------------------------------------------------------------- #


def _load_fixture() -> Any:
    """Import the shared synthetic fixture, or stop with a clear message."""
    try:
        from tests import m6_1_fixture  # noqa: PLC0415 - needed for a clean CLI error
    except Exception as error:  # noqa: BLE001 - reported, never guessed
        raise SmokeError(
            "the shared fixture tests/m6_1_fixture.py is not importable: "
            f"{type(error).__name__}: {error}"
        ) from error
    return m6_1_fixture


def _load_builder() -> Any:
    """The accepted feature builder (the only feature implementation in the lab)."""
    try:
        from recommendation.ranking.features import BeliefAwareFeatureBuilder
    except Exception as error:  # noqa: BLE001 - reported, never guessed
        raise SmokeError(
            "recommendation/ranking/features.py is not importable: "
            f"{type(error).__name__}: {error}"
        ) from error
    return BeliefAwareFeatureBuilder()


def _scenario_tasks(fixture: Any) -> dict[str, FrozenRankingTask]:
    """Return the fixture's named scenario variants as ``name -> FrozenRankingTask``."""
    variants = fixture.build_variants()
    return {str(name): task for name, task in variants.items()}


# --------------------------------------------------------------------------- #
# independent reference computations
# --------------------------------------------------------------------------- #


def _eligible_features(
    feature_set: CandidateFeatureSet,
) -> tuple[list[Any], tuple[str, ...], tuple[str, ...]]:
    """The shared eligibility projection reused from the policy module."""
    projection = shared_eligibility_projection(feature_set)
    by_id = {feature.candidate_id: feature for feature in feature_set.features}
    return (
        [by_id[candidate_id] for candidate_id in projection.eligible_ids],
        projection.eligible_ids,
        projection.excluded_ids,
    )


def _reference_a0(features: Sequence[Any]) -> tuple[str, ...]:
    """A0: trusted original order (also computed independently of the policy module)."""
    return tuple(
        feature.candidate_id
        for feature in sorted(features, key=lambda item: (item.original_rank, item.item_id))
    )


def _reference_a1(
    task: FrozenRankingTask, features: Sequence[Any]
) -> tuple[str, ...]:
    """A1: direct real-M10B rerank over an independently built M10A evidence report."""
    records = {candidate.candidate_id: candidate for candidate in task.snapshot.candidates}
    report = PreferenceEvidenceReport(
        candidates=tuple(
            CandidatePreferenceEvidence(
                original_rank=feature.original_rank,
                item_id=feature.item_id,
                parent_asin=feature.candidate_id,
                sasrec_score=records[feature.candidate_id].score,
                evidence=feature.explicit_evidence,
            )
            for feature in sorted(
                features, key=lambda item: (item.original_rank, item.item_id)
            )
        ),
    )
    return rerank_candidates(report=report).parent_asins


def _reference_a2(features: Sequence[Any]) -> tuple[str, ...]:
    """A2: independent implementation of the exact preregistered six-key policy."""
    return tuple(
        feature.candidate_id
        for feature in sorted(
            features,
            key=lambda item: (
                item.explicit_violations,
                -item.explicit_matches,
                item.inferred_ordinal_penalty,
                -item.inferred_ordinal_support,
                item.original_rank,
                item.item_id,
            ),
        )
    )


# --------------------------------------------------------------------------- #
# table rendering
# --------------------------------------------------------------------------- #


def _explicit_label(feature: Any) -> str:
    """Compact explicit-evidence label for the table."""
    return (
        f"m{feature.explicit_matches}/v{feature.explicit_violations}"
        f"/u{feature.explicit_unknown}"
    )


def _inferred_label(feature: Any) -> str:
    """Compact inferred-evidence label for the table."""
    return (
        f"m{feature.inferred_matches}/v{feature.inferred_violations}"
        f"/u{feature.inferred_unknown}/s{feature.inferred_ordinal_support}"
        f"/p{feature.inferred_ordinal_penalty}"
    )


def _table_rows(
    task: FrozenRankingTask,
    feature_set: CandidateFeatureSet,
    rankings: Sequence[Any],
) -> list[dict[str, Any]]:
    """One row per snapshot candidate, with each arm's rank (``None`` when excluded).

    Every candidate in the frozen snapshot is listed -- including ones the shared
    eligibility projection excluded -- so the table shows *why* a candidate is absent from
    all arms (its ``eligibility_status``) instead of silently hiding it.
    """
    by_arm = {
        ranking.arm: {
            candidate_id: ranking.rank_of(candidate_id)
            for candidate_id in ranking.candidate_ids
        }
        for ranking in rankings
    }
    feature_by_id = {feature.candidate_id: feature for feature in feature_set.features}
    rows: list[dict[str, Any]] = []
    for record in task.snapshot.candidates:
        feature = feature_by_id[record.candidate_id]
        rows.append(
            {
                "candidate_id": record.candidate_id,
                "original_rank": record.original_rank,
                "A0_rank": by_arm[RankingArm.A0_ORIGINAL].get(record.candidate_id),
                "A1_rank": by_arm[RankingArm.A1_EXPLICIT_RERANK].get(record.candidate_id),
                "A2_rank": by_arm[RankingArm.A2_BELIEF_TIEBREAK].get(record.candidate_id),
                "explicit_match_status": _explicit_label(feature),
                "inferred_match_status": _inferred_label(feature),
                "eligibility_status": record.eligibility.value,
            }
        )
    return rows


def _format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the smoke table as aligned text."""
    headers = (
        "candidate_id",
        "original_rank",
        "A0_rank",
        "A1_rank",
        "A2_rank",
        "explicit_match_status",
        "inferred_match_status",
        "eligibility_status",
    )
    widths = {
        header: max(len(header), max((len(str(row[header])) for row in rows), default=0))
        for header in headers
    }
    lines = ["  ".join(header.ljust(widths[header]) for header in headers)]
    for row in rows:
        lines.append("  ".join(str(row[header]).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the smoke run
# --------------------------------------------------------------------------- #


def _check_variant(
    name: str,
    task: FrozenRankingTask,
    builder: Any,
    *,
    inferred_signal: str,
) -> dict[str, Any]:
    """Build features and assert every per-variant invariant; return the evidence.

    ``inferred_signal`` fixes the expected relation between A2 and A1 for the variant:

    * ``must_differ`` -- the variant must exercise the ordinal inferred tie-break;
    * ``may_differ`` -- either outcome is valid; if it differs, only explicit ties may move;
    * ``must_equal`` -- no inferred signal exists, so A2 must degenerate to A1.
    """
    feature_set = builder.build(
        snapshot=task.snapshot, ranker_projection=task.ranker_projection
    )
    features, eligible_ids, excluded_ids = _eligible_features(feature_set)
    rankings = rank_all_arms(feature_set, task.snapshot)
    a0, a1, a2 = rankings
    feature_by_id = {feature.candidate_id: feature for feature in features}
    eligible = set(eligible_ids)
    snapshot_ids = set(task.snapshot.candidate_ids)

    # -- identity / eligibility: same set for every arm, excluded absent -------- #
    for arm in rankings:
        check(
            f"{name}:{arm.arm.value}_set",
            set(arm.candidate_ids) == eligible,
            "every arm must rank exactly the shared eligibility-projected set",
        )
        check(
            f"{name}:{arm.arm.value}_no_invention",
            set(arm.candidate_ids) <= snapshot_ids
            and len(set(arm.candidate_ids)) == len(arm.candidate_ids),
            "no arm may add, duplicate or replace a candidate",
        )
        check(
            f"{name}:{arm.arm.value}_excluded_absent",
            not (set(arm.candidate_ids) & set(excluded_ids)),
            "no arm may present a candidate the eligibility projection excluded",
        )
    check(
        f"{name}:candidate_set_equal_across_arms",
        set(a0.candidate_ids) == set(a1.candidate_ids) == set(a2.candidate_ids),
        "A0/A1/A2 must rank exactly the same candidate set",
    )
    check(
        f"{name}:candidate_set_digest_equal",
        a0.candidate_set_digest == a1.candidate_set_digest == a2.candidate_set_digest,
        "the order-insensitive candidate-set digest must be arm-independent",
    )

    # -- independent A0 / A1 / A2 parity --------------------------------------- #
    check(
        f"{name}:a0_parity",
        a0.candidate_ids == _reference_a0(features),
        "A0 must be the trusted snapshot order of the eligible pool",
    )
    check(
        f"{name}:a1_parity",
        a1.candidate_ids == _reference_a1(task, features),
        "A1 must equal a direct rerank_candidates() call over the same evidence",
    )
    check(
        f"{name}:a2_parity",
        a2.candidate_ids == _reference_a2(features),
        "A2 must equal the exact preregistered six-key policy",
    )

    # -- explicit priority: inferred never overrides --------------------------- #
    for ranking in (a1, a2):
        keys = [
            feature_by_id[candidate_id].explicit_key
            for candidate_id in ranking.candidate_ids
        ]
        check(
            f"{name}:{ranking.arm.value}_explicit_priority",
            keys == sorted(keys),
            "the arm must never place worse explicit evidence ahead of better",
        )
    for left_index in range(len(a1.candidate_ids)):
        for right_index in range(left_index + 1, len(a1.candidate_ids)):
            left = a1.candidate_ids[left_index]
            right = a1.candidate_ids[right_index]
            if feature_by_id[left].explicit_key != feature_by_id[right].explicit_key:
                check(
                    f"{name}:a1_a2_agree_on_explicit_pairs[{left}>{right}]",
                    a2.candidate_ids.index(left) < a2.candidate_ids.index(right),
                    "A1 and A2 may disagree only on explicit-priority ties",
                )

    # -- UNKNOWN is neutral ---------------------------------------------------- #
    for candidate_id, feature in feature_by_id.items():
        for evidence in feature.explicit_evidence:
            if evidence.status is EvidenceStatus.UNKNOWN:
                check(
                    f"{name}:{candidate_id}:unknown_never_violation",
                    feature.explicit_violations
                    == sum(
                        1
                        for item in feature.explicit_evidence
                        if item.status is EvidenceStatus.VIOLATION
                    ),
                    "UNKNOWN evidence must not be counted as a violation",
                )
        if feature.inferred_unknown > 0 and (
            feature.inferred_matches == 0 and feature.inferred_violations == 0
        ):
            check(
                f"{name}:{candidate_id}:unknown_inferred_zero_key",
                feature.inferred_ordinal_penalty == 0
                and feature.inferred_ordinal_support == 0,
                "UNKNOWN inferred evidence must contribute 0/0",
            )

    # -- A2 vs A1 relation ----------------------------------------------------- #
    if inferred_signal == "must_equal":
        check(
            f"{name}:a2_equals_a1",
            a2.candidate_ids == a1.candidate_ids,
            "with no inferred signal A2 must degenerate to A1",
        )
    elif inferred_signal == "must_differ":
        check(
            f"{name}:a2_differs_from_a1_when_inferred_signal_exists",
            a2.candidate_ids != a1.candidate_ids,
            "the directional variant must exercise the ordinal inferred tie-break",
        )
    if inferred_signal != "must_equal" and a2.candidate_ids != a1.candidate_ids:
        a1_positions = {
            candidate_id: index for index, candidate_id in enumerate(a1.candidate_ids)
        }
        a2_positions = {
            candidate_id: index for index, candidate_id in enumerate(a2.candidate_ids)
        }
        reordered_pairs = [
            (left, right)
            for index, left in enumerate(a1.candidate_ids)
            for right in a1.candidate_ids[index + 1 :]
            if (a1_positions[left] < a1_positions[right])
            != (a2_positions[left] < a2_positions[right])
        ]
        check(
            f"{name}:a2_reorders_only_explicit_ties",
            all(
                feature_by_id[left].explicit_key
                == feature_by_id[right].explicit_key
                for left, right in reordered_pairs
            ),
            "A2 may reorder only candidates tied on explicit priority",
        )

    return {
        "variant": name,
        "task_id": task.task_id,
        "feature_set": feature_set,
        "features": features,
        "rankings": rankings,
        "eligible_ids": eligible_ids,
        "excluded_ids": excluded_ids,
    }


def run_smoke() -> dict[str, Any]:
    """Execute every smoke assertion and return the JSON-able evidence record."""
    fixture = _load_fixture()
    builder = _load_builder()
    variants = _scenario_tasks(fixture)
    if "default" not in variants:
        raise SmokeError("the fixture must expose a 'default' scenario variant")
    base_name = "default"
    base_task = variants[base_name]

    evidence = _check_variant(
        base_name, base_task, builder, inferred_signal="may_differ"
    )
    base_features = evidence["features"]
    base_rankings = evidence["rankings"]
    rows = _table_rows(base_task, evidence["feature_set"], base_rankings)
    print(f"scenario: {base_name} ({base_task.task_id})")
    print(_format_table(rows))
    print()

    # -- retracted variant: no inferred signal -> A2 == A1 -------------------- #
    if "retracted" not in variants:
        raise SmokeError("the fixture must expose a 'retracted' scenario variant")
    retracted = _check_variant(
        "retracted", variants["retracted"], builder, inferred_signal="must_equal"
    )
    retracted_soft = sum(
        1
        for feature in retracted["features"]
        if feature.inferred_matches + feature.inferred_violations > 0
        or feature.inferred_ordinal_support + feature.inferred_ordinal_penalty > 0
    )
    check(
        "retracted:no_inferred_signal",
        retracted_soft == 0,
        "the retracted variant must carry no decided inferred evidence",
    )

    # -- directional variant: ordinal inferred tie-break only inside ties ----- #
    if "directional" in variants:
        _check_variant(
            "directional",
            variants["directional"],
            builder,
            inferred_signal="must_differ",
        )

    # -- determinism: rebuild features, re-rank, replay the M5 state ---------- #
    rebuilt = builder.build(
        snapshot=base_task.snapshot, ranker_projection=base_task.ranker_projection
    )
    check(
        "feature_rebuild_identical",
        rebuilt.model_dump(mode="json")
        == evidence["feature_set"].model_dump(mode="json"),
        "rebuilding the same snapshot/projection must be byte-identical",
    )
    reranked = rank_all_arms(evidence["feature_set"], base_task.snapshot)
    check(
        "repeat_run_identical",
        [(arm.arm, arm.candidate_ids) for arm in reranked]
        == [(arm.arm, arm.candidate_ids) for arm in base_rankings],
        "repeating the arms on the same features must produce identical orders",
    )
    replayed_task = fixture.replay_task(variant=base_name)
    replayed_features = builder.build(
        snapshot=replayed_task.snapshot,
        ranker_projection=replayed_task.ranker_projection,
    )
    check(
        "replayed_state_rebuild_identical",
        replayed_features.model_dump(mode="json")
        == evidence["feature_set"].model_dump(mode="json"),
        "features rebuilt from the replayed M5 for_ranker projection must be identical",
    )

    if _FAILURES:
        raise SmokeError("; ".join(_FAILURES))
    passed = len(_CHECKS)
    print(f"SMOKE: PASS, {passed}/{passed} checks")
    print(f"variants checked: {sorted(variants)}")
    print(DISCLAIMER)
    return {
        "scenario": base_name,
        "variant_count": len(variants),
        "rows": rows,
        "checks": passed,
        "retracted_no_inferred_signal": True,
        "disclaimer": DISCLAIMER,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke and return the process exit code (0 = pass)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=None, help="optional path for the evidence record"
    )
    arguments = parser.parse_args(argv)
    try:
        evidence = run_smoke()
    except SmokeError as error:
        print(f"SMOKE: FAIL - {error}")
        return 1
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"evidence written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
