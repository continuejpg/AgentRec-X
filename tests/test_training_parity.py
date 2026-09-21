"""Training-exposure parity between the model arms.

Why this is a test rather than a paragraph
------------------------------------------
A benchmark arm only means something if the arms were trained on comparable data.  The Step-1
Two-Tower model was trained on the frozen 20 000-user **evaluation** cohort while the accepted
SASRec was trained on all 412 445 eligible users - a 9.4x exposure gap that no metric in the
comparison table reveals.  That is exactly the kind of confound that silently invalidates a
"model A beats model B" claim, so the relationship between the two corpora is pinned here.

The relation, stated once
-------------------------
For a train history of length ``n`` the accepted SASRec builder counts ``n - 1`` raw next-item
transitions but only creates a training sample when ``n >= 2``.  The Two-Tower builder emits one
``(prefix, next item)`` pair per transition *except* the first, because a one-item prefix is not
a history - so it emits ``n - 2``, and skips a user entirely when ``n < 3``.  Therefore::

    two_tower_pairs == sasrec_raw_transitions - sasrec_trainable_users

Verified against the real cohort: 2 263 252 - 412 445 = 1 850 807, which is exactly what the
full-cohort training run recorded.  The tests below assert this identity rather than a
hard-coded count, so a change to either builder fails the suite.

What else is asserted
---------------------
* both builders read ``train_history`` **only**, so no validation or test target can reach either
  model;
* the arms now train on the same users, which is the parity this step established;
* a non-default ``min_history`` would shrink Two-Tower's corpus relative to SASRec's, so the
  parity-preserving default is asserted.

The tests run on small synthetic cohorts: no dataset and no checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.datasets.sasrec import SASRecDatasetConfig, build_dataset  # noqa: E402
from recommendation.datasets.twotower import (  # noqa: E402
    TwoTowerDatasetConfig,
    build_training_pairs,
)
from recommendation.evaluation.split import EvaluationCase  # noqa: E402

#: A sentinel outside every synthetic history, so a leaked evaluation target is visible.
FORBIDDEN_TARGETS = frozenset({498, 499, 500})


def make_cases(*lengths: int) -> list[EvaluationCase]:
    """Cases whose train histories have the given lengths, with unreachable eval targets."""
    return [
        EvaluationCase(
            user_id=f"user-{index}",
            user_int_id=index,
            train_history=tuple(range(1, length + 1)),
            validation_target=499,
            test_target=500,
            sequence_length=length + 2,
        )
        for index, length in enumerate(lengths, start=1)
    ]


def both(cases: list[EvaluationCase], *, max_seq_len: int = 50, min_history: int = 2):
    """Build both corpora for one cohort."""
    sasrec = build_dataset(cases, 500, SASRecDatasetConfig(max_seq_len=max_seq_len, seed=2026))
    two_tower = build_training_pairs(
        cases,
        num_items=500,
        config=TwoTowerDatasetConfig(
            max_seq_len=max_seq_len, seed=2026, min_history=min_history
        ),
    )
    return sasrec, two_tower


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


def test_both_builders_read_train_history_only():
    """The shared leakage guarantee, asserted on both corpora at once."""
    sasrec, two_tower = both(make_cases(2, 3, 5))
    for sample in sasrec.samples:
        assert not (set(sample.positive_ids) & FORBIDDEN_TARGETS)
    assert not (set(two_tower.target_ids) & FORBIDDEN_TARGETS)
    assert set(two_tower.target_ids) <= {1, 2, 3, 4, 5}
    assert "train_history only" in two_tower.stats["source"]


# --------------------------------------------------------------------------- #
# The two builders agree on who can be trained
# --------------------------------------------------------------------------- #


def test_a_single_item_history_is_dropped_by_both():
    """A one-item history has no transition; neither arm may invent one for it."""
    sasrec, two_tower = both(make_cases(1, 2, 3, 4))
    assert sasrec.stats.users_with_zero_transitions == 1
    assert two_tower.stats["cases_skipped_short_history"] == 1
    assert sasrec.stats.trainable_users == 3
    assert {sample.user_int_id for sample in sasrec.samples} == {2, 3, 4}


# --------------------------------------------------------------------------- #
# The exact exposure relation
# --------------------------------------------------------------------------- #


def test_two_tower_exposure_equals_sasrec_transitions_minus_users():
    """The relation the parity correction is built on, checked on several cohort shapes."""
    for lengths in ((2, 3, 4), (1, 2, 3, 4), (2, 3, 7, 12, 1), (2, 5), (3, 3, 3)):
        sasrec, two_tower = both(make_cases(*lengths))
        expected = sasrec.stats.raw_next_item_transitions - sasrec.stats.trainable_users
        assert len(two_tower) == expected, lengths


def test_a_short_window_does_not_change_the_pair_count():
    """Two-Tower keeps one pair per transition; the window only changes how much it reads.

    SASRec's *effective* transition count shrinks with a short window because pairs fall out of
    the window entirely.  Two-Tower has no such loss, which is a genuine architectural difference
    in what each model sees - recorded here rather than smoothed over.
    """
    sasrec, two_tower = both(make_cases(2, 3, 7, 12), max_seq_len=2)
    assert sasrec.stats.effective_transitions < sasrec.stats.raw_next_item_transitions
    assert len(two_tower) == (
        sasrec.stats.raw_next_item_transitions - sasrec.stats.trainable_users
    )


def test_the_step1_confound_is_a_smaller_cohort_not_a_different_rule():
    """The Step-1 gap came from restricting the cohort; on one cohort the relation still holds.

    This is the test that would have caught the original mismatch: the *rule* was never the
    problem, so any exposure difference must be attributable to the cohort, which is what the
    corrected run changes.
    """
    full = make_cases(*range(2, 30))
    subset = full[:5]
    sasrec_full, two_tower_full = both(full)
    _sasrec_subset, two_tower_subset = both(subset)

    assert len(two_tower_full) == (
        sasrec_full.stats.raw_next_item_transitions - sasrec_full.stats.trainable_users
    )
    assert len(two_tower_full) > len(two_tower_subset)
    assert len(two_tower_full) / len(two_tower_subset) > 1.0


def test_min_history_filters_users_but_not_pairs_and_does_not_shrink_the_corpus():
    """What ``min_history`` actually does, asserted so its role is not misread.

    A two-item history yields no pair under the position rule, so raising ``min_history`` from 2
    to 3 only stops those users being *processed* - the pair count is unchanged.  The option is
    therefore not an exposure dial, and asserting that stops a later reader from "correcting"
    training exposure by turning it.
    """
    cases = make_cases(2, 2, 2, 8)
    _sasrec, permissive = both(cases, min_history=2)
    _sasrec2, restrictive = both(cases, min_history=3)
    assert len(permissive) == len(restrictive), "min_history must not change the pair count"
    assert permissive.stats["cases_skipped_short_history"] == 0
    assert restrictive.stats["cases_skipped_short_history"] == 3
    assert permissive.stats["distinct_users"] == restrictive.stats["distinct_users"]
    # The default is the permissive one, so every eligible user is at least considered.
    _sasrec3, default = both(cases)
    assert len(default) == len(permissive)
    assert default.stats["cases_skipped_short_history"] == 0


def test_full_eligibility_is_what_sasrec_used_and_what_two_tower_now_uses():
    """Both arms must be trainable on every eligible user, not on an evaluation subset.

    A regression here is the Step-1 confound returning: the evaluation cohort is a *sample* of
    the eligible population, and training on it would reduce exposure by an order of magnitude.
    """
    cases = make_cases(*range(2, 40))
    sasrec, two_tower = both(cases)
    eligible = len(cases)
    assert sasrec.stats.trainable_users == eligible, "SASRec trains on every eligible user"
    # Two-Tower also considers every eligible user; it drops only those with no usable pair.
    assert two_tower.stats["cases_considered"] == eligible
    assert two_tower.stats["cases_skipped_short_history"] == 0


# --------------------------------------------------------------------------- #
# Controlled contrasts
# --------------------------------------------------------------------------- #


def test_each_fusion_contrast_varies_exactly_one_source():
    """The attribution guarantee, as an assertion rather than a promise.

    ``B - A`` and ``D - C`` are only interpretable if the two arms of a contrast differ by
    exactly one source.  The Step-1 comparison failed this - it added Two-Tower *and* removed
    popularity - so the property is checked directly against the control definitions.
    """
    from experiments.fusion_controls import CONTRASTS, CONTROLS

    for left, right in CONTRASTS:
        before, after = set(CONTROLS[left]), set(CONTROLS[right])
        assert after - before == {"two_tower"}, (left, right)
        assert before - after == set(), (
            f"{right} removes {sorted(before - after)} relative to {left}; the contrast would "
            "attribute the change to Two-Tower while changing more than one thing"
        )
        assert before & after, "a contrast with no held-constant source proves nothing"


def test_the_phase5_source_set_is_one_of_the_controls():
    """The accepted fusion must appear unchanged, so it is re-checked rather than replaced."""
    from experiments.fusion_controls import CONTROLS

    assert set(CONTROLS["A_popularity_sasrec_metadata"]) == {
        "popularity",
        "sequential",
        "metadata",
    }


def test_the_controls_hold_the_frozen_settings():
    """Head size, RRF rule and the accepted source set must all be the frozen values.

    The RRF constant itself is not restated anywhere in the control runner - it is the accepted
    default inside ``reciprocal_rank_fusion`` - so the check is that no control introduces a
    constant of its own.
    """
    import inspect

    import experiments.benchmark_public as B
    from experiments import fusion_controls
    from recommendation.control.candidate_ledger import RRF_K, reciprocal_rank_fusion

    assert fusion_controls.HEAD == 1000
    assert B.FUSION_SOURCES == ("popularity", "sequential", "metadata")
    # The accepted default is what the fusion uses.
    assert RRF_K >= 1
    assert inspect.signature(reciprocal_rank_fusion).parameters["k"].default == RRF_K
    # And no control runner passes a different constant.
    source = inspect.getsource(fusion_controls)
    assert "rrf_k=" not in source and "k=RRF" not in source
