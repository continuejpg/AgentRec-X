"""SASRec real-artifact dataset + CPU model smoke run (Milestone 3).

Builds the SASRec training dataset from the Milestone 1.5 / 2A artifact on CPU and
runs the three forwards this milestone smoke-tests.  It performs **no** training:
no optimizer, no epoch loop, no loss step, and no recommendation-quality metric.

Usage::

    .venv/bin/python -m experiments.sasrec_smoke
    .venv/bin/python -m experiments.sasrec_smoke --json out.json

IMPORTANT: the preprocessing fixture is a 100k-record *prefix* of Sports and
Outdoors.  ``max_seq_len`` here is a development integration setting only, chosen
for speed - it is NOT a justified production/benchmark value, and none of the
numbers below are benchmark results.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDatasetConfig,
    build_dataset,
    encode_inference_history,
    test_history,
    validation_history,
)
from recommendation.evaluation import FullRankingEvaluator, build_cohort_from_artifacts  # noqa: E402
from recommendation.models.sasrec import build_model, make_score_fn, smoke_forward  # noqa: E402

DEFAULT_SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
DEFAULT_MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"

#: Development integration setting only (see the module docstring).
ENGINEERING_MAX_SEQ_LEN = 20


def run(
    sequences_path: Path,
    mappings_path: Path,
    *,
    max_seq_len: int = ENGINEERING_MAX_SEQ_LEN,
    seed: int = 20240913,
) -> dict[str, Any]:
    """Build the dataset and run the CPU smoke forwards."""
    import torch

    cases, split = build_cohort_from_artifacts(str(sequences_path), str(mappings_path))
    num_items = split.catalog_size

    config = SASRecDatasetConfig(max_seq_len=max_seq_len, seed=seed, epoch=0)
    dataset = build_dataset(cases, num_items, config)
    stats = dataset.stats.as_dict()

    # ---- structural validation of every sample --------------------------- #
    checks_samples = {
        "input_ids_in_catalog": True,
        "positive_ids_in_catalog": True,
        "negative_ids_in_catalog": True,
        "negatives_are_real_items": True,
        "negatives_outside_train_history": True,
        "left_padding_only": True,
        "input_positive_alignment": True,
        "pad_only_at_padded_positive_positions": True,
        "valid_position_counts_consistent": True,
        "negative_pad_matches_positive_pad": True,
    }
    for sample in dataset:
        history = {v for v in sample.input_ids if v != PAD_ID}
        for position, (inp, pos, neg) in enumerate(
            zip(sample.input_ids, sample.positive_ids, sample.negative_ids)
        ):
            if not (PAD_ID <= inp <= num_items):
                checks_samples["input_ids_in_catalog"] = False
            if not (PAD_ID <= pos <= num_items):
                checks_samples["positive_ids_in_catalog"] = False
            if not (PAD_ID <= neg <= num_items):
                checks_samples["negative_ids_in_catalog"] = False

            if pos == PAD_ID:
                # a padded position must stay padding in all three arrays
                if neg != PAD_ID or inp != PAD_ID:
                    checks_samples["negative_pad_matches_positive_pad"] = False
                continue

            if neg == PAD_ID:
                checks_samples["negatives_are_real_items"] = False
            if neg in history:
                checks_samples["negatives_outside_train_history"] = False
            # Alignment: the arrays are shifts of one sequence, so the positive at
            # position k equals the input at the next real position.  At the last
            # real position there is no following input - that positive is the
            # history's final item, which is expected and not a misalignment.
            if position + 1 < len(sample.input_ids):
                following = sample.input_ids[position + 1]
                if following != PAD_ID and following != pos:
                    checks_samples["input_positive_alignment"] = False

        # left padding: PADs form a prefix, never interleaved with real items
        first_real = next(
            (i for i, v in enumerate(sample.input_ids) if v != PAD_ID), len(sample.input_ids)
        )
        if any(v != PAD_ID for v in sample.input_ids[:first_real]):
            checks_samples["left_padding_only"] = False
        if any(v == PAD_ID for v in sample.input_ids[first_real:]):
            checks_samples["left_padding_only"] = False

        # a real positive must always sit at or after the first real input
        real_positive_positions = [
            i for i, v in enumerate(sample.positive_ids) if v != PAD_ID
        ]
        if real_positive_positions and min(real_positive_positions) < first_real - 1:
            checks_samples["pad_only_at_padded_positive_positions"] = False
        if sample.num_valid_positions != len(real_positive_positions) or len(
            real_positive_positions
        ) != sum(1 for v in sample.valid_mask if v):
            checks_samples["valid_position_counts_consistent"] = False

    # ---- determinism: same seed -> identical digest ---------------------- #
    rebuilt = build_dataset(cases, num_items, config)
    other_seed = build_dataset(cases, num_items, SASRecDatasetConfig(max_seq_len=max_seq_len, seed=seed + 1, epoch=0))

    structural_fields_equal = all(
        (a.user_int_id, a.input_ids, a.positive_ids) == (b.user_int_id, b.input_ids, b.positive_ids)
        for a, b in zip(dataset.samples, other_seed.samples)
    )
    negatives_changed = sum(
        1 for a, b in zip(dataset.samples, other_seed.samples) if a.negative_ids != b.negative_ids
    )

    # ---- CPU model smoke ------------------------------------------------- #
    model = build_model(num_items=num_items, seed=seed, max_seq_len=max_seq_len,
                        hidden_size=32, num_blocks=2, num_heads=2, dropout=0.1)
    smoke = smoke_forward(model, batch_size=4, seq_len=max_seq_len)

    # one full-catalog pass per inference mode, using real cases
    evaluator = FullRankingEvaluator(num_items=num_items, k_values=(5, 10, 20))
    score_fn = make_score_fn(model, max_seq_len, num_items)
    sample_cases = cases[:8]
    validation_scores = [
        score_fn(validation_history(c), c.validation_target) for c in sample_cases
    ]
    test_scores = [score_fn(test_history(c), c.test_target) for c in sample_cases]

    def _finite_vectors(vectors: list[list[float]]) -> bool:
        return all(len(v) == num_items + 1 and all(x == x and abs(x) != float("inf") for x in v) for v in vectors)

    # evaluator contract: the model's vectors must pass the frozen validation
    evaluator.validate_scores(validation_scores[0])
    evaluator.validate_scores(test_scores[0])

    checks = dict(checks_samples)
    checks.update(
        {
            "deterministic_digest_for_same_seed": dataset.digest() == rebuilt.digest(),
            "structure_invariant_across_seeds": structural_fields_equal,
            "model_outputs_finite": bool(
                smoke.extra["hidden_finite"]
                and smoke.extra["positive_logits_finite"]
                and smoke.extra["negative_logits_finite"]
                and smoke.extra["scores_finite"]
                and smoke.extra["pad_score_finite"]
            ),
            "validation_score_vectors_valid": _finite_vectors(validation_scores),
            "test_score_vectors_valid": _finite_vectors(test_scores),
        }
    )

    return {
        "artifact": {
            "sequences": str(sequences_path),
            "mappings": str(mappings_path),
            "catalog_size": num_items,
        },
        "cohort": split.as_dict(),
        "dataset": stats,
        "dataset_digest": dataset.digest(),
        "seed_variation": {
            "negatives_changed_for_different_seed": negatives_changed,
            "samples": len(dataset),
            "structural_fields_unchanged": structural_fields_equal,
        },
        "model_smoke": smoke.as_dict(),
        "encoding": {
            "max_seq_len": max_seq_len,
            "example_validation_encoding": list(
                encode_inference_history(validation_history(cases[0]), max_seq_len, num_items)
            ),
            "example_test_encoding": list(
                encode_inference_history(test_history(cases[0]), max_seq_len, num_items)
            ),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "note": (
            "Engineering smoke metrics on a non-representative 100k prefix fixture. "
            "max_seq_len here is a development integration setting, not a benchmark choice. "
            "No training was performed and no recommendation-quality metric is reported."
        ),
    }


def print_report(payload: dict[str, Any]) -> None:
    """Render the smoke summary as readable text."""
    cohort, stats = payload["cohort"], payload["dataset"]
    print("=" * 78)
    print("SASRec real-artifact dataset + CPU model smoke (Milestone 3)")
    print("NO TRAINING, NO OPTIMIZER, NO QUALITY METRICS.")
    print("Engineering fixture: 100k prefix, not a benchmark dataset.")
    print("=" * 78)
    print()
    print("Users")
    print(f"  preprocessing users                : {cohort['num_users_total']}")
    print(f"  evaluation users                   : {cohort['num_users_eligible']}")
    print(f"  excluded from evaluation (len < 3) : {cohort['num_users_excluded']}")
    print(f"  trainable SASRec users             : {stats['trainable_users']}")
    print(f"  evaluation users w/ 1 train item   : {stats['users_with_one_train_item']}")
    print(f"  users with zero train transitions  : {stats['users_with_zero_transitions']}")
    print()
    print("Transitions")
    print(f"  raw train interactions             : {stats['raw_train_interactions']}")
    print(f"  raw next-item transitions          : {stats['raw_next_item_transitions']}")
    print(f"  sum(len(train_history) - 1)        : {stats['raw_next_item_transitions']} (must match)")
    print(f"  engineering max_seq_len            : {stats['max_seq_len']}  (development setting only)")
    print(f"  effective transitions after window : {stats['effective_transitions']}")
    print(f"  users truncated by max_seq_len     : {stats['users_truncated']}")
    print(f"  train interactions in window       : {stats['train_interactions_in_window']} "
          f"({100 * stats['interaction_retention']:.2f}% retained)")
    print(f"  transition retention               : {100 * stats['transition_retention']:.2f}%")
    print()
    print("Model smoke (CPU, eval mode, no optimizer)")
    smoke = payload["model_smoke"]
    print(f"  parameters                         : {smoke['config']['hidden_size']}h x {smoke['config']['num_blocks']} blocks (see config in --json output)")
    print(f"  hidden shape                       : {smoke['hidden_shape']}")
    print(f"  positive logits shape              : {smoke['positive_logits_shape']}")
    print(f"  negative logits shape              : {smoke['negative_logits_shape']}")
    print(f"  full-catalog scores shape          : {smoke['scores_shape']}")
    print(f"  all finite                         : {smoke['scores_finite']} (PAD finite: {smoke['pad_score_finite']})")
    print(f"  forward time                       : {smoke['timings']['combined_forward'] * 1000:.1f} ms")
    print()
    print(f"  dataset digest (seed={stats['seed']})           : {payload['dataset_digest'][:32]}")
    print(f"  negatives changed w/ different seed: "
          f"{payload['seed_variation']['negatives_changed_for_different_seed']} / "
          f"{payload['seed_variation']['samples']} samples")
    print()
    print("Checks")
    for name, ok in payload["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    print(f"ALL CHECKS PASSED: {payload['all_checks_passed']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SASRec dataset + model smoke run (Milestone 3)")
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--max-seq-len", type=int, default=ENGINEERING_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=20240913)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.sequences.exists() or not args.mappings.exists():
        print(f"missing artifacts:\n  {args.sequences}\n  {args.mappings}", file=sys.stderr)
        return 2

    payload = run(args.sequences, args.mappings, max_seq_len=args.max_seq_len, seed=args.seed)
    print_report(payload)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
