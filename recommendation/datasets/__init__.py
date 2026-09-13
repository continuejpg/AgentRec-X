"""Dataset builders that turn the frozen evaluation cohort into model inputs.

Milestone 3 provides the SASRec training dataset and inference encoder.  Datasets
live here; models live in :mod:`recommendation.models`; the evaluation protocol
lives in :mod:`recommendation.evaluation` and is never reimplemented.
"""

from __future__ import annotations

from .sasrec import (
    MIN_TRAIN_HISTORY_FOR_TRANSITION,
    PAD_ID,
    SASRecDataError,
    SASRecDataset,
    SASRecDatasetConfig,
    SASRecDatasetStats,
    SASRecSample,
    build_arrays,
    build_dataset,
    build_sample,
    cohort_structure_digest,
    encode_batch,
    encode_inference_history,
    sample_negative,
    test_history,
    validation_history,
)

__all__ = [
    "MIN_TRAIN_HISTORY_FOR_TRANSITION",
    "PAD_ID",
    "SASRecDataError",
    "SASRecDataset",
    "SASRecDatasetConfig",
    "SASRecDatasetStats",
    "SASRecSample",
    "build_arrays",
    "build_dataset",
    "build_sample",
    "cohort_structure_digest",
    "encode_batch",
    "encode_inference_history",
    "sample_negative",
    "test_history",
    "validation_history",
]

__version__ = "0.1.0"
