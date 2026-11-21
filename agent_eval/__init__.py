"""M3-B task-level Golden Dataset and real Web Agent evaluation."""

from agent_eval.dataset import (
    GOLDEN_DATASET_SCHEMA,
    GOLDEN_DATASET_VERSION,
    GoldenCase,
    GoldenDataset,
    load_dataset,
)
from agent_eval.metrics import compute_metrics, evaluate_case
from agent_eval.runner import AgentEvaluationRunner, EvaluationRunnerConfig, build_settings

__all__ = [
    "GOLDEN_DATASET_SCHEMA",
    "GOLDEN_DATASET_VERSION",
    "AgentEvaluationRunner",
    "EvaluationRunnerConfig",
    "GoldenCase",
    "GoldenDataset",
    "build_settings",
    "compute_metrics",
    "evaluate_case",
    "load_dataset",
]
