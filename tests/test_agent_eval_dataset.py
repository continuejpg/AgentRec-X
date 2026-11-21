"""Golden Dataset schema and reproducibility tests (no Agent startup)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.dataset import (
    CATEGORIES,
    DEFAULT_DATASET_PATH,
    DatasetValidationError,
    GoldenDataset,
    dataset_sha256,
    load_dataset,
)


def test_default_dataset_is_versioned_valid_and_complete() -> None:
    dataset = load_dataset()
    assert dataset.dataset_schema == "agentrecx.golden.agent_tasks.v1"
    assert dataset.dataset_version == "1.0.0"
    assert len(dataset.cases) == 15
    assert {case.category for case in dataset.cases} == CATEGORIES
    assert len({case.case_id for case in dataset.cases}) == 15
    assert all(case.dataset_version == dataset.dataset_version for case in dataset.cases)
    assert all(case.expected_requirements is not None for case in dataset.cases)
    assert dataset.labeling_policy.startswith("Binary relevance labels")
    assert len(dataset_sha256()) == 64


def test_relevance_labels_are_manual_binary_and_not_ranker_outputs() -> None:
    dataset = load_dataset()
    judged = [case for case in dataset.cases if case.relevance is not None]
    assert len(judged) >= 3
    for case in judged:
        labels = case.relevance.judged_ids
        assert len(labels) >= 5
        assert set(labels.values()) <= {0, 1}
        assert case.relevance.note
    # The dataset has explicit labels for the retrieval cases; the runner must use these,
    # not derive labels from the live BM25 result.
    assert {case.case_id for case in judged} >= {
        "hp_noise_warm", "laptop_update_warm", "laptop_direct_8000",
    }


def test_dataset_rejects_duplicate_case_ids() -> None:
    raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    raw["cases"][1]["case_id"] = raw["cases"][0]["case_id"]
    with pytest.raises(Exception):
        GoldenDataset.model_validate(raw)


def test_dataset_rejects_too_few_cases() -> None:
    raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    raw["cases"] = raw["cases"][:14]
    with pytest.raises(Exception):
        GoldenDataset.model_validate(raw)


def test_dataset_rejects_non_binary_relevance() -> None:
    raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    case = next(item for item in raw["cases"] if item["relevance"])
    first_id = next(iter(case["relevance"]["judged_ids"]))
    case["relevance"]["judged_ids"][first_id] = 2
    with pytest.raises(Exception):
        GoldenDataset.model_validate(raw)


def test_loader_reports_bad_json_and_wrong_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(bad)

    raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    raw["dataset_version"] = "0.9.0"
    # Cases still carry 1.0.0, so the collection validator must reject the mixture.
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(wrong)


def test_every_task_has_expected_outcome_and_requirements() -> None:
    dataset = load_dataset()
    for case in dataset.cases:
        assert case.purpose
        assert case.user_query
        assert case.expected_outcome.kind in {"recommendation", "refusal", "clarification"}
        if case.expected_outcome.kind == "refusal":
            assert case.expected_outcome.refusal_codes
        if case.execution_mode == "direct":
            assert case.clarification_reply is None
        else:
            assert case.clarification_reply
