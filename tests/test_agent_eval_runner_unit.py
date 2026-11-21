"""Fast runner configuration tests; no real MCP startup."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.dataset import load_dataset
from agent_eval.runner import (
    AgentEvaluationRunner,
    EvaluationRunnerConfig,
    EvaluationRunError,
    build_settings,
)


def test_build_settings_are_isolated_and_strict_has_no_sidecar(tmp_path: Path) -> None:
    demo = build_settings(tmp_path, price_mode="demo_historical_fx")
    assert demo.price_mode == "demo_historical_fx"
    assert demo.demo_price_sidecar is not None
    assert demo.web_state_db_path == tmp_path / "web-state.sqlite3"
    assert demo.memory_db_path == tmp_path / "m9.sqlite3"
    assert demo.otel_enabled is False

    strict = build_settings(tmp_path, price_mode="strict")
    assert strict.price_mode == "strict"
    assert strict.demo_price_sidecar is None
    with pytest.raises(EvaluationRunError):
        build_settings(tmp_path, price_mode="production")


def test_runner_selection_limit_and_unknown_case() -> None:
    dataset = load_dataset()
    runner = AgentEvaluationRunner(
        dataset,
        EvaluationRunnerConfig(case_ids=["hp_noise_warm", "tablet_3000"]),
    )
    selected = runner.selected_cases()
    assert [case.case_id for case in selected] == ["hp_noise_warm", "tablet_3000"]

    limited = AgentEvaluationRunner(
        dataset, EvaluationRunnerConfig(limit=2),
    ).selected_cases()
    assert [case.case_id for case in limited] == ["hp_noise_warm", "hp_noise_cold"]

    with pytest.raises(EvaluationRunError):
        AgentEvaluationRunner(
            dataset, EvaluationRunnerConfig(case_ids=["missing_case"])
        ).selected_cases()
