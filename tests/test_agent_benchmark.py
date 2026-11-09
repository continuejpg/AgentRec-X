from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent.adapters import LLMPreferenceAdapter, build_adapter
from benchmarks.agent.adapters import _resolve_option
from benchmarks.agent.evaluator import evaluate_run, load_human_labels
from benchmarks.agent.report import write_reports
from benchmarks.agent.schemas import BenchTask, PolicyId, TaskRun, TurnTrace
from benchmarks.agent.tasks import (
    TASKS_DIR,
    load_tasks,
    validate_tasks,
    verify_manifest,
)


class _Record:
    def __init__(self, title: str, category: str = "Sports & Outdoors") -> None:
        self.title = title
        self.categories = ("Sports & Outdoors", category)


def _fake_catalogue(tasks) -> dict[str, _Record]:
    records: dict[str, _Record] = {}
    final_terms: dict[str, set[str]] = {}
    for task in tasks:
        identities = set(task.history) | set(task.expected.final_any) | set(task.expected.must_exclude_ids)
        for identity in identities:
            records.setdefault(identity, _Record("history product"))
        for identity in task.expected.final_any:
            final_terms.setdefault(identity, set()).update(task.expected.required_title_terms_any)
    for identity, terms in final_terms.items():
        records[identity] = _Record(" ".join(sorted(terms)) + " product")
    return records


def test_task_splits_are_valid_real_schema_and_manifest() -> None:
    dev = load_tasks("dev")
    frozen = load_tasks("frozen")
    assert len(dev) == 20
    assert len(frozen) == 8
    all_tasks = (*dev, *frozen)
    records = _fake_catalogue(all_tasks)
    validate_tasks(dev, records, expected_split="dev", expected_counts={"dev": 20})
    validate_tasks(frozen, records, expected_split="frozen", expected_counts={"frozen": 8})
    verified = verify_manifest(TASKS_DIR)
    assert set(verified) == {"dev.json", "frozen.json"}
    for task in all_tasks:
        assert 3 <= len(task.turns) <= 5
        assert task.expected.final_any
        assert "expected" not in task.policy_input
        assert "turns" in task.policy_input
        assert not hasattr(task.policy_task, "expected")
        assert task.split in {"dev", "frozen"}


def test_policy_arms_do_not_invent_llm_availability() -> None:
    a = build_adapter(PolicyId.A, service=object())
    b = build_adapter(PolicyId.B, service=object())
    c = build_adapter(PolicyId.C, service=object())
    d = build_adapter(PolicyId.D, service=object())
    assert a.availability() == "llm_not_configured"
    assert b.availability() == "llm_not_configured"
    assert c.availability() is None
    assert d.availability() == "llm_not_configured"


def test_evaluator_uses_deterministic_product_role_checks() -> None:
    task = load_tasks("dev")[0]
    records = _fake_catalogue([task])
    good = next(iter(task.expected.final_any))
    run = TaskRun(
        task_id=task.task_id,
        policy=PolicyId.C,
        runnable=True,
        status="completed",
        turns=[TurnTrace(0, task.turns[-1].user_input, task.turns[-1].user_input, response_status="finished")],
        final_ids=(good,),
        retrieval_actions=("RERANK",),
    )
    evaluation = evaluate_run(run, task, records)
    assert evaluation.passed is True
    assert evaluation.human_label is None

    bad = replace(run, final_ids=(task.expected.must_exclude_ids[0],))
    failed = evaluate_run(bad, task, records)
    assert failed.passed is False
    assert any("forbidden" in failure for failure in failed.failures)


def test_pending_option_resolver_does_not_choose_none_as_a_fallback() -> None:
    options = (
        SimpleNamespace(option_id="pref:weight:lower", action="lower", value=None, label="Lower weight"),
        SimpleNamespace(option_id="pref:none", action="none", value=None, label="No additional preference"),
    )
    clarification = SimpleNamespace(
        preference=SimpleNamespace(options=options),
        research_options=(),
    )
    response = SimpleNamespace(clarification=clarification)
    assert _resolve_option(response, "I prefer the lighter option.") == "pref:weight:lower"
    assert _resolve_option(response, "forget the weight preference") is None
    assert _resolve_option(response, "No additional preference.") == "pref:none"


def test_human_label_file_is_loaded_separately(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"task_id":"dev_01_x","human_relevance":"2","note":"ok"}\n', encoding="utf-8")
    labels = load_human_labels(path)
    assert labels["dev_01_x"]["human_relevance"] == "2"
    assert load_human_labels(None) == {}


class _FakeResponse:
    recommendations = ()
    terminal = SimpleNamespace(status="finished")
    clarification = None
    grounded = SimpleNamespace(candidates=())
    observation = None
    research = None


class _FakeService:
    _task_preferences: dict = {}
    _memory_service = None

    def __init__(self) -> None:
        self.payloads = []

    def run_shopping_turn(self, payload, **kwargs):
        self.payloads.append((payload, kwargs))
        return _FakeResponse()

    def r1_diagnostics(self, task_id):
        return None

    def state_diagnostics(self, task_id):
        return None


class _FakeProposer:
    def propose(self, *, messages, current):
        return (
            "faithful rewritten query",
            [{"attribute": "price", "action": "add", "value": "30", "evidence": current}],
            1,
            2,
        )


def test_d_proposals_are_recorded_but_only_rewrite_reaches_the_request() -> None:
    context = SimpleNamespace(
        service=_FakeService(),
        rewrite_client=None,
        proposal_client=object(),
        state_db_path=None,
    )
    adapter = LLMPreferenceAdapter(context, proposer=_FakeProposer())
    task = load_tasks("dev")[0].policy_task
    run = adapter.run(task)
    assert run.runnable is True
    payload, kwargs = context.service.payloads[0]
    assert payload.message == "faithful rewritten query"
    assert payload.constraints == ()
    assert kwargs["session_id"].startswith("bench:D:")
    assert run.turns[0].llm_proposals == (
        {"attribute": "price", "action": "add", "value": "30", "evidence": task.turns[0].user_input},
    )



def test_report_writes_required_artifacts(tmp_path: Path) -> None:
    task = load_tasks("dev")[0]
    records = _fake_catalogue([task])
    run = TaskRun(
        task_id=task.task_id,
        policy=PolicyId.C,
        runnable=True,
        status="completed",
        turns=[TurnTrace(0, "q", "q", response_status="finished")],
        final_ids=(next(iter(task.expected.final_any)),),
        retrieval_actions=("RERANK",),
    )
    evaluation = evaluate_run(run, task, records)
    paths = write_reports(tmp_path, runs=[run], evaluations=[evaluation], tasks=[task])
    assert paths["trajectories"].is_file()
    assert paths["evaluations"].is_file()
    assert paths["metrics"].is_file()
    assert paths["summary"].is_file()
    assert paths["failures"].is_file()
    assert "human relevance" in paths["summary"].read_text(encoding="utf-8")
