"""Focused tests for the M6.2 CoShop external-benchmark adapter (offline lab only).

Everything here runs without network access, without CoShop installed, without pandas and
without the AgentRec-X serving/model stack:

* the external CoShop checkout is replaced by duck-typed fake dataset/spec/catalog objects;
* the official metric implementation is replaced by an injected fake metrics function;
* import isolation is asserted in a subprocess in which CoShop, pandas, torch and every
  AgentRec-X package must remain unimported.

The leakage/isolation assertions in this file are part of the milestone contract and must
never be weakened: hidden CoShop targets must not appear in the public task view, the
candidate snapshot, the ranking callable inputs or the report payload.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.external_benchmarks.coshop import adapter as coshop_adapter  # noqa: E402
from experiments.external_benchmarks.coshop.adapter import (  # noqa: E402
    ADAPTER_VERSION,
    BASELINE_NAME,
    CANDIDATE_SOURCE,
    CUSTOM_DIAGNOSTICS_LABEL,
    ENV_COSHOP_ROOT,
    OFFICIAL_METRICS_SOURCE,
    PROTOCOL_NAME,
    SNAPSHOT_SOURCE,
    CoshopContractError,
    CoshopTaskView,
    CoshopUnavailable,
    HiddenTaskTargets,
    build_task_view,
    evaluate_coshop,
    extract_hidden_targets,
    rank_baseline_candidates,
    resolve_coshop_root,
    to_candidate_snapshot,
)

ADAPTER_PATH = (
    REPO_ROOT / "experiments" / "external_benchmarks" / "coshop" / "adapter.py"
)
PACKAGE_INIT_PATH = (
    REPO_ROOT / "experiments" / "external_benchmarks" / "coshop" / "__init__.py"
)

#: Official catalog for the fake benchmark: 20 movie ids in catalog order.
CATALOG_IDS: tuple[str, ...] = tuple(f"movie-{position:03d}" for position in range(1, 21))

#: Target ids deliberately chosen to be absent from the candidate pool and catalog, so a
#: leak anywhere becomes detectable.
TARGET_NOT_IN_POOL = "target-secret-777"
SECOND_TARGET_NOT_IN_POOL = "target-secret-888"

#: Amazon / legacy AgentRec-X identity samples that must never appear in this benchmark.
AMAZON_IDENTITIES = frozenset({"B0M6HIKE01", "B0M6HIKE05", "cand-red", "cand-nometa"})


# --------------------------------------------------------------------------- #
# Duck-typed official objects (dataset / spec / catalog) and metric fakes
# --------------------------------------------------------------------------- #


class FakeCatalog:
    """Duck-typed official catalog: only an ordered ``index`` is needed."""

    def __init__(self, ids: Sequence[str]) -> None:
        self.index = tuple(ids)


class FakeSpec:
    """Duck-typed official specification with the visible fields the adapter reads."""

    def __init__(
        self,
        *,
        z0: str | None = "I want a science fiction movie with a strong story",
        initial_known_features: Sequence[str] = (),
        z0_baseline_ids: Sequence[str] | None = (),
        xstar: Sequence[str] | None = (),
    ) -> None:
        self.z0 = z0
        self.initial_known_features = list(initial_known_features)
        # ``None`` models an official payload that simply does not expose the field; the
        # adapter must reject that as a contract failure rather than invent candidates.
        self.z0_baseline_ids = None if z0_baseline_ids is None else list(z0_baseline_ids)
        self.xstar = None if xstar is None else list(xstar)


class FakeDataset:
    """Duck-typed official dataset exposing ``catalog``, ``__getitem__`` and ``__len__``."""

    def __init__(self, catalog_ids: Sequence[str], specs: Sequence[Any]) -> None:
        self.catalog = FakeCatalog(catalog_ids)
        self._specs = list(specs)

    def __getitem__(self, index: int) -> Any:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)


def make_dataset_factory(
    catalog_ids: Sequence[str] = CATALOG_IDS,
    specs: Sequence[Any] | None = None,
):
    """Return an injected dataset factory over duck-typed official objects."""
    resolved_specs = list(specs) if specs is not None else standard_specs()

    def factory(*, domain: str, version: str, split: str) -> FakeDataset:
        return FakeDataset(catalog_ids, resolved_specs)

    return factory


def standard_specs() -> list[FakeSpec]:
    """Three deterministic fake official tasks (targets never in the candidate pool)."""
    return [
        FakeSpec(
            z0="I want a science fiction movie with a strong story",
            initial_known_features=["genre"],
            z0_baseline_ids=["movie-003", "movie-005", "movie-001", "movie-003"],
            xstar=[TARGET_NOT_IN_POOL],
        ),
        FakeSpec(
            z0=None,
            initial_known_features=["genre", "year"],
            z0_baseline_ids=["movie-010", "movie-002"],
            xstar=[SECOND_TARGET_NOT_IN_POOL],
        ),
        FakeSpec(
            z0="Something light for the evening",
            initial_known_features=[],
            z0_baseline_ids=["movie-008", "movie-009"],
            xstar=[TARGET_NOT_IN_POOL],
        ),
    ]


class _FakePredictedEvaluation:
    def __init__(self, *, recall_at_k: float, recall_at_1: float, ndcg_ustar_at_k: float):
        self.recall_at_k = recall_at_k
        self.recall_at_1 = recall_at_1
        self.ndcg_ustar_at_k = ndcg_ustar_at_k


class _FakeEvaluation:
    def __init__(self, predicted: _FakePredictedEvaluation) -> None:
        self.predicted = predicted


class RecordingMetrics:
    """Injected fake of ``compute_evaluation_metrics``; records every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeEvaluation:
        self.calls.append(dict(kwargs))
        spec = kwargs["spec"]
        ranked = list(kwargs["ranked_item_ids"])
        k = int(kwargs["k"])
        targets = set(getattr(spec, "xstar", ()) or ())
        recall_at_k = 1.0 if any(item in targets for item in ranked[:k]) else 0.0
        recall_at_1 = 1.0 if ranked and ranked[0] in targets else 0.0
        return _FakeEvaluation(
            _FakePredictedEvaluation(
                recall_at_k=recall_at_k,
                recall_at_1=recall_at_1,
                ndcg_ustar_at_k=0.5,
            )
        )


def _evaluate(**overrides: Any):
    """Run one evaluation over the standard fake official objects."""
    kwargs: dict[str, Any] = {
        "limit": 2,
        "k": 5,
        "dataset_factory": make_dataset_factory(),
        "metrics_function": RecordingMetrics(),
    }
    kwargs.update(overrides)
    return evaluate_coshop(**kwargs)


# --------------------------------------------------------------------------- #
# Import isolation and deletion safety
# --------------------------------------------------------------------------- #


def test_package_import_does_not_import_coshop_or_agentrecx_stack() -> None:
    """Importing the adapter must import neither CoShop nor torch/AgentRec-X pandas."""
    code = (
        "import json, sys\n"
        "import experiments.external_benchmarks.coshop\n"
        "forbidden = []\n"
        "for name in sorted(sys.modules):\n"
        "    if name == 'coshop' or name.startswith('coshop.'):\n"
        "        forbidden.append(name); continue\n"
        "    if name.split('.')[0] in {'torch', 'pandas', 'numpy', 'recommendation'}:\n"
        "        forbidden.append(name)\n"
        "print(json.dumps(forbidden))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == []


def test_adapter_module_import_surface_is_stdlib_only() -> None:
    """No top-level import of CoShop, pandas, torch or any AgentRec-X module."""
    forbidden_roots = {"coshop", "pandas", "torch", "numpy", "recommendation"}
    for path in (ADAPTER_PATH, PACKAGE_INIT_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level_modules: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.append(node.module)
        for module in top_level_modules:
            assert module.split(".")[0] not in forbidden_roots, (
                f"top-level import {module!r} in {path.name} breaks import isolation"
            )


def test_core_packages_have_no_reverse_dependency_on_the_adapter() -> None:
    """No file under recommendation/, agent/ or api/ imports the external benchmark."""
    pattern = re.compile(
        r"^\s*(?:from|import)\s+(?:experiments\.external_benchmarks|coshop)(?:[.\s]|$)",
        re.MULTILINE,
    )
    roots = [
        root
        for root in (REPO_ROOT / "recommendation", REPO_ROOT / "agent", REPO_ROOT / "api")
        if root.is_dir()
    ]
    assert roots, "the core packages must exist for the reverse-dependency scan"
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


# --------------------------------------------------------------------------- #
# External checkout / dependency failures -> typed NOT EXECUTED verdicts
# --------------------------------------------------------------------------- #


def test_missing_root_is_unavailable(monkeypatch) -> None:
    monkeypatch.delenv(ENV_COSHOP_ROOT, raising=False)
    with pytest.raises(CoshopUnavailable, match="AGENTRECX_COSHOP_ROOT|not configured"):
        evaluate_coshop()


def test_checkout_without_official_layout_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_COSHOP_ROOT, raising=False)
    plain_dir = tmp_path / "not-a-coshop-checkout"
    plain_dir.mkdir()
    with pytest.raises(CoshopUnavailable, match=r"coshop/data/dataset\.py"):
        evaluate_coshop(coshop_root=plain_dir)


def test_env_var_checkout_is_honoured(tmp_path, monkeypatch) -> None:
    fake_root = tmp_path / "coshop-main"
    (fake_root / "coshop" / "data").mkdir(parents=True)
    (fake_root / "coshop" / "data" / "dataset.py").write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_COSHOP_ROOT, str(fake_root))
    assert resolve_coshop_root() == fake_root.resolve()

    explicit_root = tmp_path / "explicit-coshop"
    (explicit_root / "coshop" / "data").mkdir(parents=True)
    (explicit_root / "coshop" / "data" / "dataset.py").write_text("", encoding="utf-8")
    # An explicit argument always wins over the environment variable.
    assert resolve_coshop_root(explicit_root) == explicit_root.resolve()


def test_broken_external_checkout_import_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_COSHOP_ROOT, raising=False)
    broken = tmp_path / "broken-coshop"
    (broken / "coshop" / "data").mkdir(parents=True)
    (broken / "coshop" / "__init__.py").write_text("", encoding="utf-8")
    (broken / "coshop" / "data" / "dataset.py").write_text("", encoding="utf-8")
    with pytest.raises(CoshopUnavailable, match="could not be imported"):
        evaluate_coshop(limit=1, coshop_root=broken)


def test_dataset_factory_failure_is_wrapped_as_unavailable() -> None:
    def failing_factory(*, domain: str, version: str, split: str) -> Any:
        raise RuntimeError("Google Drive asset download timed out")

    with pytest.raises(CoshopUnavailable, match="dataset factory failed") as excinfo:
        evaluate_coshop(
            limit=1,
            dataset_factory=failing_factory,
            metrics_function=RecordingMetrics(),
        )
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_invalid_arguments_raise_value_error() -> None:
    factory = make_dataset_factory()
    metrics = RecordingMetrics()
    for overrides in (
        {"domain": "amazon"},
        {"version": "v3"},
        {"split": "validation"},
        {"limit": 0},
        {"k": 0},
        {"limit": "3"},
    ):
        with pytest.raises(ValueError):
            _evaluate(dataset_factory=factory, metrics_function=metrics, **overrides)


# --------------------------------------------------------------------------- #
# Public task view: visible query + visible candidate pool only
# --------------------------------------------------------------------------- #


def test_task_view_is_public_and_target_free() -> None:
    spec = FakeSpec(
        z0="I want a science fiction movie",
        initial_known_features=["genre"],
        z0_baseline_ids=["movie-003", "movie-005"],
        xstar=[TARGET_NOT_IN_POOL],
    )
    view = build_task_view(
        spec=spec, domain="movielens", version="v2", split="test", task_index=7
    )
    assert isinstance(view, CoshopTaskView)
    assert view.task_id == "coshop:movielens:v2:spec-000007"
    assert view.user_key == "coshop:movielens:v2:spec-000007:user"
    assert view.turn_key == "coshop:movielens:v2:spec-000007:turn-1"
    assert view.visible_query == "I want a science fiction movie"
    assert view.visible_query_source == "spec.z0"
    assert view.candidate_ids == ("movie-003", "movie-005")
    payload = view.public_payload()
    assert TARGET_NOT_IN_POOL not in json.dumps(payload)
    assert "xstar" not in payload
    assert "zstar" not in payload
    assert TARGET_NOT_IN_POOL not in repr(view)


def test_visible_query_falls_back_to_initial_known_features() -> None:
    spec = FakeSpec(
        z0=None,
        initial_known_features=["genre", "year"],
        z0_baseline_ids=["movie-002"],
        xstar=[TARGET_NOT_IN_POOL],
    )
    view = build_task_view(
        spec=spec, domain="hm", version="v1", split="test", task_index=3
    )
    assert "genre" in view.visible_query and "year" in view.visible_query
    assert view.visible_query_source == "spec.initial_known_features"
    assert TARGET_NOT_IN_POOL not in json.dumps(view.public_payload())

    empty = build_task_view(
        spec=FakeSpec(z0=None, initial_known_features=[], z0_baseline_ids=["movie-002"]),
        domain="hm",
        version="v1",
        split="test",
        task_index=4,
    )
    assert empty.visible_query == ""
    assert empty.visible_query_source == "none"


def test_public_task_view_never_reads_the_hidden_target() -> None:
    class TrapSpec:
        """A spec whose hidden target raises if a public view touches it."""

        z0 = "Some visible query"
        initial_known_features: list[str] = []
        z0_baseline_ids = ["movie-001", "movie-002"]

        @property
        def xstar(self):  # pragma: no cover - must never be called
            raise AssertionError("the public task view read the hidden target")

    spec = TrapSpec()
    view = build_task_view(
        spec=spec, domain="movielens", version="v2", split="test", task_index=0
    )
    snapshot = to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))
    assert view.candidate_ids == ("movie-001", "movie-002")
    assert snapshot.candidate_ids == view.candidate_ids


def test_empty_or_missing_baseline_is_a_contract_error() -> None:
    with pytest.raises(CoshopContractError, match="empty"):
        build_task_view(
            spec=FakeSpec(z0_baseline_ids=[]),
            domain="movielens",
            version="v2",
            split="test",
            task_index=0,
        )
    with pytest.raises(CoshopContractError, match="no visible-query baseline"):
        build_task_view(
            spec=FakeSpec(z0_baseline_ids=None),  # type: ignore[arg-type]
            domain="movielens",
            version="v2",
            split="test",
            task_index=0,
        )


def test_missing_or_empty_hidden_target_is_a_contract_error() -> None:
    with pytest.raises(CoshopContractError, match="empty"):
        extract_hidden_targets(spec=FakeSpec(xstar=[]), task_id="coshop:test")
    with pytest.raises(CoshopContractError, match="no hidden ground-truth target"):
        extract_hidden_targets(spec=FakeSpec(xstar=None), task_id="coshop:test")


def test_hidden_targets_are_sealed() -> None:
    hidden = extract_hidden_targets(
        spec=FakeSpec(z0_baseline_ids=["movie-001"], xstar=[TARGET_NOT_IN_POOL]),
        task_id="coshop:movielens:v2:spec-000000",
    )
    assert isinstance(hidden, HiddenTaskTargets)
    assert hidden.target_count == 1
    assert hidden.target_in_pool(["movie-001", "movie-002"]) is False
    assert hidden.target_in_pool(["movie-001", TARGET_NOT_IN_POOL]) is True
    assert hidden.target_best_rank(["movie-001", TARGET_NOT_IN_POOL, "movie-002"]) == 2
    assert hidden.target_best_rank(["movie-001", "movie-002"]) is None

    public = hidden.public_payload()
    assert set(public) == {"task_id", "target_count"}
    assert TARGET_NOT_IN_POOL not in json.dumps(public)
    assert TARGET_NOT_IN_POOL not in repr(hidden)
    # No public attribute hands out the raw target ids.
    assert not hasattr(hidden, "targets")
    assert not hasattr(hidden, "xstar")


def test_no_candidate_is_synthesized_from_the_hidden_target() -> None:
    view = build_task_view(
        spec=FakeSpec(
            z0_baseline_ids=["movie-002", "movie-004"],
            xstar=[TARGET_NOT_IN_POOL],
        ),
        domain="movielens",
        version="v2",
        split="test",
        task_index=0,
    )
    snapshot = to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))
    assert view.candidate_ids == ("movie-002", "movie-004")
    assert snapshot.candidate_ids == ("movie-002", "movie-004")
    assert TARGET_NOT_IN_POOL not in json.dumps(snapshot.model_dump())

    report = _evaluate(limit=1)
    assert report.tasks[0].ranked_candidate_ids == ("movie-003", "movie-005", "movie-001")
    assert report.custom_diagnostics["per_task"][0]["target_in_candidate_pool"] is False


# --------------------------------------------------------------------------- #
# AgentRec-X offline contract conversion
# --------------------------------------------------------------------------- #


def test_candidate_snapshot_conversion_contract() -> None:
    spec = FakeSpec(
        z0="I want a science fiction movie",
        z0_baseline_ids=["movie-003", "movie-005", "movie-001", "movie-003"],
        xstar=["movie-005"],
    )
    view = build_task_view(
        spec=spec, domain="movielens", version="v2", split="test", task_index=0
    )
    snapshot = to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))

    assert type(snapshot).__name__ == "CandidateSnapshot"
    # Candidate set equals the official baseline ids, deduplicated preserving order.
    assert snapshot.candidate_ids == ("movie-003", "movie-005", "movie-001")
    assert snapshot.candidate_count == 3
    assert snapshot.task_id == view.task_id
    assert snapshot.user_key == view.user_key
    assert snapshot.turn_key == view.turn_key
    assert snapshot.source == SNAPSHOT_SOURCE
    assert [candidate.original_rank for candidate in snapshot.candidates] == [1, 2, 3]
    assert [candidate.score for candidate in snapshot.candidates] == [0.0, 0.0, 0.0]
    assert all(
        candidate.score_kind == coshop_adapter.SCORE_KIND
        for candidate in snapshot.candidates
    )
    assert all(
        candidate.metadata_status == "missing" and candidate.metadata is None
        for candidate in snapshot.candidates
    )
    assert all(
        candidate.eligibility.value == "not_assessed"
        for candidate in snapshot.candidates
    )

    # Deterministic, distinct 1-based catalog-order item ids.
    expected = {identity: position for position, identity in enumerate(CATALOG_IDS, start=1)}
    item_ids = [candidate.item_id for candidate in snapshot.candidates]
    assert item_ids == [expected[identity] for identity in view.candidate_ids]
    assert item_ids == [3, 5, 1]
    assert len(set(item_ids)) == len(item_ids)
    assert all(isinstance(item_id, int) and item_id >= 1 for item_id in item_ids)

    # The hidden target is not carried by the snapshot (it is not a candidate here).
    assert TARGET_NOT_IN_POOL not in json.dumps(snapshot.model_dump())


def test_candidate_missing_from_catalog_is_a_contract_error() -> None:
    view = build_task_view(
        spec=FakeSpec(z0_baseline_ids=["movie-001", "movie-999"]),
        domain="movielens",
        version="v2",
        split="test",
        task_index=0,
    )
    with pytest.raises(CoshopContractError, match="not present in the official catalog"):
        to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))


def test_catalog_index_may_be_any_ordered_container() -> None:
    """The real catalog index is a pandas-like container, not a Python Sequence."""

    class IndexLike:
        def __init__(self, ids: Sequence[str]) -> None:
            self._ids = tuple(ids)

        def __iter__(self):
            return iter(self._ids)

        def __len__(self) -> int:
            return len(self._ids)

    class CatalogLike:
        index = IndexLike(CATALOG_IDS)

    view = build_task_view(
        spec=FakeSpec(z0_baseline_ids=["movie-003", "movie-007"]),
        domain="movielens",
        version="v2",
        split="test",
        task_index=0,
    )
    snapshot = to_candidate_snapshot(view, CatalogLike())
    assert [candidate.item_id for candidate in snapshot.candidates] == [3, 7]


def test_amazon_namespace_is_never_mixed_into_the_snapshot() -> None:
    view = build_task_view(
        spec=FakeSpec(z0_baseline_ids=["movie-003", "movie-005"]),
        domain="movielens",
        version="v2",
        split="test",
        task_index=0,
    )
    snapshot = to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))
    item_ids = [candidate.item_id for candidate in snapshot.candidates]
    dumped = json.dumps(snapshot.model_dump())
    for candidate in snapshot.candidates:
        assert candidate.candidate_id.startswith("movie-")
        assert candidate.candidate_id not in AMAZON_IDENTITIES
        assert candidate.metadata is None
        assert 1 <= candidate.item_id <= len(CATALOG_IDS)
        # An Amazon item_id would identify the same integer as a catalog position, so the
        # explicit provenance labels below are what prevent namespace confusion.
        assert candidate.score_kind == coshop_adapter.SCORE_KIND
    assert item_ids == [3, 5]
    assert "parent_asin" not in dumped
    assert "amazon" not in dumped.casefold()


def test_baseline_ranking_callable_uses_only_public_inputs() -> None:
    view = build_task_view(
        spec=FakeSpec(z0_baseline_ids=["movie-003", "movie-005"]),
        domain="movielens",
        version="v2",
        split="test",
        task_index=0,
    )
    snapshot = to_candidate_snapshot(view, FakeCatalog(CATALOG_IDS))
    ranked = rank_baseline_candidates(view, snapshot)
    assert ranked == view.candidate_ids == ("movie-003", "movie-005")

    import inspect

    parameters = inspect.signature(rank_baseline_candidates).parameters
    assert list(parameters) == ["task_view", "candidate_snapshot"]
    parameters = inspect.signature(build_task_view).parameters
    assert "xstar" not in parameters


def test_injected_ranker_cannot_see_targets_and_may_reorder() -> None:
    captured: dict[str, Any] = {}

    def spy_ranker(task_view: CoshopTaskView, candidate_snapshot: Any) -> tuple[str, ...]:
        captured["task_view"] = task_view
        captured["snapshot"] = candidate_snapshot
        return tuple(
            reversed([candidate.candidate_id for candidate in candidate_snapshot.candidates])
        )

    report = _evaluate(limit=1, ranker=spy_ranker)
    assert isinstance(captured["task_view"], CoshopTaskView)
    assert type(captured["snapshot"]).__name__ == "CandidateSnapshot"
    public_text = json.dumps(
        {
            "view": captured["task_view"].public_payload(),
            "snapshot": captured["snapshot"].model_dump(mode="json"),
        }
    )
    assert TARGET_NOT_IN_POOL not in public_text
    assert report.tasks[0].ranked_candidate_ids == (
        "movie-001",
        "movie-005",
        "movie-003",
    )


def test_ranker_may_not_change_the_candidate_set() -> None:
    def shrinking_ranker(task_view: CoshopTaskView, candidate_snapshot: Any) -> tuple[str, ...]:
        return (candidate_snapshot.candidates[0].candidate_id,)

    with pytest.raises(CoshopContractError, match="changed the candidate set"):
        _evaluate(limit=1, ranker=shrinking_ranker)

    def duplicating_ranker(task_view: CoshopTaskView, candidate_snapshot: Any) -> tuple[str, ...]:
        first = candidate_snapshot.candidates[0].candidate_id
        return (first, first)

    with pytest.raises(CoshopContractError, match="duplicate"):
        _evaluate(limit=1, ranker=duplicating_ranker)


# --------------------------------------------------------------------------- #
# Official metrics reuse, custom diagnostics, NOT_EXECUTED decision metrics
# --------------------------------------------------------------------------- #


def test_official_metrics_are_reused_with_k_and_without_embedding_scores() -> None:
    recorder = RecordingMetrics()
    report = _evaluate(limit=2, k=3, metrics_function=recorder)

    assert len(recorder.calls) == 2
    for call in recorder.calls:
        assert call["k"] == 3
        assert call["compute_embedding_ustar"] is False
        assert call["compute_column_ustar"] is True
        assert call["seen_item_ids"] == []
        assert "spec" in call and "catalog" in call and "ranked_item_ids" in call

    official = report.official_metrics
    assert official["source"] == OFFICIAL_METRICS_SOURCE
    assert official["k"] == 3
    assert official["compute_embedding_ustar"] is False
    assert official["tasks_evaluated"] == 2
    assert official["per_task"][0]["task_id"] == "coshop:movielens:v2:spec-000000"
    # Target is not in the visible candidate pool -> official fake recall is 0.
    assert official["per_task"][0]["recall_at_k"] == 0.0
    assert official["per_task"][0]["recall_at_1"] == 0.0
    assert official["recall_at_k"] == 0.0
    assert official["ndcg_ustar_at_k"] == 0.5
    assert report.tasks[0].official_metrics["source"] == OFFICIAL_METRICS_SOURCE


@pytest.mark.parametrize(
    ("target_position", "expected_recall", "expected_rank", "expected_in_pool"),
    [
        (1, 1.0, 1, True),
        (5, 1.0, 5, True),
        (6, 0.0, 6, True),
        (None, 0.0, None, False),
    ],
)
def test_custom_recall_at_k_cases(
    target_position: int | None,
    expected_recall: float,
    expected_rank: int | None,
    expected_in_pool: bool,
) -> None:
    baseline = [f"movie-{position:03d}" for position in range(1, 11)]
    if target_position is None:
        xstar = ["movie-999"]  # deliberately absent from the candidate pool
    else:
        xstar = [baseline[target_position - 1]]
    spec = FakeSpec(z0="query", z0_baseline_ids=baseline, xstar=xstar)
    report = evaluate_coshop(
        limit=1,
        k=5,
        dataset_factory=make_dataset_factory(CATALOG_IDS, [spec]),
        metrics_function=RecordingMetrics(),
    )
    task_custom = report.custom_diagnostics["per_task"][0]
    assert task_custom["label"] == CUSTOM_DIAGNOSTICS_LABEL
    assert task_custom["target_in_candidate_pool"] is expected_in_pool
    assert task_custom["target_original_rank_in_pool"] == expected_rank
    assert task_custom["custom_recall_at_k"] == expected_recall
    assert task_custom["retrieval_calls"] == 1
    assert task_custom["dialogue_turns"] == 0
    assert task_custom["clarifying_questions"] == 0
    assert task_custom["tool_calls"] == 1


def test_report_separates_official_custom_and_decision_metrics() -> None:
    report = _evaluate()
    body = report.as_dict()
    assert body["official_metrics"]["source"] == OFFICIAL_METRICS_SOURCE
    assert body["custom_diagnostics"]["label"] == CUSTOM_DIAGNOSTICS_LABEL
    assert body["custom_diagnostics"]["label"] != body["official_metrics"]["source"]

    expected_decision_metrics = {
        "team_accuracy",
        "conditional_selection_accuracy",
        "interaction_cost",
    }
    assert set(body["decision_metrics"]) == expected_decision_metrics
    for name, status in body["decision_metrics"].items():
        assert status["status"] == "NOT_EXECUTED"
        assert status["value"] is None
        assert status["surrogate_used"] is False
        assert status["reason"]
    team_accuracy = body["decision_metrics"]["team_accuracy"]
    assert team_accuracy["value"] is None
    # Team Accuracy must never be substituted by retrieval top-1 / recall@1.
    assert team_accuracy["value"] != body["official_metrics"]["recall_at_1"]
    assert "not top-1" in team_accuracy["reason"].lower()
    assert all(
        not isinstance(status["value"], (int, float))
        for status in body["decision_metrics"].values()
    )


def test_report_metadata_and_deterministic_digest() -> None:
    first = _evaluate()
    second = _evaluate()
    assert first.report_digest == second.report_digest
    assert first.as_dict() == second.as_dict()
    assert len(first.report_digest) == 64
    assert first.adapter_version == ADAPTER_VERSION
    assert first.protocol == PROTOCOL_NAME
    assert first.benchmark == "coshop"
    assert first.domain == "movielens"
    assert first.version == "v2"
    assert first.split == "test"
    assert first.task_count == 2
    assert first.k == 5
    assert first.baseline == BASELINE_NAME
    assert first.candidate_source == CANDIDATE_SOURCE
    assert all(isinstance(task, coshop_adapter.TaskEvaluationRecord) for task in first.tasks)

    # The digest is the SHA-256 of the canonical report body without the digest field.
    body = first.as_dict()
    body.pop("report_digest")
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert first.report_digest == expected


def test_report_provenance_and_no_hidden_target_leak() -> None:
    report = _evaluate()
    provenance = report.provenance
    assert provenance["adapter_version"] == ADAPTER_VERSION
    assert provenance["protocol"] == PROTOCOL_NAME
    assert provenance["baseline"] == BASELINE_NAME
    assert provenance["seed"] == 0
    assert provenance["dataset"] == {"domain": "movielens", "version": "v2", "split": "test"}
    assert provenance["tasks"] == {"requested_limit": 2, "evaluated": 2, "indices": [0, 1]}
    assert provenance["coshop_package_version"] is None  # no CoShop import in fake mode
    assert provenance["dataset_injected"] is True
    assert provenance["metrics_injected"] is True

    catalog_provenance = provenance["catalog"]
    assert catalog_provenance["id_count"] == len(CATALOG_IDS)
    expected_catalog_digest = hashlib.sha256(
        json.dumps(
            list(CATALOG_IDS),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert catalog_provenance["ordered_ids_sha256"] == expected_catalog_digest

    serialised = json.dumps(report.as_dict(), allow_nan=False)
    assert TARGET_NOT_IN_POOL not in serialised
    assert SECOND_TARGET_NOT_IN_POOL not in serialised
    for task in report.tasks:
        assert all(
            identity not in AMAZON_IDENTITIES for identity in task.ranked_candidate_ids
        )


def test_task_limit_is_enforced_not_silently_truncated() -> None:
    with pytest.raises(CoshopContractError, match="refusing to silently evaluate fewer"):
        _evaluate(limit=4)  # the fake official split exposes only 3 tasks


# --------------------------------------------------------------------------- #
# CLI: clean NOT EXECUTED verdict, exit code 2, no fabricated metrics
# --------------------------------------------------------------------------- #


def test_cli_not_executed_exits_2_and_writes_no_metrics(tmp_path) -> None:
    env = dict(os.environ)
    env.pop(ENV_COSHOP_ROOT, None)
    json_path = tmp_path / "not-executed-report.json"
    unavailable_root = tmp_path / "missing-coshop-checkout"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.external_benchmarks.coshop.run_evaluation",
            "--tasks",
            "1",
            "--k",
            "5",
            "--coshop-root",
            str(unavailable_root),
            "--json",
            str(json_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 2, result.stderr
    assert "NOT EXECUTED:" in result.stdout
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_EXECUTED"
    assert payload["reason"]
    assert payload["official_metrics"] is None
    assert payload["custom_diagnostics"] is None
    assert "report_digest" not in payload


def test_cli_success_path_prints_digest_and_writes_json(tmp_path, monkeypatch, capsys) -> None:
    from experiments.external_benchmarks.coshop import run_evaluation

    def fake_evaluate(**kwargs: Any):
        return evaluate_coshop(
            limit=1,
            k=5,
            dataset_factory=make_dataset_factory(),
            metrics_function=RecordingMetrics(),
        )

    monkeypatch.setattr(run_evaluation, "evaluate_coshop", fake_evaluate)
    json_path = tmp_path / "report.json"
    exit_code = run_evaluation.main(["--tasks", "1", "--k", "5", "--json", str(json_path)])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "report digest:" in printed
    assert "NOT_EXECUTED" in printed  # the decision metrics are explicitly not executed
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["report_digest"]) == 64
    assert payload["official_metrics"]["source"] == OFFICIAL_METRICS_SOURCE
    assert payload["custom_diagnostics"]["label"] == CUSTOM_DIAGNOSTICS_LABEL
    for status in payload["decision_metrics"].values():
        assert status["status"] == "NOT_EXECUTED"


# --------------------------------------------------------------------------- #
# Optional real official run (never fabricated; skipped without an external checkout)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get(ENV_COSHOP_ROOT),
    reason="official CoShop checkout is not configured (AGENTRECX_COSHOP_ROOT unset)",
)
def test_optional_real_official_run_single_task() -> None:
    """One real task when an external checkout and its official assets exist.

    If the official Google Drive assets are unavailable, this test reports the honest
    environment limitation as a skip; it never asserts a fabricated result.
    """
    try:
        report = evaluate_coshop(limit=1, k=5)
    except CoshopUnavailable as error:
        pytest.skip(f"official CoShop checkout/assets unavailable: {error}")
    assert report.task_count == 1
    assert report.official_metrics["source"] == OFFICIAL_METRICS_SOURCE
    assert report.provenance["catalog"]["id_count"] > 0
    assert len(report.report_digest) == 64
