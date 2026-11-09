"""Reusable experiment manifest, paired uncertainty and popularity slicing.

This module does **not** run a recommender.  It is the shared evidence layer around the
accepted evaluator: one frozen cohort/candidate-pool definition, one manifest that records
what was actually run, per-case metric vectors, paired bootstrap deltas, and head/body/tail
slices.  It deliberately reports dimensions separately and has no single "score" field.

Public-project patterns adopted:

* RecBole-style explicit dataset/model/protocol configuration, but expressed as a small,
  dependency-free manifest rather than a framework dependency;
* MLflow-style run provenance (code/driver/data/model hashes) without a tracking server;
* paired bootstrap and sliced reporting, not a single aggregate that can hide a regression.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "ExperimentBundle",
    "ExperimentCaseResult",
    "ExperimentError",
    "ExperimentManifest",
    "ExperimentSummary",
    "PairedDelta",
    "PopularityIndex",
    "build_bundle",
    "hash_file",
    "hash_json",
    "paired_bootstrap_delta",
    "summarize_cases",
]


class ExperimentError(ValueError):
    """The experiment contract is invalid or two arms cannot be paired."""


DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260201
_HEAD_SHARE = 0.80
_BODY_SHARE = 0.95
_HASH_CHARS = set("0123456789abcdef")


def _require_hash(name: str, value: str) -> None:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in _HASH_CHARS for char in text):
        raise ExperimentError(f"{name} must be a 64-character lowercase SHA-256 hex digest")


def hash_file(path: Path | str) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentManifest:
    """What exactly was evaluated.

    ``test_frozen=True`` is mandatory: the harness may summarize a held-out result, but it
    refuses to present a mutable/selection run as a frozen test.  Hashes are declarative: this
    object records provenance; it does not inspect the files for the caller.
    """

    name: str
    protocol_version: str
    cohort_id: str
    cohort_sha256: str
    candidate_pool_id: str
    candidate_pool_sha256: str
    driver: str
    driver_sha256: str
    code_git_head: str
    data_sha256: Mapping[str, str]
    model_sha256: Mapping[str, str]
    test_frozen: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "name",
            "protocol_version",
            "cohort_id",
            "candidate_pool_id",
            "driver",
            "code_git_head",
        ):
            if not str(getattr(self, name)).strip():
                raise ExperimentError(f"{name} must be non-empty")
        if not self.test_frozen:
            raise ExperimentError(
                "an experiment summary requires test_frozen=True; selection/validation runs "
                "must not be reported as a frozen test"
            )
        _require_hash("cohort_sha256", self.cohort_sha256)
        _require_hash("candidate_pool_sha256", self.candidate_pool_sha256)
        _require_hash("driver_sha256", self.driver_sha256)
        if not self.data_sha256 or not self.model_sha256:
            raise ExperimentError("data_sha256 and model_sha256 must both be non-empty")
        for kind, values in (("data", self.data_sha256), ("model", self.model_sha256)):
            for key, digest in values.items():
                if not str(key).strip():
                    raise ExperimentError(f"{kind} hash key must be non-empty")
                _require_hash(f"{kind}_sha256[{key!r}]", digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol_version": self.protocol_version,
            "cohort_id": self.cohort_id,
            "cohort_sha256": self.cohort_sha256,
            "candidate_pool_id": self.candidate_pool_id,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "driver": self.driver,
            "driver_sha256": self.driver_sha256,
            "code_git_head": self.code_git_head,
            "data_sha256": dict(sorted(self.data_sha256.items())),
            "model_sha256": dict(sorted(self.model_sha256.items())),
            "test_frozen": self.test_frozen,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class ExperimentCaseResult:
    """One case's metric vector for one arm."""

    case_id: str
    target_item_id: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ExperimentError("case_id must be non-empty")
        if isinstance(self.target_item_id, bool) or not isinstance(self.target_item_id, int):
            raise ExperimentError("target_item_id must be an integer")
        if not self.metrics:
            raise ExperimentError("a case result needs at least one metric")
        for name, value in self.metrics.items():
            if not str(name).strip():
                raise ExperimentError("metric names must be non-empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExperimentError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ExperimentError(f"metric {name!r} must be finite")


@dataclass(frozen=True)
class PopularityIndex:
    """Deterministic head/body/tail assignment from training item frequencies.

    Items are sorted by ``(-count, item_id)``.  Head covers the first 80% of total observed
    interactions, body the next 15%, and tail the remainder.  Items absent from the counts are
    explicitly ``unknown`` rather than silently treated as tail; callers may exclude or report
    them separately.
    """

    item_slice: Mapping[int, str]
    head_items: frozenset[int]
    body_items: frozenset[int]
    tail_items: frozenset[int]
    total_interactions: int

    @classmethod
    def from_counts(cls, counts: Mapping[int, int]) -> "PopularityIndex":
        clean: dict[int, int] = {}
        for item, count in counts.items():
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ExperimentError("popularity item keys must be positive integer ids")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ExperimentError("popularity counts must be non-negative integers")
            clean[item] = count
        total = sum(clean.values())
        assignments: dict[int, str] = {}
        if total == 0:
            return cls(
                item_slice=assignments,
                head_items=frozenset(),
                body_items=frozenset(),
                tail_items=frozenset(),
                total_interactions=0,
            )
        head: set[int] = set()
        body: set[int] = set()
        tail: set[int] = set()
        cumulative = 0
        for item, count in sorted(clean.items(), key=lambda pair: (-pair[1], pair[0])):
            if count <= 0:
                group = "tail"
            else:
                # Assign by where the item's mass ends.  The first item crossing 80% goes
                # to head; the first item crossing 95% goes to body; the rest is tail.
                before = cumulative / total
                cumulative += count
                after = cumulative / total
                if before < _HEAD_SHARE or after <= _HEAD_SHARE:
                    group = "head"
                elif before < _BODY_SHARE:
                    group = "body"
                else:
                    group = "tail"
            assignments[item] = group
            {"head": head, "body": body, "tail": tail}[group].add(item)
        return cls(
            item_slice=assignments,
            head_items=frozenset(head),
            body_items=frozenset(body),
            tail_items=frozenset(tail),
            total_interactions=total,
        )

    def slice_for(self, item_id: int) -> str:
        return self.item_slice.get(int(item_id), "unknown")


@dataclass(frozen=True)
class ExperimentSummary:
    """Per-dimension means, optionally sliced; never a single composite score."""

    n_cases: int
    metrics: Mapping[str, float]
    by_slice: Mapping[str, Mapping[str, float]]
    slice_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "metrics": dict(sorted(self.metrics.items())),
            "by_slice": {
                key: dict(sorted(value.items())) for key, value in sorted(self.by_slice.items())
            },
            "slice_counts": dict(sorted(self.slice_counts.items())),
        }


def _mean_metrics(cases: Sequence[ExperimentCaseResult], metric_names: Sequence[str]) -> dict[str, float]:
    if not cases:
        return {}
    return {
        name: statistics.fmean(float(case.metrics[name]) for case in cases)
        for name in metric_names
        if all(name in case.metrics for case in cases)
    }


def summarize_cases(
    cases: Sequence[ExperimentCaseResult],
    *,
    popularity: PopularityIndex | None = None,
) -> ExperimentSummary:
    """Summarize one arm; optional slices require a target popularity index."""
    if not cases:
        raise ExperimentError("cannot summarize an empty case set")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ExperimentError("case_id values must be unique within an arm")
    metric_names = sorted({name for case in cases for name in case.metrics})
    metrics = _mean_metrics(cases, metric_names)
    by_slice: dict[str, dict[str, float]] = {}
    slice_counts: dict[str, int] = {}
    if popularity is not None:
        grouped: dict[str, list[ExperimentCaseResult]] = {}
        for case in cases:
            group = popularity.slice_for(case.target_item_id)
            grouped.setdefault(group, []).append(case)
        for group, group_cases in sorted(grouped.items()):
            by_slice[group] = _mean_metrics(group_cases, metric_names)
            slice_counts[group] = len(group_cases)
    return ExperimentSummary(
        n_cases=len(cases),
        metrics=metrics,
        by_slice=by_slice,
        slice_counts=slice_counts,
    )


@dataclass(frozen=True)
class PairedDelta:
    """One metric's paired bootstrap result."""

    metric: str
    n: int
    mean: float
    lower: float
    upper: float
    confidence: float
    draws: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "draws": self.draws,
            "seed": self.seed,
        }


def paired_bootstrap_delta(
    baseline: Sequence[ExperimentCaseResult],
    candidate: Sequence[ExperimentCaseResult],
    *,
    metric: str,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedDelta:
    """Return the case-paired bootstrap CI for ``candidate - baseline``.

    Cases are paired by ``case_id`` and the case is the resampling unit.  A missing metric or
    a non-identical case set is an error; the function never silently intersects two arms.
    """
    if draws < 1:
        raise ExperimentError("draws must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ExperimentError("confidence must be in (0, 1)")
    base = {case.case_id: case for case in baseline}
    cand = {case.case_id: case for case in candidate}
    if set(base) != set(cand):
        raise ExperimentError("paired bootstrap requires identical case ids in both arms")
    if not base:
        raise ExperimentError("paired bootstrap requires at least one case")
    diffs: list[float] = []
    for case_id in sorted(base):
        left = base[case_id].metrics.get(metric)
        right = cand[case_id].metrics.get(metric)
        if left is None or right is None:
            raise ExperimentError(f"metric {metric!r} is missing from case {case_id!r}")
        diffs.append(float(right) - float(left))
    rng = random.Random(seed)
    n = len(diffs)
    samples = [statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, int(math.floor(alpha * (draws - 1))))
    upper_index = max(0, int(math.ceil((1.0 - alpha) * (draws - 1))))
    return PairedDelta(
        metric=metric,
        n=n,
        mean=statistics.fmean(diffs),
        lower=samples[lower_index],
        upper=samples[upper_index],
        confidence=confidence,
        draws=draws,
        seed=seed,
    )


@dataclass(frozen=True)
class ExperimentBundle:
    """A frozen manifest plus named arms with exactly aligned case ids."""

    manifest: ExperimentManifest
    arms: Mapping[str, tuple[ExperimentCaseResult, ...]]

    def __post_init__(self) -> None:
        if len(self.arms) < 1:
            raise ExperimentError("an experiment needs at least one arm")
        case_ids: set[str] | None = None
        for arm, cases in self.arms.items():
            if not str(arm).strip():
                raise ExperimentError("arm names must be non-empty")
            current = {case.case_id for case in cases}
            if len(current) != len(cases):
                raise ExperimentError(f"arm {arm!r} has duplicate case ids")
            if case_ids is None:
                case_ids = current
            elif current != case_ids:
                raise ExperimentError("all arms in a bundle must share identical case ids")

    def as_dict(self, *, popularity: PopularityIndex | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest": self.manifest.as_dict(),
            "arms": {},
        }
        summaries: dict[str, ExperimentSummary] = {}
        for arm, cases in sorted(self.arms.items()):
            summary = summarize_cases(cases, popularity=popularity)
            summaries[arm] = summary
            payload["arms"][arm] = summary.as_dict()
        names = sorted(self.arms)
        if len(names) >= 2:
            baseline = self.arms[names[0]]
            comparisons: dict[str, Any] = {}
            metric_names = sorted(
                {metric for case in baseline for metric in case.metrics}
            )
            for candidate_name in names[1:]:
                comparisons[candidate_name] = {
                    metric: paired_bootstrap_delta(
                        baseline,
                        self.arms[candidate_name],
                        metric=metric,
                    ).as_dict()
                    for metric in metric_names
                }
            payload["paired_deltas"] = {
                "baseline": names[0],
                "arms": comparisons,
            }
        return payload


def build_bundle(
    manifest: ExperimentManifest,
    arms: Mapping[str, Iterable[ExperimentCaseResult]],
    *,
    popularity: PopularityIndex | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable experiment report and validate arm alignment."""
    return ExperimentBundle(
        manifest=manifest,
        arms={name: tuple(cases) for name, cases in arms.items()},
    ).as_dict(popularity=popularity)
