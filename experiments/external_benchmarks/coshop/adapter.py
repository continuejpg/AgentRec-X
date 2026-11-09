"""CoShop external-benchmark adapter — strict isolation from AgentRec-X.

What this module is
-------------------
CoShop is an **external** conversational-shopping benchmark.  AgentRec-X is the
subject; CoShop is the measuring instrument.  This module lets AgentRec-X's
*offline lab* reuse the official CoShop task set, the official visible-query
retrieval baseline and the official metric implementation, without importing any
CoShop source into AgentRec-X, without vendoring it (the upstream repository has
no licence) and without mixing identity or target namespaces.

The official CoShop checkout lives at an external path supplied by the caller
(``coshop_root=``) or by the ``AGENTRECX_COSHOP_ROOT`` environment variable.  It
is imported lazily, only when an evaluation actually runs.  The official Google
Drive data assets must already be present in that checkout; this adapter never
downloads anything itself and never fabricates data or metrics.

Boundary rules encoded here
---------------------------
* **Import isolation.** Module import uses only the standard library.  CoShop,
  pandas, torch and AgentRec-X packages are not imported at import time.  The
  AgentRec-X offline contract (``recommendation.ranking.schemas``) is imported
  lazily inside the snapshot conversion, so the adapter is importable in a
  minimal environment and deletable without touching the core.
* **No identity mixing.** Candidate identities are the official CoShop external
  ids as strings.  The AgentRec-X ``item_id`` field is a deterministic
  catalog-order position (1-based) local to the external benchmark; no Amazon
  ``parent_asin`` / ``item_id`` value can appear here.
* **No hidden-target leakage.** A public :class:`CoshopTaskView` carries only the
  user-visible query and the official visible-query candidate ids.  The hidden
  ground-truth targets live in a separate sealed :class:`HiddenTaskTargets`
  object that only the evaluator/diagnostics touch; they are never passed to the
  snapshot conversion, the ranking callable or the report payload.
* **No target-synthesised candidates.** Candidates are exactly the official
  visible-query baseline ids (``spec.z0_baseline_ids``), deduplicated preserving
  order.  A missing/empty baseline is a typed contract failure, never rebuilt
  from targets.
* **Metrics are separated.** Official metrics come from
  ``coshop.evaluation.metrics.compute_evaluation_metrics`` (k configurable,
  ``compute_embedding_ustar=False``).  Custom retrieval diagnostics are labelled
  as such.  Dialogue/decision metrics (Team Accuracy, conditional selection
  accuracy, interaction cost) are ``NOT_EXECUTED`` because they need the CoPref
  user simulator plus a hosted LLM API; top-1/recall@1 is never substituted for
  Team Accuracy and no simulated-user result is fabricated.

Deletion safety: remove ``experiments/external_benchmarks/`` and every core test
still passes; nothing under ``recommendation/``, ``agent/`` or ``api/`` imports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ADAPTER_VERSION",
    "BASELINE_NAME",
    "CANDIDATE_SOURCE",
    "CONDITIONAL_SELECTION_REASON",
    "CUSTOM_DIAGNOSTICS_LABEL",
    "DEFAULT_DOMAIN",
    "DEFAULT_K",
    "DEFAULT_LIMIT",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT",
    "DEFAULT_VERSION",
    "ENV_COSHOP_ROOT",
    "INTERACTION_COST_REASON",
    "OFFICIAL_METRICS_SOURCE",
    "PROTOCOL_NAME",
    "SNAPSHOT_SOURCE",
    "SUPPORTED_DOMAINS",
    "SUPPORTED_SPLITS",
    "SUPPORTED_VERSIONS",
    "TEAM_ACCURACY_REASON",
    "CoshopAdapterError",
    "CoshopContractError",
    "CoshopEvaluationReport",
    "CoshopTaskView",
    "CoshopUnavailable",
    "HiddenTaskTargets",
    "TaskEvaluationRecord",
    "build_task_view",
    "evaluate_coshop",
    "extract_hidden_targets",
    "load_coshop_dataset",
    "rank_baseline_candidates",
    "resolve_coshop_root",
    "to_candidate_snapshot",
]

#: Version of this adapter (not of CoShop).
ADAPTER_VERSION = "0.1.0"

#: Frozen protocol label recorded in every report.
PROTOCOL_NAME = "agentrecx.m6_2.coshop_external_baseline.v1"

#: Environment variable naming the official CoShop checkout.
ENV_COSHOP_ROOT = "AGENTRECX_COSHOP_ROOT"

#: File that must exist inside a valid official checkout.
_COSHOP_DATA_ENTRYPOINT = Path("coshop") / "data" / "dataset.py"

#: Official dataset/version/split vocabularies (mirrors the CoShop package).
SUPPORTED_DOMAINS: tuple[str, ...] = ("hm", "movielens", "goodreads")
SUPPORTED_VERSIONS: tuple[str, ...] = ("v1", "v2")
SUPPORTED_SPLITS: tuple[str, ...] = ("test", "extended")

#: Defaults used by :func:`evaluate_coshop` and the CLI.
DEFAULT_DOMAIN = "movielens"
DEFAULT_VERSION = "v2"
DEFAULT_SPLIT = "test"
DEFAULT_LIMIT = 3
DEFAULT_K = 5

#: No sampling happens in this adapter; the seed is recorded for protocol
#: reproducibility (the official dataset also defaults to seed 0).
DEFAULT_SEED = 0

#: The baseline that is actually executed: the official visible-query retrieval
#: baseline (spec.z0 baseline ids), in its official order.  It is a retrieval
#: baseline, not an AgentRec-X model and not an M6.1 A0/A1/A2 ranking arm.
BASELINE_NAME = "coshop_official_visible_query_baseline"

#: Human-readable description of the candidate source.
CANDIDATE_SOURCE = (
    "spec.z0_baseline_ids (official visible-query vector-search retrieval order)"
)

#: Provenance labels copied into the AgentRec-X candidate snapshot.
SNAPSHOT_SOURCE = "coshop_external_benchmark"
SCORE_KIND = "coshop_official_baseline_order"

#: Official metric implementation reused (lazily imported from the CoShop checkout).
OFFICIAL_METRICS_SOURCE = "coshop.evaluation.metrics.compute_evaluation_metrics"

#: Label distinguishing adapter-computed retrieval diagnostics from official metrics.
CUSTOM_DIAGNOSTICS_LABEL = "custom_retrieval_baseline"

#: Reasons for the preregistered NOT_EXECUTED decision metrics.  These metrics are
#: defined by the official CoShop harness over a multi-turn episode with the CoPref
#: user simulator and a hosted LLM API; this adapter executes a single retrieval pass
#: and therefore cannot produce them.
TEAM_ACCURACY_REASON = (
    "Team Accuracy is the official user-simulator rerank recall_at_1 computed after "
    "a CoPref multi-turn elicitation episode; it requires the user simulator and a "
    "configured hosted LLM API. It is NOT top-1 retrieval recall/recall@1, so no "
    "surrogate is reported and no simulated-user result is fabricated."
)
CONDITIONAL_SELECTION_REASON = (
    "Conditional Selection Accuracy requires simulated user selections over multiple "
    "elicitation turns from the CoPref simulator; no hosted LLM API is configured and "
    "the adapter executes a single retrieval pass only, so nothing is fabricated."
)
INTERACTION_COST_REASON = (
    "Interaction Cost is an official episode-level cost (turns, questions, tokens, "
    "runtime) from a multi-turn elicitation episode; this adapter runs one retrieval "
    "call and zero dialogue turns, so it cannot report the official metric."
)

#: Bound on any error text copied into a typed adapter error.
_MAX_ERROR_CHARS = 320


# --------------------------------------------------------------------------- #
# Typed errors
# --------------------------------------------------------------------------- #


class CoshopAdapterError(Exception):
    """Base class for every CoShop-adapter rejection."""


class CoshopUnavailable(CoshopAdapterError):
    """The optional external dependency, checkout or official data is unavailable.

    This is an *environment* verdict, not a benchmark result: the caller must report
    ``NOT EXECUTED`` and must not fabricate metrics.
    """


class CoshopContractError(CoshopAdapterError):
    """The official CoShop payload does not satisfy the adapter's required contract.

    Examples: an empty visible-query baseline, a candidate missing from the official
    catalog, or an official object whose API drifted.  Nothing is repaired silently.
    """


# --------------------------------------------------------------------------- #
# Helpers: bounded errors, canonical JSON, numeric coercion
# --------------------------------------------------------------------------- #


def _bounded_error(error: BaseException) -> str:
    """Return a short, bounded description of ``error`` (never a raw payload)."""
    text = f"{type(error).__name__}: {error}".strip()
    return text[:_MAX_ERROR_CHARS]


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe copy of ``value`` (deterministic, no NaN/inf)."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    """Deterministic canonical JSON used for every digest in this module."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_ids(ordered_ids: Sequence[str]) -> str:
    """SHA-256 of the ordered official id list (canonical JSON encoding)."""
    return hashlib.sha256(_canonical_json(list(ordered_ids)).encode("utf-8")).hexdigest()


def _optional_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for an absent/non-finite metric value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _mean(values: Sequence[float | None]) -> float | None:
    """Mean of the non-``None`` values, rounded for digest stability, or ``None``."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 12)


def _require_text(value: Any, *, what: str) -> str:
    """Return a required non-blank official identifier string."""
    if not isinstance(value, str):
        raise CoshopContractError(f"{what} must be an official string identifier")
    text = " ".join(value.split())
    if not text:
        raise CoshopContractError(f"{what} must not be blank")
    return text


def _official_sequence(value: Any, *, what: str) -> list[Any] | None:
    """Return ``value`` as a list when it is an official id sequence, else ``None``.

    Duck-typed on purpose: the real CoShop objects may expose a list, tuple or pandas
    ``Index``-like container.  Strings/bytes are never accepted as a sequence of ids.
    """
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


# --------------------------------------------------------------------------- #
# Public task / hidden-target views
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoshopTaskView:
    """One public, ranker-facing CoShop task view.

    Deliberately has **no** field for the hidden target (``xstar``) or any full
    preference description (``zstar``): a ranking component sees the user-visible
    query and the official visible-query candidate ids only.
    """

    task_id: str
    domain: str
    version: str
    split: str
    task_index: int
    user_key: str
    turn_key: str
    visible_query: str
    visible_query_source: str
    candidate_ids: tuple[str, ...]
    source: str = "coshop_official_benchmark"

    def public_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable public view (contains no hidden target)."""
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "version": self.version,
            "split": self.split,
            "task_index": self.task_index,
            "user_key": self.user_key,
            "turn_key": self.turn_key,
            "visible_query": self.visible_query,
            "visible_query_source": self.visible_query_source,
            "candidate_ids": list(self.candidate_ids),
            "candidate_count": len(self.candidate_ids),
            "source": self.source,
        }

    def __repr__(self) -> str:  # keep large/repeated fields out of logs
        return (
            f"CoshopTaskView(task_id={self.task_id!r}, domain={self.domain!r}, "
            f"split={self.split!r}, candidate_count={len(self.candidate_ids)})"
        )


@dataclass(frozen=True)
class HiddenTaskTargets:
    """Sealed container for one task's official hidden ground-truth target ids.

    Only the evaluator/diagnostics receive an instance.  It exposes booleans and
    ordinal ranks, plus a target-free :meth:`public_payload`/:meth:`__repr__`; no
    public accessor returns the target ids themselves, so a target cannot leak
    into a candidate snapshot, a ranking input or a report.
    """

    task_id: str
    _targets: tuple[str, ...] = field(repr=False)

    @property
    def target_count(self) -> int:
        """Number of official hidden target items for this task."""
        return len(self._targets)

    def target_in_pool(self, candidate_ids: Sequence[str]) -> bool:
        """True when at least one hidden target appears in the candidate set."""
        candidates = set(candidate_ids)
        return any(target in candidates for target in self._targets)

    def target_best_rank(self, ordered_ids: Sequence[str]) -> int | None:
        """1-based best rank of any hidden target in ``ordered_ids``, or ``None``."""
        for position, identity in enumerate(ordered_ids, start=1):
            if identity in self._targets:
                return position
        return None

    def public_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable target-free view."""
        return {"task_id": self.task_id, "target_count": self.target_count}

    def __repr__(self) -> str:
        return (
            f"HiddenTaskTargets(task_id={self.task_id!r}, "
            f"target_count={self.target_count})"
        )


# --------------------------------------------------------------------------- #
# External checkout resolution and lazy import
# --------------------------------------------------------------------------- #


def resolve_coshop_root(coshop_root: str | Path | None = None) -> Path:
    """Resolve and validate the external CoShop checkout root.

    Precedence: explicit ``coshop_root`` argument, then the
    ``AGENTRECX_COSHOP_ROOT`` environment variable.  The directory must exist and
    contain ``coshop/data/dataset.py``; nothing is downloaded here.
    """
    configured = coshop_root if coshop_root is not None else os.environ.get(ENV_COSHOP_ROOT)
    if configured is None or not str(configured).strip():
        raise CoshopUnavailable(
            "CoShop is an optional external benchmark and no checkout is configured: "
            f"pass coshop_root=<path to the official CoShop repository> or set "
            f"{ENV_COSHOP_ROOT}. No metrics were produced."
        )
    root = Path(str(configured)).expanduser()
    try:
        resolved = root.resolve()
    except OSError as error:  # pragma: no cover - defensive path handling
        raise CoshopUnavailable(
            f"the configured CoShop checkout path is not usable: {_bounded_error(error)}"
        ) from error
    if not resolved.is_dir():
        raise CoshopUnavailable(
            f"the configured CoShop checkout does not exist as a directory: {resolved}"
        )
    entrypoint = resolved / _COSHOP_DATA_ENTRYPOINT
    if not entrypoint.is_file():
        raise CoshopUnavailable(
            "the configured CoShop checkout is not the official repository root: "
            f"expected {_COSHOP_DATA_ENTRYPOINT.as_posix()} under {resolved}"
        )
    return resolved


def _import_coshop_entrypoints(
    root: Path,
) -> tuple[Callable[..., Any], Callable[..., Any], str | None]:
    """Lazily import the official CoShop dataset factory and metric function.

    The external checkout is added to ``sys.path`` only for the duration of the
    import and removed afterwards.  An already-imported *different* CoShop package
    is refused (never mix identities from two checkouts).  Every failure becomes a
    :class:`CoshopUnavailable` with the original cause attached.
    """
    root_text = str(root)
    existing = sys.modules.get("coshop")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or not Path(existing_file).resolve().is_relative_to(root):
            raise CoshopUnavailable(
                "a different CoShop package is already imported in this process; "
                "refusing to mix external benchmark identities from two checkouts"
            )

    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from coshop.data import get_dataset  # type: ignore[import-not-found]
        from coshop.evaluation.metrics import (  # type: ignore[import-not-found]
            compute_evaluation_metrics,
        )
    except Exception as error:  # dependency/checkout failure -> environment verdict
        raise CoshopUnavailable(
            "the configured CoShop checkout could not be imported: install the official "
            "CoShop requirements in a separate CoShop virtual environment and run the "
            "adapter with that environment's Python; the checkout or its optional "
            f"dependencies are unavailable ({_bounded_error(error)})"
        ) from error
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:  # pragma: no cover - defensive
                pass

    module_file = getattr(sys.modules.get("coshop"), "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(root):
        raise CoshopUnavailable(
            "the imported coshop package is not the configured external checkout; "
            "refusing to continue with a foreign benchmark installation"
        )
    package_version = getattr(sys.modules["coshop"], "__version__", None)
    return get_dataset, compute_evaluation_metrics, (
        str(package_version) if package_version is not None else None
    )


def load_coshop_dataset(
    *,
    domain: str,
    version: str,
    split: str,
    coshop_root: str | Path | None = None,
) -> tuple[Any, Callable[..., Any], str | None, Path]:
    """Load the official CoShop dataset from the external checkout.

    Returns ``(dataset, metrics_function, package_version, resolved_root)``.  Any
    failure (missing checkout, missing dependency, unreachable Google Drive
    assets) becomes :class:`CoshopUnavailable`; no partial dataset is returned.
    """
    root = resolve_coshop_root(coshop_root)
    get_dataset, metrics_function, package_version = _import_coshop_entrypoints(root)
    try:
        loaded = get_dataset(
            domain,
            version=version,
            split=split,
            max_xstar=1,
            seed=DEFAULT_SEED,
        )
    except Exception as error:
        raise CoshopUnavailable(
            "the official CoShop dataset could not be loaded; its Google Drive assets "
            "must be downloaded once into the external checkout and be reachable "
            f"(no data was fabricated): {_bounded_error(error)}"
        ) from error
    return loaded, metrics_function, package_version, root


# --------------------------------------------------------------------------- #
# Official-object extraction (duck-typed; no CoShop import needed)
# --------------------------------------------------------------------------- #


def _baseline_candidate_ids(spec: Any) -> tuple[str, ...]:
    """Return the official visible-query baseline candidate ids, deduped in order."""
    raw = _official_sequence(getattr(spec, "z0_baseline_ids", None), what="baseline")
    if raw is None:
        raise CoshopContractError(
            "official spec exposes no visible-query baseline candidate list "
            "(spec.z0_baseline_ids); retrieval cannot run and candidates are never "
            "synthesized from hidden targets"
        )
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for identity in raw:
        text = _require_text(identity, what="official baseline candidate id")
        if text in seen:
            continue
        seen.add(text)
        candidate_ids.append(text)
    if not candidate_ids:
        raise CoshopContractError(
            "the official visible-query baseline candidate list is empty; retrieval "
            "cannot run and candidates are never synthesized from hidden targets"
        )
    return tuple(candidate_ids)


def _visible_query(spec: Any) -> tuple[str, str]:
    """Return ``(visible_query, source)`` from the official spec's public fields."""
    z0 = getattr(spec, "z0", None)
    if isinstance(z0, str) and z0.strip():
        return z0.strip(), "spec.z0"
    features = _official_sequence(
        getattr(spec, "initial_known_features", None), what="initial_known_features"
    )
    cleaned = (
        [text for text in (_require_feature_name(item) for item in features) if text]
        if features is not None
        else []
    )
    if cleaned:
        return (
            "Known preferences at the start of the conversation: " + ", ".join(cleaned),
            "spec.initial_known_features",
        )
    return "", "none"


def _require_feature_name(value: Any) -> str | None:
    """Return a cleaned feature name, or ``None`` when it is unusable."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def build_task_view(
    *,
    spec: Any,
    domain: str,
    version: str,
    split: str,
    task_index: int,
) -> CoshopTaskView:
    """Build the public, ranker-facing view of one official task.

    Only official *visible* fields are read: the user-visible query (``z0`` or the
    initial known features) and the official visible-query candidate ids.  The
    hidden target and all full preference descriptions are never touched.
    """
    if not isinstance(task_index, int) or isinstance(task_index, bool) or task_index < 0:
        raise CoshopContractError("task_index must be a non-negative integer")
    candidate_ids = _baseline_candidate_ids(spec)
    visible_query, query_source = _visible_query(spec)
    task_id = f"coshop:{domain}:{version}:spec-{task_index:06d}"
    return CoshopTaskView(
        task_id=task_id,
        domain=domain,
        version=version,
        split=split,
        task_index=task_index,
        user_key=f"{task_id}:user",
        turn_key=f"{task_id}:turn-1",
        visible_query=visible_query,
        visible_query_source=query_source,
        candidate_ids=candidate_ids,
        source="coshop_official_benchmark",
    )


def extract_hidden_targets(*, spec: Any, task_id: str) -> HiddenTaskTargets:
    """Seal one official task's hidden ground-truth targets for the evaluator only.

    Called by the evaluator, never by a ranking/conversion function.  A missing or
    empty target makes the task unevaluable: fail as a contract error instead of
    silently dropping it.
    """
    raw = _official_sequence(getattr(spec, "xstar", None), what="xstar")
    if raw is None:
        raise CoshopContractError(
            "official spec exposes no hidden ground-truth target list (spec.xstar); "
            "the task cannot be evaluated"
        )
    targets = tuple(_require_text(target, what="official hidden target id") for target in raw)
    if not targets:
        raise CoshopContractError(
            "the official hidden ground-truth target list is empty; the task cannot "
            "be evaluated"
        )
    return HiddenTaskTargets(task_id=_require_text(task_id, what="task_id"), _targets=targets)


# --------------------------------------------------------------------------- #
# Official catalog identity and AgentRec-X contract conversion
# --------------------------------------------------------------------------- #


def _catalog_ids(catalog: Any) -> tuple[str, ...]:
    """Return the official catalog identities in catalog order."""
    raw = _official_sequence(getattr(catalog, "index", None), what="catalog.index")
    if raw is None:
        raise CoshopContractError(
            "official dataset catalog exposes no ordered identity index; cannot build "
            "a deterministic candidate item_id mapping"
        )
    catalog_ids = tuple(_require_text(item, what="official catalog item id") for item in raw)
    if not catalog_ids:
        raise CoshopContractError("the official catalog is empty; cannot map candidates")
    if len(set(catalog_ids)) != len(catalog_ids):
        raise CoshopContractError(
            "the official catalog contains duplicate item ids; a deterministic "
            "catalog-order item_id mapping would be ambiguous"
        )
    return catalog_ids


def _catalog_item_id_map(catalog: Any) -> dict[str, int]:
    """Map each official catalog id to its 1-based catalog-order ``item_id``.

    The mapping is the agent-side integer id namespace local to this external
    benchmark.  It is distinct (a catalog position), deterministic for a given
    official catalog and *not* the Amazon ``parent_asin``/``item_id`` namespace.
    """
    return {identity: position for position, identity in enumerate(_catalog_ids(catalog), start=1)}


def _to_candidate_snapshot(
    task_view: CoshopTaskView,
    *,
    item_id_by_catalog_order: Mapping[str, int],
) -> Any:
    """Convert one public task view into the AgentRec-X offline snapshot contract.

    The contract import is lazy: the adapter stays importable (and deletable)
    without AgentRec-X, and the core lab stays unaware of the external benchmark.
    The snapshot carries no hidden target: only candidate identities, baseline
    order and provenance.
    """
    try:
        from recommendation.ranking.schemas import (  # noqa: PLC0415 - lazy isolation
            CandidateRecord,
            CandidateSnapshot,
        )
    except Exception as error:  # pragma: no cover - core lab always importable here
        raise CoshopAdapterError(
            "the AgentRec-X offline snapshot contract could not be imported: "
            f"{_bounded_error(error)}"
        ) from error

    records: list[Any] = []
    for position, candidate_id in enumerate(task_view.candidate_ids, start=1):
        item_id = item_id_by_catalog_order.get(candidate_id)
        if item_id is None:
            raise CoshopContractError(
                "an official visible-query baseline candidate is not present in the "
                "official catalog; its AgentRec-X item_id cannot be derived and is "
                "never invented"
            )
        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                item_id=item_id,
                original_rank=position,
                score=0.0,
                score_kind=SCORE_KIND,
                metadata_status="missing",
                metadata=None,
                source_ranks=(),
                rrf_score=None,
                eligibility="not_assessed",
                eligibility_detail=None,
            )
        )
    try:
        snapshot = CandidateSnapshot(
            snapshot_id=f"{task_view.task_id}:candidate-snapshot",
            user_key=task_view.user_key,
            task_id=task_view.task_id,
            turn_key=task_view.turn_key,
            candidates=tuple(records),
            source=SNAPSHOT_SOURCE,
        )
    except Exception as error:
        raise CoshopContractError(
            "the converted candidate snapshot failed AgentRec-X contract validation: "
            f"{_bounded_error(error)}"
        ) from error
    return snapshot


def to_candidate_snapshot(task_view: CoshopTaskView, catalog: Any) -> Any:
    """Build the AgentRec-X ``CandidateSnapshot`` for one task view and catalog.

    The ranker-facing snapshot equals the official visible-query baseline candidate
    set, in the official order; it never contains the hidden target and its
    ``item_id`` values are this benchmark's catalog-order positions.
    """
    if not isinstance(task_view, CoshopTaskView):
        raise CoshopContractError("to_candidate_snapshot requires a CoshopTaskView")
    return _to_candidate_snapshot(task_view, item_id_by_catalog_order=_catalog_item_id_map(catalog))


def rank_baseline_candidates(
    task_view: CoshopTaskView,
    candidate_snapshot: Any,
) -> tuple[str, ...]:
    """The ranker-facing baseline callable: ``(task_view, candidate_snapshot) -> ids``.

    It implements the official visible-query retrieval baseline by preserving the
    snapshot's trusted original order.  It is explicitly a retrieval baseline, not
    an AgentRec-X model.  It has no access to the hidden target.
    """
    if not isinstance(task_view, CoshopTaskView):
        raise CoshopContractError("rank_baseline_candidates requires a CoshopTaskView")
    snapshot_candidate_ids = tuple(
        candidate.candidate_id for candidate in candidate_snapshot.candidates
    )
    if snapshot_candidate_ids != task_view.candidate_ids:
        raise CoshopContractError(
            "the candidate snapshot does not match the public task view candidate set"
        )
    return snapshot_candidate_ids


# --------------------------------------------------------------------------- #
# Official metrics (lazily reused) and custom retrieval diagnostics
# --------------------------------------------------------------------------- #


def _run_official_metrics(
    *,
    metrics_function: Callable[..., Any],
    spec: Any,
    ranked_item_ids: Sequence[str],
    catalog: Any,
    k: int,
) -> dict[str, Any]:
    """Compute the official CoShop metrics through the official implementation.

    ``compute_embedding_ustar=False`` is always passed (the embedding utility needs
    the external vector-search server, which is not part of this baseline).  Any
    failure of the official stack becomes :class:`CoshopUnavailable`; an unexpected
    return object becomes :class:`CoshopContractError`.
    """
    try:
        evaluation = metrics_function(
            spec=spec,
            ranked_item_ids=list(ranked_item_ids),
            catalog=catalog,
            execution_max_per_retrieval=None,
            execution_max_queries=None,
            execution_global_max=None,
            k=k,
            seen_item_ids=[],
            compute_column_ustar=True,
            compute_embedding_ustar=False,
        )
    except Exception as error:
        raise CoshopUnavailable(
            "the official CoShop metric computation failed (its optional scoring "
            f"stack may be unavailable): {_bounded_error(error)}"
        ) from error
    predicted = getattr(evaluation, "predicted", None)
    if predicted is None:
        raise CoshopContractError(
            "the official metric function returned an object without a 'predicted' "
            "evaluation; the official metrics API may have drifted"
        )
    try:
        recall_at_k = _optional_float(predicted.recall_at_k)
        recall_at_1 = _optional_float(predicted.recall_at_1)
        ndcg_ustar_at_k = _optional_float(predicted.ndcg_ustar_at_k)
    except AttributeError as error:
        raise CoshopContractError(
            "the official predicted evaluation is missing recall_at_k / recall_at_1 / "
            "ndcg_ustar_at_k; the official metrics API may have drifted"
        ) from error
    return {
        "source": OFFICIAL_METRICS_SOURCE,
        "k": k,
        "compute_embedding_ustar": False,
        "recall_at_k": recall_at_k,
        "recall_at_1": recall_at_1,
        "ndcg_ustar_at_k": ndcg_ustar_at_k,
    }


def _custom_diagnostics(
    *,
    task_view: CoshopTaskView,
    ranked_ids: Sequence[str],
    hidden_targets: HiddenTaskTargets,
    k: int,
) -> dict[str, Any]:
    """Custom retrieval diagnostics (never official, never a decision metric)."""
    target_rank = hidden_targets.target_best_rank(ranked_ids)
    return {
        "label": CUSTOM_DIAGNOSTICS_LABEL,
        "baseline": BASELINE_NAME,
        "candidate_count": len(task_view.candidate_ids),
        "retrieval_calls": 1,
        "dialogue_turns": 0,
        "clarifying_questions": 0,
        "tool_calls": 1,
        "target_in_candidate_pool": hidden_targets.target_in_pool(task_view.candidate_ids),
        "target_original_rank_in_pool": target_rank,
        "custom_recall_at_k": (
            1.0 if target_rank is not None and target_rank <= k else 0.0
        ),
        "k": k,
    }


def _validate_ranking(
    ranked_ids: Any,
    *,
    task_view: CoshopTaskView,
) -> tuple[str, ...]:
    """Validate that a ranker returned a permutation of the frozen candidate set."""
    if isinstance(ranked_ids, (str, bytes)) or not isinstance(ranked_ids, Sequence):
        raise CoshopContractError(
            "the ranking callable must return a sequence of official candidate ids"
        )
    ordered = tuple(_require_text(identity, what="ranked candidate id") for identity in ranked_ids)
    if len(set(ordered)) != len(ordered):
        raise CoshopContractError(
            "the ranking callable returned duplicate candidates; an arm may reorder, "
            "never duplicate candidates"
        )
    if set(ordered) != set(task_view.candidate_ids):
        raise CoshopContractError(
            "the ranking callable changed the candidate set; an arm may reorder, never "
            "add, drop or replace a candidate"
        )
    return ordered


# --------------------------------------------------------------------------- #
# Evaluation report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskEvaluationRecord:
    """One task's official metrics plus custom diagnostics (target-free)."""

    task_id: str
    visible_query: str
    candidate_count: int
    ranked_candidate_ids: tuple[str, ...]
    official_metrics: dict[str, Any]
    custom_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CoshopEvaluationReport:
    """Deterministic, self-describing result of one CoShop baseline evaluation.

    ``official_metrics`` come from the official CoShop metric implementation;
    ``custom_diagnostics`` are adapter-computed retrieval diagnostics (labelled
    ``custom_retrieval_baseline``); ``decision_metrics`` are explicit
    ``NOT_EXECUTED`` statuses.  No field contains a hidden target id and no
    decision metric is estimated or substituted.
    """

    adapter_version: str
    protocol: str
    benchmark: str
    domain: str
    version: str
    split: str
    task_count: int
    candidate_count_total: int
    k: int
    baseline: str
    candidate_source: str
    tasks: tuple[TaskEvaluationRecord, ...]
    official_metrics: dict[str, Any]
    custom_diagnostics: dict[str, Any]
    decision_metrics: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    notes: tuple[str, ...]
    report_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable report body (including the digest)."""
        return _json_safe(
            {
                "adapter_version": self.adapter_version,
                "protocol": self.protocol,
                "benchmark": self.benchmark,
                "domain": self.domain,
                "version": self.version,
                "split": self.split,
                "task_count": self.task_count,
                "candidate_count_total": self.candidate_count_total,
                "k": self.k,
                "baseline": self.baseline,
                "candidate_source": self.candidate_source,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "visible_query": task.visible_query,
                        "candidate_count": task.candidate_count,
                        "ranked_candidate_ids": list(task.ranked_candidate_ids),
                        "official_metrics": task.official_metrics,
                        "custom_diagnostics": task.custom_diagnostics,
                    }
                    for task in self.tasks
                ],
                "official_metrics": self.official_metrics,
                "custom_diagnostics": self.custom_diagnostics,
                "decision_metrics": self.decision_metrics,
                "provenance": self.provenance,
                "notes": list(self.notes),
                "report_digest": self.report_digest,
            }
        )

    def with_digest(self) -> "CoshopEvaluationReport":
        """Return a copy with ``report_digest`` set over the canonical report body."""
        payload = self.as_dict()
        payload.pop("report_digest", None)
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return replace(self, report_digest=digest)


def _decision_metrics() -> dict[str, dict[str, Any]]:
    """The preregistered NOT_EXECUTED decision metrics (never estimated)."""
    return {
        "team_accuracy": {
            "status": "NOT_EXECUTED",
            "value": None,
            "reason": TEAM_ACCURACY_REASON,
            "surrogate_used": False,
        },
        "conditional_selection_accuracy": {
            "status": "NOT_EXECUTED",
            "value": None,
            "reason": CONDITIONAL_SELECTION_REASON,
            "surrogate_used": False,
        },
        "interaction_cost": {
            "status": "NOT_EXECUTED",
            "value": None,
            "reason": INTERACTION_COST_REASON,
            "surrogate_used": False,
        },
    }


def _report_notes() -> tuple[str, ...]:
    """The fixed interpretation notes recorded with every report."""
    return (
        "The executed baseline is the official CoShop visible-query retrieval baseline "
        f"({BASELINE_NAME}); it is a retrieval baseline, NOT an AgentRec-X model and NOT "
        "an M6.1 A0/A1/A2 ranking arm.",
        "Official metrics are copied from coshop.evaluation.metrics."
        "compute_evaluation_metrics (k configurable, compute_embedding_ustar=False).",
        f"Custom diagnostics are labelled {CUSTOM_DIAGNOSTICS_LABEL} and are never "
        "official benchmark metrics or effectiveness evidence.",
        "Team Accuracy, Conditional Selection Accuracy and Interaction Cost are "
        "NOT_EXECUTED: they need the CoPref user simulator plus a hosted LLM API; "
        "Team Accuracy is not replaced by top-1/recall@1 and no simulated-user result "
        "is fabricated.",
        "Hidden benchmark targets are sealed from the public task view, the candidate "
        "snapshot, the ranking callable and this report.",
        "CoShop is an external dependency: its source is never vendored (no upstream "
        "licence) and this adapter never downloads data by itself.",
    )


def _validate_request(
    domain: str,
    version: str,
    split: str,
    limit: int,
    k: int,
) -> None:
    """Validate public arguments up front (programming errors, not environment)."""
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"domain must be one of {SUPPORTED_DOMAINS}, got {domain!r}")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"version must be one of {SUPPORTED_VERSIONS}, got {version!r}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer number of tasks")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")


def _dataset_length(dataset: Any) -> int:
    """Return the number of tasks exposed by the official dataset split."""
    try:
        length = len(dataset)
    except Exception as error:
        raise CoshopContractError(
            "the official dataset exposes no task count; the requested task range "
            f"cannot be validated: {_bounded_error(error)}"
        ) from error
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise CoshopContractError("the official dataset task count is not a valid integer")
    return length


def _get_spec(dataset: Any, index: int) -> Any:
    """Return the official spec at ``index`` (lazily loaded by the official dataset)."""
    try:
        spec = dataset[index]
    except (KeyError, IndexError) as error:
        raise CoshopContractError(
            f"the official dataset does not expose task index {index}: {_bounded_error(error)}"
        ) from error
    except Exception as error:
        raise CoshopUnavailable(
            f"the official dataset could not load task index {index} (missing or "
            f"unreachable assets): {_bounded_error(error)}"
        ) from error
    if spec is None:
        raise CoshopContractError(
            f"the official dataset returned no spec for task index {index}"
        )
    return spec


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #


def evaluate_coshop(
    domain: str = DEFAULT_DOMAIN,
    version: str = DEFAULT_VERSION,
    split: str = DEFAULT_SPLIT,
    limit: int = DEFAULT_LIMIT,
    k: int = DEFAULT_K,
    coshop_root: str | Path | None = None,
    dataset_factory: Callable[..., Any] | None = None,
    metrics_function: Callable[..., Any] | None = None,
    ranker: Callable[[CoshopTaskView, Any], Sequence[str]] | None = None,
) -> CoshopEvaluationReport:
    """Run the official visible-query retrieval baseline over CoShop tasks.

    This executes **one retrieval pass** per task and reuses the official CoShop
    metrics; it does not simulate a user, does not converse and therefore cannot
    produce the official dialogue/decision metrics (they are recorded as
    ``NOT_EXECUTED``).

    Parameters
    ----------
    domain, version, split:
        Official CoShop dataset coordinates.
    limit:
        Number of consecutive tasks (spec indices ``0..limit-1``) to evaluate.
    k:
        Top-k passed to the official metric function (and used for the custom
        retrieval diagnostic).
    coshop_root:
        External official CoShop checkout.  Falls back to
        ``AGENTRECX_COSHOP_ROOT``.  Ignored when both ``dataset_factory`` and
        ``metrics_function`` are injected.
    dataset_factory, metrics_function:
        Test/extension seams: inject duck-typed dataset and official-metrics
        implementations to run without CoShop, pandas or network access.
    ranker:
        Optional ranking callable ``(task_view, candidate_snapshot) -> ids``.  It
        may only reorder the frozen candidate set; the default callable is the
        official visible-query baseline order.  Hidden targets are never passed.
    """
    _validate_request(domain, version, split, limit, k)

    inferred_root: Path | None = None
    package_version: str | None = None
    get_dataset: Callable[..., Any] | None = None
    if dataset_factory is None or metrics_function is None:
        inferred_root = resolve_coshop_root(coshop_root)
        get_dataset, official_metrics_function, package_version = _import_coshop_entrypoints(
            inferred_root
        )
        if metrics_function is None:
            metrics_function = official_metrics_function

    if dataset_factory is not None:
        try:
            dataset = dataset_factory(domain=domain, version=version, split=split)
        except Exception as error:
            raise CoshopUnavailable(
                "the injected official CoShop dataset factory failed; no data was "
                f"fabricated: {_bounded_error(error)}"
            ) from error
    else:
        assert get_dataset is not None  # for type checkers; guaranteed above
        try:
            dataset = get_dataset(
                domain,
                version=version,
                split=split,
                max_xstar=1,
                seed=DEFAULT_SEED,
            )
        except Exception as error:
            raise CoshopUnavailable(
                "the official CoShop dataset could not be loaded; its Google Drive "
                "assets must be downloaded once into the external checkout and be "
                f"reachable (no data was fabricated): {_bounded_error(error)}"
            ) from error

    catalog = getattr(dataset, "catalog", None)
    if catalog is None:
        raise CoshopContractError("the official dataset exposes no catalog")
    catalog_ids = _catalog_ids(catalog)
    item_id_by_catalog_order = {
        identity: position for position, identity in enumerate(catalog_ids, start=1)
    }
    catalog_digest = _sha256_ids(catalog_ids)
    task_count = _dataset_length(dataset)
    if limit > task_count:
        raise CoshopContractError(
            f"requested {limit} tasks but the official {domain}/{version}/{split} split "
            f"exposes only {task_count}; refusing to silently evaluate fewer tasks"
        )

    effective_ranker = ranker or rank_baseline_candidates
    task_records: list[TaskEvaluationRecord] = []
    for task_index in range(limit):
        spec = _get_spec(dataset, task_index)
        task_view = build_task_view(
            spec=spec,
            domain=domain,
            version=version,
            split=split,
            task_index=task_index,
        )
        hidden_targets = extract_hidden_targets(spec=spec, task_id=task_view.task_id)
        snapshot = _to_candidate_snapshot(
            task_view, item_id_by_catalog_order=item_id_by_catalog_order
        )
        ranked_ids = _validate_ranking(
            effective_ranker(task_view, snapshot), task_view=task_view
        )
        official = _run_official_metrics(
            metrics_function=metrics_function,
            spec=spec,
            ranked_item_ids=ranked_ids,
            catalog=catalog,
            k=k,
        )
        custom = _custom_diagnostics(
            task_view=task_view,
            ranked_ids=ranked_ids,
            hidden_targets=hidden_targets,
            k=k,
        )
        task_records.append(
            TaskEvaluationRecord(
                task_id=task_view.task_id,
                visible_query=task_view.visible_query,
                candidate_count=len(task_view.candidate_ids),
                ranked_candidate_ids=ranked_ids,
                official_metrics=official,
                custom_diagnostics=custom,
            )
        )

    official_metrics = _aggregate_official_metrics(task_records, k)
    custom_diagnostics = _aggregate_custom_diagnostics(task_records)
    provenance = {
        "adapter_version": ADAPTER_VERSION,
        "adapter_module": __name__,
        "protocol": PROTOCOL_NAME,
        "benchmark": "coshop",
        "benchmark_kind": "external",
        "dataset": {"domain": domain, "version": version, "split": split},
        "tasks": {
            "requested_limit": limit,
            "evaluated": len(task_records),
            "indices": list(range(len(task_records))),
        },
        "catalog": {"id_count": len(catalog_ids), "ordered_ids_sha256": catalog_digest},
        "baseline": BASELINE_NAME,
        "k": k,
        "seed": DEFAULT_SEED,
        "sampling": "none (deterministic baseline; seed recorded for protocol only)",
        "coshop_root": str(inferred_root) if inferred_root is not None else None,
        "coshop_package_version": package_version,
        "dataset_injected": dataset_factory is not None,
        "metrics_injected": metrics_function is not None,
        "official_assets": (
            "official CoShop Google Drive data assets are required in the external "
            "checkout; the adapter itself never downloads data and never fabricates it"
        ),
    }
    report = CoshopEvaluationReport(
        adapter_version=ADAPTER_VERSION,
        protocol=PROTOCOL_NAME,
        benchmark="coshop",
        domain=domain,
        version=version,
        split=split,
        task_count=len(task_records),
        candidate_count_total=sum(len(record.ranked_candidate_ids) for record in task_records),
        k=k,
        baseline=BASELINE_NAME,
        candidate_source=CANDIDATE_SOURCE,
        tasks=tuple(task_records),
        official_metrics=official_metrics,
        custom_diagnostics=custom_diagnostics,
        decision_metrics=_decision_metrics(),
        provenance=provenance,
        notes=_report_notes(),
    )
    return report.with_digest()


def _aggregate_official_metrics(
    task_records: Sequence[TaskEvaluationRecord], k: int
) -> dict[str, Any]:
    """Aggregate the per-task official metrics (means over executed tasks)."""
    recall_at_k = [record.official_metrics["recall_at_k"] for record in task_records]
    recall_at_1 = [record.official_metrics["recall_at_1"] for record in task_records]
    ndcg_ustar_at_k = [
        record.official_metrics["ndcg_ustar_at_k"] for record in task_records
    ]
    return {
        "source": OFFICIAL_METRICS_SOURCE,
        "k": k,
        "compute_embedding_ustar": False,
        "tasks_evaluated": len(task_records),
        "tasks_with_recall_at_k": sum(1 for value in recall_at_k if value is not None),
        "recall_at_k": _mean(recall_at_k),
        "recall_at_1": _mean(recall_at_1),
        "ndcg_ustar_at_k": _mean(ndcg_ustar_at_k),
        "per_task": [
            {
                "task_id": record.task_id,
                "recall_at_k": record.official_metrics["recall_at_k"],
                "recall_at_1": record.official_metrics["recall_at_1"],
                "ndcg_ustar_at_k": record.official_metrics["ndcg_ustar_at_k"],
            }
            for record in task_records
        ],
    }


def _aggregate_custom_diagnostics(
    task_records: Sequence[TaskEvaluationRecord],
) -> dict[str, Any]:
    """Aggregate the custom retrieval diagnostics (never official metrics)."""
    hits = [
        bool(record.custom_diagnostics["target_in_candidate_pool"])
        for record in task_records
    ]
    ranks = [
        record.custom_diagnostics["target_original_rank_in_pool"] for record in task_records
    ]
    return {
        "label": CUSTOM_DIAGNOSTICS_LABEL,
        "baseline": BASELINE_NAME,
        "tasks_evaluated": len(task_records),
        "candidate_count_total": sum(
            record.custom_diagnostics["candidate_count"] for record in task_records
        ),
        "retrieval_calls": sum(
            record.custom_diagnostics["retrieval_calls"] for record in task_records
        ),
        "dialogue_turns": 0,
        "clarifying_questions": 0,
        "tool_calls": sum(record.custom_diagnostics["tool_calls"] for record in task_records),
        "target_in_candidate_pool_rate": _mean([1.0 if hit else 0.0 for hit in hits]),
        "tasks_with_target_in_candidate_pool": sum(1 for hit in hits if hit),
        "mean_target_original_rank_in_pool": _mean(
            [float(rank) if rank is not None else None for rank in ranks]
        ),
        "mean_custom_recall_at_k": _mean(
            [record.custom_diagnostics["custom_recall_at_k"] for record in task_records]
        ),
        "per_task": [
            dict(record.custom_diagnostics) for record in task_records
        ],
    }
