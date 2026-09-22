"""AgentRec-X side of the public-TIGER backend boundary (L2 of the four-layer model).

This module is the **only** bridge between AgentRec-X and a TigerBackend.  It:

* writes the item-id-only handoff (``CatalogueHandoff``, ``TrainExposureHandoff``,
  ``EvalCohortHandoff``) from frozen AgentRec-X objects;
* computes ``required_frontier = K_max + |seen|`` per case, in this process, and writes only
  the resulting **integer** - never the seen-item identities;
* invokes the backend as a **subprocess** and parses its ``ScoreBatch`` stream or shards;
* converts each batch into the frozen evaluator's own
  ``(histories, targets, scores)`` shape, so ``evaluate_batched`` is used unchanged.

What this module deliberately is **not**:

* it holds **no** ML import - standard library, NumPy and ``recommendation.config`` only.  A
  backend lives in its own virtual environment, so ``torch``/``transformers`` are absent here
  by construction;
* it holds **no** backend import.  The contract is re-declared (see :class:`ScoreBatchData`) and
  the crossing is files plus a process, which is what keeps the two sides independently
  testable;
* it never reads, accepts, stores or returns ``parent_asin``.  Canonical identity belongs to
  the layers outside this module: the materialiser composes text and re-attaches identity, and
  this adapter only ever sees an opaque ``item_id``.

See ``docs/TIGER_BACKEND.md``.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "PAD_SENTINEL",
    "AdaptersError",
    "BackendProcessError",
    "BackendScoreBatch",
    "CatalogueHandoffData",
    "ContractViolation",
    "EvalCohortHandoffData",
    "FrontierStats",
    "HandoffPaths",
    "SIDResolver",
    "ScoresIndex",
    "TigerBackendAdapter",
    "TrainExposureHandoffData",
    "build_token_layout",
    "collision_audit",
    "frontier_stats",
    "required_frontier",
]

#: Must equal the backend's ``tiger_public.contracts.CONTRACT_VERSION``.  Checked on every
#: read on both sides, so a schema drift cannot pass silently.
CONTRACT_VERSION = "agentrecx.tiger_backend.v3"

#: The reserved PAD row value.  No real digit is negative, so a PAD row can never be
#: tokenised, resolved, or confused with the legal code ``0``.
PAD_SENTINEL = -1

#: The frozen cut-offs; ``K_max`` drives the certification frontier request.
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)

#: The frozen cohort seed (Phase 5).
DEFAULT_COHORT_SEED = 20260201

_LENGTH_PREFIX = struct.Struct("<I")


class AdaptersError(RuntimeError):
    """Base class for adapter failures."""


class ContractViolation(AdaptersError):
    """Raised when an artifact or argument violates the Revision-3 contract."""


class BackendProcessError(AdaptersError):
    """Raised when the backend subprocess fails, with its stderr attached."""


# --------------------------------------------------------------------------- #
# Mirrored contract records (no backend import; the schema is the contract)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CatalogueHandoffData:
    """Mirror of ``tiger_public.contracts.CatalogueHandoff``."""

    num_items: int
    num_users: int
    pad_id: int
    first_real_id: int
    mappings_sha256: str
    sequences_sha256: str
    products_sha256: str
    item_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.num_items < 1:
            raise ContractViolation(f"num_items must be >= 1, got {self.num_items}")
        if self.pad_id != 0:
            raise ContractViolation(f"pad_id must be 0, got {self.pad_id}")
        if self.first_real_id != 1:
            raise ContractViolation(f"first_real_id must be 1, got {self.first_real_id}")
        if len(self.item_ids) != self.num_items:
            raise ContractViolation(
                f"item_ids must hold num_items = {self.num_items} entries, got "
                f"{len(self.item_ids)}"
            )
        if sorted(self.item_ids) != list(range(self.first_real_id, self.num_items + 1)):
            raise ContractViolation(
                "item_ids must be exactly the real item ids 1..num_items, each once, in "
                "backend_row order"
            )
        for name in ("mappings_sha256", "sequences_sha256", "products_sha256"):
            _require_sha256(getattr(self, name), name)

    def row_of(self, item_id: int) -> int:
        if item_id == self.pad_id:
            raise ContractViolation("PAD (item id 0) is not an item and has no backend row")
        if not self.first_real_id <= item_id <= self.num_items:
            raise ContractViolation(f"item id {item_id} is outside 1..{self.num_items}")
        return item_id - self.first_real_id


@dataclass(frozen=True)
class TrainExposureHandoffData:
    """Mirror of ``tiger_public.contracts.TrainExposureHandoff``.

    ``rows`` are ``train_history`` item ids and nothing else.  The builder that produces this
    record (:func:`TigerBackendAdapter.materialise_train_exposure`) is handed
    ``EvaluationCase.train_history`` values, so a target is never in scope.
    """

    protocol: str
    protocol_version: str
    field_source: str
    users: int
    examples: int
    rows: tuple[tuple[int, ...], ...]
    examples_sha256: str
    catalogue_sha256: str

    def validate(self, *, catalogue: CatalogueHandoffData | None = None) -> None:
        if self.field_source != "EvaluationCase.train_history":
            raise ContractViolation(
                f"field_source must be EvaluationCase.train_history, got {self.field_source!r}"
            )
        if self.protocol != "temporal_leave_two_out":
            raise ContractViolation(
                f"protocol must be temporal_leave_two_out, got {self.protocol!r}"
            )
        if self.protocol_version != "agentrecx.eval_protocol.v1":
            raise ContractViolation(
                "protocol_version must be agentrecx.eval_protocol.v1, got "
                f"{self.protocol_version!r}"
            )
        expected = sum(len(row) - 1 for row in self.rows)
        if self.examples != expected:
            raise ContractViolation(
                f"examples must equal the next-item transition count {expected} over "
                f"{len(self.rows)} rows, got {self.examples}"
            )
        if len(self.rows) > self.examples:
            raise ContractViolation("a history row contributes at least one transition")
        _require_sha256(self.examples_sha256, "examples_sha256")
        _require_sha256(self.catalogue_sha256, "catalogue_sha256")
        if catalogue is not None and self.catalogue_sha256 != catalogue.mappings_sha256:
            raise ContractViolation(
                "catalogue_sha256 does not match the catalogue mappings_sha256"
            )
        for index, row in enumerate(self.rows):
            if len(row) < 2:
                raise ContractViolation(
                    f"exposure row {index} has {len(row)} items; at least 2 are required"
                )
            _assert_real_item_ids(row, catalogue, context=f"exposure row {index}")


@dataclass(frozen=True)
class EvalCohortHandoffData:
    """Mirror of ``tiger_public.contracts.EvalCohortHandoff``.

    There is no target field, and no seen set: ``required_frontier`` is an integer count
    computed in this process.
    """

    cohort_seed: int
    cohort_size: int
    protocol_version: str
    k_values: tuple[int, ...]
    case_ids: tuple[int, ...]
    test_histories: tuple[tuple[int, ...], ...]
    required_frontier: tuple[int, ...]
    catalogue_sha256: str

    def validate(self, *, catalogue: CatalogueHandoffData | None = None) -> None:
        if self.protocol_version != "agentrecx.eval_protocol.v1":
            raise ContractViolation(
                "protocol_version must be agentrecx.eval_protocol.v1, got "
                f"{self.protocol_version!r}"
            )
        if not self.k_values:
            raise ContractViolation("k_values must not be empty")
        for name, values in (
            ("test_histories", self.test_histories),
            ("required_frontier", self.required_frontier),
            ("case_ids", self.case_ids),
        ):
            if len(values) != self.cohort_size:
                raise ContractViolation(
                    f"{name} must hold cohort_size = {self.cohort_size} entries, got "
                    f"{len(values)}"
                )
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ContractViolation("case_ids must be unique")
        k_max = max(self.k_values)
        if any(int(value) < k_max for value in self.required_frontier):
            raise ContractViolation(
                f"every required_frontier must be >= max(k_values) = {k_max}"
            )
        _require_sha256(self.catalogue_sha256, "catalogue_sha256")
        if catalogue is not None and self.catalogue_sha256 != catalogue.mappings_sha256:
            raise ContractViolation(
                "catalogue_sha256 does not match the catalogue mappings_sha256"
            )
        for index, history in enumerate(self.test_histories):
            _assert_real_item_ids(history, catalogue, context=f"cohort case {index} history")


@dataclass(frozen=True)
class FrontierStats:
    """Diagnostics recorded **before** a CERTIFIED attempt, so its cost is known in advance."""

    cases: int
    k_max: int
    minimum: int
    maximum: int
    mean: float
    p95: int
    p99: int
    histogram: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.mean, 6),
            "p95": self.p95,
            "p99": self.p99,
            "k_max": self.k_max,
            "cases": self.cases,
            "histogram": dict(self.histogram),
        }


@dataclass(frozen=True)
class BackendScoreBatch:
    """Mirror of ``tiger_public.contracts.ScoreBatch``, plus its validated conversion."""

    case_ids: tuple[int, ...]
    scores: np.ndarray
    reachable: np.ndarray
    reachable_real_count: np.ndarray
    status: str
    certification: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    generation: Mapping[str, Any]


@dataclass(frozen=True)
class HandoffPaths:
    """Where the three handoff files live, so both sides agree without configuration drift."""

    root: Path

    @property
    def catalogue(self) -> Path:
        return self.root / "catalogue.json"

    @property
    def catalogue_items(self) -> Path:
        return self.root / "catalogue_items.jsonl"

    @property
    def products_text(self) -> Path:
        return self.root / "products_text.jsonl"

    @property
    def exposure(self) -> Path:
        return self.root / "train_exposure.jsonl"

    @property
    def exposure_record(self) -> Path:
        return self.root / "train_exposure.json"

    @property
    def cohort(self) -> Path:
        return self.root / "eval_cohort.json"

    @property
    def cohort_items(self) -> Path:
        return self.root / "eval_cohort.jsonl"

    @property
    def request(self) -> Path:
        return self.root / "score_request.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def require(self, *names: str) -> None:
        for name in names:
            path = self.root / name
            if not path.is_file():
                raise ContractViolation(f"missing handoff file {path}")


def required_frontier(history: Sequence[int], *, k_max: int = max(DEFAULT_K_VALUES)) -> int:
    """``K_max + |unique seen|`` for one case.

    The seen set is the evaluator's own - every distinct item in the full test-time history,
    which may be longer than the backend's truncated model window - so the result is **not**
    assumed small.
    """
    if k_max < 1:
        raise ContractViolation(f"k_max must be >= 1, got {k_max}")
    seen = {int(item) for item in history}
    if 0 in seen:
        raise ContractViolation("history contains item id 0 (PAD); histories must be real items")
    return k_max + len(seen)


def frontier_stats(frontiers: Sequence[int], *, k_max: int = max(DEFAULT_K_VALUES)) -> FrontierStats:
    """min/max/mean/p95/p99 plus a histogram, recorded before any certified run."""
    values = np.asarray([int(value) for value in frontiers], dtype=np.int64)
    if values.size == 0:
        raise ContractViolation("frontier_stats needs at least one case")
    if int(values.min()) < k_max:
        raise ContractViolation(
            f"minimum required_frontier {int(values.min())} is below k_max {k_max}"
        )
    buckets = (0, 25, 50, 100, 250, 500)
    histogram: dict[str, int] = {}
    lower = int(values.min())
    for upper in buckets:
        if upper <= lower:
            continue
        count = int(((values > lower) & (values <= upper)).sum())
        if count:
            histogram[f"({lower},{upper}]"] = count
        lower = max(lower, upper)
    histogram[f">{lower}"] = int((values > lower).sum())
    return FrontierStats(
        cases=int(values.size),
        k_max=k_max,
        minimum=int(values.min()),
        maximum=int(values.max()),
        mean=float(values.mean()),
        p95=int(np.percentile(values, 95)),
        p99=int(np.percentile(values, 99)),
        histogram={key: value for key, value in histogram.items() if value},
    )


def build_token_layout(*, levels: int, codebook_size: int, dedup_levels: int = 0,
                       dedup_vocab_size: int | None = None) -> dict[str, Any]:
    """Derive the token layout identically to the backend's ``build_token_layout``.

    Duplicated on purpose: the adapter must be able to check a SID's tokenisation without
    importing the backend package.  The arithmetic is the contract, and a test asserts both
    sides agree.
    """
    if levels < 1 or codebook_size < 1:
        raise ContractViolation("levels and codebook_size must be >= 1")
    if dedup_levels not in (0, 1):
        raise ContractViolation(f"dedup_levels must be 0 or 1, got {dedup_levels}")
    dedup_size = codebook_size if dedup_vocab_size is None else int(dedup_vocab_size)
    offsets = [index * codebook_size for index in range(levels + dedup_levels)]
    code_space = (levels + dedup_levels) * codebook_size
    return {
        "levels": levels,
        "dedup_levels": dedup_levels,
        "codebook_size": codebook_size,
        "dedup_vocab_size": dedup_size,
        "per_item_tokens": levels + dedup_levels,
        "level_offsets": offsets,
        "special": {"pad": code_space, "bos": code_space + 1, "eos": code_space + 2},
        "vocab_size": code_space + 3,
        "code_space": code_space,
        "sid_to_token": "token = level_offsets[l] + code_l",
        "sentinel_tokenisable": False,
    }


def collision_audit(
    real_codes: Sequence[Sequence[int]], *, dedup_levels: int, dedup_vocab_size: int
) -> dict[str, Any]:
    """Both collision blocks, computed identically to the backend's.

    The overflow check reads ``pre_dedup.largest_group``; acceptance reads ``post_dedup``.
    Keeping both preserves the representation-quality diagnostic instead of collapsing it.
    """
    total = len(real_codes)
    if total == 0:
        raise ContractViolation("collision_audit needs at least one item")
    pre: dict[tuple[int, ...], int] = {}
    for codes in real_codes:
        key = tuple(int(value) for value in codes)
        pre[key] = pre.get(key, 0) + 1
    pre_hits = {key: count for key, count in pre.items() if count > 1}
    pre_items = sum(pre_hits.values())
    pre_largest = max(pre.values())
    # The dedup digit is a per-group ordinal in first-come item order, exactly as the backend
    # assigns it, so the post-dedup block reports the assignment that would actually be built.
    post: dict[tuple[int, ...], int] = {}
    ordinals: dict[tuple[int, ...], int] = {}
    for codes in real_codes:
        key = tuple(int(value) for value in codes)
        if dedup_levels:
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            full = key + (ordinal,)
        else:
            full = key
        post[full] = post.get(full, 0) + 1
    post_hits = {key: count for key, count in post.items() if count > 1}
    post_items = sum(post_hits.values())

    def rate(count: int) -> float:
        return round(count / total, 6)

    return {
        "pre_dedup": {
            "collision_groups": len(pre_hits),
            "items_in_collision": pre_items,
            "collision_rate": rate(pre_items),
            "largest_group": pre_largest,
            "distinct_sids": len(pre),
        },
        "post_dedup": {
            "collision_groups": len(post_hits),
            "items_in_collision": post_items,
            "collision_rate": rate(post_items),
            "largest_group": max(post.values()) if post else 0,
            "distinct_sids": len(post),
        },
        "dedup_vocab_size": int(dedup_vocab_size),
        "dedup_scheme": "collision_ordinal",
        "first_member_zero": True,
        "largest_group_le_dedup_vocab": pre_largest <= dedup_vocab_size,
        "overflow_refusal_triggered": pre_largest > dedup_vocab_size,
    }


@dataclass(frozen=True)
class ScoresIndex:
    """The backend's sharded-score index, validated against the cohort it claims to cover."""

    num_items: int
    cohort_size: int
    batch_size: int
    shards: tuple[Mapping[str, Any], ...]
    row_offsets: tuple[int, ...]
    total_rows: int
    status: str
    certification: Mapping[str, Any] = field(default_factory=dict)
    retrieval: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ScoresIndex:
        if payload.get("format") != "agentrecx.tiger.scores_index.v3":
            raise ContractViolation(
                f"scores index format must be agentrecx.tiger.scores_index.v3, got "
                f"{payload.get('format')!r}"
            )
        _require_contract_version(payload.get("contract_version"))
        try:
            index = cls(
                num_items=int(payload["num_items"]),
                cohort_size=int(payload["cohort_size"]),
                batch_size=int(payload["batch_size"]),
                shards=tuple(payload.get("shards") or ()),
                row_offsets=tuple(int(value) for value in payload.get("row_offsets") or ()),
                total_rows=int(payload["total_rows"]),
                status=str(payload["status"]),
                certification=dict(payload.get("certification") or {}),
                retrieval=dict(payload.get("retrieval") or {}),
            )
        except KeyError as error:
            raise ContractViolation(f"scores index is missing {error.args[0]!r}") from error
        if index.status not in ("APPROXIMATE", "CERTIFIED"):
            raise ContractViolation(f"unknown retrieval status {index.status!r}")
        if len(index.shards) != len(index.row_offsets):
            raise ContractViolation("row_offsets must have one entry per shard")
        if index.row_offsets and index.row_offsets[0] != 0:
            raise ContractViolation("row_offsets must start at 0")
        summed = sum(int(shard.get("rows", 0)) for shard in index.shards)
        if summed != index.total_rows:
            raise ContractViolation(
                f"shard rows sum to {summed} but total_rows is {index.total_rows}"
            )
        if index.total_rows != index.cohort_size:
            raise ContractViolation(
                f"total_rows {index.total_rows} does not cover cohort_size "
                f"{index.cohort_size}"
            )
        return index


class SIDResolver:
    """Deterministic ``SID -> item_id`` resolution against the catalogue assignment.

    Built from rows ``1..num_items`` of the assignment, so the reserved PAD row is **ignored by
    construction** and ``resolve`` fails closed on a code no item occupies: never a nearest
    neighbour, a partial match or a default.
    """

    def __init__(self, assignment: Sequence[Sequence[int]], *, item_ids: Sequence[int]) -> None:
        if len(assignment) < 2:
            raise ContractViolation("an assignment must hold at least PAD plus one item")
        if len(assignment) - 1 != len(item_ids):
            raise ContractViolation(
                f"assignment holds {len(assignment) - 1} items but item_ids holds "
                f"{len(item_ids)}"
            )
        self._item_ids = tuple(int(value) for value in item_ids)
        table: dict[tuple[int, ...], list[int]] = {}
        for index, codes in enumerate(assignment[1:], start=0):
            key = tuple(int(value) for value in codes)
            if any(value < 0 for value in key):
                raise ContractViolation(
                    f"real item at backend_row {index} holds a negative digit; only the "
                    "reserved PAD row may carry the sentinel"
                )
            table.setdefault(key, []).append(self._item_ids[index])
        self._table = {key: tuple(sorted(members)) for key, members in table.items()}

    @property
    def code_count(self) -> int:
        return len(self._table)

    def members(self, semantic_id: Sequence[int]) -> tuple[int, ...]:
        return self._table.get(tuple(int(value) for value in semantic_id), ())

    def resolve(self, semantic_id: Sequence[int]) -> int | None:
        members = self.members(semantic_id)
        return members[0] if members else None

    def collision_sizes(self) -> tuple[int, ...]:
        return tuple(len(members) for members in self._table.values())


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class TigerBackendAdapter:
    """Drive the backend over a filesystem + subprocess boundary.

    The adapter never imports a backend module and never imports a numeric stack beyond NumPy.
    ``backend_python`` defaults to the running interpreter so the Step-2.3 placeholder (which
    needs only NumPy) can be exercised in this environment; a real deployment points it at
    ``backends/tiger_public/.venv/bin/python``.
    """

    def __init__(
        self,
        handoff_dir: str | Path,
        *,
        backend_root: str | Path | None = None,
        backend_python: str | Path | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.paths = HandoffPaths(Path(handoff_dir))
        if backend_root is None:
            backend_root = Path(__file__).resolve().parents[2] / "backends" / "tiger_public"
        self.backend_root = Path(backend_root)
        self.backend_python = str(backend_python or sys.executable)
        self.timeout_seconds = float(timeout_seconds)

    # -- materialisation (L1 writes; this method only knows item ids) ------ #

    def materialise_catalogue(
        self,
        *,
        item_ids: Sequence[int],
        num_users: int,
        mappings_sha256: str,
        sequences_sha256: str,
        products_sha256: str,
    ) -> CatalogueHandoffData:
        """Write ``catalogue.json`` + ``catalogue_items.jsonl`` from an explicit id table."""
        catalogue = CatalogueHandoffData(
            num_items=len(item_ids),
            num_users=int(num_users),
            pad_id=0,
            first_real_id=1,
            mappings_sha256=mappings_sha256,
            sequences_sha256=sequences_sha256,
            products_sha256=products_sha256,
            item_ids=tuple(int(value) for value in item_ids),
        )
        catalogue.validate()
        self.paths.root.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.paths.catalogue,
            {
                "format": "agentrecx.tiger.catalogue.v3",
                "contract_version": CONTRACT_VERSION,
                "num_items": catalogue.num_items,
                "num_users": catalogue.num_users,
                "pad_id": catalogue.pad_id,
                "first_real_id": catalogue.first_real_id,
                "item_id_span": [catalogue.first_real_id, catalogue.num_items],
                "mappings_sha256": catalogue.mappings_sha256,
                "sequences_sha256": catalogue.sequences_sha256,
                "products_sha256": catalogue.products_sha256,
            },
        )
        _write_jsonl(
            self.paths.catalogue_items,
            (
                {"backend_row": row, "item_id": item_id}
                for row, item_id in enumerate(catalogue.item_ids)
            ),
        )
        return catalogue

    def materialise_products_text(
        self, texts: Sequence[str], *, catalogue: CatalogueHandoffData | None = None
    ) -> str:
        """Write pre-composed item text, keyed by opaque ``item_id`` only.

        The text is composed in L1, from the AgentRec-X catalogue artifact; this method never
        sees a text field name, and a canonical-identity string has no way in or out.  The
        caller may pass the catalogue it just wrote, so materialisation never has to re-read
        (and therefore never has to depend on the manifest existing yet).
        """
        resolved = catalogue or self.read_catalogue()
        if len(texts) != resolved.num_items:
            raise ContractViolation(
                f"products_text must hold {resolved.num_items} entries, got {len(texts)}"
            )
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ContractViolation(
                    f"products_text entry {index} must be a string, got {type(text).__name__}"
                )
        return _write_jsonl(
            self.paths.products_text,
            (
                {"item_id": item_id, "text": text}
                for item_id, text in zip(resolved.item_ids, texts, strict=True)
            ),
        )

    def materialise_train_exposure(
        self,
        *,
        train_histories: Sequence[Sequence[int]],
        catalogue: CatalogueHandoffData | None = None,
    ) -> TrainExposureHandoffData:
        """Write exposure rows from ``train_history`` values **only**.

        The parameter is named ``train_histories`` and the record declares
        ``field_source``; there is no argument through which a validation or test target could
        arrive, and the function never receives an ``EvaluationCase``.
        """
        resolved = catalogue or self.read_catalogue()
        rows = tuple(tuple(int(item) for item in history) for history in train_histories)
        for index, row in enumerate(rows):
            if len(row) < 2:
                raise ContractViolation(
                    f"train_history {index} has {len(row)} items; at least 2 are required to "
                    "form a next-item example"
                )
        # ``examples`` counts next-item transitions, one per (history, target) pair, which is
        # the same unit the accepted SASRec builder reports.  A row therefore contributes
        # ``len(row) - 1`` and never ``1``.
        examples = sum(len(row) - 1 for row in rows)
        payload = "".join(
            json.dumps({"row": list(row)}, sort_keys=True) + "\n" for row in rows
        )
        exposure = TrainExposureHandoffData(
            protocol="temporal_leave_two_out",
            protocol_version="agentrecx.eval_protocol.v1",
            field_source="EvaluationCase.train_history",
            users=len(rows),
            examples=examples,
            rows=rows,
            examples_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            catalogue_sha256=resolved.mappings_sha256,
        )
        exposure.validate(catalogue=resolved)
        _write_json(
            self.paths.exposure_record,
            {
                "format": "agentrecx.tiger.train_exposure.v3",
                "contract_version": CONTRACT_VERSION,
                "protocol": exposure.protocol,
                "protocol_version": exposure.protocol_version,
                "field_source": exposure.field_source,
                "users": exposure.users,
                "examples": exposure.examples,
                "examples_sha256": exposure.examples_sha256,
                "catalogue_sha256": exposure.catalogue_sha256,
            },
        )
        _write_jsonl(
            self.paths.exposure,
            ({"case_id": index, "items": list(row)} for index, row in enumerate(rows)),
        )
        return exposure

    def materialise_eval_cohort(
        self,
        *,
        test_histories: Sequence[Sequence[int]],
        catalogue: CatalogueHandoffData | None = None,
        cohort_seed: int = DEFAULT_COHORT_SEED,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> tuple[EvalCohortHandoffData, FrontierStats]:
        """Write cohort histories plus the integer frontier request.  No target is written.

        ``required_frontier`` is computed here, in AgentRec-X, from the evaluator's own seen
        set.  Only the integer is serialised: the backend receives ``required_frontier`` and
        never the identities behind it.
        """
        resolved = catalogue or self.read_catalogue()
        cutoffs = tuple(int(value) for value in k_values)
        k_max = max(cutoffs)
        histories = tuple(tuple(int(item) for item in history) for history in test_histories)
        frontiers = tuple(required_frontier(history, k_max=k_max) for history in histories)
        case_ids = tuple(range(len(histories)))
        # The adapter's own seen-set derivation must never reach the artifact, so it is not
        # stored anywhere; only the count above survives this block.
        cohort = EvalCohortHandoffData(
            cohort_seed=int(cohort_seed),
            cohort_size=len(histories),
            protocol_version="agentrecx.eval_protocol.v1",
            k_values=cutoffs,
            case_ids=case_ids,
            test_histories=histories,
            required_frontier=frontiers,
            catalogue_sha256=resolved.mappings_sha256,
        )
        cohort.validate(catalogue=resolved)
        stats = frontier_stats(frontiers, k_max=k_max)
        _write_json(
            self.paths.cohort,
            {
                "format": "agentrecx.tiger.eval_cohort.v3",
                "contract_version": CONTRACT_VERSION,
                "cohort_seed": cohort.cohort_seed,
                "cohort_size": cohort.cohort_size,
                "protocol_version": cohort.protocol_version,
                "k_values": list(cohort.k_values),
                "required_frontier_stats": stats.as_dict(),
                "catalogue_sha256": cohort.catalogue_sha256,
            },
        )
        _write_jsonl(
            self.paths.cohort_items,
            (
                {
                    "case_id": case_id,
                    "history": list(history),
                    "required_frontier": frontier,
                }
                for case_id, history, frontier in zip(
                    cohort.case_ids, cohort.test_histories, cohort.required_frontier, strict=True
                )
            ),
        )
        return cohort, stats

    def write_score_request(
        self, *, status: str, batch_size: int, sample_items: int = 64, seed: int = 2026
    ) -> None:
        if status not in ("APPROXIMATE", "CERTIFIED"):
            raise ContractViolation(f"status must be APPROXIMATE or CERTIFIED, got {status!r}")
        _write_json(
            self.paths.request,
            {
                "format": "agentrecx.tiger.score_request.v3",
                "contract_version": CONTRACT_VERSION,
                "status": status,
                "batch_size": int(batch_size),
                "sample_items": int(sample_items),
                "seed": int(seed),
            },
        )

    def write_manifest(self) -> dict[str, Any]:
        """Hash every handoff file so a reader can verify before it trusts anything."""
        files: dict[str, str] = {}
        for path in sorted(self.paths.root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            files[str(path.relative_to(self.paths.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        payload = {
            "format": "agentrecx.tiger.manifest.v3",
            "contract_version": CONTRACT_VERSION,
            "files": files,
        }
        _write_json(self.paths.manifest, payload)
        return payload

    # -- reading ----------------------------------------------------------- #

    def read_catalogue(self, *, verify: bool = True) -> CatalogueHandoffData:
        payload = self._read_record("catalogue.json", verify=verify)
        rows: list[int] = []
        for number, record in enumerate(_iter_jsonl(self.paths.catalogue_items), start=1):
            if set(record) != {"backend_row", "item_id"}:
                raise ContractViolation(
                    "catalogue_items.jsonl records must hold exactly "
                    f"{{'backend_row', 'item_id'}}, got {sorted(record)}"
                )
            if int(record["backend_row"]) != number - 1:
                raise ContractViolation(
                    f"catalogue_items.jsonl line {number} declares backend_row "
                    f"{record['backend_row']}; rows must ascend contiguously from 0"
                )
            rows.append(int(record["item_id"]))
        span = payload.get("item_id_span") or [0, 0]
        catalogue = CatalogueHandoffData(
            num_items=int(payload["num_items"]),
            num_users=int(payload["num_users"]),
            pad_id=int(payload["pad_id"]),
            first_real_id=int(payload["first_real_id"]),
            mappings_sha256=str(payload["mappings_sha256"]),
            sequences_sha256=str(payload["sequences_sha256"]),
            products_sha256=str(payload["products_sha256"]),
            item_ids=tuple(rows),
        )
        catalogue.validate()
        if (int(span[0]), int(span[1])) != (catalogue.first_real_id, catalogue.num_items):
            raise ContractViolation(
                f"item_id_span {tuple(span)} disagrees with num_items {catalogue.num_items}"
            )
        return catalogue

    def read_cohort(
        self, *, catalogue: CatalogueHandoffData | None = None, verify: bool = True
    ) -> EvalCohortHandoffData:
        payload = self._read_record("eval_cohort.json", verify=verify)
        resolved = catalogue or self.read_catalogue(verify=verify)
        case_ids: list[int] = []
        histories: list[tuple[int, ...]] = []
        frontiers: list[int] = []
        for number, record in enumerate(_iter_jsonl(self.paths.cohort_items), start=1):
            if set(record) != {"case_id", "history", "required_frontier"}:
                raise ContractViolation(
                    "eval_cohort.jsonl records must hold exactly "
                    f"{{'case_id', 'history', 'required_frontier'}}, got {sorted(record)}"
                )
            history = record["history"]
            if not isinstance(history, list) or not history:
                raise ContractViolation(
                    f"eval_cohort.jsonl line {number} needs a non-empty 'history' list"
                )
            case_ids.append(int(record["case_id"]))
            histories.append(tuple(int(item) for item in history))
            frontiers.append(int(record["required_frontier"]))
        cohort = EvalCohortHandoffData(
            cohort_seed=int(payload["cohort_seed"]),
            cohort_size=int(payload["cohort_size"]),
            protocol_version=str(payload["protocol_version"]),
            k_values=tuple(int(value) for value in payload.get("k_values") or ()),
            case_ids=tuple(case_ids),
            test_histories=tuple(histories),
            required_frontier=tuple(frontiers),
            catalogue_sha256=str(payload["catalogue_sha256"]),
        )
        cohort.validate(catalogue=resolved)
        return cohort

    def read_semantic_ids(self) -> dict[str, Any]:
        """Read a stage-2 ``semantic_ids.json`` from a directory the caller names.

        The adapter has no default location for it: Step 2.3 produces none.  The method exists
        so the *schema* checks live in one place when Step 2.4 lands.
        """
        raise NotImplementedError(
            "Step 2.3 has no stage-2 producer; see docs/TIGER_BACKEND.md section 11"
        )

    # -- backend invocation ------------------------------------------------ #

    def _backend_command(self, *args: str) -> list[str]:
        return [
            self.backend_python,
            "-m",
            "tiger_public.cli",
            *args,
        ]

    def run_stage(
        self,
        stage: str,
        *args: str,
        env_extra: Mapping[str, str] | None = None,
        stdout_to: Any | None = None,
    ) -> subprocess.CompletedProcess:
        """Run one backend stage as a subprocess, attaching stderr to any failure."""
        import os

        if stage not in ("build-features", "fit-sid", "train", "score"):
            raise ContractViolation(f"unknown backend stage {stage!r}")
        if not self.backend_root.is_dir():
            raise BackendProcessError(f"backend root {self.backend_root} does not exist")
        environment = dict(os.environ)
        source_root = str(self.backend_root / "src")
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{existing}" if existing else source_root
        )
        if env_extra:
            environment.update({str(key): str(value) for key, value in env_extra.items()})
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._backend_command(stage, *args),
            cwd=str(self.backend_root),
            env=environment,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendProcessError(
                f"backend stage {stage!r} failed with exit code {completed.returncode}\n"
                f"stderr: {completed.stderr.decode('utf-8', 'replace').strip()}"
            )
        return completed

    def score_stream(
        self, *, status: str, batch_size: int, sample_items: int = 64, seed: int = 2026
    ) -> Iterator[BackendScoreBatch]:
        """Run ``score --stream`` and yield validated batches as they arrive.

        The stream is the production path: peak memory is one batch rather than the whole
        cohort, and every batch is validated and converted before it is yielded.
        """
        import os

        if status not in ("APPROXIMATE", "CERTIFIED"):
            raise ContractViolation(f"status must be APPROXIMATE or CERTIFIED, got {status!r}")
        catalogue = self.read_catalogue()
        cohort = self.read_cohort(catalogue=catalogue)
        self.paths.require("semantic_ids.json", "tiger.json")
        environment = dict(os.environ)
        source_root = str(self.backend_root / "src")
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{existing}" if existing else source_root
        )
        argv = self._backend_command(
            "score",
            "--cohort",
            str(self.paths.root),
            "--sid",
            str(self.paths.root),
            "--ckpt",
            str(self.paths.root),
            "--status",
            status.lower(),
            "--batch-size",
            str(int(batch_size)),
            "--sample-items",
            str(int(sample_items)),
            "--seed",
            str(int(seed)),
            "--stream",
        )
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=str(self.backend_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdout is not None
            while True:
                header_line = process.stdout.readline()
                if not header_line:
                    break
                if not header_line.strip():
                    continue
                header = json.loads(header_line.decode("utf-8"))
                scores = _read_framed_array(process.stdout)
                reachable = _read_framed_array(process.stdout)
                yield self._validate_batch(
                    header=header,
                    scores=scores,
                    reachable=reachable,
                    catalogue=catalogue,
                    cohort=cohort,
                )
        finally:
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
            code = process.wait(timeout=self.timeout_seconds)
            if code != 0:
                raise BackendProcessError(
                    f"backend score stream failed with exit code {code}\n"
                    f"stderr: {stderr.decode('utf-8', 'replace').strip()}"
                )

    def read_score_shards(
        self, *, status: str | None = None
    ) -> Iterator[BackendScoreBatch]:
        """Read and validate a sharded score artifact, in shard order."""
        catalogue = self.read_catalogue()
        cohort = self.read_cohort(catalogue=catalogue)
        index_path = self.paths.root / "scores" / "index.json"
        if not index_path.is_file():
            raise ContractViolation(f"missing shard index {index_path}")
        index = ScoresIndex.from_payload(json.loads(index_path.read_text(encoding="utf-8")))
        if index.num_items != catalogue.num_items:
            raise ContractViolation(
                f"score index num_items {index.num_items} disagrees with the catalogue "
                f"{catalogue.num_items}"
            )
        if index.cohort_size != cohort.cohort_size:
            raise ContractViolation(
                f"score index cohort_size {index.cohort_size} disagrees with the cohort "
                f"{cohort.cohort_size}"
            )
        if status is not None and index.status != status:
            raise ContractViolation(
                f"score index status {index.status!r} is not the requested {status!r}"
            )
        for shard in index.shards:
            path = self.paths.root / str(shard["name"])
            if not path.is_file():
                raise ContractViolation(f"missing score shard {path}")
            with np.load(path, allow_pickle=False) as payload:
                required = {"scores", "reachable", "reachable_real_count", "case_ids"}
                missing = required - set(payload.files)
                if missing:
                    raise ContractViolation(
                        f"shard {path.name} is missing arrays {sorted(missing)}"
                    )
                yield self._validate_batch(
                    header={
                        "format": "agentrecx.tiger.scores.v3",
                        "contract_version": CONTRACT_VERSION,
                        "case_ids": [int(value) for value in payload["case_ids"]],
                        "status": index.status,
                        "reachable_real_count": [
                            int(value) for value in payload["reachable_real_count"]
                        ],
                        "certification": dict(index.certification or {}),
                        "retrieval": dict(index.retrieval or {}),
                        "generation": {},
                    },
                    scores=np.asarray(payload["scores"]),
                    reachable=np.asarray(payload["reachable"]),
                    catalogue=catalogue,
                    cohort=cohort,
                )

    # -- validation and conversion ----------------------------------------- #

    def _validate_batch(
        self,
        *,
        header: Mapping[str, Any],
        scores: np.ndarray,
        reachable: np.ndarray,
        catalogue: CatalogueHandoffData,
        cohort: EvalCohortHandoffData,
    ) -> BackendScoreBatch:
        """Validate one batch against the Revision-3 contract and prepare it for the evaluator.

        Order of checks matters: the *receipt* is verified first, then the shape, then the
        empty-frontier refusal (a tail score is only defined when at least one real item is
        reachable).  Nothing is filled before those pass, so a malformed batch can never
        produce a plausible score matrix.
        """
        if header.get("format") != "agentrecx.tiger.scores.v3":
            raise ContractViolation(
                f"score batch format must be agentrecx.tiger.scores.v3, got "
                f"{header.get('format')!r}"
            )
        _require_contract_version(header.get("contract_version"))
        status = str(header.get("status"))
        if status not in ("APPROXIMATE", "CERTIFIED"):
            raise ContractViolation(f"unknown retrieval status {status!r}")
        case_ids = tuple(int(value) for value in header.get("case_ids") or ())
        if not case_ids:
            raise ContractViolation("a score batch must hold at least one case")
        if len(set(case_ids)) != len(case_ids):
            raise ContractViolation("score batch case_ids must be unique")
        batch = len(case_ids)
        expected = (batch, catalogue.num_items + 1)
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        reachable = np.ascontiguousarray(reachable, dtype=np.bool_)
        if tuple(scores.shape) != expected:
            raise ContractViolation(
                f"scores shape {tuple(scores.shape)} must be [batch, num_items + 1] = "
                f"{expected}"
            )
        if tuple(reachable.shape) != expected:
            raise ContractViolation(
                f"reachable shape {tuple(reachable.shape)} must equal the scores shape "
                f"{expected}"
            )
        if not np.isfinite(scores).all():
            raise ContractViolation("scores must be finite; NaN/Inf is never a valid score")
        if np.any(scores[:, 0] != 0.0):
            raise ContractViolation("scores[:, 0] (the PAD column) must be exactly 0.0")
        if np.any(reachable[:, 0]):
            raise ContractViolation("the PAD column must never be reachable")

        real_reachable = reachable[:, 1:]
        counts = real_reachable.sum(axis=1).astype(np.int32)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            offenders = [int(case_ids[index]) for index in empty]
            raise ContractViolation(
                "refusing the run: no real item is reachable for case_ids "
                f"{offenders}; a tail score is undefined without at least one reachable item "
                "and no NaN/Inf fallback is permitted"
            )
        declared = np.asarray(
            [int(value) for value in header.get("reachable_real_count") or counts],
            dtype=np.int32,
        )
        if declared.shape == counts.shape and not np.array_equal(declared, counts):
            raise ContractViolation(
                "declared reachable_real_count disagrees with the reachable matrix"
            )

        # Tail fill: strictly below every reachable item in that row, so the evaluator's
        # finiteness check keeps its meaning and the documented tail convention holds.
        filled = scores.copy()
        filled[:, 0] = 0.0
        for row in range(batch):
            mask = reachable[row, 1:]
            tail = float(filled[row, 1:][mask].min()) - 1.0
            filled[row, 1:][~mask] = tail
        if not np.isfinite(filled).all():
            raise ContractViolation("the tail fill produced a non-finite score")

        return BackendScoreBatch(
            case_ids=case_ids,
            scores=filled,
            reachable=reachable,
            reachable_real_count=counts,
            status=status,
            certification=dict(header.get("certification") or {}),
            retrieval=dict(header.get("retrieval") or {}),
            generation=dict(header.get("generation") or {}),
        )

    def evaluation_batches(
        self,
        batches: Iterable[BackendScoreBatch],
        *,
        targets: Sequence[int],
        cohort: EvalCohortHandoffData | None = None,
    ) -> Iterator[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Any]]:
        """Convert validated batches into the frozen evaluator's own input shape.

        The returned triples are exactly what ``recommendation.evaluation.batched
        .evaluate_batched`` consumes: ``(histories, targets, scores)``.  The evaluator keeps
        ownership of PAD exclusion, seen-item masking, tie-breaking, ranking and metrics - this
        method neither masks nor sorts, and it never returns a ``parent_asin``.

        ``case_ids`` is verified against the frozen cohort order before any batch is yielded,
        so a reordered or duplicated stream is refused rather than scored.
        """
        resolved = cohort or self.read_cohort()
        expected = resolved.case_ids
        offset = 0
        for batch in batches:
            stop = offset + len(batch.case_ids)
            if batch.case_ids != expected[offset:stop]:
                raise ContractViolation(
                    f"score batch receipt {batch.case_ids} does not match the frozen cohort "
                    f"order {expected[offset:stop]}"
                )
            histories = resolved.test_histories[offset:stop]
            targets_slice = tuple(int(value) for value in targets[offset:stop])
            if len(targets_slice) != len(histories):
                raise ContractViolation(
                    f"targets holds {len(targets)} entries but the cohort has "
                    f"{resolved.cohort_size} cases"
                )
            offset = stop
            yield histories, targets_slice, _to_torch(batch.scores)
        if offset != resolved.cohort_size:
            raise ContractViolation(
                f"the score stream covered {offset} cases but the cohort has "
                f"{resolved.cohort_size}"
            )

    # -- helpers ----------------------------------------------------------- #

    def _read_record(self, name: str, *, verify: bool) -> dict[str, Any]:
        path = self.paths.root / name
        if not path.is_file():
            raise ContractViolation(f"missing handoff file {path}")
        if verify:
            self._verify_manifest(required=(name,))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ContractViolation(f"{name} must hold a JSON object")
        _assert_no_forbidden_keys(payload, context=name)
        _require_contract_version(payload.get("contract_version", CONTRACT_VERSION))
        return payload

    def _verify_manifest(self, *, required: Sequence[str]) -> dict[str, Any]:
        """Refuse a missing file, a non-digest, and a **content** mismatch.

        The digest comparison is the point: declaring a plausible-looking hash is not the same
        as the bytes on disk matching it.  Only the files a given read actually depends on are
        re-hashed, so a large handoff costs one hash per file read rather than a full sweep.
        """
        if not self.paths.manifest.is_file():
            raise ContractViolation(f"missing handoff manifest {self.paths.manifest}")
        payload = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            raise ContractViolation("the handoff manifest must declare a non-empty 'files'")
        for name, digest in files.items():
            text = str(digest or "")
            if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
                raise ContractViolation(
                    f"manifest declares a non-sha256 digest for {name!r}: {digest!r}"
                )
            if not (self.paths.root / name).is_file():
                raise ContractViolation(f"manifest declares {name!r}, which does not exist")
        missing = [name for name in required if name not in files]
        if missing:
            raise ContractViolation(
                f"the handoff manifest does not declare the required file(s) {missing}"
            )
        for name in required:
            actual = hashlib.sha256((self.paths.root / name).read_bytes()).hexdigest()
            if actual != str(files[name]).lower():
                raise ContractViolation(
                    f"{name} does not match its manifest digest; the handoff has been modified "
                    "since it was written"
                )
        return payload


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #

#: A handoff record may never carry one of these; their presence means a target leaked.
_FORBIDDEN_KEYS = frozenset(
    {
        "target",
        "targets",
        "label",
        "labels",
        "valid",
        "validation",
        "test",
        "test_target",
        "validation_target",
        "seen",
        "parent_asin",
    }
)


def _assert_no_forbidden_keys(payload: Mapping[str, Any], *, context: str) -> None:
    for key in payload:
        if str(key).lower() in _FORBIDDEN_KEYS:
            raise ContractViolation(
                f"{context} declares forbidden key {key!r}; a handoff artifact must not "
                "carry a target, a split name, a seen set, or canonical identity"
            )


def _require_sha256(value: str, name: str) -> None:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ContractViolation(f"{name} must be a 64-character hex sha256 digest")


def _require_contract_version(value: Any) -> None:
    if value != CONTRACT_VERSION:
        raise ContractViolation(
            f"contract_version must be {CONTRACT_VERSION!r}, got {value!r}; the adapter and "
            "the backend must agree on the schema before any artifact crosses the boundary"
        )


def _assert_real_item_ids(
    item_ids: Sequence[int], catalogue: CatalogueHandoffData | None, *, context: str
) -> None:
    for item_id in item_ids:
        value = int(item_id)
        if value == 0 and (catalogue is None or catalogue.pad_id == 0):
            raise ContractViolation(f"{context} contains item id 0 (PAD)")
        if catalogue is not None and not catalogue.first_real_id <= value <= catalogue.num_items:
            raise ContractViolation(
                f"{context} contains item id {value}, outside "
                f"{catalogue.first_real_id}..{catalogue.num_items}"
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(dict(record), sort_keys=True, ensure_ascii=True) + "\n" for record in records
    )
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ContractViolation(f"missing handoff file {path}")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractViolation(f"{path.name} line {number} is not JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ContractViolation(f"{path.name} line {number} must be a JSON object")
        yield payload


def _read_framed_array(handle: Any) -> np.ndarray:
    prefix = handle.read(_LENGTH_PREFIX.size)
    if len(prefix) != _LENGTH_PREFIX.size:
        raise ContractViolation("the score stream ended mid-frame")
    (length,) = _LENGTH_PREFIX.unpack(prefix)
    payload = handle.read(length)
    if len(payload) != length:
        raise ContractViolation("the score stream ended mid-array")
    import io

    return np.load(io.BytesIO(payload), allow_pickle=False)


def _to_torch(scores: np.ndarray) -> Any:
    """Hand the evaluator a float32 torch tensor without importing torch at module scope.

    ``torch`` is a direct AgentRec-X dependency (``recommendation/inference/ranking.py``
    imports NumPy directly and the evaluator takes a tensor), so this is not a new dependency;
    the import is deferred only so the adapter module stays importable in a NumPy-only
    environment.
    """
    import torch

    return torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
