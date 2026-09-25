"""Step 2.7 TIGER engines: qualification fixtures, not serving code.

Two engines satisfy the seam :class:`recommendation.control.tiger_source.TigerSourceTool` needs:

``CertifiedEvidenceTigerEngine``
    Serves proposals from the **frozen certified H7 evidence** (``retrieval_candidates.jsonl`` +
    ``retrieval_summary.json``).  This makes the identity, grounding, ledger, deduplication and
    determinism gates run against *real* TIGER output while re-running no retrieval at all, so no
    TIGER semantic can be perturbed by qualification.

``FakeTigerEngine``
    A deterministic, injected engine used to drive the failure matrix.  It fabricates nothing that
    reaches a ledger: every failure mode it can produce is one the tool must refuse.

Neither is a serving engine and neither is registered anywhere.  Step 2.7 qualifies the source;
it does not enable it.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "CERTIFIED",
    "APPROXIMATE",
    "EngineRecommendation",
    "EngineResult",
    "CertifiedEvidenceTigerEngine",
    "FakeTigerEngine",
    "FAILURE_MODES",
]

CERTIFIED = "CERTIFIED"
APPROXIMATE = "APPROXIMATE"

#: The failure modes the qualification matrix drives through the tool.
FAILURE_MODES = (
    "backend_unavailable",
    "checkpoint_unavailable",
    "sid_artifact_unavailable",
    "invalid_sid",
    "unresolved_identity",
    "budget_exhausted",
    "non_finite_score",
    "timeout",
    "empty_result",
)

RESOLVER = {"candidate_item_for_path": "tiger_public.retrieve.SidResolver",
            "resolution": "token_path -> integer item_id (trie leaf)",
            "collision_policy": "smallest_item_id_wins"}


@dataclass(frozen=True)
class EngineRecommendation:
    parent_asin: str
    rank: int
    score: float
    semantic_id: tuple[int, ...]
    item_id: int


@dataclass(frozen=True)
class EngineResult:
    retrieval_mode: str
    recommendations: tuple[EngineRecommendation, ...]
    checkpoint_sha256: str
    sid_artifact_sha256: str
    resolver: Mapping[str, Any] = field(default_factory=lambda: dict(RESOLVER))


# --------------------------------------------------------------------------- #
# Frozen-evidence engine (real TIGER output, no retrieval re-run)
# --------------------------------------------------------------------------- #
class CertifiedEvidenceTigerEngine:
    """Serve one case's certified TIGER candidates from the frozen H7 artifact.

    Parameters
    ----------
    artifact_dir:
        The accepted H7 output directory.
    mappings_path:
        The frozen preprocessing mappings, used for the two-way identity conversion the seam
        requires (``parent_asin_to_item_id``) and for the item-id -> parent_asin readout.
    """

    def __init__(self, artifact_dir: str | Path, *, mappings_path: str | Path,
                 cache: dict[int, dict] | None = None, summary: dict | None = None) -> None:
        self.artifact_dir = Path(artifact_dir)
        payload = json.loads(Path(mappings_path).read_text(encoding="utf-8"))
        self._item2id: dict[str, int] = payload["item2id"]
        self._id2item: list[str | None] = payload["id2item"]
        self.num_items = int(payload["num_items"])
        self._summary = summary if summary is not None else json.loads(
            (self.artifact_dir / "retrieval_summary.json").read_text()
        )
        self._cache = cache if cache is not None else self._load_records()
        self._case_index = 0

    def _load_records(self) -> dict[int, dict]:
        records: dict[int, dict] = {}
        with (self.artifact_dir / "retrieval_candidates.jsonl").open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records[int(record["case_id"])] = record
        return records

    @property
    def summary(self) -> Mapping[str, Any]:
        return self._summary

    @property
    def checkpoint_sha256(self) -> str:
        return str(self._summary["hashes"]["checkpoint_tiger_pt"])

    @property
    def sid_artifact_sha256(self) -> str:
        return str(self._summary["hashes"]["sid_semantic_ids"])

    @property
    def retrieval_mode(self) -> str:
        return str(self._summary["status"])

    def for_case(self, case_index: int) -> "CertifiedEvidenceTigerEngine":
        """A view of this engine bound to one cohort case (cheap; shares the loaded records)."""
        clone = CertifiedEvidenceTigerEngine.__new__(CertifiedEvidenceTigerEngine)
        clone.artifact_dir = self.artifact_dir
        clone._item2id = self._item2id
        clone._id2item = self._id2item
        clone.num_items = self.num_items
        clone._summary = self._summary
        clone._cache = self._cache
        clone._case_index = int(case_index)
        return clone

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if parent_asin not in self._item2id:
            raise KeyError(parent_asin)
        return int(self._item2id[parent_asin])

    def recommend(self, history_item_ids: Sequence[int], k: int = 10) -> EngineResult:
        record = self._cache.get(self._case_index)
        if record is None:
            raise KeyError(f"no frozen evidence for case {self._case_index}")
        if record.get("budget_exhausted") or record.get("status") != CERTIFIED:
            raise RuntimeError("frozen evidence is not certified")
        recs: list[EngineRecommendation] = []
        for candidate in record["candidates"][: int(k)]:
            item_id = int(candidate["item_id"])
            parent_asin = self._id2item[item_id]
            if parent_asin is None:
                raise KeyError(f"item {item_id} unmapped")
            recs.append(EngineRecommendation(
                parent_asin=str(parent_asin), rank=int(candidate["rank"]),
                score=float(candidate["score"]),
                semantic_id=tuple(int(v) for v in candidate["semantic_id"]),
                item_id=item_id))
        return EngineResult(retrieval_mode=CERTIFIED, recommendations=tuple(recs),
                            checkpoint_sha256=self.checkpoint_sha256,
                            sid_artifact_sha256=self.sid_artifact_sha256)


# --------------------------------------------------------------------------- #
# Injected engine for the failure matrix
# --------------------------------------------------------------------------- #
class FakeTigerEngine:
    """Deterministic engine with one injectable failure mode.

    ``mode`` is what it *reports*; the tool decides whether that is acceptable.  A failure mode
    either raises (source unavailable) or returns a deliberately malformed / mislabelled result,
    so the tool's refusal is exercised rather than its happy path.
    """

    def __init__(self, *, recommendations: Sequence[EngineRecommendation] = (),
                 mode: str = CERTIFIED, fail: str | None = None,
                 checkpoint_sha256: str = "a" * 64,
                 sid_artifact_sha256: str = "b" * 64) -> None:
        self.recommendations = tuple(recommendations)
        self.mode = mode
        self.fail = fail
        self.checkpoint_sha256 = checkpoint_sha256
        self.sid_artifact_sha256 = sid_artifact_sha256
        self.calls = 0

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if self.fail == "unresolved_identity":
            raise KeyError(parent_asin)
        # stable across processes: a salted hash would make failure tests flaky
        return zlib.crc32(parent_asin.encode('utf-8')) % 1000 + 1

    def recommend(self, history_item_ids: Sequence[int], k: int = 10) -> EngineResult:
        self.calls += 1
        if self.fail == "backend_unavailable":
            raise ConnectionError("backend process is not reachable")
        if self.fail == "timeout":
            raise TimeoutError("retrieval exceeded its time budget")
        if self.fail == "checkpoint_unavailable":
            raise FileNotFoundError("tiger.pt is missing")
        if self.fail == "sid_artifact_unavailable":
            raise FileNotFoundError("semantic_ids.json is missing")
        if self.fail == "invalid_sid":
            bad = EngineRecommendation(parent_asin="A1", rank=1, score=-1.0,
                                       semantic_id=(-1, 0, 0, 0), item_id=1)
            return EngineResult(self.mode, (bad,), self.checkpoint_sha256, self.sid_artifact_sha256)
        if self.fail == "non_finite_score":
            bad = EngineRecommendation(parent_asin="A1", rank=1, score=float("nan"),
                                       semantic_id=(0, 0, 0, 0), item_id=1)
            return EngineResult(self.mode, (bad,), self.checkpoint_sha256, self.sid_artifact_sha256)
        if self.fail == "empty_result":
            return EngineResult(self.mode, (), self.checkpoint_sha256, self.sid_artifact_sha256)
        if self.fail == "budget_exhausted":
            # Budget exhaustion downgrades the run; the engine reports the downgraded mode and
            # the tool must refuse it when certification is required.
            return EngineResult(APPROXIMATE, self.recommendations, self.checkpoint_sha256,
                                self.sid_artifact_sha256)
        return EngineResult(self.mode, self.recommendations[: int(k)],
                            self.checkpoint_sha256, self.sid_artifact_sha256)
