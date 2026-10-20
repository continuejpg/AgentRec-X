"""Learned Ranker (LightGBM LambdaRank) for the experimental serving profile.

One shared feature builder feeds **both** the offline materialiser and this online adapter, so the
15-value protocol cannot drift into two formulas.  Rules taken from the frozen schema and proven
against the materialised JSONL (R-LR-03A'', max |oracle - stored| = 5e-7):

* per-source ``rank`` is the source's own rank (gaps preserved; never re-indexed after filtering);
* per-source ``score`` and ``rrf_score`` are quantised to 6 decimals;
* ``rrf_rank`` is the rank by **full-precision** RRF descending, then ``item_id`` ascending;
* missing source -> ``mask=0, rank=0, inv_rank=0.0, score=0.0``;
* ordering: ``ranker_score`` desc, then ``rrf_score`` desc, then ``item_id`` asc.

LightGBM is optional at runtime: if it is absent, the model cannot be loaded, the schema disagrees or
prediction fails, the caller falls back to full-precision RRF and the reason is counted.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FEATURES: tuple[str, ...] = (
    "seq_mask", "seq_rank", "seq_inv_rank", "seq_score",
    "tt_mask", "tt_rank", "tt_inv_rank", "tt_score",
    "lex_mask", "lex_rank", "lex_inv_rank", "lex_score",
    "rrf_score", "rrf_rank", "source_count",
)
SOURCES = ("seq", "tt", "lex")
RRF_K = 60
SCORE_DECIMALS = 6
#: Deployable locations of the frozen artifacts (overridable by environment for other deployments).
MODEL_PATH = Path(os.environ.get("AGENTRECX_RANKER_MODEL", "runs/lr_v1_formal/model.txt"))
SCHEMA_PATH = Path(os.environ.get("AGENTRECX_RANKER_SCHEMA", "runs/lr_v1_formal/feature_schema.json"))


def rrf_full(rank_by_source: Mapping[str, int]) -> float:
    """Full-precision RRF: sum 1/(60+rank) over the sources that returned the item."""
    return sum(1.0 / (RRF_K + rank) for rank in rank_by_source.values() if rank > 0)


def rrf_ranks(candidates: Sequence[str], rrf_by_asin: Mapping[str, float]) -> dict[str, int]:
    """Rank by full-precision RRF descending, then ASIN ascending (the materialiser's rule)."""
    ordered = sorted(candidates, key=lambda a: (-rrf_by_asin.get(a, 0.0), a))
    return {asin: index + 1 for index, asin in enumerate(ordered)}


def build_feature_rows(
    item_ids: Sequence[int],
    parent_asins: Sequence[str],
    rank_by_source: Mapping[str, Mapping[str, int]],
    score_by_source: Mapping[str, Mapping[str, float]],
    rrf_by_asin: Mapping[str, float],
    rrf_universe: Sequence[str] | None = None,
) -> list[list[float]]:
    """The single 15-value builder shared by offline materialisation and serving.

    ``rrf_universe`` is the pool ``rrf_rank`` is ranked within.  The materialiser ranked on the
    RRF output **before** history filtering, so serving must pass that same universe (defaults to
    the candidate list for callers that genuinely rank afterwards).
    """
    universe = [a for a in (rrf_universe if rrf_universe is not None else parent_asins)]
    ranks = rrf_ranks(universe, rrf_by_asin)
    rows: list[list[float]] = []
    for item_id, asin in zip(item_ids, parent_asins):
        row: list[float] = []
        for source in SOURCES:
            rank = int(rank_by_source.get(source, {}).get(asin, 0) or 0)
            score = float(score_by_source.get(source, {}).get(asin, 0.0) or 0.0)
            row.extend([1.0 if rank > 0 else 0.0, float(rank),
                        round(1.0 / rank, 6) if rank else 0.0, round(score, SCORE_DECIMALS)])
        row.extend([round(float(rrf_by_asin.get(asin, 0.0)), SCORE_DECIMALS),
                    float(ranks.get(asin, 0)), float(sum(1 for s in SOURCES
                                                         if rank_by_source.get(s, {}).get(asin, 0)))])
        rows.append(row)
    return rows


def order_by_ranker(scores: Sequence[float], rrf: Sequence[float], item_ids: Sequence[int]) -> list[int]:
    """Deterministic ordering: ranker desc, rrf desc, item_id asc. Returns indices."""
    return sorted(range(len(item_ids)),
                  key=lambda i: (-float(scores[i]), -float(rrf[i]), int(item_ids[i])))


@dataclass
class LearnedRanker:
    """Loads the frozen LightGBM model once and ranks a legal candidate pool."""

    model_path: Path
    schema_path: Path | None = None
    expected_sha256: str | None = None
    _booster: Any = None
    _best_iteration: int | None = None
    fallback_count: int = 0
    fallback_reasons: list[str] = field(default_factory=list)
    load_error: str | None = None
    last_used_model: bool = False

    def __post_init__(self) -> None:
        self.load()

    # -- lifecycle ---------------------------------------------------------- #
    def load(self) -> bool:
        path = Path(self.model_path)
        try:
            if not path.exists():
                raise FileNotFoundError(f"model not found: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.expected_sha256 and digest != self.expected_sha256:
                raise ValueError("model SHA256 mismatch")
            schema = json.loads(Path(self.schema_path).read_text()) if self.schema_path else None
            if schema is not None:
                if tuple(schema.get("features", ())) != FEATURES:
                    raise ValueError("feature schema order mismatch")
            import lightgbm as lgb          # optional dependency: absence is a fallback reason
            booster = lgb.Booster(model_file=str(path))
            if booster.num_feature() != len(FEATURES):
                raise ValueError(f"model expects {booster.num_feature()} features, schema has {len(FEATURES)}")
            self._booster = booster
            meta = path.parent / "metrics.json"
            self._best_iteration = (json.loads(meta.read_text()).get("best_iteration")
                                    if meta.exists() else None)
            self.load_error = None
            return True
        except Exception as exc:  # noqa: BLE001 - every failure must become a counted fallback
            self._booster = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    # -- inference ---------------------------------------------------------- #
    def rank(self, parent_asins: Sequence[str], item_ids: Sequence[int],
             rank_by_source: Mapping[str, Mapping[str, int]],
             score_by_source: Mapping[str, Mapping[str, float]],
             rrf_by_asin: Mapping[str, float],
             rrf_universe: Sequence[str] | None = None) -> tuple[list[int], str | None]:
        """Return ranked indices; on any problem return None so the caller uses full-precision RRF."""
        if self._booster is None:
            self.fallback_count += 1
            self.fallback_reasons.append(self.load_error or "model not loaded")
            self.last_used_model = False
            return [], self.fallback_reasons[-1]
        try:
            rows = build_feature_rows(item_ids, parent_asins, rank_by_source, score_by_source, rrf_by_asin, universe)
            if not rows:
                raise ValueError("no candidates")
            import numpy as np
            preds = self._booster.predict(np.asarray(rows, dtype=np.float32),
                                          num_iteration=self._best_iteration or 0)
            if not all(abs(float(p)) < float("inf") for p in preds):
                raise ValueError("non-finite prediction")
            if len(preds) != len(item_ids):
                raise ValueError("prediction length mismatch")
            rrf_col = [round(float(rrf_by_asin.get(a, 0.0)), SCORE_DECIMALS) for a in parent_asins]
            self.last_used_model = True
            return order_by_ranker(list(preds), rrf_col, item_ids), None
        except Exception as exc:  # noqa: BLE001 - counted, then fallback
            self.fallback_count += 1
            reason = f"{type(exc).__name__}: {exc}"
            self.fallback_reasons.append(reason)
            self.last_used_model = False
            return [], reason

    def order(self, parent_asins: Sequence[str], item_ids: Sequence[int],
              rank_by_source: Mapping[str, Mapping[str, int]],
              score_by_source: Mapping[str, Mapping[str, float]],
              rrf_by_asin: Mapping[str, float],
              rrf_universe: Sequence[str] | None = None) -> tuple[list[int], str | None]:
        idx, reason = self.rank(parent_asins, item_ids, rank_by_source, score_by_source, rrf_by_asin, rrf_universe)
        if idx:
            return idx, None
        # full-precision RRF fallback on the same legal pool (never re-calls the recall models)
        fallback = sorted(range(len(item_ids)),
                          key=lambda i: (-float(rrf_by_asin.get(parent_asins[i], 0.0)), int(item_ids[i])))
        return fallback, reason




class LearnedRankerEngine:
    """Wraps the accepted fusion engine: same recall sources, learned final ordering.

    ``recommend`` runs each source **once**, builds the 15-value features with the shared builder,
    lets LightGBM order the full legal pool and then truncates to ``k``.  On any ranker problem it
    returns the wrapper's own full-precision RRF ordering (same pool, no extra recall) and records
    the reason.  Identity/legal checks stay the fusion engine's (its ``item_id_for`` and
    ``is_known_identity`` are the trusted ones), so a fallback can never bypass them.
    """

    def __init__(self, fusion_engine: Any, ranker: LearnedRanker, head_depth: int = 100) -> None:
        self.fusion = fusion_engine
        self.ranker = ranker
        self.head_depth = head_depth
        self.last_fallback_reason: str | None = None
        self.last_used_model: bool = False

    # ---- engine-contract passthroughs (the app's route checks these) -------- #
    def is_ready(self) -> bool:
        return bool(getattr(self.fusion, "is_ready", lambda: True)())

    @property
    def score_kind(self) -> str:
        return "lightgbm_lambdarank"

    def model_metadata(self) -> dict:
        meta = dict(getattr(self.fusion, "model_metadata", lambda: {})() or {})
        meta.update({"ranker": "lightgbm_lambdarank", "ranker_loaded": self.ranker._booster is not None,
                     "fallback_count": self.ranker.fallback_count,
                     "load_error": self.ranker.load_error})
        return meta

    # ---- helpers ---------------------------------------------------------- #
    def _recall(self, history: Sequence[str]):
        from recommendation.inference.sasrec import Recommendation
        ranks: dict[str, dict[str, int]] = {s: {} for s in SOURCES}
        scores: dict[str, dict[str, float]] = {s: {} for s in SOURCES}
        item_of: dict[str, int] = {}
        seq = self.fusion.sequential.recommend(list(history), k=self.head_depth)
        for rec in seq.recommendations:
            ranks["seq"][rec.parent_asin] = int(rec.rank); scores["seq"][rec.parent_asin] = float(rec.score)
            item_of[rec.parent_asin] = int(rec.item_id)
        # Two-Tower input must match r_lr01_materialize.py exactly: the FULL integer history in
        # original order (never ids inferred from SASRec results, never ASINs cast to int).
        ids = []
        for asin in history:
            try:
                ids.append(int(self.fusion.item_id_for(asin)))
            except Exception:
                continue
        if ids:
            tt = self.fusion.two_tower.recommend(ids, k=self.head_depth)
            for rec in tt.recommendations:
                ranks["tt"][rec.parent_asin] = int(rec.rank); scores["tt"][rec.parent_asin] = float(rec.score)
                item_of.setdefault(rec.parent_asin, int(rec.item_id))
        usable = [self.fusion.catalogue[a] for a in history[-5:]
                  if a in self.fusion.catalogue and getattr(self.fusion.catalogue[a], "title", None)]
        if usable:
            from recommendation.inference.lexical import query_text_for
            for position, (asin, score) in enumerate(
                    self.fusion.lexical.scored(query_text_for(usable), limit=self.head_depth), start=1):
                ranks["lex"][asin] = position; scores["lex"][asin] = float(score)
                item_of.setdefault(asin, 0)
        return ranks, scores, item_of

    # ---- the serving entry point ------------------------------------------ #
    def recommend(self, history_parent_asins: Sequence[str], k: int = 10) -> Any:
        from recommendation.inference.sasrec import Recommendation, RecommendationResult
        history = [str(x) for x in history_parent_asins]
        ranks, scores, item_of = self._recall(history)
        masked = set(history)
        # The universe the materialiser ranked within: every item any source returned, history
        # included (the materialiser filtered history AFTER ranking).
        # The materialiser ranked inside the full RRF dict produced from the three source lists
        # (history and unresolvable ASINs still occupy their rank slots).  Build exactly that dict.
        fused_keys: list[str] = []
        for source in SOURCES:
            for asin in ranks[source]:
                if asin not in fused_keys:
                    fused_keys.append(asin)
        rrf_all = {a: rrf_full({s: ranks[s].get(a, 0) for s in SOURCES}) for a in fused_keys}
        universe = list(rrf_all)
        candidates = [a for a in universe if a not in masked]
        # identity: only what the trusted mapping recognises survives (fail-closed on unknown)
        resolved: list[str] = []
        for asin in candidates:
            try:
                item_of[asin] = int(self.fusion.item_id_for(asin))
            except Exception:
                continue
            resolved.append(asin)
        # rrf_rank is computed over the WHOLE fused universe, so the ranker must receive the
        # full score map: restricting it to the legal pool would push history/unresolvable
        # members to 0.0 (they sort last) and shift every pool rank upwards.
        rrf_by_asin = dict(rrf_all)
        item_ids = [item_of[a] for a in resolved]
        order, reason = self.ranker.order(resolved, item_ids, ranks, scores, rrf_by_asin, universe)
        self.last_fallback_reason = reason
        self.last_used_model = bool(reason is None)
        # scores: the model's own value when it ranked, otherwise the full-precision RRF score
        model_scores = None
        if reason is None:
            try:
                import numpy as np
                rows = build_feature_rows(item_ids, resolved, ranks, scores, rrf_by_asin, universe)
                model_scores = list(self.ranker._booster.predict(          # noqa: SLF001
                    np.asarray(rows, dtype=np.float32), num_iteration=self.ranker._best_iteration or 0))
            except Exception:  # pragma: no cover - order() already succeeded; keep RRF scores
                model_scores = None
        top = order[:k]
        recs = tuple(
            Recommendation(rank=index + 1, item_id=int(item_ids[i]), parent_asin=resolved[i],
                           score=float(model_scores[i]) if model_scores is not None
                           else float(rrf_by_asin.get(resolved[i], 0.0)))
            for index, i in enumerate(top))
        return RecommendationResult(
            recommendations=recs, requested_k=int(k), history_length=len(history),
            effective_history_length=min(len(history), 50), history_truncated=len(history) > 50,
            eligible_candidates=len(resolved),
            timings_ms={"lexical": 0.0},
            score_kind=("lightgbm_lambdarank" if reason is None else "rrf_rank_fusion"),
            provenance={"ranker": {"used": reason is None,
                                   "fallback_count": int(self.ranker.fallback_count),
                                   "fallback_reason": reason,
                                   "model": "lightgbm_lambdarank"}})
