"""Serving V1: fixed multi-source retrieval with deterministic reciprocal-rank fusion.

What this is
------------
A thin :class:`~recommendation.tools.recommendation.RecommendationEngine` that consults a **fixed**
set of retrievers for every request and combines their *orderings* with the repository's accepted
:func:`~recommendation.control.candidate_ledger.reciprocal_rank_fusion`.

```text
history -> sequential (SASRec)  ┐
        -> two_tower            ├─ RRF(K=60) -> fused candidates -> existing enricher
        -> catalogue lexical    ┘                                 -> existing matcher
                                                                  -> existing reranker
```

It is deliberately *not* adaptive: the same sources run for every user, in the same order, with the
same constant.  Nothing here selects a source, tunes a weight or scores a candidate with a model
score of its own.

Why rank fusion and not score fusion
------------------------------------
The three sources produce incomparable numbers: a SASRec logit, a Two-Tower cosine and a BM25 value.
Summing them would fabricate a quantity.  RRF consumes only each source's *ordering*, which is the
one thing every retriever can be trusted to mean.

Scores, honestly labelled
-------------------------
A fused score is **not** a model score.  ``FusionInferenceEngine.score_kind`` is
``"rrf_rank_fusion"``, the per-item provenance records each source's own rank and score kind, and
:class:`~recommendation.tools.schemas.ToolRecommendation` carries that label to the response so a
fused value can never be read as a SASRec or Two-Tower score.

Invariants preserved from the accepted serving path
---------------------------------------------------
* ``PAD`` (``item_id`` 0) is never a candidate — the source engines already exclude it, and every
  identity here must resolve inside the trusted catalogue;
* already-consumed history is excluded from every source's ranking, so fusion cannot re-introduce
  it;
* identity is the canonical ``parent_asin``; an identity the trusted side cannot confirm is dropped
  (fail closed) rather than presented;
* output is deterministic: the source engines are deterministic and RRF breaks ties on identity.

The fused candidates flow into the **existing** evidence, matcher and reranker stages untouched:
this engine only decides *which* candidates exist and in what fused order, exactly as the accepted
SASRec engine did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from recommendation.control.candidate_ledger import RRF_K, reciprocal_rank_fusion
from recommendation.inference.lexical import LEXICAL_SCORE_KIND, CatalogueLexicalIndex, query_text_for
from recommendation.inference.sasrec import Recommendation, RecommendationResult
from recommendation.tools.schemas import MAX_K

__all__ = [
    "FUSION_PROFILES",
    "FUSION_SCORE_KIND",
    "FUSION_SOURCES",
    "FUSION_SOURCES_V2_TIGER",
    "TIGER_SERVING_HEAD",
    "TIGER_SOURCE",
    "TIGER_SOURCE_SCORE_KIND",
    "DEFAULT_SOURCE_HEAD",
    "FusionError",
    "FusionInferenceConfig",
    "FusionInferenceEngine",
]

#: What a fused score is.  Never a model score; the label exists so it cannot be read as one.
FUSION_SCORE_KIND = "rrf_rank_fusion"

#: The Serving V1 source set, in the frozen order they are consulted.
#:
#: Three sources, chosen for engineering simplicity.  This is the same *combination* the audited
#: D-style arm used (sequential + metadata + two-tower, without popularity).  That is a statement
#: about which arm the configuration resembles, **not** a claim that three sources beat four: the
#: audited B/D contrasts are two different experiments and are not interchangeable.
FUSION_SOURCES: tuple[str, ...] = ("sequential", "two_tower", "catalog_lexical")

#: The TIGER source, as the fusion layer names it.
TIGER_SOURCE = "tiger"

#: The score kind TIGER candidates carry.  It must equal
#: ``recommendation.control.tiger_source.TIGER_SCORE_KIND``; a test asserts the equality so the
#: two declarations cannot drift apart.
TIGER_SOURCE_SCORE_KIND = "tiger_sid_sequence_logprob"

#: Serving V2: the accepted three sources plus TIGER as a peer generative retriever.
#:
#: A **separate profile**, never a replacement: ``FUSION_SOURCES`` keeps serving deployments
#: exactly as they were, and a V2 deployment opts in by naming this profile.
FUSION_SOURCES_V2_TIGER: tuple[str, ...] = FUSION_SOURCES + (TIGER_SOURCE,)

#: Named profiles.  A profile fixes the source *set*; the depths are recorded separately because
#: TIGER cannot serve the 100-deep head the other three are read at.
FUSION_PROFILES: dict[str, tuple[str, ...]] = {
    "serving_v1": FUSION_SOURCES,
    # Gate C-1/C-2: single-backend serving profiles, added for service acceptance.  They select one
    # source each and change nothing else: same scoring, same seen mask, same top-k, same
    # tie-breaking inside that source.  ``serving_v1`` remains the default and is untouched.
    "serving_v1_sequential_only": ("sequential",),
    "serving_v1_two_tower_only": ("two_tower",),
    # R-LR-03: the same three sources, but the final ordering comes from the trained LightGBM
    # LambdaRank (fallback: full-precision RRF on the same pool).  Defaults stay untouched.
    "serving_v1_learned_ranker": FUSION_SOURCES,
    "serving_v2_tiger": FUSION_SOURCES_V2_TIGER,
    # Same four sources, but the three engines that can serve depth 100 keep it while TIGER stays
    # at its own ceiling.  A *different* configuration, recorded as such rather than presented as
    # equivalent to the uniform one.
    "serving_v2_tiger_mixed_head": FUSION_SOURCES_V2_TIGER,
}

#: TIGER's own serving ceiling: its engine refuses ``k > 20`` (``TigerWorkerConfig.k_max``).
TIGER_SERVING_HEAD = 20

#: Depths for the V2 profiles.  ``uniform`` reads every source at TIGER's ceiling so the fusion
#: compares like with like; ``mixed`` keeps the V1 depth of 100 for the three engines that can
#: serve it and 20 for TIGER, which is a *different* and explicitly labelled configuration.
PROFILE_DEPTH_POLICIES: dict[str, dict[str, int] | str] = {
    "serving_v1": "uniform",
    "serving_v2_tiger": "uniform_tiger_head",
    "serving_v2_tiger_mixed_head": "v1_head_with_tiger_ceiling",
}

#: How deep each source's ranking is read before fusion.
#:
#: This is the **serving** head, and it is bounded by the accepted serving contract: the accepted
#: engines refuse ``k > MAX_K`` because that is the limit the Tool and the HTTP schema enforce.  The
#: offline fusion experiments read depth-1000 head lists straight out of a score matrix, which the
#: serving engines deliberately do not offer.
#:
#: The difference is material and is reported rather than papered over: B1's paired evidence was
#: established for a depth-1000 fusion, while Serving V1 fuses depth-100 rankings.  Deeper serving
#: heads would mean relaxing an accepted serving guard, which is not this milestone's decision.
DEFAULT_SOURCE_HEAD = MAX_K

#: Score kind reported per source, for provenance.  These describe the *source's* own numbers.
SOURCE_SCORE_KINDS: dict[str, str] = {
    "sequential": "sasrec_logit",
    "two_tower": "two_tower_cosine",
    "catalog_lexical": LEXICAL_SCORE_KIND,
    TIGER_SOURCE: TIGER_SOURCE_SCORE_KIND,
}


class FusionError(RuntimeError):
    """The fusion engine cannot produce a trustworthy result."""


class UnknownCatalogueIdentity(FusionError):
    """A caller-supplied history identity is outside the served catalogue.

    A **client** input error by construction: the raising check inspects only the request's own
    history list against the trusted identity map.  A dedicated subclass keeps genuine engine
    failures out of the 4xx path.
    """


@runtime_checkable
class SequentialEngineLike(Protocol):
    """The accepted sequential engine: history identities in, ranked candidates out."""

    def recommend(self, history_parent_asins: Sequence[str], k: int = ...) -> RecommendationResult: ...


@runtime_checkable
class TwoTowerEngineLike(Protocol):
    """The Two-Tower engine: integer-id history in, ranked candidates out."""

    def recommend(self, history_item_ids: Sequence[int], k: int = ...) -> RecommendationResult: ...

    def parent_asin_to_item_id(self, parent_asin: str) -> int: ...


@dataclass(frozen=True)
class FusionInferenceConfig:
    """Fixed fusion configuration.  Nothing here is tunable at request time."""

    head_depth: int = DEFAULT_SOURCE_HEAD
    #: The repository's accepted constant; not a knob.
    rrf_k: int = RRF_K
    sources: tuple[str, ...] = FUSION_SOURCES
    #: The named profile this configuration realises, recorded so a response says which serving
    #: contract produced it rather than leaving a reader to infer it from the source list.
    profile: str = "serving_v1"
    #: Per-source read depth.  ``None`` means "every source at :attr:`head_depth`", which is what
    #: Serving V1 does; a V2 profile declares the depths explicitly because TIGER cannot serve 100.
    source_head_depths: Mapping[str, int] | None = None

    def depth_for(self, source: str) -> int:
        """How deep ``source`` is read.  Absent an explicit entry, :attr:`head_depth`."""
        if self.source_head_depths is None:
            return int(self.head_depth)
        return int(self.source_head_depths.get(source, self.head_depth))

    @property
    def head_depths(self) -> dict[str, int]:
        """The resolved depth of every configured source, for reporting."""
        return {name: self.depth_for(name) for name in self.sources}

    @classmethod
    def for_profile(
        cls,
        profile: str,
        *,
        head_depth: int = DEFAULT_SOURCE_HEAD,
        rrf_k: int = RRF_K,
    ) -> "FusionInferenceConfig":
        """Build the configuration of a named profile.

        ``serving_v1`` is unchanged (uniform :data:`DEFAULT_SOURCE_HEAD`).  ``serving_v2_tiger``
        reads every source at TIGER's ceiling so the four-way fusion is comparable; the
        ``_mixed_head`` variant keeps 100 for the three engines that support it, and both record
        their depths so the difference is never silent.
        """
        if profile not in FUSION_PROFILES:
            raise FusionError(
                f"unknown fusion profile {profile!r}; known: {sorted(FUSION_PROFILES)}"
            )
        sources = FUSION_PROFILES[profile]
        policy = PROFILE_DEPTH_POLICIES.get(profile, "uniform")
        if policy == "uniform":
            return cls(head_depth=head_depth, rrf_k=rrf_k, sources=sources, profile=profile)
        if policy == "uniform_tiger_head":
            uniform = min(int(head_depth), TIGER_SERVING_HEAD)
            return cls(
                head_depth=uniform, rrf_k=rrf_k, sources=sources, profile=profile,
                source_head_depths={name: uniform for name in sources},
            )
        depths = {name: int(head_depth) for name in sources if name != TIGER_SOURCE}
        depths[TIGER_SOURCE] = TIGER_SERVING_HEAD
        return cls(
            head_depth=int(head_depth), rrf_k=rrf_k, sources=sources, profile=profile,
            source_head_depths=depths,
        )


@dataclass
class FusionInferenceEngine:
    """Fixed multi-source serving engine (Serving V1)."""

    sequential: SequentialEngineLike
    two_tower: TwoTowerEngineLike
    lexical: CatalogueLexicalIndex
    catalogue: Mapping[str, Any]
    #: Resolves a canonical identity to the model item id; required so a fused candidate can be
    #: reported in the accepted shape and so an unknown identity fails closed.
    item_id_for: Callable[[str], int]
    config: FusionInferenceConfig = field(default_factory=FusionInferenceConfig)
    #: Optional confirmation that an identity belongs to the trusted catalogue.  When supplied, an
    #: identity it rejects is dropped instead of presented.
    is_known_identity: Callable[[str], bool] | None = None
    #: The TIGER serving engine (``recommendation.backends.tiger_engine.TigerServingEngine``).
    #: Required **iff** ``"tiger"`` is in ``config.sources``: a profile that names TIGER must not
    #: quietly fuse three sources and report four.
    tiger: Any = None

    #: Declared so consumers label the fused score correctly.
    score_kind = FUSION_SCORE_KIND

    # -- trusted identity surface ------------------------------------------ #
    #
    # The Agent layer grounds candidates with ``GroundingVerifier``, which requires the identity
    # map's three methods.  The fusion engine already holds that map - ``item_id_for`` and
    # ``is_known_identity`` are the sequential engine's own - so it exposes the same surface by
    # delegation instead of letting a second map exist.  Read-only: no fusion behaviour changes.

    def _identity_source(self) -> Any:
        for name in ("has_parent_asin", "parent_asin_to_item_id", "item_id_to_parent_asin"):
            if not callable(getattr(self.sequential, name, None)):
                raise FusionError(
                    f"the sequential engine does not provide {name}(); the Agent layer needs a "
                    "trusted identity map to ground candidates"
                )
        return self.sequential

    def has_parent_asin(self, parent_asin: str) -> bool:
        """Whether the trusted catalogue knows this identity."""
        return bool(self._identity_source().has_parent_asin(parent_asin))

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Trusted identity -> integer id (the same direction the sources use)."""
        return int(self._identity_source().parent_asin_to_item_id(parent_asin))

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Integer id -> trusted identity, for reporting a fused candidate."""
        return str(self._identity_source().item_id_to_parent_asin(int(item_id)))

    @property
    def device(self) -> Any:
        """The device the underlying engines serve on.

        The serving layer (``/health``, the demo runtime) reads ``engine.device`` from whatever
        engine is injected; a fusion engine that omitted it broke those paths at runtime — which is
        exactly what the browser smoke caught and the in-process tests, using injected stubs, could
        not.
        """
        return getattr(self.sequential, "device", "cpu")

    def __post_init__(self) -> None:
        if self.config.head_depth < 1:
            raise FusionError("head_depth must be >= 1")
        if self.config.rrf_k < 1:
            raise FusionError("rrf_k must be >= 1")
        unknown = [name for name in self.config.sources if name not in SOURCE_SCORE_KINDS]
        if unknown:
            raise FusionError(f"unknown fusion source(s): {unknown}")
        if TIGER_SOURCE in self.config.sources:
            if self.tiger is None:
                raise FusionError(
                    "the configured profile names the tiger source but no TIGER engine was "
                    "injected; refusing to fuse fewer sources than the profile declares"
                )
            depth = self.config.depth_for(TIGER_SOURCE)
            if depth > TIGER_SERVING_HEAD:
                raise FusionError(
                    f"tiger head depth {depth} exceeds its serving ceiling "
                    f"{TIGER_SERVING_HEAD}; the accepted engine refuses k > {TIGER_SERVING_HEAD}"
                )

    # -- readiness / metadata ---------------------------------------------- #

    def is_ready(self) -> bool:
        """True when every required collaborator reports itself ready."""
        for collaborator in (self.sequential, self.two_tower):
            ready = getattr(collaborator, "is_ready", None)
            if callable(ready) and not ready():
                return False
        if TIGER_SOURCE in self.config.sources:
            # TIGER is ready only once its long-lived worker has completed the handshake.  A
            # configured-but-cold worker must never make the deployment report ready.
            if not bool(getattr(self.tiger, "started", False)):
                return False
        return self.lexical.indexed_documents > 0

    def model_metadata(self) -> dict[str, Any]:
        """Describe what is actually serving, for the health/model surface.

        The sequential engine's own metadata is included, because the served model *is* that
        checkpoint plus the other two sources: ``/v1/model`` requires the underlying model's
        identity fields, so the fusion description is added to them rather than replacing them.
        """
        payload: dict[str, Any] = {}
        underlying = getattr(self.sequential, "model_metadata", None)
        if callable(underlying):
            payload.update(underlying())
        payload.update({
            "score_kind": FUSION_SCORE_KIND,
            "serving_fusion": "rrf",
            "fusion_sources": list(self.config.sources),
            "source_score_kinds": {name: SOURCE_SCORE_KINDS[name] for name in self.config.sources},
            "rrf_k": self.config.rrf_k,
            "source_head_depth": self.config.head_depth,
            "serving_fusion_profile": self.config.profile,
            "source_head_depths": self.config.head_depths,
            "depth_policy": PROFILE_DEPTH_POLICIES.get(self.config.profile, "uniform"),
            "tiger_health": self.tiger_health(),
            "lexical_documents": self.lexical.indexed_documents,
            "catalogue_size": self.lexical.catalogue_size,
            "active_engine": "fusion_rrf",
        })
        return payload

    # -- one request -------------------------------------------------------- #

    def _sequential_ranking(self, history: Sequence[str]) -> list[str]:
        result = self.sequential.recommend(list(history), self.config.head_depth)
        return [item.parent_asin for item in result.recommendations]

    def _two_tower_ranking(self, history: Sequence[str]) -> list[str]:
        item_ids: list[int] = []
        for parent_asin in history:
            item_ids.append(int(self.two_tower.parent_asin_to_item_id(parent_asin)))
        if not item_ids:
            return []
        result = self.two_tower.recommend(item_ids, self.config.head_depth)
        return [item.parent_asin for item in result.recommendations]

    def tiger_health(self) -> dict[str, Any] | None:
        """TIGER's lifecycle state, or ``None`` when the profile does not include TIGER."""
        if TIGER_SOURCE not in self.config.sources:
            return None
        health = getattr(self.tiger, "health", None)
        if callable(health):
            return dict(health())
        return {"state": "ready" if getattr(self.tiger, "started", False) else "not_started"}

    def _tiger_ranking(
        self, history: Sequence[str], *, meta: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Read TIGER's head and convert it into the unified candidate shape.

        TIGER returns ``TigerRecommendation`` objects carrying ``item_id``, ``parent_asin``,
        ``rank``, ``score`` (a Semantic-ID sequence log-probability) and ``semantic_id``.  The
        fused layer only ever sees identities and ranks; the model's own numbers are recorded per
        identity instead of being mixed into the fused score, which stays rank-based.
        """
        encoder = getattr(self.tiger, "parent_asin_to_item_id", None)
        if not callable(encoder):
            encoder = self.two_tower.parent_asin_to_item_id
        item_ids = [int(encoder(value)) for value in history]
        if not item_ids:
            return []
        result = self.tiger.recommend(item_ids, k=self.config.depth_for(TIGER_SOURCE))
        mode = str(getattr(result, "retrieval_mode", "") or "")
        identities: list[str] = []
        for item in getattr(result, "recommendations", ()) or ():
            identity = str(getattr(item, "parent_asin", "") or "")
            if not identity:
                continue
            meta.setdefault(identity, {}).update({
                "rank": int(getattr(item, "rank", 0) or 0),
                "score": float(getattr(item, "score", 0.0) or 0.0),
                "score_kind": TIGER_SOURCE_SCORE_KIND,
                "item_id": int(getattr(item, "item_id", 0) or 0),
                "semantic_id": list(getattr(item, "semantic_id", ()) or ()),
                "retrieval_mode": mode,
            })
            identities.append(identity)
        return identities

    def _lexical_ranking(self, history: Sequence[str]) -> list[str]:
        # Only the most recent items contribute, and only their titles: the query is bounded by
        # construction because an unbounded one costs seconds per request (see query_text_for).
        records = [self.catalogue.get(parent_asin) for parent_asin in history]
        query = query_text_for([record for record in records if record is not None])
        return self.lexical.ranking(query, limit=self.config.head_depth)

    def _ranking_for(
        self, source: str, history: Sequence[str], *, tiger_meta: dict[str, dict[str, Any]]
    ) -> list[str]:
        if source == "sequential":
            return self._sequential_ranking(history)
        if source == "two_tower":
            return self._two_tower_ranking(history)
        if source == "catalog_lexical":
            return self._lexical_ranking(history)
        if source == TIGER_SOURCE:
            return self._tiger_ranking(history, meta=tiger_meta)
        raise FusionError(f"no ranking builder for source {source!r}")  # pragma: no cover

    def recommend(
        self,
        history_parent_asins: Sequence[str],
        k: int = 10,
    ) -> RecommendationResult:
        """Return the fused top-``k`` for ``history_parent_asins``.

        Raises
        ------
        FusionError
            k is not a positive int, the history is empty, or the history contains an identity
            outside the served catalogue.  History is never partially consumed: a caller either
            supplies a usable history or gets an explicit failure.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise FusionError(f"k must be a positive int, got {k!r}")
        history = [str(value) for value in history_parent_asins]
        if not history:
            raise FusionError("history must not be empty")
        if self.is_known_identity is not None:
            unknown = [value for value in history if not self.is_known_identity(value)]
            if unknown:
                raise UnknownCatalogueIdentity(
                    f"history contains {len(unknown)} identity(ies) outside the served catalogue, "
                    f"first: {unknown[0]!r}"
                )

        started = time.perf_counter()
        timings_ms: dict[str, float] = {}
        seen = set(history)

        rankings: list[list[str]] = []
        source_rankings: dict[str, list[str]] = {}
        # Per-request, never instance state: two concurrent requests must not be able to see each
        # other's TIGER metadata.
        tiger_meta: dict[str, dict[str, Any]] = {}
        for source in self.config.sources:
            began = time.perf_counter()
            ranked = self._ranking_for(source, history, tiger_meta=tiger_meta)
            timings_ms[f"source_{source}_ms"] = round((time.perf_counter() - began) * 1000.0, 3)
            cleaned: list[str] = []
            for identity in ranked:
                if not isinstance(identity, str) or not identity:
                    continue
                if identity in seen:
                    # Already-consumed history is never re-presented, whatever a source returns.
                    continue
                if self.is_known_identity is not None and not self.is_known_identity(identity):
                    # Fail closed: an identity the trusted side cannot confirm is dropped.
                    continue
                if identity not in cleaned:
                    cleaned.append(identity)
            source_rankings[source] = cleaned
            if cleaned:
                rankings.append(cleaned)

        began = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k) if rankings else []
        timings_ms["fusion_ms"] = round((time.perf_counter() - began) * 1000.0, 3)

        source_rank_by_identity: dict[str, dict[str, int]] = {}
        for source, ranked in source_rankings.items():
            for position, identity in enumerate(ranked, start=1):
                source_rank_by_identity.setdefault(identity, {})[source] = position

        recommendations: list[Recommendation] = []
        provenance: dict[str, dict[str, Any]] = {}
        for position, (identity, fused_score) in enumerate(fused, start=1):
            if len(recommendations) >= k:
                break
            ranks = source_rank_by_identity.get(identity, {})
            item_id = int(self.item_id_for(identity))
            recommendations.append(
                Recommendation(
                    rank=len(recommendations) + 1,
                    item_id=item_id,
                    parent_asin=identity,
                    score=float(fused_score),
                )
            )
            provenance[identity] = {
                "item_id": item_id,
                "fused_rank": len(recommendations),
                "fused_score": float(fused_score),
                "score_kind": FUSION_SCORE_KIND,
                "sources": [name for name in self.config.sources if name in ranks],
                "source_ranks": {name: ranks[name] for name in self.config.sources if name in ranks},
                "source_score_kinds": {
                    name: SOURCE_SCORE_KINDS[name] for name in self.config.sources if name in ranks
                },
            }
            if identity in tiger_meta:
                # TIGER's own view of this candidate: sequence log-probability, Semantic ID and
                # the mode that actually ran.  Kept beside the fused rank, never folded into it.
                provenance[identity]["tiger"] = dict(tiger_meta[identity])

        timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return RecommendationResult(
            recommendations=recommendations,
            requested_k=k,
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=len(fused),
            timings_ms=timings_ms,
            score_kind=FUSION_SCORE_KIND,
            provenance=provenance,
        )
