"""The TIGER-FP32 candidate source (Step 2.7 CandidateSource qualification).

What it is
----------
A :class:`~recommendation.control.candidate_plane.CandidateSourceTool` that consults the accepted
TIGER-FP32 generative retriever.  It follows the same seam as
:class:`~recommendation.control.two_tower_source.TwoTowerSourceTool`: it returns
``(parent_asin, source_rank, score)`` triples and the plane grounds every identity with the
trusted :class:`~recommendation.control.GroundingVerifier` before the ledger records anything.

What it deliberately does not do
--------------------------------
* **It never grounds an identity.**  Grounding belongs to the plane and the trusted verifier.
  A TIGER-generated Semantic ID is a *proposal*; canonical identity stays AgentRec-X's.
* **It never chooses its own history.**  The trusted history arrives as an argument, so the tool
  cannot answer for a user it was not given.
* **It never mixes its scores with another source's.**  Its scores are the frozen sequence
  log-probabilities of an item's Semantic-ID path, labelled :data:`TIGER_SCORE_KIND`.  The only
  cross-source combination in this repository is rank-based fusion, which uses ordering alone.
* **It never promotes a retrieval status.**  ``APPROXIMATE`` is never reported as ``CERTIFIED``
  at this layer; with ``require_certified=True`` an APPROXIMATE result is refused outright, and
  otherwise the true mode is carried into every proposal.

Failure isolation
-----------------
Every way this source can fail is turned into :class:`CandidateSourceUnavailable`, which the
plane classifies as a source-local failure.  It never returns a shortened list, a substituted
score, a fabricated identity or a partial result, because each of those would make the recorded
provenance a lie.  The one non-exceptional case is a *valid* retrieval that genuinely found
nothing: that returns an empty list and records an explicit ``EMPTY`` outcome, so "found nothing"
is never confused with "could not be consulted".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .arguments import CandidateSource
from .candidate_plane import CandidateSourceUnavailable

__all__ = [
    "TIGER_SCORE_KIND",
    "TIGER_SOURCE_NAME",
    "TIGER_SOURCE_VERSION",
    "TigerProposal",
    "TigerSourceTool",
]

#: What a TIGER score means: the sum of full-vocabulary log-probabilities over an item's complete
#: Semantic-ID token path (``score_rule.version == 1``).  Deliberately distinct from
#: ``sasrec_logit`` and ``two_tower_cosine`` so no consumer can compare scales by accident.
TIGER_SCORE_KIND = "tiger_sid_sequence_logprob"

#: The source's own name/version, carried into provenance so a later change is detectable.
TIGER_SOURCE_NAME = "tiger_fp32"
TIGER_SOURCE_VERSION = 1

#: The two retrieval statuses, restated here so this module never imports the backend.
CERTIFIED = "CERTIFIED"
APPROXIMATE = "APPROXIMATE"

_HEX12 = re.compile(r"^[0-9a-f]{12,64}$")


@dataclass(frozen=True)
class TigerProposal:
    """One TIGER sighting of one candidate, with the provenance the task requires.

    Retained per call in :attr:`TigerSourceTool.last_proposals`.  Everything here describes
    *this source's* view; nothing is a fused or global quantity.
    """

    parent_asin: str
    rank: int
    score: float
    semantic_id: tuple[int, ...]
    item_id: int
    retrieval_mode: str
    checkpoint_sha256: str
    sid_artifact_sha256: str
    resolver: Mapping[str, Any]
    source: str = TIGER_SOURCE_NAME
    source_version: int = TIGER_SOURCE_VERSION

    def provenance_token(self) -> str:
        """A compact, deterministic provenance string for the ledger's ``source_query`` field.

        The shared provenance schema records free text here and bounds it at 200 characters, so
        the retrieval mode and the artifact hashes ride along as a short token rather than
        forcing a TIGER-specific field onto the shared schema.
        """
        return "mode=%s;ckpt=%s;sid=%s;resolver=%s;src=%s/v%d" % (
            self.retrieval_mode,
            self.checkpoint_sha256[:12],
            self.sid_artifact_sha256[:12],
            str(self.resolver.get("candidate_item_for_path", "unknown"))[:40],
            self.source,
            self.source_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "rank": self.rank,
            "score": self.score,
            "semantic_id": list(self.semantic_id),
            "item_id": self.item_id,
            "retrieval_mode": self.retrieval_mode,
            "checkpoint_sha256": self.checkpoint_sha256,
            "sid_artifact_sha256": self.sid_artifact_sha256,
            "resolver": dict(self.resolver),
            "source": self.source,
            "source_version": self.source_version,
        }


class TigerSourceTool:
    """Adapt a TIGER retrieval engine to the candidate-source seam.

    Parameters
    ----------
    engine:
        Anything with ``recommend(history_item_ids, k=...)`` returning an object exposing
        ``retrieval_mode``, ``recommendations`` (each carrying ``parent_asin``, ``rank``,
        ``score``, ``semantic_id``, ``item_id``) and the artifact hashes.  Normally a serving
        engine; injected so a test needs no checkpoint.
    require_certified:
        When true (the default, and the canonical/evaluation posture), an APPROXIMATE retrieval
        is refused rather than recorded.  Set false only for interactive use, where the true
        mode still travels with every proposal.
    """

    def __init__(
        self,
        engine: Any,
        *,
        require_certified: bool = True,
        max_k: int = 100,
    ) -> None:
        if not callable(getattr(engine, "recommend", None)):
            raise CandidateSourceUnavailable(
                "a tiger source needs an engine with a callable recommend() method"
            )
        if not callable(getattr(engine, "parent_asin_to_item_id", None)):
            raise CandidateSourceUnavailable(
                "a tiger source needs an engine that can resolve a trusted history; "
                "the identity mapping is never guessed"
            )
        if max_k < 1:
            raise CandidateSourceUnavailable(f"max_k must be >= 1, got {max_k}")
        self._engine = engine
        self._require_certified = bool(require_certified)
        self._max_k = int(max_k)
        self._last_proposals: tuple[TigerProposal, ...] = ()
        self._last_outcome: dict[str, Any] = {"state": "NOT_CALLED"}

    # -- metadata ---------------------------------------------------------- #

    @property
    def source(self) -> CandidateSource:
        """The candidate source this tool implements."""
        return CandidateSource.TIGER

    @property
    def score_kind(self) -> str:
        """What this tool's scores mean: a Semantic-ID sequence log-probability."""
        return TIGER_SCORE_KIND

    @property
    def engine(self) -> Any:
        """The wrapped engine (exposed for lifecycle inspection)."""
        return self._engine

    @property
    def require_certified(self) -> bool:
        """True when an APPROXIMATE retrieval must be refused."""
        return self._require_certified

    def provenance_token(self) -> str | None:
        """The current call's provenance token, for the plane's ``source_query`` field.

        Returns ``None`` before any successful call so the plane records nothing rather than a
        misleading token.
        """
        if not self._last_proposals:
            return None
        return self._last_proposals[0].provenance_token()

    @property
    def last_proposals(self) -> tuple[TigerProposal, ...]:
        """The full proposals retained from the most recent successful call."""
        return self._last_proposals

    @property
    def last_outcome(self) -> Mapping[str, Any]:
        """An explicit, inspectable outcome for the most recent call."""
        return dict(self._last_outcome)

    # -- execution --------------------------------------------------------- #

    def propose(
        self,
        *,
        arguments: Any,
        limit: int,
        history: tuple[str, ...] = (),
    ) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, score)`` triples in TIGER's own order.

        ``history`` comes from the plane's trusted reader.  Every failure mode is mapped to
        :class:`CandidateSourceUnavailable`; nothing is shortened, substituted or invented.
        """
        if not history:
            self._last_outcome = {"state": "UNAVAILABLE", "reason": "no_trusted_history"}
            raise CandidateSourceUnavailable(
                "the tiger source needs trusted history; the run supplied none"
            )
        try:
            encoded = [self._engine.parent_asin_to_item_id(value) for value in history]
        except Exception as exc:  # noqa: BLE001 - any mapping failure is an unusable source
            self._last_outcome = {"state": "UNAVAILABLE", "reason": "history_unmappable"}
            raise CandidateSourceUnavailable(
                "the tiger source could not resolve the trusted history: "
                f"{type(exc).__name__}"
            ) from exc

        k = max(1, min(int(limit), self._max_k))
        try:
            result = self._engine.recommend(encoded, k=k)
        except CandidateSourceUnavailable:
            self._last_outcome = {"state": "UNAVAILABLE", "reason": "engine_unavailable"}
            raise
        except Exception as exc:  # noqa: BLE001 - any engine failure is a source-local failure
            self._last_outcome = {"state": "UNAVAILABLE",
                                  "reason": f"engine_error:{type(exc).__name__}"}
            raise CandidateSourceUnavailable(
                f"the tiger engine failed: {type(exc).__name__}"
            ) from exc

        mode = self._mode_of(result)
        if self._require_certified and mode != CERTIFIED:
            self._last_outcome = {"state": "REFUSED", "reason": "not_certified",
                                  "retrieval_mode": mode}
            raise CandidateSourceUnavailable(
                f"certified retrieval required but the engine reported {mode!r}; an "
                "APPROXIMATE result is never reported as CERTIFIED"
            )

        recommendations = list(getattr(result, "recommendations", ()) or ())
        if not recommendations:
            self._last_proposals = ()
            self._last_outcome = {"state": "EMPTY", "retrieval_mode": mode,
                                  "reason": "retrieval_returned_no_candidates"}
            return []

        checkpoint = str(getattr(result, "checkpoint_sha256", "") or "")
        sid_hash = str(getattr(result, "sid_artifact_sha256", "") or "")
        resolver = dict(getattr(result, "resolver", {}) or {})
        proposals: list[TigerProposal] = []
        for position, item in enumerate(recommendations, start=1):
            proposals.append(
                self._validate(item, position=position, mode=mode, checkpoint=checkpoint,
                               sid_hash=sid_hash, resolver=resolver)
            )
        self._last_proposals = tuple(proposals)
        self._last_outcome = {"state": "OK", "retrieval_mode": mode,
                              "candidates": len(proposals)}
        return [(p.parent_asin, p.rank, p.score) for p in proposals]

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _mode_of(result: Any) -> str:
        mode = getattr(result, "retrieval_mode", None)
        if mode not in (CERTIFIED, APPROXIMATE):
            raise CandidateSourceUnavailable(
                f"the tiger engine reported retrieval_mode {mode!r}; it must be "
                f"{CERTIFIED!r} or {APPROXIMATE!r}"
            )
        return str(mode)

    def _validate(self, item: Any, *, position: int, mode: str, checkpoint: str,
                  sid_hash: str, resolver: Mapping[str, Any]) -> TigerProposal:
        """Refuse a malformed proposal instead of recording a fabricated or partial one."""
        parent_asin = getattr(item, "parent_asin", None)
        if not isinstance(parent_asin, str) or not parent_asin.strip():
            raise CandidateSourceUnavailable(
                f"candidate {position} has no usable parent_asin; the source will not emit a "
                "candidate whose identity is unknown"
            )
        rank = getattr(item, "rank", None)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != position:
            raise CandidateSourceUnavailable(
                f"candidate {position} reports rank {rank!r}; ranks must be 1-based and "
                "contiguous in the source's own order"
            )
        score = getattr(item, "score", None)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise CandidateSourceUnavailable(
                f"candidate {position} has a non-numeric score {score!r}"
            )
        score = float(score)
        if not math.isfinite(score):
            raise CandidateSourceUnavailable(
                f"candidate {position} has a non-finite score {score!r}"
            )
        raw_sid = getattr(item, "semantic_id", None)
        if not raw_sid:
            raise CandidateSourceUnavailable(
                f"candidate {position} carries no Semantic ID; a TIGER candidate is a "
                "Semantic-ID proposal and that provenance is required"
            )
        try:
            semantic_id = tuple(int(value) for value in raw_sid)
        except (TypeError, ValueError) as exc:
            raise CandidateSourceUnavailable(
                f"candidate {position} has an unreadable Semantic ID: {type(exc).__name__}"
            ) from exc
        if any(value < 0 for value in semantic_id):
            raise CandidateSourceUnavailable(
                f"candidate {position} has a negative Semantic-ID digit {semantic_id!r}"
            )
        item_id = getattr(item, "item_id", None)
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            raise CandidateSourceUnavailable(
                f"candidate {position} reports item_id {item_id!r}; a resolved catalogue id "
                "is required for provenance"
            )
        if not _HEX12.match(checkpoint or ""):
            raise CandidateSourceUnavailable(
                "the tiger engine did not report a usable checkpoint sha256"
            )
        if not _HEX12.match(sid_hash or ""):
            raise CandidateSourceUnavailable(
                "the tiger engine did not report a usable SID artifact sha256"
            )
        return TigerProposal(
            parent_asin=parent_asin,
            rank=rank,
            score=score,
            semantic_id=semantic_id,
            item_id=item_id,
            retrieval_mode=mode,
            checkpoint_sha256=checkpoint,
            sid_artifact_sha256=sid_hash,
            resolver=resolver,
        )
