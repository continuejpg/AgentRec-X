"""The Two-Tower candidate source (post-Phase-5 model expansion).

What it is
----------
A :class:`~recommendation.control.candidate_plane.CandidateSourceTool` that consults the
Two-Tower retrieval engine through the **accepted recommendation stack**: the same
``RecommendationTool`` contract, the same trusted-history reader, and the same plane that
grounds every identity before the ledger records it.

What it deliberately does not do
--------------------------------
* **It never grounds an identity.**  It returns ``(parent_asin, rank, score)`` triples and the
  plane's :class:`~recommendation.control.GroundingVerifier` decides whether each product
  exists.  An identity this tool reports that the mapping does not know is recorded as an
  UNGROUNDED audit entry, exactly as for every other source.
* **It never chooses its own history.**  The trusted history is supplied by the plane through
  the same call signature the history source uses, so the tool cannot invent a user.
* **It never mixes its scores with another source's.**  Its scores are cosine similarities,
  labelled :data:`~recommendation.inference.TWO_TOWER_SCORE_KIND`; the ledger records that
  label, and the only cross-source combination in this repository is rank-based fusion, which
  uses ordering alone.
"""

from __future__ import annotations

from typing import Any

from .arguments import CandidateSource
from .candidate_plane import CandidateSourceUnavailable

__all__ = ["TWO_TOWER_SCORE_KIND", "TwoTowerSourceTool"]

#: Re-exported so the control plane does not import the inference package for a string.
TWO_TOWER_SCORE_KIND = "two_tower_cosine"


class TwoTowerSourceTool:
    """Adapt a Two-Tower serving engine to the candidate-source seam.

    Parameters
    ----------
    engine:
        Anything with ``recommend(history_item_ids, k=...)`` returning a result whose
        ``recommendations`` carry ``parent_asin``, ``rank`` and ``score``.  Normally the
        accepted ``TwoTowerInferenceEngine``; injected so a test needs no checkpoint.
    """

    def __init__(self, engine: Any) -> None:
        if not callable(getattr(engine, "recommend", None)):
            raise CandidateSourceUnavailable(
                "a two-tower source needs an engine with a callable recommend() method"
            )
        self._engine = engine

    @property
    def source(self) -> CandidateSource:
        """The candidate source this tool implements."""
        return CandidateSource.TWO_TOWER

    @property
    def score_kind(self) -> str:
        """What this tool's scores mean: a learned cosine, not a SASRec logit."""
        return TWO_TOWER_SCORE_KIND

    @property
    def engine(self) -> Any:
        """The wrapped engine (exposed for lifecycle inspection)."""
        return self._engine

    def propose(
        self,
        *,
        arguments: Any,
        limit: int,
        history: tuple[str, ...] = (),
    ) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, score)`` triples in the engine's own order.

        ``history`` comes from the plane's trusted reader.  An empty history is reported as an
        unavailable source rather than silently answered from nothing: the plane classifies a
        source failure, and a fabricated empty result set would look like "retrieval found
        nothing" instead of "this source could not be consulted".
        """
        if not history:
            raise CandidateSourceUnavailable(
                "the two-tower source needs trusted history; the run supplied none"
            )
        # The plane supplies *identities*; the engine is keyed by integer id, so the mapping
        # conversion happens here.  An identity the mapping does not know is an unavailable
        # source for this call rather than a silently shortened history.
        try:
            encoded = [self._engine.parent_asin_to_item_id(value) for value in history]
        except Exception as exc:  # noqa: BLE001 - any mapping failure is an unusable source
            raise CandidateSourceUnavailable(
                f"the two-tower source could not resolve the trusted history: "
                f"{type(exc).__name__}"
            ) from exc
        result = self._engine.recommend(encoded, k=max(1, min(int(limit), 100)))
        return [
            (str(item.parent_asin), int(item.rank), float(item.score))
            for item in result.recommendations
        ]
