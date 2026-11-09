"""Typed errors for the M6.1 offline belief-aware ranking foundation.

Every rejection in this package is one of these types so a caller can tell why a
snapshot, feature build, policy or evaluation was refused.  These are offline-lab
components: raising one of these must never affect the serving path.
"""

from __future__ import annotations

__all__ = [
    "CandidateSnapshotError",
    "FeatureBuildError",
    "RankingError",
    "RankingPolicyError",
    "RankingProtocolError",
]


class RankingError(Exception):
    """Base class for every M6.1 ranking-foundation rejection."""


class CandidateSnapshotError(RankingError):
    """A candidate snapshot is invalid, foreign, stale or inconsistent with its refs."""


class FeatureBuildError(RankingError):
    """Trusted candidate features could not be built without inventing evidence."""


class RankingPolicyError(RankingError):
    """A ranking policy was asked to do something outside its contract."""


class RankingProtocolError(RankingError):
    """The frozen paired evaluation protocol or its inputs are invalid."""
