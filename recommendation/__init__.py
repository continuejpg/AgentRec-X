"""AgentRec-X recommendation package - Milestone 1 (Amazon Reviews 2023 preprocessing).

The package ``__init__`` deliberately re-exports nothing.  Import the modules
directly so that ``python -m recommendation.preprocess`` behaves predictably::

    from recommendation import config
    from recommendation.io_utils import load_interactions
    from recommendation.preprocess import run_preprocessing, preprocess_interactions

Milestone 1 contains no models: ItemCF, SASRec, evaluation, retrieval and agent
components arrive in later milestones.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
