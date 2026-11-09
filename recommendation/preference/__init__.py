"""P14-B0 legal history-state read path and preference-aware scorer V0.

This package adds two pieces on top of the accepted M9/M5 memory and the accepted
search/eligibility components:

* ``history``: a query-independent, budget-bounded read of the *current user's* M9
  state with provenance.  It never reads evaluator labels and never reasons from the
  current shopping query alone; it reconstructs the active state that M9 already
  applied and then optionally ranks preferences by deterministic query-term overlap.
* ``scorer``: a deterministic, explainable reranker over a FIXED candidate set.  It
  does not search, does not use ground truth, and cannot make a hard-constraint
  violator pass by relevance alone.
"""

__all__ = ["HISTORY_READ_SCHEMA", "SCORER_VERSION"]

HISTORY_READ_SCHEMA = "agentrecx.p14b0.history_state_read.v1"
SCORER_VERSION = "preference_scorer_v0.1"
