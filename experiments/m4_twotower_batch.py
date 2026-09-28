"""Exact batched Two-Tower retrieval — prototype for the M4 materialisation bottleneck.

Why this exists
---------------
The per-user reference path measures ~30.3 s/user (~168 h for 20,000 users), which is not viable.
The cost is not the model: :meth:`TwoTowerInferenceEngine.score_catalog` builds a **one-row**
tensor, calls ``full_catalog_scores``, checks finiteness, and materialises **156,747 Python floats**
via ``.tolist()`` -- per user.

``full_catalog_scores`` is natively batched: it takes ``[batch, seq_len]`` and returns
``[batch, num_items + 1]``. This prototype uses that batching and converts to Python only the
**top-100** rows it keeps.

Exactness
---------
This is **not** an approximation layer. It changes batching and nothing else:

* same checkpoint, same model instance;
* same trusted history, same ``encode_inference_history`` (left-pad, most recent ``max_seq_len``);
* same catalogue and same score matrix semantics (``full_catalog_scores``, index 0 = PAD);
* same masking and same tie-break, because ranking is delegated to the repository's own
  :func:`~recommendation.inference.ranking.rank_top_k` -- the identical function the reference
  calls, so ``lexsort((candidate_ids, -candidate_scores))`` is not reimplemented;
* top-``k`` only, with ``k`` bounded exactly as the reference bounds it (``min(limit, 100)``).

No ANN, no quantisation, no reordering, no float32 downcast.

The equivalence check in :mod:`experiments.m4_twotower_batch_bench` compares this path against
``TwoTowerSourceTool`` on the same users and reports order mismatches and score deltas.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recommendation.datasets.sasrec import encode_inference_history  # noqa: E402
from recommendation.inference.ranking import rank_top_k  # noqa: E402
from recommendation.inference.twotower import (  # noqa: E402
    TWO_TOWER_SCORE_KIND,
    TwoTowerInferenceConfig,
    TwoTowerInferenceEngine,
)

__all__ = ["BatchedTwoTowerScorer", "BatchedHead"]

#: The reference bounds ``k`` with ``max(1, min(limit, 100))``; kept identical here.
MAX_K = 100


@dataclass(frozen=True)
class BatchedHead:
    """One user's exact top-``k`` result."""

    user_int_id: int
    rows: tuple[tuple[str, int, float], ...]
    finite: bool


class BatchedTwoTowerScorer:
    """Exact batched full-catalogue Two-Tower retrieval over many users at once.

    Parameters
    ----------
    config:
        The engine configuration; the checkpoint is loaded once, exactly as the reference does.
    """

    def __init__(self, config: TwoTowerInferenceConfig) -> None:
        self._engine = TwoTowerInferenceEngine(config)
        self._model = self._engine.model
        self._max_seq_len = self._engine.max_seq_len
        self._num_items = self._engine.num_items
        self._identity = self._engine

    # -- properties -------------------------------------------------------- #

    @property
    def score_kind(self) -> str:
        """The repository's own label for these scores."""
        return TWO_TOWER_SCORE_KIND

    @property
    def num_items(self) -> int:
        """Catalogue size (index 0 is PAD)."""
        return self._num_items

    @property
    def max_seq_len(self) -> int:
        """The history window the accepted model was trained with."""
        return self._max_seq_len

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Resolve an identity through the engine's accepted mapping."""
        return self._engine.parent_asin_to_item_id(parent_asin)

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Resolve an item id back to its identity."""
        return self._engine.item_id_to_parent_asin(item_id)

    # -- batched scoring --------------------------------------------------- #

    def score_batch(self, histories: Sequence[Sequence[int]]) -> np.ndarray:
        """Return a ``[batch, num_items + 1]`` float64 score matrix for the given histories.

        The finiteness check mirrors ``score_catalog``: the reference rejects a non-finite matrix
        rather than ranking it, because a NaN target would sort first.
        """
        if not histories:
            return np.zeros((0, self._num_items + 1), dtype=np.float64)
        encoded = [
            encode_inference_history(list(history), self._max_seq_len, self._num_items)
            for history in histories
        ]
        tensor = torch.tensor(encoded, dtype=torch.long, device=self._engine.device)
        with torch.inference_mode():
            scores = self._model.full_catalog_scores(tensor)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("the two-tower model produced non-finite catalog scores")
        # float64 on CPU: the reference converts each score with float(), and rank_top_k compares
        # them with lexsort. Converting the whole matrix once keeps the same values.
        return scores.detach().to("cpu", torch.float64).numpy()

    def heads_for_batch(
        self,
        users: Sequence[tuple[int, Sequence[int]]],
        *,
        k: int = MAX_K,
    ) -> list[BatchedHead]:
        """Exact top-``k`` heads for a batch of ``(user_int_id, history_item_ids)`` pairs.

        Ranking is delegated to the repository's :func:`rank_top_k` per row, so masking
        (PAD + seen items) and the tie-break are the accepted serving semantics, not a restatement
        of them.
        """
        if not users:
            return []
        bounded_k = max(1, min(int(k), MAX_K))
        matrix = self.score_batch([history for _uid, history in users])
        heads: list[BatchedHead] = []
        for row, (user_int_id, history) in enumerate(users):
            ranked = rank_top_k(
                matrix[row],
                num_items=self._num_items,
                seen_item_ids=history,
                k=bounded_k,
            )
            heads.append(
                BatchedHead(
                    user_int_id=int(user_int_id),
                    rows=tuple(
                        (
                            self.item_id_to_parent_asin(item.item_id),
                            int(item.rank),
                            float(item.score),
                        )
                        for item in ranked
                    ),
                    finite=True,
                )
            )
        return heads

    # -- convenience ------------------------------------------------------- #

    def one(self, history: Sequence[int], *, k: int = MAX_K) -> list[tuple[str, int, float]]:
        """Exact top-``k`` for a single history, through the batched path (batch size 1)."""
        heads = self.heads_for_batch([(0, history)], k=k)
        return list(heads[0].rows) if heads else []


def build_scorer(
    checkpoint: pathlib.Path | str,
    mappings: pathlib.Path | str,
    *,
    categories: pathlib.Path | str | None = None,
    device: str = "cpu",
) -> BatchedTwoTowerScorer:
    """Construct a :class:`BatchedTwoTowerScorer` from the frozen M4 checkpoint and mapping."""
    checkpoint_path = pathlib.Path(checkpoint)
    config = TwoTowerInferenceConfig(
        checkpoint_path=checkpoint_path,
        categories_path=(
            pathlib.Path(categories)
            if categories is not None
            else checkpoint_path.parent / "metadata_categories.json"
        ),
        mappings_path=pathlib.Path(mappings),
        device=device,
    )
    return BatchedTwoTowerScorer(config)
