"""M4 frozen source adapters.

Four :class:`~recommendation.control.candidate_plane.CandidateSourceTool` implementations that
serve **pre-materialised** per-user heads instead of running a live retriever.

Why these exist
---------------
Preregistration §12.3 measured that a live pass is not viable on the target machine: building the
BM25 index and the TF-IDF similar-item index in one process is OOM-killed, and a single
similar-item query costs seconds. Materialising the heads once is therefore a **prerequisite**, not
an optimisation, and it is also what licences §13's refusal to claim live retrieval latency.

What these adapters guarantee
-----------------------------
* **Identical bytes for both arms.** A head is a pure function of the frozen catalogue, the frozen
  history and the frozen rules; both arms read the same array. Nothing here consults the model, the
  ledger, the clock or any global statistic.
* **One call per ``(user, source)``.** Preregistration §9 measured that a second query of the same
  source double-counts entries in RRF (an identity's fused score rose 0.0164 -> 0.0323 and overtook
  the previous leader). Each adapter instance is bound to one user and refuses a second call, so a
  duplicate result can **never reach** ``CandidateLedger``.

No frozen file is modified: these classes implement the repository's own tool protocol, which is a
documented extension seam (``candidate_plane.py`` validates the declared ``source``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from recommendation.control.arguments import CandidateSource

__all__ = [
    "DuplicateSourceQuery",
    "FrozenHeadTool",
    "AccountedSources",
]


class DuplicateSourceQuery(RuntimeError):
    """Raised when one ``(user, source)`` pair is queried a second time.

    Preregistration §9 forbids this. Raising here is the *hard isolation* guarantee: the exception
    is raised inside ``propose``, which runs **before** the plane grounds or records anything
    (``candidate_plane.py:391-419``), so the duplicate never reaches the ledger. The loop classifies
    the resulting failure and terminates the run per §9.
    """


@dataclass
class FrozenHeadTool:
    """One user's pre-materialised head for one source.

    Parameters
    ----------
    source:
        Which :class:`CandidateSource` this adapter stands in for.
    score_kind:
        The label the **real** tool declares, carried through so ledger provenance is identical to
        a live run. Resolved from the real tool by the materialisation pass, never invented here.
    head:
        ``(parent_asin, rank, score)`` triples, best first, already truncated to the frozen depth.
    """

    source: CandidateSource
    score_kind: str
    head: tuple[tuple[str, int, float], ...]
    #: Shared one-shot flag. A list is used so an adapter that serves the head through another
    #: object (the HISTORY source goes through the accepted ``RecommendationTool``) can share the
    #: *same* flag instead of each holding its own, which would let one path bypass the rule.
    state: list[bool] = field(default_factory=lambda: [False], repr=False)

    # -- protocol ---------------------------------------------------------- #

    def propose(
        self,
        *,
        arguments: Any = None,
        limit: int = 0,
        history: Sequence[str] = (),
    ) -> list[tuple[str, int, float]]:
        """Return this source's frozen ranking, honouring the requested ``limit``.

        ``arguments`` is accepted and deliberately **ignored**. The head was already produced by the
        frozen rules of preregistration §5 during materialisation, so there is nothing to derive and
        nothing to validate here: ignoring the arguments is what guarantees a head can never be
        paired with a query that did not produce it.

        Accepting both argument shapes matters. ``SELECT_SOURCE`` carries
        :class:`~recommendation.control.arguments.SelectSourceArguments` (a ``source`` and a
        ``limit``), while the dedicated actions carry ``terms`` or ``seed_parent_asin``. The frozen
        plane's own argument-keyed adapters read only their dedicated field
        (``candidate_plane.py:216`` reads ``terms``), so a live ``SELECT_SOURCE{catalog_search}``
        would return an empty list. M4 serves the same four sources uniformly through
        ``SELECT_SOURCE``, so this adapter must not depend on which shape arrived.

        The ``limit`` is still honoured by slicing, because depth is a frozen experimental
        parameter (§3) and a live tool would have honoured it too.
        """
        self._claim()
        if limit <= 0:
            return []
        return [tuple(row) for row in self.head[:limit]]  # type: ignore[misc]

    def _claim(self) -> None:
        """Consume the single permitted call, or raise if it was already consumed."""
        if self.state[0]:
            raise DuplicateSourceQuery(
                f"source {self.source.value!r} has already been queried for this user; "
                "preregistration §9 forbids a second query because it double-counts in RRF"
            )
        self.state[0] = True

    @property
    def consumed(self) -> bool:
        """True once this tool has served its one permitted call."""
        return self.state[0]


@dataclass
class AccountedSources:
    """Builds one fresh source tool **per user** from a materialised head table.

    The per-user rebuild is what makes the §9 one-call rule enforceable without any cross-user
    state: a fresh tool per user starts unconsumed, and a second query on the same user finds its
    tool already consumed.
    """

    #: source -> user_int_id -> frozen ``(parent_asin, rank, score)`` rows.
    heads: Mapping[CandidateSource, Mapping[int, tuple[tuple[str, int, float], ...]]]
    #: source -> the ``score_kind`` the real tool declares.
    score_kinds: Mapping[CandidateSource, str]

    def for_user(self, user_int_id: int) -> dict[CandidateSource, FrozenHeadTool]:
        """Return one fresh :class:`FrozenHeadTool` per materialised source for this user."""
        built: dict[CandidateSource, FrozenHeadTool] = {}
        for source, per_user in self.heads.items():
            rows = per_user.get(int(user_int_id), ())
            built[source] = FrozenHeadTool(
                source=source,
                score_kind=self.score_kinds.get(source, "unspecified"),
                head=tuple(rows),
            )
        return built
