"""M4 frozen query / seed construction rules.

The two argument-keyed M4 sources (``CATALOG_SEARCH`` and ``SIMILAR_ITEM``) take arguments from
the *policy*.  If those arguments were model-chosen, M4 would compare traversal **and** query
formulation at once, and §1.1 of ``docs/M4_PREREGISTRATION.md`` explicitly excludes query
construction from the research question.

This module removes that confound by making both arguments a **pure function of the trusted
test-time history ``h``**.  It is hash-bound in the preregistration (``§12`` artifact 6), so it
cannot be edited after the freeze without invalidating the run.

The definition of ``h`` is preregistration §5.1: the parent ASINs of ``case.test_history``, which
is ``train_history + (validation_target,)`` under ``temporal_leave_two_out``.  Callers pass the
already-mapped tuple; this module never reads the mapping itself.

Neither function is told anything about the model, the ledger, the candidate set, or the
catalogue's global statistics, and neither is random.  ``TOKEN_PATTERN`` is imported from the
trusted ``similar_item`` module rather than re-declared, so the two sources tokenise identically.

Frozen constants are named and exported so a test can assert them.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from recommendation.control.arguments import MAX_SEARCH_TERMS
from recommendation.control.similar_item import TOKEN_PATTERN

__all__ = [
    "MAX_HISTORY_ITEMS_SCANNED",
    "MAX_TERMS",
    "MIN_TERM_LENGTH",
    "deterministic_terms",
    "newest_indexed",
]

#: How many of the most recent history items are scanned for search terms.  Scanning the newest
#: items first is what makes the query reflect the user's current intent rather than their oldest
#: purchase.
MAX_HISTORY_ITEMS_SCANNED = 3

#: Upper bound on the returned term list.  Deliberately below
#: :data:`~recommendation.control.arguments.MAX_SEARCH_TERMS` so the rule can never produce a term
#: list the argument model would refuse, and so a single verbose title cannot crowd out the terms
#: contributed by the other scanned items.
MAX_TERMS = 8

#: Terms shorter than this carry almost no discriminative signal (sizes, units, fragments) and are
#: dropped.  This is a fixed rule, not a tuned parameter.
MIN_TERM_LENGTH = 3


def deterministic_terms(
    history: Sequence[str],
    *,
    describe: Callable[[str], str | None] | None = None,
) -> tuple[str, ...]:
    """Return the frozen lexical terms for ``CATALOG_SEARCH`` given trusted history ``h``.

    ``history`` is chronological (oldest first), so the newest items are at the end.  The newest
    ``MAX_HISTORY_ITEMS_SCANNED`` items are scanned newest-first; for each, its descriptive text is
    tokenised with the repository's own tokeniser and its terms are added in first-seen order.
    At most ``MAX_TERMS`` terms are returned.

    ``describe`` maps a history identity to its descriptive text (normally the catalogue title).
    It is injected so this function stays pure and testable, and so it reads the catalogue through
    the same trusted handle the rest of the system uses.  A ``None`` or empty description
    contributes no terms; an identity the catalogue does not know is skipped rather than guessed at.

    Returns an empty tuple when nothing usable is found.  An empty tuple is a *decision* the caller
    must honour -- the caller skips that source rather than sending a degenerate query -- and it can
    never exceed ``MAX_SEARCH_TERMS``.
    """
    if describe is None:
        return ()
    recent = list(history)[-MAX_HISTORY_ITEMS_SCANNED:]
    terms: list[str] = []
    seen: set[str] = set()
    for identity in reversed(recent):
        text = describe(identity)
        if not text:
            continue
        for token in TOKEN_PATTERN.findall(str(text).casefold()):
            if len(token) < MIN_TERM_LENGTH or token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= MAX_TERMS:
                return tuple(terms)
    return tuple(terms)


def newest_indexed(
    history: Sequence[str],
    *,
    is_indexed: Callable[[str], bool],
) -> str | None:
    """Return the newest history identity the ``SIMILAR_ITEM`` index knows, or ``None``.

    ``history`` is chronological, so this scans from the end.  ``is_indexed`` is the index's own
    membership test (``SimilarItemSource.covers``), injected so this function performs no lookup of
    its own and stays pure.

    Returns ``None`` when no history item is indexed.  ``None`` means "this source has no legal
    seed for this user", and the caller skips ``SIMILAR_ITEM`` rather than inventing a seed --
    inventing one would be implicit target selection, which the repository removed in Phase 2.1.
    """
    for identity in reversed(list(history)):
        if is_indexed(identity):
            return identity
    return None


def iter_terms(history: Sequence[str], describe: Callable[[str], str | None]) -> Iterable[str]:
    """Convenience wrapper yielding :func:`deterministic_terms` one term at a time."""
    return iter(deterministic_terms(history, describe=describe))
