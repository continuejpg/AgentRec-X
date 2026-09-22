"""The frozen canonical item-score rule (Step 2.3 skeleton, schema only).

``docs/TIGER_BACKEND.md`` section 5 freezes one definition, and **every** retrieval algorithm
must rank under it.  The definition is stated here, in the module whose name says so, because
splitting it across the search implementations is exactly how beam search and branch-and-bound
would come to disagree about what "score" means.

The rule
--------

For a catalogue item ``i`` with Semantic-ID token path ``t(i) = (t_1, ..., t_M)``::

    score(i | history) = SUM over l = 1..M of log P_model(t_l | history, t_<l)

where ``P_model`` is the softmax over the model's **full** token vocabulary - the decoder's raw
logits, unmodified.  ``EOS`` is **excluded** from the sum; it is how the *search* recognises a
completed path, not a term in the score.

Why the distinction matters
---------------------------

``model scoring semantics`` define the item **order**; the ``search algorithm`` defines the item
**coverage**.  The catalogue trie is a search constraint only.  Renormalising the softmax over
only the currently valid child tokens would make an item's score depend on which other items
happened to be in the search frontier - i.e. on the search budget - so it is forbidden unless
separately selected, versioned and pre-registered.

What Step 2.3 implements
------------------------

No model.  :data:`SCORE_RULE` is the frozen declaration that a Step-2.5 checkpoint must record
verbatim, and :func:`sequence_log_probability` is the one implementation of the sum, written
against a log-probability matrix so it can be validated now with synthesised inputs and reused
unchanged once a real model exists.  :func:`assert_monotone_prefix_bound` states the
admissibility property the certified search relies on (section 5.3).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from tiger_public.contracts import ContractError, TokenLayout

__all__ = [
    "SCORE_RULE",
    "assert_monotone_prefix_bound",
    "prefix_upper_bounds",
    "sequence_log_probability",
    "token_path_for",
    "validate_score_rule",
]

#: The frozen rule.  A checkpoint records this by value; changing any field requires a new
#: ``version`` and a full re-benchmark, because the two rules are not comparable.
SCORE_RULE: dict[str, Any] = {
    "version": 1,
    "eos_in_score": False,
    "softmax": "full_vocabulary",
    "child_renormalisation": False,
    "terms": "sum of log P over the item's complete SID token path",
    "note": "changing any field requires a new version and a full re-benchmark",
}


def validate_score_rule(rule: dict[str, Any] | None) -> None:
    """Refuse a score rule that is not the frozen one.

    A checkpoint that declares ``eos_in_score: true``, a renormalised softmax, or child
    renormalisation is rejected rather than silently scored under a different rule.
    """
    candidate = dict(rule or {})
    for key in ("eos_in_score", "softmax", "child_renormalisation"):
        if candidate.get(key) != SCORE_RULE[key]:
            raise ContractError(
                f"score_rule.{key} must be {SCORE_RULE[key]!r}, got {candidate.get(key)!r}; "
                "the item-score rule is frozen (docs/TIGER_BACKEND.md section 5)"
            )
    if int(candidate.get("version", 0)) != int(SCORE_RULE["version"]):
        raise ContractError(
            f"score_rule.version must be {SCORE_RULE['version']}, got "
            f"{candidate.get('version')!r}"
        )


def sequence_log_probability(
    log_probabilities: np.ndarray,
    tokens: Sequence[int],
) -> np.ndarray:
    """Sum ``log P(t_l | history, t_<l)`` over one item's complete token path.

    Parameters
    ----------
    log_probabilities:
        ``[batch, M, V]`` **full-vocabulary** log-softmax values, one row per decoding step.
        The caller produces these with a plain ``log_softmax`` over the whole vocabulary; this
        function never renormalises, and it ignores any trie or frontier information, because
        the score must not depend on the search.
    tokens:
        The item's token path, length ``M``.

    Returns
    -------
    ``[batch]`` float32.  ``EOS`` is not included: ``tokens`` is the item's SID path only.
    """
    table = np.asarray(log_probabilities, dtype=np.float64)
    if table.ndim != 3:
        raise ContractError(
            f"log_probabilities must be [batch, steps, vocab], got shape {tuple(table.shape)}"
        )
    path = tuple(int(token) for token in tokens)
    if table.shape[1] != len(path):
        raise ContractError(
            f"log_probabilities has {table.shape[1]} steps but the token path has {len(path)}"
        )
    vocab = table.shape[2]
    total = np.zeros(table.shape[0], dtype=np.float64)
    for step, token in enumerate(path):
        if not 0 <= token < vocab:
            raise ContractError(
                f"token {token} at step {step} is outside the vocabulary [0, {vocab})"
            )
        total += table[:, step, token]
    return total.astype(np.float32)


def prefix_upper_bounds(prefix_log_probabilities: np.ndarray) -> np.ndarray:
    """Cumulative prefix sums, i.e. the branch-and-bound upper bounds.

    Because every term is a log-probability and therefore ``<= 0``, the sum over a prefix is an
    upper bound on the sum over any extension of that prefix.  That is what makes the certified
    search exact rather than merely deep.
    """
    table = np.asarray(prefix_log_probabilities, dtype=np.float64)
    if table.ndim != 3:
        raise ContractError(
            f"log_probabilities must be [batch, steps, vocab], got shape {tuple(table.shape)}"
        )
    return np.cumsum(table, axis=1).astype(np.float32)


def assert_monotone_prefix_bound(log_probabilities: np.ndarray) -> None:
    """Assert the admissibility property the certified search depends on.

    Two facts must hold for the frozen rule: every term is ``<= 0`` (a log-probability from a
    full-vocabulary softmax), and extending a prefix never increases the accumulated sum.  A
    violation means the input is not a log-softmax - for example a renormalised or
    temperature-scaled distribution - and certifying against it would be unsound.
    """
    table = np.asarray(log_probabilities, dtype=np.float64)
    if np.any(table > 1e-6):
        raise ContractError(
            "log-probabilities must be <= 0; a positive value means the input is not a "
            "full-vocabulary log-softmax and the prefix bound would not be admissible"
        )
    cumulative = np.cumsum(table, axis=1)
    if table.shape[1] > 1 and np.any(np.diff(cumulative, axis=1) > 1e-6):
        raise ContractError("prefix sums must be non-increasing along the token path")


def token_path_for(semantic_id: Sequence[int], layout: TokenLayout) -> tuple[int, ...]:
    """The model token path of one item; the PAD sentinel is refused, not aliased."""
    return layout.tokenise(semantic_id)
