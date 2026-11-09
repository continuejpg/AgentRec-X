"""Shared vocabulary for the M5-MVP unified decision state engine.

This module is the typed core every other ``recommendation.state`` module imports.  It
deliberately has no model, dataset, network or service dependency: the unified decision
state is pure conversational state, and a state engine that cannot be imported without
the recommendation stack could not be replayed offline.

Reuse policy
------------
The accepted preference-memory schema already owns the narrow preference ontology, and
this module **imports it rather than restating it**:

* :class:`~recommendation.memory.schemas.PreferenceKind`
* :class:`~recommendation.memory.schemas.PreferencePolarity`
* :class:`~recommendation.memory.schemas.PreferenceStatus`
* ``MAX_VALUE_LENGTH`` / ``MAX_SOURCE_TEXT_LENGTH`` / ``SECRET_REDACTION`` and
  :func:`~recommendation.memory.schemas.contains_secret_like_text` -- the M5 value
  bounds and the credential check are the accepted M9 ones, not a second opinion.

:class:`PreferenceScope` and :class:`PreferenceLifecycle` are the one deliberate
duplication in this module.  Their accepted definitions live in
``recommendation.control.task_state``, but importing that module executes
``recommendation/control/__init__.py``, which imports the recommendation tool chain and
therefore PyTorch.  The M5 state layer must be importable and replayable with no model
stack (the smoke example requires exactly that), while reusing the same four lifecycle
members and three scope members unchanged.  The duplication is therefore *pinned*:

* the members and their semantics are copied value-for-value from the accepted enums;
* ``tests/test_state_schemas.py`` asserts value parity and property parity against
  ``recommendation.control.task_state`` whenever that module is importable, so drift
  fails a test instead of silently creating a second vocabulary;
* when the control plane is wired to the state engine in a later milestone, the accepted
  enums can re-export these types (or the reverse) without changing any stored value.

Ordinal strength, not confidence
--------------------------------
:class:`BeliefStrength` is an **ordinal** signal.  It exists so "the user said it" can
outrank "the system guessed it" deterministically.  It is not a probability, it is not
calibrated, and no module may convert it to a probability-like score.  That is why it is
a small integer enum rather than a float: a float would invite exactly the pretence of
calibration the milestone forbids.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from recommendation.memory.schemas import (
    MAX_SOURCE_TEXT_LENGTH,
    MAX_VALUE_LENGTH,
    SECRET_REDACTION,
    PreferenceKind,
    PreferencePolarity,
    PreferenceStatus,
    contains_secret_like_text,
)

__all__ = [
    "BeliefOrigin",
    "BeliefStrength",
    "ConstraintAuthority",
    "DIGEST_LENGTH",
    "HARD_CONSTRAINT_AUTHORITIES",
    "HypothesisBasis",
    "KnowledgeStatus",
    "MAX_ATTRIBUTE_LENGTH",
    "MAX_COMPARED_REFS",
    "MAX_EVENT_ID_LENGTH",
    "MAX_EVIDENCE_REFS",
    "MAX_FEEDBACK_TEXT_LENGTH",
    "MAX_REASON_LENGTH",
    "MAX_REF_LENGTH",
    "MAX_REQUEST_TEXT_LENGTH",
    "MAX_SESSION_ID_LENGTH",
    "MAX_SOURCE_LENGTH",
    "MAX_SOURCE_TEXT_LENGTH",
    "MAX_TASK_ID_LENGTH",
    "MAX_TASK_INTENT_LENGTH",
    "MAX_USER_KEY_LENGTH",
    "MAX_VALUE_LENGTH",
    "MIN_COMPARED_REFS",
    "ObservationKind",
    "PreferenceDirection",
    "PreferenceKind",
    "PreferenceLifecycle",
    "PreferencePolarity",
    "PreferenceScope",
    "PreferenceStatus",
    "SECRET_REDACTION",
    "STATE_SCHEMA_VERSION",
    "bounded_identifier",
    "bounded_text",
    "canonical_json",
    "clean_attribute",
    "clean_preference_value",
    "contains_secret_like_text",
    "deterministic_id",
    "redacted_text",
]

#: Version of the unified-decision-state contract.  Stored in the database ``meta``
#: table and carried by every serialized state, so a future migration is explicit.
STATE_SCHEMA_VERSION = 1

#: Length of a canonical SHA-256 hex digest stored in a state reference.
DIGEST_LENGTH = 64

#: Bounds kept here rather than inline so validation and documentation cannot drift.
MAX_EVENT_ID_LENGTH = 128
MAX_USER_KEY_LENGTH = 128
MAX_TASK_ID_LENGTH = 128
MAX_SESSION_ID_LENGTH = 128
MAX_SOURCE_LENGTH = 80
MAX_ATTRIBUTE_LENGTH = 120
MAX_REF_LENGTH = 200
MAX_REQUEST_TEXT_LENGTH = 1000
MAX_FEEDBACK_TEXT_LENGTH = 500
MAX_REASON_LENGTH = 240
MAX_TASK_INTENT_LENGTH = 40
MAX_EVIDENCE_REFS = 8
MAX_COMPARED_REFS = 8
MIN_COMPARED_REFS = 2


class PreferenceScope(str, Enum):
    """How long a preference or constraint is meant to apply.

    Copied value-for-value from
    :class:`recommendation.control.task_state.PreferenceScope` (see the module
    docstring for why this module cannot import it directly).

    * :attr:`SESSION` -- applies to the current session only.  Never persisted.
    * :attr:`TASK` -- applies to the current task/request only.  Never persisted.
    * :attr:`LONG_TERM` -- crosses tasks.  May only come from an explicit user
      statement (or the read-only legacy-memory adapter), never from an inference.
    """

    SESSION = "session"
    TASK = "task"
    LONG_TERM = "long_term"

    @property
    def is_ephemeral(self) -> bool:
        """True when this scope must never be treated as durable memory."""
        return self is not PreferenceScope.LONG_TERM


class PreferenceLifecycle(str, Enum):
    """Where a preference belief came from and how much authority it carries.

    Copied value-for-value from
    :class:`recommendation.control.task_state.PreferenceLifecycle`.

    * :attr:`INFERRED_HYPOTHESIS` -- derived by the system; soft evidence only.  It may
      inform a soft ranking signal or a clarification question, and it may never become
      a hard constraint or a durable explicit preference on its own.
    * :attr:`EXPLICIT` -- the user stated it.
    * :attr:`CONFIRMED` -- the user confirmed a hypothesis.  Confirmation is an explicit
      user act; it never happens automatically.
    * :attr:`RETRACTED` -- the signal was withdrawn.  The belief is kept for provenance
      instead of being deleted.

    Only ``EXPLICIT`` and ``CONFIRMED`` may ever be proposed for durable memory; that
    property is reused from the accepted enum and re-pinned here.
    """

    EXPLICIT = "explicit"
    INFERRED_HYPOTHESIS = "inferred_hypothesis"
    CONFIRMED = "confirmed"
    RETRACTED = "retracted"

    @property
    def may_be_committed(self) -> bool:
        """True when this belief may be proposed for durable long-term memory."""
        return self in (PreferenceLifecycle.EXPLICIT, PreferenceLifecycle.CONFIRMED)

    @property
    def is_soft_evidence(self) -> bool:
        """True when this belief may only ever be soft evidence."""
        return self is PreferenceLifecycle.INFERRED_HYPOTHESIS


class BeliefStrength(str, Enum):
    """Discrete, ordinal strength of a preference belief.  NOT a probability.

    ``WEAK`` is the only strength a system inference may use by default; ``MODERATE``
    is allowed for an inference with a named trusted basis; ``STRONG`` is reserved for
    user-stated and user-confirmed beliefs.  :attr:`rank` gives the deterministic
    ordering weight; the values are names, not calibrated scores, and must never be
    presented as confidence percentages.
    """

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

    @property
    def rank(self) -> int:
        """Ordinal rank (1-3), used as the deterministic soft-signal weight."""
        return _BELIEF_STRENGTH_RANKS[self]


_BELIEF_STRENGTH_RANKS: dict[BeliefStrength, int] = {
    BeliefStrength.WEAK: 1,
    BeliefStrength.MODERATE: 2,
    BeliefStrength.STRONG: 3,
}


class KnowledgeStatus(str, Enum):
    """Estimate of whether the user understands one product attribute.

    The ladder is deliberately **not** a proof ladder: displaying an explanation moves
    the user to :attr:`EXPLANATION_PROVIDED` only.  Nothing except explicit evidence
    from the user may move a belief to :attr:`UNDERSTANDING_DEMONSTRATED`.
    """

    UNKNOWN = "unknown"
    NEEDS_EXPLANATION = "needs_explanation"
    EXPLANATION_PROVIDED = "explanation_provided"
    UNDERSTANDING_DEMONSTRATED = "understanding_demonstrated"


class BeliefOrigin(str, Enum):
    """Provenance category of a preference belief (audit, not authority)."""

    USER_STATED = "user_stated"
    USER_CONFIRMED = "user_confirmed"
    SYSTEM_INFERRED = "system_inferred"
    LEGACY_MEMORY = "legacy_memory"


class PreferenceDirection(str, Enum):
    """Optional direction of a feature preference ("lighter", "cheaper")."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    EXACT = "exact"


class ObservationKind(str, Enum):
    """What kind of interaction observation was recorded.

    Observations are evidence, never beliefs: a comparison observation must not be
    converted into a feature preference without separate supporting evidence.
    """

    PRODUCT_COMPARISON = "product_comparison"


class ConstraintAuthority(str, Enum):
    """Who authorized a hard constraint.  Only explicit user statements are hard.

    * :attr:`USER_EXPLICIT` -- stated by the user as a constraint for this task or for
      long-term use.  May define feasibility.
    * :attr:`USER_CONFIRMED` -- reserved for a future milestone.  A confirmed soft
      preference is still a ranking signal, not a filter.
    * :attr:`LEGACY_EXPLICIT` -- an explicit entry read from the accepted M9 memory.
      The M5-MVP compatibility adapter projects such entries as long-term *beliefs*,
      not as hard constraints, so the old memory system cannot change feasibility.
    """

    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED = "user_confirmed"
    LEGACY_EXPLICIT = "legacy_explicit"


#: The only constraint authority that may become a hard filter in M5-MVP.
HARD_CONSTRAINT_AUTHORITIES: frozenset[ConstraintAuthority] = frozenset(
    {ConstraintAuthority.USER_EXPLICIT}
)


class HypothesisBasis(str, Enum):
    """Named trusted source a preference hypothesis was derived from.

    The basis is provenance, not proof: every one of these values still produces an
    ``INFERRED_HYPOTHESIS``, never an explicit preference.
    """

    BROWSING_HISTORY = "browsing_history"
    INTERACTION_TRACE = "interaction_trace"
    CATALOGUE_EVIDENCE = "catalogue_evidence"
    PRIOR_PREFERENCE = "prior_preference"
    OTHER = "other"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def bounded_identifier(value: object, *, field: str, max_length: int) -> str:
    """Strip and validate an opaque identifier/token.

    Identifiers are exact strings: surrounding whitespace is removed, but an embedded
    newline or control character is refused rather than silently normalised, because
    that would change identity.  A value that is too long is refused, not truncated.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must not be blank")
    if len(stripped) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    if any(character in stripped for character in ("\x00", "\n", "\r", "\t")):
        raise ValueError(f"{field} must not contain control characters")
    return stripped


def bounded_text(value: object, *, field: str, max_length: int) -> str:
    """Collapse whitespace and truncate untrusted free text to ``max_length``.

    Used for raw user words kept for audit.  Unlike a preference value, a long raw
    sentence is bounded by truncation (the state does not need the whole paragraph);
    a preference value that is too long is refused by
    :func:`clean_preference_value` instead.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    collapsed = " ".join(value.split())
    return collapsed[:max_length]


def redacted_text(value: object, *, field: str, max_length: int) -> str:
    """Bound untrusted raw text and replace credential-looking content.

    Raw user text is retained only as bounded context.  If it looks like a credential
    (API key, card number, password assignment), the whole span is replaced with
    ``[redacted]``: the state is not a secret archive, and partial redaction of an
    arbitrary token is easy to get wrong.
    """
    bounded = bounded_text(value, field=field, max_length=max_length)
    if contains_secret_like_text(bounded):
        return SECRET_REDACTION
    return bounded


def clean_preference_value(value: object) -> str:
    """Normalise a preference value with the accepted M9 rules.

    Same observable rules as the accepted preference-memory schema: collapse
    whitespace, require non-blank, refuse values longer than ``MAX_VALUE_LENGTH`` and
    refuse credential-looking values.  Importing the accepted helpers is not possible
    (they are private), so the rules are restated here and kept deliberately identical.
    """
    if not isinstance(value, str):
        raise ValueError("preference value must be a string")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("preference value must not be blank")
    if len(collapsed) > MAX_VALUE_LENGTH:
        raise ValueError(
            f"preference value must be at most {MAX_VALUE_LENGTH} characters"
        )
    if contains_secret_like_text(collapsed):
        raise ValueError("preference value looks like a credential and must not be stored")
    return collapsed


def clean_attribute(value: object) -> str:
    """Normalise an attribute label (e.g. ``waterproofing``)."""
    if not isinstance(value, str):
        raise ValueError("attribute must be a string")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("attribute must not be blank")
    if len(collapsed) > MAX_ATTRIBUTE_LENGTH:
        raise ValueError(
            f"attribute must be at most {MAX_ATTRIBUTE_LENGTH} characters"
        )
    if contains_secret_like_text(collapsed):
        raise ValueError("attribute looks like a credential and must not be stored")
    return collapsed


def canonical_json(value: Any) -> str:
    """Return the canonical JSON encoding used for hashes and digests.

    Sorted keys, no insignificant whitespace, UTF-8 text preserved.  The same data
    must always produce the same bytes, otherwise event idempotency and replay digests
    would not be deterministic.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_id(prefix: str, *parts: object) -> str:
    """Return a stable opaque id derived from the given identity parts.

    Identity is content -- user, task, scope, kind, value, polarity -- not a counter or
    a clock, so rebuilding the same state produces the same ids.  The ``\\x1f``
    separator keeps neighbouring parts from concatenating ambiguously.
    """
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"
