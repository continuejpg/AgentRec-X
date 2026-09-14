"""Server-owned demo profiles: the source of a demo session's trusted history.

Three states must never be conflated (Milestone 11 design note)::

    trusted behavioural history   application-owned, chronological parent_asin sequence
                                  consumed only by RecommendationTool / SASRec
    preference memory             explicit conversational statements (Milestone 9)
    browser transcript            purely visual history in the page (this milestone)

A :class:`DemoProfile` supplies the **first** of those.  It is chosen by the
application *before* the conversation starts, it is read-only for the whole session,
and no chat message can extend, reorder or replace it.

Profiles come from the accepted processed sequences artifact using the same
deterministic rule the accepted Milestone 7C integration uses:

1. walk the stored user records in order (ascending ``user_int_id``, so the result is
   independent of dict/hash ordering);
2. take the first records that satisfy the minimum length filter;
3. supply ``parent_asins[:-2] + [parent_asins[-2]]`` -- the training prefix plus the
   validation target, with the final leave-one-out **test target excluded**, so no
   future evaluation target is ever shown to the demo or fed to the model.

Selection never inspects scores or candidates, so a profile cannot be tuned toward a
preferred recommendation output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recommendation import config as project_config

__all__ = [
    "DEFAULT_DEMO_PROFILE_COUNT",
    "DEFAULT_PROFILE_PREFIX",
    "MIN_PROFILE_HISTORY_LENGTH",
    "DemoProfile",
    "build_demo_profiles",
    "demo_profiles_from_artifact",
    "sequences_artifact_path",
]

#: How many demo profiles are offered when the caller does not say otherwise.  Small and
#: fixed: this is a local demo, not a user directory.
DEFAULT_DEMO_PROFILE_COUNT = 3

#: Prefix of the generated, application-owned profile identifiers.
DEFAULT_PROFILE_PREFIX = "demo-user"

#: Minimum supplied history length.  Matches the accepted integration convention so a
#: profile always has enough sequential context for SASRec to produce candidates.
MIN_PROFILE_HISTORY_LENGTH = 3

#: Human-readable labels.  They describe the *arity* of the demo slot, not the real
#: user behind it: the demo never claims to know who the source user is.
_DISPLAY_NAMES: tuple[str, ...] = (
    "Demo shopper 1",
    "Demo shopper 2",
    "Demo shopper 3",
    "Demo shopper 4",
    "Demo shopper 5",
    "Demo shopper 6",
)


@dataclass(frozen=True)
class DemoProfile:
    """An application-owned demo profile and its trusted recommendation history.

    ``profile_id`` is a safe, generated identifier.  ``source_user_int_id`` is kept so a
    diagnostic can prove *which* accepted record a profile came from, but it is **not**
    part of any public response schema (see the Milestone 11 serialization boundary).
    """

    profile_id: str
    display_name: str
    trusted_user_history: tuple[str, ...]
    source_user_int_id: int
    source_length: int

    @property
    def history_length(self) -> int:
        """Number of supplied history items."""
        return len(self.trusted_user_history)

    @property
    def history_distinct(self) -> int:
        """Distinct supplied items (duplicates are preserved in the history itself)."""
        return len(set(self.trusted_user_history))


def sequences_artifact_path() -> Path:
    """Path to the accepted processed sequences artifact."""
    return project_config.PROCESSED_DIR / f"{project_config.DEFAULT_CATEGORY}_sequences.json"


def demo_profiles_from_artifact(
    *,
    count: int = DEFAULT_DEMO_PROFILE_COUNT,
    min_length: int = MIN_PROFILE_HISTORY_LENGTH,
    prefix: str = DEFAULT_PROFILE_PREFIX,
    sequences_file: Path | None = None,
) -> tuple[DemoProfile, ...]:
    """Build the fixed demo profile set from the accepted sequences artifact.

    Raises
    ------
    FileNotFoundError
        The accepted sequences artifact is absent.
    LookupError
        Fewer eligible records exist than ``count``.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    path = sequences_file or sequences_artifact_path()
    if not path.exists():
        raise FileNotFoundError(f"accepted sequences artifact not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[DemoProfile] = []
    for record in payload["sequences"]:
        asins = list(record["parent_asins"])
        if len(asins) < 3:
            continue
        # Training prefix + validation target; the leave-one-out test target is excluded.
        history = tuple(asins[:-2] + [asins[-2]])
        if len(history) <= min_length:
            continue
        if not any(history):
            continue
        index = len(profiles) + 1
        profiles.append(
            DemoProfile(
                profile_id=f"{prefix}-{index}",
                display_name=(
                    _DISPLAY_NAMES[index - 1]
                    if index - 1 < len(_DISPLAY_NAMES)
                    else f"Demo shopper {index}"
                ),
                trusted_user_history=history,
                source_user_int_id=int(record["user_int_id"]),
                source_length=len(asins),
            )
        )
        if len(profiles) == count:
            break

    if len(profiles) < count:
        raise LookupError(
            f"the accepted sequences artifact yielded {len(profiles)} eligible demo "
            f"profile(s), but {count} were requested"
        )
    return tuple(profiles)


def build_demo_profiles(
    *,
    count: int = DEFAULT_DEMO_PROFILE_COUNT,
    min_length: int = MIN_PROFILE_HISTORY_LENGTH,
    sequences_file: Path | None = None,
    profiles: tuple[DemoProfile, ...] | None = None,
) -> dict[str, DemoProfile]:
    """Return the demo profile registry keyed by ``profile_id``.

    Supplying ``profiles`` bypasses artifact loading, which is what the offline tests
    use so no 300 MB artifact is required.
    """
    if profiles is None:
        profiles = demo_profiles_from_artifact(
            count=count, min_length=min_length, sequences_file=sequences_file
        )
    registry = {profile.profile_id: profile for profile in profiles}
    if len(registry) != len(profiles):
        raise ValueError("demo profile ids must be unique")
    if not registry:
        raise ValueError("at least one demo profile is required")
    return registry
