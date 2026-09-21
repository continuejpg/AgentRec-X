"""Record and replay around the structured-model seam (Phase 3).

Why this exists
---------------
A live model call is the only part of this system that is neither free nor reproducible.  Phase 3
has to *measure real model behaviour* and still be able to re-run the measurement later without
paying for it again, so the model seam gained a recording layer between the policy and the
provider.

::

    LLMAgentPolicy
          |  ModelRequest
          v
    RecordingModelClient            <- this module
          |  canonical fingerprint
          +--> REPLAY  : look up, return the recorded ModelResponse, never touch the network
          +--> LIVE    : call the wrapped client, return its response, record nothing
          +--> RECORD  : call the wrapped client, write the response
          +--> RECORD_IF_MISSING : replay when present, otherwise call and write

Nothing else changes.  The recorded response is the *provider's* response, so replay reproduces
what the model said - it does not cache a policy decision, and the policy still parses, validates
and proposes exactly as it would on a live answer.

What a recording is not
-----------------------
It is not an answer key.  The fixture the evaluation plane grades against is the case
declaration; a recording is only the model's own text.  A replayed run that passes did so by
reproducing the model's decisions, not by being told the expected ones.

Where the recorder sits
-----------------------
At the model-client boundary, never inside a domain tool.  The tools, the ledger, the constraint
kernel, the validator and the completion guard are untouched by recording, so a replay cannot
change what the trusted runtime does with a response - only which response it receives.

Privacy
-------
A recording holds the prompt and the context the policy disclosed, because reproducing a
decision requires knowing what the model was asked.  It holds **no** credential, no
authorization header, and no provider chain-of-thought: the adapter reads
``reasoning_content`` never, and the repository never asks for it.  Recordings are evaluation
artifacts - they are written outside version control by default (see
``DEFAULT_RECORDINGS_DIR``) and are meant to be reviewed before anyone commits them.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .model_client import ModelCallError, ModelRequest, ModelResponse, StructuredModelClient

__all__ = [
    "DEFAULT_RECORDINGS_DIR",
    "recording_identity",
    "Fingerprint",
    "ModelSink",
    "ReplayMode",
    "RecordingFormatError",
    "RecordingModelClient",
    "RecordingStore",
    "RECORDING_FORMAT_VERSION",
    "fingerprint_request",
    "replay_client",
]

#: Version of the recording format.  Bumped when the fingerprint inputs or the entry shape
#: change, so an old recording is rejected as incompatible rather than mis-replayed.
RECORDING_FORMAT_VERSION = 1

#: Where recordings go by default.  **Outside version control on purpose**: a recording holds a
#: prompt and a model's verbatim answer, which is fine as an experiment artifact and wrong as an
#: unreviewed repository fixture.  A deployment that wants fixtures in-tree points this at a
#: reviewed directory instead.
DEFAULT_RECORDINGS_DIR = Path("/tmp/agentrecx-recordings")


class ReplayMode(str, Enum):
    """How a recording client treats the provider.

    The important guarantee is asymmetric: ``REPLAY`` never falls back to the network.  A
    silent fallback would make a "reproducible" run quietly depend on a live endpoint, which is
    exactly the property the mode exists to provide.
    """

    #: Call the provider every time.  Write nothing.
    LIVE = "live"
    #: Call the provider and write each response.
    RECORD = "record"
    #: Never call the provider; fail when a response was not recorded.
    REPLAY = "replay"
    #: Replay when recorded, otherwise call and write.
    RECORD_IF_MISSING = "record_if_missing"

    @property
    def calls_provider(self) -> bool:
        """True when this mode may reach the network."""
        return self is not ReplayMode.REPLAY

    @property
    def writes_recordings(self) -> bool:
        """True when this mode may write a recording."""
        return self in (ReplayMode.RECORD, ReplayMode.RECORD_IF_MISSING)


class RecordingFormatError(ModelCallError):
    """A recording could not be used: missing, unreadable, or from an incompatible version."""

    def __init__(self, message: str, *, code: str = "recording_invalid") -> None:
        super().__init__(message, code=code)


@dataclass(frozen=True)
class Fingerprint:
    """A stable identity for one model request, plus the canonical body it hashes.

    ``digest`` is what a lookup keys on.  ``canonical`` is kept so a test (or a person debugging
    a replay miss) can see exactly what was hashed instead of guessing at a dict ordering.
    """

    digest: str
    canonical: dict[str, Any]

    def short(self) -> str:
        """A short prefix, for a human-readable trace line."""
        return self.digest[:12]


def _canonical_json(value: Any) -> str:
    """Serialise to canonical JSON: sorted keys, no incidental whitespace, stable separators.

    The point is that two materially identical requests produce the same bytes regardless of how
    their dictionaries were built, and two materially different ones never collide.  ``default``
    covers the tuple-in-a-dict case, which JSON has no native form for.
    """
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default
    )


def _json_default(value: Any) -> Any:
    """Render an object JSON cannot represent natively, deterministically."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, (tuple, list)):
        return list(value)
    return str(value)


def fingerprint_request(
    request: ModelRequest,
    *,
    provider: str,
    model: str,
    prompt_version: int | None = None,
    parse_protocol_version: int | None = None,
) -> Fingerprint:
    """Compute the canonical fingerprint of one model request.

    Keyed on everything that can change the answer:

    * the **provider and model**, because two models answer the same prompt differently and a
      recording made against one must not be replayed against another;
    * the **prompt version**, so a prompt revision invalidates a recording rather than silently
      replaying an answer to a question that is no longer being asked;
    * the **parse-protocol version**, because a change to how answers are interpreted can change
      which response a run needs, so a trace recorded under the old protocol must not be replayed
      as though it were made under the new one;
    * the full **context payload** - observation, candidate references, available actions,
      constraint state, budget - because that is what the policy actually decides from;
    * the **action schema**, because an offered-menu change changes the decision;
    * the **correction**, because a retry after a parse failure is a different request.

    Deliberately *not* keyed on the user request alone.  A policy decision depends on runtime
    state, so two turns of the same conversation with different observations must not share a
    recording.
    """
    body = {
        "format": RECORDING_FORMAT_VERSION,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "parse_protocol_version": parse_protocol_version,
        "system_prompt_sha256": _sha256(request.system_prompt),
        "context_payload": request.context_payload,
        "action_schema": list(request.action_schema),
        "correction": request.correction,
    }
    canonical = _canonical_json(body)
    return Fingerprint(digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical=body)


def _sha256(text: str) -> str:
    """Hex SHA-256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RecordingStore:
    """An append-only JSONL recording file.

    JSONL rather than one JSON document because a live run appends incrementally: a crash
    halfway through a suite leaves every earlier exchange readable instead of an unparseable
    fragment.  The whole file is loaded into an index on first use, so a replay does not depend
    on the order entries were written.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._index: dict[str, dict[str, Any]] | None = None
        self._appended = 0

    @property
    def path(self) -> Path:
        """The recording file this store reads and writes."""
        return self._path

    @property
    def appended(self) -> int:
        """How many entries this process wrote (diagnostics only)."""
        return self._appended

    def _load(self) -> dict[str, dict[str, Any]]:
        """Read and index the file, tolerating a missing one as empty."""
        if self._index is not None:
            return self._index
        index: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            for line_number, line in enumerate(
                self._path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RecordingFormatError(
                        f"recording {self._path.name} line {line_number} is not JSON"
                    ) from exc
                if not isinstance(entry, dict):
                    raise RecordingFormatError(
                        f"recording {self._path.name} line {line_number} is not an object"
                    )
                version = entry.get("format_version")
                if version != RECORDING_FORMAT_VERSION:
                    # Refused rather than guessed: a recording from another format version may
                    # describe a different request shape, and replaying it would produce a
                    # plausible but wrong answer.
                    raise RecordingFormatError(
                        f"recording {self._path.name} is format {version!r}, expected "
                        f"{RECORDING_FORMAT_VERSION}"
                    )
                digest = entry.get("fingerprint")
                if isinstance(digest, str) and digest:
                    # A list, not a single entry.  A request may legitimately repeat within a run
                    # (an identical context makes an identical request), and keeping every
                    # response means an overwrite can never hide a collision: a replay consumes
                    # them in order, so N identical requests get the N answers they originally
                    # received.
                    index.setdefault(digest, []).append(entry)
        self._index = index
        return index

    def identity(self) -> dict[str, Any] | None:
        """The provider/model/prompt-version this recording was made against.

        Read from the file rather than restated by the caller.  A fingerprint includes the
        provider and model, so a replay must use the *recording's* identity, not the identity of
        whatever environment happens to be configured now - otherwise every lookup misses and the
        failure looks like a corrupt recording rather than a mismatched argument.
        """
        index = self._load()
        for entries in index.values():
            entry = entries[0]
            return {
                "provider": entry.get("provider"),
                "model": entry.get("model"),
                "prompt_version": entry.get("prompt_version"),
                "parse_protocol_version": entry.get("parse_protocol_version"),
                "format_version": entry.get("format_version"),
            }
        return None

    def __len__(self) -> int:
        """How many exchanges the recording holds (responses, not distinct requests)."""
        return sum(len(entries) for entries in self._load().values())

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and digest in self._load()

    def entries_for(self, digest: str) -> list[dict[str, Any]]:
        """Every recorded exchange for a fingerprint, in the order it was written."""
        return list(self._load().get(digest, ()))

    def response_count(self, digest: str) -> int:
        """How many responses were recorded for a fingerprint."""
        return len(self._load().get(digest, ()))

    def response_for(self, digest: str, *, index: int = 0) -> ModelResponse:
        """Rebuild one recorded :class:`ModelResponse`, or raise a clear error.

        ``index`` selects among repeated recordings of the same request.  The latency in a
        replayed response is the **recorded** latency, not a fresh measurement: reporting a
        replay as though it had just taken that long would be a fabrication, so the value is
        carried as the record of what the original call cost and the replay client marks the
        response as replayed so a reader can tell the difference.
        """
        entries = self._load().get(digest, ())
        if not entries:
            raise RecordingFormatError(
                f"no recording for request {digest[:12]} in {self._path.name}; "
                "run with --record first (replay never falls back to the network)",
                code="recording_missing",
            )
        if index >= len(entries):
            # More replays than recordings for this request.  Refusing is the honest outcome: a
            # silent wrap would hand back an answer that belongs to an earlier occurrence.
            raise RecordingFormatError(
                f"recording {digest[:12]} holds {len(entries)} response(s); replay asked for "
                f"occurrence {index}",
                code="recording_exhausted",
            )
        response = entries[index].get("response")
        if not isinstance(response, dict):
            raise RecordingFormatError(
                f"recording {digest[:12]} has no response object", code="recording_invalid"
            )
        return ModelResponse.model_validate(_response_fields(response))

    def write(
        self,
        *,
        fingerprint: Fingerprint,
        request: ModelRequest,
        response: ModelResponse,
        provider: str,
        model: str,
        prompt_version: int | None,
        parse_protocol_version: int | None = None,
    ) -> None:
        """Append one exchange.  Creates the parent directory if needed.

        ``timestamp`` is informational only and is deliberately excluded from the fingerprint:
        two runs of the same request must share a recording even though their clocks differ.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "format_version": RECORDING_FORMAT_VERSION,
            "fingerprint": fingerprint.digest,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "parse_protocol_version": parse_protocol_version,
            "timestamp": time.time(),
            "request": {
                # The system prompt is stored verbatim: it is a repository artifact describing
                # this policy, carries no credential and no user data, and reproducing a
                # decision needs to know which contract was in force.  Its hash is also in the
                # fingerprint, so a modified prompt cannot replay an old answer.
                "system_prompt": request.system_prompt,
                "context_payload": request.context_payload,
                "action_schema": list(request.action_schema),
                "correction": request.correction,
            },
            "response": {
                "text": response.text,
                "model_id": response.model_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "request_id": response.request_id,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "estimated_cost": response.estimated_cost,
            },
        }
        # Load the index *before* appending, and append to that same object.  Writing first and
        # then calling ``_load`` looked equivalent but was not: on a store whose file did not
        # exist yet, the write created it and the following ``_load`` read it back, so the entry
        # was counted twice and ``len(store)`` disagreed with the file.
        index = self._load()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(entry) + "\n")
        index.setdefault(fingerprint.digest, []).append(entry)
        self._appended += 1


def recording_identity(path: Path | str) -> dict[str, Any]:
    """Read a recording's provider/model identity, or raise a clear error.

    Used to point a replay at the identity the recording was made against.  An empty recording
    has no identity and is reported as such rather than defaulted, because guessing would produce
    silent misses.
    """
    identity = RecordingStore(path).identity()
    if identity is None:
        raise RecordingFormatError(f"the recording at {path} has no entries", code="recording_empty")
    return identity


def _response_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the response fields read back from a recording.

    A whitelist rather than ``**response`` so a future field cannot enter a ``ModelResponse``
    through a hand-edited recording, and so an unknown key is dropped instead of raising on a
    recording written by a newer format revision that kept the same version.
    """
    allowed = (
        "text",
        "model_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "request_id",
        "finish_reason",
        "latency_ms",
        "estimated_cost",
    )
    return {key: response[key] for key in allowed if key in response}


class ModelSink:
    """Where a live run's per-call observations go, for the experiment summary.

    Deliberately tiny: the evaluation needs counts and latency, and this collects them without
    the model client having to know what an experiment report looks like.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_call(
        self,
        *,
        fingerprint: Fingerprint,
        response: ModelResponse,
        mode: ReplayMode,
        replayed: bool,
    ) -> None:
        """Append one model call's non-sensitive diagnostics."""
        self.calls.append(
            {
                "fingerprint": fingerprint.short(),
                "mode": mode.value,
                "replayed": replayed,
                "model_id": response.model_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
                "estimated_cost": response.estimated_cost,
                "empty": response.is_empty,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate the calls: counts, tokens, latency.  Unknowns stay unknown.

        Latency and tokens are summed over **every** call, including replayed ones, because a
        replayed call carries the measurement taken when it was recorded.  Excluding them made a
        replay report no latency at all, which is the opposite of useful: the point of recording
        is to be able to compare efficiency offline.  The ``live_calls``/``replayed_calls``
        counts are what tell a reader which kind of measurement they are looking at.
        """
        live = [c for c in self.calls if not c["replayed"]]
        replayed = [c for c in self.calls if c["replayed"]]
        latencies = [c["latency_ms"] for c in self.calls if isinstance(c["latency_ms"], (int, float))]
        costs = [c["estimated_cost"] for c in self.calls if isinstance(c["estimated_cost"], (int, float))]
        return {
            "model_calls": len(self.calls),
            "live_calls": len(live),
            "replayed_calls": len(replayed),
            "empty_responses": sum(1 for c in self.calls if c["empty"]),
            "input_tokens": _sum_optional(c["input_tokens"] for c in self.calls),
            "output_tokens": _sum_optional(c["output_tokens"] for c in self.calls),
            "total_tokens": _sum_optional(c["total_tokens"] for c in self.calls),
            "model_latency_ms_total": round(sum(latencies), 3) if latencies else None,
            "model_latency_ms_mean": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
            "model_latency_ms_max": round(max(latencies), 3) if latencies else None,
            "estimated_cost_total": round(sum(costs), 8) if costs else None,
        }


def _sum_optional(values: Any) -> int | None:
    """Sum the non-``None`` values, or ``None`` when none were reported.

    Never treats an unknown as a zero: a total that silently omitted an unreported call would
    read as a measured number.
    """
    present = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return sum(present) if present else None


class RecordingModelClient:
    """A :class:`StructuredModelClient` that records, replays, or passes through.

    The wrapped client is called only when the mode permits it, so ``REPLAY`` is a hard offline
    guarantee rather than a convention.  The policy above this layer is unchanged and unaware of
    the mode, which is what keeps a replayed run a genuine run of the same policy.
    """

    def __init__(
        self,
        wrapped: StructuredModelClient,
        *,
        store: RecordingStore,
        mode: ReplayMode = ReplayMode.LIVE,
        provider: str = "unknown",
        model: str | None = None,
        prompt_version: int | None = None,
        parse_protocol_version: int | None = None,
        sink: ModelSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wrapped = wrapped
        self._store = store
        self._mode = mode
        self._provider = provider
        self._model = model or getattr(wrapped, "model", None) or "unknown"
        self._prompt_version = prompt_version
        self._parse_protocol_version = parse_protocol_version
        self._sink = sink
        self._clock = clock
        self._replayed = 0
        self._live = 0
        #: How many times each fingerprint has been replayed, so repeated identical requests
        #: consume their own recorded occurrences in the order they were written.
        self._replay_counts: dict[str, int] = {}

    # -- metadata ---------------------------------------------------------- #

    @property
    def mode(self) -> ReplayMode:
        """The mode this client was constructed with."""
        return self._mode

    @property
    def provider(self) -> str:
        """The provider profile name this client records against."""
        return self._provider

    @property
    def model(self) -> str:
        """The endpoint model identifier this client records against."""
        return self._model

    @property
    def store(self) -> RecordingStore:
        """The recording store in use."""
        return self._store

    @property
    def replayed_calls(self) -> int:
        """How many calls were answered from a recording."""
        return self._replayed

    @property
    def live_calls(self) -> int:
        """How many calls reached the wrapped client."""
        return self._live

    def fingerprint(self, request: ModelRequest) -> Fingerprint:
        """The fingerprint this client would look up for ``request``."""
        return fingerprint_request(
            request,
            provider=self._provider,
            model=self._model,
            prompt_version=self._prompt_version,
            parse_protocol_version=self._parse_protocol_version,
        )

    # -- the seam ---------------------------------------------------------- #

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer ``request`` according to the mode."""
        fingerprint = self.fingerprint(request)

        if self._mode is ReplayMode.REPLAY:
            response = self._replay(fingerprint.digest)
            self._observe(fingerprint, response, replayed=True)
            return response

        if self._mode is ReplayMode.RECORD_IF_MISSING and fingerprint.digest in self._store:
            if self._replay_counts.get(fingerprint.digest, 0) < self._store.response_count(
                fingerprint.digest
            ):
                response = self._replay(fingerprint.digest)
                self._observe(fingerprint, response, replayed=True)
                return response
            # Recorded occurrences are used up; this occurrence must come from the provider.
        

        response = self._call_wrapped(request)
        self._live += 1
        if self._mode.writes_recordings:
            self._store.write(
                fingerprint=fingerprint,
                request=request,
                response=response,
                provider=self._provider,
                model=self._model,
                prompt_version=self._prompt_version,
                parse_protocol_version=self._parse_protocol_version,
            )
        self._observe(fingerprint, response, replayed=False)
        return response

    def _replay(self, digest: str) -> ModelResponse:
        """Return the next recorded response for a fingerprint.

        Occurrences are consumed in order, so a run that made the same request twice replays the
        two answers it originally received.  Exhausting them raises rather than wrapping: a
        replay that silently reused an earlier answer would report a decision the model never
        made for this occurrence.
        """
        index = self._replay_counts.get(digest, 0)
        response = self._store.response_for(digest, index=index)
        self._replay_counts[digest] = index + 1
        self._replayed += 1
        return response

    def _call_wrapped(self, request: ModelRequest) -> ModelResponse:
        """Call the wrapped client, normalising any unexpected failure into our error type.

        A wrapped client is required to raise :class:`ModelCallError`; a provider SDK leaking its
        own exception type past this point would bypass the policy's bounded-failure path.
        """
        try:
            return self._wrapped.complete(request)
        except ModelCallError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised, never swallowed
            raise ModelCallError(
                f"the wrapped model client raised {type(exc).__name__}", code="provider_error"
            ) from exc

    def _observe(self, fingerprint: Fingerprint, response: ModelResponse, *, replayed: bool) -> None:
        """Hand one call to the sink, when a sink was configured."""
        if self._sink is not None:
            self._sink.record_call(
                fingerprint=fingerprint, response=response, mode=self._mode, replayed=replayed
            )


def replay_client(
    *,
    wrapped: StructuredModelClient | None,
    recording: Path | str,
    provider: str,
    model: str | None = None,
    prompt_version: int | None = None,
    parse_protocol_version: int | None = None,
    sink: ModelSink | None = None,
) -> RecordingModelClient:
    """Build a client for a replay-or-record experiment.

    When a recording exists the mode is ``REPLAY``, so the provider is never contacted - and a
    missing entry raises rather than silently calling out.  When it does not exist the mode is
    ``RECORD`` and ``wrapped`` must be supplied.  This is the shape the experiment commands use:
    run once to record, then re-run offline to reproduce.
    """
    store = RecordingStore(recording)
    if store.path.exists():
        return RecordingModelClient(
            wrapped or _UnreachableClient(),
            store=store,
            mode=ReplayMode.REPLAY,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            parse_protocol_version=parse_protocol_version,
            sink=sink,
        )
    if wrapped is None:
        raise RecordingFormatError(
            f"no recording at {store.path} and no provider client to record with",
            code="recording_missing",
        )
    return RecordingModelClient(
        wrapped,
        store=store,
        mode=ReplayMode.RECORD,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        parse_protocol_version=parse_protocol_version,
        sink=sink,
    )


class _UnreachableClient:
    """A stand-in wrapped client that refuses to be called.

    Replay never calls it, so reaching it means the mode semantics broke.  Raising a clear error
    is better than a ``None`` type error somewhere deeper.
    """

    def complete(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover - guard
        raise ModelCallError(
            "an offline replay client was asked to call a provider", code="replay_violation"
        )
