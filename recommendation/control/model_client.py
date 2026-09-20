"""The structured-model seam for a model-driven policy (Phase 1).

This module is the **only** place in the control plane that knows a model exists.  It defines
the narrow interface an :class:`~recommendation.control.model_policy.LLMAgentPolicy` needs, and
nothing else in AgentRec-X imports a provider SDK, reads an API key, or opens a socket.

Why a seam rather than a direct call
------------------------------------
Three requirements force it, and all three are about testability rather than elegance:

* **The test suite must not need a network or a key.**  Every policy test injects a scripted
  client, so the suite is deterministic and offline.  This repository's dependency policy is
  explicit that no provider SDK is imported by the core (``requirements.txt`` records the same
  reasoning for the accepted decision seam), and that stays true.
* **No provider lock-in.**  A real provider is a small adapter behind
  :class:`StructuredModelClient`; the control plane, the validator and the executor never learn
  which one is in use.
* **The model is untrusted input.**  A client returns *text*.  Turning that text into an action
  is the policy's job, and the policy treats it exactly as it would treat any other untrusted
  payload: parse, then validate.

The interface is deliberately synchronous, matching the accepted synchronous loop.

What a client may not do
------------------------
A client has no access to the catalogue, the ledger, behavioural history, preference memory or
the executor.  It receives a prompt and returns text.  It cannot express an action, and it
cannot cause one to run: only the policy turns its output into an :class:`ActionProposal`, and
only the validator authorises that proposal.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MODEL_SEAM_VERSION",
    "ModelCallError",
    "ModelRequest",
    "ModelResponse",
    "ScriptedModelClient",
    "StructuredModelClient",
]

#: Version of the model seam.
MODEL_SEAM_VERSION = 1


class ModelCallError(RuntimeError):
    """A model client could not produce a response.

    Raised for a provider exception, a timeout, an empty response or any other transport-level
    failure.  It is distinct from a *parse* failure: transport failures are normalised by the
    policy into a bounded failure path, while a parse failure means the model answered and the
    answer was unusable.
    """

    #: Stable, payload-free code for the trajectory, e.g. ``provider_error`` or ``timeout``.
    code: str = "model_error"

    def __init__(self, message: str, *, code: str = "model_error") -> None:
        super().__init__(message)
        self.code = code


class ModelRequest(BaseModel):
    """One structured-model call.

    Carries the instruction and the bounded context the policy decided to disclose.  It has no
    field for the catalogue, the ledger, behavioural history or a memory store, so a client
    cannot be handed any of them by accident - the policy's projection is the whole payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = MODEL_SEAM_VERSION
    #: The policy contract: choose one action, use only the offered actions, invent nothing.
    system_prompt: str
    #: The bounded, structured view of the current control state.
    context_payload: dict[str, Any] = Field(default_factory=dict)
    #: JSON-schema-ish description of each offered action and its arguments.
    action_schema: tuple[dict[str, Any], ...] = ()
    #: Optional instruction suffix for retries, carrying the parse error from the attempt before.
    correction: str | None = Field(default=None, max_length=400)

    def as_messages(self) -> list[dict[str, str]]:
        """Render the request as a two-message conversation.

        A provider adapter maps this onto its own message type; keeping the shape local is what
        keeps a provider SDK out of the control plane.
        """
        import json

        body = {
            "context": self.context_payload,
            "available_actions": list(self.action_schema),
        }
        content = json.dumps(body, indent=2, sort_keys=True, default=str)
        if self.correction:
            content = f"{content}\n\nYour previous answer was rejected: {self.correction}\n" "Answer again with one valid action object."
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]


class ModelResponse(BaseModel):
    """A model's raw answer.  Untrusted text, plus non-authoritative diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The answer text.  Expected to contain one JSON action object; not assumed to.
    text: str = ""
    #: Which model answered, for observability.  Never a control signal.
    model_id: str | None = Field(default=None, max_length=120)
    #: Optional token accounting, when a provider reports it.  Diagnostic only.
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @property
    def is_empty(self) -> bool:
        """True when the model returned nothing usable."""
        return not self.text.strip()


@runtime_checkable
class StructuredModelClient(Protocol):
    """The one method the policy requires from a model.

    ::

        complete(request: ModelRequest) -> ModelResponse

    Implementations must raise :class:`ModelCallError` for a transport failure rather than
    returning a sentinel, so the policy can distinguish "the model could not answer" from "the
    model answered badly".
    """

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the model's answer for one request."""
        ...


class ScriptedModelClient:
    """A deterministic offline client: the test and evaluation double.

    It answers with a caller-supplied script, and it records every request it received - which
    is how a test proves *what the policy disclosed to the model* as well as what it chose.

    Parameters
    ----------
    script:
        Answers in order.  An answer may be a ``str`` (raw text, so a test can supply malformed
        output), a :class:`ModelResponse`, a mapping (serialised to JSON), or an ``Exception``
        instance, which is raised to simulate a provider failure.
    repeat_last:
        When the script runs out, repeat the final answer instead of raising.  A malformed
        answer combined with ``repeat_last=True`` is exactly the "model never answers properly"
        case the bounded-failure test needs.
    """

    def __init__(self, script: list[Any], *, repeat_last: bool = False) -> None:
        if not script:
            raise ValueError("a scripted client needs at least one answer")
        self._script = list(script)
        self._repeat_last = bool(repeat_last)
        self.requests: list[ModelRequest] = []

    @property
    def call_count(self) -> int:
        """How many times this client was asked for an answer."""
        return len(self.requests)

    @property
    def last_request(self) -> ModelRequest | None:
        """The most recent request, or ``None`` before the first call."""
        return self.requests[-1] if self.requests else None

    def _answer_for(self, index: int) -> Any:
        if index < len(self._script):
            return self._script[index]
        if self._repeat_last:
            return self._script[-1]
        raise ModelCallError(
            "the scripted model has no further answers", code="script_exhausted"
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the next scripted answer."""
        import json

        index = len(self.requests)
        self.requests.append(request)
        entry = self._answer_for(index)

        if isinstance(entry, Exception):
            raise ModelCallError(
                f"the scripted model raised {type(entry).__name__}",
                code="provider_error",
            )
        if isinstance(entry, ModelResponse):
            return entry
        if isinstance(entry, str):
            return ModelResponse(text=entry, model_id="scripted")
        if isinstance(entry, dict):
            return ModelResponse(
                text=json.dumps(entry), model_id="scripted"
            )
        raise ModelCallError(
            f"a scripted answer must be str, ModelResponse, mapping or Exception, got "
            f"{type(entry).__name__}",
            code="bad_script_entry",
        )

    # -- convenience ------------------------------------------------------- #

    def actions(self) -> tuple[str | None, ...]:
        """The action named by each scripted answer, for compact assertions."""
        import json

        names: list[str | None] = []
        for entry in self._script:
            if isinstance(entry, ModelResponse):
                payload: Any = entry.text
            elif isinstance(entry, dict):
                payload = entry
            else:
                payload = None
            if isinstance(payload, dict):
                names.append(payload.get("action"))
            elif isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except Exception:  # noqa: BLE001 - a malformed answer has no action
                    names.append(None)
                else:
                    names.append(
                        parsed.get("action") if isinstance(parsed, dict) else None
                    )
            else:
                names.append(None)
        return tuple(names)
