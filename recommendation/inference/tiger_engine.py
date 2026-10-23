"""The AgentRec-X side of a **live** TIGER serving engine.

This is the L2 client of the four-layer boundary: it owns the trusted identity map and the
lifecycle of a long-lived backend worker, and it owns nothing about the model.  It reuses the
accepted generative path unchanged - the worker calls ``tiger_public.tiger.load_generator``,
``tiger_public.retrieve.build_history_input`` / ``TigerCaseScorer`` / ``retrieve`` and
``tiger_public.trie.CatalogueTrie`` - so no generator, RQ-VAE or trie is reimplemented here.

Contract
--------
``TigerSourceTool`` (``recommendation/control/tiger_source.py``) accepts any engine exposing::

    engine.parent_asin_to_item_id(parent_asin) -> int
    engine.recommend(history_item_ids, k=...)  -> result

where ``result`` carries ``retrieval_mode``, ``recommendations`` (each with ``item_id``,
``parent_asin``, ``rank``, ``score``, ``semantic_id``), ``checkpoint_sha256``,
``sid_artifact_sha256`` and ``resolver``.  That is exactly what :class:`TigerServingEngine`
returns.

Boundary rules kept here
------------------------
* **No canonical identity crosses.** Requests carry integer item ids only; the response carries
  integer item ids plus model outputs.  ``parent_asin`` is attached *here*, from the trusted
  map, and is never sent to the worker.
* **No target is visible.** The engine has no notion of a test target and never asks for one.
* **Nothing is invented.** An item id the trusted map cannot resolve, a mode the backend did not
  actually run, a malformed frame or a dead worker all raise; no result is shortened,
  substituted or relabelled (see :class:`TigerContractViolation`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "APPROXIMATE",
    "CERTIFIED",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_HANDOFF_DIR",
    "DEFAULT_SID_DIR",
    "TigerAssetsMissing",
    "TigerContractViolation",
    "TigerEngineError",
    "TigerEngineNotReady",
    "TigerRecommendation",
    "TigerServingEngine",
    "TigerServingResult",
    "TigerWorkerConfig",
    "TigerWorkerError",
    "TigerWorkerTimeout",
    "set_shared_engine",
    "shared_engine",
]

CERTIFIED = "CERTIFIED"
APPROXIMATE = "APPROXIMATE"

REPO = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO / "backends" / "tiger_public"
DEFAULT_BACKEND_SRC = BACKEND_ROOT / "src"
DEFAULT_BACKEND_PYTHON = BACKEND_ROOT / ".venv" / "bin" / "python"
DEFAULT_CHECKPOINT_DIR = REPO / "runs" / "tiger_public_2026" / "checkpoint"
DEFAULT_SID_DIR = REPO / "runs" / "tiger_public_2026" / "sid"
DEFAULT_HANDOFF_DIR = REPO / "runs" / "tiger_backend_handoff_prod"

CONTRACT = "agentrecx.tiger.serve.v1"


# --------------------------------------------------------------------------- #
# Errors - every failure mode is explicit, none is a silent fallback
# --------------------------------------------------------------------------- #


class TigerEngineError(RuntimeError):
    """Base class for every serving-engine failure."""


class TigerAssetsMissing(TigerEngineError):
    """The accepted checkpoint / sid / handoff artifacts are not where they must be."""


class TigerWorkerError(TigerEngineError):
    """The backend worker failed, refused a request, or violated the frame contract."""

    def __init__(self, message: str, *, code: str = "worker_error", stderr: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.stderr = stderr


class TigerWorkerTimeout(TigerWorkerError):
    """The worker did not answer within the configured budget."""


class TigerContractViolation(TigerEngineError):
    """The worker answered, but the answer cannot be trusted as a TIGER result."""


class TigerEngineNotReady(TigerEngineError):
    """A request arrived before the long-lived worker finished loading.

    A serving process must not block an HTTP request for the minutes a CPU cold start takes, so
    such a deployment constructs the engine with ``blocking_start=False`` and lets
    :meth:`TigerServingEngine.prewarm` load in the background.  Until then this is raised, and
    :meth:`TigerServingEngine.health` reports ``loading``.
    """


# --------------------------------------------------------------------------- #
# Configuration and identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TigerWorkerConfig:
    """Everything the worker needs, and nothing about the model itself."""

    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    sid_dir: Path = DEFAULT_SID_DIR
    catalogue_dir: Path = DEFAULT_HANDOFF_DIR
    exposure_path: Path | None = None
    generator_layout_path: Path | None = None
    backend_src: Path = DEFAULT_BACKEND_SRC
    python_executable: Path = DEFAULT_BACKEND_PYTHON
    device: str = "cpu"
    k_max: int = 20
    max_hist_items: int = 20
    beam: int = 512
    prefix_budget: int = 200_000
    batch_prefixes: int = 256

    @property
    def resolved_exposure(self) -> Path:
        return Path(self.exposure_path or (self.catalogue_dir / "train_exposure.jsonl"))

    @property
    def resolved_layout(self) -> Path:
        return Path(self.generator_layout_path or (self.checkpoint_dir / "generator_layout.json"))

    def required_paths(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("checkpoint weights", Path(self.checkpoint_dir) / "tiger.pt"),
            ("checkpoint metadata", Path(self.checkpoint_dir) / "tiger.json"),
            ("generator layout", self.resolved_layout),
            ("sid assignment", Path(self.sid_dir) / "semantic_ids.json"),
            ("sid token layout", Path(self.sid_dir) / "layout.json"),
            ("handoff catalogue", Path(self.catalogue_dir) / "catalogue.json"),
            ("train exposure", self.resolved_exposure),
            ("backend sources", Path(self.backend_src)),
            ("backend interpreter", Path(self.python_executable)),
        )

    def missing(self) -> list[str]:
        return [f"{label} ({path})" for label, path in self.required_paths() if not path.exists()]

    def command(self) -> list[str]:
        return [
            str(self.python_executable),
            "-m",
            "tiger_public.serve_worker",
            "--checkpoint-dir", str(self.checkpoint_dir),
            "--sid", str(self.sid_dir),
            "--catalogue", str(self.catalogue_dir),
            "--exposure", str(self.resolved_exposure),
            "--generator-layout", str(self.resolved_layout),
            "--device", str(self.device),
            "--k-max", str(int(self.k_max)),
            "--max-hist-items", str(int(self.max_hist_items)),
            "--beam", str(int(self.beam)),
            "--prefix-budget", str(int(self.prefix_budget)),
            "--batch-prefixes", str(int(self.batch_prefixes)),
        ]


class IdentityResolver(Protocol):
    """The trusted identity map: exactly the two directions the source seam needs."""

    def parent_asin_to_item_id(self, parent_asin: str) -> int: ...

    def item_id_to_parent_asin(self, item_id: int) -> str: ...


class _MappingIdentity:
    """Adapter for a plain ``{parent_asin: item_id}`` mapping (tests, small fixtures)."""

    def __init__(self, mapping: Mapping[str, int]) -> None:
        self._forward = {str(k): int(v) for k, v in mapping.items()}
        self._reverse = {value: key for key, value in self._forward.items()}
        if len(self._reverse) != len(self._forward):
            raise TigerEngineError("the identity mapping is not injective")

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        try:
            return self._forward[str(parent_asin)]
        except KeyError as exc:
            raise KeyError(f"unknown parent_asin {parent_asin!r}") from exc

    def item_id_to_parent_asin(self, item_id: int) -> str:
        try:
            return self._reverse[int(item_id)]
        except KeyError as exc:
            raise KeyError(f"unknown item_id {item_id!r}") from exc


# --------------------------------------------------------------------------- #
# Result types (the shape TigerSourceTool consumes)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TigerRecommendation:
    """One TIGER candidate, with identity re-attached in the trusted layer."""

    item_id: int
    parent_asin: str
    rank: int
    score: float
    semantic_id: tuple[int, ...]
    token_path: tuple[int, ...] = ()
    resolver: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "parent_asin": self.parent_asin,
            "rank": self.rank,
            "score": self.score,
            "semantic_id": list(self.semantic_id),
            "token_path": list(self.token_path),
        }


@dataclass(frozen=True)
class TigerServingResult:
    """What ``recommend`` returns: the source's own view, plus provenance."""

    retrieval_mode: str
    recommendations: tuple[TigerRecommendation, ...]
    checkpoint_sha256: str
    sid_artifact_sha256: str
    resolver: Mapping[str, Any]
    k: int = 0
    required_frontier: int = 0
    history_length: int = 0
    wall_seconds: float = 0.0
    summary: Mapping[str, Any] = field(default_factory=dict)
    requested_mode: str = CERTIFIED

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(item.item_id for item in self.recommendations)

    @property
    def parent_asins(self) -> tuple[str, ...]:
        return tuple(item.parent_asin for item in self.recommendations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "requested_mode": self.requested_mode,
            "k": self.k,
            "required_frontier": self.required_frontier,
            "history_length": self.history_length,
            "wall_seconds": self.wall_seconds,
            "checkpoint_sha256": self.checkpoint_sha256,
            "sid_artifact_sha256": self.sid_artifact_sha256,
            "resolver": dict(self.resolver),
            "recommendations": [item.as_dict() for item in self.recommendations],
            "summary": dict(self.summary),
        }


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #


class Transport(Protocol):
    """One bidirectional frame channel to a worker."""

    def start(self, timeout: float) -> dict[str, Any]: ...

    def request(self, frame: dict[str, Any], timeout: float) -> dict[str, Any]: ...

    def close(self) -> None: ...

    def stderr_tail(self) -> str: ...


class SubprocessTransport:
    """A real worker process: JSONL on its stdin/stdout, diagnostics on its stderr."""

    def __init__(self, config: TigerWorkerConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._frames: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, timeout: float) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self._config.backend_src), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self._process = subprocess.Popen(
                self._config.command(),
                cwd=str(REPO),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise TigerAssetsMissing(
                f"could not start the TIGER worker with {self._config.python_executable}: {exc}"
            ) from exc
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_reader.start()
        frame = self._next_frame(timeout, what="the worker handshake")
        if frame.get("op") == "error":
            raise TigerWorkerError(
                f"the TIGER worker refused to start: {frame.get('code')}: {frame.get('message')}",
                code=str(frame.get("code") or "startup_failed"),
                stderr=self.stderr_tail(),
            )
        if frame.get("op") != "ready":
            raise TigerWorkerError(
                f"the worker's first frame was {frame.get('op')!r}, expected 'ready'",
                code="bad_handshake",
                stderr=self.stderr_tail(),
            )
        return frame

    def close(self) -> None:
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except (OSError, ValueError):  # pragma: no cover - defensive
                pass

    # -- frames ------------------------------------------------------------- #

    def request(self, frame: dict[str, Any], timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise TigerWorkerError(
                f"the TIGER worker is not running (exit={None if process is None else process.returncode})",
                code="worker_not_running",
                stderr=self.stderr_tail(),
            )
        assert process.stdin is not None
        try:
            process.stdin.write(json.dumps(frame, sort_keys=True) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise TigerWorkerError(
                f"the TIGER worker's stdin closed: {exc}",
                code="worker_stdin_closed",
                stderr=self.stderr_tail(),
            ) from exc
        response = self._next_frame(timeout, what=f"request {frame.get('request_id')!r}")
        if response.get("op") == "error":
            raise TigerWorkerError(
                f"the TIGER worker reported {response.get('code')}: {response.get('message')}",
                code=str(response.get("code") or "worker_error"),
                stderr=self.stderr_tail(),
            )
        return response

    def stderr_tail(self) -> str:
        return " | ".join(list(self._stderr)[-6:])

    # -- pumps -------------------------------------------------------------- #

    def _next_frame(self, timeout: float, *, what: str) -> dict[str, Any]:
        try:
            frame = self._frames.get(timeout=timeout)
        except queue.Empty as exc:
            raise TigerWorkerTimeout(
                f"the TIGER worker did not answer {what} within {timeout:.0f}s",
                code="timeout",
                stderr=self.stderr_tail(),
            ) from exc
        if frame is None:
            raise TigerWorkerError(
                f"the TIGER worker closed its stdout while waiting for {what}",
                code="worker_exited",
                stderr=self.stderr_tail(),
            )
        return frame

    def _pump_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._frames.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr.append(f"non-JSON on stdout: {line[:200]}")
        self._frames.put(None)

    def _pump_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        for line in process.stderr:
            line = line.rstrip()
            if line:
                self._stderr.append(line[:300])


TransportFactory = Callable[[TigerWorkerConfig], Transport]


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class TigerServingEngine:
    """A live TIGER engine: one long-lived worker, many single-request retrievals.

    The worker is started lazily on the first :meth:`start`/:meth:`recommend` and kept alive, so
    the generator is loaded once per process rather than once per request.
    """

    def __init__(
        self,
        identity: IdentityResolver | Mapping[str, int],
        *,
        config: TigerWorkerConfig | None = None,
        mode: str = CERTIFIED,
        allow_mode_downgrade: bool = False,
        startup_timeout: float = 900.0,
        request_timeout: float = 300.0,
        expected_checkpoint_sha256: str | None = None,
        expected_sid_sha256: str | None = None,
        transport_factory: TransportFactory | None = None,
        verify_assets_on_start: bool = True,
        blocking_start: bool = True,
    ) -> None:
        if mode not in (APPROXIMATE, CERTIFIED):
            raise TigerEngineError(f"mode must be {APPROXIMATE} or {CERTIFIED}, got {mode!r}")
        if isinstance(identity, Mapping):
            identity = _MappingIdentity(identity)
        for name in ("parent_asin_to_item_id", "item_id_to_parent_asin"):
            if not callable(getattr(identity, name, None)):
                raise TigerEngineError(f"the identity map must provide {name}()")
        self._identity = identity
        self._config = config or TigerWorkerConfig()
        self._mode = mode
        self._allow_mode_downgrade = bool(allow_mode_downgrade)
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._expected_checkpoint = expected_checkpoint_sha256
        self._expected_sid = expected_sid_sha256
        self._transport_factory: TransportFactory = transport_factory or SubprocessTransport
        self._verify_assets = bool(verify_assets_on_start)
        self._transport: Transport | None = None
        self._handshake: dict[str, Any] | None = None
        self._request_counter = 0
        #: ``True`` (default) loads the worker inside the first request; ``False`` refuses with
        #: :class:`TigerEngineNotReady` until :meth:`prewarm` has finished, which is what a
        #: serving process wants.
        self._blocking_start = bool(blocking_start)
        self._state = "not_started"
        self._last_error: str | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def config(self) -> TigerWorkerConfig:
        return self._config

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def started(self) -> bool:
        return self._handshake is not None

    def missing_assets(self) -> list[str]:
        return self._config.missing()

    def start(self) -> dict[str, Any]:
        """Start the worker (idempotent) and return its handshake."""
        if self._handshake is not None:
            return self._handshake
        if self._verify_assets:
            missing = self.missing_assets()
            if missing:
                self._state = "failed"
                self._last_error = "assets_missing: " + "; ".join(missing)
                raise TigerAssetsMissing(
                    "TIGER serving needs artifacts that are not present: " + "; ".join(missing)
                )
        transport = self._transport_factory(self._config)
        handshake = transport.start(self._startup_timeout)
        if str(handshake.get("contract")) != CONTRACT:
            transport.close()
            raise TigerContractViolation(
                f"the worker announced contract {handshake.get('contract')!r}, expected {CONTRACT!r}"
            )
        try:
            self._check_pinned_hashes(dict(handshake.get("identity") or {}))
        except Exception as exc:
            # Never leave a started worker behind when the run is refused.
            transport.close()
            self._state = "failed"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        self._transport = transport
        self._handshake = handshake
        self._state = "ready"
        self._last_error = None
        return handshake

    def prewarm(self, *, background: bool = True, timeout: float | None = None) -> dict[str, Any]:
        """Load the worker once, up front, and report the state.

        Serving deployments call this at boot with ``background=True``: the process answers
        ``/health`` immediately (``loading``) and becomes ``ready`` when the model is in memory.
        """
        if self._handshake is not None:
            return self.health()

        def _load() -> None:
            try:
                self.start()
            except Exception as exc:  # noqa: BLE001 - surfaced through health()
                self._state = "failed"
                self._last_error = f"{type(exc).__name__}: {exc}"

        self._state = "loading"
        if background:
            thread = threading.Thread(target=_load, name="tiger-prewarm", daemon=True)
            thread.start()
            if timeout is not None:
                thread.join(timeout)
        else:
            _load()
        return self.health()

    def close(self) -> None:
        transport, self._transport = self._transport, None
        self._handshake = None
        if transport is not None:
            transport.close()
        self._state = "not_started"

    def health(self) -> dict[str, Any]:
        """Lifecycle state plus the artifacts it would serve from.

        ``state`` is one of ``not_started`` (configured, worker not launched), ``loading``
        (prewarm in flight), ``ready`` (handshake completed), ``failed`` (the worker refused or
        died - ``error`` carries the reason).  ``assets_missing`` lists every missing artifact so
        a cold deployment says *what* is absent instead of only that it is not ready.
        """
        identity = dict((self._handshake or {}).get("identity") or {})
        return {
            "state": self._state,
            "started": self._handshake is not None,
            "device": str(self._config.device),
            "retrieval_mode": str(self._mode),
            "checkpoint_dir": str(self._config.checkpoint_dir),
            "checkpoint_sha256": str(identity.get("checkpoint_sha256") or ""),
            "sid_artifact_sha256": str(identity.get("sid_artifact_sha256") or ""),
            "error": self._last_error,
            "assets_missing": self.missing_assets(),
        }

    def __enter__(self) -> "TigerServingEngine":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- reporting ---------------------------------------------------------- #

    def identity_report(self) -> dict[str, Any]:
        """The worker's declared artifact identity (hashes and layout)."""
        if self._handshake is None:
            raise TigerWorkerError("the TIGER worker has not started yet", code="not_started")
        return dict(self._handshake.get("identity") or {})

    def asset_report(self) -> dict[str, Any]:
        """Serving provenance: contract, device, artifact hashes, resolver policy."""
        if self._handshake is None:
            raise TigerWorkerError("the TIGER worker has not started yet", code="not_started")
        identity = self.identity_report()
        return {
            "contract": str(self._handshake.get("contract")),
            "device": str(self._handshake.get("device")),
            "load_seconds": float(self._handshake.get("load_seconds") or 0.0),
            "checkpoint_sha256": str(identity.get("checkpoint_sha256") or ""),
            "sid_artifact_sha256": str(identity.get("sid_artifact_sha256") or ""),
            "catalogue_sha256": str(identity.get("catalogue_sha256") or ""),
            "generator_layout_sha256": str(identity.get("generator_layout_sha256") or ""),
            "exposure_sha256": str(identity.get("exposure_sha256") or ""),
            "resolver": dict(self._handshake.get("resolver") or {}),
            "paths": {
                "checkpoint_dir": str(self._config.checkpoint_dir),
                "sid_dir": str(self._config.sid_dir),
                "catalogue_dir": str(self._config.catalogue_dir),
            },
        }

    def _check_pinned_hashes(self, identity: Mapping[str, Any]) -> None:
        for label, expected, key in (
            ("checkpoint", self._expected_checkpoint, "checkpoint_sha256"),
            ("sid artifact", self._expected_sid, "sid_artifact_sha256"),
        ):
            if expected is None:
                continue
            actual = str(identity.get(key) or "")
            if actual != expected:
                raise TigerContractViolation(
                    f"the running {label} is {actual or 'unknown'!r}, expected the pinned "
                    f"{expected!r}; refusing to serve from an unpinned artifact"
                )

    # -- identity ----------------------------------------------------------- #

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Trusted direction used by ``TigerSourceTool`` to encode the history."""
        return int(self._identity.parent_asin_to_item_id(parent_asin))

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Trusted direction used to re-attach canonical identity to a candidate."""
        return str(self._identity.item_id_to_parent_asin(int(item_id)))

    # -- inference ---------------------------------------------------------- #

    def recommend(
        self,
        history_item_ids: Sequence[int],
        *,
        k: int = 20,
        mode: str | None = None,
        required_frontier: int | None = None,
        max_hist_items: int | None = None,
    ) -> TigerServingResult:
        """Run one live generative retrieval for a caller-supplied history.

        The history is AgentRec-X integer item ids; nothing else is accepted.  The result's
        ``retrieval_mode`` is the mode the worker **actually ran**, never an upgrade.
        """
        history = self._validate_history(history_item_ids)
        requested = str(mode or self._mode)
        if requested not in (APPROXIMATE, CERTIFIED):
            raise TigerEngineError(f"mode must be {APPROXIMATE} or {CERTIFIED}, got {requested!r}")
        k_max = int(self._config.k_max)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= k_max:
            raise TigerEngineError(f"k must be an integer in 1..{k_max}, got {k!r}")
        if required_frontier is not None and (
            isinstance(required_frontier, bool)
            or not isinstance(required_frontier, int)
            or required_frontier < 1
        ):
            raise TigerEngineError(
                f"required_frontier must be a positive int or None, got {required_frontier!r}"
            )
        if self._handshake is None:
            if not self._blocking_start:
                raise TigerEngineNotReady(
                    "the TIGER worker is still loading; its health state is "
                    f"{self.health()['state']!r}"
                )
            self.start()
        assert self._transport is not None

        self._request_counter += 1
        request_id = self._request_counter
        # The frontier is a project rule, not a guess: the frozen cohort defines
        # ``required_frontier == K_max + |unique seen|`` (experiments/step26_gates.py G16.1), which
        # is what lets a certified top-k survive a seen-item mask.  The worker keeps the same
        # fallback for callers that talk to it directly; sending it explicitly makes every
        # request frame self-describing.
        if required_frontier is None:
            required_frontier = int(k) + len(set(history))
        frame: dict[str, Any] = {
            "op": "recommend",
            "request_id": request_id,
            "history": list(history),
            "k": int(k),
            "mode": requested,
            "required_frontier": int(required_frontier),
        }
        if max_hist_items is not None:
            frame["max_hist_items"] = int(max_hist_items)

        response = self._transport.request(frame, self._request_timeout)
        if response.get("request_id") not in (None, request_id):
            raise TigerContractViolation(
                f"the worker answered request {response.get('request_id')!r} while "
                f"{request_id!r} was outstanding"
            )
        reported = str(response.get("retrieval_mode") or "")
        if reported not in (APPROXIMATE, CERTIFIED):
            raise TigerContractViolation(
                f"the worker reported retrieval_mode {reported!r}; expected "
                f"{APPROXIMATE!r} or {CERTIFIED!r}"
            )
        if requested == CERTIFIED and reported != CERTIFIED and not self._allow_mode_downgrade:
            raise TigerContractViolation(
                f"CERTIFIED retrieval was requested but the worker ran {reported}; an "
                "approximate result is never reported as certified"
            )
        recommendations = self._build_recommendations(response, reported)
        return TigerServingResult(
            retrieval_mode=reported,
            recommendations=recommendations,
            checkpoint_sha256=str(response.get("checkpoint_sha256") or ""),
            sid_artifact_sha256=str(response.get("sid_artifact_sha256") or ""),
            resolver=dict(response.get("resolver") or {}),
            k=int(response.get("k", k)),
            required_frontier=int(response.get("required_frontier") or 0),
            history_length=int(response.get("history_length") or len(history)),
            wall_seconds=float(response.get("wall_seconds") or 0.0),
            summary=dict(response.get("summary") or {}),
            requested_mode=requested,
        )

    # -- validation --------------------------------------------------------- #

    @staticmethod
    def _validate_history(history_item_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(history_item_ids, (str, bytes)) or not isinstance(history_item_ids, Iterable):
            raise TigerEngineError("history_item_ids must be a sequence of integer item ids")
        values = list(history_item_ids)
        if not values:
            raise TigerEngineError("history_item_ids must not be empty")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TigerEngineError(
                    f"history_item_ids must be integers, got {type(value).__name__}"
                )
            if value == 0:
                raise TigerEngineError("item id 0 is PAD and is never a history item")
            if value < 1:
                raise TigerEngineError(f"item ids must be >= 1, got {value!r}")
        return tuple(values)

    def _build_recommendations(
        self, response: Mapping[str, Any], mode: str
    ) -> tuple[TigerRecommendation, ...]:
        raw_items = response.get("items")
        if not isinstance(raw_items, list):
            raise TigerContractViolation("the worker response carries no 'items' list")
        seen_ids: set[int] = set()
        seen_ranks: set[int] = set()
        out: list[TigerRecommendation] = []
        for position, raw in enumerate(raw_items, start=1):
            if not isinstance(raw, Mapping):
                raise TigerContractViolation(f"item {position} is not an object")
            item_id = raw.get("item_id")
            if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
                raise TigerContractViolation(
                    f"item {position} has item_id {item_id!r}; a Semantic ID is not an identity "
                    "and PAD is never a candidate"
                )
            if item_id in seen_ids:
                raise TigerContractViolation(f"item_id {item_id} appears twice in one response")
            seen_ids.add(item_id)
            rank = raw.get("rank", position)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise TigerContractViolation(f"item {item_id} has rank {rank!r}")
            if rank in seen_ranks:
                raise TigerContractViolation(f"rank {rank} is used twice in one response")
            seen_ranks.add(rank)
            score = raw.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TigerContractViolation(f"item {item_id} has a non-numeric score {score!r}")
            score = float(score)
            if not math.isfinite(score):
                raise TigerContractViolation(f"item {item_id} has a non-finite score")
            semantic_id = raw.get("semantic_id")
            if not isinstance(semantic_id, (list, tuple)) or not semantic_id:
                raise TigerContractViolation(
                    f"item {item_id} carries no semantic_id; TIGER provenance requires it"
                )
            semantic = tuple(int(value) for value in semantic_id)
            token_path = tuple(int(v) for v in (raw.get("token_path") or ()))
            try:
                parent_asin = self.item_id_to_parent_asin(item_id)
            except Exception as exc:  # noqa: BLE001 - an unmappable id is a hard failure
                raise TigerContractViolation(
                    f"item_id {item_id} from the worker is not in the trusted identity map; "
                    "results are never substituted or dropped"
                ) from exc
            out.append(
                TigerRecommendation(
                    item_id=item_id,
                    parent_asin=parent_asin,
                    rank=rank,
                    score=score,
                    semantic_id=semantic,
                    token_path=token_path,
                    resolver=dict(raw.get("resolver") or {}),
                )
            )
        ordered = tuple(sorted(out, key=lambda item: item.rank))
        # ``TigerSourceTool`` validates ``rank == position`` (tiger_source.py:295), i.e. the
        # sequence must be ranked 1..N with no gaps.  Checking it here turns an opaque
        # source-unavailable into a precise contract violation.
        expected = tuple(range(1, len(ordered) + 1))
        actual = tuple(item.rank for item in ordered)
        if actual != expected:
            raise TigerContractViolation(
                f"the worker's ranks are {actual}, expected the contiguous 1..{len(ordered)} "
                "that the source seam requires"
            )
        return ordered


# --------------------------------------------------------------------------- #
# Operator entry point: asset self-check and one live recommendation
# --------------------------------------------------------------------------- #


def _parse_history(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.replace(" ", "").split(",") if part)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - operator CLI
    from recommendation.inference.tiger_engine import TigerServingEngine as _Engine  # self-import guard

    parser = argparse.ArgumentParser(description="TIGER serving engine self-check / one request")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--sid", type=Path, default=DEFAULT_SID_DIR)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=(CERTIFIED, APPROXIMATE), default=CERTIFIED)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--history", default="", help="comma-separated AgentRec-X item ids")
    parser.add_argument("--identity-json", type=Path, default=None,
                        help="optional {parent_asin: item_id} JSON for a live request")
    args = parser.parse_args(argv)

    config = TigerWorkerConfig(
        checkpoint_dir=args.checkpoint_dir,
        sid_dir=args.sid,
        catalogue_dir=args.catalogue,
        device=args.device,
    )
    missing = config.missing()
    if missing:
        print("TIGER serving assets are incomplete:")
        for entry in missing:
            print(f"  - {entry}")
        return 3
    print("all declared assets are present")
    if not args.history:
        print("(pass --history/--identity-json to run a live request)")
        return 0
    if args.identity_json is None:
        print("a live request needs --identity-json {parent_asin: item_id}")
        return 2
    identity = json.loads(Path(args.identity_json).read_text(encoding="utf-8"))
    engine = _Engine(identity, config=config, mode=args.mode)
    with engine:
        print(json.dumps(engine.asset_report(), indent=1, sort_keys=True))
        result = engine.recommend(_parse_history(args.history), k=args.k)
    print(json.dumps(result.as_dict(), indent=1, sort_keys=True)[:2000])
    return 0


if __name__ == "__main__":  # pragma: no cover - operator CLI
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Process-wide engine registry
# --------------------------------------------------------------------------- #

#: The one TIGER engine a process serves from.
#:
#: A deployment must load the model **once**: the serving fusion and the Agent both need TIGER,
#: and two engines would mean two worker processes, two multi-minute cold starts and two
#: answers to "which checkpoint is serving".  Whichever component builds the engine first
#: registers it here; every later builder adopts it.
_SHARED_ENGINE: "TigerServingEngine | None" = None


def set_shared_engine(engine: "TigerServingEngine | None") -> None:
    """Register the process-wide TIGER engine (``None`` clears it)."""
    global _SHARED_ENGINE
    _SHARED_ENGINE = engine


def shared_engine() -> "TigerServingEngine | None":
    """The registered engine, or ``None`` when this process has not built one yet."""
    return _SHARED_ENGINE
