"""Local demo launcher support for AgentRec-X (Milestone 11.5).

This module is the **decision layer** behind ``scripts/start_demo.sh`` and
``scripts/setup_demo.sh``.  The shell scripts own the process (interpreter selection,
``exec``, foreground/Ctrl+C); this module owns everything that benefits from being
typed, imported and unit-tested:

* locating the repository root independently of the current working directory;
* loading and verifying the committed artifact manifest
  (:data:`MANIFEST_PATH`, ``config/demo_runtime_artifacts.json``);
* tiered artifact verification -- **fast** (existence + regular file + exact byte
  size) for every normal start, **deep** (full SHA-256) for setup and explicit
  ``--verify``;
* classifying the configured TCP port as *free*, *already serving AgentRec-X*, or
  *occupied by a foreign process*;
* reporting the Python / NumPy / PyTorch-CPU environment.

Trust boundaries
----------------
This module **never**:

* installs or upgrades anything (``pip`` is invoked only by ``scripts/setup_demo.sh``);
* starts, stops, signals or kills any process -- it only *observes* a port.  The module
  deliberately imports neither ``subprocess`` nor ``signal``, so it has no
  process-management capability at all, and ``tests/test_local_demo.py`` guards that;
* writes, moves or regenerates an accepted artifact;
* replaces the runtime's own verification.  The checkpoint digest check inside
  :class:`~recommendation.inference.sasrec.SASRecInferenceEngine` remains authoritative;
  the launcher's checks are an *additional*, earlier gate.

Why fast verification is by size and not by hash
------------------------------------------------
The five artifacts total ~832 MB (~856 MB with the catalogue metadata).  Hashing that
on every start would add seconds to a demo whose whole purpose is a quick local start.
An exact byte-size comparison catches truncation, a partial copy and the realistic
WSL2/repository-mistake cases at the cost of one ``stat`` per file.  Full SHA-256 is
available on demand via ``--verify`` and is always used by ``setup_demo.sh``.

Usage
-----
::

    python -m recommendation.local_demo doctor     [--verify] [--json]
    python -m recommendation.local_demo preflight  [--verify] [--host H --port P]
    python -m recommendation.local_demo env        [--json]
    python -m recommendation.local_demo manifest-verify
    python -m recommendation.local_demo artifact-list
    python -m recommendation.local_demo port       [--host H --port P] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "ARTIFACT_MANIFEST_FORMAT",
    "ArtifactCheck",
    "ArtifactReport",
    "ArtifactSpec",
    "EnvironmentInfo",
    "LauncherError",
    "MANIFEST_PATH",
    "PortStatus",
    "REPO_ROOT",
    "WHEEL_PATH",
    "classify_port",
    "default_venv_python",
    "environment_info",
    "load_artifact_manifest",
    "load_artifact_specs",
    "normalise_omp_num_threads",
    "report_to_dict",
    "resolve_repo_root",
    "verify_artifacts",
]


class LauncherError(RuntimeError):
    """Raised when the local demo cannot be prepared safely."""


# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #


def resolve_repo_root(start: Path | None = None) -> Path:
    """Return the AgentRec-X repository root, independent of the CWD.

    Marker-based rather than a fixed number of ``..`` hops, so the module keeps
    working if it is ever moved inside a sub-package, and so a *wrong* checkout is
    detected instead of silently accepted.

    Searches upward from ``start`` (when given) and from this file's own location.  It
    deliberately does **not** fall back to the current working directory: the module's
    own path already guarantees a correct answer, and a CWD fallback would make the
    result depend on where the caller happened to be standing.
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    candidates.append(Path(__file__).resolve())

    markers = (Path("recommendation") / "config.py", Path("AGENTS.md"))
    for candidate in candidates:
        base = candidate if candidate.is_dir() else candidate.parent
        for directory in (base, *base.parents):
            if all((directory / marker).exists() for marker in markers):
                return directory.resolve()
    raise LauncherError(
        "could not locate the AgentRec-X repository root (looked for "
        f"{', '.join(str(m) for m in markers)} from {candidates[0]})"
    )


#: Repository root, derived from this file's location.
REPO_ROOT = resolve_repo_root()

#: The accepted runtime artifact manifest (small committed metadata, never artifacts).
MANIFEST_PATH = REPO_ROOT / "config" / "demo_runtime_artifacts.json"

#: Marker file proving a virtual environment is usable.
WHEEL_PATH = "pyvenv.cfg"

#: Canonical relative location of the project virtualenv interpreter.
VENV_PYTHON_RELPATH = Path(".venv") / "bin" / "python"

#: Artifact manifest format tag.
ARTIFACT_MANIFEST_FORMAT = "agentrecx.demo_runtime_artifacts.v1"

#: Expected artifact roles.  A manifest missing or adding a role is rejected so the
#: launcher can never silently stop checking something the demo needs.
EXPECTED_ARTIFACT_ROLES: tuple[str, ...] = (
    "checkpoint",
    "manifest",
    "mappings",
    "catalog",
    "sequences",
)

#: Supported Python series, matching the accepted environment.
SUPPORTED_PYTHON: tuple[int, int] = (3, 10)


def default_venv_python(repo_root: Path | None = None) -> Path:
    """Return ``<repo>/.venv/bin/python`` (Linux/WSL2 layout)."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / VENV_PYTHON_RELPATH


# --------------------------------------------------------------------------- #
# Artifact manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactSpec:
    """One expected runtime artifact, as recorded in the committed manifest."""

    role: str
    path: str
    size_bytes: int
    sha256: str
    description: str = ""
    verified_at_runtime: bool = False

    def absolute(self, repo_root: Path | None = None) -> Path:
        """Resolve the artifact path against the repository root."""
        root = Path(repo_root) if repo_root is not None else REPO_ROOT
        return root / self.path


def load_artifact_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the committed artifact manifest.

    Raises
    ------
    LauncherError
        The manifest is missing, is not JSON, has the wrong format tag, or does not
        describe exactly :data:`EXPECTED_ARTIFACT_ROLES`.
    """
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.is_file():
        raise LauncherError(f"artifact manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise LauncherError(f"artifact manifest is not valid JSON: {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise LauncherError(f"artifact manifest must be a JSON object: {manifest_path}")
    if payload.get("format") != ARTIFACT_MANIFEST_FORMAT:
        raise LauncherError(
            f"artifact manifest format {payload.get('format')!r} != "
            f"{ARTIFACT_MANIFEST_FORMAT!r}"
        )

    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise LauncherError("artifact manifest has no 'artifacts' list")

    roles = [entry.get("role") for entry in entries if isinstance(entry, dict)]
    if sorted(roles) != sorted(EXPECTED_ARTIFACT_ROLES):
        raise LauncherError(
            f"artifact manifest roles {sorted(roles)} != {sorted(EXPECTED_ARTIFACT_ROLES)}"
        )
    if len(entries) != len(EXPECTED_ARTIFACT_ROLES):
        raise LauncherError("artifact manifest has duplicate roles")
    return payload


def load_artifact_specs(path: str | Path | None = None) -> tuple[ArtifactSpec, ...]:
    """Return the manifest's artifacts as validated :class:`ArtifactSpec` objects."""
    payload = load_artifact_manifest(path)
    specs: list[ArtifactSpec] = []
    for entry in payload["artifacts"]:
        spec = ArtifactSpec(
            role=str(entry.get("role")),
            path=str(entry.get("path", "")),
            size_bytes=int(entry.get("size_bytes", -1)),
            sha256=str(entry.get("sha256", "")),
            description=str(entry.get("description", "")),
            verified_at_runtime=bool(entry.get("verified_at_runtime", False)),
        )
        if not spec.path or Path(spec.path).is_absolute():
            raise LauncherError(
                f"artifact {spec.role!r} path must be repository-relative, got {spec.path!r}"
            )
        if ".." in Path(spec.path).parts:
            raise LauncherError(
                f"artifact {spec.role!r} path must not escape the repository: {spec.path!r}"
            )
        if spec.size_bytes <= 0:
            raise LauncherError(
                f"artifact {spec.role!r} has no positive size_bytes in the manifest"
            )
        if len(spec.sha256) != 64 or any(c not in "0123456789abcdef" for c in spec.sha256):
            raise LauncherError(
                f"artifact {spec.role!r} sha256 must be 64 lowercase hex characters"
            )
        specs.append(spec)
    return tuple(specs)


# --------------------------------------------------------------------------- #
# Tiered artifact verification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactCheck:
    """The result of checking one artifact."""

    role: str
    path: str
    mode: str  # "fast" | "deep"
    status: str  # "ok" | "missing" | "not_a_regular_file" | "size_mismatch" | "sha256_mismatch"
    message: str
    size_bytes: int | None = None
    actual_sha256: str | None = None

    @property
    def ok(self) -> bool:
        """True when the artifact passed its check."""
        return self.status == "ok"


@dataclass
class ArtifactReport:
    """Aggregate result over every expected artifact."""

    mode: str
    checks: list[ArtifactCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every artifact passed."""
        return bool(self.checks) and all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[ArtifactCheck]:
        """The checks that did not pass."""
        return [check for check in self.checks if not check.ok]


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """Return the SHA-256 of a file, streamed so large files are never fully read.

    Mirrors :func:`recommendation.training.checkpoint.sha256_file` but is defined here
    to keep this module importable without importing PyTorch.
    """
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(
    specs: Sequence[ArtifactSpec] | None = None,
    *,
    deep: bool = False,
    repo_root: Path | None = None,
    compute_sha256: bool = True,
) -> ArtifactReport:
    """Verify every expected artifact.

    ``deep=False`` (fast mode) checks existence, regular-file type and exact byte size.
    ``deep=True`` additionally hashes the file and compares against the manifest.

    Every artifact is checked even after a failure, so one missing file does not hide
    the state of the others.
    """
    resolved = tuple(specs) if specs is not None else load_artifact_specs()
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    mode = "deep" if deep else "fast"
    report = ArtifactReport(mode=mode)

    for spec in resolved:
        target = spec.absolute(root)
        if not target.exists():
            report.checks.append(
                ArtifactCheck(
                    role=spec.role,
                    path=spec.path,
                    mode=mode,
                    status="missing",
                    message=f"artifact missing: {spec.path}",
                )
            )
            continue
        if not target.is_file():
            report.checks.append(
                ArtifactCheck(
                    role=spec.role,
                    path=spec.path,
                    mode=mode,
                    status="not_a_regular_file",
                    message=f"artifact is not a regular file: {spec.path}",
                )
            )
            continue

        size = target.stat().st_size
        if size != spec.size_bytes:
            report.checks.append(
                ArtifactCheck(
                    role=spec.role,
                    path=spec.path,
                    mode=mode,
                    status="size_mismatch",
                    message=(
                        f"{spec.path} is {size} bytes, expected {spec.size_bytes} "
                        f"(difference {size - spec.size_bytes:+d})"
                    ),
                    size_bytes=size,
                )
            )
            continue

        if deep and compute_sha256:
            actual = sha256_file(target)
            if actual != spec.sha256:
                report.checks.append(
                    ArtifactCheck(
                        role=spec.role,
                        path=spec.path,
                        mode=mode,
                        status="sha256_mismatch",
                        message=(
                            f"{spec.path} sha256 {actual} != expected {spec.sha256}"
                        ),
                        size_bytes=size,
                        actual_sha256=actual,
                    )
                )
                continue
            report.checks.append(
                ArtifactCheck(
                    role=spec.role,
                    path=spec.path,
                    mode=mode,
                    status="ok",
                    message=f"{spec.path} ok ({size} bytes, sha256 verified)",
                    size_bytes=size,
                    actual_sha256=actual,
                )
            )
            continue

        report.checks.append(
            ArtifactCheck(
                role=spec.role,
                path=spec.path,
                mode=mode,
                status="ok",
                message=f"{spec.path} ok ({size} bytes)",
                size_bytes=size,
            )
        )

    return report


def report_to_dict(report: ArtifactReport) -> dict[str, Any]:
    """Return a JSON-serialisable view of an artifact report."""
    return {
        "mode": report.mode,
        "ok": report.ok,
        "checks": [
            {
                "role": check.role,
                "path": check.path,
                "status": check.status,
                "ok": check.ok,
                "message": check.message,
                "size_bytes": check.size_bytes,
                "actual_sha256": check.actual_sha256,
            }
            for check in report.checks
        ],
    }


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


@dataclass
class EnvironmentInfo:
    """Python / NumPy / PyTorch-CPU environment facts."""

    python_version: str
    python_executable: str
    python_supported: bool
    venv_python: str
    venv_exists: bool
    venv_self_contained: bool
    numpy_version: str | None
    numpy_error: str | None
    torch_version: str | None
    torch_error: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool | None
    torch_has_nvidia_packages: bool
    device: str
    cpu_only_ok: bool
    in_wsl: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "python_supported": self.python_supported,
            "venv_python": self.venv_python,
            "venv_exists": self.venv_exists,
            "venv_self_contained": self.venv_self_contained,
            "numpy_version": self.numpy_version,
            "numpy_error": self.numpy_error,
            "torch_version": self.torch_version,
            "torch_error": self.torch_error,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_cuda_available": self.torch_cuda_available,
            "torch_has_nvidia_packages": self.torch_has_nvidia_packages,
            "device": self.device,
            "cpu_only_ok": self.cpu_only_ok,
            "in_wsl": self.in_wsl,
        }

    @property
    def ok(self) -> bool:
        """True when the environment can serve the CPU-only demo."""
        return bool(
            self.python_supported
            and self.venv_exists
            and self.numpy_version is not None
            and self.torch_version is not None
            and self.cpu_only_ok
        )


def python_supported(version_info: Sequence[int] | None = None) -> bool:
    """True when the interpreter is the supported Python series."""
    info = tuple(version_info) if version_info is not None else sys.version_info
    return (int(info[0]), int(info[1])) == SUPPORTED_PYTHON


def _detect_nvidia_packages(site_packages: Path) -> bool:
    """True when CUDA userspace wheels are installed alongside torch.

    The PyPI CUDA wheels install a top-level ``nvidia`` package directory (cuDNN,
    cuBLAS, ...).  Its presence is the clearest on-disk evidence that a CPU-only
    environment has been polluted by an unqualified ``pip install torch``.
    """
    if not site_packages.is_dir():
        return False
    return any(
        entry.name.lower() == "nvidia"
        for entry in site_packages.iterdir()
        if entry.is_dir()
    )


def _detect_wsl() -> bool:
    """True when running under WSL (best effort, never raises)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:  # pragma: no cover - non-Linux
        return False


def environment_info(
    *,
    repo_root: Path | None = None,
    import_numpy: bool = True,
    import_torch: bool = True,
    venv_self_contained: bool | None = None,
) -> EnvironmentInfo:
    """Collect the environment facts the launcher reports.

    NumPy and PyTorch are imported lazily and their failures recorded rather than
    raised, so a missing dependency produces a readable report instead of a traceback.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    venv_python = default_venv_python(root)
    venv_exists = venv_python.is_file()

    if venv_self_contained is None:
        venv_self_contained = _read_venv_self_contained(root / ".venv" / WHEEL_PATH)

    numpy_version: str | None = None
    numpy_error: str | None = None
    if import_numpy:
        try:  # pragma: no cover - depends on the installed environment
            import numpy  # noqa: PLC0415 - deliberate lazy import

            numpy_version = str(numpy.__version__)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            numpy_error = f"{type(exc).__name__}: {exc}"

    torch_version: str | None = None
    torch_error: str | None = None
    cuda_version: str | None = None
    cuda_available: bool | None = None
    has_nvidia = False
    device = "cpu"
    if import_torch:
        try:  # pragma: no cover - depends on the installed environment
            import torch  # noqa: PLC0415 - deliberate lazy import

            torch_version = str(torch.__version__)
            cuda_version = getattr(torch.version, "cuda", None)
            cuda_available = bool(torch.cuda.is_available())
            has_nvidia = _detect_nvidia_packages(Path(torch.__file__).resolve().parent.parent)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            torch_error = f"{type(exc).__name__}: {exc}"

    cpu_only_ok = (
        torch_version is not None
        and cuda_version is None
        and cuda_available is False
        and not has_nvidia
    )

    return EnvironmentInfo(
        python_version=".".join(str(p) for p in sys.version_info[:3]),
        python_executable=sys.executable,
        python_supported=python_supported(),
        venv_python=str(venv_python),
        venv_exists=venv_exists,
        venv_self_contained=bool(venv_self_contained),
        numpy_version=numpy_version,
        numpy_error=numpy_error,
        torch_version=torch_version,
        torch_error=torch_error,
        torch_cuda_version=cuda_version,
        torch_cuda_available=cuda_available,
        torch_has_nvidia_packages=has_nvidia,
        device=device,
        cpu_only_ok=cpu_only_ok,
        in_wsl=_detect_wsl(),
    )


def _read_venv_self_contained(pyvenv_cfg: Path) -> bool:
    """Read ``include-system-site-packages`` from a ``pyvenv.cfg`` (default False)."""
    try:
        text = pyvenv_cfg.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip().lower() == "include-system-site-packages":
            return value.strip().lower() == "true"
    return False


def normalise_omp_num_threads(
    value: str | None, *, fallback: int = 8
) -> str | None:
    """Return a libgomp-safe ``OMP_NUM_THREADS`` value, or ``None`` to leave it unset.

    ``libgomp`` rejects an empty or non-positive ``OMP_NUM_THREADS`` on *every* torch
    import, which would kill the server before it starts (see ``conftest.py`` and the
    USAGE troubleshooting table).  This sandbox has been observed exporting ``0``.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return str(fallback)
    try:
        number = int(text)
    except ValueError:
        return str(fallback)
    # A valid non-positive value must be replaced (libgomp rejects it); a valid
    # positive value is the caller's deliberate choice and is preserved as-is.
    return text if number >= 1 else str(fallback)


# --------------------------------------------------------------------------- #
# Port classification
# --------------------------------------------------------------------------- #


@dataclass
class PortStatus:
    """What the launcher found on the configured port."""

    state: str  # "free" | "agentrecx_running" | "foreign"
    host: str
    port: int
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def free(self) -> bool:
        """True when nothing is listening."""
        return self.state == "free"

    @property
    def is_agentrecx(self) -> bool:
        """True when the listener is an AgentRec-X demo server."""
        return self.state == "agentrecx_running"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _probe_host(host: str) -> str:
    """Return an address a connect probe can actually use."""
    return "127.0.0.1" if host in ("", "0.0.0.0", "::", "*") else host


def port_is_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when a TCP connection to ``host:port`` succeeds."""
    try:
        with socket.create_connection((_probe_host(host), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _http_get_json(url: str, timeout: float) -> tuple[int | None, dict[str, Any] | None, str]:
    """GET a JSON endpoint; returns ``(status, payload, error)`` and never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost only
            status = int(response.status)
            raw = response.read(64 * 1024)
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - any failure means "not AgentRec-X"
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return status, None, "response was not JSON"
    return status, payload if isinstance(payload, dict) else None, ""


def classify_port(host: str, port: int, *, timeout: float = 2.0) -> PortStatus:
    """Classify the configured port without ever touching a foreign process.

    Returns ``"free"``, ``"agentrecx_running"`` or ``"foreign"``.  A port that answers
    like an AgentRec-X demo server is reported as such; anything else that is listening
    is ``"foreign"`` and must be left completely alone.
    """
    port = int(port)
    if not port_is_listening(host, port, timeout=min(timeout, 1.0)):
        return PortStatus(
            state="free",
            host=host,
            port=port,
            detail=f"nothing is listening on {host}:{port}",
        )

    base = f"http://{_probe_host(host)}:{port}"
    status, payload, error = _http_get_json(f"{base}/v1/demo/health", timeout)
    evidence: dict[str, Any] = {"demo_health_status": status, "demo_health_error": error}
    if status == 200 and payload is not None and {"status", "model_loaded", "demo_ready"} <= set(payload):
        model_status, model_payload, model_error = _http_get_json(f"{base}/v1/model", timeout)
        evidence["model_status"] = model_status
        evidence["model_error"] = model_error
        if model_status == 200 and model_payload is not None and model_payload.get("model_type") == "SASRec":
            evidence["demo_ready"] = payload.get("demo_ready")
            evidence["profiles"] = payload.get("profiles")
            evidence["device"] = model_payload.get("device")
            return PortStatus(
                state="agentrecx_running",
                host=host,
                port=port,
                detail=(
                    f"an AgentRec-X demo server is already serving {host}:{port} "
                    f"(model_loaded={payload.get('model_loaded')}, "
                    f"demo_ready={payload.get('demo_ready')}, "
                    f"profiles={payload.get('profiles')})"
                ),
                evidence=evidence,
            )
        evidence["reason"] = "demo health matched but /v1/model did not look like SASRec"

    return PortStatus(
        state="foreign",
        host=host,
        port=port,
        detail=(
            f"{host}:{port} is occupied by something that is not an AgentRec-X demo "
            f"server ({error or evidence.get('reason') or 'no matching health contract'})"
        ),
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Preflight aggregation
# --------------------------------------------------------------------------- #


@dataclass
class PreflightResult:
    """Everything the launcher needs to decide whether it may start."""

    environment: EnvironmentInfo
    artifacts: ArtifactReport
    port: PortStatus | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the demo may be started safely."""
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "environment": self.environment.to_dict(),
            "artifacts": report_to_dict(self.artifacts),
            "port": self.port.to_dict() if self.port is not None else None,
        }


def run_preflight(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    deep: bool = False,
    repo_root: Path | None = None,
    import_torch: bool = False,
) -> PreflightResult:
    """Run the artifact, environment and (optionally) port checks.

    ``import_torch`` defaults to ``False`` so the fast preflight stays cheap; the
    ``doctor`` command asks for the full environment report explicitly.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    env = environment_info(repo_root=root, import_numpy=True, import_torch=import_torch)
    report = verify_artifacts(deep=deep, repo_root=root)
    status = classify_port(host, port) if port is not None else None

    problems: list[str] = []
    if not env.python_supported:
        problems.append(
            f"Python {env.python_version} is not supported; expected "
            f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}.x"
        )
    if not env.venv_exists:
        problems.append(
            f"virtualenv interpreter not found: {env.venv_python} "
            "(run ./scripts/setup_demo.sh)"
        )
    if env.numpy_version is None:
        problems.append(f"NumPy is not importable: {env.numpy_error}")
    if import_torch and env.torch_version is None:
        problems.append(f"PyTorch is not importable: {env.torch_error}")
    if import_torch and env.torch_version is not None and not env.cpu_only_ok:
        problems.append(
            "PyTorch is not the accepted CPU-only build "
            f"(torch={env.torch_version}, cuda={env.torch_cuda_version}, "
            f"cuda_available={env.torch_cuda_available}, "
            f"nvidia_packages={env.torch_has_nvidia_packages})"
        )
    if not report.ok:
        for check in report.failures:
            problems.append(check.message)
    # A busy port is deliberately NOT a preflight problem.  The environment and the
    # artifacts can be perfectly fine while the port is taken, and the launcher reports
    # the three port outcomes as its own step -- that is what lets it tell
    # "AgentRec-X already running" (exit 0) apart from "foreign process" (exit 3)
    # without either being reported as a broken environment.
    return PreflightResult(environment=env, artifacts=report, port=status, problems=problems)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_artifact_report(report: ArtifactReport, stream: Any = None) -> None:
    """Print a human-readable artifact report."""
    out = stream if stream is not None else sys.stdout
    label = "SHA-256" if report.mode == "deep" else "size"
    print(f"Artifacts ({report.mode} verification: existence + regular file + {label}):", file=out)
    for check in report.checks:
        mark = "OK  " if check.ok else "FAIL"
        print(f"  [{mark}] {check.role:<10} {check.message}", file=out)


def _print_environment(env: EnvironmentInfo, stream: Any = None) -> None:
    """Print a human-readable environment report."""
    out = stream if stream is not None else sys.stdout
    print("Environment:", file=out)
    print(f"  python          : {env.python_version} ({env.python_executable})", file=out)
    print(f"  venv interpreter: {env.venv_python} (exists={env.venv_exists})", file=out)
    print(f"  numpy           : {env.numpy_version or env.numpy_error}", file=out)
    print(f"  torch           : {env.torch_version or env.torch_error}", file=out)
    if env.torch_version is not None:
        print(f"  torch.version.cuda : {env.torch_cuda_version}", file=out)
        print(f"  cuda available  : {env.torch_cuda_available}", file=out)
        print(f"  nvidia packages : {env.torch_has_nvidia_packages}", file=out)
    print(f"  device          : {env.device} (CUDA not required)", file=out)
    print(f"  WSL             : {env.in_wsl}", file=out)
    print(f"  cpu-only ok     : {env.cpu_only_ok}", file=out)


def _cmd_env(args: argparse.Namespace) -> int:
    """``env``: report the environment, optionally as JSON."""
    env = environment_info(import_torch=True)
    if args.json:
        print(json.dumps(env.to_dict(), indent=2, sort_keys=True))
    else:
        _print_environment(env)
    return 0 if env.ok else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    """``doctor``: full environment + artifact report (deep when asked)."""
    env = environment_info(import_torch=True)
    report = verify_artifacts(deep=args.verify)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": env.ok and report.ok,
                    "environment": env.to_dict(),
                    "artifacts": report_to_dict(report),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if (env.ok and report.ok) else 1

    _print_environment(env)
    print()
    _print_artifact_report(report)

    problems: list[str] = []
    if not env.python_supported:
        problems.append(f"unsupported Python {env.python_version}")
    if not env.venv_exists:
        problems.append(f"venv interpreter missing: {env.venv_python}")
    if env.numpy_version is None:
        problems.append(f"numpy not importable: {env.numpy_error}")
    if env.torch_version is None:
        problems.append(f"torch not importable: {env.torch_error}")
    elif not env.cpu_only_ok:
        problems.append("torch is not the accepted CPU-only build")
    problems.extend(check.message for check in report.failures)

    print()
    if problems:
        print("DOCTOR: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("DOCTOR: PASS")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    """``preflight``: the read-only gate run before every start."""
    result = run_preflight(host=args.host, port=args.port, deep=args.verify)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ok else 1

    print(f"Repository root : {REPO_ROOT}")
    print(f"Interpreter     : {result.environment.venv_python}")
    print()
    _print_artifact_report(result.artifacts)
    print()
    if result.port is not None:
        print(f"Port            : {result.port.state} - {result.port.detail}")
    else:
        print("Port            : not checked")
    print()
    if result.problems:
        print("PREFLIGHT: FAIL")
        for problem in result.problems:
            print(f"  - {problem}")
        return 1
    print("PREFLIGHT: PASS")
    return 0


def _cmd_manifest_verify(args: argparse.Namespace) -> int:
    """``manifest-verify``: validate the committed manifest itself."""
    specs = load_artifact_specs()
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifest": str(MANIFEST_PATH),
                    "format": ARTIFACT_MANIFEST_FORMAT,
                    "artifacts": [
                        {
                            "role": spec.role,
                            "path": spec.path,
                            "size_bytes": spec.size_bytes,
                            "sha256": spec.sha256,
                        }
                        for spec in specs
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"Artifact manifest: {MANIFEST_PATH}")
    print(f"Format           : {ARTIFACT_MANIFEST_FORMAT}")
    for spec in specs:
        print(f"  {spec.role:<10} {spec.size_bytes:>12}  {spec.sha256[:16]}...  {spec.path}")
    print("MANIFEST: PASS")
    return 0


def _cmd_artifact_list(args: argparse.Namespace) -> int:
    """``artifact-list``: print ``role<TAB>absolute path`` for the shell."""
    for spec in load_artifact_specs():
        print(f"{spec.role}\t{spec.absolute()}")
    return 0


def _cmd_port(args: argparse.Namespace) -> int:
    """``port``: classify a port without disturbing whatever is on it."""
    status = classify_port(args.host, args.port)
    if args.json:
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"{status.state}: {status.detail}")
    return 0 if status.state != "foreign" else 2


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher-support CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m recommendation.local_demo",
        description=(
            "Local demo launcher support: artifact verification, environment report and "
            "port classification. Read-only: this never installs, starts or kills anything."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="full environment + artifact report")
    doctor.add_argument("--verify", action="store_true", help="full SHA-256 verification")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    preflight = subparsers.add_parser("preflight", help="read-only gate run before start")
    preflight.add_argument("--host", default="127.0.0.1")
    preflight.add_argument("--port", type=int, default=None)
    preflight.add_argument("--verify", action="store_true", help="full SHA-256 verification")
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(func=_cmd_preflight)

    env = subparsers.add_parser("env", help="Python / NumPy / PyTorch-CPU report")
    env.add_argument("--json", action="store_true")
    env.set_defaults(func=_cmd_env)

    manifest = subparsers.add_parser("manifest-verify", help="validate the committed manifest")
    manifest.add_argument("--json", action="store_true")
    manifest.set_defaults(func=_cmd_manifest_verify)

    listing = subparsers.add_parser("artifact-list", help="print expected artifact paths")
    listing.set_defaults(func=_cmd_artifact_list)

    port = subparsers.add_parser("port", help="classify a TCP port")
    port.add_argument("--host", default="127.0.0.1")
    port.add_argument("--port", type=int, required=True)
    port.add_argument("--json", action="store_true")
    port.set_defaults(func=_cmd_port)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except LauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
