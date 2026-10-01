#!/usr/bin/env python3
"""Opt-in artifact recoverability tooling: list, verify, fetch and fresh-clone smoke.

What this is
------------
The canonical artifact metadata lives in ``config/demo_runtime_artifacts.json``.  This script
only *reads* it -- it never defines a second artifact list.  The ``recoverability`` section of
that manifest holds the B0-scope artifacts and their durable (Hugging Face) location.

Design rules, each of which is enforced rather than documented:

* **Nothing here runs implicitly.**  ``pytest`` and demo startup never call this module, and
  only the ``fetch`` and ``smoke`` subcommands may touch the network.
* **A download is never trusted.**  The local size and SHA-256 are recomputed after transfer;
  a mismatch deletes the partial file and fails closed.
* **There is no local-copy fallback.**  ``fetch`` always downloads, even when a correct file
  already exists, and ``smoke`` refuses to run when the target artifacts are already present.
  Restoring from the development machine's own disk is not recoverability evidence.
* **A credential is never printed.**  ``HF_TOKEN`` is read from the environment or from a
  git-ignored ``.env`` / ``.env.local``, sent only as a request header, and is dropped when a
  redirect leaves the original host.

Exit codes: ``0`` success, ``2`` blocked (not configured / durable location PENDING),
``3`` verification failure, ``4`` transport failure.

Usage
-----
    python scripts/fetch_artifacts.py list
    python scripts/fetch_artifacts.py verify
    python scripts/fetch_artifacts.py fetch --all
    python scripts/fetch_artifacts.py smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Repository root, derived from this file's location (``<repo>/scripts/fetch_artifacts.py``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Canonical manifest.  Single source of truth for artifact identity and durable location.
DEFAULT_MANIFEST = REPO_ROOT / "config" / "demo_runtime_artifacts.json"

#: Format tag of the manifest's recoverability section.
RECOVERY_FORMAT = "agentrecx.artifact_recovery.v1"

#: Lifecycle of a durable location.  ``READY`` claims that a clean restore has been proven, so only
#: a passing B0-c round-trip may reach it; an upload on its own never sets it.
STATUS_PENDING = "PENDING"
STATUS_UPLOADED_UNVERIFIED = "UPLOADED_UNVERIFIED"
STATUS_READY = "READY"
ALLOWED_STATUS: tuple[str, ...] = (STATUS_PENDING, STATUS_UPLOADED_UNVERIFIED, STATUS_READY)

#: Statuses that name a real remote location and may therefore be fetched.  Fetchable is not the
#: same as verified: ``UPLOADED_UNVERIFIED`` explicitly means "not yet proven recoverable".
FETCHABLE_STATUS: tuple[str, ...] = (STATUS_UPLOADED_UNVERIFIED, STATUS_READY)

#: What each status means, so an unverified location is never read as a guarantee.
STATUS_MEANING: dict[str, str] = {
    STATUS_PENDING: "no remote location exists yet",
    STATUS_UPLOADED_UNVERIFIED: (
        "a remote location exists and its metadata is queryable, but a clean restore has NOT been "
        "proven; B0-c must fetch, verify, load and serve it before this may become READY"
    ),
    STATUS_READY: "a clean restore from this location has been proven (B0-c passed)",
}

#: Repository kinds this tool understands.  The kind changes the URL shape and nothing else, so an
#: unknown value is refused rather than guessed.
MODEL_REPO_TYPE = "model"
DATASET_REPO_TYPE = "dataset"
ALLOWED_REPO_TYPES: tuple[str, ...] = (MODEL_REPO_TYPE, DATASET_REPO_TYPE)

#: Collection segment used by the **API**, per repository kind.
_API_COLLECTION: dict[str, str] = {
    MODEL_REPO_TYPE: "models",
    DATASET_REPO_TYPE: "datasets",
}

#: Prefix used by the **resolve** path.  Only a dataset repository carries one: a model file lives
#: at ``/<repo_id>/resolve/...`` while a dataset file lives at ``/datasets/<repo_id>/resolve/...``.
_RESOLVE_PREFIX: dict[str, str] = {
    MODEL_REPO_TYPE: "",
    DATASET_REPO_TYPE: "datasets/",
}

#: Environment variables.  ``HF_TOKEN`` is a credential and is never echoed.
ENV_REPO_ID = "HF_REPO_ID"
ENV_TOKEN = "HF_TOKEN"
ENV_REVISION = "HF_REVISION"
DEFAULT_REVISION = "main"

#: Credential files, highest precedence last.  The real process environment still wins over both.
ENV_FILES: tuple[str, ...] = (".env", ".env.local")

#: The consumer key a smoke run needs, as recorded in ``required_for``.
SMOKE_CONSUMER = "two_tower_serving_smoke"

CHUNK_BYTES = 1 << 20

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_VERIFY = 3
EXIT_NETWORK = 4


class RecoveryError(RuntimeError):
    """A recoverability precondition is not met.

    Messages are safe to print: they never contain a credential value.
    """


class VerificationError(RecoveryError):
    """A local or downloaded artifact failed size or digest verification."""


class BlockedError(RecoveryError):
    """The durable location is not usable yet (missing configuration or PENDING)."""


class TransportError(RecoveryError):
    """A fetch failed at the transport layer."""


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecoveryEntry:
    """One artifact that a durable-restore step is expected to be able to fetch."""

    role: str
    path: str
    size_bytes: int
    sha256: str
    description: str
    required_for: tuple[str, ...]
    durable_status: str
    repo_id: str | None
    revision: str | None
    remote_path: str | None

    def absolute(self, repo_root: Path) -> Path:
        """Resolve the artifact path against ``repo_root``."""
        return Path(repo_root) / self.path


@dataclass(frozen=True)
class DurableLocation:
    """Where a READY artifact can be fetched from, and the URIs that follow from it.

    The repository kind is part of the location because it changes the path shape: a dataset file
    lives under ``/datasets/<repo_id>/...``.  The full URI is **derived here and never stored**, so
    a recorded address cannot drift away from the recorded repository, revision and path.
    """

    endpoint: str
    repo_type: str
    repo_id: str
    revision: str
    remote_path: str

    @property
    def collection(self) -> str:
        """The API collection segment for this repository kind."""
        try:
            return _API_COLLECTION[self.repo_type]
        except KeyError:  # pragma: no cover - guarded by manifest validation
            raise RecoveryError(
                f"unknown repo_type {self.repo_type!r}; allowed: {ALLOWED_REPO_TYPES}"
            ) from None

    @property
    def resolve_prefix(self) -> str:
        """The resolve-path prefix: ``datasets/`` for a dataset repository, empty for a model."""
        try:
            return _RESOLVE_PREFIX[self.repo_type]
        except KeyError:  # pragma: no cover - guarded by manifest validation
            raise RecoveryError(
                f"unknown repo_type {self.repo_type!r}; allowed: {ALLOWED_REPO_TYPES}"
            ) from None

    @property
    def uri(self) -> str:
        """The download URI, derived from the parts so the two cannot drift apart."""
        return (
            f"{self.endpoint.rstrip('/')}/{self.resolve_prefix}{self.repo_id}"
            f"/resolve/{self.revision}/{self.remote_path}"
        )

    @property
    def api_uri(self) -> str:
        """The repository's metadata endpoint (no trailing slash)."""
        return f"{self.endpoint.rstrip('/')}/api/{self.collection}/{self.repo_id}"

    def tree_uri(self, revision: str | None = None) -> str:
        """The file-tree endpoint for a revision."""
        return f"{self.api_uri}/tree/{revision or self.revision}"


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the canonical manifest, or raise :class:`RecoveryError`."""
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise RecoveryError(f"canonical manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"canonical manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecoveryError(f"canonical manifest must be a JSON object: {manifest_path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def load_recovery_entries(
    manifest: Mapping[str, Any],
) -> tuple[RecoveryEntry, ...]:
    """Return the manifest's recoverability entries, validated.

    Validation is structural and offline: field presence and types, a repository-relative
    path, a positive size, a lowercase 64-hex digest and a known durable status.  It never
    touches the network and never reads an artifact.
    """
    section = manifest.get("recoverability")
    _require(isinstance(section, dict), "canonical manifest has no 'recoverability' section")
    assert isinstance(section, dict)  # narrowed for type checkers
    _require(
        section.get("format") == RECOVERY_FORMAT,
        f"recoverability format {section.get('format')!r} != {RECOVERY_FORMAT!r}",
    )
    entries = section.get("artifacts")
    _require(isinstance(entries, list) and bool(entries), "recoverability has no artifacts")

    parsed: list[RecoveryEntry] = []
    for raw in entries:
        _require(isinstance(raw, dict), "every recoverability entry must be a JSON object")
        assert isinstance(raw, dict)
        role = raw.get("role")
        _require(isinstance(role, str) and bool(role), "recoverability entry without a role")
        path = raw.get("path")
        _require(
            isinstance(path, str) and bool(path),
            f"artifact {role!r} has no path",
        )
        as_path = Path(path)
        _require(not as_path.is_absolute(), f"artifact {role!r} path must be relative: {path}")
        _require(".." not in as_path.parts, f"artifact {role!r} path escapes the repository")
        size = raw.get("size_bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"artifact {role!r} needs a positive integer size_bytes",
        )
        digest = raw.get("sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"artifact {role!r} needs a lowercase 64-hex sha256",
        )
        required_for = raw.get("required_for", [])
        _require(
            isinstance(required_for, list) and all(isinstance(x, str) for x in required_for),
            f"artifact {role!r} required_for must be a list of strings",
        )
        durable = raw.get("durable")
        _require(isinstance(durable, dict), f"artifact {role!r} has no durable block")
        assert isinstance(durable, dict)
        status = durable.get("status")
        _require(
            status in ALLOWED_STATUS,
            f"artifact {role!r} durable.status {status!r} not in {ALLOWED_STATUS}",
        )
        for field in ("repo_id", "revision", "remote_path"):
            value = durable.get(field)
            _require(
                value is None or (isinstance(value, str) and bool(value)),
                f"artifact {role!r} durable.{field} must be null or a non-empty string",
            )
        parsed.append(
            RecoveryEntry(
                role=role,
                path=path,
                size_bytes=size,
                sha256=digest,
                description=str(raw.get("description", "")),
                required_for=tuple(required_for),
                durable_status=str(status),
                repo_id=durable.get("repo_id"),
                revision=durable.get("revision"),
                remote_path=durable.get("remote_path"),
            )
        )
    roles = [entry.role for entry in parsed]
    _require(len(set(roles)) == len(roles), f"duplicate recoverability roles: {sorted(roles)}")
    return tuple(parsed)


def manifest_endpoint(manifest: Mapping[str, Any]) -> str:
    """Return the configured service endpoint (no trailing slash)."""
    endpoint = manifest.get("recoverability", {}).get("endpoint")
    _require(
        isinstance(endpoint, str) and endpoint.startswith(("http://", "https://")),
        "recoverability.endpoint must be an http(s) URL",
    )
    return endpoint.rstrip("/")


def manifest_repo_type(manifest: Mapping[str, Any]) -> str:
    """Return the frozen repository kind, refusing anything outside the allowed set.

    Fail-closed on purpose: an unrecognised kind would silently produce the wrong URL shape, and a
    404 caused by a guessed path is far more expensive to diagnose than a refusal here.
    """
    section = manifest.get("recoverability")
    _require(isinstance(section, dict), "canonical manifest has no 'recoverability' section")
    assert isinstance(section, dict)
    repo_type = section.get("repo_type")
    _require(
        repo_type in ALLOWED_REPO_TYPES,
        f"recoverability.repo_type {repo_type!r} is not one of {ALLOWED_REPO_TYPES}",
    )
    return str(repo_type)


def check_repo_id(repo_id: str) -> str:
    """Reject a repository id that already carries a collection prefix.

    ``HF_REPO_ID`` must stay the bare ``owner/name``.  If it carried ``datasets/`` as well, the
    derived URI would be ``/datasets/datasets/...`` and every fetch would 404.
    """
    for prefix in ("datasets/", "models/"):
        if repo_id.startswith(prefix):
            raise BlockedError(
                f"repo_id {repo_id!r} starts with {prefix!r}, but it must be the bare "
                f"'owner/name': the repository kind is frozen in the manifest and the collection "
                f"segment is derived from it. Remove the prefix to avoid a doubled path."
            )
    return repo_id


# --------------------------------------------------------------------------- #
# environment / secrets
# --------------------------------------------------------------------------- #


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` file.

    Blank lines and ``#`` comments are ignored, surrounding quotes are stripped and an
    existing value is never overwritten by a later file.  A missing file yields ``{}``.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = value
    return values


def load_environment(repo_root: Path) -> dict[str, str]:
    """Merge configuration sources, lowest precedence first.

    Precedence: **real process environment > ``.env.local`` > ``.env`` > the manifest default.**
    A variable set in the real environment is never overridden by a file, and an empty value is
    treated as unset rather than as a deliberate empty string.
    """
    merged: dict[str, str] = {}
    for name in ENV_FILES:  # lowest precedence first; a later assignment wins
        for key, value in read_env_file(Path(repo_root) / name).items():
            merged[key] = value
    merged.update(os.environ)
    return {key: value for key, value in merged.items() if value}


def resolve_durable(
    manifest: Mapping[str, Any],
    entry: RecoveryEntry,
    environ: Mapping[str, str],
) -> DurableLocation:
    """Resolve the fetchable location of ``entry``, or raise :class:`BlockedError`.

    A PENDING entry is refused explicitly.  An UPLOADED_UNVERIFIED entry *is* fetchable — that is
    what makes B0-c's clean restore possible — but it carries no recoverability claim.  The
    intended failure mode is "the owner has not published this yet", never "silently used whatever
    was on disk".
    """
    if entry.durable_status not in FETCHABLE_STATUS:
        raise BlockedError(
            f"artifact {entry.role!r} has no fetchable location yet: durable.status is "
            f"{entry.durable_status!r} ({STATUS_MEANING.get(entry.durable_status, 'unknown')}). "
            f"Upload the file, then record repo_id / revision / remote_path and set status to "
            f"'{STATUS_UPLOADED_UNVERIFIED}' in the canonical manifest. This tool will not fall "
            f"back to a local copy."
        )
    repo_id = environ.get(ENV_REPO_ID) or entry.repo_id
    if not repo_id:
        raise BlockedError(
            f"artifact {entry.role!r} has no repository: set {ENV_REPO_ID} (or record "
            f"durable.repo_id in the canonical manifest)"
        )
    remote_path = entry.remote_path
    if not remote_path:
        raise BlockedError(f"artifact {entry.role!r} has no durable.remote_path")
    revision = environ.get(ENV_REVISION) or entry.revision or DEFAULT_REVISION
    return DurableLocation(
        endpoint=manifest_endpoint(manifest),
        repo_type=manifest_repo_type(manifest),
        repo_id=check_repo_id(repo_id),
        revision=revision,
        remote_path=remote_path,
    )


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, entry: RecoveryEntry) -> None:
    """Verify an artifact's size and digest, or raise :class:`VerificationError`."""
    if not path.is_file():
        raise VerificationError(f"artifact {entry.role!r} is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != entry.size_bytes:
        raise VerificationError(
            f"artifact {entry.role!r} size mismatch: expected {entry.size_bytes}, got {actual_size}"
        )
    actual = sha256_file(path)
    if actual != entry.sha256:
        raise VerificationError(
            f"artifact {entry.role!r} sha256 mismatch: expected {entry.sha256}, got {actual}"
        )


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


class _TokenSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when a redirect leaves the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is None:
            return None
        if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new_request.headers.pop("Authorization", None)
        return new_request


def _open(uri: str, token: str | None):  # type: ignore[no-untyped-def]
    """Open ``uri`` for reading.  The only network call in this module."""
    request = urllib.request.Request(uri, headers={"User-Agent": "AgentRec-X-artifact-fetch/1"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(_TokenSafeRedirectHandler)
    return opener.open(request, timeout=60)


def download(uri: str, destination: Path, token: str | None = None) -> Path:
    """Stream ``uri`` into ``destination`` atomically.

    The bytes land in ``<destination>.part`` and are moved into place only by the caller's
    verification step, so a failed transfer can never be mistaken for a valid artifact.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with _open(uri, token) as response, open(partial, "wb") as handle:
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise TransportError(f"fetch failed with HTTP {exc.code} for {uri}") from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise TransportError(f"fetch failed for {uri}: {exc.reason}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise TransportError(f"fetch failed while writing {partial}: {exc}") from exc
    return partial


def fetch_entry(
    entry: RecoveryEntry,
    location: DurableLocation,
    repo_root: Path,
    token: str | None = None,
) -> Path:
    """Download, verify and install one artifact.  Never reuses an existing local copy."""
    destination = entry.absolute(repo_root)
    partial = download(location.uri, destination, token)
    try:
        verify_file(partial, entry)
    except VerificationError:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)
    return destination


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


def select_entries(
    entries: Sequence[RecoveryEntry],
    roles: Sequence[str],
    *,
    consumer: str | None = None,
) -> tuple[RecoveryEntry, ...]:
    """Select entries by role, by consumer, or all of them."""
    if roles:
        known = {entry.role for entry in entries}
        unknown = sorted(set(roles) - known)
        if unknown:
            raise RecoveryError(f"unknown artifact role(s): {unknown}; known: {sorted(known)}")
        wanted = set(roles)
        return tuple(entry for entry in entries if entry.role in wanted)
    if consumer is not None:
        selected = tuple(e for e in entries if consumer in e.required_for)
        if not selected:
            raise RecoveryError(f"no artifact declares required_for={consumer!r}")
        return selected
    return tuple(entries)


# --------------------------------------------------------------------------- #
# lightweight probe
# --------------------------------------------------------------------------- #

#: Probe outcomes.  ``SKIPPED`` is not a failure: it means the check could not be attempted for a
#: configuration reason (nothing to test yet), which is reported rather than guessed at.
PROBE_OK = "ok"
PROBE_AUTH = "authentication/permission"
PROBE_NOT_FOUND = "not-found"
PROBE_HTTP = "http-error"
PROBE_TIMEOUT = "timeout"
PROBE_CONNECTION = "connection-failure"
PROBE_SKIPPED = "skipped"

#: What each outcome means, so a failure is never misread as a different failure.
PROBE_REMEDY: dict[str, str] = {
    PROBE_TIMEOUT: "the endpoint did not answer in time: a network egress problem, not a credential problem",
    PROBE_CONNECTION: "the endpoint could not be reached: network/firewall, not a credential problem",
    PROBE_AUTH: f"check {ENV_TOKEN} validity, its scope, and that it may read this repository",
    PROBE_NOT_FOUND: (
        "check the repo_id spelling/case, the repo_type (model vs dataset), the revision, "
        "and the remote_path"
    ),
    PROBE_HTTP: "the server answered with an unexpected status; read the code before anything else",
    PROBE_SKIPPED: "nothing to test yet: record the missing field, then re-run the probe",
}

PROBE_TIMEOUT_SECONDS = 15


def classify_http_status(code: int) -> str:
    """Map an HTTP status to a probe outcome.  Never collapses 401 and 404 together."""
    if code in (401, 403):
        return PROBE_AUTH
    if code == 404:
        return PROBE_NOT_FOUND
    return PROBE_HTTP


def classify_transport_error(exc: BaseException) -> tuple[str, str]:
    """Map a raised transport error to ``(outcome, detail)``.

    Distinguishing "we could not reach it" from "it said no" is the whole point: a blocked egress
    and a bad token look identical if both are reported as "probe failed".
    """
    if isinstance(exc, urllib.error.HTTPError):
        return classify_http_status(exc.code), f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return PROBE_TIMEOUT, f"timeout after {PROBE_TIMEOUT_SECONDS}s: {reason}"
        return PROBE_CONNECTION, f"could not connect: {reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return PROBE_TIMEOUT, f"timeout after {PROBE_TIMEOUT_SECONDS}s: {exc}"
    if isinstance(exc, OSError):
        return PROBE_CONNECTION, f"could not connect: {exc}"
    return PROBE_CONNECTION, str(exc)


def normalise_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    """Lower-case header names so lookups do not depend on the server's casing."""
    return {str(key).lower(): str(value) for key, value in dict(headers).items()}


def size_from_headers(headers: Mapping[str, str]) -> tuple[int | None, str]:
    """Return ``(size, source)`` from a HEAD response, or ``(None, "unavailable")``.

    Preference order follows what the service actually guarantees for a large file: the LFS
    ``x-linked-size`` first, then ``Content-Length``.  A server that provides neither is reported as
    *unavailable* — never as a match and never as a mismatch.
    """
    for name in ("x-linked-size", "content-length"):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            return int(str(raw).strip()), name
        except (TypeError, ValueError):
            continue
    return None, "unavailable"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects, so a HEAD stays on the service and never reaches the CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _http_get_json(uri: str, token: str | None, timeout: int) -> tuple[int, Any]:
    """GET ``uri`` and decode JSON.  Raises the transport error for the caller to classify."""
    request = urllib.request.Request(uri, headers={"User-Agent": "AgentRec-X-artifact-probe/1"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(_TokenSafeRedirectHandler)
    with opener.open(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _http_head(uri: str, token: str | None, timeout: int) -> tuple[int, dict[str, str]]:
    """Issue a HEAD and return ``(status, headers)`` — headers only, never a body.

    Redirects are deliberately not followed: the service's own response carries the size hint
    (``x-linked-size``), and stopping here guarantees the probe cannot pull an artifact body.
    A 3xx is reported as reachable, because it means the path exists.
    """
    request = urllib.request.Request(
        uri, method="HEAD", headers={"User-Agent": "AgentRec-X-artifact-probe/1"}
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, normalise_headers(response.headers)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            return exc.code, normalise_headers(exc.headers or {})
        raise


@dataclass
class ProbeResult:
    """One probe check's outcome."""

    label: str
    outcome: str
    detail: str
    size_note: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome not in (PROBE_OK, PROBE_SKIPPED)


def _report(result: ProbeResult) -> None:
    print(f"  [{result.outcome:>22s}] {result.label}")
    if result.detail:
        print(f"      {result.detail}")
    if result.size_note:
        print(f"      {result.size_note}")


def probe_repository(
    manifest: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    timeout: int = PROBE_TIMEOUT_SECONDS,
) -> tuple[DurableLocation | None, list[ProbeResult]]:
    """Checks A and B: repository access and revision/tree access.  No artifact bytes."""
    results: list[ProbeResult] = []
    repo_id = environ.get(ENV_REPO_ID)
    if not repo_id:
        results.append(
            ProbeResult("A. repo access", PROBE_SKIPPED, f"{ENV_REPO_ID} is not set")
        )
        results.append(
            ProbeResult("B. revision/tree access", PROBE_SKIPPED, "no repository to query")
        )
        return None, results

    repo_type = manifest_repo_type(manifest)
    try:
        repo_id = check_repo_id(repo_id)
    except BlockedError as exc:
        results.append(ProbeResult("A. repo access", PROBE_SKIPPED, str(exc)))
        results.append(ProbeResult("B. revision/tree access", PROBE_SKIPPED, "no usable repo_id"))
        return None, results

    revision = environ.get(ENV_REVISION) or DEFAULT_REVISION
    location = DurableLocation(
        endpoint=manifest_endpoint(manifest),
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
        remote_path="",
    )
    token = environ.get(ENV_TOKEN) or None

    try:
        status, payload = _http_get_json(location.api_uri, token, timeout)
    except Exception as exc:  # noqa: BLE001 - classified, never re-raised raw
        outcome, detail = classify_transport_error(exc)
        results.append(ProbeResult("A. repo access", outcome, f"GET {location.api_uri}: {detail}"))
    else:
        served = payload.get("id") if isinstance(payload, dict) else None
        results.append(
            ProbeResult(
                "A. repo access",
                PROBE_OK,
                f"GET {location.api_uri}: HTTP {status}, id={served!r}",
            )
        )

    tree_uri = location.tree_uri()
    try:
        status, payload = _http_get_json(tree_uri, token, timeout)
    except Exception as exc:  # noqa: BLE001
        outcome, detail = classify_transport_error(exc)
        results.append(ProbeResult("B. revision/tree access", outcome, f"GET {tree_uri}: {detail}"))
    else:
        count = len(payload) if isinstance(payload, list) else "?"
        results.append(
            ProbeResult(
                "B. revision/tree access",
                PROBE_OK,
                f"GET {tree_uri}: HTTP {status}, {count} top-level entries",
            )
        )
    return location, results


def probe_artifact(
    manifest: Mapping[str, Any],
    entry: RecoveryEntry,
    environ: Mapping[str, str],
    *,
    timeout: int = PROBE_TIMEOUT_SECONDS,
) -> ProbeResult:
    """Check C: the artifact's own path, via HEAD only (never a body)."""
    label = f"C. remote artifact path ({entry.role})"
    if not entry.remote_path:
        return ProbeResult(
            label,
            PROBE_SKIPPED,
            f"durable.remote_path is not recorded yet (status {entry.durable_status}); "
            "upload the file and record the path, then re-run the probe",
        )
    try:
        location = resolve_durable(manifest, entry, environ)
    except BlockedError as exc:
        return ProbeResult(label, PROBE_SKIPPED, str(exc))

    token = environ.get(ENV_TOKEN) or None
    try:
        status, headers = _http_head(location.uri, token, timeout)
    except Exception as exc:  # noqa: BLE001
        outcome, detail = classify_transport_error(exc)
        return ProbeResult(label, outcome, f"HEAD {location.uri}: {detail}")

    size, source = size_from_headers(headers)
    if size is None:
        note = (
            "size: unavailable -- the server returned no x-linked-size or Content-Length, so the "
            "manifest size_bytes could not be checked (this is neither a pass nor a failure)"
        )
    elif size == entry.size_bytes:
        note = f"size: {size} ({source}) == manifest size_bytes {entry.size_bytes} -- MATCH"
    else:
        note = f"size: {size} ({source}) != manifest size_bytes {entry.size_bytes} -- MISMATCH"
    return ProbeResult(label, PROBE_OK, f"HEAD {location.uri}: HTTP {status}", note)


def probe_exit_code(results: Sequence[ProbeResult], size_mismatch: bool) -> int:
    """Collapse probe results into the tool's exit-code contract."""
    outcomes = {result.outcome for result in results}
    if outcomes & {PROBE_TIMEOUT, PROBE_CONNECTION}:
        return EXIT_NETWORK
    if size_mismatch or outcomes & {PROBE_AUTH, PROBE_NOT_FOUND, PROBE_HTTP}:
        return EXIT_VERIFY
    if PROBE_SKIPPED in outcomes:
        return EXIT_BLOCKED
    return EXIT_OK


def run_probe(
    entries: Sequence[RecoveryEntry],
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    timeout: int = PROBE_TIMEOUT_SECONDS,
) -> int:
    """Lightweight network validation: access, revision, and each artifact's path.  No bodies."""
    recoverability = manifest.get("recoverability", {})
    print("PROBE - lightweight network validation; no artifact body is transferred")
    print(f"  backend   : {recoverability.get('backend')}")
    try:
        print(f"  repo_type : {manifest_repo_type(manifest)}")
    except RecoveryError as exc:
        print(f"  repo_type : INVALID -- {exc}")
        return EXIT_BLOCKED
    environ = load_environment(repo_root)
    print(f"  repo_id   : {environ.get(ENV_REPO_ID) or '(unset)'}")
    print(f"  revision  : {environ.get(ENV_REVISION) or DEFAULT_REVISION}")
    print(
        "  token     : "
        + ("present (value never printed)" if environ.get(ENV_TOKEN) else f"absent ({ENV_TOKEN})")
    )
    print()

    _, results = probe_repository(manifest, environ, timeout=timeout)
    for entry in entries:
        results.append(probe_artifact(manifest, entry, environ, timeout=timeout))
    for result in results:
        _report(result)

    mismatch = any("MISMATCH" in result.size_note for result in results)
    code = probe_exit_code(results, mismatch)
    print()
    if code == EXIT_OK:
        print("PROBE PASS - access, revision and every artifact path verified")
    else:
        failing = [r for r in results if r.failed]
        print(f"PROBE RESULT: exit {code}")
        for result in failing:
            remedy = PROBE_REMEDY.get(result.outcome)
            if remedy:
                print(f"  {result.outcome}: {remedy}")
    if any(PROBE_SKIPPED == result.outcome for result in results) and code == EXIT_BLOCKED:
        print("  (skipped checks are configuration gaps, not failures)")
    return code


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def run_list(entries: Sequence[RecoveryEntry], manifest: Mapping[str, Any]) -> int:
    """Print the recoverability view: identity, derived location and status.

    Offline, and it never prints a credential — only whether one is present.
    """
    recoverability = manifest.get("recoverability", {})
    endpoint = manifest_endpoint(manifest)
    repo_type = manifest_repo_type(manifest)
    environ = load_environment(REPO_ROOT)
    repo_id = environ.get(ENV_REPO_ID)
    revision = environ.get(ENV_REVISION) or DEFAULT_REVISION

    print(f"canonical manifest : {DEFAULT_MANIFEST}")
    print(f"backend            : {recoverability.get('backend')}")
    print(f"endpoint           : {endpoint}")
    print(f"repo_type          : {repo_type}")
    print(
        "repo_id            : "
        + (repo_id if repo_id else f"NOT SET -- export {ENV_REPO_ID} or record durable.repo_id")
    )
    print(f"revision           : {revision}")
    print(
        "token              : "
        + ("present (value never printed)" if environ.get(ENV_TOKEN) else f"absent ({ENV_TOKEN})")
    )
    print()
    for entry in entries:
        print(f"{entry.role}  [{entry.durable_status}]")
        print(f"  local path   : {entry.path}")
        print(f"  size / sha256: {entry.size_bytes} / {entry.sha256}")
        try:
            location = resolve_durable(manifest, entry, environ)
        except BlockedError as exc:
            print(f"  repo_id      : {repo_id or '(unset)'}")
            print(f"  revision     : {revision}")
            print(f"  remote_path  : {entry.remote_path or '(unset)'}")
            print(f"  derived URI  : NOT FETCHABLE -- {exc}")
        else:
            print(f"  repo_id      : {location.repo_id}")
            print(f"  revision     : {location.revision}")
            print(f"  remote_path  : {location.remote_path}")
            print(f"  derived URI  : {location.uri}")
        print()
    return EXIT_OK


def run_verify(
    entries: Sequence[RecoveryEntry],
    repo_root: Path,
) -> int:
    """Verify local copies.  Offline; no fallback and no download."""
    failures: list[str] = []
    for entry in entries:
        target = entry.absolute(repo_root)
        try:
            verify_file(target, entry)
        except VerificationError as exc:
            failures.append(str(exc))
            print(f"FAIL  {entry.role}: {exc}")
        else:
            print(f"OK    {entry.role}: {target}")
    if failures:
        print(f"\n{len(failures)} artifact(s) failed verification")
        return EXIT_VERIFY
    print(f"\nall {len(entries)} artifact(s) verified")
    return EXIT_OK


def run_fetch(
    entries: Sequence[RecoveryEntry],
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    roles: Sequence[str],
) -> int:
    """Fetch and verify the selected artifacts.  Requires an explicit selection."""
    if not roles:
        raise RecoveryError(
            "fetch requires an explicit selection: pass --role <role> (repeatable) or --all"
        )
    environ = load_environment(repo_root)
    token = environ.get(ENV_TOKEN) or None
    total = sum(entry.size_bytes for entry in entries)
    print(f"fetching {len(entries)} artifact(s), {total} bytes total")
    for entry in entries:
        location = resolve_durable(manifest, entry, environ)
        print(f"  {entry.role} <- {location.uri}")
        target = fetch_entry(entry, location, repo_root, token)
        print(f"  {entry.role} verified -> {target}")
    return EXIT_OK


def run_smoke(
    entries: Sequence[RecoveryEntry],
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    k: int = 10,
) -> int:
    """The fresh-clone contract: absent locally, fetched, verified, loaded, served.

    Step 1 is a refusal, not a warning.  If the target files already exist, the run stops:
    a copy that never crossed the network cannot demonstrate recoverability.
    """
    present = [e for e in entries if e.absolute(repo_root).exists()]
    if present:
        names = ", ".join(f"{e.role} ({e.path})" for e in present)
        raise BlockedError(
            f"fresh-clone smoke requires the target artifacts to be ABSENT, but found: {names}. "
            f"Restoring from a development machine's own directory is not recoverability "
            f"evidence. Move them away (or run in a fresh clone), then retry."
        )

    environ = load_environment(repo_root)
    token = environ.get(ENV_TOKEN) or None

    print("[1/4] fetch from the durable URI")
    for entry in entries:
        location = resolve_durable(manifest, entry, environ)
        print(f"      {entry.role} <- {location.uri}")
        fetch_entry(entry, location, repo_root, token)

    print("[2/4] SHA256 verification (re-checked against the canonical manifest)")
    for entry in entries:
        verify_file(entry.absolute(repo_root), entry)
        print(f"      {entry.role} OK {entry.sha256}")

    print("[3/4] load TwoTowerInferenceEngine with the canonical mappings")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from recommendation.inference.twotower import (  # noqa: PLC0415 - opt-in path
        TwoTowerInferenceConfig,
        TwoTowerInferenceEngine,
    )

    by_role = {entry.role: entry for entry in entries}
    checkpoint = by_role["two_tower_checkpoint"].absolute(repo_root)
    mappings = by_role["mappings"].absolute(repo_root)
    engine = TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(checkpoint_path=checkpoint, mappings_path=mappings)
    )
    if not engine.is_ready():
        raise RecoveryError("TwoTowerInferenceEngine loaded but reports not ready")
    print(f"      ready, {engine.num_items} catalogue items")

    print(f"[4/4] recommend(k={k}) and identity round-trip")
    history = list(range(1, 6))
    result = engine.recommend(history, k=k)
    items = list(result.recommendations)
    if len(items) != k:
        raise RecoveryError(f"recommend(k={k}) returned {len(items)} items")
    if any(item.item_id == 0 for item in items):
        raise RecoveryError("PAD (item_id 0) was returned as a recommendation")
    for item in items:
        resolved = engine.parent_asin_to_item_id(item.parent_asin)
        if resolved != item.item_id:
            raise RecoveryError(
                f"identity mismatch for {item.parent_asin!r}: {resolved} != {item.item_id}"
            )
    print(f"      {len(items)} items, PAD absent, parent_asin round-trip OK")
    print("\nSMOKE PASS - recovered without any local copy of the artifacts")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    common.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    common.add_argument(
        "--role",
        action="append",
        default=[],
        help="artifact role to act on (repeatable)",
    )
    common.add_argument(
        "--all",
        action="store_true",
        help="act on every recoverability artifact (required for fetch without --role)",
    )
    common.add_argument(
        "--timeout",
        type=int,
        default=PROBE_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds for probe (default {PROBE_TIMEOUT_SECONDS})",
    )

    parser = argparse.ArgumentParser(
        prog="fetch_artifacts.py",
        description=(
            "List, verify, fetch or smoke-test the artifacts declared in the canonical manifest. "
            "Only 'fetch' and 'smoke' use the network."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", parents=[common], help="show the recoverability table (offline)")
    sub.add_parser(
        "verify", parents=[common], help="verify local copies against the manifest (offline)"
    )
    sub.add_parser("fetch", parents=[common], help="download and verify (network)")
    sub.add_parser(
        "probe",
        parents=[common],
        help="lightweight network validation: access, revision, artifact paths (network, no bodies)",
    )
    sub.add_parser(
        "smoke",
        parents=[common],
        help="fresh-clone smoke: absent, fetch, verify, load, serve (network)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        entries = load_recovery_entries(manifest)
        repo_root = Path(args.repo_root)

        if args.command == "list":
            return run_list(entries, manifest)
        if args.command == "verify":
            return run_verify(select_entries(entries, args.role), repo_root)
        if args.command == "fetch":
            selected = select_entries(entries, args.role) if args.role else None
            if selected is None:
                if not args.all:
                    raise RecoveryError(
                        "fetch requires --role <role> or --all; refusing to download by default"
                    )
                selected = entries
            return run_fetch(selected, manifest, repo_root, roles=args.role or ["--all"])
        if args.command == "probe":
            return run_probe(
                select_entries(entries, args.role),
                manifest,
                repo_root,
                timeout=args.timeout,
            )
        if args.command == "smoke":
            selected = (
                select_entries(entries, args.role) if args.role else select_entries(
                    entries, (), consumer=SMOKE_CONSUMER
                )
            )
            return run_smoke(selected, manifest, repo_root)
    except BlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    except TransportError as exc:
        print(f"TRANSPORT FAILED: {exc}", file=sys.stderr)
        return EXIT_NETWORK
    except RecoveryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
