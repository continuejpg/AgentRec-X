"""Tests for the local demo launcher support (Milestone 11.5).

Scope: the launcher itself.  These tests deliberately do **not** touch the accepted
Milestone 11 behaviour -- ``tests/test_demo_*.py``, ``tests/test_api.py`` and
``experiments/web_demo_smoke.py`` remain unchanged and authoritative for the service.

Everything here is offline and fast:

* synthetic artifact trees under ``tmp_path`` (never the real ~832 MB artifacts);
* no sockets bound, no server started, no subprocess that starts a service;
* the port classifier is exercised against a real loopback listener and against a
  monkeypatched probe, but never signals anything.

The one committed, non-synthetic thing under test is
``config/demo_runtime_artifacts.json`` itself, whose digests must agree with the
runtime's own accepted constants.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

from recommendation import local_demo
from recommendation.api.app import ACCEPTED_CHECKPOINT_SHA256

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "config" / "demo_runtime_artifacts.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def make_spec(role: str, payload: bytes, *, path: str | None = None) -> local_demo.ArtifactSpec:
    """Build an ArtifactSpec describing ``payload`` at ``path``."""
    return local_demo.ArtifactSpec(
        role=role,
        path=path if path is not None else f"artifacts/{role}.bin",
        size_bytes=len(payload),
        sha256=_sha256(payload),
        description=f"synthetic {role}",
    )


def write_spec(root: Path, spec: local_demo.ArtifactSpec, payload: bytes) -> Path:
    """Materialise ``payload`` at ``spec``'s location under ``root``."""
    target = root / spec.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def write_manifest(path: Path, specs: list[local_demo.ArtifactSpec]) -> Path:
    """Write a structurally valid manifest describing ``specs``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": local_demo.ARTIFACT_MANIFEST_FORMAT,
                "category": "Synthetic",
                "artifacts": [
                    {
                        "role": spec.role,
                        "path": spec.path,
                        "size_bytes": spec.size_bytes,
                        "sha256": spec.sha256,
                        "description": spec.description,
                    }
                    for spec in specs
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def full_role_specs(root: Path) -> tuple[list[local_demo.ArtifactSpec], list[bytes]]:
    """Build one correct spec per expected role, with distinct payloads."""
    specs: list[local_demo.ArtifactSpec] = []
    payloads: list[bytes] = []
    for index, role in enumerate(local_demo.EXPECTED_ARTIFACT_ROLES):
        payload = f"synthetic-{role}-payload".encode() * (index + 1)
        spec = make_spec(role, payload, path=f"tree/{role}.bin")
        specs.append(spec)
        payloads.append(payload)
    return specs, payloads


# --------------------------------------------------------------------------- #
# committed manifest (the only non-synthetic input)
# --------------------------------------------------------------------------- #


def test_committed_manifest_exists_and_is_valid() -> None:
    """The repository ships a structurally valid artifact manifest."""
    specs = local_demo.load_artifact_specs(MANIFEST)
    assert len(specs) == len(local_demo.EXPECTED_ARTIFACT_ROLES)
    assert {spec.role for spec in specs} == set(local_demo.EXPECTED_ARTIFACT_ROLES)


def test_committed_manifest_paths_are_repo_relative() -> None:
    """Every manifest path is relative and stays inside the repository."""
    for spec in local_demo.load_artifact_specs(MANIFEST):
        path = Path(spec.path)
        assert not path.is_absolute(), spec.path
        assert ".." not in path.parts, spec.path


def test_committed_manifest_checkpoint_digest_matches_runtime_constant() -> None:
    """The manifest and the runtime must agree on the accepted checkpoint identity.

    Prevents the two copies of the accepted digest from drifting apart silently.
    """
    by_role = {spec.role: spec for spec in local_demo.load_artifact_specs(MANIFEST)}
    assert by_role["checkpoint"].sha256 == ACCEPTED_CHECKPOINT_SHA256


def test_committed_manifest_agrees_with_real_artifact_sizes() -> None:
    """Recorded sizes match the artifacts on disk when they are present.

    Skips (rather than fails) on a checkout without the git-ignored artifacts, which
    is the same convention the accepted integration tests use.
    """
    specs = local_demo.load_artifact_specs(MANIFEST)
    missing = [spec.path for spec in specs if not (REPO_ROOT / spec.path).is_file()]
    if missing:
        pytest.skip(f"accepted artifacts not present: {missing}")
    for spec in specs:
        assert (REPO_ROOT / spec.path).stat().st_size == spec.size_bytes, spec.path


# --------------------------------------------------------------------------- #
# manifest loading / validation
# --------------------------------------------------------------------------- #


def test_load_artifact_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(local_demo.LauncherError, match="not found"):
        local_demo.load_artifact_manifest(tmp_path / "nope.json")


def test_load_artifact_manifest_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": "wrong", "artifacts": []}), encoding="utf-8")
    with pytest.raises(local_demo.LauncherError, match="format"):
        local_demo.load_artifact_manifest(path)


def test_load_artifact_manifest_rejects_missing_role(tmp_path: Path) -> None:
    """Dropping a role must fail loudly so the launcher cannot silently check less."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format": local_demo.ARTIFACT_MANIFEST_FORMAT,
                "artifacts": [
                    {"role": "checkpoint", "path": "a", "size_bytes": 1, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(local_demo.LauncherError, match="roles"):
        local_demo.load_artifact_manifest(path)


def _manifest_with_corruption(tmp_path: Path, **corrupt: object) -> Path:
    """Write a complete, otherwise-valid manifest with one field overridden.

    The manifest must still carry every expected role, otherwise role validation fires
    before the field validation under test.
    """
    payload: dict[str, object] = {
        "format": local_demo.ARTIFACT_MANIFEST_FORMAT,
        "category": "Synthetic",
        "artifacts": [
            {
                "role": role,
                "path": f"tree/{role}.bin",
                "size_bytes": 10,
                "sha256": "a" * 64,
            }
            for role in local_demo.EXPECTED_ARTIFACT_ROLES
        ],
    }
    entries = payload["artifacts"]
    assert isinstance(entries, list)
    entries[0].update(corrupt)  # type: ignore[union-attr]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("bad_path", ["/etc/passwd", "../outside.bin"], ids=["absolute", "dotdot"])
def test_load_artifact_specs_rejects_escaping_paths(tmp_path: Path, bad_path: str) -> None:
    path = _manifest_with_corruption(tmp_path, path=bad_path)
    with pytest.raises(local_demo.LauncherError, match="repository-relative|escape"):
        local_demo.load_artifact_specs(path)


def test_load_artifact_specs_rejects_bad_digest(tmp_path: Path) -> None:
    path = _manifest_with_corruption(tmp_path, sha256="not-a-digest")
    with pytest.raises(local_demo.LauncherError, match="sha256"):
        local_demo.load_artifact_specs(path)


def test_load_artifact_specs_rejects_non_positive_size(tmp_path: Path) -> None:
    path = _manifest_with_corruption(tmp_path, size_bytes=0)
    with pytest.raises(local_demo.LauncherError, match="positive size"):
        local_demo.load_artifact_specs(path)


def test_load_artifact_specs_round_trips(tmp_path: Path) -> None:
    specs, _ = full_role_specs(tmp_path)
    path = write_manifest(tmp_path / "m.json", specs)
    loaded = local_demo.load_artifact_specs(path)
    assert [spec.role for spec in loaded] == [spec.role for spec in specs]
    assert [spec.sha256 for spec in loaded] == [spec.sha256 for spec in specs]


# --------------------------------------------------------------------------- #
# tiered artifact verification
# --------------------------------------------------------------------------- #


def test_fast_verification_passes_on_exact_files(tmp_path: Path) -> None:
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    report = local_demo.verify_artifacts(specs, deep=False, repo_root=tmp_path)
    assert report.ok
    assert all(check.status == "ok" for check in report.checks)
    assert report.mode == "fast"


def test_deep_verification_passes_and_hashes(tmp_path: Path) -> None:
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    report = local_demo.verify_artifacts(specs, deep=True, repo_root=tmp_path)
    assert report.ok
    assert report.mode == "deep"
    assert all(check.actual_sha256 == check_sha for check, check_sha in
               zip(report.checks, [spec.sha256 for spec in specs]))


def test_fast_verification_detects_truncated_file(tmp_path: Path) -> None:
    """A truncated artifact is caught by exact size, which is the realistic case."""
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    truncated = specs[0]
    (tmp_path / truncated.path).write_bytes(payloads[0][:-1])

    report = local_demo.verify_artifacts(specs, deep=False, repo_root=tmp_path)
    assert not report.ok
    assert [check.role for check in report.failures] == ["checkpoint"]
    assert report.failures[0].status == "size_mismatch"


def test_fast_verification_accepts_corrupted_content_of_the_right_size(tmp_path: Path) -> None:
    """Fast mode is explicitly a size check: same-size corruption is NOT detected.

    This documents the tier boundary rather than asserting a bug.
    """
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    target = specs[0]
    (tmp_path / target.path).write_bytes(b"X" * len(payloads[0]))
    assert local_demo.verify_artifacts(specs, deep=False, repo_root=tmp_path).ok


def test_deep_verification_detects_corrupted_content_of_the_right_size(tmp_path: Path) -> None:
    """Deep mode catches the same-size substitution that fast mode cannot."""
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    target = specs[0]
    (tmp_path / target.path).write_bytes(b"X" * len(payloads[0]))

    report = local_demo.verify_artifacts(specs, deep=True, repo_root=tmp_path)
    assert not report.ok
    assert report.failures[0].status == "sha256_mismatch"
    assert report.failures[0].actual_sha256 == _sha256(b"X" * len(payloads[0]))


def test_verification_reports_every_missing_artifact_not_just_the_first(tmp_path: Path) -> None:
    """One problem must not hide the state of the other artifacts."""
    specs, _ = full_role_specs(tmp_path)
    report = local_demo.verify_artifacts(specs, deep=False, repo_root=tmp_path)
    assert not report.ok
    assert len(report.failures) == len(specs)
    assert all(check.status == "missing" for check in report.failures)


def test_verification_rejects_a_directory_in_place_of_a_file(tmp_path: Path) -> None:
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs[1:], payloads[1:]):
        write_spec(tmp_path, spec, payload)
    (tmp_path / specs[0].path).mkdir(parents=True)

    report = local_demo.verify_artifacts(specs, deep=False, repo_root=tmp_path)
    assert not report.ok
    assert report.failures[0].status == "not_a_regular_file"


def test_verification_report_is_json_serialisable(tmp_path: Path) -> None:
    specs, payloads = full_role_specs(tmp_path)
    for spec, payload in zip(specs, payloads):
        write_spec(tmp_path, spec, payload)
    report = local_demo.verify_artifacts(specs, deep=True, repo_root=tmp_path)
    payload = json.dumps(local_demo.report_to_dict(report))
    assert json.loads(payload)["ok"] is True


# --------------------------------------------------------------------------- #
# repository-root discovery
# --------------------------------------------------------------------------- #


def test_resolve_repo_root_finds_the_real_repository() -> None:
    assert local_demo.resolve_repo_root() == REPO_ROOT


def test_resolve_repo_root_is_independent_of_nested_start(tmp_path: Path) -> None:
    nested = REPO_ROOT / "recommendation" / "api"
    assert local_demo.resolve_repo_root(nested) == REPO_ROOT


def _isolate_module_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's ``__file__`` into an isolated tree with no repository markers."""
    fake_module = tmp_path / "elsewhere" / "recommendation" / "local_demo.py"
    fake_module.parent.mkdir(parents=True, exist_ok=True)
    fake_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(local_demo, "__file__", str(fake_module))


def test_resolve_repo_root_rejects_a_non_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree that is not an AgentRec-X checkout must be rejected, not guessed."""
    _isolate_module_location(tmp_path, monkeypatch)
    with pytest.raises(local_demo.LauncherError, match="could not locate"):
        local_demo.resolve_repo_root(tmp_path / "elsewhere")


def test_resolve_repo_root_ignores_the_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CWD is never consulted, even when it *is* the real repository."""
    _isolate_module_location(tmp_path, monkeypatch)
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(local_demo.LauncherError, match="could not locate"):
        local_demo.resolve_repo_root()


# --------------------------------------------------------------------------- #
# OMP_NUM_THREADS normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("0", "8"),
        ("", "8"),
        ("   ", "8"),
        ("-1", "8"),
        ("not-a-number", "8"),
        ("4", "4"),
        (" 16 ", "16"),
    ],
    ids=["unset", "zero", "empty", "spaces", "negative", "junk", "positive", "padded"],
)
def test_normalise_omp_num_threads(value: str | None, expected: str | None) -> None:
    assert local_demo.normalise_omp_num_threads(value) == expected


def test_normalise_omp_num_threads_preserves_a_valid_user_setting() -> None:
    """An explicit positive value is the user's choice and must survive untouched.

    It must not be silently dropped to unset, which would change the thread count
    behind the user's back.
    """
    assert local_demo.normalise_omp_num_threads("4") == "4"
    assert local_demo.normalise_omp_num_threads("24") == "24"


def test_normalise_omp_num_threads_libgomp_values_are_safe() -> None:
    """Whatever it returns must be acceptable to libgomp (unset or a positive int)."""
    for raw in ("0", "", "-5", "junk", "3"):
        result = local_demo.normalise_omp_num_threads(raw)
        assert result is None or int(result) >= 1


# --------------------------------------------------------------------------- #
# environment report
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not local_demo.environment_info().venv_exists,
    reason=(
        "this checkout has no project virtualenv; run ./scripts/setup_demo.sh first "
        "(a fresh clone has no .venv, and this test describes the venv's stack)"
    ),
)
def test_environment_info_reports_the_accepted_cpu_stack() -> None:
    env = local_demo.environment_info()
    assert env.python_supported is True
    assert env.venv_exists is True
    assert env.numpy_version is not None
    assert env.torch_version is not None
    # CPU-only contract, exactly as the milestone requires.
    assert env.torch_cuda_version is None
    assert env.torch_cuda_available is False
    assert env.torch_has_nvidia_packages is False
    assert env.device == "cpu"
    assert env.cpu_only_ok is True
    assert env.ok is True


def test_environment_info_is_json_serialisable() -> None:
    payload = json.dumps(local_demo.environment_info().to_dict())
    assert json.loads(payload)["device"] == "cpu"


def test_environment_info_records_import_failures_without_raising(monkeypatch) -> None:
    """A broken dependency must produce a report, not a traceback."""
    import builtins

    real_import = builtins.__import__

    def deny_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("synthetic numpy failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_numpy)
    env = local_demo.environment_info()
    assert env.numpy_version is None
    assert "synthetic numpy failure" in (env.numpy_error or "")
    assert env.ok is False


def test_virtualenv_is_self_contained() -> None:
    """The accepted environment does not inherit system site-packages."""
    env = local_demo.environment_info(import_numpy=False, import_torch=False)
    assert env.venv_self_contained is False
    assert env.venv_python.endswith(".venv/bin/python")


def test_read_venv_self_contained_parses_the_flag(tmp_path: Path) -> None:
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("home = /usr/bin\ninclude-system-site-packages = true\n", encoding="utf-8")
    assert local_demo._read_venv_self_contained(cfg) is True
    cfg.write_text("home = /usr/bin\ninclude-system-site-packages = false\n", encoding="utf-8")
    assert local_demo._read_venv_self_contained(cfg) is False
    assert local_demo._read_venv_self_contained(tmp_path / "absent.cfg") is False


# --------------------------------------------------------------------------- #
# port classification
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    """Return a currently free loopback port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_classify_port_reports_free_when_nothing_listens() -> None:
    status = local_demo.classify_port("127.0.0.1", _free_port())
    assert status.state == "free"
    assert status.free is True
    assert status.is_agentrecx is False


def test_classify_port_reports_foreign_for_a_non_http_listener() -> None:
    """A listener that is not an AgentRec-X server must be classified, not killed."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])
    try:
        status = local_demo.classify_port("127.0.0.1", port, timeout=1.0)
        assert status.state == "foreign"
        assert status.free is False
        assert status.is_agentrecx is False
    finally:
        server.close()


def test_classify_port_reports_foreign_for_a_plain_http_server() -> None:
    """A real HTTP server that is not AgentRec-X is foreign."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"hello": "world"}')

        def log_message(self, *args):  # silence the test output
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = local_demo.classify_port("127.0.0.1", port, timeout=2.0)
        assert status.state == "foreign"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_classify_port_detects_a_real_agentrecx_health_contract() -> None:
    """A listener answering the AgentRec-X health contract is recognised as such.

    Uses a stub server rather than the real demo so the unit suite stays offline and
    fast; the real server is covered by the launch smoke.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            body = {
                "/v1/demo/health": {
                    "status": "ok",
                    "model_loaded": True,
                    "metadata_loaded": True,
                    "demo_ready": True,
                    "profiles": 3,
                },
                "/v1/model": {"model_type": "SASRec", "device": "cpu"},
            }.get(self.path, {})
            payload = json.dumps(body).encode()
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = local_demo.classify_port("127.0.0.1", port, timeout=2.0)
        assert status.state == "agentrecx_running"
        assert status.is_agentrecx is True
        assert status.evidence["profiles"] == 3
        assert status.evidence["device"] == "cpu"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_probe_host_maps_wildcard_bindings_to_loopback() -> None:
    assert local_demo._probe_host("0.0.0.0") == "127.0.0.1"
    assert local_demo._probe_host("") == "127.0.0.1"
    assert local_demo._probe_host("::") == "127.0.0.1"
    assert local_demo._probe_host("192.168.1.5") == "192.168.1.5"


def test_port_status_is_json_serialisable() -> None:
    status = local_demo.classify_port("127.0.0.1", _free_port())
    assert json.loads(json.dumps(status.to_dict()))["state"] == "free"


# --------------------------------------------------------------------------- #
# preflight aggregation
# --------------------------------------------------------------------------- #


def test_run_preflight_fails_when_the_virtualenv_is_absent(tmp_path: Path) -> None:
    result = local_demo.run_preflight(port=None, repo_root=tmp_path, import_torch=False)
    assert result.ok is False
    assert any("virtualenv interpreter not found" in problem for problem in result.problems)


def test_run_preflight_reports_port_state(tmp_path: Path) -> None:
    result = local_demo.run_preflight(
        host="127.0.0.1", port=_free_port(), repo_root=tmp_path, import_torch=False
    )
    assert result.port is not None
    assert result.port.state == "free"


def test_run_preflight_keeps_every_port_outcome_out_of_problems(tmp_path: Path) -> None:
    """No port outcome may appear in ``problems``.

    ``problems`` means "the environment or artifacts are broken".  A busy port means
    neither, and the launcher must be able to reach its own port branch: exit 0 for an
    existing AgentRec-X instance, exit 3 for a foreign process.  If either were listed
    here, preflight would fail first with exit 1 and those branches would be dead.
    """
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])
    try:
        result = local_demo.run_preflight(
            host="127.0.0.1", port=port, repo_root=tmp_path, import_torch=False
        )
        assert result.port is not None and result.port.state == "foreign"
        assert not any(
            "not an AgentRec-X" in problem or "already serving" in problem
            for problem in result.problems
        ), result.problems
    finally:
        server.close()


def test_run_preflight_records_an_agentrecx_listener_without_failing_it(
    tmp_path: Path,
) -> None:
    """An existing AgentRec-X instance is *recorded* on the port, not raised as a problem."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            body = {
                "/v1/demo/health": {
                    "status": "ok",
                    "model_loaded": True,
                    "metadata_loaded": True,
                    "demo_ready": True,
                    "profiles": 3,
                },
                "/v1/model": {"model_type": "SASRec", "device": "cpu"},
            }.get(self.path, {})
            payload = json.dumps(body).encode()
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = local_demo.run_preflight(
            host="127.0.0.1", port=port, repo_root=tmp_path, import_torch=False
        )
        assert result.port is not None and result.port.state == "agentrecx_running"
        assert not any("already serving" in problem for problem in result.problems)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_run_preflight_is_json_serialisable(tmp_path: Path) -> None:
    result = local_demo.run_preflight(port=None, repo_root=tmp_path, import_torch=False)
    payload = json.loads(json.dumps(result.to_dict()))
    assert "problems" in payload and "artifacts" in payload


# --------------------------------------------------------------------------- #
# Windows one-click wrapper (delegation only)
# --------------------------------------------------------------------------- #

WIN_CMD = REPO_ROOT / "start-agentrecx.cmd"
WIN_RESOLVER = REPO_ROOT / "scripts" / "windows" / "start_agentrecx.ps1"
WIN_BROWSER = REPO_ROOT / "scripts" / "windows" / "open_demo_browser.ps1"


def _cmd_text() -> str:
    return WIN_CMD.read_text(encoding="utf-8", errors="replace")


def _cmd_code_lines() -> str:
    """The .cmd with REM comment lines removed (checks target executable lines)."""
    return "\n".join(
        line for line in _cmd_text().splitlines()
        if not line.strip().lower().startswith("rem")
    )


def test_windows_wrapper_files_exist() -> None:
    """The one-click entry point, its resolver and its browser helper are present."""
    assert WIN_CMD.is_file(), "start-agentrecx.cmd is missing"
    assert WIN_RESOLVER.is_file(), "scripts/windows/start_agentrecx.ps1 is missing"
    assert WIN_BROWSER.is_file(), "scripts/windows/open_demo_browser.ps1 is missing"


def test_windows_wrapper_never_uses_the_current_directory() -> None:
    """Regression: a UNC current directory is illegal in cmd.exe.

    The real failure printed "UNC paths are not supported. Defaulting to Windows
    directory." Resolution must come from %~dp0 only.
    """
    code = _cmd_code_lines()
    assert "%~dp0" in code
    for forbidden in ("%CD%", "cd /d", "pushd"):
        assert forbidden.lower() not in code.lower(), (
            f"the wrapper must not depend on the current directory: {forbidden!r}"
        )


def test_windows_wrapper_does_not_call_wslpath_or_parse_unc_itself() -> None:
    """Regression: the old wrapper called wslpath inside `for /f` and produced a
    garbled repository path. Conversion must be lexical only, all WSL discovery
    must live in PowerShell, and the shim must not parse the UNC share at all.
    """
    code = _cmd_code_lines()
    assert "wslpath" not in code.lower(), "batch must not call wslpath"
    # Two for /f loops are allowed, and BOTH must iterate a plain string or file.
    # What is banned is command substitution (the backquoted `usebackq` form),
    # because running wsl.exe inside a for /f body is exactly what garbled the
    # path in the failed first version.
    for_lines = [l.strip() for l in code.splitlines() if "for /f" in l.lower()]
    assert len(for_lines) == 2, for_lines
    for line in for_lines:
        assert "`" not in line, f"no command substitution allowed: {line}"
    assert any("usebackq tokens=1* delims==" in l for l in for_lines), "response reader"
    assert any("tokens=1,* delims=" in l for l in for_lines), "share splitter"
    # wsl.exe must be invoked exactly once, and only with a verified distro.
    wsl_invocations = [
        line for line in code.splitlines()
        if line.strip().lower().startswith("wsl.exe ")
    ]
    assert len(wsl_invocations) == 1, wsl_invocations
    assert "-d %AGENTRECX_DISTRO_OK%" in wsl_invocations[0]


def test_windows_wrapper_derives_prefix_and_share_separately() -> None:
    """A \\\\wsl.localhost\\<Distro>\\<path> URL has TWO components to drop.

    Dropping only the UNC prefix leaves "Distro/path" glued together and shifts the
    Linux path up one level (for example /Ubuntu-22.04/home/user/AgentRec-X). This
    test pins the two-step derivation so that regression cannot come back.
    """
    code = _cmd_code_lines()
    assert '%AGENTRECX_WIN:~0,16%"=="\\\\wsl.localhost\\"' in code
    assert '%AGENTRECX_WIN:~0,7%"=="\\\\wsl$\\"' in code
    assert 'set "AGENTRECX_REST=%AGENTRECX_WIN:~16%"' in code
    assert 'set "AGENTRECX_REST=%AGENTRECX_WIN:~7%"' in code
    # The share component must then be split off before building the Linux path.
    assert 'for /f "tokens=1,* delims=\\" %%d in ("%AGENTRECX_REST%")' in code
    assert 'set "AGENTRECX_AFTER=%%e"' in code
    assert 'set "AGENTRECX_LINUX=/%AGENTRECX_AFTER%"' in code
    assert r"%AGENTRECX_LINUX:\=/%" in code


def test_windows_wrapper_unc_conversion_drops_both_prefix_and_distro() -> None:
    """Behavioural check of the derivation, mirroring cmd's tokens=1,* split."""
    BS = chr(92)
    prefix16 = BS * 2 + "wsl.localhost" + BS
    prefix7 = BS * 2 + "wsl$" + BS

    def derive(dp0: str) -> tuple[str, str]:
        win = dp0[:-1] if dp0.endswith(BS) else dp0
        rest = ""
        if win[:16].lower() == prefix16.lower():
            rest = win[16:]
        elif win[:7].lower() == prefix7.lower():
            rest = win[7:]
        parts = [p for p in rest.split(BS) if p]
        distro = parts[0] if parts else ""
        after = BS.join(parts[1:]) if len(parts) > 1 else ""
        linux = ("/" + after.replace(BS, "/")) if after else win
        return distro, linux

    # The UNC shape is the contract under test, so the expectation is pinned
    # synthetically. Deriving it from this checkout made the test depend on where
    # the repository lives: on a Windows drive the launcher has no UNC share to
    # parse at all, and the resolver's drive translation takes over instead.
    linux_expected = "/home/zxt/AgentRec-X"
    tail = BS.join(linux_expected.strip("/").split("/")) + BS
    for prefix in (prefix16, prefix7):
        form = prefix + "Ubuntu-22.04" + BS + tail
        distro, linux = derive(form)
        assert distro == "Ubuntu-22.04", form
        assert linux == linux_expected, f"{form} -> {linux}"


def test_resolver_maps_a_windows_drive_path_to_its_mnt_mount() -> None:
    """A checkout on a Windows drive: ``D:\\repo`` is ``/mnt/d/repo`` inside WSL.

    A drive path has no UNC share to parse, so the batch shim hands the resolver
    the raw Windows path and the resolver must translate it before verifying
    anything. The translation is only a candidate: ``Test-RepoRoot`` still has to
    prove the path inside a distro, so a wrong guess fails verification exactly
    like an unverifiable path always did.
    """
    import re

    BS = chr(92)
    ps = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace")
    pattern = r"^([A-Za-z]):[\\/](.*)$"
    assert pattern in ps, "the resolver does not translate a Windows drive path"
    assert "'/mnt/'" in ps, "the translation must target the WSL automount root"
    assert ".ToLower()" in ps, "the drive letter must be lower case for /mnt/<drive>"

    def to_linux(win: str) -> str:
        match = re.match(pattern, win)
        if not match:
            return win
        return "/mnt/" + match.group(1).lower() + "/" + match.group(2).replace(BS, "/")

    assert to_linux("D:" + BS + "IT" + BS + "agentrec" + BS + "X") == "/mnt/d/IT/agentrec/X"
    assert to_linux("d:/IT/agentrec/X") == "/mnt/d/IT/agentrec/X"
    assert to_linux("/home/zxt/AgentRec-X") == "/home/zxt/AgentRec-X"


def test_windows_wrapper_verifies_the_distro_before_launching() -> None:
    """A -d value must never be guessed: it comes from the verified response."""
    code = _cmd_code_lines()
    assert '"%AGENTRECX_DISTRO_OK%"==""' in code, "no empty-distro guard"
    assert '"%AGENTRECX_REPO_OK%"==""' in code, "no empty-repo guard"
    assert 'if not "%AGENTRECX_STATUS%"=="ok"' in code, "no status guard"
    # The launch must come after the guards.
    lines = code.splitlines()
    guard = next(i for i, l in enumerate(lines) if 'if not "%AGENTRECX_STATUS%"=="ok"' in l)
    launch = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("wsl.exe "))
    assert guard < launch, "the launch is not gated by the status guard"


def test_resolver_enumerates_real_distros_and_never_guesses() -> None:
    """The PowerShell resolver must ask WSL which distros exist."""
    ps = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace")
    assert "'--list'" in ps and "'--quiet'" in ps, "must enumerate distros via wsl --list --quiet"
    assert "$installed -contains $candidate" in ps, "candidate must be checked against the list"
    assert "Test-RepoRoot" in ps, "must verify the repository inside the distro"


def test_resolver_is_windows_powershell_51_compatible() -> None:
    """Regression: ProcessStartInfo.ArgumentList does not exist on .NET Framework.

    Windows PowerShell 5.1 is the default on Windows 11, so the resolver must use
    the 5.1-safe Start-Process form and avoid PS7-only syntax.
    """
    ps = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace")
    assert "ArgumentList.Add" not in ps, "ProcessStartInfo.ArgumentList is .NET Core only"
    assert "Start-Process" in ps
    # PS7-only operators must not appear.
    for forbidden in ("??", "?."):
        assert forbidden not in ps, f"PS7-only syntax not allowed: {forbidden!r}"


def test_resolver_verifies_launcher_exists_and_is_executable() -> None:
    """The repository check must specifically require an executable start_demo.sh."""
    ps = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace")
    assert "test -x scripts/start_demo.sh" in ps
    assert "recommendation/config.py" in ps
    assert "AGENTS.md" in ps


def test_resolver_never_prints_an_unverified_path() -> None:
    """A path may only be reported on the ok branch, after verification."""
    ps = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace")
    # The verified pair is passed only to Write-Response -VerifiedDistro/-VerifiedRepo.
    assert "-VerifiedDistro $foundDistro" in ps
    assert "-VerifiedRepo $linuxPath.TrimEnd('/')" in ps


def test_windows_wrapper_is_delegation_only() -> None:
    """The Windows side must not duplicate launcher, artifact or app logic."""
    code = _cmd_code_lines().lower()
    resolver = WIN_RESOLVER.read_text(encoding="utf-8", errors="replace").lower()
    for forbidden in ("best.pt", "sha256", "pip install", "torch", "numpy", "uvicorn"):
        assert forbidden not in code, f"cmd must not contain {forbidden!r}"
        assert forbidden not in resolver, f"resolver must not contain {forbidden!r}"
    assert "scripts/start_demo.sh" in code or "start_demo.sh" in resolver


def test_windows_wrapper_never_kills_processes() -> None:
    """No process management on the Windows side either."""
    code = _cmd_code_lines().lower() + WIN_RESOLVER.read_text(
        encoding="utf-8", errors="replace"
    ).lower()
    for forbidden in ("taskkill", "tskill", "stop-process", "wmic process"):
        assert forbidden not in code, f"must not contain {forbidden!r}"


def test_windows_wrapper_does_not_hardcode_a_username() -> None:
    """Resolution must not depend on a particular Windows or WSL user name."""
    text = _cmd_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("rem"):
            continue
        assert "/home/zxt" not in stripped, f"hardcoded user path: {stripped!r}"
        assert "C:\\Users\\zxt" not in stripped, f"hardcoded user path: {stripped!r}"


def test_windows_wrapper_keeps_crlf_for_cmd() -> None:
    """cmd.exe wants CRLF, and a core.autocrlf=false checkout must not break it."""
    raw = WIN_CMD.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "cmd file must not carry a UTF-8 BOM"
    assert b"\r\n" in raw, "cmd file must use CRLF line endings"
    assert all(byte < 128 for byte in raw), "cmd file must stay pure ASCII"


def test_windows_scripts_have_no_stray_non_comment_hash_lines() -> None:
    """Regression: a leading '#' inside a .cmd is not a comment and broke parsing."""
    for line in _cmd_text().splitlines():
        stripped = line.strip()
        assert not stripped.startswith("#"), f"invalid .cmd comment syntax: {stripped!r}"


def test_gitattributes_pins_windows_line_endings() -> None:
    """The CRLF requirement is enforced by git, not by luck."""
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.cmd text eol=crlf" in attributes
    assert "*.sh text eol=lf" in attributes


def test_browser_helper_is_convenience_only_and_loopback_only() -> None:
    """Opening the browser must not supervise the service, and must stay local."""
    ps = WIN_BROWSER.read_text(encoding="utf-8", errors="replace")
    code = "\n".join(line for line in ps.splitlines() if not line.strip().startswith("#"))
    assert "/health" in code, "readiness probe missing"
    assert "/demo/" in code, "demo URL missing"
    assert "Start-Process" in code
    assert "127.0.0.1" in code
    for other in ("0.0.0.0", "localhost:", "http://192.", "http://10."):
        assert other not in code, f"browser helper must stay loopback-only: {other!r}"
    for forbidden in ("Stop-Process", "taskkill", "Start-Service"):
        assert forbidden not in code, f"browser helper must not contain {forbidden!r}"


def test_browser_helper_avoids_powershell_automatic_variables() -> None:
    """$Host is a PowerShell automatic variable; the bind host must not shadow it."""
    ps = WIN_BROWSER.read_text(encoding="utf-8", errors="replace")
    assert "[string]$BindHost" in ps
    assert "[string]$Host" not in ps

# --------------------------------------------------------------------------- #
# trust-boundary guards
# --------------------------------------------------------------------------- #


def test_local_demo_has_no_process_control_capability() -> None:
    """The launcher must not even be *able* to signal or spawn a process.

    This is the guard for the "never kill an unrelated process" requirement.  It is
    asserted against the imported module rather than its source text: a clean
    subprocess imports the module and reports what it pulled in.  Having no
    ``subprocess``/``signal`` binding means there is no process-management machinery
    to misuse, and ``os`` is never referenced through the module.
    """
    import subprocess

    code = (
        "import sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "import recommendation.local_demo as m;"
        "assert not hasattr(m, 'subprocess'), 'module exposes subprocess';"
        "assert not hasattr(m, 'signal'), 'module exposes signal';"
        "assert 'subprocess' not in sys.modules, 'subprocess imported at module load';"
        "assert 'signal' not in sys.modules, 'signal imported at module load';"
        "print('clean')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout


def test_local_demo_never_invokes_pip() -> None:
    """Normal start must not mutate the environment; only setup_demo.sh installs."""
    import subprocess

    code = (
        "import sys, io, contextlib;"
        "sys.path.insert(0, sys.argv[1]);"
        "import recommendation.local_demo as m;"
        "assert not hasattr(m, 'subprocess'), 'pip could be shelled out via subprocess';"
        "print('clean')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr

    # No pip invocation is constructible either: there is no shell-out facility and
    # no bare "-m pip" command string.
    source = Path(local_demo.__file__).read_text(encoding="utf-8")
    assert '"-m", "pip"' not in source
    assert "'-m', 'pip'" not in source
    assert "os.system" not in source


def test_start_script_never_runs_pip() -> None:
    """The start path must be read-only; only setup_demo.sh may install."""
    start = (REPO_ROOT / "scripts" / "start_demo.sh").read_text(encoding="utf-8")
    code_lines = [
        line for line in start.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    offenders = [line for line in code_lines if "pip install" in line or "pip3 install" in line]
    assert not offenders, f"start_demo.sh must not install: {offenders}"
    assert "setup_demo.sh" in start, "start should point at the setup script"


def test_local_demo_does_not_import_torch_at_module_load() -> None:
    """Importing the launcher support must not drag in PyTorch.

    Run in a subprocess so the assertion cannot be polluted by other test modules
    that legitimately import torch.  This guards the fast preflight path: the whole
    point of the fast tier is that it stays cheap.
    """
    import subprocess

    code = (
        "import sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "import recommendation.local_demo as m;"
        "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if 'torch' in k);"
        "assert 'numpy' not in sys.modules, 'numpy imported at module load';"
        "print('clean')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout
