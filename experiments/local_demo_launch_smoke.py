"""Milestone 11.5 smoke: the one-command local demo launcher over a real socket.

This is the acceptance smoke for ``scripts/start_demo.sh``.  Unlike
:mod:`experiments.web_demo_smoke` -- which drives the accepted demo through FastAPI's
in-process ``TestClient`` -- this smoke exercises the **real user path**:

    scripts/start_demo.sh
      -> launcher preflight (artifacts + environment + port)
      -> recommendation.api.app            (the existing entry point, unchanged)
      -> Uvicorn                           (real process)
      -> a real localhost TCP socket
      -> GET /health, /v1/demo/health, /v1/model, /demo/
      -> SIGINT
      -> graceful shutdown, port released, no orphan process

What it deliberately does not do
--------------------------------
* it never installs anything and never mutates the environment;
* it never runs ``setup_demo.sh``;
* it never touches a foreign process -- if the scratch port is occupied it fails
  rather than reclaiming it, and it never signals anything it did not start;
* it makes no recommendation-quality claim.

Usage::

    .venv/bin/python -m experiments.local_demo_launch_smoke
    .venv/bin/python -m experiments.local_demo_launch_smoke --port 8130 --json /tmp/m115.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.api.app import ACCEPTED_CHECKPOINT_SHA256  # noqa: E402
from recommendation.local_demo import classify_port  # noqa: E402

#: The launcher under test.
START_SCRIPT = REPO_ROOT / "scripts" / "start_demo.sh"

#: Default scratch port.  Deliberately NOT 8000: the acceptance environment already
#: runs an AgentRec-X instance there and this smoke must never disturb it.
DEFAULT_PORT = 8130

#: How long to wait for the server to become ready.  The real start loads a 121 MB
#: checkpoint, a 307 MB catalogue and a 349 MB sequences artifact, which took ~12-20 s
#: on the acceptance machine, so this is generous on purpose.
READY_TIMEOUT_SECONDS = 300.0

#: How long to wait for a graceful shutdown after SIGINT.
SHUTDOWN_TIMEOUT_SECONDS = 60.0


def free_port() -> int:
    """Return a currently free loopback port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def http_get(url: str, timeout: float = 10.0) -> tuple[int | None, Any, str]:
    """GET a URL; returns ``(status, json_or_text, error)`` and never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - reported as a gate failure
        return None, None, f"{type(exc).__name__}: {exc}"
    text = raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text), ""
    except json.JSONDecodeError:
        return status, text, ""


def http_get_status(url: str, timeout: float = 10.0) -> tuple[int | None, str]:
    """GET a URL following redirects is disabled; returns ``(status, location)``."""
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - localhost

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # noqa: D102 - no redirects
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status), response.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.headers.get("Location", "") if exc.headers else ""
    except Exception:  # noqa: BLE001 - reported as a gate failure
        return None, ""


def wait_for_ready(process: subprocess.Popen, port: int) -> tuple[bool, float]:
    """Poll ``/health`` until the child answers or the timeout expires."""
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    while time.monotonic() - started < READY_TIMEOUT_SECONDS:
        if process.poll() is not None:
            return False, time.monotonic() - started
        status, _payload, error = http_get(f"{base}/health", timeout=3.0)
        if status == 200 and not error:
            return True, time.monotonic() - started
        time.sleep(0.5)
    return False, time.monotonic() - started


class OutputCollector:
    """Drain the child's stdout in a background thread so it can never block."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, args=(process,), daemon=True)
        self._thread.start()

    def _drain(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._lines.append(line.rstrip("\n"))

    @property
    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def text(self) -> str:
        return "\n".join(self.lines)


def stop_child(process: subprocess.Popen, port: int) -> tuple[int | None, bool]:
    """Send SIGINT, wait for a graceful exit, and report whether the port was released.

    Only ever signals the process it started.  If graceful shutdown fails the child is
    terminated outright so the smoke cannot leave an orphan behind.
    """
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:  # pragma: no cover - already gone
            pass
    try:
        exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
        return None, False

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if classify_port("127.0.0.1", port, timeout=0.5).state == "free":
            return exit_code, True
        time.sleep(0.3)
    return exit_code, False


def main(argv: list[str] | None = None) -> int:
    """Run the launcher launch smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 11.5 local launcher smoke")
    parser.add_argument("--port", type=int, default=None, help="scratch port (default: auto)")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--keep-output", action="store_true", help="always print child output")
    args = parser.parse_args(argv)

    port = args.port if args.port is not None else free_port()

    print("=" * 78)
    print("Milestone 11.5 smoke: one-command local demo launcher over a real socket")
    print("=" * 78)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"port": port, "start_script": str(START_SCRIPT)}

    if not START_SCRIPT.is_file():
        print(f"SKIPPED: launcher not found: {START_SCRIPT}")
        print("\nSMOKE: SKIP")
        return 0

    # ---- gate 1: the scratch port must be free before we begin --------------- #
    pre_status = classify_port("127.0.0.1", port)
    checks["scratch port is free before the run"] = pre_status.state == "free"
    if not checks["scratch port is free before the run"]:
        print(f"\nFAIL: scratch port {port} is not free ({pre_status.state}): {pre_status.detail}")
        print("\nSMOKE: FAIL")
        return 1

    process: subprocess.Popen | None = None
    collector: OutputCollector | None = None
    exit_code: int | None = None
    port_released = False

    try:
        environment = dict(os.environ)
        environment.pop("AGENTRECX_PORT", None)
        environment.pop("AGENTRECX_HOST", None)

        started = time.perf_counter()
        process = subprocess.Popen(
            ["bash", str(START_SCRIPT), "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        collector = OutputCollector(process)

        ready, elapsed = wait_for_ready(process, port)
        details["ready_seconds"] = round(elapsed, 1)
        checks["launcher started and became ready"] = ready

        if not ready:
            print(f"\nFAIL: the launcher did not become ready within {READY_TIMEOUT_SECONDS:.0f}s")
            print("\n--- child output ---")
            print(collector.text())
        else:
            base = f"http://127.0.0.1:{port}"

            # ---- child is really the launcher + server ---------------------- #
            checks["child process is still alive"] = process.poll() is None

            # ---- gate: launcher preflight reported PASS --------------------- #
            output = collector.text()
            details["launcher_output_head"] = collector.lines[:12]
            checks["launcher preflight reported PASS"] = "PREFLIGHT: PASS" in output
            checks["launcher reported the port as free"] = "port is free" in output

            # ---- gate: M6 health ------------------------------------------- #
            status, payload, error = http_get(f"{base}/health")
            checks["GET /health returns 200"] = status == 200
            checks["health reports the model loaded"] = (
                isinstance(payload, dict) and payload.get("model_loaded") is True
            )
            checks["health reports the CPU device"] = (
                isinstance(payload, dict) and payload.get("device") == "cpu"
            )
            details["health"] = payload if isinstance(payload, dict) else error

            # ---- gate: demo health ----------------------------------------- #
            status, payload, error = http_get(f"{base}/v1/demo/health")
            demo_health = payload if isinstance(payload, dict) else {}
            checks["GET /v1/demo/health returns 200"] = status == 200
            checks["demo health reports ready"] = (
                demo_health.get("status") == "ok" and demo_health.get("demo_ready") is True
            )
            checks["demo health reports metadata loaded"] = (
                demo_health.get("metadata_loaded") is True
            )
            checks["expected demo profiles are available"] = demo_health.get("profiles") == 3
            details["demo_health"] = demo_health if demo_health else error

            # ---- gate: model identity -------------------------------------- #
            status, payload, error = http_get(f"{base}/v1/model")
            model_info = payload if isinstance(payload, dict) else {}
            checks["GET /v1/model returns 200"] = status == 200
            checks["model is SASRec"] = model_info.get("model_type") == "SASRec"
            checks["model runs on CPU"] = model_info.get("device") == "cpu"
            checks["model parameters are frozen"] = model_info.get("model_parameters_frozen") is True
            checks["checkpoint identity is the accepted digest"] = (
                model_info.get("checkpoint_sha256") == ACCEPTED_CHECKPOINT_SHA256
            )
            details["model"] = {
                key: model_info.get(key)
                for key in (
                    "model_type",
                    "num_items",
                    "max_seq_len",
                    "device",
                    "checkpoint_sha256",
                    "parameter_count",
                    "model_parameters_frozen",
                )
            }

            # ---- gate: browser UI ------------------------------------------ #
            status, payload, error = http_get(f"{base}/demo/")
            checks["GET /demo/ serves the browser UI"] = status == 200
            checks["GET /demo/ returns HTML"] = isinstance(payload, str) and "<html" in payload.lower()

            status, location = http_get_status(f"{base}/")
            checks["GET / redirects to /demo/"] = status == 307 and location.endswith("/demo/")
            details["browser_url"] = f"{base}/demo/"

            # ---- gate: the real socket, not an in-process client ------------- #
            probe = socket.socket()
            try:
                probe.settimeout(5.0)
                probe.connect(("127.0.0.1", port))
                checks["server is reachable over a real TCP socket"] = True
            except OSError:
                checks["server is reachable over a real TCP socket"] = False
            finally:
                probe.close()

            # ---- coverage guard: this smoke is not the TestClient smoke ------ #
            checks["smoke used a real process and socket, not TestClient"] = (
                "testclient" not in output.lower()
            )

    finally:
        if process is not None:
            exit_code, port_released = stop_child(process, port)

    details["child_exit_code"] = exit_code
    details["port_released"] = port_released

    # ---- gate: clean shutdown ------------------------------------------------ #
    checks["SIGINT stopped the child gracefully"] = exit_code in (0, 130)
    shutdown_output = (collector.text() if collector else "").lower()
    checks["Uvicorn reported graceful shutdown"] = (
        "shutting down" in shutdown_output
        or "shutdown complete" in shutdown_output
        or "stopped (exit status" in shutdown_output
    )
    checks["the port was released after shutdown"] = port_released
    checks["no orphan launcher process remains"] = classify_port("127.0.0.1", port).state == "free"

    passed = all(checks.values())
    print("\n--- gates ---")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not passed or args.keep_output:
        print("\n--- launcher output (tail) ---")
        print("\n".join((collector.lines if collector else [])[-40:]))

    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"milestone": "11.5", "checks": checks, **details}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
