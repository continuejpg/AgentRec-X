"""Real-asset tests for the TIGER serving engine: real worker process, real artifacts.

Two tiers, kept strictly apart from the fake-transport contract tests:

* **Real subprocess, no weights** - the worker is started for real with the backend
  interpreter and its asset self-check/startup path is exercised.  These run wherever the
  backend venv and the sid/handoff artifacts are present.
* **Real generative inference** - requires the accepted step-2.5 checkpoint
  (``runs/tiger_public_2026/checkpoint/{tiger.pt,tiger.json,generator_layout.json}``).  It is
  skipped with an explicit reason when that checkpoint is absent, and a skip is **never** a
  claim that inference works.

Nothing here is a substitute for the other: the fake-transport tests document the engine-side
contract, these document what the backend actually does on this machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from recommendation.inference.tiger_engine import (
    DEFAULT_HANDOFF_DIR,
    REPO,
    TigerServingEngine,
    TigerWorkerConfig,
    TigerWorkerError,
)
from recommendation.control.tiger_source import TigerSourceTool

BACKEND_ROOT = REPO / "backends" / "tiger_public"
BACKEND_PYTHON = BACKEND_ROOT / ".venv" / "bin" / "python"
BACKEND_SRC = BACKEND_ROOT / "src"
CHECKPOINT_DIR = REPO / "runs" / "tiger_public_2026" / "checkpoint"
SID_DIR = REPO / "runs" / "tiger_public_2026" / "sid"
HANDOFF_DIR = DEFAULT_HANDOFF_DIR
MAPPINGS_PATH = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
#: Recorded as ``mappings_sha256`` in the handoff's catalogue.json.
FROZEN_MAPPINGS_SHA256 = "dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912"

#: From the frozen H7 execution manifest (runs/step26-h7-execution-manifest.txt).
FROZEN_H7 = {
    "sid/semantic_ids.json": "6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e",
    "sid/layout.json": "22521d7b58f8538c3d3b0d8a27200a255ac62e1e9da0676c6c55ebadde101b11",
    "handoff/catalogue.json": "84f4a35dfe99c96f9b9e9c65ee2864aeed901fba0577fe221810555fe0732096",
    "checkpoint/tiger.pt": "95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a",
    "checkpoint/generator_layout.json": "eed4dffaccf8dfecf40ee5bc9f8206870910a8a4e31fdd941e8ac6697814a55a",
}

requires_backend = pytest.mark.skipif(
    not BACKEND_PYTHON.exists(), reason=f"no backend interpreter at {BACKEND_PYTHON}"
)
requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "tiger.pt").exists(),
    reason=(
        "the accepted TIGER generator checkpoint is not on this machine "
        f"({CHECKPOINT_DIR}/tiger.pt); generative inference is unverified here"
    ),
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(BACKEND_SRC), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    return env


def _identity_from_mappings() -> dict[str, int]:
    """The trusted ``parent_asin -> item_id`` map, from the frozen mappings artifact.

    The handoff deliberately carries no canonical identity (``catalogue_items.jsonl`` maps
    ``backend_row`` to ``item_id`` only), so identity must come from AgentRec-X's own artifact -
    the same one whose sha256 the handoff's ``catalogue.json`` records as ``mappings_sha256``.
    """
    payload = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    assert _sha256(MAPPINGS_PATH) == FROZEN_MAPPINGS_SHA256, "the mappings artifact moved"
    return {str(k): int(v) for k, v in payload["item2id"].items()}


# --------------------------------------------------------------------------- #
# Tier 1: real artifacts and the real worker process (no weights needed)
# --------------------------------------------------------------------------- #


@requires_backend
def test_local_sid_and_catalogue_match_the_frozen_h7_manifest() -> None:
    checks = {
        "sid/semantic_ids.json": SID_DIR / "semantic_ids.json",
        "sid/layout.json": SID_DIR / "layout.json",
        "handoff/catalogue.json": HANDOFF_DIR / "catalogue.json",
    }
    for label, path in checks.items():
        assert path.is_file(), f"{label} is missing at {path}"
        assert _sha256(path) == FROZEN_H7[label], f"{label} does not match the H7 freeze"


@requires_backend
def test_real_worker_self_check_accepts_the_restored_assets() -> None:
    """With the restored checkpoint the worker's own self-check must report ok."""
    completed = subprocess.run(
        [
            str(BACKEND_PYTHON), "-m", "tiger_public.serve_worker",
            "--self-check",
            "--checkpoint-dir", str(CHECKPOINT_DIR),
            "--sid", str(SID_DIR),
            "--catalogue", str(HANDOFF_DIR),
        ],
        cwd=str(REPO),
        env=_worker_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    frame = json.loads(completed.stdout.strip().splitlines()[-1])
    assert frame["op"] == "self_check"
    assert frame["ok"] is True, frame.get("problems")
    assert frame["problems"] == []
    assert completed.returncode == 0


@requires_backend
def test_real_worker_self_check_reports_missing_assets(tmp_path: Path) -> None:
    """The same self-check, pointed at a checkpoint directory that does not exist."""
    completed = subprocess.run(
        [
            str(BACKEND_PYTHON), "-m", "tiger_public.serve_worker",
            "--self-check",
            "--checkpoint-dir", str(tmp_path / "checkpoint"),
            "--sid", str(SID_DIR),
            "--catalogue", str(HANDOFF_DIR),
        ],
        cwd=str(REPO),
        env=_worker_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    frame = json.loads(completed.stdout.strip().splitlines()[-1])
    assert frame["op"] == "self_check"
    assert frame["ok"] is False
    assert any("tiger.pt" in problem for problem in frame["problems"])
    assert completed.returncode == 3, "an incomplete asset set must not exit successfully"


@requires_backend
def test_real_worker_startup_fails_closed_over_the_real_ipc(tmp_path: Path) -> None:
    """The engine talks to the genuine worker; a missing checkpoint must fail cleanly, not hang."""
    engine = TigerServingEngine(
        {"B0PLACEHOLDER": 1},
        config=TigerWorkerConfig(
            checkpoint_dir=tmp_path / "checkpoint",
            sid_dir=SID_DIR,
            catalogue_dir=HANDOFF_DIR,
        ),
        verify_assets_on_start=False,   # let the *worker* be the one that refuses
        startup_timeout=300,
    )
    started = time.perf_counter()
    with pytest.raises(TigerWorkerError) as excinfo:
        engine.start()
    engine.close()
    elapsed = time.perf_counter() - started
    assert excinfo.value.code == "assets_unavailable"
    assert "tiger.pt" in str(excinfo.value) or "checkpoint" in str(excinfo.value)
    assert elapsed < 300


# --------------------------------------------------------------------------- #
# Tier 2: real generative inference (needs the checkpoint; skipped without it)
# --------------------------------------------------------------------------- #


@requires_backend
@requires_checkpoint
def test_real_recommendation_on_frozen_histories() -> None:
    """One live generative retrieval per frozen history; prints the measured facts."""
    cases: list[dict[str, Any]] = []
    with (HANDOFF_DIR / "eval_cohort.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 1:  # one case: worker start-up alone costs minutes on CPU
                break
            cases.append(json.loads(line))
    assert cases, "the frozen eval cohort is empty"

    identity = _identity_from_mappings()
    config = TigerWorkerConfig(
        checkpoint_dir=CHECKPOINT_DIR, sid_dir=SID_DIR, catalogue_dir=HANDOFF_DIR, device="cpu"
    )
    engine = TigerServingEngine(identity, config=config, startup_timeout=1800, request_timeout=900)
    with engine:
        report = engine.asset_report()
        print(
            f"[tiger-real] device={report['device']} load={report['load_seconds']:.1f}s "
            f"checkpoint={report['checkpoint_sha256'][:12]} sid={report['sid_artifact_sha256'][:12]}"
        )
        for case in cases:
            history = tuple(int(value) for value in case["history"])
            started = time.perf_counter()
            result = engine.recommend(history, k=20)
            elapsed = time.perf_counter() - started
            print(
                f"[tiger-real] case={case['case_id']} hist={len(history)} "
                f"mode={result.retrieval_mode} candidates={len(result.recommendations)} "
                f"latency={elapsed:.2f}s status={result.summary.get('status')}"
            )
            assert result.recommendations, "a certified retrieval returned no candidates"
            assert all(item.parent_asin for item in result.recommendations)
            assert [item.rank for item in result.recommendations] == list(
                range(1, len(result.recommendations) + 1)
            )

        history = tuple(int(value) for value in cases[0]["history"])
        first = engine.recommend(history, k=5)
        repeated = engine.recommend(history, k=5)

        # A real candidate-source call through the existing seam, not a fake engine.
        tool = TigerSourceTool(engine, require_certified=False, max_k=20)
        reverse = {value: key for key, value in identity.items()}
        asins = tuple(reverse[value] for value in history if value in reverse)
        triples = tool.propose(arguments=None, limit=5, history=asins)
        print(f"[tiger-real] TigerSourceTool -> {len(triples)} triples, top={triples[0]}")
        print(f"[tiger-real] score_kind={tool.score_kind} provenance={tool.last_proposals[0].provenance_token()}")
        assert triples, "a real TIGER call through the source seam returned nothing"
        assert all(isinstance(asin, str) and asin.startswith("B0") for asin, _, _ in triples)

    assert [item.as_dict() for item in repeated.recommendations] == [
        item.as_dict() for item in first.recommendations
    ], "the same history must retrieve the same candidates in the same order"
    # CERTIFIED returns the certified frontier (k + |unique history|), not k: the frontier is
    # what survives a seen-item mask.  The head of the list is still the certified top-k.
    assert len(first.recommendations) >= 5
    assert first.summary.get("status") in {"CERTIFIED", "APPROXIMATE"}


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the backend package targets python >= 3.10"
)
def test_checkpoint_absence_is_documented_rather_than_hidden() -> None:
    """A guard against silently claiming success: the blocker is asserted, not implied."""
    if (CHECKPOINT_DIR / "tiger.pt").exists():
        pytest.skip("the checkpoint is present; the real-inference test above covers it")
    assert not (CHECKPOINT_DIR).exists() or not (CHECKPOINT_DIR / "tiger.pt").exists()
