"""Resumable checkpoints and the experiment manifest (Milestone 5).

A checkpoint must carry everything needed to resume *semantically* correctly: not
just weights, but the optimizer state, the epoch/step counters, the best-validation
record, the early-stopping patience counter, the negative-sampling epoch, and the
RNG states so negative resampling continues rather than restarting.

Writes are atomic (temp file -> flush/fsync -> ``os.replace``) so an interrupted run
never leaves a half-written checkpoint behind.

Checkpoints are binaries and must stay git-ignored; this module never writes into the
repository tree unless the caller points it at an ignored path.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Format tag written into every checkpoint so a loader can reject foreign files.
CHECKPOINT_FORMAT = "agentrecx.sasrec.checkpoint.v1"

#: Manifest format tag.
MANIFEST_FORMAT = "agentrecx.experiment.v1"


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, malformed, or incompatible."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Write ``payload`` to ``path`` atomically via a temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        with open(temp_name, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def write_json_atomic(payload: Any, path: str | Path) -> Path:
    """Write JSON atomically, with an fsync before the rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def capture_rng_state() -> dict[str, Any]:
    """Capture Python and torch RNG states (CPU, and CUDA when present)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class TrainingState:
    """Everything needed to resume training without losing semantics."""

    epoch: int = 0
    global_step: int = 0
    best_epoch: int = -1
    best_metric: float = float("-inf")
    best_hr: float = float("-inf")
    patience_counter: int = 0
    negative_epoch: int = 0
    epochs_without_improvement: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_hr": self.best_hr,
            "patience_counter": self.patience_counter,
            "negative_epoch": self.negative_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: TrainingState,
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    max_seq_len: int,
    seed: int,
    num_items: int,
    dataset_identity: dict[str, Any] | None = None,
    validation_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a resumable checkpoint and return its path."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "saved_at": time.time(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "state": state.as_dict(),
        "model_config": model_config,
        "trainer_config": trainer_config,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "num_items": num_items,
        "dataset_identity": dataset_identity or {},
        "validation_metrics": validation_metrics or {},
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }
    return atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model/optimizer/RNG state.

    Returns the raw payload so callers can read the training state and the recorded
    validation metrics.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface any deserialisation failure
        raise CheckpointError(f"could not read checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{path} is not an {CHECKPOINT_FORMAT} checkpoint "
            f"(got format={payload.get('format') if isinstance(payload, dict) else type(payload).__name__})"
        )

    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload


def state_from_payload(payload: dict[str, Any]) -> TrainingState:
    """Rebuild a :class:`TrainingState` from a loaded checkpoint payload."""
    raw = payload.get("state", {})
    return TrainingState(
        epoch=int(raw.get("epoch", 0)),
        global_step=int(raw.get("global_step", 0)),
        best_epoch=int(raw.get("best_epoch", -1)),
        best_metric=float(raw.get("best_metric", float("-inf"))),
        best_hr=float(raw.get("best_hr", float("-inf"))),
        patience_counter=int(raw.get("patience_counter", 0)),
        negative_epoch=int(raw.get("negative_epoch", 0)),
        epochs_without_improvement=int(raw.get("epochs_without_improvement", 0)),
    )


# --------------------------------------------------------------------------- #
# Checkpoint selection (validation only)
# --------------------------------------------------------------------------- #


@dataclass
class SelectionRecord:
    """The best-so-far validation record under the frozen selection rule."""

    best_epoch: int = -1
    best_ndcg10: float = float("-inf")
    best_hr10: float = float("-inf")
    patience: int = 0

    def consider(self, epoch: int, ndcg10: float, hr10: float) -> bool:
        """Offer an epoch's validation metrics; return True if it becomes the best.

        Order (Milestone 5 §22): higher validation NDCG@10 wins; on an exact tie,
        higher validation HR@10 wins; if still tied, the *earlier* epoch wins.  Test
        metrics are never consulted.
        """
        improved = False
        if ndcg10 > self.best_ndcg10:
            improved = True
        elif ndcg10 == self.best_ndcg10:
            if hr10 > self.best_hr10:
                improved = True
            elif hr10 == self.best_hr10 and self.best_epoch < 0:
                improved = True  # first observation
            # otherwise: earlier epoch already holds the record -> no improvement
        if improved:
            self.best_epoch = epoch
            self.best_ndcg10 = ndcg10
            self.best_hr10 = hr10
            self.patience = 0
        else:
            self.patience += 1
        return improved

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "best_epoch": self.best_epoch,
            "best_validation_ndcg@10": self.best_ndcg10,
            "best_validation_hr@10": self.best_hr10,
            "patience_counter": self.patience,
        }


# --------------------------------------------------------------------------- #
# Experiment manifest
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentManifest:
    """Machine-readable record of one canonical run."""

    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def update(self, **sections: Any) -> None:
        """Merge sections into the manifest."""
        self.payload.update(sections)

    def as_dict(self) -> dict[str, Any]:
        """Return the manifest with its format tag."""
        return {"format": MANIFEST_FORMAT, "run_id": self.run_id, **self.payload}

    def write(self, path: str | Path) -> Path:
        """Atomically write the manifest."""
        return write_json_atomic(self.as_dict(), path)


def git_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Return best-effort git metadata (branch / HEAD / dirty flag).

    Never raises: a missing git binary or repository simply yields ``available=False``.
    """
    import subprocess

    root = str(repo_root)
    meta: dict[str, Any] = {"available": False}
    try:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if head.returncode != 0:
            return meta
        branch = subprocess.run(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        meta = {
            "available": True,
            "commit": head.stdout.strip(),
            "branch": branch.stdout.strip() if branch.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return meta
    return meta


def environment_metadata() -> dict[str, Any]:
    """Return PyTorch/CUDA/GPU metadata for the manifest."""
    meta: dict[str, Any] = {
        "python_version": __import__("sys").version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
    }
    if torch.cuda.is_available():
        meta["device_name"] = torch.cuda.get_device_name(0)
        meta["device_capability"] = list(torch.cuda.get_device_capability(0))
        meta["total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
    return meta


__all__ = [
    "CHECKPOINT_FORMAT",
    "MANIFEST_FORMAT",
    "CheckpointError",
    "ExperimentManifest",
    "SelectionRecord",
    "TrainingState",
    "atomic_torch_save",
    "capture_rng_state",
    "environment_metadata",
    "git_metadata",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "sha256_file",
    "state_from_payload",
    "write_json_atomic",
]
