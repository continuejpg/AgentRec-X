"""Long-lived TIGER serving worker: one process, many single-request retrievals.

Why this exists
---------------
``retrieve_cli`` is built for the frozen cohort: it reads ``case_id``/``history``/
``required_frontier`` out of the v3 eval-cohort handoff and walks it.  Serving needs the
opposite shape - an arbitrary, caller-supplied history, one request at a time, with the
generator loaded **once** instead of once per request.

This module adds only the loop and the transport.  It reuses, unmodified:

* :func:`tiger_public.tiger.load_generator` - the accepted checkpoint loader, including its
  dependency-hash refusal;
* :func:`tiger_public.retrieve.build_history_input` - byte-for-byte the training-time encoder
  construction;
* :class:`tiger_public.retrieve.TigerCaseScorer` - the accepted next-token scorer;
* :func:`tiger_public.retrieve.retrieve` - APPROXIMATE beam / CERTIFIED branch-and-bound,
  status never upgraded;
* :class:`tiger_public.trie.CatalogueTrie` and :class:`tiger_public.retrieve.SidResolver`.

Boundary
--------
``parent_asin`` never appears in this file.  Requests carry AgentRec-X integer item ids and
responses carry integer item ids plus the model's own outputs.  Identity is re-attached by the
AgentRec-X layer (``recommendation/backends/tiger_engine.py``).

Transport
---------
One JSON object per line on stdin, one JSON object per line on stdout.  stdout carries protocol
frames **only**; every human-readable line goes to stderr.  A request that fails produces one
``error`` frame and the worker keeps serving; a startup failure produces one fatal ``error``
frame and exit code :data:`EXIT_STARTUP_FAILURE`.

Request frame::

    {"op": "recommend", "request_id": 1, "history": [12, 44, 91], "k": 20,
     "mode": "CERTIFIED", "required_frontier": 23, "max_hist_items": 20}

``required_frontier`` defaults to ``k + |unique history|``, the convention the frozen cohort
uses (``experiments/step26_gates.py`` G16.1: ``required_frontier == K_max + |unique seen|``).
The frontier is what makes a certified answer survive a seen-item mask, so the default is the
project's own rule rather than a number invented here.

Response frame::

    {"op": "recommend_result", "request_id": 1, "retrieval_mode": "CERTIFIED", "k": 20, ...,
     "items": [{"item_id": ..., "rank": 1, "score": ..., "semantic_id": [...],
                "token_path": [...], "resolver": {...}}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from tiger_public.generator_layout import load_generator_layout
from tiger_public.io import read_catalogue, read_semantic_ids, sha256_file
from tiger_public.retrieve import (
    APPROXIMATE,
    CERTIFIED,
    RetrievalConfig,
    SidResolver,
    TigerCaseScorer,
    build_history_input,
    retrieve,
)
from tiger_public.tiger import load_generator
from tiger_public.trie import CatalogueTrie

__all__ = ["CONTRACT", "EXIT_STARTUP_FAILURE", "main", "serve"]

#: Protocol identity for the request/response frames above.
CONTRACT = "agentrecx.tiger.serve.v1"

#: Exit code used when the worker cannot start at all (missing checkpoint, bad assets).
EXIT_STARTUP_FAILURE = 3

#: Error codes the AgentRec-X side maps to typed failures.
CODE_ASSETS = "assets_unavailable"
CODE_REQUEST = "request_invalid"
CODE_RETRIEVAL = "retrieval_failed"
CODE_INTERNAL = "internal_error"


# --------------------------------------------------------------------------- #
# Asset assembly (identical to retrieve_cli's, minus the frozen cohort)
# --------------------------------------------------------------------------- #


class WorkerStartupError(RuntimeError):
    """Raised when the worker cannot assemble its assets or load the generator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Assets:
    """The loaded, shared state of one worker process.

    Everything here is built once.  Per request only the encoder tokens, the scorer and the
    retrieval run are new.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        sid_dir: Path,
        catalogue_dir: Path,
        exposure_path: Path,
        generator_layout_path: Path | None,
        device: str,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sid_dir = Path(sid_dir)
        self.catalogue_dir = Path(catalogue_dir)
        self.exposure_path = Path(exposure_path)
        self.generator_layout_path = Path(
            generator_layout_path or (self.checkpoint_dir / "generator_layout.json")
        )
        self.device = device

        for label, path in (
            ("checkpoint directory", self.checkpoint_dir),
            ("sid directory", self.sid_dir),
            ("catalogue/handoff directory", self.catalogue_dir),
            ("train exposure", self.exposure_path),
            ("generator layout", self.generator_layout_path),
            ("checkpoint weights", self.checkpoint_dir / "tiger.pt"),
            ("checkpoint metadata", self.checkpoint_dir / "tiger.json"),
            ("sid assignment", self.sid_dir / "semantic_ids.json"),
            ("sid token layout", self.sid_dir / "layout.json"),
        ):
            if not path.exists():
                raise WorkerStartupError(
                    CODE_ASSETS,
                    f"{label} is missing at {path}; TIGER serving needs the accepted step-2.5 "
                    "checkpoint directory (tiger.pt + tiger.json + generator_layout.json) and "
                    "the matching sid/ and handoff/ artifacts",
                )

        started = time.perf_counter()
        self.catalogue = read_catalogue(self.catalogue_dir)
        self.semantic_ids = read_semantic_ids(self.sid_dir)
        self.layout = load_generator_layout(self.generator_layout_path)
        self.assignment = self.semantic_ids.assignment
        if len(self.assignment) - 1 != self.catalogue.num_items:
            raise WorkerStartupError(
                CODE_ASSETS,
                "the assignment holds "
                f"{len(self.assignment) - 1} items but the catalogue declares "
                f"{self.catalogue.num_items}; refusing to retrieve against a mismatched catalogue",
            )
        self.trie = CatalogueTrie(self.assignment, layout=self.layout)
        self.resolver = SidResolver(self.trie)
        self.generator, self.metadata, _resume = load_generator(
            out_dir=self.checkpoint_dir,
            sid_dir=self.sid_dir,
            exposure_path=self.exposure_path,
            catalogue_dir=self.catalogue_dir,
            generator_layout_path=self.generator_layout_path,
            device=self.device,
        )
        self.load_seconds = time.perf_counter() - started
        self.identity = self._identity()

    @property
    def resolver_record(self) -> dict[str, Any]:
        """The resolver provenance that rides on every result."""
        return {
            "candidate_item_for_path": "smallest_item_id_wins",
            "collision_policy": str(getattr(SidResolver, "collision_policy", "unknown")),
            "levels": int(self.layout.get("levels", 0)),
            "dedup_levels": int(self.layout.get("dedup_levels", 0)),
            "collisions_observed": int(getattr(self.resolver, "collisions_observed", 0)),
        }

    def _identity(self) -> dict[str, Any]:
        return {
            "catalogue_items": int(self.catalogue.num_items),
            "assignment_items": len(self.assignment) - 1,
            "catalogue_sha256": str(getattr(self.catalogue, "catalogue_sha256", "") or ""),
            "sid_artifact_sha256": sha256_file(self.sid_dir / "semantic_ids.json"),
            "sid_layout_sha256": sha256_file(self.sid_dir / "layout.json"),
            "generator_layout_sha256": sha256_file(self.generator_layout_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_dir / "tiger.pt"),
            "exposure_sha256": sha256_file(self.exposure_path),
            "layout": dict(self.layout),
        }

    def recommend(
        self,
        history: Sequence[int],
        *,
        k: int,
        mode: str,
        required_frontier: int | None,
        max_hist_items: int,
    ) -> dict[str, Any]:
        """Run one retrieval for one caller-supplied history."""
        frontier = (
            int(required_frontier)
            if required_frontier is not None
            else int(k) + len(set(int(value) for value in history))
        )
        started = time.perf_counter()
        encoder_tokens = build_history_input(
            history,
            assignment=self.assignment,
            layout=self.layout,
            max_hist_items=max_hist_items,
        )
        scorer = TigerCaseScorer(
            self.generator,
            encoder_tokens=encoder_tokens,
            batch_prefixes=self.batch_prefixes,
            device=self.device,
        )
        config = RetrievalConfig(
            mode=mode,
            k=k,
            beam=self.beam,
            prefix_budget=self.prefix_budget,
            batch_prefixes=self.batch_prefixes,
        )
        result = retrieve(
            self.trie,
            scorer,
            config=config,
            required_frontier=frontier,
            case_id=-1,
            resolver=self.resolver,
        )
        items = [
            {
                "item_id": int(candidate.item_id),
                "rank": int(candidate.rank),
                "score": float(candidate.score),
                "semantic_id": [int(value) for value in candidate.semantic_id],
                "token_path": [int(value) for value in candidate.token_path],
                "resolver": dict(candidate.resolver.as_dict()),
            }
            for candidate in result.candidates
        ]
        return {
            "retrieval_mode": str(result.status),
            "requested_mode": str(mode),
            "k": int(k),
            "required_frontier": int(frontier),
            "history_length": len(history),
            "unique_history_items": len(set(int(value) for value in history)),
            "encoder_tokens": len(encoder_tokens),
            "wall_seconds": time.perf_counter() - started,
            "summary": {
                "status": str(result.status),
                "prefixes_expanded": int(result.prefixes_expanded),
                "nodes_pruned_by_bound": int(result.nodes_pruned_by_bound),
                "budget_exhausted": bool(result.budget_exhausted),
                "threshold_final": float(result.threshold_final),
                "exhaustion_reason": result.exhaustion_reason,
                "candidates": len(items),
            },
            "items": items,
        }

    # Budget knobs are plain attributes so the CLI can set them after construction.
    beam: int = 512
    prefix_budget: int = 200_000
    batch_prefixes: int = 256


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def _emit(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, sort_keys=True) + "\n")
    sys.stdout.flush()


def _error(code: str, message: str, *, request_id: Any = None, fatal: bool = False) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "op": "error",
        "code": code,
        "message": message[:800],
        "fatal": bool(fatal),
    }
    if request_id is not None:
        frame["request_id"] = request_id
    return frame


def _validate_request(frame: dict[str, Any], *, k_max: int, max_hist_items: int) -> dict[str, Any]:
    if frame.get("op") != "recommend":
        raise ValueError(f"unknown op {frame.get('op')!r}")
    history = frame.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty list of integer item ids")
    for value in history:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"history items must be integer ids >= 1 (PAD 0 is not an item), got {value!r}")
    k = frame.get("k", 20)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= k_max:
        raise ValueError(f"k must be an integer in 1..{k_max}, got {k!r}")
    mode = str(frame.get("mode") or CERTIFIED)
    if mode not in (APPROXIMATE, CERTIFIED):
        raise ValueError(f"mode must be {APPROXIMATE} or {CERTIFIED}, got {mode!r}")
    frontier = frame.get("required_frontier")
    if frontier is not None:
        if isinstance(frontier, bool) or not isinstance(frontier, int) or frontier < 1:
            raise ValueError(f"required_frontier must be a positive int or null, got {frontier!r}")
    per_request_hist = frame.get("max_hist_items", max_hist_items)
    if isinstance(per_request_hist, bool) or not isinstance(per_request_hist, int) or per_request_hist < 1:
        raise ValueError(f"max_hist_items must be a positive int, got {per_request_hist!r}")
    return {
        "history": [int(value) for value in history],
        "k": int(k),
        "mode": mode,
        "required_frontier": None if frontier is None else int(frontier),
        "max_hist_items": int(per_request_hist),
    }


def serve(assets: Assets, *, k_max: int, max_hist_items: int) -> int:
    """Read request frames until stdin closes; return a process exit code."""
    _emit(
        {
            "op": "ready",
            "contract": CONTRACT,
            "device": assets.device,
            "default_mode": CERTIFIED,
            "k_max": int(k_max),
            "max_hist_items": int(max_hist_items),
            "load_seconds": round(assets.load_seconds, 3),
            "resolver": assets.resolver_record,
            "identity": assets.identity,
        }
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(_error(CODE_REQUEST, f"frame is not JSON: {exc}"))
            continue
        if not isinstance(frame, dict):
            _emit(_error(CODE_REQUEST, "frame must be a JSON object"))
            continue
        op = frame.get("op")
        if op == "shutdown":
            _emit({"op": "bye"})
            return 0
        if op == "ping":
            _emit({"op": "pong", "contract": CONTRACT, "identity": assets.identity})
            continue
        request_id = frame.get("request_id")
        try:
            request = _validate_request(frame, k_max=k_max, max_hist_items=max_hist_items)
        except ValueError as exc:
            _emit(_error(CODE_REQUEST, str(exc), request_id=request_id))
            continue
        try:
            payload = assets.recommend(
                request["history"],
                k=request["k"],
                mode=request["mode"],
                required_frontier=request["required_frontier"],
                max_hist_items=request["max_hist_items"],
            )
        except Exception as exc:  # noqa: BLE001 - a failed request must not kill the worker
            _emit(
                _error(
                    CODE_RETRIEVAL,
                    f"{type(exc).__name__}: {exc}",
                    request_id=request_id,
                )
            )
            continue
        payload.update(
            {
                "op": "recommend_result",
                "request_id": request_id,
                "contract": CONTRACT,
                "checkpoint_sha256": assets.identity["checkpoint_sha256"],
                "sid_artifact_sha256": assets.identity["sid_artifact_sha256"],
                "resolver": assets.resolver_record,
            }
        )
        _emit(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="long-lived TIGER serving worker (JSONL on stdio)")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="the accepted step-2.5 generator directory (tiger.pt + tiger.json)")
    parser.add_argument("--sid", type=Path, required=True, help="the matching sid/ directory")
    parser.add_argument("--catalogue", type=Path, required=True,
                        help="the handoff directory (catalogue.json + train_exposure.jsonl)")
    parser.add_argument("--exposure", type=Path, default=None)
    parser.add_argument("--generator-layout", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--max-hist-items", type=int, default=20)
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--prefix-budget", type=int, default=200_000)
    parser.add_argument("--batch-prefixes", type=int, default=256)
    parser.add_argument("--self-check", action="store_true",
                        help="verify the assets and exit without loading the generator")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exposure = args.exposure or (args.catalogue / "train_exposure.jsonl")
    layout_path = args.generator_layout or (args.checkpoint_dir / "generator_layout.json")

    if args.self_check:
        # Asset-level check only: no torch import, no weights.  Used by the AgentRec-X side's
        # integration test to exercise the real subprocess protocol without the checkpoint.
        problems = [
            f"{label} missing at {path}"
            for label, path in (
                ("checkpoint directory", args.checkpoint_dir),
                ("checkpoint weights", args.checkpoint_dir / "tiger.pt"),
                ("checkpoint metadata", args.checkpoint_dir / "tiger.json"),
                ("generator layout", layout_path),
                ("sid directory", args.sid),
                ("sid assignment", args.sid / "semantic_ids.json"),
                ("sid token layout", args.sid / "layout.json"),
                ("handoff catalogue", args.catalogue / "catalogue.json"),
                ("train exposure", exposure),
            )
            if not path.exists()
        ]
        _emit(
            {
                "op": "self_check",
                "contract": CONTRACT,
                "ok": not problems,
                "problems": problems,
                "paths": {
                    "checkpoint_dir": str(args.checkpoint_dir),
                    "sid_dir": str(args.sid),
                    "catalogue_dir": str(args.catalogue),
                    "exposure": str(exposure),
                    "generator_layout": str(layout_path),
                },
            }
        )
        return 0 if not problems else EXIT_STARTUP_FAILURE

    try:
        assets = Assets(
            checkpoint_dir=args.checkpoint_dir,
            sid_dir=args.sid,
            catalogue_dir=args.catalogue,
            exposure_path=exposure,
            generator_layout_path=layout_path,
            device=args.device,
        )
    except WorkerStartupError as exc:
        _emit(_error(exc.code, exc.message, fatal=True))
        return EXIT_STARTUP_FAILURE
    except Exception as exc:  # noqa: BLE001 - startup must report, not traceback into a hang
        _emit(_error(CODE_INTERNAL, f"{type(exc).__name__}: {exc}", fatal=True))
        return EXIT_STARTUP_FAILURE

    assets.beam = int(args.beam)
    assets.prefix_budget = int(args.prefix_budget)
    assets.batch_prefixes = int(args.batch_prefixes)
    print(
        f"[tiger-serve] ready device={assets.device} items={assets.catalogue.num_items} "
        f"load={assets.load_seconds:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return serve(assets, k_max=int(args.k_max), max_hist_items=int(args.max_hist_items))


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
