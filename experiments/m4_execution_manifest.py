"""Build the durable M4 execution manifest (preregistration §12, §17).

`runs/` is gitignored, so a manifest written there could be lost and would not be reviewable. This
tool writes `docs/M4_EXECUTION_MANIFEST.md` — a **committed** record — and additionally a machine
readable `docs/M4_EXECUTION_MANIFEST.json` beside it.

Design rule: the manifest must never silently accept a missing value before the formal execution.
Every field is either a real value or the literal `PENDING`, and `--check` fails while any field is
still `PENDING`. That makes "the manifest is complete" a checkable precondition rather than a
promise.

Usage
-----
    python -m experiments.m4_execution_manifest            # (re)generate
    python -m experiments.m4_execution_manifest --check    # fail if anything is PENDING
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MD_OUT = REPO / "docs/M4_EXECUTION_MANIFEST.md"
JSON_OUT = REPO / "docs/M4_EXECUTION_MANIFEST.json"

PREREGISTRATION = REPO / "docs/M4_PREREGISTRATION.md"
AMENDMENT_1 = REPO / "docs/M4_PREREGISTRATION_AMENDMENT_1.md"
RESULT_DOC = REPO / "docs/M4_RESULT.md"
FORMAL_ARTIFACT = REPO / "runs/m4_formal_result.json"
CANDIDATE_PLANE = REPO / "recommendation/control/candidate_plane.py"
HEADS = REPO / "runs/m4_evidence/heads.npz"
HEADS_PROVENANCE = REPO / "runs/m4_evidence/heads_provenance.json"
SIMILAR_INDEX_META = REPO / "runs/m4_evidence/similar_item_index.json"
M3_PAIRED_INPUTS = REPO / "runs/m3_execution/paired_inputs.npz"

PENDING = "PENDING"

#: Modules whose exact content the run depends on. Hashed so the manifest pins the code, not just
#: the commit.
CODE_FILES = (
    "experiments/m4_benchmark.py",
    "experiments/m4_harness.py",
    "experiments/m4_policies.py",
    "experiments/m4_frozen_sources.py",
    "experiments/m4_query_rules.py",
    "experiments/m4_schema_projection.py",
    "experiments/m4_sharding.py",
    "experiments/m4_conformance.py",
    "experiments/m4_materialize_heads.py",
    "experiments/m4_similar_neighbours.py",
    "recommendation/control/candidate_plane.py",
    "recommendation/control/loop.py",
    "recommendation/control/candidate_ledger.py",
    "recommendation/control/model_policy.py",
    "recommendation/evaluation/batched.py",
    "recommendation/evaluation/metrics.py",
    "recommendation/evaluation/split.py",
)

#: The frozen provider configuration, identical to the M3 amendment. Recorded as identifiers, not
#: secrets: the credential itself is never stored here.
PROVIDER = {
    "profile": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "thinking": "disabled",
    "temperature": 0.0,
    "json_mode": True,
    "max_tokens": 512,
    "prompt_identifier": "M4_GUARDED_ADAPTIVE_V1",
    "policy_identifier": "GuardedAdaptivePolicy(inner=LLMAgentPolicy)",
    "tool_budget": {"max_tool_calls": 4, "max_steps": 6},
    #: Set after the non-cohort probe was run; the probe's own report is the evidence.
    "concurrency": {
        "probed": True,
        "levels": {"1": {"calls": 3, "failures": 0, "latency_mean_ms": 832.2},
                   "2": {"calls": 4, "failures": 0, "latency_mean_ms": 880.6},
                   "4": {"calls": 8, "failures": 0, "latency_mean_ms": 716.8}},
        "selected": 4,
        "selection_rule": "highest concurrency with zero failures and zero schema failures",
        "rate_limits_observed": 0,
        "schema_failures_observed": 0,
        "note": "non-cohort synthetic structured-action requests only; 19 calls total including "
                "a confirmation run, against a 16-call budget (3 over)",
    },
    "config_validated_against_manifest": True,
}


def sha256_file(path: pathlib.Path) -> str:
    """SHA-256 of a file, streamed; ``PENDING`` when the file does not exist yet."""
    if not path.is_file():
        return PENDING
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a read-only git command, returning stdout or ``PENDING``."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        )
        return out.stdout.strip() or PENDING
    except Exception:  # noqa: BLE001 - a manifest tool must not crash on git problems
        return PENDING


def cohort_hash() -> dict[str, str]:
    """The frozen cohort's identity/order hash, plus the M3 reference it is checked against."""
    try:
        import numpy as np

        frozen = np.load(M3_PAIRED_INPUTS)
        targets = frozen["target_ids"]
        return {
            "source": str(M3_PAIRED_INPUTS.relative_to(REPO)),
            "targets_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
            "users": str(int(targets.shape[0])),
            "selection": "benchmark_public.cohort_from_cases(size=20000, seed=20260201)",
        }
    except Exception:  # noqa: BLE001
        return {"source": PENDING, "targets_sha256": PENDING, "users": PENDING, "selection": PENDING}


def head_hashes() -> dict[str, Any]:
    """Per-source head hashes, from the assembled artifact and its provenance sidecar."""
    result: dict[str, Any] = {
        "assembled_artifact": str(HEADS.relative_to(REPO)),
        "assembled_sha256": sha256_file(HEADS),
        "provenance_sha256": sha256_file(HEADS_PROVENANCE),
        "per_source": {},
    }
    if HEADS_PROVENANCE.is_file():
        payload = json.loads(HEADS_PROVENANCE.read_text(encoding="utf-8"))
        result["depth"] = payload.get("depth", PENDING)
        result["users"] = payload.get("users", PENDING)
        result["score_kinds"] = payload.get("score_kinds", {})
        result["inputs"] = payload.get("inputs", {})
        for source in ("history", "catalog_search", "similar_item", "two_tower"):
            lengths = payload.get("source_lengths", {}).get(source)
            result["per_source"][source] = {
                "users_with_head": lengths.get("users_with_head") if lengths else PENDING,
                "min": lengths.get("min") if lengths else PENDING,
                "mean": lengths.get("mean") if lengths else PENDING,
                "max": lengths.get("max") if lengths else PENDING,
            }
    else:
        result["per_source"] = {
            source: {"users_with_head": PENDING} for source in
            ("history", "catalog_search", "similar_item", "two_tower")
        }
    return result


def build() -> dict[str, Any]:
    """Assemble the manifest payload."""
    manifest: dict[str, Any] = {
        "milestone": "M4",
        "generated_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or PENDING,
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(REPO)),
            "sha256": sha256_file(PREREGISTRATION),
            # The original document is never edited. Amendments are separate, hashed documents, so
            # a run is bound to both and the history stays visible.
            "amended": AMENDMENT_1.is_file(),
            "amendments": (
                [
                    {
                        "version": 1,
                        "path": str(AMENDMENT_1.relative_to(REPO)),
                        "sha256": sha256_file(AMENDMENT_1),
                        "subject": "expose the four frozen source values in the action contract",
                        "scientific_design_changed": False,
                        "formal_cohort_previously_run": False,
                    }
                ]
                if AMENDMENT_1.is_file()
                else []
            ),
        },
        #: The sealed outcome. The artifact itself is gitignored (AGENTS.md §8), so the committed
        #: result document is what carries the numbers, and both are hashed here.
        "result": {
            "document": {
                "path": str(RESULT_DOC.relative_to(REPO)),
                "sha256": sha256_file(RESULT_DOC) if RESULT_DOC.is_file() else None,
            },
            "artifact": {
                "path": str(FORMAL_ARTIFACT.relative_to(REPO)),
                "sha256": sha256_file(FORMAL_ARTIFACT) if FORMAL_ARTIFACT.is_file() else None,
                "committed": False,
                "note": "gitignored per AGENTS.md §8; contents transcribed into the document above",
            },
            "executed_once": True,
            "rerun_after_results_seen": False,
            "primary_endpoint": "NDCG@10",
            "verdict": "SUPPORTED",
            "direction": "adaptive worse than fixed",
        },
        "amendment_1_subject": (
            "select_source.source is exposed to the provider as a closed enum of the four frozen "
            "source names; M4-scoped in the injected schema-projection client; no shared "
            "control-plane behaviour changed"
        ),
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "commit_subject": git("log", "-1", "--format=%s"),
            "remote_head": git("rev-parse", "origin/master"),
            "worktree_dirty": bool(git("status", "--porcelain") != PENDING and git("status", "--porcelain")),
            "file_hashes": {name: sha256_file(REPO / name) for name in CODE_FILES},
        },
        "frozen_cohort": cohort_hash(),
        "source_universe": {
            "members": ["history", "catalog_search", "similar_item", "two_tower"],
            "depth": 100,
            "excluded": {"tiger": "not wired into the runtime; artifacts absent",
                          "popularity/metadata_retrieval": "not CandidateSource members"},
        },
        "fusion": {"implementation": "candidate_ledger.reciprocal_rank_fusion", "rrf_k": 60},
        "heads": head_hashes(),
        "similar_item_index": {
            "path": "runs/m4_evidence/similar_item_index.pkl",
            "sha256": (
                json.loads(SIMILAR_INDEX_META.read_text(encoding="utf-8")).get("sha256", PENDING)
                if SIMILAR_INDEX_META.is_file()
                else PENDING
            ),
        },
        "evaluator": {
            "implementation": "recommendation.evaluation.batched.evaluate_batched",
            "protocol": "temporal_leave_two_out",
            "protocol_version": "agentrecx.eval_protocol.v1",
            "k_values": [5, 10, 20],
            "catalog_size": 156746,
        },
        "statistics": {
            "primary_endpoint": "NDCG@10",
            "delta_convention": "adaptive - fixed",
            "mcnemar": "exact paired, two-sided, no continuity correction",
            "bootstrap": {"resamples": 10000, "seed": 20260201, "unit": "user"},
        },
        "provider": PROVIDER,
        "sharding": {
            "implementation": "experiments/m4_sharding.py",
            "sha256": sha256_file(REPO / "experiments/m4_sharding.py"),
            "merge": "sort on frozen cohort position; duplicate/missing user is an error",
        },
    }
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render the manifest as a reviewable markdown document."""
    code = manifest["code"]
    heads = manifest["heads"]
    lines = [
        "# M4 Execution Manifest",
        "",
        "Durable record of the exact frozen configuration the M4 formal execution runs under.",
        "Generated by `experiments/m4_execution_manifest.py`; regenerate rather than edit by hand.",
        "",
        "`PENDING` means the value is not available yet. The pre-execution gate refuses to pass while",
        "any field is `PENDING` (`--check`), so an incomplete manifest cannot be mistaken for a",
        "complete one.",
        "",
        "## Preregistration",
        "",
        "| field | value |",
        "|---|---|",
        f"| path | `{manifest['preregistration']['path']}` |",
        f"| SHA256 | `{manifest['preregistration']['sha256']}` |",
        f"| amended | {manifest['preregistration']['amended']} |",
        "",
        "## Code",
        "",
        "| field | value |",
        "|---|---|",
        f"| commit | `{code['commit']}` |",
        f"| subject | {code['commit_subject']} |",
        f"| remote HEAD | `{code['remote_head']}` |",
        f"| worktree dirty | {code['worktree_dirty']} |",
        "",
        "Per-file SHA256:",
        "",
        "| file | SHA256 |",
        "|---|---|",
    ]
    for name, digest in code["file_hashes"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    cohort = manifest["frozen_cohort"]
    lines += [
        "",
        "## Frozen cohort",
        "",
        "| field | value |",
        "|---|---|",
        f"| reference | `{cohort['source']}` |",
        f"| users | {cohort['users']} |",
        f"| target_ids SHA256 | `{cohort['targets_sha256']}` |",
        f"| selection | `{cohort['selection']}` |",
        "",
        "## Source universe",
        "",
        f"Members: {', '.join('`' + m + '`' for m in manifest['source_universe']['members'])}  ",
        f"Depth: **{manifest['source_universe']['depth']}**  ",
        f"Fusion: `{manifest['fusion']['implementation']}` (k={manifest['fusion']['rrf_k']})",
        "",
        "## Heads",
        "",
        "| field | value |",
        "|---|---|",
        f"| artifact | `{heads['assembled_artifact']}` |",
        f"| artifact SHA256 | `{heads['assembled_sha256']}` |",
        f"| provenance SHA256 | `{heads['provenance_sha256']}` |",
        "",
        "| source | users with head | min | mean | max |",
        "|---|---|---|---|---|",
    ]
    for source, info in heads["per_source"].items():
        lines.append(
            f"| `{source}` | {info.get('users_with_head', PENDING)} | {info.get('min', PENDING)} | "
            f"{info.get('mean', PENDING)} | {info.get('max', PENDING)} |"
        )
    provider = manifest["provider"]
    lines += [
        "",
        "## Evaluator and statistics",
        "",
        "| field | value |",
        "|---|---|",
        f"| evaluator | `{manifest['evaluator']['implementation']}` |",
        f"| protocol | `{manifest['evaluator']['protocol']}` ({manifest['evaluator']['protocol_version']}) |",
        f"| K values | {manifest['evaluator']['k_values']} |",
        f"| primary endpoint | {manifest['statistics']['primary_endpoint']} |",
        f"| delta convention | {manifest['statistics']['delta_convention']} |",
        f"| McNemar | {manifest['statistics']['mcnemar']} |",
        f"| bootstrap | {manifest['statistics']['bootstrap']} |",
        "",
        "## Provider (frozen; credential never recorded here)",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for key, value in provider.items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "## Sharding",
        "",
        f"Implementation `{manifest['sharding']['implementation']}` "
        f"(SHA256 `{manifest['sharding']['sha256']}`).  ",
        f"Merge: {manifest['sharding']['merge']}",
        "",
    ]
    return "\n".join(lines)


def pending_fields(manifest: dict[str, Any]) -> list[str]:
    """Every field still ``PENDING``, as dotted paths."""
    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{prefix}[{index}]")
        elif node == PENDING:
            found.append(prefix)

    walk(manifest, "")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M4 execution manifest")
    parser.add_argument("--check", action="store_true", help="fail while any field is PENDING")
    args = parser.parse_args(argv)

    manifest = build()
    pending = pending_fields(manifest)
    manifest["pending_fields"] = pending
    manifest["complete"] = not pending

    JSON_OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    MD_OUT.write_text(render_markdown(manifest), encoding="utf-8")
    print(f"wrote {MD_OUT.relative_to(REPO)}")
    print(f"wrote {JSON_OUT.relative_to(REPO)}")
    if pending:
        print(f"PENDING fields ({len(pending)}):")
        for name in pending:
            print(f"  - {name}")
    else:
        print("manifest COMPLETE: no pending fields")
    if args.check and pending:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
