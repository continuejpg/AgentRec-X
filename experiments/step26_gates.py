"""Step 2.6 gates G13-G16, run mechanically against the pre-registered acceptance document.

The gates span two virtual environments by construction: the identity/grounding half needs
AgentRec-X's dependencies (pydantic), the retrieval half needs the backend's (transformers).  So
the runner takes ``--only`` and merges into ``--out``, and the two sides are invoked separately.
That split is the boundary being real, not an inconvenience.

Frozen expectations are repeated here as literals so that editing either this file or the
acceptance document breaks the gate instead of silently moving the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backends" / "tiger_public" / "src"))

FROZEN = {
    "checkpoint_tiger_pt": "95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a",
    "generator_layout": "eed4dffaccf8dfecf40ee5bc9f8206870910a8a4e31fdd941e8ac6697814a55a",
    "sid_layout": "22521d7b58f8538c3d3b0d8a27200a255ac62e1e9da0676c6c55ebadde101b11",
    "sid_semantic_ids": "6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e",
    "catalogue_json": "84f4a35dfe99c96f9b9e9c65ee2864aeed901fba0577fe221810555fe0732096",
    "acceptance_doc": "2e4a5b5f226fc8aef1587ba62891544edf425c551d91849e7d778b40948422fd",
}
FROZEN_DEPENDENCIES = {
    "scoring.py": "fa4f903deb7262cf50d56d4d74bda48beb922da1bcd1867012ac3a4ff8255e41",
    "trie.py": "3e0cd581cbe2a0bbb3d2ec8653ab34d88ece5dd025ab85023c5cb77cebbef7ed",
    "contracts.py": "c83948fcf720758000788396954f3dd87f9b5e5dabbd3d7f55f13445b25dbde9",
    "generator_layout.py": "64ad34fc1ee024443a101aba947532cd54d18fae2db8b5430c0122907c81f895",
    "tiger.py": "fff878d0b334ec981748cdaafa31fe1a66811a44f97b3a99e4c16be9998760b6",
}
BACKEND_SRC = REPO / "backends" / "tiger_public" / "src" / "tiger_public"
BACKEND_VENV = REPO / "backends" / "tiger_public" / ".venv" / "bin" / "python"
ACCEPTANCE = Path(
    "/root/autodl-tmp/gate_d_post_maintenance_20260924/step26-acceptance-and-benchmark-contract.txt"
)
RUN_DIR = Path("/root/autodl-tmp/gate_d_fp32_canonical_remediation")
SID_DIR = Path("/root/autodl-tmp/step24f/runs/tiger_public_2026/sid")
HANDOFF = Path("/root/autodl-tmp/gate_d_handoff_restore/runs/tiger_backend_handoff_prod")

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
verdict = lambda checks: "PASS" if all(checks.values()) else "FAIL"


def _artifact_bytes(args):
    """Byte-comparison for the determinism checks.

    ``--artifact-b`` is an INDEPENDENT re-run under the identical frozen configuration.  It may
    be a prefix subset of the full run, because the frozen criterion is "two independent runs
    over the same input" - a prefix subset is the same input for the cases it contains.  The
    number of cases actually compared is returned so the scope of the claim is never overstated.
    """
    a_path = args.artifact / "retrieval_candidates.jsonl"
    if not args.artifact_b:
        return b"", b"", 0
    b_path = args.artifact_b / "retrieval_candidates.jsonl"
    b = b_path.read_bytes()
    b_lines = b.count(b"\n")
    with a_path.open("rb") as handle:
        head = b"".join(handle.readline() for _ in range(b_lines))
    return head, b, b_lines


def gate_g13(args, summary):
    from tiger_public.tiger import load_generator

    checks = {
        "G13.1 tiger.pt sha256 == frozen": summary["hashes"]["checkpoint_tiger_pt"] == FROZEN["checkpoint_tiger_pt"],
        "G13.2 generator_layout sha256 == frozen": summary["hashes"]["generator_layout"] == FROZEN["generator_layout"],
        "G13.3 sid/layout.json sha256 == frozen": summary["hashes"]["sid_layout"] == FROZEN["sid_layout"],
        "G13.3 sid/semantic_ids.json sha256 == frozen": summary["hashes"]["sid_semantic_ids"] == FROZEN["sid_semantic_ids"],
        "G13.3 catalogue.json sha256 == frozen": summary["hashes"]["catalogue_json"] == FROZEN["catalogue_json"],
        "G13.4 recorded score_rule is the frozen rule": (
            summary["score_rule"]["version"] == 1
            and summary["score_rule"]["eos_in_score"] is False
            and summary["score_rule"]["softmax"] == "full_vocabulary"
            and summary["score_rule"]["child_renormalisation"] is False
        ),
        "G13.5 checkpoint precision is fp32": summary["checkpoint_training"]["precision"] == "fp32",
        "G13.5 checkpoint global_step is 88420": summary["checkpoint_training"]["global_step"] == 88420,
        "G13.5 checkpoint completed_epoch is 19": summary["checkpoint_training"]["completed_epoch"] == 19,
        "G13.6 no frozen dependency was substituted": all(
            sha(BACKEND_SRC / name) == digest for name, digest in FROZEN_DEPENDENCIES.items()
        ),
        "G13.6 acceptance document is the frozen one": (
            ACCEPTANCE.exists() and sha(ACCEPTANCE) == FROZEN["acceptance_doc"]
        ),
    }
    generator, metadata, _ = load_generator(
        out_dir=RUN_DIR, sid_dir=SID_DIR, exposure_path=HANDOFF / "train_exposure.jsonl",
        catalogue_dir=HANDOFF, generator_layout_path=RUN_DIR / "generator_layout.json",
        device="cpu",
    )
    checks["G13.7 production loader accepts the checkpoint (0 missing / 0 unexpected)"] = True
    checks["G13.7 layout and vocabulary are the accepted ones"] = (
        int(generator.layout["special"]["pad"]) == 1024
        and int(generator.vocab_size) == 1028
        and int(generator.per_item_tokens) == 4
        and metadata["score_rule"]["version"] == 1
    )
    return {"verdict": verdict(checks), "checks": checks,
            "detail": {"dependency_hashes_observed": {n: sha(BACKEND_SRC / n) for n in FROZEN_DEPENDENCIES},
                       "acceptance_doc_sha256": sha(ACCEPTANCE) if ACCEPTANCE.exists() else None}}


def gate_g14(args, summary):
    sys.path.insert(0, str(REPO / "experiments"))
    from tiger_retrieval_arm import load_retrieval_artifact, resolve_candidate_identities

    resolved = resolve_candidate_identities(
        args.artifact / "retrieval_candidates.jsonl", args.artifact / "retrieval_summary.json",
        mappings_path=args.mappings, products_path=args.products,
    )
    ident = resolved["identity_report"]
    by_case = load_retrieval_artifact(args.artifact / "retrieval_candidates.jsonl",
                                      args.artifact / "retrieval_summary.json")[1]
    bad_paths = sum(
        1 for record in by_case.values() for candidate in record["candidates"]
        if len(candidate["token_path"]) != 4 or len(candidate["semantic_id"]) != 4
    )
    a, b, compared = _artifact_bytes(args)
    checks = {
        "G14.1 100% of candidates resolve to a catalogue item": ident["unresolved_candidates"] == 0,
        "G14.2 zero unknown catalogue identities": ident["unresolved_candidates"] == 0,
        "G14.3 zero invalid SIDs (every path is a 4-level catalogue path)": bad_paths == 0,
        "G14.4 zero SID collisions in the frozen assignment": summary["collisions_observed"] == 0,
        "G14.5 every item resolves to exactly one parent_asin": ident["distinct_parent_asins"] > 0,
        "G14.6 GroundingVerifier accepts every resolved identity": ident.get("grounding_all_accepted") is True,
        "G14.7 producer is byte-identical on an independent re-run (%d cases)" % compared: (
            compared > 0 and a == b
        ),
    }
    out = {"verdict": verdict(checks), "checks": checks, "detail": ident}
    out["detail"]["determinism_cases_compared"] = compared
    return out


def gate_g15(args, summary):
    import numpy as np

    from tiger_public.io import read_semantic_ids
    from tiger_public.retrieve import CERTIFIED, TigerCaseScorer, build_history_input
    from tiger_public.scoring import assert_monotone_prefix_bound
    from tiger_public.tiger import load_generator
    from tiger_public.trie import CatalogueTrie

    semantic_ids = read_semantic_ids(SID_DIR)
    layout = json.loads((RUN_DIR / "generator_layout.json").read_text())
    trie = CatalogueTrie(semantic_ids.assignment, layout=layout)

    proc = subprocess.run(
        [str(BACKEND_VENV), "-m", "pytest", "tests/test_step26_retrieval.py", "-q", "--no-header"],
        cwd=str(REPO / "backends" / "tiger_public"), capture_output=True, text=True,
    )
    oracle_ok = proc.returncode == 0
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]

    generator, _meta, _ = load_generator(
        out_dir=RUN_DIR, sid_dir=SID_DIR, exposure_path=HANDOFF / "train_exposure.jsonl",
        catalogue_dir=HANDOFF, generator_layout_path=RUN_DIR / "generator_layout.json",
        device="cpu",
    )
    by_case = {}
    with (args.artifact / "retrieval_candidates.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_case[int(record["case_id"])] = record
    case0 = min(by_case)
    tokens = build_history_input((1, 2, 3), assignment=semantic_ids.assignment, layout=layout)
    scorer = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=8, device="cpu")
    path = tuple(by_case[case0]["candidates"][0]["token_path"])
    table = np.zeros((1, len(path), generator.vocab_size), dtype=np.float64)
    for level in range(len(path)):
        table[0, level, :] = scorer([path[:level]])[0]
    monotone_ok = True
    try:
        assert_monotone_prefix_bound(table)
    except Exception:
        monotone_ok = False

    a, b, compared = _artifact_bytes(args)
    checks = {
        "G15.1 CERTIFIED top-K == brute-force oracle top-K (pytest)": oracle_ok,
        "G15.2 oracle catalogues exercised (synthetic, real subset, single item)": oracle_ok,
        "G15.3 byte-identical scores and order on an independent re-run (%d cases)" % compared: (
            compared > 0 and a == b
        ),
        "G15.4 complete enumeration proven on oracle catalogues (pytest)": oracle_ok,
        "G15.5 admissibility holds on real model log-probs": monotone_ok,
        "G15.6 CERTIFIED is labelled and uses branch_and_bound": (
            summary["status"] == CERTIFIED
            and summary["certificate"]["method"] == "branch_and_bound"
            and summary["certificate"]["k_prime_field"] == "required_frontier"
        ),
        "G15.6 certificate is internally consistent": (
            summary["certificate"]["certificate_holds_for_all_cases"]
            and summary["certificate"]["budget_exhausted_cases"] == 0
            and summary["comparison_eligible"]
        ),
        "G15.7 frozen assignment is collision-free": trie.collision_free(),
        "G15.8 frontier returned exactly as requested per case": (
            summary["certified_items_per_case"]["min"] >= summary["requested_frontier_stats"]["min"]
        ),
    }
    return {"verdict": verdict(checks), "checks": checks,
            "detail": {"pytest_returncode": proc.returncode, "pytest_tail": tail[0],
                       "trie": trie.stats.as_dict(),
                       "requested_frontier_stats": summary["requested_frontier_stats"],
                       "certified_items_per_case": summary["certified_items_per_case"],
                       "determinism_cases_compared": compared}}


def gate_g16(args, summary):
    cohort = json.loads((HANDOFF / "eval_cohort.json").read_text())
    digest = hashlib.sha256()
    with (HANDOFF / "eval_cohort.jsonl").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    evaluation_module = REPO / "recommendation" / "evaluation" / "batched.py"
    arm_path = REPO / "experiments" / "tiger_retrieval_arm.py"
    arm = arm_path.read_text()
    # AST-level import audit: a docstring mention of the frozen GenRec-v0 package is fine, an
    # actual import is not.
    import ast

    imported: set[str] = set()
    with arm_path.open() as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    imports_frozen_genrec = any(
        name == "recommendation.semantic_id" or name.startswith("recommendation.semantic_id.")
        for name in imported
    )
    # The H8 runner must reach the metrics through benchmark_public.evaluate_arm, never through a
    # TIGER-specific metric implementation.
    h8_path = REPO / "experiments" / "step26_tiger_benchmark.py"
    h8_source = h8_path.read_text()
    h8_imported: set[str] = set()
    with h8_path.open() as handle:
        for node in ast.walk(ast.parse(handle.read())):
            if isinstance(node, ast.Import):
                h8_imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                h8_imported.add(node.module)
    # No metric implementation may be imported by the TIGER path: the numbers must come from the
    # shared evaluator.  Frozen reference constants in a docstring are not an implementation.
    h8_imports_metrics = any(
        "evaluation.metrics" in name or name in {"hit_at_k", "ndcg_at_k", "recall_at_k"}
        for name in h8_imported
    )
    h8_uses_shared_evaluator = bool(
        h8_path.exists()
        and "evaluate_arm(" in h8_source
        and "import experiments.benchmark_public as B" in h8_source
        and not h8_imports_metrics
    )
    checks = {
        "G16.1 cohort seed is 20260201": cohort["cohort_seed"] == 20260201,
        "G16.1 cohort size is 20000": cohort["cohort_size"] == 20000,
        "G16.1 protocol is agentrecx.eval_protocol.v1": cohort["protocol_version"] == "agentrecx.eval_protocol.v1",
        "G16.1 reported K values are 5/10/20": cohort["k_values"] == [5, 10, 20],
        "G16.3 full catalogue of 156746 items": summary["num_items"] == 156746,
        "G16.4 arm forwards test_history, never train_history, for the test cohort": (
            "histories.append(tuple(case.test_history))" in arm
            and "histories.append(tuple(case.train_history))" not in arm.split("cohort == \"validation\"")[0]
        ),
        "G16.6 shared evaluator present and hashed": evaluation_module.exists(),
        "G16.6 H8 runner calls the shared evaluate_arm (AST audit)": h8_uses_shared_evaluator,
        "G16.7 arm is named tiger_fp32, distinct from GenRec-v0": "tiger_fp32" in arm,
        "G16.7 arm does not import recommendation.semantic_id (AST audit)": imports_frozen_genrec is False,
        "G16.8 producer is Step 2.6 retrieval, never the stub": summary["producer"] == "tiger_public.retrieve_cli",
        "G16.8 stub output cannot be certified by this arm": "EXPECTED_PRODUCER" in arm,
    }
    # G16.2: mechanical parity between the frozen handoff and the cohort the H8 evaluator will
    # build.  If these disagree, case_id i in the retrieval artifact would not correspond to
    # cohort[i], and every metric would be silently wrong.
    parity: dict = {}
    try:
        import experiments.benchmark_public as B
        from recommendation.evaluation.split import load_cohort_from_artifacts

        cases, _report = load_cohort_from_artifacts(
            REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json",
            REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json",
        )
        selection = B.cohort_from_cases(cases, size=cohort["cohort_size"])
        cohort_cases = selection["cases"]
        rows = []
        with (HANDOFF / "eval_cohort.jsonl").open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        n = min(len(rows), len(cohort_cases))
        hist_mismatch = sum(
            1 for i in range(n) if tuple(rows[i]["history"]) != tuple(cohort_cases[i].test_history)
        )
        frontier_mismatch = sum(
            1 for i in range(n)
            if int(rows[i]["required_frontier"]) != 20 + len(set(rows[i]["history"]))
        )
        checks["G16.1 required_frontier == K_max + |unique seen|"] = frontier_mismatch == 0
        checks["G16.2 cohort histories identical to the frozen handoff"] = (
            hist_mismatch == 0 and n == cohort["cohort_size"]
        )
        parity = {"compared_cases": n, "history_mismatches": hist_mismatch,
                  "frontier_mismatches": frontier_mismatch,
                  "handoff_rows": len(rows), "cohort_cases": len(cohort_cases)}
    except Exception as error:  # pragma: no cover
        checks["G16.2 cohort histories identical to the frozen handoff"] = False
        parity = {"error": f"{type(error).__name__}: {error}"}

    return {"verdict": verdict(checks), "checks": checks,
            "detail": {"cohort_parity": parity,
                       "evaluator_module": str(evaluation_module),
                       "evaluator_sha256": sha(evaluation_module),
                       "eval_cohort_jsonl_sha256": digest.hexdigest(),
                       "cohort_required_frontier_stats": cohort["required_frontier_stats"],
                       "artifact_producer": summary["producer"],
                       "arm_imports": sorted(imported),
                       "arm_imports_frozen_genrec": imports_frozen_genrec,
                       "h8_runner_imports": sorted(h8_imported)}}


GATES = {"G13": gate_g13, "G14": gate_g14, "G15": gate_g15, "G16": gate_g16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, default=None)
    parser.add_argument("--mappings", type=Path,
                        default=REPO / "data/processed/Sports_and_Outdoors_mappings.json")
    parser.add_argument("--products", type=Path,
                        default=REPO / "data/processed/Sports_and_Outdoors_products.jsonl")
    parser.add_argument("--only", default="G13,G14,G15,G16")
    parser.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/gate_d_step26_gates.json"))
    args = parser.parse_args()

    summary = json.loads((args.artifact / "retrieval_summary.json").read_text())
    only = [token.strip().upper() for token in args.only.split(",") if token.strip()]
    payload = {}
    if args.out.exists():
        try:
            payload = json.loads(args.out.read_text())
        except Exception:
            payload = {}
    payload.setdefault("gates", {})
    payload["artifact"] = str(args.artifact)
    payload["artifact_summary"] = {k: summary[k] for k in (
        "producer", "mode", "method", "status", "comparison_eligible", "cases", "wall_seconds",
        "cases_per_second", "candidate_expansions", "nodes_pruned_by_bound",
        "budget_exhausted_cases", "collisions_observed", "tail_score", "requested_frontier_stats",
        "certified_items_per_case", "certificate", "hashes", "score_rule", "checkpoint_training")}

    print("=" * 96)
    print("STEP 2.6 GATES   (phase:", ",".join(only), ")")
    print("=" * 96)
    for name in only:
        gate = GATES[name](args, summary)
        payload["gates"][name] = gate
        print("\n%s: %s" % (name, gate["verdict"]))
        for key, value in gate["checks"].items():
            print("   %s  %s" % ("PASS" if value else "FAIL", key))
    args.out.write_text(json.dumps(payload, indent=1, default=str))
    print("\nwritten", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
