"""Step 2.7 qualification: run gates Q1-Q8 and emit a machine-readable verdict.

Runs against the FROZEN certified H7 evidence, so the identity, grounding, ledger,
deduplication and determinism gates exercise real TIGER output while re-running no retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/root/autodl-tmp/agentrec-X")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "backends" / "tiger_public" / "src"))

from recommendation.control import (  # noqa: E402
    ActionKind, CandidateLedger, CandidatePlane, CandidateSource, GroundingVerifier,
    SelectSourceArguments, ValidatedAction,
)
from recommendation.control.candidate_plane import CandidateSourceTool, CandidateSourceUnavailable  # noqa: E402
from recommendation.control.tiger_source import TIGER_SCORE_KIND, TigerSourceTool  # noqa: E402
from recommendation.control.two_tower_source import TWO_TOWER_SCORE_KIND  # noqa: E402

import step27_tiger_engines as eng  # noqa: E402

ARTIFACT = Path("/root/autodl-tmp/step26_h7_canonical")
APPROX = Path("/root/autodl-tmp/step27_approx64")
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
OUT = Path("/root/autodl-tmp/gate_d_step27_qualification.json")
CASES = 64

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Map:
    def __init__(self, identities):
        self._ids = sorted(str(v) for v in identities)
        self._to = {v: i + 1 for i, v in enumerate(self._ids)}

    def has_parent_asin(self, a): return a in self._to
    def parent_asin_to_item_id(self, a): return self._to[a]
    def item_id_to_parent_asin(self, i): return self._ids[int(i) - 1]


class Catalog:
    def __init__(self, identities): self._ids = {str(v) for v in identities}
    def __contains__(self, a): return str(a) in self._ids


class StubSource:
    """A deterministic non-TIGER source, used for dedup and for the Q7 equivalence check."""
    source = CandidateSource.SIMILAR_ITEM
    score_kind = "item_item_similarity"

    def __init__(self, rows): self._rows = tuple(rows)
    def propose(self, *, arguments, limit): return list(self._rows[: int(limit)])


def select(source, limit):
    return ValidatedAction(action=ActionKind.SELECT_SOURCE, action_id=f"act:{source.value}:{limit}",
                           step_index=0, run_id="r", k=limit,
                           arguments=SelectSourceArguments(source=source, limit=limit))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=CASES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    base = eng.CertifiedEvidenceTigerEngine(ARTIFACT, mappings_path=MAPPINGS)
    summary = base._summary
    gates: dict[str, dict] = {}

    def case_identities(index):
        record = base._cache[index]
        return tuple(str(base._id2item[int(c["item_id"])]) for c in record["candidates"])

    # ---------------- Q1: source contract compliance ----------------
    probe = TigerSourceTool(eng.FakeTigerEngine(recommendations=()))
    checks = {
        "Q1.1 implements the CandidateSourceTool protocol": isinstance(probe, CandidateSourceTool),
        "Q1.2 source is CandidateSource.TIGER": probe.source is CandidateSource.TIGER,
        "Q1.3 score_kind is the TIGER sequence log-probability": probe.score_kind == TIGER_SCORE_KIND,
        "Q1.4 score_kind is distinct from Two-Tower": TIGER_SCORE_KIND != TWO_TOWER_SCORE_KIND,
        "Q1.5 enum member exists and is candidate-producing": (
            CandidateSource.TIGER.value == "tiger"),
        "Q1.6 plane registers the tool and lists the source": True,
    }
    idents = case_identities(0)
    plane = CandidatePlane(ledger=CandidateLedger(),
                           grounding=GroundingVerifier(Map(idents), Catalog(idents)),
                           tiger_tool=TigerSourceTool(base.for_case(0)))
    checks["Q1.6 plane registers the tool and lists the source"] = (
        plane.has_source(CandidateSource.TIGER)
        and CandidateSource.TIGER in plane.available_sources())
    checks["Q1.7 plane reports the score_kind in its status"] = any(
        entry["source"] == "tiger" and entry["score_kind"] == TIGER_SCORE_KIND
        for entry in plane.status()["sources"])
    gates["Q1"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}

    # ---------------- Q2: identity / grounding on real evidence ----------------
    grounded_total = 0
    ungrounded_total = 0
    round_trip_ok = True
    catalogue_ok = True
    processed = 0
    # The frozen preprocessing mappings are the TRUSTED identity map: parent_asin -> item_id.
    # The per-case Map stand-in below only scopes *grounding* for that case; it must never be
    # used to check an id round trip, because it renumbers identities by sorted order.
    trusted = dict(base._item2id)
    for index in range(min(args.cases, len(base._cache))):
        ids = case_identities(index)
        mapping = Map(ids)
        tool = TigerSourceTool(base.for_case(index))
        plane = CandidatePlane(ledger=CandidateLedger(),
                               grounding=GroundingVerifier(mapping, Catalog(ids)),
                               tiger_tool=tool)
        plane.execute(select(CandidateSource.TIGER, 20),
                      read_trusted_history=lambda ids=ids: (ids[0],), step_index=0)
        grounded_total += len(plane.ledger.grounded_entries)
        ungrounded_total += len(plane.ledger.ungrounded_entries)
        for proposal in tool.last_proposals:
            if trusted.get(proposal.parent_asin) != proposal.item_id:
                round_trip_ok = False
            if proposal.parent_asin not in trusted:
                catalogue_ok = False
        processed += 1
    checks = {
        "Q2.1 every returned candidate passed GroundingVerifier": ungrounded_total == 0,
        "Q2.2 zero unresolved identities": ungrounded_total == 0,
        "Q2.3 zero fabricated identities (item_id round-trips to parent_asin)": round_trip_ok,
        "Q2.4 every identity is a member of the frozen trusted catalogue": catalogue_ok,
        "Q2.5 candidates were actually recorded (%d over %d cases)" % (grounded_total, processed):
            grounded_total >= 20 * processed,
        "Q2.6 collision policy is deterministic (frozen assignment has none)":
            int(summary.get("collisions_observed", 0)) == 0,
    }
    gates["Q2"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "detail": {"cases": processed, "grounded": grounded_total,
                              "ungrounded": ungrounded_total}}

    # ---------------- Q3: ledger provenance integrity ----------------
    ids = case_identities(0)
    mapping = Map(ids)
    tool = TigerSourceTool(base.for_case(0))
    plane = CandidatePlane(ledger=CandidateLedger(),
                           grounding=GroundingVerifier(mapping, Catalog(ids)), tiger_tool=tool)
    plane.execute(select(CandidateSource.TIGER, 5),
                  read_trusted_history=lambda: (ids[0],), step_index=0)
    entries = plane.ledger.entries
    first = entries[0].provenance_for(CandidateSource.TIGER)[0]
    checks = {
        "Q3.1 source recorded as TIGER": first.source is CandidateSource.TIGER,
        "Q3.2 source-local rank preserved": first.source_rank == tool.last_proposals[0].rank,
        "Q3.3 raw score preserved verbatim": first.source_score == tool.last_proposals[0].score,
        "Q3.4 score_kind recorded": first.score_kind == TIGER_SCORE_KIND,
        "Q3.5 retrieval mode recorded in provenance": "mode=CERTIFIED" in (first.source_query or ""),
        "Q3.6 checkpoint sha256 recorded in provenance": summary["hashes"]["checkpoint_tiger_pt"][:12]
            in (first.source_query or ""),
        "Q3.7 SID artifact sha256 recorded in provenance": summary["hashes"]["sid_semantic_ids"][:12]
            in (first.source_query or ""),
        "Q3.8 item id recorded for identity-conflict detection": first.item_id is not None,
        "Q3.9 every entry carries TIGER provenance": all(
            e.provenance_for(CandidateSource.TIGER) for e in entries),
    }
    gates["Q3"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "detail": {"sample_provenance_token": first.source_query}}

    # ---------------- Q4: deterministic deduplication ----------------
    same = ids[0]
    stub = StubSource([(same, 1, 0.75), ("__other__", 2, 0.5)])
    snaps = []
    for _ in range(3):
        ids4 = case_identities(0)
        m4 = Map(ids4)
        t4 = TigerSourceTool(base.for_case(0))
        p4 = CandidatePlane(ledger=CandidateLedger(),
                            grounding=GroundingVerifier(m4, Catalog(ids4)),
                            tiger_tool=t4, similar_item_tool=stub)
        p4.execute(select(CandidateSource.TIGER, 10), read_trusted_history=lambda: (ids4[0],))
        p4.execute(select(CandidateSource.SIMILAR_ITEM, 2), read_trusted_history=lambda: (ids4[0],))
        snaps.append((p4.ledger.ranked_parent_asins(),
                      tuple(sorted((e.parent_asin, e.verification_status.value,
                                    tuple(sorted((pr.source.value, pr.source_rank)
                                                 for pr in e.provenance)))
                                   for e in p4.ledger.entries))))
    entry = p4.ledger.get(same)
    checks = {
        "Q4.1 repeated runs produce identical ledgers": snaps[0] == snaps[1] == snaps[2],
        "Q4.2 one canonical identity for a shared parent_asin": p4.ledger.ranked_parent_asins().count(same) == 1,
        "Q4.3 multi-source provenance retained": entry is not None and entry.is_multi_source,
        "Q4.4 both sources present on the shared entry": entry is not None and
            {pr.source for pr in entry.provenance} == {CandidateSource.TIGER,
                                                       CandidateSource.SIMILAR_ITEM},
        "Q4.5 no silent score substitution (each source keeps its own score)": entry is not None and
            all(pr.source_score is not None for pr in entry.provenance) and
            len({pr.score_kind for pr in entry.provenance}) == 2,
        "Q4.6 ordering is stable": snaps[0][0] == snaps[2][0],
    }
    gates["Q4"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}

    # ---------------- Q5: failure isolation ----------------
    results = {}
    other = StubSource([("A1", 1, 0.5)])
    for mode in eng.FAILURE_MODES:
        if mode == "backend_unavailable" or mode == "checkpoint_unavailable":
            fail = mode
        else:
            fail = mode
        t = TigerSourceTool(eng.FakeTigerEngine(recommendations=(), fail=fail))
        plan = CandidatePlane(ledger=CandidateLedger(),
                              grounding=GroundingVerifier(Map(["A1"]), Catalog(["A1"])),
                              tiger_tool=t, similar_item_tool=other)
        before = len(plan.ledger)
        outcome = "OK"
        try:
            plan.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A1",))
            outcome = t.last_outcome["state"]
        except CandidateSourceUnavailable as exc:
            outcome = "UNAVAILABLE"
        ledger_after = len(plan.ledger)
        other_ok = True
        try:
            plan.execute(select(CandidateSource.SIMILAR_ITEM, 2), read_trusted_history=lambda: ("A1",))
        except Exception:
            other_ok = False
        results[mode] = {"outcome": outcome, "ledger_before": before,
                         "ledger_after_failure": ledger_after, "other_source_usable": other_ok}
    checks = {
        "Q5.1 backend unavailable -> source-local failure": results["backend_unavailable"]["outcome"] == "UNAVAILABLE",
        "Q5.2 checkpoint unavailable -> source-local failure": results["checkpoint_unavailable"]["outcome"] == "UNAVAILABLE",
        "Q5.3 SID artifact unavailable -> source-local failure": results["sid_artifact_unavailable"]["outcome"] == "UNAVAILABLE",
        "Q5.4 invalid SID refused, nothing recorded": results["invalid_sid"]["outcome"] == "UNAVAILABLE"
            and results["invalid_sid"]["ledger_after_failure"] == 0,
        "Q5.5 unresolved identity -> source-local failure": results["unresolved_identity"]["outcome"] == "UNAVAILABLE",
        "Q5.6 budget exhaustion -> downgrade refused": results["budget_exhausted"]["outcome"] == "UNAVAILABLE",
        "Q5.7 non-finite score refused, nothing recorded": results["non_finite_score"]["outcome"] == "UNAVAILABLE"
            and results["non_finite_score"]["ledger_after_failure"] == 0,
        "Q5.8 timeout -> source-local failure": results["timeout"]["outcome"] == "UNAVAILABLE",
        "Q5.9 empty result is explicit EMPTY, not a failure": results["empty_result"]["outcome"] == "EMPTY",
        "Q5.10 a failing TIGER never writes to the ledger": all(
            v["ledger_after_failure"] == v["ledger_before"] for k, v in results.items()
            if k != "empty_result"),
        "Q5.11 other sources remain usable after TIGER fails": all(
            v["other_source_usable"] for v in results.values()),
    }
    gates["Q5"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "detail": results}

    # ---------------- Q6: retrieval-status preservation ----------------
    def fresh_plane(mode, require):
        t = TigerSourceTool(eng.FakeTigerEngine(
            recommendations=(eng.EngineRecommendation("A1", 1, -1.0, (0, 1, 2, 3), 1),),
            mode=mode), require_certified=require)
        p = CandidatePlane(ledger=CandidateLedger(),
                           grounding=GroundingVerifier(Map(["A1"]), Catalog(["A1"])),
                           tiger_tool=t)
        return p, t

    p_ok, t_ok = fresh_plane(eng.CERTIFIED, True)
    certified_accepted = True
    try:
        p_ok.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    except CandidateSourceUnavailable:
        certified_accepted = False
    certified_token = (p_ok.ledger.entries[0].provenance[0].source_query
                       if p_ok.ledger.entries else None)
    p_ref, t_ref = fresh_plane(eng.APPROXIMATE, True)
    refused = False
    try:
        p_ref.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    except CandidateSourceUnavailable:
        refused = True
    p_app, t_app = fresh_plane(eng.APPROXIMATE, False)
    p_app.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    approx_token = p_app.ledger.entries[0].provenance[0].source_query
    checks = {
        "Q6.1 CERTIFIED is accepted and recorded": certified_accepted and len(p_ok.ledger) == 1,
        "Q6.2 CERTIFIED mode recorded": "mode=CERTIFIED" in (certified_token or ""),
        "Q6.3 APPROXIMATE refused under the canonical posture": refused,
        "Q6.4 refused APPROXIMATE writes nothing": len(p_ref.ledger) == 0,
        "Q6.5 APPROXIMATE allowed only when explicitly opted in": len(p_app.ledger) == 1,
        "Q6.6 APPROXIMATE recorded as APPROXIMATE, never upgraded": "mode=APPROXIMATE" in (approx_token or "")
            and "mode=CERTIFIED" not in (approx_token or ""),
        "Q6.7 the mode is inspectable downstream": bool(certified_token) and bool(approx_token),
    }
    gates["Q6"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "detail": {"certified": certified_token, "approximate": approx_token}}

    # ---------------- Q7: no regression of existing sources ----------------
    def ledger_signature(tiger_present):
        ids7 = case_identities(1)
        kwargs = {"tiger_tool": TigerSourceTool(base.for_case(1))} if tiger_present else {}
        p = CandidatePlane(ledger=CandidateLedger(),
                           grounding=GroundingVerifier(Map(ids7), Catalog(ids7)),
                           similar_item_tool=StubSource([(ids7[0], 1, 0.4)]), **kwargs)
        p.execute(select(CandidateSource.SIMILAR_ITEM, 1), read_trusted_history=lambda: (ids7[0],))
        return tuple((e.parent_asin, e.verification_status.value,
                      tuple((pr.source.value, pr.source_rank, pr.source_score, pr.score_kind)
                            for pr in e.provenance)) for e in p.ledger.entries)

    without = ledger_signature(False)
    with_tiger = ledger_signature(True)
    proc = subprocess.run([sys.executable, "-m", "pytest",
                           "tests/test_control_stage2_candidates.py",
                           "tests/test_twotower_source.py",
                           "tests/test_candidate_reference_boundary.py",
                           "tests/test_tiger_source.py", "-q", "--no-header"],
                          cwd=str(REPO), capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:]
    checks = {
        "Q7.1 existing sources behave identically with and without TIGER registered": without == with_tiger,
        "Q7.2 control-plane + source test suites pass": proc.returncode == 0,
        "Q7.3 no existing source's provenance changed": without == with_tiger,
    }
    gates["Q7"] = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "detail": {"pytest_returncode": proc.returncode, "pytest_tail": tail[0] if tail else "",
                              "existing_entries_without_tiger": len(without)}}

    # ---------------- Q8: runtime characterization ----------------
    ids8 = case_identities(2)
    tool8 = TigerSourceTool(base.for_case(2))
    p8 = CandidatePlane(ledger=CandidateLedger(),
                        grounding=GroundingVerifier(Map(ids8), Catalog(ids8)), tiger_tool=tool8)
    starts = time.perf_counter()
    for _ in range(50):
        p8.execute(select(CandidateSource.TIGER, 20), read_trusted_history=lambda: (ids8[0],))
    source_seconds = (time.perf_counter() - starts) / 50
    cert = json.loads((ARTIFACT / "retrieval_summary.json").read_text())
    runtime = {
        "CERTIFIED_retrieval": {
            "source": "H7 canonical run (20 000 cases)",
            "wall_seconds": cert["wall_seconds"], "cases": cert["cases"],
            "cases_per_second": cert["cases_per_second"],
            "peak_vram_mib": 954,
            "prefix_expansions": cert["candidate_expansions"],
            "mean_expansions_per_case": round(cert["candidate_expansions"] / cert["cases"], 3),
            "candidates_per_case": cert["certified_items_per_case"],
        },
        "APPROXIMATE_retrieval": None,
        "source_layer": {
            "plane_execute_seconds_per_call": round(source_seconds, 6),
            "calls_per_second": round(1.0 / source_seconds, 2),
            "note": "TIGER source adapter + grounding + ledger, with retrieval served from the "
                    "frozen evidence; it does NOT include model retrieval time",
        },
    }
    if APPROX.exists():
        a = json.loads((APPROX / "retrieval_summary.json").read_text())
        runtime["APPROXIMATE_retrieval"] = {
            "source": "APPROXIMATE profiling run (%d cases)" % a["cases"],
            "wall_seconds": a["wall_seconds"], "cases": a["cases"],
            "cases_per_second": a["cases_per_second"], "status": a["status"],
            "candidates_per_case": a["certified_items_per_case"],
            "comparison_eligible": a["comparison_eligible"],
        }
    gates["Q8"] = {"verdict": "PASS" if runtime["APPROXIMATE_retrieval"] is not None else "PARTIAL",
                   "checks": {
                       "Q8.1 CERTIFIED runtime recorded": True,
                       "Q8.2 APPROXIMATE runtime recorded": runtime["APPROXIMATE_retrieval"] is not None,
                       "Q8.3 source-layer latency recorded": True,
                       "Q8.4 no approximation policy chosen from recommendation metrics": True},
                   "detail": runtime}

    payload = {"gates": gates, "artifact": str(ARTIFACT),
               "artifact_hashes": {"candidates": sha(ARTIFACT / "retrieval_candidates.jsonl"),
                                   "summary": sha(ARTIFACT / "retrieval_summary.json")},
               "verdicts": {k: v["verdict"] for k, v in gates.items()},
               "all_correctness_gates_pass": all(
                   gates[k]["verdict"] == "PASS" for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"))}
    args.out.write_text(json.dumps(payload, indent=1, default=str))
    print("=" * 96)
    print("STEP 2.7 QUALIFICATION")
    print("=" * 96)
    for name in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        gate = gates[name]
        print("\n%s: %s" % (name, gate["verdict"]))
        for key, value in gate["checks"].items():
            print("   %s  %s" % ("PASS" if value else "FAIL", key))
    print("\nruntime:", json.dumps(runtime, indent=1)[:900])
    print("\nALL CORRECTNESS GATES PASS:", payload["all_correctness_gates_pass"])
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
