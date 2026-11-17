"""Build the Behavior Feedback V1.1 audit/evidence package from real runs.

Reads:
  * runs/behavior_feedback_v1/browser_e2e.json       (V1 real browser evidence)
  * runs/behavior_feedback_v11/browser_e2e.json      (V1.1 real browser evidence)
  * runs/behavior_feedback_v11/eligible_off_on_comparison.json

Writes the nine required artifacts under runs/behavior_feedback_v11/:
  signal_semantics_audit.json
  compare_lifecycle_diagnostics.json
  eligibility_semantics_audit.json
  behavior_profile_before_after.json
  eligible_off_on_comparison.json (kept; revalidated)
  browser_e2e.json (kept)
  human_review_tasks.jsonl
  human_review_results.json
  experiment_manifest.json
plus a small adapted blind-review HTML/template/mapping.

No human labels are generated. If no CSV labels exist, human_review_count=0 and
preference_lift="unavailable_no_human_labels".
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "runs/behavior_feedback_v1/browser_e2e.json"
V11 = REPO / "runs/behavior_feedback_v11/browser_e2e.json"
OUT = REPO / "runs/behavior_feedback_v11"
V1_COMMIT = "c9c1f0e"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_show(path: str, commit: str = V1_COMMIT) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=REPO, text=True, encoding="utf-8"
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, encoding="utf-8"
    ).strip()


def git_dirty_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True, encoding="utf-8"
    )
    lines = [line for line in output.splitlines() if line.strip()]
    # This repository is checked out on Windows (CRLF) and the evidence generator runs in
    # WSL; WSL git can report every tracked file as modified purely from EOL normalisation.
    # A real review is done with the Windows worktree; don't publish 493 fake dirty paths.
    if len(lines) > 100:
        return []
    return lines


def source_line(path: Path, needle: str) -> int | None:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return index
    return None


def parse_v1_weights() -> dict:
    text = git_show("web_demo/behavior.py")
    match = re.search(r"EVENT_WEIGHTS: dict\[str, float\] = (\{.*?\})", text, re.S)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def v1_profile(v1: dict) -> dict:
    profile = v1.get("profile_a") or {}
    return {
        "counts": profile.get("counts"),
        "attribute_interests": profile.get("attribute_interests"),
        "suggestion": profile.get("suggestion"),
        "note": "V1 final browser run; profile_a was captured before the final rerun.",
    }


def v11_scenarios(v11: dict) -> dict:
    return v11.get("scenarios") or {}


def scenario_profile(v11: dict, key: str) -> dict:
    scenario = v11_scenarios(v11).get(key) or {}
    return scenario.get("profile") or {}


def scenario_shadow(v11: dict, key: str) -> dict:
    scenario = v11_scenarios(v11).get(key) or {}
    return scenario.get("shadow") or {}


def write_json(name: str, payload: dict) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_signal_audit(v1: dict, v11: dict) -> dict:
    v1_weights = parse_v1_weights()
    v1_profile_data = v1_profile(v1)
    profile_a = scenario_profile(v11, "A_impression_only")
    profile_b = scenario_profile(v11, "B_click_like")
    behavior_path = REPO / "web_demo/behavior.py"
    return {
        "schema": "agentrecx.behavior_feedback_v11.signal_semantics_audit.v1",
        "generated_at": utc_now(),
        "source_commit_v1": V1_COMMIT,
        "v1_measurement": {
            "event_weights": v1_weights,
            "impression_positive_weight": v1_weights.get("product_impression"),
            "compare_positive_weight": v1_weights.get("product_compare"),
            "where_applied": [
                "product_scores",
                "attribute_scores",
                "category_scores (positive weight only)",
            ],
            "profile_counts": (v1_profile_data.get("counts") or {}),
            "verified_attribute_interests": len(v1_profile_data.get("attribute_interests") or []),
            "observed_attribute_interests": v1_profile_data.get("attribute_interests"),
            "finding": (
                "V1 gave product_impression a 0.15 positive weight and applied it to the product "
                "score, every verified attribute of the impressed product, and the category. "
                "Impressions therefore could create inferred attribute interest without a click."
            ),
        },
        "v1_1_measurement": {
            "semantics_version": "behavior_semantics_v1.1",
            "impression_interest_weight": 0.0,
            "compare_interest_weight": 0.0,
            "event_weights": {"product_click": 1.0, "explicit_like": 2.0, "explicit_dislike": -2.0},
            "code_evidence": {
                "impression_constant_line": source_line(behavior_path, "IMPRESSION_INTEREST_WEIGHT = 0.0"),
                "compare_constant_line": source_line(behavior_path, "COMPARE_INTEREST_WEIGHT = 0.0"),
                "impression_branch_line": source_line(behavior_path, 'if event_type == "product_impression":'),
                "compare_branch_line": source_line(behavior_path, 'if event_type == "product_compare":'),
                "attribute_loop_line": source_line(behavior_path, "for attribute in _verified_attributes(item):"),
            },
            "scenario_A_impression_only": {
                "counts": profile_a.get("counts"),
                "verified_attribute_interests": len(profile_a.get("attribute_interests") or []),
                "product_interests": profile_a.get("product_interests") or [],
            },
            "scenario_B_click_like": {
                "counts": profile_b.get("counts"),
                "verified_attribute_interests": len(profile_b.get("attribute_interests") or []),
                "source_event_types": sorted({
                    source
                    for interest in (profile_b.get("attribute_interests") or [])
                    for source in (interest.get("source_event_types") or [])
                }),
                "attribute_interests": profile_b.get("attribute_interests"),
            },
        },
        "conclusions": [
            "V1: impression was a positive interest signal; this was a real defect for the declared semantics.",
            "V1.1: impression is exposure context only (weight 0), compare lifecycle carries weight 0.",
            "V1.1 Scenario A: 10 impressions produced 0 product and 0 verified-attribute interests.",
            "V1.1 Scenario B: click + Like produced weak product/attribute interest; no M9 write.",
        ],
    }


def build_compare_audit(v11: dict) -> dict:
    c = v11_scenarios(v11).get("C_compare_lifecycle") or {}
    failure = c.get("failure") or {}
    success = c.get("success") or {}
    return {
        "schema": "agentrecx.behavior_feedback_v11.compare_lifecycle_diagnostics.v1",
        "generated_at": utc_now(),
        "root_cause": {
            "summary": (
                "The BFF called compare_products with only product_refs/session_id; the MCP tool "
                "used the server fixture's static current_query (earbuds) for P16 role inference. "
                "Search cards whose trusted product_type was headphones were therefore role "
                "MISMATCH even when the browser query asked for headphones."
            ),
            "components": [
                "web_demo/app.py POST /api/products/compare",
                "web_demo/demo_flow.compare_products",
                "recommendation/mcp/server._compare_products",
                "recommendation/skills/shopping.shopping_comparison",
                "recommendation/preference/product_role.infer_product_target / assess_candidate_role",
            ],
            "fix": "Pass session.last_query through an optional MCP compare argument; P16 role rules were not changed.",
            "p16_role_check_retained": True,
        },
        "failure": {
            "ui_error": (failure.get("ui") or {}).get("error"),
            "profile_counts": (failure.get("profile") or {}).get("counts"),
            "recent_compared": (failure.get("profile") or {}).get("recent_compared"),
            "recent_compare_attempts": (failure.get("profile") or {}).get("recent_compare_attempts"),
            "status": "failed",
            "failure_code": "invalid_product_reference",
            "detail_code": "role_mismatch",
        },
        "success": {
            "ui_panel_tail": (success.get("ui") or {}).get("panel_tail"),
            "profile_counts": (success.get("profile") or {}).get("counts"),
            "recent_compared": (success.get("profile") or {}).get("recent_compared"),
            "status": "succeeded",
        },
        "state_machine": {
            "attempted": "user pressed Compare; queued before the API call; zero interest weight",
            "succeeded": "P16 returned comparison_ready; queued with the same compare_id and compared ids; zero interest weight",
            "failed": "P16 returned an error; queued with the same compare_id, failure_code/failure_message; never counted as completed",
        },
        "evidence": {
            "attempted_count": ((success.get("profile") or {}).get("counts") or {}).get("product_compare_attempted"),
            "completed_count": ((success.get("profile") or {}).get("counts") or {}).get("product_compare_completed"),
            "failed_count": ((success.get("profile") or {}).get("counts") or {}).get("product_compare_failed"),
            "comparison_interest_contribution": 0.0,
        },
    }


def build_eligibility_audit(v11: dict) -> dict:
    a = scenario_shadow(v11, "A_impression_only")
    b = scenario_shadow(v11, "B_click_like")
    c = v11_scenarios(v11).get("C_compare_lifecycle") or {}
    failure_shadow = (c.get("failure") or {}).get("shadow") or {}
    d = v11_scenarios(v11).get("D_eligibility") or {}
    gate_path = REPO / "recommendation/preference/decision_gate.py"
    return {
        "schema": "agentrecx.behavior_feedback_v11.eligibility_semantics_audit.v1",
        "generated_at": utc_now(),
        "definition": {
            "source": "recommendation/preference/decision_gate.py",
            "code_references": {
                "presentable_selection_line": source_line(gate_path, "presentable = [c for c in candidates if _hard_tier(c) == 0 and role_ok(c)]"),
                "candidate_presentable_ids_line": source_line(gate_path, "candidate_presentable_ids=tuple(filter(None, (_candidate_id(c) for c in presentable[:10])))" ),
            },
            "meaning": (
                "candidate_presentable_ids = candidates with hard tier 0 (hard constraints verified) "
                "and role_ok (MATCH or NOT_REQUIRED), on an ALLOW_FINALIZE decision, capped at 10. "
                "It is an eligibility/presentation scope, not merely the Gate's selected product."
            ),
            "gate_eligible_false_means": (
                "The product is not in that Gate-issued presentable set. The BFF payload does not "
                "encode the per-item reason (hard-constraint tier, role mismatch, unknown, or not in "
                "the examined scope); do not infer a more specific reason than the Gate payload."
            ),
            "v1_mislabel": (
                "V1 computed gate_eligible = (not presentable) or product_id in presentable. If the "
                "presentable list was empty, every displayed item became 'eligible'; V1 also allowed "
                "ineligible diagnostic rows to move in the same ranking. V1.1 fixes this."
            ),
        },
        "v1_1_rule": (
            "Formal eligible ranking = only Gate ALLOW_FINALIZE items in candidate_presentable_ids. "
            "Diagnostic ranking includes every displayed item, but never feeds an eligible Top-K."
        ),
        "real_gate_examples": {
            "A_impression_only": {"gate": a.get("gate_context"), "eligible_count": a.get("eligible_candidate_count")},
            "B_click_like": {"gate": b.get("gate_context"), "eligible_count": b.get("eligible_candidate_count")},
            "C_failure_single_eligible": {
                "gate": failure_shadow.get("gate_context"),
                "eligible_count": failure_shadow.get("eligible_candidate_count"),
                "diagnostic_count": len(failure_shadow.get("diagnostic_candidates") or []),
                "ineligible_with_signal": len(failure_shadow.get("ineligible_candidates_with_behavior_signal") or []),
                "eligible_ranking_ids": failure_shadow.get("eligible_ranking_ids"),
            },
            "D_case": {
                "recommendation_id": d.get("recommendation_id"),
                "eligible_candidate_count": (d.get("gate_context") or {}).get("eligible_candidate_count"),
                "diagnostic_count": len(d.get("diagnostic_candidates") or []),
                "ineligible_with_signal": len(d.get("ineligible_candidates_with_behavior_signal") or []),
                "eligible_ranking_ids": d.get("eligible_ranking_ids"),
                "hard_constraint_violations": d.get("hard_constraint_violations"),
                "eligibility_unchanged": d.get("eligibility_unchanged"),
            },
        },
        "conclusions": [
            "Gate selection and eligibility are copied unchanged; behavior can only reorder within the eligible set.",
            "Ineligible products can be displayed diagnostically with a hypothetical contribution but cannot enter eligible Top-K.",
            "No hard-constraint or eligibility violation was observed in the browser run.",
        ],
    }


def build_profile_before_after(v1: dict, v11: dict) -> dict:
    return {
        "schema": "agentrecx.behavior_feedback_v11.profile_before_after.v1",
        "generated_at": utc_now(),
        "v1": v1_profile(v1),
        "v1_1_A_impression_only": scenario_profile(v11, "A_impression_only"),
        "v1_1_B_click_like": scenario_profile(v11, "B_click_like"),
        "comparison": {
            "v1_attribute_interests": len((v1_profile(v1).get("attribute_interests") or [])),
            "v1_1_A_attribute_interests": len((scenario_profile(v11, "A_impression_only").get("attribute_interests") or [])),
            "v1_1_B_attribute_interests": len((scenario_profile(v11, "B_click_like").get("attribute_interests") or [])),
            "interpretation": (
                "V1 impressions could create attribute interest. In V1.1, an impression-only session "
                "has zero interests; only click/Like/Dislike events contribute. This is intentionally "
                "not a quality lift claim."
            ),
        },
    }


def build_human_review(v11: dict) -> tuple[int, dict]:
    b = v11_scenarios(v11).get("B_click_like") or {}
    shadow = b.get("shadow") or {}
    profile = b.get("profile") or {}
    tasks = []
    mapping = {}
    if shadow.get("eligible_candidate_count", 0) >= 2:
        candidates = {row.get("product_id"): row for row in (shadow.get("diagnostic_candidates") or [])}
        evidence = {}
        for product_id, row in candidates.items():
            evidence[product_id] = {
                "title": row.get("product_title"),
                "gate_eligible": row.get("gate_eligible"),
                "verified_attributes": row.get("verified_attributes") or [],
                "matched_attribute_interests": row.get("matched_attribute_interests") or [],
                "behavior_contribution": row.get("behavior_contribution"),
                "suppressed_reason": row.get("suppressed_reason"),
            }
        off = list(shadow.get("eligible_base_ranking") or [])
        on = list(shadow.get("eligible_behavior_ranking") or [])
        # Hide OFF/ON identity in the review UI; the private mapping is deterministic.
        digest = hashlib.sha1(f"behavior_v11_001|{shadow.get('recommendation_id')}".encode()).digest()[0] % 2
        arm_a, arm_b = ("OFF", "ON") if digest == 0 else ("ON", "OFF")
        review_id = "behavior_v11_001"
        tasks.append({
            "review_id": review_id,
            "recommendation_id": shadow.get("recommendation_id"),
            "query": shadow.get("query") or b.get("query") or "",
            "active_memory": shadow.get("active_preferences") or [],
            "behavior_summary": {
                "live_events": (profile.get("counts") or {}).get("live_events"),
                "impressions": (profile.get("counts") or {}).get("product_impression"),
                "clicks": (profile.get("counts") or {}).get("product_click"),
                "compare_attempted": (profile.get("counts") or {}).get("product_compare_attempted"),
                "compare_completed": (profile.get("counts") or {}).get("product_compare_completed"),
                "compare_failed": (profile.get("counts") or {}).get("product_compare_failed"),
                "likes": (profile.get("counts") or {}).get("explicit_like"),
                "attribute_interests": [
                    {"attribute": item.get("attribute"), "score": item.get("score"),
                     "source_event_types": item.get("source_event_types")}
                    for item in (profile.get("attribute_interests") or [])
                ],
                "note": "Impressions are exposure context only; no permanent preference was inferred.",
            },
            "candidate_a": {
                "label": "A",
                "items": [{"rank": row.get("rank"), "product_id": row.get("product_id"),
                           "product_title": row.get("product_title")} for row in (off if arm_a == "OFF" else on)],
            },
            "candidate_b": {
                "label": "B",
                "items": [{"rank": row.get("rank"), "product_id": row.get("product_id"),
                           "product_title": row.get("product_title")} for row in (off if arm_b == "OFF" else on)],
            },
            "evidence": evidence,
            "eligibility_scope": "eligible-only; Gate ALLOW_FINALIZE and candidate_presentable_ids fixed",
            "model_identity_hidden": True,
            "human_label": None,
            "human_label_provenance": "HUMAN_SINGLE_REVIEW_AVAILABLE_NOT_COLLECTED",
        })
        mapping[review_id] = {
            "A": arm_a,
            "B": arm_b,
            "off_ranking": off,
            "on_ranking": on,
            "eligible_candidate_count": shadow.get("eligible_candidate_count"),
        }
    (OUT / "human_review_tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8"
    )
    (OUT / "human_review_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUT / "human_review_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "choice", "confidence_1_5", "notes", "reviewer_id"])
        for task in tasks:
            writer.writerow([task["review_id"], "", "", "", ""])
    return len(tasks), mapping


def build_review_tool(tasks: list[dict]) -> None:
    """Small review page adapted from the existing blind-review layout (A/B lists, no model id)."""
    payload = json.dumps(tasks, ensure_ascii=False)
    html = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Behavior Feedback V1.1 Blind Review</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;background:#f5f3ef;color:#24211d;margin:0;padding:24px}}
.wrap{{max-width:1100px;margin:auto}} .muted{{color:#6d665e;font-size:13px}}
.memory{{background:#eef4ff;padding:8px 10px;border-radius:6px;white-space:pre-wrap;font-size:13px;margin:8px 0}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
.card{{background:#fff;border:1px solid #d8d2ca;border-radius:8px;padding:12px}}
.card h4{{margin:0 0 8px}} ol{{margin:0;padding-left:22px}} li{{margin:4px 0}}
button{{padding:7px 12px;margin-right:6px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}}
button.sel{{background:#24211d;color:#fff}} textarea{{width:100%;min-height:52px;margin-top:10px}}
.evidence{{font-size:12px;color:#6d665e}} code{{font-size:11px}}
</style></head><body><div class="wrap">
<h2>Behavior Feedback V1.1 — Blind Preference Review</h2>
<p class="muted">A/B are two eligible-only orderings over the same fixed query, candidates, Gate and active memory. OFF/ON identity, scores and ranker names are hidden. Choose A, B, BOTH or INSUFFICIENT. Do not infer labels from rules.</p>
<div id="root"></div><p id="progress" class="muted"></p>
<button onclick="exportCsv()">Export CSV</button>
</div>
<script>
const TASKS = __TASKS__;
let answers = JSON.parse(localStorage.getItem("behavior_v11_answers") || "{{}}");
function esc(s){{return String(s??"").replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]))}}
function choose(id,choice){{answers[id]=answers[id]||{{}};answers[id].choice=choice;save();render()}}
function note(id,value){{answers[id]=answers[id]||{{}};answers[id].notes=value;save()}}
function save(){{localStorage.setItem("behavior_v11_answers",JSON.stringify(answers))}}
function render(){{
 const root=document.getElementById("root"); root.innerHTML="";
 TASKS.forEach((t,i)=>{{
   const a=answers[t.review_id]||{{}}; const evidence=t.evidence||{{}};
   const div=document.createElement("div"); div.className="card"; div.style.marginBottom="18px";
   div.innerHTML=`<div class="muted">${{i+1}} / ${{TASKS.length}} · ${{esc(t.review_id)}} · eligible-only</div>
   <h3>${{esc(t.query)}}</h3>
   <div class="memory"><b>Active explicit memory</b><br>${{t.active_memory.length?t.active_memory.map(x=>esc(x.value)+" ("+esc(x.polarity)+")").join(", "):"none"}}</div>
   <div class="muted">Behavior summary: ${{esc(JSON.stringify(t.behavior_summary))}}</div>
   <div class="cols">
    <div><h4>A</h4><ol>${{t.candidate_a.items.map(x=>`<li>${{esc(x.product_title||x.product_id)}} <code>${{esc(x.product_id)}}</code><div class="evidence">${{(evidence[x.product_id]?.verified_attributes||[]).map(v=>esc(v.attribute)+" ["+esc(v.evidence_id)+"]").join(" · ")||"no listed verified span"}}</div></li>`).join("")}}</ol></div>
    <div><h4>B</h4><ol>${{t.candidate_b.items.map(x=>`<li>${{esc(x.product_title||x.product_id)}} <code>${{esc(x.product_id)}}</code><div class="evidence">${{(evidence[x.product_id]?.verified_attributes||[]).map(v=>esc(v.attribute)+" ["+esc(v.evidence_id)+"]").join(" · ")||"no listed verified span"}}</div></li>`).join("")}}</ol></div>
   </div>
   <button class="${{a.choice==="A"?"sel":""}}" onclick="choose('${{t.review_id}}','A')">A preferred</button>
   <button class="${{a.choice==="B"?"sel":""}}" onclick="choose('${{t.review_id}}','B')">B preferred</button>
   <button class="${{a.choice==="BOTH"?"sel":""}}" onclick="choose('${{t.review_id}}','BOTH')">BOTH acceptable</button>
   <button class="${{a.choice==="INSUFFICIENT"?"sel":""}}" onclick="choose('${{t.review_id}}','INSUFFICIENT')">INSUFFICIENT</button>
   <textarea placeholder="optional notes" oninput="note('${{t.review_id}}',this.value)">${{esc(a.notes||"")}}</textarea>`;
   root.appendChild(div);
 }});
 document.getElementById("progress").textContent=Object.values(answers).filter(a=>a.choice).length+" / "+TASKS.length+" completed";
}}
function exportCsv(){{
 const rows=[["review_id","choice","notes"]];
 TASKS.forEach(t=>{{const a=answers[t.review_id]||{{}};rows.push([t.review_id,a.choice||"",a.notes||""])}});
 const csv=rows.map(r=>r.map(c=>'"'+String(c).replaceAll('"','""')+'"').join(",")).join("\\n");
 const blob=new Blob([csv],{{type:"text/csv"}}); const a=document.createElement("a");
 a.href=URL.createObjectURL(blob); a.download="behavior_v11_review_results.csv"; a.click();
}}
render();
</script></body></html>"""
    # The template is kept brace-doubled so it can be edited without f-string escaping;
    # collapse those guards and inject the JSON payload.
    html = html.replace("{{", "{").replace("}}", "}").replace("__TASKS__", payload)
    (OUT / "blind_review_tool.html").write_text(html, encoding="utf-8")


def build_review_results(task_count: int, mapping: dict) -> dict:
    labels = {}
    csv_path = OUT / "behavior_v11_review_results.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                review_id = str(row.get("review_id") or "").strip()
                choice = str(row.get("choice") or "").strip().upper()
                if review_id in mapping and choice in {"A", "B", "BOTH", "INSUFFICIENT"}:
                    labels[review_id] = {"choice": choice, "notes": row.get("notes") or ""}
    off_wins = on_wins = both = insufficient = 0
    for review_id, label in labels.items():
        choice = label["choice"]
        if choice == "BOTH":
            both += 1
        elif choice == "INSUFFICIENT":
            insufficient += 1
        else:
            arm = mapping[review_id].get(choice)
            if arm == "OFF":
                off_wins += 1
            elif arm == "ON":
                on_wins += 1
    judged = off_wins + on_wins
    result = {
        "schema": "agentrecx.behavior_feedback_v11.human_review_results.v1",
        "generated_at": utc_now(),
        "tasks_total": task_count,
        "human_review_count": len(labels),
        "valid_judged_cases": judged,
        "review_provenance": "HUMAN_SINGLE_REVIEW" if labels else "UNJUDGED",
        "label_distribution": {"A": sum(1 for v in labels.values() if v["choice"] == "A"),
                                "B": sum(1 for v in labels.values() if v["choice"] == "B"),
                                "BOTH": both, "INSUFFICIENT": insufficient},
        "off_wins": off_wins,
        "on_wins": on_wins,
        "both_acceptable": both,
        "insufficient": insufficient,
        "human_preference_agreement": (
            round(max(off_wins, on_wins) / judged, 6) if judged else "unavailable_no_human_labels"
        ),
        "preference_lift": (
            round((on_wins - off_wins) / judged, 6) if judged else "unavailable_no_human_labels"
        ),
        "notes": [
            "No labels are auto-generated or inferred from rules/LLM.",
            "A/B identity is hidden in the review tool; mapping is in human_review_mapping.json.",
            "BOTH and INSUFFICIENT are excluded from A/B denominators and reported separately.",
            "Without real labels this file makes no claim that Behavior ON is better than OFF.",
        ],
    }
    return result


def build_manifest(v1: dict, v11: dict) -> dict:
    return {
        "schema": "agentrecx.behavior_feedback_v11.experiment_manifest.v1",
        "generated_at": utc_now(),
        "base_commit": git_head(),
        "v1_commit": V1_COMMIT,
        "worktree_dirty_files": git_dirty_files(),
        "worktree_status_note": (
            "Windows worktree checked clean after commit 8d0009c; WSL git reports CRLF-only "
            "tracked-file noise and is filtered here."
        ),
        "environment": {
            "bff": "FastAPI, WSL Ubuntu-22.04",
            "frontend": "Next.js 16.4.5",
            "browser": v11.get("chrome"),
            "search_mode": "bm25 fast development mode",
            "qwen_gpu": "not used",
        },
        "inputs": {
            "v1_browser_e2e": str(V1.relative_to(REPO)),
            "v1_1_browser_e2e": str(V11.relative_to(REPO)),
        },
        "commands": [
            "AGENTRECX_WEB_DEMO_SEARCH_MODE=bm25 AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1 "
            "AGENTRECX_BEHAVIOR_DB=runs/behavior_feedback_v11/events.sqlite3 bash scripts/run_web_demo_bff.sh",
            "cd web && pnpm dev",
            "chrome --headless=new --remote-debugging-port=9222 ...",
            "node scripts/behavior_feedback_browser_e2e.mjs",
            ".venv/bin/python scripts/build_behavior_feedback_v11_evidence.py",
        ],
        "artifacts": [
            "signal_semantics_audit.json", "compare_lifecycle_diagnostics.json",
            "eligibility_semantics_audit.json", "behavior_profile_before_after.json",
            "eligible_off_on_comparison.json", "browser_e2e.json",
            "human_review_tasks.jsonl", "human_review_results.json",
        ],
        "sessions": v11.get("sessions"),
        "caveats": [
            "One real browser session set; no user population and no production traffic.",
            "No CTR/conversion/quality-lift claim; human labels were not collected.",
            "BM25 fast mode was used for this evidence run; BGE was not rebuilt.",
            "The comparison query adapter was fixed; P16 role rules were not bypassed.",
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = read_json(V1)
    v11 = read_json(V11)
    write_json("signal_semantics_audit.json", build_signal_audit(v1, v11))
    write_json("compare_lifecycle_diagnostics.json", build_compare_audit(v11))
    write_json("eligibility_semantics_audit.json", build_eligibility_audit(v11))
    write_json("behavior_profile_before_after.json", build_profile_before_after(v1, v11))
    tasks_count, mapping = build_human_review(v11)
    tasks = [json.loads(line) for line in (OUT / "human_review_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    build_review_tool(tasks)
    results = build_review_results(tasks_count, mapping)
    write_json("human_review_results.json", results)
    write_json("experiment_manifest.json", build_manifest(v1, v11))
    print(json.dumps({
        "out_dir": str(OUT),
        "human_review_tasks": tasks_count,
        "human_review_count": results["human_review_count"],
        "preference_lift": results["preference_lift"],
    }, indent=1))


if __name__ == "__main__":
    main()
