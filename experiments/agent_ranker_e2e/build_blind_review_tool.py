"""Generate a standalone local blind-review HTML tool for Phase 4.4."""

from __future__ import annotations

import json
from pathlib import Path

R = Path("runs/agent_ranker_phase43")
OUT = Path("runs/agent_ranker_phase44")
TASKS = R / "blind_review_tasks.jsonl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.dumps(tasks, ensure_ascii=False)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AgentRec-X Blind Preference Review</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#f7f8fa;color:#1b1d22}}
.card{{background:#fff;border:1px solid #d9dde5;border-radius:10px;padding:18px;margin:16px 0}}
.memory{{background:#eef4ff;padding:8px 10px;border-radius:6px;white-space:pre-wrap;font-size:13px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.prod{{border:1px solid #e2e5eb;border-radius:8px;padding:10px;font-size:13px}}
button{{margin:4px 6px 0 0;padding:7px 12px;border-radius:6px;border:1px solid #b8c0cc;background:#fff;cursor:pointer}}
button.sel{{background:#1f6feb;color:#fff;border-color:#1f6feb}}
textarea{{width:100%;min-height:50px;margin-top:8px}}
.bar{{position:sticky;top:0;background:#f7f8fa;padding:10px 0;z-index:2}}
.muted{{color:#667085;font-size:12px}} .tag{{display:inline-block;background:#eef0f4;border-radius:4px;padding:2px 6px;font-size:11px;margin:2px}}
</style></head><body>
<h1>AgentRec-X Blind Preference Review</h1>
<p class="muted">40 disagreement-set pairs. Model identity, scores, BGE rank and rule labels are hidden. Choose A / B / BOTH / INSUFFICIENT.</p>
<div class="bar"><button onclick="exportCSV()">Export CSV</button> <button onclick="resetAll()">Reset progress</button> <span id="progress" class="muted"></span></div>
<div id="root"></div>
<script>
const TASKS = {payload};
const KEY = "agentrecx_phase44_blind_review_v1";
const answers = JSON.parse(localStorage.getItem(KEY) || "{{}}");
function esc(s){{return String(s ?? "").replace(/[&<>]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));}}
function save(){{localStorage.setItem(KEY, JSON.stringify(answers)); renderProgress();}}
function choose(id, choice){{answers[id] = answers[id] || {{}}; answers[id].choice=choice; save(); render();}}
function note(id, value){{answers[id] = answers[id] || {{}}; answers[id].notes=value; save();}}
function renderProgress(){{const n=TASKS.filter(t=>answers[t.review_id] && answers[t.review_id].choice).length; document.getElementById("progress").textContent=n+" / "+TASKS.length+" completed";}}
function product(p){{return `<div class="prod"><b>${{esc(p.title||p.item_id)}}</b><div class="muted">${{esc(p.item_id)}}</div><div>attributes: ${{esc((p.attributes||[]).join(", ") || "none")}}</div><div class="muted">unknown: ${{esc((p.unknown_attributes||[]).join(", "))}}</div><div class="muted">evidence: ${{esc((p.evidence||[]).join("; "))}}</div></div>`;}}
function render(){{
 const root=document.getElementById("root"); root.innerHTML="";
 TASKS.forEach((t,i)=>{{
  const a=answers[t.review_id]||{{}};
  const mem=t.active_memory||{{}};
  const pos=(mem.positive||[]).map(x=>x.value); const neg=(mem.negative||[]).map(x=>x.value);
  const div=document.createElement("div"); div.className="card";
  div.innerHTML=`<div class="muted">${{i+1}} / ${{TASKS.length}} · ${{esc(t.scenario_type)}} · ${{esc(t.review_id)}}</div>
   <h3>${{esc(t.query)}}</h3>
   <div class="memory"><b>Active Memory</b><br>Positive: ${{esc(pos.join(", ")||"none")}}<br>Negative: ${{esc(neg.join(", ")||"none")}}</div>
   <div class="cols">${{product(t.candidate_a)}}${{product(t.candidate_b)}}</div>
   <div><button class="${{a.choice==='A'?'sel':''}}" onclick="choose('${{t.review_id}}','A')">A</button>
   <button class="${{a.choice==='B'?'sel':''}}" onclick="choose('${{t.review_id}}','B')">B</button>
   <button class="${{a.choice==='BOTH'?'sel':''}}" onclick="choose('${{t.review_id}}','BOTH')">BOTH</button>
   <button class="${{a.choice==='INSUFFICIENT'?'sel':''}}" onclick="choose('${{t.review_id}}','INSUFFICIENT')">INSUFFICIENT</button></div>
   <textarea placeholder="optional notes" oninput="note('${{t.review_id}}',this.value)">${{esc(a.notes||"")}}</textarea>`;
  root.appendChild(div);
 }});
 renderProgress();
}}
function exportCSV(){{
 const rows=[["review_id","choice","notes"]];
 TASKS.forEach(t=>{{const a=answers[t.review_id]||{{}}; rows.push([t.review_id,a.choice||"",a.notes||""]);}});
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(",")).join("\\n");
 const blob=new Blob([csv],{{type:"text/csv"}}); const url=URL.createObjectURL(blob);
 const a=document.createElement("a"); a.href=url; a.download="agentrecx_phase44_blind_review_results.csv"; a.click(); URL.revokeObjectURL(url);
}}
function resetAll(){{if(confirm("Reset all local choices?")){{for(const k of Object.keys(answers))delete answers[k]; save(); render();}}}}
render();
</script></body></html>"""
    (OUT / "blind_review_tool.html").write_text(html, encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "tool": str(OUT / "blind_review_tool.html")}, indent=1))


if __name__ == "__main__":
    main()
