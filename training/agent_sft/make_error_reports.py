"""Build confusion/per-operation diagnostics for the frozen test and the independent hard test."""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path
from .schema import parse_and_validate

ROOT = Path(__file__).resolve().parents[2]
OPS = ["set_goal","update_goal","add_preference","update_preference","retract_preference","set_constraint","update_constraint","retract_constraint","resolve_reference","clarify","noop","abstain"]
SETS = {"test": ROOT/"training/agent_sft/data/test.jsonl", "hard": ROOT/"training/agent_sft/hard_test/hard_test.jsonl"}
PRED = {
    "test": {"rule": ROOT/"training/agent_sft/eval/rule_local/predictions.jsonl", "base": ROOT/"training/agent_sft/eval/base/predictions.jsonl", "sft": ROOT/"training/agent_sft/eval/sft/predictions.jsonl"},
    "hard": {"rule": ROOT/"training/agent_sft/eval/hard_rule/predictions.jsonl", "base": ROOT/"training/agent_sft/eval/hard_base/predictions.jsonl", "sft": ROOT/"training/agent_sft/eval/hard_sft/predictions.jsonl"},
}

def load(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def ops(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list): return ()
    return tuple((str(o.get("op","")),str(o.get("attribute","")),str(o.get("value","")),str(o.get("direction","")),str(o.get("scope",""))) for o in payload["operations"] if isinstance(o,dict))

def report(dataset: str):
    test={r["sample_id"]:r for r in load(SETS[dataset])}
    rows=[]; summary={}
    for provider,path in PRED[dataset].items():
        preds={r["sample_id"]:r for r in load(path)}
        counts={k:{"gold":0,"pred":0,"tp":0} for k in OPS}; conf=Counter(); errors=Counter(); valid=exact=0; cases=[]
        for sid,g in test.items():
            p=preds.get(sid,{}); raw=str(p.get("prediction_raw","")); pred,errs=parse_and_validate(raw,g["user_message"]); gold=json.loads(g["assistant"]); go,po=ops(gold),ops(pred)
            if not errs: valid+=1
            gt={o[0] for o in go}; pt={o[0] for o in po} if not errs else set()
            for k in OPS:
                counts[k]["gold"]+=int(k in gt); counts[k]["pred"]+=int(k in pt); counts[k]["tp"]+=int(k in gt and k in pt)
            if not errs and po==go: exact+=1
            if not errs and po!=go:
                if not po: errors["missing_operations"]+=1
                elif not go: errors["extra_operations"]+=1
                elif go[0][0]!=po[0][0]: errors["wrong_operation_type"]+=1
                elif go[0][1]!=po[0][1]: errors["wrong_attribute"]+=1
                elif str(go[0][2]).casefold()!=str(po[0][2]).casefold(): errors["wrong_value"]+=1
                elif go[0][3]!=po[0][3]: errors["wrong_direction"]+=1
                elif go[0][4]!=po[0][4]: errors["wrong_scope"]+=1
                else: errors["evidence_or_multi_operation"]+=1
                if len(cases)<10: cases.append({"sample_id":sid,"scenario":g["scenario"],"user_message":g["user_message"],"gold":g["assistant"],"prediction_raw":raw})
            conf[(("none" if not go else go[0][0]),("none" if not po else po[0][0]))]+=1
        per={}
        for k,c in counts.items():
            pr=c["tp"]/c["pred"] if c["pred"] else 0.0; rec=c["tp"]/c["gold"] if c["gold"] else 0.0
            f=2*pr*rec/(pr+rec) if pr+rec else 0.0
            per[k]={**c,"precision":pr,"recall":rec,"f1":f}
        supported=[k for k in OPS if per[k]["gold"]>0]
        summary[provider]={"valid_json":valid,"exact":exact/len(test),"macro_all":sum(per[k]["f1"] for k in OPS)/len(OPS),"macro_supported":sum(per[k]["f1"] for k in supported)/len(supported),"supported":supported,"errors":dict(errors)}
        for k in OPS: rows.append({"dataset":dataset,"provider":provider,"operation":k,**per[k]})
        for (g,p),n in sorted(conf.items()): rows.append({"dataset":dataset,"provider":provider,"operation":"__confusion__","gold_operation":g,"predicted_operation":p,"count":n})
        (ROOT/f"training/agent_sft/reports/P2_SFT_{dataset.upper()}_ERROR_CASES_{provider}.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8")
    out=ROOT/f"training/agent_sft/reports/P2_SFT_{dataset.upper()}_OPERATION_METRICS.csv"
    with out.open("w",encoding="utf-8",newline="") as h:
        fields=["dataset","provider","operation","gold","pred","tp","precision","recall","f1","gold_operation","predicted_operation","count"]; w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (ROOT/f"training/agent_sft/reports/P2_SFT_{dataset.upper()}_ERROR_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(dataset,json.dumps(summary,ensure_ascii=False,indent=2))
for d in ("test","hard"): report(d)
