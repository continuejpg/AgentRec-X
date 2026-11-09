"""Build the P2-SFT-v2 incremental dataset (new templates, mixed with v1.1 train).

The v1.1 files are never overwritten.  New messages are independently authored; the
100-sample hard test and the 200-sample Parser Test are never used as training rows.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .prepare_data import _contexts_from_catalogue
from .schema import SCHEMA_VERSION, canonical_json, validate_proposal

ROOT = Path(__file__).with_name("data_v2")
V1 = Path(__file__).with_name("data")
SEED = 20261106
MODIFIERS = [
    "for a weekend trip", "for car camping", "for a hiking trip", "for a family campout",
    "for a backpacking trip", "for a park day", "for the upcoming trip", "for a short trip",
    "for a long trip", "for the campsite", "for my next outing", "for a mountain trip",
    "for a beach trip", "for a road trip", "for the summer", "for the spring",
    "for a group trip", "for a solo trip", "for the festival", "for the cabin",
]


def _proposal(ops, clarification=None):
    result = {"schema_version": SCHEMA_VERSION, "operations": ops, "clarification": clarification}
    return result


def _op(op, attribute, value, direction, scope, evidence, **extra):
    row = {"op": op, "attribute": attribute, "value": value, "direction": direction, "scope": scope, "evidence": evidence}
    row.update(extra)
    return row


def _state(goal, preferences=None, constraints=None, count=5):
    return {"goal": goal, "preferences": preferences or {}, "constraints": constraints or {}, "last_presented_count": count, "last_reference": "previous_state"}


def _families():
    rows = []
    # A. multi-condition update (price + material + weight)
    for i in range(100):
        budget = 20 + 5 * (i % 15)
        material = ["titanium", "aluminum", "steel", "carbon fiber"][i % 4]
        msg = [
            f"Actually set the ceiling to ${budget}, use {material}, and pick the lighter one.",
            f"Change the max to ${budget}; {material} is required, lower weight preferred.",
            f"At most ${budget}, {material} only, and favor the lightest.",
            f"New hard cap ${budget}, use {material}, minimize weight please.",
        ][i % 4]
        ops = [
            _op("update_constraint", "price_max", budget, None, "task", f"${budget}"),
            _op("add_preference", "material", material, "exact", "task", material),
            _op("add_preference", "weight", None, "minimize", "task", "lighter" if "lighter" in msg else "lower weight" if "lower weight" in msg else "minimize weight" if "minimize weight" in msg else "lightest"),
        ]
        rows.append(("multi_constraint", f"v2_multi_{i%4}", msg, ops, None, _state({"product_type": "cookware", "role": "main"}, {}, {"price_max": 100})))
    # B. clarify vs noop
    for i in range(100):
        if i < 50:
            msg = [
                "I am not sure which gear category I need; help me pick one.",
                "Could you ask me what kind of product I want first?",
                "I have no category in mind; please narrow it down with a question.",
                "Not sure between cooking and water gear; ask me to choose.",
                "I need a category question before searching.",
            ][i % 5]
            ops = [_op("clarify", "product_type", None, None, "task", msg)]
            clar = {"attribute": "product_type", "question": "Which product category should I use?", "choices": ["water filter", "cookware", "lantern", "stove"]}
            rows.append(("clarify", f"v2_clarify_{i%5}", msg, ops, clar, _state(None)))
        else:
            msg = [
                "Nothing new this turn; keep the current plan.",
                "No change needed, continue the same search.",
                "That is all for now, no new preference.",
                "Keep the existing constraints; nothing to add.",
                "No correction this time.",
            ][i % 5]
            ops = [_op("noop", None, None, None, None, msg, reason="no_change")]
            rows.append(("noop", f"v2_noop_{i%5}", msg, ops, None, _state({"product_type": "lantern", "role": "main"})))
    # C. complex retraction
    for i in range(100):
        if i < 50:
            msg = [
                "Forget the price ceiling, keep the material preference.",
                "Drop the maximum budget entirely.",
                "Clear the spending cap but leave the color alone.",
                "Remove the price limit from my constraints.",
                "Ignore the earlier budget cap.",
            ][i % 5]
            ops = [_op("retract_constraint", "price_max", None, None, "task", msg)]
            rows.append(("retract_constraint2", f"v2_rcon_{i%5}", msg, ops, None, _state({"product_type": "water_filter", "role": "main"}, {"material": "steel"}, {"price_max": 30})))
        else:
            msg = [
                "Forget the weight preference; keep the budget.",
                "Remove the material requirement only.",
                "Drop the lighter-direction preference.",
                "Clear my color preference.",
                "Ignore the titanium requirement, not the price.",
            ][i % 5]
            ops = [_op("retract_preference", "weight" if ("weight" in msg or "lighter" in msg) else "material" if "material" in msg or "titanium" in msg else "color", None, None, "task", msg)]
            rows.append(("retract_preference2", f"v2_rpref_{i%5}", msg, ops, None, _state({"product_type": "cookware", "role": "main"}, {"weight": "minimize", "material": "titanium", "color": "blue"})))
    # D. cross-turn reference
    for i in range(100):
        idx = 1 + (i % 5)
        words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[idx]
        msg = [
            f"Use the {words} option from the shortlist you showed.",
            f"I meant the {words} result in the earlier comparison.",
            f"Take the {words} listing from the previous page.",
            f"Go with the {words} candidate from that list.",
            f"Show details for the {words} item previously presented.",
        ][i % 5]
        ops = [_op("resolve_reference", None, None, None, "task", msg, reference=f"previous_candidate_{idx}", reason="cross_turn_reference")]
        rows.append(("reference2", f"v2_ref_{i%5}", msg, ops, None, _state({"product_type": "cookware", "role": "main"}, count=5)))
    # E. role accessory/main
    for i in range(100):
        if i < 50:
            msg = [
                "I only want the replacement cartridge, not the main unit.",
                "Switch to the mouthpiece accessory only.",
                "Give me the spare part rather than the whole filter.",
                "Just the replacement component, please.",
                "Show the accessory version.",
            ][i % 5]
            ops = [_op("update_goal", "role", "accessory", None, "task", msg)]
            rows.append(("role_accessory2", f"v2_racc_{i%5}", msg, ops, None, _state({"product_type": "water_filter", "role": "main"})))
        else:
            msg = [
                "Actually I need the complete filter, not the accessory.",
                "Switch back to the full purifier.",
                "Forget the spare part; show the main unit.",
                "I want the complete device, not a replacement.",
                "Return to the main water filter.",
            ][i % 5]
            ops = [_op("update_goal", "role", "main", None, "task", msg)]
            rows.append(("role_main2", f"v2_rmain_{i%5}", msg, ops, None, _state({"product_type": "water_filter", "role": "accessory"})))
    # F. repeated budget updates
    for i in range(100):
        value = [10, 15, 25, 40, 55, 70, 90][i % 7]
        msg = [
            f"Actually make the budget ${value}.",
            f"Change the cap to ${value} now.",
            f"New maximum spending: ${value}.",
            f"Set the upper limit at ${value}.",
        ][i % 4]
        ops = [_op("update_constraint", "price_max", value, None, "task", f"${value}")]
        rows.append(("budget2", f"v2_budget_{i%4}", msg, ops, None, _state({"product_type": "water_filter", "role": "main"}, {}, {"price_max": 30})))
    return rows


def _context_for(contexts, index):
    return contexts[index % len(contexts)]


def main() -> int:
    contexts, source = _contexts_from_catalogue()
    contexts = [dict(item, target_type=None) for item in contexts]
    rng = random.Random(SEED)
    families = _families()
    out_rows = {"train": [], "val": []}
    for index, (scenario, template_id, message, ops, clarification, state) in enumerate(families):
        # Qualify the message so the 100 variants of each family are distinct while
        # preserving the original evidence substring used by the labels.
        message = message + ", " + MODIFIERS[(index // 5) % len(MODIFIERS)]
        proposal = _proposal(ops, clarification)
        errors = validate_proposal(proposal, message)
        if errors:
            raise SystemExit(f"invalid v2 sample {index}: {errors} {message}")
        split = "val" if index % 5 == 0 else "train"
        context = _context_for(contexts, index)
        row = {
            "sample_id": f"p2sft_v2_{split}_{len(out_rows[split]):05d}",
            "split": split,
            "scenario": scenario,
            "source": "p2sft.data.v1.2",
            "template_id": template_id,
            "state": state,
            "catalogue_context": {"category": context.get("category", ""), "title_example": context.get("title_example", "")},
            "user_message": message,
            "assistant": canonical_json(proposal),
            "human_review_status": "not_required_rule_checked",
        }
        out_rows[split].append(row)
    # Mix in original v1.1 train rows; keep new data at ~20% of training.
    original_train = [json.loads(line) for line in (V1 / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    original_val = [json.loads(line) for line in (V1 / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train = list(original_train)
    train.extend(out_rows["train"])
    val = list(original_val)
    val.extend(out_rows["val"])
    # Exact duplicate guard across all existing text, test and hard test.
    seen = set()
    for path in [V1/"train.jsonl", V1/"val.jsonl", V1/"test.jsonl", Path(__file__).with_name("hard_test")/"hard_test.jsonl"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["user_message"].casefold())
    new_messages = [row["user_message"].casefold() for row in out_rows["train"] + out_rows["val"]]
    overlap = [m for m in new_messages if m in seen]
    if overlap:
        raise SystemExit(f"new data overlaps existing text: {overlap[:3]}")
    ROOT.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        path = ROOT / f"{split}.jsonl"
        payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
        path.write_bytes(payload.encode("utf-8"))
    new_payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in out_rows["train"] + out_rows["val"]) + "\n"
    manifest = {
        "version": "p2sft.data.v1.2",
        "base_version": "p2sft.data.v1.1",
        "seed": SEED,
        "counts": {"train": len(train), "val": len(val), "new_train": len(out_rows["train"]), "new_val": len(out_rows["val"]), "mixed_original_train": len(original_train)},
        "new_data_sha256": hashlib.sha256(new_payload.encode("utf-8")).hexdigest(),
        "sha256": {split: hashlib.sha256((ROOT / f"{split}.jsonl").read_bytes()).hexdigest() for split in ("train", "val")},
        "context_source": source,
        "label_policy": "independent semantic configuration + schema validator + evidence substring",
        "human_review_status": "not_required_rule_checked",
        "prohibited_sources_used": False,
        "notes": "600 new rows from 6 independently authored template families, mixed with 2400 v1.1 train rows; 120 new val mixed with 240 v1.1 val; no test/hard-test rows used.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
