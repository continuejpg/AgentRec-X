"""Freeze a small independent hard shopping-expression test set (P2-SFT phase 2).

The sentences, labels and contexts here are newly authored for this diagnostic set; no
train/val/test template or frozen Agent benchmark sentence is copied.  Labels are
defined by this explicit semantic configuration and validated against the same versioned
operation schema.  Samples remain exploratory until the prepared human review is done.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import SCHEMA_VERSION, canonical_json, validate_proposal

OUT = Path(__file__).with_name("hard_test")
VERSION = "p2sft.hard_test.v1"
CONTEXTS = [
    {"category": "Water Filters", "title_example": "portable water filter"},
    {"category": "Camping Cookware", "title_example": "camping cookware set"},
    {"category": "Electric Lanterns", "title_example": "rechargeable camping lantern"},
    {"category": "Camping Stoves", "title_example": "foldable camping stove"},
    {"category": "Self-Inflating Sleeping Mats", "title_example": "self-inflating sleeping mat"},
]


def _state(goal=None, preferences=None, constraints=None, last_count=5):
    return {"goal": goal, "preferences": preferences or {}, "constraints": constraints or {}, "last_presented_count": last_count, "last_reference": "previous_state"}


def _proposal(ops, clarification=None):
    return {"schema_version": SCHEMA_VERSION, "operations": ops, "clarification": clarification}


def _op(op, attribute, value, direction, scope, evidence, **extra):
    value_obj = {
        "op": op, "attribute": attribute, "value": value, "direction": direction,
        "scope": scope, "evidence": evidence,
    }
    value_obj.update(extra)
    return value_obj


def _cases():
    cases = []
    # 1. multi-condition update
    for i, (budget, material) in enumerate([(40, "titanium"), (70, "aluminum"), (25, "steel")]):
        for j in range(4):
            msg = [
                f"Actually cap it at ${budget}, use {material}, and favor the lighter one.",
                f"Change the ceiling to ${budget}; I want {material} and lower weight.",
                f"Set the max to ${budget}, please use {material}, lighter is better.",
                f"Budget ${budget} max, {material} material, and the least heavy option.",
            ][j]
            ops = [
                _op("update_constraint", "price_max", budget, None, "task", f"${budget}"),
                _op("add_preference", "material", material, "exact", "task", material),
                _op("add_preference", "weight", None, "minimize", "task", "lighter" if "lighter" in msg.lower() else "lower weight" if "lower weight" in msg.lower() else "least heavy option"),
            ]
            cases.append((msg, _state({"product_type": "cookware", "role": "main"}, {}, {"price_max": 100}), ops, None, "multi_condition"))
    # 2. budget repeated correction
    for i, (msg, value) in enumerate([
        ("Start over with a ceiling of $60.", 60),
        ("No, set the cap to $20 now.", 20),
        ("Actually the absolute upper limit is $90.", 90),
        ("Revise the maximum to $35.", 35),
        ("Change my spending limit to $75.", 75),
    ]):
        ev = f"${value}"
        cases.append((msg, _state({"product_type": "water_filter", "role": "main"}, {}, {"price_max": 30}), [_op("update_constraint", "price_max", value, None, "task", ev)], None, "budget_repeat"))
    # 3. complex retraction
    for msg, attr, evidence in [
        ("Forget the weight preference, keep the color.", "weight", "Forget the weight preference"),
        ("Drop the material requirement and leave the budget alone.", "material", "Drop the material requirement"),
        ("Ignore the lighter request; keep the blue preference.", "weight", "Ignore the lighter request"),
        ("Remove the titanium preference.", "material", "Remove the titanium preference"),
        ("Clear the weight direction now.", "weight", "Clear the weight direction"),
    ]:
        cases.append((msg, _state({"product_type": "cookware", "role": "main"}, {"weight": "minimize", "material": "titanium"}, {"price_max": 50}), [_op("retract_preference", attr, None, None, "task", evidence)], None, "complex_retraction"))
    # 4. cross-turn reference
    for msg, ref in [
        ("Take the third result from the previous shortlist.", "previous_candidate_3"),
        ("I meant the second one you showed.", "previous_candidate_2"),
        ("Use the fourth option from the earlier list.", "previous_candidate_4"),
        ("The last one in the previous comparison.", "previous_candidate_5"),
        ("Go back to the first result in that list.", "previous_candidate_1"),
    ]:
        cases.append((msg, _state({"product_type": "stove", "role": "main"}, last_count=5), [_op("resolve_reference", None, None, None, "task", msg, reference=ref, reason="cross_turn_reference")], None, "cross_turn_reference"))
    # 5. implicit but unconfirmed
    for msg in [
        "These look a bit heavy, but I haven't decided.",
        "Maybe lighter might be nice, not sure yet.",
        "I am considering titanium, but do not lock it in.",
        "The budget seems high; I have not confirmed a new one.",
        "It could be blue, but I am only thinking out loud.",
    ]:
        cases.append((msg, _state({"product_type": "cookware", "role": "main"}), [_op("noop", None, None, None, None, msg, reason="no_change")], None, "implicit_unconfirmed"))
    # 6. negation / contradiction
    for msg, role_evidence in [
        ("Not the drinking bottle; I want the complete filter device.", "complete filter device"),
        ("Do not show a bottle, show the full water purifier.", "full water purifier"),
        ("The bottle version is wrong; use the complete filter instead.", "complete filter"),
        ("I don't mean a filtered bottle, I mean the main purifier.", "main purifier"),
        ("Not a bottle at all - give me the complete water filter.", "complete water filter"),
    ]:
        cases.append((msg, _state({"product_type": "filtered_bottle", "role": "main"}), [_op("update_goal", "role", "main", None, "task", role_evidence)], None, "negation_role"))
    # 7. accessory role changes
    for msg, ev in [
        ("Just the spare mouthpiece, not the whole filter.", "spare mouthpiece"),
        ("I only need the replacement cartridge.", "replacement cartridge"),
        ("Switch to the mouthpiece accessory only.", "mouthpiece accessory"),
        ("Give me the replacement part rather than the main unit.", "replacement part"),
        ("Actually the spare valve is enough.", "spare valve"),
    ]:
        cases.append((msg, _state({"product_type": "water_filter", "role": "main"}), [_op("update_goal", "role", "accessory", None, "task", ev)], None, "accessory_role"))
    # 8. clarify
    for msg in [
        "I have no idea which outdoor category to pick; help me choose.",
        "Could you help me decide between a filter and cookware?",
        "Not sure what kind of gear this should be yet.",
        "Please ask me a category question first.",
        "I cannot decide between lantern and sleeping pad.",
    ]:
        cases.append((msg, _state(), [_op("clarify", "product_type", None, None, "task", msg)], {"attribute": "product_type", "question": "Which product category should I use?", "choices": ["water filter", "cookware", "lantern", "stove"]}, "clarify"))
    # 9. noop
    for msg in [
        "Carry on; nothing new from me.",
        "That is fine, keep the current plan.",
        "No extra constraint this turn.",
        "Nothing to change right now.",
        "Continue with what you already have.",
    ]:
        cases.append((msg, _state({"product_type": "lantern", "role": "main"}), [_op("noop", None, None, None, None, msg, reason="declined_preference")], None, "noop"))
    # 10. abstain
    for msg in [
        "I want a submarine for the campsite.",
        "Find me a licensed helicopter.",
        "Can you sell me a teleporter?",
        "I need a spacecraft, not camping gear.",
        "Book me a rocket launch instead.",
    ]:
        cases.append((msg, _state({"product_type": "water_filter", "role": "main"}), [_op("abstain", None, None, None, None, msg, reason="unsupported_request")], None, "abstain"))
    # 11. hard price + material + direction (another wording family)
    for i, (msg, budget, material, evidence_weight) in enumerate([
        ("Actually $55 max, steel only, and prefer the lightest.", 55, "steel", "lightest"),
        ("Change ceiling to $45, use aluminum, prioritize lower weight.", 45, "aluminum", "lower weight"),
        ("Set a $65 cap, titanium, minimize weight.", 65, "titanium", "minimize weight"),
        ("Make it at most $30, use steel, lighter wins.", 30, "steel", "lighter"),
        ("At most $80, aluminum, choose the lightest possible.", 80, "aluminum", "lightest"),
    ]):
        cases.append((msg, _state({"product_type": "cookware", "role": "main"}, {}, {"price_max": 100}), [
            _op("update_constraint", "price_max", budget, None, "task", f"${budget}"),
            _op("add_preference", "material", material, "exact", "task", material),
            _op("add_preference", "weight", None, "minimize", "task", evidence_weight),
        ], None, "multi_condition_2"))
    # 12. retract constraint
    for msg in [
        "Forget the maximum budget now.",
        "Clear the spending ceiling.",
        "Remove the price cap entirely.",
        "Drop the upper price limit.",
        "Ignore the budget constraint from earlier.",
    ]:
        cases.append((msg, _state({"product_type": "stove", "role": "main"}, {}, {"price_max": 60}), [_op("retract_constraint", "price_max", None, None, "task", msg)], None, "retract_constraint"))
    # 13. role correction back to main after accessory
    for msg in [
        "Actually give me the complete filter, not just the mouthpiece.",
        "Switch back to the full filtration system.",
        "Forget the replacement part; show the complete unit.",
        "I need the main purifier instead of the accessory.",
        "Return to the complete water filter.",
    ]:
        cases.append((msg, _state({"product_type": "water_filter", "role": "accessory"}), [_op("update_goal", "role", "main", None, "task", msg)], None, "role_correction"))
    # 14. noop after withdrawal
    for msg in [
        "I withdrew that; nothing else to add.",
        "The item is unchanged, no new preference.",
        "Ignore my last thought; keep everything as is.",
        "No correction needed after all.",
        "That withdrawal is enough for now.",
    ]:
        cases.append((msg, _state({"product_type": "lantern", "role": "main"}), [_op("noop", None, None, None, None, msg, reason="no_change")], None, "noop_after_withdraw"))
    # 15. additional hard combinations (kept independent of train/val/test templates)
    extra_multi = [
        ("Cap at $42, use titanium, and favor the lightest option.", 42, "titanium", "lightest option"),
        ("At most $58, aluminum, minimize the weight please.", 58, "aluminum", "minimize the weight"),
        ("Set $33 as the ceiling, steel, and pick the least heavy.", 33, "steel", "least heavy"),
        ("Change the cap to $77, titanium, lighter wins.", 77, "titanium", "lighter"),
        ("Keep it below $26, use aluminum, and prefer light weight.", 26, "aluminum", "light weight"),
    ]
    for msg, budget, material, weight_evidence in extra_multi:
        cases.append((msg, _state({"product_type": "cookware", "role": "main"}, {}, {"price_max": 100}), [
            _op("update_constraint", "price_max", budget, None, "task", f"${budget}"),
            _op("add_preference", "material", material, "exact", "task", material),
            _op("add_preference", "weight", None, "minimize", "task", weight_evidence),
        ], None, "multi_condition_3"))
    extra_retract = [
        ("Forget the weight preference; keep the material as is.", "weight", "Forget the weight preference"),
        ("Drop the lighter direction, nothing else.", "weight", "Drop the lighter direction"),
        ("Remove the titanium requirement, keep the budget.", "material", "Remove the titanium requirement"),
        ("Clear my weight preference but leave the color.", "weight", "Clear my weight preference"),
        ("Ignore the material preference only.", "material", "Ignore the material preference"),
    ]
    for msg, attr, ev in extra_retract:
        cases.append((msg, _state({"product_type": "cookware", "role": "main"}, {"weight": "minimize", "material": "titanium", "color": "blue"}), [_op("retract_preference", attr, None, None, "task", ev)], None, "complex_retraction_2"))
    extra_negation = [
        ("That is a bottle, not what I asked; show the complete purifier.", "complete purifier"),
        ("No bottle: the full filtration unit is the target.", "full filtration unit"),
        ("I reject the bottle option; choose the complete filter.", "complete filter"),
        ("Not the bottle category; use the main purifier.", "main purifier"),
        ("Bottle is wrong; I need the complete water treatment device.", "complete water treatment device"),
    ]
    for msg, ev in extra_negation:
        cases.append((msg, _state({"product_type": "filtered_bottle", "role": "main"}), [_op("update_goal", "role", "main", None, "task", ev)], None, "negation_role_2"))
    for msg in [
        "Please ask which category I mean before searching.",
        "I need a clarifying question about the product type.",
        "Which category would you like me to choose from?",
    ]:
        cases.append((msg, _state(), [_op("clarify", "product_type", None, None, "task", msg)], {"attribute": "product_type", "question": "Which product category should I use?", "choices": ["water filter", "cookware", "lantern", "stove"]}, "clarify_2"))
    for msg in [
        "Nothing new; continue the previous search.",
        "No new preference, go on.",
        "Keep everything unchanged for this turn.",
    ]:
        cases.append((msg, _state({"product_type": "lantern", "role": "main"}), [_op("noop", None, None, None, None, msg, reason="no_change")], None, "noop_2"))
    for msg in [
        "I need a working time machine for camping.",
        "Find me a lunar rover.",
    ]:
        cases.append((msg, _state({"product_type": "water_filter", "role": "main"}), [_op("abstain", None, None, None, None, msg, reason="unsupported_request")], None, "abstain_2"))
    return cases


def main() -> int:
    cases = _cases()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (message, state, operations, clarification, scenario) in enumerate(cases):
        proposal = _proposal(operations, clarification)
        errors = validate_proposal(proposal, message)
        if errors:
            raise SystemExit(f"case {index} invalid: {errors}\n{message}")
        rows.append({
            "sample_id": f"hard_{index:04d}",
            "split": "hard_test",
            "scenario": scenario,
            "source": "independent_manual_semantic_config_p2sft_hard_v1",
            "catalogue_context": CONTEXTS[index % len(CONTEXTS)],
            "state": state,
            "user_message": message,
            "assistant": canonical_json(proposal),
            "human_review_status": "pending",
        })
    path = OUT / "hard_test.jsonl"
    content = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    path.write_bytes(content.encode("utf-8"))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    review = rows[:20]
    review_path = OUT / "hard_test_manual_review_20.csv"
    import csv
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "scenario", "user_message", "gold_json", "reviewer", "reviewed_at", "label_status", "notes"])
        writer.writeheader()
        for row in review:
            writer.writerow({"sample_id": row["sample_id"], "scenario": row["scenario"], "user_message": row["user_message"], "gold_json": row["assistant"], "reviewer": "", "reviewed_at": "", "label_status": "pending", "notes": ""})
    manifest = {
        "version": VERSION,
        "samples": len(rows),
        "sha256": digest,
        "human_review_status": "pending",
        "human_reviewed_count": 0,
        "source": "independent manual semantic configuration; no train/val/test template copy; not LLM-generated",
        "label_status": "frozen_pending_human_review",
        "contains_product_identity": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
