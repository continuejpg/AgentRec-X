"""Prepare the independent P2-SFT preference-operation dataset.

Labels are generated from explicit semantic configurations in this file, not from any
model output or from the frozen Agent benchmark.  Product contexts are drawn from the
trusted Sports & Outdoors catalogue (category + title excerpt only; never parent_asin).
Train/validation/test use disjoint template banks so near-duplicate wording cannot leak
across the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .schema import SCHEMA_VERSION, canonical_json, validate_proposal

GENERATOR_VERSION = "p2sft.data.v1"
DEFAULT_SEED = 20261105
DATA_DIR = Path(__file__).with_name("data")

# Template banks are intentionally disjoint across split.  Test is not a paraphrase of a
# training template: it has its own template_id and wording family.
TEMPLATES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "price_update": {
        "train": [
            ("train_price_a", "Actually make the budget under ${budget}."),
            ("train_price_b", "Can you change my limit to ${budget}?"),
            ("train_price_c", "I want the price cap at ${budget} instead."),
        ],
        "val": [("val_price_a", "New budget: ${budget} maximum.")],
        "test": [("test_price_a", "Please revise the spending ceiling to ${budget}.")],
    },
    "price_retract": {
        "train": [("train_pretract_a", "Forget the budget limit."), ("train_pretract_b", "Drop the price cap please.")],
        "val": [("val_pretract_a", "Remove my maximum price.")],
        "test": [("test_pretract_a", "The budget no longer matters; clear it.")],
    },
    "soft_weight": {
        "train": [
            ("train_weight_a", "I prefer a lighter weight."),
            ("train_weight_b", "Lighter is better for me."),
        ],
        "val": [("val_weight_a", "Please favor lighter options.")],
        "test": [("test_weight_a", "If possible, choose the least heavy one.")],
    },
    "material_pref": {
        "train": [("train_material_a", "Titanium cookware is my preference.")],
        "val": [("val_material_a", "Titanium is my material preference.")],
        "test": [("test_material_a", "I would rather have the titanium version.")],
    },
    "new_goal": {
        "train": [
            ("train_goal_a", "I need a {target_phrase} for {use}."),
            ("train_goal_b", "Please find me a {target_phrase} for {use}."),
        ],
        "val": [("val_goal_a", "My new target is a {target_phrase}.")],
        "test": [("test_goal_a", "Switch me to a {target_phrase} suitable for {use}.")],
    },
    "role_correction": {
        "train": [("train_role_a", "Correction: I want the complete filter device rather than the drinking bottle.")],
        "val": [("val_role_a", "Correction: I want the full filter device, not the bottle.")],
        "test": [("test_role_a", "Not the bottle version; give me the complete purifier instead.")],
    },
    "accessory_role": {
        "train": [("train_accessory_a", "I only need a replacement mouthpiece for a water filter.")],
        "val": [("val_accessory_a", "Please narrow it to the replacement mouthpiece only.")],
        "test": [("test_accessory_a", "The spare mouthpiece alone is what I actually require.")],
    },
    "ambiguity": {
        "train": [("train_amb_a", "I am unsure which outdoor item category fits me.")],
        "val": [("val_amb_a", "I am not sure what kind of outdoor product to get.")],
        "test": [("test_amb_a", "Please help me choose a category; I have no idea yet.")],
    },
    "noop": {
        "train": [("train_noop_a", "Nothing further from me."), ("train_noop_b", "That is all, continue as is.")],
        "val": [("val_noop_a", "Nothing else to add.")],
        "test": [("test_noop_a", "I have no further requirement right now.")],
    },
    "unsupported": {
        "train": [("train_unsupported_a", "I need a spaceship.")],
        "val": [("val_unsupported_a", "Find me a private jet.")],
        "test": [("test_unsupported_a", "I want a teleportation device.")],
    },
    "cross_turn_reference": {
        "train": [("train_ref_a", "Show me the second option.")],
        "val": [("val_ref_a", "Compare the third one for me.")],
        "test": [("test_ref_a", "Tell me about the fourth result I just saw.")],
    },
}

_TARGETS = {
    "water_filter": ("complete water filter", "water filter", ["hiking", "camping", "emergency preparedness"]),
    "cookware": ("camping cookware set", "cookware", ["backpacking", "camp cooking", "group trips"]),
    "lantern": ("camping lantern", "lantern", ["night hiking", "tent lighting"]),
    "stove": ("camping stove", "stove", ["backpacking", "boiling water"]),
    "sleeping_pad": ("sleeping pad", "sleeping pad", ["tent camping", "cold weather"]),
}

_FALLBACK_CONTEXTS = [
    {"category": "Water Filters", "title_example": "portable water filtration system"},
    {"category": "Camping Cookware", "title_example": "camping cookware and pot set"},
    {"category": "Electric Lanterns", "title_example": "rechargeable camping lantern"},
    {"category": "Camping Stoves", "title_example": "portable camping stove"},
    {"category": "Self-Inflating Sleeping Mats", "title_example": "self-inflating sleeping pad"},
    {"category": "Sports Water Bottles", "title_example": "filtered water bottle"},
]


def _contexts_from_catalogue(limit: int = 60) -> tuple[list[dict[str, str]], str]:
    try:
        from recommendation.api.app import DEFAULT_CATALOG
        from recommendation.catalog import MetadataIndex

        records = MetadataIndex.load(DEFAULT_CATALOG).records
    except Exception:
        return list(_FALLBACK_CONTEXTS), "fallback_static_categories"
    wanted = ("water filters", "water purifiers", "cookware", "lantern", "stove", "sleeping")
    grouped: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in records.values():
        categories = tuple(str(value) for value in (getattr(record, "categories", ()) or ()))
        title = " ".join(str(getattr(record, "title", "") or "").split())
        if not categories or not title:
            continue
        category = categories[-1]
        lowered = (category + " " + title).casefold()
        if not any(term in lowered for term in wanted):
            continue
        target = _infer_target({"category": category, "title_example": title})
        if target is None:
            continue
        key = (category.casefold(), title.casefold()[:80])
        if key in seen:
            continue
        grouped.setdefault(target, []).append({"category": category, "title_example": title[:120]})
        seen.add(key)
    # Balance across target types so a long catalogue iteration cannot fill the context
    # budget with sleeping mats before water filters are ever seen.
    per_target = max(1, limit // max(1, len(grouped)))
    contexts: list[dict[str, str]] = []
    for target in sorted(grouped):
        contexts.extend(grouped[target][:per_target])
    return (contexts or list(_FALLBACK_CONTEXTS)), "real_sports_outdoors_catalogue"


def _state_template(scenario: str) -> dict[str, Any]:
    base = {
        "goal": None,
        "preferences": {},
        "constraints": {},
        "last_presented_count": 5,
        "last_reference": "previous_state",
    }
    if scenario in {"price_update", "price_retract"}:
        base["goal"] = {"product_type": "water_filter", "role": "main"}
        base["constraints"] = {"price_max": 10}
    elif scenario in {"soft_weight", "material_pref", "role_correction", "accessory_role"}:
        base["goal"] = {"product_type": "cookware" if scenario == "material_pref" else "water_filter", "role": "main"}
        if scenario == "soft_weight":
            base["preferences"] = {"weight": "minimize"}
        if scenario == "role_correction":
            base["goal"] = {"product_type": "filtered_bottle", "role": "main"}
    elif scenario == "cross_turn_reference":
        base["goal"] = {"product_type": "cookware", "role": "main"}
        base["last_reference"] = "previous_candidate_2"
    return base


def _ops(scenario: str, message: str, context: dict[str, str], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if scenario == "price_update":
        evidence = re.search(r"under \$?\d+|limit to \$?\d+|cap at \$?\d+|budget: \$?\d+ maximum|ceiling to \$?\d+", message, re.I)
        evidence_text = evidence.group(0) if evidence else message
        amount = re.search(r"\d+(?:\.\d+)?", evidence_text)
        if amount is None:
            raise ValueError(f"price update has no numeric budget: {message!r}")
        value: int | float = float(amount.group(0))
        if value.is_integer():
            value = int(value)
        return ([{"op": "update_constraint", "attribute": "price_max", "value": value, "direction": None, "scope": "task", "evidence": evidence_text}], None)
    if scenario == "price_retract":
        return ([{"op": "retract_constraint", "attribute": "price_max", "value": None, "direction": None, "scope": "task", "evidence": message}], None)
    if scenario == "soft_weight":
        return ([{"op": "add_preference", "attribute": "weight", "value": None, "direction": "minimize", "scope": "task", "evidence": message}], None)
    if scenario == "material_pref":
        return ([{"op": "add_preference", "attribute": "material", "value": "titanium", "direction": "exact", "scope": "task", "evidence": message}], None)
    if scenario == "new_goal":
        target = context["target_type"]
        target_phrase = _TARGETS[target][0]
        evidence = target_phrase if target_phrase in message else _TARGETS[target][1]
        return ([{"op": "set_goal", "attribute": "product_type", "value": target, "direction": None, "scope": "task", "evidence": evidence}], None)
    if scenario == "role_correction":
        for phrase in ("complete water filter", "full filter device", "complete purifier"):
            if phrase in message:
                return ([{"op": "update_goal", "attribute": "role", "value": "main", "direction": None, "scope": "task", "evidence": phrase}], None)
        return ([{"op": "update_goal", "attribute": "role", "value": "main", "direction": None, "scope": "task", "evidence": message}], None)
    if scenario == "accessory_role":
        for phrase in ("replacement mouthpiece", "replacement mouthpiece only", "spare mouthpiece", "mouthpiece"):
            if phrase in message:
                return ([{"op": "update_goal", "attribute": "role", "value": "accessory", "direction": None, "scope": "task", "evidence": phrase}], None)
        return ([{"op": "update_goal", "attribute": "role", "value": "accessory", "direction": None, "scope": "task", "evidence": message}], None)
    if scenario == "ambiguity":
        return ([{"op": "clarify", "attribute": "product_type", "value": None, "direction": None, "scope": "task", "evidence": message}], {
            "attribute": "product_type",
            "question": "Which outdoor product category do you want?",
            "choices": ["water filter", "camping cookware", "lantern", "camping stove"],
        })
    if scenario == "noop":
        return ([{"op": "noop", "attribute": None, "value": None, "direction": None, "scope": None, "evidence": message, "reason": "declined_preference"}], None)
    if scenario == "unsupported":
        return ([{"op": "abstain", "attribute": None, "value": None, "direction": None, "scope": None, "evidence": message, "reason": "unsupported_request"}], None)
    if scenario == "cross_turn_reference":
        reference = "previous_candidate_2"
        number = "second"
        if "third" in message:
            reference, number = "previous_candidate_3", "third"
        elif "fourth" in message:
            reference, number = "previous_candidate_4", "fourth"
        return ([{"op": "resolve_reference", "attribute": None, "value": None, "direction": None, "scope": "task", "evidence": message, "reference": reference, "reason": "cross_turn_reference"}], None)
    raise ValueError(f"unknown scenario {scenario}")


def _message_for(scenario: str, template: str, context: dict[str, str], rng: random.Random) -> str:
    target = context["target_type"]
    target_phrase = _TARGETS[target][0]
    use = rng.choice(_TARGETS[target][2])
    budget = rng.choice([30, 50, 80])
    return template.format(target_phrase=target_phrase, use=use, budget=budget)


def _sample(sample_id: str, split: str, scenario: str, template_id: str, template: str, context: dict[str, str], rng: random.Random) -> dict[str, Any]:
    message = _message_for(scenario, template, context, rng)
    operations, clarification = _ops(scenario, message, context, rng)
    proposal = {"schema_version": SCHEMA_VERSION, "operations": operations}
    if clarification is not None:
        proposal["clarification"] = clarification
    else:
        proposal["clarification"] = None
    errors = validate_proposal(proposal, message)
    if errors:
        raise ValueError(f"generated invalid sample {sample_id}: {errors}")
    return {
        "sample_id": sample_id,
        "split": split,
        "scenario": scenario,
        "template_id": template_id,
        "state": _state_template(scenario),
        "catalogue_context": {"category": context["category"], "title_example": context["title_example"]},
        "user_message": message,
        "assistant": canonical_json(proposal),
        "source": GENERATOR_VERSION,
    }


def _split_templates(split: str) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for scenario, bank in TEMPLATES.items():
        for template_id, template in bank[split]:
            items.append((scenario, template_id, template))
    return items


def generate(*, train_size: int = 2400, val_size: int = 240, test_size: int = 200, seed: int = DEFAULT_SEED, contexts: list[dict[str, str]] | None = None) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    contexts = contexts or [dict(value, target_type=_infer_target(value)) for value in _contexts_from_catalogue()[0]]
    contexts = [item for item in contexts if item.get("target_type") or _infer_target(item)]
    for item in contexts:
        item.setdefault("target_type", _infer_target(item))
    by_target: dict[str, list[dict[str, str]]] = {}
    for item in contexts:
        by_target.setdefault(str(item["target_type"]), []).append(item)
    if not by_target:
        raise ValueError("no usable catalogue contexts")
    result: dict[str, list[dict[str, Any]]] = {}
    for split, size in (("train", train_size), ("val", val_size), ("test", test_size)):
        bank = _split_templates(split)
        rows: list[dict[str, Any]] = []
        for index in range(size):
            scenario, template_id, template = bank[index % len(bank)]
            desired = {
                "price_update": "water_filter",
                "price_retract": "water_filter",
                "soft_weight": "water_filter",
                "material_pref": "cookware",
                "role_correction": "water_filter",
                "accessory_role": "water_filter",
                "cross_turn_reference": "cookware",
            }.get(scenario)
            if scenario == "new_goal":
                desired = list(by_target.keys())[rng.randrange(len(by_target))]
            candidates = by_target.get(desired) or contexts
            context = candidates[rng.randrange(len(candidates))]
            sample = _sample(f"p2sft_{split}_{index:05d}", split, scenario, template_id, template, context, rng)
            rows.append(sample)
        result[split] = rows
    return result


def _infer_target(context: dict[str, str]) -> str | None:
    text = (context.get("category", "") + " " + context.get("title_example", "")).casefold()
    if "water filters" in text or "water purifiers" in text:
        return "water_filter"
    if "sports water bottles" in text:
        return None
    if "lantern" in text:
        return "lantern"
    if "stove" in text:
        return "stove"
    if "sleeping" in text:
        return "sleeping_pad"
    if "cookware" in text or "pots" in text:
        return "cookware"
    return None


def write_dataset(out_dir: Path, *, train_size: int = 2400, val_size: int = 240, test_size: int = 200, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    contexts, context_source = _contexts_from_catalogue()
    rows = generate(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed, contexts=[dict(value, target_type=_infer_target(value)) for value in contexts])
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, samples in rows.items():
        path = out_dir / f"{split}.jsonl"
        content = "\n".join(json.dumps(sample, ensure_ascii=False, sort_keys=True) for sample in samples) + "\n"
        path.write_bytes(content.encode("utf-8"))
        hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        counts[split] = len(samples)
    train_templates = {sample["template_id"] for sample in rows["train"]}
    test_templates = {sample["template_id"] for sample in rows["test"]}
    val_templates = {sample["template_id"] for sample in rows["val"]}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "counts": counts,
        "sha256": hashes,
        "context_source": context_source,
        "context_count": len(contexts),
        "split_rule": "disjoint template banks; test templates never appear in train/val",
        "template_ids": {"train": sorted(train_templates), "val": sorted(val_templates), "test": sorted(test_templates)},
        "leakage_checks": {
            "test_train_template_overlap": sorted(test_templates & train_templates),
            "test_val_template_overlap": sorted(test_templates & val_templates),
            "train_val_template_overlap": sorted(train_templates & val_templates),
            "test_train_exact_message_overlap": len(
                {sample["user_message"] for sample in rows["test"]}
                & {sample["user_message"] for sample in rows["train"]}
            ),
        },
        "label_policy": "independent semantic configuration + schema validator; no runtime C output used",
        "contains_product_identity": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare independent P2-SFT preference-operation data")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--val-size", type=int, default=240)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = write_dataset(args.out, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
