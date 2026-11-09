from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from training.agent_sft.evaluate_parser import evaluate, load_jsonl
from training.agent_sft.format import completion_only_encoding, validate_dataset_row
from training.agent_sft.rule_parser import rule_text
from training.agent_sft.schema import SCHEMA_VERSION, validate_proposal
from training.agent_sft.train import _select_completion_only, load_jsonl as load_train_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "training/agent_sft/data"
_ASIN = re.compile(r"\bB[0-9A-Z]{9}\b")


class FakeTokenizer:
    chat_template = "fake-qwen3-non-thinking"
    eos_token = "</s>"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        assert tokenize is False
        assert enable_thinking is False
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [ord(character) + 1 for character in text]}


def test_manifest_counts_hashes_and_leakage_checks() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 2400, "val": 240, "test": 200}
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["leakage_checks"]["test_train_template_overlap"] == []
    assert manifest["leakage_checks"]["test_val_template_overlap"] == []
    assert manifest["leakage_checks"]["test_train_exact_message_overlap"] == 0
    assert manifest["context_source"] == "real_sports_outdoors_catalogue"
    for split, digest in manifest["sha256"].items():
        content = (DATA_DIR / f"{split}.jsonl").read_bytes()
        import hashlib

        assert hashlib.sha256(content).hexdigest() == digest


def test_every_sample_is_schema_valid_and_has_evidence_substring() -> None:
    for split in ("train", "val", "test"):
        rows = load_train_jsonl(DATA_DIR / f"{split}.jsonl")
        assert rows
        for row in rows:
            assert validate_dataset_row(row) == []
            payload = json.loads(row["assistant"])
            for operation in payload["operations"]:
                assert operation["evidence"] in row["user_message"]


def test_assistant_never_contains_product_identity_or_unversioned_payload() -> None:
    for split in ("train", "val", "test"):
        for row in load_train_jsonl(DATA_DIR / f"{split}.jsonl"):
            assistant = row["assistant"]
            assert not _ASIN.search(assistant)
            assert SCHEMA_VERSION in assistant
            assert '"schema_version"' in assistant


def test_price_update_label_value_matches_spoken_budget() -> None:
    rows = [row for row in load_train_jsonl(DATA_DIR / "train.jsonl") if row["scenario"] == "price_update"]
    assert rows
    for row in rows:
        payload = json.loads(row["assistant"])
        operation = payload["operations"][0]
        numeric = re.search(r"\d+(?:\.\d+)?", operation["evidence"])
        assert numeric is not None
        assert float(operation["value"]) == float(numeric.group(0))


def test_completion_only_mask_marks_prompt_and_keeps_completion() -> None:
    row = load_train_jsonl(DATA_DIR / "train.jsonl")[0]
    prompt_ids, labels, info = completion_only_encoding(FakeTokenizer(), row, eos_token="</s>", max_seq_length=2048)
    assert info["truncated"] is False
    assert len(prompt_ids) == len(labels) == info["total_tokens"]
    assert labels[: info["prompt_tokens"]] == [-100] * info["prompt_tokens"]
    assert any(value != -100 for value in labels[info["prompt_tokens"]:])
    assert info["completion_tokens"] > 0


def test_preflight_rejects_or_accepts_with_same_mask_contract() -> None:
    rows = load_train_jsonl(DATA_DIR / "val.jsonl")[:4]
    encoded = _select_completion_only(FakeTokenizer(), rows, 2048, "</s>")
    assert len(encoded) == 4
    for item in encoded:
        assert sum(1 for value in item["labels"] if value != -100) > 0


def test_schema_rejects_asins_bad_evidence_and_unknown_ops() -> None:
    message = "I need a complete water filter."
    good = {"schema_version": SCHEMA_VERSION, "operations": [{"op": "set_goal", "attribute": "product_type", "value": "water_filter", "direction": None, "scope": "task", "evidence": "complete water filter"}], "clarification": None}
    assert validate_proposal(good, message) == []
    asin = json.loads(json.dumps(good))
    asin["operations"][0]["value"] = "B09CJN4XGM"
    assert any("identity" in error or "ASIN" in error for error in validate_proposal(asin, message))
    bad_evidence = json.loads(json.dumps(good))
    bad_evidence["operations"][0]["evidence"] = "not in message"
    assert any("substring" in error for error in validate_proposal(bad_evidence, message))
    bad_op = json.loads(json.dumps(good))
    bad_op["operations"][0]["op"] = "buy_product"
    assert any("invalid op" in error for error in validate_proposal(bad_op, message))


def test_rule_parser_is_valid_and_evaluable_locally() -> None:
    rows = load_jsonl(DATA_DIR / "test.jsonl")[:20]
    for row in rows:
        payload = json.loads(rule_text(row["user_message"]))
        assert validate_proposal(payload, row["user_message"]) == []

    def predictor(row):
        return rule_text(row["user_message"]), 0.001

    result = evaluate(rows, predictor)
    assert result["metrics"]["schema_valid_rate"] == 1.0
    assert result["metrics"]["samples"] == 20
    assert 0.0 <= result["metrics"]["operation_exact_match"] <= 1.0
    assert len(result["predictions"]) == 20


def test_no_frozen_benchmark_text_is_used_as_training_source() -> None:
    frozen_text = set()
    frozen_path = Path(__file__).resolve().parent.parent / "benchmarks/agent/tasks/dev.json"
    for task in json.loads(frozen_path.read_text(encoding="utf-8")):
        for turn in task["turns"]:
            frozen_text.add(turn["user_input"].lower())
    for split in ("train", "val", "test"):
        for row in load_train_jsonl(DATA_DIR / f"{split}.jsonl"):
            assert row["user_message"].lower() not in frozen_text
