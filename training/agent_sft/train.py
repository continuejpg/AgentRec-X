"""Qwen3.5-9B LoRA SFT for shopping preference operation proposals.

Run this on AutoDL (A100 80GB recommended).  Local CPU use is limited to data/schema
contract checks; this script deliberately lazy-imports torch/transformers/PEFT.

Example:
  python -m training.agent_sft.train --config training/agent_sft/configs/qwen3_5_9b_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping

from .format import completion_only_encoding, validate_dataset_row
from .modeling import MODEL_ID, describe_model, load_text_model, load_tokenizer
from .schema import SCHEMA_VERSION


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - AutoDL installs PyYAML
            raise RuntimeError("PyYAML is required for YAML configs") from exc
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return value


def _validate_rows(rows: list[dict[str, Any]], split: str) -> None:
    if not rows:
        raise ValueError(f"{split} dataset is empty")
    errors: list[str] = []
    for index, row in enumerate(rows):
        for error in validate_dataset_row(row):
            errors.append(f"{split}[{index}] {error}")
        if len(errors) > 20:
            break
    if errors:
        raise ValueError("dataset schema validation failed:\n" + "\n".join(errors[:20]))


def _select_completion_only(tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int, eos_token: str | None) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    truncated = 0
    for row in rows:
        input_ids, labels, info = completion_only_encoding(
            tokenizer,
            row,
            eos_token=eos_token,
            max_seq_length=max_seq_length,
        )
        if info["truncated"]:
            truncated += 1
        if info["completion_tokens"] <= 0:
            raise ValueError(f"sample {row.get('sample_id')} has no completion tokens")
        encoded.append({"input_ids": input_ids, "labels": labels, **info})
    if truncated:
        raise ValueError(f"{truncated}/{len(rows)} samples exceed max_seq_length={max_seq_length}; refusing to train on silently truncated completions")
    return encoded


class _JsonlDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class _CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention = []
        labels = []
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention.append([1] * len(feature["input_ids"]) + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _chat_template(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise RuntimeError("Qwen tokenizer has no chat_template; refusing to guess a template")
    return str(template)


def _preflight(tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int, eos_token: str | None) -> dict[str, Any]:
    sample = rows[: min(8, len(rows))]
    encoded = _select_completion_only(tokenizer, sample, max_seq_length, eos_token)
    return {
        "checked_samples": len(sample),
        "prompt_tokens_max": max(item["prompt_tokens"] for item in encoded),
        "total_tokens_max": max(item["total_tokens"] for item in encoded),
        "masked_prompt_tokens": sum(1 for item in encoded for value in item["labels"][: item["prompt_tokens"]] if value == -100),
        "completion_loss_tokens": sum(
            len(item["labels"]) - item["prompt_tokens"] for item in encoded
        ),
        "chat_template": _chat_template(tokenizer)[:200],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B preference-operation LoRA SFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None, help="local Qwen3.5-9B path (defaults to config model_name_or_path)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-adapter", default=None, help="continue training from an existing LoRA adapter")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 20261105))
    random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    val_rows = load_jsonl(args.data_dir / "val.jsonl")
    _validate_rows(train_rows, "train")
    _validate_rows(val_rows, "val")
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("data manifest schema_version does not match the proposal schema")

    model_name = str(args.model or config.get("model_name_or_path", MODEL_ID))
    if "Qwen3.5-9B" not in model_name and model_name != MODEL_ID:
        raise ValueError(f"this branch is Qwen3.5-only; got model {model_name!r}")
    tokenizer = load_tokenizer(model_name, trust_remote_code=bool(config.get("trust_remote_code", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer.eos_token:
        raise RuntimeError("tokenizer has no eos token; completion-only labels need it")
    preflight = _preflight(tokenizer, train_rows, int(config.get("max_seq_length", 2048)), tokenizer.eos_token)
    train_encoded = _select_completion_only(tokenizer, train_rows, int(config.get("max_seq_length", 2048)), tokenizer.eos_token)
    val_encoded = _select_completion_only(tokenizer, val_rows, int(config.get("max_seq_length", 2048)), tokenizer.eos_token)

    dtype = torch.bfloat16 if bool(config.get("bf16", True)) else torch.float16
    quantization_config = None
    if bool(config.get("load_in_4bit", False)):
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = load_text_model(
        model_name,
        dtype=dtype,
        device_map=None,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        quantization_config=quantization_config,
        use_kernels=bool(config.get("use_kernels", False)),
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model_info = describe_model(model)
    if not model_info["model_type_recognized"]:
        raise RuntimeError(f"loaded model is not the documented Qwen3.5 CausalLM: {model_info}")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules") or "all-linear",
    )
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("LoRA target modules selected no trainable parameters for Qwen3.5-9B")

    output_dir = args.output_dir or Path(config.get("output_dir", "training/agent_sft/output/qwen3_5_9b_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(train_encoded) // max(1, int(config.get("per_device_train_batch_size", 2)) * int(config.get("gradient_accumulation_steps", 8))))
    total_steps = max(1, int(steps_per_epoch * float(config.get("num_train_epochs", 2))))
    warmup_steps = max(0, int(round(total_steps * float(config.get("warmup_ratio", 0.03)))))
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(config.get("num_train_epochs", 2)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_steps": warmup_steps,
        "logging_steps": int(config.get("logging_steps", 10)),
        "eval_steps": int(config.get("eval_steps", 50)),
        "save_steps": int(config.get("save_steps", 50)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": bool(config.get("dataloader_pin_memory", True)),
        "seed": seed,
        "data_seed": seed,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
    }
    try:
        training_args = TrainingArguments(**training_kwargs, eval_strategy="steps")
    except TypeError:  # transformers < 4.46 spells this evaluation_strategy
        training_args = TrainingArguments(**training_kwargs, evaluation_strategy="steps")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_JsonlDataset(train_encoded),
        eval_dataset=_JsonlDataset(val_encoded),
        data_collator=_CompletionCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "chat_template.jinja").write_text(_chat_template(tokenizer), encoding="utf-8")
    (adapter_dir / "schema_version.txt").write_text(SCHEMA_VERSION + "\n", encoding="utf-8")
    history = list(trainer.state.log_history)
    (output_dir / "training_log_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "trainer_metrics.json").write_text(json.dumps(eval_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "model_name_or_path": model_name,
        "model_info": model_info,
        "lora_trainable_parameters": trainable,
        "schema_version": SCHEMA_VERSION,
        "adapter_dir": str(adapter_dir),
        "seed": seed,
        "config": config,
        "data_manifest": data_manifest,
        "preflight": preflight,
        "train_result": dict(train_result.metrics),
        "eval_metrics": eval_metrics,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": "training loss is diagnostic only; use parser eval + end-to-end D0/D1 for effectiveness",
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"adapter_dir": str(adapter_dir), "eval_loss": eval_metrics.get("eval_loss"), "report": str(output_dir / "training_report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
