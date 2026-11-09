"""P4B Planner LoRA SFT: reuse the existing Qwen3.5 LoRA stack, planner chat data only.

This is a bootstrap experiment.  It trains on planner chat rows (system/user/assistant),
masking the prompt and optimizing only the assistant Planner Action JSON.
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

from training.agent_sft.format import _apply_chat_template
from training.agent_sft.modeling import MODEL_ID, describe_model, load_text_model, load_tokenizer
from training.agent_sft.train import _CompletionCollator, _JsonlDataset, _chat_template, _load_config, load_jsonl

SCHEMA_VERSION = "agentrecx.planner_sft.v0"
CONTRACT_VERSION = "agentrecx.planner_action.v1"


def _encode(tokenizer: Any, row: Mapping[str, Any], max_seq_length: int) -> dict[str, Any]:
    messages = list(row["messages"])
    if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"{row.get('sample_id')}: expected system/user/assistant messages")
    prompt_text = _apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{row.get('sample_id')}: chat template prefix is not tokenizer-stable")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if labels[-1] == -100:
        raise ValueError(f"{row.get('sample_id')}: no completion labels")
    if len(full_ids) > max_seq_length:
        raise ValueError(f"{row.get('sample_id')}: {len(full_ids)} tokens > {max_seq_length}")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(full_ids) - len(prompt_ids),
        "total_tokens": len(full_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Planner Action LoRA SFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/planner_sft_v0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    seed = int(config.get("seed", 20261107))
    random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    train_rows = load_jsonl(args.data_dir / "chat_train.jsonl")
    val_rows = load_jsonl(args.data_dir / "chat_val.jsonl")
    if not train_rows or not val_rows:
        raise ValueError("planner chat train/val data is empty")
    model_name = str(args.model or config.get("model_name_or_path", MODEL_ID))
    if "Qwen3.5-9B" not in model_name and model_name != MODEL_ID:
        raise ValueError(f"P4B is Qwen3.5-9B-only, got {model_name!r}")
    tokenizer = load_tokenizer(model_name, trust_remote_code=bool(config.get("trust_remote_code", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_seq_length = int(config.get("max_seq_length", 2048))
    train_encoded = [_encode(tokenizer, row, max_seq_length) for row in train_rows]
    val_encoded = [_encode(tokenizer, row, max_seq_length) for row in val_rows]
    preflight = {
        "train_samples": len(train_encoded),
        "val_samples": len(val_encoded),
        "max_total_tokens": max(item["total_tokens"] for item in train_encoded + val_encoded),
        "prompt_tokens_min": min(item["prompt_tokens"] for item in train_encoded),
        "completion_tokens_min": min(item["completion_tokens"] for item in train_encoded),
        "masked_prompt_tokens": sum(item["labels"][: item["prompt_tokens"]].count(-100) for item in train_encoded),
        "chat_template_prefix": _chat_template(tokenizer)[:300],
    }
    dtype = torch.bfloat16 if bool(config.get("bf16", True)) else torch.float16
    model = load_text_model(model_name, dtype=dtype, device_map=None, trust_remote_code=bool(config.get("trust_remote_code", True)))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model_info = describe_model(model)
    if not model_info["model_type_recognized"]:
        raise RuntimeError(f"not Qwen3.5 CausalLM: {model_info}")
    model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules") or "all-linear",
    )
    model = get_peft_model(model, lora)
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("LoRA selected no trainable parameters")

    output_dir = args.output_dir or Path(config.get("output_dir", "training/planner_sft/output/qwen3_5_9b_planner_lora_v0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = int(config.get("per_device_train_batch_size", 2))
    accum = int(config.get("gradient_accumulation_steps", 8))
    steps_per_epoch = max(1, len(train_encoded) // max(1, batch * accum))
    total_steps = max(1, int(steps_per_epoch * float(config.get("num_train_epochs", 2))))
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(config.get("num_train_epochs", 2)),
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": accum,
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_steps": max(0, int(round(total_steps * float(config.get("warmup_ratio", 0.03))))),
        "logging_steps": int(config.get("logging_steps", 1)),
        "eval_steps": int(config.get("eval_steps", 10)),
        "save_steps": int(config.get("save_steps", 10)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": seed,
        "data_seed": seed,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
    }
    try:
        training_args = TrainingArguments(**training_kwargs, eval_strategy="steps")
    except TypeError:
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
    report = {
        "stage": "P4B-planner-lora-sft-v0",
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "model_name_or_path": model_name,
        "model_info": model_info,
        "adapter_dir": str(adapter_dir),
        "preflight": preflight,
        "lora_trainable_parameters": trainable,
        "train_result": dict(train_result.metrics),
        "eval_metrics": eval_metrics,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "training_log_history.json").write_text(json.dumps(list(trainer.state.log_history), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"adapter_dir": str(adapter_dir), "eval_metrics": eval_metrics, "train_metrics": dict(train_result.metrics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
