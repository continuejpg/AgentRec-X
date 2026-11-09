"""G0 GPU preflight for Qwen3.5-9B.

Run on AutoDL before any training.  It verifies the model class, tokenizer/chat template,
non-thinking call, JSON generation, completion-only masks and LoRA target selection.
It writes a machine-readable report and exits non-zero on any blocker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from .format import completion_only_encoding, validate_dataset_row
from .modeling import MODEL_ID, describe_model, load_text_model, load_tokenizer
from .schema import parse_and_validate
from .train import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B G0 compatibility preflight")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--out", type=Path, default=Path("training/agent_sft/reports/g0_preflight.json"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"stage": "G0", "model": args.model, "checks": {}, "blocked": []}
    try:
        rows = load_jsonl(args.data_dir / "train.jsonl")
        sample = rows[0]
        errors = validate_dataset_row(sample)
        report["checks"]["dataset_schema"] = not errors
        if errors:
            report["blocked"].append(f"dataset: {errors[:3]}")
        tokenizer = load_tokenizer(args.model, trust_remote_code=True)
        report["tokenizer_class"] = type(tokenizer).__name__
        report["chat_template_present"] = bool(getattr(tokenizer, "chat_template", None))
        # Qwen3.5 must use the non-thinking template; a failure is a blocker rather than a
        # silent fallback to thinking mode.
        try:
            template = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Return the JSON object."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            report["checks"]["non_thinking_template"] = True
            report["template_preview"] = str(template)[:200]
        except TypeError as exc:
            report["checks"]["non_thinking_template"] = False
            report["blocked"].append(f"enable_thinking=False unsupported: {exc}")
        encoded = [
            completion_only_encoding(tokenizer, row, eos_token=tokenizer.eos_token, max_seq_length=2048)
            for row in rows[:32]
        ]
        report["lengths"] = {
            "max_total_tokens": max(info["total_tokens"] for _ids, _labels, info in encoded),
            "max_prompt_tokens": max(info["prompt_tokens"] for _ids, _labels, info in encoded),
            "truncated_count": sum(1 for _ids, _labels, info in encoded if info["truncated"]),
            "completion_tokens_min": min(info["completion_tokens"] for _ids, _labels, info in encoded),
            "masked_prompt_tokens": sum(
                1
                for _ids, labels, info in encoded
                for value in labels[: info["prompt_tokens"]]
                if value == -100
            ),
        }
        report["checks"]["completion_only_mask"] = (
            report["lengths"]["truncated_count"] == 0
            and report["lengths"]["completion_tokens_min"] > 0
            and report["lengths"]["masked_prompt_tokens"] > 0
        )
        if not report["checks"]["completion_only_mask"]:
            report["blocked"].append("completion-only mask/truncation check failed")
        model = load_text_model(args.model, dtype=__import__("torch").bfloat16, device_map="auto", trust_remote_code=True)
        import torch

        report["model_info"] = describe_model(model)
        report["checks"]["model_type"] = bool(report["model_info"].get("model_type_recognized"))
        if not report["checks"]["model_type"]:
            report["blocked"].append("loaded model is not the documented Qwen3.5 CausalLM")
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", None)
        report["architecture"] = {
            "layer_types": list(layer_types) if layer_types is not None else None,
            "linear_attention_layers": (
                list(layer_types).count("linear_attention") if layer_types is not None else None
            ),
            "full_attention_layers": (
                list(layer_types).count("full_attention") if layer_types is not None else None
            ),
            "causal_conv1d_available": importlib.util.find_spec("causal_conv1d") is not None,
            "fla_available": importlib.util.find_spec("fla") is not None,
            "kernel_mode": "fast_kernels" if importlib.util.find_spec("causal_conv1d") and importlib.util.find_spec("fla") else "pytorch_fallback",
        }
        report["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
            "torch_cuda_version": torch.version.cuda,
            "total_memory_gib": (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                if torch.cuda.is_available()
                else None
            ),
            "allocated_gib_after_load": (
                torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else None
            ),
        }
        # One real greedy JSON generation through the same prompt formatter used by SFT.
        from .format import prompt_text

        prompt = prompt_text(tokenizer, sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
        elapsed = time.perf_counter() - started
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        payload, schema_errors = parse_and_validate(text, sample["user_message"])
        report["generation"] = {"latency_ms": elapsed * 1000.0, "raw_preview": text[:500], "schema_errors": schema_errors}
        report["checks"]["json_generation"] = payload is not None and not schema_errors
        if not report["checks"]["json_generation"]:
            report["blocked"].append("generated text did not parse into the versioned operation schema")
        # Verify PEFT can select trainable linear modules on this actual architecture.
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
        model = get_peft_model(model, lora)
        trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
        lora_modules = [name for name, _module in model.named_modules() if ".lora_" in name or name.endswith("lora_A") or name.endswith("lora_B")]
        module_distribution: dict[str, int] = {}
        for name in lora_modules:
            suffix = name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
            module_distribution[suffix] = module_distribution.get(suffix, 0) + 1
        report["lora"] = {
            "target_modules": "all-linear",
            "trainable_parameters": trainable,
            "lora_module_count": len(lora_modules),
            "module_distribution": module_distribution,
        }
        report["gpu"]["allocated_gib_after_lora"] = (
            torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else None
        )
        report["gpu"]["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else None
        )
        report["checks"]["lora_targets"] = trainable > 0 and len(lora_modules) > 0
        if trainable <= 0:
            report["blocked"].append("LoRA selected zero trainable parameters")
        report["final_recommendation"] = "proceed_to_G1" if not report["blocked"] else "blocked"
    except Exception as exc:  # noqa: BLE001 - preflight reports blockers instead of pretending
        report["blocked"].append(f"{type(exc).__name__}: {exc}")
        report["final_recommendation"] = "blocked"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "blocked": report["blocked"], "recommendation": report.get("final_recommendation")}, ensure_ascii=False))
    return 0 if not report["blocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
