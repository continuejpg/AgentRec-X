"""G1 smoke test: a few real Qwen3.5-9B LoRA optimisation steps.

This is not the formal run.  It checks finite loss, backward pass, adapter save/reload and
one schema-parseable generation.  Run only after G0 passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .format import completion_only_encoding, prompt_text
from .modeling import MODEL_ID, describe_model, load_text_model, load_tokenizer
from .schema import parse_and_validate
from .train import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B G1 LoRA smoke train")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--out", type=Path, default=Path("training/agent_sft/output/qwen3_5_9b_smoke"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model

    rows = [row for row in load_jsonl(args.data_dir / "train.jsonl") if row["scenario"] in {"price_update", "soft_weight", "noop"}][:4]
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    encoded = [completion_only_encoding(tokenizer, row, eos_token=tokenizer.eos_token, max_seq_length=args.max_seq_length) for row in rows]
    model = load_text_model(args.model, dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    max_len = max(len(item[0]) for item in encoded)
    for input_ids, labels, _info in encoded:
        pad = max_len - len(input_ids)
        batch["input_ids"].append(input_ids + [tokenizer.pad_token_id] * pad)
        batch["attention_mask"].append([1] * len(input_ids) + [0] * pad)
        batch["labels"].append(labels + [-100] * pad)
    tensors = {key: torch.tensor(value, dtype=torch.long).to(model.device) for key, value in batch.items()}
    losses: list[float] = []
    for _step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors)
        loss = output.loss
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    args.out.mkdir(parents=True, exist_ok=True)
    adapter = args.out / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (adapter / "schema_version.txt").write_text("agentrecx.preference_operation.v1\n", encoding="utf-8")
    # Reload the adapter on the same base weights and generate one response.
    from peft import PeftModel

    base = model.get_base_model()
    reloaded = PeftModel.from_pretrained(base, str(adapter))
    reloaded = reloaded.to(device)
    reloaded.eval()
    sample = rows[0]
    prompt = prompt_text(tokenizer, sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(reloaded.device)
    with torch.no_grad():
        generated = reloaded.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
    text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    payload, errors = parse_and_validate(text, sample["user_message"])
    import importlib.util

    report = {
        "stage": "G1",
        "model": describe_model(base),
        "losses": losses,
        "finite": all(value == value and abs(value) != float("inf") for value in losses),
        "trainable_parameters": trainable,
        "reload_trainable_parameters": sum(int(parameter.numel()) for parameter in reloaded.parameters() if parameter.requires_grad),
        "adapter": str(adapter),
        "reload_generation_preview": text[:300],
        "reload_schema_errors": errors,
        "reload_json_valid": payload is not None and not errors,
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else None,
        },
        "kernels": {
            "causal_conv1d_available": importlib.util.find_spec("causal_conv1d") is not None,
            "fla_available": importlib.util.find_spec("fla") is not None,
        },
        "data_hash": json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8")).get("sha256"),
        "note": "G1 smoke only; not a quality or final training result",
    }
    (args.out / "smoke_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"losses": losses, "json_valid": report["reload_json_valid"], "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["finite"] and report["reload_json_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
