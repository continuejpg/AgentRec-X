"""Chat-template and completion-only formatting helpers for P2-SFT.

These helpers are intentionally model-agnostic and import no GPU/TRL package, so the
local CPU contract tests can verify prompt construction and label masks before AutoDL.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .schema import SCHEMA_VERSION, parse_and_validate

SYSTEM_PROMPT = """You are a shopping-preference operation parser for AgentRec-X.
Return ONLY one compact JSON object matching schema_version {schema}.
Do not output product ASINs, URLs, explanations, or code fences.
The model proposes operations; M5 lifecycle and R1 remain the authority.
Every operation evidence must be an exact substring of the current user message.

The object must have exactly this shape:
{{"schema_version":"{schema}","operations":[{{"op":"update_constraint","attribute":"price_max","value":30,"direction":null,"scope":"task","evidence":"under $30"}}],"clarification":null}}

Every operation MUST use exactly these keys:
op, attribute, value, direction, scope, evidence
(and optionally reference, reason where applicable).
Allowed op values: set_goal, update_goal, add_preference, update_preference,
retract_preference, set_constraint, update_constraint, retract_constraint,
resolve_reference, clarify, noop, abstain.
Do not rename fields (never use operation, constraint_field or new_value).
For an ambiguous goal use one clarify operation plus a clarification object.
For no change use one noop operation with a known reason; for unsupported requests
use one abstain operation. Do not invent user confirmation.
""".format(schema=SCHEMA_VERSION)


def user_prompt(row: Mapping[str, Any]) -> str:
    state = json.dumps(row.get("state", {}), ensure_ascii=False, sort_keys=True)
    context = json.dumps(row.get("catalogue_context", {}), ensure_ascii=False, sort_keys=True)
    return (
        f"Current effective state: {state}\n"
        f"Catalogue context (no identity): {context}\n"
        f"Current user message: {row['user_message']}\n"
        "Produce the preference-operation proposal."
    )


def messages(row: Mapping[str, Any], *, include_assistant: bool = False) -> list[dict[str, str]]:
    result = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(row)},
    ]
    if include_assistant:
        result.append({"role": "assistant", "content": str(row["assistant"])})
    return result


def _apply_chat_template(tokenizer: Any, msgs: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return str(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )
    except TypeError as exc:
        # Do not silently fall back to the model's default (possibly thinking) template.
        raise RuntimeError(
            "Qwen3.5 tokenizer chat template does not accept enable_thinking=False; "
            "the G0 preflight must resolve this before training"
        ) from exc


def prompt_text(tokenizer: Any, row: Mapping[str, Any]) -> str:
    return _apply_chat_template(tokenizer, messages(row), add_generation_prompt=True)


def training_text(tokenizer: Any, row: Mapping[str, Any], eos_token: str | None = None) -> tuple[str, str]:
    prompt = prompt_text(tokenizer, row)
    assistant = str(row["assistant"])
    suffix = eos_token if eos_token is not None else ""
    return prompt, prompt + assistant + suffix


def completion_only_encoding(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    eos_token: str | None = None,
    max_seq_length: int | None = None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Tokenize prompt+completion and mask every prompt token with -100."""
    prompt, full = training_text(tokenizer, row, eos_token=eos_token)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prefix is not tokenizer-stable")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    info = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(full_ids) - len(prompt_ids),
        "total_tokens": len(full_ids),
        "truncated": bool(max_seq_length is not None and len(full_ids) > max_seq_length),
    }
    if max_seq_length is not None:
        prompt_ids = prompt_ids[:max_seq_length]
        full_ids = full_ids[:max_seq_length]
        labels = labels[:max_seq_length]
    return full_ids, labels, info


def validate_dataset_row(row: Mapping[str, Any]) -> list[str]:
    """CPU-check one prepared sample: schema, evidence, no ASIN, no target leak."""
    errors: list[str] = []
    for key in ("sample_id", "split", "scenario", "state", "catalogue_context", "user_message", "assistant"):
        if key not in row:
            errors.append(f"missing {key}")
    if errors:
        return errors
    try:
        assistant = str(row["assistant"])
        payload = json.loads(assistant)
    except (TypeError, ValueError) as exc:
        return [f"assistant is not JSON: {type(exc).__name__}"]
    from .schema import validate_proposal

    errors.extend(validate_proposal(payload, str(row["user_message"])))
    rendered = json.dumps(row, ensure_ascii=False)
    if SCHEMA_VERSION not in assistant:
        errors.append("assistant does not declare the versioned schema")
    return errors
