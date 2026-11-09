"""Qwen3.5-9B model compatibility boundary for P2-SFT.

Verified against the official Transformers Qwen3.5 documentation (model family
`Qwen3.5`, text-only class `Qwen3_5ForCausalLM`, checkpoint `Qwen/Qwen3.5-9B`).
Qwen3.5 is a hybrid-attention, natively multimodal family; P2-SFT uses the **text-only
CausalLM** class and does not load the vision path.

No model name is silently substituted: if the installed Transformers build does not
expose Qwen3.5, loading fails with an explicit compatibility error.
"""

from __future__ import annotations

import importlib
from typing import Any

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_FAMILY = "Qwen3.5"
MODEL_CLASS_NAME = "Qwen3_5ForCausalLM"
EXPECTED_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
REQUIRED_TRANSFORMERS = ">=5.9.0"


class ModelCompatibilityError(RuntimeError):
    """The installed environment cannot load Qwen3.5-9B as documented."""


def transformers_version() -> str:
    try:
        module = importlib.import_module("transformers")
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def _causal_lm_class() -> Any:
    try:
        module = importlib.import_module("transformers")
        cls = getattr(module, MODEL_CLASS_NAME, None)
        if cls is None:
            raise ModelCompatibilityError(
                f"transformers {transformers_version()} does not expose {MODEL_CLASS_NAME}; "
                f"install transformers{REQUIRED_TRANSFORMERS} or a build documented to support Qwen3.5"
            )
        return cls
    except ImportError as exc:
        raise ModelCompatibilityError(
            f"transformers is not installed; Qwen3.5 requires transformers{REQUIRED_TRANSFORMERS}"
        ) from exc


def load_tokenizer(model_id: str = MODEL_ID, *, trust_remote_code: bool = True) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.eos_token is None:
        raise ModelCompatibilityError("Qwen3.5 tokenizer has no eos token")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ModelCompatibilityError("Qwen3.5 tokenizer has no chat_template")
    return tokenizer


def load_text_model(
    model_id: str = MODEL_ID,
    *,
    dtype: Any = None,
    device_map: str | None = None,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    quantization_config: Any = None,
    use_kernels: bool = False,
) -> Any:
    """Load only the text-only Qwen3.5 CausalLM path."""
    import torch

    cls = _causal_lm_class()
    resolved_dtype = dtype if dtype is not None else torch.bfloat16
    kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if use_kernels:
        kwargs["use_kernels"] = True
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        # `use_kernels` is only on newer builds; retry without it rather than silently
        # changing the model class.
        if "use_kernels" in kwargs:
            kwargs.pop("use_kernels", None)
            return cls.from_pretrained(model_id, **kwargs)
        raise ModelCompatibilityError(f"Qwen3.5 CausalLM load failed: {exc}") from exc


def describe_model(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", ""))
    return {
        "class": type(model).__name__,
        "model_type": model_type,
        "model_family_expected": MODEL_FAMILY,
        "model_id_expected": MODEL_ID,
        "class_expected": MODEL_CLASS_NAME,
        "parameter_count": sum(int(value.numel()) for value in getattr(model, "parameters", lambda: [])() if hasattr(value, "numel")),
        "dtype": str(getattr(model, "dtype", "")),
        "model_type_recognized": model_type in EXPECTED_MODEL_TYPES or type(model).__name__ == MODEL_CLASS_NAME,
    }
