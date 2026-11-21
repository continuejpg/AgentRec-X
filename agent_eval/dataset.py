"""Versioned Golden Dataset for task-level Web Agent evaluation.

This is deliberately separate from the *recommender* evaluation plane
(:mod:`recommendation.evaluation`) and from the older trajectory plane
(:mod:`recommendation.evaluation.agent`).  The unit of evaluation here is a real
Web task: a user query, an optional clarification reply, an expected business outcome,
expected structured requirements, and an optional manual relevance label set.

The dataset file is the source of truth.  ``load_dataset`` only reads and validates it;
the runner never rewrites the Golden cases.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CATEGORIES",
    "GOLDEN_DATASET_SCHEMA",
    "GOLDEN_DATASET_VERSION",
    "DEFAULT_DATASET_PATH",
    "BudgetExpectation",
    "DatasetValidationError",
    "GoldenCase",
    "GoldenDataset",
    "HardConstraints",
    "OutcomeExpectation",
    "RelevanceLabels",
    "RequirementsExpectation",
    "dataset_sha256",
    "load_dataset",
]

GOLDEN_DATASET_SCHEMA = "agentrecx.golden.agent_tasks.v1"
GOLDEN_DATASET_VERSION = "1.0.0"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden" / "golden_agent_tasks_v1.json"
CATEGORIES = frozenset({
    "normal",
    "category_use_case",
    "budget",
    "budget_update",
    "no_price",
    "no_within_budget",
    "cold_resume",
    "replay",
})
ExecutionMode = Literal["warm", "cold_resume", "replay", "direct"]
OutcomeKind = Literal["recommendation", "refusal", "clarification"]


class DatasetValidationError(ValueError):
    """The Golden Dataset is malformed or does not cover the required task classes."""


class BudgetExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_amount: float | None = Field(default=None, ge=0)
    min_amount: float | None = Field(default=None, ge=0)
    currency: str | None = None

    @model_validator(mode="after")
    def _one_bound(self) -> "BudgetExpectation":
        if self.max_amount is None and self.min_amount is None:
            raise ValueError("a budget expectation needs max_amount or min_amount")
        if self.max_amount is not None and self.min_amount is not None:
            raise ValueError("only one budget bound is supported by the current task schema")
        return self

    def as_dict(self) -> dict:
        return {
            "max_amount": self.max_amount,
            "min_amount": self.min_amount,
            "currency": self.currency,
        }


class RequirementsExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    budget: BudgetExpectation | None = None
    category: str | None = None
    use_cases: list[str] = Field(default_factory=list)

    @field_validator("use_cases")
    @classmethod
    def _unique_cases(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("use_cases must contain non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("use_cases must not contain duplicates")
        return value

    def as_dict(self) -> dict:
        return {
            "budget": self.budget.as_dict() if self.budget else None,
            "category": self.category,
            "use_cases": list(self.use_cases),
        }


class OutcomeExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: OutcomeKind
    refusal_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusal_consistency(self) -> "OutcomeExpectation":
        if self.kind == "refusal" and not self.refusal_codes:
            raise ValueError("a refusal expectation needs at least one refusal code")
        if self.kind != "refusal" and self.refusal_codes:
            raise ValueError("refusal_codes are only valid for outcome kind=refusal")
        return self


class HardConstraints(BaseModel):
    """Machine-checkable constraints on a recommended product card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    price_kind: str | None = None
    product_role: str | None = None

    def as_dict(self) -> dict:
        return {
            "max_price": self.max_price,
            "currency": self.currency,
            "price_kind": self.price_kind,
            "product_role": self.product_role,
        }


class RelevanceLabels(BaseModel):
    """Manual binary judgments for every id in the retrieval top-5 pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judged_ids: dict[str, int] = Field(min_length=1)
    note: str = ""

    @field_validator("judged_ids")
    @classmethod
    def _binary_grades(cls, value: dict[str, int]) -> dict[str, int]:
        for item_id, grade in value.items():
            if not str(item_id).strip():
                raise ValueError("relevance id must be non-empty")
            if grade not in (0, 1):
                raise ValueError(
                    "only binary relevance grades are supported; do not invent nDCG grades"
                )
        return value

    def as_dict(self) -> dict:
        return {"judged_ids": dict(self.judged_ids), "note": self.note}


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    dataset_version: str
    category: Literal[
        "normal", "category_use_case", "budget", "budget_update",
        "no_price", "no_within_budget", "cold_resume", "replay",
    ]
    purpose: str = Field(min_length=1)
    user_query: str = Field(min_length=1)
    clarification_reply: str | None = None
    execution_mode: ExecutionMode
    #: Most tasks use the explicit demo historical FX mode.  A no-price case uses strict
    #: mode so "no usable price" is a real configuration, not a fabricated label.
    price_mode: Literal["demo_historical_fx", "strict"] = "demo_historical_fx"
    expected_requirements: RequirementsExpectation
    expected_outcome: OutcomeExpectation
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    acceptable_product_ids: list[str] = Field(default_factory=list)
    relevance: RelevanceLabels | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _mode_and_reply_consistency(self) -> "GoldenCase":
        reply = self.clarification_reply
        if self.execution_mode == "direct" and reply is not None:
            raise ValueError("direct cases must not declare a clarification reply")
        if self.execution_mode in {"warm", "cold_resume", "replay"} and reply is None:
            raise ValueError(
                f"execution_mode={self.execution_mode!r} requires a clarification reply"
            )
        if self.execution_mode == "replay" and self.expected_outcome.kind != "recommendation":
            raise ValueError("the current replay contract only covers completed recommendations")
        if self.expected_outcome.kind == "refusal" and self.acceptable_product_ids:
            raise ValueError("a refusal case cannot pin an acceptable product")
        if self.acceptable_product_ids and len(set(self.acceptable_product_ids)) != len(
            self.acceptable_product_ids
        ):
            raise ValueError("acceptable_product_ids must be unique")
        return self

    @property
    def judged(self) -> bool:
        return self.relevance is not None

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "dataset_version": self.dataset_version,
            "category": self.category,
            "purpose": self.purpose,
            "user_query": self.user_query,
            "clarification_reply": self.clarification_reply,
            "execution_mode": self.execution_mode,
            "price_mode": self.price_mode,
            "expected_requirements": self.expected_requirements.as_dict(),
            "expected_outcome": {
                "kind": self.expected_outcome.kind,
                "refusal_codes": list(self.expected_outcome.refusal_codes),
            },
            "hard_constraints": self.hard_constraints.as_dict(),
            "acceptable_product_ids": list(self.acceptable_product_ids),
            "relevance": self.relevance.as_dict() if self.relevance else None,
            "notes": self.notes,
        }


class GoldenDataset(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True,
    )

    dataset_schema: Literal["agentrecx.golden.agent_tasks.v1"] = Field(alias="schema")
    dataset_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    description: str = Field(min_length=1)
    labeling_policy: str = Field(min_length=1)
    catalog: dict = Field(default_factory=dict)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_collection(self) -> "GoldenDataset":
        if not 15 <= len(self.cases) <= 20:
            raise ValueError(
                f"the Golden Dataset must contain 15-20 representative tasks, got {len(self.cases)}"
            )
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        missing = CATEGORIES - {case.category for case in self.cases}
        if missing:
            raise ValueError(f"dataset does not cover required categories: {sorted(missing)}")
        for case in self.cases:
            if case.dataset_version != self.dataset_version:
                raise ValueError(
                    f"case {case.case_id!r} has dataset_version={case.dataset_version!r}, "
                    f"expected {self.dataset_version!r}"
                )
        return self

    def case_by_id(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def as_dict(self) -> dict:
        return {
            "schema": self.dataset_schema,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "description": self.description,
            "labeling_policy": self.labeling_policy,
            "catalog": dict(self.catalog),
            "cases": [case.as_dict() for case in self.cases],
        }


def load_dataset(path: Path | str | None = None) -> GoldenDataset:
    """Load and fully validate a Golden Dataset; never silently repair it."""
    source = Path(path) if path is not None else DEFAULT_DATASET_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"Golden Dataset not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Golden Dataset is not valid JSON: {exc}") from exc
    try:
        dataset = GoldenDataset.model_validate(raw)
    except Exception as exc:  # pydantic error; expose a loader-level failure
        raise DatasetValidationError(f"Golden Dataset validation failed: {exc}") from exc
    if dataset.dataset_schema != GOLDEN_DATASET_SCHEMA:
        raise DatasetValidationError(
            f"unsupported Golden Dataset schema {dataset.dataset_schema!r}"
        )
    return dataset


def dataset_sha256(path: Path | str | None = None) -> str:
    source = Path(path) if path is not None else DEFAULT_DATASET_PATH
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
