"""Versioned query set for the M3-C retrieval experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "QUERY_DATASET_SCHEMA",
    "DEFAULT_QUERIES_PATH",
    "QuerySet",
    "RetrievalQuery",
    "load_queries",
    "queries_sha256",
]

QUERY_DATASET_SCHEMA = "agentrecx.retrieval_eval.queries.v1"
DEFAULT_QUERIES_PATH = Path(__file__).resolve().parent / "queries_v1.json"


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    query: str = Field(min_length=1)
    expected_category: Literal["laptop", "headphones", "tablet", "mouse", "keyboard"]
    expected_use_cases: list[str] = Field(default_factory=list)
    expected_features: list[str] = Field(default_factory=list)
    purpose: str = Field(min_length=1)

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expected_category": self.expected_category,
            "expected_use_cases": list(self.expected_use_cases),
            "expected_features": list(self.expected_features),
            "purpose": self.purpose,
        }


class QuerySet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    query_schema: Literal["agentrecx.retrieval_eval.queries.v1"] = Field(alias="schema")
    dataset_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    description: str = Field(min_length=1)
    queries: list[RetrievalQuery] = Field(min_length=8, max_length=10)

    @model_validator(mode="after")
    def _unique_and_cover(self) -> "QuerySet":
        ids = [query.query_id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval query_id values must be unique")
        categories = {query.expected_category for query in self.queries}
        missing = {"laptop", "headphones", "tablet"} - categories
        if missing:
            raise ValueError(f"query set must cover {sorted(missing)}")
        return self

    def as_dict(self) -> dict:
        return {
            "schema": self.query_schema,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "description": self.description,
            "queries": [query.as_dict() for query in self.queries],
        }


def load_queries(path: Path | str | None = None) -> QuerySet:
    source = Path(path) if path is not None else DEFAULT_QUERIES_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        return QuerySet.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - visible loader failure
        raise ValueError(f"retrieval query set validation failed: {exc}") from exc


def queries_sha256(path: Path | str | None = None) -> str:
    source = Path(path) if path is not None else DEFAULT_QUERIES_PATH
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return digest
