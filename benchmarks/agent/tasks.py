"""Task loading and real-catalogue validation.

The validator proves that every expected product exists in the served Sports & Outdoors
catalogue, that its title/category evidence supports the assertion, and that expected
answers are not present in the policy-visible user text.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import BenchTask, BenchTurn, ExpectedOutcome

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
MANIFEST_NAME = "manifest.json"
_ASIN = re.compile(r"\bB[0-9A-Z]{9}\b")
_ALLOWED_ACTIONS = {"RERANK", "RETRIEVE"}
_ALLOWED_ATTRIBUTES = {
    "weight",
    "price",
    "price_max",
    "price_min",
    "color",
    "material",
    "brand",
    "feature",
    "category",
}


class TaskValidationError(ValueError):
    """A benchmark task is malformed or unsupported by the real catalogue."""


def _record(records: Mapping[str, Any], identity: str) -> Any:
    record = records.get(str(identity))
    if record is None:
        raise TaskValidationError(f"identity {identity!r} is not in the served catalogue")
    return record


def _title(record: Any) -> str:
    return " ".join(str(getattr(record, "title", "") or "").lower().split())


def _categories(record: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in (getattr(record, "categories", ()) or ()))


def task_from_mapping(raw: Mapping[str, Any]) -> BenchTask:
    """Parse one JSON object into the typed task representation."""
    if not isinstance(raw, Mapping):
        raise TaskValidationError("task must be a JSON object")
    expected_raw = raw.get("expected") or {}
    if not isinstance(expected_raw, Mapping):
        raise TaskValidationError("expected must be an object")
    turns_raw = raw.get("turns") or []
    if not isinstance(turns_raw, list):
        raise TaskValidationError("turns must be a list")
    turns: list[BenchTurn] = []
    for item in turns_raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            text = str(item.get("user_input", ""))
        else:
            raise TaskValidationError("each turn must be a string or object")
        turns.append(BenchTurn(user_input=text))
    expected = ExpectedOutcome(
        final_any=tuple(str(value) for value in expected_raw.get("final_any", ())),
        final_min=int(expected_raw.get("final_min", 1)),
        required_title_terms_any=tuple(
            str(value).lower() for value in expected_raw.get("required_title_terms_any", ())
        ),
        must_exclude_title_terms=tuple(
            str(value).lower() for value in expected_raw.get("must_exclude_title_terms", ())
        ),
        must_exclude_ids=tuple(str(value) for value in expected_raw.get("must_exclude_ids", ())),
        active_preference_attributes=tuple(
            str(value).lower()
            for value in expected_raw.get("active_preference_attributes", ())
        ),
        actions_any=tuple(str(value).upper() for value in expected_raw.get("actions_any", ())),
    )
    return BenchTask(
        task_id=str(raw.get("task_id", "")),
        split=str(raw.get("split", "")),
        scenario=str(raw.get("scenario", "")),
        history=tuple(str(value) for value in raw.get("history", ())),
        turns=tuple(turns),
        expected=expected,
        preference_discovery=bool(raw.get("preference_discovery", False)),
    )


def load_tasks(split: str, *, tasks_dir: Path | None = None) -> tuple[BenchTask, ...]:
    """Load one split from ``benchmarks/agent/tasks/<split>.json``."""
    directory = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    path = directory / f"{split}.json"
    if not path.is_file():
        raise TaskValidationError(f"missing task split: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TaskValidationError(f"{path} must contain a JSON list")
    return tuple(task_from_mapping(value) for value in payload)


def _validate_task(task: BenchTask, records: Mapping[str, Any], expected_split: str) -> None:
    prefix = f"{task.task_id or '<missing-id>'}: "
    if not task.task_id:
        raise TaskValidationError(prefix + "task_id is required")
    if task.split != expected_split:
        raise TaskValidationError(prefix + f"split {task.split!r} does not match file split")
    if task.scenario not in {
        "goal_correction_withdraw",
        "accessory_then_restore",
        "accessory_target",
        "goal_change_to_filter",
        "budget_conflict",
        "cookware_material",
        "cookware_lightweight_pref",
        "filter_lightweight_withdraw",
        "ambiguous_to_cookware",
        "cross_turn_reference",
        "filter_bottle_role",
        "goal_change_sleeping_to_lantern",
        "goal_change_tablets_to_filter",
        "gravity_filter_pref",
        "filter_straw_pref",
        "cookware_to_stove",
        "filter_budget_pref",
        "filter_color_correction",
        "cookware_set_reference",
        "broad_gear_to_filter",
        "frozen_filter_correction",
        "frozen_cookware_to_filter",
        "frozen_filter_accessory",
        "frozen_pad_to_lantern",
        "frozen_budget_conflict",
        "frozen_filter_pref",
        "frozen_ambiguous_cookware",
        "frozen_bottle_to_filter",
    }:
        raise TaskValidationError(prefix + f"unknown scenario {task.scenario!r}")
    if not 3 <= len(task.turns) <= 5:
        raise TaskValidationError(prefix + "tasks must have 3-5 user turns")
    if not task.history:
        raise TaskValidationError(prefix + "trusted history must not be empty")
    for identity in task.history:
        _record(records, identity)
    expected_ids = set(task.expected.final_any) | set(task.expected.must_exclude_ids)
    for identity in expected_ids:
        record = _record(records, identity)
        limit = max(1, len(expected_ids))
        if not _title(record) and not _categories(record):
            raise TaskValidationError(prefix + f"{identity} has no trusted title/category")
        del limit
    if not task.expected.final_any:
        raise TaskValidationError(prefix + "final_any must contain at least one real product")
    if not 1 <= task.expected.final_min <= len(task.expected.final_any):
        raise TaskValidationError(prefix + "final_min must be in [1, len(final_any)]")
    if not task.expected.required_title_terms_any:
        raise TaskValidationError(prefix + "required_title_terms_any is required")
    expected_titles = [
        _title(_record(records, identity)) for identity in task.expected.final_any
    ]
    evidence = any(
        term in title
        for term in task.expected.required_title_terms_any
        for title in expected_titles
    )
    if not evidence:
        raise TaskValidationError(
            prefix + "no expected final product has the asserted catalogue title evidence"
        )
    bad_actions = set(task.expected.actions_any) - _ALLOWED_ACTIONS
    if bad_actions:
        raise TaskValidationError(prefix + f"unknown actions: {sorted(bad_actions)}")
    bad_attributes = set(task.expected.active_preference_attributes) - _ALLOWED_ATTRIBUTES
    if bad_attributes:
        raise TaskValidationError(prefix + f"unknown attributes: {sorted(bad_attributes)}")
    for turn in task.turns:
        text = turn.user_input
        if not text.strip():
            raise TaskValidationError(prefix + "user turns must be non-empty")
        if _ASIN.search(text):
            raise TaskValidationError(prefix + "user input must not contain ASINs")
        for identity in expected_ids:
            if identity.casefold() in text.casefold():
                raise TaskValidationError(prefix + "user input leaks an expected product id")


def validate_tasks(
    tasks: Sequence[BenchTask],
    records: Mapping[str, Any],
    *,
    expected_split: str,
    expected_counts: Mapping[str, int] | None = None,
) -> None:
    """Validate every task against the real catalogue; raise on the first failure."""
    if not tasks:
        raise TaskValidationError("task split is empty")
    seen: set[str] = set()
    for task in tasks:
        _validate_task(task, records, expected_split)
        if task.task_id in seen:
            raise TaskValidationError(f"duplicate task_id {task.task_id!r}")
        seen.add(task.task_id)
    if expected_counts and expected_split in expected_counts:
        wanted = int(expected_counts[expected_split])
        if len(tasks) != wanted:
            raise TaskValidationError(
                f"{expected_split} expected {wanted} tasks, got {len(tasks)}"
            )


def verify_manifest(tasks_dir: Path | None = None) -> dict[str, str]:
    """Check the frozen task files against their recorded SHA-256 hashes."""
    directory = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise TaskValidationError(f"missing task manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    if not isinstance(files, Mapping):
        raise TaskValidationError("manifest files must be an object")
    verified: dict[str, str] = {}
    for name, expected in files.items():
        path = directory / str(name)
        if not path.is_file():
            raise TaskValidationError(f"manifest names missing file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected):
            raise TaskValidationError(
                f"hash mismatch for {name}: expected {expected}, got {actual}"
            )
        verified[str(name)] = actual
    return verified


def write_manifest(tasks_dir: Path | None = None, *, extra: Mapping[str, Any] | None = None) -> Path:
    """Regenerate the hash manifest deliberately (not invoked by the runner)."""
    directory = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    files = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload: dict[str, Any] = {"files": files}
    if extra:
        payload.update(dict(extra))
    destination = directory / MANIFEST_NAME
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def iter_policy_inputs(tasks: Iterable[BenchTask]) -> Iterable[Mapping[str, Any]]:
    """Convenience for auditing that adapters never receive expected answers."""
    return (task.policy_input for task in tasks)
