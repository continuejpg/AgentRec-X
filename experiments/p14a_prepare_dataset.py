#!/usr/bin/env python
"""P14-A dataset preparation: official Shopping Companion train split -> agent/eval JSONL.

The released Shopping Companion benchmark (config ``benchmark``, split ``train``) ships
1,600 *stage-level* instances: 800 shopping tasks x two stages, where ``stage_1`` is
preference recovery and ``stage_2`` is product recommendation.  This converter reads the
already-downloaded ``train.parquet`` and writes three artifacts:

``train_tasks.jsonl`` (AGENT-VISIBLE, 800 rows)
    One row per official task (``question_id``), joining the two stage instances.  Schema
    tag ``agentrecx.p14a.shopping_task.v1``.  The official ``stage_2`` prompt has four
    messages::

        [system, user <question>, user <oracle-memory block>, user <final instruction>]

    The oracle-memory block (``"The most relevant user dialogue memories are as follows:
    ..."``) is ground-truth memory derived from the task's answer sessions.  It is REMOVED
    from the agent-visible input; the row keeps ``system``, ``user`` (the question again)
    and ``final_user``.  Nothing in this file may reveal the answer.

``train_eval.jsonl`` (EVALUATOR-ONLY, 800 rows)
    Schema tag ``agentrecx.p14a.eval.v1``: the parsed ``ground_truth`` object, the exact
    removed oracle-memory block, its SHA-256, and the SHA-256 of the original four-message
    stage_2 prompt (canonical JSON) so the removal is auditable/reversible.  This file is
    for the independent post-run evaluator only and must never be passed to the agent
    runtime.

``manifest.json``
    Source hash, row counts, pairing validation, oracle-isolation evidence, output hashes,
    timestamps and the exact command.  The converter re-opens what it wrote and FAILS
    (non-zero exit) if a forbidden key/substring or a ground-truth value leaks into the
    agent-visible file.

Isolation scan and the official system prompts
----------------------------------------------
The agent-visible scan bans the substrings ``product_id``, ``ground_truth``,
``reward_model``, the oracle-memory marker and the eval-only field names.  The *official*
stage_2 system prompt itself contains the literal token ``product_id`` four times, because
it documents the answer format (``@REC::product_id@`` for a single product,
``@REC::product_id1,product_id2,...@`` for a bundle).  That text is static official
instruction boilerplate, not a task answer, so the converter keeps both system prompts
byte-identical to the release (their SHA-256 are recorded) and requires every forbidden
substring occurrence to be *inside* those two exact prompt strings.  Outside them the
count must be zero.  Scrubbing the release prompt instead would silently change the task
instructions, so it is deliberately not done.

Requirements / usage
--------------------
``pyarrow`` is needed only here, in the tooling environment; the runtime adapter
(``recommendation/shopping_companion/dataset.py``) is stdlib-only.

    PYTHONPATH=...pyarrow-libs python experiments/p14a_prepare_dataset.py \\
        --parquet C:\\...\\p14a_scratch\\train.parquet \\
        --out-dir data/p14a/dataset

``test.parquet`` is never opened (passing it is refused explicitly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Schema tags of the two output files (frozen by the P14-A dataset contract).
TASKS_SCHEMA = "agentrecx.p14a.shopping_task.v1"
EVAL_SCHEMA = "agentrecx.p14a.eval.v1"
MANIFEST_SCHEMA = "agentrecx.p14a.prepare_manifest.v1"

#: First characters of the official stage_2 oracle-memory message.
ORACLE_MEMORY_MARKER = "The most relevant user dialogue memories are as follows:"

#: Constant final stage_2 user message of the official release.
FINAL_USER_MESSAGE = "I will not provide you with any more information. Please complete the task."

#: ``Current Date: 2025/09/25 (Thu)`` prefix of the official question.
QUESTION_DATE_RE = re.compile(r"Current Date:\s*(\d{4}/\d{2}/\d{2})\s*\(([A-Za-z]{3})\)")
QUESTION_DATE_ONLY_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
QUESTION_ID_RE = re.compile(r"^([0-9a-f]{32})_(\d+)$")

#: Tool allow-lists frozen by the P14-A dataset contract (per stage).
ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    "stage_1": ("mem_search", "mem_view", "mem_summarize_by_date"),
    "stage_2": ("product_search", "product_view"),
}

#: Exact key order of an agent-visible task row (schema ``agentrecx.p14a.shopping_task.v1``).
TASK_FIELDS: tuple[str, ...] = (
    "schema",
    "task_id",
    "conversation_id",
    "question_id",
    "question",
    "question_date",
    "question_type",
    "split",
    "source_index",
    "stage_1",
    "stage_2",
    "allowed_tools",
    "tools_kwargs",
)
STAGE_1_FIELDS: tuple[str, ...] = ("system", "user")
STAGE_2_FIELDS: tuple[str, ...] = (
    "system",
    "user",
    "final_user",
    "oracle_memory_removed",
    "oracle_chars_removed",
)

#: Exact key order of an evaluator-only row (schema ``agentrecx.p14a.eval.v1``).
EVAL_FIELDS: tuple[str, ...] = (
    "schema",
    "task_id",
    "conversation_id",
    "ground_truth",
    "oracle_memory_block",
    "oracle_memory_block_sha256",
    "stage_2_original_prompt_sha256",
)

#: Substrings that must not appear in agent-visible content outside the official prompts.
FORBIDDEN_AGENT_SUBSTRINGS: tuple[str, ...] = (
    ORACLE_MEMORY_MARKER,
    "ground_truth",
    "reward_model",
    "product_id",
    "oracle_memory_block",
    "stage_2_original_prompt_sha256",
    "answer_session_ids",
)

#: JSON keys that must never appear anywhere in an agent-visible row.
FORBIDDEN_AGENT_KEYS: frozenset[str] = frozenset(
    {
        "ground_truth",
        "reward_model",
        "oracle_memory_block",
        "oracle_memory_block_sha256",
        "stage_2_original_prompt_sha256",
        "product_id",
        "answer_session_ids",
        "dialogue",
    }
)

#: Markers of a real dialogue session (oracle-memory body) in the official release.
#: The second/third entries are the literal JSON-escaped forms, because every dialogue
#: line would appear inside a JSON string in the output files.
DIALOGUE_SESSION_MARKERS: tuple[str, ...] = ("[Date: ", "\\nuser: ", "\\nassistant: ")

#: Question types emitted by the release.
QUESTION_TYPES: frozenset[str] = frozenset({"single_product", "add_on_deals"})

_JSON_SEPARATORS = (",", ":")


class ConversionError(RuntimeError):
    """The released parquet does not satisfy the P14-A conversion contract."""


# --------------------------------------------------------------------------- #
# small stable helpers
# --------------------------------------------------------------------------- #


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for prompt hashing (sorted keys, no extra spaces)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=_JSON_SEPARATORS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def _require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), f"{name} must be a string, got {type(value).__name__}")
    _require(allow_empty or bool(value.strip()), f"{name} must not be empty")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer, got {type(value).__name__}",
    )
    if minimum is not None:
        _require(value >= minimum, f"{name} must be >= {minimum}, got {value}")
    return value


def _json_line(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=_JSON_SEPARATORS)


def _display_path(path: Path) -> str:
    """Stable POSIX-style path string for the committed manifest."""
    return path.as_posix()


def _iter_keys(value: Any) -> list[str]:
    """Recursively collect every JSON object key in ``value``."""
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_iter_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_iter_keys(child))
    return keys


def _count_marker_outside_prompts(
    line: str,
    marker: str,
    system_values: Sequence[str],
) -> tuple[int, int]:
    """Return ``(inside_official_prompts, outside_official_prompts)`` occurrences.

    The two system prompts are JSON-encoded inside the line; the forbidden markers used
    here are plain ASCII and cannot straddle a JSON escape, so counting on the raw line is
    exact for them.
    """
    inside = sum(text.count(marker) for text in system_values)
    total = line.count(marker)
    return inside, total - inside


# --------------------------------------------------------------------------- #
# parquet -> paired stage rows
# --------------------------------------------------------------------------- #


def _refuse_test_split(parquet_path: Path) -> None:
    _require(
        parquet_path.name.lower() != "test.parquet",
        "refusing to convert 'test.parquet': the test split is reserved and must never be touched",
    )


def load_parquet_rows(parquet_path: Path) -> list[dict[str, Any]]:
    """Read the official train parquet into python dictionaries (pyarrow, tooling only)."""
    _refuse_test_split(parquet_path)
    _require(parquet_path.is_file(), f"parquet file does not exist: {parquet_path}")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the tooling environment
        raise ConversionError(
            "pyarrow is required to read the parquet; run this converter with the tooling "
            "interpreter (the runtime adapter has no pyarrow dependency)"
        ) from exc
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:  # noqa: BLE001 - pyarrow raises many exception types
        raise ConversionError(f"cannot read parquet {parquet_path}: {exc}") from exc
    rows = table.to_pylist()
    _require(isinstance(rows, list) and rows, f"parquet {parquet_path} contains no rows")
    return rows


def _validate_stage_row(row: Mapping[str, Any], position: int) -> None:
    """Validate the released row shape that this converter depends on."""
    where = f"parquet row {position} (1-based {position + 1})"
    _require(isinstance(row, Mapping), f"{where}: row must be a mapping")
    _require(set(row) == {"data_source", "agent_name", "prompt", "ability", "reward_model", "extra_info"},
             f"{where}: unexpected column set {sorted(row)}")
    _require(row["data_source"] == "shopping_companion", f"{where}: unexpected data_source {row['data_source']!r}")
    _require(row["agent_name"] == "tool_agent", f"{where}: unexpected agent_name {row['agent_name']!r}")
    ability = row["ability"]
    _require(ability in {"stage_1", "stage_2"}, f"{where}: unexpected ability {ability!r}")

    prompt = row["prompt"]
    _require(isinstance(prompt, list), f"{where}: prompt must be a list")
    expected_roles = ("system", "user") if ability == "stage_1" else ("system", "user", "user", "user")
    roles = []
    for message in prompt:
        _require(isinstance(message, Mapping), f"{where}: prompt messages must be mappings")
        _require(set(message) == {"content", "role"}, f"{where}: unexpected prompt message keys {sorted(message)}")
        roles.append(message["role"])
        _require_str(message["content"], f"{where}: prompt content")
    _require(tuple(roles) == expected_roles,
             f"{where}: {ability} prompt roles {roles} != expected {list(expected_roles)}")

    reward = row["reward_model"]
    _require(isinstance(reward, Mapping) and set(reward) == {"ground_truth", "style"},
             f"{where}: unexpected reward_model shape")
    _require_str(reward["ground_truth"], f"{where}: reward_model.ground_truth")
    _require_str(reward["style"], f"{where}: reward_model.style")

    extra = row["extra_info"]
    _require(isinstance(extra, Mapping), f"{where}: extra_info must be a mapping")
    required = {"ability", "conversation_id", "index", "question", "question_id", "question_type", "split", "tools_kwargs"}
    _require(required <= set(extra), f"{where}: extra_info misses {sorted(required - set(extra))}")
    _require(extra["ability"] == ability, f"{where}: extra_info.ability {extra['ability']!r} != {ability!r}")
    _require_str(extra["question_id"], f"{where}: extra_info.question_id")
    _require_str(extra["conversation_id"], f"{where}: extra_info.conversation_id")
    _require_str(extra["question"], f"{where}: extra_info.question")
    _require(extra["question_type"] in QUESTION_TYPES,
             f"{where}: unexpected question_type {extra['question_type']!r}")
    _require(extra["split"] == "train", f"{where}: split must be 'train', got {extra['split']!r}")
    _require_int(extra["index"], f"{where}: extra_info.index", minimum=0)
    _require(isinstance(extra["tools_kwargs"], Mapping) and extra["tools_kwargs"],
             f"{where}: extra_info.tools_kwargs must be a non-empty mapping")


def pair_stage_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], Counter]:
    """Group released rows by ``question_id`` and require exactly one stage_1 + one stage_2."""
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    stage_counts: Counter = Counter()
    for position, row in enumerate(rows):
        _validate_stage_row(row, position)
        question_id = row["extra_info"]["question_id"]
        ability = row["ability"]
        _require(ability not in by_question[question_id],
                 f"question {question_id!r} has more than one {ability} row")
        by_question[question_id][ability] = dict(row)
        stage_counts[ability] += 1
    _require(all(set(pair) == {"stage_1", "stage_2"} for pair in by_question.values()),
             "the release is not fully paired: every question_id must have one stage_1 and one stage_2 row")
    return dict(by_question), stage_counts


# --------------------------------------------------------------------------- #
# record builders
# --------------------------------------------------------------------------- #


def extract_question_date(question: str, task_id: str) -> str:
    """Extract ``YYYY/MM/DD`` from the official ``Current Date: ...`` question prefix."""
    match = QUESTION_DATE_RE.search(question)
    _require(match is not None, f"task {task_id!r}: question has no 'Current Date: YYYY/MM/DD (Day)' prefix")
    date = match.group(1)
    _require(QUESTION_DATE_ONLY_RE.match(date) is not None, f"task {task_id!r}: unparsable question date {date!r}")
    return date


def _stage_system_prompt(row: Mapping[str, Any], ability: str) -> str:
    return row["prompt"][0]["content"]


def _assert_no_oracle_text(value: str, where: str) -> None:
    """Fail when a retained agent-visible field carries the oracle marker or an answer brand."""
    for marker in (ORACLE_MEMORY_MARKER, "ground_truth", "reward_model"):
        _require(marker not in value, f"{where}: retained agent-visible text contains forbidden marker {marker!r}")


def build_task_record(question_id: str, pair: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build one agent-visible task row from the two released stage rows."""
    stage_1 = pair["stage_1"]
    stage_2 = pair["stage_2"]
    extra_1 = stage_1["extra_info"]
    extra_2 = stage_2["extra_info"]

    _require(extra_1["question_id"] == question_id == extra_2["question_id"],
             f"task {question_id!r}: question_id is inconsistent across stages")
    match = QUESTION_ID_RE.match(question_id)
    _require(match is not None, f"task {question_id!r}: question_id is not '<conversation_id>_<n>'")
    conversation_id = _require_str(extra_1["conversation_id"], f"task {question_id!r}: conversation_id")
    _require(conversation_id == extra_2["conversation_id"],
             f"task {question_id!r}: conversation_id differs across stages")
    _require(match.group(1) == conversation_id,
             f"task {question_id!r}: question_id prefix does not match conversation_id")
    _require(extra_1["question_type"] == extra_2["question_type"],
             f"task {question_id!r}: question_type differs across stages")
    _require(extra_1["index"] == extra_2["index"],
             f"task {question_id!r}: source index differs across stages")
    _require(extra_1["split"] == extra_2["split"] == "train", f"task {question_id!r}: expected split 'train'")
    _require(extra_1["tools_kwargs"] == extra_2["tools_kwargs"],
             f"task {question_id!r}: tools_kwargs differ across stages")

    question = _require_str(extra_1["question"], f"task {question_id!r}: question")
    _require(question == extra_2["question"], f"task {question_id!r}: question differs across stages")
    question_date = extract_question_date(question, question_id)

    prompt_1 = stage_1["prompt"]
    prompt_2 = stage_2["prompt"]
    system_1 = _require_str(prompt_1[0]["content"], f"task {question_id!r}: stage_1 system prompt")
    system_2 = _require_str(prompt_2[0]["content"], f"task {question_id!r}: stage_2 system prompt")
    user_1 = _require_str(prompt_1[1]["content"], f"task {question_id!r}: stage_1 user message")
    user_2 = _require_str(prompt_2[1]["content"], f"task {question_id!r}: stage_2 question message")
    _require(user_1 == question, f"task {question_id!r}: stage_1 user message is not the official question")
    _require(user_2 == question, f"task {question_id!r}: stage_2 first user message is not the official question")

    oracle_block = _require_str(prompt_2[2]["content"], f"task {question_id!r}: stage_2 oracle-memory block")
    _require(oracle_block.startswith(ORACLE_MEMORY_MARKER),
             f"task {question_id!r}: stage_2 message 2 is not the oracle-memory block")
    final_user = _require_str(prompt_2[3]["content"], f"task {question_id!r}: stage_2 final user message")
    _require(final_user == FINAL_USER_MESSAGE,
             f"task {question_id!r}: stage_2 final user message differs from the official constant")

    # The oracle-memory block must be isolated to the evaluator file and to nothing else.
    for name, value in (
        ("stage_1 system", system_1),
        ("stage_2 system", system_2),
        ("stage_1 user", user_1),
        ("stage_2 user", user_2),
        ("stage_2 final_user", final_user),
        ("question", question),
    ):
        _assert_no_oracle_text(value, f"task {question_id!r} {name}")
        _require(oracle_block not in value, f"task {question_id!r}: oracle-memory block appears in {name}")

    return {
        "schema": TASKS_SCHEMA,
        "task_id": question_id,
        "conversation_id": conversation_id,
        "question_id": question_id,
        "question": question,
        "question_date": question_date,
        "question_type": extra_1["question_type"],
        "split": "train",
        "source_index": int(extra_1["index"]),
        "stage_1": {
            "system": system_1,
            "user": user_1,
        },
        "stage_2": {
            "system": system_2,
            "user": user_2,
            "final_user": final_user,
            "oracle_memory_removed": True,
            "oracle_chars_removed": len(oracle_block),
        },
        "allowed_tools": {stage: list(tools) for stage, tools in ALLOWED_TOOLS.items()},
        "tools_kwargs": dict(extra_1["tools_kwargs"]),
    }


def build_eval_record(question_id: str, pair: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build one evaluator-only row (ground truth + exact removed oracle block)."""
    stage_1 = pair["stage_1"]
    stage_2 = pair["stage_2"]
    extra_1 = stage_1["extra_info"]
    oracle_block = stage_2["prompt"][2]["content"]
    original_prompt = [
        {"role": message["role"], "content": message["content"]}
        for message in stage_2["prompt"]
    ]

    ground_truth_text = _require_str(
        stage_2["reward_model"]["ground_truth"], f"task {question_id!r}: reward_model.ground_truth"
    )
    try:
        ground_truth = json.loads(ground_truth_text)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"task {question_id!r}: ground_truth is not valid JSON: {exc}") from exc
    _require(isinstance(ground_truth, dict) and ground_truth,
             f"task {question_id!r}: parsed ground_truth must be a non-empty object")
    _require(
        ground_truth_text == stage_1["reward_model"]["ground_truth"],
        f"task {question_id!r}: ground_truth differs across stages",
    )

    return {
        "schema": EVAL_SCHEMA,
        "task_id": question_id,
        "conversation_id": extra_1["conversation_id"],
        "ground_truth": ground_truth,
        "oracle_memory_block": oracle_block,
        "oracle_memory_block_sha256": sha256_text(oracle_block),
        "stage_2_original_prompt_sha256": sha256_text(canonical_json(original_prompt)),
    }


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write compact UTF-8 JSONL and return its hash/size facts."""
    payload = "\n".join(_json_line(row) for row in rows) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    raw = path.read_bytes()
    return {"path": _display_path(path), "rows": len(rows), "bytes": len(raw), "sha256": sha256_bytes(raw)}


# --------------------------------------------------------------------------- #
# post-write isolation verification (the converter FAILS here on any leak)
# --------------------------------------------------------------------------- #


def _reopen_tasks(tasks_path: Path, expected: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    lines = tasks_path.read_text(encoding="utf-8").splitlines()
    _require(len(lines) == len(expected), f"{tasks_path}: expected {len(expected)} lines, found {len(lines)}")
    reloaded: list[dict[str, Any]] = []
    for line_number, (line, expected_row) in enumerate(zip(lines, expected), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"{tasks_path} line {line_number}: invalid JSON after writing: {exc}") from exc
        _require(isinstance(row, dict), f"{tasks_path} line {line_number}: row is not a JSON object")
        _require(
            tuple(row.keys()) == TASK_FIELDS,
            f"{tasks_path} line {line_number}: top-level keys {list(row.keys())} != {list(TASK_FIELDS)}",
        )
        _require(tuple(row["stage_1"].keys()) == STAGE_1_FIELDS,
                 f"{tasks_path} line {line_number}: stage_1 keys {list(row['stage_1'])} != {list(STAGE_1_FIELDS)}")
        _require(tuple(row["stage_2"].keys()) == STAGE_2_FIELDS,
                 f"{tasks_path} line {line_number}: stage_2 keys {list(row['stage_2'])} != {list(STAGE_2_FIELDS)}")
        _require(row["schema"] == TASKS_SCHEMA, f"{tasks_path} line {line_number}: wrong schema tag")
        _require(row["stage_2"]["oracle_memory_removed"] is True,
                 f"{tasks_path} line {line_number}: oracle_memory_removed must be true")
        _require_int(row["stage_2"]["oracle_chars_removed"],
                     f"{tasks_path} line {line_number}: oracle_chars_removed", minimum=1)
        if tuple(row.keys()) != tuple(expected_row.keys()):  # pragma: no cover - writer invariant
            raise ConversionError(f"{tasks_path} line {line_number}: key order drift")
        reloaded.append(row)
    return lines, reloaded


def scan_agent_file(
    tasks_path: Path,
    expected: Sequence[Mapping[str, Any]],
    official_prompts: Sequence[str],
    eval_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-open the agent-visible file and fail on any oracle/answer leak.

    Returns the isolation-evidence block stored in the manifest.
    """
    lines, reloaded = _reopen_tasks(tasks_path, expected)
    file_text = tasks_path.read_text(encoding="utf-8")

    _require(len(official_prompts) == 2,
             "expected exactly two distinct official system prompts (stage_1, stage_2)")

    # 1. no forbidden JSON key anywhere.
    forbidden_keys_found: list[dict[str, Any]] = []
    for line_number, row in enumerate(reloaded, start=1):
        offending = sorted(set(_iter_keys(row)) & FORBIDDEN_AGENT_KEYS)
        if offending:
            forbidden_keys_found.append({"line": line_number, "keys": offending})
    _require(not forbidden_keys_found,
             f"{tasks_path}: forbidden JSON keys in agent-visible content: {forbidden_keys_found[:3]}")

    # 2. derive the exact official system prompts and require every forbidden substring
    #    occurrence to be confined to them (outside count must be zero).
    prompts_by_stage: dict[str, set[str]] = {"stage_1": set(), "stage_2": set()}
    for row in reloaded:
        prompts_by_stage["stage_1"].add(row["stage_1"]["system"])
        prompts_by_stage["stage_2"].add(row["stage_2"]["system"])
    _require(
        all(len(values) == 1 for values in prompts_by_stage.values()),
        f"{tasks_path}: expected one distinct system prompt per stage, got "
        f"{ {stage: len(values) for stage, values in prompts_by_stage.items()} }",
    )
    stage_1_prompt = next(iter(prompts_by_stage["stage_1"]))
    stage_2_prompt = next(iter(prompts_by_stage["stage_2"]))
    _require(
        {stage_1_prompt, stage_2_prompt} == set(official_prompts),
        f"{tasks_path}: system prompt differs from the official source prompts",
    )

    marker_inside: Counter = Counter()
    marker_outside: Counter = Counter()
    for line_number, (line, row) in enumerate(zip(lines, reloaded), start=1):
        system_values = (row["stage_1"]["system"], row["stage_2"]["system"])
        for marker in FORBIDDEN_AGENT_SUBSTRINGS:
            inside, outside = _count_marker_outside_prompts(line, marker, system_values)
            marker_inside[marker] += inside
            marker_outside[marker] += outside
            _require(
                outside == 0,
                f"{tasks_path} line {line_number}: forbidden substring {marker!r} appears outside the "
                f"official system prompts ({outside} occurrence(s))",
            )

    # 3. ground-truth values must not appear in agent-visible text.
    gt_product_ids: set[str] = set()
    gt_product_names: set[str] = set()
    for eval_row in eval_rows:
        ground_truth = eval_row["ground_truth"]
        preferences = ground_truth.get("preferences")
        products = preferences if isinstance(preferences, list) else [ground_truth]
        for product in products:
            if not isinstance(product, Mapping):
                continue
            product_id = product.get("product_id")
            if isinstance(product_id, str) and product_id:
                gt_product_ids.add(product_id)
            product_name = product.get("product_name")
            if isinstance(product_name, str) and product_name:
                gt_product_names.add(product_name)
    _require(gt_product_ids and gt_product_names, "ground truth contains no product identities to scan for")

    digit_runs = set(re.findall(r"\d{7,}", file_text))
    leaked_ids = sorted(gt_product_ids & digit_runs)
    _require(not leaked_ids, f"{tasks_path}: ground-truth product ids leak into agent-visible text: {leaked_ids[:5]}")

    leaked_names = sorted(name for name in gt_product_names if name in file_text)
    _require(not leaked_names, f"{tasks_path}: ground-truth product names leak into agent-visible text: {leaked_names[:3]}")

    # 4. the removed oracle block (dialogue sessions) must not be present, not even a fingerprint.
    leaked_oracle = 0
    for eval_row in eval_rows:
        block = eval_row["oracle_memory_block"]
        fingerprint = block[len(ORACLE_MEMORY_MARKER):][:120].strip()
        if fingerprint and fingerprint in file_text:
            leaked_oracle += 1
    _require(leaked_oracle == 0,
             f"{tasks_path}: {leaked_oracle} oracle-memory dialogue fingerprint(s) appear in agent-visible text")

    dialogue_markers = {marker: file_text.count(marker) for marker in DIALOGUE_SESSION_MARKERS}
    _require(all(count == 0 for count in dialogue_markers.values()),
             f"{tasks_path}: dialogue-session markers found in agent-visible text: {dialogue_markers}")

    # 5. the agent file must never be mistaken for the evaluator file.
    eval_only_markers = {
        marker: file_text.count(marker)
        for marker in ("oracle_memory_block_sha256", "stage_2_original_prompt_sha256", '"ground_truth"', '"reward_model"')
    }
    _require(all(count == 0 for count in eval_only_markers.values()),
             f"{tasks_path}: evaluator-only markers found in agent-visible text: {eval_only_markers}")

    _require(
        sum(row["stage_2"]["oracle_chars_removed"] for row in reloaded) > 0,
        f"{tasks_path}: oracle_chars_removed totals must be positive",
    )

    return {
        "top_level_keys_exact": True,
        "forbidden_keys_found": forbidden_keys_found,
        "forbidden_substrings": {
            marker: {
                "inside_official_system_prompts": marker_inside[marker],
                "outside_official_system_prompts": marker_outside[marker],
            }
            for marker in FORBIDDEN_AGENT_SUBSTRINGS
        },
        "official_system_prompts": {
            "usage": "each row carries the stage_1 prompt once and the stage_2 prompt once",
            "stage_1": {
                "chars": len(stage_1_prompt),
                "sha256": sha256_text(stage_1_prompt),
                "rows_using_exact_prompt": len(reloaded),
            },
            "stage_2": {
                "chars": len(stage_2_prompt),
                "sha256": sha256_text(stage_2_prompt),
                "rows_using_exact_prompt": len(reloaded),
                "literal_product_id_occurrences_in_prompt": stage_2_prompt.count("product_id"),
            },
        },
        "official_system_prompt_note": (
            "Both system prompts are the official release text, byte-identical. The stage_2 system "
            "prompt documents the answer format and therefore contains the literal token 'product_id' "
            "(4 occurrences); all forbidden-substring occurrences are confined to these two official "
            "prompt strings and are counted here. The prompts are NOT scrubbed because that would change "
            "the official task instructions."
        ),
        "ground_truth_identity_scan": {
            "unique_product_ids_checked": len(gt_product_ids),
            "unique_product_names_checked": len(gt_product_names),
            "product_ids_in_agent_file": 0,
            "product_names_in_agent_file": 0,
        },
        "oracle_text_fingerprints_checked": len(eval_rows),
        "oracle_text_fingerprints_in_agent_file": 0,
        "dialogue_session_markers_in_agent_file": dialogue_markers,
        "evaluator_only_markers_in_agent_file": eval_only_markers,
    }


def verify_eval_file(
    eval_path: Path,
    expected: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-open the evaluator-only file and verify ground truth + oracle reversibility."""
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    _require(len(lines) == len(expected), f"{eval_path}: expected {len(expected)} lines, found {len(lines)}")
    reloaded: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"{eval_path} line {line_number}: invalid JSON: {exc}") from exc
        _require(tuple(row.keys()) == EVAL_FIELDS,
                 f"{eval_path} line {line_number}: keys {list(row.keys())} != {list(EVAL_FIELDS)}")
        _require(row["schema"] == EVAL_SCHEMA, f"{eval_path} line {line_number}: wrong schema tag")
        _require(isinstance(row["ground_truth"], dict) and bool(row["ground_truth"]),
                 f"{eval_path} line {line_number}: ground_truth must be a non-empty object")
        _require(
            sha256_text(row["oracle_memory_block"]) == row["oracle_memory_block_sha256"],
            f"{eval_path} line {line_number}: oracle_memory_block_sha256 does not match the stored block",
        )
        _require(re.fullmatch(r"[0-9a-f]{64}", row["stage_2_original_prompt_sha256"]) is not None,
                 f"{eval_path} line {line_number}: stage_2_original_prompt_sha256 is not a sha256 hex digest")
        reloaded.append(row)

    task_by_id = {row["task_id"]: row for row in task_rows}
    eval_by_id = {row["task_id"]: row for row in reloaded}
    _require(len(task_by_id) == len(task_rows), f"{eval_path}: duplicate task ids in agent-visible file")
    _require(len(eval_by_id) == len(reloaded), f"{eval_path}: duplicate task ids in evaluator file")
    _require(set(task_by_id) == set(eval_by_id), f"{eval_path}: evaluator task ids do not match the agent file")

    # The stored stage_2 prompt hash must be reproducible from the agent row + oracle block,
    # which proves the oracle block sat exactly at message index 2 of the original prompt.
    reversible = 0
    for task_id, eval_row in eval_by_id.items():
        task = task_by_id[task_id]
        prompt = [
            {"role": "system", "content": task["stage_2"]["system"]},
            {"role": "user", "content": task["stage_2"]["user"]},
            {"role": "user", "content": eval_row["oracle_memory_block"]},
            {"role": "user", "content": task["stage_2"]["final_user"]},
        ]
        _require(
            sha256_text(canonical_json(prompt)) == eval_row["stage_2_original_prompt_sha256"],
            f"{eval_path}: stage_2_original_prompt_sha256 is not reproducible for task {task_id!r}",
        )
        reversible += 1

    oracle_blocks = [row["oracle_memory_block"] for row in reloaded]
    return {
        "rows": len(reloaded),
        "rows_with_oracle_memory_block": sum(1 for block in oracle_blocks if block.startswith(ORACLE_MEMORY_MARKER)),
        "rows_with_ground_truth_object": sum(1 for row in reloaded if isinstance(row["ground_truth"], dict)),
        "oracle_memory_block_sha256_matches": sum(
            1 for row in reloaded if sha256_text(row["oracle_memory_block"]) == row["oracle_memory_block_sha256"]
        ),
        "unique_oracle_memory_blocks": len(set(oracle_blocks)),
        "stage_2_original_prompt_is_reversible": reversible,
        "eval_text_contains_oracle_marker": lines and ORACLE_MEMORY_MARKER in eval_path.read_text(encoding="utf-8"),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def _pairing_validation(
    rows: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage_counts: Counter,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    conversation_ids = [task["conversation_id"] for task in tasks]
    conversation_multiplicity = Counter(Counter(conversation_ids).values())
    suffix_to_type = Counter(
        f"_{task['question_id'].rsplit('_', 1)[-1] if '_' in task['question_id'] else '?'}"
        f":{task['question_type']}"
        for task in tasks
    )
    return {
        "ok": True,
        "source_rows": len(rows),
        "stage_counts": dict(stage_counts),
        "unique_question_ids": len(pairs),
        "tasks_with_exactly_one_stage_1": sum(1 for pair in pairs.values() if "stage_1" in pair),
        "tasks_with_exactly_one_stage_2": sum(1 for pair in pairs.values() if "stage_2" in pair),
        "unpaired_stage_rows": 0,
        "unique_conversation_ids": len(set(conversation_ids)),
        "tasks_per_conversation": {str(count): n for count, n in sorted(conversation_multiplicity.items())},
        "conversation_id_consistent_across_stages": len(tasks),
        "question_identical_across_stages": len(tasks),
        "question_type_consistent_across_stages": len(tasks),
        "source_index_consistent_across_stages": len(tasks),
        "question_id_format_valid": len(tasks),
        "question_id_prefix_matches_conversation_id": len(tasks),
        "question_date_extracted": len(tasks),
        "question_id_suffix_question_type_counts": dict(sorted(suffix_to_type.items())),
        "question_type_counts": dict(Counter(task["question_type"] for task in tasks)),
        "stage_1_prompt_shape": dict(Counter("system+user" for _ in tasks)),
        "stage_2_prompt_shape": dict(Counter("system+user+oracle_memory+final_user" for _ in tasks)),
        "stage_2_message_count_before": 4,
        "stage_2_message_count_after": 3,
        "stage_2_message_count_before_distribution": {"4": len(tasks)},
        "stage_2_message_count_after_distribution": {"3": len(tasks)},
    }


def convert(parquet_path: Path, out_dir: Path) -> dict[str, Any]:
    """Run the whole conversion and return the manifest dictionary."""
    started_at = _utc_now()
    started = datetime.now(timezone.utc)
    _refuse_test_split(parquet_path)
    _require(parquet_path.is_file(), f"parquet file does not exist: {parquet_path}")
    parquet_bytes = parquet_path.read_bytes()
    parquet_sha256 = sha256_bytes(parquet_bytes)

    rows = load_parquet_rows(parquet_path)
    pairs, stage_counts = pair_stage_rows(rows)

    ordered_question_ids = sorted(pairs)
    tasks = [build_task_record(question_id, pairs[question_id]) for question_id in ordered_question_ids]
    eval_rows = [build_eval_record(question_id, pairs[question_id]) for question_id in ordered_question_ids]

    official_prompts = sorted(
        {
            _stage_system_prompt(pair[ability], ability)
            for pair in pairs.values()
            for ability in ("stage_1", "stage_2")
        }
    )
    _require(len(official_prompts) == 2,
             f"expected 2 distinct official system prompts, found {len(official_prompts)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "train_tasks.jsonl"
    eval_path = out_dir / "train_eval.jsonl"
    tasks_facts = write_jsonl(tasks_path, tasks)
    eval_facts = write_jsonl(eval_path, eval_rows)

    agent_scan = scan_agent_file(tasks_path, tasks, official_prompts, eval_rows)
    eval_scan = verify_eval_file(eval_path, eval_rows, tasks)

    oracle_chars = [task["stage_2"]["oracle_chars_removed"] for task in tasks]
    oracle_total = sum(int(chars) for chars in oracle_chars)
    eval_text = eval_path.read_text(encoding="utf-8")
    finished_at = _utc_now()
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": finished_at,
        "timestamps": {"started_at": started_at, "finished_at": finished_at, "duration_seconds": round(duration, 3)},
        "converter": {
            "path": "experiments/p14a_prepare_dataset.py",
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "source": {
            "parquet": {
                "path": str(parquet_path),
                "resolved_path": str(parquet_path.resolve()),
                "bytes": len(parquet_bytes),
                "sha256": parquet_sha256,
                "rows": len(rows),
            },
            "dataset": {
                "name": "yuzhan2205/Shopping-companion",
                "config": "benchmark",
                "split": "train",
                "test_split_used": False,
            },
        },
        "outputs": {
            "train_tasks": {**tasks_facts, "schema": TASKS_SCHEMA, "visibility": "agent"},
            "train_eval": {
                **eval_facts,
                "schema": EVAL_SCHEMA,
                "visibility": "evaluator-only",
                "eval_text_contains_oracle_marker": ORACLE_MEMORY_MARKER in eval_text,
            },
        },
        "row_counts": {
            "source_rows": len(rows),
            "tasks": len(tasks),
            "stage_1_instances": stage_counts.get("stage_1", 0),
            "stage_2_instances": stage_counts.get("stage_2", 0),
            "agent_visible_rows": tasks_facts["rows"],
            "evaluator_only_rows": eval_facts["rows"],
        },
        "ordering": {
            "train_tasks": "sorted by task_id (question_id)",
            "train_eval": "same order as train_tasks",
            "task_id_definition": "official extra_info.question_id (unique per shopping task)",
            "conversation_id_definition": (
                "official extra_info.conversation_id; one conversation carries the _1 single_product "
                "task and the _2 add_on_deals task, so it is not unique per task"
            ),
        },
        "pairing_validation": _pairing_validation(rows, pairs, stage_counts, tasks),
        "oracle_isolation": {
            "stage_2_message_count_before": 4,
            "stage_2_message_count_after": 3,
            "stage_2_removed_message_role": "user",
            "oracle_blocks_removed": len(tasks),
            "oracle_chars_removed": {
                "total": oracle_total,
                "min": min(int(c) for c in oracle_chars),
                "max": max(int(c) for c in oracle_chars),
                "mean": round(oracle_total / len(oracle_chars), 3),
            },
            "agent_visible_file": {**agent_scan, "path": str(tasks_path), "rows": tasks_facts["rows"],
                                   "sha256": tasks_facts["sha256"]},
            "evaluator_only_file": {**eval_scan, "path": str(eval_path), "rows": eval_facts["rows"],
                                    "sha256": eval_facts["sha256"]},
            "assertion": {
                "passed": True,
                "description": (
                    "After writing, the converter re-opened train_tasks.jsonl and raised unless: exact top-level "
                    "keys; no forbidden JSON key; no forbidden substring outside the two official system prompts; "
                    "no ground-truth product id/name; no oracle-memory dialogue fingerprint; no dialogue-session "
                    "marker; no evaluator-only field name. train_eval.jsonl was re-opened and its oracle sha256, "
                    "ground_truth and prompt-reversibility were re-verified."
                ),
            },
        },
        "commands": [
            " ".join(
                [
                    "python",
                    "experiments/p14a_prepare_dataset.py",
                    "--parquet",
                    str(parquet_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )
        ],
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pyarrow": _pyarrow_version(),
        },
        "notes": [
            "The oracle-memory block is ground-truth memory derived from answer_session_ids. It is removed "
            "from agent-visible input and stored only in train_eval.jsonl.",
            agent_scan["official_system_prompt_note"],
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    _require(json.loads(manifest_path.read_text(encoding="utf-8"))["schema"] == MANIFEST_SCHEMA,
             "written manifest is not readable")
    return manifest


def _pyarrow_version() -> str:
    try:
        import pyarrow  # type: ignore[import-not-found]

        return str(pyarrow.__version__)
    except ImportError:  # pragma: no cover - tooling environment always has it
        return "unavailable"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the official Shopping Companion train.parquet into P14-A agent-visible tasks "
        "and evaluator-only ground truth.",
    )
    parser.add_argument("--parquet", required=True, type=Path, help="path to the released train.parquet")
    parser.add_argument(
        "--out-dir",
        default=Path("data/p14a/dataset"),
        type=Path,
        help="output directory (default: data/p14a/dataset)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = convert(args.parquet, args.out_dir)
    except ConversionError as exc:
        print(f"[P14-A] CONVERSION FAILED: {exc}", file=sys.stderr)
        return 1

    isolation = manifest["oracle_isolation"]
    agent_scan = isolation["agent_visible_file"]
    print(f"[P14-A] source parquet : {manifest['source']['parquet']['path']}")
    print(f"[P14-A] source sha256  : {manifest['source']['parquet']['sha256']}")
    print(
        f"[P14-A] rows read      : {manifest['row_counts']['source_rows']} "
        f"-> {manifest['row_counts']['tasks']} tasks "
        f"(stage_1={manifest['row_counts']['stage_1_instances']}, "
        f"stage_2={manifest['row_counts']['stage_2_instances']})"
    )
    print(
        f"[P14-A] question types : "
        f"{manifest['pairing_validation']['question_type_counts']}"
    )
    print(
        f"[P14-A] stage_2 prompt  : {isolation['stage_2_message_count_before']} messages -> "
        f"{isolation['stage_2_message_count_after']} agent-visible messages; "
        f"{isolation['oracle_blocks_removed']} oracle blocks removed, "
        f"{isolation['oracle_chars_removed']['total']} chars total"
    )
    print(f"[P14-A] wrote tasks    : {manifest['outputs']['train_tasks']['path']} "
          f"({manifest['outputs']['train_tasks']['rows']} rows, "
          f"sha256={manifest['outputs']['train_tasks']['sha256']})")
    print(f"[P14-A] wrote eval     : {manifest['outputs']['train_eval']['path']} "
          f"({manifest['outputs']['train_eval']['rows']} rows, "
          f"sha256={manifest['outputs']['train_eval']['sha256']})")
    print(
        "[P14-A] agent scan     : PASS - forbidden keys=0; forbidden substrings outside official prompts=0; "
        f"product_id occurrences inside official prompts={agent_scan['forbidden_substrings']['product_id']['inside_official_system_prompts']}"
    )
    print(
        f"[P14-A] evaluator scan : PASS - {isolation['evaluator_only_file']['rows_with_oracle_memory_block']} "
        f"oracle blocks, {isolation['evaluator_only_file']['rows_with_ground_truth_object']} ground-truth objects, "
        f"reversible prompt hashes={isolation['evaluator_only_file']['stage_2_original_prompt_is_reversible']}"
    )
    print(f"[P14-A] manifest       : {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
