"""P14-A dataset adapter: official Shopping Companion train tasks + evaluator-only labels.

This module is the **only** reader of the P14-A dataset artifacts produced by
``experiments/p14a_prepare_dataset.py``:

* :func:`load_tasks` reads ``train_tasks.jsonl`` (schema
  ``agentrecx.p14a.shopping_task.v1``).  These rows are **agent-visible**: each one joins
  the two official stage-level instances of one shopping task (``stage_1`` = preference
  recovery via the memory tools, ``stage_2`` = product recommendation via the product
  tools).  The official stage_2 oracle-memory message (the ground-truth "most relevant
  user dialogue memories" block) has already been removed from this file.
* :func:`load_eval` reads ``train_eval.jsonl`` (schema ``agentrecx.p14a.eval.v1``).  These
  rows are **evaluator-only**: the parsed task ground truth plus the exact oracle-memory
  block and prompt hashes needed to audit (and reconstruct) the removed message.

Hard boundary
-------------
``train_eval.jsonl`` is for the independent post-run evaluator and **must never be passed
to the agent runtime** - not to the planner, planner bridge, memory adapter, catalogue
tools, trajectory recorder or any agent-facing code path.  Reading it in the runtime
would hand the agent the answer.  Keep the two files with this loader: the evaluator calls
:func:`load_eval` / :func:`load_task_eval_map` *after* the run, nothing else does.

Standard library only: no pyarrow, no numpy, no ML dependency.  The conversion itself
(pyarrow -> JSONL) is a separate one-off tooling step; the runtime consumes JSONL.

Why system prompts are not scanned by :func:`assert_agent_input_has_no_oracle`
-----------------------------------------------------------------------------
Both official system prompts are kept byte-identical to the release.  The stage_2 system
prompt documents the answer format and therefore contains the literal token
``product_id`` (``@REC::product_id@``); that is static instruction boilerplate, not a
task answer.  The guard is applied to every *payload* field (question, user messages,
final user instruction, identifiers, tool kwargs); the system prompts are validated as
non-empty official instruction text that must not contain oracle/ground-truth markers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "EVAL_SCHEMA",
    "FINAL_USER_MESSAGE",
    "FORBIDDEN_AGENT_MARKERS",
    "ORACLE_MEMORY_MARKER",
    "P14DatasetError",
    "QUESTION_TYPES",
    "STAGE_1_ALLOWED_TOOLS",
    "STAGE_2_ALLOWED_TOOLS",
    "TASKS_SCHEMA",
    "ShoppingTask",
    "ShoppingTaskStage",
    "TaskEval",
    "assert_agent_input_has_no_oracle",
    "load_eval",
    "load_task_eval_map",
    "load_tasks",
    "pair_stages",
    "select_task",
    "validate_stage_pairing",
]

#: Schema tags of the two artifact files (frozen by the P14-A dataset contract).
TASKS_SCHEMA = "agentrecx.p14a.shopping_task.v1"
EVAL_SCHEMA = "agentrecx.p14a.eval.v1"

#: First characters of the oracle-memory message that was removed from stage_2 input.
ORACLE_MEMORY_MARKER = "The most relevant user dialogue memories are as follows:"

#: Constant final stage_2 user message of the official release.
FINAL_USER_MESSAGE = "I will not provide you with any more information. Please complete the task."

#: Frozen tool allow-lists per stage.
STAGE_1_ALLOWED_TOOLS: tuple[str, ...] = ("mem_search", "mem_view", "mem_summarize_by_date")
STAGE_2_ALLOWED_TOOLS: tuple[str, ...] = ("product_search", "product_view")

#: Question types emitted by the official release.
QUESTION_TYPES: frozenset[str] = frozenset({"single_product", "add_on_deals"})

#: Markers that identify evaluator-only/oracle content.  None of these may appear in
#: agent-visible payload text; each raises :exc:`P14DatasetError` through
#: :func:`assert_agent_input_has_no_oracle`.
FORBIDDEN_AGENT_MARKERS: tuple[str, ...] = (
    ORACLE_MEMORY_MARKER,
    "ground_truth",
    "reward_model",
    "product_id",
    "oracle_memory_block",
    "answer_session_ids",
)

_TASK_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)
_STAGE_1_FIELDS: frozenset[str] = frozenset({"system", "user"})
_STAGE_2_FIELDS: frozenset[str] = frozenset(
    {"system", "user", "final_user", "oracle_memory_removed", "oracle_chars_removed"}
)
_EVAL_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "task_id",
        "conversation_id",
        "ground_truth",
        "oracle_memory_block",
        "oracle_memory_block_sha256",
        "stage_2_original_prompt_sha256",
    }
)
_QUESTION_DATE_RE = re.compile(r"Current Date:\s*(\d{4}/\d{2}/\d{2})\s*\([A-Za-z]{3}\)")
_QUESTION_DATE_ONLY_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_QUESTION_ID_RE = re.compile(r"^([0-9a-f]{32})_(\d+)$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_PROMPT_FORBIDDEN_MARKERS: tuple[str, ...] = (
    ORACLE_MEMORY_MARKER,
    "ground_truth",
    "reward_model",
)


class P14DatasetError(ValueError):
    """A P14-A dataset artifact is missing, malformed or violates the frozen schema."""


# --------------------------------------------------------------------------- #
# small validation helpers
# --------------------------------------------------------------------------- #


def _fail(message: str) -> None:
    raise P14DatasetError(message)


def _expect_exact_keys(mapping: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = set(mapping)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _fail(f"{where}: schema keys mismatch (missing={missing}, unexpected={extra})")


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected a string, got {type(value).__name__}")
    if not value.strip():
        _fail(f"{where}: expected a non-empty string")
    return value


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{where}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        _fail(f"{where}: expected an integer >= {minimum}, got {value}")
    return value


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _read_jsonl(path: str | Path, *, file_role: str) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        _fail(f"{file_role} file not found: {source}")
        raise AssertionError("unreachable") from exc
    except OSError as exc:
        _fail(f"{file_role} file cannot be read: {source}: {exc}")
        raise AssertionError("unreachable") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            _fail(f"{source} line {line_number}: blank line in {file_role} JSONL (truncated file?)")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{source} line {line_number}: invalid JSON in {file_role} file: {exc}")
            raise AssertionError("unreachable") from exc
        if not isinstance(row, dict):
            _fail(f"{source} line {line_number}: expected a JSON object, got {type(row).__name__}")
        rows.append(row)
    if not rows:
        _fail(f"{source}: {file_role} file contains no rows")
    return rows


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShoppingTaskStage:
    """One agent-visible input stage of a P14-A shopping task.

    ``stage_1`` carries only ``system`` + ``user``; ``stage_2`` additionally carries the
    official ``final_user`` instruction.  The oracle-memory block is never a field here.
    """

    system: str
    user: str
    final_user: str | None = None
    oracle_memory_removed: bool = False
    oracle_chars_removed: int = 0

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        """Chat messages for this stage (system, user [, final user])."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
        if self.final_user is not None:
            messages.append({"role": "user", "content": self.final_user})
        return tuple(messages)


@dataclass(frozen=True)
class ShoppingTask:
    """One agent-visible P14-A shopping task (both official stages joined)."""

    schema: str
    task_id: str
    conversation_id: str
    question_id: str
    question: str
    question_date: str
    question_type: str
    split: str
    source_index: int
    stage_1: ShoppingTaskStage
    stage_2: ShoppingTaskStage
    allowed_tools: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    tools_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stages(self) -> tuple[ShoppingTaskStage, ShoppingTaskStage]:
        """The two stages in canonical order ``(stage_1, stage_2)``."""
        return (self.stage_1, self.stage_2)

    def stage(self, ability: str) -> ShoppingTaskStage:
        """Return the stage named ``"stage_1"`` or ``"stage_2"``; anything else raises."""
        if ability == "stage_1":
            return self.stage_1
        if ability == "stage_2":
            return self.stage_2
        _fail(f"task {self.task_id!r}: unknown stage ability {ability!r}")
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class TaskEval:
    """EVALUATOR-ONLY labels for one P14-A shopping task.

    Never give this object (or the file it was loaded from) to the agent runtime.
    """

    schema: str
    task_id: str
    conversation_id: str
    ground_truth: Mapping[str, Any]
    oracle_memory_block: str
    oracle_memory_block_sha256: str
    stage_2_original_prompt_sha256: str


# --------------------------------------------------------------------------- #
# agent-visible guard
# --------------------------------------------------------------------------- #


def assert_agent_input_has_no_oracle(text: str, *, source: str = "agent input") -> None:
    """Raise :exc:`P14DatasetError` if ``text`` contains evaluator-only/oracle content.

    The guard checks every marker in :data:`FORBIDDEN_AGENT_MARKERS` (the oracle-memory
    marker, ground-truth/reward-model field names, target-product identity keys and the
    answer-session identifiers).  It is intended for the agent-visible payload; it is
    deliberately not applied to the two official system prompts, which are static
    instruction text (the stage_2 prompt documents ``@REC::product_id@``).
    """
    if not isinstance(text, str):
        _fail(f"{source}: expected text to scan, got {type(text).__name__}")
    for marker in FORBIDDEN_AGENT_MARKERS:
        if marker in text:
            _fail(
                f"{source}: forbidden evaluator-only/oracle marker {marker!r} reached agent-visible input"
            )


def _assert_official_prompt(system: str, where: str) -> None:
    """Validate an official (static) system prompt that may carry format boilerplate."""
    text = _require_str(system, where)
    for marker in _OFFICIAL_PROMPT_FORBIDDEN_MARKERS:
        if marker in text:
            _fail(f"{where}: official system prompt contains oracle/ground-truth marker {marker!r}")


# --------------------------------------------------------------------------- #
# task row -> ShoppingTask
# --------------------------------------------------------------------------- #


def _validate_allowed_tools(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    mapping = _require_mapping(raw, where)
    _expect_exact_keys(mapping, frozenset({"stage_1", "stage_2"}), where)
    result: dict[str, tuple[str, ...]] = {}
    for stage, expected in (("stage_1", STAGE_1_ALLOWED_TOOLS), ("stage_2", STAGE_2_ALLOWED_TOOLS)):
        tools = mapping[stage]
        if not isinstance(tools, (list, tuple)) or not all(isinstance(tool, str) for tool in tools):
            _fail(f"{where}.{stage}: expected a list of tool names")
        if tuple(tools) != expected:
            _fail(f"{where}.{stage}: allowed tools {list(tools)} != frozen {list(expected)}")
        result[stage] = tuple(tools)
    return result


def _validate_tools_kwargs(
    raw: Any, *, task_id: str, question_id: str, conversation_id: str, where: str
) -> dict[str, Any]:
    mapping = _require_mapping(raw, where)
    if not mapping:
        _fail(f"{where}: tools_kwargs must not be empty")
    for tool_name, spec in mapping.items():
        tool_where = f"{where}.{tool_name}"
        if not isinstance(tool_name, str) or not tool_name.strip():
            _fail(f"{where}: tool names must be non-empty strings")
        spec_mapping = _require_mapping(spec, tool_where)
        create_kwargs = spec_mapping.get("create_kwargs")
        if create_kwargs is None:
            continue
        create_mapping = _require_mapping(create_kwargs, f"{tool_where}.create_kwargs")
        for key, expected in (("question_id", question_id), ("conversation_id", conversation_id)):
            if key not in create_mapping:
                continue
            actual = create_mapping[key]
            if not isinstance(actual, str) or actual != expected:
                _fail(
                    f"{tool_where}.create_kwargs.{key} = {actual!r} does not match task "
                    f"{task_id!r} ({expected!r}); cross-task tool scoping would leak memory"
                )
    return dict(mapping)


def _build_stage_1(raw: Any, where: str) -> ShoppingTaskStage:
    mapping = _require_mapping(raw, where)
    _expect_exact_keys(mapping, _STAGE_1_FIELDS, where)
    system = _require_str(mapping["system"], f"{where}.system")
    _assert_official_prompt(system, f"{where}.system")
    user = _require_str(mapping["user"], f"{where}.user")
    return ShoppingTaskStage(system=system, user=user)


def _build_stage_2(raw: Any, where: str) -> ShoppingTaskStage:
    mapping = _require_mapping(raw, where)
    _expect_exact_keys(mapping, _STAGE_2_FIELDS, where)
    system = _require_str(mapping["system"], f"{where}.system")
    _assert_official_prompt(system, f"{where}.system")
    user = _require_str(mapping["user"], f"{where}.user")
    final_user = _require_str(mapping["final_user"], f"{where}.final_user")
    if final_user != FINAL_USER_MESSAGE:
        _fail(f"{where}.final_user: does not match the official final instruction")
    removed = mapping["oracle_memory_removed"]
    if removed is not True:
        _fail(f"{where}.oracle_memory_removed: must be true, got {removed!r}")
    chars = _require_int(mapping["oracle_chars_removed"], f"{where}.oracle_chars_removed", minimum=1)
    return ShoppingTaskStage(
        system=system,
        user=user,
        final_user=final_user,
        oracle_memory_removed=True,
        oracle_chars_removed=chars,
    )


def _task_from_row(row: Mapping[str, Any], where: str) -> ShoppingTask:
    _expect_exact_keys(row, _TASK_FIELDS, where)
    if row["schema"] != TASKS_SCHEMA:
        _fail(f"{where}.schema: expected {TASKS_SCHEMA!r}, got {row['schema']!r}")
    task_id = _require_str(row["task_id"], f"{where}.task_id")
    question_id = _require_str(row["question_id"], f"{where}.question_id")
    conversation_id = _require_str(row["conversation_id"], f"{where}.conversation_id")
    if task_id != question_id:
        _fail(f"{where}: task_id {task_id!r} != question_id {question_id!r} (schema v1 ties them)")
    match = _QUESTION_ID_RE.match(question_id)
    if match is None:
        _fail(f"{where}.question_id: {question_id!r} is not '<conversation_id>_<n>'")
    if match.group(1) != conversation_id:
        _fail(f"{where}.question_id: prefix does not match conversation_id {conversation_id!r}")
    question = _require_str(row["question"], f"{where}.question")
    question_date = _require_str(row["question_date"], f"{where}.question_date")
    if _QUESTION_DATE_ONLY_RE.match(question_date) is None:
        _fail(f"{where}.question_date: {question_date!r} is not YYYY/MM/DD")
    date_match = _QUESTION_DATE_RE.search(question)
    if date_match is None:
        _fail(f"{where}.question: missing official 'Current Date: YYYY/MM/DD (Day)' prefix")
    if date_match.group(1) != question_date:
        _fail(f"{where}.question_date: {question_date!r} != question prefix date {date_match.group(1)!r}")
    question_type = row["question_type"]
    if question_type not in QUESTION_TYPES:
        _fail(f"{where}.question_type: unexpected {question_type!r}")
    split = row["split"]
    if split != "train":
        _fail(f"{where}.split: expected 'train', got {split!r}")
    source_index = _require_int(row["source_index"], f"{where}.source_index", minimum=0)
    stage_1 = _build_stage_1(row["stage_1"], f"{where}.stage_1")
    stage_2 = _build_stage_2(row["stage_2"], f"{where}.stage_2")
    allowed_tools = _validate_allowed_tools(row["allowed_tools"], f"{where}.allowed_tools")
    tools_kwargs = _validate_tools_kwargs(
        row["tools_kwargs"],
        task_id=task_id,
        question_id=question_id,
        conversation_id=conversation_id,
        where=f"{where}.tools_kwargs",
    )

    # Oracle isolation for every payload field the agent can see.
    assert_agent_input_has_no_oracle(question, source=f"{where}.question")
    assert_agent_input_has_no_oracle(stage_1.user, source=f"{where}.stage_1.user")
    assert_agent_input_has_no_oracle(stage_2.user, source=f"{where}.stage_2.user")
    assert_agent_input_has_no_oracle(stage_2.final_user or "", source=f"{where}.stage_2.final_user")
    assert_agent_input_has_no_oracle(
        json.dumps(tools_kwargs, ensure_ascii=False, sort_keys=True),
        source=f"{where}.tools_kwargs",
    )
    if stage_1.system == stage_2.system:
        _fail(f"{where}: stage_1 and stage_2 system prompts are identical; release integrity check failed")

    return ShoppingTask(
        schema=TASKS_SCHEMA,
        task_id=task_id,
        conversation_id=conversation_id,
        question_id=question_id,
        question=question,
        question_date=question_date,
        question_type=question_type,
        split=split,
        source_index=source_index,
        stage_1=stage_1,
        stage_2=stage_2,
        allowed_tools=allowed_tools,
        tools_kwargs=tools_kwargs,
    )


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def load_tasks(path: str | Path) -> list[ShoppingTask]:
    """Load the AGENT-VISIBLE task file (schema ``agentrecx.p14a.shopping_task.v1``).

    Strict by design: a missing field, an unexpected key, a bad schema tag, an
    inconsistent question date or any evaluator-only marker in the payload raises
    :exc:`P14DatasetError` instead of being skipped.
    """
    rows = _read_jsonl(path, file_role="agent-visible task")
    tasks: list[ShoppingTask] = []
    for line_number, row in enumerate(rows, start=1):
        tasks.append(_task_from_row(row, f"{Path(path)} line {line_number}"))
    return tasks


def _eval_from_row(row: Mapping[str, Any], where: str) -> TaskEval:
    _expect_exact_keys(row, _EVAL_FIELDS, where)
    if row["schema"] != EVAL_SCHEMA:
        _fail(f"{where}.schema: expected {EVAL_SCHEMA!r}, got {row['schema']!r}")
    task_id = _require_str(row["task_id"], f"{where}.task_id")
    conversation_id = _require_str(row["conversation_id"], f"{where}.conversation_id")
    ground_truth = _require_mapping(row["ground_truth"], f"{where}.ground_truth")
    if not ground_truth:
        _fail(f"{where}.ground_truth: expected a non-empty object")
    oracle_block = _require_str(row["oracle_memory_block"], f"{where}.oracle_memory_block")
    if not oracle_block.startswith(ORACLE_MEMORY_MARKER):
        _fail(f"{where}.oracle_memory_block: does not start with the oracle-memory marker")
    block_sha = _require_str(row["oracle_memory_block_sha256"], f"{where}.oracle_memory_block_sha256")
    if _SHA256_HEX_RE.match(block_sha) is None:
        _fail(f"{where}.oracle_memory_block_sha256: not a sha256 hex digest")
    actual_sha = hashlib.sha256(oracle_block.encode("utf-8")).hexdigest()
    if actual_sha != block_sha:
        _fail(f"{where}.oracle_memory_block_sha256: does not match the stored oracle-memory block")
    prompt_sha = _require_str(row["stage_2_original_prompt_sha256"], f"{where}.stage_2_original_prompt_sha256")
    if _SHA256_HEX_RE.match(prompt_sha) is None:
        _fail(f"{where}.stage_2_original_prompt_sha256: not a sha256 hex digest")
    return TaskEval(
        schema=EVAL_SCHEMA,
        task_id=task_id,
        conversation_id=conversation_id,
        ground_truth=dict(ground_truth),
        oracle_memory_block=oracle_block,
        oracle_memory_block_sha256=block_sha,
        stage_2_original_prompt_sha256=prompt_sha,
    )


def load_eval(path: str | Path) -> list[TaskEval]:
    """Load the EVALUATOR-ONLY labels (schema ``agentrecx.p14a.eval.v1``).

    This is the only function allowed to read ``train_eval.jsonl``.  It must be called by
    the independent post-run evaluator only, never by the agent runtime.
    """
    rows = _read_jsonl(path, file_role="evaluator-only")
    return [_eval_from_row(row, f"{Path(path)} line {line_number}") for line_number, row in enumerate(rows, start=1)]


def load_task_eval_map(path: str | Path) -> dict[str, TaskEval]:
    """Load the evaluator-only rows keyed by ``task_id`` (post-run evaluator only)."""
    evals = load_eval(path)
    mapping: dict[str, TaskEval] = {}
    for task_eval in evals:
        if task_eval.task_id in mapping:
            _fail(f"evaluator-only file {Path(path)}: duplicate task_id {task_eval.task_id!r}")
        mapping[task_eval.task_id] = task_eval
    return mapping


# --------------------------------------------------------------------------- #
# pairing / validation / deterministic selection
# --------------------------------------------------------------------------- #


def _stage_pair(task: ShoppingTask, where: str) -> tuple[ShoppingTaskStage, ShoppingTaskStage]:
    stage_1, stage_2 = task.stage_1, task.stage_2
    if not isinstance(stage_1, ShoppingTaskStage) or not isinstance(stage_2, ShoppingTaskStage):
        _fail(f"{where}: task {task.task_id!r} does not carry two ShoppingTaskStage objects")
    if stage_1.final_user is not None:
        _fail(f"{where}: task {task.task_id!r} stage_1 must not carry a final_user message")
    if stage_1.oracle_memory_removed or stage_1.oracle_chars_removed:
        _fail(f"{where}: task {task.task_id!r} stage_1 must not carry oracle-removal metadata")
    if stage_2.final_user is None or not stage_2.oracle_memory_removed:
        _fail(f"{where}: task {task.task_id!r} stage_2 is not a valid recommendation stage")
    return stage_1, stage_2


def pair_stages(tasks: Iterable[ShoppingTask]) -> dict[str, tuple[ShoppingTaskStage, ...]]:
    """Pair each task's ``stage_1`` and ``stage_2`` into ``{task_id: (stage_1, stage_2)}``.

    Raises :exc:`P14DatasetError` for an empty input, duplicate task ids, or a task that
    does not carry exactly one preference-recovery stage and one recommendation stage.
    """
    pairs: dict[str, tuple[ShoppingTaskStage, ...]] = {}
    for task in tasks:
        if not isinstance(task, ShoppingTask):
            _fail(f"pair_stages: expected ShoppingTask objects, got {type(task).__name__}")
        if task.task_id in pairs:
            _fail(f"pair_stages: duplicate task_id {task.task_id!r}")
        pairs[task.task_id] = _stage_pair(task, "pair_stages")
    if not pairs:
        _fail("pair_stages: no tasks to pair")
    return pairs


def validate_stage_pairing(tasks: Iterable[ShoppingTask]) -> dict[str, int]:
    """Validate the frozen P14-A pairing invariants and return integer counts.

    Raises :exc:`P14DatasetError` on the first violation.  The returned mapping contains
    only integer counters (e.g. ``{"tasks": 800, "stage_1": 800, ...}``).
    """
    task_list = list(tasks)
    pairs = pair_stages(task_list)
    unique_conversation_ids: set[str] = set()
    counters = {
        "tasks": len(task_list),
        "stage_1": 0,
        "stage_2": 0,
        "paired_tasks": len(pairs),
        "unique_task_ids": len({task.task_id for task in task_list}),
        "unique_question_ids": len({task.question_id for task in task_list}),
        "unique_conversation_ids": 0,
        "single_product": 0,
        "add_on_deals": 0,
        "oracle_memory_removed": 0,
        "oracle_chars_removed_total": 0,
        "stage_1_question_equals_stage_2": 0,
        "question_date_extracted": 0,
    }
    if counters["unique_task_ids"] != len(task_list):
        _fail("validate_stage_pairing: duplicate task_id values")
    if counters["unique_question_ids"] != len(task_list):
        _fail("validate_stage_pairing: duplicate question_id values")
    for task in task_list:
        stage_1, stage_2 = _stage_pair(task, "validate_stage_pairing")
        counters["stage_1"] += 1
        counters["stage_2"] += 1
        unique_conversation_ids.add(task.conversation_id)
        if task.question_type == "single_product":
            counters["single_product"] += 1
        elif task.question_type == "add_on_deals":
            counters["add_on_deals"] += 1
        else:  # pragma: no cover - load_tasks already rejects this
            _fail(f"validate_stage_pairing: task {task.task_id!r} has unexpected question_type")
        if stage_1.user != task.question or stage_2.user != task.question:
            _fail(f"validate_stage_pairing: task {task.task_id!r} stage user messages are not the question")
        counters["stage_1_question_equals_stage_2"] += 1
        if _QUESTION_DATE_RE.search(task.question) is None or task.question_date not in task.question:
            _fail(f"validate_stage_pairing: task {task.task_id!r} question_date is inconsistent with question")
        counters["question_date_extracted"] += 1
        if not stage_2.oracle_memory_removed:
            _fail(f"validate_stage_pairing: task {task.task_id!r} stage_2 oracle memory is not marked removed")
        counters["oracle_memory_removed"] += 1
        counters["oracle_chars_removed_total"] += stage_2.oracle_chars_removed
    counters["unique_conversation_ids"] = len(unique_conversation_ids)
    counters["ok"] = 1
    return {key: int(value) for key, value in counters.items()}


def select_task(tasks: Iterable[ShoppingTask], index: int = 0) -> ShoppingTask:
    """Deterministically pick one task: sort by ``task_id`` then take ``index % len``.

    ``index`` may be any integer (negative values wrap); the same ``(tasks, index)``
    always returns the same task regardless of the input iteration order.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        _fail(f"select_task: index must be an integer, got {type(index).__name__}")
    ordered = sorted(tasks, key=lambda task: _task_id_of(task))
    if not ordered:
        _fail("select_task: no tasks available")
    return ordered[index % len(ordered)]


def _task_id_of(task: Any) -> str:
    if not isinstance(task, ShoppingTask):
        _fail(f"select_task: expected ShoppingTask objects, got {type(task).__name__}")
    return task.task_id
