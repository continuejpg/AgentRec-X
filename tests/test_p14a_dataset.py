"""P14-A dataset adapter tests: stage pairing, oracle isolation, deterministic selection.

The fixtures are a deterministic real slice of the official Shopping Companion train split,
converted by ``experiments/p14a_prepare_dataset.py``.  Selection rule: walking the converted
``train_tasks.jsonl`` in its canonical order (sorted by ``task_id``), take the first two
``single_product`` tasks and the first one ``add_on_deals`` task.

* ``tests/data/p14a/tasks_fixture.jsonl`` - AGENT-VISIBLE rows (schema
  ``agentrecx.p14a.shopping_task.v1``); the official stage_2 oracle-memory message is gone.
* ``tests/data/p14a/eval_fixture.jsonl`` - EVALUATOR-ONLY rows (schema
  ``agentrecx.p14a.eval.v1``); contains the real train-split ground truth and the exact
  removed oracle-memory block.  That is allowed here (train split, evaluator-only), and the
  isolation assertions below use it to prove the agent file does not leak it.

Everything runs offline with the standard library only: no pyarrow, no torch, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.shopping_companion.dataset import (  # noqa: E402
    EVAL_SCHEMA,
    FINAL_USER_MESSAGE,
    FORBIDDEN_AGENT_MARKERS,
    ORACLE_MEMORY_MARKER,
    P14DatasetError,
    STAGE_1_ALLOWED_TOOLS,
    STAGE_2_ALLOWED_TOOLS,
    TASKS_SCHEMA,
    ShoppingTask,
    assert_agent_input_has_no_oracle,
    load_eval,
    load_task_eval_map,
    load_tasks,
    pair_stages,
    select_task,
    validate_stage_pairing,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "p14a"
TASKS_FIXTURE = FIXTURE_DIR / "tasks_fixture.jsonl"
EVAL_FIXTURE = FIXTURE_DIR / "eval_fixture.jsonl"

#: Exact top-level key order frozen by schema ``agentrecx.p14a.shopping_task.v1``.
TASK_FIELDS = (
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
STAGE_1_FIELDS = ("system", "user")
STAGE_2_FIELDS = ("system", "user", "final_user", "oracle_memory_removed", "oracle_chars_removed")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys |= _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _all_keys(child)
    return keys


@pytest.fixture(scope="module")
def tasks() -> list[ShoppingTask]:
    return load_tasks(TASKS_FIXTURE)


@pytest.fixture(scope="module")
def eval_map():
    return load_task_eval_map(EVAL_FIXTURE)


@pytest.fixture(scope="module")
def raw_task_rows() -> list[dict]:
    return _read_rows(TASKS_FIXTURE)


# --------------------------------------------------------------------------- #
# fixtures + schema
# --------------------------------------------------------------------------- #


def test_fixtures_exist_and_cover_both_question_types(tasks: list[ShoppingTask]) -> None:
    assert TASKS_FIXTURE.is_file() and EVAL_FIXTURE.is_file()
    assert len(tasks) == 3
    types = [task.question_type for task in tasks]
    assert types.count("single_product") == 2
    assert types.count("add_on_deals") == 1
    assert [task.question_id for task in tasks] == sorted(task.question_id for task in tasks)


def test_agent_rows_have_the_exact_frozen_schema(raw_task_rows: list[dict]) -> None:
    assert raw_task_rows
    for row in raw_task_rows:
        assert tuple(row.keys()) == TASK_FIELDS
        assert row["schema"] == TASKS_SCHEMA
        assert tuple(row["stage_1"].keys()) == STAGE_1_FIELDS
        assert tuple(row["stage_2"].keys()) == STAGE_2_FIELDS
        assert row["split"] == "train"
        assert row["question_type"] in {"single_product", "add_on_deals"}
        assert isinstance(row["source_index"], int) and not isinstance(row["source_index"], bool)
        assert row["allowed_tools"] == {
            "stage_1": list(STAGE_1_ALLOWED_TOOLS),
            "stage_2": list(STAGE_2_ALLOWED_TOOLS),
        }
        # No evaluator-only JSON key may appear anywhere in an agent-visible row.
        forbidden_keys = {"ground_truth", "reward_model", "oracle_memory_block", "product_id", "dialogue"}
        assert not (_all_keys(row) & forbidden_keys)


# --------------------------------------------------------------------------- #
# loaders + pairing
# --------------------------------------------------------------------------- #


def test_load_tasks_and_pairing_counts(tasks: list[ShoppingTask]) -> None:
    pairs = pair_stages(tasks)
    assert len(pairs) == 3
    for task in tasks:
        assert pairs[task.task_id] == (task.stage_1, task.stage_2)

    counts = validate_stage_pairing(tasks)
    assert all(isinstance(value, int) for value in counts.values()), counts
    assert counts["tasks"] == 3
    assert counts["paired_tasks"] == 3
    assert counts["stage_1"] == 3
    assert counts["stage_2"] == 3
    assert counts["unique_task_ids"] == 3
    assert counts["unique_question_ids"] == 3
    assert counts["unique_conversation_ids"] == 2  # one conversation carries the _1 and _2 task
    assert counts["single_product"] == 2
    assert counts["add_on_deals"] == 1
    assert counts["oracle_memory_removed"] == 3
    assert counts["oracle_chars_removed_total"] == sum(task.stage_2.oracle_chars_removed for task in tasks)
    assert counts["stage_1_question_equals_stage_2"] == 3
    assert counts["question_date_extracted"] == 3
    assert counts["ok"] == 1


def test_exactly_one_preference_recovery_and_one_recommendation_stage(tasks: list[ShoppingTask]) -> None:
    for task in tasks:
        assert isinstance(task, ShoppingTask)
        assert task.task_id == task.question_id
        assert task.question_id.startswith(task.conversation_id)

        stage_1, stage_2 = task.stages
        assert isinstance(stage_1.final_user, type(None))
        assert stage_1.oracle_memory_removed is False
        assert stage_1.oracle_chars_removed == 0
        assert stage_2.final_user == FINAL_USER_MESSAGE
        assert stage_2.oracle_memory_removed is True
        assert stage_2.oracle_chars_removed > 0
        assert stage_1 is task.stage("stage_1")
        assert stage_2 is task.stage("stage_2")
        assert [message["role"] for message in stage_1.messages] == ["system", "user"]
        assert [message["role"] for message in stage_2.messages] == ["system", "user", "user"]

        # Both stage user messages are the official question (same question in both stages).
        assert stage_1.user == task.question
        assert stage_2.user == task.question


def test_pairing_rejects_duplicates_and_mixed_stages(tasks: list[ShoppingTask]) -> None:
    with pytest.raises(P14DatasetError):
        pair_stages([tasks[0], tasks[0]])
    with pytest.raises(P14DatasetError):
        pair_stages([])
    with pytest.raises(P14DatasetError):
        validate_stage_pairing([tasks[0], tasks[0]])
    # A stage_1 that carries a stage_2-only field is not a valid pair.
    mixed = replace(tasks[0], stage_1=replace(tasks[0].stage_1, final_user="extra"))
    with pytest.raises(P14DatasetError):
        pair_stages([mixed])
    # A stage_2 without the oracle-removal metadata is not a valid recommendation stage.
    unmarked = replace(tasks[0], stage_2=replace(tasks[0].stage_2, oracle_memory_removed=False))
    with pytest.raises(P14DatasetError):
        pair_stages([unmarked])


# --------------------------------------------------------------------------- #
# deterministic selection
# --------------------------------------------------------------------------- #


def test_select_task_is_deterministic(tasks: list[ShoppingTask]) -> None:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    assert select_task(tasks, 0) == ordered[0]
    assert select_task(tasks, 0) == select_task(tasks, 0)
    assert select_task(list(reversed(tasks)), 0) == ordered[0]
    assert select_task(tasks, 3) == ordered[0]  # index % len
    assert select_task(tasks, -1) == ordered[-1]
    assert select_task(tasks, 1) != select_task(tasks, 0)
    with pytest.raises(ValueError):
        select_task([])
    with pytest.raises(P14DatasetError):
        select_task(tasks, "0")  # type: ignore[arg-type]
    with pytest.raises(P14DatasetError):
        select_task(tasks, True)


# --------------------------------------------------------------------------- #
# oracle isolation: agent side
# --------------------------------------------------------------------------- #


def test_stage_2_agent_input_contains_no_oracle(tasks: list[ShoppingTask], raw_task_rows: list[dict]) -> None:
    agent_text = TASKS_FIXTURE.read_text(encoding="utf-8")
    assert ORACLE_MEMORY_MARKER not in agent_text
    assert "ground_truth" not in agent_text
    assert "reward_model" not in agent_text
    assert "[Date: " not in agent_text  # a real dialogue session never reaches the agent file
    assert '\\nuser: ' not in agent_text
    assert '\\nassistant: ' not in agent_text

    # The ONLY "product_id" occurrences in the agent file are inside the official stage_2
    # system prompt (4 per row: the documented answer format).  Nothing else may carry it.
    expected_occurrences = sum(row["stage_2"]["system"].count("product_id") for row in raw_task_rows)
    assert expected_occurrences == 4 * len(raw_task_rows)
    assert agent_text.count("product_id") == expected_occurrences

    for task, row in zip(tasks, raw_task_rows):
        # The oracle-memory message is not stored anywhere in the agent row, not even as a
        # fingerprint; only its character count survives (as isolation metadata).
        assert task.stage_2.oracle_chars_removed == row["stage_2"]["oracle_chars_removed"]
        payload_fields = {
            "question": task.question,
            "stage_1_user": task.stage_1.user,
            "stage_2_user": task.stage_2.user,
            "stage_2_final_user": task.stage_2.final_user or "",
            "question_id": task.question_id,
            "conversation_id": task.conversation_id,
            "tools_kwargs": json.dumps(task.tools_kwargs, ensure_ascii=False, sort_keys=True),
        }
        for field_name, payload in payload_fields.items():
            assert_agent_input_has_no_oracle(payload, source=field_name)

    # Documented schema fact: the official stage_2 system prompt legitimately documents the
    # answer format with the literal token ``product_id``; it is static instruction text, not
    # an answer, and it is the ONLY place where that token may appear.
    for task in tasks:
        assert "@REC::product_id@" in task.stage_2.system
        assert ORACLE_MEMORY_MARKER not in task.stage_2.system
        assert "ground_truth" not in task.stage_2.system
        assert "reward_model" not in task.stage_2.system
        assert "@REC::product_id@" not in task.stage_1.system


def test_guard_rejects_oracle_markers() -> None:
    assert_agent_input_has_no_oracle("Current Date: 2025/08/30 (Sat)\ncollar")
    for marker in FORBIDDEN_AGENT_MARKERS:
        with pytest.raises(ValueError):
            assert_agent_input_has_no_oracle(f"leaked: {marker}")
    with pytest.raises(P14DatasetError):
        assert_agent_input_has_no_oracle('{"product_id": "3449560723"}')
    with pytest.raises(P14DatasetError):
        assert_agent_input_has_no_oracle(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# oracle isolation: evaluator side + both directions
# --------------------------------------------------------------------------- #


def test_eval_file_contains_oracle_and_ground_truth(eval_map, tasks: list[ShoppingTask]) -> None:
    agent_text = TASKS_FIXTURE.read_text(encoding="utf-8")
    assert set(eval_map) == {task.task_id for task in tasks}

    eval_rows = load_eval(EVAL_FIXTURE)
    assert len(eval_rows) == 3
    for row in eval_rows:
        assert row.schema == EVAL_SCHEMA
        assert row.task_id in eval_map
        assert row.oracle_memory_block.startswith(ORACLE_MEMORY_MARKER)
        assert (
            hashlib.sha256(row.oracle_memory_block.encode("utf-8")).hexdigest()
            == row.oracle_memory_block_sha256
        )
        assert row.oracle_memory_block not in agent_text  # the removed message stays removed
        assert row.ground_truth  # parsed dict, not a JSON string
        if row.task_id.endswith("_1"):
            assert {"product_id", "product_name", "dialogue"} <= set(row.ground_truth)
        else:
            assert {"n", "preferences", "voucher", "voucher_type"} <= set(row.ground_truth)

    # The eval file is not the agent file: no task row schema and no stage dicts there.
    eval_text = EVAL_FIXTURE.read_text(encoding="utf-8")
    assert TASKS_SCHEMA not in eval_text
    assert '"stage_1"' not in eval_text and '"stage_2"' not in eval_text

    # Target identities from the ground truth must not appear in the agent-visible text.
    target_ids: set[str] = set()
    for row in eval_rows:
        ground_truth = row.ground_truth
        products = ground_truth["preferences"] if "preferences" in ground_truth else [ground_truth]
        for product in products:
            target_ids.add(str(product["product_id"]))
    assert target_ids
    assert not (target_ids & set(re.findall(r"\d{7,}", agent_text)))


def test_eval_file_is_reversible_from_agent_row_plus_oracle_block(
    tasks: list[ShoppingTask], eval_map
) -> None:
    for task in tasks:
        row = eval_map[task.task_id]
        original_prompt = [
            {"role": "system", "content": task.stage_2.system},
            {"role": "user", "content": task.stage_2.user},
            {"role": "user", "content": row.oracle_memory_block},
            {"role": "user", "content": task.stage_2.final_user},
        ]
        reconstructed = hashlib.sha256(_canonical_json(original_prompt).encode("utf-8")).hexdigest()
        assert reconstructed == row.stage_2_original_prompt_sha256


def test_agent_and_eval_files_are_not_interchangeable(tasks: list[ShoppingTask]) -> None:
    # load_tasks must reject the evaluator-only file and vice versa: the boundary is explicit.
    with pytest.raises(P14DatasetError):
        load_tasks(EVAL_FIXTURE)
    with pytest.raises(P14DatasetError):
        load_eval(TASKS_FIXTURE)


# --------------------------------------------------------------------------- #
# question-date extraction
# --------------------------------------------------------------------------- #


def test_question_date_extraction(tasks: list[ShoppingTask]) -> None:
    pattern = re.compile(r"Current Date:\s*(\d{4}/\d{2}/\d{2})\s*\([A-Za-z]{3}\)")
    for task in tasks:
        assert task.question.startswith("Current Date: ")
        match = pattern.search(task.question)
        assert match is not None
        assert task.question_date == match.group(1)
        assert re.fullmatch(r"\d{4}/\d{2}/\d{2}", task.question_date)
        assert task.question_date in task.question
        assert "Current Date:" not in task.question_date


# --------------------------------------------------------------------------- #
# malformed input is raised, never swallowed
# --------------------------------------------------------------------------- #


def _good_task_row() -> dict:
    return _read_rows(TASKS_FIXTURE)[0]


def test_malformed_task_files_raise(tmp_path: Path) -> None:
    good = _good_task_row()

    missing = json.loads(json.dumps(good))
    del missing["question"]
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "missing.jsonl", [missing]))

    unknown = json.loads(json.dumps(good))
    unknown["oracle_memory_block"] = "leak"
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "unknown.jsonl", [unknown]))

    wrong_schema = json.loads(json.dumps(good))
    wrong_schema["schema"] = "agentrecx.p13.something"
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "schema.jsonl", [wrong_schema]))

    wrong_date = json.loads(json.dumps(good))
    wrong_date["question_date"] = "2025/01/01"
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "date.jsonl", [wrong_date]))

    injected = json.loads(json.dumps(good))
    injected["stage_2"]["user"] = f"{ORACLE_MEMORY_MARKER}\n\n[Date: 2025/01/01 (Wed)]\nuser: leaked"
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "injected.jsonl", [injected]))

    wrong_tool_scope = json.loads(json.dumps(good))
    wrong_tool_scope["tools_kwargs"]["mem_search"]["create_kwargs"]["conversation_id"] = "0" * 32
    with pytest.raises(P14DatasetError):
        load_tasks(_write_rows(tmp_path / "scope.jsonl", [wrong_tool_scope]))


def test_bad_json_blank_lines_and_partial_files_raise(tmp_path: Path) -> None:
    good = _good_task_row()
    with pytest.raises(P14DatasetError):
        load_tasks(tmp_path / "does_not_exist.jsonl")

    not_json = tmp_path / "not_json.jsonl"
    not_json.write_text("not json\n", encoding="utf-8")
    with pytest.raises(P14DatasetError):
        load_tasks(not_json)

    blank = tmp_path / "blank.jsonl"
    blank.write_text(
        json.dumps(good, ensure_ascii=False) + "\n\n" + json.dumps(good, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(P14DatasetError):
        load_tasks(blank)

    # A partially broken file must fail as a whole instead of returning the good prefix.
    broken = tmp_path / "partial.jsonl"
    _write_rows(broken, [good, {"schema": TASKS_SCHEMA}])
    with pytest.raises(P14DatasetError):
        load_tasks(broken)


def test_duplicate_task_ids_raise(tmp_path: Path) -> None:
    good = _good_task_row()
    path = _write_rows(tmp_path / "dupes.jsonl", [good, json.loads(json.dumps(good))])
    loaded = load_tasks(path)  # loading alone cannot see the duplicate
    assert len(loaded) == 2
    with pytest.raises(P14DatasetError):
        validate_stage_pairing(loaded)


def test_malformed_eval_files_raise(tmp_path: Path) -> None:
    good_eval = _read_rows(EVAL_FIXTURE)[0]

    no_gt = json.loads(json.dumps(good_eval))
    del no_gt["ground_truth"]
    with pytest.raises(P14DatasetError):
        load_eval(_write_rows(tmp_path / "no_gt.jsonl", [no_gt]))

    bad_sha = json.loads(json.dumps(good_eval))
    bad_sha["oracle_memory_block_sha256"] = "0" * 64
    with pytest.raises(P14DatasetError):
        load_eval(_write_rows(tmp_path / "bad_sha.jsonl", [bad_sha]))

    bad_schema = json.loads(json.dumps(good_eval))
    bad_schema["schema"] = TASKS_SCHEMA
    with pytest.raises(P14DatasetError):
        load_eval(_write_rows(tmp_path / "bad_schema.jsonl", [bad_schema]))


def test_loader_accepts_str_and_path(tasks: list[ShoppingTask]) -> None:
    from_path = load_tasks(TASKS_FIXTURE)
    from_str = load_tasks(str(TASKS_FIXTURE))
    assert [task.task_id for task in from_path] == [task.task_id for task in from_str] == [
        task.task_id for task in tasks
    ]
    assert [row.oracle_memory_block_sha256 for row in load_eval(str(EVAL_FIXTURE))] == [
        row.oracle_memory_block_sha256 for row in load_eval(EVAL_FIXTURE)
    ]


# --------------------------------------------------------------------------- #
# dependency boundary
# --------------------------------------------------------------------------- #


def test_dataset_module_is_stdlib_only() -> None:
    """Importing the runtime adapter must not pull pyarrow/torch into the process."""
    script = (
        "import sys;"
        "import recommendation.shopping_companion.dataset as dataset;"
        "assert 'pyarrow' not in sys.modules, 'pyarrow imported by runtime adapter';"
        "assert 'torch' not in sys.modules, 'torch imported by runtime adapter';"
        "print(dataset.TASKS_SCHEMA)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert TASKS_SCHEMA in completed.stdout
