# P3C Final Fix Report — SOURCE FIXED, RUNTIME VERIFICATION BLOCKED

## 1. 已应用的源码修改

### `recommendation/demo/agent_service.py::_integrate_state_turn`

```python
if proposal_actions is not None:
    current_turn_kinds: tuple[str, ...] = ()
else:
    current_turn_kinds = tuple(kind for kind, _value in trusted_pairs)

merged = shopping_adapter.merge_constraint_pairs(
    trusted_pairs,
    outcome.authorized_constraints,
    current_turn_kinds=current_turn_kinds,
)
self._constraint_trace[task_id] = {
    ...
    "current_turn_kinds": list(current_turn_kinds),
    "merged": [list(item) for item in merged],
}
```

- LLM Proposal 模式不再把 `price_max` 放进 `current_turn_kinds`；
- Rule 模式只使用真实 `trusted_pairs` 的 kind，不再混入 `_task_price_constraints` 历史缓存；
- trace 的 `current_turn_kinds` 与实际传给 merge 的变量完全一致。

### `recommendation/state/shopping_adapter.py::merge_constraint_pairs`

只在 **同 kind 的 trusted constraint 实际存在** 时，才允许 `current_turn_kinds` 抑制
authorized constraint：

```python
trusted_kinds: set[str] = set()
# ...
trusted_kinds.add(signature[0])
# ...
if signature[0] in current_kinds and signature[0] in trusted_kinds:
    continue
```

新增回归测试：`tests/test_p3c_final_merge.py`
- authorized 50 + 空 trusted + current_kind price_max → 保留 50；
- trusted 30 + authorized 50 + current_kind price_max → trusted 30 胜出；
- 无 current 抑制 → authorized 加入。

源码 diff：`P3C_FINAL_SOURCE_DIFF.txt`

---

## 2. 运行验证状态：BLOCKED

当前本地 Python 环境在执行最小测试/Stub API 时出现依赖导入失败：

```text
ModuleNotFoundError: No module named 'anyio._lazyimport'
ImportError: cannot import name 'BaseModel' from 'pydantic' (unknown location)
pytest module has no attribute console_main
```

因此**无法完成可信的四轮 Stub API 重跑和测试结果确认**。

不把这一轮的任何旧运行结果当作修复后证据：
- 不声称 `30 → 50 → None → None` 已实测通过；
- 不声称 R1 实际输入已是 50；
- 不声称第二轮 Final 为空已定位完成。

`P3C_FINAL_FOUR_TURN_TRACE.csv` 仍是旧 trace，仅作历史参考，不是修复后验收。

---

## 3. 验收状态

| 要求 | 状态 |
|---|---|
| `$50` 不再被 merge 丢弃（源码） | 已修（未运行验证） |
| R1 实际输入与 effective state 一致 | **UNVERIFIED** |
| R1 diagnostics 不再 stale | **UNVERIFIED** |
| 撤回后旧值不复活 | **UNVERIFIED** |
| 第二轮 Final=[] 原因 | **UNVERIFIED / 未完成** |
| 必要回归测试通过 | **BLOCKED：环境导入错误** |

---

## 4. 下一最小步骤

1. 修复本地 Python 环境（或使用可用的项目虚拟环境）；
2. 运行 `tests/test_p3c_final_merge.py`；
3. 重跑四轮 Stub API，读取 `constraint_trace.merged`、`r1_actual`、`candidate_filter_diagnostics.active_price_max`；
4. 若第二轮 Final 仍为空，只从 candidate final4 到 API final0 的终端/收敛路径继续定位，不放宽 role/硬约束。

---

## 5. 交付

- `P3C_FINAL_FIX_REPORT.md`（本文件）
- `P3C_FINAL_FOUR_TURN_TRACE.csv`（旧 trace，不是修复后验收）
- `P3C_FINAL_CANDIDATE_TRACE.csv`
- `P3C_FINAL_SOURCE_DIFF.txt`
- `tests/test_p3c_final_merge.py`

未启动 GPU、未训练、未运行完整 Benchmark、未修改 Serving V1/Gold/Evaluator、未 RL、
未 commit/push/deploy。

**结论：源码修复已应用，但运行验收 BLOCKED，不能宣称 P3C 最终修复通过。**
