# P3C Runtime Acceptance Report

## 1. 环境恢复

已恢复隔离 CPU 测试环境：

```text
C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env
```

基于 `D:\IT\CONDA\envs\cxr\python.exe`，使用项目已有依赖定义安装：

```text
pydantic==2.10.3
fastapi / uvicorn / httpx / anyio / pytest / langgraph
```

未升级整个依赖树，未修改训练环境。

验证：

```text
python -m pytest tests/test_p3c_final_merge.py -q
3 passed
```

---

## 2. 真实丢失位置

`recommendation/demo/agent_service.py::run_shopping_turn` 的末端防御性注入块：

```python
proposed_kinds = {...}
existing = {(kind, value) for item in constraints}   # 过滤前快照
constraints = tuple(
    item for item in constraints if kind not in proposed_kinds
)
added = tuple(
    ... for item in proposal_actions.constraints
    if (kind, value) not in existing                 # 已被过滤掉，仍按旧快照判断
)
```

`_integrate_state_turn` 正确返回了 `price_max=30/50`，但该块先按 kind 移除它，再用过滤前
的 `existing` 判定“已存在”，导致 `added=()`，最终 `constraints=[]`。

**这就是 `returned_constraints=[price_max=50]` → `final_constraints=[]` 的确切位置。**

### 修复

- `_integrate_state_turn`：Proposal 模式 `current_turn_kinds=()`；Rule 模式仅使用真实
  `trusted_pairs` 的 kind，不再混入 `_task_price_constraints` 历史缓存；
- `merge_constraint_pairs`：只有同 kind trusted constraint 实际存在时，`current_turn_kinds`
  才能抑制 authorized；
- 防御性注入块：移除错误的过滤前 `existing` 判定；先按 kind 替换，再直接加入当前 proposal
  约束；
- `_constraint_trace.current_turn_kinds` 与实际传入 merge 的参数一致。

源码 diff：`P3C_RUNTIME_SOURCE_DIFF.txt`

---

## 3. 四轮 Stub API 实际输入

STUB_NOT_REAL_QWEN；真实 Shopping API 路由。

| Turn | merged | final constraints | R1 actual constraints | Eligibility | Final |
|---:|---|---|---|---|---|
| 1 | price_max=30 | price_max=30 | **price_max=30** | requirement=30，eligible=10，unresolved=4 | 3 IDs |
| 2 | price_max=50 | price_max=50 | **price_max=50** | requirement=50，eligible=6，unresolved=4 | []（见下） |
| 3 | [] | [] | [] | 无预算要求 | 3 IDs |
| 4 | [] | [] | null（未进入 R1） | 无预算要求 | 3 IDs |

M5 `merged`、最终 constraints、R1 实际调用参数三轮一致：`30 → 50 → None → None`。

---

## 4. 第二轮 Final=[] 的真实原因

不是价格过滤：

第二轮终端：

```text
terminal.status = waiting_for_user
termination_reason = awaiting_user
termination_detail = "awaiting the user's answer to: candidate_set"
```

即 R1 后要求用户回答候选集问题（candidate_set），该 turn 按协议不展示商品，因此 Final=[]。
Eligibility 本身是 6 个 verified eligible、4 个 unresolved，符合 $50；不是预算过滤清空。

历史 P3C 的“预算导致空候选”结论不适用于当前复测。

---

## 5. 验收结论

| 项目 | 状态 |
|---|---|
| 环境恢复 | PASS |
| merge 单元测试 | PASS（3 passed） |
| 四轮 API | PASS |
| merged 30→50→None→None | PASS |
| R1 实际输入 30→50→None→None | PASS |
| Eligibility 使用 30/50 最新值 | PASS |
| 第二轮 Final=[] | 已解释：候选集澄清等待，不是预算 |
| 默认 Serving V1 | 未修改 |

---

## 6. 交付

- `P3C_RUNTIME_ACCEPTANCE_REPORT.md`
- `P3C_RUNTIME_FOUR_TURN_TRACE.csv`
- `P3C_RUNTIME_SOURCE_DIFF.txt`
- `tests/test_p3c_final_merge.py`（3 passed）
- Stub 轨迹：`P3C2B_STUB_API_TRAJECTORY.jsonl`

未启动 GPU、未训练、未运行完整 Benchmark、未修改 Serving V1/Gold/Evaluator、未 RL、
未 commit/push/deploy。
