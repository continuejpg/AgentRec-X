# P3D Planner Action Contract — CPU 原型报告

## 1. 实现范围

新增最小、显式 opt-in 的 Planner Action Contract：

- `recommendation/demo/planner_contract.py`
  - `PlannerAction`: `ASK / RETRIEVE / RERANK / FINALIZE`
  - `PlannerObservation`
  - `ActionProposal`
  - `validate_action(...)`：最大步数、空候选 FINALIZE、空候选 RERANK、无歧义 ASK、
    RERANK 后重检索等合法性校验
  - `StubPlanner`：**CPU 确定性测试替身，不是 Qwen**
  - `run_planner(...)`：Observation → Action → Tool → Observation 循环记录
- `recommendation/demo/planner_service_executor.py`
  - RETRIEVE：调用现有 `DemoAgentService.run_shopping_turn`
  - RERANK：调用现有 M5 → M6.1 `_m5_soft_rerank`
  - FINALIZE：返回 service 已产生/校验的候选
  - 不生成候选、不绕过 role/price/catalogue provenance 校验

默认 Rule Agent 路径未改动；Planner 仅通过显式构造使用。

## 2. 真实 CPU 多步轨迹

轨迹：`P3D_PLANNER_TRACE.jsonl`

执行配置：

```text
AGENTRECX_STATE_MODE=apply_constraints
AGENTRECX_R1_DYNAMIC_RETRIEVAL=on
AGENTRECX_M5_SOFT_RERANK=on
```

步骤：

| Step | Action | Tool | Observation after |
|---:|---|---|---|
| 1 | RETRIEVE | `run_shopping_turn` | 5 个真实候选 |
| 2 | RERANK | `_m5_soft_rerank`（M6.1） | 候选 5，M6.1 applied=false（该候选无受支持 soft evidence） |
| 3 | FINALIZE | `finalize` | 终止，final 5 IDs |

状态证据：

```text
state_version / state_digest = 5a2b3b768f3cb4c24b510444b57de6affe616d7c2851eab751f93e9be45dbe37
M5 mode = apply_constraints
M5 hard_constraint_count = 1, kinds=[price_max]
R1 actual constraints = [[price_max, 30]]
```

R1 确实收到 `price_max=30`，不是空约束。

## 3. 测试

- `tests/test_planner_contract.py`：Planner action 序列、合法性、max steps、空候选拒绝。
- `tests/test_p3c_final_merge.py`：merge 语义。
- `tests/test_p3c_budget_cross_layer.py`：四轮 Stub API，断言 R1 实际约束
  `30 → 50 → None → None`。

执行结果：

```text
tests/test_planner_contract.py + tests/test_p3c_final_merge.py
7 passed

tests/test_p3c_final_merge.py + tests/test_p3c_budget_cross_layer.py
4 passed
```

## 4. 限制

- Planner 是 **Stub CPU planner**，不是真实 Qwen，不能声称模型已具备自主规划能力；
- RERANK 调用现有 M6.1，但本轮候选没有支持 soft evidence，`applied=false`；
- Planner 包装的是现有 `run_shopping_turn` 工具，不是新的推荐执行器；
- 未接入默认 API，默认 Rule Agent 和 Serving V1 不变；
- 未训练 SFT/RL，未启动 GPU。

## 5. 清理与保留

保留：

- `tests/test_planner_contract.py`（新增行为）
- `tests/test_p3c_final_merge.py`
- `tests/test_p3c_budget_cross_layer.py`（跨层预算回归）

未删除：

- 既有 P3C 轨迹/报告/数据，避免降低可复现性和覆盖证据；
- 临时 Stub/验收脚本位于系统临时目录，不在仓库根目录新增调试脚本。

## 6. 下一步（需 GPU）

有 GPU 后可用真实 Qwen Planner 替换 `StubPlanner`，保持同一 Action Contract / executor，
只替换 planner 输出；本轮不声称已完成。

## 交付

- `P3D_PLANNER_PROTOTYPE_REPORT.md`
- `P3D_PLANNER_TRACE.jsonl`
- `recommendation/demo/planner_contract.py`
- `recommendation/demo/planner_service_executor.py`
- `tests/test_planner_contract.py`

未修改 Serving V1/Gold/Evaluator，未训练/启动 GPU，未运行完整 Benchmark，未 RL，
未 commit/push/deploy。
