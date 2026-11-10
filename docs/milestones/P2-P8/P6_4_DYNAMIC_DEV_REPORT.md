# P6.4 Dynamic Dev 输入与角色状态传递修复报告

## 1. 状态

本轮只完成两个已证实问题：role_target 传递修复、6 个 Dynamic Dev 的输入/Runner 整理。  
未运行 Holdout、Qwen、GPU、训练、Fusion-20k；未修改 Serving V1 / Eligibility / 角色判断规则 / Gold；未 commit。

真实执行结果（6 个 Dev）：

```text
PASS    3   budget_update, goal_switch, role_constraint
FAIL    1   normal_termination
UNKNOWN 2   preference_retract, insufficient_candidates
```

## 2. role_target 修复

问题位置：HTTP shopping 路径把 `session_id = payload.user_key` 与 `task_id = session.session_id`
分开传入；`run_shopping_turn` 把角色目标写入 `_role_current_targets[task_key]`（task_key=task_id），
而 `_execute_plan` 通过 `_role_target_for(payload.user_key)` 读取。规则路径没有 proposal 分支
已有的镜像步骤，因此 `role_target=null`。

实际修复（`recommendation/demo/agent_service.py`）：

1. `run_shopping_turn` 在 `_update_role_target` 之后、每个 turn 都调用
   `_mirror_role_target_to_user_key(task_key, payload.user_key)`；
2. 新增该方法：仅把当前 task 的 `_role_current_targets[task_key]` 与
   `_role_main_targets[task_key]` 镜像到该 Session 的 `user_key`；
3. 不修改角色判断、accessory gate 或 Eligibility；镜像键是 server-owned、每个 demo session
   独立的 `user_key`，不共享给其他 Session。

源码 diff：`P6_4_DYNAMIC_DEV_SOURCE_DIFF.txt`。

针对性回归：`tests/test_p6_4_role_target_transfer.py`，1 passed。测试证明：
- 首轮建立 `water_filter/main`；
- 第二轮无类型词（“Show me more options.”）仍能读到同一 role target；
- 第二个 Session 的 user_key 不同且没有继承第一个 Session 的 role target。

## 3. Dev 输入与共享状态执行方式

新增 `data/p6_benchmark_v0/dynamic_dev_p64.jsonl`（保留原 dev 文件，Holdout 未改）：

- 6 个任务使用相同初始 history：
  ```text
  B019YN5ZSA, B001U9YK8I, B01BXDLZ7O, B0BGMX9FCF, B07GYZSV29
  ```
- 初始 M5 为空；
- 操作式 turn 改为自然语言购物表达（例如
  `Actually, my budget is now $60. Please keep looking for a water filter.`，
  `Forget about the lighter options. I still need a water filter.`，
  `Actually, I would like a water bottle instead.`）；
- 每个 turn 带 `state_operations`（set_goal / set_price_max / remove_price_max /
  add_preference / retract_preference / set_role）和 `expected_effective_state`；
- `evaluation_protocol = planner_only_controlled_evaluation`。

Runner `experiments/p6_benchmark/run_dynamic_dev_p64.py`：

- 复用 Serving V1 Fusion（`fusion_rrf`, sources=sequential/two_tower/catalog_lexical, RRF K=60,
  head=100）、DemoAgentService、M5 apply_constraints、CandidateLedger、Eligibility、商品角色规则；
- 每轮先用 runner 的确定性 `state_operations` 更新 canonical M5 状态，再把该状态中的
  hard constraints 通过 trusted `AgentRecommendRequest.constraints` 交给 Agent；
- natural-language message 仍发给 Agent；soft preference / goal / role 使用服务端确定性
  role/soft 状态抽取，不把“Parser 理解能力”计入 Planner 分数；
- 期望状态仅用于执行控制与评估，**不向 Agent 传入 expected state 或 Gold 商品**；
- 最终评分只使用 promotion 后的 `response.recommendations`；
- 价格缺失无法判定目录是否存在合格商品时，仍返回 UNKNOWN。

## 4. 6 个任务的真实最终状态与候选

| task | 结果 | 最终展示候选（promotion 后） | 最终 M5 / role_target | 评分依据 |
|---|---|---|---|---|
| budget_update | **PASS** | B095RLKB7Q (MATCH, $228.95), B08XJQ9HPX (MATCH, $29.89) | price_max 已移除；water_filter/main | 2/2 角色 MATCH；无价格违反；状态转换 40→60→remove 全部正确 |
| preference_retract | **UNKNOWN** | B0BX5QFWQN (MATCH), B091VYKC96 (UNKNOWN) | 无 hard constraint；water_filter/main；lighter 已撤回 | 1/2 MATCH，1 个角色证据 UNKNOWN → 不判成功 |
| goal_switch | **PASS** | B0BFR4VJTW (MATCH), B0BCJ6LBJ4 (MATCH) | 无 hard constraint；filtered_bottle/main | 2/2 MATCH；role target 已切换 |
| role_constraint | **PASS** | B07CRN6L4Q (MATCH, $199.95) | 无 hard constraint；water_filter/accessory | 1/1 accessory MATCH；role target 到达过滤层 |
| normal_termination | **FAIL** | **[]** | price_max=50；cookware/main | 目录有 258 个独立验证 eligible 商品，但最终展示为空 |
| insufficient_candidates | **UNKNOWN** | **[]** | price_max=1；role_target=None（无候选，未产生角色选择） | catalogue 213 个 role-match 中 78 个价格未知，不能证明无合格商品；Agent 追问价格而非报告无合格商品 |

说明：`normal_termination` 的 Agent 文本称 “Top 4 candidate(s)…”，但最终
`response.recommendations=[]`（角色过滤后无 MATCH）；评分严格以最终展示列表为准，因此 FAIL。
`insufficient_candidates` 没有展示商品且保留了 $1 约束，但 catalogue 价格证据不足，按规则 UNKNOWN。

## 5. P6 动态指标（现有 `dynamic_metrics`）

```text
VerifiedTaskSuccess         = 0.500000   # 3/6
HardConstraintViolationRate = 0.000000   # 最终展示项没有可证实违反
ProductRoleCorrectness      = 0.857143   # 6/7 最终展示项 MATCH
PreferenceUpdateConsistency = 0.833333   # 5/6；insufficient 因 role evidence 缺失保守记 false
EligibleCandidateCoverage   = 0.666667   # 4/6
ToolCallsPerTask            = 5.166667   # 31/6
UnknownRate                 = 0.333333   # 2/6
```

## 6. Tool Calls 统计口径

- `max_tool_calls=4` 是**单个 Agent run / 单个 turn** 的上限；
- `ToolCallsPerTask` 是同一任务所有 turn 的 `observation.tool_calls_used` 之和再平均；
- 本轮 6 任务共 13 个 turn、31 次 tool call；每 turn 为 2–3 次，均未超过 4；
- 每任务总和：8 / 6 / 4 / 4 / 4 / 5。P6.3 的 5.0 同为此口径，不是协议违规。

## 7. 是否能进入 Qwen Base / SFT-v0 对照 / P7

**Qwen 对照：LIMITED（可以开始接线，但尚未完成公平对照）。**

已具备：
- role_target 修复及回归；
- 6 个 Dev 任务的自然语言输入、统一初始 history/M5 和确定性状态转换；
- Runner 已将 hard constraints 以 trusted 请求字段统一交给 Agent，评分只用最终展示列表。

尚缺：
- 本轮没有运行 Qwen Base/SFT-v0（约束禁止启动 GPU/Qwen）；Planner 执行器还没有接到同一个
  受控 Runner；
- Qwen Planner 的多步 RETRIEVE/RERANK 路径是否保持同一 M5/role 状态，需要在 GPU 环境验证；
- `normal_termination` 的真实失败和 `preference_retract`/`insufficient` 的 UNKNOWN 应保留，
  不修改规则制造 PASS；
- Holdout 尚未运行。

**P7：NOT READY。** 目前只完成了 Rule Agent 侧的输入/状态修复与真实 baseline；Qwen 公平对照、
更大质量评估和 Holdout 都还没有完成，不能进入 P7。

## 8. 交付

- `recommendation/demo/agent_service.py`（role_target 镜像修复）
- `tests/test_p6_4_role_target_transfer.py`（针对性回归，1 passed）
- `data/p6_benchmark_v0/dynamic_dev_p64.jsonl`（6 个 Dev 的自然语言/受控状态输入）
- `experiments/p6_benchmark/run_dynamic_dev_p64.py`（共享状态 Runner）
- `P6_4_DYNAMIC_DEV_REPORT.md`
- `P6_4_DYNAMIC_DEV_SOURCE_DIFF.txt`
- 运行产物：`runs/p6_4_dynamic/dev_20261107_160255/`（2-task 最小验证）
  `runs/p6_4_dynamic/dev_20261107_160922/`（6-task 完整运行）
