# P7-B Qwen Base vs Planner SFT-v0 真实 Dev 对照报告

## 0. 结论先行

- 硬件：RTX 4090D 24GB，CUDA 可用；Qwen3.5-9B 与 Planner SFT-v0 adapter 均存在。
- Soft-State：**确认存在 M5 + M9 两条污染路径，已最小修复**；CPU 回归 14 passed。
- 真实 6-task Dev 已跑：**Rule P6.4 / Base / SFT-v0 各 6 任务**。
- 结果：三档 `VerifiedTaskSuccess` 都是 **0.5**；但失败任务不同。
- 结论：**没有证据表明 Base 或 SFT-v0 优于 Rule**；单次 6-task 结果不构成统计显著性。
- P7 质量状态仍 **NOT ESTABLISHED**；SFT-v1 失败案例**不足以直接开始**。

## 1. 运行环境与数据

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D, 24564 MiB |
| Qwen | `/root/autodl-tmp/models/Qwen3.5-9B-ms`（4 shard） |
| SFT adapter | `.../qwen3_5_9b_planner_lora_v0/adapter` |
| Base server | `127.0.0.1:8009`，无 adapter |
| SFT server | `127.0.0.1:8008`，加载 adapter |
| 本地 Runner | `experiments/p6_benchmark/run_dynamic_dev_p64.py` |
| 协议 | 6 Dev tasks；k=5；max_steps=6；每 Agent run max_tool_calls=4；Fusion `rrf3`；`state_mode=apply_constraints` |
| 评分 | promotion 后 `response.recommendations`；不使用 Action 格式/工具成功/诊断数量替代任务成功 |

运行产物：

- Rule：`runs/p6_4_dynamic/dev_20261107_160922/`（P6.4，未重跑）
- Base 正式：`runs/p6_4_dynamic/base_20261107_183429/`
- SFT-v0 正式：`runs/p6_4_dynamic/sft_20261107_182912/`
- Base budget 单任务诊断重跑：`runs/p6_4_dynamic/base_20261107_183800/`（仅用于稳定性判断，**不替换**正式 6-task 结果）

## 2. Soft-State 污染与修复

### 2.1 确认的问题

Planner 改写后的 RETRIEVE query 原先会作为 `payload.message` 进入 Agent：

1. `_execute_plan` 从 query 抽取 M5/M9 preference；
2. Agent graph 的 M9 persistence 从 `state["user_message"]` 再次写入；
3. 重复 RETRIEVE 因此可能新增、重复、或重新激活已撤回偏好；
4. role target 也可能被 query 改写。

Base 早期正式运行中 `preference_retract` 的 turn0 `lighter_active=false`，直接证明了污染存在；SFT 同一任务为 true，说明这是 query/记忆路径差异而非任务本身。

### 2.2 修复

仍允许 Planner 自由改写 query，query 继续用于检索；只切断 query 对用户状态的写入：

- `agent_service.py`：
  - 新增 `planner_retrieval` / `apply_user_state` / `state_message`；
  - 第一次 Planner RETRIEVE 使用 `state_message`（真实用户文本）做一次 M5/M9 状态应用；
  - 后续 RETRIEVE/RERANK 不抽取、不写 M5/M9；
  - Planner query 不再改写 role target。
- `agent/pipeline.py` + `agent/state.py` + `control/loop.py`：
  - 新增 `memory_user_message` 与 `skip_memory_persist` seam；
  - M9 写入只使用真实用户 turn，后续 Planner 步骤显式跳过。
- `planner_service_executor.py`：
  - `state_message` 与 `_state_applied` 记录本轮 canonical 状态；
  - 第一次 RETRIEVE 应用一次，后续不再应用。
- `tests/test_p7_b_soft_state.py`：
  - 撤回偏好不被 Planner query 重新激活；
  - 重复 RETRIEVE 不会改写 M9；
  - 第一次 RETRIEVE 会把真实用户 turn 的偏好写入 M9，后续改写 query 不会改 M9。

测试：**14 passed**（soft-state 2 + planner contract 11 + role target 1）。  
源码摘要：`P7_B_SOFT_STATE_SOURCE_DIFF.txt`

## 3. 6-task 逐任务结果

### 3.1 状态表

| 任务 | Rule P6.4 | Base | SFT-v0 |
|---|---|---|---|
| budget_update | PASS | **FAIL**（Planner 空 action） | PASS |
| preference_retract | UNKNOWN（1 个 role-UNKNOWN） | **PASS** | UNKNOWN（1 个 role-UNKNOWN） |
| goal_switch | PASS | PASS | PASS |
| role_constraint | PASS | PASS | PASS |
| normal_termination | FAIL（无展示） | FAIL（FINALIZE contract error） | FAIL（第二 turn FINALIZE contract error） |
| insufficient_candidates | UNKNOWN（catalogue 价格未知） | UNKNOWN（catalogue 价格未知） | UNKNOWN（catalogue 价格未知） |

计数：

| Arm | PASS | FAIL | UNKNOWN |
|---|---:|---:|---:|
| Rule | 3 | 1 | 2 |
| Base | 3 | 2 | 1 |
| SFT-v0 | 3 | 1 | 2 |

Base 的 `budget_update` 在正式 6-task 运行中失败；随后单独诊断重跑同一任务为 PASS。  
**不把诊断 PASS 合并进正式结果**；这是 Base Planner 输出/执行不稳定的证据，而不是可复现的稳定失败。

### 3.2 最终真实展示 ID

| 任务 | Rule | Base | SFT-v0 |
|---|---|---|---|
| budget_update | `B095RLKB7Q, B08XJQ9HPX` | `[]`（执行失败） | `B095RLKB7Q, B08XJQ9HPX` |
| preference_retract | `B0BX5QFWQN, B091VYKC96` | `B0B6Z7PTZ8, B0BX5QFWQN, B09CJN4XGM, B0BFR4VJTW, B08THB27YF` | `B0BX5QFWQN, B091VYKC96` |
| goal_switch | `B0BFR4VJTW, B0BCJ6LBJ4` | `B0BFR4VJTW, B0BCJ6LBJ4, B08TQQ2VNG, B07HF7M273, B094ZG2YK5` | `B0BFR4VJTW, B0BCJ6LBJ4` |
| role_constraint | `B07CRN6L4Q` | `B07CRN6L4Q` | `B07CRN6L4Q` |
| normal_termination | `[]` | `[]` | `[]` |
| insufficient_candidates | `[]` | `[]` | `[]` |

- SFT-v0 在 budget / pref / goal / role 四个任务上的最终 ID 与 Rule **完全相同**（均为 Rule 的候选序列）。
- Base 的最终列表不同（budget 为 5 项；goal 为 5 项）。
- FINALIZE 到真实 `response.recommendations` 的映射：Base/SFT 所有已记录展示列表的 **identity mismatch = 0**；未出现 `planner_identity_mismatch`。执行失败的任务没有最终展示，不参与映射统计。

### 3.3 Action 序列 / RERANK / 重试 / 工具调用 / 延迟

| 任务 | Base actions | SFT actions | Base RERANK/重复 RETRIEVE | SFT RERANK/重复 RETRIEVE | Base tools / lat | SFT tools / lat |
|---|---|---|---|---|---|
| budget | 无记录（失败） | R,RK,F ×3 turns | - / - | 3 / 0 | 0 / 0s | 8 / 36.9s |
| preference | R,RK,F; R,F | R,RK,F; R,RK,F | 1/0 / 0 | 2/0 / 0 | 5 / 20.0s | 6 / 23.7s |
| goal | R,F; R,F | R,RK,F; R,F | 0/0 / 0 | 1/0 / 0 | 4 / 14.2s | 4 / 15.8s |
| role | R,F; R,F | R,RK,F; R,RK,F | 0/0 / 0 | 2/0 / 0 | 4 / 16.9s | 4 / 19.7s |
| normal | 无记录（失败） | R,RK,F（仅第一 turn） | - / - | 1/0 / 0 | 0 / 0s | 2 / 11.2s |
| insufficient | R; R,ASK | R; R | 0/0 / 0 | 0/0 / 0 | 5 / 9.6s | 5 / 11.3s |

- 每个已记录 turn 的工具调用数最大为 **3**，均未超过单次 Agent run 的 `max_tool_calls=4`；正式运行中没有重复相同 RETRIEVE。
- RERANK：Base 1 次、SFT 9 次被调用；**applied=0**，`ranking_reason=no_active_m5_soft_preference`，没有观察到 RERANK 带来的排序变化。
- 延迟（本地 runner + 隧道 + Qwen server）：
  - Base：60.6s / 8 个已记录 turn（均值 7.58s/turn）；
  - SFT：118.6s / 12 个已记录 turn（均值 9.88s/turn）；
  - Rule 是 CPU 本地，100.7s / 13 turn，不能与远端 Qwen 做同口径延迟比较。
- Base/SFT 均各自运行在独立进程中，task 独立 Session；Rule 未重跑。

### 3.4 preference_retract 单任务前置检查（补充）

在正式 6-task 之后补做 C.3 指定的 `preference_retract` 单任务检查：

- Base：`runs/p6_4_dynamic/base_20261107_184241/`
- SFT-v0：`runs/p6_4_dynamic/sft_20261107_184708/`

| 检查项 | Base | SFT-v0 |
|---|---|---|
| 最终状态 | PASS | UNKNOWN（1 个 role-UNKNOWN） |
| M5 turn0 lighter active | true | true |
| M5 turn1 lighter active（撤回后） | false | false |
| role_target | water_filter/main（正确） | water_filter/main（正确） |
| Action turn0 | RETRIEVE, RETRIEVE, RERANK, FINALIZE | RETRIEVE, RERANK, FINALIZE |
| Action turn1 | RETRIEVE, FINALIZE | RETRIEVE, RERANK, FINALIZE |
| 重复 RETRIEVE | turn0 1 次（同一 query） | 0 |
| FINALIZE ID 映射 | 两 turn 均与真实 recommendations 一致 | 两 turn 均一致 |
| identity mismatch | 0 | 0 |
| tool calls / 延迟 | 8 / 29.5s | 6 / 19.1s |

补充检查确认：

- Planner query 未污染 M5；turn0 的 lighter preference 存在，turn1 撤回后不存在；
- role_target 正确传入并在两 turn 保持；
- 有 FINALIZE 的两个 turn，商品 ID 均与真实展示列表一致；
- Base 在该任务出现一次重复 RETRIEVE（同一 query），但没有造成状态错误；正式 6-task 该任务无重复。

## 4. 指标对照

| Metric | Rule P6.4 | Base | SFT-v0 |
|---|---:|---:|---:|
| VerifiedTaskSuccess | 0.5000 | 0.5000 | 0.5000 |
| HardConstraintViolationRate | 0.0000 | 0.0000 | 0.0000 |
| ProductRoleCorrectness | 0.8571 | 1.0000 | 0.8571 |
| PreferenceUpdateConsistency | 0.8333 | 0.5000 | 0.6667 |
| EligibleCandidateCoverage | 0.6667 | 0.5000 | 0.6667 |
| ToolCallsPerTask | 5.1667 | 3.0000 | 4.8333 |
| UnknownRate | 0.3333 | 0.1667 | 0.3333 |

说明：

- RoleCorrectness 只统计**实际展示**商品；Base 的 1.0 不代表 Base 总体更好，因为 Base 有两个任务没有展示，未被纳入分母。
- Base 的 ToolCalls 较低主要是因为 budget/normal 在执行早期失败、没有工具调用，不可解释为效率优势。
- 三档均无 hard-constraint violation；没有模型因违反价格/角色硬约束而展示不合格商品。
- 这是单次 6-task Dev 结果，**不宣称统计显著性**。

## 5. 失败归因

### 5.1 真实 Planner 决策/契约错误

1. **Base budget_update**：`HttpPlannerError: unknown planner action ''`，Planner 返回空 action，发生在任何工具调用之前。  
   随后同任务诊断重跑 PASS，说明 Base 存在 **输出/执行不稳定**，不能把它当作稳定固定失败。
2. **Base normal_termination**：`PlannerContractError: FINALIZE is not allowed now`；在无可 finalize 候选时提前 FINALIZE，任何展示发生前失败。
3. **SFT-v0 normal_termination**：第一 turn 成功（4 个候选）；第二 turn 出现 `FINALIZE is not allowed now`，Planner 在候选状态不满足契约时仍调用 FINALIZE。  
   这是真实的 Planner / Action 契约错误，不是候选检索失败。

### 5.2 共同工具或商品证据限制

1. **preference_retract UNKNOWN**（Rule 与 SFT）：最终一个商品 role-UNKNOWN；这是 catalogue 元数据/role evidence 的限制，不是 Planner 选择错误。SFT 的 M5 preference 状态本身是正确的。
2. **insufficient_candidates UNKNOWN**：catalogue 有 213 个 role-matching 商品、78 个价格未知，Eligibility 无法判定；三个 arm 都无法把“存在但价格未知”变成可验证的合格商品。Base 选择 ASK，SFT 没有 ASK，但二者都没有伪造合格展示。
3. **normal_termination 的共同底层问题**：Rule 也失败于“catalogue 有合格商品但没有展示”，说明 role gate / 候选证据链是共同限制；Base 在此之上还叠加了 Planner 错误，SFT 在第二 turn 变成契约错误。
4. **RERANK 无 applied evidence**：Planner 多次调用 RERANK，但 M5 没有 active soft preference，`catalogue` 身份与排序不变；这不是 Planner 质量收益。

### 5.3 不是错误的项

- Base 的 preference_retract、goal_switch、role_constraint PASS；
- SFT 的 budget、goal_switch、role_constraint PASS；
- 所有最终展示均通过 role/价格硬约束检查；
- FINALIZE ID 映射未发现身份错位。

## 6. 质量结论

- Base 与 SFT-v0 的 `VerifiedTaskSuccess` 都是 **0.5**，没有胜过 Rule 的 0.5；
- SFT-v0 的通过集合与 Rule 基本一致（budget/pref/goal/role 的最终 ID 也完全相同），呈现明显的 Rule-like planner 行为；
- Base 的通过任务不同，且 budget 出现不可复现的空 action；当前只能判定为 **不稳定**，不能判定为质量提升；
- 三档的 hard-constraint violation 都是 0；
- 单次 6-task、无随机重复，不能给出统计显著性结论；
- **P7 质量状态：NOT ESTABLISHED。**

## 7. 是否足以开始 Planner SFT-v1

**目前不足。**

可用失败/分析案例：

- Base：空 action（非稳定复现）；
- Base：提前 FINALIZE；
- SFT：第二 turn 提前 FINALIZE；
- 共同：role-UNKNOWN 商品、价格未知导致的 insufficient UNKNOWN。

问题：

- 只有一次 6-task 运行，没有多 seed / 重复；
- Base 的 budget 失败不可稳定复现，不能直接作为训练标签；
- 共同工具/证据限制不是 Planner SFT 的正样本目标；
- 不能把失败轨迹自动当 SFT 正标签。

建议下一阶段先做：

1. 对 Base / SFT 各重复运行 2–3 次，确认失败稳定性；
2. 把“空 action”“无候选 FINALIZE”做成 Planner 格式/契约压力测试集；
3. 由人工/规则区分 Planner 决策错误与共同证据限制；
4. 再决定是否构造 SFT-v1 数据；不要自动用失败轨迹当正标签。

## 8. 复现命令

```powershell
# Base
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm base --planner-url http://127.0.0.1:18009

# SFT-v0
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm sft --planner-url http://127.0.0.1:18008
```

服务端启动命令见 P7-B 阻断报告 §6；本轮使用 RTX 4090D，Base 8009 / SFT 8008。
