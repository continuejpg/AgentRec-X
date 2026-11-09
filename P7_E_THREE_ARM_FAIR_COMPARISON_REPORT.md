# P7-E 三组同版本 Dynamic Dev 对照报告

## 0. 运行版本与结论

三组均在 **P7-D 修复后的同一代码**上运行，未混用 P7-B 旧结果：

- Rule CPU：`runs/p6_4_dynamic/rule_20261107_201603/`
- Qwen Base：`runs/p6_4_dynamic/base_20261107_202024/`
- Planner SFT-v0：`runs/p6_4_dynamic/sft_20261107_203139/`

结论：

- 三组 `VerifiedTaskSuccess` 都是 **0.5**；
- Rule 与 SFT-v0：**3 PASS / 0 FAIL / 3 UNKNOWN**；
- Qwen Base：**3 PASS / 2 FAIL / 1 UNKNOWN**；
- P7-D 明显改善了 `normal_termination` 的召回：Rule/SFT 从空展示/合同失败变为 5 项展示、UNKNOWN；Base 仍在该任务上提前 FINALIZE 失败；
- SFT-v0 没有相对 Base 的可验证 **VerifiedTaskSuccess** 增量（都是 0.5）；它减少了执行失败，但与 Rule 基本一致；
- SFT-v0 仍然执行大量无效 RERANK（9 次 invoked，0 applied，0 consumed）；
- 6 个 Dev 任务不足以宣称统计显著性。

## 1. 六任务逐项结果

| 任务 | Rule | Base | SFT-v0 |
|---|---|---|---|
| budget_update | PASS | **FAIL**（HTTP 空 action） | PASS |
| preference_retract | UNKNOWN | **PASS** | UNKNOWN |
| goal_switch | PASS | PASS | PASS |
| role_constraint | PASS | PASS | PASS |
| normal_termination | UNKNOWN | **FAIL**（非法 FINALIZE） | UNKNOWN |
| insufficient_candidates | UNKNOWN | UNKNOWN | UNKNOWN |

计数：

| Arm | PASS | FAIL | UNKNOWN |
|---|---:|---:|---:|
| Rule | 3 | 0 | 3 |
| Base | 3 | 2 | 1 |
| SFT-v0 | 3 | 0 | 3 |

## 2. 最终展示 ID 与角色证据

| 任务 | Rule final IDs | Base final IDs | SFT final IDs |
|---|---|---|---|
| budget_update | `B095RLKB7Q, B08XJQ9HPX` | `[]` | `B095RLKB7Q, B08XJQ9HPX` |
| preference_retract | `B0BX5QFWQN, B091VYKC96` | `B0B6Z7PTZ8, B0BX5QFWQN, B09CJN4XGM, B0BFR4VJTW, B08THB27YF` | `B0BX5QFWQN, B091VYKC96` |
| goal_switch | `B0BFR4VJTW, B0BCJ6LBJ4` | `B07HF7M273, B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` | `B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |
| role_constraint | `B07CRN6L4Q` | `B07CRN6L4Q` | `B07CRN6L4Q` |
| normal_termination | `B07S8GNB4T, B09FXNJZTV, B08557VLNT, B09YCYNCSK, B091VYKC96` | `[]` | 同 Rule |
| insufficient_candidates | `[]` | `[]` | `[]` |

最终商品角色状态：

| 任务 | Rule MATCH/MISMATCH/UNKNOWN | Base | SFT |
|---|---|---|---|
| budget_update | 2 / 0 / 0 | 0 / 0 / 0（失败） | 2 / 0 / 0 |
| preference_retract | 1 / 0 / 1 | **5 / 0 / 0** | 1 / 0 / 1 |
| goal_switch | 2 / 0 / 0 | **5 / 0 / 0** | **4 / 0 / 0** |
| role_constraint | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |
| normal_termination | 2 / 0 / 3 | 0 / 0 / 0（失败） | 2 / 0 / 3 |
| insufficient_candidates | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

观察：

- **Base 在 preference_retract 找到 5 个全部 MATCH 的商品并 PASS**；Rule/SFT 都有 1 个 role-UNKNOWN，因此为 UNKNOWN。
- **Base 与 SFT 在 goal_switch 找到了 Rule 没有展示的额外 MATCH 商品**（Base 5 个、SFT 4 个，都全部 MATCH；Rule 2 个）。
- normal_termination 三组最终 ID 相同（Rule/SFT），但都有 3 个 role-UNKNOWN；Base 未产生展示。

## 3. Action、Tool 与成本

| 任务 | Rule tool / lat(s) | Base actions / tool / lat(s) | SFT actions / tool / lat(s) |
|---|---|---|---|
| budget | 8 / 8.1 | 无记录（HTTP 失败）/ 0 / 0 | R,RK,F ×3 / 8 / 56.0 |
| preference | 6 / 3.5 | R,R,RK,F; R,F / 8 / 28.7 | R,RK,F ×2 / 6 / 35.7 |
| goal | 4 / 3.7 | R,F; R,R,R,F / 8 / 27.7 | R,RK,F ×2 / 4 / 36.0 |
| role | 4 / 3.8 | R,F ×2 / 4 / 28.8 | R,RK,F ×2 / 4 / 36.1 |
| normal | 4 / 3.0 | 无记录（非法 FINALIZE）/ 0 / 0 | R,RK,F ×2 / 4 / 34.1 |
| insufficient | 5 / 3.5 | R; R,ASK / 5 / 24.8 | R; R / 5 / 23.3 |
| **总计** | **31 / 25.6s / 13 turns** | **25 / 110.0s / 8 turns** | **31 / 221.2s / 13 turns** |

说明：

- `R = RETRIEVE`, `RK = RERANK`, `F = FINALIZE`。
- Base/SFT 的延迟是本地 Runner + SSH 隧道 + 远端 Qwen；Rule 是本地 CPU，**不能直接做延迟比较**。
- 单次 `run_shopping_turn` 仍受 `max_tool_calls=4`；Base 的 goal turn1 累计 6 次 tool calls，来自 3 次 RETRIEVE 的累加，单次 Agent run 仍不超过 4。

### 重复 RETRIEVE / RERANK

| Arm | 重复相同 Query RETRIEVE | RERANK invoked | applied | consumed soft evidence |
|---|---:|---:|---:|---:|
| Rule | 0 | 0（无 Planner） | 0 | 0 |
| Base | 3（preference 1 + goal 2） | 1 | 0 | 0 |
| SFT-v0 | 0 | **9** | **0** | **0** |

SFT-v0 仍在所有 PASS/UNKNOWN 轨迹中大量调用 RERANK，但没有 active soft preference，`applied=0`、`consumed=0`，没有可验证收益。

## 4. 核心指标

| Metric | Rule | Base | SFT-v0 |
|---|---:|---:|---:|
| VerifiedTaskSuccess | 0.5000 | 0.5000 | 0.5000 |
| HardConstraintViolationRate | 0.0000 | 0.0000 | 0.0000 |
| ProductRoleCorrectness | 0.6667 | 1.0000 | 0.7143 |
| PreferenceUpdateConsistency | 0.8333 | 0.5000 | 0.8333 |
| EligibleCandidateCoverage | 0.8333 | 0.5000 | 0.8333 |
| ToolCallsPerTask | 5.1667 | 4.1667 | 5.1667 |
| UnknownRate | 0.5000 | 0.1667 | 0.5000 |

Base 的 RoleCorrectness=1.0 只统计实际展示商品；其两个失败任务没有展示，不能解释为总体角色质量更高。Base 的 ToolCalls 较低也主要因为 budget/normal 早期失败。

## 5. 失败与归因

### Planner / 输出失败

1. **Base budget_update**：`HttpPlannerError: unknown planner action ''`；无工具调用、无 turn record，属于 Planner HTTP 输出/格式错误。
2. **Base normal_termination**：`PlannerContractError: FINALIZE is not allowed now`；在无候选 observation 上提前 FINALIZE，属于 Planner 契约/决策错误。
3. SFT-v0 本轮没有 Planner Contract/HTTP 执行错误。

### 共同工具 / 商品证据限制

1. **normal_termination UNKNOWN**：Rule/SFT 最终 5 项中 3 项 role-UNKNOWN；P7-D 已让 Query 保留 cookware，召回改善（role MATCH 0→2，final 0→5），但商品证据不足导致 UNKNOWN，不是再次丢失 Query 目标。
2. **preference_retract UNKNOWN**：Rule/SFT 的最终列表各有一个 role-UNKNOWN 商品。
3. **insufficient_candidates UNKNOWN**：catalogue 有 213 个 role-matching、78 个价格未知；三个 Arm 都无法验证合格商品，Base 选择 ASK，SFT 未 ASK。
4. **Base normal 仍有 Planner 契约问题**，不能归因于 role gate 或 P7-D Query 修复。

## 6. P7-D 是否消除了共同检索失败？

**部分消除。**

- `normal_termination` Rule：从 FAIL/空展示变为 UNKNOWN/5 项展示；
- `normal_termination` SFT：从上一轮的 Contract 失败变为 UNKNOWN/5 项展示；
- Base：仍在 normal 上提前 FINALIZE，无法进入展示阶段；
- 因此 P7-D 修复了“follow-up 丢失 cookware 目标”这一层；剩余 UNKNOWN 主要是最终 role evidence 不足，不是 Query 目标丢失。

## 7. SFT-v0 是否有可验证增量收益？

**没有清晰的正向增量。**

- 对 Base：SFT 将 Base 的 budget FAIL 和 normal FAIL 变成 PASS/UNKNOWN，且无 Planner Contract/HTTP 错误；但 Base 的 preference_retract PASS 变成 SFT 的 UNKNOWN；
- `VerifiedTaskSuccess` 两者都是 0.5；
- 对 Rule：SFT 的通过任务和 final IDs 与 Rule 基本一致（budget/pref 的 ID 相同，goal 多两个 MATCH 商品，normal 相同），没有出现新的 Verified 成功；
- SFT 的 RERANK 9 次 invoked、0 applied，未产生质量或成本收益；
- 因此目前只有“更少执行失败/更接近 Rule”的弱证据，没有可验证的学习型 Planner 质量增量。

## 8. 下一步是否值得开始 SFT-v1？

**目前不值得直接开始。**

缺少：

1. 更大且稳定的 Dev/验证集，而非仅 6 个任务；
2. 多次重复运行，确认 Base 的空 action / 非法 FINALIZE 是否稳定；
3. 明确区分 Planner 决策错误与 role/price evidence 限制的失败标签；
4. SFT-v0 相对 Rule/Base 的可靠增量指标（当前 Verified 都是 0.5）；
5. RERANK 等无效 action 的成本约束或训练信号；
6. Holdout 前不得用当前 6 个 Dev 结果做统计显著性声明。

## 9. 交付物

- Rule：`runs/p6_4_dynamic/rule_20261107_201603/`
- Base：`runs/p6_4_dynamic/base_20261107_202024/`
- SFT-v0：`runs/p6_4_dynamic/sft_20261107_203139/`

本轮未运行 Holdout、未训练、未扩充 Benchmark、未修改 Gold/Evaluator/Serving V1/Eligibility、未新增 Agent 框架、未 commit/push/deploy。GPU 实例未关闭。
