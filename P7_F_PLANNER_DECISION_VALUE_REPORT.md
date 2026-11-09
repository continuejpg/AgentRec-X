# P7-F Planner 实际推荐决策收益分析

## 0. 结论

**有限 GO：Query 选择确有可验证的推荐结果效应；但完整 Planner SFT-v1 目前 NO-GO。**

理由：

- Base 在 `preference_retract` 用更短的 Query（`water filter`）取得 5 个全部 role-MATCH 的最终商品并 PASS；Rule/SFT 使用包含撤回文本的 Query，只得到 1 MATCH + 1 UNKNOWN，任务 UNKNOWN。
- `goal_switch` 中 Base/SFT 的干净新目标 Query 也比 Rule 的 old+new Query 多找到 role-MATCH 商品。
- 但这是 **Query 文本效应**，不是 RERANK 或更复杂 Planner 动作的收益；Base 自己第一 turn 的短 Query 也出现过更少候选，说明不是单调的“Planner 更聪明”。
- SFT-v0 的 RERANK 在实际存储轨迹中为 **11 次 invoked / 0 applied / 0 consumed**（P7-E 文字报告写 9 次，按 `turns.jsonl` 重新计数为 11）；Planner 在做 RERANK 决策时看不到 active soft evidence，无法预判它一定无效。
- 当前动作空间主要只有 RETRIEVE Query 有实际效果；RERANK 是盲动作，FINALIZE/ASK 不改变候选。这不支持训练一个广义 Planner SFT-v1，但支持一个很小的 Query 策略开发/数据构造任务。

因此：

- **GO（窄范围）：** 针对 Query 选择整理受控对照数据，进入 targeted Query-policy 开发；
- **NO-GO（完整 SFT-v1）：** 目前证据不足以训练通用 Planner。

## 1. 数据来源

只复用 P7-E 同版本真实轨迹：

- Rule：`runs/p6_4_dynamic/rule_20261107_201603/`
- Base：`runs/p6_4_dynamic/base_20261107_202024/`
- SFT-v0：`runs/p6_4_dynamic/sft_20261107_203139/`

没有重新运行、没有补造日志；source-level raw top-N 在现有记录中仍为 **UNVERIFIED**。

## 2. preference_retract：Base 为什么找到更多 MATCH

### 2.1 关键 Query 与结果

| Arm | turn | Planner / Agent Query | role MATCH | final |
|---|---|---|---:|---|
| Rule | 0 | 用户原文：`I need a water filter...` | 5 | 5 |
| Rule | 1 | 含撤回文本：`Forget about... water filter` | 1 | 2（1 MATCH + UNKNOWN） |
| Base | 0 | `water filter lighter options` | 3 | 1 |
| Base | 1 | **`water filter`** | **5** | **5（全部 MATCH）** |
| SFT | 0 | 用户原文 | 5 | 5 |
| SFT | 1 | 含撤回文本：`Forget about... water filter` | 1 | 2（1 MATCH + UNKNOWN） |

Base 最终 ID：

```text
B0B6Z7PTZ8, B0BX5QFWQN, B09CJN4XGM, B0BFR4VJTW, B08THB27YF
```

Rule/SFT 最终 ID：

```text
B0BX5QFWQN, B091VYKC96
```

### 2.2 因果判断

Base 的额外 MATCH 不是来自 Role gate、Eligibility 或来源选择：

- 三组使用同一 CandidateLedger / Eligibility / role gate；
- Base 和 SFT 都通过同一 `ServicePlannerExecutor`；SFT 的 Query 几乎等于 Rule 的用户原文，最终结果也与 Rule 相同（1 MATCH + 1 UNKNOWN）；
- Base 的 turn1 Query 去掉了 `Forget about ...` 与旧偏好词，只保留 `water filter`；同一工具返回更多 MATCH 候选。

**结论：这是 Planner Query 文本的可验证效果（LIMITED）。**  
但 Base 的 turn0 短 Query 只得到 1 个 final，而 Rule/SFT 原文得到 5 个；因此 Base 的收益是“最终这一 turn 的 Query 较好”，不是整体策略稳定优于 Rule。

## 3. goal_switch：额外 MATCH 商品的来源

| Arm | turn1 Query | role MATCH | final match IDs |
|---|---|---:|---|
| Rule | `water filter for hiking + water bottle`（旧+新） | 2 | `B0BFR4VJTW, B0BCJ6LBJ4` |
| Base | `water bottle`（同一 Query 重复 3 次） | 5 | `B07HF7M273, B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |
| SFT | 新目标完整用户句 | 5（role feedback）/ 4 final | `B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |

Base/SFT 相比 Rule 额外展示的 MATCH 商品：

- `B094ZG2YK5`
- `B0987D9PKM`
- Base 另含 `B07HF7M273`

这些是真实 role-MATCH，不是 UNKNOWN 充数。  
但三组任务都 PASS；**更多 MATCH 商品本身不等于推荐质量更好**，也没有改变任务级成功状态。

Query 差异仍是主要解释：Rule 把旧目标 `water filter for hiking` 和新目标混在一起；Base 用干净 `water bottle`，SFT 用新目标完整句。

## 4. RERANK 为什么完全无效

### 4.1 实际计数

按 P7-E 存储的 `turns.jsonl` 重新计数：

| Arm | RERANK invoked | applied | consumed soft evidence |
|---|---:|---:|---:|
| Rule | 0 | 0 | 0 |
| Base | 1 | 0 | 0 |
| SFT-v0 | **11** | **0** | **0** |

> P7-E 文字报告曾写 SFT 9 次；本次按持久化 planner steps 重新计数为 11 次。任务结论不变：全部无效。

SFT 的 11 次分布：

- budget_update：3
- preference_retract：2
- goal_switch：2
- role_constraint：2
- normal_termination：2

每次 tool observation：

```text
active_soft_preference_count = 0
consumed_soft_preference_count = 0
rerank_applied = false
ranking_reason = no_active_m5_soft_preference
```

### 4.2 Planner 事前能否判断？

**不能，至少不能从当前可见 Observation 可靠判断。**

- `ServicePlannerExecutor._retrieval_feedback()` 只暴露 raw/RRF/role/hard-constraint 计数，不暴露 active soft preference 的有无；
- `PlannerObservation.m5_state` 只有 `preference_count`、`constraint_kinds` 等，没有“soft preference 是否可被 M6.1 消费”的字段；
- RERANK 的 `active_soft_preference_count`、`ranking_reason` 只在执行后的 tool observation 中出现；
- RERANK action 没有候选子集或偏好参数，执行器固定调用 `_m5_soft_rerank(existing=())`。

因此 Planner 在做 RERANK 决策时看不到未来应用结果，无法区分“可能有 soft evidence”与“必然是 no-op”。这是 **动作/观察空间设计问题**，不是可以靠现有 Observation 预判的 Planner 错误。

## 5. 当前动作空间是否足以学习推荐策略？

**只够支持很窄的 Query 策略，不足以支持完整 Planner SFT-v1。**

| Action | 实际自由度 | 问题 |
|---|---|---|
| RETRIEVE | 可自由写 query 文本 | 当前唯一有可验证结果差异的动作；但只能改变文本，不能选来源/权重 |
| RERANK | 无参数，固定消费 M5 soft evidence | 观察不到证据；本实验 12 次调用全部 no-op（Base 1 + SFT 11） |
| FINALIZE | 终止并展示当前候选 | 高概率空候选时非法；不能补候选 |
| ASK | 提问 | 可用于证据不足，但不改变候选 |

没有 action 能：

- 选择 history / catalog_search / sequential / two_tower 的来源组合；
- 调整融合权重或当前意图与历史兴趣的结合；
- 指定 RERANK 使用哪一类偏好或候选子集；
- 给 Planner 提供 active soft evidence 的动态可见性。

所以：**Query 级别的策略学习有空间；完整 agent policy 学习没有足够可执行自由度。**

## 6. 五个问题的直接回答

### 1) Base 为什么在部分任务找到更多 MATCH？

因为 Base Planner 产生了更短、更干净、更贴近当前有效目标的 RETRIEVE Query：

- preference：`water filter`（去掉撤回/旧偏好文本）；
- goal：`water bottle`（不再混入旧目标 `water filter for hiking`）。

不是 Role gate/Eligibility 放宽，也不是商品资格变化。

### 2) 是 Planner/Query 的实际效果，还是工具路径/其他因素？

**主要是 Query 文本效果，但为 LIMITED 而非完全受控因果证明：**

- 同工具、同来源路径下，SFT 使用用户原文得到与 Rule 相近结果；Base 使用不同 Query 得到额外 MATCH；
- 没有 counterfactual 固定状态重跑同一 Query，因此不能说“Query 差异是唯一因果变量”；
- 未观察到工具路径本身造成差异。

### 3) RERANK 为什么完全无效？

- 当前 M5 没有 active soft preference，`_m5_soft_rerank` 返回 `ranking=None`；
- Planner Observation 不暴露 active soft evidence，无法事前判断；
- RERANK 无参数，执行器固定消费全局 M5 soft state；
- 结果：Base 1 次 + SFT 11 次全部 applied=0 / consumed=0。

### 4) 动作空间是否足够支持有意义的策略学习？

**仅对 Query 策略足够；对完整 Planner 策略不足。**

### 5) 下一步唯一最值得做什么？

**把 RETRIEVE Query 变成显式、可复用的 Query 策略变量，并收集受控 Query→结果对照数据。**

具体最小方向（本轮不实现）：

- 以当前 effective `role_target.type` + 最新有效 hard constraint 构成 canonical Query 模板；
- 在同一任务/状态/候选来源下记录 canonical Query、用户原文 Query、Planner Query 的 role MATCH / final success；
- 先验证 Query 模板是否稳定优于原文，再决定 SFT-v1 数据构造。

RERANK 在 active soft evidence 不可见之前不应进入训练数据；它目前是无效动作。

## 7. GO / NO-GO

**GO（窄范围，仅 Query 策略）：**  
存在可验证的 Query 决策效果（Base preference PASS、goal 额外 MATCH），值得进入 targeted Query-policy 开发和受控数据构造。

**NO-GO（完整 Planner SFT-v1）：**  
目前只有 6 个 Dev 任务、单次运行；RERANK 不可用；动作空间只有 Query 有效；SFT-v0 相对 Rule/Base 无 Verified 增量；不足以训练通用 Planner。

本轮不实现任何代码、不新增测试、不训练、不运行 Holdout、不重跑 Benchmark、不修改 Serving V1/Gold/Evaluator/Eligibility、不 commit/push/deploy，GPU 实例未关闭。

交付：`P7_F_PLANNER_DECISION_VALUE_REPORT.md`
