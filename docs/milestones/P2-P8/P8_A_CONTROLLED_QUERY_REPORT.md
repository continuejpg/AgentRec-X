# P8-A 受控 Query 策略与推荐结果实验报告

## 0. 结论

**简单确定性 Canonical Query 可以复现 Base 在 `preference_retract` 的全部收益；在 `goal_switch` 只能部分复现，Base Query 多 1 个 role-MATCH 商品，但没有改变任务成功状态。**

- preference_retract：
  - Raw → 2 个最终商品（1 MATCH + 1 UNKNOWN）；
  - Canonical `water filter` → **5 个全部 MATCH**；
  - Planner/Base 真实 Query `water filter` → 与 Canonical **完全相同**：5 个全部 MATCH、相同 ID。
- goal_switch：
  - Raw → 4 个最终商品，全部 MATCH；
  - Canonical `filtered bottle` → 4 个最终商品，全部 MATCH；
  - Planner/Base 真实 Query `water bottle` → 5 个最终商品，全部 MATCH；
  - Base/Planner 比 Canonical 多 1 个 MATCH 商品，但三个变体都满足当前任务条件（PASS 等价）。

**结论：Query 差异是实际原因；但 Base 的额外收益很小，且 preference 的收益完全可由确定性 Canonical 规则复现。当前不值得为 Query 策略训练 SFT。下一步应研究可学习的来源选择/融合策略，而不是 Query 文本策略。**

## 1. 实验设置

两个场景、三个变体，共 6 次受控检索；每次使用独立 Session、独立 CandidateLedger、相同 history / Canonical M5 / role_target / Serving V1 Fusion / k=5。

| 项 | 值 |
|---|---|
| state mode | `apply_constraints` |
| Fusion | `rrf3`（history + catalog_search） |
| history | P6.4/P7-E 相同 5 个 ASIN |
| role / M5 | 先执行相同 seed turns，再执行最终受控 Query |
| 最终 Query | 只影响检索；`planner_retrieval=True`, `apply_user_state=False`，不写 M5/M9 |
| 评分 | 只使用最终 `response.recommendations` |

实际检索使用同一 Query 构造路径；记录的是进入检索器的 `query_terms`，不是仅仅输入字符串。

产物：`runs/p8_a/results.json`

### Canonical Query 定义

```text
Canonical = 当前有效 role_target.type + 当前仍有效的 hard constraints
```

- preference_retract：`water filter`（无 hard constraint）
- goal_switch：`filtered bottle`（无 hard constraint）

### 三个变体

| 变体 | preference_retract | goal_switch |
|---|---|---|
| Raw | 最终用户 Turn 原文 `Forget about...` | 目标切换 Turn 原文 `Actually... water bottle...` |
| Canonical | `water filter` | `filtered bottle` |
| Planner | P7-E Base 真实 Query `water filter` | P7-E Base 真实 Query `water bottle` |

## 2. 受控结果

### preference_retract

| 变体 | 实际 query_terms | raw/RRF | 最终展示 | final role | final IDs |
|---|---|---:|---|---|---|
| Raw | `water filter, Forget, about, the, lighter, options., I, still` | 10/15 | 2 | 1 MATCH + 1 UNKNOWN | `B0BX5QFWQN, B091VYKC96` |
| Canonical | `water filter, water, filter` | 10/15 | **5** | **5 MATCH** | `B0B6Z7PTZ8, B0BX5QFWQN, B09CJN4XGM, B0BFR4VJTW, B08THB27YF` |
| Planner | 同 Canonical | 10/15 | **5** | **5 MATCH** | 同 Canonical |

**Canonical 与 Base Planner Query 完全一致：相同 query_terms、相同最终 ID、相同 role 结果。**  
Raw 的差异来自撤回指令文本进入了 query terms；不是状态、来源或 Eligibility 差异。

### goal_switch

| 变体 | 实际 query_terms | raw/RRF | 最终展示 | final role | final IDs |
|---|---|---:|---|---|---|
| Raw | `filtered bottle, water filter for hiking, water bottle, Actually...` | 10/15 | 4 | 4 MATCH | `B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |
| Canonical | `water filter for hiking, water bottle, filtered, bottle` | 10/15 | 4 | 4 MATCH | `B07HF7M273, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |
| Planner | `filtered bottle, water filter for hiking, water bottle, water, bottle` | 10/15 | **5** | **5 MATCH** | `B07HF7M273, B0BFR4VJTW, B0BCJ6LBJ4, B094ZG2YK5, B0987D9PKM` |

- Canonical 与 Raw 的最终数量都是 4，但集合不同：Canonical 有 `B07HF7M273`，Raw 有 `B0BFR4VJTW`。
- Planner/Base 同时包含两者，因此多 1 个 MATCH 商品。
- 三个变体都只展示 role-MATCH 商品；按当前任务条件均可判定为 PASS。
- `candidate_filter.final_candidate_count` 在 goal 变体中记录为 5；实际 `response.recommendations` 展示为 Raw/Canonical 4、Planner 5，本报告以真实展示 ID 为准。
- 因为候选池相同而展示数量/集合不同，**多 1 个商品不等于质量更高**。

### 任务条件判定

| 场景 | Raw | Canonical | Planner |
|---|---|---|---|
| preference_retract | UNKNOWN（1 个 role-UNKNOWN） | **PASS**（5 MATCH） | **PASS**（5 MATCH） |
| goal_switch | **PASS**（4 MATCH） | **PASS**（4 MATCH） | **PASS**（5 MATCH） |

## 3. 状态与工具是否一致

- 每个变体都先执行相同的 seed turns：
  - preference：turn0 `I need a water filter...` + turn1 retraction；
  - goal：turn0 `water filter for hiking` + turn1 `water bottle instead`。
- 每个变体的 seeded state：
  - preference：`event_count=10`, `preference_count=1`, `hard_constraint_count=0`；
  - goal：`event_count=8`, `preference_count=2`, `hard_constraint_count=0`；
  - role target 相同（per scenario）。
- 最终受控 Query 使用 `apply_user_state=False`，不写 M5/M9；三次变体的 source_calls 都是 `history=1, catalog_search=1`，raw/RRF 计数相同（10/15）。

**因此，观察到的最终差异只能来自 Query 文本/query_terms 差异，而不是状态、来源或工具配置差异。**

限制：

- 单源 SASRec / Two-Tower / BM25 原始 top-N 仍未持久化，source-level raw counts **UNVERIFIED**；
- goal_switch 的 Canonical 实际 query_terms 仍包含旧 `water filter for hiking`，来自既有 M9/query 上下文，不是 Canonical 规则显式加入；这说明现有 Query 构造仍有 legacy 上下文残留。

## 4. Base Query 是否超出 Canonical

| 场景 | Base 相对 Canonical 的额外收益 |
|---|---|
| preference_retract | **0**；Canonical 与 Base Query 完全相同 |
| goal_switch | **+1 个 role-MATCH 商品**；任务成功状态不变 |

所以：

- preference 的 Base 收益完全是简单 Canonical 规则可复现的；
- goal 的 Base 收益只有“多一个 MATCH 商品”，没有任务级 PASS 差异，也没有证据说明它比 Canonical 更符合个性化偏好；
- 不能把“多一个 MATCH”直接判定为推荐质量更好。

## 5. 五个问题的回答

### 1) Canonical Query 能否复现 Base 的改善？

- preference_retract：**能，完全复现**（5 个相同 MATCH ID）。
- goal_switch：**部分复现**（4 个 MATCH，Base 5 个；Base 多 `B0BFR4VJTW`，Canonical 有 `B07HF7M273`，集合不同）。
- 任务级：两个场景都没有因 Canonical 变差；goal 仍满足条件。

### 2) Base Query 是否有超出简单 Canonical 规则的实际收益？

- preference：无。
- goal：只多 1 个 role-MATCH 展示商品；没有任务成功状态变化；不足以证明质量增益。
- 结论：**没有稳定、可验证的额外收益。**

### 3) 差异是否来自受控 Query 改变，而不是状态或工具配置改变？

**是，主要来自 Query。**

- Seed state 相同；
- source_calls / raw / RRF 相同；
- role_target / history / Eligibility / k 相同；
- 只有 query_terms 和最终展示不同。

但 goal 的 Canonical 实际 query terms 仍含 legacy 旧目标词，说明 Query 构造不完全“纯 Canonical”；这是 residual confound，标记 LIMITATION，而非状态差异。

### 4) 当前 Query 策略是否值得用 SFT 学习？

**不值得。**

- preference 的 Base 收益已被确定性 Canonical 规则完全复现；
- goal 只有 1 个商品的差异，且不改变 PASS；
- 没有跨任务稳定的 Query 策略收益；
- 用 SFT 学习 Query 会引入不必要的模型依赖，规则/模板更合适。

### 5) 下一步是否应转向可学习的个性化召回来源选择或融合策略？

**是。**

- Query 文本效应已很小，且主要可规则化；
- RERANK 在 P7-F 已证明是盲 no-op；
- 当前动作空间没有来源选择/融合权重动作，因此下一步应在**来源选择或融合策略**上建立可控实验；
- 但在训练前，应先给 Planner/服务暴露可观测的来源候选证据和 soft evidence，保证动作是可判断的。

## 6. 建议

**NO-GO for Query SFT。**  
**GO（下一研究阶段）for source-selection / fusion policy**，前提是先把现有工具的来源/soft evidence 做成可观测、可执行的 action，而不是继续训练 Query 文本。

本轮未训练、未运行 Holdout、未重复完整 P7、未修改 Serving V1/Gold/Evaluator/Eligibility/M5/Planner Contract、未 commit/push/deploy。GPU 实例未关闭。

交付：

- `P8_A_CONTROLLED_QUERY_REPORT.md`
- `runs/p8_a/results.json`
