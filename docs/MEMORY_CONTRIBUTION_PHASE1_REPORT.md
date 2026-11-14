# AgentRec-X — Memory Contribution Benchmark Phase 1 报告

> 状态：**Phase 1 已完成（本地 CPU，真实实验）**
> 范围：只回答问题“长期 Memory 是否改变排序、是否改善 observed acceptable 指标、状态语义是否合理”。
> 本轮**没有**训练模型、没有启动 GPU/AutoDL、没有调用 Qwen、没有重建 BGE/候选池、没有修改 Planner/Retrieval/Gate/Scorer 生产代码、没有修改 P18。

---

## 0. 执行摘要（先给 Q1–Q5 的答案）

| 问题 | 结论 |
|---|---|
| **Q1：长期 Memory 是否真的改变了当前排序？** | **是。** 在 19 个 recommend task 中，B（Memory OFF）与 C（Memory ON）有 **12/19 个任务完整排序不同**，**5/19 个任务 Top-1 不同**，**14/14 个有 Active Memory 的任务都出现了候选分数变化**。 |
| **Q2：在 P20 已知 acceptable 标签下，Memory ON 是否优于 Memory OFF？** | **在这套 provisional observed-acceptable 标签上，是。** C 的 Hit@1 为 **18/19**，B 为 **14/19**（+4）；MRR@10 为 **0.973684 vs 0.850877**；acceptable 首次排名改善 **5 个任务**，回退 **0 个任务**。Gate finalized acceptable 为 **18 vs 9**（B 有 10 个任务未 finalize，主要是无 memory/hard 判据触发 ASK，见 §2.4）。但标签不是人工 Gold，N=19，不能外推为真实用户满意度。 |
| **Q3：Memory 更新、撤回、冲突和 UNKNOWN 是否得到合理处理？** | **是（在构造型状态测试范围内）。** 16 个 counterfactual cases、**126/126 checks 通过**；更新只保留新状态，撤回后偏好不再参与排序，冲突触发 `NEEDS_USER_CLARIFICATION`，UNKNOWN 不被当作匹配或冲突。 |
| **Q4：现有固定权重 PreferenceAwareScorer 对 Memory 的利用有哪些具体局限？** | 见 §6。核心是：没有 query-soft 通道（所以 B 与 A 完全相同）、固定权重且 query relevance 占主导、positive 上限 0.2/term、UNKNOWN 无任何信号、无时间衰减/置信度/用户级权重学习、冲突检测只覆盖 positive-positive 互斥对。 |
| **Q5：Phase 2 应该优先补充什么数据？** | **优先补真实标签和真实纵向偏好数据，而不是先训练。** ① 更多 task + 真正人工 relevance/choice 标签；② 真实用户的纵向 memory 事件（写入/更新/撤回/冲突）与后续选择；③ 独立的 query-soft-preference 标注；④ 商品属性 evidence coverage 与 known/unknown 标签；⑤ counterfactual state + 真实选择的成对数据。当前 19 tasks / 14 active-memory tasks 不足以训练 Learned Memory Ranker。 |

**真实用户满意度：`not established`。** 本轮没有任何独立用户反馈或线上 A/B。

---

## 1. 实验设置与可复现性

### 1.1 复用的资产

| 资产 | 路径 | 用途 |
|---|---|---|
| Ranking v1 固定池数据集 | `runs/preference_ranking_v1/ranking_dataset.jsonl` | 19 个 recommend task；每任务固定 Original-BGE Top-20；provisional acceptable 标签 |
| P20 30-task suite | `runs/p20_agent_eval/task_suite.json` | 任务定义、query、history fixture 引用 |
| Original BGE trace | `runs/advanced_retrieval_v2/task_arm_original_bge.json` | 固定候选池来源（只读 provenance，未重新检索） |
| Event-Sourced Preference Memory Reader | `recommendation.preference.history.HistoryStateReader` | 唯一合法的 active historical preference 读取路径 |
| PreferenceAwareScorer | `recommendation.preference.scorer.PreferenceAwareScorer` | 未修改；复用其已有分数与特征 |
| DecisionGate | `recommendation.preference.decision_gate.DecisionGate` | 未修改；三个 arm 使用同一 Gate 与同一预算状态 |
| Electronics 商品 metadata / evidence | Ranking v1 `candidate_pool[].attributes`（已由既有构建脚本从 enriched catalogue 提取） | 固定池商品属性、form factor、role/hard 状态；本轮不重新加载或重建 catalogue |

### 1.2 实验范围

- **19** 个 `expected_decision == recommend` 任务（P20 30-task suite 中的 recommend 子集）。
- 每个任务固定 **20** 个候选，合计 **380** 个 candidate-task 记录。
- **14** 个任务有 Active Memory，**5** 个没有。
- **210** 个 provisional acceptable 标签；其中 **202 个**在共享 eligibility 集合内，**8 个**被现有 structural role filter 排除（均为 `role_MISMATCH`）。
- 所有分数、排序、Gate 行为都由本地 CPU 计算；没有模型推理。

### 1.3 三个 arm 的严格定义

所有 arm 共享：

- 同一 20 候选固定池；
- 同一商品 metadata/evidence；
- 同一 `prepare_hard(query)` 硬约束；
- 同一 structural eligibility 过滤：

```text
hard_tier == 0
and role_status in {MATCH, NOT_REQUIRED}
```

- 同一 DecisionGate 版本、同一预算状态（无剩余 retrieval / product-evidence round）。

| Arm | 输入 | 说明 |
|---|---|---|
| **A — Relevance Only** | 无 preference；仅 Original-BGE rank | 在共享 eligible 集合内按 BGE 原始排名 |
| **B — Existing Scorer / Memory OFF** | `query_preferences` only | 保留当前 query 条件；关闭 historical memory 的 matched/conflict/unknown 加减分 |
| **C — Existing Scorer / Memory ON** | `query_preferences + history_preferences` | 使用 Memory Reader 的 active preferences |

**Memory OFF 不是“清空所有 Preferences”。** 当前 query 的硬条件保留在 `hard_requirements` 中；如果未来出现当前 query 的显式软偏好，它会被标记为 `origin="current_query"` 并保留在 B。当前 19 条 query 没有任何显式软偏好语句，因此 B 的 query-soft 列表为空，但**没有任何 query 硬条件被删除**。

### 1.4 指标与标签边界

- **Observed Acceptable Hit@1/3/5**：只对已有 provisional acceptable 标签计算。
- **MRR@10**：只针对已知 acceptable 标签；没有把未标注候选当成负例。
- 未标注候选始终是 `unjudged`，不是 negative。
- 这不是完整人工 Gold nDCG，也不是 full-catalogue Recall；只是固定池、单一 provisional 标签集上的比较。
- 所有 arm 的 ranking 都是在共享 eligible 集合内排序；因此 hard violation 在构造上为 0。pre-ranking 被排除的候选原因单独记录。

### 1.5 复现命令

```bash
# 本地 WSL CPU Python（仓库内已有 .venv）
.venv/bin/python -m experiments.memory_contribution.run_fixed_pool
.venv/bin/python -m experiments.memory_contribution.run_counterfactual
.venv/bin/python -m experiments.memory_contribution.build_manifest
```

输出目录：

```text
runs/memory_contribution_phase1/
  fixed_pool_comparison.json
  per_task_results.jsonl
  arm_results.jsonl
  0051_diagnostic.json
  counterfactual_cases.jsonl
  counterfactual_summary.json
  manifest.json
```

---

## 2. A / B / C 三组指标对照

### 2.1 总体（19 recommend tasks）

| 指标 | A Relevance Only | B Scorer / Memory OFF | C Scorer / Memory ON |
|---|---:|---:|---:|
| tasks | 19 | 19 | 19 |
| **Observed Acceptable Hit@1** | **14/19** | **14/19** | **18/19** |
| Observed Acceptable Hit@3 | 19/19 | 19/19 | 19/19 |
| Observed Acceptable Hit@5 | 19/19 | 19/19 | 19/19 |
| MRR@10（known acceptable） | 0.850877 | 0.850877 | **0.973684** |
| mean first acceptable rank | 1.368421 | 1.368421 | **1.052632** |
| tasks with acceptable in ranking | 19/19 | 19/19 | 19/19 |

### 2.2 关键观察：A 与 B 完全相同

`a_vs_b.full_ranking_changed_tasks = 0`，19/19 任务排序完全一致。

原因不是 benchmark 遗漏，而是当前生产 Scorer 的输入结构：

- 当前 query 的条件只进入 `hard_requirements`（由 `prepare_hard` 提取）；
- `PreferenceAwareScorer.preferences` 的软偏好通道只有 historical memory；
- 当前 19 条 query 没有被抽取成 query-soft preferences；
- 因此 Memory OFF 时 scorer 的公式退化为 `0.5 × rank_relevance`，与 A 的 BGE 顺序等价。

**这是当前架构的一个真实局限，不是本实验的错误。** 它也说明仅用现有 query 无法测量“query-soft preference”这一通道的贡献。

### 2.3 Active-Memory 子集

| 子集 | tasks | A Hit@1 | B Hit@1 | C Hit@1 | B MRR | C MRR |
|---|---:|---:|---:|---:|---:|---:|
| 有 Active Memory | 14 | 9/14 | 9/14 | **13/14** | 0.797619 | **0.964286** |
| 无 Active Memory | 5 | 5/5 | 5/5 | 5/5 | 1.0 | 1.0 |

- 14 个 active-memory 任务的 Hit@1 从 9 提升到 13；提升全部来自 Memory。
- 5 个 no-memory 任务上 C 与 B/A 完全相同，符合预期。

### 2.4 Gate 行为

| Arm | ALLOW_FINALIZE | NEED_USER_CLARIFICATION | Gate selected = rank1 | Gate finalized acceptable |
|---|---:|---:|---:|---:|
| A | 9 | 10 | 9 | 9 |
| B | 9 | 10 | 9 | 9 |
| C | **19** | 0 | 19 | **18** |

- A/B 有 10 个任务因为“没有 hard requirement，也没有 active preference”触发 `insufficient_decision_criteria`，Gate 不 finalize。这是 Gate 的既有语义，不是排序失败。
- C 的 19 个任务都能 finalize；其中 18 个 Gate selected 是 acceptable，**0051 是唯一 selected 非 acceptable 的案例**。
- 因此“ranking Hit@1”和“Gate-finalized acceptable”是两个不同指标，本报告分别展示。

### 2.5 硬约束

| Arm | Violations@1 | Violations@3 | Violations@5 |
|---|---:|---:|---:|
| A | 0 | 0 | 0 |
| B | 0 | 0 | 0 |
| C | 0 | 0 | 0 |

pre-ranking 共享过滤排除了：

```text
hard_tier_1 (UNKNOWN hard evidence): 6 candidate instances
hard_tier_2 (violated hard evidence): 0 candidate instances
role_MISMATCH / other role exclusions: 107 candidate instances
```

记忆评分没有绕过任何 hard filter；三个 arm 的 eligible 集合完全相同。

---

## 3. B vs C：Memory 的独立贡献

### 3.1 排序变化

| 变化类型 | 任务数 | 任务 |
|---|---:|---|
| Top-1 变化 | **5** | `p14b3_eval100_0065`, `p14e2_lite_0045`, `p14e2_lite_0046`, `p14e2_lite_0048`, `p14e2_lite_0049` |
| Top-3 集合变化 | **7** | 上述 5 个 + `p14b3_eval100_0045`, `p14e2_lite_0047`, `p14e2_lite_0052` |
| 完整排序变化 | **12** | `p14b3_eval100_0045`, `p14b3_eval100_0065`, `p14e2_lite_0045`, `p14e2_lite_0046`, `p14e2_lite_0047`, `p14e2_lite_0048`, `p14e2_lite_0049`, `p14e2_lite_0050`, `p14e2_lite_0051`, `p14e2_lite_0052`, `p14e2_lite_0053`, `p14e2_lite_0054` |
| 任一候选分数变化 | **14** | 所有有 Active Memory 的任务 |
| 有 Active Memory 且确实受影响 | **14 / 14** | 分数变化或排序变化 |

### 3.2 对 observed acceptable 的影响

| 指标 | 数值 |
|---|---:|
| acceptable 首次排名 improved | **5 个任务** |
| acceptable 首次排名 regressed | **0 个任务** |
| acceptable 首次排名 unchanged | 14 个任务 |
| Hit@1 benefit（C hit1 且 B 不 hit1） | **4 个任务** |
| Hit@1 regression（B hit1 且 C 不 hit1） | **0 个任务** |

改善任务：

- `p14b3_eval100_0065`：B first acceptable rank 2 → C rank 1
- `p14e2_lite_0045`：2 → 1
- `p14e2_lite_0046`：2 → 1
- `p14e2_lite_0048`：3 → 1
- `p14e2_lite_0051`：3 → 2（改善但 Top-1 仍未变为 acceptable，见 §6）

### 3.3 有收益的典型案例

1. **`p14b3_eval100_0065`** — Memory：`in_ear positive`、`wireless positive`。
   B 的 Top-1 是 `B07KGN6852`（未标注），C 的 Top-1 变为 `B01KIMIFME`（acceptable，in-ear）。
   原因：`in_ear` 正向匹配把 in-ear 候选整体抬升。

2. **`p14e2_lite_0045`** — Memory：`over_ear positive`、`noise_cancelling positive`、`in_ear negative`。
   B 的 Top-1 是 `B08HM3MLJF`（未标注，in-ear），C 的 Top-1 变为 `B0BLSQ5T75`（acceptable，over-ear + noise-cancelling）。
   这是 positive 与 negative 同时起作用的案例。

3. **`p14e2_lite_0046`** — Memory：`wireless positive`、`silent positive`、`rgb negative`。
   B 的 Top-1 是 `B09PYFYBTJ`（未标注），C 的 Top-1 变为 `B0BK3LYMR2`（acceptable）。

4. **`p14e2_lite_0048`** — Memory：`microphone positive`、`wireless positive`、`wired negative`。
   B 的 acceptable 首次排名为 3，C 提升到 1（`B0716JGTVR`）。
   注意：B 的 Top-1 与 Top-2 都是 unjudged，因此这里的“B 失败”只是相对于 provisional labels，不表示真实用户不喜欢它们。

### 3.4 没有变化的任务与原因

7 个任务 B/C 完整排序相同：

| 任务 | 原因 |
|---|---|
| `p14b3_eval100_0001` | 无 Active Memory |
| `p14e2_lite_0041` | 无 Active Memory |
| `p14e2_lite_0042` | 无 Active Memory |
| `p14e2_lite_0043` | 无 Active Memory |
| `p14e2_lite_0044` | 无 Active Memory |
| `p14b3_eval100_0073` | Active Memory 只有 `wireless positive`；hard requirement 已经是 wireless，所有 eligible 候选都带 wireless 证据，因此所有候选获得相同 +0.2，顺序不变（分数变化、排序不变） |
| `p14e2_lite_0055` | 同上：只有 `wireless positive`，所有 eligible 候选同分抬升，顺序不变 |

**“排序变了”不等于“质量提高”。** 例如 `p14e2_lite_0049` 的 Top-1 从 `B08D32JM5F` 变成 `B097PNMZ42`，两者都在 provisional acceptable 集合中；这个变化只能证明 Memory 改变了排序，不能证明新 Top-1 更符合真实用户偏好。

---

## 4. `0051` 诊断

### 4.1 输入

```text
query: Which earbuds work for workouts? I'd rather hear one clear suggestion than a long list.
active memory:
  - over_ear negative
  - rechargeable positive
  - microphone positive
hard requirements: none
shared eligible set: 3 candidates
```

### 4.2 A / B / C 排名与分数

| item | label | BGE rank | A rank | B rank | C rank | score B | score C | query component | memory match | unknown |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B07BKPKXTT` | unjudged | 8 | 1 | 1 | **1** | 0.315789 | 0.315789 | 0.315789 | 0.0 | rechargeable, microphone, over_ear |
| `B004K6R43G` | unjudged | 13 | 2 | 2 | **3** | 0.184211 | 0.184211 | 0.184211 | 0.0 | rechargeable, microphone, over_ear |
| `B08D32JM5F` | **acceptable** | 16 | 3 | 3 | **2** | 0.105263 | **0.305263** | 0.105263 | **+0.2 microphone** | rechargeable, over_ear |

### 4.3 为什么 Existing Scorer 仍选择 `B07`

1. Memory OFF（B）时，`B07` 的分数是 query 分 **0.315789**，`B08` 是 **0.105263**；差距 **0.210526**。
2. Memory ON（C）时，`B08` 匹配 `microphone`，获得 **+0.2**；`B07` 没有匹配项，也没有冲突项。
3. 0.2 的正向 bonus **小于** 0.210526 的 query relevance 差距；最终：
   - `B07` = 0.315789
   - `B08` = 0.305263
   - 差距 **0.010526**
4. `B07` 没有 microphone/rechargeable 证据。这些是 **UNKNOWN**，不是 negative，也不会被扣分；`B07` 的 `over_ear` 也是 UNKNOWN（它是 in-ear），所以 negative `over_ear` 不构成冲突。
5. 因此 Memory **把 `B08` 从 rank 3 抬到 rank 2，把差距从 0.210526 缩小到 0.010526（缩小 0.2）**，但没有改变 Top-1。
6. `B08` 是 acceptable 商品；`B07` 不是。Gate 仍选 `B07`，所以 0051 是 Hit@1 失败案例。

### 4.4 `B07` 的 Unknown 不等于“没有麦克风”

`B07` 的目录属性只有 `brand`, `in_ear`, `wireless`；没有 `microphone`，也没有 `rechargeable`。Scorer 正确地把它记录为 `unknown_preferences`，没有当作“确定没有麦克风”，也没有因此扣分。
本轮**没有**为了修复 0051 改 Scorer 权重；这是一个诊断案例，不是调参目标。

---

## 5. Counterfactual Memory 状态测试

### 5.1 规模与结果

- **16** 个 counterfactual cases；
- **16/16** cases 通过；
- **126/126** checks 通过；
- **15/16** cases 的排序发生变化，**10/16** cases 的 Top-1 发生变化；
- Memory Consistency Pass Rate（构造型）：**1.000000**；
- Hard Constraint Violations：**0**；
- 2 个冲突案例均触发 `NEEDS_USER_CLARIFICATION`，Top-1 Gold 被标记为不允许。

| 类别 | cases | passed | checks |
|---|---:|---:|---:|
| positive preference | 4 | 4 | 35 |
| negative preference | 2 | 2 | 12 |
| preference update | 2 | 2 | 18 |
| preference withdrawal | 2 | 2 | 14 |
| preference conflict | 2 | 2 | 12 |
| UNKNOWN evidence | 4 | 4 | 35 |

### 5.2 状态处理语义

每个 case 都通过真实 M9 写入路径（`ShoppingCompanionMemory.ingest_sessions`）和真实 reader 生成，不另建 Memory 实现；baseline/mutated 使用同一 query、同一候选池、同一商品 metadata、同一 hard constraints，只改变 memory sessions。

#### Positive Preference

- `cf001`：加入 `rechargeable positive` 后，真实有 rechargeable 证据的 `B0B55YF7CL` 获得 +0.2 并成为 Top-1。
- `cf002`：加入 `on_ear positive` 后，on-ear 候选 `B0BQ38B3NC` 成为 Top-1。
- `cf003`：加入 `microphone positive` 后，有 microphone 证据的候选获得 +0.2，无证据候选保持 UNKNOWN。
- `cf015`（0051 池）：`B08D32JM5F` 从 rank 3 提升到 rank 2，但 Top-1 仍为 `B07`，与 §4 一致。

#### Negative Preference

- `cf004`：`over_ear negative` 使有 over-ear 证据的 `B0BLSQ5T75` 获得 -0.35，Top-1 变为 `B0BQ38B3NC`。
- `cf005`：`microphone negative` 使有 microphone 证据的候选被扣分，Top-1 改变。
- 缺少证据的候选只产生 UNKNOWN，不会被错误扣分。

#### Preference Update

- `cf006`：`over-ear` → `rechargeable` 的显式更正只保留新状态；旧状态是 `superseded`，不叠加。
- `cf007`：`microphone` → `rechargeable` 同样只保留新状态。
- 验证点：`active_preferences` 只含新值；旧值出现在 `inactive_raw` 的 `superseded` 记录中；旧值不再在 scorer 的 `matched_preferences` 中。

#### Preference Withdrawal

- `cf008`：撤回 `microphone` 后 active 为空，旧值为 `removed`；该偏好不再影响排序。
- `cf009`：撤回 `over_ear negative` 后，over-ear 候选的扣分消失，分数回升。
- 撤回不是“再加一个反向偏好”，而是从 active 状态中移除。

#### Preference Conflict

- `cf010`：`over_ear positive` + `in_ear positive` → Memory Reader 报告冲突；Gate 返回 `NEED_USER_CLARIFICATION`，不强行选择其中一个。
- `cf011`：`wireless positive` + `wired positive` → 同样触发 clarification。
- 两个案例都标记 `requires_clarification=true`，`top1_gold_allowed=false`。

#### UNKNOWN Evidence

- `cf012`：正向 microphone 偏好下，无 microphone 证据的候选不匹配、不冲突、不加分。
- `cf013`：负向 over_ear 偏好下，无 over_ear 证据的候选不冲突、不扣分。
- `cf014`：无 rechargeable 证据的候选 UNKNOWN；有 rechargeable 证据的候选获得 +0.2。
- `cf016`（0051 池）：所有 eligible 候选都是 in-ear；`over_ear negative` 全部是 UNKNOWN，排序不变、无冲突。

### 5.3 这些测试能说明什么

- 能说明：**Memory 状态处理与 Scorer 的确定性响应符合现有语义**；更新/撤回/冲突/UNKNOWN 没有出现明显错误。
- 不能说明：真实用户满意度、真实偏好强度、或“有冲突时用户会怎么选”。
- `Memory Consistency Pass Rate = 1.0` 是**构造型测试通过率**，不是自然分布上的准确率。

---

## 6. 当前固定权重 Scorer 的具体局限

1. **没有 query-soft preference 通道。**
   当前 query 条件只进入 hard requirements；19 条 query 没有显式软偏好。因此 Memory OFF（B）与 Relevance Only（A）完全一致，无法单独测量 query-soft 信号的贡献。

2. **Query relevance 权重固定且占主导。**
   `query_weight=0.5`，且 query relevance 是池内 rank 的线性函数。单个 positive 只 +0.2，最多 3 个 term；在 0051 中，0.2 不足以跨越 0.210526 的 relevance 差距。

3. **Positive 上限固定。**
   `max_positive_terms=3`；更多正向偏好不会继续加分，多个相似偏好的边际贡献被截断。

4. **UNKNOWN 完全没有信号。**
   UNKNOWN 不加分也不扣分；Scorer 不能主动获取证据，也不能区分“目录没写”与“商品确实没有”。这是 0051 的直接原因之一。

5. **没有时间衰减/置信度。**
   所有 active preferences 等权；新偏好和旧偏好没有不同权重（更新/撤回本身是正确处理的，但生效后的权重不随新旧变化）。

6. **冲突检测只覆盖 positive-positive 互斥对。**
   现有 `conflicts()` 检查 `wireless/wired`、`over_ear/in_ear`、`on_ear/in_ear` 的正向互斥；正向与负向同一属性的组合不会被报告为冲突。Scorer 会同时应用正向加分和负向扣分，Gate 也不会因此 ASK。

7. **没有用户级/任务级权重学习。**
   所有用户、所有 query 使用同一组固定权重；不能学习“某类用户更看重 microphone”或“某类任务 query relevance 应占更低权重”。

8. **没有上下文/多偏好交互。**
   Preference 之间只有简单加和；没有组合特征、交叉特征或 pairwise 学习。

这些局限直接支持 Phase 2 的方向：学习型 ranker 需要真实偏好/选择标签和 query-soft 通道，而不是继续在固定权重上手工调参。

---

## 7. 当前结果能证明什么、不能证明什么

### 能证明（在明确范围内）

1. **Memory 确实改变了排序。**
   19 个固定池任务中 12 个完整排序变化，5 个 Top-1 变化，14/14 个 active-memory 任务出现分数变化。

2. **在 provisional observed-acceptable 标签上，Memory ON 优于 Memory OFF。**
   Hit@1 18/19 vs 14/19，MRR 0.973684 vs 0.850877；5 个任务首次 acceptable 排名改善，0 个任务回退。

3. **Memory 状态处理是合理的（构造型范围）。**
   16 个 cases / 126 checks 通过；更新、撤回、冲突、UNKNOWN 都符合现有 M9 Reader/Scorer/Gate 语义。

4. **0051 的失败机理是清楚的。**
   microphone 的正向 +0.2 不足以抵消 0.210526 的 query relevance 差距；UNKNOWN 没有被错误当成冲突或“没有麦克风”。

### 不能证明

1. **不能证明真实用户满意度提高。**
   没有独立用户反馈、没有线上 A/B；`real user satisfaction = not established`。

2. **不能把“排序变化”直接当成“质量提高”。**
   未标注候选仍是 unjudged；0049 的 Top-1 更换只是 acceptable→acceptable，标签无法判断哪一个更好。

3. **不能外推到完整目录 / 完整 Recall / nDCG。**
   这是固定池、19 个任务、单目录、provisional 标签的评估。

4. **不能证明 Learned Memory Ranker 会更好。**
   本轮没有训练；只测了现有固定权重 Scorer。

5. **不能证明统计显著性。**
   N=19，没有置信区间/显著性检验；+4 Hit@1 是方向性结果，不是统计显著结论。

6. **不能证明 B/A 差异的语义贡献。**
   B 与 A 完全相同；query-soft 通道在当前生产代码中不存在。

---

## 8. Phase 2 是否必要、应优先补什么数据

### 结论

**有必要进入 Phase 2 数据准备，但不应立即进入 GPU 训练。**
当前结果足以证明“Memory 有排序价值、状态语义基本正确”，但**不足以训练 Learned Memory Ranker**：

- 只有 19 个 recommend task，14 个有 Active Memory；
- 标签是 provisional automated/verified，不是人工 Gold；
- memory 来自构造型 persona fixture；
- 没有 query-soft preference 标注；
- 没有真实用户对“推荐结果”的反馈；
- 商品属性 UNKNOWN 普遍存在。

### Phase 2 优先补充的数据（按优先级）

1. **真实 relevance/choice 标签 + 更多任务**
   把 19-task 扩到数百/数千 task；每个候选需要 human acceptable / graded relevance，而不是单一 acceptable set。至少要让 pairwise 或 graded 标签能够训练 listwise/pointwise ranker。

2. **真实纵向用户偏好与选择数据**
   真实的“用户说过偏好 → 后续选择/放弃”的序列；包含写入、更新、撤回、冲突事件和时间戳。当前 P20 是构造模板，不能证明真实行为。

3. **独立的 query-soft-preference 标注**
   将“当前 query 的软偏好”与“历史 memory”分成两个字段/两个来源，才能训练并评估二者的交互。这是打破 B=A 的前提。

4. **商品属性 evidence coverage 与 known/unknown 标签**
   需要标注“目录中确实有该属性”“目录中明确没有”“目录未覆盖”，才能学习 UNKNOWN 的正确处理；当前 Scorer 只能按 attribute presence 处理。

5. **Counterfactual state + 真实选择的成对数据**
   同一 query/候选池下，改变 memory 状态后的真实用户选择（或回顾性偏好），用于训练 memory-conditioned ranker；构造型测试不能替代。

6. **安全的负样本/未选择样本协议**
   Amazon ratings 只有 observed interaction，未交互不等于负例；需要明确 candidate exposure 和未选择语义，否则会制造伪负样本。

### 为什么不能直接沿用 Phase A 的外部数据

- ESCI 与当前 76,263 商品目录只有 **626** 个精确交集商品；
- TREC 2025 只有 **115** 个精确交集商品，且 test-only；
- Amazon Reviews 只有 **14,157** 个用户有 ≥3 个目录内交互。

这些数据可用于外部研究或辅助预训练，但不足以直接训练“当前 Agent 目录上的 Learned Memory Ranker”。

---

## 9. 工程变更、验证与未决问题

### 9.1 新增文件

```text
experiments/memory_contribution/__init__.py
experiments/memory_contribution/common.py
experiments/memory_contribution/run_fixed_pool.py
experiments/memory_contribution/run_counterfactual.py
experiments/memory_contribution/build_manifest.py
tests/test_memory_contribution.py
docs/MEMORY_CONTRIBUTION_PHASE1_REPORT.md
```

### 9.2 修改的生产代码

**无。**

- 未修改 `recommendation/preference/scorer.py`
- 未修改 `recommendation/preference/decision_gate.py`
- 未修改 `recommendation/preference/history.py`
- 未修改 Planner / Retrieval / BGE / P20 suite / P18 Manifest
- 未改生产默认 Ranking 行为

Memory OFF 是通过 benchmark runner 的局部输入控制实现的：
`query_preferences` 保留，`history_preferences` 在 B 中被过滤；C 使用两者。

### 9.3 生成但未入库的产物

`runs/` 按仓库规则不入 Git；本轮产物在：

```text
runs/memory_contribution_phase1/
```

`manifest.json` 记录了所有输入/输出文件的 SHA-256、配置、运行环境和保护声明。

### 9.4 测试

```text
tests/test_memory_contribution.py
  9 passed
```

相关既有测试：

```text
tests/test_p14b0_memory_scorer.py
tests/test_p14b1_decision_gate.py
tests/test_preference_construction.py
tests/test_memory_contribution.py
  61 passed
```

测试覆盖：

- Memory OFF 保留 query preference、仅删除 history 通道；
- query-soft 提取器不会从商品请求中凭空造偏好；
- Hit@k/MRR 对 unjudged 候选不强加负例；
- shared eligibility 只用 hard tier 与 role；
- M9 update/withdrawal/conflict 语义；
- Scorer 对 UNKNOWN 不奖励、不判冲突；
- negative preference 只在有证据时扣分。

### 9.5 交叉核对

- 新 C 排序与已有 Ranking v1 `C_hard_filtered_scorer` 排序：**19/19 完全一致**。
- 新 C Hit@1 = 18/19，与已有记录一致。
- 已有 gate selected acceptable = 18/19，与 C Gate finalized acceptable 一致。

### 9.6 未决问题

1. B = A：query-soft preference 通道缺失，无法在当前数据上单独测量该信号。
2. 8 个 provisional acceptable 标签被 role filter 排除（202/210 eligible）；这是共享 eligibility 的结果，需在标签升级时核对 role 定义。
3. 标签为 provisional，不是 human Gold；不能报告“完整人工 nDCG”。
4. N=19，无统计显著性/置信区间。
5. 0051 仍是 Hit@1 失败；本轮按任务要求不调权重。
6. Counterfactual 测试是构造型，不代表真实用户分布。
7. `runs/memory_contribution_phase1/` 未提交 Git；需用本地复现命令重建。

### 9.7 结论与下一里程碑建议

- **Memory Contribution Phase 1：完成。**
- **是否进入 GPU 训练：否。**
- **建议下一里程碑：Phase 2 数据准备**，优先做真实标签、真实纵向偏好/选择数据、query-soft 标注、属性 evidence 标注；在数据规模与标签质量达到训练条件前，不训练 Learned Memory Ranker。

---

## 10. Git Commit

本轮只创建本地 Git commit，**不 Push**。代码、测试与本报告在同一个本地提交中；最终 HEAD 以 `git log -1` 为准。

保护性检查：

```text
P18 Frozen Manifest modified: no
P18 Reserved used: no
Planner modified: no
Retrieval modified: no
DecisionGate modified: no
Scorer modified: no
Production ranking default changed: no
GPU / AutoDL / Qwen / LightGBM used: no
Git Push: no
```
