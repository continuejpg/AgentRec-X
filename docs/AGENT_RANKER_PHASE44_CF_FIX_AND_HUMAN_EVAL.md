# Phase 4.4 — Counterfactual E2E Fix, GPU Validation & Blind Preference Evaluation

> 本轮修复了 Phase 4.3 的 Hybrid 16/16 `budget_exhausted`，完成 **Rule Planner 96 条 + Qwen Planner 96 条** Counterfactual E2E；并交付 40 对盲评 HTML 工具。
> 不训练/替换模型；生产默认仍为 `PreferenceAwareScorer`。没有 HUMAN_GOLD。

---

## 0. 结论先行

### Hybrid 根因

**实验集成 Bug，不是 Hybrid 公式或 Gate Bug。**

Phase 4.3 的 CF Runner 对同一 `(case, state)` 只创建 **一个 `ElectronicsPlannerExecutor`**，然后依次运行 `existing -> structured -> hybrid`。第一个 arm finalized 后，`executor.terminal_status='finalized'`、`selected_candidate_id`、`last_rows`、ledger 等状态泄漏到下一个 arm。后续 arm 只执行了 `RETRIEVE` 就被旧的终止状态/状态机截断，记录成 `budget_exhausted`。

最小修复：

```text
每个 (case, state, arm) 创建全新 ElectronicsPlannerExecutor
+ 新 FixedPoolSource
+ 新 CachedRanker
```

没有增加 `MAX_STEPS`，没有跳过 Gate，没有强制 FINALIZE，没有改 Hybrid 权重。

### 修复后 Rule Planner 96 条

| Arm | Trajectories | FINALIZE | AWAITING_CLARIFICATION | BUDGET_EXHAUSTED | Errors | Hard Violations |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 32 | 28 | 4 | 0 | 0 | 0 |
| Natural | 32 | 24 | 8 | 0 | 0 | 0 |
| Hybrid | 32 | **28** | **4** | **0** | 0 | 0 |

**Hybrid 由 16/16 budget_exhausted 变为 28 FINALIZE + 4 ASK + 0 budget_exhausted。**

### 修复后 Qwen Planner 96 条

| Arm | Trajectories | FINALIZE | ASK | BUDGET_EXHAUSTED | Errors | Hard |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 32 | 28 | 3 | 0 | 1 | 0 |
| Natural | 32 | 24 | 7 | 0 | 1 | 0 |
| Hybrid | 32 | **28** | **3** | **0** | 1 | 0 |

Qwen 的 3 个 error 都来自 **同一 conflict case `cf010`** 的 planner 输出解析失败（unknown action `''`），不是 Ranker/Hybrid 失败。其余 93 条正常完成。

---

## 1. Hybrid 预算耗尽：修复前后轨迹

结果：`runs/agent_ranker_phase44/hybrid_budget_fix.json`

### 修复前

16 个 mutated Hybrid 全部：

```text
actions = [RETRIEVE]
status  = budget_exhausted
gate    = ALLOW_FINALIZE (but no FINALIZE action)
```

### 修复后

同一个 `cf001` 的 Hybrid：

```text
actions = [RETRIEVE, RERANK, FINALIZE]
status  = finalized
gate    = ALLOW_FINALIZE
selected= B0B55YF7CL
```

### 根因代码路径

Phase 4.3 `run_e2e_counterfactual.py`：

```python
executor = ElectronicsPlannerExecutor(...)          # once per case/state
for arm in ["existing", "structured", "hybrid"]:
    executor.scorer = CachedRanker(...)             # replaced, but executor state not reset
    run_planner(...)
```

修复：

```python
for arm in ["existing", "natural", "hybrid"]:
    executor = ElectronicsPlannerExecutor(...)      # fresh per arm
    executor.search_source = FixedPoolSource(ids)
    executor.scorer = CachedRanker(...)
    run_planner(...)
```

### 排序算法未改变

```text
pool_z(ShopRanker Query Only)
+ 0.5 * positive_match_rate
- 0.75 * negative_conflict_rate
```

权重、公式、阈值均未改变。没有根据 P20/Silver/CF 重调。

---

## 2. 完整 Counterfactual 矩阵

16 Cases × 2 States × 3 Arms = **96 条/ planner**。

固定变量：

- Query
- Original BGE 固定候选池
- Candidate Metadata
- Eligibility
- Gate
- Action Budget `MAX_STEPS=8`
- Memory Event Sequence

只变化：

- Ranker Arm
- Baseline / Mutated Memory State

每个状态都经过真实：

```text
ShoppingCompanionMemory.ingest_sessions
-> HistoryStateReader
-> active preferences
-> Ranker
-> DecisionGate
```

没有只改 Prompt 字符串冒充状态变化。Baseline 为空时使用空 active state；Mutated 使用真实 `mutated_memory_sessions` 写入 Memory Writer。

### Rule Planner per arm

| Arm | FINALIZE | ASK | Budget | Hard |
|---|---:|---:|---:|---:|
| Existing | 28 | 4 | 0 | 0 |
| Natural | 24 | 8 | 0 | 0 |
| Hybrid | 28 | 4 | 0 | 0 |

### Qwen Planner per arm

| Arm | FINALIZE | ASK | Error | Hard |
|---|---:|---:|---:|---:|
| Existing | 28 | 3 | 1 | 0 |
| Natural | 24 | 7 | 1 | 0 |
| Hybrid | 28 | 3 | 1 | 0 |

### 正确 ASK

- 2 个 conflict cases：`cf010`、`cf011`
- 每个 case 有 baseline/mutated 两个状态
- Rule：conflict 的 mutated 状态全部 ASK（6/6 arms across cases/states）
- Qwen：`cf011` mutated 全部 ASK；`cf010` mutated 因 planner 解析错误记录 error
- 没有为了 FINALIZE 率强行推荐；ASK 不计入普通推荐失败。

### Rule vs Qwen

- Existing/Hybrid 的 FINALIZE/ASK 数量几乎一致；
- Natural 在 Rule 下 ASK 更多（8 vs 7）；
- Qwen 的 3 个 error 是 planner JSON 解析问题，不是 Ranker 返回问题；
- 两者都 hard violation 0。

---

## 3. Top-1 / Top-3 / Pairwise Ranking Changes

独立 Ranker 评分（固定候选池、同 Memory 状态）结果：
`runs/agent_ranker_phase44/cf_ranking_metrics.json`

| Arm | Top-1 changed baseline→mutated | Top-3 set changed | Pair order changed |
|---|---:|---:|---:|
| Existing | 10/16 | 11/16 | 14/16 |
| Natural | 12/16 | 14/16 | 15/16 |
| Hybrid | 10/16 | 11/16 | 14/16 |

解释：

- 这是 **baseline vs mutated 的 mechanistic 变化**，不是正确性；
- Natural 对 Memory 状态变化最敏感；
- Hybrid 较接近 Existing 的变化幅度；
- 没有独立 preference gold，不能把变化方向称为正确/错误。

### Rule/Qwen E2E Top-1

E2E 最终 selected 记录在 `cf_matrix_rule.json` / `cf_matrix_qwen.json`；所有 finalized selected 均在 eligible 候选中，hard violation 0。

### Key-case behavior（Rule E2E）

| Case group | Observed behavior | Interpretation |
|---|---|---|
| Negative `cf004/cf005` | Existing/Hybrid mutated finalized; Natural mutated ASK | No independent direction Gold; Natural is more conservative |
| Update `cf006/cf007` | All arms finalized; Hybrid changed top1 after mutated state | New active state affects ranking; no correctness Gold |
| Withdrawal `cf008/cf009` | All/most arms finalized; no stale preference contribution in adapter | Adapter withdrawal invariance passes; E2E direction not Gold |
| UNKNOWN `cf012-cf014` | All arms finalized; Natural asks on cf013 | UNKNOWN not forced as absence; Gate may ask |
| Conflict `cf010/cf011` | Mutated states ASK; baseline has no conflict -> finalize | Correct ASK semantics |
| `0051` `cf015/cf016` | Baseline ASK; mutated finalized; Natural/Hybrid select acceptable `B08D32JM5F`, Existing selects `B07BKPKXTT` | Memory state changes finalization and selection |

**注意：**上述行为是 mechanistic；没有独立用户偏好 Gold，不能把 Natural 的 ASK 或 Hybrid 的 selected 直接判为正确。

---

## 4. Natural Memory 的完整 CF 表现

Rule：

```text
baseline: 13 FINALIZE + 3 ASK
mutated:  11 FINALIZE + 5 ASK
```

Qwen：

```text
baseline: 13 FINALIZE + 3 ASK
mutated:  11 FINALIZE + 4 ASK + 1 planner error
```

Natural 对 memory 状态更敏感：multiple negative/withdrawal/conflict cases 进入 ASK；
这不是“正确率”，只是 Gate 对候选排序的响应。没有独立方向 Gold。

---

## 5. Hybrid 修复后的正确率与终止情况

Rule：

```text
28 FINALIZE
4 ASK
0 budget_exhausted
0 error
0 hard violation
```

Qwen：

```text
28 FINALIZE
3 ASK
0 budget
1 planner parse error (cf010)
0 hard violation
```

Hybrid 修复后可以完成正常 Agent E2E，不再因执行器状态泄漏而全部预算耗尽。
但它的 P20 E2E 仍是 18/19，低于 Natural 19/19；不能因此直接进入生产。

---

## 6. 40 对 Blind Preference Evaluation

复用 Phase 4.3 的 40 对 disagreement-set 任务，没有重新生成大量相似样本。

### 工具

`runs/agent_ranker_phase44/blind_review_tool.html`

独立本地 HTML：

- 展示 Query；
- 展示 Active Memory（Positive / Negative）；
- 展示 A/B 商品、attributes、evidence、UNKNOWN；
- 支持 A / B / BOTH / INSUFFICIENT；
- 支持 notes；
- localStorage 保存进度；
- Export CSV。

隐藏：

- 模型身份；
- 模型分数；
- 原始 BGE rank；
- Silver 规则标签；
- 哪个模型推荐了哪个商品。

### 标注状态

```text
tasks: 40
HUMAN_GOLD: 0
HUMAN_SINGLE_REVIEW: 0
```

本轮没有实际人工输入，所以：

- 不报告 Pairwise Agreement；
- 不报告按 Query Family 的 human result；
- 不把工具里未填的任务称为 Gold。

若开发者完成审核，只能标记为 `HUMAN_SINGLE_REVIEW`；这 40 对是 **Disagreement-Set Human Preference Evaluation**，不是随机用户满意度评测。

---

## 7. GPU 部署成本复核

### Concurrent Resident

实测：

```text
Qwen Planner: 17,551 MB
ShopRanker 20-candidate batch 额外峰值: 1,917.29 MB
nvidia-smi total used: 19,067 MB
OOM: 0
```

ShopRanker 20-candidate 与 Qwen Planner 并发驻留 **PASS**，没有 OOM。
但余量只有约 5.5GB，且这是轻量 batch；不能直接推断所有 E2E 并发高峰都安全。

### Sequential

P20 E2E 实际采用：

- Qwen Planner 单独驻留；
- ShopRanker scores 来自独立 GPU 运行；
- Ranker 排序注入 E2E；
- Memory Writer/Reader、Gate、BGE 实际执行。

这是稳定的生产候选测试路径，但包含一次离线 Ranker 评分，不是每任务实时加载 ShopRanker。

### 延迟

P20 E2E（Phase 4.3，公共接口未因本轮修复改变）：

| Arm | p50 | p95 |
|---|---:|---:|
| Existing | 10.87s | 13.43s |
| Natural | 10.46s | 13.78s |
| Hybrid | 10.59s | 13.88s |

拆分：

```text
Qwen Planner p50:       3.48–3.71 s
BGE retrieval p50:      0.28–0.35 s
Memory Adapter CPU:     0.0704 ms / 20 candidates
Ranker-only Shop:       109.98 ms
Ranker-only Hybrid:     110.09 ms
Final answer template:  0.012 ms
```

- CF Qwen matrix wall：约 **1012 s**（96 trajectories）；
- CF Shop scoring wall：7.763 s，peak 3,572 MB；
- CF Rule matrix：CPU，无 GPU model generation；
- OOM：0。

---

## 8. P20 回归

Phase 4.3 P20 E2E 未受本轮 CF 修复影响（公共 E2E 接口未变）：

| Arm | Success | FINALIZE | Hard | Answer Evidence | E2E p50 |
|---|---:|---:|---:|---:|---:|
| Existing | 18/19 | 19/19 | 0 | 19/19 | 10.87s |
| Natural | **19/19** | 19/19 | 0 | 19/19 | 10.46s |
| Hybrid | 18/19 | 19/19 | 0 | 19/19 | 10.59s |

`0051`：

- Existing 错误；
- Natural / Hybrid 选择 `B08D32JM5F` acceptable。

仍不能因 1 个 task 替换生产。

---

## 9. 最终候选决策

### Existing PreferenceAwareScorer

**保留生产默认。**
P20 E2E 18/19，hard 0，稳定性高。

### ShopRanker Natural Memory

**继续 Shadow；可进入下一轮独立用户偏好评测；暂不进入生产试用。**

- P20 E2E 19/19；
- CF Natural 24 FINALIZE / 8 ASK（Rule），24/7+1 error（Qwen）；
- 对 Memory 变化敏感，但没有独立 Gold；
- 没有足够证据替换生产。

### Fixed Hybrid

**修复后仍只保留为实验 Shadow，不进入生产试用。**

- 修复后 Rule 28 FINALIZE/4 ASK/0 budget；Qwen 28/3/1 planner error；
- 但 P20 E2E 18/19，**0054 失败**；
- Silver 0.731 是规则同源；
- Negative direction 11/17，不稳定；
- 没有独立人类标签。

### DCN-V2

**不进入 Agent 生产候选。**
保留历史 ESCI 结果与迁移失败诊断。

---

## 10. 工程交付

### 关键文件

```text
runs/agent_ranker_phase44/
  hybrid_budget_fix.json
  cf_matrix_rule.json
  cf_matrix_qwen.json
  cf_ranking_metrics.json
  cf_shop_scores.json
  phase44_summary.json
  blind_review_tool.html
  gpu_runtime.json
  experiment_manifest.json

docs/AGENT_RANKER_PHASE44_CF_FIX_AND_HUMAN_EVAL.md
```

### 代码

```text
experiments/agent_ranker_e2e/run_cf_matrix.py
experiments/agent_ranker_e2e/score_cf_gpu.py
experiments/agent_ranker_e2e/build_cf_ranking_metrics.py
experiments/agent_ranker_e2e/build_hybrid_fix_report.py
experiments/agent_ranker_e2e/build_blind_review_tool.py
experiments/agent_ranker_e2e/build_phase44_summary.py
experiments/agent_ranker_e2e/build_phase44_runtime.py
```

### 约束

```text
生产 Scorer / Gate / Planner / P18 修改: no
ShopRanker 重训: no
Qwen 重训: no
DCN 重训: no
BGE 重建: no
Hybrid 权重调整: no
HUMAN_GOLD 伪造: no
Git Push: no
```

**Phase 4.4 完成：Hybrid budget_exhausted 已定位为 executor 状态复用 Bug 并最小修复；Rule 96 条、Qwen 96 条 CF E2E 已运行；Natural CF 完整覆盖；40 对盲评工具可用但 Human Gold 仍为 0；生产默认保持 Existing Scorer。**
