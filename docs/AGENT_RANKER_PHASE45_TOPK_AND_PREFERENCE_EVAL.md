# Phase 4.5 — Top-K Agent E2E, ASK Quality & Independent Preference Evaluation

> 本轮收敛当前 Ranking Refresh：实际运行 P20 **Top-1/3/5 Agent E2E Shadow**（57 条），完成 Qwen Top-3 Explanation Shadow、ASK 质量诊断、cf010 解析定位、40 对盲评导入接口，并给出三种 Ranker 的最终候选决策。
> 没有训练/替换模型；生产默认仍为 `PreferenceAwareScorer`。HUMAN_GOLD 仍为 0。

---

## 0. 结论先行

### Top-K P20 Agent E2E（19 tasks × 3 rankers）

| Ranker | Hit@1 | Hit@3 | Hit@5 | MRR | Valid Top-K items | Evidence | Hard Violations | ASK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing Scorer | **17/19** | **19/19** | **19/19** | 0.947368 | 85 | 85/85 | 0 | 0 |
| ShopRanker Natural | **19/19** | **19/19** | **19/19** | **1.000000** | 85 | 85/85 | 0 | 0 |
| Fixed Hybrid | **18/19** | **19/19** | **19/19** | 0.973684 | 85 | 85/85 | 0 | 0 |

- `0051`：
  - Existing：acceptable `B08D32JM5F` 在 Top-3 第 2 位，Hit@1 失败；
  - Natural / Hybrid：`B08D32JM5F` Top-1。
- Top-K 每件商品都经过独立 eligibility/evidence 检查；hard violation **0**。
- 85 个有效 Top-K 商品均带 evidence；19 个任务中 4 个任务只有 2–3 个可展示候选，未强行补足 Top-5。

### ASK 质量

Rule Planner 的 4 个 arm 分歧 case：

- `cf004_negative_over_ear / mutated`
- `cf005_negative_microphone / mutated`
- `cf009_withdraw_negative_over_ear / baseline`
- `cf013_unknown_negative_over_ear / mutated`

Qwen Planner 在**同样的 4 个 case/state** 上也出现同样分歧。因此总共 8 个 (planner, case, state) 分歧项。
这些 case 中 Natural 触发 ASK，Existing/Hybrid 选择 FINALIZE。
**没有独立用户偏好 Gold，全部标为 `UNJUDGED`；不能仅凭 Existing 会 FINALIZE 就判定 Natural 的 ASK 错误。**

Qwen 聚合（Phase 4.4 实际 96 条）：

| Arm | FINALIZE | ASK | Error |
|---|---:|---:|---:|
| Existing | 28 | 3 | 1 |
| Natural | 24 | 7 | 1 |
| Hybrid | 28 | 3 | 1 |

Qwen 的 3 个 error 全部是同一 conflict case `cf010` 的 Planner JSON 截断，不是 ASK 策略问题。
完整 Qwen 96 条已恢复保存：[cf_matrix_qwen_full.json](runs/agent_ranker_phase44/cf_matrix_qwen_full.json)。

### cf010 Parser

- 原始 Qwen 返回长度 **1128**，在 `"RER...` 处截断；
- Parser 收到 **invalid planner JSON**，不是空 action；
- 两次上限重试产生同样截断；
- **不是 Parser Bug**，是模型生成截断；未跳过 Validator、未强转 ASK、未强制 FINALIZE。

### Human

```text
40 tasks
HUMAN_GOLD: 0
HUMAN_SINGLE_REVIEW: 0
```

没有人工标签，模型偏好准确率 **unavailable**。

---

## 1. ASK Decision Quality

结果：`runs/agent_ranker_phase45/ask_quality_diagnostics.json`

### 1.1 分歧 case

Rule Planner 中 Natural 比 Existing/Hybrid 多 4 个 ASK：

| Case | State | Existing | Natural | Hybrid | Classification |
|---|---|---|---|---|---|
| cf004 negative_over_ear | mutated | FINALIZE | ASK | FINALIZE | UNJUDGED |
| cf005 negative_microphone | mutated | FINALIZE | ASK | FINALIZE | UNJUDGED |
| cf009 withdraw_negative_over_ear | baseline | FINALIZE | ASK | FINALIZE | UNJUDGED |
| cf013 unknown_negative_over_ear | mutated | FINALIZE | ASK | FINALIZE | UNJUDGED |

- 没有独立用户标签证明哪一边正确；
- Natural 的 ASK 可能来自其 top1 排序把 negative/unknown 相关候选放到 Gate 更容易触发澄清的位置；
- 这属于 **Ranker 排序 → Gate 状态耦合**，不是 DecisionGate 阈值被修改的证据；
- 没有调整生产 Gate，也没有为了 FINALIZE 率修改 Natural 的分数尺度。

### 1.2 明确冲突

- `cf010`、`cf011` mutated 状态：所有 arm 都保持 ASK（Rule）或在 Qwen 中记录错误/ASK；
- Rule 下 conflict ASK 6/6（两 case × 两 states × 三 arms 中所有 mutated 项）；
- 没有把明确冲突强行 FINALIZE。

### 1.3 结论

**Natural 的额外 ASK 是“可能的过度澄清”候选，但本轮无法证明为错误；状态为 UNJUDGED。**
需要独立人类偏好/证据审查，而不是改 Gate。

---

## 2. cf010 Qwen Parser 定位

结果：

- [cf010_parse_diagnosis.json](runs/agent_ranker_phase45/cf010_parse_diagnosis.json)
- [cf010_trace.jsonl](runs/agent_ranker_phase45/cf010_trace.jsonl)

### 2.1 证据

```text
trace records: 18
invalid payloads: 6
attempts per invalid: 1 and 2
raw lengths: 1128
tail: ... allowed_actions list does not include ASK, only RETRIEVE, FINALIZE, RER
response errors: ["invalid planner JSON"]
```

- Qwen 生成长 rationale，`max_new_tokens=256` 截断 JSON；
- Parser 正确拒绝无效 JSON；
- 现有 bounded retry 仍是同一截断；
- 没有自动转 ASK 或 FINALIZE。

### 2.2 结论

**不是 Parser Bug；是生成截断 + 无效 JSON。**
不修改生产 Planner。若未来需要解决，应单独测试更大的 max tokens / 更短 rationale prompt，并记录为推理配置变化。

---

## 3. Top-K Agent E2E Shadow

代码：`experiments/agent_ranker_e2e/run_topk_e2e.py`
结果：`runs/agent_ranker_phase45/topk_e2e.json`

### 3.1 链路

```text
Qwen Planner
-> Original BGE Top-20 retrieval
-> Event-Sourced Memory Writer/Reader
-> 专业 Ranker (Existing / Natural / Hybrid)
-> Eligibility + per-candidate Evidence validation
-> DecisionGate
-> Top-K Presenter
```

- Qwen 只负责 Planner/决策，不重新排列商品；
- Top-K 只来自专业 Ranker 的已排序候选；
- 每个 Top-K 候选独立检查 hard eligibility 和 evidence；
- Gate ASK 时不强行输出 Top-K；
- 不清洗 unjudged 商品为负例；
- 没有完整分级标签，**不报告标准 nDCG**。

### 3.2 P20 Top-K 结果

| Arm | Hit@1 | Hit@3 | Hit@5 | MRR | Valid | Evidence | Hard | ASK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing | 17/19 | 19/19 | 19/19 | 0.947368 | 85 | 85/85 | 0 | 0 |
| Natural | **19/19** | **19/19** | **19/19** | **1.0** | 85 | 85/85 | 0 | 0 |
| Hybrid | 18/19 | 19/19 | 19/19 | 0.973684 | 85 | 85/85 | 0 | 0 |

- Valid items 85 / 95 possible：4 tasks 只有 2–3 个可展示候选（不足 5 个），没有补造；
- 每件商品都带 ranker score、verified evidence、memory match/conflict、unknown、hard eligibility；
- Top-K Presenter p50 **0.03 ms**；
- E2E p50：Existing 10.89s、Natural 9.98s、Hybrid 10.19s；
- hard violation **0**。

### 3.3 0051

```text
Existing top3:  B07BKPKXTT, B08D32JM5F, B004K6R43G
Natural top3:   B08D32JM5F, B004K6R43G, B07BKPKXTT
Hybrid top3:    B08D32JM5F, B004K6R43G, B07BKPKXTT
```

Existing 的 acceptable 只在第 2 位；Natural/Hybrid Top-1 正确。

---

## 4. Grounded Top-K Explanation Shadow

结果：`runs/agent_ranker_phase45/topk_explanations.json`
验证：`runs/agent_ranker_phase45/explanation_metrics.json`

对 Natural Top-3 的 19 个任务生成两种解释：

### A. Deterministic Evidence Template

- 每件商品只陈述 verified attributes；
- UNKNOWN 明确保留；
- 不添加价格/评分/库存/保证；
- 所有 19 个任务覆盖全部 Top-K items。

### B. Qwen `/explain` Shadow

- Qwen 不选择商品、不改顺序；
- 实际 19 个任务调用；
- 平均延迟 **5346.2 ms**，最大 **8824.4 ms**；
- Candidate ID consistency **17/19**；
- Ranking order consistency **17/19**；
- Attribute claims **174**；
- Unsupported attributes **1**（`sweat-resistant` 不在 verified attributes 中）；
- 2 个任务只返回首个商品（0050、0051），因此 order consistency 失败。

结论：Qwen 解释可用作 Shadow，但不能自动视为忠实；需要人工复核。
生产回答仍使用确定性 Evidence Template。

---

## 5. 40 对盲评导入与独立偏好

### 工具

- [blind_review_tool.html](runs/agent_ranker_phase44/blind_review_tool.html)
- [blind_review_tasks.jsonl](runs/agent_ranker_phase43/blind_review_tasks.jsonl)
- [blind_review_template.csv](runs/agent_ranker_phase43/blind_review_template.csv)

### 导入接口

代码：`experiments/agent_ranker_e2e/import_blind_review.py`

支持：

- A / B / BOTH / INSUFFICIENT；
- Notes；
- Reviewer ID；
- Task ID；
- 输入 CSV 不存在/为空时：human_review_count=0；
- BOTH/INSUFFICIENT 不进入 A/B agreement 分母；
- 不自动生成标签。

### 当前状态

```text
tasks: 40
HUMAN_GOLD: 0
HUMAN_SINGLE_REVIEW: 0
review_provenance: UNJUDGED
model_agreement: unavailable_no_human_labels
```

没有真实人工输入，因此不报告：

- Existing / Natural / Hybrid Pairwise Agreement；
- Positive / Negative / Withdrawal / UNKNOWN human group 指标；
- 真实用户满意度。

这 40 对是 **Disagreement-Set Human Preference Evaluation**，不是随机用户样本。

---

## 6. GPU 与部署成本

### Top-K E2E

```text
Qwen Planner resident: 17,551 MB
Top-K Presenter p50: 0.03 ms
E2E p50: 9.98–10.89 s
E2E p95: 13.92–14.45 s
hard violations: 0
OOM: 0
```

### Qwen / ShopRanker 并发

```text
Qwen Planner: 17,551 MB
ShopRanker 20-candidate concurrent batch peak: 1,917.29 MB
nvidia total: 19,067 MB
No OOM
```

### 其他

```text
ShopRanker CF scoring: 7.763 s, peak 3,572.42 MB
Qwen CF matrix wall: ~1012 s (Phase 4.4)
CPU Memory Adapter: 0.0704 ms / 20 candidates
Ranker-only Shop: 109.978 ms
Ranker-only Hybrid: 110.09 ms
Qwen Top-3 explanation mean: 5.35 s
```

部署：Qwen + ShopRanker 并发 20-candidate 实测可运行，但余量约 5.5GB；生产路径仍应保留顺序加载/卸载作为稳定方案。

---

## 7. 最终 Ranker Candidate Decision

### Existing PreferenceAwareScorer

**保留生产默认。**
P20 Top-1 17/19，Top-3/5 19/19，hard 0；稳定、低风险。

### ShopRanker Natural Memory

**当前最强实验候选；保留 Shadow，可进入下一轮受控试用评估，但不能直接替换生产。**

支持证据：

- P20 Top-1/3/5 **19/19**；
- Top-K evidence 85/85，hard 0；
- 0051 Top-1 正确；
- E2E p50 ~9.98s。

限制：

- ASK 比 Existing 多 4 个（Rule），Qwen 聚合 Natural 7 vs Existing 3；ASK 正确性无独立 Gold；
- 没有 HUMAN_GOLD；
- 40 对盲评尚未审核；
- Qwen 解释不是 100% faithful（17/19 order，1 unsupported）。

### Fixed Hybrid

**保留实验 Shadow，不进入生产试用。**

- P20 Top-K Hit@1 18/19，0054 仍失败；
- Silver 0.731 与规则同源；
- Negative direction 11/17；
- CF/ASK 没有独立 Gold；
- 不能仅凭 Top-K 指标宣称真实偏好收益。

### DCN-V2

**不进入 Agent 生产候选。**

---

## 8. 阻碍生产替换的最小剩余证据

1. **40 对独立人工盲评**（至少 15–20 对初审，最终 40 对）；
2. 人工标签下的 Natural vs Existing vs Hybrid pairwise agreement；
3. ASK 质量的人工审查：Natural 的额外 ASK 是合理谨慎还是过度澄清；
4. Top-K 的完整分级标签（当前只有 observed acceptable，不能算完整 nDCG）；
5. 对 Qwen Explanation 的人工 claim 审核，确认 unsupported/semantic 问题。

在这些完成前：

- 生产默认：**Existing Scorer**；
- 最强实验候选：**ShopRanker Natural**；
- 继续 Shadow：**Hybrid**；
- 排除：**DCN-V2 B**。

---

## 9. 工程交付

### 代码

```text
experiments/agent_ranker_e2e/run_topk_e2e.py
experiments/agent_ranker_e2e/run_topk_explanations.py
experiments/agent_ranker_e2e/build_ask_quality.py
experiments/agent_ranker_e2e/import_blind_review.py
experiments/agent_ranker_e2e/build_phase45_summary.py
```

### 结果

```text
runs/agent_ranker_phase45/
  topk_e2e.json
  topk_metrics.json
  topk_explanations.json
  explanation_metrics.json
  ask_quality_diagnostics.json
  cf010_parse_diagnosis.json
  cf010_trace.jsonl
  human_preference_eval.json
  phase45_summary.json
  gpu_runtime.json
  experiment_manifest.json
```

### 约束

```text
生产 Scorer / Gate / Planner / P18 modified: no
ShopRanker / Qwen / DCN retrained: no
BGE rebuilt: no
Hybrid weights changed: no
HUMAN_GOLD fabricated: no
Git Push: no
```

**Phase 4.5 完成：Top-K Agent E2E 实际运行；Natural Top-1/3/5 19/19；ASK 质量差异已定位为 UNJUDGED 的排序/Gate 耦合；cf010 是 Qwen JSON 截断而非 Parser Bug；40 对盲评工具和导入接口可用但 Human Gold=0；生产默认保持 Existing Scorer。**
