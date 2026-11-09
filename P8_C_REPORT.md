# P8-C 可控个性化召回融合与多用户策略评测报告

## 0. 结论

**Source Policy 已经真实进入 Agent 的 history/catalog_search 融合计算，默认 balanced 与旧行为完全兼容；多用户 Dev 回放也证明策略会改变排序和 Top-K 集合。但合法相关性收益只出现在极少数可达目标上，不足以开始 SFT-v1。**

| 项 | 状态 |
|---|---|
| Source Policy 实现 | **VERIFIED** |
| balanced 向后兼容 | **VERIFIED**（数值与旧 unweighted RRF 相同） |
| 未知 policy 拒绝 | **VERIFIED** |
| Planner Contract / Executor / Observation | **VERIFIED** |
| 多用户真实 Dev 回放 | **VERIFIED（120 users / 120 evaluated）** |
| 真实 Qwen Base 最小接入 | **VERIFIED（最小调用；未输出非默认 policy）** |
| 推荐质量增量 | **LIMITED** |
| SFT-v1 | **NO-GO（本轮不训练）** |

## 1. Source Policy 实现

源码位置：

- `recommendation/control/candidate_ledger.py`：
  - `SOURCE_POLICIES`
  - `SOURCE_POLICY_WEIGHTS`
  - `normalize_source_policy`
  - `source_policy_weights`
  - `weighted_reciprocal_rank_fusion`
- `recommendation/control/loop.py`：`_LoopEngine` / `LoopController` 接收 policy；`_fuse_presentable` 使用同一 weighted RRF。
- `recommendation/demo/agent_service.py`：`run_shopping_turn → run → _execute_plan → _compose_controller → _build_controller` 全链路传 policy；CandidateLedger 和 LoopController 使用同一 policy。
- `recommendation/demo/planner_contract.py`：`PlannerObservation.source_policy/source_feedback`；RETRIEVE 可选 `arguments.source_policy`；非法值抛出 `invalid_source_policy`。
- `recommendation/demo/planner_service_executor.py`：从 ActionProposal 读取 policy，传给 `run_shopping_turn`；构建真实 source feedback。
- `recommendation/demo/planner_http_proposer.py`、`training/agent_sft/planner_prompt.py`：向 Planner 暴露 policy 与 feedback。
- `experiments/p6_benchmark/run_dynamic_dev_p64.py`：`--source-policy` 仅用于受控评测，默认 balanced。

权重：

| Policy | history | catalog_search |
|---|---:|---:|
| balanced | 1.0 | 1.0 |
| intent_focus | 1.0 | 2.0 |
| history_focus | 2.0 | 1.0 |

融合公式：

```text
score(d) = Σ_s weight[s] / (60 + rank_s(d))
```

- 同一 ASIN 去重一次；
- tie-break 使用 identity，稳定；
- 只改变 Agent 层 source RRF，不改变 Serving V1 内部的 SASRec/Two-Tower/BM25；
- policy 默认 balanced，且数值上与旧 unweighted RRF 完全一致；
- 未知 policy 在 Planner Contract、Service、Ledger 均被拒绝。

### Observation / feedback

`ServicePlannerExecutor` 只从实际响应构建：

- `source_policy` 与实际 weights；
- history / catalog candidate counts；
- overlap count/IDs；
- fused IDs；
- 每个候选的 source/rank/score_kind contribution；
- role target/counts；
- Eligibility evaluated/eligible/ineligible/unresolved；
- M5 preference/hard constraint/soft preference summary。

为保留历史兼容，history capability 中无 ledger provenance 的 grounded candidates 以单独 accounting 统计，并明确标注；不是伪造 catalog provenance。

### 最小测试

`tests/test_p8c_source_policy.py` 4 passed：

- balanced == 旧 unweighted RRF；
- 权重真实改变融合顺序并去重；
- CandidateLedger 使用 policy；
- Planner Contract 接受合法 policy、拒绝未知 policy。

合并 targeted 测试：

```text
tests/test_p8c_source_policy.py
tests/test_planner_contract.py
tests/test_control_stage2_candidates.py
40 passed
```

源码 diff：`P8_C_SOURCE_DIFF.txt`

## 2. 真实来源与融合证据

### 2.1 真实工具执行

- History：真实 Serving V1 history tool；每用户执行 1 次，Top-50；
- Catalog：真实 Hybrid BM25+BGE；每用户执行 1 次，Top-50；
- BGE index：P8-B 的 `runs/p8_bge/bge_base_dense.npz`；
- 同一份不可变候选快照分别 replay 三种 policy；没有重复调用 BGE。

一个可达样本（`classical-dev-00028`）：

```text
query: stoves & grills camping stoves camping furniture tables
target: B0BKFGDBBM
history top-10: B0BKFGDBBM(1), B0BX5QFWQN(2), ...
catalog top-10: B07C3WK867(1), B0BC2HCYZT(2), ... (target not present)
```

| Policy | target rank | rank change vs balanced |
|---|---:|---|
| balanced | 2 | baseline |
| intent_focus | absent | order+set changed |
| history_focus | **1** | order+set changed |

这证明 source policy 不是只改最终展示顺序，而是真实改变 Agent RRF 中的 source contribution。

### 2.2 Agent 端三策略

同一真实 Agent 请求：

```text
I need a water filter under $40 for camping.
```

- 三策略的最终 role-gated recommendations 都是 `B08XJQ9HPX, B07L47J44Q`；
- 这说明当 role gate 只保留少量合格商品时，最终展示可能相同，但内部 source fusion 已经执行；
- source feedback 记录了实际 history/catalog counts 和 policy weights。

另一个受控动态任务 `normal_termination`：

| Policy | final IDs | task | role |
|---|---|---|---|
| balanced | `B07S8GNB4T, B09FXNJZTV, B08557VLNT, B09YCYNCSK, B091VYKC96` | UNKNOWN | 2/5 |
| intent_focus | `..., B08557VLNT, B09YCYNCSK, B08MDQYHKR` | UNKNOWN | 2/5 |
| history_focus | `B07S8GNB4T, B09FXNJZTV, B091VYKC96, B08557VLNT, B09YCYNCSK` | UNKNOWN | 2/5 |

- Source policy 改变了最终集合与顺序；
- role MATCH 数、task 状态没有改善；
- 不能把“集合变化”当作质量提升。

硬约束任务 `budget_update` 三策略：

```text
balanced/intent_focus/history_focus
final IDs = B095RLKB7Q, B08XJQ9HPX
PASS, 2/2 role MATCH, HardConstraintViolation=0
```

说明在硬约束较强时，policy 差异可能被 Eligibility/角色过滤吸收。

## 3. 多用户策略对照

### 3.1 数据与合法性

- 数据：`data/p6_benchmark_v0/classical_dev.jsonl`，Dev split；
- 用户：**120**；评估成功 **120**，跳过 0；
- History：每用户真实 `history_parent_asins`（train history）；
- Target：`validation_parent_asin`；`test_parent_asin` 与 frozen Holdout 未使用；
- Query：从最近两个历史商品的类别/标题构造，**不使用 validation target**；
- 查询来源：`last_history_categories`；
- 每个来源 Top-50；
- 评测明确标记为 **next-item relevance proxy**，不是真实用户搜索满意度。

Candidate Cache / Replay：

- `runs/p8_c/source_policy_eval/candidate_cache.jsonl`
- 每行保存真实 history 排名、Hybrid Catalog 排名、query、target provenance；
- 三策略在同一缓存上运行同一 `weighted_reciprocal_rank_fusion`，没有重复 BGE 检索；
- `per_user.jsonl`、`summary.json` 保存完整结果。

### 3.2 Candidate 可达性

| 项 | 数量 |
|---|---:|
| users evaluated | 120 |
| target in history pool | 7 |
| target in catalog pool | 1 |
| target in union | **8** |
| mean history candidates | 50.0 |
| mean catalog candidates | 50.0 |
| mean source overlap | 1.1 |
| catalog hybrid query mean latency | 0.675s |

**112/120 的 validation target 不在任一来源 Top-50 中；任何权重策略都无法凭空恢复它。**

### 3.3 指标（全部 120 用户）

| Policy | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| balanced | 0.025 | 0.01126 | 0.00708 |
| intent_focus | 0.000 | 0.00000 | 0.00000 |
| history_focus | 0.025 | 0.01718 | 0.01458 |

在 **target 可达的 8 个样本**上：

| Policy | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| balanced | 0.375 | 0.16883 | 0.10625 |
| intent_focus | 0.000 | 0.00000 | 0.00000 |
| history_focus | 0.375 | **0.25770** | **0.21875** |

- history_focus 在可达子集上提升 NDCG/MRR，但不提升 Recall；
- intent_focus 在这些样本上把 target 排到 Top-10 外，表现更差；
- 只有 8 个可达样本，不能做统计显著性结论。

### 3.4 排序变化

对全部 120 用户：

| Policy | order changed | set changed | mean Jaccard vs balanced |
|---|---:|---:|---:|
| balanced | 0% | 0% | 1.000 |
| intent_focus | 100% | 98.33% | 0.3885 |
| history_focus | 100% | 98.33% | 0.3870 |

**策略执行能力明确存在；质量收益不明确。**

### 3.5 按历史长度分组

| History length | n | balanced NDCG | history_focus NDCG | intent_focus NDCG |
|---|---:|---:|---:|---:|
| 0–3 | 37 | 0.01164 | 0.01705 | 0.000 |
| 4–7 | 55 | 0.01673 | 0.02601 | 0.000 |
| 8+ | 28 | 0.000 | 0.000 | 0.000 |

没有任何一组支持 intent_focus；8+ 组全部为 0，属于候选不可达。

### 3.6 Oracle

在 8 个可达样本中，按 validation target 事后选择：

```text
best policy counts: balanced=5, history_focus=3
mean best NDCG@10 = 0.25770
```

该 oracle **使用验证标签，仅用于分析**，不是 Planner 可达到的成绩，也不能作为 SFT 正标签。

## 4. 真实 Qwen Planner 最小接入

- GPU：RTX 4090D，Qwen3.5-9B Base + `/plan` 可用；
- 直接真实 `/plan` observation 包含：
  - `source_policy="balanced"`；
  - `source_feedback`（history/catalog counts、overlap、fused IDs）；
- 真实 Qwen 输出：

```json
{"action":"RETRIEVE","arguments":{"query":"water filter under $40 for camping"},"rationale":"..."}
```

**Qwen 没有输出 `source_policy`，因此正确使用了 balanced；没有强迫模型假装学会策略选择。**
- `/plan` 延迟约 4.1s；
- 一次真实 Runner `budget_update` Qwen Base 任务：PASS；
- source feedback 实际为 history=5, catalog=10, overlap=0，soft preference summary 可用；
- 没有执行完整 Qwen 三组 Benchmark；没有训练/改 Adapter。

## 5. 五个问题的回答

### 1) Source Policy 是否真正改变推荐执行？

**是。**  
它不是改 response 顺序：weighted RRF 在 CandidateLedger/Loop 融合处真实计算；多用户回放中 100% order changed，98.3% set changed；动态任务最终 ID 也发生改变。

### 2) 是否在不同真实用户历史上观察到策略质量差异？

**LIMITED。**  
在 8 个 target 可达用户上，history_focus 的 NDCG/MRR 明显高于 balanced，intent_focus 为 0；但 112/120 target 不可达，样本量只有 8，不能支持稳定质量结论。

### 3) 固定 balanced 是否仍是总体最好策略？

- 全部 120 用户：history_focus NDCG/MRR 更高，balanced 与 history_focus Recall 相同，intent_focus 为 0；
- 可达子集：balanced 5/8 个样本是最佳，history_focus 3/8；
- 因此 **不能断言 fixed balanced 最好，也不能断言 history_focus 稳定更好**；数据不足以给出统计结论。

### 4) 是否存在合理的学习型策略选择空间？

**存在但很小，且当前不可靠。**  
可观察信号：history length、source overlap、role/eligibility、source feedback；策略确实能改变排序。  
但候选可达性太低，且策略收益只集中在 8 个样本；现有信号不足以训练一个可信的策略选择器。

### 5) 下一阶段是否应该开始 Planner SFT-v1？

**NO-GO（本轮）**。  
理由：

- 没有可重复、可归因的稳定质量增量；
- 120 用户中只有 8 个 target 可达；
- intent_focus 在所有主要指标上都更差；
- 固定 balanced 与 history_focus 的差异不足以构造可靠标签；
- 当前不具备“策略选择标签”的独立验证数据。

## 6. 最终判断

| 维度 | 判断 |
|---|---|
| Source Policy 实现 | **GO / VERIFIED** |
| 排名改变能力 | **VERIFIED** |
| 推荐质量增益 | **LIMITED** |
| SFT-v1 训练 | **NO-GO（本轮不训练）** |

### 下一阶段优先事项

1. 扩大 Dev 用户与**可达目标**数量，而不是直接训练；
2. 分析为什么 catalog_search 对 next-item target 几乎不可达，并单独收集 query-source 对照；
3. 在更多真实可评估样本上验证 history_focus 与 balanced 的差异；
4. 保留 Source Policy 接口和 source feedback，但暂不生成未审核 SFT 正标签。

## 7. 交付物

- 报告：`P8_C_REPORT.md`
- 源码 diff：`P8_C_SOURCE_DIFF.txt`
- Source Policy 代码：`recommendation/control/candidate_ledger.py`、`recommendation/control/loop.py`、`recommendation/demo/planner_contract.py`、`recommendation/demo/planner_service_executor.py` 等
- 多用户回放：`experiments/p8c_policy_eval.py`
  - cache：`runs/p8_c/source_policy_eval/candidate_cache.jsonl`
  - per-user：`runs/p8_c/source_policy_eval/per_user.jsonl`
  - summary：`runs/p8_c/source_policy_eval/summary.json`
- 动态策略检查：`runs/p6_4_dynamic/rule_balanced_20261107_234611/` 等
- Planner 集成测试：`tests/test_p8c_source_policy.py`（4 passed）
- Qwen 最小验证：`runs/p6_4_dynamic/base_20261107_234359/`

## 8. 未覆盖 / 未解决问题

- 未运行全仓库测试；只运行了与本轮相关的 40 个 targeted 测试（全部通过）。
- 本轮开始前已观察到 `tests/test_r1_dynamic_retrieval.py` 的一个既有失败（首个合成水过滤器 turn 的 role gate 把 synthetic history candidates 全部过滤，导致 recommendations 为空）；该失败路径为 BM25/role 路径，不是本轮 Source Policy 代码，未扩展排查。
- Qwen 只做了真实 `/plan` 单次调用和一次 `budget_update` 任务；未运行完整 Qwen 三组 benchmark。
- 112/120 Dev 用户的 validation target 不进入任一来源 Top-50；这是本轮质量结论的最大限制。
- `intent_focus` 在真实 Dev 可达样本中没有收益，不能把它当作已验证的有效策略。

本轮未训练 SFT-v1、未运行 Holdout、未修改 Gold/Evaluator/Serving V1 默认行为/M5/Eligibility/Role Gate、未 commit/push/deploy；GPU 实例未关闭。
