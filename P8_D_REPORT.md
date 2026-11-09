# P8 后续：多用户策略数据与训练依据报告

## 0. 结论

**已建立可复用的 300 用户 Dev 候选/策略数据集，并单独完成当前购物意图检索检查。但不具备训练 SFT-v1 的条件。**

| 项 | 判断 |
|---|---|
| 多用户数据产物 | **GO / VERIFIED** |
| 当前购物意图检索 | **LIMITED（无真实 Query-Product relevance 标签）** |
| Source Policy 排序差异 | **VERIFIED** |
| 推荐质量增益 | **LIMITED：只提升已命中目标的排序，不增加 Recall** |
| Policy 选择学习问题 | **NO-GO：没有 balanced/intent 严格胜出的样本；仅有固定 history_focus 优势** |
| SFT-v1 | **NO-GO（本轮不训练、不生成正标签）** |

## 1. 样本规模、来源与划分

| 项 | 值 |
|---|---|
| 数据集 | `runs/p8_d/multiuser/` |
| 总样本 | **300** |
| 已评估 | 300；skipped 0 |
| Canonical P6 Dev | 157（剔除 3 个 target-in-history case） |
| 扩展 Dev | 143 |
| 扩展来源 | 同一 `Sports_and_Outdoors_sequences.json`，同一 `sha256(user_id)%5` 用户划分 |
| Frozen Holdout | **0 个 Holdout 用户被选中** |
| Target | `items[-2]` validation target |
| Test target | `items[-1]` 未写入数据集、未用于 Query |
| History | `items[:-2]`，时间有序；target_in_history = 0 |
| History candidate budget | Top-50 |
| Catalog candidate budget | Top-50 |
| Candidate cache 复用 | 118 个 P8-C case |
| 构建耗时 | 385.79s（真实工具；非全 412,445 用户扫描） |

Temporal protocol 仍是 P6 的 `temporal_leave_two_out`；扩展样本只是同一 Dev split 中
canonical 文件因 `MAX_DEV=160` 未纳入的额外用户，不是新的 Holdout/test 口径。

### 可复用产物

- `runs/p8_d/multiuser/cases.jsonl`
- `runs/p8_d/multiuser/candidate_cache.jsonl`
- `runs/p8_d/multiuser/policy_results.jsonl`
- `runs/p8_d/multiuser/summary.json`
- 数据说明：`P8_D_DATA_CARD.md`

每条 candidate/policy 记录明确包含：

- 输入 history；
- validation target；
- `query` 与 `query_source`；
- history/catalog raw candidate ranks；
- target 是否在各来源中；
- `reachable_by_source`；
- 三 Policy 的 Top-10、target rank、Recall/NDCG/MRR；
- strict win / tie / unreachable 标记；
- pre-retrieval / post-retrieval features。

## 2. 两类标签分别代表什么

### 2.1 真实交互标签（next-item relevance proxy）

标签是用户在真实 Amazon Sports 交互序列中的下一个验证商品（`validation_parent_asin`）。
它表示“该用户在该时间点之后实际交互的商品”，用于 next-item ranking proxy。

**它不是用户搜索 Query 的相关性标签，也不是用户搜索满意度。**
Query 是从历史商品类别/标题推导的代理查询，不是真实用户搜索词。

### 2.2 当前购物意图标签（synthetic task proxy）

使用 `data/p6_benchmark_v0/dynamic_dev.jsonl` 的 6 个 Dev 合成任务：

- 有明确商品角色（main/accessory）；
- 有硬价格约束；
- 有合成 seed ASINs。

这里没有人类 Query-Product relevance 标签；只报告：

- role MATCH/UNKNOWN/MISMATCH；
- price eligibility；
- synthetic seed coverage。

合成 seed ASIN 不是人工相关性标签，不能拿来声称真实检索质量。本轮未使用 ESCI，
也没有把 ESCI 的商品 ID 映射伪装成当前 catalogue 的标签。

## 3. 来源候选可达率

300 个真实 Dev history 上：

| 来源 | target 可达数 |
|---|---:|
| History Top-50 | 14 |
| Catalog Hybrid Top-50 | 1 |
| Union | **15 / 300（5.0%）** |
| 仅 history | 14 |
| 仅 catalog | 1 |
| 双来源 | 0 |

候选规模：

- history Top-50 mean = 50.0；
- catalog Top-50 mean = 50.0；
- union mean = 98.98；
- 两来源平均重叠 = **1.02**。

**285/300 的 validation target 不进入任一来源 Top-50；任何 Policy 都无法恢复这些目标。**

## 4. 三种 Source Policy 的真实质量差异

全 300 用户：

| Policy | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| balanced | 0.0200 | 0.01105 | 0.00831 |
| intent_focus | 0.0000 | 0.00000 | 0.00000 |
| history_focus | 0.0200 | **0.01497** | **0.01333** |

Target 可达的 15 个用户：

| Policy | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| balanced | 0.4000 | 0.2210 | 0.1662 |
| intent_focus | 0.0000 | 0.0000 | 0.0000 |
| history_focus | 0.4000 | **0.2995** | **0.2667** |

严格 pairwise（可达样本）：

- history_focus vs balanced：**5 个更好，10 个相同，0 个更差**；
- history_focus vs intent_focus：6 个更好，9 个相同，0 个更差；
- balanced 与 intent_focus 都没有严格胜出样本。

因此：

- **Source Policy 确实改变排序**：全 300 中 intent/history 的 order changed 100%，set changed 99%；
- **但没有增加 Recall**：balanced/history 都是 0.0200（全部），可达子集都是 0.4000；
- 收益只来自把已进入候选池的 target 从较低位置前移；
- `intent_focus` 在本数据上没有收益，反而稀释了 history target。

## 5. 严格胜出、并列与不可达

| 结果 | 数量 |
|---|---:|
| 严格唯一胜出（strict unique win） | **5** |
| 并列（tie） | **10** |
| 不可达（unreachable） | **285** |
| 可达但三个 Policy 都在 Top-10 外 | 9 |

5 个严格胜出样本：

```text
all winners = history_focus
balanced unique wins = 0
intent_focus unique wins = 0
```

10 个 tie 中：

- 9 个是目标可达但都排在 Top-10 外，三个策略都是 0；
- 1 个是 balanced/history 同时把 target 排在 Top-1，intent_focus 为 0，形成 balanced/history 并列。

**没有观察到 balanced 或 intent_focus 严格优于 history_focus 的样本。**
因此在这个 Dev 数据上，“每个用户选择不同 Policy”的问题并没有显现；固定 history_focus
已经弱支配其他两个策略的合法 NDCG。

## 6. Planner 可见特征是否足以预测更优 Policy

可用的 pre-retrieval 特征：

- history length；
- history category count / entropy；
- query length / token count；
- query role target；
- price constraint（当前意图任务）。

这些特征均不包含 validation target、测试目标或 Gold 答案。

结果：

- 严格胜出样本只有 **5** 个，且全部是同一个 policy；
- 没有 balanced/intent 的严格胜出样本，无法构造三类标签；
- 10 个 tie 中没有可用于区分策略的合法质量标签；
- 因此无法训练/评估一个可信的 Policy 选择器；`learnability.status=insufficient`。

即使使用 post-retrieval 的 overlap、role/eligibility 等字段，也只能解释“谁可能有候选”，
不能在这份数据中证明“哪个 Policy 更好”，而且这些字段在首次 RETRIEVE 选择 Policy 时
尚未全部可见。

## 7. 当前购物意图单独评测

6 个 synthetic Dev 购物请求，真实 BM25/BGE/Hybrid，Top-10 inspected：

| Mode | mean role MATCH rate | mean role MISMATCH rate | mean role UNKNOWN rate | mean price eligible rate* | synthetic seed recall** |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.400 | 0.550 | 0.050 | 0.267 | 0.0 |
| BGE | **0.783** | **0.200** | 0.017 | **0.333** | 0.0 |
| Hybrid | **0.783** | 0.217 | 0.000 | 0.300 | 0.0 |

\* At most 3 tasks have a price constraint.
\** Synthetic seed coverage is not human Query-Product relevance.

可说明：

- BGE/Hybrid 在 role proxy 上比 BM25 更集中；
- 但这不是真实搜索相关性评测；
- 合成 seed 全部未命中 Top-10，说明这些 seed 不能作为“相关性 ground truth”；
- 没有可靠的 Query-Product relevance 标签，因此当前购物意图下不能声称 BGE 更“相关”，
  只能报告角色和约束代理。

## 8. 是否具备训练 SFT-v1 的条件

### 结论：**NO-GO**

原因：

1. **No selection problem observed**：5/5 严格胜出都是 history_focus，balanced/intent 没有
   严格胜出；固定 history_focus 已经弱支配其他策略。
2. **No Recall gain**：history_focus 只提升已命中目标的 NDCG/MRR，Recall 完全不变。
3. **Huge unreachable mass**：285/300 target 不在候选池，权重策略无法解决候选可达性问题。
4. **Catalog signal missing**：只有 1 个 target 由 catalog 单独可达，无法形成 history/catalog
   互补决策训练信号。
5. **No reliable current-intent relevance labels**：合成角色/价格/seed 不能当作 SFT 策略标签。
6. **Learnability sample insufficient**：只有 5 个 strict winners，且只有 1 个类别。

本轮没有生成 SFT-v1 正标签，没有训练 LoRA，也没有把 oracle 标签包装成 Planner 监督。

### 若未来继续，最低前置条件

- 扩大真正 target 可达的样本，而不是继续提高样本总数；
- 增加 catalog/query 相关性覆盖，使 history/catalog 都有合法胜出案例；
- 收集独立的人类 Query-Product relevance 标签，或独立的搜索意图任务；
- 在策略选择前可见的 observation 中，验证特征与最优 policy 的真实关联；
- 保持 balanced 默认，不直接用 validation target 或事后 oracle 生成标签。

## 9. 基线冻结与交付物

冻结基线：

- HEAD：`4dec6d2`（工作区有既有未提交改动；本轮未 commit）
- History：Serving V1 现有 SASRec/Two-Tower/BM25 fusion
- Catalog：BM25 + BGE Hybrid
- BGE：`BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Index：`runs/p8_bge/bge_base_dense.npz`，156,746 × 768
- Policies：balanced / intent_focus / history_focus，weighted RRF K=60
- Source budgets：history 50 / catalog 50
- 未修改 Serving V1、Gold/Evaluator、M5、Eligibility、Role Gate、BGE 索引

交付：

- `P8_D_REPORT.md`
- `P8_D_DATA_CARD.md`
- `P8_D_SOURCE_DIFF.txt`
- `experiments/p8d_build_multiuser_dataset.py`
- `experiments/p8d_intent_retrieval.py`
- `runs/p8_d/multiuser/`
- `runs/p8_d/current_intent/retrieval_eval.json`
- `runs/p8_d/baseline_manifest.json`

## 10. 未覆盖 / 限制

- 300 是受本机内存/工具耗时约束的开发规模；未做全 412,445 用户扫描。
- Query 来自历史商品文本，不是真实用户搜索 Query。
- 当前购物意图任务只有 6 个合成 Dev 请求，没有真实相关性标签。
- 扩展 Dev 用户仍来自同一 P6 Dev split；它们不是 frozen Holdout，但不能被描述为新的独立泛化 cohort。
- 本轮没有运行完整 Benchmark 或 Holdout；没有训练 SFT-v1。
- 未 commit/push/deploy；GPU 实例未关闭。
