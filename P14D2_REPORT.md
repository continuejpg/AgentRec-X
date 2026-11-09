# P14-D2 — Electronics Dense Retrieval Repair

> 范围：P14-D2。修复 P13-R Electronics catalogue 的身份对齐 BGE Dense Index，在 135 条真实 Amazon-C4 Dev 上做自然检索对照，并在 BGE 固定候选池上重做 Qwen3 Query Only / Real History / Shuffled History 消融。
> 不训练/微调，不补 62% history metadata，不扩展 Benchmark，不改 V0/Gate/Planner，不运行公开 Test 或项目 Reserved Test，不 commit。

## 0. 结论

- **BGE Dense Index 修复成功**：76,263 个身份与 P13-R accepted catalogue **完全一致**（0 missing / 0 extra / 0 duplicate），替代了旧 P8 index（156,746 个 identity，与当前 catalogue 交集 0，样本为 Books；未复用）。
- **自然检索显著改善**（Original Query）：BM25 Recall@20/100 = **10/20**，BGE = **27/66**，Hybrid RRF = **22/47**。
- **Hybrid RRF 没有在 BGE 上产生增量**（22 < 27）；固定参数、未调权重。
- 在 BGE original-query Top-20 固定候选池（target reach **27/135**）上重做历史消融：
  * Query Only 13/135，Real History **15/135**，Shuffled History **16/135**；
  * Real History 略高于 Query Only，但 **低于 shuffled 负对照**，不能把差异归因于真实历史；
  * 目标可达从 10 提升到 27 后，真实历史仍没有独特增益。
- **结论：密集检索修复成功并可采纳到检索层；但真实历史个性化 Ranker 仍无证据支持继续投资。**

## 1. 旧索引问题与修复

旧 `runs/p8_bge/bge_base_dense.npz`：

- item_count = **156,746**；
- 与 P13-R Electronics accepted item_id 交集 = **0**；
- 样本标题：Curious George at the Baseball Game / BodyBoss Ultimate Body Fitness Workout Guide / Ben Hogan's Five Lessons（Books 类）；
- 判定：错误的 catalogue identity，**不复用**。

新索引 `runs/p14d2/bge_index.npz`：

- 文档身份来自 `ElectronicsCatalogue.load()` 的 **76,263** 个 accepted item，按 item_id 排序；
- identity set 与 catalogue **完全相等**：0 missing、0 extra、0 duplicates；
- 向量维度 768，CLS pooling + L2 normalize；文档数 76,263；
- 使用 `BAAI/bge-base-en-v1.5`（revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`）；
- 文档文本 = title + "\n" + catalogue text，max_length 512，batch 32，fp16；
- index 文件 213,173,605 bytes；`catalogue_docs` 与 `bge_retrieval` 均可核验；
- identity hash：`673ff68e7e3618e3ecbc38d297a11a89ff94b1544a1e096a4f570ec188bde65c`；
- index sha256：`fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d`；
- query vectors 与 BGE retrieval 结果均由同一个 index/模型生成；没有用 evaluator label 排序。

## 2. 自然检索对照（135 条真实 Dev）

Original Query 和 Released Rewritten Query **分开统计，不混合**：

| Query | Method | Recall@20 | Recall@100 |
| --- | --- | ---: | ---: |
| Original | BM25 | 10/135 | 20/135 |
| Original | **BGE** | **27/135** | **66/135** |
| Original | Hybrid RRF (k=60) | 22/135 | 47/135 |
| Rewritten | BM25 | 18/135 | 30/135 |
| Rewritten | BGE | 20/135 | 56/135 |
| Rewritten | Hybrid RRF (k=60) | 17/135 | 48/135 |

- BGE 在 Original Query 上 Recall@20 从 10 → 27（+17），Recall@100 从 20 → 66（+46）；
- Rewritten Query 上 BGE Recall@100 56 vs BM25 30，但 Recall@20 只有 20 vs 18；
- **Hybrid RRF 在两个 query 上都没有超过单独的 BGE**；固定 k=60、等权重，未根据 Dev 调参；
- 目标 not in catalogue = **1/135**；
- 若施加 Product Type/Role 过滤，目标会被过滤的任务 = **6/135**（该过滤是本项目模拟任务的设计，不适合作为真实 catalogue 召回前置，已单独报告）；
- 不注入 Target：所有自然检索均在完整 catalogue 上执行。

## 3. 固定候选池与历史消融

采用**运行前固定规则**：选择 Recall@20 最高的 BGE/Hybrid，且必须超过 BM25；BGE original-query Top-20 胜出，作为固定候选池（target reach 27/135）。Target 不注入、所有方法共享该池。

Qwen3-Reranker-4B（NF4 4-bit，同一 P14-D1/D1-R instruction，同一 original query）：

| 方法 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 条件 n=27 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE Top-20（不重排） | 5/135 | 12/135 | 19/135 | 0.063 | 5/27 |
| **Query Only** | **13/135** | **22/135** | **27/135** | 0.125 | **13/27** |
| Real History (recovered metadata) | **15/135** | 22/135 | 26/135 | 0.139 | 15/27 |
| Shuffled History | **16/135** | 23/135 | 27/135 | 0.139 | 16/27 |

- Real History 比 Query Only +2 Hit@1，但 **Shuffled History 也比 Query Only +3，且比 Real History 高 1**；
- Real vs Query Only：improved 9 / worsened 5；Shuffled vs Query Only：improved 8 / worsened 6；两者高度相似；
- **没有可归因于真实历史的独特增益**；历史摘要的差异无法压过负对照。
- 这与此前结论一致：密集检索修复了 target 可达性，但历史信号仍不足。

## 4. 成本

| 项 | 成本 |
| --- | --- |
| BGE index | 76,263 docs；加载 1.83 s；编码 78.67 s；峰值 519.1 MiB 额外显存 |
| Query encode | original 0.09 s；rewritten 0.03 s |
| BGE index 文件 | 213 MB（本地 runs/p14d2） |
| Qwen history ablation | 8,100 pairs；加载 21.29 s；总 450.89 s；峰值 6,256.2 MiB |
| 训练/微调 | 无 |

## 5. 最终回答

**Dense Retrieval 是否改善真实目标可达性？** **是**：Original Query Recall@20 10→27，Recall@100 20→66；身份已对齐，旧错误索引未被复用。

**Hybrid 是否提供增量？** **否**：固定 RRF(k=60) 在两个 query 上均低于 BGE；没有调参篡改结果。

**在可比较候选池上，真实历史是否优于 Query Only 与 Shuffled？** **否（无独特增益）**：Real 15 vs Query 13 vs Shuffled 16（Hit@1）；真实历史略高于 Query Only，但低于 shuffled，且 gains/losses 结构与 shuffled 几乎一致。**不能声称真实历史带来增益。**

**是否继续投资真实历史个性化 Ranker？** **不继续（当前证据 NO-GO）。** Dense retrieval 修复值得保留到检索层；真实历史 Ranker 需要新的数据/监督来源（更高 history metadata 覆盖、目标可达率、真实用户行为目标），不是当前的 135 条 Dev 上继续调模型。

## 6. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `DENSE_RETRIEVAL_REPAIR` | **PASS** | 76,263 identity 精确对齐；BGE Recall@20/100 显著超过 BM25 |
| `HYBRID_RETRIEVAL` | **FAIL** | 固定 RRF 无增量，低于 BGE |
| `HISTORY_INCREMENT` | **FAIL** | Real 15 vs Query13 vs Shuffled16；无独特增益 |
| **P14-D2** | **GO（仅检索修复）** | 索引修复和有身份验证的真实对照完成；不代表真实历史 Ranker GO |
| `CONTINUE_REAL_HISTORY_RANKER` | **NO-GO（当前）** | 无历史增量证据；不继续在这批数据上调 Ranker |

> Dev 仍是 P13-R 已使用过的 exposed split；target 时间未独立重算；标签为 observed positive；无人工核验。未训练/SFT、未改 V0/Gate/Planner/Memory/旧标签、未运行公开 Test/Reserved Test、未 reset/停止 AutoDL 实例、未 commit。

## 7. 产物

- `P14D2_REPORT.md`、`P14D2_SOURCE_DIFF.txt`
- `runs/p14d2/`：`bge_index.npz`（213MB）、`query_vectors.npz`、`bge_retrieval.json`、`catalogue_docs.jsonl`、`queries.jsonl`、`natural_retrieval.json`、`pools/improved_natural.jsonl`、`scores_history_ablation.json`、`comparison.json`、`per_task.jsonl`、`dense_index_manifest.json`、`model_manifest.json`、`decision.json`
- `experiments/p14d2/{natural_retrieval,evaluate}.py`；`tests/test_p14d2_dense.py`：6 passed（相关回归另行记录）
