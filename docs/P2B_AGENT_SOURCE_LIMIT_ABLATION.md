# P2B.3 — Agent 源预算消融：案例级代理查询 + 全链路埋点

> 本阶段只做 Phase 2B.3（A→E）。边界见 §9。结论按 **直接证据 / 有限样本观察 / 未验证假设**
> 三级标注；工程质量与推荐效果分开判定（§1.4）。

## 0. TL;DR

- **2B.2 的“目标仅 5% 进池”确认是输入混杂，已消除。** 旧驱动对 40 个案例发同一条消息
  `"recommend something for me"`，导致 `catalog_search` 查询恒定：两个商品在 **40/40** 个
  Top-5 里固定出现（第三个 25–27/40）。2B.3 用 `history[-2:]` 标题逐案例构造代理查询后，同一
  指标下最高单商品频次降到 **4/40**。
- **同一 40 案例、`requested_k=5`、四臂（5/20/50/100）重跑**：池 9.9 → 39.0 → 95.1 → 187.1，
  `HR@5` 四臂恒为 **1/40**，`pool_coverage` 仅 1/40 → 2/40（配对 Δ=+0.025，95% CI **[0, 0.075]**）。
- **四分支归因**：`never_returned` 39/39/38/38，`filtered` **恒为 0**，`pooled_not_top_k` 0/0/1/1，
  `top_k` 1/1/1/1。没有目标被 grounding / 去重 / 过滤丢失；**38/40 在 depth=100 内两个源都没返回**。
- **本实验能说的**：在这套输入下，**Agent 源预算不是 Top-5 的瓶颈**（池放大 19 倍，命中率不变）。
- **本实验不能说的**：不能据此断言“源召回差”。代理查询与目标标题的中位 Jaccard 仅 **0.029**，
  它衡量的是一种内容近似查询，不是真实购买意图。

---

## 1. 结论

### 1.1 直接证据（artifact 里可复核的读数）

| # | 证据 | 位置 |
|---|---|---|
| E1 | 40 个案例的代理查询 **40/40 互不相同**；每臂最终 Top-5 列表 **40/40 互不相同** | `cases.json`、`samples.jsonl.ranked` |
| E2 | 与 2B.2 是**同一批 40 案例**（`case_index` 2000–2039，顺序一致）、同一 `requested_k=5`、同一控制面；四臂唯一变量是 `AGENTRECX_AGENT_SOURCE_LIMIT` | `config.json`、`cases.json` |
| E3 | 每源 `requested_limit == effective_limit`（capped=0），两源每臂都返回满额（short=0），失败调用 0；池 = `2·arm − 跨源重复`（0.1 / 0.975 / 4.875 / 12.925） | `samples.jsonl.sources.*`、`dedup.*` |
| E4 | `history` 是 **pre_fused**：Serving V1，`score_kind=rrf_rank_fusion`，内层 `sequential / two_tower / catalog_lexical`，`head_depth=100`，`rrf_k=60` | `config.json.history_source`、`samples.jsonl.sources.history` |
| E5 | 用接受的 `reciprocal_rank_fusion` 回放的融合顺序与运行时**逐案例一致 160/160**（presented 与 final prefix 两处校验） | `samples.jsonl.fusion` |
| E6 | `filtered = 0`（四臂）：没有任何目标被 grounding 或去重丢弃 | `attribution.json.by_arm` |
| E7 | 在 depth=100 内：`history` 只返回 **2/40** 个目标（源内 rank 20、21），`catalog_search` 只返回 **1/40**（rank 1） | `summary.json.arms`、`samples.jsonl.target_outcome` |
| E8 | 延迟四臂基本持平：p50 1528.8 / 1586.5 / 1557.4 / 1617.9 ms；`history` ≈1.15–1.17 s，`catalog_search` ≈0.33–0.35 s | `summary.json.arms.*.p50_ms` |
| E9 | **直通性校验**：新驱动 `--message-mode fixed` 重跑同一 40 案例，复现 2B.2 的聚合读数（池覆盖 0/1/2/2、HR 0/0/0/0、均值池 10.0/39.93/99.8/199.75）→ 观测器不改变行为 | `/tmp/p2b_40a_repro_v1`（临时复现，非交付产物） |

### 1.2 有限样本观察（n=40，正例 2）

- **O1 池覆盖几乎不动**：`pool_coverage` = 1/40 (2.5%) / 1/40 / 2/40 (5.0%) / 2/40。50 与 100 相对
  5 的配对差 +0.025，95% CI **[0, 0.075]**，**跨 0**，不能排除“无差异”。
- **O2 Top-5 命中不变**：`HR@5` = `NDCG@5` = 1/40（0.025）四臂相同；唯一命中在 rank 1。
- **O3 加深预算可以把 history-only 目标推得更深**：case 2011 在 arm 50 进池（`fused_rank=38`），
  到 arm 100 反而变成 **43**。机制见 §5.3（RRF 下 catalog 头部条目压过 history rank 21）。
- **O4 同一用户，输入一变结论就变**：case 2001 在 2B.2 进池但不进 Top-5；2B.3 里因为代理查询
  逐字命中了它的标题词，成为 `catalog_search` rank 1、最终 rank 1。这说明“是否命中”对输入高度敏感。

### 1.3 未验证假设 / 不能下的结论

- **H1 38/40 未返回 ≠ 源召回差。** 代理查询与目标标题中位 Jaccard 0.029，18/40 零共享词；
  BM25 与 Serving V1 的语义泛化能力在这个输入下没有被公平测量。真实用户意图下的召回**未知**。
- **H2 只测了 Agent 预算，没测内层深度。** `history` 的内层 `head_depth` 固定 100；加深 Agent
  预算只是保留更多已融合的头部，不会改变 Serving V1 的召回。更深内层是否救得回目标是**未验证**的。
- **H3 唯一 BM25 命中是“同系列近重复”。** case 2001 的种子商品与目标同属 Razor EcoSmart 系列，
  查询词与目标标题共享 7 个词。这类复购场景下 catalog 会异常有效，但样本只有 1 例。
- **H4 40 例是连续块**（2000–2039），不是随机抽样；不能外推到全量用户或其它类目。
- **H5 计时不可跨机器比较**：包含 `/mnt/d` 9p 文件系统与单进程热缓存开销。

### 1.4 判定

| 层 | 判定 | 依据 |
|---|---|---|
| 工程闭环 | **PASS** | 40 案例 × 4 臂全部 `status=ok`；同案例配对；埋点自洽（E3/E5）；固定模式复现 2B.2（E9）；旧目录未覆盖；相关测试 37 passed |
| 推荐质量 | **NOT ESTABLISHED** | n=40、正例 2；HR@5 差为 0，池覆盖差 CI 跨 0 |
| 瓶颈位置 | **有限证据支持“不在 Agent 源预算这一层”** | 池 9.9→187.1（≈19×），Top-5 命中不变；`filtered=0`；38/40 在 depth=100 外 |

---

## 2. 本阶段改了什么（A / B / C）

### 2.1 A — 案例级代理查询

**规则**（`experiments/p2b_agent_source_limit_ablation.py:build_proxy_query`）：

1. 取 `history[-2:]`（`history = asins[:-2]`，即序列倒数第 3、第 4 个商品）；
2. 每个商品取标题，用与受信词法源相同的 token 形状 `[^\W_]+` 小写化，各取前 **6** 个词；
3. 按顺序拼接、去重（保留首次出现），空格连接；最多 12 个词（≤ `MAX_SEARCH_TERMS=16`）。

**泄漏防护**：

- `items[-1]`（test target）与 `items[-2]`（validation target）**不在 `history` 里**，结构上不可达；
- `attach_proxy_queries` 断言 `target not in seeds`；
- `config.json` 记录 `proxy_query.seed = "history[-2:]"`、`seed_offsets_from_sequence_end = [-3, -4]`、
  `held_out_items_read = []`；
- 40 个案例中 0 个缺少种子标题（`proxy_titles` 均非空）。

**为什么放在 `message` 里**：生产 `catalog_search` 的查询只由 `context.user_request`（即请求
`message`）经 `_search_arguments_from_message` 构造。让案例的 message 等于代理查询，就**不需要改
生产查询构造**，同一套 token 化/截断规则原样生效。

**标注**：`cases.json` 每个案例带 `query_source: "history-title-proxy"`、`proxy_seed_asins`、
`proxy_titles`、`proxy_terms`、`proxy_query`。缺标题的案例会被排除并记录
`excluded_reason`（本次为 0）。

### 2.2 B — 最小埋点

**不新建账本**：所有源级事实来自 ①源调用观测器 ②响应里的 `grounded` 视图（现有 ledger provenance）
③trajectory/observation。观测器是**直通**的，只记录、不改写返回、不重排、不重评分。

> 本路径只注册 `history` 与 `catalog_search`，没有 TIGER，因此任务单里提到的
> `TigerSourceTool.last_outcome` / `last_proposals` 在本实验中没有对象可复用；实际复用的是
> ledger provenance、`grounded` 视图和 `observation`。

**为什么用 in-place 方法替换而不是包装对象**：`RecommendFromHistoryCapability` 会
`isinstance(tool, RecommendationTool)` 校验，替换对象会让合法部署变成“不支持”；因此只替换实例的
`run` / `candidates` 方法，对象本身与其类型不变。

字段与位置：

| 需求 | 字段（`samples.jsonl`） |
|---|---|
| 每源 requested / effective limit | `sources.history.requested_limit` / `.effective_limit`；`sources.catalog_search.*` |
| 是否被下调 | `sources.*.capped`（本次全为 false） |
| 返回数 | `sources.*.returned` |
| 源内排名（目标） | `sources.*.rank_of_target` |
| 耗时 | `sources.*.ms`；`sources.history.internal.timings_ms`（Serving V1 内层拆分） |
| 失败 / 截断 | `sources.*.failed`、`.exhausted_short_of_limit`、`.internal.history_truncated` |
| 去重前 / 后 | `dedup.returned_total` / `dedup.unique` / `dedup.duplicates` / `.overlap_history_catalog` |
| 目标是否被过滤 | `target_outcome.filtered` / `.filtered_reason` / `.grounded_status` |
| fused_rank | `target_outcome.fused_rank`（附 `fusion.matches_presented_order` 校验） |
| 最终排名 | `target_outcome.final_rank`、`ranked`（完整 Top-k） |
| **pre_fused** | `sources.history.pre_fused = true`；`sources.catalog_search.pre_fused = false` |

`history` 的 `internal` 还保留 **目标在 Serving V1 内部** 的 `fused_rank` / `sources` /
`source_ranks`，用于说明“外层 Agent 预算”和“内层融合”是两件事。

### 2.3 C — 运行

- 四臂 `(5, 20, 50, 100)`，同一 40 案例、同一 `requested_k=5`、`sources=("history","catalog_search")`；
- 输出写到**新目录** `runs/p2b_agent_source_limit_40b/`，驱动拒绝覆盖任何已存在目录；
- 旧目录 `runs/p2b_agent_source_limit_40/` 未触碰（summary.json 修改时间仍为 2026-10-24 09:18:36）。

---

## 3. 输入与产物

| 文件 | 规模 | SHA-256 |
|---|---|---|
| `runs/p2b_agent_source_limit_40b/config.json` | 1,415 B | `a0cdb7eff4e4186a9f63ddec3a38d5c74e73862c354415ecd716ec2c57f01714` |
| `runs/p2b_agent_source_limit_40b/cases.json` | 57,105 B | `db7f8072f03e95e183c02d8df5e3d702323b7dd8c4bd212eb095431f2fab4d4c` |
| `runs/p2b_agent_source_limit_40b/samples.jsonl` | 561,329 B | `c66600cbe82f6f1e7b4d32e840ad895bf36bbfe9c76fa034f09b4aaefc07e5d7` |
| `runs/p2b_agent_source_limit_40b/summary.json` | 4,970 B | `548d787b9be1ddcb8cc8d75e1d1bf27e12f76571b36fb1bb82e4a4d1e92082e1` |
| `runs/p2b_agent_source_limit_40b/attribution.json` | 10,016 B | `a60e7174d6d11912c0bda741ccd4b52ade0191f01135393421983124af18b488` |
| `experiments/p2b_agent_source_limit_ablation.py`（驱动） | — | `ea076db6d020cbc052e4ca5a06ff6ba123820c9c9e273d4438287d23d4fc2368` |

`SHA256SUMS.json` 由驱动在写完全部产物后生成；`config.json.driver_sha256` 与上表驱动哈希一致
（已用 `Get-FileHash` 逐项复核）。

关键配置（`config.json`）：

```json
{
  "cases_requested": 40, "cases_usable": 40, "excluded_cases": [],
  "skip": 2000, "requested_k": 5, "message_mode": "proxy",
  "history_source": {"pre_fused": true, "engine_score_kind": "rrf_rank_fusion",
                     "profile": "serving_v1",
                     "inner_sources": ["sequential", "two_tower", "catalog_lexical"],
                     "inner_head_depth": 100, "inner_rrf_k": 60},
  "catalog_search_source": {"pre_fused": false, "score_kind": "bm25", "catalogue_size": 156746},
  "git_head": "d9aeed96272f5ac7f476ddf1c1e69407c0656794"
}
```

> `git_head` 是运行时的提交；本次改动（驱动 + 文档 + 新测试）**未提交**，驱动内容以
> `driver_sha256` 为准。

---

## 4. 结果

### 4.1 四臂汇总（`summary.json`）

| arm | 均值池 | pool_coverage | HR@5 | NDCG@5 | never_returned | filtered | pooled_not_top_k | top_k | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 9.90 | 1/40 (2.5%) | 1/40 | 1/40 | 39 | 0 | 0 | 1 | 1528.8 | 2049.0 |
| 20 | 39.03 | 1/40 (2.5%) | 1/40 | 1/40 | 39 | 0 | 0 | 1 | 1586.5 | 2023.0 |
| 50 | 95.13 | 2/40 (5.0%) | 1/40 | 1/40 | 38 | 0 | 1 | 1 | 1557.4 | 2091.8 |
| 100 | 187.08 | 2/40 (5.0%) | 1/40 | 1/40 | 38 | 0 | 1 | 1 | 1617.9 | 2065.3 |

配对差（相对 arm 5，bootstrap 2000 次，seed 20261024）：

| 指标 | 20 − 5 | 50 − 5 | 100 − 5 |
|---|---|---|---|
| `hr@5` | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] |
| `ndcg@5` | 0.0 [0, 0] | 0.0 [0, 0] | 0.0 [0, 0] |
| `pool_coverage` | 0.0 [0, 0] | **+0.025 [0, 0.075]** | **+0.025 [0, 0.075]** |

### 4.2 源级读数（预算确已到达源）

| arm | history req/eff/ret | catalog req/eff/ret | 跨源重复/例 | 池 = 2·arm − 重复 | history ms | catalog ms |
|---|---|---|---|---|---|---|
| 5 | 5/5/5 | 5/5/5 | 0.10 | 9.90 | 1152.4 | 334.8 |
| 20 | 20/20/20 | 20/20/20 | 0.975 | 39.03 | 1167.1 | 346.5 |
| 50 | 50/50/50 | 50/50/50 | 4.875 | 95.13 | 1166.9 | 330.8 |
| 100 | 100/100/100 | 100/100/100 | 12.925 | 187.08 | 1170.8 | 328.5 |

- `capped=0`：没有任何组件把预算改小；`short=0`：两源从未少于请求量。
- **延迟几乎不随预算增长**：`history` 恒 ~1.15–1.17 s，`catalog_search` 恒 ~0.33 s。原因：
  Serving V1 的**内层**永远读 `head_depth=100`，BM25 永远给全库 156,746 条打分；Agent 预算只
  决定“拷贝/采纳多少条”，不决定检索深度。这也解释了 E3 的线性池增长。

---

## 5. 归因（D）

### 5.1 四分支

定义：`never_returned` = 该臂深度内无源返回；`filtered` = 有源返回但未成为 grounded 池候选；
`pooled_not_top_k` = 进池但 fused_rank > k；`top_k` = 进入最终 k。

| arm | 从未出现在任何来源 | 被过滤掉 | 进池但排名靠后 | 进 Top-5 |
|---|---|---|---|---|
| 5 | 39 | 0 | 0 | 1 |
| 20 | 39 | 0 | 0 | 1 |
| 50 | 38 | 0 | 1 | 1 |
| 100 | 38 | 0 | 1 | 1 |

- **`filtered=0` 是有限定条件的读数**：本实验没有硬约束（`constraints` 为空），能丢弃目标的
  只有 grounding。为 0 说明没有目标因身份校验/去重丢失，**不能外推**到有约束的场景。
- **`never_returned` 是主分支（95%）**，且它的含义是“在 depth ≤ 100 内没有返回”，不是“源永远
  找不到”。深度上限是本实验的边界，不是源的全体排序。

### 5.2 只有两个目标可见

| case | 目标 | 目标标题（截断） | 与代理查询共享词 | 源内排名 | fused_rank | 最终 |
|---|---|---|---|---|---|---|
| 2001 | `B07X5FNDMT` | Razor EcoSmart SUP Electric Scooter – 16" Air-Filled Tires… | 7（`ecosmart sup electric scooter 16 air up`） | catalog 1（全臂）；history 20（arm≥20） | 1 | **1** |
| 2011 | `B087H8X7WH` | FINIS Training Swim Fins | 1（`swim`） | history 21（arm≥50） | 38（arm 50）/ **43**（arm 100） | — |

- 2001 是**同系列近重复**：种子商品 `B09GXNCVJ9` 与目标同属 Razor EcoSmart 系列，查询词几乎
  逐字命中目标标题；它解释了**唯一**的 Top-5 命中，不能作为语义泛化的证据。
- 2011 是 **Two-Tower 内层**捞回来的（`history_internal.sources=["two_tower"]`，内层 rank 4），
  BM25 没返回它。它证明 Serving V1 的内部融合有内容泛化能力，但外层预算只决定把它保留到哪一步。

### 5.3 为什么 2011 越深反而越靠后

`RRF(k=60)` 下，只有 history 来源的候选 rank 21 得分 `1/(60+21)=1/81≈0.01235`。任何 catalog
候选只要 rank ≤ 20，得分 `≥1/80` 就高于它；rank 21 持平（按 identity 破平）。因此：

- arm 50：catalog 只有 50 条，池里约 40 条排在它前面（含两源重复合并）→ `fused_rank=38`；
- arm 100：catalog 扩到 100 条，更多条目获得高于 1/81 的权重 → `fused_rank=43`。

**这是“池更大 ≠ 排名更好”的直接观测**：加深预算会增加**竞争条目**，history-only 目标的相对位次
可能下降。要改变这一点需要改融合权重/排序，而本阶段明确禁止。

---

## 6. 与 2B.2（`runs/p2b_agent_source_limit_40/`）对照

两次运行是同一 40 案例、同一 `k`、同一路径；差异只有“消息”和“埋点”。

| 指标 | 2B.2（40a） | 2B.3（40b） |
|---|---|---|
| 每案例消息 | 同一条固定串 | 案例级代理查询 |
| catalog 查询数 | 1（恒定） | **40/40 不同** |
| 最高单商品 Top-5 频次 | **40/40**（`B07DW3N2GZ`、`B07Z7JT5VW`；第三名 25–27/40） | **4/40**（arm 5/20）、**3/40**（arm 50/100） |
| 每臂不同 Top-5 列表 | 39/40 | 40/40 |
| pool_coverage（5/20/50/100） | 0 / 1 / 2 / 2 | 1 / 1 / 2 / 2 |
| HR@5（四臂） | 0 / 0 / 0 / 0 | 1/40（全部臂） |
| 池内案例 | `{2001}` / `{2001, 2011}` | 同左 |

**读法**：

1. 40a 的“5% 进池”不是召回结论：固定消息把两个商品钉进 40/40 个 Top-5，池覆盖是那个输入的函数。
2. 40b 的“5% 进池”也不能当召回结论：它恰好来自两个**内容上可检索**的用户（一个同系列近重复、
   一个 Two-Tower 泛化），其余 38 个在 depth=100 外。
3. 两次的**池内案例集合相同**（2001、2011），说明“哪些目标能被检索到”主要由**用户/商品内容**
   决定，而不是由消息决定；消息决定的是**用哪个源、以什么 rank** 检索到。

**直通性校验（E9）**：用新驱动 `--message-mode fixed` 重跑同一 40 案例，池覆盖 **0/1/2/2**、
HR **0/0/0/0**、均值池 **10.0 / 39.93 / 99.8 / 199.75**，与 2B.2 记录一致——观测器没有改变
任何行为。该次复现还补出了旧运行没有的 `fused_rank`：两个进池目标分别固定在 rank **40** 与
**42**，即“进池但远在 Top-5 之外”；2B.3 中同样的两个目标变成 1 与 38/43（因为 catalog 列表变了）。

---

## 7. 限制（必须与结果一起读）

1. **代理查询不是真实意图**：标题词拼接是内容近似，不是用户想买什么；中位 Jaccard 0.029，
   18/40 零共享词。它解决了“所有案例同一查询”的混杂，但没有解决“查询是否代表意图”。
2. **深度上限 100**：`catalog_search` 与 Agent 预算都不超过 100；`never_returned` 只覆盖这一深度。
3. **`history` 内层深度固定 100**：本消融只动外层预算，Serving V1 的召回不受影响（E4/H2）。
4. **样本**：40 例连续块、2 个正例；二项 95% CI 很宽（1/40 约 0.1%–13%）。没有做显著性检验，
   `pool_coverage` 的 CI 已跨 0。
5. **没有硬约束**：`filtered` 分支在本实验里只可能由 grounding 触发，为 0 不代表有约束时也为 0。
6. **计时**：`/mnt/d` 9p 文件系统、单进程热缓存；p50 差异（1529→1618 ms）不应当作性能结论。
7. **`catalog_search` 的 `matched` 标志**未单独记录（`candidates()` 不暴露），样本里的
   `returned=0` 即“零命中”。

---

## 8. 复现

```bash
# 本阶段结果（WSL Ubuntu-22.04 + 仓库自带 Linux venv；Windows 侧无法直接执行 .venv/bin/python）
.venv/bin/python -m experiments.p2b_agent_source_limit_ablation \
    --cases 40 --out runs/p2b_agent_source_limit_40b

# 复现 2B.2 的固定消息行为（同一驱动、同一 40 案例集；输出到新目录，避免覆盖）
# 本次已验证：池覆盖 0/1/2/2、HR 0/0/0/0、均值池 10.0/39.93/99.8/199.75
.venv/bin/python -m experiments.p2b_agent_source_limit_ablation \
    --cases 40 --out /tmp/p2b_40a_repro_v1 --message-mode fixed

# 相关测试（只跑这三个文件，不跑全量）
.venv/bin/python -m pytest -q \
    tests/test_p2b_proxy_query.py tests/test_p2b_metrics.py tests/test_agent_source_limit.py
# -> 37 passed
```

派生指标（不在 `summary.json` 里，可由 `samples.jsonl` 复算）：

```powershell
# 每臂 Top-5 中单商品最高出现次数：40a = 40/40，40b = 4/40（arm 5/20）或 3/40（arm 50/100）
Get-Content runs/p2b_agent_source_limit_40b/samples.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.arm -eq 5 } |
  ForEach-Object { $_.ranked } | Group-Object | Sort-Object Count -Descending | Select-Object -First 3
```

---

## 9. 变更文件与本阶段边界

**新增/修改**

| 文件 | 变更 |
|---|---|
| `experiments/p2b_agent_source_limit_ablation.py` | A：代理查询构造与标注；B：直通观测器、per-source/去重/融合回放/四分支；C：`--message-mode`、`attribution.json`、`SHA256SUMS.json`、config 增补 |
| `tests/test_p2b_proxy_query.py` | 新增 14 项纯函数测试（代理查询、泄漏防护、四分支） |
| `runs/p2b_agent_source_limit_40b/` | 新产物（`runs/` 已被 `.gitignore` 忽略） |
| `docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md` | 本文档 |

**没有做**（§约束）：没有改生产查询构造、RRF、排序策略、`requested_k` 默认值；没有动冻结评测器 /
预注册 / 历史报告 / 检查点；没有跑全量测试、没有训练、没有起服务、没有调用外部 LLM；没有自动提交 Git。

**建议下一步**（不在本阶段执行）：

- 用真实/半真实查询（而非标题代理）重跑同一四臂，才能把“代理查询弱”与“源召回弱”分开；
- 若要验证 `history` 内层深度，需要单独的 `head_depth` 消融（会触及 Serving V1 配置，需另行预注册）；
- 对 2001 这类“同系列近重复”单独分层统计，避免它主导小样本结论。
