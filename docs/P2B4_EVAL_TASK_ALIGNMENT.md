# P2B.4 — 评测任务对齐与召回来源诊断

> 本阶段只做「确认数据语义 → 选定评测任务 → 诊断召回来源 → 给出下一项最小改动」。
> 不改生产召回/排序/查询构造，不实施生产改动，不提交 Git。
> 结论按 **直接证据 / 有限样本观察 / 未验证假设** 标注。

## 0. 结论（先看这一节）

1. **没有真实购物查询与相关商品标签。** 仓库里只有 Amazon Reviews 2023 的交互序列
   （每用户一条，时间有序）+ 商品元数据 + 单一 next-item 标签。所有“查询”都是**历史的确定性
   函数**（M4 的 `deterministic_terms(h)`、2B.3 的标题代理查询），不是用户写的。→ **分支 2**。
2. **2B.2 / 2B.3 用错了任务历史。** 驱动用 `history = asins[:-2]`（train_history），而冻结协议
   `agentrecx.eval_protocol.v1` 的 test history 是 `asins[:-1]`（train_history + validation）。
   修正后同一 40 案例、同一 depth=50，Serving V1 头里目标的排名从 **20 / 21** 变成 **1 / 8**。
   因此 2B.2/2B.3 的数字是“隔一步预测”，**不能**与 P3 / Phase 5 / M4 的数字并列。
3. **修正任务后的三来源对照（40 例，depth=50，k=5）**：history-only 命中 2/40，catalog-only
   **0/40**，融合 2/40。catalog 每例提供约 48 个历史里没有的候选，但**没有一个是目标**；
   融合相对 history-only 的质量增益为 0，p50 从 1261 ms 涨到 1505 ms（+19%）。
4. **20k 冻结 M4 头（raw source 层）**：catalog_search 对 SASRec / Two-Tower 确有互补性
   （depth 50：540 个命中目标既不在 SASRec 也不在 Two-Tower 头里）。但 Agent 的 `history`
   源是 Serving V1（内部已含 **catalog_lexical** 与 Two-Tower），40 例的 composition 层证据
   显示独立 `catalog_search` 的**边际目标覆盖为 0**。
5. **下一项生产改动：本阶段没有可实施项。** 唯一被证据支持的最小改动在**评测驱动**（把历史改成
   `test_history`）。生产侧是否需要改，取决于下一轮 300–500 例三臂对照的验收（§7.5）。

---

## 1. A — 当前实验状态与提交方案

### 1.1 状态（读取时）

| 项 | 值 |
|---|---|
| 分支 | `feat/agent-source-limit-ablation` |
| HEAD | `d9aeed9 docs(p2b): hand off the source-limit ablation state and the next phase` |
| 工作树 | 2 modified + 2 untracked（2B.3），2 个新实验驱动 + 本文档（2B.4） |
| 运行产物 | `runs/` 被 `.gitignore` 忽略，未提交 |

```
 M docs/P2B_HANDOFF.md
 M experiments/p2b_agent_source_limit_ablation.py
?? docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md
?? tests/test_p2b_proxy_query.py
?? experiments/p2b4_three_source_comparison.py
?? experiments/p2b4_source_diagnosis.py
?? docs/P2B4_EVAL_TASK_ALIGNMENT.md
```

2B.3 的产物校验和已复核（与 `runs/p2b_agent_source_limit_40b/SHA256SUMS.json`、
`config.json.driver_sha256` 一致）：`config.json a0cdb7ef…`、`cases.json db7f8072…`、
`samples.jsonl c66600cb…`、`summary.json 548d787b…`、`attribution.json a60e7174…`、
驱动 `ea076db6…`。

### 1.2 提交方案（不自动执行）

**Commit 1 — 2B.3（先提交，2B.4 的驱动 `import` 它）**

```
experiments/p2b_agent_source_limit_ablation.py
tests/test_p2b_proxy_query.py
docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md
docs/P2B_HANDOFF.md
```

建议信息：`feat(p2b3): case-level proxy queries + per-source instrumentation for the source-limit ablation`

**Commit 2 — 2B.4（纯诊断 + 决策，不含生产代码）**

```
experiments/p2b4_three_source_comparison.py
experiments/p2b4_source_diagnosis.py
docs/P2B4_EVAL_TASK_ALIGNMENT.md
```

建议信息：`docs(p2b4): align the evaluation task and diagnose source complementarity`

**提交前检查（本轮已跑）**

```bash
.venv/bin/python -m pytest -q tests/test_p2b_proxy_query.py tests/test_p2b_metrics.py tests/test_agent_source_limit.py
# 37 passed
git diff --stat && git status --porcelain
# runs/ 下产物不入库；SHA-256 见 §8
```

---

## 2. B — 数据核对（只查与评测直接相关的入口）

### 2.1 可用数据及其真实语义

| 资产 | 内容 | 标签 |
|---|---|---|
| `data/processed/Sports_and_Outdoors_sequences.json` | `agentrecx.sequences.v1`，412,445 用户各一条**时间有序**交互序列（`item_ids` / `parent_asins` / `unix_ms` / `ratings`） | 每条序列只有“最后一个商品”可作为 next-item 正例；`items[-2]` 是 validation 目标 |
| `data/processed/Sports_and_Outdoors_products.jsonl` | 156,746 个目录商品的**元数据**（title/store/categories/features/description/price_text…） | 无相关性、无点击、无查询 |
| `data/processed/Sports_and_Outdoors_mappings.json` | `user2id` / `item2id` / `id2user` / `id2item`，PAD=0，item 1..156,746 | 身份映射，不是标签 |
| `runs/m4_evidence/heads.npz` | M4 冻结：20k 用户的四源 depth-100 头 + 冻结查询串 | 目标 = 同一批用户的 test target（分析时才 join） |
| `data/artifacts/demo/preference_memory.sqlite3` | demo 偏好记忆（无写入） | 不是相关性标签 |

**不存在**：查询日志、点击/曝光日志、qrels、人工相关性标注、用户会话文本。

### 2.2 五个问题的直接回答

| # | 问题 | 回答 | 证据 |
|---|---|---|---|
| 1 | 有真实购物查询 + 相关商品标签吗？ | **没有** | `data/` 只有上表 4 类资产；仓库中的“查询”都是历史的纯函数（`experiments/m4_query_rules.py`），或 2B.3 的标题代理；没有任何 qrels 文件 |
| 2 | 只有交互序列 + 下一商品标签吗？ | **是**（另有一条 validation 目标，test 时它属于历史） | `docs/EXPERIMENTS.md` §4；`recommendation/evaluation/split.py`；单正例 ⇒ HR@K == Recall@K |
| 3 | 有真实需求文本但没有可靠相关性标签吗？ | **没有** | M4 的 `user_message` 是常量 `"recommend something useful for me"`；2B.3 的 message 是标题拼接；demo 消息是固定样例。没有任何用户写的请求被保存 |
| 4 | 这 40 个案例能评估什么？ | **能评估 next-item prediction，但必须先修 history**；**不能**评估自然语言商品搜索 | 见 §3 |
| 5 | 与冻结实验队列重叠吗？ | P3 / Gate 队列：**0**；20k 冻结 cohort：**3/40**；R-LR 窗口：40/40 落在窗口内（划分未落盘） | 见 §2.3 |

### 2.3 与冻结队列的重叠（精确到案例）

| 队列 | 定义 | 与本轮 40 例（`case_index` 2000–2039）的重叠 |
|---|---|---|
| P3 统一评测（`runs/p3_unified*`，已执行 1,000 例） | 前 1,000 个通过协议的有效 case | **0** |
| Gate B-1/B-2/C-1/C-2 | 前 100 个有效 case | **0** |
| M3 / M4 / Phase 5 / B1 的 20,000 用户 cohort（seed `20260201`） | 分层确定性抽样，`user_int_ids` 记录在 `runs/m4_evidence/heads.npz` | **3 / 40**：case `2020`、`2022`、`2025` |
| R-LR（LightGBM）5,000 组队列 | valid case #1001–6000，按 `sha256(case_id)%100` 分 train/val/test；划分成员未落盘 | **40/40 落在此窗口内**；无法从现有产物排除训练/测试重叠 |

**读法**：3/40 与 20k cohort 重叠（若按 20,000 / 412,445 ≈ 4.85% 均匀估算，期望约 1.9 例；3 例在噪声内），必须在报告里写明；本轮用的是
Serving V1（不是 M3/M4/Phase5 的任何模型），这 3 例不影响本轮的来源读数。**但**若以后用这 40 例
评估 LightGBM profile，必须先按 `sha256(case_id)%100` 复算划分，排除 R-LR 训练重叠。

---

## 3. C — 任务判定：分支 2（next-item prediction）

### 3.1 选分支 2 的依据

分支 1 要求“真实查询 + 相关商品标签”，分支 3 要求“真实查询但缺标签”。二者都需要**用户写的查询**，
仓库里不存在（§2.2 第 1/3 条）。因此只能选：

> **任务名：next-item prediction（下一商品预测）**，协议 `agentrecx.eval_protocol.v1`，
> 单正例，全目录排序；**不是**自然语言购物查询评测。

### 3.2 关键发现：2B.2 / 2B.3 的 history 不符合冻结协议

| 项 | 冻结协议（`recommendation/evaluation/split.py`） | 2B.2 / 2B.3 驱动 |
|---|---|---|
| test history | `train_history + (validation_target,) = items[:-1]` | `asins[:-2]`（**train_history**，丢掉 validation 交互） |
| test target | `items[-1]` | `items[-1]` |
| 谁在用正确版本 | P3、Phase 5、B1、M4、`gate_c12` 全部用 `case.test_history` | 只有 2B 驱动用 `asins[:-2]` |

**直接证据（同一 40 案例、同一 depth=50，只改 history 语义）**：

| case | 2B.3（`asins[:-2]`）history 目标 rank | 2B.4（`test_history`）history 目标 rank |
|---|---|---|
| 2001 | 20 | **1** |
| 2011 | 21 | **8** |

其余 38 例两边都不在 depth=50 头内。**结论**：2B.2 / 2B.3 衡量的是“隔一步预测”，比冻结任务更难，
其 5% 覆盖率不能与 Phase 5 的 20.5%（Serving V1 内层 sequential @1000）等数字并列。

**机制**：case 2001 的序列尾部是 paddle board → EcoSmart 滑板车 → 伞绳（paracord，validation）→
目标滑板车。丢掉 validation 后，Serving V1 的最近交互变成伞绳/paddle board，目标掉到 rank 20；
补回 validation 后，内部 `catalog_lexical` 用完整历史元数据在 rank 1 命中目标。

### 3.3 对 2B.2 / 2B.3 结论的影响

- “Agent 源预算不是 Top-5 瓶颈”这一**定性**结论仍成立（池放大 19 倍、命中不变），但它的数字
  属于“隔一步”任务，**不再作为任务对齐后的依据**。
- 2B.3 的 `filtered=0`、`history pre_fused=true`、埋点自洽（160/160 融合回放）不受任务修正影响，
  仍是有效工程证据。

---

## 4. D — 统一比较指标

| 维度 | 指标 | 口径 |
|---|---|---|
| 候选生成 | 目标候选池覆盖率、去重后池大小 | 每 case 一个正例；池 = presentable grounded 集合 |
| 最终推荐 | HR@5、NDCG@5 | k=5，与冻结评测一致 |
| 来源互补性 | 各源独有候选数、目标是否仅被某源返回（`history_only` / `catalog_only` / `both` / `neither`）、目标在源内的 rank | 同一 case 的两源输出直接对比 |
| 运行成本 | p50 / p95、来源失败率、请求失败率 | 同一进程、同一案例集 |

规则：**同一 case 配对**；报 n 与 95% CI；`0` 命中不等于“方案等效”，只等于“本样本未观测到差异”。

---

## 5. 召回来源诊断（两套互补证据）

### 5.1 20k 冻结 M4 头：raw source 层

来源：`runs/m4_evidence/heads.npz`（M4 预注册，`test_history` 语义，depth 100），目标由
`sequences.json` 的 `parent_asins[-1]` join。复现：`experiments/p2b4_source_diagnosis.py`。

| depth | 来源 | 命中 | 命中率 | 唯一命中* | 命中时平均 rank |
|---|---|---|---|---|---|
| 50 | history（SASRec） | 716 | 3.58% | 377 | 19.4 |
| 50 | catalog_search（BM25） | 595 | 2.97% | 170 | 15.7 |
| 50 | two_tower | 813 | 4.06% | 455 | 20.6 |
| 50 | similar_item | 613 | 3.07% | 189 | 16.5 |
| 50 | **四源并集** | **1,936** | **9.68%** | — | — |
| 100 | history（SASRec） | 1,121 | 5.61% | 560 | 39.3 |
| 100 | catalog_search（BM25） | 805 | 4.03% | 204 | 30.8 |
| 100 | two_tower | 1,254 | 6.27% | 639 | 39.3 |
| 100 | similar_item | 819 | 4.10% | 207 | 30.8 |
| 100 | **四源并集** | **2,741** | **13.71%** | — | — |

\*唯一命中 = 该源命中且其余三源都没命中。

互补性（depth 50 / 100）：

| 对 | both | 左独有 | 右独有 | 两头平均 Jaccard |
|---|---|---|---|---|
| history vs catalog_search | 30 / 56 | 686 / 1,065 | 565 / 749 | 0.002 / 0.003 |
| history vs two_tower | 322 / 528 | 394 / 593 | 491 / 726 | 0.106 / 0.118 |
| two_tower vs catalog_search | 43 / 92 | 770 / 1,162 | 552 / 713 | 0.003 / 0.005 |

- **catalog 在 raw source 层很互补**：depth 50 有 **540** 个用户的命中目标既不在 SASRec 也不在
  Two-Tower 头里（depth 100 为 **687**）。
- **但这是对 raw 模型说的**。Agent 的 `history` 源是 Serving V1，内部已经含 `catalog_lexical` 与
  Two-Tower；这些“catalog 独有”的目标很可能已被内部 lexical 通道覆盖——§5.2 的直接证据支持这一点。

### 5.2 Agent composition 层：40 例、修正任务、depth=50

来源：`runs/p2b4_three_source_40/`（本轮执行；history = `test_history`，query = 冻结 M4 规则）。

| 方案 | 池均值 | 目标进池 | HR@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| history only（Serving V1） | 50.0 | **2/40 (5.0%)** | 1/40 (2.5%) | 1261.4 | 1608.2 |
| catalog_search only | 50.0 | **0/40 (0.0%)** | 0/40 | 295.9 | 438.4 |
| history + catalog（融合） | 98.0 | **2/40 (5.0%)** | 1/40 (2.5%) | 1504.5 | 1871.3 |

来源归因（融合臂逐 case 对比两源 rank）：

| 目标被谁返回 | 例数 | 说明 |
|---|---|---|
| both | 0 | — |
| **history only** | **2** | case 2001（rank 1）、case 2011（rank 8） |
| **catalog only** | **0** | — |
| neither | 38 | 两源 depth=50 内都没有 |

独有候选（每例）：catalog 独有 **48.0**、history 独有 **48.0**、共享 2.0 —— 候选集几乎不相交，
但 catalog 的独有候选**不含目标**。融合相对 history-only：`HR@5` Δ=0、`pool_coverage` Δ=0
（95% CI [0, 0]），p50 **+243 ms（+19%）**。

两个命中的机制（Serving V1 内部 provenance）：

| case | 外部 history rank | 内部来源 | 内部 rank |
|---|---|---|---|
| 2001 | 1 | `catalog_lexical` | 1 |
| 2011 | 8 | `two_tower` | 1 |

**case 2001 特别说明**：M4 代理查询是 `exfoil 1000lb black paracord 250 spools rope parachute` ——
最近商品（伞绳）的标题把 8 个 term 配额占满，滑板车相关词根本没进查询，所以 catalog_search 没命中；
而 Serving V1 内部的 `catalog_lexical` 用更宽的历史元数据构造查询，rank 1 命中。**这说明 catalog
的 0/40 是“查询覆盖”问题，不是“BM25 无用”**；也说明独立 catalog 源与内部 lexical 通道高度重叠。

### 5.3 哪个来源最可能提供额外价值？

| 证据层 | 判断 | 强度 |
|---|---|---|
| raw source（20k） | catalog_search 对 SASRec / Two-Tower 有真实互补性 | 强（n=20,000） |
| Agent composition（40 例） | catalog_search 相对 Serving V1 的边际目标覆盖 = **0** | 弱（n=40，2 正例；且被查询覆盖污染） |
| 机制 | Serving V1 内部已有 lexical + Two-Tower，独立 catalog 主要在**重排**而非**扩召回** | 直接（provenance） |

**结论**：在分支 2 下，最可能提供额外价值的**不是**“把 catalog_search 加深/加宽”，而是
**history 源内部已存在的通道**；catalog_search 的价值只有在真实用户查询（分支 1）或查询覆盖修好
之后才能体现。当前证据不支持对它做生产投入。

---

## 6. 三来源最小对照设计（已执行最小版 + 可扩展版）

### 6.1 设计（与 2B.4 任务单一致）

| 项 | 值 |
|---|---|
| 案例 | 40 例，`case_index` 2000–2039（与 2B.3 同一批用户） |
| history | `case.test_history = items[:-1]`（冻结协议） |
| query | 冻结 M4 `deterministic_terms(h)`：最近 3 个历史商品、最多 8 个 term、长度 <3 丢弃 |
| 深度 | 每源 50（`AGENTRECX_AGENT_SOURCE_LIMIT=50`），最终 k=5 |
| 方案 | `history` / `catalog_search` / `history+catalog_search` |
| 配对 | 同 case、同 query、同 k；报告 n 与 bootstrap 95% CI |
| 固定项 | Serving V1 profile、RRF、排序、`requested_k=5` 均未改 |

### 6.2 已执行命令

```bash
.venv/bin/python -m experiments.p2b4_three_source_comparison --cases 40 --out runs/p2b4_three_source_40
```

### 6.3 扩展版（下一轮，未执行）

同一驱动、同一设计，案例扩到 300–500（`--cases 300`），用于把 2 个正例变成 ~15–25 个正例：

```bash
.venv/bin/python -m experiments.p2b4_three_source_comparison --cases 300 --out runs/p2b4_three_source_300
```

**验收门槛（先写死，避免事后挑指标）**

| 编号 | 条件 | 不满足时的动作 |
|---|---|---|
| C1 | `catalog_search` 的目标覆盖率 > 0，且融合覆盖率 − history-only 覆盖率的配对 95% CI 下界 > 0 | 不投生产；不加深 catalog 深度 |
| C2 | 融合覆盖率 ≥ 两单源覆盖率的较大者（融合不得丢目标） | 视为融合缺陷，先查 plane/RRF |
| C3 | 融合 p50 ≤ 1.5 × history-only p50（本轮 1.19×） | 需要单独的延迟预算决策 |
| C4 | history 语义 = `test_history`、query 不含 target、`query_source` 有标注、三臂同案例 | 结果作废重跑 |

**样本量说明**：以本轮 5% 覆盖率为基线，300 例约 15 个正例；2 个百分点的差异仍会 CI 跨 0。
所以 C1 的目的不是“证明显著”，而是“确认是否有任何独有命中”；没有独有命中就不必再加深。

---

## 7. E — 决策与交付

### 7.1 可用数据及其真实语义

见 §2.1。一句话：**交互序列 + 单正例 + 商品元数据；没有查询日志，也没有相关性标签。**

### 7.2 选定的评测任务与依据

**next-item prediction（冻结 test split）**，依据 §2.2 与 §3.1。任何报告必须同时写：
- `history_semantics = case.test_history (train_history + validation_target)`；
- `query_source = frozen-m4-deterministic-terms`（代理，不是用户查询）；
- 评分口径 = 单正例全目录排序，HR@K == Recall@K。

### 7.3 三种来源方案的最小对照设计

见 §6。已执行 40 例最小版；扩展版命令与验收门槛（C1–C4）已固定。

### 7.4 哪个来源最可能提供额外价值

**在分支 2 下：没有证据支持独立 `catalog_search` 提供额外召回价值；Serving V1 的
`history` 源是唯一观测到命中的来源。** 它的内部 `catalog_lexical` / `two_tower` 已经覆盖了
catalog 通道能带来的东西。catalog 的 raw-source 互补性（20k：540/687 个独有命中）是对
SASRec/Two-Tower 说的，不是对 Serving V1 说的。

### 7.5 下一项最小生产代码改动

**本阶段：无。** 理由：

1. 评测任务刚刚对齐；2B.2/2B.3 的数字不能作为生产决策依据（§3.2）。
2. 40 例 composition 证据显示 catalog 边际覆盖为 0，但被查询覆盖污染（§5.2），不足以判死刑，
   也不足以支持投入。
3. 生产查询构造、RRF、默认候选深度在本阶段冻结（§限制）。

**如果**扩展版满足 C1–C4，**那时**的最小生产改动候选（按优先级）：

| 候选 | 改动量 | 触发条件 |
|---|---|---|
| 把 `catalog_search` 加入 demo 默认 plan（当前默认只有 `history`） | 1 行常量 + 文档 | C1 成立且产品接受 +19% p50 |
| 调整 catalog 查询的历史覆盖（per-title 配额） | 生产查询构造，需预注册 | 扩展版证明“查询覆盖”是唯一瓶颈（分支 1 或代理实验） |
| 加深 Serving V1 内层 head（100 → 1000） | 生产 profile，需预注册 | 不在本阶段范围；也不符合“默认候选深度不变”的约束 |

在此之前**不改生产代码**。

### 7.6 具体验收指标与运行命令

```bash
# 1) 任务对齐后的三来源对照（本轮已跑；扩展版把 --cases 改成 300）
.venv/bin/python -m experiments.p2b4_three_source_comparison --cases 40 --out runs/p2b4_three_source_40

# 2) 20k 冻结头来源诊断（本轮已跑）
.venv/bin/python -m experiments.p2b4_source_diagnosis --out runs/p2b4_source_diagnosis

# 3) 回归测试（只跑三个文件，不跑全量）
.venv/bin/python -m pytest -q tests/test_p2b_proxy_query.py tests/test_p2b_metrics.py tests/test_agent_source_limit.py
```

**必须报告的指标**：三方案的目标进池率、HR@5/NDCG@5、池大小、每源独有候选数、目标源内 rank、
p50/p95、失败率；配对差与 95% CI；样本量与正例数。

**判定规则**：

- 若 `catalog_only = 0` 且融合覆盖 = history-only → 记“本样本未观测到 catalog 的边际召回价值”，
  不写“catalog 无效”。
- 若融合 < 任一单源 → 记融合缺陷，不是来源问题。
- 若 95% CI 跨 0 → 只报方向，不报优劣。

---

## 8. 产物与校验和

| 文件 | 规模 | SHA-256 |
|---|---|---|
| `runs/p2b4_three_source_40/config.json` | 1,551 B | `234bf7e19410e65d70d3016a0aa19cecbbf0a132f4dc2cfb625cbcd03640ed68` |
| `runs/p2b4_three_source_40/cases.json` | 27,430 B | `2aea0d294480232efbcd65c5c7fff5dbbf44551513bfc8d6be3d517df1bb8c42` |
| `runs/p2b4_three_source_40/samples.jsonl` | 142,587 B | `46f7f639be9f90cd5d187be2488914dbd9e99a8be23eff90eaaa052351262134` |
| `runs/p2b4_three_source_40/summary.json` | 3,023 B | `83419eca03b0764294077f0c9b1674d58c76460bdc7fab874ed31050b934cb03` |
| `runs/p2b4_source_diagnosis/summary.json` | 4,554 B | `59c7017adc42d0bd4da6a5dbb7a1506ddf6464293a00abd29c638f7359741bfa` |
| `experiments/p2b4_three_source_comparison.py` | — | `8b7b9d7807a8a1089d46ebf6093d7b87fc2ad10225f577908d704874e2ad5fbf` |
| `experiments/p2b4_source_diagnosis.py` | — | `c4c25c2c39782901fe7355762d205c22dce2ed7370b5f2b93e34e9d854cf6192` |

两个运行目录各自带 `SHA256SUMS.json`；`runs/` 被 `.gitignore` 忽略，不入库。

---

## 9. 限制（必须与结果一起读）

1. **样本**：40 例、2 个正例；`2/40` 的 95% CI 约 [0.6%, 16.9%]，`0/40` 约 [0%, 8.8%]。
   任何“来源优劣”都只能作方向性阅读。
2. **代理查询**：M4 规则是历史的纯函数，不是用户查询；它可能因为一个长标题占满 term 配额而漏掉
   目标（case 2001）。这既是限制，也是“为什么不能把代理查询结果说成真实查询效果”的证据。
3. **raw source ≠ Agent composition**：20k 的 `history` 是 SASRec，不是 Serving V1；两张表的
   结论不能互换。
4. **队列重叠**：3/40 与 20k 冻结 cohort 重叠；R-LR 窗口覆盖全部 40 例，划分成员未落盘。
5. **计时**：`/mnt/d` 9p 文件系统、单进程热缓存；p50 差异（1261 → 1505 ms）只在本机有意义。
6. **M4 头的查询**是冻结的 `deterministic_terms(h)`；如果生产查询构造变了，本诊断需要重跑。
