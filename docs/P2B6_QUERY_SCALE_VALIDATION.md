# P2B.6 — 查询策略规模验证与主线切换

> 目标：在正确的 test history 上，用 1,000 例有界验证 `catalog_search` 的查询覆盖与来源增益，
> 按**预先写死**的停止规则决定是否结束 catalog 查询策略主线，并交付 Learned Ranker 的
> 最小独立评估方案（不执行）。
>
> 不改生产查询构造 / RRF / 排序 / 默认深度；不动冻结评测资产；不提交 Git。
> 结论按 **直接证据 / 有限样本观察 / 未验证假设** 标注。

## 0. TL;DR

- **样本**：1,000 个新案例（artifact index 6710 起，与 2B.5 的 6500–6709 **不重叠**）；
  扫描 1,082、按冻结 20k cohort 精确排除 57、可用 1,000；973 个不同目标，25 个目标重复（最多 3 次），
  test history 中位数 5。
- **三臂（depth=50，k=5，无 TIGER，失败 0）**：

| 臂 | 池覆盖 | HR@5 | NDCG@5 | p50 |
|---|---|---|---|---|
| A history-only | 46/1000 (4.6%) | **11/1000 (1.1%)** | **0.0070** | 1116 ms |
| B + current query | 62/1000 (6.2%) | 10/1000 (1.0%) | 0.0063 | 1301 ms |
| C + balanced query | 59/1000 (5.9%) | 9/1000 (0.9%) | 0.0066 | 1230 ms |

- **配对**：B−A 覆盖 **+1.6 pp [0.9, 2.4]**，HR −0.1 pp [−0.8, 0.6]；C−A 覆盖 **+1.3 pp [0.6, 2.0]**，
  HR −0.2 pp；**C−B 覆盖 −0.3 pp [−1.1, +0.5]**，HR −0.1 pp。均衡查询在规模上**没有**复现
  2B.5 的正向信号，点估计反而更低。
- **停止规则**：R1 ✗（C−B CI 下界 < 0）、**R2 ✓（2 个 catalog 独有目标进入 Top-5，均在 B）**、
  R3 ✗（B 2.2% < 3.0%；C 2.1% < 4.0%）。
- **机制（直接证据）**：catalog 每例新增约 47 个候选，RRF 下 **A 的 11 个 Top-5 命中里有 7 个被挤出
  Top-5**；B 靠 4 个重排提升 + 2 个 catalog 独有命中补回 6 个，净 −1。**catalog 的价值是重排，
  但重排的净效果为负**。
- **决策**：**结束查询策略（C vs B）主线**；均衡查询不设为生产默认；R2 只给 catalog-as-source
  一个“可进入独立生产验证”的资格，但净指标为负，本阶段**不启动**；主线切换到 Learned Ranker
  （方案见 [`LR_INDEPENDENT_EVAL_PLAN.md`](LR_INDEPENDENT_EVAL_PLAN.md)）。

---

## 1. A — 当前状态

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/agent-source-limit-ablation` / `d9aeed9` |
| 未提交变更 | 2B.3（2 改 + 2 新）、2B.4（2 驱动 + 文档）、2B.5（驱动 + 测试 + 文档）、2B.6（驱动 + 文档） |
| 历史产物 | `runs/p2b_agent_source_limit_40b/`、`runs/p2b4_*`、`runs/p2b5_query_coverage_200/` 均保留、未覆盖 |
| 本轮新增 | `experiments/p2b6_query_scale_validation.py`、`docs/P2B6_QUERY_SCALE_VALIDATION.md`、`docs/LR_INDEPENDENT_EVAL_PLAN.md` |
| 运行产物 | `runs/p2b6_query_scale_1000/`（`runs/` 被 gitignore） |

**必须标注的口径问题**：2B.2 / 2B.3 的驱动使用 `history = asins[:-2]`（train history），
不是冻结协议的 `test_history = asins[:-1]`。它们的指标（含“5% 进池”与 2B.3 的源预算结论）
**不能**与 2B.4 之后的结果直接比较；2B.4 已经用同一 40 案例证明，补回 validation 交互后
target 在 Serving V1 头内的 rank 从 20/21 变成 1/8。

**提交方案（不执行）**：按依赖顺序拆 4 个提交：2B.3 → 2B.4 → 2B.5 → 2B.6；`runs/` 不入库。
2B.6 驱动 `import` 2B.5 的辅助函数（`load_cases`、`attach_queries`、`build_sample`、`_aggregate` 等），
因此必须排在 2B.5 之后。

---

## 2. B — 样本与查询

### 2.1 样本

```
skip = 6710（artifact index；2B.5 的 200 例止于 6709）
scanned = 1082
excluded_by_frozen_cohort = 57        # user_int_id ∈ M3/M4/Phase5/B1 的 20,000 用户
selected = 1025（含 25 个缓冲）
usable = 1000（两条查询都非空的案例；unusable = 0）
```

- **任务**：`next_item_prediction`，`history = case.test_history = items[:-1]`，`target = items[-1]`。
- **目标分布**：973 个不同目标；25 个目标出现 >1 次，最大频次 3；无目标缺失元数据。
  目标 `main_category` 前几类：`Sports & Outdoors` 653、无分类 101、`AMAZON FASHION` 73、
  `Amazon Home` 67、`Tools & Home Improvement` 34、`Automotive` 22。
- **history 长度**：min 4 / median 5 / max 103（均 ≤ 500，不触发请求上限）。
- **队列隔离**：20k cohort 精确排除（57 例，5.3%，与 20,000/412,445 ≈ 4.85% 一致）；
  P3（前 1,000）、R-LR 窗口（valid #1001–6000）与 2B.5 样本均按位置避开。R-LR 的逐例划分
  未落盘，因此“按位置在窗口外、划分不可验证”的限制仍然成立（记录在 config）。

### 2.2 查询（与 2B.5 完全相同，未改规则）

| 模式 | 规则 | 来源 |
|---|---|---|
| current | 最近 3 个历史商品标题、≤8 term、长度<3 丢弃、**先到先得** | 冻结 M4 `deterministic_terms` |
| balanced | 同窗口/上限/最小长度，**最近 3 个标题轮转配额**（每标题≤3） | 2B.5 实验实现 |

两条查询只读 `test_history` 的标题；target 不作为参数、不参与构造（2B.5 单测
`test_the_target_is_never_read` 覆盖）。查询与目标 ASIN 无关，**不含**未来交互。

---

## 3. C — 三臂结果（1,000 例，depth=50，k=5）

### 3.1 汇总（`summary.json`；括号内为精确二项 95% CI）

| 臂 | 池覆盖 | HR@5 | NDCG@5 | 池均值 | catalog 独有候选/例 | p50 / p95 ms |
|---|---|---|---|---|---|---|
| A history-only | 46 (4.6% [3.39, 6.09]) | 11 (1.1% [0.55, 1.96]) | 0.0070 | 50.0 | — | 1116 / 1464 |
| B + current | 62 (6.2% [4.79, 7.88]) | 10 (1.0% [0.48, 1.83]) | 0.0063 | 97.09 | 47.09 | 1301 / 1749 |
| C + balanced | 59 (5.9% [4.52, 7.54]) | 9 (0.9% [0.41, 1.70]) | 0.0066 | 96.47 | 46.47 | 1230 / 1657 |

来源失败 0、请求失败 0。B/C 的 p50 相对 A 增加 **+16.6% / +10.2%**。

### 3.2 配对差（同 case，bootstrap 10,000 次）

| 对比 | 覆盖率 Δ [95% CI] | HR@5 Δ [95% CI] | NDCG@5 Δ [95% CI] |
|---|---|---|---|
| B − A | **+0.016 [0.009, 0.024]** | −0.001 [−0.008, 0.006] | −0.0007 [−0.0057, 0.0044] |
| C − A | **+0.013 [0.006, 0.020]** | −0.002 [−0.008, 0.004] | −0.0005 [−0.0053, 0.0046] |
| **C − B** | **−0.003 [−0.011, 0.005]** | −0.001 [−0.005, 0.003] | +0.0002 [−0.0028, 0.0033] |

- **覆盖确实增加了**（B/C 对 A 的 CI 下界 > 0），但 **Top-5 没有增加**（B/C 点估计略低于 A）。
- **C 没有复现 2B.5 的正向信号**：2B.5 的 C−B 是 +0.005 [−0.010, 0.025]；1,000 例上是
  **−0.003 [−0.011, 0.005]**。200 例的正差是噪声。

### 3.3 目标被谁返回

| 臂 | both | history only | catalog only | neither | catalog 命中率 |
|---|---|---|---|---|---|
| A | — | 46 | — | 954 | — |
| B | 6 | 40 | **16** | 938 | **22/1000 = 2.2%** |
| C | 8 | 38 | **13** | 941 | **21/1000 = 2.1%** |

- B 的 catalog 命中 22 = both 6 + catalog-only 16；C 为 21 = both 8 + catalog-only 13。
- **C 相对 B：catalog 命中找回 8、丢掉 9**（净 −1）；这正是“均衡查询在规模上不占优”的来源。
- 两臂的 history 命中都是 46（没有丢失），只是 Top-5 位次被重排。

### 3.4 候选排名证据（demotion 是主效应）

A 的 11 个 Top-5 命中在 B/C 中的去向：

| 指标 | B | C |
|---|---|---|
| 保留在 Top-5 | **4/11**（6948, 6956, 7088, 7258） | **5/11**（6948, 6956, 7088, 7181, 7258） |
| 被挤出 Top-5 | **7/11**（6751 1→7、6781 4→8、7090 2→7、7125 4→14、7181 3→6、7265 2→9、7566 3→9） | **6/11**（6751 1→6、6781 4→8、7090 2→6、7125 4→16、7265 2→10、7566 3→8） |
| 新进 Top-5 | 6 个：4 个重排提升（6756 18→4、7239 37→3、7520 8→1、7702 22→1）+ **2 个 catalog 独有**（7135 →3、7452 →3） | 4 个：6756 18→1、7329 30→4、7520 8→1、7702 22→4；**catalog 独有 0** |
| 净 HR@5 | 11 − 7 + 6 = **10** | 11 − 6 + 4 = **9** |

**R2 的两个命中**（均来自 B/current 查询）：

| case | 臂 | catalog 源内 rank | 融合后 rank | 最终 rank |
|---|---|---|---|---|
| 7135 | B | 2 | 3 | 3 |
| 7452 | B | 2 | 3 | 3 |

**机制**：catalog 每例给池子新增约 47 个候选，每个候选在 RRF 里都有 `1/(60+rank)` 权重；
history-only 命中只有 history 一份贡献，因此被系统性推后。2B.3 在 40 例上看到的“2011 越深越靠后”
（38→43）在 1,000 例上表现为 7/11 的 Top-5 被挤出。**这是 RRF 的既有行为，本轮没有修改它。**

---

## 4. D — 停止规则与判定

### 4.1 规则核对（阈值未改；R3 分母与基线已按 2B.5 核实）

| 规则 | 定义 | 2B.6 读数 | 结果 |
|---|---|---|---|
| **R1** | ≥1,000 例上 C−B 配对覆盖率 CI 下界 > 0 | n=1,000；Δ=−0.003，CI [−0.011, 0.005] | **✗** |
| **R2** | ≥2 个 catalog 独有目标在融合后进入 Top-5 | 2 个（case 7135、7452，均 B） | **✓** |
| **R3** | catalog 目标命中率 ≥ 2× 2B.5 基线 | 基线：B 3/200=1.50% → 3.00%；C 4/200=2.00% → 4.00%。实测：B 22/1000=2.2%（CI [1.38, 3.31]）；C 21/1000=2.1%（CI [1.30, 3.19]） | **✗** |

**R3 口径核对**：2B.5 的“catalog 命中”= 融合臂中 `sources.catalog_search.rank_of_target` 非空的
案例数（不是最终 Top-5 命中），分母 = 200 个可用案例。复算：B = both 1 + catalog_only 2 = 3/200；
C = 1 + 3 = 4/200。2B.6 用同一口径（B 6+16、C 8+13）。

### 4.2 判定

**不是“没有任何条件满足”**：R2 成立。但必须把 R2 的对象说清楚：

1. **R2 是 current 查询（B）的成绩**：两个 catalog 独有命中都是 B 以 catalog rank 2 拿到的；
   均衡查询（C）的 catalog 独有 Top-5 是 **0**。
2. **均衡查询（C）被否定**：R1 不成立且点估计为负；C−B 覆盖、HR 都不占优；200 例的正差没有复现。
   → **不设为生产默认，不再扩样。**
3. **catalog 作为额外来源（B）**：R2 给它“值得进入独立生产验证”的资格，但同一 1,000 例上
   HR@5 10 vs A 11、NDCG@5 0.0063 vs 0.0070，净收益为负（重排收益被 demotion 抵消）。
   → 本阶段**不启动**生产验证；只记录“有独有 Top-5 能力、但当前融合方式净负”的有界发现。
   若未来启动，必须是独立预注册实验，并带 Top-5 demotion 保护指标（见 §7）。

**主线决策**：

- **结束 catalog 查询策略（C vs B）研究主线**；不继续扩样（1,000 例已满足 R1 的规模门槛）。
- 不修改生产默认值、来源计划、RRF、查询构造或候选深度。
- **切换到 Learned Ranker 独立评估**（方案见 §5 与独立文档）。

---

## 5. E — Learned Ranker 最小评估方案（本轮不执行）

完整方案：[`docs/LR_INDEPENDENT_EVAL_PLAN.md`](LR_INDEPENDENT_EVAL_PLAN.md)。要点：

| 项 | 方案 |
|---|---|
| 问题 | 冻结 LR V1 是否在**同一候选池**上优于**正确的全精度 RRF** |
| 为什么重做 | 训练管线把 RRF 限制在池子集（HR@20 3/766）；修正后 RRF 是 30/766，LR 21/766，Δ NDCG@20 = −0.001246 [−0.006189, 0.00407]；A6 另一版 Δ = −0.001587 [−0.006543, 0.003704]。既有结论 NOT ESTABLISHED |
| 数据 | **新 cohort**：artifact index 9000 起，2,000 例；与 LR 训练（#1001–6000）、P3、2B.5/2B.6 均不重叠；排除 20k cohort |
| 固定 | 三源 top-100 并集、depth=100、全精度 RRF、冻结模型/schema、目录/检查点、k=5/10/20 |
| 单变量 | 只换排序器（RRF → LR），不换深度、不换候选池、不重训 |
| 主终点 | NDCG@20（与训练目标、A6 口径一致） |
| 终点/门槛 | G1 配对 ΔNDCG@20 CI 下界 > 0；G2 ΔHR@5 不回归；G3 池身份/特征/哈希/fallback；G4 cohort 独立 |
| 结论上限 | 即使 G1+G2 成立，也只代表“值得独立生产验证”，不等于切换默认 |
| 前置 | 建 `.venv-ranker`（lightgbm==4.7.0 / numpy==1.26.4）；默认 `serving_v1` 不导入 LightGBM |
| 边界 | 这是固定三源 `/v1/recommend` 路径的评估；Agent 动态平面明确拒绝 `LearnedRankerEngine` |

---

## 6. 复现、产物与校验和

```bash
# 2B.6 三臂验证（本轮已跑，约 70 分钟）
.venv/bin/python -m experiments.p2b6_query_scale_validation \
    --cases 1000 --skip 6710 --out runs/p2b6_query_scale_1000

# 相关测试（本轮未新增查询构造，复用 2B.5 的 13 项；共 50 passed）
.venv/bin/python -m pytest -q \
    tests/test_p2b5_query_coverage.py tests/test_p2b_proxy_query.py \
    tests/test_p2b_metrics.py tests/test_agent_source_limit.py
```

| 文件 | 规模 | SHA-256 |
|---|---|---|
| `runs/p2b6_query_scale_1000/config.json` | 2,675 B | `86a36b811be2fe2cf1e4149bc3653dc5f90c0a73418d8e78971745de4d87999c` |
| `runs/p2b6_query_scale_1000/cases.json` | 880,823 B | `107e733c53ef948ba6aa733fc155a9fbffda42988888280844b78a583cbd0b08` |
| `runs/p2b6_query_scale_1000/samples.jsonl` | 7,529,251 B | `a29ed916d946505731d176b4fbc5abb468bbd69e8f5bd835a06603331fe50d42` |
| `runs/p2b6_query_scale_1000/summary.json` | 21,841 B | `f9a86c27b0da03cd3c26f6d568505c4d9d57aea6ae9e53640627278a31259f49` |
| `experiments/p2b6_query_scale_validation.py` | — | `653aa6e90890a2833a4d74423f29cae4d7df8a122c68fcdda986f297f5de989e` |

`config.json` 记录：`skip=6710`、`selection={scanned:1082, excluded_by_frozen_cohort:57, selected:1025,
usable:1000, unusable_queries:0}`、`stop_rules`（R1/R2/R3 定义与 2B.5 基线）、`sample_isolation`、
`sequence_artifact`、`git_head=d9aeed9`、`driver_sha256`（与上表一致，已复核）。
2B.5 驱动未被修改（哈希 `3f896d7d…` 与其产物配置一致）。

---

## 7. 限制与未验证假设

1. **样本量**：1,000 例、A/B/C 命中 46/62/59；单臂 CI 宽约 ±1.5 pp。R1 的规模门槛已满足，
   所以 R1 的“不成立”不是功效不足，而是点估计为负。
2. **R2 只有 2 例**（0.2%）；二项 95% CI 约 [0.02%, 0.72%]。它证明“catalog 独有目标**可以**进
   Top-5”，不能证明“通常能”。
3. **R3 的边界**：B 的 2.2% 低于 3.0% 阈值，但 CI 上界 3.31% 与阈值重叠；结论是“未达到预注册
   翻倍”，不是“翻倍不可能”。
4. **代理查询**：两条查询都由历史标题构造，不代表真实用户意图；本实验不能描述成自然语言购物
   搜索的表现。
5. **history 是 pre_fused**：Serving V1 已含内部 `catalog_lexical`；独立 catalog 与内部通道重叠，
   所以“catalog 独有候选”不等于“新信息”。
6. **demotion 的因果**：本轮只观测到“加入 catalog 后 7/11 的 A-Top-5 被挤出”，没有做融合权重
   实验（约束禁止改 RRF）。若要区分“catalog 候选质量差”与“RRF 权重不适合多源”，需要另立实验。
7. **队列隔离**：20k cohort 精确排除；R-LR 只按位置在窗口外，划分成员不可验证（config 已标注）。

---

## 8. 变更文件与技术债

| 文件 | 变更 |
|---|---|
| `experiments/p2b6_query_scale_validation.py` | 复用 2B.5 的三臂驱动；1,000 例；R1/R2/R3 评估；新产物目录 |
| `docs/P2B6_QUERY_SCALE_VALIDATION.md` | 本文档 |
| `docs/LR_INDEPENDENT_EVAL_PLAN.md` | LR 最小独立评估方案（未执行） |
| `runs/p2b6_query_scale_1000/` | 新产物（gitignore，未覆盖旧目录） |

**技术债（记录，不再本轮处理）**：2B.6 驱动 `import` 了 2B.5 的私有辅助函数
（`_aggregate`、`_paired`、`_hit_attribution` 等）。对一次性实验可接受；若还要第三次复用，
应把这些函数提升为公开的实验工具模块（`experiments/_shared_*.py`），而不是继续跨驱动引用私有名。

没有修改：生产查询构造、RRF、排序、默认候选深度、来源计划、冻结 evaluator / 预注册 / 历史报告 /
检查点；没有跑全量测试或全量评测；没有自动提交 Git。
