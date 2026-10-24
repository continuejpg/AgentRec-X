# P2B.5 — Catalog 查询覆盖与来源增益

> 本阶段只验证两件事：① 冻结的 catalog 查询构造是否限制目标覆盖；② 独立 catalog 来源是否给
> Agent 控制面带来增量价值。不改生产查询构造 / RRF / 排序 / 默认深度；不提交 Git。
> 结论按 **直接证据 / 有限样本观察 / 未验证假设** 标注。

## 0. TL;DR

- **样本**：200 个协议正确案例（`test_history = items[:-1]`），已排除冻结 20k cohort 的 10 个用户；
  P3 前缀与 R-LR 窗口按位置避开。四臂 A/B/C/D，depth=50，k=5，无 TIGER，来源失败 0。
- **A（history-only）**：目标进池 7/200（3.5%），但**全部在 5 名以外**（融合 rank 7–39），HR@5 = 0。
- **B（+当前查询）**：进池 9/200（4.5%），HR@5 1/200；catalog 独有命中 2 例（rank 27、46）。
- **C（+均衡查询）**：进池 10/200（5.0%），HR@5 1/200；catalog 独有命中 3 例（rank 21、44、40）。
- **C vs B**：净 +1 个 catalog 命中（找回 2、丢掉 1）；配对覆盖率差 **+0.005，95% CI [−0.010, 0.025]**，
  HR 差 0 → **NOT ESTABLISHED**，只能说“均衡预算修正了一个可复现的失败模式，但收益未成立”。
- **唯一 Top-5 提升来自重排，不是扩召回**：case 6676 在 A 已进池（rank 25），catalog 给它 rank 5/33，
  RRF 把它推到 rank 1。其余新增覆盖的融合排名在 43–92。
- **副作用（直接证据）**：catalog 给池子加约 47 个独有候选后，**7 个 history 命中有 6 个被推后**
  （如 7→13、8→22、39→78）。本样本 A 没有 Top-5 命中，所以没测到“挤掉 Top-5”的伤害，但这正是
  下一轮要盯的风险。
- **下一项生产改动：无。** catalog 未表现出可信的增量；均衡查询也不是稳定改进（C−B CI 跨 0）。
  建议保持默认 plan 为 history-only，继续只在实验侧探索。

---

## 1. A — 基线与数据

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/agent-source-limit-ablation` / `d9aeed9` |
| 2B.3 / 2B.4 变更 | 保持未提交、未覆盖（4 个文件 + 2 个 2B.4 驱动 + 2 份文档） |
| 本轮新增 | `experiments/p2b5_query_coverage.py`、`tests/test_p2b5_query_coverage.py`、本文档 |
| 运行产物 | `runs/p2b5_query_coverage_200/`（`runs/` 被 gitignore） |

**任务历史（唯一正确版本）**：`history = case.test_history = items[:-1]`，
`target = items[-1]`（`recommendation/evaluation/split.py`、P3/Phase 5/B1/M4 同一口径）。
2B.2/2B.3 的 `asins[:-2]` 是 train-only，已在 2B.4 里被替换；本轮不再使用。

**样本选择**：

```
skip = 6500（artifact index）
scanned = 210
excluded_by_frozen_cohort = 10     # user_int_id ∈ M3/M4/Phase5/B1 的 20,000 用户
selected = 200
```

- **精确排除**：20k cohort 由 `runs/m4_evidence/heads.npz` 的 `user_int_ids` 定义（seed 20260201），
  逐用户排除，不做近似。
- **按位置避开**：P3 用前 1,000 例；R-LR 用 valid #1001–6000；本轮从 artifact index 6500 开始，
  两者都在窗口外。
- **不可验证的部分**：R-LR 每例的 train/val/test 划分未落盘（只有 `sha256(case_id)%100` 规则），
  因此只能声明“按位置在窗口外”，不能逐案例证明不在训练集。该限制已在 `config.json` 的
  `sample_isolation` 里写明。未发现其他冻结队列。

**查询只用 target 之前的信息**：两条查询都只读 `history`（`test_history`）里的商品标题；
`target` 不作为参数传入（单元测试 `test_the_target_is_never_read` 断言 `describe` 只收到 history
身份，且目标身份不在其中）。查询不含未来交互、不含目标商品标题、不含目标 ASIN。

---

## 2. B — 实验设计

固定项：`AGENTRECX_AGENT_SOURCE_LIMIT=50`、`requested_k=5`、同 200 案例 / 历史 / 目标、
同一目录（156,746）、同一 Serving V1 profile 与检查点、不启用 TIGER、RRF/排序未改。

| 臂 | 来源 | 查询 | 说明 |
|---|---|---|---|
| **A** | history | 不调用 catalog | Serving V1（pre_fused，内部 sequential+two_tower+catalog_lexical） |
| **B** | history + catalog | **current**：冻结 M4 `deterministic_terms(h)`（最近 3 个商品、≤8 term、长度<3 丢弃、**先到先得**） | 现行实验口径 |
| **C** | history + catalog | **balanced**：同一窗口/上限/最小长度，改为**最近 3 个标题轮转配额**（每标题最多 3 个 term，最多 8 个） | 仅在实验驱动实现 |
| **D** | catalog only | current | 去掉 history，量 raw 词法源单独贡献 |

C 的分配规则（`experiments/p2b5_query_coverage.py:balanced_title_terms`）：
最近 3 个标题、每个最多 3 个候选 term、轮转取（第 1 轮每标题第 1 个词 → 第 2 轮每标题第 2 个词 …），
去重、总上限 8。它修的是 B 的已知失败模式：一个啰嗦的最近标题会把 8 个配额吃光。

**记录**（每案例每臂）：两源 `requested/effective limit`、返回数、目标源内 rank、耗时、失败；
去重后池大小；目标进池、融合 rank、最终 rank；catalog 独有候选数；HR@5/NDCG@5；
p50/p95；来源失败率。history 标 `pre_fused=true`，catalog 标 `pre_fused=false`，两者不混为同级原始模型。

---

## 3. C — 结果

### 3.1 四臂汇总（`summary.json`）

| 臂 | 进池 | HR@5 | NDCG@5 | 池均值 | catalog 独有候选/例 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| A history-only | 7/200 (3.5%) | 0/200 | 0 | 50.0 | 0 | 1096.6 | 1477.7 |
| B + current | 9/200 (4.5%) | 1/200 (0.5%) | 0.005 | 97.49 | 47.49 | 1342.8 | 1833.2 |
| C + balanced | 10/200 (5.0%) | 1/200 (0.5%) | 0.005 | 96.21 | 46.21 | 1264.9 | 1673.8 |
| D catalog-only | 3/200 (1.5%) | 1/200 (0.5%) | 0.0019 | 50.0 | 50.0 | 262.1 | 434.8 |

精确二项 95% CI（单臂覆盖率）：A [1.42%, 7.08%]，B [2.08%, 8.37%]，C [2.42%, 9.00%]，
D [0.31%, 4.32%]。来源失败 0，请求失败 0。

### 3.2 配对差（同 case，bootstrap 5000 次）

| 对比 | 覆盖率 Δ [95% CI] | HR@5 Δ | NDCG@5 Δ |
|---|---|---|---|
| B − A | **+0.010 [0.000, 0.025]** | +0.005 [0.000, 0.015] | +0.005 |
| C − A | **+0.015 [0.000, 0.035]** | +0.005 [0.000, 0.015] | +0.005 |
| **C − B** | **+0.005 [−0.010, 0.025]** | 0.000 [0.000, 0.000] | 0.000 |
| D − A | −0.020 [−0.050, 0.005] | +0.005 [0.000, 0.015] | +0.0019 |

三个“对 A 的正差”下界都贴 0；**C − B 跨 0**。按 2B.4 的判据（CI 下界 > 0 才算成立），本轮
**没有任何来源或查询策略达到“可信增量”**。

### 3.3 目标被谁返回（每臂，融合臂按两源 rank）

| 臂 | both | history only | catalog only | neither |
|---|---|---|---|---|
| A | — | 7 | — | 193 |
| B | 1 | 6 | 2 | 191 |
| C | 1 | 6 | 3 | 190 |
| D | — | — | 3 | 197 |

- B 的 catalog 独有命中：case 6607（catalog rank 27）、6707（46）。
- C 的 catalog 独有命中：6607（21）、6650（44）、6664（40）。
- **C 相对 B：找回 6650、6664；丢掉 6707**（净 +1）。
- B/C 都保留了 A 的全部 7 个 history 命中（both 1 + history_only 6 = 7），融合没有丢目标。

### 3.4 候选排名证据（关键）

括号内是“进池但未进最终 5 名”的池内 rank；无括号是最终 rank；`-` 表示未进池。

| case | A | B | C | D | history 源内 rank | B 的 catalog rank | C 的 catalog rank |
|---|---|---|---|---|---|---|---|
| 6513 | (28) | (56) | (52) | - | 28 | - | - |
| 6542 | (7) | (13) | (14) | - | 7 | - | - |
| 6554 | (10) | (19) | (20) | - | 10 | - | - |
| 6588 | (39) | (78) | (69) | - | 39 | - | - |
| 6607 | - | (55) | (43) | (27) | - | 27 | 21 |
| 6612 | (8) | (22) | (16) | - | 8 | - | - |
| 6650 | - | - | (86) | - | - | - | 44 |
| 6664 | - | - | (74) | - | - | - | 40 |
| 6676 | (25) | **1** | **1** | **5** | 25 | 5 | 33 |
| 6701 | (8) | (15) | (19) | - | 8 | - | - |
| 6707 | - | (92) | - | (46) | - | 46 | - |

两条直接结论：

1. **唯一 Top-5 提升是重排**：6676 在 A 已经进池（rank 25）；catalog 的 rank 5（current）/33（balanced）
   与 history rank 25 相加，RRF 把它推到 rank 1。它不需要“扩召回”。
2. **catalog 会系统性地把 history-only 候选推后**：7 个 history 命中有 6 个 rank 变差
   （7→13、8→22、10→19、25→1 例外、28→56、39→78、8→15）。原因是 catalog 每例新增约 47 个候选，
   每个都带 RRF 权重；history-only 候选只有 history 一份贡献。**若某个 history 命中本来在 5 名内，
   加深 catalog 可能把它挤出 Top-5**——本样本 A 最好 rank 是 7，所以没有观测到这种伤害，这是样本
   边界，不是“不会发生”的证据。

### 3.5 均衡查询（C）到底修了什么

| case | current 查询 | balanced 查询 | 结果 |
|---|---|---|---|
| 6650 | `emgreat black carbon fiber ultra light cycling bike` | `emgreat polar black v650 loop carbon cycling activity` | current 未命中；balanced rank 44 |
| 6664 | `supermats heavy duty equipment mat 20gs made for` | `supermats bikeroo h2o4k9 heavy bike neoprene duty seat` | current 未命中；balanced rank 40 |
| 6707 | `yes4all sandbags sand kettlebell heavy duty for fitness` | `yes4all taxidermists spri sandbags woodshop wrist sand ttw` | current rank 46；balanced 未命中 |

机制与设计预期一致：6650/6664 的 current 查询被“最近标题”占满，均衡后带入了更早标题里与目标相关的词
（`cycling`/`bike`/`seat` 等），从而命中；但 6707 说明均衡也会把本来有效的查询稀释掉。200 例中
两条查询 **0 例完全相同**，term 集合平均 Jaccard 0.319。

---

## 4. D — 结果判定

| 判据 | 本轮读数 | 判定 |
|---|---|---|
| C 是否优于 B | 覆盖率 +1 例（+0.5 pp），HR 相同；C−B 95% CI [−0.010, 0.025] | **方向为正，但不可信 → NOT ESTABLISHED** |
| B/C 是否优于 A | B +2 例、C +3 例覆盖；HR 各 +1 例；对 A 的 CI 下界均为 0 | **不成立（未达可信门槛）** |
| catalog 是否有独有候选 | 是：每例约 47 个；其中 2–3 例包含目标，catalog rank 21–46 | 有独有**候选**，但独有**有效命中**很少且在源头就靠后 |
| 覆盖增加但排序不变？ | 新增覆盖的融合 rank 43–92；唯一 Top-5 变化来自重排（6676） | 记录排名证据，不改 RRF |
| 样本量 | 200 例，A/B/C 命中 7/9/10 | **NOT ESTABLISHED**（CI 宽，正例少） |

**综合判定**：

- **“catalog_search 能否给 Agent 带来增量召回” → NOT ESTABLISHED。** 方向为正（+2/+3 个目标），
  但配对不确定性覆盖 0，且新增命中全部在 5 名外。
- **“均衡查询是否比当前查询好” → NOT ESTABLISHED，但机制成立。** 它修复了“最近标题吃满配额”的
  可复现失败模式（2 例），也制造了 1 例新失败；净收益 +1 例。
- **“catalog 完全无用” → 不能这么说。** 它确实提供了 history 没有的候选（47/例）和 2–3 个目标；
  6676 的 Top-5 提升也依赖 catalog 的 rank 5。它的价值目前是**重排**，不是**扩召回**。

---

## 5. E — 下一项生产改动建议

**建议：不实施生产改动。**

| 候选 | 结论 | 理由 |
|---|---|---|
| 把 `catalog_search` 加入 demo 默认 plan | **不做** | 覆盖率增益 CI 跨 0；Top-5 增益是 1 例重排；p50 +22%（B）/ +15%（C） |
| 把均衡查询用于生产查询构造 | **不做** | C−B CI 跨 0；且生产查询是用户消息，不是标题代理，二者语义不同 |
| 继续调 catalog 查询（窗口/配额/停用词） | 仅实验侧 | 只有在更大样本上 C−B 的 CI 下界 > 0 才值得 |
| 改 RRF / 融合权重 | **不做** | 约束禁止；且 6676 说明现有 RRF 已能利用 catalog 排序信号 |

**什么会改变这个决定**（预先写死，避免事后挑指标）：

1. 在 ≥1,000 例、同一 4 臂设计上，C−B 的配对覆盖率差 95% CI 下界 **> 0**；或
2. catalog 独有命中中至少有 2 例的融合 rank ≤ 5（不是靠 history 已有命中重排）；或
3. catalog 命中率本身提升到 ≥2×（当前 3–4/200）。

在此之前，默认 plan 保持 `history`-only；catalog 保持 opt-in。

> 对照 2B.4 预写的门槛 C1（`catalog_search` 覆盖率 > 0 且融合 − history 的配对 95% CI 下界 > 0）：
> 前半条**满足**（D 臂 3/200 = 1.5%，catalog 在 B/C 臂有 2–3 个独有命中）；后半条**不满足**
> （B−A 下界 0.000，C−A 下界 0.000）。所以 C1 整体仍判“未通过”。

---

## 6. 复现与产物

```bash
# 四臂实验（本轮已跑，约 17 分钟）
.venv/bin/python -m experiments.p2b5_query_coverage --cases 200 --out runs/p2b5_query_coverage_200

# 查询构造单测 + 既有相关测试（本轮 50 passed；未跑全量）
.venv/bin/python -m pytest -q \
    tests/test_p2b5_query_coverage.py tests/test_p2b_proxy_query.py \
    tests/test_p2b_metrics.py tests/test_agent_source_limit.py
```

| 文件 | 规模 | SHA-256 |
|---|---|---|
| `runs/p2b5_query_coverage_200/config.json` | 2,439 B | `7f22fd42b367a5e9f62175b5df6141c710b3471c43c49fefc8e36ef2e6c57871` |
| `runs/p2b5_query_coverage_200/cases.json` | 175,878 B | `1976d8ff1626b0b13ceb060408277c96cacd759d95a44aae35891aac69c0f337` |
| `runs/p2b5_query_coverage_200/samples.jsonl` | 1,881,401 B | `c287d30d853da266684b4283acde740796a2d32762a4196964b719efdf47891c` |
| `runs/p2b5_query_coverage_200/summary.json` | 8,186 B | `4ac9c1fbc977534cb4895aa84e563655ade0d2a783260c49470eb61b1045afac` |
| `experiments/p2b5_query_coverage.py` | — | `3f896d7d065f8aa9a222900ce6b5beb849596266189967cf8cbf7780bce61e7a` |
| `tests/test_p2b5_query_coverage.py` | — | `81bd71c5319dc7169ee04a30506d05270598a1bc5c5f1a2581fd1460377d3b47` |

`config.json` 记录：`skip=6500`、`selection={scanned:210, excluded_by_frozen_cohort:10, selected:200}`、
`sequence_sha256=d3d83426…`、`git_head=d9aeed9`、`driver_sha256=3f896d7d…`、两条查询规则的完整参数、
`sample_isolation`（20k 精确排除 / P3 与 R-LR 按位置避开 / R-LR 划分不可验证）。

---

## 7. 限制

1. **样本**：200 例、7–10 个正例；单臂 95% CI 宽约 4–6 个百分点，配对差 CI 跨 0。不能据此判定优劣。
2. **Top-5 的天花板**：A 的 7 个命中最好 rank 7，因此本轮**测不到**“catalog 把 Top-5 命中挤掉”的
   风险；只能从 rank 证据推断它可能发生。
3. **代理查询**：两条查询都只由标题拼成，不代表用户意图；case 6650/6664/6707 的差异是“最近标题是否
   吃满配额”，不是语义理解差异。
4. **history 是 pre_fused**：Serving V1 已经融合 3 个内部源；catalog 与内部 `catalog_lexical` 有重叠，
   所以“catalog 独有候选”不等于“新信息”，只是“未出现在 Serving V1 的 depth-50 头里”。
5. **队列隔离**：20k cohort 精确排除（10 例）；R-LR 只按位置在窗口外，划分成员未落盘，无法逐例证明。
6. **计时**：`/mnt/d` 9p + 单进程热缓存；p50 差异（1097 → 1265/1343 ms）只在本机有意义。

---

## 8. 变更文件

| 文件 | 变更 |
|---|---|
| `experiments/p2b5_query_coverage.py` | 四臂驱动 + `balanced_title_terms`（实验侧）+ 20k cohort 排除 + 产物校验和 |
| `tests/test_p2b5_query_coverage.py` | 13 项查询构造测试（轮转公平性、上限、去重、缺标题、只读 history、目标不可见、确定性） |
| `docs/P2B5_CATALOG_QUERY_COVERAGE.md` | 本文档 |
| `runs/p2b5_query_coverage_200/` | 新产物（gitignore，未覆盖任何旧目录） |

没有修改：生产查询构造、RRF、排序、默认候选深度、冻结 evaluator / 预注册 / 历史报告 / 检查点；
没有跑全量测试或全量评测；没有自动提交 Git。
