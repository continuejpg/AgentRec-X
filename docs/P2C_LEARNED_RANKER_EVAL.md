# P2C — Learned Ranker 独立评估

> 目标：收口 Phase 2B 版本，验证冻结 LightGBM Learned Ranker 是否优于 RRF 基线，并按预先
> 写死的 G1–G4 判定是否值得进入受控部署。
>
> 结论：**G1/G2 失败，G3/G4 通过 → 保留 RRF 默认路径，不启用 Learned Ranker。**
> 之前"LR 优于 RRF"的说法来自一个被反转的 RRF 基线，已用代码追踪与 766 组确定性复现解释清楚。
>
> 不修改生产默认排序、不重训、不动冻结评测资产；本轮 2C 实验代码未提交。

## 0. TL;DR

- **Phase 2B 收口**：4 个按依赖顺序的本地提交（2B.3 → 2B.4 → 2B.5 → 2B.6），工作树干净；
  未 push、未 merge、未提交 `runs/`。
- **RRF 基线根因**：训练脚本把 RRF 基线按 `1/(60 + rrf_score)` 排序。该变换严格递减，而
  `rank_metrics` 按 `score` 降序排 → 基线被**反转**。确定性复现：正确 RRF HR@20 **30/766**，
  训练脚本 RRF **3/766**，冻结 LR **21/766**；三者与历史 JSON 逐项一致。
- **前置验证**：模型/Schema SHA 一致；15 维特征顺序与缺失规则一致；与冻结训练数据逐值
  **max diff = 0.0**（292/292 候选）；cohort 与训练/P3/2B.5/2B.6 均不重叠。
- **独立 2,000 例结果**：RRF 全面优于冻结 LR。
  - NDCG@20：RRF **0.012201** vs LR **0.008267**，配对 Δ **−0.003934**，95% CI
    **[−0.006397, −0.001685]**；
  - HR@5：RRF **0.0130** vs LR **0.0060**，Δ **−0.0070**，95% CI **[−0.0115, −0.0025]**。
- **判定**：G1 ✗、G2 ✗、G3 ✓、G4 ✓ → **不部署、不切默认、不重训**；Learned Ranker V1 的
  质量结论记为 **NOT ESTABLISHED（本 cohort 上为负）**。

---

## 1. A — Phase 2B 版本收口

按依赖顺序提交（均在本分支 `feat/agent-source-limit-ablation`，**未 push**）：

| 阶段 | Commit | 文件 | 变更 |
|---|---|---|---|
| 2B.3 | `a728203` | `experiments/p2b_agent_source_limit_ablation.py`、`tests/test_p2b_proxy_query.py`、`docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md`、`docs/P2B_HANDOFF.md` | +1354 / −107 |
| 2B.4 | `9c2df95` | `experiments/p2b4_three_source_comparison.py`、`experiments/p2b4_source_diagnosis.py`、`docs/P2B4_EVAL_TASK_ALIGNMENT.md` | +1215 |
| 2B.5 | `f215183` | `experiments/p2b5_query_coverage.py`、`tests/test_p2b5_query_coverage.py`、`docs/P2B5_CATALOG_QUERY_COVERAGE.md` | +1196 |
| 2B.6 | `9eabbfe` | `experiments/p2b6_query_scale_validation.py`、`docs/P2B6_QUERY_SCALE_VALIDATION.md`、`docs/LR_INDEPENDENT_EVAL_PLAN.md` | +826 |

- 依赖关系已按顺序满足：2B.4 驱动 import 2B.3 的指标与观测器；2B.5 import 2B.3；2B.6 import 2B.5。
- 每组提交后核对 `git diff --cached --name-only` 与 `git status`；2B 提交完成后工作树干净
  （2C 的实验脚本随后新增、未提交，见 §8）。
- `runs/` 全部由 `.gitignore` 忽略，未入库；无 push / merge / 历史改写。
- **2B.2 / 2B.3 的口径警告仍成立**：其驱动使用 `history = asins[:-2]`（train history），
  不等同冻结协议 `test_history = asins[:-1]`；其指标不可与 2B.4 之后的数字直接比较（2B.4 §3.2）。

---

## 2. B — 前置验证

### 2.1 运行环境

| 项 | 结果 |
|---|---|
| `.venv-ranker`（按部署文档创建） | Python 3.10.12 / LightGBM 4.7.0 / NumPy 1.26.4 ✓ |
| `.venv-ranker` 能否跑完整评估驱动 | **不能**：没有 fastapi 等项目依赖；只适合"加载模型 + 预测" |
| 实际评估环境 | 项目 `.venv`（已有全部依赖，NumPy 1.26.4）+ 安装 `lightgbm==4.7.0` |
| 对默认路径的影响 | 无：`serving_v1` 不 import LightGBM；未改任何仓库文件/依赖清单 |

> 环境决策已记录：评估使用项目 `.venv`；`.venv-ranker` 保留为模型加载环境的验证（本地
> `.git/info/exclude` 排除，不入库）。

### 2.2 冻结模型与特征契约

| 检查 | 值 | 结果 |
|---|---|---|
| `model.txt` SHA-256 | `c1b37d4a2afe9187000c73bc7bb22cc811dda3a2fe9bfe2118fa513db6eb9` | 与部署文档一致 |
| `feature_schema.json` SHA-256 | `ad1d0be168dc9f183c45683c26e24a0086e882632570e2feffa8f9971598a422` | 与部署文档一致 |
| 特征数 / 顺序 | 15 维；与 `learned_ranker.FEATURES` 和 schema 完全一致 | ✓ |
| 缺失值规则 | `mask=0, rank=0, inv_rank=0.0, score=0.0`；`build_feature_rows` 实现一致 | ✓ |
| 模型输入维度 | `booster.num_feature() == 15` | ✓ |
| 推理轮数 | `best_iteration = 22`（`runs/lr_v1_formal/metrics.json`） | ✓ |

### 2.3 特征一致性（G3 的强证据）

用与评估驱动相同的引擎调用重建 **query 0 / user 1001** 这一已物化组，与
`runs/lr_v1_r2/rlr01_groups.jsonl` 逐值比较：

```
pool 292 stored rows 292 same asins True
15/15 features: max |recomputed - stored| = 0.000000000
overall max diff: 0.0
```

复现：`.venv/bin/python -m experiments.r_lr05_feature_parity`（退出码 0 当且仅当 ≤ 5e-7）。

### 2.4 样本独立性（G4）

| 项 | 值 |
|---|---|
| cohort | 2,000 个有效案例，valid case **#9001–#11098** |
| 扫描 / 排除 | 扫描 2,098；按冻结 20k cohort 精确排除 **98**；选取 2,000 |
| 用户唯一性 | 2,000 个不同 `user_int_id`（1 序列 = 1 用户） |
| 位置窗口 | P3 [1,1000]、LR 训练 [1001,6000]、2B.5 [6500,6709]、2B.6 [6710,7791] |
| 是否重叠 | **无**（first_valid_case 9001 > 7791） |
| 与模型训练/选择重叠 | 无（模型训练用 valid #1001–6000；本 cohort 为 #9001+） |

---

## 3. B — RRF 基线差异根因（3/766 vs 30/766）

### 3.1 代码追踪

`experiments/r_lr02b_train.py`（以及 `r_lr02_train.py`）在评估 RRF 基线时这样构造分数：

```python
# r_lr02b_train.py:117,137（r_lr02_train.py:95,99 同）
rrf_va = np.where(rrf[...] > 0, 1.0 / (60.0 + rrf[...]), 0.0)
rrf_te = np.where(rrf[...] > 0, 1.0 / (60.0 + rrf[...]), 0.0)
rrf_test = rank_metrics(y, grp, rrf_te, rrf, item)   # score = rrf_te
```

而 `rank_metrics` 的第一排序键是 `-score`：

```python
# r_lr02b_train.py:72-73
order = np.lexsort((item[a:b], -rrf_main[a:b] if score is not rrf else -rrf[a:b],
                    -score[a:b]))
```

`1/(60 + x)` 是 `x` 的**严格递减**函数。把它当作 `score` 降序排序，等价于按 **RRF 分数升序**
排序 —— 基线被精确反转。正确基线的做法是直接把 `rrf_score` 当作排序分数（`score = rrf`），
次序为 `(-rrf, item_id)`。

### 3.2 确定性复现（同一 766 组、同一标签）

复现脚本：`.venv/bin/python -m experiments.r_lr05_baseline_rootcause`

| 排序器 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@20 |
|---|---|---|---|---|---|
| **正确 RRF**（score = `rrf_score`） | 6/766 | 16/766 | **30/766** | 0.004582 | 0.013627 |
| **训练脚本 RRF**（score = `1/(60+rrf)`） | 0/766 | 0/766 | **3/766** | 0.000000 | 0.000919 |
| **冻结 LR**（22 棵树） | 8/766 | 13/766 | 21/766 | 0.007675 | 0.012390 |

与历史产物逐项对齐：

- `runs/lr_v1_formal/metrics.json`：RRF `HR@20=3/766, NDCG@20=0.000919`；LR
  `HR@5=0.010444 (8/766), HR@10=0.016971 (13/766), HR@20=0.027415 (21/766), NDCG@20=0.01239` ✓
- `runs/lr_v1_formal/corrected_test_comparison.json`：RRF `HR@20=30/766, NDCG@20=0.013636`；
  LR `21/766, 0.01239`；Δ NDCG@20 = −0.001246 [−0.006189, 0.00407] ✓（0.013627 vs 0.013636 的
  9e-6 差异来自物化时 `rrf_score` 存的是 6 位小数，而 A6 用全精度；HR 完全相同）。

### 3.3 结论

- `metrics.json` 中"LR 比 RRF 高 0.0114 NDCG@20"是**基线反转造成的伪差值**；
- 用正确基线，766 组上 RRF 实际略高于 LR（Δ −0.0012，CI 跨 0），无法证明 LR 更优；
- 因此本阶段的正式比较必须使用 §4 的独立 cohort 与显式全精度 RRF 基线。**没有选择对 LR
  更有利的基线**：两个基线都报告，且以代码/确定性复现为准。

---

## 4. C — 固定候选池，比较两个排序器

### 4.1 设计（单变量）

| 项 | 固定值 |
|---|---|
| 任务 | `next_item_prediction`，`history = case.test_history = items[:-1]`，`target = items[-1]` |
| 候选池 | SASRec / Two-Tower / catalogue-lexical 各 top-**100** 的并集；去历史、去重、只保留已知身份 |
| 基线 | **全精度 RRF**：`sum 1/(60+source_rank)` over 完整 fused universe；池序 `(-rrf, item_id)` |
| 实验组 | 冻结 LR（`model.txt` + `feature_schema.json`，22 棵树，`build_feature_rows` + `order_by_ranker`） |
| 唯一变量 | 排序器（RRF → LR）；召回深度、来源、目录、检查点、候选集合均不变 |
| 指标 | 主 **NDCG@20**；次 **HR@5**；其余 HR/NDCG@5/10/20 一并报告 |
| 统计 | 按 case 配对 bootstrap 10,000 次，seed 20261019（预注册） |
| 运行 | 单进程 CPU；p50 929 ms/case（recall + features + predict） |

### 4.2 结果（`runs/lr_v2_independent_eval/`）

n = 2,000；候选池覆盖（目标在池内）**239/2000 = 11.95%**；平均池 273.88；空池 0。

| 指标 | RRF | 冻结 LR | Δ（LR − RRF），95% CI |
|---|---|---|---|
| **NDCG@20**（主） | **0.012201** | 0.008267 | **−0.003934 [−0.006397, −0.001685]** |
| **HR@5**（次） | **0.0130**（26/2000） | 0.0060（12/2000） | **−0.0070 [−0.0115, −0.0025]** |
| HR@10 | 0.0190（38） | 0.0135（27） | −0.0055 [−0.0105, −0.0005] |
| HR@20 | 0.0290（58） | 0.0220（44） | −0.0070 [−0.0125, −0.0020] |
| NDCG@5 | 0.007818 | 0.003623 | −0.004195 [−0.007051, −0.001496] |
| NDCG@10 | 0.009682 | 0.006042 | −0.003640 [−0.006384, −0.001116] |

**所有 6 个指标的配对 CI 都整体为负**，不存在"LR 只在 NDCG@20 差、HR@5 更好"的分裂。

### 4.3 排名证据

- 池内 239 个正例中，LR 把目标排得更好 **107** 例、更差 **125** 例、相同 **7** 例；
- 平均目标排名：RRF **101.44** → LR **111.67**；中位数 76 → 99；
- LR 在 Top-5 命中上从 26 降到 12；这不是阈值噪声，而是整体排序变差。

---

## 5. D — G1–G4 判定与结论

| 门槛 | 定义 | 读数 | 结果 |
|---|---|---|---|
| **G1** | 配对 ΔNDCG@20 95% CI 下界 > 0 | −0.003934，CI 上界 −0.001685 | **✗** |
| **G2** | ΔHR@5 ≥ 0 或 CI 下界 > −0.002 | −0.0070，CI [−0.0115, −0.0025] | **✗** |
| **G3** | 模型/Schema 哈希、15 特征、池身份、零 fallback | 全部一致；feature_failures=0；fallback=0；pool_identity_failures=0；parity max diff 0.0 | **✓** |
| **G4** | cohort 独立（位置 + 冻结 cohort 排除 + 用户唯一） | #9001–11098；排除 98；2,000 用户；窗口无重叠 | **✓** |

**判定：`all_satisfied = false`。**

- **保留 RRF 默认路径**；不启用 `serving_v1_learned_ranker`；不改生产排序策略。
- **不重训、不调参、不扩大无效实验**。冻结 LR V1 在本独立 cohort 上明确更差；其历史"优势"
  是基线反转造成的。
- 质量结论：**NOT ESTABLISHED（本 cohort 上为负）**。工程结论：评估链路与特征契约 **PASS**。

**下一步建议**（不在本轮执行）：

1. 保留本负结果，不再用同一个冻结模型做重复评估；
2. 若未来重启 LR 方向，必须先用正确 RRF 基线**重新训练/选择**，再用新的独立 cohort 验证；
3. 本 cohort（#9001–11098）已用于一次正式比较，不应再用于模型选择；
4. 不因为"模型已经训练"就启用它——本轮数据不支持任何形式的默认切换。

---

## 6. 产物、SHA 与复现

```bash
# 独立评估（本轮已跑，2,000 例，约 32 分钟）
.venv/bin/python -m experiments.r_lr05_independent_eval \
    --cases 2000 --skip 9000 --out runs/lr_v2_independent_eval

# 基线根因复现（766 组，约 2 分钟）
.venv/bin/python -m experiments.r_lr05_baseline_rootcause

# 特征一致性（292 行，约 1 分钟）
.venv/bin/python -m experiments.r_lr05_feature_parity
```

| 文件 | 规模 | SHA-256 |
|---|---|---|
| `runs/lr_v2_independent_eval/config.json` | 2,868 B | `3a765e7a6e21bba9d59390d6d62b9595b78e6548fac364075f46a750abcc3504` |
| `runs/lr_v2_independent_eval/cases.json` | 285,865 B | `c37d4bda1b04985041e3dcf635dacff3b7ea92c26e1fae65bdc70a3bed2fdafa` |
| `runs/lr_v2_independent_eval/samples.jsonl` | 751,219 B | `feea2f55a993ee0d5a0375fe503d4f470ff6543d9dcc8743dd32cde8d156d426` |
| `runs/lr_v2_independent_eval/summary.json` | 3,638 B | `a5f44e3fac5d3915aa48aa5fdabe5f032410b7f0b77b65701708cf3a0ac3b845` |
| `experiments/r_lr05_independent_eval.py` | — | `7804aa1dae59c928a4cf31b9f3d669045e41216b94e0e1d1a4a43d2078767006` |
| `experiments/r_lr05_baseline_rootcause.py` | — | `4e71608f49d948d06d4d415fb4ac2126ae8ac4f21bab3f4878b81188ed9e9771` |
| `experiments/r_lr05_feature_parity.py` | — | `26902c948ee94b39c696292fe020b3d454400fd698fb84da8666b3227de8fd3e` |

`config.json` 记录：cohort 窗口与排除数、模型/Schema SHA、15 特征、`head_depth=100`、
`best_iteration=22`、输入产物哈希（sequences/mappings/SASRec/Two-Tower）、G1–G4 定义、
`git_head=9eabbfe`、`driver_sha256`（与上表一致）。

---

## 7. 限制

1. **单模型、单 cohort**：只评估冻结的 LR V1（22 棵树）；结论不外推到其它 ranker 或重训模型。
2. **池内指标**：两个排序器都只对三源并集排序；池外目标计为 miss，与部署行为一致，但不是
   全目录 HR/NDCG。
3. **cohort 是连续块**（valid #9001–11098，排除 98 个 20k cohort 用户）；不是随机抽样。
   1 case = 1 用户，配对 bootstrap 以用户为单位。
4. **候选召回 11.95%** 与本模型训练 cohort 的 13.06% 接近，但目标分布仍可能有区域差异。
5. **延迟**是单进程 CPU 的端到端（recall + features + predict），不是生产 SLA。
6. **不重训**：本结论只针对冻结模型；不能回答"重新训练一个正确基线下的 ranker 是否会更好"。
7. 2C 的驱动与验证脚本**未提交**（按约束）；2B 的 4 个提交仅在本地分支，未 push。

---

## 8. 变更文件（2C，均未提交）

| 文件 | 作用 |
|---|---|
| `experiments/r_lr05_independent_eval.py` | 2,000 例独立评估驱动；RRF vs 冻结 LR；G1–G4；产物 + SHA |
| `experiments/r_lr05_baseline_rootcause.py` | 3/766 vs 30/766 的确定性复现（766 组） |
| `experiments/r_lr05_feature_parity.py` | 15 维特征与冻结训练数据逐值一致性检查 |
| `docs/P2C_LEARNED_RANKER_EVAL.md` | 本文档 |
| `runs/lr_v2_independent_eval/` | 新产物（gitignore，未覆盖历史目录） |

没有修改：生产默认排序、RRF、查询构造、候选深度、冻结 evaluator / 预注册 / 历史报告 /
模型检查点；没有重训；没有跑全仓测试；没有 push / merge。
