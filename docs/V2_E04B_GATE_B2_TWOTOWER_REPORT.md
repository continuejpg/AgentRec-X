# AgentRec-X V2 / E-04B-2 Gate B-2 — Two-Tower 真实模型加载、推理与同案例质量基线

> 状态：**PASS**。真实 Two-Tower checkpoint 已离线加载并**实际前向推理** ✓；与 Gate B-1 的 SASRec
> **逐案例同序对齐**（100/100，`first_mismatch: null`）✓；标定资产来源已查明并**判定不参与评分** ✓；
> 受限 full-catalogue HR/NDCG 已实测并给出配对对照 ✓；资产哈希加载前后一致 ✓；结果写入全新目录 ✓。
> 本轮**未**加载 SASRec（对照只用 Gate B-1 归档结果 ✓）、**未**加载 TIGER、**未**启动 Worker、
> **未**调用在线 LLM、**未**联网/下载、**未**安装依赖、**未**训练或微调、**未**改模型结构/评分算法、
> **未**改生产逻辑/RRF/Agent/Memory/RAG/Serving、**未**执行任何 Git 写操作 ✓。

---

## 1. 开始时分支、HEAD、工作树

```text
branch : v2/e04a-pass-checkpoint      commit : b3af9da79a87aada73cf389ee4c22df5f2169d54
status : 14 个状态条目 = 1 个已跟踪修改（docs/ARCHITECTURE.md）+ 13 个未跟踪文件
git diff --stat : 2 行（同上的单个文件）     —— 与 Gate B-1 结束时一致 ✓
```

## 2. 权重与标定资产哈希（加载前后一致 ✓）

| 资产 | 大小 | SHA-256（前 24 位，全值见 `gate_b2_summary.json`） |
|---|---|---|
| `runs/twotower_public_2026_full/checkpoint.pt` | 162,221,222 B | `0e56be01f872ed0cc46eca69…` |
| `runs/twotower_public_2026_full/metadata_categories.json` | 2,741,318 B | 见 summary ✓ |
| `runs/b1_evidence/two_tower_head.npz`（**未参与评分** ✗） | 52 MB | 见 summary ✓ |

`hash_check_after.unchanged = true` ✓ —— 资产未被修改 ✓（脚本**只读** ✓）。

## 3. 模型架构与评分路径

| 项 | 值 |
|---|---|
| `run_id` / 架构 | `twotower-public-gru-seed2026` / `two_tower_dssm` |
| `model_config` | embedding 128、hidden 128、**GRU pooling**、`max_seq_len=50`、`num_items=156746`、dropout 0.1、`logq_correction=true` |
| 元数据塔输入 | `main_category` / `store` / `details:Color` / `details:Material`（基数 22/1235/256/256，最低频次 20 ✓） |
| 训练目标 | `in_batch_sampled_softmax` + log-Q 校正（`run.json:objective` ✓） |
| 引擎元数据（实测） | `TwoTowerDSSM`、`num_items=156746`、`max_seq_len=50`、`score_kind="two_tower_cosine"`、`parameter_count=40,551,297`、`device=cpu` |
| 用户历史编码 | `recommendation/datasets/sasrec.py:503 encode_inference_history(history, max_seq_len, num_items)`（**与 SASRec 同一输入约定**：左侧 PAD 填充、取最新 50 项 ✓） |
| 评分路径 | `TwoTowerInferenceEngine.score_catalog`（`recommendation/inference/twotower.py:219`）：**原始全目录余弦分数**，`torch.inference_mode()`，非有限即抛错 ✓ |
| 排名/掩码 | 既有 `evaluation/metrics.py`：`sorted_ranking` + `excluded_seen` + `rank_of_target` ✓（**未**新写指标 ✓） |

## 4. 标定资产的作用与使用决定（Phase 1 重点）

**结论：`runs/b1_evidence/two_tower_head.npz` 不参与本轮评分** ✓。依据（可核对）：

1. 它由 `experiments/b1_fusion_paired_evidence.py:484-511` 写出（`rankings["two_tower"]` → npz ✓），
   是 **B1 配对证据实验**（arms A–D：popularity / sequential / metadata / two_tower）的**排名缓存** ✗，
   不是训练或推理产物；
2. **生产推理路径不引用它** ✗：`grep` 显示仅 `experiments/` 使用 ✓；`recommendation/inference/twotower.py`
   从不加载该文件 ✓（测试 `test_the_calibration_cache_is_not_part_of_inference` 以源码断言锁定 ✓）；
3. 其来源清单 `runs/b1_evidence/B1_EXECUTION_MANIFEST.json` 自述为 **COMPLETE 的实验记录**
   （`classification: {"B_minus_A": "SUPPORTED", "D_minus_C": "SUPPORTED"}` ✓、`preregisration`/`gates` ✓），
   即实验归档 ✗，非推理配置 ✗；
4. 采用"事后排名缓存"会改变排序路径与对照基座 ✗，与"按生产路径评测"的要求冲突 ✓。

因此本轮按**生产路径**评测（原始模型分数 ✓），并在结果中记录 `calibration_decision.used_for_scoring = false` ✓
及其理由 ✓。**未**为提高指标切换评分方式 ✓。

## 5. 真实加载与推理证据

```text
加载耗时 1.57 s；is_ready = true；加载后 RSS 704.5 MB（加载前可用 6,4xx MB ✓）
真实前向：每例 encode_inference_history → score_catalog（全目录 156,746+1 分数，有限性校验通过 ✓）
共执行 125 例次（5 + 20 + 100，嵌套前缀 ✓）：model_failures = 0 ✓，分数全部有限 ✓
```

## 6. 与 SASRec 的逐案例对齐检查（Phase 2）

```text
同一选择规则重建（"artifact 顺序的前 N 个有效 case"，无抽样、无调参 ✓）
b1_cases = 100，rebuilt_cases = 100，same_ids_in_same_order = true，first_mismatch = null ✓
```

对齐方式与边界（不夸大 ✓）：归档的 Gate B-1 结果保存了**逐案例 case_id（用户整数 id，非敏感 ✓）与 rank** ✓，
本轮据此**逐一比对 case_id 与顺序** ✓；history / test_target / seen mask 由**同一冻结规则与同一 `split_sequence`**
在同 artifact 上重新推导 ✓（归档未保存历史原文 ✗，故不能逐字段比对历史，属**可核对范围内的最强对齐** ✓）。
**未**因 Two-Tower 的可用性或命中情况重新挑选用户 ✓。

## 7. 运行时防泄漏证据（Phase 3）

| 检查 | 结果 |
|---|---|
| 用户塔输入 | **仅** `test_history`（`encode_inference_history` ✓）；`target_ids_fed_to_model = 0`（三阶段 ✓） |
| target 的使用 | **仅**冻结掩码 `excluded_seen(test_history, test_target)` ✓ 与最终 `rank_of_target` ✓ |
| 是否用目标文本/未来行为/目标 embedding | **否** ✓ |
| 是否用 test_target 调整分数或选择评分配置/用户 | **否** ✓（评分配置与案例规则均在跑分前固定 ✓） |
| 训练/测试边界 | checkpoint 的 `run.json:dataset` 自述 cohort 412,445、`sampled: false`、catalog 156,746 ✓，与评测同数据划分 ✓；无跨划分资产参与评分 ✓ |

## 8. 5/20/100 例指标（Phase 4/5，实测）

| 阶段 | selected | valid | scored | skipped | failures | invalid | pad | seen | 候选输出 | 命中 @5/@10/@20 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 耗时 | RSS 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Smoke 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 100 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.24 s | 857.9 MB |
| Exploratory 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 400 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.44 s | 865.8 MB |
| **Exploratory 100** | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 2000 | **2/2/3** | **0.02** | **0.02** | **0.03** | **0.016309** | **0.016309** | **0.018663** | 22.35 s | 872.4 MB |

* **嵌套前缀**：5 ⊂ 20 ⊂ 100 ⇒ **不可相加为 125 个独立用户** ✓，各阶段单独成表 ✓。
* 全目录口径（156,746 商品全评分 + 屏蔽掩码 ✓），**非** sampled negatives ✓；分母 = `cases_scored` ✓。
* Smoke 不以命中为成功条件 ✓。

## 9. SASRec / Two-Tower 同案例对照（100 例，配对 ✓）

| 指标 | SASRec（Gate B-1 归档，非重跑 ✓） | Two-Tower（本轮实测） | 差值（TT − SASRec） | 命中数（SASRec / TT，分母 100） |
|---|---|---|---|---|
| HR@5 | 0.010000 | **0.020000** | **+0.010000** | 1 / 2 |
| HR@10 | 0.010000 | **0.020000** | **+0.010000** | 1 / 2 |
| HR@20 | 0.020000 | **0.030000** | **+0.010000** | 2 / 3 |
| NDCG@5 | 0.006309 | **0.016309** | **+0.010000** | — |
| NDCG@10 | 0.006309 | **0.016309** | **+0.010000** | — |
| NDCG@20 | 0.008756 | **0.018663** | **+0.009907** | — |

**配对不确定性（逐案例 rank 可得，故给出真实配对检验 ✓）**：
精确二项 McNemar（双侧，无外部依赖 ✓）—— HR@5/@10：不一致对 1（SASRec-only 0 / TT-only 1），**p = 1.0**；
HR@20：不一致对 3（SASRec-only 1 / TT-only 2），**p = 1.0** ✓。
⇒ **100 例探索性对照下，Two-Tower 与 SASRec 的差异不具统计显著性** ✓；
**不得**据此宣称任一模型总体更优 ✗，也不得外推为全数据集结论 ✗。

## 10. 命中数与分母

```text
SASRec（归档）: hits@5/10/20 = 1 / 1 / 2，分母 100
Two-Tower（本轮）: hits@5/10/20 = 2 / 2 / 3，分母 100
```

## 11. CPU 耗时与峰值 RSS

```text
加载 1.57 s（RSS 704.5 MB）；smoke 5 = 1.24 s；20 例 = 4.44 s；100 例 = 22.35 s
RSS 峰值 872.4 MB；结束时可用内存 5,886.6 MB ✓
batch size = 1，逐例全目录评分，未缓存全量用户分数矩阵 ✓；内存守护（可用 <900 MB 停止）未触发 ✓
CPU-only、单实例、未并行加载其他神经模型 ✓
```

## 12. 模型失败与非法候选

```text
model_failures = 0 ✓；cases_skipped = 0 ✓；invalid_candidates = 0 ✓；pad_candidates = 0 ✓；seen_candidates = 0 ✓（三阶段一致）
```

## 13. 历史正式指标与本轮结果的可比性

* Two-Tower 的 `run.json` **不含** formal test metrics ✗（仅架构/训练/数据描述 ✓）⇒ **无历史质量数字可对照** ✓，
  本轮不与任何历史 TT 数字比较 ✗。
* `runs/b1_evidence/*` 的 B1 配对证据（`B_minus_A: SUPPORTED` 等）来自**不同的臂协议与队列** ✗，
  **不构成**与本轮同协议对照 ✓，故未引用为基线 ✗。
* SASRec 侧的历史 sealed 全量指标已由 Gate B-1 报告单独讨论 ✓（同协议、不同样本 ⇒ 仅统计相容 ✓）。
* 本轮 100 例为 **exploratory** ✓，**不得**表述为全数据集质量水平 ✗。

## 14. 新增文件及测试结果

| 文件 | 说明 |
|---|---|
| `experiments/e04b_gate_b2_twotower.py`（新增） | Phase 1–5：资产哈希、加载、同案例重建与对齐、运行时隔离、全目录评分、指标与资源记录、精确配对检验；结果目录唯一化（不覆盖 ✓） |
| `tests/test_v2_e04b_gate_b2.py`（新增，6 项） | 模型/身份契约、真实前向有限性、排序确定性与标签隔离、标定缓存不参与推理、产物契约、同案例同序对齐 |
| `runs/e04b_gate_b2_twotower/{gate_b2_summary.json, stage_smoke_5.json, stage_exploratory_20.json, stage_exploratory_100.json}`（新增，被忽略目录 ✓） | 原始计数、逐例 case_id 与 rank、配对检验、资源与哈希记录 |
| `docs/V2_E04B_GATE_B2_TWOTOWER_REPORT.md`（新增） | 本报告 |

```text
新增 B-2(6) + B-1(5) + Gate A(6) + E-04B-P(12) + P0(4)  → 33 passed / 0 failed ✓
Two-Tower / ranking / mask / metric 相关既有回归            → 379 passed / 0 failed ✓
复现命令：
  .venv/bin/python -m experiments.e04b_gate_b2_twotower --smoke 5 --exploratory 20,100
  .venv/bin/python -m pytest tests/test_v2_e04b_gate_b2.py -q
```

## 15. 已知遗留失败（单独登记，未擅自修复 ✓）

`tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity` **仍失败** ✗ ——
扫描 `recommendation/backends/` 的标识符是否含规范身份名，命中 `tiger_backend.py` / `tiger_engine.py` /
`__init__.py`（TIGER 轮遗留 ✗，本轮 `git status` 无 `backends/` 改动 ✓）。
**本轮测试并非全部通过** ✗：上述项为**既有、与本轮无关**的失败 ✓，未修复、未放宽断言 ✓。

## 16. 结束 HEAD、git status

```text
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54（未变 ✓）
status : 1 个已跟踪修改（docs/ARCHITECTURE.md）+ 15 个未跟踪文件（原 13 + 本轮 3，runs/ 被忽略 ✓）
         0 删除、0 reset/checkout/restore/clean/stash/commit ✓；.env 未读取值亦未修改 ✓
资产   : checkpoint / categories / head cache 哈希加载前后一致 ✓（只读 ✓）
```

## 17. 最终结论：**PASS**

```text
✓ 真实 checkpoint 成功加载（1.57 s，is_ready，参数 40,551,297）
✓ 真实 Two-Tower 前向推理成功（125 例次，0 失败，分数有限）
✓ 真实商品身份与映射正确（156,746 三方一致；PAD/invalid/seen 候选全 0；无 0-based 偏移）
✓ 目标隔离与掩码符合冻结协议（target_ids_fed_to_model = 0；target 仅用于掩码与指标）
✓ 与 SASRec 的案例逐一对齐（100/100，同序，first_mismatch=null）
✓ 完成受限 full-catalogue HR/NDCG（5/20/100 例，分子分母齐全）
✓ 标定资产来源与使用方式清楚（B1 实验排名缓存；判定不参与评分；生产路径无引用）
✓ 资源与结果可复现（规则/seed/K/掩码/批大小/评分方式/哈希全部记录；目录唯一化）
✓ 未破坏现有工作树与资产（哈希一致；1 modified + 15 untracked）
```

**明确未声称** ✗：任一模型总体更优、Agent 自主规划/Memory/Replanning/多源 RRF/RAG 收益、真实用户满意度、线上效果。
**停止点**：不进入 Gate B-3、不加载 TIGER、不启动 Worker、不联网、不安装依赖、不训练、不调用在线 LLM ✓。
