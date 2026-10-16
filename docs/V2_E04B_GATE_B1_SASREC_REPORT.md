# AgentRec-X V2 / E-04B-2 Gate B-1 — SASRec 真实模型加载、推理与推荐质量基线

> 状态：**PASS**。真实 SASRec checkpoint 在 CPU 上实际加载 ✓、真实前向推理实际执行 ✓、
> 全目录评分与冻结掩码/排序口径一致 ✓、受限 HR/NDCG 已实测 ✓、目标隔离有运行时证据 ✓、
> 结果写入全新目录 ✓、既有检查点与资产未受影响 ✓。
> 本轮**未**加载 Two-Tower、**未**加载 TIGER、**未**启动 TIGER Worker、**未**调用任何在线 LLM、
> **未**联网/下载、**未**安装依赖、**未**训练或微调、**未**修改模型结构或推理算法、
> **未**改 Serving `/v1/recommend`、**未**改 RRF、**未**执行任何 Git 写操作 ✓。

---

## 1. 开始时 HEAD、分支与工作树

```text
branch : v2/e04a-pass-checkpoint      commit : b3af9da79a87aada73cf389ee4c22df5f2169d54
status : 11 个状态条目 = 1 个已跟踪修改（docs/ARCHITECTURE.md，P0-CLOSE 的授权文档修正）
         + 10 个未跟踪文件（E-04B 各轮产物）；git diff --stat = 2 行（同上一个文件）
结束时: 增加本轮 3 个新文件（脚本 / 测试 / 本报告）与 runs/e04b_gate_b1_sasrec/（被忽略目录）✓
```

## 2. 本轮实际授权边界

授权并已执行：只读访问 checkpoint/配置/映射/评测数据 ✓、CPU 本地加载 SASRec ✓、用**现有**推理代码
做小批量真实预测 ✓、复用既有 evaluator/metrics ✓、读取必要真实序列与映射 ✓、新增独立脚本/测试/报告 ✓、
结果写入新目录 ✓。**未**执行任何 §1 未授权项 ✓（见文首清单）。

## 3. checkpoint、配置与模型架构证据

| 项 | 值 |
|---|---|
| 权重路径 | `runs/sasrec_canonical_2026/best.pt`（121,675,470 B） |
| SHA-256（实测） | `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912` |
| 与 `run.json` 记录一致 | **是** ✓（`checkpoints.best.sha256`，加载时以 `expected_checkpoint_sha256` 强校验 ✓） |
| 架构（`engine.model_metadata()`） | `SASRec`、`num_items=156746`、`max_seq_len=50`、`hidden=64`、`blocks=2`、`heads=2`、`dropout=0.2`、`parameter_count=10,135,104`、`model_parameters_frozen=true`、`device=cpu` |
| 与 `run.json:model_config` 一致 | **是** ✓（pre-norm / gelu / padding_idx 0 等一致 ✓） |
| 记录协议（`run.json:cohort`） | `temporal_leave_two_out` / `agentrecx.eval_protocol.v1` / `min_sequence_length=3` ✓ **与冻结协议一致** ✓ |
| 历史正式指标（**仅作对照，未替代实测** ✗） | `formal_test_metrics`：HR@5 0.00830 / HR@10 0.01357 / HR@20 0.02149、NDCG@5 0.00533 / NDCG@10 0.00702 / NDCG@20 0.00901（`test_sealed=true`） |
| 依赖 | 现有环境即可（`torch 2.1.2+cpu`、`numpy 1.26.4`）——**未**安装任何依赖 ✓ |

## 4. 模型加载是否成功

**是** ✓：`SASRecInferenceEngine` 构造成功，`is_ready() == True`，加载耗时 **1.42 s**，
进程 RSS 195.4 MB → **498.6 MB**（加载后）✓。加载即校验 checkpoint 摘要（见 §3 ✓）。

## 5. 推理链路是否真正成功执行

**是** ✓（不是"能 torch.load"✗，而是真的构造模型实例 + 前向 + 输出校验 ✓）：
每例调用现有 `encode_history` → `score_catalog`（**全目录**，`torch.inference_mode()`，
内部对 `isfinite` 校验，非有限即抛 `InferenceError` ✓）→ 既有 `sorted_ranking` 取 Top-K ✓。
三阶段共 125 例（5 + 20 + 100），`model_failures = 0` ✓、输出分数全部有限 ✓。

## 6. 输入历史与测试标签隔离检查（运行时证据）

| 检查 | 结果 |
|---|---|
| 送入模型的内容 | 仅 `case.test_history`（经 `item_id_to_parent_asin` 映射）✓；**未**传入 `test_target` ✗ |
| 运行时证明 | 脚本记录 `target_ids_fed_to_model`，三阶段均为 **0** ✓；测试断言 `set(encoded) ⊆ set(test_history)` ✓ |
| 目标的使用范围 | **仅**两处：冻结掩码 `excluded_seen(test_history, test_target)` ✓ 与最终 `rank_of_target` 命中/排序 ✓ |
| 是否用目标标题/描述/未来信息 | **否** ✓（本轮完全不读商品文本 ✗） |
| 是否事后挑选易命中样本 | **否** ✓：规则在跑分前固定为"artifact 顺序的前 N 个有效 case"（无抽样、无调参 ✓） |
| 是否绕过冻结协议另起定义 | **否** ✓：切分/掩码/排序/指标全部复用 `evaluation/split.py`、`evaluation/metrics.py` ✓ |

## 7. 商品 ID、PAD、seen mask 合法性检查

```text
真实商品数      : mappings.num_items = 156746 = item2id 条目数 = 数据文件 num_items = 引擎 num_items ✓
PAD             : padding.pad_id = 0，first_real_id = 1 ✓（两表一致，无 0-based/1-based 偏移 ✓）
Top-K 违规      : pad_candidates = 0、invalid_candidates = 0、seen_candidates = 0（三阶段全部为 0 ✓）
可映射真实身份   : 所有送入与返回的 id 均经 engine.item_id_to_parent_asin 合法映射 ✓
掩码口径        : excluded_seen(test_history, test_target)（重复交互目标仍可评分 ✓，Gate A 已审计 ✓）
```

## 8. Smoke 测试样本与结果（5 例）

```text
cases_selected=5  cases_valid=5  cases_scored=5  cases_skipped=0  model_failures=0
输出候选=100（5×20）  invalid=0  pad=0  seen=0  耗时=0.65 s  RSS 峰值=636.1 MB
命中：HR@5/10/20 = 0 / 0 / 0（5 例探索性样本，非质量结论 ✗）
⇒ Smoke 只证明**真实推理链路可执行** ✓（不等同于"命中目标" ✗）
```

## 9.–11. 各阶段样本数与指标（实测，非复制历史 ✗）

| 阶段 | selected | valid | scored | skipped | 失败 | 候选输出 | 命中 @5/@10/@20 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Smoke（5） | 5 | 5 | 5 | 0 | 0 | 100 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Exploratory（20） | 20 | 20 | 20 | 0 | 0 | 400 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Exploratory（100）** | 100 | 100 | 100 | 0 | 0 | 2000 | **1/1/2** | **0.01** | **0.01** | **0.02** | **0.006309** | **0.006309** | **0.008756** |

* 指标分母均为 `cases_scored`（5 / 20 / 100 ✓），命中数与分母同时记录 ✓。
* 100 例中的两次命中分别位于最终合法排序的 **rank 2** 与 **rank 16**（由 NDCG 反推并与 HR 计数一致 ✓）：
  NDCG@5 = 1/log2(3)/100 = 0.006309 ✓、NDCG@20 = (0.6309 + 1/log2(17))/100 = 0.008756 ✓。
* 口径：**full-catalogue**（156,746 商品全评分 + 屏蔽 mask ✓），**非** sampled negatives ✓；
  单正例下 HR ≡ Recall（`metrics.py:311` 与 `:39` ✓）。

## 12. CPU 运行时间与内存峰值

```text
加载：1.42 s；RSS 195.4 → 498.6 MB
smoke 5    ：0.65 s，RSS 峰值 636.1 MB
expl 20    ：1.88 s，RSS 峰值 636.1 MB
expl 100   ：9.40 s，RSS 峰值 636.1 MB（可用内存结束时 6,197 MB ✓）
batch size ：1（每例一次前向 + 一次全目录评分；未缓存全量用户分数 ✓）
内存守护   ：可用 <900 MB 即中止（本轮未触发 ✓）；CPU-only、未并行加载其他模型 ✓
```

## 13. 失败案例与资源限制

* **模型执行失败：0** ✓；**跳过：0** ✓；**非法/PAD/已见候选：0** ✓（三阶段一致 ✓）。
* 资源限制：CPU-only（无 CUDA ✓）、内存 7 GB 上限 ⇒ batch size 1、逐步扩样 ✓；
  未做更大规模（>100 例）评测：**属本轮范围限制** ✗，非失败 ✓。
* 与既有测试的一个**既有**失败（与本轮无关 ✗）：`tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity`
  失败 ✗ —— 该测试扫描 `recommendation/backends/` 中的标识符是否含规范身份名，命中
  `recommendation/backends/{tiger_backend,tiger_engine,__init__}.py`（**均非本轮改动**：本轮 `git status` 无 `backends/` 条目 ✓）。
  这是 TIGER 后端轮的遗留边界问题 ✗，需在后续授权轮处理，本轮**未**修改这些文件 ✓（详见 §16）。

## 14. 与历史 `formal_test_metrics` 的差异及可比性

| 指标 | 历史（sealed，412,445 例全量） | 本轮（exploratory，100 例） | 可比性 |
|---|---|---|---|
| HR@5 | 0.00830 | 0.01000 | 同量级 ✓ |
| HR@10 | 0.01357 | 0.01000 | 同量级 ✓ |
| HR@20 | 0.02149 | 0.02000 | 同量级 ✓ |
| NDCG@5 | 0.00533 | 0.00631 | 同量级 ✓ |
| NDCG@10 | 0.00702 | 0.00631 | 同量级 ✓ |
| NDCG@20 | 0.00901 | 0.00876 | 同量级 ✓ |

**可比性说明**：协议、身份空间、掩码、排序与 K 一致 ✓，但**样本不同**（100 vs 412,445 ✓）。
在 HR@20 ≈ 2% 的量级下，100 例的二项标准误约 1.4 个百分点 ⇒ 本轮数值与历史值**统计上相容** ✓，
但**不构成对 sealed 结果的复现** ✗，**不得**据此宣称全数据集质量水平 ✗。
本轮也**未**改动案例选择规则以求提高指标 ✓。

## 15. 新增文件与测试结果

| 文件 | 说明 |
|---|---|
| `experiments/e04b_gate_b1_sasrec.py`（新增） | Phase 1–4：资产校验、真实加载、运行时隔离检查、全目录评分、指标与资源记录；结果目录唯一化（不覆盖 ✓） |
| `tests/test_v2_e04b_gate_b1.py`（新增，5 项） | 摘要一致性、身份空间、真实前向有限性、排序确定性与掩码/标签隔离、阶段产物契约 |
| `runs/e04b_gate_b1_sasrec/{gate_b1_summary.json, stage_smoke_5.json, stage_exploratory_20.json, stage_exploratory_100.json}`（新增，被忽略目录 ✓） | 原始计数、逐例 case_id（用户整数 id，无原始身份 ✗）、资源与命令记录 |
| `docs/V2_E04B_GATE_B1_SASREC_REPORT.md`（新增） | 本报告 |

```text
新增脚本/测试 + Gate A 协议 + E-04B-P 协议 + P0  → 27 passed / 0 failed ✓
SASRec 及相关既有回归（-k "sasrec or inference or mask or metric or identity"）
                                              → 356 passed, 23 skipped, 1 failed（§13 的既有失败）✓
复现命令：
  .venv/bin/python -m experiments.e04b_gate_b1_sasrec --smoke 5 --exploratory 20,100
  .venv/bin/python -m pytest tests/test_v2_e04b_gate_b1.py -q
```

## 16. 结束 HEAD、git status

```text
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54（v2/e04a-pass-checkpoint，未变 ✓）
status : 1 个已跟踪修改（docs/ARCHITECTURE.md，P0-CLOSE 授权项 ✓）
         + 13 个未跟踪文件（原 10 个 + 本轮 3 个）；0 删除、0 reset/stash/commit ✓
资产   : runs/sasrec_canonical_2026/*、data/processed/* 未被修改 ✓（只读 + 校验摘要 ✓）
未处理项：§13 的 TIGER 边界测试失败（既有 ✗，非本轮引入 ✓，未擅自修改 ✓）
```

## 17. 最终结论：**PASS**

```text
✓ checkpoint 与实际模型结构兼容（metadata 与 run.json 一致，摘要强校验通过）
✓ SASRec 真实加载（1.42 s，is_ready，参数冻结）
✓ 真实历史驱动真实前向推理（125 例，0 失败，全目录评分，分数有限）
✓ 返回合法真实商品（pad/invalid/seen 候选全为 0，身份可映射）
✓ 目标隔离与掩码口径符合冻结协议（运行时 target_ids_fed_to_model = 0）
✓ 完成受限真实 HR/NDCG（5/20/100 例分级，分子分母齐全）
✓ 结果可复现（规则/seed/K/掩码/批大小/评分方式全部记录；目录唯一化）
✓ 资源、样本与指标清楚；工作树与资产受保护
```

**明确未声称** ✗：Agent 自主规划收益、Memory 个性化收益、Replanning 收益、多源 RRF 收益、RAG 推荐收益、
真实用户满意度、线上效果 —— 均属后续已冻结的对应实验 ✓。**本轮只评价真实 SASRec 单模型的下一物品能力** ✓。

**停止点**：不进入 Gate B-2、不加载 Two-Tower、不加载 TIGER、不调用在线 LLM、不训练、不联网、不安装依赖 ✓。
