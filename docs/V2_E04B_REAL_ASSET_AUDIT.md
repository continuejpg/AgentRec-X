# AgentRec-X V2 / E-04B-1 — 真实推荐资产盘点、评测协议审核与可执行基线确认

> 状态：**只读审计**。本轮**未**发起任何网络请求、**未**加载任何模型权重、**未**修改生产代码或测试、
> **未**执行任何 Git 写操作。所有判断均以实际文件元数据、`file:line` 与既有工件为证。
> ⚠️ 本文件中的 **READY_LOCAL 仅表示"工件/代码/映射/依赖齐备、具备离线推理条件"**，
> **不代表本轮已经真实运行过推理**；真实运行属 E-04B-2。

---

## 1. 当前 Commit、分支与工作树状态

```text
branch : v2/e04a-pass-checkpoint
commit : b3af9da79a87aada73cf389ee4c22df5f2169d54   （E-04A 检查点，未被修改/重写/覆盖 ✓）
status : 工作树干净（提交后无 modified / 无 untracked）；`.env`、缓存与 `runs/`、`data/` 均被 .gitignore 忽略 ✓
本轮新增：仅本文档（`docs/V2_E04B_REAL_ASSET_AUDIT.md`，未跟踪）✓
```

## 2. 四后端真实资产状态矩阵

| 后端 | 状态 | 判定依据（关键证据） | E-04B-2 可执行性 |
|---|---|---|---|
| **SASRec**（顺序推荐） | **READY_LOCAL** | 引擎 `recommendation/inference/sasrec.py:232` `SASRecInferenceEngine`；默认权重 `recommendation/api/app.py:76` → `runs/sasrec_canonical_2026/best.pt`（**117 MB，存在** ✓）；映射 `app.py:78` → `data/processed/Sports_and_Outdoors_mappings.json`（**38 MB** ✓）；真实训练/评测记录 `runs/sasrec_canonical_2026/run.json`（含 `formal_test_metrics`/`formal_validation_metrics`/`cohort`/`git`/`environment` ✓） | 可（CPU，分批） |
| **Two-Tower**（双塔检索） | **READY_LOCAL** | 引擎 `recommendation/inference/twotower.py:70` `TwoTowerInferenceEngine`；默认目录 `app.py:83` → `runs/twotower_public_2026_full`，其中 `checkpoint.pt`（**155 MB，存在** ✓）+ `metadata_categories.json` + `run.json`（`architecture`/`model_config`/`trainer_config`/`dataset`/`git` ✓）；另有 `runs/twotower_public_2026/checkpoint.pt`（155 MB ✓）与头标定 `runs/b1_evidence/two_tower_head.npz`（52 MB ✓） | 可（CPU） |
| **BM25 / catalog_search**（词法） | **READY_LOCAL** | 实现 `recommendation/control/catalog_search.py:74-107`（`score_kind="bm25"` ✓，BM25 常量与 idf 复用 `recommendation/rag/retrieval.py:35` ✓）；语料 `data/processed/Sports_and_Outdoors_products.jsonl`（**294 MB，156,746 行** ✓）；无权重依赖 ✓ | 可（最便宜，先跑） |
| **Similar-item（M4 附加源）** | **READY_LOCAL** | 索引 `runs/m4_evidence/similar_item_index.pkl`（**392 MB** ✓）+ `heads.npz`（93 MB ✓）+ `shards/catalog_search.*.json` ✓；来源开关 `AGENTRECX_AGENT_SIMILAR_ITEMS`（`agent_service`）✓ | 可选（内存占用最高 ⚠️） |
| **TIGER**（生成式 semantic-id） | **READY_LOCAL（有条件）** | 权重 `runs/tiger_public_2026/checkpoint/tiger.pt`（**44 MB** ✓）+ `tiger.json` + **`generator_layout.json`** + `manifest.json` + `score_rule.json` ✓；特征 `features/item_features.npy`（**470 MB** ✓）+ `item_features.json` ✓；SID `sid/{tokenizer.pt 6.7MB, tokenizer.json, semantic_ids.json 5.6MB, layout.json}` ✓；后端服务 `backends/tiger_public/src/tiger_public/serve_worker.py`（真实进程 + JSONL IPC ✓）；L2 适配 `recommendation/backends/tiger_engine.py:86-96`（`TigerAssetsMissing`/`TigerWorkerError`/`TigerContractViolation` ✓ 明确拒绝语义）；身份映射 `runs/tiger_backend_handoff_prod/catalogue_items.jsonl`（156,746 行 `{backend_row,item_id}` ✓，**不含 `parent_asin`** ✓ 符合边界规则）；文档 `docs/TIGER_SERVING.md:3` 记录 **T-01C 已在真实权重上端到端验证** ✓ | 可，但**最重**：CPU 首次启动约 12 分钟（`TIGER_SERVING.md:113`），须放在最后且需依赖自查 ⚠️ |

**条件与残余风险（TIGER）**：`backends/tiger_public/requirements-ml.txt` 声明 `transformers==5.17.0`、
`sentence-transformers==6.1.0`，而当前 `.venv` **未安装** `transformers/sentencepiece/tokenizers` ✗；
但**后端包的第三方 import 实际只有 `torch` + `numpy`** ✓（全包 grep 结果），且 `torch 2.1.2+cpu`、`numpy 1.26.4`
均已就绪 ✓。因此判定为"有条件 READY"：**E-04B-2 首次运行即需用受控的 gated 真实测试复跑确认**
（`tests/test_tiger_engine_real.py` ✓ 自带 `requires_backend`/`requires_checkpoint` 跳过条件 ✓）；
若运行中暴露缺失依赖，则按"禁止自动安装"规则**停下并报告**，状态改判 `ASSET_INCOMPLETE`（缺 Python 依赖）✓。
另：`backends/tiger_public/.venv/bin/python` 是**指向仓库 `.venv` 的符号链接** ⚠️（并非真正隔离环境），
这与 `requirements-ml.txt` 中"装进 backend 自己的 venv"的建议不一致 —— 记录为**已知偏差** ✓。

## 3. 本地实际存在的权重、索引、数据与映射清单（仅元数据）

| 类别 | 路径 | 大小/行数 | 格式 |
|---|---|---|---|
| SASRec 权重 | `runs/sasrec_canonical_2026/best.pt` | 117 MB | torch checkpoint + `run.json` |
| Two-Tower 权重 | `runs/twotower_public_2026_full/checkpoint.pt`、`runs/twotower_public_2026/checkpoint.pt` | 155 MB ×2 | torch checkpoint + `run.json` |
| Two-Tower 头标定 | `runs/b1_evidence/two_tower_head.npz` | 52 MB | npz |
| Similar-item 索引 | `runs/m4_evidence/similar_item_index.pkl`（+ `heads.npz` 93 MB） | 392 MB | pickle / npz |
| TIGER 权重 | `runs/tiger_public_2026/checkpoint/tiger.pt` | 44 MB | torch + json 布局/评分规则 |
| TIGER 特征 | `runs/tiger_public_2026/features/item_features.npy` | 470 MB | npy（+ `item_features.json`） |
| TIGER SID | `runs/tiger_public_2026/sid/{tokenizer.pt,tokenizer.json,semantic_ids.json,layout.json}` | 6.7 MB / 5.6 MB | torch + json |
| TIGER 交接（后端侧） | `runs/tiger_backend_handoff_prod/{catalogue_items,eval_cohort,train_exposure,products_text}.jsonl` | 156,746 / **20,000** / 412,445 / 156,746 行 | JSONL，**无 `parent_asin`** ✓ |
| 商品目录（AgentRec-X 侧） | `data/processed/Sports_and_Outdoors_products.jsonl` | 294 MB / 156,746 行 | JSONL |
| 交互序列 | `data/processed/Sports_and_Outdoors_sequences.json` | 333 MB，`format="agentrecx.sequences.v1"`，**并行数组按时间升序**（最旧在前）✓ | JSON |
| ID 映射 | `data/processed/Sports_and_Outdoors_mappings.json` | 38 MB；`num_users=412445`、`num_items=156746`、`pad_id=0`、`first_real_id=1` ✓ | JSON |
| 玩具目录（对照基线） | `runs/demo_catalog/mappings.json`（`demo-*` 身份） | 4 KB | JSON —— ⚠️ **仅 demo；不得作为真实质量基线** ✗ |

## 4. 真实数据集与评测划分情况

* 数据集：**Amazon Reviews 2023 / Sports_and_Outdoors**（项目自建预处理产物，`data/processed/*`，mtime 2026-09-14）。
* 规模：**412,445 用户 / 156,746 商品**（映射文件与交接文件三方一致 ✓：`train_exposure` 412,445 行、
  `catalogue_items` 与 `products_text` 各 156,746 行 ✓）。
* 时间信息：序列文件自述"**并行数组按时间升序（oldest first）**" ✓、切分协议为
  `temporal_leave_two_out`（`recommendation/evaluation/split.py:46` ✓）→ **具备时间序切分条件** ✓。
* 切分语义：`split_sequence`（`split.py:202` ✓）；每个用户留出**验证 + 测试各一个正例**，
  `validation_history`/`test_history`（`:92`/`:97`）与 `validation_seen`/`test_seen`（`:107`/`:112`）明确暴露
  "构造历史时可见 / 评估时必须屏蔽"的集合 ✓。
* 候选与屏蔽：`build_catalog`（`split.py:150`）**排除 PAD(0)** ✓；`validate_item_ids`（`:168`）拒绝 0/负数/越界 ✓。
* 后端交接（TIGER）：`eval_cohort.jsonl` 仅含 `{case_id, history, required_frontier}` ✓ —— **不含 test/validation target** ✓，
  这是"评测标签不进后端"的结构性保证 ✓（对应 `AGENTS.md` §19.2 规则 2）。
* 负采样：`recommendation/evaluation/{evaluator,batched}.py` 中**未发现 sampled-negative 路径** ✓；
  协议按 `build_catalog(num_items)` 的全目录打分 + seen 屏蔽 ✓ → **full-catalogue 指标** ✓，
  因此**不得**与任何 sampled-negative 结果直接比较 ✓（报告将始终标注该口径 ✓）。

## 5. 数据泄漏与用户身份风险

| 风险 | 结论 | 证据 |
|---|---|---|
| 用测试集未来交互构造 Memory，再宣称 Memory 提升 | **代码路径不存在** ✓：`recommendation/evaluation/agent/*` 与 `recommendation/demo/*` 中**没有任何** `test_target`/`validation_target`/`required_frontier` 的读取或注入 ✓ | 全仓 grep 结果为空 ✓ |
| 评测期 Memory 来源 | Agent 评测的偏好由**用例声明的约束**构造（`recommendation/evaluation/agent/runner.py:187-195` ✓），历史为"应用自有、绝不来自模型"（`:992` ✓） | ✓ |
| 目标商品标签被用于选源/初始化偏好/Replanning | 结构性不可达：`required_frontier` 只进后端、不进 Agent；Replan 只读受限 Observation（E-03B §3B 字段清单）✓ | E-03B 报告 + `replan_policy.build_replan_request_payload` ✓ |
| 商品身份空间一致性 | 156,746 = 映射 `num_items` = `catalogue_items` 行数 = `products_text` 行数 ✓；PAD=0 隔离 ✓ | 上表 ✓ |
| 跨数据集/跨版本混用 | 未发现（仅 Sports_and_Outdoors 一套 processed 产物 ✓）；`runs/demo_catalog` 为玩具目录，**须显式排除** ✗ | ⚠️ 使用纪律 |
| RAG 目录事实 vs 质量标签隔离 | RAG 只读 `products_text.jsonl`（商品文本）✓ 与 `data/processed/*_products.jsonl` 元数据 ✓；标签（holdout 正例）从不进入 RAG/解释链 ✓ | E-02B 报告 + `product_knowledge.retrieve_product_knowledge` ✓ |
| 用户身份/隐私 | 报告不含任何用户历史或标识原文；映射仅用于**行数/规模核对** ✓ | 本文档 ✓ |

## 6. 已有评测入口与指标实现（可复用，避免重造）

| 能力 | 实现（file:line） |
|---|---|
| 时间序切分协议 | `recommendation/evaluation/split.py:46`（协议名）、`:202`（`split_sequence`）、`:92-115`（历史/seen 集合） |
| 候选与身份校验 | `split.py:150`（`build_catalog`，排除 PAD）、`:168`（`validate_item_ids`） |
| 指标 | `recommendation/evaluation/metrics.py:311`（`hit_at_k`）、`:317`（`ndcg_at_k`）；`:39` 与 `:495`（`compare_hr_recall`）明确**单正例下 HR 与 Recall 数值等价** ✓（报告必须据此措辞） |
| 评测驱动 | `recommendation/evaluation/evaluator.py`、`batched.py`（分批以适配小内存 ✓）、`recommendation/evaluation/agent/`（Agent 轨迹评测：`runner.py`、`attribution.py`、`cases.py`、`metrics.py`） |
| 公开基准入口 | `experiments/benchmark_twotower.py:92-98`（`--cohort/--two-tower-dir/--sasrec-checkpoint/--skip-fusion`）、`experiments/benchmark_genrec.py:104-110`（`--cohort/--genrec-dir/--two-tower-dir/--beams-per-level`） |
| 既有配对统计/证据 | `runs/b1_evidence/{b1_paired_statistics.json,mcnemar.json,B1_EXECUTION_MANIFEST.json}`、`runs/m2-paired-uncertainty.json`、`runs/m3_execution/{run_summary.json,m3_ndcg_statistics.json}`、`runs/genrec_benchmark/comparison.json`（均含 `cohort`/`k_values`/`evidence_sha256` 等，可作为**方法学模板** ✓） |
| Agent 行为录制（可复盘） | `runs/m3_execution/agent_behavior.jsonl`（8.6 MB）、`agent_rankings.jsonl`（1.2 MB）——**回放 ≠ 在线工具成本** ✓（§6 公平性要求） |

**当前环境容量**：24 vCPU；**内存仅 7 GB（可用约 6 GB）** ⚠️；磁盘 `/mnt/d` 剩余 452 GB ✓；
**GPU 不可用**（`torch 2.1.2+cpu`，`cuda.is_available()=False`，NVML 被宿主阻止）✗ → 全部走 CPU ✓。

## 7. 哪些质量比较当前可执行

在**同一 20,000 用例队列的子集**（建议 200–500 例起步）、同一 full-catalogue + seen 屏蔽口径、
同一 top-k（5/10/20）、同一商品身份空间下：

* **A. 单来源**：BM25、SASRec、Two-Tower、TIGER（含 Similar-item 可选）✓ 全部具备 READY_LOCAL 条件 ✓。
* **B. 固定组合 + RRF**：融合实现已存在（`recommendation/inference/fusion.py:226` `FusionInferenceEngine` ✓）；
  RRF 常数与并列规则**不得修改** ✓（E-02A-V 契约）。
* **C. 无 Memory 的融合**：默认路径 ✓。
* **E. 候选不足时的确定性追加来源**：E-03B 的确定性 Gate + 追加逻辑已在离线测试中验证 ✓，
  可在**受控队列**上回放（不含模型决策）✓。
* **F. Agent 轨迹复盘**：使用 `runs/m3_execution/*.jsonl` 录制 ✓（并**同时报告**额外工具成本 ✓）。

## 8. 哪些质量结论当前无法成立

1. **Memory 增益不可量化** ✗：离线队列中**没有真实用户自述偏好数据**；M9 的条目来自用户文本（E-01B）
   与"已展示商品的反馈"（E-03C）✓，二者在离线 cohort 上**不存在** ✓ → 只能验证**偏好排序契约**
   （"同一候选池、不同偏好 ⇒ 顺序变化" ✓，E-02A/E-03C 已离线验证），**不能**声称 Recall/NDCG 提升 ✓。
2. **线上效果与生产稳定性** ✗：本轮（乃至 E-04B 全部）无法触及 —— 无 A/B、无真实流量、无长期运行证据 ✓。
3. **玩具目录结果** ✗：`runs/demo_catalog`（`demo-*` 身份）与任何 toy catalog 结果**不作为真实质量基线** ✓。
4. **跨口径比较** ✗：sampled-negative 结果与 full-catalogue 结果不得并列 ✓；HR 与 Recall 在单正例下等价，
   报告须显式说明 ✓（`metrics.py:39`）。
5. **TIGER 的 CPU 时延/吞吐** ✗：首次启动约 12 分钟（CPU 主导，`TIGER_SERVING.md:113`），
   在 7 GB 内存与无 GPU 条件下不构成生产性能结论 ✓。

## 9. E-04B-2 最小离线执行计划（按依赖与成本排序）

| 步骤 | 内容 | 输入/权重 | 入口 | 资源（估） | 离线 | 最小成功判据 | 主要风险 |
|---|---|---|---|---|---|---|---|
| **0** | 协议与身份自检（不加载任何模型） | mappings + sequences + `catalogue_items.jsonl` | 复用 `split.py`/`build_catalog`/`validate_item_ids` | 秒级 / <1 GB | ✓ | 三方规模一致（412,445 / 156,746）✓、PAD 从不出现 ✓、cohort 200 例可切分 ✓ | 无 |
| **1** | **BM25 单源基线**（最便宜） | `data/processed/*_products.jsonl`（流式读） | `recommendation/control/catalog_search.py` + `evaluation/*` | CPU 分钟级 / 1–2 GB ⚠️ | ✓ | 200 例全目录 Recall@K/NDCG@K/HR@K 产出，去重率与无效 ID=0 ✓ | 294 MB 语料一次性载入内存不足 → **必须分批** ✓ |
| **2** | **SASRec 单源** | `runs/sasrec_canonical_2026/best.pt`（117 MB） | `recommendation/inference/sasrec.py:232` | CPU 分钟级 / ~2 GB | ✓ | 与 `run.json` 记录的 formal 指标**同口径抽样**可复现（允许规模差异） | CPU 推理较慢；须分批 |
| **3** | **Two-Tower 单源** | `runs/twotower_public_2026_full/checkpoint.pt`（155 MB） | `recommendation/inference/twotower.py:70`；对照 `experiments/benchmark_twotower.py` | CPU 分钟级 / ~2 GB | ✓ | 同上 | 头标定文件缺失时须显式报告，不得用近似头替代 ✗ |
| **4** | **RRF 融合（B/C）** | 步骤 1–3 的候选（同队列同 k） | `recommendation/inference/fusion.py:226` | 秒级 | ✓ | 同一身份空间、RRF 参数不变、逐来源 provenance 可核对 ✓ | 候选预算不一致 ⇒ 必须同时报告每源预算 ✓ |
| **5** | **确定性追加来源（E）** | 步骤 4 的短候选场景 | E-03B 确定性 Gate（离线测试已覆盖）+ 回放 | 秒级 | ✓ | 追加后候选合法、k 生效、无未验证商品 ✓ | 不得宣称为"模型自主规划"（该结论仅 E-04A-R1 有 1 次真实证据）✓ |
| **6** | **TIGER 真实服务联调**（最后，最重） | `runs/tiger_public_2026/{checkpoint,features,sid}` + handoff | 先跑 `tests/test_tiger_engine_real.py`（gated ✓），再走 `TigerSourceTool` | CPU **~12 分钟启动** + 470 MB 特征 / ~3–4 GB ⚠️ | ✓ | 真实生成 ID 能经 `catalogue_items.jsonl` 合法映射回 `item_id`；缺失/越界有明确拒绝 ✓ | 依赖缺失（transformers/sentence-transformers 未装 ✗）→ 若触发则**停止报告**，不自动安装 ✓ |
| **7** | **四源融合 + 公平性报告** | 1–6 全部 | 同上 | 分钟级 | ✓ | 同队列/同 k/同屏蔽；报告每源预算与工具成本 ✓ | 内存 7 GB 上限 → 串行执行、逐源释放 ✓ |

**外部获取需求**：上述步骤 **0–7 都不需要下载** ✓（工件已在本地 ✓）。
若 E-04B-2 发现某工件缺失，需要外部获取（项目私有 HF 仓，见 `.env` 的 `HF_REPO_ID/HF_REVISION` ✓ 与
`scripts/fetch_artifacts.py` ✓），**必须先取得你的授权**，本轮**未**执行任何下载 ✓。

## 10. 缺失资产与外部依赖

* **Python 依赖**：`transformers` / `sentencepiece` / `tokenizers` / `sentence-transformers` / `scipy` / `sklearn` **未安装** ✗；
  其中仅 TIGER 后端**声明**需要 `transformers`+`sentence-transformers`（`requirements-ml.txt` ✓），
  但后端源码实际只 import `torch`+`numpy` ✓ → 由 E-04B-2 的真实运行判定是否真正缺失 ✓。
* **GPU/CUDA**：不可用 ✗（CPU-only torch ✓）→ 全部 CPU 执行 ✓。
* **内存**：7 GB 上限 ⚠️ → 必须串行、分批（`evaluation/batched.py` 已提供分批驱动 ✓）。
* **缺失工件**：**未发现**关键工件缺失 ✓（SASRec/Two-Tower/TIGER/SID/特征/交接/目录/映射齐备 ✓）。

## 11. 是否需要用户进一步授权

* **下载**：不需要（本地齐备）✓；如后续发现缺失，需单独授权 ✓（本轮未下载 ✓）。
* **加载大型模型**：E-04B-2 **需要**在本地加载 117 MB / 155 MB / 44 MB 权重 + 470 MB 特征 ✓（纯离线 ✓）→
  属于"最小离线联调"范围，**建议授权**；若你希望进一步限制，可先只跑步骤 0–1（BM25，不加载任何权重）✓。
* **依赖安装**：**未授权、也未执行** ✓（若 TIGER 真实运行确需 `transformers`，将停下报告并另行申请 ✓）。
* **真实 LLM**：**不需要**（E-04B 不调用 DeepSeek 在线 API ✓）。

## 12. 本轮是否触发网络请求、模型加载、生产代码修改或 Git 写操作

| 项 | 结果 |
|---|---|
| 网络请求（含 HF 下载、在线 LLM） | **无** ✓（未运行任何 `from_pretrained`/`snapshot_download`/`hf_hub_download` ✓） |
| 模型加载 / 大模型推理 / 训练 | **无** ✓（仅读取文件元数据、JSON 键名与行数 ✓；`torch` 仅用于查询版本与 `cuda.is_available()` ✓） |
| 生产代码 / 测试修改 | **无** ✓ |
| Git 写操作（add/commit/push/reset/restore/clean/stash/checkout） | **无** ✓（HEAD 仍 `b3af9da`，工作树仅新增本文档 ✓） |
| `.env` / 凭据 | 未读取值、未修改 ✓（仅确认键名存在性 ✓） |
| 大型资产 | 未删除、未移动、未覆盖 ✓ |

## 附：本轮 PASS 判据自评

| 判据 | 结果 |
|---|---|
| 四来源资产状态有据可查 | ✓（§2 矩阵，逐条 `file:line` + 文件元数据） |
| 真实数据集与身份映射边界明确 | ✓（§4：412,445 / 156,746 三方一致；PAD 隔离；后端交接无 `parent_asin`） |
| 数据划分与评测协议具可执行方案 | ✓（§6/§9：`temporal_leave_two_out` + full-catalogue + seen 屏蔽 + 现成入口） |
| 明确下一轮可直接加载什么、缺少什么 | ✓（§9/§10：可直接加载 SASRec/Two-Tower/TIGER/BM25 工件；缺 Python 依赖与 GPU，内存受限） |
| 无未授权在线调用、外部下载、大规模训练 | ✓（§12） |
| 现有检查点不受影响 | ✓（HEAD 与工作树未变） |

**E-04B-1 = PASS（本子任务）**；**E-04B 整项仍未完成** ✗ —— 真实推理与质量跑分属 E-04B-2，
且 Memory 增益与线上效果在本阶段**无法成立**（§8）✓。
