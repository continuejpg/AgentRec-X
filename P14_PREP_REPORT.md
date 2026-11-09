# P14-0 — Repository Cleanup & Architecture Preparation 报告

> 性质：**有边界的工程整理**。本轮不接入 Shopping Companion、不构建新索引、不训练/微调模型、
> 不改推荐或 Agent 决策行为、不运行冻结 Holdout / P13-R reserved test、不 commit/push/deploy。
>
> 依据：本报告所有数字来自本轮真实命令输出；清理前状态由两份快照（`before_inventory.json` /
> `after_inventory.json`，path+size+sha256）与 `git status --porcelain -uall` 支撑；归档与删除
> 均有逐文件 SHA256 记录（`P14_PREP_ARCHIVE_MANIFEST.json`、`P14_PREP_SOURCE_DIFF.txt`）。

## 0. 结论

**P14-0 = PASS（工程整理维度）。质量/效果结论：NOT APPLICABLE（本轮不评测效果，也不改变行为）。**

| 验收项 | 结果 |
| --- | --- |
| D1 代码清洁度（保留/合并/归档/删除 + 前后数量体积） | **PASS**：删除缓存/临时目标 **742 个（6,793 文件，203,544,979 B ≈ 194.1 MiB）**= 首轮 706 目标/6,087 文件/190,090,044 B + 测试后二次清理 36 目标/706 文件/13,454,935 B；归档 60 个历史文件（43 个 git 跟踪实验脚本 + 17 个根目录临时文件，全部 SHA256 可恢复）；`recommendation/` 运行代码 0 删除 0 修改；experiments `.py` 146 → 103（根目录 124 → 81） |
| D2 旧功能回归 | **LIMITED（工程结论：未发现清理引入的回归）**：WSL Linux `.venv`（Python 3.10.12，CPU）分批运行 186 个测试文件；chunk1–4 完成 **3,129 passed / 4 failed / 29 skipped**，4 个失败逐项定位在**本轮未触碰**的既有代码/测试缺口；chunk5（37 个文件）按收敛指令在约 29% 时安全终止，其中 P14-0 必需领域的文件已由 451-passed 定向批次覆盖。另：compileall PASS、导入 smoke PASS、契约测试 26 passed。完整“清理前”同套件基线因 WSL OOM 未能取得（见 §6.3） |
| D3 归档/删除验证 | **PASS**：6 组 tar.gz 全部 60/60 文件解包 SHA256 逐字节一致；manifest 记录 original_path/sha256/archive_path/reason/restore；43 个跟踪文件另有 git HEAD 可恢复 |
| D4 新架构准备 | **PASS**：`docs/P14_PREP_INTERFACE.md`（三边界、复用表、四契约、namespace 规则、GAP G1–G10、P14 接入建议）+ `recommendation/contracts/namespaces.py` 契约种子 + 26 个回归测试 |
| D5 不污染技术路线 | **PASS**：Serving V1 / SASRec / Two-Tower / Planner 契约 / M5 / CandidateLedger / Role Gate / Eligibility / Source Policy 代码未改；唯一 tracked 内容修改为 `.gitignore`（+4 行环境目录忽略规则）；`data/`、`runs/`、`models/` 无文件内容变化 |

**一句话**：仓库从“8 阶段历史脚本 + 根目录产物堆积 + 约 181 MiB 可重建缓存”整理为
“共享 Agent 核心 + 两类保留研究入口 + 可恢复历史归档 + P14 契约种子”，旧能力与负结果全部保留。

**最终判定：P14-0 = LIMITED** —— 工程整理（D1/D3/D4/D5）为 PASS，回归验证（D2）为 LIMITED，
原因是 WSL 7.7 GB 内存限制下无法完成单进程全量回归、分批回归未跑完最后一个 chunk（少量历史测试未完成），
且未能取得逐项可比的“清理前”基线。**未发现任何由本轮清理引入的关键回归**（4 个失败均位于本轮
0 改动、且不在归档依赖链上的模块）。按“不追求 100% 历史测试通过、如实交付”的收敛要求，不再重跑。

---

## 1. 清理前的真实仓库问题（C1 盘点）

盘点方法：`git status --porcelain -uall`、按目录统计、静态 import/AST 依赖图
（578 个 Python 模块、8,474 条 import 边、942 个文本文件引用扫描）、动态引用文本扫描
（`importlib` / `-m module` / 路径字符串）。

| # | 问题 | 证据 |
| --- | --- | --- |
| 1 | 工作区长期未提交：HEAD `4dec6d2`（2026-10-25），P5–P13-R 全部工作只存在于工作区 | 28 个已跟踪文件被修改；`git status --porcelain -uall` 共 482 行（28 个 ` M ` + 454 个 `??` 未跟踪文件）；无 stash；HEAD 树 592 个文件（414 个 `.py`） |
| 2 | 根目录堆积过程产物：61 个 `.md`、36 个 `.txt`、14 个 `.csv`、12 个 `.jsonl`、5 个 `.json`、15 个 `.log`；其中 15 个日志与 2 个一次性调试脚本无任何引用 | 引用扫描：`p7*.log`、`p8a_run.log`、`inspect_p6_case.py`、`verify_p8d.py` 在 docs/reports/tests 中 0 命中 |
| 3 | 28 个 `pip-unpack-*`/`pip-install-*` 临时目录散落根目录（45,059,519 B） | 全部为 pip 构建临时目录，无引用 |
| 4 | 675 处 `__pycache__`、6,041 个 `.pyc`（100,979,301 B），其中 `runs/p8_bge/py_deps` 依赖 overlay 内 639 处 68.97 MiB、`tests/__pycache__` 20.93 MiB | 生成物，可重建 |
| 5 | `.venv-bge/` 是半成品 Windows venv：缺 `pyvenv.cfg`、仅 numpy `.pyd` + `python.exe`（15 文件 43,710,976 B），全仓库 0 引用 | 不可用环境残留 |
| 6 | 根目录 `run.json` 是 CWD 误写产物，与 `runs/p10/bge_sasrec/run.json` 字节相同 | SHA256 相同（`5070AFCC…`） |
| 7 | `experiments/` 混有 5 个历史阶段：根目录 124 个 `.py`，共 146 个 `.py`；其中 43 个为已完成阶段的独立 runner，无 test/current-code 依赖 | AST 闭包：43 个候选被测试/保留代码导入数 = 0 |
| 8 | 存在“看起来重复、语义不同”的实现：BM25 两套（`control/catalog_search.py` vs `inference/lexical.py`）、多套指标实现（`recommendation/evaluation/` vs `p6_benchmark/metrics.py` vs `p13/retrieval_metrics.py` vs `benchmarks/agent/`）；RRF 实际只有一份实现 | import 图 + 源码核对；见 §4 |
| 9 | 大体积外部产物：`runs/` 4.17 GB、`models/` 1.25 GB、`backends/` 1.18 GB（含 backend venv）、`.venv/` 1.02 GB、`data/` 0.78 GB — 全部 git-ignored，不属于源码清理范围 | 目录统计；本轮未移动/压缩/删除 |
| 10 | P13-R 检索输入 `data/p13r/electronics/catalog.jsonl` 与 236 MB BGE 向量索引不在本地（`runs/p13r` 只有结果） | 文件检查；记录为 P14 前置条件，本轮不处理 |
| 11 | 旧 docs 只描述到 10 月下旬（V2/Learned Ranker），P5–P13-R 只有根目录报告 + `docs/P13R_CONTEXT_HANDOFF.md` | docs 时间戳/索引检查；本轮不改旧 docs |

**结论**：问题集中在“历史入口与缓存堆积”，不在 `recommendation/` 运行代码（其所有非 `__init__`
模块都有 inbound import，无孤儿模块）。

---

## 2. 四类分类结果（C2）

### 2.1 第一类：核心运行代码 —— 全部保留，零行为修改

范围与判据：`recommendation/` 当前 **186 个 `.py`**（HEAD 跟踪 143 个；43 个为本轮前已存在的未跟踪 P5–P13-R 模块，2 个为本轮新增的 `contracts/`），经静态依赖分析确认 **没有零引用模块**；所有非 `__init__` 模块要么被运行路径引用，要么被测试覆盖。

| 能力 | 保留实现（代表） |
| --- | --- |
| Agent runtime / Planner | `recommendation/demo/agent_service.py`、`demo/planner_contract.py`、`demo/planner_service_executor.py`、`demo/planner_http_proposer.py`、`control/loop.py` |
| M5 事件记忆 | `recommendation/state/`（`decision_state.py`、`event_store.py`、`belief_updater.py`、`state_projector.py`、`state_manager.py`、`shopping_adapter.py`、`legacy_adapter.py`） |
| M9 偏好记忆 | `recommendation/memory/`（`service.py`、`store.py`、`schemas.py`、`query_adapter.py`） |
| 候选与证据 | `control/candidate_plane.py`、`control/candidate_ledger.py`、`control/grounding.py`、`control/evidence.py`、`control/evidence_executor.py` |
| 约束 / 角色 / Eligibility | `control/constraint_eligibility.py`、`control/grounded_reasoning.py`、`demo/product_role.py` |
| 工具契约 | `recommendation/tools/`（`recommendation.py`、`schemas.py`） |
| Serving V1 | `inference/{sasrec,twotower,lexical,fusion}.py`、`api/app.py`、`api/schemas.py` |
| 搜索 | `control/catalog_search.py`（Agent 候选源）、`inference/lexical.py`（Serving V1 词法源）、`user-facing catalog/metadata.py` |
| 评估 | `recommendation/evaluation/`（`split.py`、`metrics.py`、`evaluator.py`、`batched.py`、`experiment.py`、`decision.py`） |
| 传统模型（第二类实现，代码在 `recommendation/{models,datasets,training,baselines}/`） | SASRec、Two-Tower、ItemCF、LightGCN、BGE-SASRec、Semantic-ID v0 |

**本轮对第一类做过的修改：0 个现有文件内容修改、0 个删除。**
唯一新增：`recommendation/contracts/`（namespace 契约种子，见 §5），不改变任何现有调用路径。

### 2.2 第二类：有研究价值的传统算法与历史证据 —— 保留

保留判据：可复现、结构清楚、或承载冻结结论（含负结果），即使已退出默认调用链。

| 保留内容 | 位置 |
| --- | --- |
| SASRec 规范训练/评测入口 | `experiments/sasrec_canonical.py`、`sasrec_formal_test.py`、`sasrec_smoke.py`、`sasrec_tiny_overfit.py`、`sasrec_inference_smoke.py`；`runs/sasrec_canonical_2026/`（116 MB checkpoint） |
| Two-Tower | `experiments/train_twotower.py`、`benchmark_twotower.py`、`benchmark_public.py`、`phase5_benchmark.py`、`fusion_controls.py`；`runs/twotower_public_2026*/` |
| ItemCF / LightGCN / BGE-SASRec | `recommendation/baselines/itemcf.py`、`itemcf_csr.py`、`experiments/itemcf_smoke.py`、`p10_train_lightgcn.py`、`p10_train_bge_sasrec.py`、`p10_eval_four.py`、`runs/p10/` |
| Semantic-ID v0（冻结） | `recommendation/semantic_id/`、`experiments/train_semantic_id.py`、`benchmark_genrec.py`、`runs/semantic_id_public_2026/` |
| TIGER 外部后端边界（PARTIAL） | `backends/tiger_public/`（29 tracked 文件）、`recommendation/backends/tiger_backend.py`、`inference/tiger_engine.py`、`control/tiger_source.py`、`experiments/step27_tiger_engines.py`、`materialize_tiger_backend.py`、`tests/test_tiger_*.py` |
| Learned Ranker 服务与证据 | `recommendation/ranking/`（`inference/learned_ranker.py` 在线路径）、`runs/lr_v1_formal/`、`runs/lr_v1_r2/`、`docs/LR_INDEPENDENT_EVAL_PLAN.md` |
| P11/P12 任务与证据 | `data/p11_benchmark/`、`data/p12/`、`runs/p12/`、`P11_REPORT.md`、`P12_REPORT.md`、`P11_BENCHMARK_SCHEMA.md` |
| P13 / P13-R 数据与负结果 | `data/p13/`、`data/p13r/`、`runs/p13/`、`runs/p13r/`、`P13_REPORT.md`、`P13R_REPORT.md`、`docs/P13R_CONTEXT_HANDOFF.md` |
| P6/P8/P9 研究评测入口 | `experiments/p6_benchmark/`、`p6b/p6c/p6d/p6e_*.py`、`p8c_policy_eval.py`、`p8d_*.py`、`p9_*.py`、`runs/p6_*`、`runs/p8_*`、`runs/p9/` |
| 模型权重 / 训练 adapter（唯一资产） | `runs/` 内 checkpoint、`training/agent_sft/output/*.tar.gz`、`models/bge-base-en-v1.5/`、`backends/tiger_public/.venv`（环境）、`.venv`、`.venv-ranker` |

### 2.3 第三类：历史实验代码 —— 压缩归档（可恢复）

归档判据（任一满足即**保留**，否则归档）：
1. 被测试静态 import（37 个 experiments 模块，闭包后共 103 个保留）；
2. 被 `README.md` / `docs/USAGE.md` / `docs/ARCHITECTURE.md` / `docs/EXPERIMENTS.md` 等**当前文档**
   列为可运行入口（M7–M11 smoke、SASRec、demo_scenario、p3/p4 评测等）；
3. CI（`.github/workflows/verify.yml`）直接使用（`build_demo_catalog`、`demo_scenario`）；
4. 第二类/当前研究主线（P5–P13-R）入口。

归档结果（6 组，共 60 个文件；43 个 git 跟踪 + 17 个未跟踪根目录临时文件）：

| 分组 | 阶段 | 文件数 | 归档大小 | 归档 sha256（前 12 位） |
| --- | --- | ---: | ---: | --- |
| `v2_pipeline_gates` | S2/S4/S6：V2 E-04 gates、Gate C、TIGER step24/25、t03lite | 14 | 62,764 B | `14ee1b1387a9` |
| `tiger_canonical_step26_28` | S4：TIGER canonical remediation Step 2.6/2.7/2.8 | 10 | 40,076 B | `10750556452d` |
| `learned_ranker_v1` | S7/R-LR：LightGBM LambdaRank V1 训练/物化/对齐脚本 | 6 | 17,249 B | `462cf0826ffa` |
| `p2b_agent_diagnostics` | P2B/P2C：source-limit 消融诊断、审计、延迟探针 | 6 | 19,699 B | `3d941cf84b13` |
| `m4_frozen_materialization` | S5 M4：冻结执行/物化工具 | 7 | 24,709 B | `ce35283f8f19` |
| `root_scratch_p7p8` | P7/P8：根目录运行日志 + 2 个一次性调试脚本 | 17 | 3,849 B | `8e2f94237fd6` |
| **合计** | | **60** | **168,346 B** | 逐文件 SHA256 见 manifest |

- 归档位置：`archive/legacy_experiments/<group>.tar.gz`（tar 成员名 = 仓库相对路径，解包即原位恢复）；
- 说明文档：`archive/legacy_experiments/README.md`；
- 清单：`P14_PREP_ARCHIVE_MANIFEST.json`（original_path / sha256 / size / archive_path / reason / restore）；
- 独立复核（Lead 亲验）：6/6 组 tar listing 与 manifest 完全一致；60/60 文件解包后 SHA256+size 一致。

**未归档、仍留在原地的历史代码及原因**（避免“为压缩而压缩”）：
`experiments/e04b2_gate_a.py`、`e04b_p0_rule_baseline.py`、`e04b_protocol_v2.py`、`m3_agent_*.py`、
`m4_{benchmark,conformance,frozen_sources,harness,policies,query_rules,schema_projection,sharding}.py`、
`p2b_agent_source_limit_ablation.py`、`p2b5_query_coverage.py`、`step27_tiger_engines.py`、
`materialize_tiger_backend.py`、`llm_policy_smoke.py`、`live_agent_evaluation.py`、
`_harness.py`、`_metrics.py` —— 全部被现有测试静态 import，删除/归档会破坏回归套件。

### 2.4 第四类：确认无价值的垃圾代码/产物 —— 删除

删除清单（706 个目标、6,087 个文件、190,090,044 B，逐目标记录在 `P14_PREP_SOURCE_DIFF.txt` §5）：

| 类别 | 目标数 | 文件数 | 字节数 | 依据 |
| --- | ---: | ---: | ---: | --- |
| `__pycache__` 字节码树（含 runs 依赖 overlay 内 639 处） | 675 | 6,041 | 100,979,301 | 生成物，导入时重建；非唯一资产 |
| `pip-unpack-*` / `pip-install-*` 等 pip 临时目录 | 28 | 24 | 45,059,519 | pip 构建残留，无引用 |
| `.venv-bge/` 半成品 venv | 1 | 15 | 43,710,976 | 缺 `pyvenv.cfg`，仅有 numpy 二进制；0 引用；Linux `.venv` 才是运行环境 |
| `.pytest_cache/` | 1 | 6 | 336,531 | pytest 缓存 |
| 根目录 `run.json`（CWD 误写） | 1 | 1 | 3,717 | 与 `runs/p10/bge_sasrec/run.json` SHA256 相同 |
| 裸 `.pyc`（`__pycache__` 之外） | 0 | 0 | 0 | — |

**明确不删除、不动的内容**：模型权重/checkpoint（`runs/`、`models/`、`training/agent_sft/output/*.tar.gz`）、
原始与处理后数据（`data/`）、冻结 benchmark（`data/p6_benchmark_v0/`）、P11/P12 标注与评论证据
（`data/p11_benchmark/`、`data/p12/`、`runs/p12/`）、P13/P13-R 报告与数据、`runs/` 全部实验结果、
`backends/tiger_public/`（含其独立 venv，环境可重建但不删）、`.venv` / `.venv-ranker`、
用户尚未提交的 28 个修改文件。

**已知冗余但保留**：`models/bge-base-en-v1.5/` 同时存在 `model.safetensors`、`pytorch_model.bin`、
`onnx/model.onnx` 三种同源权重（约 1.25 GB）。按“不得假设可重新下载”的数据安全规则，本轮**不删除**
任何模型权重格式，只记录（P14+ 可在确认加载路径后单独决策）。

---

## 3. 清理前后的数量与体积（D1）

### 3.1 源码与工作区

| 对比项 | 清理前 | 清理后 | 变化 |
| --- | --- | --- | --- |
| git 跟踪源码（HEAD） | 592 文件 / 414 `.py` | 592 / 414（HEAD 未变；工作区实存 549） | 43 个归档文件从工作区移除，未提交、HEAD 不变 |
| 工作区非忽略源码（快照口径：path+size+sha256） | **1,021 文件 / 391,691,558 B** | **959 文件 / 347,678,431 B** | **-62 文件 / -44,013,127 B（-41.97 MiB）** = 60 归档 + `.venv-bge` 15 文件 41.7 MB + `run.json`；新增文件抵消少量 |
| 生成缓存/临时产物（单独盘点） | 706 目标 / 6,087 文件 / 190,090,044 B | 0（清理后测试又生成 36 目标/706 文件/13,454,935 B，已二次清除） | 合计删除 **203,544,979 B ≈ 194.1 MiB** 可重建缓存/临时产物 |
| `.py` 文件数：recommendation / tests / experiments | 184 / 197 / 146（experiments 根 124） | **186 / 198 / 103**（experiments 根 81） | +2 契约种子 / +1 契约测试 / -43 归档 |
| 工作区总计（含 git-ignored 产物与环境，排除 `.git`） | ≈9,070 MB / ≈80,560 文件 | **8,894.6 MB / 74,650 文件** | ≈ -175 MB / ≈ -5,910 文件（缓存波动导致与单类合计略有差异） |
| `runs/`（实验结果，保留） | 4,171.9 MB / 11,809 | 4,102.9 MB / 6,945 | -69.0 MB / -4,864（仅删除其中 `__pycache__`，结果文件未动） |
| `models/`（模型权重，保留） | 1,252.0 MB / 14 | 1,252.0 MB / 14 | 0 |
| `data/`（原始/处理后数据，保留） | 777.1 MB / 38 | 777.1 MB / 38 | 0 |
| `training/`（训练代码 + LoRA adapter，保留） | 319.0 MB / 137 | 318.9 MB / 115 | -0.1 MB / -22（仅 `__pycache__`） |
| `backends/`（TIGER 后端 + 独立 venv，保留） | 1,183.3 MB / 35,118 | 1,182.8 MB / 35,097 | ~0（仅测试侧 `__pycache__`） |
| `.venv/` + `.venv-ranker/`（运行环境，保留） | 1,018.4 + 227.7 MB | 不变 | 0 |
| 源码树合计（recommendation/tests/experiments/benchmarks/docs/scripts/archive） | ≈31.7 MB / ≈1,645 文件 | **≈11.2 MB / ≈698 文件** | ≈ -20.5 MB / -947（缓存 + 归档源文件 + 历史脚本） |
| 归档压缩效果 | 60 文件原始 **642,490 B** | **168,346 B**（6 个 tar.gz） | 压缩约 3.8×；另加 README 与 SHA256 manifest |

### 3.2 目录结构变化（源码视角）

```text
清理前                                    清理后
-------------------------------------     -------------------------------------
recommendation/    186 py (164 tracked)   recommendation/    186 py (164 tracked + contracts/ 2)
tests/             197 py                 tests/             198 py (+1 契约测试)
experiments/       146 py (5 子包混杂)     experiments/       103 py (保留当前入口/研究/测试依赖)
  experiments/*.py  124 (43 历史 runner)    experiments/*.py    81
archive/            —                     archive/legacy_experiments/  6 tar.gz + README.md
根目录             61 md / 36 txt / ...    根目录             61 md / 36 txt / ...（研究证据保留）
  + 15 log + 2 debug + pip-*/28            （日志/调试脚本/pip 目录全部清除）
```

### 3.3 重复实现减少情况（C3）

**结论：本轮消除“重复临时资产”703 项（缓存/pip/半成品环境/误写 manifest），但未做任何
“为合并而合并”的源码合并。**

- 源码级内容完全重复扫描（SHA256 分组）：全仓库仅 4 组，均为产物/空文件，无重复模块实现：
  - `P2_FINAL_AGENT_COMPARISON.csv` = `training/agent_sft/eval/P2_BRIDGE_AGENT_COMPARISON.csv`
    （同一证据两处副本，均保留：根目录是报告引用路径，training 是运行产物路径）；
  - `P3C3B_SOURCE_DIFF.txt` = `P3C3_SOURCE_DIFF.txt`（历史快照相同，保留两份以对应各自里程碑）；
  - `P3C3C_FOUR_TURN_TRACE.csv` = `P3C_FINAL_FOUR_TURN_TRACE.csv`（同上）；
  - `experiments/__init__.py` = `backends/tiger_public/tests/__init__.py`（都是空包标记）。
- 语义重复排查与处置（不做强制合并，理由写入 `docs/P14_PREP_INTERFACE.md` §6/§9）：
  - **BM25 两套实现**：`control/catalog_search.py::CatalogSearchSource.search`（Agent 候选源，
    带 grounding/provenance）vs `inference/lexical.py::CatalogueLexicalIndex.scored`（Serving V1
    词法源，参与 RRF）。调用方/语义/测试均不同，合并会改变默认 Serving 行为 ⇒ 保留为两条明确边界。
  - **RRF**：实际只有一份实现（`control/candidate_ledger.py::reciprocal_rank_fusion`），
    `inference/fusion.py` 直接 import 复用 ⇒ 无重复。
  - **指标计算**：`recommendation/evaluation/`（full-ranking 规范协议）、`experiments/p6_benchmark/metrics.py`
    （P6 v0）、`experiments/p13/retrieval_metrics.py`（fixed-pool proxy）、`benchmarks/agent/`（task-level）
    分属 **4 个不可互换的协议**；强行合并会产生“同名不同义”的指标 ⇒ 禁止合并，只写清权威分工
    （接口文档 §6 硬警告）。
  - **多套状态/候选对象**：`agent/state.py`、`demo/state_integration.py`、`state/`（M5）职责不同；
    `control/schemas.py`、`demo/schemas.py`、`tools/schemas.py` 分属控制面/演示面/工具面 ⇒ 不合并。

---

## 4. 新旧架构职责边界（C4/C5）

完整版本见 `docs/P14_PREP_INTERFACE.md`（234 行，逐符号核对）。摘要：

### 边界一：Legacy Recommendation（冻结，只读复用）
Sports 5-core 的 SASRec/Two-Tower/Catalogue BM25 + `/v1/recommend` 默认 `serving_v1`
（`FUSION_SOURCES=("sequential","two_tower","catalog_lexical")`、`RRF_K=60`、source head=100）与其
item_id mapping / checkpoint / `agentrecx.eval_protocol.v1` 均冻结；P14 只经服务级接口只读调用，
**不得把旧 Sports 整数 item_id 默认映射到新 Shopping Companion 商品**。

### 边界二：Shared Agent Core（P14 增强，不重写）
M5 事件状态、M9 偏好记忆、Planner Action 契约（ASK/RETRIEVE/RERANK/FINALIZE）、Tool contract、
CandidateLedger + Source Policy、Hard Constraint / Eligibility、Evidence Provenance、Agent
Trajectory、Evaluator。内核不得硬编码 benchmark 字段或商品类目；当前 Sports/water-filter 特化
（`demo/product_role.py`、`demo/shopping_intent.py`）留在 legacy demo 层，不上升。

### 边界三：Dataset / Catalog / Backend Adapter（P14-0 只定义责任）
每个后端一个薄适配模块 + 契约测试；不建通用插件框架/第二套 Runtime；跨目录 id 不得静默归一。

### P14 四个最小工具契约（详见接口文档 §3）
- **Search**：`query + catalog_namespace + top_k/budget + retrieval_options` →
  `hits[{item_id, rank, score, score_kind, matched_fields}] + source + retrieval_metadata`；
- **Recommend**：`user_preferences + current_query + FIXED candidates` →
  `ranked[] + matched_preferences + conflicting_preferences + unknown_fields + evidence_source`；
  只重排固定候选集，不得重搜全目录、不得绕过硬约束；
- **Memory**：`session/user context + requested_scope` → `preferences + source_event/evidence + effective_status + timing/conflict`；
- **Product Info**：`product_id + catalog_namespace + requested_attributes` →
  `attributes{value/missing/unreadable/unsupported} + provenance + missing/unknown`。

### Catalogue namespace（P14-0 已落地的最小结构）
`recommendation/contracts/namespaces.py`：
`CatalogNamespace`（`^[a-z][a-z0-9_]{1,63}$`）、`ProductRef{namespace,item_id,parent_asin?}`、
`namespaced_id = "<namespace>:<item_id>"`、`ensure_same_namespace()`（混用抛 `NamespaceMismatchError`）、
`KNOWN_CATALOG_NAMESPACES = {amazon_sports_5core, amazon_electronics_c4, demo_catalog}`、
`PAD_ITEM_ID="0"` 仅在 legacy 整数空间内被拒绝为真实商品。配套 `tests/test_contracts_namespaces.py`
（26 passed）。不改变任何现有调用路径。

---

## 5. 必要合并与重构（C3 结论）

| 动作 | 结果 | 理由 |
| --- | --- | --- |
| 合并重复函数/模块 | **0** | 全仓库无内容重复的源码模块；语义相近者分属不同协议/契约 |
| 新增抽象 | **1**（仅 namespace 契约种子 2 文件 + 1 测试） | C5 要求商品 ID 带显式命名空间；“只建立当前必要的共同结构与验证规则” |
| 新增 Registry / Plugin / 第二 Runtime | **0** | 明确禁止；P14 接入点用既有 + ≤8 个薄 adapter |
| 修改现有运行代码 | **0** | C6 冻结行为；唯一 tracked 修改是 `.gitignore`（新增 `.venv-*/` 忽略规则，防止环境目录再次污染） |

---

## 6. 回归验证（D2）

### 6.1 环境与命令

- **执行位置：本机 Windows（工作区 `D:\IT\CODE\PROJECTION\agentrec\X`）+ WSL Ubuntu-22.04**。
  本轮**未连接、未启动、未占用 AutoDL / RTX 4090D**，**未使用 GPU**（无 BGE 批量编码、无 Qwen 推理、无训练）。
- Windows 宿主机无法直接跑完整套件（既有环境限制，非本轮引入）：`resource` 模块仅 Unix；
  `backends/tiger_public/.venv/bin/python` 是 Linux 符号链接（Windows 上 `WinError 1920`）。
  因此使用仓库自带 Linux 环境：**仓库 `.venv`（Python 3.10.12，CPU-only）**，与 CI
  `ubuntu-latest / Python 3.10` 同一口径。
- 命令（清理后分批 + 定向批次；单批均设上限，未同时运行多个大型 pytest）：

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q --no-header -p no:cacheprovider <分批文件列表>
```

> 单进程全量 `pytest tests` 在本机 WSL 7.7 GB 内存下不可完成（98% 时 OOM），该限制在 P14-0 前已存在；
> 本轮按收敛要求改为分批 + 必需领域定向批次，不再重跑全量。

### 6.2 结果（本轮收敛前实际完成）

环境：**本机 Windows + WSL Ubuntu-22.04 + 仓库 `.venv`（Python 3.10.12，CPU-only）**；
未使用 GPU、未连接/占用 AutoDL。

| 运行 | 范围 | 结果 |
| --- | --- | --- |
| 第一次全量单进程 | `pytest tests`（约全部 7,000+ 项） | 进行到 **98% 被 WSL OOM killer 终止**（dmesg：`anon-rss 7,250,532 kB`，WSL 内存 7.7 GB）；无 summary，不用于结论 |
| 分批 run chunk1 | 38 个测试文件 | **846 passed, 3 skipped, 2 failed** in 316.5 s |
| 分批 run chunk2 | 37 个测试文件 | **845 passed, 13 skipped, 1 failed** in 195.7 s |
| 分批 run chunk3 | 37 个测试文件 | **648 passed, 13 skipped, 1 failed** in 117.0 s |
| 分批 run chunk4 | 37 个测试文件 | **790 passed, 0 failed** in 609.7 s |
| 分批 run chunk5 | 37 个测试文件 | **未完成**：按收敛指令在约 29% 时安全终止；已跑部分无失败，未计入统计 |
| 必需领域定向批次 | 20 个关键测试文件（Serving V1 / Planner / CandidateLedger / Eligibility / Memory / M5 / P6 / P12 / P13 / P13-R / SASRec checkpoint / Two-Tower / fusion_tiger / docs guards） | **451 passed, exit 0** in 34.0 s（硬上限 280 s） |
| 契约种子测试 | `tests/test_contracts_namespaces.py` | **26 passed** |
| 字节编译 | `compileall recommendation tests experiments` | **PASS（exit 0）** |
| 导入 smoke | `create_app(enable_demo=True)`（15 条 OpenAPI 路由）+ `ProductRef` | **PASS** |
| P13 / P13-R 既有测试 | `tests/test_p13_pipeline.py`、`tests/test_p13r_pipeline.py` | 包含在 451 passed 定向批次中，**全部通过** |

**分批合计（chunk1–4，不代表全仓库）**：**3,129 passed / 4 failed / 29 skipped**。

### 6.3 失败与未完成说明（分类记录）

**4 个失败（逐项定位；全部位于本轮 0 改动的代码/测试路径，与归档删除/缓存清理无依赖关系）：**

| # | 失败测试 | 失败信息 | 分类/原因 |
| --- | --- | --- | --- |
| 1–2 | `tests/test_http_tiger_surface.py::test_closing_the_agent_leaves_a_borrowed_engine_running`、`…stops_an_engine_it_created` | `AttributeError: 'types.SimpleNamespace' object has no attribute '_m8_choice_store'` @ `recommendation/demo/agent_service.py:742` | 测试 stub 与 P5–P13-R 未提交 `agent_service.py` 的属性缺口；该文件本轮**未改动**、不在归档链上 ⇒ **与清理无关** |
| 3 | `tests/test_m5_1_shopping_integration.py::test_legacy_memory_disagreement_is_reported_at_kind_level` | `assert [] == ['price_max']`（legacy memory ↔ M5 state 差异报告语义与期望不一致） | 涉及 `demo/state_integration.py` 等**本轮前已修改**的未提交文件；本轮未触碰 ⇒ **与清理无关** |
| 4 | `tests/test_p1b3_soft_rerank.py::test_unconfirmed_or_unsupported_preference_never_enters_m61` | `soft_preferences_from_ranker_projection(...)` 返回非空，测试期望 `()` | P1B3 时代 soft-preference 语义与测试期望不一致；相关文件本轮未触碰 ⇒ **与清理无关** |

> 诚实说明：由于第一次全量运行 OOM、且按要求不再反复恢复旧树重跑，**无法逐项证明这些失败在 P14-0
> 之前就已存在**。能证明的是：它们所在的模块与测试不在 P14-0 改动集内，也不导入任何被归档模块，
> 因此 **没有证据表明是清理引入的新回归**；本报告不把它们标成“已确证 pre-existing”。

**未完成（LIMITED 的原因）：**
- chunk5 的 37 个测试文件未跑完（其中 Agent memory/state、Eligibility、M5 acceptance、P6 classical、
  P12 evidence、SASRec checkpoint/ranking、TwoTower、docs guards 等**必需领域**已由 451-passed 定向批次覆盖；
  剩余为 m3/m4/v2 schema/p4 api 等历史测试）。
- 单进程全量套件在本机 WSL 内存限制下不可完成；该限制在 P14-0 之前即存在（测试读取真实 `data/`、`runs/`
  大文件并在 `/mnt/d` 9p 文件系统上做大量 I/O，进程内存累积至 OOM）。
- 未运行：GPU 相关、训练、冻结 Holdout、P13-R reserved test（本轮不允许）。

---

## 7. 归档与删除验证（D3）

| 验证 | 命令/证据 | 结果 |
| --- | --- | --- |
| tar 成员 = manifest 文件清单 | `tar -tzf <group>.tar.gz` 与 manifest `original_path` 排序比较 | 6/6 组一致 |
| 解包字节一致 | 解到全新临时目录，逐文件 SHA256 + size 对比 | **60/60 一致，0 失败** |
| tarball 自身完整性 | 对 `.tar.gz` 计算 SHA256 与 manifest `archive_sha256` 对比 | 6/6 一致 |
| 原文件已移除 | 计划内 60 个路径 `Test-Path` | 60/60 不存在 |
| git 跟踪文件恢复路径 | 43 个均为 tracked（`git ls-files` 可查），`git checkout HEAD -- <path>` 可恢复 | 已验证跟踪状态 |
| 未跟踪文件恢复路径 | 17 个仅在 tarball（无 git 历史） | manifest 记录，**不得只依赖 git** |
| 删除项可恢复性 | 缓存/pip/`__pycache__`：导入或运行即重建；`.venv-bge`：坏环境，可用 `.venv`/依赖重建；`run.json`：`runs/p10/bge_sasrec/run.json` 同内容 | 均非唯一资产 |
| 原有未提交修改未误伤 | 清理前 `git status` 的 28 个 ` M ` 文件与清理后逐一对照 | **28/28 保留、0 内容变化**（`git diff` 只新增本轮 `.gitignore` 一项修改；43 个 ` D ` 全部是计划内归档文件） |

恢复命令（在仓库根目录）：

```bash
tar -xzf archive/legacy_experiments/<group>.tar.gz -C .
```

---

## 8. 代码质量检查

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 字节编译（CI 同款） | `.venv/bin/python -m compileall -q recommendation tests experiments` | **PASS（exit 0）** |
| 新契约测试 | `.venv/bin/python -m pytest tests/test_contracts_namespaces.py -q` | **26 passed** |
| 导入 smoke | `create_app(enable_demo=True)` + `ProductRef(namespace=demo_catalog)` | **PASS（15 条 OpenAPI 路由；`demo_catalog:demo-socks`）** |
| 手工 demo smoke（等价 CI 路径） | CI 的 `build_demo_catalog` + `demo_scenario` 由 `tests/test_packaging.py`（chunk4 通过）与 `test_docs.py`（定向批次通过）覆盖 | **PASS（间接，经测试）** |
| 文档守卫 | `pytest tests/test_docs.py -q` | **PASS（含在 451 passed 批次）** |

---

## 9. P14 接入建议（D4）

**原样复用（不改）**：`recommendation/state/`（M5）、`recommendation/memory/`（M9）、
`demo/planner_contract.py` 契约、`control/{candidate_ledger,grounding,constraint_eligibility,trajectory}.py`、
`recommendation/evaluation/`、`benchmarks/agent/`。

**适配后复用**：`control/catalog_search.py` + `control/product_research.py`（加 namespace/固定候选集输入）、
`preference_matching/` + `reranking/`（组合成 P14 Recommend 后端）、`catalog/metadata.py`（目录 registry）。

**P14 需要新建的 adapter（≤8，接口文档 §7 有落点）**：Planner→runtime 桥、dataset loader、
catalogue namespace 注册、Search backend、Recommendation/preference scorer、Product info provider、
benchmark task builder、P14 evaluator 版本。

**进入 P14 前的两个明确决策**（接口文档 §11/GAP）：① Planner 契约目前只在实验 harness/tests 跑，
`/agent/shopping` 运行时是 `ActionKind`——P14 必须明确桥接方式，不能另建第二 planner；
② P13-R 的 `catalog_meta.jsonl` 身份/生产者漂移（76,918 条中 72 条 item_id 非 ASIN，
`experiments/p13r/retrieval.py` 把 C4 item_id 当 `parent_asin`）——复用该目录前必须显式登记
namespace 与 producer/hash。

**进入 P14 的数据前置条件（本轮不处理）**：`data/p13r/electronics/catalog.jsonl` 与 BGE 向量索引
本地缺失（远端/重建）；P14 Electronics/Companion 数据与标签方案仍待决策（见 P13-R 开放问题）。

---

## 10. 未完成 / 保留事项（诚实清单）

1. **未 commit/push/deploy**（按任务约束）。工作区仍有 28 个 P5–P13-R 未提交修改 + 本轮改动；
   本轮没有 reset/覆盖它们（逐文件对照 before/after `git status` 验证）。
2. **未改旧 docs**（`docs/ONBOARDING.md`、`docs/README.md` 索引等）：其内容到 10 月下旬，
   且 `docs/README.md` 已被用户修改（未提交），本轮避免与用户未提交工作冲突；P14 可再统一刷新。
3. **根目录 61 份阶段报告未迁移**：它们是 P11/P12/P13/P13-R 等冻结证据，被报告间相互引用；
   迁移会破坏历史引用。按“证据保留优先”留在原位。
4. **模型三种格式权重重复保留**（见 §2.4）：非“可重新下载”资产的保守处理。
5. **P13-R `catalog.jsonl` / BGE 索引本地缺失**：不影响本轮整理，但影响 P14 复用。
6. **全量测试的慢测试**：见 §6.3；属既有测试特征（读大量真实数据），非本轮引入。
7. **`backends/tiger_public/.venv` 与 `.venv`/`.venv-ranker` 未纳入源码清单**：它们是环境，
   不是源码；仅确认本轮未改动其内容（git status/时间戳核对）。

---

## 11. 交付物清单

| 交付物 | 状态 |
| --- | --- |
| `P14_PREP_REPORT.md`（本文件） | ✅ |
| `P14_PREP_SOURCE_DIFF.txt`（修改/归档/删除全清单，由前后快照生成） | ✅ |
| `P14_PREP_ARCHIVE_MANIFEST.json`（6 组/60 文件 SHA256 + 恢复方法） | ✅ |
| `docs/P14_PREP_INTERFACE.md`（边界/复用/契约/GAP/接入建议） | ✅ |
| `archive/legacy_experiments/`（6 tar.gz + README.md） | ✅ |
| 清理后的代码（0 删除/0 修改现有运行代码；新增 `recommendation/contracts/` + 测试） | ✅ |
| 回归验证（WSL 全量测试前后对比、compileall、demo smoke） | 见 §6/§8 |
