# P6 Classical Full-Ranking Evaluation Report

本轮任务（用户已确认）：实现只读 full-ranking scorer 适配器，在 P6 classical 数据上运行
SASRec / Two-Tower，**两个 history 口径都跑、分开标注**；先 dev，脚本冻结后一次性跑 holdout；
不训练、不启动 GPU、不修改默认 Serving V1 / Gold / Evaluator / 协议数据。

状态一句话：**实现 PASS，dev/holdout 已实际执行，质量结论 UNVERIFIED（样本功效不足），
Fusion 严格 full-ranking BLOCKED，dynamic 评测 BLOCKED。**

## 1. 新增文件

```text
experiments/p6_benchmark/full_ranking.py      只读 scorer 适配器（SASRec / Two-Tower）
experiments/p6_benchmark/run_classical.py     classical full-ranking 评测 runner（CPU）
experiments/p6_benchmark/uncertainty.py       post-hoc bootstrap 不确定性分析（非冻结指标）
tests/test_p6_classical_eval.py               9 项 CPU 单测
P6_CLASSICAL_EVAL_REPORT.md                   本报告
P6_CLASSICAL_EVAL_SOURCE_DIFF.txt             新增源码 diff
runs/p6_classical_eval/                       运行产物（`runs/` 已被 gitignore）
  smoke_2cases/                               smoke（2 cases × 2 模型 × 2 口径 + 确定性检查）
  dev_20261107_125847/                        dev 160 cases
  holdout_20261107_130105/                    holdout 40 cases（一次性）
```

未修改任何既有项目文件；未 commit / push / deploy；未训练；未使用 GPU。

## 2. 协议与两个 history 口径

### 2.1 评测语义

- 输入：`data/p6_benchmark_v0/classical_{dev,holdout}.jsonl` 冻结 case；
- 打分：模型对 **全 catalogue** 输出原始分数向量（长度 `num_items + 1`，index 0 = PAD）；
- 排序：`recommendation/evaluation/metrics.sorted_ranking`，即仓库统一 kernel：
  - candidates = `1..num_items` 减去 seen history；
  - **target 从 mask 中移除后仍可参与**（`excluded_seen = set(history) - {target}`）；
  - tie rule：分数高者在前，同分 item id 小者在前；
- 指标：`experiments/p6_benchmark/metrics.py` 的 P6 定义：
  `Recall@10 / Recall@50 / NDCG@10 / MRR@10 / CatalogueCoverage@10`；
- **full-ranking 与 top-k 的区别**：runner 对每个 case 计算完整 catalogue 排序，只把前 50 个
  ID 传给指标函数。P6 的所有 cutoff ≤ 50，因此前 50 截断与完整排序的指标值**完全一致**；
  这不是 retrieval head，也不是把 top-k 结果当 full ranking。

### 2.2 两个口径（本轮都运行、分开报告）

| 口径 | model input history | 说明 |
|---|---|---|
| `p6_shared_history` | `items[:-2]` | P6 benchmark v0 交付文件原样；validation/test target 共用同一 history |
| `eval_protocol_v1_test_history` | `items[:-2] + validation_target` | 仓库统一 `agentrecx.eval_protocol.v1` test 模式；validation 交互是更早的已观测事件 |

两者区别：每个 case canonical 口径多 1 个 history item（长度 +1）。之所以分开，是因为
P6 v0 文件与仓库正式指标不是同一口径，不能混报。

### 2.3 对 P6 v0 交付物的核对结论

本轮只读核对发现，P6 v0 报告与实现存在以下**文档/口径不一致**（本轮未修改 P6 文件，
留待 P6 v1 决定）：

1. `build_benchmark.py` 只从 `split.py` import 了 `MIN_SEQUENCE_LENGTH`，**没有复用**
   `temporal_leave_two_out` / `EvaluationCase`；报告 §3 “复用 split.py” 的表述不准确。
2. `protocol.json` 仍把该切分命名为 `temporal_leave_two_out`，但它与 `split.py` 的 test
   模式不同（P6 test history 不含 validation target）。本轮以两个口径并列的方式回避混报。
3. `leakage_report.json` 的 `history_target_overlap_cases = 3`：实测这 3 例是
   **validation target** 在 history 中；**test target 在 history 中 = 0**（详见 §5）。
   报告 §4 的 “target 曾出现在历史中” 表述容易被读成 test target，P6 v1 应修正措辞。
4. 数据本身自洽：asin↔item_id 一致、最大 item id 156724 ≤ 156746、时间戳严格递增、
   dev/holdout user 不交叉、test target item overlap=2（已记录，不伪装 item-disjoint）。

### 2.4 未评测的 arm

- **Fusion / Serving V1**：`FusionInferenceEngine.recommend` 只返回 fused top-k
  （每源 head=100），没有全 catalogue score vector；未进入任何 source head 的商品没有
  定义分数。**本轮没有伪造 tail 顺序，也没有为 Fusion 重写底座**，因此 Serving V1
  Fusion 的严格 full-ranking 记为 BLOCKED / 未评测。
- 既有 `runs/phase5_benchmark_public/arms.json` 是 20,000-case canonical 评测，但其
  `fixed_fusion` 源是 popularity + sequential + metadata，**不是 Serving V1 的
  sequential + two_tower + lexical**，不能冒充 P6 Fusion 结果。
- dynamic（Traditional / Rule / Qwen Base / SFT-v0）未评测：GPU 关闭、远端无
  `recommendation/` 包、本地也没有 dynamic runner；见 §9。

## 3. 环境与资产

```text
本地 Python: C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env
Python 3.10.19 / torch 2.5.1 / device=cpu
```

- 本地 torch 能枚举到 RTX 5070 Laptop（8 GiB），但该 torch 构建不支持 sm_120，
  CUDA 实际不可用；本轮全部 CPU，且按约束不启动远端 GPU。
- SASRec checkpoint：`runs/sasrec_canonical_2026/best.pt`
  （sha256 `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912`，
  与 `run.json` 记录一致，runner 启动时强制校验）。
- Two-Tower checkpoint：`runs/twotower_public_2026/checkpoint.pt`
  （sha256 `abbfecdacc71807a4860e651e6ab8355f13164986cf9c184bdc3e54ab6ada02f`）。
- 数据：`data/p6_benchmark_v0/classical_{dev,holdout}.jsonl`；
  mappings：`data/processed/Sports_and_Outdoors_mappings.json`；
  dev split sha256 前缀 `c137cf5f…`，holdout `560fab03…`（完整值在 summary.json）。
- Git HEAD：`4dec6d2081902f1e89265a06ea4ded595087dd43`，分支
  `feat/agent-source-limit-ablation`；dev 运行时 summary 记录工作区有 179 条 dirty 条目
  （包含此前未提交的历史工作以及本轮新增文件；本轮未 commit）。

## 4. 实际执行命令

```powershell
# 单测（13 passed）
python -m pytest -q tests/test_p6_classical_eval.py tests/test_p6_benchmark.py

# 2-case smoke：2 模型 × 2 口径 + 每模型/口径各 2 case 的重复打分确定性检查
python experiments/p6_benchmark/run_classical.py --split dev --limit 2 `
  --check-determinism 2 --out runs/p6_classical_eval/smoke_2cases

# dev 160 cases（脚本冻结）
python experiments/p6_benchmark/run_classical.py --split dev

# holdout 40 cases：脚本未再修改，一次性运行
python experiments/p6_benchmark/run_classical.py --split holdout

# post-hoc bootstrap（不参与评测、不用于选模）
python experiments/p6_benchmark/uncertainty.py --results-dir <dev_dir>
python experiments/p6_benchmark/uncertainty.py --results-dir <holdout_dir>
```

## 5. 结果

### 5.1 dev（n=160）

括号内为 post-hoc bootstrap 95% CI（百分位法，seed=20261107，10000 次；**非冻结协议指标**）。

| model | variant | Recall@10（95%CI） | Recall@50 | NDCG@10 | MRR@10 | Cov@10 | hit@10 / hit@50 | mean rank |
|---|---|---|---|---|---|---|---|---|
| SASRec | p6_shared_history | 0.0125 [0.0000, 0.0312] | 0.0312 | 0.0058 | 0.0036 | 0.001652 | 2 / 5 | 22619.8 |
| SASRec | eval_protocol_v1 | 0.0125 [0.0000, 0.0312] | 0.0250 | 0.0060 | 0.0040 | 0.001665 | 2 / 4 | 21113.0 |
| Two-Tower | p6_shared_history | 0.0000 [0.0000, 0.0000] | 0.0125 | 0.0000 | 0.0000 | 0.003266 | 0 / 2 | 45036.8 |
| Two-Tower | eval_protocol_v1 | 0.0125 [0.0000, 0.0312] | 0.0312 | 0.0046 | 0.0023 | 0.003381 | 2 / 5 | 43098.5 |

dev 的 top-10 hit 集中在 case `classical-dev-00028` 与 `classical-dev-00188`（SASRec 两个口径、
Two-Tower canonical 口径均命中）；这只是 2 个命中，不能外推为模型优劣。

### 5.2 holdout（n=40，只运行一次）

| model | variant | Recall@10（95%CI） | Recall@50 | NDCG@10 | MRR@10 | Cov@10 | hit@10 / hit@50 | mean rank |
|---|---|---|---|---|---|---|---|---|
| SASRec | p6_shared_history | 0.0250 [0.0000, 0.0750] | 0.1000 | 0.0079 | 0.0031 | 0.000944 | 1 / 4 | 24528.0 |
| SASRec | eval_protocol_v1 | 0.0000 [0.0000, 0.0000] | 0.0250 | 0.0000 | 0.0000 | 0.000970 | 0 / 1 | 24463.8 |
| Two-Tower | p6_shared_history | 0.0000 [0.0000, 0.0000] | 0.0500 | 0.0000 | 0.0000 | 0.001174 | 0 / 2 | 26794.6 |
| Two-Tower | eval_protocol_v1 | 0.0000 [0.0000, 0.0000] | 0.0500 | 0.0000 | 0.0000 | 0.001167 | 0 / 2 | 25525.2 |

holdout 的 1 个 top-10 hit 是 `classical-holdout-00054`（SASRec/p6 口径，rank 8）；
其余 top-50 hit 均未进入 top-10。

### 5.3 运行成本与产物

| 运行 | cases | wall time | 产物 |
|---|---:|---:|---|
| smoke_2cases | 2 | 6.37 s | `runs/p6_classical_eval/smoke_2cases/summary.json` |
| dev_20261107_125847 | 160 | 124.77 s | `summary.json`、`per_case.jsonl`、`uncertainty.json` |
| holdout_20261107_130105 | 40 | 31.90 s | 同上 |

引擎加载：SASRec 0.535 s、Two-Tower 0.918 s；dev 单口径耗时 SASRec ≈18.8 s、
Two-Tower ≈42–43 s；holdout 全部 ≈32 s。

产物 sha256（完整）：

```text
dev/summary.json       b131485ae4e6247cc3a2519260a09e9044c5d08b1ef62676fa9bff3e00714c7f
dev/per_case.jsonl     674b11e930c2f643c0272c45b1649d184205d4811efa73513b405552dafa211c
dev/uncertainty.json   7e2ede80570119c8f00e35151b9b3643ff4ebf0e06019b6cdf67a56094bea48d
holdout/summary.json   6f748d72ab819f5b42b1c27a047a618567c7351bc72c135e9e0e36e57d657bde
holdout/per_case.jsonl 3d983ea2d19fabe190375eb1eaf6c5972d966898669c2daab84b76e0b7b0e7fd
holdout/uncertainty.json db75375e1bfc46f13950df508a979337f797cce9f1e64f3d0f4e944f6bc01a2d
```

## 6. 验证与 sanity check

1. **单测 13 passed**：`tests/test_p6_classical_eval.py`（9 项）+
   `tests/test_p6_benchmark.py`（4 项），覆盖：
   - 两个 history 口径的定义与差异；
   - 已知分数下 P6 指标精确值（含 masked history 不出现、target re-admitted）；
   - target rank > 50 时四个 cutoff 指标全为 0、coverage 正确；
   - scorer 必须返回 `num_items + 1` 个分数；
   - 200 条冻结 P6 case 的 test target 不在 P6 口径 history 中；canonical 口径若出现
     test target，只能是 validation target == test target 的重复交互；
   - 确定性检查、bootstrap 函数。
2. **smoke 确定性 PASS**：2 cases × 2 模型 × 2 口径，重复打分完整排序完全一致
   （`deterministic=true`，`mismatches=[]`）；SASRec 两个 case 的 rank（2337 / 39981）
   与独立 ad-hoc smoke 完全一致。
3. **full-ranking 一致性 PASS**：每个 case 检查
   `(exact_rank <= 50) == (target ∈ top50)`；per-case 均值与 summary 聚合值在 1e-12 内一致
   （uncertainty.py 内置断言）。
4. **防泄漏审计 PASS**：
   - P6 口径：`test_target_in_model_history_cases = 0/160 (dev), 0/40 (holdout)`；
   - canonical 口径：dev 有 1 例（`validation_target == test_target` 的重复交互，
     协议本来就允许 target re-admit），holdout 为 0；
   - target 从未作为标签传给 scorer（scorer 只接收 history）。
5. **holdout 一次性 PASS**：holdout 结果目录只有一个，且是在 dev 检查完成、脚本冻结后运行；
   `uncertainty.py` 是后加的独立分析工具，不改变任何排序/指标。

## 7. 解读（必须与实现 PASS 区分）

- **实现层面 PASS**：GPU 之外的全链路（checkpoint 校验 → full-catalogue score →
  canonical ranking → P6 指标 → 产物/哈希）真实跑通，结果可复现、可审计。
- **质量结论 UNVERIFIED**：
  - P6 v0 dev=160 / holdout=40，`Recall@10` 的每个 hit 就是 0.625%（dev）/ 2.5%（holdout）；
  - 95% CI 全部很宽且大量重叠（例如 dev SASRec R@10 [0, 0.0312]），
    **无法据此声称 SASRec 优于 Two-Tower，也无法声称任一 history 口径更好**；
  - dev 与 holdout 的方向都发生了翻转（dev canonical 的 NDCG 略高；holdout 是 P6 口径的
    R@10 更高），这正是小样本噪声的表现；
  - 因此本报告不给出任何模型排名或“P6 质量 PASS”的结论。
- 与既有正式结果的对照仅作合理性检查：SASRec 在 412,445 用户 canonical 正式口径的
  HR@10 = 0.013565（`run.json`），本轮 dev canonical 为 2/160 = 0.0125，在 160 样本分辨率内
  不矛盾。这不是复现，也不能替代正式 cohort 结果。
- `CatalogueCoverage@10` 是 full-catalogue 全排序下 top-10 集合的覆盖率，数值天然很小
  （0.0009–0.0034），不能与 retrieval head 的 coverage 直接比较。
- bootstrap 只是 post-hoc 不确定性展示，不是冻结协议；`CatalogueCoverage@10` 是集合指标，
  未做 bootstrap。

## 8. 判定汇总

| 项 | 判定 |
|---|---|
| 只读 full-ranking scorer 适配器 | **PASS** |
| P6 classical dev 160 实际运行 | **PASS（实现/流程）** |
| P6 classical holdout 40 一次性运行 | **PASS（实现/流程）** |
| dev/holdout 指标可复现、产物完整 | **PASS** |
| P6 质量结论 / 模型优劣 | **UNVERIFIED（统计功效不足）** |
| Serving V1 Fusion 严格 full-ranking | **BLOCKED / 未评测**（只有 head-fusion top-k，无全 catalogue 分数） |
| P6 dynamic 四组公平评测 | **BLOCKED**（GPU 关闭、远端无 `recommendation/` 包、无 runner） |
| Planner SFT-v0 任务级收益 | **UNVERIFIED**（沿用 P4C 结论，本轮未重新评测） |
| Planner SFT-v0 Adapter 备份 | **BLOCKED**（实例关闭、SSH 不可用） |

## 9. 限制与下一步建议

限制：

1. dev/holdout 太小，无法支撑质量结论；要回答“SASRec vs Two-Tower / 两个口径”的问题，
   需要更大 cohort（例如已冻结的 20,000 case canonical cohort）或更合适的配对统计设计。
2. holdout 只做**用户级**隔离，test target item overlap=2（v0 已记录），不是严格
   item-disjoint。
3. Fusion 的 full-ranking 不可直接定义；若要做 Serving V1 评测，只能明确选择
   “head-fusion top-k”协议，或另立 pre-registered 的全排序融合定义，不能伪造 tail。
4. dynamic Agent 评测缺 runner 与 GPU，本轮未触碰。
5. P6 v0 报告中的 `split.py` 复用表述、`temporal_leave_two_out` 命名、
   `history_target_overlap` 措辞需要在 P6 v1 修订；本轮按“两个口径并列”如实报告。

建议下一步（按优先级）：

1. 先决定 P6 v1 协议：修正文档/命名，明确是否统一到 `eval_protocol.v1`，或保留 shared-history
   变体并改名；
2. 在更大 cohort（20k 或全量 canonical）上复跑本 runner（同一代码可直接复用），以获得
   有统计意义的质量结论；
3. 如需 Fusion，单独立项定义 head-fusion 协议，并在报告中明确它不是 full ranking；
4. dynamic 评测等 GPU/实例与 runner 就绪后再做，并保持四组公平控制。

## 10. 交付

- `experiments/p6_benchmark/full_ranking.py`
- `experiments/p6_benchmark/run_classical.py`
- `experiments/p6_benchmark/uncertainty.py`
- `tests/test_p6_classical_eval.py`（9 passed，与既有 P6 测试合并 13 passed）
- `P6_CLASSICAL_EVAL_REPORT.md`（本文件）
- `P6_CLASSICAL_EVAL_SOURCE_DIFF.txt`
- `runs/p6_classical_eval/{smoke_2cases,dev_20261107_125847,holdout_20261107_130105}/`

未训练、未启动 GPU、未修改默认 Serving V1 / Gold / Evaluator / P6 数据、未 commit / push / deploy。
