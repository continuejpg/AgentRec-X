# P6 Recommendation Quality Benchmark v0

## 1. 目标与状态

本轮只建立评测协议、时间切分、任务 Schema 和最小可执行验证；没有训练模型、没有启动
GPU、没有运行完整 Benchmark。

状态：**PROTOCOL READY (DEV SAMPLE), FULL EVAL PENDING**。

## 2. 代码与数据

新增：

- `experiments/p6_benchmark/build_benchmark.py`
- `experiments/p6_benchmark/metrics.py`
- `tests/test_p6_benchmark.py`

生成目录：`data/p6_benchmark_v0/`

```text
classical_dev.jsonl       160 cases
classical_holdout.jsonl    40 cases
dynamic_dev.jsonl           6 tasks
dynamic_holdout.jsonl       4 tasks
leakage_report.json
protocol.json
```

## 3. 经典个性化评测协议

复用：

- 数据：`data/processed/Sports_and_Outdoors_sequences.json`
- split：`recommendation/evaluation/split.py` 的 `temporal_leave_two_out`
- 指标：`recommendation/evaluation/metrics.py` 的 hit/NDCG 定义
- 现有 SASRec / Two-Tower / Fusion 推理接口

时间切分：

```text
sequence = [i1, i2, ..., i(n-2), i(n-1), in]
history  = [i1, ..., i(n-2)]
validation target = i(n-1)
test target       = in
```

规则：

- 只用前 n-2 个交互作为模型输入；
- 验证目标为 n-1，测试目标为 n；
- 历史时间戳必须严格不晚于目标时间；
- 目标商品从 seen-mask 中移除后仍可作为正样本；
- 不进行 sampled-negative；使用 full-catalogue ranking；
- 已有历史商品被 mask，PAD 0 永不参与。

指标：

```text
Recall@10  = 1[rank(target) <= 10]
Recall@50  = 1[rank(target) <= 50]
NDCG@10    = 1/log2(rank+1) if rank <= 10 else 0
MRR@10     = 1/rank if rank <= 10 else 0
CatalogueCoverage@10 = |union(top10 recommendations)| / catalogue_size
```

## 4. 时间切分与泄漏检查结果

样本来自真实 Sports & Outdoors 序列：

```text
num_users_total = 412,445
num_items       = 156,746
dev cases       = 160
holdout cases   = 40
user_overlap    = 0
dev test targets = 160 unique
holdout test targets = 40 unique
test target overlap dev/holdout = 2
history-target overlap cases = 3
```

结论：

- 用户级 dev/holdout 完全隔离；
- holdout 不用于 SFT、prompt 调参或模型选择；
- 有 2 个测试目标商品在 dev/holdout 间重合，已明确记录，不假装 item-disjoint；
- 有 3 个 case 的 target 曾出现在该用户历史中，遵循现有 protocol（target 从 mask 中移除），
  不是未来泄漏；
- 所有历史时间戳早于验证目标，验证目标早于测试目标；
- 未使用冻结评测任务做调参。

如果后续要求严格的 item-disjoint holdout，需要重新采样并接受更小的 holdout 集合；当前
v0 选择用户级隔离，并显式报告 item overlap。

## 5. 动态购物任务 Schema

`dynamic_{dev,holdout}.jsonl` 每条任务：

```json
{
  "task_id": "dynamic-dev-budget_update",
  "split": "dev",
  "synthetic_user_request": true,
  "template_id": "dev_budget_update",
  "source": {"catalogue": "amazon_reviews_2023:meta_categories", "not_frozen": true},
  "initial_request": "I need a water filter under $40 for camping.",
  "turns": [{"message": "..."}],
  "catalogue_seed_asins": ["real parent_asin", "..."],
  "checks": {
    "must_preserve_hard_constraints": true,
    "must_not_present_role_mismatch": true,
    "must_report_unknown_when_no_eligible": false,
    "forbidden_presentation": ["hard-constraint violator", "role MISMATCH as match"]
  },
  "max_steps": 6,
  "max_tool_calls": 4
}
```

dev/holdout 使用互不重叠的 `template_id`；任务需求为合成，商品目录为真实数据。

设计覆盖：

- 当前购物目标；
- 预算更新；
- 偏好撤回；
- 目标切换；
- replacement accessory / 商品角色；
- 合格候选不足；
- 正常终止。

## 6. 动态指标定义

结果侧需要每个 task 一条记录，至少包含：

- `verified_success`
- `hard_constraint_violations`
- `role_correct / role_total`
- `preference_update_consistent`
- `eligible_candidate_present`
- `tool_calls`
- `unknown`

计算：

```text
VerifiedTaskSuccess        = count(verified_success) / tasks
HardConstraintViolationRate= count(violations > 0) / tasks
ProductRoleCorrectness     = sum(role_correct) / sum(role_total)
PreferenceUpdateConsistency= count(preference_update_consistent) / tasks
EligibleCandidateCoverage  = count(eligible_candidate_present) / tasks
ToolCallsPerTask           = mean(tool_calls)
UnknownRate                = count(unknown) / tasks
```

`VerifiedTaskSuccess` 只由可检查的约束/角色/终止条件决定，不依赖模型 rationale，也不使用
不可见的 Gold 商品身份。

## 7. 公平实验协议

固定：

- 同一 Sports & Outdoors catalogue；
- 同一初始用户历史和 M5 有效状态；
- 同一 candidate sources：history + catalog_search；
- 同一 k=5，max_steps=6，max_tool_calls=4；
- 同一 Eligibility 和 role 规则；
- 同一评测脚本和 dev/holdout 划分；
- 不修改 Gold/Evaluator，不为提高分数放宽约束。

对照组：

| Arm | 用途 |
|---|---|
| Traditional Recommendation | Serving V1 直接推荐 |
| Rule Agent | 默认 Rule Agent |
| Qwen Base Planner | 真实 Base `/plan` |
| Planner SFT-v0 | P4B Adapter `/plan` |

开发集用于调 prompt / 数据 / 选择配置；holdout 只用于最终报告。已有冻结任务不复用于调参。

## 8. 最小验证

命令：

```text
python -m pytest tests/test_p6_benchmark.py -q
```

结果：

```text
4 passed
```

覆盖：

- 指标已知值计算；
- dev/holdout 用户不交叉；
- 时间戳历史早于目标；
- 动态任务 template 不交叉；
- 动态指标定义。

脚本构建：

```text
python experiments/p6_benchmark/build_benchmark.py
```

真实输出见 `data/p6_benchmark_v0/leakage_report.json`。

## 9. 现有模型/评测接口复用判断

- `split.py` / `EvaluationCase` / temporal_leave_two_out：可直接复用；
- `metrics.py` 的 ranking 与 hit/NDCG 定义：可复用；
- `SASRecInferenceEngine` 和 `TwoTowerInferenceEngine` 可提供 catalogue score vector，
  但需要 evaluator 适配器把 score vector 交给现有 `FullRankingEvaluator`；
- `FusionInferenceEngine.recommend()` 返回 fused top-k，**不直接返回全 catalogue score
  vector**；若要计算严格 full-ranking Recall@10/50，需要复用其内部 source score 或
  增加只读 full-score 接口，不能把 top-k 当作 full ranking；
- 本轮的指标脚本接受 ranked ID 列表，方便与 Rule/Planner final candidates 对齐；
  full-ranking 适配仍是下一步代码工作，不是本轮伪造结果。

## 10. 需要 GPU 执行的具体任务

1. 用 SASRec / Two-Tower / Fusion 在完整 160/40 classical 划分上生成真实 full-ranking
   scores（CPU 可能很慢；GPU 可选加速，不启动本轮）。
2. 运行四个 arm 的动态任务评测：Traditional、Rule Agent、Qwen Base、Planner SFT-v0。
3. 真实 Qwen Base / Planner SFT-v0 的多步轨迹需要 GPU `/plan` 服务。
4. 若严格 item-disjoint 重采样后需要重新跑全部 classical arms。
5. 完整 holdout 最终报告和配对 bootstrap 置信区间（尚未实现）。

## 11. 结论

- **时间切分与用户级隔离：PASS（小样本）**；
- **Schema/指标定义：PASS（4 tests）**；
- **动态任务样本：PASS（6 dev / 4 holdout，模板隔离）**；
- **完整模型质量：UNVERIFIED**（未运行完整 classical/dynamic 评测）；
- **公平四组对照：UNVERIFIED**（协议已定义，GPU 任务待执行）。

## 交付

- `P6_BENCHMARK_V0_REPORT.md`
- `P6_BENCHMARK_V0_SOURCE_DIFF.txt`
- `data/p6_benchmark_v0/`（protocol、split cases、dynamic tasks、leakage report）
- `experiments/p6_benchmark/build_benchmark.py`
- `experiments/p6_benchmark/metrics.py`
- `tests/test_p6_benchmark.py`（4 passed）

未训练、未启动 GPU、未修改默认 Serving V1/Gold/Evaluator、未 commit/push/deploy。
