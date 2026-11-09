# P4A Planner SFT v0 Report

## 1. 结论

已生成 **320 条** Planner 决策 SFT v0 数据，格式与现有
`agentrecx.planner_action.v1` Contract 一致；label 来自规则/工具语义，**未使用 Base Qwen
输出作为标签**。本轮未训练、未跑 Benchmark。

## 2. 数据产物

目录：`data/planner_sft_v0/`

| 文件 | 内容 |
|---|---|
| `train.jsonl` | 267 条训练样本 |
| `val.jsonl` | 53 条验证样本 |
| `chat_train.jsonl` | 267 条 Qwen Chat 转换 |
| `chat_val.jsonl` | 53 条 Qwen Chat 转换 |
| `manifest.json` | 数量/动作/场景/来源统计 |
| `schema.json` | 字段与 Action 枚举约束 |
| `README.md` | data card、schema、split、标签规则、限制 |

生成/转换脚本：

`training/planner_sft/build_v0.py`

## 3. 规模与分布

```text
samples: 320
train: 267
val: 53

actions:
RETRIEVE 100
ASK       50
RERANK    90
FINALIZE  80

scenarios:
retrieve_initial_clear       60
retrieve_after_empty         40
ask_ambiguous                30
rerank_soft_weight           50
rerank_soft_feature          40
finalize_after_rerank        50
finalize_no_supported_soft   30
ask_missing_budget           20
```

Validation 覆盖全部四种 Action：RETRIEVE 16、RERANK 15、FINALIZE 14、ASK 8。
Split 按 `group_id = scenario:keyword` 隔离，train/val 无同名 group。

## 4. 数据来源

- 用户请求：**synthetic_shopping_scenario**，不是真实用户行为；
- 商品身份/标题/价格/weight/color/material：真实 Amazon Reviews 2023 catalogue metadata；
- label：rule + tool semantics；
- `qwen_output_used = false`；
- 不含 final candidate set、Gold ASIN、future observation、evaluator label。

## 5. 代表性样本与标签依据

1. `retrieve_initial_clear-000`
   - query: “I need a water filter under $20 for backpacking.”
   - state: no candidates, price_max=20, allowed=[ASK, RETRIEVE]
   - target: **RETRIEVE** (`{"query": ...}`)
   - basis: 无候选且请求已足够明确。

2. `retrieve_after_empty-000`
   - query: “Try again for a water filter under $20 for backpacking.”
   - last_tool: `run_shopping_turn` 返回 0
   - target: **RETRIEVE**
   - basis: 上次检索为空，但请求仍明确，应重试检索而非询问。

3. `ask_ambiguous-002`
   - query: “Can you help me pick something for family camping? I have not decided what I need.”
   - state: no candidates, `needs_user_input=true`
   - target: **ASK** (`{"question": ...}`)
   - basis: 商品类型和预算缺失。

4. `rerank_soft_weight-000`
   - query: “I need a water filter under $20 for backpacking; lighter is better.”
   - 5 个真实候选，含 weight 证据，未 rerank
   - target: **RERANK**
   - basis: soft weight 偏好 + 有候选 + 尚未经过 M6.1。

5. `rerank_soft_feature-000`
   - query: “I need a water filter under $20 for backpacking; I prefer black.”
   - 5 个候选，含 color 证据
   - target: **RERANK**
   - basis: soft color 偏好 + 候选证据可支持重排。

6. `finalize_after_rerank-000`
   - query: “Show me the best water filter under $20 for backpacking.”
   - 5 个候选，`reranked=true`
   - target: **FINALIZE**
   - basis: 已重排的已校验候选，不应再次检索。

7. `finalize_no_supported_soft-001`
   - query: “I need a cookware under $25 for car camping; I care about noise level.”
   - 5 个候选，soft `noise_level` 无 catalogue 证据，未 rerank
   - target: **FINALIZE**
   - basis: M6.1 无可消费证据，保留已校验候选集。

8. `ask_missing_budget-000`
   - query: “I need a water filter for backpacking, but I do not know what budget is reasonable.”
   - no candidates, `needs_human_budget`
   - target: **ASK**
   - basis: 关键预算缺失，先问用户。

## 6. 验证

命令：

```text
python -m pytest tests/test_planner_sft_v0.py tests/test_planner_contract.py -q
14 passed
```

覆盖：

- 320 条样本、sample_id 唯一；
- train/val group 不交叉；
- val 覆盖全部四种 Action；
- 每条 `target.action ∈ allowed_actions` 且通过 `validate_action`；
- RERANK/FINALIZE 有候选与 evidence；
- ASK 有 `needs_user_input` 且无候选；
- 无 future/gold 字段；
- chat 样本的 assistant 是 Planner Action，不是 Preference Operations。

数据质量附加检查：

```text
price_max 约束下候选价格越界数 = 0
```

## 7. 训练前阻塞项

1. **chat 文件只是 messages，不是 tokenizer IDs**：后续训练脚本需套 Qwen chat template、
   仅对 assistant 计算 loss、确定 max length / packing。
2. **标签来自规则语义，不是真实工具执行反馈**：尚未逐样本调用真实
   `ServicePlannerExecutor` 验证 target 可执行。
3. **候选 evidence 不是实时 M6.1 evidence packet**：来自 catalogue metadata。
4. **数据规模仍偏小且请求为模板化合成**；需真实 Planner 轨迹补充，不能把 320 条当作
   充分 SFT 数据。
5. **没有 frozen Planner 评测集**；train/val 只做开发验证，不可当 Benchmark。
6. **未训练**：本轮仅数据；未消耗 GPU 训练。

## 8. 交付

- `data/planner_sft_v0/`（320 条 + chat + manifest + schema + README）
- `training/planner_sft/build_v0.py`
- `tests/test_planner_sft_v0.py`
- `P4A_PLANNER_SFT_V0_SOURCE_DIFF.txt`

未启动训练、未改 Gold/Evaluator、未运行完整 Benchmark、未 commit/push/deploy。
