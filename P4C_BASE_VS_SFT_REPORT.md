# P4C Base vs Planner SFT-v0 最小真实 Agent 对照

## 1. 运行条件

| 项目 | Base | SFT-v0 |
|---|---|---|
| 模型 | Qwen3.5-9B Base | Qwen3.5-9B + Planner LoRA-v0 |
| Adapter | **None** | `training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter` |
| 服务 | `/plan`, port 8009 | `/plan`, port 8008 |
| 解码 | greedy, max_new_tokens=256 | greedy, max_new_tokens=256 |
| 工具 | 同一个 `DemoAgentService` | 同一个 `DemoAgentService` |
| 场景 | 4 个新开发场景 | 相同 4 个场景 |
| 每场景状态 | 独立 session/task id | 独立 session/task id |

工具与环境相同：M5 `apply_constraints`、R1 dynamic on、M5 soft rerank on、
M7 grounded actions on、candidate ledger 不变。4 个场景均未进入 P4A train/val。

## 2. 8 条真实轨迹

证据：`P4C_BASE_VS_SFT_TRACES.jsonl`

| 模型 | 场景 | 实际 Action 序列 | 终止 | Final 候选数 | 非法 RERANK |
|---|---|---|---|---:|---:|
| Base | ask_novel | ASK | ASK | 0 | 0 |
| Base | retrieve_clear | RETRIEVE → RERANK → FINALIZE | FINALIZE | 2 | 1 |
| Base | avoid_invalid_rerank | RETRIEVE → RERANK → FINALIZE | FINALIZE | 1 | 1 |
| Base | effective_rerank | RETRIEVE → RERANK → FINALIZE | FINALIZE | 2 | 1 |
| SFT-v0 | ask_novel | ASK | ASK | 0 | 0 |
| SFT-v0 | retrieve_clear | RETRIEVE → RERANK → FINALIZE | FINALIZE | 5 | 1 |
| SFT-v0 | avoid_invalid_rerank | RETRIEVE → RERANK → FINALIZE | FINALIZE | 5 | 1 |
| SFT-v0 | effective_rerank | RETRIEVE → RERANK → FINALIZE | FINALIZE | 5 | 1 |

所有轨迹：Action JSON schema/Validator **合法**；无模型调用错误；无未终止。
但“合法”不等于“合理”，也不等于任务成功。

## 3. 逐场景事实

### ASK 场景

Query:

```text
We are setting up a cabin and need some gear, but I have not decided what kind of product or budget.
```

- Base: **ASK**
  - question: “Could you please specify the type of cabin gear you are looking for (e.g., sleeping bags, tents, cookware) and your approximate budget?”
- SFT-v0: **ASK**
  - question: “Which product type, trip style, and budget should I use?”

结论：两者都正确避免无信息检索；SFT 的问题明显复述训练模板，没有针对 cabin 场景。

### 清晰 RETRIEVE 场景

Query:

```text
Find me a rechargeable headlamp for night hiking under $45.
```

- Base query: `rechargeable headlamp night hiking under $45` → 2 candidates
- SFT query: 完整原句 → 5 candidates

两者都 RETRIEVE，但 SFT 直接复制用户原句，Base 做了简短改写；候选数不同，
不能据此声称哪一方检索更正确。

### 避免无效 RERANK 场景

Query:

```text
I need a water filter under $50 for hiking.
```

无明确 soft preference。

- Base: RETRIEVE → **RERANK** → FINALIZE；M6.1 `rerank_applied=false`
- SFT-v0: RETRIEVE → **RERANK** → FINALIZE；M6.1 `rerank_applied=false`

两者都进行了无效 RERANK，均未避免。

### 有效偏好重排场景

Query:

```text
I need a water filter under $35 for backpacking; lighter weight is important.
```

- Base: RETRIEVE → RERANK → FINALIZE；`consumed_soft_preference_count=0`，
  reason=`active_m5_soft_preference_not_supported_by_m6_1`
- SFT-v0: 同样三步；同样的 `rerank_applied=false`

因此本场景**没有形成有效 M6.1 重排**，只能证明两者都调用了 RERANK；不能证明
偏好被有效消费或排序改善。

## 4. Base vs SFT 结论

### 改善

没有可确认的改善。

### 退步

没有在 Action 合法性、终止或硬约束执行上出现退步。

### 没有差异

- ASK 场景：两者都 ASK；
- 无效 RERANK：两者都未避免；
- 有效偏好重排：两者都调用 RERANK，但 M6.1 均未应用。

### 可疑差异（不是质量结论）

SFT 的 RETRIEVE query 原样复制用户输入，导致本轮候选数 5；Base 做了缩写，候选数 2。
最终任务结果均标记 **UNKNOWN**，因此不能把候选数更多当作更好。

## 5. 任务成功判定

全部 8 条：**UNKNOWN**。

原因：

- 只验证了 Planner Action 与工具调用；
- 未定义该 4 个新场景的 Gold final item；
- 无法确认候选是否真正满足用户需求；
- 不应以 rationale 或候选数量替代任务成功。

硬约束：

- 6 条进入 RETRIEVE 的轨迹均 `hard_constraint_evaluated=true`；
- 未见跳过硬约束；
- 但最终候选资格仍需独立验证，不能直接判任务成功。

## 6. 实际暴露的问题

1. **SFT 没有学会“何时不 RERANK”**：没有受支持 soft evidence 时仍固定选择 RERANK。
2. **SFT 复述训练模板**：ASK question 是训练集统一模板，没有根据 cabin 场景变化。
3. **有效 soft preference 证据不足**：water filter 候选没有 M6.1 可消费的 weight evidence，
   RERANK 调用有效但无效果。
4. **开发场景仍太接近训练模板**：虽然 query 未在 train/val，但结构和动作模式与训练场景相似，
   SFT 的 53/53 val 不能外推。
5. **Base 与 SFT 的检索 query 行为不同**：SFT 复制原句，Base 缩写；需要更多样本才能判断影响。

## 7. 下一版数据最值得改进的 2 点

1. **增加 evidence-aware RERANK 负例**：候选存在但无支持 soft evidence 时，标签应为
   FINALIZE；同时增加“有证据才 RERANK”的正例，训练模型读取 M6.1 evidence，而不是
   见到候选就 RERANK。
2. **去模板化 + 新场景泛化**：增加不同商品类别、不同 ASK 问法、不同 retrieval query
   改写、以及多步真实 Observation；不要复用统一 ASK question 和 rationale。

## 8. 交付

- `P4C_BASE_VS_SFT_TRACES.jsonl`
- `P4C_BASE_VS_SFT_SUMMARY.json`
- 远端 Adapter 路径：
  `training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter`

未重新训练、未扩充数据、未运行完整 Benchmark、未修改 Gold/Evaluator/Serving V1、
未 commit/push/deploy。GPU 与两个 `/plan` 服务保持运行。
