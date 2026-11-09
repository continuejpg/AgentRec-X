# P4D Adaptive Retrieval Planner Report

## 1. 实现范围

复用现有 Planner Contract / HTTP Proposer / Validator / ServicePlannerExecutor / R1 /
CandidateLedger / Eligibility，只补充 Planner 可见的检索反馈字段：

- `last_retrieve_query`
- `retrieval_feedback`：
  - raw/RRF/grounded counts
  - role_match / role_mismatch / role_unknown
  - role_rejection_reasons
  - hard eligible / ineligible / unresolved
  - final_candidate_count
- `role_target`
- `eligibility`（active constraints、price_max、合格数）
- `candidate_evidence`（当前候选标题/价格/weight）

未新增 Action 类型；`RETRIEVE` 仍通过 `ServicePlannerExecutor` 调用现有
`run_shopping_turn`；Validator 仍拒绝非法 Action；R1/Eligibility/role 逻辑未改写。

源码 diff：`P4D_ADAPTIVE_RETRIEVAL_SOURCE_DIFF.txt`

## 2. 真实轨迹

模型：**Qwen3.5-9B Base**，无 Adapter，`/plan` 真实推理，`max_new_tokens=512`。
每条轨迹使用独立 session/task，M5 状态隔离。

证据：`P4D_ADAPTIVE_RETRIEVAL_TRACES.jsonl`

| 场景 | Action 序列 | 终止 | Final | 结果 |
|---|---|---|---|---|
| retry_after_role_feedback | RETRIEVE ×3 → FINALIZE | **未终止** | — | **FAIL**：重复相同 query，最后 FINALIZE 无候选被 Validator 拒绝 |
| retry_after_vague_retrieval | RETRIEVE → RERANK → FINALIZE | FINALIZE | 1 | 完成，但未再检索 |
| finalize_after_eligible | RETRIEVE → RERANK → FINALIZE | FINALIZE | 1 | 完成，但有一次不必要 RERANK |
| strict_budget_no_eligible | ASK | ASK | 0 | 未检索，保留硬约束，等待用户 |

## 3. Observation → Action 证据

### 场景 1：角色/反馈后重检索

- 初始 query: `I need a replacement filter cartridge for my water filter under $20.`
- 第 1 次 RETRIEVE query: `water filter replacement cartridge under $20`
- 工具反馈（第 2 步可见）：

```text
raw_recall_count=10
rrf_candidate_count=3
grounded_candidate_count=3
hard eligible=3, ineligible=9, unresolved=3
final_candidate_count=3
```

但 `candidate_ids=[]`（当前 API 最终未展示候选），模型没有改写 query：

```text
第 2 次 RETRIEVE: 完全相同 query
第 3 次 RETRIEVE: 完全相同 query
之后 FINALIZE（无候选）→ PlannerContractError: FINALIZE is not allowed now
```

结论：**没有形成有效的自适应 Query 调整**；这是模型行为失败，不是工具伪造成败。

### 场景 2：模糊需求

- Initial query: `I need something to make stream water safe to drink on a hike under $25.`
- 第 1 次 RETRIEVE: `hiking water filter under $25`
- 反馈：grounded=5，hard eligible=5，ineligible=8，unresolved=2，final_candidate_count=5
- 下一步模型选择 **RERANK**，随后 FINALIZE，final=B08NGGJMD6。
- M6.1 `rerank_applied=false`（无支持 soft evidence）。

未发生第二次 RETRIEVE；模型没有尝试针对“饮水安全”改写 query。

### 场景 3：已有合格候选

- Query: `I need a water filter under $50 for camping.`
- RETRIEVE → RERANK → FINALIZE
- 反馈：grounded=9，eligible=9，ineligible=3，unresolved=3，final_candidate_count=5
- 最终 1 个 API 展示候选 B08NGGJMD6。
- 模型仍先做了一次 RERANK；M6.1 未实际应用。

### 场景 4：严格预算

- Query: `I need a water filter under $1.`
- 模型直接 **ASK**：

```text
A water filter under $1 is not feasible. Would you like to adjust your budget to a realistic amount, such as under $50 or $100?
```

- 没有 RETRIEVE；没有生成绕过预算的 query；没有 final。
- 硬约束未被改写，符合“找不到合格商品时严格保留硬约束”的期望，但任务结果 UNKNOWN。

## 4. 检索改进 / 失败 / 不必要调用

### 改进

没有可确认的检索改进。模型没有根据反馈生成新的 RETRIEVE query。

### 失败

1. **重复相同 query**：场景 1 连续三次相同 RETRIEVE，最终无候选。
2. **失败后错误 FINALIZE**：无候选时仍尝试 FINALIZE，被 Validator 正确拒绝。
3. **无效 RERANK**：场景 2/3 都调用 M6.1，但 `rerank_applied=false`。
4. **过度依赖 ASK / RERANK**：没有将“检索到 1 个候选”转化为合理的 FINALIZE 决策。

### 不必要调用

- 场景 2、3 在只有 1 个最终候选时仍调用 RERANK；
- 场景 1 三次 RETRIEVE query 完全不变，没有产生新的 Observation 价值。

## 5. 硬约束与任务成功

- 场景 2、3 的 M5 hard constraint 分别保留 `price_max=25`、`price_max=50`；
- 场景 4 没有绕过 `$1` 约束，直接 ASK；
- 场景 1 没有产生 final，无法确认任务成功；
- 所有场景任务结果：**UNKNOWN**；仅验证了 Planner Action 和工具调用。

## 6. 后续 Planner SFT-v1 真实样本

已提取 11 条原始行为样本：

`P4D_SFT_V1_CANDIDATE_SAMPLES.jsonl`

每条包含：

- 真实 user request / observation；
- 真实模型 Action + arguments + raw；
- 工具反馈；
- `curation_status="needs_human_label"`；
- `do_not_use_as_positive_label=true`。

这些是行为轨迹，不是自动正标签；下一版数据应先人工/规则标注“应重检索”与“应终止”。

## 7. 最值得改进的 2 点

1. **Feedback-aware RETRIEVE 标签**：需要针对“候选角色/资格不合格、candidate_ids 为空、
   final_candidate_count 不一致”等反馈构造正负例，训练模型改写 query 或 FINALIZE，而不是
   重复同一 query。
2. **终止条件校准**：候选数=1、无 soft evidence 时应 FINALIZE，而不是 RERANK；
   需要真实“应终止”负例，避免模型把 RETRIEVE/RERANK 当成默认序列。

## 8. 交付

- `P4D_ADAPTIVE_RETRIEVAL_TRACES.jsonl`
- `P4D_SFT_V1_CANDIDATE_SAMPLES.jsonl`
- `P4D_ADAPTIVE_RETRIEVAL_SOURCE_DIFF.txt`
- 测试：`tests/test_planner_contract.py` 11 passed

未训练新模型、未扩充数据、未运行完整 Benchmark、未修改 Gold/Evaluator/Serving V1、
未 commit/push/deploy。Base 与 SFT 服务仍运行，GPU 未关闭。
