# P3D 可选 Qwen Planner 接入报告

## 1. 复用调查

调查/复用顺序：

1. 项目已有 `recommendation/demo/planner_contract.py`、`planner_service_executor.py`；
2. 项目已有 LangGraph（已安装、已在 Agent loop/AgentGraph 使用），但没有为 Planner
   新引入任何框架；
3. 已有 `DemoAgentService.run_shopping_turn`、M5、R1、M6.1、CandidateLedger；
4. 现有 HTTP 客户端实现为 `training/agent_sft/http_proposer.py`，但那是 Preference
   Operation 协议，不能直接作为 Planner Action 协议复用；因此新增独立适配层。

结论：采用“现有 Planner Contract + 一个独立 HTTP Action Proposer + 标准库 urllib”
的最小方案。未引入新第三方依赖，未复制外部代码。

## 2. 新增/修改代码

新增：

- `recommendation/demo/planner_http_proposer.py`
  - 显式 opt-in：
    - `AGENTRECX_QWEN_PLANNER=on`
    - `AGENTRECX_QWEN_PLANNER_URL`
  - 请求包含：M5 Effective State 摘要、候选摘要、allowed_actions、上一轮工具观察、
    remaining_steps；
  - 严格解析 `action / arguments / rationale`；
  - 拒绝 Preference Operation JSON（含 `operations` 字段）；
  - 未知 action、非法参数、服务不可用均抛错，**无规则 fallback**。

修改：

- `recommendation/demo/planner_contract.py`
  - `PlannerObservation` 增加 `m5_state / allowed_actions / last_tool_observation / remaining_steps`；
  - `run_planner` 每步填入 allowed actions、剩余步数、上一轮 tool observation；
  - 增加 `validate_action_arguments`（ASK question、RETRIEVE query）；
  - 非法 action 不在 allowed 中时拒绝；
- `recommendation/demo/planner_service_executor.py`
  - Observation 附加 M5 摘要（只读）。

未修改默认 Rule Agent、Serving V1、Preference Parser、Gold/Evaluator。

## 3. CPU Stub HTTP 轨迹

文件：`P3D_HTTP_PLANNER_TRACE.jsonl`  
明确标记：`STUB_HTTP_PLANNER_NOT_QWEN`，不是真实 Qwen。

轨迹：

| Step | Action | Tool | Observation |
|---:|---|---|---|
| 1 | RETRIEVE | `run_shopping_turn` | 真实候选 |
| 2 | RERANK | M6.1 `_m5_soft_rerank` | 候选 1 |
| 3 | FINALIZE | `finalize` | 终止 |

M5 soft 诊断：

```text
active_soft_preference_count=2
consumed_soft_preference_count=1
soft_preference_source=m5_for_ranker
rerank_applied=false
ranking_reason=m6_1_rerank_not_applicable
candidate_count=1
```

结论：M6.1 被调用且消费了 1 个 soft preference；但由于最终候选只有 1 个，无法产生排序
变化，所以 `rerank_applied=false`、`ranking_changed=false`。这是如实记录，不声称排序改善。

## 4. RETRIEVE 与 M5 幂等性

`ServicePlannerExecutor`：

- 每次 RETRIEVE 使用 `turn_prefix + 递增 `_turn` 生成新的 server turn id；
- 不重用同一 turn id；
- 本轮 trace 只有一次 RETRIEVE，M5 event_count=6；
- 每次调用仍走现有 `run_shopping_turn` 的 M5.2 turn identity/idempotency 机制。

未额外构造重复 RETRIEVE 场景，但 executor 不会复用轮次 id；如需 API 级重复提交验证，
应在后续有完整服务环境时补一个同 idempotency_key 的测试。

## 5. 测试

环境：

```text
C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env
```

测试：

```text
tests/test_planner_contract.py
- Stub planner RETRIEVE→RERANK→FINALIZE
- finalize empty rejected
- ASK need signal rejected
- max steps
- HTTP parser accepts action contract
- rejects Preference Operation schema
- rejects unknown action
- RETRIEVE missing query rejected
- HTTP unavailable raises planner_unavailable

9 passed
```

另：

```text
tests/test_p3c_final_merge.py + tests/test_p3c_budget_cross_layer.py
4 passed
```

## 6. 外部依赖

未引入新第三方依赖：

- HTTP 客户端：Python 标准库 `urllib`；
- LangGraph 为项目已有依赖，未新增/升级；
- 未引入多个 Agent 框架。

## 7. 未完成 / 限制

- 未运行真实 Qwen Planner（无 GPU）；
- RERANK 场景 candidate_count=1，M6.1 消费但无法产生排序变化；
- 未接入默认 API，Planner 必须显式 opt-in；
- 未训练 SFT/RL，未运行完整 Benchmark。

## 8. 交付

- `P3D_QWEN_PLANNER_REPORT.md`
- `P3D_HTTP_PLANNER_TRACE.jsonl`
- `recommendation/demo/planner_http_proposer.py`
- 修改：`planner_contract.py`、`planner_service_executor.py`
- 测试：`tests/test_planner_contract.py`

未 commit/push/deploy。
