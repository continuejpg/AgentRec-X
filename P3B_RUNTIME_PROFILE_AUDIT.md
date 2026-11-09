# P3B 运行 Profile 审计

## 1. 实际开关与优先级（源码）

### M5 State Mode

- `recommendation/demo/state_integration.py:203-221`：`resolve_state_mode` 的优先级是
  **显式参数 > `AGENTRECX_STATE_MODE` > `off`**；
- `state_integration.py:128`：`DEFAULT_STATE_MODE = "off"`；
- `runtime.py:934-947`：`ShoppingStateIntegration.from_env()` 在 Agent service 组装时调用；
- `agent_service.py:1967`：`_integrate_state_turn` 真正执行 M5 写入/投影。

### R1 Dynamic Retrieval

- `recommendation/demo/retrieval_plan.py:33-38`：`dynamic_retrieval_enabled()` 只读取
  `AGENTRECX_R1_DYNAMIC_RETRIEVAL`，默认 **off**；
- `agent_service.py:1330-1350` 和 `:1463` 只有在 `dynamic_retrieval_enabled()` 为真时走 R1 plan；
- Benchmark runner 的 `FIXED_BENCHMARK_ENV` 在 `benchmarks/agent/runner.py:33-40` 将
  `AGENTRECX_R1_DYNAMIC_RETRIEVAL` 设为 `on`，再由 `_apply_profile`（`:44-49`）写入环境。

### M5 Soft Rerank（M6.1）

- `agent_service.py:433-436`：构造函数参数优先；否则读取
  `AGENTRECX_M5_SOFT_RERANK`，默认 `"off"`；
- `agent_service.py:1848-1916`：`_m5_soft_rerank` 只在 `_soft_m5_rerank` 为真时调用；
- `agent_service.py:2076-2088`：只有 M5 apply 时生成并保存 `for_ranker` projection；
- `agent_service.py:1384-1385`：候选路径上的 soft-M5 重排入口；
- `recommendation/reranking/reranker.py:342`：默认 M6.1 `PreferenceReranker` 本身不读取 M5；
  M5 soft 是**额外的可选重排**，不是默认 reranker 的自动输入。

### Proposal 注入

- `benchmarks/agent/adapters.py:527`：`LLMPreferenceAdapter`；
- `adapters.py:579`：读取 `last_payload`；
- `adapters.py:592-597`：调用 `run_shopping_turn(..., proposal_operations=...)`；
- 默认 API route `recommendation/api/demo_routes.py:792-834` **没有传** proposal；
- `agent_service.py:1069` 接收 proposal；`agent_service.py:2030` 有 proposal 时关闭规则同轮 M5 写入。

## 2. 三个实际运行的 Profile

受控脚本在独立运行时中构建相同的 catalogue、SASRec/Two-Tower、Top-K=5；分别设置：

| Profile | AGENTRECX_STATE_MODE | R1 dynamic | M5 soft rerank | Proposal |
|---|---|---|---|---|
| DEFAULT | 未设置 → off | off | off | 否 |
| RULE_APPLIED | apply_constraints | on | on | 否（规则解析） |
| LLM_APPLIED | apply_constraints | on | on | 是（已校验 operation payload） |

实际观测（`runs/p3b_controlled/controlled_results.json`）：

- DEFAULT：`state_mode=off`、`soft_m5_rerank=False`、`state_diagnostics=None`、`r1_diagnostics=None`；
- RULE_APPLIED：`state_mode=apply_constraints`、`soft_m5_rerank=True`，M5 events 存在，R1 诊断存在；
- LLM_APPLIED：同 RULE，且最终轮 proposal 有效并进入 M5（`proposal_diagnostics.valid`）。

## 3. 结论

1. 默认 Agent API 的 M5、R1、soft-M5 **确实都是关闭的**，不是“设了环境变量就一定生效”能解决的问题；
2. `RULE_APPLIED`/`LLM_APPLIED` 中三个开关在运行时实际生效（有 state/R1 诊断为证）；
3. `LLM_APPLIED` 的 proposal 只在 benchmark D 路径可注入；默认 API 没有 proposal 参数；
4. M5 soft projection 即使生成，也不保证 M6.1 调用（见 controlled 结果中部分 `soft_rerank_diagnostics=None`）。
