# P3C-2 Qwen 可选 Agent API 集成报告

## 1. 完成情况

本阶段完成了代码层集成与失败路径实现，但 **AutoDL 当前实例无可用 GPU**，因此无法真实启动
Qwen3.5-9B model_server，也无法完成“第一轮预算+重量、第二轮改预算+撤回重量”的真实
Qwen API 多轮验证。该部分明确标记为 **BLOCKED_NO_GPU**，没有伪造 Qwen 轨迹。

实例检查：

- 实例可 SSH，代码/模型/SFT-v2 adapter 持久化目录仍在；
- `nvidia-smi` 无 GPU 输出；
- 因此未启动 `model_server.py`，未执行真实 Qwen 推理。

## 2. 代码修改

### 2.1 Qwen 可选 API 模式

新增：

- `recommendation/demo/qwen_proposal.py`
  - `qwen_proposal_enabled()`：只有 `AGENTRECX_QWEN_PROPOSAL=on` 才启用；
  - `QwenProposalClient.from_env()`：要求 `AGENTRECX_QWEN_PROPOSAL_URL`；
  - `propose(messages, current)`：调用 model_server `/propose`；
  - 模型不可用/无 payload 抛 `QwenProposalError`，**不回退规则 proposal**。

修改：

- `recommendation/api/demo_routes.py::agent_shopping`
  - 只有 `qwen_proposal_enabled()` 为真时才构造 QwenProposalClient；
  - proposal 通过既有 `run_shopping_turn(..., proposal_operations=...)` 注入；
  - 模型错误映射为 `503 qwen_proposal_unavailable`/`qwen_proposal_invalid_*`；
  - 默认路径 `proposal_operations=None`，行为不变。

默认 `/v1/recommend` 没有被修改。

### 2.2 候选链路诊断

修改 `recommendation/demo/agent_service.py`：

- 新增 `_candidate_filter_diagnostics`；
- 在 `_execute_plan` 角色筛选后记录：
  - `raw_recall_count`：当前可见的 grounded ledger identity 数；
  - `rrf_candidate_count` / `grounded_candidate_count`：presentable candidates；
  - role matched/mismatch/unknown 及 rejection reasons；
  - hard constraint evaluated/eligible/ineligible/unresolved；
  - final candidate count / IDs；
- 新增 `candidate_filter_diagnostics(session_id)` 只读访问器。

限制：本次受控运行发生在该 instrumentation 之前，因此
`P3C2_CANDIDATE_FILTER_TRACE.csv` 中部分列为
`UNKNOWN_NOT_INSTRUMENTED_BEFORE_RUN`，需要下一次真实执行重新生成；没有用推测值填充。

## 3. 测试

新增 `tests/test_p3c2_qwen_api.py`：

- Qwen 默认关闭；
- 启用但缺 URL 被拒绝；
- URL 不可达时抛 `qwen_proposal_unavailable`，不 fallback。

与 P3C-1 测试合计：

```text
6 passed
```

## 4. 真实验证状态

### 未完成

- Qwen SFT-v2 真实 model_server 启动；
- 真实两轮 Shopping API 轨迹；
- Qwen proposal 经 API 写入 M5 的端到端证据；
- 候选细粒度 role/hard 计数真实运行。

原因：实例无 GPU，不能加载 9B 模型。已保留：

- 远端 SFT-v2 adapter；
- 本地 adapter 备份：
  `training/agent_sft/output/qwen3_5_9b_lora_v2_adapter.tar.gz`；
- 模型文件在 `/root/autodl-tmp/models/Qwen3.5-9B-ms`。

### 已实现但未运行

- `AGENTRECX_QWEN_PROPOSAL=on` + URL 的 API 路径；
- 模型不可用时 503，禁止 fallback；
- 候选过滤 trace 代码。

## 5. 候选归零问题

P3C-1 受控结果仍然显示：

```text
accessory_to_main:
  grounded_candidate_pool = 10
  final_ids = []
```

当前只能确认发生在 **grounded pool 之后、final 之前**。新代码会记录 role/hard 计数，但
这次没有 GPU/新受控运行来生成这些计数，故不声称具体阶段；也没有放宽角色过滤或使用
Gold ID。

## 6. 回答

1. **Qwen SFT-v2 是否真正进入可选 Agent API？**
   - 代码路径已实现；默认关闭；当前实例无 GPU，未做真实 Qwen API 运行。因此只能回答
     “已实现、未运行验证”，不能回答“已真实接入并验证”。
2. **它是否通过 API 实际更新 M5？**
   - 未验证；没有真实 Qwen 多轮轨迹。不能声称已更新。
3. **配件 → 主商品候选在哪一步归零？**
   - 已知 grounded=10、final=0；具体 role/hard 阶段当前仍为
     `UNKNOWN_BETWEEN_GROUNDED_AND_FINAL`，新增诊断可下一次运行定位。
4. **默认 Serving V1 是否保持不变？**
   - 是；未修改 `/v1/recommend` 或 `serving_v1`。默认 Agent API 也未启用 Qwen。
5. **当前架构是否可以进一步引入 LLM Planner？**
   - 不具备证据。Qwen 目前只是 proposal parser；planner 动作空间/奖励/环境不完整，
     且候选归零与 M6.1 消费问题未解决；不应先做 LLM Planner/RL。

## 7. 交付

- `P3C2_QWEN_API_INTEGRATION_REPORT.md`
- `P3C2_CANDIDATE_FILTER_TRACE.csv`
- `P3C2_AGENT_API_TRAJECTORY.jsonl`（明确 BLOCKED，不是成功轨迹）
- 源码：`recommendation/demo/qwen_proposal.py`、`recommendation/api/demo_routes.py`、
  `recommendation/demo/agent_service.py`
- 测试：`tests/test_p3c2_qwen_api.py`

未重新训练、未修改 Gold/Evaluator/frozen/Serving V1，未 commit/push/deploy。
