# P3D Qwen Prompt-based Planner Report

## 1. 调查与复用

- 复用现有 `DemoAgentService.run_shopping_turn`、M5、R1、M6.1、CandidateLedger；
- 复用 `planner_contract.py` 的 Action / Observation / Validator；
- 复用 `planner_service_executor.py` 的真实工具执行；
- 复用 `training/agent_sft/modeling.py` 的 Qwen 加载逻辑；
- 不引入新 Agent 框架、不引入新第三方依赖。

现有 Preference Parser 协议（`/propose`、Preference Operation JSON）保持独立，没有把
operations 当作 Planner Action。

## 2. 新增/修改

### Planner Prompt

新增 `training/agent_sft/planner_prompt.py`：

- 独立 system prompt：只允许 `ASK / RETRIEVE / RERANK / FINALIZE`；
- 严格输出：

```json
{"action":"RETRIEVE","arguments":{"query":"..."},"rationale":"..."}
```

- 输入包含：
  - 当前 query；
  - M5 state 摘要；
  - candidate_ids / candidate_count / state_version；
  - allowed_actions；
  - last_tool_observation；
  - remaining_steps；
  - needs_user_input / reranked。
- 明确禁止 Preference Operations、operations、ASIN、URL。

### Model Server

修改 `training/agent_sft/model_server.py`：

- 保留 `/propose`（Preference Parser）；
- 新增独立 `/plan`：
  - 使用 `planner_prompt.planner_prompt`；
  - 返回 `text / payload / errors / latency_ms / input_tokens / output_tokens`；
  - `_parse_planner_text` 只接受 Planner Action 协议，拒绝 `operations`；
- Planner 使用 Base 模型时**不加载 Preference Parser LoRA**:
  启动时不要传 `--adapter`；传了 adapter 的进程不应作为 Planner baseline。

### HTTP Proposer

`recommendation/demo/planner_http_proposer.py`：

- 显式 opt-in：
  - `AGENTRECX_QWEN_PLANNER=on`
  - `AGENTRECX_QWEN_PLANNER_URL`
- POST `/plan`，每步根据最新 Observation 请求；
- 解析 action/arguments/rationale；
- 拒绝 Preference Operation schema、未知 action、非法参数；
- 模型不可用/JSON 无效均抛错，无 Rule fallback。

## 3. CPU 协议验证

测试：`tests/test_planner_contract.py`（新增 prompt/HTTP 测试，未新增重复文件）。

结果：

```text
11 passed
```

覆盖：

- Planner 输入包含 `m5_state / allowed_actions / remaining_steps / RERANK`；
- HTTP `/plan` 请求-响应 round trip；
- Preference Operation JSON 被拒绝；
- 未知 action 被拒绝；
- RETRIEVE 缺 query 被拒绝；
- 模型不可用抛 `planner_unavailable`，无 fallback；
- 原有 Stub Planner RETRIEVE→RERANK→FINALIZE 序列仍通过。

CPU Stub HTTP Planner 轨迹仍来自现有
`P3D_HTTP_PLANNER_TRACE.jsonl`，明确标记 `STUB_HTTP_PLANNER_NOT_QWEN`；不是 Qwen 输出。

## 4. 真实 Qwen 状态

远端检查：

```text
SFT-v2 adapter 文件存在
base model config.json 存在
GPU_UNAVAILABLE（nvidia-smi 不可用）
```

因此：

- 未加载真实 Qwen3.5-9B Base；
- 未产生真实 Planner Action；
- 未验证模型是否能根据不同 Observation 改变动作；
- 真实模型验收标记 **BLOCKED_NO_GPU**，没有伪造轨迹。

## 5. 未验证 / 限制

1. 真实 Qwen Prompt Planner 输出质量、动作合法率和延迟未验证；
2. 每步重新请求模型的真实多步轨迹未运行；
3. 当前 Stub Planner 是固定策略，不代表模型自主规划；
4. 未接入默认 Rule Agent 或 `/v1/recommend`；
5. 未训练 Planner SFT/RL，未运行完整 Benchmark；
6. 未修改 Serving V1/Gold/Evaluator；未 commit/push/deploy。

## 6. 有 GPU 时的最小命令

```bash
# Base model only; do NOT pass --adapter for the Planner baseline
python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --port 8008
```

客户端：

```bash
AGENTRECX_QWEN_PLANNER=on \
AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18008 \
python <planner-cpu-trace-driver>
```

## 7. 交付

- `P3D_QWEN_PROMPT_PLANNER_REPORT.md`
- `P3D_QWEN_PROMPT_PLANNER_SOURCE_DIFF.txt`
- `training/agent_sft/planner_prompt.py`
- `training/agent_sft/model_server.py`（`/plan`）
- `recommendation/demo/planner_http_proposer.py`
- `tests/test_planner_contract.py`（11 passed）
- CPU Stub 轨迹：`P3D_HTTP_PLANNER_TRACE.jsonl`（非 Qwen）
