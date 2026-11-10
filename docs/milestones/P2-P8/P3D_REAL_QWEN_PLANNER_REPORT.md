# P3D Real Qwen Planner GPU Acceptance Report

## 1. GPU / 模型加载

```text
GPU: NVIDIA RTX 6000D
GPU memory: 85,651 MiB total
Model: /root/autodl-tmp/models/Qwen3.5-9B-ms
Model ID: Qwen/Qwen3.5-9B
Class: Qwen3_5ForCausalLM
Adapter: None  ← Base model only, no Preference Parser LoRA
```

Planner server 启动命令：

```bash
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 8008
```

加载完成：`proposal server on 127.0.0.1:8008 adapter=None`  
GPU 占用约 17.97 GiB。

## 2. 真实多步轨迹 PASS

证据：`P3D_REAL_QWEN_PLANNER_TRACE.jsonl`  
模型：真实 Qwen3.5-9B Base，不是 Stub。

### 用户需求

```text
I need a water filter under $30 and I prefer the lighter option.
```

### Step 1

- Observation: 无候选，allowed_actions=[ASK, RETRIEVE]
- 模型原始输出：

```json
{"action":"RETRIEVE","arguments":{"query":"water filter under $30 lighter option"},"rationale":"..."}
```

- 解析 Action: RETRIEVE
- Validator: PASS
- Tool: `run_shopping_turn`，真实返回 5 个候选
- latency: 3109 ms；input 362 / output 76 tokens

### Step 2

- 新 Observation: 5 候选，reranked=false
- 模型原始输出：

```json
{"action":"RERANK","arguments":{},"rationale":"We have 5 candidates ..."}
```

- 解析 Action: RERANK
- Validator: PASS
- Tool: M6.1 `_m5_soft_rerank`
- M6.1 结果：consumed_soft_preference_count=0，
  `active_m5_soft_preference_not_supported_by_m6_1`，排序未变化
- latency: 2703 ms；input 678 / output 66 tokens

### Step 3

- 新 Observation: 5 候选，reranked=true，remaining_steps=1
- 模型原始输出：

```json
{"action":"FINALIZE","arguments":{},"rationale":"The observation shows 5 candidates ..."}
```

- 解析 Action: FINALIZE
- Validator: PASS
- Tool: `finalize`
- latency: 2887 ms；input 706 / output 93 tokens

### 终止

```text
termination = FINALIZE
final_ids = [
  B09WD13RLR, B0BX5QFWQN, B09LW2KHPM, B016HM7QRE, B091VYKC96
]
```

M5:

```text
mode=apply_constraints
hard_constraint_count=1 (price_max=30)
state_digest=5446024959f5a0eab3203e5a857b6128160288bec043d030563effb3e735dea7
```

## 3. 对照轨迹 PASS（模型动作不同）

证据：`P3D_REAL_QWEN_CONTROL_TRACE.jsonl`

用户需求：

```text
I do not know which camping gear I need; help me choose.
```

模型真实输出：

```json
{"action":"ASK","arguments":{"question":"What type of camping trip are you planning ...?"},"rationale":"..."}
```

结果：

```text
termination=ASK
final_ids=[]
```

与首条轨迹（RETRIEVE→RERANK→FINALIZE）动作不同，说明模型确实读取了不同 Observation，
而不是固定序列。

该对照暴露并修复了一个真实接口问题：ASK 动作现在被标记为 terminal（awaiting user），
不会再被 Planner Loop 重复执行到 max_steps。

## 4. 实际源码修改

- `recommendation/demo/planner_http_proposer.py`
  - 解包 model_server 的 envelope `{payload,text,...}`；
  - 修复前把 envelope 当 Action，导致 `unknown planner action ''`；
  - 记录 `last_raw` / `last_response`。
- `recommendation/demo/planner_service_executor.py`
  - ASK 返回 terminal=True + needs_user_input=True，正确结束等待用户。
- `training/agent_sft/model_server.py`
  - 独立 `/plan` 路由；
  - `_parse_planner_text` 严格拒绝 Preference Operation。
- `training/agent_sft/planner_prompt.py`
  - 独立 Planner Prompt。

Diff：`P3D_REAL_QWEN_PLANNER_SOURCE_DIFF.txt`

## 5. 测试

```text
tests/test_planner_contract.py
11 passed
```

覆盖 Action Contract、HTTP 协议、Preference schema 拒绝、未知 action、模型不可用无 fallback。

## 6. 结论与限制

PASS：

- 真实 Qwen3.5-9B Base 加载成功；
- 真实模型生成 Planner Action；
- Validator 与真实工具执行贯通；
- 后续 Action 使用上一步真实 Observation；
- 不同 Observation 触发不同动作（ASK vs RETRIEVE/RERANK/FINALIZE）；
- 无 Stub、无固定动作序列。

限制：

- RERANK 被模型选择，但 M6.1 未应用（当前候选无受支持 soft evidence）；这是真实行为，
  不能声称重排生效；
- 只运行了两条最小轨迹，不是完整 Benchmark；
- 未训练 Planner SFT/RL；
- 未修改默认 Rule Agent / Serving V1 / Gold / Evaluator；
- 未 commit/push/deploy。

## 交付

- `P3D_REAL_QWEN_PLANNER_TRACE.jsonl`（真实 Qwen）
- `P3D_REAL_QWEN_CONTROL_TRACE.jsonl`（真实 Qwen 对照）
- `P3D_REAL_QWEN_PLANNER_SOURCE_DIFF.txt`
- `P3D_QWEN_PROMPT_PLANNER_REPORT.md`（Prompt/接口说明）
