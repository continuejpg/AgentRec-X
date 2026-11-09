# P7-A 统一动态任务 Runner 接入 Qwen Planner 报告

## 1. 状态

本轮只做 CPU 接线与静态/最小 Contract 验证，**没有运行 Qwen Base / SFT-v0 真实推理**，GPU 未启动。

- 统一 Runner 已支持 `--arm rule|base|sft`；
- Rule 仍走既有 `DemoAgentService.run_shopping_turn`，P6.4 结果保留、未重跑；
- Base / SFT 走现有 `HttpPlannerProposer` + `PlannerAction` Validator + `ServicePlannerExecutor`，没有第二套 Planner；
- 静态检查确认三档 Arm 配置可解析，Base/SFT 构造出 `HttpPlannerProposer`，不发出网络请求；
- 现有测试：`tests/test_planner_contract.py` + `tests/test_p6_4_role_target_transfer.py` 共 **12 passed**。

## 2. 统一 Runner 用法

入口：`experiments/p6_benchmark/run_dynamic_dev_p64.py`

```powershell
$venv='C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONIOENCODING='utf-8'

# Rule（P6.4 已跑过；保留结果）
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py --arm rule

# Qwen Base（需先有本地 planner endpoint）
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm base --planner-url http://127.0.0.1:18009

# Planner SFT-v0
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm sft --planner-url http://127.0.0.1:18008
```

`--planner-url` 是 **base URL**，proposer 自己追加 `/plan`。`--planner-model` 可选，只作为请求元数据。
Base 与 SFT 的差异仅是 endpoint（以及远端是否加载 adapter），Runner 和执行路径完全相同。

源码 diff：`P7_A_ARM_WIRING_SOURCE_DIFF.txt`

## 3. Arm 切换与共享条件

| 条件 | Rule | Base / SFT-v0 |
|---|---|---|
| 任务文件 | `dynamic_dev_p64.jsonl` 6 Dev | 同 |
| 初始真实 history | 同 5 个 ASIN | 同 |
| M5 状态更新 | runner `state_operations` 每 user turn 应用一次 | 同 |
| hard constraints | trusted `AgentRecommendRequest.constraints` | 同 |
| Serving V1 Fusion | sequential + two_tower + catalog_lexical / RRF K=60 | 同 |
| CandidateLedger / Eligibility | 既有 `DemoAgentService` | 同 |
| k / max_steps / per-turn tool budget | 5 / 6 / 4 | 同 |
| 最终评分 | `_diagnose` + 最终展示列表 | 同 |
| Planner | 无（direct Rule Agent） | `run_planner` + `HttpPlannerProposer` + `ServicePlannerExecutor` |

每档 Arm 独立进程运行；每个 task 独立 Session；不跨 Arm 共享可变 M5/ledger 状态。

新增的 wiring（仅接线，不改 Planner 规则）：

- `ServicePlannerExecutor` 增加只读 wiring 字段：
  - `effective_role_target` / `effective_state`：来自 runner 的同一 user-turn 有效状态；
  - `last_response` / `tool_calls_used`：记录实际 Agent 响应和真实 tool 调用数；
  - `_observation` 的 `role_target` 优先使用 canonical effective role；`m5_state` 携带 effective state。
- `DemoAgentService.prime_task_role_target(task_key,user_key,message)`：在 Planner episode 开始前，
  从**真实用户 turn** 播种角色目标（复用 P6.4 的 `_update_role_target` + session mirror），避免被
  planner 改写后的 RETRIEVE query 改变本 turn 的角色。

## 4. 静态接线验证结果

```text
--arm rule  --check-wiring
  proposer: null
  executor: ServicePlannerExecutor
  contract_runner: run_planner
  validator: validate_action
  network_request_issued: false
  runtime_built: false

--arm base --planner-url http://127.0.0.1:8009 --check-wiring
  proposer: HttpPlannerProposer
  proposer_url: http://127.0.0.1:8009
  network_request_issued: false

--arm sft --planner-url http://127.0.0.1:8008 --check-wiring
  proposer: HttpPlannerProposer
  proposer_url: http://127.0.0.1:8008
  network_request_issued: false
```

没有使用 Stub 输出；没有伪造 Qwen 轨迹。

## 5. 接线阻塞与限制

**接口层没有阻塞**：Base/SFT 已能选择现有 HTTP Proposer、Validator 和 Executor。  
存在一个需要在 GPU 轮验证/最小修复的**公平性接线问题**：

1. `ServicePlannerExecutor` 的 RETRIEVE 会调用
   `run_shopping_turn(payload.model_copy(message=planner_query))`。Hard constraints 和 role target
   已由 runner/canonical state 固定；但 **soft preference（如 “lighter options”）的 M5 抽取**
   仍基于 planner 改写后的 RETRIEVE query，而不是当前用户 turn 原文。  
   本轮没有扩大修改范围；在 Qwen Base/SFT 对照前需要验证，必要时把 canonical soft state 也注入
   M5（或在 executor 中显式携带当前 user-turn 原文）。
2. Planner 的 FINALIZE 返回 candidate IDs；Runner 将其映射到最近一次真实
   `response.recommendations` 对象，若身份不一致会记录
   `planner_identity_mismatch`，不会用工具诊断数量替代展示商品。该映射需要在真实轨迹上确认。
3. 当前没有 GPU / model server endpoint；真实 Qwen 调用未执行，因此 **Base/SFT 的真实指标为 NOT RUN，不是 PASS/FAIL**。

## 6. 下一次启用 GPU 后的准确命令

### 6.1 远端启动 model server

Base（不加载 adapter）：

```bash
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --port 8009
```

SFT-v0：

```bash
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --adapter /root/autodl-tmp/agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter \
  --port 8008
```

启动前先确认 adapter 目录仍存在（P6.4 交接记录的最重要远程风险）。若显存不允许两个 9B 实例同时驻留，
按 Arm 依次启动/运行；本轮不启动云实例。

### 6.2 本地隧道

```powershell
ssh -N -L 18009:127.0.0.1:8009 -L 18008:127.0.0.1:8008 -p 29824 `
  -i C:\Users\zxt\.ssh\agentrecx_autodl `
  -o IdentitiesOnly=yes root@connect.cqa1.seetacloud.com
```

### 6.3 本地运行 6 个 Dev 任务

```powershell
$venv='C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONIOENCODING='utf-8'

& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm base --planner-url http://127.0.0.1:18009

& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm sft --planner-url http://127.0.0.1:18008
```

产物分别写入 `runs/p6_4_dynamic/base_<timestamp>/` 与 `runs/p6_4_dynamic/sft_<timestamp>/`
（summary/turns/tasks；最终评分基于真实 `response.recommendations`）。

## 7. 结论

- **接口接通：PASS（静态）**；
- **真实 Qwen Base/SFT 推理：NOT RUN / UNVERIFIED**（GPU 未启动，不得报告模型 PASS）；
- **公平对照条件：LIMITED** —— 同一 Runner、任务、history、Serving V1、Eligibility、k/max_steps/tool budget
  与评分已统一；role_target/hard constraints 已接通；但 planner query 上的 soft-preference M5 抽取
  仍是需要验证的公平性缺口；
- **P7：NOT READY**。下一轮应先启用 GPU 完成 Base/SFT 6-task 受控运行，并验证/修复 soft-state 注入，
  再讨论质量结论与 Holdout；本轮不扩大 Benchmark、不运行 Holdout、不训练。
