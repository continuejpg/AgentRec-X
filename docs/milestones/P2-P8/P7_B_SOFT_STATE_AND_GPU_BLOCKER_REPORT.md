# P7-B Soft-State 修复与 GPU 阻断报告

## 1. 状态一句话

**Soft-State 污染已确认并完成最小修复（CPU，13 passed）；真实 Qwen Base / SFT-v0 对照未运行，原因是当前远端实例没有可用 GPU。**  
不伪造任何 Planner 结果；Rule P6.4 基线保留，未重跑。

## 2. Soft-State 污染与修复

### 2.1 是否污染：是

`ServicePlannerExecutor.execute(RETRIEVE)` 原先执行：

```python
response = self.service.run_shopping_turn(
    self.payload.model_copy(update={"message": query}), ...
)
```

`run_shopping_turn` 随后把 `payload.message` 当作当前用户话语：

- `_execute_plan` 从 `payload.message` 抽取 M9/M5 preferences 并写入状态；
- `_integrate_state_turn` 把抽取结果并入 M5；
- 重复 RETRIEVE 可能因此重新写入/重新激活已撤回的偏好；
- `_update_role_target` 也可能被改写后的 query 改变。

结论：仅当 Planner query 与用户原话语恰好一致时才无害；这是真实污染风险，不能带入 Base/SFT 对照。

### 2.2 最小修复

保持 Planner 可自由改写 query，只把 query 用于检索：

- `run_shopping_turn` / `run()` / `_execute_plan()` 增加 `planner_retrieval`,
  `apply_user_state`, `state_message`；
- 第一次 Planner RETRIEVE：
  - 使用 `state_message`（真实用户 turn 原文）做 M5/M9 抽取与状态应用；
  - 不使用 Planner query 抽取偏好；
  - role target 由 runner 从真实用户 turn 预播种，不再由 query 改写。
- 同一 user turn 的后续 RETRIEVE/RERANK：
  - `apply_user_state=False`，不再抽取偏好、不再写 M5 events、不再重复应用状态操作；
  - 仍使用原始 `state_message` 已应用的状态和 trusted constraints。
- Planner query 仍完整保留用于检索；没有固定或限制 Qwen 的 query 文本。

修改文件：

- `recommendation/demo/agent_service.py`
- `recommendation/demo/planner_service_executor.py`
- `experiments/p6_benchmark/run_dynamic_dev_p64.py`
- `tests/test_p7_b_soft_state.py`（新增针对性回归）

源码 diff：`P7_B_SOFT_STATE_SOURCE_DIFF.txt`

### 2.3 最小验证

```text
tests/test_p7_b_soft_state.py                         1 passed
tests/test_planner_contract.py                        11 passed
tests/test_p6_4_role_target_transfer.py               1 passed
合计                                                  13 passed

场景：先写入 "lighter options" → 撤回 → Planner RETRIEVE query 含
"lighter options" → 断言已撤回偏好不会被重新激活；重复 RETRIEVE 亦然。
```

Rule 路径默认参数为 `planner_retrieval=False`、`apply_user_state=True`、
`state_message=None`，行为保持 P6.4 原路径；P6.4 Rule 结果仍可直接引用，未重跑。

## 3. GPU 检查与阻断

实际 SSH 检查（未启动/关闭任何云实例）：

```text
SSH root@connect.cqa1.seetacloud.com:29824        OK
/root/autodl-tmp/models/Qwen3.5-9B-ms             存在
.../qwen3_5_9b_planner_lora_v0/adapter            存在
torch 2.11.0+cu128, torch.cuda.is_available()     False
torch.cuda.device_count()                         0
/dev/nvidia*                                     不存在
/usr/bin/nvidia-smi                               0 字节且 Permission denied
model_server 进程 / 8008 / 8009 端口              不存在
```

结论：当前远端是 **CPU-only / 无 GPU 分配状态**，无法运行 Qwen Base 或 SFT-v0。
按任务停止条件，不运行真实模型、不伪造结果；需要先由用户把实例切到 GPU 模式或分配 GPU。

## 4. Base / SFT-v0 6-task 实际结果

**NOT RUN（BLOCKED）。**  
没有真实 Qwen 输出，因此：

- Base：未执行，不能报告 PASS/FAIL/UNKNOWN；
- SFT-v0：未执行，不能报告 PASS/FAIL/UNKNOWN；
- 与 P6.4 Rule 的逐任务对照：**暂不可做**；
- Base vs SFT 推荐质量、约束合规、工具成本：**无真实数据**；
- FINALIZE ID 映射：只有静态接线和代码路径，**未在真实 Qwen 轨迹验证**；
- SFT-v1 失败案例：**没有真实 Qwen 失败轨迹**，不能用 P6.4 Rule 失败冒充 Planner 失败。

## 5. 已接通的验证内容（CPU）

- `--arm rule|base|sft` 统一 Runner 已接通（P7-A）；
- Base/SFT 使用现有 `HttpPlannerProposer` + `run_planner` + `ServicePlannerExecutor`；
- 每 user turn 的 runner state operations 只应用一次；
- Planner query 不再写入用户偏好；
- `ServicePlannerExecutor` 记录真实 Agent response 和 `tool_calls_used`；
- FINALIZE 返回的 IDs 与最近一次真实 `response.recommendations` 做身份映射；若失败会记录
  `planner_identity_mismatch`（真实轨迹待验证）；
- 已知待验证边界：如果某个 user turn 的 Planner episode 在任何 RETRIEVE 之前就以 ASK 终止，
  该 turn 的 canonical state operations 尚未写入 M5（当前实现只在第一次 RETRIEVE 应用状态）。
  单任务 GPU 检查必须先验证这一点；若 Qwen 出现该路径，需要把状态应用前移到 episode 开始。 

## 6. GPU 就绪后的准确运行命令

### 6.1 确认 GPU / 启动 model server（远端）

```bash
cd /root/autodl-tmp/agentrecx

# Base，无 adapter
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --port 8009

# SFT-v0，加载 Planner LoRA adapter
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --adapter /root/autodl-tmp/agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter \
  --port 8008
```

若显存不足以同时驻留 Base + SFT，按 Arm 依次启动/停止 model server（不关闭/操作云实例）。

### 6.2 本地隧道

```powershell
ssh -N -L 18009:127.0.0.1:8009 -L 18008:127.0.0.1:8008 -p 29824 `
  -i C:\Users\zxt\.ssh\agentrecx_autodl `
  -o IdentitiesOnly=yes root@connect.cqa1.seetacloud.com
```

### 6.3 先做单任务最小检查（budget_update）

```powershell
$venv='C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONIOENCODING='utf-8'

& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm base --planner-url http://127.0.0.1:18009 --only budget_update

& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm sft --planner-url http://127.0.0.1:18008 --only budget_update
```

检查：Action 全部通过 Validator、每步 M5 constraint/role 状态、FINALIZE ID 是否为真实
`response.recommendations`、`planner_identity_mismatch` 是否出现。

### 6.4 单任务通过后跑 6 个 Dev

```powershell
& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm base --planner-url http://127.0.0.1:18009

& "$venv\Scripts\python.exe" experiments/p6_benchmark/run_dynamic_dev_p64.py `
  --arm sft --planner-url http://127.0.0.1:18008
```

产物：`runs/p6_4_dynamic/base_<timestamp>/`、`runs/p6_4_dynamic/sft_<timestamp>/`。
评分仍只使用 promotion 后的 `response.recommendations`。

## 7. 结论

- **Soft-State 修复：PASS（CPU；13 tests passed）**；
- **Qwen Base / SFT-v0 真实对照：BLOCKED（GPU 不可用）**；
- **Rule P6.4 基线：保留，未被本轮默认路径修改**；
- **P7-B 目标：未完成真实对照；不宣称 Base/SFT 质量或 P7 就绪**；
- 下一步需用户先恢复/分配 GPU，然后按 §6 执行；本轮不训练、不运行 Holdout、不修改
  Gold/Evaluator/Serving V1、不 commit、不操作云实例。
