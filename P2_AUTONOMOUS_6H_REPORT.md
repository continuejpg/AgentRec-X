# P2 6 小时自主执行报告

## 0. 实际状态

在无人值守窗口内完成了：

1. Proposal → M5 → R1/M6.1 执行桥实现；
2. 真实 Qwen3.5-9B Base/SFT HTTP 推理；
3. C/D0/D1-v1 统一配置重跑 20 dev 任务；
4. D1 商品目标/accessory 映射修复；
5. SFT-v2 增量数据与一次真实增量训练；
6. D1-v2 20 dev 任务与 Parser 最终困难集诊断；
7. 失败归因、RL readiness 审计。

未完成/跳过：

- 原始 200 Parser Test 的 SFT-v2 评测已完成（Exact 1.00，接近饱和，谨慎解释）；
- Agent RL：入口条件不满足，未启动，未伪造；
- frozen benchmark、Serving/推荐模型修改、commit/push/deploy：均未做。

## 1. 最终 C/D0/D1-v1/D1-v2

| arm | V2 PASS/FAIL/UNKNOWN | legacy | p50/p95 ms | tokens |
|---|---|---|---:|---:|
| C | 12/5/3 | 8 | 9173/20167 | 0 |
| D0 Base | 12/5/3 | 8 | 12139/19755 | 25,973 |
| D1 SFT-v1 | 12/5/3 | 7 | 17823/33086 | 25,760 |
| D1-v2 SFT-v2 | 12/5/3 | 7 | 15341/20625 | 25,721 |

逐任务结果：`P2_FINAL_AGENT_COMPARISON.csv`。

### 关键事实

- 执行桥真实写入 M5：D1 52/64 个 proposal 有效，D0 只有 1/61；
- D0 的无效 JSON **没有被规则回退掩盖**；
- D1-v2 的 Parser 困难集 Exact 0.66→0.92，但 Agent V2 仍 12/5/3；
- D1-v2 延迟比 D1-v1 低（p50 17.8s→15.3s），但未超过 C/D0 的成功率；
- 所有臂失败的 8 个任务集合基本相同：dev_05/06/07/12/13/14/17/18。

## 2. 失败归因

| 类别 | 结论 |
|---|---|
| Parser/JSON | D0 60/61 invalid；D1 12/64 invalid；SFT-v2 显著改善解析器 |
| Proposal→M5 | 已真实工作；D1 valid 52，M5 有 events |
| M5 lifecycle | 17/20 PASS；dev_06/07/14 FAIL，C/D0/D1 同 |
| Goal/role/候选 | dev_13/18 role FAIL；dev_12 空候选，属角色/召回 |
| 预算多轮 | dev_05/17 UNKNOWN，非桥接独有 |
| R1/工具动作 | D1 RETRIEVE 34 / RERANK 18，D0 32/17；动作本身有效 |

当前瓶颈：**Agent 执行/状态/角色候选层**，不是“proposal 只写日志”。SFT 继续提升 Parser
并不会自动提高 V2。

## 3. SFT-v2 负面/混合结果

- 训练真实完成：180 steps、1 epoch、train_loss 0.00493、eval_loss 0.04979；
- 困难集 Parser：Exact 0.66→0.92，Slot 0.535→0.951；
- Agent V2：无提升（12/5/3 不变）；
- 因此保留 SFT-v1 和 SFT-v2 两个 adapter，不宣称 SFT-v2 全面优于旧模型；
- 原始 Parser Test SFT-v2 结果已完成；Exact 1.00，属于接近饱和结果。

## 4. Agent RL

**未启动。** 进入条件不满足：

- 没有可训练的 ASK/RETRIEVE/RERANK/FINALIZE 策略动作；
- 没有 reward/工具成本/非法动作惩罚训练接口；
- 没有多步 rollout 到 policy update 的闭环；
- R1/M6.1 是固定执行器，模型只输出 Parser JSON。

详见 `P2_RL_READINESS.md`。没有用单步 Parser GRPO 冒充 Agent RL，也没有 RL checkpoint。

## 5. 最终回答

1. **D1 修复后是否优于 D0/C？** 否；V2 均 12/5/3，legacy C/D0/D1 为 8/8/7。
2. **SFT-v2 是否实际训练，性能是否改善？** 实际训练；困难集 Parser 明显改善（Exact .92），原始 test Exact 1.00（接近饱和，谨慎解释）。
3. **增量 SFT 是否改善端到端 Agent Task Success？** 否，V2 不变。
4. **RL 环境是否具备真实多步训练条件？** 否；缺策略动作、reward 和多步训练闭环。
5. **是否完成实际 RL 策略更新？** 否；未运行 RL Pilot。
6. **RL 相对 SFT 是否有有效增益？** 无数据，不能声称。
7. **当前最值得继续研究的瓶颈？** M5 lifecycle、goal/role 语义、候选召回/角色证据；不是继续加大 Parser SFT 或直接 RL。

## 6. 产物

- `P2_FINAL_AGENT_COMPARISON.csv`
- `P2_SFT_V2_TRAINING_REPORT.md`
- `P2_RL_READINESS.md`
- `P2_BRIDGE_IMPLEMENTATION_REPORT.md`
- `P2_BRIDGE_TASK_FAILURE_ANALYSIS.md`
- 轨迹：`runs/p2bridge_{c,d0,d1,d1v2}/trajectories.jsonl`
- SFT-v2 adapter：AutoDL `/root/autodl-tmp/agentrecx/training/agent_sft/output/qwen3_5_9b_lora_v2/adapter`
- 数据：`training/agent_sft/data_v2/`
- 环境：RTX 6000D 83GiB、torch 2.11+cu128、transformers 5.22.1、peft 0.21.2。
