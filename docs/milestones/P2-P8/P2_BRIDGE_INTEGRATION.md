# P2-Bridge 集成报告

状态：**已实现并真实运行 C/D0/D1 20 个 dev 任务**。

完整实现/结果见 [P2_BRIDGE_IMPLEMENTATION_REPORT.md](P2_BRIDGE_IMPLEMENTATION_REPORT.md)。

## 1. 数据流

```text
Qwen3.5-9B Base/SFT
  -> operation JSON
  -> schema + evidence validator
  -> ProposalActions
  -> ExplicitPreference/ExplicitConstraint/ExplicitRetraction
  -> M5 StateIntegration (SQLite events)
  -> effective state
  -> R1
  -> M6.1
  -> Top-K
```

- `proposal_bridge.py` 只做验证与映射，不直接写 DB；
- `agent_service.py` 在 R1 之前把 proposal 作为本轮唯一 M5 输入；
- 有效 proposal 关闭同轮规则提取器的 M5 双写；
- invalid proposal 被拒绝，不产生 M5 事件，也不会被规则回退掩盖；
- D0/D1 使用相同 Prompt、schema、解码、R1/M6.1、candidate、Top-K；仅权重不同；
- C 用相同 `apply_constraints` 配置重跑，不使用历史 12/20 直接比较。

## 2. 验收结果

- `tests/test_proposal_bridge.py` 4 项 + P2 回归 13 项通过；
- smoke：`dev_06_cookware_material` 3 轮真实 D1，3 个 proposal 均写入 M5；
- D0 proposal 有效 1/61（60 invalid JSON，真实失败）；
- D1 proposal 有效 52/64（12 invalid，已记录拒绝）；
- C/D0/D1 V2 均为 PASS 12 / FAIL 5 / UNKNOWN 3；
- legacy C 8、D0 8、D1 7；
- D1 的 proposal 写入没有转化为 V2 成功率提升。

## 3. 相关产物

- [P2_BRIDGE_IMPLEMENTATION_REPORT.md](P2_BRIDGE_IMPLEMENTATION_REPORT.md)
- [P2_BRIDGE_AGENT_COMPARISON.csv](../../../training/agent_sft/eval/P2_BRIDGE_AGENT_COMPARISON.csv)
- [P2_BRIDGE_AGENT_SUMMARY.json](../../../training/agent_sft/eval/P2_BRIDGE_AGENT_SUMMARY.json)
- [P2_BRIDGE_TASK_FAILURE_ANALYSIS.md](P2_BRIDGE_TASK_FAILURE_ANALYSIS.md)
- smoke：`training/agent_sft/eval/P2_BRIDGE_SMOKE_TRAJECTORY.jsonl`
- 逐轮轨迹：`runs/p2bridge_c/trajectories.jsonl`、`runs/p2bridge_d0/trajectories.jsonl`、`runs/p2bridge_d1/trajectories.jsonl`
