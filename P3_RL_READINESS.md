# P3 Agent RL Readiness

## 1. 结论

**未启动真实 Agent RL Pilot，也没有 RL Checkpoint 或参数更新。**

完成了最小 action/environment 合同与确定性 unit test，但它**没有接入真实 AgentRec-X
任务环境**，不能当作 RL 训练证据。

## 2. 已完成的 RL 前置工程

新增 `training/agent_sft/agent_rl_env.py`：

- Action：`ASK / RETRIEVE / RERANK / FINALIZE`；
- 合法动作检查、重复动作拒绝、未知动作惩罚；
- observation/state transition、最大步数、终止条件；
- reward 使用：
  - 商品角色是否满足；
  - 硬价格约束是否满足；
  - 工具成本；
  - 非法/重复动作惩罚；
- reward **不使用目标 ASIN**，也不使用模型自己的商品角色判断。

测试：

```text
tests/test_agent_rl_env.py + tests/test_proposal_bridge.py
9 passed
```

此外已有真实 P3 轨迹分析：

- Proposal 52 个有效 operation → 38 个 M5 events；
- M5 变化只导致 3/20 任务最终商品集合变化；
- C/D0/D1/D1-v2 V2 PASS 集合完全相同。

## 3. 进入真实 RL 仍缺的条件

| 条件 | 状态 |
|---|---|
| 真实 AgentRec-X 多步环境接口 | 未完成（当前只是 toy contract） |
| 模型可选择 ASK/RETRIEVE/RERANK/FINALIZE 并产生 logprob | 未实现 |
| 每个动作返回真实工具观察和后续任务状态 | 未实现 |
| 训练任务（非 frozen Benchmark、非已分析 dev） | 未建立 |
| reward 与环境真实反馈绑定 | toy 已测试；真实链路未绑定 |
| 多步 rollout→policy update 闭环 | 未实现 |
| 至少一次真实参数更新验证 | 未发生 |

## 4. 为什么没有强行训练

当前证据表明：

- Parser 改进没有转化为 Agent V2；
- M5 状态到 R1/M6.1 的传递很弱；
- 失败任务由生命周期/角色/候选证据驱动；
- 没有可训练的动作选择接口。

在这些条件不满足时做 GRPO，只能得到“单步 JSON/动作模板”优化，不是真实 Agent RL。
因此保存 readiness 与合同，不生成虚假 RL 指标。

## 5. 最小后续步骤

1. 将 `agent_rl_env` 接入现有 M5/R1/M6.1，而非 toy items；
2. 为每个 action 保存 policy logprob；
3. 建立独立 RL training tasks，禁止使用 frozen/dev 已分析任务；
4. 绑定真实 hard-constraint/role/tool-cost reward；
5. 跑一次最小在线更新并验证参数变化；
6. 通过后才启动 Pilot。
