# P2 Agent RL Readiness

## 1. 结论

**未启动 Agent RL Pilot。** 当前条件不满足，且没有用 Parser 单步 GRPO 冒充多轮 Agent RL。

已完成的是：Proposal → M5 → R1/M6.1 执行桥、真实 D0/D1 推理与 20 dev 任务对照。

## 2. 条件检查

| 条件 | 状态 | 说明 |
|---|---|---|
| Proposal→M5→R1/M6.1 执行链路稳定 | PARTIAL | 桥已真实工作；20 任务 V2 C/D0/D1 均为 12/5/3，D1 未提升 |
| 存在模型可选择的多步动作 | NO | 当前 D 只生成 preference operations；RETRIEVE/RERANK/FINALIZE 由固定 R1/M6.1 执行，不是策略动作 |
| 环境返回真实工具观察和后续状态 | PARTIAL | M5 状态和 R1 结果可观测；但没有为策略提供 action-observation 训练接口 |
| 可验证完成奖励/非法惩罚/工具成本 | NO | Evaluator V2 是评测器，不是训练 reward；没有 reward 单元测试/反作弊/工具成本 |
| 奖励不依赖固定 Gold ASIN 或模型自评角色 | NO | V2 具备角色证据，但未定义为可训练 reward |
| 可收集多步真实 rollout 并关联策略更新 | NO | 当前只记录轨迹，没有 policy logprob/动作选择/训练入口 |
| 至少一个无 frozen 任务的最小在线训练闭环 | NO | 尚未实现 |

## 3. 为什么不能直接做 GRPO

当前模型输出是 **Parser JSON**，不是 Agent 动作序列。R1/M6.1 的 RETRIEVE、RERANK、FINALIZE
由系统固定执行，模型无法选择；因此：

- 单轮 Parser 的 schema/operation reward 只能训练“解析器”，不能称为 Agent RL；
- 任务成功受 M5 生命周期、角色证据、候选召回共同影响，单步奖励会错误归因；
- 没有可执行的 policy step/logprob，无法做真实策略梯度。

## 4. 已完成的 RL 前置工作

- 真实 M5 状态写入与 SQLite replay 单元测试；
- C/D0/D1 真实逐轮轨迹；
- Proposal 有效性/拒绝统计（D0 1/61，D1 52/64）；
- Evaluator V2 逐任务结果；
- 失败归因：D0 Parser/JSON、D1 goal/role/M5 lifecycle/候选召回；
- 可用于后续 rollout 的本地 Agent 运行入口和模型 HTTP proposer。

## 5. 若继续做 RL 的最小必要步骤

1. 定义策略动作空间：ASK、RETRIEVE、RERANK、FINALIZE（或 operation+action 联合）；
2. 实现 action→environment state transition 和 multi-step trajectory；
3. 定义 reward：任务完成 + hard constraint/role + 工具成本 − 非法动作；
4. 增加 reward 单元测试和反作弊测试；
5. 增加 policy logprob/训练入口和 KL 诊断；
6. 先在独立训练任务上跑一个最小在线闭环，验证参数更新；
7. 通过后才允许 Pilot 与开发集比较。

当前没有执行以上训练入口，因此没有 RL checkpoint、RL 指标或 RL 增益声明。
