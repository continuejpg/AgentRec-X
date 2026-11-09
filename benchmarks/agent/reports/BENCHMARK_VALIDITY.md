# Benchmark 可信度审计

## 结论先行

当前 benchmark 可以作为**开发集诊断工具**，但还**不足以支持 A/B/C/D 策略因果比较**：

1. A/B/D 在当前环境没有 LLM provider，全部未执行；
2. C 与 A/B 不是相同的执行路径（C 有 M5/R1/M7，A/B 是 stateless rewrite）；
3. C 的 2/20 结果混有确定性策略失败、用户模拟器限制和 Gold 粒度问题；
4. D 的“结构化偏好提议”还没有真正写入 M5 或改变后续动作，不能算已验证收益；
5. 当前没有人工相关性标签，不能把 deterministic pass 当成满意度。

本审计没有修改任务、Gold、冻结协议、Serving V1，也没有生成 A/B/D 的模拟结果。

## 1. 共享部分与真实对照变量

共享：

- 同一真实 Sports & Outdoors catalogue 与 SASRec / Two-Tower / BM25 运行时；
- 同一 `k=5`，同一 `history,catalog_search` source plan；
- 同一 RRF 与 trusted metadata；
- BGE、ItemCF 强制关闭；
- 同一 benchmark evaluator 和任务数据。

C/D 额外拥有：

- M5 unified task state；
- R1 `RETRIEVE` / `RERANK`；
- M7 grounded interaction；
- `preference_discovery` 按任务开启。

A/B 不具备这些状态能力。A/B/D 还需要 LLM，但当前 provider 缺失。

## 2. 公平性风险

| 风险 | 证据 | 结论 |
|---|---|---|
| 执行路径不同 | A/B 用 `service.run` stateless；C/D 用 `run_shopping_turn` | 不是纯 policy ablation，包含 runtime 路径差异 |
| 交互预算不同 | C/D 可能 M7 提问、R1 重新检索；A/B 没有澄清回合 | C/D 有额外交互机会，不能声称是同等预算对照组 |
| 用户模拟器不同 | C/D 的 benchmark 用户模拟器只在能匹配 option 时回答 M7；否则忽略 pending question | 会改变澄清轨迹，尤其 dev_08 |
| 候选预算相同 | 四组均 k=5、相同 source plan | 已尽量固定，但 C/D 的 R1 可额外 tool call |
| 商品资格检查 | A/B 在 benchmark 后处理中调用 `promote_explicit_type_matches`；C/D 在 shopping service 内处理 | 类似但不等价，可能造成角色过滤差异 |
| LLM 不可用 | `llm_not_configured: LLMPolicyError` | A/B/D 没有可比较输出 |
| 人工标签缺失 | 0 labels | 不能评估满意度或真实相关性 |

## 3. Session / Memory / SQLite 隔离

- A/B：stateless，不传 `user_key`，理论上不读写 M5 memory。
- C/D：每条任务/策略生成唯一 session id，例如
  `bench:C:dev_01_goal_correction_withdraw:<uuid>`；
  `task_id` 与 session 同级；`_task_preferences`、`_r1_*`、M7/M8 缓存均按 task id 分键。
- 四组共享同一个 `DemoRuntime`、同一个 preference memory SQLite 和同一个 decision-state SQLite。
  不是每个 policy 独立数据库；隔离依赖 session/task key，而不是独立 DB 文件。
- 因此：
  - 跨任务读取不应发生，因为 task id / session id 唯一；
  - 运行顺序**理论上**不应改变推荐候选；
  - 但全局 DB 锁、事件诊断容量（`MAX_DIAGNOSTIC_TASKS=16`）和日志/延迟会随运行顺序变化；
  - 当前没有按 policy 单独重启模型或单独 state DB，所以不能宣称“完全独立状态目录”。

审计建议：下一阶段应打印每个 task 使用的 session/task/db key，或将 C/D 拆到独立 state DB；在当前版本中，至少应把“共享 runtime + 唯一 key”和“独立 DB”明确区分。

## 4. D 组 LLM proposal 实际链路

代码事实：

```text
LLMPreferenceProposer.propose()
  -> rewritten_message
  -> [validated structured proposals]  (recorded as llm_proposals in trace)

LLMPreferenceAdapter.run()
  -> AgentRecommendRequest(message=rewritten_message, ...)
  -> run_shopping_turn()
```

也就是说：

- LLM 的结构化 `preferences` 只经过本地校验并记录；
- **没有**作为 `constraints` 注入请求；
- **没有**作为 M5 extraction 注入；
- **没有**单独的 proposal → preference event 通道；
- 只有 `rewritten_message` 影响一次 C pipeline 的文本理解。

因此 D 不是“已实现结构化偏好收益”。它最多是“LLM rewrite + 结构化 proposal 记录”；真正能改变策略的是 rewrite 文本，而不是 proposal 对象。Provider 缺失时 D 完全不可运行，也没有模拟数据。

## 5. Gold 与 Evaluator 可信度

已发现的 Gold 粒度问题：

- `dev_06`、`dev_07`、`dev_10`、`dev_15`、`dev_20` 的最终商品角色正确，但不在 expected ASIN 列表；
- 这些不能仅凭 `final_any` 精确身份判定为“策略彻底失败”；
- `dev_07` 同时含 forbidden water filter，`dev_20` 同时含 survival kit/tent，说明角色正确也不等于完全正确；
- `dev_15` 用户说“lighter weight”，但 task expected active attributes 为空，Gold 状态语义值得后续复核（本阶段不修改）；
- 冻结集不参与本次审计，未运行、未调参。

Evaluator 当前输出是一个 deterministic pass，但没有把 `role_ok`、`identity_ok`、`forbidden_role_ok` 拆成独立指标；这是下一阶段最优先可观测性修复。

## 6. 运行顺序审计

静态检查显示：

- C/D 的 session id 唯一；
- service task-preference/R1/M7 caches 按 task key；
- state DB 事件按 task_id 查询；
- A/B 不写 memory；
- 因此顺序改变不应改变同一任务的候选或状态。

未将“顺序改变”作为已执行实验：没有为了验证顺序而重复跑完整 20 任务扫描。当前只能报告“设计上应按 key 隔离”，不能报告“已通过 A/B 顺序重复实验验证”。

## 7. 有效性判定

- **A/B/C/D 全策略因果结论：不可信**（A/B/D 无运行、交互预算不同、D proposal 未应用）。
- **C 组内部失败诊断：可用于开发集**，因为保留了原始 2/20、逐轮事件、R1 动作、完整 candidate pool 和最终 Top-K。
- **人工满意度：未测量**。
- **冻结集结论：未形成**；不得用冻结集结果反向修改本报告。

详见 [C_FAILURE_AUDIT.md](C_FAILURE_AUDIT.md) 与 [C_FAILURE_AUDIT.csv](C_FAILURE_AUDIT.csv)。
