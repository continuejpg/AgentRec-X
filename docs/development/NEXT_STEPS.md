# AgentRec-X Next Steps

> **Status: planning only.** M1-G added documentation; it did not implement any item on this
> page. The baseline these steps start from is documented in
> [`M1_BASELINE.md`](M1_BASELINE.md). Quality of recommendation remains **NOT ESTABLISHED**;
> engineering correctness and recommendation quality are separate claims.

## 1. 当前已完成能力

| 能力 | 状态 | 证据/边界 |
|---|---|---|
| Durable Warm Reply | 完成（M1-C） | handle 证明 → claim → heartbeat → EXTERNAL_STARTED → M9 → recommend → checkpoint → finalize → 200；`tests/test_web_warm_reply_e2e.py`、`test_web_reply_executor.py` |
| Cold Resume（MCP 子进程重启） | 完成（M1-D，CPU/BM25 单轮） | continuation v1 → validate → restore new handle → reply；`tests/test_web_cold_resume_e2e.py`、`test_web_continuation.py` |
| Task requirements 合并 | 完成（M1-E） | query 5000 + reply 8000/gaming/laptop → 8000/gaming/laptop；原始 5000 仅历史；`tests/test_web_requirement_update_e2e.py` |
| Budget Gate | 完成（M1-E） | 7999/8000 符合、8001 超；无价/跨币种 UNKNOWN；超预算首选可回退到 verified 候选；无候选明确拒绝 |
| 过期预算检索清理 | 完成（M1-F） | BM25 文本不再含 5000/8000；`retrieval_query` 单独记录；原始 query 保留 |
| 真实历史 USD → CNY 演示价 | 完成（M1-F，仅 demo mode） | 25,736 个真实 priced item；固定 7.00、Decimal/2dp；MSI `$899 → ¥6,293`；`tests/test_demo_price.py`、`test_web_demo_price_e2e.py` |
| Request Ledger / Checkpoint / Finalize / Replay | 完成 | `CLAIMED → EXTERNAL_STARTED → RESULT_DURABLE → COMPLETED`；AMBIGUOUS fail-closed；Replay 零 MCP |
| M9 偏好持久化与 Session 隔离 | 完成（英文显式偏好） | `preference_memory` schema v1；`user_key` 隔离；reply_turn 幂等；m9 共享 schema 初始化幂等 |
| 严格 / 演示价格模式隔离 | 完成 | 默认 strict；仅 `AGENTRECX_PRICE_MODE=demo_historical_fx` 启用；strict 重启 replay 不重算 |
| 本地 CPU/BM25 真实检索 | 完成 | 真实 C4 目录 + BM25；不接 GPU、不接外部模型 |

## 2. 未完成能力（不要误报为已完成）

### 2.1 多轮 ASK

当前只支持 **一次 ASK → 一次 reply → 一次推荐**。如果 reply 后 `recommend_products`
再次返回 `awaiting_clarification`，Coordinator 返回
**503 `clarification_reply_ambiguous`**：M9 可能已经写入，但结果不能最终展示，也禁止自动重跑。
要支持多轮，需要：

- 新的 pending/continuation 轮次，而不是复用已消费的 pending；
- 每轮唯一的 reply turn 幂等键；
- ASK 问题的上下文/已合并 requirements 的持久化；
- 对“同一 session 只有一个 active task/OPEN pending”的约束做显式状态机扩展。

### 2.2 qwen_live 持久化 Pending

qwen_live 的 clarification handle 依赖 MCP 子进程内的 planner/executor continuation
（对象/闭包，不是 JSON 快照）。当前：

- qwen_live 的持久化 pending 仍是 **409**；
- 本轮/现有 M1 路径不启用 Qwen；
- 要跨进程恢复，需要把 planner 状态事件化/可重放，而不是序列化对象。

### 2.3 动态 Task 恢复与故障恢复边界

现已支持“**精确匹配的快照**恢复”，不支持“从任意历史状态重建任务”：

- 快照缺失/损坏/过期/版本不符/绑定不符 → 409；
- 旧 handle 永不复用；restore 无业务副作用；
- EXTERNAL_STARTED 后进程崩溃 → AMBIGUOUS，禁止自动重跑；
- Web SQLite 与 M9 SQLite 不是同一事务域，不承诺 exactly-once；
- 未验证任意 crash point（claim 后/外部调用中/checkpoint 前）的完整恢复矩阵。

### 2.4 真实价格与商品分类

- 默认 strict 模式没有 CNY 价格；demo 模式使用真实历史 USD 价格 × 固定演示汇率，
  不是实时售价；50,527 个 matched record 没有可解析价格；sidecar 没有 per-item 报价日期。
- laptop 角色已加入 `product_role`，但仍是标题词序/配件词表启发式，不是完整商品 taxonomy；
  top 20 通常只有少数主商品笔记本有历史价格。
- 不接实时价格 API，不使用 GPU，不更换商品数据集。

### 2.5 展示与交付层

- 前端优化、固定演示流程/脚本、GitHub README 完善：**本轮不实施**，作为后续集中处理事项。
- 商品卡 demo 文案有 payload/viewmodel 断言；尚未做浏览器视觉回归。

## 3. 已知技术限制

1. **Cold Resume 范围**：CPU/BM25、单轮；`pending.mode` 必须是 `cpu_live`。
2. **多轮 follow-up ASK**：503 ambiguous，不自动重试。
3. **qwen_live**：持久化 pending 409；需要 planner 状态事件化。
4. **跨库一致性**：Web State 与 M9 两库；M9 写成功后 finalize 失败 → AMBIGUOUS；不宣称 exactly-once。
5. **并发 reply**：同时到达的回复由 claim/lease/fencing 决出唯一赢家；孤儿 restore handle 随 TTL
   失效，不跨进程回收。
6. **预算拒绝语义**：`budget_unverifiable`/`budget_exceeded` 返回 HTTP 200 + 无商品 refusal；
   通用 `no_verified_candidate_in_scope` 仍按既有语义走 503，不混同。
7. **价格数据缺口**：50,527 条无价；没有 per-item 报价日期；固定 7.00 是演示汇率，不是市场汇率。
8. **有界候选池**：BM25 top-K 不是全目录穷举；“候选池无结果”不等于“全目录无商品”。
9. **M9 中文偏好**：M9 抽取器是英文规则；中文“游戏/8000”只存在于 task requirements，不写永久 M9。
10. **Demo 子进程启动成本**：加载 143 MB sidecar 约 13s/子进程；真实 E2E 较慢，不能当默认启动。

## 4. 值得进一步测试的风险

| 风险 | 为什么重要 | 建议测试 |
|---|---|---|
| 配置泄漏（strict vs demo） | demo 汇率不应影响 strict 或已提交结果 | 启 demo 后以 strict 重启 replay；检查 MCP 子进程 `--price-mode` |
| Gate 回退选择 | 超预算/无价首选不能被包装成成功 | 构造 top-1 over-budget、lower-ranked within；全 over 必须拒绝 |
| 过期预算误删/漏删 | 预算数字既不能进 BM25，也不能误删型号数字 | 中英文预算短语、`Dell 5000`/`RTX 8000` 等反例 |
| 多轮 ASK | 现在明确 503，未来改状态机时最易重复写 M9 | 第二轮 ASK 的 pending/continuation/idempotency 矩阵 |
| Qwen live 恢复 | 对象状态不能序列化，容易变成假恢复 | 子进程重启、snapshot 缺失/损坏、planner 重放 |
| Reset Fencing | reset 后旧 claim 不能提交 | reset 跨越 claim/checkpoint/finalize 的并发测试 |
| 跨库 exactly-once | M9 写入与 Web finalize 不在一个事务 | M9 成功后 finalize 失败、再 replay；确认 AMBIGUOUS |
| M9 共享 DB 并发 | 两个 MCP 子进程同时建库曾出现 UNIQUE 竞态 | 空 DB 双 child 启动；schema v1 重开 |
| 真实 ID/价格漂移 | sidecar 数据更新会改变匹配率与价格 | 重新计算 matched/priced/≤8000，并冻结报告快照 |
| 前端展示 | payload 有 demo 文案，不等于浏览器真的显示 | 视觉/E2E 断言 price display + note |

## 5. 建议的后续开发顺序

1. **冻结并复验基线**（最低成本）
   - 在 M1 文档/配置矩阵上补契约测试；
   - 每次代码变更后重跑 33 文件 M1 回归（369 是上一轮的历史数字，不能冒充本轮新结果）；
   - 保持 strict 默认，确认 demo 模式不影响它。
2. **多轮 ASK 状态机**（最高业务价值）
   - 为第二轮 ASK 新建 pending/continuation；
   - 明确 reply turn 幂等键、task requirements 合并、失败/重试边界；
   - 新增真实 CPU/BM25 多轮 E2E，并验证不会重复 M9 写入。
3. **qwen_live 持久化 pending**（在多轮状态机之后）
   - planner/executor continuation 事件化或可重放；
   - 明确“恢复失败必须 409，不自动重跑”；
   - 不启用 Qwen 时仍保持现有 strict/CPU 路径不变。
4. **动态 Task 恢复与故障恢复矩阵**
   - 覆盖 claim 后、EXTERNAL_STARTED 后、checkpoint 前、finalize 前的 crash；
   - 明确 RESULT_DURABLE/COMPLETED/AMBIGUOUS 的可恢复边界；
   - 保留“跨库不承诺 exactly-once”的诚实声明。
5. **真实价格/汇率与商品 taxonomy**（仅当展示需要更真实时）
   - 引入可核实的 CNY 价格源或有来源的汇率；
   - 扩大 Product Role/类型表，不再依赖标题启发式；
   - 继续保留 strict 模式的 UNKNOWN 和 demo 模式的显式标注。
6. **展示层集中处理**（最后）
   - 前端价格/演示文案渲染、固定演示脚本、GitHub README；
   - 本轮明确不做。

## 6. 本轮明确不做

- 不启用 Qwen / 不配置外部模型；
- 不开发多轮 ASK；
- 不迁移 Agent 框架；
- 不使用 GPU；
- 不改业务代码、前端或 GitHub README；
- 不新增演示脚本。

## 7. 完成标准（后续每个阶段）

一个阶段只有在满足以下条件时才算完成：

1. 代码/配置存在且与文档一致；
2. 单元测试 + 真实（或明确可控的）E2E 通过；
3. 回归测试通过，并记录 exit code；
4. 关键失败路径（无商品、无价格、重复请求、重启、reset）有断言；
5. 文档更新；
6. Git diff 与工作区检查；
7. 明确报告哪些是历史证据、哪些是本轮新跑结果。
