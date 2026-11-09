# P14-E2 Lite — Focused Benchmark Expansion 报告

> 范围：仅 P14-E2 Lite。在冻结的 P14-E 40 条 pilot_dev 之上新增 **80 条**任务，切分为 train 40 / dev 20 / reserved_test 20；未训练模型、未启动 Qwen、未改 Scorer/Gate/Memory/Planner/Serving、未改 P14-E 标签、未 commit。
> 数据：`data/p14e2/benchmark/`；校验：`runs/p14e2/validation/summary.json`；Smoke：`runs/p14e2/smoke/`；源码差异：[`P14E2_LITE_SOURCE_DIFF.txt`](P14E2_LITE_SOURCE_DIFF.txt)。

## 0. 一句话结论

新增 **80 条完整任务**（另有原 40 条 pilot_dev 保持冻结，合计 120 条），三类切分按 persona/family/same-query pair 隔离；数据校验 **pass=true**（0 failure，0 warning），原 P14-E 数据哈希未变、reserved_test 无任何推理记录。6 条 train/dev 真实工具 Smoke 中 4 条达到预期决策（2 条推荐、1 条 ASK、1 条有界拒绝），2 条推荐任务出现“Top-1 不在标注合格集”的已知 Scorer/Gate 限制而非数据不可读。判定 **GO**（仅表示可进入 P14-C0 Qwen Base 的 train/dev 推理诊断；标签仍 provisional，reserved_test 仍封存）。

## 1. 实际新增多少条任务？

**80 条**，ID `p14e2_lite_0001`–`p14e2_lite_0080`：

| Split | 任务数 | 用途 |
| --- | ---: | --- |
| train | **40** | Qwen Base / Scorer 开发与调参观察 |
| dev | **20** | 开发期诊断（不用于训练权重） |
| reserved_test | **20** | 封存；本轮只做 Schema/泄漏/ID 校验，未搜索、未 Scorer、未 Gate、未 Planner |
| 新总量 | **80** | 加上冻结的 P14-E pilot_dev 40 = 120 |

任务类型（实际）：

| 类型 | 数量 |
| --- | ---: |
| no_history_recommend（无历史但可推荐） | 15 |
| history_influenced（历史偏好影响选择） | 20 |
| same_query_persona_pair（8 对） | 16 |
| preference_update / preference_withdrawal / preference_conflict | 4 / 4 / 4 |
| insufficient_information（真正需要 ASK） | 10 |
| no_verified_match / unverifiable_constraint（证据不足/无法验证） | 4 / 3 |
| **合计** | **80** |

期望决策：recommend 59、clarification_required 14、no_verified_match 4、insufficient_evidence 3。

## 2. 与原 40 条相比补充了哪些场景？

- **无历史也可推荐（15）**：新增“只有当前显式硬约束”的任务族（如 “must be wireless / noise cancelling / over-ear”），历史只有中性事件、无可恢复偏好；用于防止 Decision Gate 学成“没有历史就一律 ASK/拒绝”。
- **历史偏好影响选择（20）**：当前 query 不重复历史软偏好，依赖合法 history-state read 恢复；覆盖 headphones/earbuds/keyboard/mouse/speaker 五类。
- **同 Query 不同 Persona（16 = 8 对）**：train 5 对、dev 2 对、reserved 1 对；每对 query 完全相同、persona 不同，且 pair 内两任务在同一 split；pair 间 query 互不重复，并加入与 pilot 不同的 pair 级上下文语句。
- **动态偏好（12）**：update 4、withdrawal 4、conflict 4；保留显式事件顺序、撤回/更新语义，更新后旧值必须失效。
- **ASK（10）**：真正缺少关键信息（无 preference、无 hard requirement、query 泛化），而非商品事实缺失；商品事实类问题由 Gate 要求补证据而不是问用户。
- **证据不足/不可验证（7）**：4 条“当前采样目录内无可验证组合”，3 条 price/warranty 等 catalog_meta 无字段的 UNKNOWN，硬约束不被当作满足。
- 所有非 pair query 附加了确定性、无属性词的上下文短语，保证**不同 split 间不存在相同 query+history 签名**；原 P14-E 的 40 条与新数据无近重复（校验通过）。

## 3. 是否存在数据或标签泄漏？

**没有发现泄漏，校验通过（leakage_hits={}）**：

- Agent-visible 仅含 query、history 表达、工具、namespace、tool_budget、公开约束、构造元数据；不含 effective/withdrawn preferences、expected decision、verified ids、persona 结构、task_type/family_id、label basis/intent note；
- `family_id`、task_type、label_basis、review reasons 仅存在于 evaluator-only；
- **Original P14-E 未修改**：`agent_tasks.jsonl`/`evaluator_only.jsonl` 的 sha256 与 manifest 一致；
- **Reserved-test 隔离**：本轮没有 reserved_test 的搜索/Scorer/Gate/Planner 记录（校验 smoke 目录 0 命中），只做了 Schema/泄漏/ID/证据引用校验；
- 新 persona 与 P14-E pilot personas **零重叠**；family/persona/same-query pair 不跨 split（校验通过）；
- 80 条任务没有任何近重复签名，且没有与冻结 P14-E 任务重复的 query+history 签名。

## 4. 哪些标签仍需人工审核？

- 新增 **23/80** 条标记为 `pending_human_review`（其余 `automated_checked`，**没有** `human_verified`）：same-query 对照 pair、no_verified_match/unverifiable 证据边界、少量候选任务；
- 原 P14-E 的 **15 条 pending 保持原状**，未升级；
- 审核清单：`data/p14e2/benchmark/review_checklist.json`（列出 task_id、split、类型、原因、有效偏好、硬约束、verified 数量）；
- 敏感性：本轮没有把 pending 任务从数据中删除，Smoke 与校验包含全部任务；正式报告结论标注 provisional，不用 pending 任务单独提升结果；reserved_test 全部封存。

## 5. 是否有足够商品证据支撑评价？

**有，且已如实标注边界**（`runs/p14e2/validation/summary.json`）：

- 全部 80 条任务验证的 verified 商品引用 **1,460** 个，全部在 P13-R accepted catalogue 与 namespace `amazon_electronics_c4` 中；
- 硬约束可验证（每个声明的硬约束在当前池内至少有一个证据支持的商品）**77/80**；3 条不可验证任务是 price/warranty 的显式 UNKNOWN（不应被视为满足）；
- 候选池 UNKNOWN 属性平均比例 **85.55%**（只在 17 个文本证据属性上评分，price/brand/size/material/stock 不参与）；
- no_verified_match 4 条、unverifiable 3 条均是“当前采样目录/字段范围内”的结论，不是全目录断言；
- 商品事实来自真实 catalog_meta 文本证据，未编造/未用 LLM 补全。

## 6. 现有 Runtime 是否能够读取新增任务？

**能读取（Smoke 已证明），但推荐质量仍需诊断**：

- `runs/p14e2/smoke/` 的 6 条 train/dev 任务真实调用了 M9/M5 history-state read、BM25（P13-R）、Product Info、CandidateLedger、Eligibility、冻结的 Preference Scorer 与 Decision Gate，数据可正常读取与执行；
- 结果：no_history_recommend → ALLOW_FINALIZE 且 Top-1 正确；history_influenced → ALLOW_FINALIZE 且正确；insufficient_information → NEED_USER_CLARIFICATION；no_verified_match → NO_VERIFIED_CANDIDATE_IN_SCOPE；same_query pair 成员与 withdrawal 任务 FINALIZE 但 Top-1 不在标注 verified 集（**已知 P14-B0/B1 的 Scorer/Gate 限制**，与本轮数据构造无关；本轮未调 Scorer/Gate 权重）；
- 资源：Smoke wall **12.59 s**、峰值 RSS **831.1 MB**（CPU）；未用 GPU。

结论：新增任务与现有 Runtime/工具/数据契约兼容；Reserved Test 未执行任何推理。

## 7. 是否可以开始 P14-C0 的 Qwen Base 推理？

**可以（在 train/dev 上，provisional）**，但需遵守：

1. Qwen Base 只使用 **train/dev** 做推理诊断；本轮不训练、不调 Scorer/Gate；
2. **reserved_test 保持封存**：不运行 Qwen、不搜索、不 Gate、不看逐任务预测；后续 freeze 决策前不用于任何调参；
3. 23 条 pending + 原 15 条 pending 仍为 provisional；正式结论需单独标注，并做人审敏感性分析（当前 review checklist 已给出）；
4. Smoke 暴露的 2 条 Top-1 未命中应作为 P14-C0 的诊断输入（检索覆盖/证据不足/Scorer 排序），不能据此修改标签或 Gate 来美化；
5. 进入 P14-C0 前建议：先冻结这一版 120 条数据的 manifest/hash，记录 Qwen Base 配置与 prompt，并保留本报告的 GO 仅为“可开始推理”，不代表推荐质量成立。

## 8. 校验汇总与测试

| 项 | 结果 |
| --- | --- |
| 数据校验 | `pass=true`，failures=[]，warnings=[] |
| 任务数 / 切分 | 80；train 40 / dev 20 / reserved_test 20 |
| 类型分布 | 15/20/16/4/4/4/10/4/3（与目标一致） |
| provenance replay | 80/80 |
| persona/family/pair 跨 split | 0 |
| 近重复 / 与 pilot 重复 | 0 |
| Agent 泄漏 | 0 |
| 原 P14-E 哈希 | 未变 |
| Reserved 推理记录 | 0 |
| 测试 | P14-E2 10 项 + 相关回归（P14-B1/B0/P14-E）共 **45 passed**（27.65 s） |
| 资源 | 构造/校验 CPU；Smoke 12.59 s / 831 MB |

新增源码：`experiments/p14e2/{__init__,build_lite,validate_lite,smoke}.py`；`tests/test_p14e2_lite.py`；数据 Schema/Manifest：`data/p14e2/benchmark/schema_v0.2.json`、`manifest.json`。未修改共享运行时（Scorer/Gate/Memory/Planner/Serving 均为冻结只读）。

## 9. 最终判定

**P14-E2 Lite = GO**（provisional：数据可进入 P14-C0 Qwen Base 的 train/dev 推理诊断；标签仍需人工审核，reserved_test 封存；不代表推荐质量或公开 Benchmark 泛化结论）。

继续条件：Qwen Base 配置/prompt 记录、沿用现有 Gate/Scorer 不调参、reserved_test 不曝光、pending 审核完成后更新结论。
