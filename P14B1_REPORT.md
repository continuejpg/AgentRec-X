# P14-B1 — Evidence-Aware Recommendation Decision Gate

> 范围：仅 P14-B1。新增确定性 Decision Gate、接入 P14-A Executor 的 FINALIZE 钩子、40 条 Pilot Dev 四组对照、5 条真实工具运行时轨迹。
> 未训练模型、未改 P14-E 标签、未扩 Benchmark、未改 Serving/Planner 默认动作、未 commit。

## 0. 一句话结论

9 条错误 Top-1 的具体原因是**“没有决策充分性校验”**，不是排序错误：3 条历史偏好冲突、1 条历史被撤回后无有效判据、5 条需求信息不足；Gate 用 Agent-visible 证据（Scorer 的 hard/preference 状态、合法恢复的 active 偏好、query 硬约束、工具预算）区分 `ALLOW_FINALIZE / ASK / 补商品证据 / 扩大检索 / 有界拒绝`。在**同一冻结候选池**上，Search+Preferences + Gate 把 14 条非推荐任务的错误 FINALIZE 从 **14/14 降到 0/14**，同时 26 条可推荐任务仍全部 FINALIZE、正确 Top-1 保持 **18/26（0 误拦截）**。结论仍为 **pilot-dev provisional**，总判断 **LIMITED**。

## 1. 9 条错误推荐的真正原因

逐条证据（`runs/p14b0/non_recommend_audit.jsonl` 与 `runs/p14b1/candidate_decisions.jsonl`）：

| 类别 | 任务 | 实际证据 | 原因 |
| --- | --- | --- | --- |
| 历史偏好冲突未解决 | 0026/0027/0028 | active 偏好同时包含 `wireless` 与 `wired`（都是 positive），query 无硬约束可裁决 | Scorer 只能按 BM25+匹配打分，选中其中一个；缺少“冲突需要用户裁决”的检查 |
| 撤回后无判据 | 0030 | `wireless` 已被历史显式撤回，active 偏好为 0，query 无硬约束/无属性约束 | 无任何个性化判据仍强行给 Top-1，等于猜测 |
| 当前需求信息不足 | 0031–0035 | active 偏好 0、硬约束 0、query 仅“I need some new earbuds/…” | 同上：缺少“当前请求是否足以区分候选”的检查 |
| 硬约束违规/UNKNOWN | 无 | 这 9 条均无 hard violation/unknown 参与（B 的 top1 hard tier 均为 0） | 不是 Eligibility 问题 |

因此 Gate 的必要规则是：**冲突→ASK；无判据→ASK；硬约束 UNKNOWN→先补证据/再扩检索/有界拒绝；明确违规→禁止 FINALIZE**。

## 2. Gate 使用的 Agent-visible 证据

`recommendation/preference/decision_gate.py`（`decision_gate_v0.1`，纯确定性；无搜索、无标签读取）：

- Scorer 行：`hard_tier`（verified/unknown/violated）、`matched_preferences`、`conflicting_preferences`、`unknown_preferences`、`evidence_refs`（catalogue span + memory_id）；
- 合法 history-state read 的 active 偏好与 provenance（`memory_id/source_turn_id/source_text`）；`inactive` 仅用于审计，不参与 FINALIZE；
- 当前 query 中显式硬约束（agent-visible 文本解析，非 evaluator 字段）；
- 由 active 偏好计算出的冲突对、query 是否有可区分判据、工具预算（`GateBudget`）、`ask_pending`；
- 绝不读取：evaluator-only persona、expected_decision、reference/verified ids、ground truth、任务类型/审核状态。

Gate 输出：`action / allow_finalize / reason_code / reason / selected_candidate_id / candidate_presentable_ids / hard_constraint_status / preference_conflict_status / unknown_fields / evidence_refs / ask_question / bounded_scope / budget_used / gate_version`。

## 3. 允许 FINALIZE 的条件

必须同时满足：

1. 没有未解决的 ASK（`ask_pending=False`）；
2. 没有未解决的 active 偏好冲突；
3. 存在明确判据：当前显式硬约束、或恢复出的 active 偏好、或 query 含有可区分要求（`query_specificity`）；
4. 选定候选的硬约束全部 **verified**（UNKNOWN 不算满足，violated 直接排除）；
5. 选定候选不与显式负偏好冲突；
6. 软偏好 UNKNOWN 不阻塞，但会进入 `unknown_fields`，输出“limited recommendation/不确定说明”（不虚构满足）。

## 4. 触发 ASK 的条件

- `ask_pending`：已有未回答澄清，禁止绕过；
- **active 偏好冲突**（如 wireless vs wired，无当前硬约束可裁决）：问“哪一个优先级更高”；
- **无任何判据**（无 active 偏好、无硬约束、query 仅泛泛请求）：问“最看重什么（用途/预算/必需功能）”；
- 选定候选与显式负偏好冲突（Scorer 未找到无冲突候选）：请用户决定是否接受/继续找。
每次 ASK 都产生真实可回答的问题，而不是“请提供更多信息”这类空话；ASK 后进入等待状态，任何未回答 ASK 不能 FINALIZE。

## 5. 继续查商品 / 扩大检索 / 有界拒绝的条件

- `NEED_MORE_PRODUCT_EVIDENCE`：存在硬约束 UNKNOWN 的候选且商品证据预算未用尽 → 用真实 `product_view` 查事实（不把 UNKNOWN 当满足）；
- `NEED_MORE_CANDIDATES`：证据轮后仍无 verified 候选且检索预算未用尽 → 用扩展 query 再检索一次（真实动作，单独记录，不混入固定池指标）；
- `NO_VERIFIED_CANDIDATE_IN_SCOPE`：预算耗尽仍无 verified 候选 → 只声称“当前已检查的检索范围内没有可验证合格商品”，明确 `bounded_scope=True`，**不断言全目录不存在**；
- `BUDGET_EXHAUSTED`：预算耗尽时保留 UNKNOWN 并不确定推荐。

## 6. Gate 是否真正进入 Runtime？

- 已修改 `recommendation/shopping_companion/planner_bridge.py`（新增可选 `finalize_gate` 钩子与 `max_gate_retries`）：默认 `None` 时旧行为完全不变；安装 Gate 后 `_finalize` 先询问 Gate，不允许时转为 `needs_user_input`（ASK）、下一轮 RERANK/RETRIEVE，或 bounded terminal refusal；未解决 ASK 不得 FINALIZE；有重试上限避免无限循环。
- 真实工具运行时轨迹：`experiments/p14b1/runtime_smoke.py` 在 6 条代表性任务上真实调用 Memory（M9/M5 history-state read）、BM25（P13-R catalogue）、Product Info（真实证据，UNKNOWN 保留）、CandidateLedger、Eligibility（CATEGORY verified）、Preference Scorer、Decision Gate，轨迹见 `runs/p14b1/runtime/`：

| 任务 | 类型 | 结果 | 工具步数 |
| --- | --- | --- | --- |
| 0001 | regular recommend | FINALIZE，Top-1 verified（correct） | 5 |
| 0029 | withdrawal recommend | FINALIZE，Top-1 verified（correct） | 5 |
| 0026 | preference conflict | `awaiting_clarification`（真实问题，未伪造回答） | 5 |
| 0031 | insufficient information | `awaiting_clarification` | 5 |
| 0036 | no verified match | `no_verified_candidate_in_scope`（先补证据/扩检索，预算耗尽后有界拒绝） | 6 |
| 0039 | unverifiable price | `no_verified_candidate_in_scope`（UNKNOWN 保留） | 6 |

资源：runtime smoke wall **12.97 s**、峰值 RSS **830.5 MB**（CPU）；未用 GPU。

**诚实说明**：本轮真实轨迹使用同一 Planner Action 词汇与真实工具/组件，但并非在 P14-A 生产 Runtime 上跑完整 Electronics 任务（需要商品/工具 adapter）；Executor 钩子已实现但未做完整端到端生产运行。因此 `RUNTIME_INTEGRATION` 判为 **LIMITED**，不是 PASS。

## 7. 错误推荐是否减少？

固定候选池（P14-B0 冻结的 BM25 Top-20；Gate 不增加候选、不注入 target）：

| 变体 | 可推荐任务 FINALIZE | 正确 Top-1 | 错误 Top-1 | 不必要 ASK | 不必要拒绝 | 非推荐任务错误 FINALIZE | 正确 ASK | 正确 UNKNOWN 处理 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Search Only, Gate Off | 26/26 | 9/26 | 17 | 0 | 0 | **14/14** | 0 | 0 |
| Search Only, Gate On | 24/26 | 10/26 | 14 | 2 | 0 | **0/14** | 9 | 5 |
| Search+Prefs, Gate Off | 26/26 | 18/26 | 8 | 0 | 0 | **14/14** | 0 | 0 |
| **Search+Prefs, Gate On** | **26/26** | **18/26** | **8** | **0** | **0** | **0/14** | **9** | **5** |

- 错误 FINALIZE：14/14 → **0/14**；正确 Top-1 不下降（18/26 保持）；不必要 ASK/拒绝均为 **0**；硬约束违规 0。
- Search Only + Gate On 的 2 个“不必要 ASK”来自：没有偏好重排时 BM25 首选与显式负偏好冲突，Gate 让用户决定；在 Search+Prefs 路径不存在（说明 Gate 与 Scorer 职责互补，不是互相取代）。
- 尚未减少的 8 个错误 Top-1 均为检索/覆盖/证据问题（4 条 retrieval unreachable 不计入 Ranker 错误；4 条 both wrong 需靠更好候选/证据，不是决策门能修）。

## 8. 正常推荐是否被过度阻断？

- 26/26 可推荐任务全部 FINALIZE；正确 Top-1 保持 18/26；不必要 ASK/拒绝 = 0；
- 排除 15 条 pending 后（16 条 recommend）：Gate On 仍 16/16 finalize、正确 12/16（0.75，与 Gate Off 相同）、非推荐错误 FINALIZE 0/9；说明改进不是靠 pending 任务造成的；
- 软偏好 UNKNOWN 允许带不确定性推荐（`unknown_fields` 显式记录），不要求 17 个属性全部已知，避免过度拒绝。

## 9. UNKNOWN 是否被正确处理？

- 硬约束 UNKNOWN ≠ verified：Gate 先请求真实的 Product Info/再扩大检索；预算耗尽后输出 `NO_VERIFIED_CANDIDATE_IN_SCOPE`，不声称全目录不存在（`bounded_scope=True`）；
- 软偏好 UNKNOWN：不阻止 FINALIZE，但进入 `unknown_fields`，Top-1 输出带“limited recommendation”原因，不虚构满足；
- 商品事实仍只来自 P13-R 文本证据；price/brand/size/material/stock 保持 UNKNOWN（无补全）；
- 发现并在报告中保留：Gate 的硬约束状态目前由 Scorer/证据层产生，Eligibility 只用于类别等结构化约束；统一两个 hard-status 生产者在生产 Runtime 中是后续工作（这也是 RUNTIME_INTEGRATION LIMITED 的原因之一）。

## 10. 是否仍有人工审核依赖？

是：15/40 pending（其中 10/26 recommend）保留，未升级为 human_verified；P14-E evaluator-only 标签哈希测试未变（`test_p14e_labels_and_b0_artifacts_unchanged`）；报告给出全量与排除 pending 的敏感性；结论标注 provisional / pilot-dev。Gate 与 Evaluator 分离实现（Gate 不读标签；评价程序独立读取标签后评分）。

## 11. 是否出现新旧代码污染？

- P14-B1 仅修改 1 个共享文件：`recommendation/shopping_companion/planner_bridge.py`（新增可选 FINALIZE 钩子；`gate=None` 时旧行为不变）；
- 新增：`recommendation/preference/decision_gate.py`、`experiments/p14b1/{__init__,diagnose,runtime_smoke}.py`、`tests/test_p14b1_decision_gate.py`；
- 未修改：Serving V1、SASRec/Two-Tower、P13-R 数据、P14-E 标签/任务、P14-B0 Scorer 权重/结果、Memory（P14-B0 的修复保持）、Planner 默认动作、CandidateLedger/Eligibility 语义；
- 回归：P14-B1 新测试 + 相关 Memory/Planner/Executor/P14-A/P14-E/P14-B0/Serving/Eligibility 共 **174 passed**（38.1 s）；未运行全量 pytest。

## 12. 是否可以继续扩大 Benchmark？

**有条件可以**（P14-B1 = LIMITED）：

1. 先完成 15 条 pending 人工审核（尤其 10 条 recommend），否则数字仍 provisional；
2. 把 Gate 接入生产 Runtime 的完整链路（当前有 Executor 钩子 + 真实工具决策循环，但未在 P14-A 生产 Runtime 上跑完整 Electronics 任务）；
3. 统一 hard-status 生产：Scorer/P14-E 文本证据 vs `CandidateEligibilityEvaluator` 的结构化事实，避免两套状态在未来漂移；
4. 改善候选覆盖（4 条 unreachable）与 grounding（4 条 both wrong），否则 Gate 只能保证“不错误推荐”，不能提升召回；
5. 扩展 200–500 时仍须 persona/模板/同 query group 隔离与独立 reserved-test；本轮不扩。

### 最终判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `DECISION_GATE` | **PASS** | 规则确定、可解释、复用 Scorer/Ledger/证据；14/14 错误 FINALIZE 归零，未误拦正常推荐；测试覆盖 D1 全项 |
| `RUNTIME_INTEGRATION` | **LIMITED** | Executor 钩子已实现、真实工具决策轨迹 6 条正确；但未在生产 Runtime 完整跑 Electronics 任务，hard-status 生产者待统一 |
| `SAFETY_AND_COVERAGE` | **PASS** | 硬约束 UNKNOWN 不冒充满足、明确违规禁止 FINALIZE、有界拒绝不越权断言、ASK 不绕过、预算上限明确；候选覆盖限制如实报告 |
| `EVALUATION_ISOLATION` | **PASS** | Gate 不读标签；标签哈希未变；Gate/Evaluator 分离；pending 保留；无 task_id 特判 |
| `LEGACY_REGRESSION` | **PASS** | 174 项相关测试通过；Serving/Planner 默认动作/旧模型未改；P14-B0 结果保留 |
| **P14-B1** | **LIMITED**（pilot-dev 决策逻辑可用，进入数据扩展前的条件见 §12） | Gate 显著修复错误推荐且不牺牲正常推荐；仍受人工标签、生产 Runtime 集成与候选覆盖限制 |

## 附：产物

- `runs/p14b1/summary.json`（四变体、pending 敏感性、reachability）
- `runs/p14b1/candidate_decisions.jsonl`（逐任务四变体决策与 gate trace）
- `runs/p14b1/runtime/`（6 条真实工具轨迹 + summary）
- 源码差异：[`P14B1_SOURCE_DIFF.txt`](P14B1_SOURCE_DIFF.txt)
- 未修改 `data/p14e/benchmark/*`；未 commit / 未用 GPU / 未操作 AutoDL
