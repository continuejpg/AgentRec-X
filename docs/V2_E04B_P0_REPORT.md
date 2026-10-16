# AgentRec-X V2 / E-04B P0 — 规则 Agent（B2）离线行为基线报告

> 状态：**PASS**。B2 = 冻结协议中的「规则工具选择」arm，其**真实**入口是被测过的
> `recommendation/evaluation/agent/` 控制面；本轮在其上产出**可复现的行为基线** ✓。
> 本轮**未**加载 SASRec / Two-Tower / TIGER 权重、**未**启动 TIGER Worker、**未**调用在线 LLM、
> **未**联网、**未**安装依赖、**未**修改任何**已跟踪**文件、**未**执行任何 Git 写操作 ✓。

---

## 1. 初始 HEAD 与工作树

```text
branch : v2/e04a-pass-checkpoint      commit : b3af9da79a87aada73cf389ee4c22df5f2169d54
开始时：7 个未跟踪文件（Gate A 4 个 + E-04B-P 3 个）、0 modified —— 与上一轮报告一致 ✓
结束时：9 个未跟踪文件（新增 experiments/e04b_p0_rule_baseline.py、tests/test_v2_e04b_p0.py）、0 modified ✓
```

## 2. B2 实际代码入口及可执行性

| 项 | 事实 |
|---|---|
| 规则策略实现（真实源码） | `recommendation/evaluation/agent/runner.py:216-513`（确定性策略库：`_RecommendThenFinish`、`_ConstraintFilteringPolicy`、`_RecoverAfterEmpty`、`_EvidenceSeekingPolicy`、`_SimilarItemRecoveryPolicy`、`DecidingOncePolicy`）+ `build_adaptive_policy:554` |
| 评测入口（真实可调用） | `run_case:1038`、`run_suite:1052`、`CaseRunner:661`、`CaseOutcome:625`、`SuiteReport:996` |
| 指标对象 | `recommendation/evaluation/agent/metrics.py:38 TrajectoryMetrics`（24 个字段 ✓） |
| 任务登记册 | `recommendation/evaluation/agent/cases.py:46 EvaluationCase`、`:551 load_cases()` → **18 个 case**（含 `history-driven`、`novel-product-need`、`missing-requirement`、`hard-constraint`、`task-override`、`empty-source-recovery`、`premature-finish-recovery` 等） |
| 控制器工厂与 reasoner | 复用 **既有、已被测试**的 `tests/test_evaluation_agent_plane.py::suite_factory` / `build_reasoner` ✓（P0 **未**新建第二套 harness ✓） |
| 决策来源 | **全部由规则产生** ✓：`policy_factory=build_adaptive_policy`，**不构建任何 ModelClient**；本轮 `llm_calls = 0` ✓（记录在结果文件 ✓） |
| 不加载权重时可执行的部分 | **整条真实控制流程**：`LoopController` + `ActionValidator` + `CandidatePlane` + `GroundingVerifier` + `CompletionGuard` + 预算/终止 ✓（外壳为合成夹具 ✓） |
| 只能用受控 Fixture 替代的部分 | 商品目录与顺序推荐结果（`CANDIDATE_ROWS` 合成夹具 ✓）、history 源失败/空结果按 case 声明注入 ✓ |
| 历史归档能提供的证据 | `runs/m3_execution/{agent_behavior.jsonl,agent_rankings.jsonl}`（**过去**运行的录制 ✓，本轮**未**用作新测量 ✗） |
| 本轮可做的**新**测量 | 行为类指标（§5 全部 ✓）|
| 本轮**不可**测量 | HR/NDCG、任务成功率、偏好满足率、Agent 优于固定策略 ✗（无真实相关性/适配标签，且为合成夹具 ✓） |

## 3. 行为场景、输入合同与来源标签

场景按 `cases.py` 的 `category` 归类（**受控合成任务**，非真实用户意图 ✗）：

| 冻结场景 | 覆盖该场景的 case（category） | 输入合同 | 来源标签 |
|---|---|---|---|
| A 历史充分、无文本需求 | `1_history_driven`（`history-driven` 等） | 夹具历史（`CANDIDATE_ROWS` ✓）+ 声明式任务文本 | 合成夹具 ✓ |
| B 明确文本请求 | `2_explicit_novel_need`（`novel-product-need`） | 声明式需求文本、无充分历史 | 合成夹具 ✓ |
| C 混合信息 | `4_hard_constraint`、`5_task_override` 等 | 历史 + 当前需求 + 硬约束声明 | 合成夹具 ✓ |
| D 候选不足 / 零候选 | `empty-source-recovery`（`recovery-required` 走**空 history**）+`premature-finish-recovery` | 确定性轻量工具制造的不足/零候选 | 合成夹具 ✓；**规则策略自带恢复能力** ✓，**不是** E-03B 的 LLM Replanning ✗ |
| E 非法来源与预算 | `unauthorised`/预算类 case（`forbidden_actions`、`max_tool_calls`） | 声明式禁用动作 + 预算上限 | 合成夹具 ✓ |

## 4. 每个场景的真实动作及终止轨迹（本轮新执行，18 例，墙钟 **0.79 s**）

```text
history-driven        : tool_calls=1 steps=2 → COMPLETED
novel-product-need    : tool_calls=1 steps=2 → COMPLETED
missing-requirement   : tool_calls=0 steps=1 → ASKED_CLARIFICATION      （规则选择"先澄清" ✓）
hard-constraint       : tool_calls=1 steps=6 → COMPLETED
（完整 18 行逐例轨迹见 runs/e04b_p0/p0_baseline.json 的 cases 数组 ✓）
终止分布：COMPLETED 15 / ASKED_CLARIFICATION 1 / UNAUTHORISED_ACTION 1 / COMPLETION_REFUSED 1 ✓
逐例通过：17 / 18；唯一未通过 = premature-finish-recovery（**既有、已被测试显式记录的缺口** ✓，
          见 tests/test_evaluation_agent_plane.py:594 附近断言 failed ⊆ {premature-finish-recovery} ✓）
```

## 5. 指标定义、分子、分母（全部为**新测量** ✓，`runs/e04b_p0/p0_baseline.json`）

| 指标 | 定义 | 分子 / 分母 | 值 |
|---|---|---|---|
| 合法工具选择率 | 提议通过 `ActionValidator` 的 case / 全部 case | 17 / 18 | **0.944444** |
| 工具调用次数 | 真实工具调用总数 / 每 case | 33 / 18 | **1.8333** |
| **实际执行来源集合** | **该 harness 的结果对象未暴露逐来源明细** | — | `not_exposed_by_harness`（改由 `required_actions_met` / `forbidden_actions_used` 核对 ✓） |
| 工具失败 / 空结果次数 | 失败、空结果 / 工具调用总数 | 0 / 33、0 / 33 | 0.0 ✓ |
| **Grounded 候选数量** | **未暴露** | — | `not_exposed_by_harness`（改由 `hallucinated_candidates == 0` 核对 ✓） |
| 预算合规率 | `steps ≤ max_steps 且 tool_calls ≤ max_tool_calls` 的 case / 全部 | 18 / 18 | **1.0** |
| 终止状态分布 | 见 §4 | 18 / 18 | 15/1/1/1 |
| 规则决策与轨迹一致性 | 满足声明必需动作且未使用禁用动作的 case / 全部 | 17 / 18 | **0.944444** |
| 卫生计数 | 多余调用 / 约束违反 / 未强制检查 / 幻觉候选 | 0 / 0 / 0 / 0 | ✓ |

**口径纪律**：所有比率的分子分母都写进结果文件 ✓；`legal_tool_selection` **只表示"合法"**，
**不表示"最合适"** ✗（无人工适配标注）；18 例合成夹具的指标**不得**外推为真实场景表现 ✗。

## 6. Fixture / 真实行为 / 历史回放的明确区分

| 记录项 | 值 |
|---|---|
| `data_kind` | **`synthetic_fixture`** ✓（商品目录与顺序推荐为受控夹具；非真实交互 ✗） |
| `measurement_kind` | **`new_execution`** ✓（本轮真实执行 ✓，非回放 ✗） |
| 历史回放 | 本轮**未使用**；归档 `runs/m3_execution/*.jsonl` 仅登记为可用回放证据 ✓ |
| 真实用户历史 | 本轮**未读取**任何真实用户序列/标识 ✓（P0 不走 processed 数据） |

## 7. 工具预算、步骤预算及调用成本

```text
声明的 arm 预算（冻结协议 ARMS["B2.rule_tool_selection"]）：tool_calls ≤ 2，llm_calls = 0
逐 case 预算：由 cases.py 的 max_steps / max_tool_calls 声明 ✓，实际 18/18 合规 ✓
实际：tool_calls 总 33（均值 1.83/case），steps 合规 ✓，llm_calls = 0 ✓，墙钟 0.79 s ✓
（未记录内存峰值：本 arm 不加载权重、无大索引，内存不是其约束 ✓）
```

## 8. 当前不能计算的任务质量指标（明确列出，且**未被**伪造）

`task_success_rate`、`preference_satisfaction_rate`、`hr_at_k`、`ndcg_at_k`、
`agent_beats_fixed_policy` —— 结果文件中写为 `not_claimed` ✓。
原因：① 无真实自然语言请求及其相关性标注（E-04B-P §10 的标注缺口 ✗）；
② 本轮为合成夹具（无真实目录/权重 ✗）；③ B2 是**规则**策略，其"自主性增量"必须与 B1/B3/B4 在
**同预算同 case** 下比较才能声称，而 B3/B4 需 LLM 授权（当前额度 4/4 已用尽 ✗）。
可报告的替代结论仅限**受控夹具下的行为正确性**（§5 ✓）——测试已断言这些质量指标**不在**输出中 ✓。

## 9. 与冻结协议的一致性

* 复用 `experiments/e04b_protocol_v2.py` 的机读定义与校验：`assert_same_case_ids`（同一 case 集合 ✓）、
  `metric_is_computable`（无标签时 HR **不可计算** ✓，脚本内断言 ✓）、`ARMS[B2]` 预算与
  `arm_status == "EXECUTABLE_OFFLINE"` ✓（测试断言 ✓）。
* 场景/K/成本定义/防泄漏规则**未改动** ✓；`K` 未使用（本 arm 无 relevance 指标 ✓）。
* 防泄漏：无隐藏目标进入请求 ✓（不改请求构造）、不用 `test_target` 构造查询 ✓（本 arm 无查询构造）、
  不从正例反选工具 ✓（策略由 `build_adaptive_policy` 依 case 声明决定 ✓）、不读其他 Session Memory ✓、
  不从未来交互造偏好 ✓、合成反馈未标为真实 ✓、同一 case 的可用工具与预算固定 ✓。

## 10. 修改文件与测试数量

| 文件 | 说明 |
|---|---|
| `experiments/e04b_p0_rule_baseline.py`（新增） | P0 脚本：复用既有 runner/夹具，产出行为指标（含分子分母）、协议校验与结果文件 |
| `tests/test_v2_e04b_p0.py`（新增，4 项） | 比率口径、`not_claimed` 纪律、arm 状态与预算、录制基线的完整性（含既有缺口登记 ✓） |
| `runs/e04b_p0/p0_baseline.json`（新增，被忽略目录 ✓） | 完整结果（arm/协议/seed/预算/来源菜单/逐例轨迹/指标） |
| 已跟踪文件 | **0 修改** ✓（`git status` 无 `M`，`git diff --stat` 为空 ✓） |

```text
本轮测试：tests/test_v2_e04b_p0.py + test_v2_e04b_protocol.py + test_v2_gate_a_protocol.py → 22 passed ✓
§8 相关回归：上列 + test_evaluation_agent_plane.py + test_agent_control_plane.py +
            test_llm_agent_policy.py + test_control_stage2_candidates.py +
            test_candidate_reference_boundary.py + test_constraint_eligibility.py
            → 230 passed, 1 failed（见下）
```

⚠️ **该 1 项失败与本轮无关，是既有问题** ✗：`test_evaluation_agent_plane.py::test_every_implemented_stage_is_documented_and_marked`
断言 `docs/ARCHITECTURE.md` 含字符串 `'## 7. Candidate plane (Stage 2, IMPLEMENTED)'`，
而该文档实际写作 `## 7. Candidate plane (Stage 2) — RESEARCH ONLY`（`docs/ARCHITECTURE.md:511`）✓。
证据表明非本轮引入：**本轮未修改任何已跟踪文件**（`git status` 无 `M`、`git diff --stat` 为空 ✓），
且该断言只涉及架构文档的措辞 ✓。**未**自行修改文档或放宽断言 ✓（需你决定是更新文档还是更新断言）。

## 11. 结束 HEAD、git status

```text
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54（v2/e04a-pass-checkpoint，未变 ✓）
status : 9 个未跟踪文件（本轮新增 2 个），0 modified、0 deleted
         未执行 add/commit/push/reset/restore/clean/stash/checkout ✓；未触碰 .env ✓
```

## 12. 最终结论

**P0 = PASS。**

```text
B2 的真实规则策略入口可执行            ✓（run_suite + build_adaptive_policy，18 例 0.79 s）
受控场景产生合法且可追踪的选择与轨迹    ✓（逐例 tool_calls/steps/terminal 已记录；预算 18/18 合规）
统一记录行为、预算与成本指标            ✓（9 类指标，全部带分子/分母；llm_calls=0、工具 33 次、耗时已记录）
不伪造推荐质量或 Agent 增量收益         ✓（not_claimed 明确列出，测试断言不得出现）
```

**未完成/未声称**：真实推荐质量（Layer A 的真实权重指标属 Gate B ✗）、任务成功率与偏好满足率（需标注 ✗）、
B3/B4 的自主性增量（需 LLM 授权 ✗）、`premature-finish-recovery` 这一**既有**行为缺口（已被既有测试登记 ✓，本轮如实标注 ✓）。
**停止点**：不进入 Gate B、不加载任何模型权重、不调用在线 LLM ✓。

---

# 13. P0-CLOSE 更正说明（corrigendum）

> 本节**只纠正描述与审计字段的真实语义** ✓。**未**修改 §1–§12 的任何数值、**未**重写规则策略、
> **未**改变任何失败案例的结论 ✓；冻结的原始结果文件 `runs/e04b_p0/p0_baseline.json` **保持原样** ✓。

## 13.1 §7 文档错误的根因

`docs/ARCHITECTURE.md` 的**平面级**小节标题沿用了**轨道级**的标记：整个 OFFLINE LAB 轨道确实是研究轨道
（`docs/ARCHITECTURE.md:190` `## 0b … (RESEARCH ONLY)`、`:430` `## 6 … RESEARCH ONLY` ✓），
但 §7–§12 六个**平面**各自都已有实现代码与专属测试，标题却仍写着 `— RESEARCH ONLY` ✗，
与 `tests/test_evaluation_agent_plane.py:686-691` 的文档契约（要求 `(…, IMPLEMENTED)`）冲突 ✗。
根因：**平面实现状态与轨道状态的标记混用**（不是能力缺失，也不是测试过严）。

| 项 | 修正前 | 修正后 |
|---|---|---|
| §7（Candidate plane） | `## 7. Candidate plane (Stage 2) — RESEARCH ONLY`（`:511`） | `## 7. Candidate plane (Stage 2, IMPLEMENTED)`（`:511`）+ 3 行状态注记（`:513-516`） |
| §8（Interaction & personalization） | `… (Stage 3) — RESEARCH ONLY`（`:582`） | `… (Stage 3, IMPLEMENTED)`（`:584`） |
| §9（Reasoning plane） | `… (Stage 4) — RESEARCH ONLY` | `… (Stage 4, IMPLEMENTED)`（`:697`） |
| §10（Model-driven policy） | `… (Phase 1) — RESEARCH ONLY` | `… (Phase 1, IMPLEMENTED)`（`:795`） |
| §11（Hard-constraint enforcement） | `… (Phase 2) — RESEARCH ONLY` | `… (Phase 2, IMPLEMENTED)`（`:917`） |
| §12（Evaluation planes） | `… (Stage 6) — RESEARCH ONLY` | `… (Stage 6, IMPLEMENTED)`（`:1064`） |
| **保留**（轨道级 / 未实现项） | `§0b`、`§6` 的 `RESEARCH ONLY` ✓；§7 内 `### Not implemented (documented, not claimed)`（`:567`）**原样保留** ✓ | 未改动 ✓ |

**实现证据（file:line）**：Candidate Plane `recommendation/control/candidate_plane.py`（21,648 B）
经 `recommendation/demo/agent_service.py:51-52,527` 接入真实服务 ✓，契约测试 `tests/test_control_stage2_candidates.py` ✓；
§8 `demo/memory_ranking.py` + `tests/test_control_stage3_interaction.py`、`test_demo_memory_ranking.py` ✓；
§9 `demo/product_knowledge.py`、`demo/explanation.py` + `tests/test_product_knowledge_rag.py` ✓；
§10 `demo/llm_policy.py` + `tests/test_llm_agent_policy.py` ✓；§11 `tests/test_constraint_eligibility.py` ✓；
§12 `evaluation/agent/runner.py`（47,971 B）+ `tests/test_evaluation_agent_plane.py` ✓。
**未**删除或放宽任何测试断言 ✓。

## 13.2 P0 行为指标口径更正（五问五答）

**Q1「17/18」到底是什么？** 它**不是**逐提议的 Validator 合法率 ✗，而是**用例声明的动作契约合规率** ✓：
`tool_selection_correct = not required_missing and not forbidden_used`
（`recommendation/evaluation/agent/metrics.py:260`）——即"该 case 声明的必需动作全部出现 **且** 未使用任何被声明禁止的动作"。
该口径**不说明**所选工具是否"最合适" ✗（那需要人工适配标注，本仓没有 ✗）。
结果文件中该键已由 `legal_tool_selection` 更名为 **`declared_action_contract_compliance`**（17/18 = 0.944444，数值未变 ✓）。

**Q2 `UNAUTHORISED_ACTION` 是谁产生的？** 是**规则策略依 case 自身声明行动、被 `ActionValidator` 拒绝** ✓，
不是 Fixture 直接写死的终态 ✗，也不是规则主动"想违规" ✓。两处证据：
① `missing-requirement` 声明 `forbidden_actions=["recommend_from_history"]`，规则**避开**它并转为 `ASKED_CLARIFICATION`
（tool_calls=0, steps=1, passed ✓）——负向声明的正向用法 ✓；
② `premature-finish-recovery`（category `5a_guarded_completion`）在 history 源抛错时尝试推进，
被校验/守卫**拒绝**，终态 `UNAUTHORISED_ACTION`（tool_calls=3, steps=3, passed=false ✗）。

**Q3 `premature-finish-recovery` 失败的准确行为？** 该 case 由既有夹具工厂注入
`history_source_fails=True`（history 引擎抛错 ✓）→ 规则尝试在证据不足时推进/结束 →
`CompletionGuard`/`ActionValidator` **正确拒绝**了过早完成 ✓，但该 case 期望的是另一条**恢复路径**，
因此它**未满足声明的期望**并被计为未通过 ✗。即：**安全行为正确、恢复期望未达成** ✓；
这是**既有**且已被 `tests/test_evaluation_agent_plane.py` 显式登记的缺口（该测试断言
`failed ⊆ {"premature-finish-recovery"}` ✓），本轮**未**改变该结论 ✓。

**Q4 为什么 `empty-source-recovery` 与"工具空结果 0/33"同时出现？** 因为 **0/33 是幽灵零值** ✗：
`TrajectoryMetrics` **没有** `tool_empty_results`（也没有 `tool_failures`）字段（字段表见
`recommendation/evaluation/agent/metrics.py:38-110` ✓），我上一轮用 `getattr(..., 0)` 取默认值 0 ✗ ——
它**从未被测量过** ✗。真实的空源证据来自**用例声明 + 轨迹** ✓：
`recovery-required` 类 case 以**空 history 引擎**驱动（`rows=()` ✓），首个来源返回 **0 候选**（合法空结果，非错误 ✓），
规则随即切换来源并完成 ✓：`empty-source-recovery` tool_calls=1/steps=2/COMPLETED、
`empty-source-recovery-required` tool_calls=2/steps=3/COMPLETED、`similar-item-recovery` tool_calls=2/steps=3/COMPLETED ✓。
该二项现已改标 **`not_exposed_by_harness`** ✓（并写明替代核对方式 ✓），不再报 0/33 ✗。

**Q5 哪些计数来自真实 Trajectory，哪些只是 CaseRunner 的断言？**

| 类别 | 字段 |
|---|---|
| **来自真实轨迹的测量** ✓ | `tool_calls`、`steps`、`unnecessary_tool_calls`、`terminal_outcome`，以及由运行自身记录派生的 `hallucinated_candidates`、`constraint_violations`、`constraint_not_enforced`、`memory_effect_correct` |
| **CaseRunner 的用例级断言结果**（非逐步测量）▲ | `tool_selection_correct`、`required_actions_met`/`required_actions_missing`、`forbidden_actions_used`、`terminal_allowed`、`reached_completion`、`unverified_completion` |
| **该 harness 未暴露** ✗ | `executed_sources`、`grounded_candidates_total`、`tool_failures`、`tool_empty_results`（现一律标 `not_exposed_by_harness`，**禁止**再以 0 呈现 ✗） |

## 13.3 更正产物

| 文件 | 说明 |
|---|---|
| `runs/e04b_p0/p0_metric_semantics.json`（新增） | 机读更正记录：改名映射、幽灵零值名单、指标来源分类、四类 case 证据、未变结论 |
| `runs/e04b_p0_corrected/p0_baseline.json`（新增） | 用更正后的字段名重跑一次的基线（**新目录，未覆盖**冻结文件 ✓），数值与冻结版一致（17/18 ✓） |
| `docs/ARCHITECTURE.md`（修正） | §7–§12 标题改为 `IMPLEMENTED` ✓，保留轨道级与未实现项标记 ✓ |
| `experiments/e04b_p0_rule_baseline.py`（修正） | 指标命名与"缺字段不得报 0"的纪律 ✓（不改策略、不改夹具 ✓） |

冻结文件 `runs/e04b_p0/p0_baseline.json` 仍保留原键名与原始数值 ✓（可追溯 ✓）。
