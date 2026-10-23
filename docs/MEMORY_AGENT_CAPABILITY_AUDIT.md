> **审计快照（2026-10-16，HEAD `4facd07` 前）。** 这是一次只读核验的结论，**不是当前架构说明**。
> 当前入口见 [`ONBOARDING.md`](ONBOARDING.md)，架构事实见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
> 其中的 L1–L4 证据等级与行号引用为当时状态，之后的精排/Agent 集成交付未回填到本文。

# MEMORY / AGENT 能力核验矩阵（只读源码核验 · 分支 v2/e04a-pass-checkpoint · HEAD 4facd0712e8355841e41810621fa6460a37e663a）

> 证据等级：L0 PLANNED / L1 CODED / L2 CONNECTED / L3 TESTED / L4 DEMONSTRATED（受控多轮行为证据）。
> 每条均注明**验证环境**：确定性规则执行 / 真实模型 / 真实 LLM 调用 / Stub 测试。

## 1. 能力核验矩阵

| 能力 | 代码入口 | 实际调用链 | 测试证据 | 等级 | 能否写入简历 |
|---|---|---|---|---|---|
| Structured Memory（本体 + 容器 + 服务） | `recommendation/memory/schemas.py:78,105,112,134,211,264`（`PreferenceKind/Polarity/Mode/Status`、`PreferenceCandidate`、`PreferenceRemoval`）；`memory/service.py:168,182,231`（`PreferenceMemoryService`、`process_turn`） | 见链 A | `tests/test_memory_service.py`、`test_memory_store.py` | **L3** ✓ | **可以**（措辞见 RESUME_CLAIMS） |
| 长期偏好（跨请求/跨重启持久化） | `demo/runtime.py:93,132-138` → `preference_memory.sqlite3`，环境变量 `AGENTRECX_MEMORY_DB`（`MEMORY_DB_ENV_VAR`） | 链 A 末尾写入 store | 同上 + `tests/test_memory_store.py` | **L3** ✓ | **可以**（强调"SQLite 持久化 + 可配置路径"） |
| 当前轮 / 当前任务状态 | `recommendation/evaluation/agent/cases.py:46`（用例契约）、`control/context.py`（PolicyContext） | 链 C | `tests/test_agent_state.py`、`test_agent_control_plane.py` | **L3** ✓ | 可以（表述为"任务/轮次状态"，非长期记忆） |
| 否定偏好与排除条件 | `PreferencePolarity.avoid`（`schemas.py:105`）+ `PreferenceMode.remove`（`:112`）+ 匹配器 `preference_matching/matcher.py` | 链 B（过滤/降权） | `tests/test_preference_matching.py`、`test_preference_reranking.py` | **L3** ✓ | 可以 |
| 偏好来源证据 | `PreferenceCandidate.source_text`（`schemas.py:211`，`_clean_source_text:201` + 密钥样式拦截 `contains_secret_like_text:171`） | 链 A | `test_memory_service.py` | **L3** ✓ | 可以（"保留来源片段并做敏感信息拦截"） |
| 更新版本 / supersede / 撤销 | `memory/service.py:120 _supersedes`、`PreferenceStatus.superseded/removed`（`schemas.py:134`） | 链 A | `test_demo_preference_pipeline_fixes.py`、`test_v2_product_feedback.py`（23 项，商品身份级反馈 promote/demote/replace/retract） | **L3** ✓ | 可以 |
| 跨用户隔离 | 以 `user_key`/session 作用域存取（`memory/service.py` + `demo/runtime.py`）；另有 `identity` 守卫 `memory/extraction.py:158,167,172,395` | 链 A | `tests/test_agent_state_isolation.py` | **L3** ✓ | 可以 |
| Memory-conditioned BM25（检索改写） | `memory/query_adapter.py::project_active_preferences`（被 `demo/agent_service.py:553` 调用）；策略开关 `demo/agent_policy.py:156,166 use_memory` | 链 B 第 1 段 | `tests/test_memory_query_adapter.py`、`test_memory_bm25_retrieval.py` | **L3** ✓ | 可以（"按需把生效偏好折入词法查询"） |
| 商品属性证据（来自可信目录） | `demo/product_knowledge.py` + `control/grounding.py:63,67,164,219`（`identity_map` 契约）；证据视图 `KnowledgeEvidenceView{parent_asin,item_id,field_path,value,source='catalog_metadata'}` | 链 B 第 3 段 | `tests/test_product_knowledge_rag.py` + Gate B/C 的合法性检查（非法/PAD/重复 = 0） | **L3** ✓ | 可以（强调"字段级证据 + 目录来源，非模型自述"） |
| Memory-aware 候选过滤 | `demo/agent_service.py:512-525`（注入 `preference_matcher`/`constraint` 平面）；`preference_matching/matcher.py:458`（PRODUCT 身份匹配） | 链 B 第 2 段 | `test_preference_matching.py`、`test_constraint_eligibility.py` | **L3** ✓ | 可以 |
| Memory-aware Ranking | `demo/agent_service.py:206,246-249,512-525`（`memory_ranking_enabled`）、`demo/memory_ranking.py` | 链 B 第 2 段 | `tests/test_demo_memory_ranking.py`、`test_demo_llm_agent_memory_acceptance.py`；**并有一次真实 Provider 端到端证据**：E-04A 运行中 planner 上下文携带受限投影 `[{kind: color, value: black}]`，最终排序中 `k-black` 由第 2 位升至**第 1 位** ✓ | **L4** ✓ | **可以**（注明环境：真实模型 + 真实 Provider 1 次调用，Memory 由 M9 直接写入） |
| Agent 工具选择（确定性策略） | `recommendation/evaluation/agent/runner.py:216-513` 策略库 + `demo/agent_policy.py` | 链 C | P0 基线：18 例、100% 预算合规、0 约束违反（`runs/e04b_p0/`） | **L4** ✓（行为基线 ✓） | 可以（"确定性策略 + 合法性校验"） |
| Agent 工具选择（LLM 提议） | `demo/tool_selection.py:214` | 链 C | `tests/test_llm_tool_selection.py`、`test_llm_agent_policy.py`（Stub）+ **E-04A 真实调用 1 次**：模型选择 `["catalog_search","history"]`，经 `ActionValidator` 通过并实际执行 ✓（额度 4/4 已用尽，**不可复现** ✗） | **L3→L4（一次性）** ◐ | 可以，但必须写"真实 Provider 已一次性验证，当前额度用尽" |
| Agent 有界重规划（至多一次） | `demo/observation.py` + `demo/replan_policy.py`（Gate 判定 + 追加来源） | 链 C | `tests/test_v2_observation_replan_gate.py`（16）、`test_v2_bounded_replanning.py`（24）+ **E-04A-R1 真实调用 1 次**：`status=appended_candidates`、追加来源真实执行、**同一份预算**（tool 2/4、steps 3/6）✓ | **L4** ✓（1 次真实 + 离线执行型测试） | 可以 |
| **Memory → Agent 动作变化** | 规划器读取的是**受限价值级投影**（`project_active_preferences` ✓），但**策略层没有以 memory 为条件选择动作** ✗ | 链 C 中**中断**：`agent_policy` 的动作集合由「请求声明的来源 + 预算 + 观测」决定；`use_memory` 只影响**词法查询**，不改变动作选择 | 无成对对照测试 ✗ | **EVIDENCE_MISSING** ✗ | **不可以**（除非改写为"Memory 影响检索/排序，而非规划"） |

## 2. 覆盖度小结（按 L 级）
L4：Memory-aware Ranking ✓、确定性工具选择行为基线 ✓、有界重规划 ✓（含 1 次真实 Provider）
L3：结构/持久化/否定/来源/版本/隔离/BM25 改写/属性证据/候选过滤 ✓
EVIDENCE_MISSING ✗：**Memory 改变工具计划**这一条

## 3. 报告末尾五问
1. **Memory 实际实现到哪一层？** 到"**结构化偏好本体 + SQLite 持久化 + 跨轮更新（add/replace/remove/supersede）+ 跨用户隔离 + 来源证据**"层 ✓（L3），并在**排序**上达到 L4 ✓。
2. **能否影响检索、排序或 Agent 动作？** 检索 ✓（折入词法查询，L3）、排序 ✓（L4，含一次真实 Provider 端到端）、**Agent 动作 ✗（EVIDENCE_MISSING）**。
3. **M-02 / M-03 / M-04 等级？** M-02 = **L3（部分 L4）** ✓；M-03 = **L3，其中 Memory-aware Ranking 为 L4** ✓；M-04 = **L3–L4（有界重规划 L4；LLM 提议一次性 L4、当前不可复现）** ◐，其中"Memory-aware 规划"为 **EVIDENCE_MISSING** ✗。
4. **最值得强调的两项真实技术贡献：** ① **结构化偏好记忆 + 跨轮更新 + 持久化与隔离**（L3→L4 ✓）；② **受控工具预算下的合法性校验与"至多一次"有界重规划**（含 1 次真实 Provider 验证 ✓）。
5. **不能写入简历：** "Memory 改变 Agent 规划/工具选择" ✗、"线上大规模验证" ✗、"真实 LLM 持续在线闭环" ✗、"TIGER 质量准入" ✗。
