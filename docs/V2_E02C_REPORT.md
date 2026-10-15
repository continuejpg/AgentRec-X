# AgentRec-X V2 / E-02C — Memory + 多源融合 + 排序 + Product RAG 联合验收

> 前序：[`docs/V2_E02A_REPORT.md`](V2_E02A_REPORT.md)（Memory-aware Ranking）、
> [`docs/V2_E02A_V_REPORT.md`](V2_E02A_V_REPORT.md)（多来源累计候选与 RRF 呈现）、
> [`docs/V2_E02B_REPORT.md`](V2_E02B_REPORT.md)（Product Knowledge RAG 与可信解释）。
> 结论：**PASS** —— 同一 Session 的真实 Memory 更新 → 后续 Agent 合法选择并执行来源 → 多源候选正确融合 →
> 当前 ACTIVE Memory 对真实候选可信重排 → 对最终展示商品检索目录知识 → 只输出经过验证的商品解释；
> 更正/撤销、双会话隔离、无证据不编造、RAG 失败不影响推荐、旧模式与 Serving 不回归全部成立。
> **本轮未修改任何生产代码**（验收为主），并如实报告 1 个非阻塞的展示层发现（§10）。

---

## 0. 一句话结论

```text
同一 Session，两个真实来源（history capability + catalog_search plane），4 件稳定身份商品
（j-red / j-black / j-blue / j-plain，其中 j-plain 无颜色属性，j-red 被两个来源同时召回）：

Turn 1 无 Memory  fused=[j-red, j-black, j-blue, j-plain]  final 同（applied=false, moved=0）
Turn 3 prefer black fused 不变 → final=[j-black, j-red, j-blue, j-plain]（moved=2）
        black: M10A match(details.Color=black) + 解释引用 j-black:details.Color="black"（带"符合你所陈述的偏好"）
Turn 5 prefer blue（supersede black）→ final=[j-blue, j-red, j-black, j-plain]（moved=3）
        blue: match(details.Color=blue) + 解释引用 j-blue:details.Color="blue"；black 不再 match
Turn 7 REMOVE      final 回到 fused 基线；所有解释不再含任何"符合偏好"表述
全程：商品身份、sasrec_score/score_kind、fusion_sources、original_order、偏好匹配证据字段均未被 RAG 改动
```

## 1. 初始 HEAD、工作树

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall   →  26 modified + 42 untracked
```

本轮开始前阅读了 `recommendation/demo/runtime.py`（`llm_agent_turn` / `explain_products`）、
`agent_service.py`（`run_selection` / `_execute_plan` / `_memory_ranking`）、`agent_trajectory.py`
（`presentable_candidates` / `build_recommendations_view`）、`memory_ranking.py`、`explanation.py`、
`product_knowledge.py`、`control/loop.py`（`_record_candidate_ranking` / `_fuse_presentable`）与
`agent_chat_adapter.py` 的现有实现。未执行任何禁用 git 操作；未安装依赖；未训练/加载模型；
未调用在线 Provider（全部使用仓库自带 `ScriptedModelClient` 与确定性轻量工具）；未改 Serving。

## 2. 完整联合执行顺序（源码行号核对）

| # | 环节 | 位置与事实 |
|---|---|---|
| 1 | 合法 Memory 身份绑定 | `sessions.py:270` 由 session UUID 派生 `user_key="demo-session:<session_id>"`；`demo_routes.py:621` 传入 `runtime.llm_agent_turn(user_key=session.user_key, …)`。客户端与模型都无法指定（请求体带 `user_key` → 422，测试断言）。 |
| 2 | 写前 ACTIVE Memory 读取 | `runtime.py:471` `_active_preference_snapshot(user_key)` → `:472` `project_active_preferences(snapshot)`（值级投影）。 |
| 3 | ToolSelectionPlanner | `:475-481` `LLMAgentTurn(..., preference_context=planner_preference_context(projection))` → `:483` `turn.plan()`：一次真实 `ModelRequest`（`user_request` / `available_tools` / `tool_budget` / `context.active_preferences`≤10）→ 校验后的 `SourceSelection`。 |
| 4 | 本轮 Memory 写入 | `:528` `_process_turn_memory(user_key, message, turn_id)`，服务端 turn id、每轮一次；**在规划之后、执行之前**。 |
| 5 | Controller 实际执行 | `:537` `turn.run(preference_snapshot=snapshot)` → `run_selection` → 合法 plan；`recommend_from_history` 走 accepted capability，`search_catalog` 走 plane（`ActionValidator` 校验 + `GroundingVerifier` 记录）。 |
| 6 | grounded 候选累计 | `control/loop.py:979-1057` `_record_candidate_ranking` / `_fuse_presentable` / `presentable_tool_result`（E-02A-V）。 |
| 7 | RRF 融合与 k 截断 | `agent_trajectory.presentable_candidates` → `build_recommendations_view(..., ranking=, limit=payload.k)`：排序之后才截断。 |
| 8 | Memory-aware Ranking | `agent_service._memory_ranking`：呈现集合 → M8 enricher → M10A match → M10B rerank，使用**写前**快照。 |
| 9 | Product Knowledge Retrieval | `runtime.py:605 explain_products` → `product_knowledge.retrieve_product_knowledge`（按身份查表 + accepted `retrieve_evidence`，仅最终展示商品）。 |
| 10 | Claim 生成与验证 | `explanation.generate_explanations`（一次 `ModelRequest`）→ `validate_explanations`（身份/evidence_id/禁跨商品/field+value 逐字）→ `render_reason` 模板渲染。 |
| 11 | ChatResponse 组装 | `agent_chat_adapter.build_agent_chat_response(..., explanations=, rag=)`：卡片 + `audit.rag` + `ranked_with`（写前）+ trace + agent。 |

**"先读后写"确认**：面板 `active_preferences` 是**写后**状态，`audit.ranked_with_preferences` 与 RAG 请求的
`stated_preferences` 都是**写前**状态；`test_the_write_happens_after_the_read` 断言同一轮既陈述 `blue` 又请求
推荐时，排序与解释仍只用 `black`（该轮写入的 `blue` 从下一轮才生效）。

## 3. ADD → 推荐 → supersede → 推荐 → REMOVE → 推荐 的执行轨迹（实测）

```text
Turn 1  请求 "RedWidget BlackWidget"（全程不变）
        Planner context: active_preferences 无（[]）；plan=[history, catalog_search]
        actions {recommend_from_history:1, search_catalog:1, finish:1}
        grounded=4  sources_present=[history, catalog_search]  multi_source_count=0（见 §4 边界）
        fused/原序 = [j-red, j-black, j-blue, j-plain]；applied=false moved=0
        知识证据：j-red:title="RedWidget"、j-black:title="BlackWidget"（j-blue/j-plain 无匹配事实）
        解释：2 grounded（仅目录标题事实，无偏好表述）+ 2 insufficient_evidence

Turn 2  "I prefer black."（direct）
        memory_update = {changed:true, added:[{color,prefer,black}]}；store 真值 {color/prefer/black}

Turn 3  同请求
        Planner context: active_preferences=[{color,prefer,black}]；同一 plan，actions 同上
        M10A：j-black match(details.Color=black)；j-red/j-blue unknown；j-plain unknown（无颜色属性）
        final=[j-black, j-red, j-blue, j-plain]（applied=true, moved=2）
        知识证据：j-black:details.Color="black"（score 1.99243）→ reason
          'Recommended because its catalogue details.Color is "black", which matches a preference you stated.'
        j-red 解释仍只引其标题，**不含**任何偏好匹配表述
        分数/来源：j-black 1.137496/bm25/[catalog_search]（与 Turn 1 相同）；j-blue 0.7/sasrec_logit/[] 不变

Turn 4  "Actually, I prefer blue instead."（direct）
        memory_update = {changed:true, added:[{color,prefer,blue}], superseded:[{color,prefer,black}]}

Turn 5  同请求
        Planner context: active_preferences=[{color,prefer,blue}]
        M10A：j-blue match(details.Color=blue)；j-black unknown（旧值不再 match）
        final=[j-blue, j-red, j-black, j-plain]（applied=true, moved=3）
        知识证据：j-blue:details.Color="blue" → reason 带"符合你所陈述的偏好"；j-black 仅标题，无匹配表述

Turn 6  "I don't care about color anymore."（direct）
        memory_update = {changed:true, removed:[{color,prefer,blue}]}；store ACTIVE 为空

Turn 7  同请求
        Planner context: active_preferences=[]；applied=false；final 回到 fused 基线
        解释：j-black 仍引标题 'Recommended because its catalogue title is "BlackWidget".'
             但**不含**"符合偏好"表述（撤销后不得再声称符合）
```

## 4. 多来源候选、RRF 与 Memory 最终排序对照（六层分别核对）

| 层 | 实测 |
|---|---|
| 1. 实际已执行来源 | `action_counts={recommend_from_history:1, search_catalog:1, finish:1}`，轨迹 `[recommend_from_history, search_catalog, finish]` |
| 2. 本轮累计合法候选 | 4 件：j-red、j-black、j-blue、j-plain（`grounded_count=4`） |
| 3. Ledger 中真实存在的来源记录 | **仅 `catalog_search`**（j-black#1、j-red#2；`source_query` 为本轮请求）。顺序来源走 accepted capability，**不写 Ledger** ⇒ `multi_source_count=0`，但 `sources_present=[history, catalog_search]` 如实列出两个来源（E-02A-V 边界 1，本轮未改） |
| 4. RRF 融合顺序 | 对两个来源各自 ranking 调用既有 `reciprocal_rank_fusion`（K=60、并列按 identity）：history `[red,blue,black,plain]` + lexical `[black,red]` → red 1/61+1/62=0.032522、black 1/63+1/61=0.032266、blue 1/62、plain 1/64 ⇒ `[j-red, j-black, j-blue, j-plain]`，与响应 `audit.original_order` **逐项一致**（测试同时与手算/公开函数比对） |
| 5. Memory 最终顺序 | Turn 1/7 同层 4；Turn 3 `[j-black, j-red, j-blue, j-plain]`；Turn 5 `[j-blue, j-red, j-black, j-plain]`；`avoid red` → `[j-black, j-blue, j-plain, j-red]`（moved=4，red 因 violation 殿后） |
| 6. 最终 ChatResponse 商品集合 | 恒为同 4 件（`k=4`）；重叠商品 j-red **只有一张卡片**（去重按稳定身份），其 `fusion_sources=[catalog_search]`、`score=1.137496/bm25` |

`k` 截断（`k=2`）：两个来源各自受 k 限制 → 累计 3 件（`grounded_count=3`），融合 `[red, black, blue]`，
响应取前 2 `[j-red, j-black]`；RAG 的 `products_examined=2`，被截断的 j-blue **不出现在任何卡片或引用中**
（`test_k_bounds_the_presentation_and_rag_never_resurrects_a_cut_candidate`）。

## 5. 商品级 evidence_id / field_path / value / reason 溯源（j-black，Turn 3）

```text
item_id            : 701
原始来源           : catalog_search #1（source_score=1.137496，score_kind=bm25，source_query="RedWidget BlackWidget"）
                     （顺序来源也返回了 j-black，但它走 capability，不写 Ledger，故无 provenance 记录）
RRF rank (original): 2
Memory rank        : 1（M10A：color/prefer/black == details.Color/black → match）
发布分数/来源       : 1.137496 / score_kind=bm25 / fusion_sources=["catalog_search"]（RAG 前后逐字节相同）
evidence_id        : "j-black:details.Color"
field_path / value : details.Color = "black"（source=catalog_metadata，retrieval_score=1.99243）
偏好匹配证据        : kind=color, value=black, status=match, metadata_field=details.Color, metadata_value=black
validated reason   : 'Recommended because its catalogue details.Color is "black", which matches a preference you stated.'
```

验证链本身（`explanation.py:228`）：身份必须 ∈ 本轮呈现集合 → `evidence_id` 必须 ∈ 该商品自己的 pack →
禁止跨商品引用 → `field_path`/`value` 必须与单元**逐字**一致 → 只有通过者进入模板渲染，且"符合偏好"表述
仅当 M10A 在**同一 field/value** 上记录了 `match` 时才追加。偏好证据（`evidence`）与商品知识证据
（`knowledge_evidence`）始终是两个字段。

## 6. 无证据、幻觉、注入、模型失败的降级表现（全部实测）

| 情形 | 实测行为 |
|---|---|
| 商品无目录记录 | `metadata_status="missing"`、`reason_status="insufficient_evidence"`、`reason=null`（不编造） |
| 无相关 Evidence Pack | 该商品 `insufficient_evidence`，不进模型调用要求（其它商品照常）；全部商品都无证据时**不调用模型**（`failure="no_evidence"`） |
| 检索层零分回退 | 见 §7 |
| 虚构商品 ID | 整条解释被拒（`explanations_rejected≥1`），卡片 `reason=null`，推荐不变 |
| 虚构 evidence_id | 同上（`unknown_evidence_id`） |
| 跨商品引用（真实存在但属于别的商品） | 拒绝（`cross_product_evidence`） |
| 字段路径真实但值错误 | 拒绝（`unsupported_claim`）；响应体不含任何编造属性词（测试断言 `Item Weight`/`180 g`/`premium`/`aerospace`/`lightweight` 均不出现） |
| 非法 JSON | `failure="malformed_output"`（`model_called=true`），无 reason |
| Provider 超时/传输失败 | `failure="model_failed"`、`model_called=false`，推荐与全部旧字段不变 |
| 商品描述中的 Prompt Injection | 注入文本只作为 `products[*].evidence[*].value` 数据出现；plan/身份/权限不变；`source_text`（用户原话整段）、`memory_id`、`logical_seq`、session namespace 均不出现在请求或响应；被拒绝的假商品计数 +1 |
| 失败回退 | 绝不生成通用营销文案：所有失败路径只产生 `reason=null` + 现有 abstain 状态 |

## 7. 零分检索回退与 `model_called` 审计语义核查结论

**一、零分回退**：accepted 检索层在"范围内没有任何词命中"时按文档化规则回退首个字段（score 0.0）。
实测：`j-plain`（仅标题）在 `"NothingMatchesThisRequest"` 下得到 `title` 单元、`retrieval_score=0.0`、
`fallback_reason="no_lexical_match"`；交付的解释是
`'Recommended because its catalogue title is "PlainGadget".'` —— 只陈述真实目录事实，**不含**"相关/满足需求/符合偏好"
任何表述，也不使用该回退去声称命中。审计区分方式：`knowledge_evidence[].retrieval_score=0.0` +
`fallback_reason` 明确标注"无词法匹配"；而正相关命中会有 >0 的 `retrieval_score`（如 1.99）。
**结论：目录存在、回退片段、正相关命中三者可区分，无冒充。**

**二、`model_called` 语义**（逐态实测）：

| 状态 | `audit.rag` 实测 | 判定依据 |
|---|---|---|
| 未尝试（能力关闭） | `rag = null`；模型仅被调用 1 次（工具规划） | 开关默认关，绝不发起解释调用 |
| 无可解释内容（无任何商品有证据） | `enabled=true, model_called=false, failure="no_evidence"` | 检索后判定，调用前返回 |
| 已尝试但失败/超时 | `enabled=true, model_called=false, failure="model_failed"` | 传输失败经 `ModelCallError` 归一化 |
| 调用成功但输出非法 | `enabled=true, model_called=true, failure="malformed_output"` | 解析失败 |
| 调用成功但未通过验证 | `enabled=true, model_called=true, failure=null, explanations_rejected≥1` | 验证拒绝，推荐不变 |

四种状态两两可区分（`rag is null` / `model_called=false`+code / `model_called=true`+failure / `model_called=true`+rejected），
语义自洽，**无需修正**；本轮不扩展成本/遥测字段。

## 8. 两 Session 隔离

```text
Session A: 存 black；Session B: 存 blue（同一 harness、不同 session/user_key）
交替同请求推荐：
  A → pre-write=[black]，final 首位 j-black，j-blue match=0
  B → pre-write=[blue] ，final 首位 j-blue ，j-black match=0
  再次 A → 与 A 上一轮完全一致（B 的回合未影响 A），store 各自保持 {black} / {blue}
RAG：两轮各自的 Evidence Pack 只含该轮已展示商品，且每个 evidence_id 都以本商品身份为前缀
身份：请求体附带 user_key → 422；模型文本无法指定 namespace（注入测试）
并发：4 个 session（black/blue/black/blue）4 线程并发同请求 → 每个 session 首位商品与自身偏好一致，无串用
```

## 9. 四种 Feature Flag 组合（实测）

| Memory Ranking | RAG | 商品 ID 集合 | 最终顺序首位 | 原始分数/来源 | 偏好匹配证据 | 商品知识证据 | 解释状态 | 模型调用次数 |
|---|---|---|---|---|---|---|---|---|
| 关 | 关 | 4 件（同） | j-red（融合原序） | 1.137496/bm25/[catalog_search] | `match_count=0` | `[]` | `not_attempted`，`audit.rag=null` | 1（仅规划） |
| 开 | 关 | 4 件（同） | j-black（记忆重排） | 同上（不变） | `match_count=1` | `[]` | `not_attempted`，`audit.rag=null` | 1 |
| 关 | 开 | 4 件（同） | j-red（融合原序） | 同上 | `match_count=0` | 有（j-red/j-black 标题） | `grounded`/`insufficient_evidence` | 2（规划+解释） |
| 开 | 开 | 4 件（同） | j-black（记忆重排） | 同上 | `match_count=1` | j-black `details.Color` 等 | j-black `grounded`（带匹配表述） | 2 |

四种组合中：**商品 ID 集合恒定**；顺序只随 Memory Ranking 开关变化；分数/score_kind/provenance 恒定不变；
偏好匹配证据只随 Ranking 开关变化；商品知识证据与解释状态只随 RAG 开关变化；关闭时模型调用次数为 1。

## 10. 生产代码是否修改

**未修改任何生产代码。** 本轮只新增验收测试
`tests/test_v2_phase4_joint_acceptance.py`（20 项）与本文档。核对方式（可复现）：

```bash
$ find recommendation -name "*.py" -newer docs/V2_E02B_REPORT.md      # 以 E-02B 收尾工件为界
(无输出)
$ git status --short -uall | grep -c '^ M'                             # 26 → 26（modified 集合未增）
26
```

**本轮发现 1 个非阻塞展示层问题（如实报告，未修改）**：

```text
现象：多来源 且/或 Memory 重排的 llm_agent 回合中，响应的 message 叙述与卡片顺序/集合不一致。
实测（Turn 3，prefer black）：
  cards（authoritative，结构化字段）: [j-black, j-red, j-blue, j-plain]
  audit.original_order / reranked_order: [j-red, j-black, j-blue, j-plain] / [j-black, j-red, j-blue, j-plain]
  message 中的编号列表            : 1. j-red  2. j-blue  3. j-black  4. j-plain  ← 顺序策略来源自身的顺序
实测（k=2）：cards=[j-red, j-black]，message 仍列出 j-blue（已被 k 截断）
根因：message 由 accepted renderer 在 controller 运行**内部**生成（基于该次运行的候选/富化状态），
      而 phase-4 的候选累计（E-02A-V）、Memory 重排（E-02A）与 k 截断发生在运行**之后**的投影层；
      单来源 deterministic/llm 路径下两者一致，因此旧模式不受影响。
影响：`recommendations` / `audit` / 每卡 `reason`+`knowledge_evidence` 全部自洽且为权威展示；
      仅 `message` 的散文编号可能与卡片不一致（浏览器以上下两处呈现，用户可能看到矛盾）。
未修改原因：修复需在"重新渲染叙述（需把累计候选+重排结果回灌 accepted renderer，涉及重新富化）"与
      "在 agent 路径替换/补一句确定性最终顺序说明（会丢失既有逐候选证据行）"之间做产品取舍，
      属于 accepted renderer 的展示决策，超出本轮"验收为主、无明确最小修法不改生产代码"的授权范围
      （§11 亦要求此类情况形成可复现证据而非擅自扩大修改）。
建议后续（最小选项）：在 agent 路径仅在"最终顺序 ≠ 运行自述顺序"时，用最终卡片顺序重排 message 中
      的编号行（保持逐候选内容不变），并在适配器层记录一次审计标志；本轮留待明确授权后实施。
```

其余旧路径核对：`ChatResponse` 顶层键集合未变（`tests/test_demo_llm_agent_mode.py::CHAT_RESPONSE_KEYS`、
`tests/test_demo_llm_mode.py` 均通过）；新增字段全部为可选（`reason`/`reason_status`/`knowledge_evidence`/
`audit.rag`），deterministic/llm/Agent Recommend/Serving 响应结构与旧字段不变
（`test_other_paths_are_untouched_by_the_joint_capabilities`）。

## 11. 测试命令、PASS/FAIL 数量

```bash
.venv/bin/python -m pytest tests/test_v2_phase4_joint_acceptance.py \
  tests/test_product_knowledge_rag.py tests/test_demo_memory_ranking.py \
  tests/test_multi_source_fusion_presentation.py tests/test_demo_llm_agent_memory_acceptance.py \
  tests/test_demo_llm_agent_memory.py tests/test_agent_state_isolation.py \
  tests/test_agent_control_plane.py tests/test_agent_chat_adapter.py tests/test_llm_tool_selection.py \
  tests/test_llm_selected_controller.py tests/test_demo_llm_agent_mode.py tests/test_demo_agent_api.py \
  tests/test_demo_api.py tests/test_demo_multiturn.py tests/test_demo_llm_mode.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py tests/test_preference_reranking.py \
  tests/test_agent_reranking.py tests/test_agent_memory.py tests/test_provider_record_replay.py \
  tests/test_demo_control_plane.py tests/test_demo_recommendation_trace.py tests/test_demo_sessions.py \
  tests/test_demo_web.py tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_control_stage2_candidates.py tests/test_constraint_eligibility.py \
  tests/test_candidate_reference_boundary.py tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
```

```text
983 passed in 108.98s        （35 个文件，0 failed；本轮新增 20 项联合验收测试）
```

覆盖：本轮新增 E-02C API 级联合验收；E-02B 商品知识 RAG；E-02A Memory Ranking；
E-02A-V 多源融合呈现；E-01B 多轮 Memory；Agent 控制平面 / Grounding / 约束资格；
Tool Selection 与 LLM-selected controller；Agent Chat Adapter；Legacy deterministic/llm Chat；
Legacy Agent Recommend；Serving `/v1/recommend`（`test_demo_api.py`、`test_serving_fusion.py`）；
Provider record/replay。未删除或放宽任何正确性断言。

开发过程中修正的 3 处**测试自身**假设（非产品缺陷，已核实后改写断言）：两个来源各自受 `k` 限制导致
累计候选数为 3（而非 4）；capability 路径按 accepted 设计信任 Tool 自身候选（不受测试身份映射限制）；
以及 narration 与卡片顺序差异（改为断言卡片级契约并在 §10 记录该边界）。

## 12. 尚未完成的真实 Provider 与真实资产联调

1. **真实在线 Provider 未调用**（未授权）：解释与工具规划均由仓库自带 `ScriptedModelClient` /
   确定性 oracle 假模型产生；真实模型是否逐字复制 `field_path`/`value`、是否滥用 abstain 未经验证。
   （本轮期间 harness 自身的模型服务曾返回 "Service is too busy"，与仓库测试无关：全部测试不发网络请求。）
2. **真实 300 MB 目录资产未加载**：本轮使用合成目录（title / store / details.Color / price / Item Weight）；
   真实资产的字段覆盖率分布未知。
3. **自由文本解释未实现**：按 E-02B 的保守路线只发布"验证引用 + 模板"句子，系统不声称通用事实校验。
4. **无质量实验**：未做 NDCG/Recall 或人工可读性评估。
5. **§10 的 narration 一致性问题**为已知待办（需要明确授权后按最小选项修复）。

## 13. 最终 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall
开始：26 modified + 42 untracked
结束：26 modified + 44 untracked
```

本轮新增 2 个未跟踪文件：`tests/test_v2_phase4_joint_acceptance.py`、`docs/V2_E02C_REPORT.md`。
modified 集合未新增成员（**本轮未触碰任何已跟踪的生产文件**）。前序 TIGER / Demo / M9 / M10 /
E-01 / E-02A / E-02A-V / E-02B 成果与 11 份文档逐条确认仍在，无删除、无回退。

## 14. 最终结论

**PASS。** §13 完成标准逐项由真实 API 级执行测试支持：

```text
同一 Session 的真实 Memory 更新（M9 ADD/supersede/REMOVE 均为 store 真值）
  → 后续 Agent 合法选择并执行来源（plan 校验 + capability + plane + ActionValidator/GroundingVerifier）
  → 多源候选正确融合（本轮全部合法执行来源的 grounded 候选，既有 RRF，K=60 与 tie-break 未改，k 在融合后生效）
  → 当前 ACTIVE Memory 对真实候选可信重排（M10A 三态证据 + M10B 既有键，写前快照）
  → 对最终展示商品检索目录知识（按身份限定、可追溯 evidence_id、有界、缺失即声明）
  → 只输出经过验证的商品解释（身份/引用/禁跨商品/逐字一致；失败即 reason=null）
```

并且：更正与撤销后排序与解释同步更新（旧值不再 match，也不再有"符合偏好"表述）；两个 Session 偏好、
重排证据、RAG 上下文互不串用（含 4 线程并发）；无证据不编造（含零分回退只陈述真实事实）；
RAG 全部失败路径不改变推荐身份/集合/顺序/原始分数/provenance；旧 deterministic/llm、Agent Recommend、
Serving 与 provider replay 不回归（983 passed / 0 failed）。
§10 的 narration 顺序一致性问题与本报告 §12 的边界均已如实记录，作为后续明确授权后的最小修复项。
本轮到此停止，不进入第 5 大阶段。
