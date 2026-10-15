# AgentRec-X V2 / E-03C — Product Feedback Loop

> 前序：[`docs/V2_E03B_REPORT.md`](V2_E03B_REPORT.md)（Bounded Replanning）。
> 本轮完成阶段 5 最后一个任务：**用户对上一轮真实展示商品的反馈，可信绑定身份、写入 M9，并在下一轮
> 真实推荐中产生符合语义的可观察变化**。
> 结论：**PASS**（附一次必须披露的工作树事故及其恢复，见 §17）。

---

## 0. 一句话结论

```text
实测闭环（真实 Chat API / Session / M9 / Controller / M10A-M10B）：
  turn1 recommend            cards=[k-hist, k-black, k-blue, k-red]   （session 记录展示位置 1..4）
  turn2 "I like the second one."
        → audit.feedback {status:resolved, action:like, reference:"the second one",
                          parent_asin:k-black, item_id:804, final_position:2,
                          written:true, entry_status:active, effect:promote}
        → active_preferences=[product/prefer/k-black]，added=[product/prefer/k-black]（无任何泛化条目）
  turn3 recommend（同一 plan、同一候选池）
        → cards=[k-black, k-hist, k-blue, k-red]   ← 被喜欢的商品升到第一，卡片带 feedback:"liked"
        → 候选集合、分数、score_kind、fusion_sources 与 turn1 完全一致（只有顺序变化）
  turn4 "I don't like the first one."（此时第一件=k-black，即用户实际看到的第一件）
        → {status:resolved, action:dislike, parent_asin:k-black, final_position:1, effect:demote}
  turn5 → cards=[k-blue, k-red, k-black, …]（k-black 降到末位，卡片 feedback:"disliked"）
  turn6 "Forget what I said about the first one." → {action:retract, written:true, entry_status:removed}
  turn7 → 顺序回到 turn5 之前的状态：失效反馈不再影响排序
```

## 1. 初始 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall
开始：26 modified + 52 untracked   （与 E-03B 报告结束状态一致）
结束：32 modified + 54 untracked
```

未 reset/checkout/restore/clean/stash，未删未跟踪文件，未装依赖，未训练模型，未调用在线 Provider，
未改 `/v1/recommend`。

## 2. 当前 Session 是否有真实已展示商品身份映射

**开始时不具备**（`DemoSession` 只有 session/user_key/profile/history/turn/decision_mode ✓）。
本轮新增最小映射（`recommendation/demo/sessions.py:127-154`）：

```text
last_presentation      = 最近一次「推荐轮」实际展示的卡片：{parent_asin, item_id, position(最终展示位), title}
last_presentation_turn = 产生它的 server-owned turn id
```

* 写入来源是**响应自己的 recommendations**（`runtime.py:726`），因此不可能包含被 `k` 截断、未 grounded
  或从未展示的商品；`position` 是用户看到的最终位次，不是 RRF 原始 rank（原始顺序另存于
  `audit.original_order`，两者不混用）。
* 生命周期：展示商品的轮次替换它；**direct 轮不展示商品 ⇒ 不清空**（`record_presentation` 对空列表
  直接返回旧记录），因此用户仍可对「上一次真正看到的列表」反馈；`reset` 清空整个 session。
* 逐 session 保存 ⇒ 结构上无法跨会话引用。

## 3. M9 对商品级偏好的实际支持情况

**开始时：不支持**。`PreferenceKind` = category/feature/brand/price_max/price_min/color/material/
free_form_constraint，全部是**属性级**；把商品 ID 塞进 `color`/`feature` 属于误用，本轮未这样做。

最小向后兼容扩展（无迁移、无新库）：

* 新增 `PreferenceKind.PRODUCT = "product"`（`memory/schemas.py:102`），`value` 是**规范 `parent_asin`**，
  语义是「用户对这一件商品的态度」，刻意不是「对它的品牌/颜色/类别的态度」；
* 该 kind **不在** `TEXT_KINDS` 内 ⇒ 永不参与 lexical query 改写（identity 不是文本）；
* M10A 新增身份规则（`preference_matching/matcher.py:458`）：以候选自己的 `parent_asin` 与取值比较，
  命中即 MATCH，AVOID 命中即 VIOLATION，**其它商品是 UNKNOWN 而非 VIOLATION**；支持矩阵登记为
  `SUPPORTED`（`preference_matching/schemas.py:142`）；
* M9 写入路径新增三个**可选**参数（`memory/service.py:220-230`）：`extra_candidates` /
  `extra_removals` / `suppress_text_spans`，全部照常重新校验、去重、supersede、写审计；
* 原有 ADD/REPLACE/REMOVE、supersede、投影契约、`TEXT_KINDS`、`_supersedes` 语义均未改动。

## 4. 反馈身份绑定与合法性验证方案

`recommendation/demo/feedback.py`（纯函数、无 IO）承担身份决策，模型完全不参与：

```text
classify_feedback(message) → action ∈ {like, dislike, exclude, correct, retract, none}
attitude_of(message)       → prefer | avoid | None      （correct 只表示「更正」，不表示态度）
_reference_tokens(message) → (text, kind, value)        kind ∈ {ordinal, last, parent_asin, item_id, deictic}
resolve_reference(message, presented) → ProductFeedbackResolution
```

判定顺序与拒绝语义：

| 输入 | 结果 |
|---|---|
| 序数（the second one / 2nd） | 绑定到该 **展示位次** 的卡片 |
| the last one | 绑定到展示列表末位 |
| 形如 asin 的 token（8–14 位大写字母数字） | 仅当等于本会话展示过的 `parent_asin` 才绑定 |
| `item 802` | 仅当等于本会话展示过的 item_id 才绑定 |
| 标题片段 / this one / it | 仅当唯一匹配（或只展示过一件）才绑定 |
| 无指代 | `ambiguous`（不写） |
| 指代越界（第 7 件 / 第 3 件但只展示 2 件） | `ambiguous`（不写） |
| 未展示过的身份 / item id | `unknown_product`（不写） |
| 尚无展示历史 | `no_presentation`（不写） |
| 非商品反馈（如 “I prefer blue”） | `none`（按普通 M9 偏好处理，**不**作为反馈） |

任何 `status != resolved` ⇒ **零写入**，且句子若指向「被展示的商品」（出现 ordinal/identity/`you showed
me` 等表达）则其泛化读法一并被丢弃，避免 “I don't like the black shoes you showed me” 变成
`color=black` 偏好（§5）。写什么、状态如何，一律以 M9 自己的快照/摘要为准（见 §7）。

## 5. LIKE / DISLIKE / AVOID / REMOVE 的实际语义

| 用户语义 | 存储 | 下一轮实际效果 |
|---|---|---|
| LIKE | `product` + `prefer` + `parent_asin` | M10A 对该商品 MATCH ⇒ M10B `match_count` 更高 ⇒ **上移**（实测升到第一） |
| DISLIKE | `product` + `avoid` | M10A 对该商品 VIOLATION ⇒ 排序下沉（实测到末位） |
| 明确排除（“don't recommend it again”） | `product` + `avoid`（同一 kind） | 同样是**降序**；`effect:"demote"` 如实上报；当融合池 > k 时该商品**确实不出现在答案中**（实测 k=2 时不出现），但它没有被从候选集合中删除——本部署的排序按 M10A evidence 进行，**不把降序冒充硬排除** |
| CORRECT（“actually I like it instead”） | 同一 kind，`REPLACE` + `replaces=<asin>` | 目标商品的旧态度被 supersede，新态度生效（实测 avoid→prefer） |
| REMOVE / 撤销（“forget what I said about it”） | 值域撤回（`PreferenceRemoval(value=<asin>)`，由 `mode=REMOVE` 的候选转换而来） | 该商品反馈被 tombstone（`entry_status:"removed"`）；实测排序**回到基线**；未命中任何记录时 `written:false`（不谎报） |

## 6. 两轮及多轮真实 Chat 执行轨迹

见 §0（turn1→turn7 的实测输出）。关键点：

* turn3 的 plan 与候选池与 turn1 完全相同（同一 fake 计划 `[history]` + 同一追加来源 `catalog_search`），
  只有**顺序**变化 ⇒ 变化归因于反馈而非偶然；
* turn4 的 “the first one” 指向的是 **ranking 之后**用户看到的第一件（k-black），而不是融合序第一件
  （k-hist）——§9E 的核心断言；
* turn6 的撤销把 `k-black` 的条目 tombstone，turn7 顺序回到撤销前，证明失效反馈不再参与排序。

## 7. 反馈前后同一合法候选池的 ID 与 rank 对照

```text
turn1（基线，k=4）      : k-hist(1) k-black(2) k-blue(3) k-red(4)
turn3（like k-black）   : k-black(1) k-hist(2) k-blue(3) k-red(4)      ← 同一 4 件、同 provenance/分数
turn5（dislike k-black）: k-blue(1)  k-red(2)  k-black(3)…
turn7（retract 后）     : 回到 turn5 之前的状态
多来源对照（history+catalog_search）: 反馈使 k-red 升到第一，fusion_sources / score_kind /
                                     grounded 去重结果与反馈前逐项一致
```

审计可分辨的六个阶段（§10）：`action/span`（用户说了什么）→ `reference`（解析出的指代）→
`parent_asin/item_id/final_position`（验证过的身份）→ `written/entry_status`（真正写入的记录，来自
store 自己的摘要）→ `active_product_feedback`（下一轮会读到的 ACTIVE 集合）→ 下一轮 cards/顺序
（最终候选与排序变化）。`written` 由 store 快照判定，**不是**模型文本；`active` 由 store 投影判定，
**不是**「面板里有一个条目」。

## 8. 撤销和更正后的状态变化

实测：`retract` 命中条目 ⇒ `written:true, entry_status:"removed"`，`active_product_feedback:[]`，
下一轮顺序回到基线；`retract` 未命中 ⇒ `written:false, entry_status:null`（如实报告，无副作用）。
`correct` 使用 `REPLACE`+`replaces=<asin>` ⇒ 只 supersede 该商品的旧条目（实测 avoid→prefer），
其它 kind 的偏好（如 color=blue）不受影响（回归 `test_demo_llm_agent_memory_acceptance.py` 全绿）。

## 9. 两 Session 隔离证据

Session A 喜欢 `k-black` ⇒ A 的下一轮 `k-black` 第一、卡片带标记；Session B 同 plan 同池 ⇒ 顺序不变、
`active_preferences==[]`、无标记；B 试图 `Forget what I said about the second one.` ⇒ `written:false`
（B 从未展示过 A 的商品列表 ⇒ 无可撤回对象）。M9 命名空间不同（`user_key` 不同）✓。

## 10. 非法引用、未展示商品、k 截断测试

`tests/test_v2_product_feedback.py` 覆盖：未展示身份（`unknown_product`）、未展示 item id
（`unknown_product`）、越界位次（`ambiguous`）、多卡片下的 `it`（`ambiguous`）、无展示历史
（`no_presentation`）、非反馈语句（`none`）、以及 **k=2 时被截断的 k-red 用 `item 802` 指代仍
`unknown_product`**（`position` 只记录展示过的两件）。全部 `written:false`、`active_preferences==[]`、
下一轮顺序与反馈前一致。

## 11. Replanning、RAG 和叙述一致性回归

* **E-03B**：开启 `AGENTRECX_AGENT_REPLANNING=1` 时，反馈轮后依旧追加一个来源（`status:"appended_candidates"`，
  `tool_calls==2`），追加来源的商品**同样**遵守有效反馈（标记与排序一致），模型上下文只多不少地仍只含
  计数/来源/预算——反馈不能增加预算、也不产生第二次决策（实测 `replan_calls==1`）。
* **E-02B RAG**：卡片 `feedback` 标记与 `reason`/`knowledge_evidence` 分离；RAG 只解释目录事实，
  不因「用户不喜欢」生成任何商品缺陷；叙述仍逐行等于最终 cards 的身份与顺序（`test_v2_narration_consistency.py` 全绿）。

## 12. 修改文件与 file:line

| 文件:行 | 内容 |
|---|---|
| `recommendation/demo/feedback.py`（**新增**） | `classify_feedback:200`、`attitude_of:189`、`_PRESENTATION_RE:112`、`_IDENTITY_RE:126`、`PresentedCard:131`、`ProductFeedbackResolution:153`、`resolve_reference:253`、`feedback_candidate:365`、`product_feedback_view:414`、`feedback_marks:461` |
| `recommendation/demo/sessions.py:127-154` | `DemoSession.last_presentation` / `last_presentation_turn` / `record_presentation` |
| `recommendation/memory/schemas.py:102` | `PreferenceKind.PRODUCT` |
| `recommendation/memory/service.py:158,220-230,337` | `_normalise_span`、`process_turn(extra_candidates, extra_removals, suppress_text_spans)`、`_with_caller_statements`、PRODUCT 的 REMOVE ⇒ 值域撤回 |
| `recommendation/memory/extraction.py:158,167,172,395-398` | 引用/身份/「被指向的商品」三类守卫：引用不是偏好取值 |
| `recommendation/preference_matching/matcher.py:458`、`schemas.py:142` | PRODUCT 身份匹配规则与支持矩阵 |
| `recommendation/demo/schemas.py:377,418,503` | 卡片 `feedback` 标记、`AuditView.feedback`、`ProductFeedbackView` |
| `recommendation/demo/serialization.py:551,567` | `build_chat_response(feedback=, feedback_marks=)`（legacy 直答路径） |
| `recommendation/demo/agent_chat_adapter.py:330,483` | 卡片标记与 `audit.feedback` |
| `recommendation/demo/runtime.py:594-601,726,753,856,1137` | 直答分支解析+写入+合并、展示记录、`resolve_product_feedback`、`_merge_memory_write` |
| `docs/USAGE.md` | E-03C 契约（六阶段审计、拒绝语义、语义边界、与 RAG/Replanning 的关系） |
| `tests/test_v2_product_feedback.py`（**新增，23 项**） | §9 A–J 的 API 级执行测试 |

未改动：`/v1/recommend`、RRF 公式、GroundingVerifier、CandidatePlane/Ledger、E-03B Gate、RAG 协议、
`ChatResponse` 既有字段语义（新字段全部为可选）。

## 13. 测试命令与 PASS/FAIL 数量

```bash
.venv/bin/python -m pytest tests/test_v2_product_feedback.py \
  tests/test_v2_bounded_replanning.py tests/test_v2_observation_replan_gate.py \
  tests/test_v2_narration_consistency.py tests/test_v2_phase4_joint_acceptance.py \
  tests/test_multi_source_fusion_presentation.py tests/test_demo_memory_ranking.py \
  tests/test_product_knowledge_rag.py tests/test_demo_llm_agent_memory_acceptance.py \
  tests/test_agent_state_isolation.py tests/test_agent_control_plane.py \
  tests/test_llm_selected_controller.py tests/test_demo_llm_agent_mode.py \
  tests/test_provider_record_replay.py tests/test_demo_llm_agent_memory.py \
  tests/test_agent_chat_adapter.py tests/test_llm_tool_selection.py tests/test_demo_agent_api.py \
  tests/test_demo_api.py tests/test_demo_multiturn.py tests/test_demo_llm_mode.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_demo_control_plane.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_sessions.py tests/test_demo_web.py tests/test_demo_fusion_view.py \
  tests/test_llm_agent_availability_guard.py tests/test_control_stage2_candidates.py \
  tests/test_constraint_eligibility.py tests/test_candidate_reference_boundary.py \
  tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
```

```text
1058 passed in 93.22s        （39 个文件，0 failed；E-03B 基线 1035 + 本轮新增 23）
```

开发中自查并修正的真实缺陷（均已修复并保留测试）：PRODUCT 匹配误用不存在的 `candidate` 变量；
`PreferenceKind` 未导入；`@staticmethod` 装饰器错位；直答轮抑制条件过宽导致普通偏好丢失
（“I prefer blue”）；更正语义把 “actually I like” 存成 avoid（态度与模式混淆）；
撤回未命中也报告 removed；identity 引用长度限制过窄；“seventh one”/身份 token/被指向的商品
被当作偏好取值。测试侧修正均经实测确认产品行为正确后才改写，未放宽任何安全断言。

## 14. 未完成的真实 Provider/资产与在线质量验证

* 全部测试使用 fake/Scripted ModelClient（按 payload 形状作答），**未调用在线 LLM**；真实 provider 的
  时延/限流/输出分布未验证。
* 未加载真实 349 MB checkpoint / 300 MB 元数据 / 6.21 GB item-item / TIGER 后端；`two_tower`/`tiger`
  只覆盖「未注册即不可选」。
* 反馈的**长期**效果（多轮累积、遗忘曲线、NDCG 级离线评估）未评估：本轮只验证「同池、同计划、
  不同反馈 ⇒ 可解释的顺序/展示变化」。
* 硬排除（真正的候选级删除）未实现：按 §4B 的允许范围，实现的是排序级降序并如实上报为 `demote`。

## 15. 结束时 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall      →  32 modified + 54 untracked
```

本轮新增未跟踪：`recommendation/demo/feedback.py`、`tests/test_v2_product_feedback.py`、
`docs/V2_E03C_REPORT.md`；本轮改动的已跟踪文件：`memory/schemas.py`、`memory/service.py`、
`memory/extraction.py`、`preference_matching/matcher.py`、`preference_matching/schemas.py`、
`demo/schemas.py`、`demo/serialization.py`、`demo/sessions.py`、`demo/runtime.py`、`docs/USAGE.md`
（modified 26 → 32）；`demo/agent_chat_adapter.py` 为既有未跟踪文件（本轮续改）。
`schemas.py` 的 modified 计数变化中，有一部分来自 §17 的重建：重建后的文件与 HEAD 基线 + 本会话
新增契约等价（由 1058 项测试与全部 pinned key/byte-for-byte 断言验证），但**不是**被覆盖前那份
文件的逐字副本（docstring 与字段书写顺序为重建时重写）。

## 16. 最终结论

**PASS。**

```text
用户对已展示真实商品表达反馈
  → 可信绑定身份（session 自己的展示记录；序数/身份/item id/唯一标题；其余一律拒绝）
  → 反馈被正确记录（M9 新 product kind；ADD/REPLACE/REMOVE 与原有语义一致；store 快照为准）
  → 下一轮读取有效反馈（active_product_feedback 与 sorted 前的 ACTIVE 集合一致）
  → 相同合法候选池中的排序出现符合语义的可解释变化（like 上移、dislike/exclude 下沉、
     correct 替换、retract 复原；候选集合、provenance、分数、k、Grounding 不变）
```

撤销、会话隔离、身份安全（未展示/k 截断/伪造身份/模糊指代全部零写入）、RAG（不把用户态度写成商品
缺陷、标记与知识证据分离）、Replanning（追加来源同受反馈约束、预算与 Gate 不变）与旧路径
（deterministic / llm / Agent Recommend / Serving）回归全部通过，1058 项测试 0 失败。
本轮到此停止，不进入 E-04A。

## 17. 必须披露的事故与恢复（工作树）

**事故**：本轮中途，一个修改脚本把 `recommendation/demo/agent_chat_adapter.py` 的文本误写入
`recommendation/demo/schemas.py`（脚本末尾误用 `p.write_text(t)`，`p` 指向 schemas.py），
导致该文件的**未提交**内容被覆盖（HEAD 版本为 644 行、24 个类；被覆盖前包含本会话 E-01…E-03B 的
全部新增，例如 `RagAuditView`、`ReplanAuditView`、`AgentExecutionView` 再导出、`ChatResponse.agent`、
卡片 `reason/knowledge_evidence` 等）。被破坏的副本已保存在 `/tmp/destroyed_schemas.py.txt`
（md5 `b695c793ac33e82e6bcfe70a146a4b6f`）。

**恢复**：以 `git show HEAD:recommendation/demo/schemas.py` 为基线，依据**未被破坏的消费者**重建契约：
`demo/__init__.py` 与 `agent_chat_adapter.py` 的 import 列表、`serialization.py` 的构造点、以及测试中
pinned 的 key set（`CHAT_RESPONSE_KEYS`）与逐字段断言（`audit.rag` 精确字典、`knowledge_evidence`
字段、`reason_status` 默认值、卡字段等）。逐条重建后迭代运行相关测试直至全绿。

**验证**：重建版本通过 **1058 项**测试，包含全部 pinned key set、byte-for-byte 审计比较、RAG 字典
精确比较与响应往返（`ChatResponse.model_validate_json(dump) == response`）。因此可以确认其**契约等价**；
但必须说明：docstring 文字、字段书写顺序、注释措辞是重建时重新撰写的，不是原件的逐字还原。
恢复过程中额外发现并修复的问题：`Any` 未导入导致 `ChatResponse` 前向引用未解析、`agent` 字段类型
退化为 `Any` 导致往返不相等、`KnowledgeEvidenceView` 字段集合不完整（缺 `parent_asin`/`item_id`/
`truncated`）、`RagAuditView.reason_source` 默认值缺失。以上均已修复并由测试锁定。

**教训与后续**：本轮之后任何批量改写脚本都必须对**目标路径**做断言（例如断言 `p.name` 与预期
一致、写入前打印绝对路径与 diff 摘要），并且不应以「读取 A、写入 B」的形式共享变量。
建议在 E-04 之前把当前工作树提交（本任务禁止自动 commit），以便此类事故可由版本历史直接恢复。
