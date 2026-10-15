# AgentRec-X V2 / E-03C-SCHEMA-INTEGRITY — 事故恢复与 API 契约核验

> 触发事件：E-03C 开发中，一个修改脚本把 `recommendation/demo/agent_chat_adapter.py` 的文本误写入
> `recommendation/demo/schemas.py`，覆盖了该文件**未提交**的内容。当前文件是基于 HEAD 基线、
> 消费者代码与测试契约重建的版本。
> 本轮性质：**只做核验与最小修复，不开发新功能，不进入 E-04A**。
> 结论：**PASS（可观察契约核验通过；事故前字节级一致性无法证明）**。

---

## 0. 结论摘要

```text
可恢复性：未找到任何可信的事故前副本（git 全部 786 个 blob / stash / dangling 对象、
          pyc 缓存、编辑器备份、Harness 会话记录全部分布式检查后均无命中）。
          ⇒ 原始未提交版本不可逐字恢复，本轮只证明「可观察契约兼容」。

核验范围：4 个契约模块、53 个模型、371 个字段，逐字段记录类型 / required / default /
          约束 / 嵌套类型 / 验证器（见 docs/V2_E03C_SCHEMA_INVENTORY.md，由运行时对象生成）。

本轮发现并修复的真实缺陷：4 个字段的注解停留在未解析的 ForwardRef
          （RecommendationCard.knowledge_evidence、AuditView.rag / .replan / .feedback）
          ⇒ 已通过「把新视图类定义移到引用它们的模型之前」修复，OpenAPI 现在发布真实 $ref。

回归：40 个文件、1070 项测试、0 失败（含本轮新增 12 项 Schema 完整性测试）。
```

## 1. 实际 HEAD 和初始工作树

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall
32 modified + 55 untracked          （与 E-03C 报告结束状态一致）
```

**动手前的保全**（§1 要求）：

```text
/tmp/e03c_schema_integrity/schemas.reconstructed.py   ← 本轮修改前的重建版（840 行）
    md5 ed088de82b654162cca02f996acfeb5a
/tmp/destroyed_schemas.py.txt                          ← 事故被覆盖内容（= adapter 文本，非原件）
    md5 b695c793ac33e82e6bcfe70a146a4b6f
/tmp/e03c_schema_integrity/inventory.txt               ← 运行时生成的 Schema 清单原始输出
```

未执行 reset / checkout / restore / clean / stash / commit，未删除任何未跟踪文件。

## 2. 原始未提交 Schema 是否找到可信副本

**没有找到。** 逐项定向检查结果：

| 来源 | 检查方式 | 结果 |
|---|---|---|
| Git 历史 | `git log`/HEAD 版本；`git stash list`；`git fsck --lost-found` | 只有 HEAD 基线（无本会话 E-01…E-03 新增）✗ |
| Git 对象库 | 枚举 **786 个 blob**，每个前 400 行搜索 `class RagAuditView` | **0 命中** ✗（该未提交版本从未进入对象库） |
| 悬挂对象 | 列出 dangling commit / blob（5 个 commit + blob） | 均不含本文件 ✗ |
| pyc 缓存 | `recommendation/demo/__pycache__/schemas.cpython-310.pyc` | mtime 15:10，字符串常量显示 `ProductFeedbackView` 等**重建后**符号 ✗（事故后被覆盖重编译） |
| 编辑器/备份 | `*.bak|*.orig|*.swp|*~`、`~/.config/Code/User/History` | 无 ✗ |
| Harness 会话记录 | `~/.dsh/sessions/**/session.v3.jsonl.zstd`（42 份，zstd 解压后全文搜索 4 个标记） | 存储的最新会话为 2026-09-28；**本会话尚未落盘**，0 命中 ✗ |
| 历史报告 | `docs/V2_E0*.md`（13 份） | 只有**契约级**描述（字段名/语义），无完整源码 ✗ |

**因此明确记录**：原始未提交版本不可逐字恢复。本轮及后续只能证明**可观察契约兼容**
（类型、字段、默认值、序列化、OpenAPI、历史回归），**不能声称字节级恢复**；
未通过猜测还原注释、字段书写顺序或任何未经测试验证的默认值。

## 3./4. 与当前重建版本的差异 / 无法恢复的部分

由于不存在原件，**没有「原件 vs 重建」的差异可列出** ✗。可确认与不可确认的边界：

* **可确认**（由可执行证据锁定）：公开模型集合、字段名、字段类型、required/optional、
  默认值、约束（`ge/le/min_length/max_length`）、`extra="forbid"` 与 `frozen=True`、
  Literal/Enum 取值、嵌套模型关系、JSON 往返、OpenAPI 形状、以及 E-01A…E-03C 全部历史回归。
* **不可确认**（恢复限制，必须保留）：docstring 文字、字段书写顺序、注释措辞、字段描述的具体
  措辞（其中字段描述会进入 OpenAPI `description` ✗ 属于「可观察但未逐字恢复」的部分）、
  以及任何**未被测试或消费者观察到的**内部细节。

## 5. Schema 全部公开模型与字段的核验结果

完整逐模型清单：**[`docs/V2_E03C_SCHEMA_INVENTORY.md`](V2_E03C_SCHEMA_INVENTORY.md)**
（`53` 个模型 / `371` 个字段，由运行时对象生成 ✓）。核验要点：

| 模块 | 模型数 | 说明 |
|---|---|---|
| `demo/schemas.py` | 28 | Chat 请求/响应、卡片、审计、记忆面板、证据、RAG/Replan/Feedback 块、trace 系列 |
| `demo/agent_schemas.py` | 12 | Agent 响应、观察、门控决策、Replan 执行、grounded/eligibility/terminal/trajectory/timing |
| `memory/schemas.py` | 8 | PreferenceKind / Polarity / Mode / Status / Candidate / Removal / Extraction / Entry / Snapshot |
| `preference_matching.schemas.py` | 5 | EvidenceStatus / ReasonCode / Evidence / Counts / Report / MatchingSupport |

重点模型（当前真实定义）：

```text
ChatRequest        message(required,1..2000)  k(default 5, strict int, 1..100)  decision_mode(default None)
ChatResponse       required = session_id, turn_id, turn, route, message, audit
                   可选/默认：api_version, active_preferences(()) , memory_update(default factory),
                   recommendations(()), trace(None), agent(None)
AuditView          required = reranking_applied, candidate_count, moved_count
                   默认：original_order(()), reranked_order(()), ranked_with_preference_count(0),
                   ranked_with_preferences(()), rag(None), replan(None), feedback(None)
RecommendationCard required = original_rank, parent_asin, item_id, sasrec_score, metadata_status
                   默认：fused_rank(None), fusion_sources(()), fusion_source_ranks(factory dict),
                   reranked_rank(None), metadata(None), match/violation/unknown_count(0),
                   evidence(()), fallback_reason(None), movement_summary(None), reason(None),
                   reason_status('not_attempted'), knowledge_evidence(()), feedback(None)
KnowledgeEvidenceView required = evidence_id, parent_asin, item_id, field_path, value
                   默认：source('catalog_metadata'), retrieval_score(0.0), truncated(False)
RagAuditView       默认：enabled(False), model_called(False), 计数(0), failure(None),
                   reason_source('template_from_validated_claims')
ReplanAuditView    默认：continue_recommended(False), reason_code(''), eligible_sources(()),
                   预算(0), decision_source('deterministic_rules'), status(None),
                   model_called(False), chosen_source(None), 计数(0)
ProductFeedbackView required = status；其余默认（action('none')、reference('')、span('')、
                   detail('')、parent_asin(None)、item_id(None)、final_position(None)、
                   written(False)、entry_status(None)、effect(None)、两个元组(())
AgentRecommendResponse required = route, message, control_plane, grounded, eligibility,
                   terminal, trajectory；可选：observation/replan/replan_execution/timing
ActivePreferenceView = {kind, polarity, value}   ← 刻意**不含** memory_id 等 store 句柄
EvidenceView         = {kind, value, polarity, status, metadata_field, metadata_value, detail}
```

全部 53 个模型的 `model_config` 均为 `extra="forbid" + frozen=True`；字典字段一律使用
`default_factory`（`RecommendationCard.fusion_source_ranks`、`AgentObservationView.source_calls`
/`source_outcomes`、`ReplanObservationView.source_outcomes`、`AgentTerminalView.limits`、
`AgentExecutionView.action_counts` ✓）；全部模型可 `model_rebuild()` 成功（无未解析引用 ✓）。

## 6. 类型、默认值、序列化、OpenAPI 兼容结果

```text
① 类型安全：两个响应契约模块中，**没有任何公开字段的注解是 Any**（脚本扫描，0 命中）；
            也没有任何未解析 ForwardRef（修复后 0 命中）。
② 默认值/可选性：上表全部逐项断言（tests/test_v2_schema_integrity.py 固定 8 组字段集合）。
③ 序列化：真实 Chat 响应与 Agent 响应均满足
            model_validate(model_dump()) == model 且 model_validate_json(model_dump_json()) == model；
            嵌套块在往返后仍是模型实例（ProductFeedbackView / ReplanAuditView / AgentExecutionView）。
④ 向后兼容：构造一份**E-02B 之前形状**的响应 JSON（无 rag/replan/feedback、卡片无
            reason/knowledge_evidence/feedback、无 agent/trace）→ 正常校验，新字段取文档默认值，
            再次序列化后仍与自身相等。
⑤ 拒绝行为：额外字段（顶层与嵌套）、`k="5"`、空 message、未知 decision_mode、
            缺失 required 的 KnowledgeEvidenceView → 全部 ValidationError。
⑥ OpenAPI：GET /openapi.json 正常生成（11 条路径 / 61 个 schema），
            ChatResponse、AuditView、RecommendationCard、KnowledgeEvidenceView、RagAuditView、
            ReplanAuditView、ProductFeedbackView、AgentRecommendResponse、AgentObservationView、
            AgentReplanView、AgentExecutionView 全部在列；
            四个曾被 ForwardRef 卡住的字段现在发布真实 $ref：
              RecommendationCard.knowledge_evidence → items.$ref #/…/KnowledgeEvidenceView
              AuditView.rag/.replan/.feedback       → anyOf[0].$ref #/…/RagAuditView|ReplanAuditView|ProductFeedbackView
              ChatResponse.agent                    → anyOf[0].$ref #/…/AgentExecutionView
            required 集合与契约一致（ChatResponse / AgentRecommendResponse 已断言）。
```

历史契约交叉核对（§4）：把 13 份 `docs/V2_E0*.md` 中出现的 **89 条带点字段路径**逐一映射到
当前模型 ✓ —— **全部命中**；未命中的 55 条经人工分诊均为**文件名、模块/函数名或内部属性**
（如 `agent_service.py`、`self._served`、`entry.memory_id`、`ledger.ungrounded_entries`）✗
不是响应字段 ✓。分阶段抽检：E-01A（`terminal.*`、`agent.action_counts`）、E-01B
（`memory_update.*`、`active_preferences[].*`）、E-02A（`audit.ranked_with_preferences`、
`reranked_rank`、`match_count`/`violation_count`、`evidence[]`）、E-02A-V（`fusion_sources`、
`fusion_source_ranks`、`grounded.sources_present`）、E-02B（`reason`、`reason_status`、
`knowledge_evidence[].*`、`audit.rag.*`）、E-02C（卡片身份/顺序与叙述一致）、E-03A
（`observation.*`、`replan.*`）、E-03B（`replan_execution.*`、`audit.replan.status/model_called/
chosen_source/initial_candidate_count/final_candidate_count`）、E-03C（`audit.feedback.*`、
卡片 `feedback`）**均在位且由测试断言** ✓。

## 7. 本轮发现并修复的实际错误

| # | 缺陷 | 证据 | 修复 |
|---|---|---|---|
| 1 | `RecommendationCard.knowledge_evidence`、`AuditView.rag`、`AuditView.replan`、`AuditView.feedback` 的注解是**未解析 ForwardRef**（重建时把新视图类放在引用它们的模型之后） | 运行时 `model_fields[...].annotation` 打印 `ForwardRef(...)`；`model_rebuild()` 后才解析 | 把 `KnowledgeEvidenceView`/`RagAuditView`/`ReplanAuditView`/`ProductFeedbackView` 四个类定义**移到 `RecommendationCard` 之前**（纯顺序调整，无语义变更）；修复后 0 个 ForwardRef，OpenAPI 发布真实 `$ref` |
| 2 | 恢复过程中（上一轮）已修并保留锁定：`Any` 未导入导致的未解析注解、`ChatResponse.agent` 退化为 `Any`、`KnowledgeEvidenceView` 缺 `parent_asin`/`item_id`/`truncated`、`RagAuditView.reason_source` 缺默认值 | 上一轮报告 §17 | 已修复，本轮由 `test_the_contract_referenced_views_stay_typed_and_defaulted` 与 OpenAPI 断言固定 |

除上述外，本轮**未**做任何其他改动：没有顺便重构 Schema 体系，没有改动字段集合、类型或默认值。

## 8. 事故影响范围核查结果

```text
① 是否只有 demo/schemas.py 被误覆盖？
   ⇒ 是。破坏脚本只有一次 p.write_text(t) 写入，目标即该文件。
     证据：agent_chat_adapter.py / serialization.py / runtime.py 均**不含**任何 schema 类定义
           （`^class AuditView|ChatResponse|ProductFeedbackView` 计数全为 0）；
           adapter 的唯一标记 `recommendation_narration` 只出现在 adapter 文件本身（1 个文件命中）。
② 是否还有文件被同一脚本错误写入？
   ⇒ 没有。事故窗口（12:45–12:55）内被修改的其它文件只有 sessions.py 与 memory/schemas.py，
     两者是本轮 E-03C 的**有意改动**（展示映射、PRODUCT kind），内容与设计一致。
③ 重建后的 imports / 类定义与消费者是否匹配？
   ⇒ 是。`import recommendation.demo`、`agent_chat_adapter`、`serialization` 均正常导入；
     demo/__init__ 的 27 个导出名与 adapter 的 13 个导入名全部可解析（本轮测试覆盖）。
④ 是否删除了字段注释以外的业务验证逻辑？
   ⇒ 未发现缺失。全部模型的 `model_config` 仍为 extra=forbid + frozen；
     现存验证器（field_validator）在本轮清单中逐模型列出；
     ChatRequest 的 message 去空白、k 严格整数、decision_mode Literal 等行为由测试断言。
⑤ 是否存在因 Any 放行非法输入？
   ⇒ 不存在。公开字段 0 个 `Any`；额外字段/错误类型全部被拒绝（测试断言）。
```

## 9. 全部测试命令和结果

```bash
# 1) Schema 专项（本轮新增）
.venv/bin/python -m pytest tests/test_v2_schema_integrity.py -q
#    → 12 passed

# 2) 40 文件合并回归（E-03C 的 39 文件 + 本轮 Schema 专项）
.venv/bin/python -m pytest tests/test_v2_schema_integrity.py tests/test_v2_product_feedback.py \
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
#    → 1070 passed in 84.12s   （0 failed）

# 3) 清单生成（证据）
.venv/bin/python - <<'PY'   # 见 /tmp/e03c_schema_integrity/inventory.txt 或 docs/V2_E03C_SCHEMA_INVENTORY.md
...  直接遍历 vars(module) 中的 BaseModel 子类，打印 model_config / model_fields / validators
PY
```

覆盖到的历史阶段：E-01A/CLOSE/B1/B2、E-02A/AV/B/C-联合/C-叙述、E-03A/B/C（含本轮新增），
Legacy Chat（deterministic / llm）、Agent Recommend、Serving 融合、Provider replay、Session 隔离。

## 10. 尚未覆盖的兼容性风险

1. **字节级不一致风险（无法消除）**：字段描述文字、docstring、字段书写顺序、注释。其中字段描述
   会进入 OpenAPI 的 `description`：当前描述是重建时重写的，语义等价但**措辞未必与事故前相同** ✗。
   若外部有客户端依赖描述文字（极少见 ✗ 但无法排除），属于未验证区域。
2. **未被消费者或测试触及的隐藏细节**：如果事故前文件含有未被任何代码/测试引用的字段或常量，
   重建不会包含它们 ✗，本轮无法发现（这也是 §4「不能因当前消费者不引用就认定非契约」的反向风险：
   我们只能证明**被观察到**的契约完整）。缓释措施：把现有 1070 项测试视为契约的**下界**，
   任何未来发现的历史依赖都应视为新证据并补齐。
3. **验证器行为**：清单列出了现存 `field_validator`，但历史文件是否还有**已删除模型的验证器**
   无从核对 ✗（当前所有模型的构造/拒绝行为均有测试覆盖）。
4. **OpenAPI 的 required/anyOf 形状**已核验 ✓；但**历史快照不可得** ✗，因此不能做「新旧 OpenAPI diff」，
   只能对照契约文档与测试（§5 已明确不临时造快照冒充原件）。

## 11. 建议的安全版本检查点策略（需用户授权，本轮未执行）

现状风险：**全部 V2（E-01A…E-03C）成果仍未提交**，包括 57 个未跟踪文件 ✗ —— 这正是不久前事故
无法从版本历史恢复的根本原因。

建议（按优先级，全部需用户明确授权后执行）：

1. **立即建立检查点（推荐）**：把当前工作树完整提交到一个新分支，保留所有未跟踪文件：
   ```bash
   git switch -c v2/phase5-e03c-checkpoint          # 不改动 master
   git add -A                                       # 含 57 个未跟踪文件，逐一人工过目后确认
   git commit -m "V2 phase 5 checkpoint: E-01A..E-03C (1058 tests + schema integrity 12)"
   ```
   注意：`git add -A` 会包含 `backends/`、数据工件等 ✗ 提交前需按 `.gitignore` 与人工清单筛选
   （大文件/凭据/venv 不得入库 ✗）。
2. **事故证据归档**：把 `/tmp/destroyed_schemas.py.txt`、`/tmp/e03c_schema_integrity/`
   （重建版快照 + 清单原始输出）复制到仓库外的安全位置（或 `docs/incidents/` 下的纯文本记录 ✗
   不含可执行代码），以便后续审计 ✓。
3. **高风险文件人工复核清单**（建议逐行过目）：
   `recommendation/demo/schemas.py`（事故文件）、`recommendation/demo/agent_chat_adapter.py`、
   `recommendation/demo/serialization.py`、`recommendation/demo/runtime.py`、
   `recommendation/memory/service.py`、`recommendation/memory/extraction.py`、
   `recommendation/preference_matching/matcher.py`、`recommendation/demo/feedback.py`。
4. **流程改进**：批量改写脚本必须在写入前断言目标路径（如 `assert p.name == "schemas.py"`）、
   写入前打印绝对路径与 diff 摘要；禁止「读 A、写 B」共享变量；每次批量改写后立即 `git diff --stat`。
5. **检查点纪律**：此后每个里程碑（E-04A 之前）都建立一次检查点提交，避免连续数轮成果再次
   处于「不可恢复」状态。

**不得**把当前重建代码标记为「可信历史基线」✗：它可以作为**新的检查点起点** ✓，
但历史一致性的限制（§2/§10）必须随检查点说明一起保留。

## 12. 最终 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall      →  32 modified + 57 untracked
```

本轮新增未跟踪：`docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md`、`docs/V2_E03C_SCHEMA_INVENTORY.md`、
`tests/test_v2_schema_integrity.py`（+2 = 上一轮 55 → 57 ✓）。
本轮改动的已跟踪文件：仅 `recommendation/demo/schemas.py`（视图类顺序修复）✓ ——
modified 计数不变（32），因为该文件本就在 modified 集合内 ✓。

## 13. 最终结论

**PASS（可观察契约核验通过；事故前字节级一致性无法证明）。**

```text
① 可取得的可信历史契约（HEAD 基线 + 消费者代码 + 13 份阶段报告 + 1070 项测试）
   ↔ 当前重建 Schema：字段、类型、required/optional、默认值、约束、嵌套、序列化、
     向后兼容与 OpenAPI 形状**全部一致** ✓
② 事故未造成其他未处理损坏：单文件、单次写入，其余文件核查无异常 ✓
③ 本轮发现并修复 1 个真实缺陷（4 个字段的 ForwardRef → OpenAPI 形状退化），
   并以 12 项专项测试锁定 ✓
④ 历史回归（E-01A…E-03C、Legacy、Serving、Provider replay、Session 隔离）1070 项 0 失败 ✓
⑤ 恢复限制（§2/§10）：原始未提交文件不可逐字恢复；字段描述措辞/字段顺序/注释不可证明 ✗
```

本轮到此停止：不自动提交、不进入第 6 阶段。
