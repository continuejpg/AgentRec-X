# P14-E Shopping Preference Benchmark — Specification v0.1

> 性质：独立的 AgentRec-X **购物偏好** Benchmark 草案（pilot_dev）。真实商品来自 P13-R Electronics C4 sampled catalogue；偏好历史为**构造模板**，不是真实用户购买记录。
> 版本：`agentrecx.p14e.shopping_preference.v0.1`；生成器 `p14e-0.1.0`；评测逻辑 `p14e-labeling-0.1.0`；固定随机种子 `20261108`。

## 1. Benchmark 目标

在真实商品目录上评价 Agent 的以下能力，而不是只测“最终命中了哪个商品 ID”：

1. 从合法的长期历史中**恢复**当前有效偏好，并保留来源；
2. 处理**偏好更新、撤回、冲突**（不应继续使用失效偏好）；
3. 在**当前需求 + 历史偏好 + 硬约束**共同作用下，检索并推荐满足证据的商品；
4. 在证据不足/无匹配时**合理 ASK 或拒绝**，而不是编造满足；
5. 全过程中不把 evaluator-only 的 persona/标签/答案泄漏给 Agent。

Pilot 目的：冻结数据/协议/隔离方式，验证可扩展到 200–500 条；**不**训练模型、**不**运行正式四组 Agent Benchmark。

## 2. 数据来源与局限

| 项 | 现实情况 |
| --- | --- |
| 商品目录 | `data/p13r/electronics/catalog_meta.jsonl`（Amazon-C4 sampled Electronics metadata；sha256 见 P13-R manifest） |
| 目录规模 | 76,918 行；身份校验通过 **76,263**（655 条隔离：book/media 482、空/过短文本 173） |
| 可分类商品 | **11,092** 条（标题可识别为 headphones/earbuds/keyboard/mouse/speaker），构成 benchmark 的候选池；其余 65,171 条 untyped 商品仍进入检索索引，但不进入 labeled pool |
| Namespace | **`amazon_electronics_c4`**（独立于 P14-A Shopping Companion 与旧 Sports 整数 ID 空间） |
| 属性来源 | **只有 `item_id` / `category=Electronics` / `text`（标题+描述）**。属性由文本正则证据抽取（wireless、noise cancelling、over-ear、mechanical 等 17 个属性）；每条属性带 evidence span |
| 缺失字段 | **price / brand / size / material / stock = UNKNOWN**（源文件没有对应字段；不编造、不 LLM 补全） |
| 历史偏好 | 全部为**构造模板**（`history_source=constructed_template`，`profile_source=synthetic_scenario_profile`），不得称为真实购买记录 |
| 可用性 | 仅 pilot_dev（40 条）；非 frozen test；不可改名为从未见过的独立 Test |

已知局限：属性证据仅覆盖文本中出现的词（UNKNOWN 比例高）；目录是 C4 sampled 而非完整 Electronics 目录；`no_verified_match` 只表示“当前 76,263 条样本内无可验证匹配”，不等于现实世界不存在。

## 3. Schema v0.1

机器可读 schema：`data/p14e/benchmark/schema_v0.1.json`。两个**物理分离**的文件：

### 3.1 `agent_tasks.jsonl`（Agent-visible）

必填：`schema, task_id, split, user_id, persona_id, current_query, current_time, history_sessions, allowed_tools, catalog_namespace, tool_budget, public_constraints, construction_metadata`。

- `history_sessions`：`{session_id, date, turns:[{role:"user", content}]}`；只有用户原始表达，不含结构化偏好标签；
- `construction_metadata`：`history_source / profile_source / construction_method / generator_version / random_seed / review_status`；
- 不包含：persona 结构、有效/撤回偏好、硬约束结构、候选标签、expected decision、evidence snippet、target id、task_type、intent note。

### 3.2 `evaluator_only.jsonl`（Evaluator-only）

必填：`task_id, task_type, split, user_id, persona_id, persona, effective_preferences, withdrawn_preferences, hard_constraints, catalog_namespace, product_type_pool, expected_decision, decision_rationale, labels, construction_metadata`。

- `effective_preferences` / `withdrawn_preferences` 带 `source_event_id`、`source_text`、`source_type`；
- `labels`：verified candidate count/sample、pool size、hard verified/unknown/violated 计数、候选证据样例；
- `hard_constraints`：结构化（attribute/form_factor/unverifiable）与 `source_text`；不可验证项显式标 `unverifiable`。

另有 `history_events.jsonl`（冻结事件 + 偏好候选标注 + 来源）、`personas.jsonl`（合成 persona 定义）、`manifest.json`、`review_checklist.json`。

## 4. 偏好历史构造规则（固定顺序）

**persona/偏好设计 → 历史事件 → 冻结历史 → 当前 query → 评价规则**。严禁先选商品再编历史。

1. **Persona**：按商品类型/场景定义正/负偏好集合；只使用能在真实目录文本中找到证据的词表；若有组合无证据支持，则换用有支持的同义组合并记录（不得为制造差异而编造属性）。
2. **历史事件**：模板生成 2–5 个按时间排序的 session；每个事件记录：
   `event_id, session_id, timestamp, sequence, raw_text, preference_candidates[{kind,value,polarity,status}], source_type, history_source, profile_source, construction_method, generator_version, review_status`。
   语义覆盖：`assert`（肯定/否定）、`withdraw`（撤回）、`supersede`（更新旧偏好）；自然语言使用对 M9 extractor 友好的表达，但**数据集本身不依赖 memory 工具是否提取成功**。
3. **冻结**：写入 `history_events.jsonl` 后，`agent_tasks.history_sessions` 与其逐字一致（验证 40/40）。
4. **当前 query**：模拟自然购物请求；**不重复**历史软偏好；仅在需要时显式给出当前硬约束（如 "It needs to be wireless."），并与历史软偏好区分。
5. **evaluation rules**：独立 labeler 在真实目录上重新计算，见 §5；生成阶段不选 target product。

来源字段（每条任务/事件）：`history_source=constructed_template`、`profile_source=synthetic_scenario_profile`、`construction_method=template_generation_v1`、`generator_version=p14e-0.1.0`、`random_seed=20261108`、`review_status=automated_checked|pending_human_review`。未进行人类核查 ⇒ 不允许标 `human_verified`。

任务类型分布（40 条）：regular_history 12、same_query_persona_pair 10（5 对同 Query/不同历史）、preference_update 3 + preference_withdrawal 2 + preference_conflict 3（动态共 8）、insufficient_information 5、no_verified_match 3 + unverifiable_constraint 2（合理拒绝/证据不足共 5）。

## 5. 评价协议（可扩展，不实现复杂训练）

### 5.1 Search / Candidate Reachability

- 对 `expected_decision=recommend` 的任务，用 Agent 实际检索结果（或独立 BM25/BGE 探针）与 **verified candidate set** 求交；
- 可报告：已标注合格商品命中（hit@k）、候选池可达性、当前目录覆盖、未标注数量；
- **未标注商品不是负例**；`no_verified_match` 只相对于当前目录/证据。

### 5.2 Preference Satisfaction

对候选逐条计算（labeler，非 LLM）：
- 属性：`satisfied`（文本有直接证据）/`violated`（负偏好且文本有证据）/`unknown`（无证据，不判负）；
- 硬约束：`verified` / `violated`（互斥 form factor 反向证据）/`unknown`；
- **硬约束违规与软偏好不满足分开计数**；
- 可接受候选（acceptable）：硬约束全部 verified + 无软偏好 violated + 至少一个正偏好 satisfied（若存在正偏好）。

### 5.3 Memory Recovery

- 评价 Agent 从**合法历史**恢复当前有效偏好：召回/精确率/F1；逐条可追溯到 `source_event_id`；
- 撤回偏好必须**不再 active**；更新后旧偏好不得 active；冲突偏好应触发澄清而不是任意选边；
- M9 输出为自由文本，评价端允许**有记录的规范化**（`P14E` smoke 使用关键词 canonicalisation），规范化规则需版本化；
- 若 Agent/工具无法用当前 query 命中历史词（M9 lexical read 语义），允许使用**由 agent-visible 历史构造的 probe query**并单独报告，不能使用 evaluator 标签生成 query。

### 5.4 Agent Decision

按 `expected_decision` 判断：
- `recommend`：最终推荐任一位于 verified set 的商品即可（允许多个合法答案）；可同时评价是否显式引用偏好来源；
- `clarification_required`：发出与缺失信息相关的 ASK/澄清，或明确说明需要澄清；不得直接给确定推荐；
- `no_verified_match` / `insufficient_evidence`：不得把 `unknown` 当满足；允许合理拒绝/说明证据不足；
- 不只用“命中预设商品 ID”评价 Agent。

## 6. 防泄漏与防自我验证

1. **文件物理隔离**：Agent-visible 与 evaluator-only 分文件；测试扫描禁止键/子串（0 命中）。
2. **构造与标签分离**：生成器不计算标签；labeler 只读冻结资产+目录，重算标签（确定性）。
3. **事实来源**：属性必须来自真实文本证据并保留 span；缺失 → UNKNOWN，不补全。
4. **无 target 注入**：候选池不由参考商品构造；检索/ledger/eligibility 不接收 target id。
5. **人工审核**：无真实人类审核 ⇒ 只能报 `automated_checked` / `pending_human_review`；Harness 自己的复核不算独立人工标注。15/40 任务进入 review_checklist。
6. **保留风险**：文本正则可能误匹配（如 accessory 标题含 "headphones"）；已通过任务抽样与 review 队列标记，未声称已确认。

## 7. 数据切分与扩展策略（冻结原则）

- 本轮 40 条 = **pilot_dev**；只用于结构验证/人工审查/协议调试；不得改名为 frozen test。
- 扩展到 200–500 时：
  1. **persona/user 级隔离**（同一 persona 不跨 train/dev/reserved-test）；
  2. **任务模板族隔离/近重复检测**（同模板族归入同一 split）；
  3. **same-query 对照组作为完整 group**，不能拆到不同 split；
  4. 历史引用不得跨不该共享的 split（同一 user 的历史只在其 split 内）；
  5. **reserved-test 独立生成、封存**，不参与选模型/调权重；生成后冻结哈希；
  6. train/dev 的标签和 review 状态可迭代；reserved-test 只读。

## 8. 当前限制（必须随结果一起报告）

- 属性词表基于文本证据，UNKNOWN 比例高（约 92.75% 的候选池商品至少缺一个请求属性证据）；
- P13-R 目录是 C4 sampled，不是完整 Electronics；无 price/brand/material/stock；
- 偏好历史是模板合成，不能作为真实购买偏好证据；
- M9 的 read API 是 lexical 的，current query 往往不含历史属性词；更新/撤回语义依赖 extractor（smoke 中 form-factor 撤回未被 M9 识别，造成 1 个 stale active preference）；
- 15/40 任务待人工审查；在完成审查前只能给出 LIMITED 结论。
