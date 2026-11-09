# P16 — Two Reusable Shopping Skills

> 范围：P16 only。在现有 Agent Core 上交付两个可执行、可复用的 Skills：
> `shopping-comparison` 与 `preference-clarification`。
> 不新增 Planner/Memory/Search/RAG Runtime，不训练模型，不重建索引，不运行 Reserved Test。

## 0. 结论

- **两个 Skills 都是真实可调用工作流**，不是 Markdown 占位：
  - 通过现有 `ElectronicsPlannerExecutor.invoke_skill` 调用；
  - 复用既有 BGE catalogue、Product Info、P15 Evidence Bundle、Claim Validation、M9 Memory 和 Gate。
- **Shopping Comparison**：4 个有效比较案例、9 个真实商品、72 个有效 claims、0 个 rejected claims、54 个 UNKNOWN 字段；0 次 Qwen 调用；平均 Skill 延迟约 **1.49 ms**（这些是结构化比较任务，不能与 P15 不同任务的 11.6 s 直接算优化百分比）。
- **Preference Clarification**：5 个 ASK/等待案例；3 个真实回复更新；冲突回复成功移除 `wired` 并保留 `wireless`；withdrawal 移除 `wireless` 且保留 inactive audit；重复回复 idempotent；未回复不 FINALIZE；跨用户读取被拒绝。
- **P16 = GO（工程技能可复用）**；语义真实性仍未人工核验，`semantic_truth_verified=false`，不能把自动结构验证当作自然语言真实性证明。

## 1. Skills 与最小注册

目录：

- `skills/shopping-comparison/SKILL.md`
- `skills/preference-clarification/SKILL.md`

每个 `SKILL.md` 都有 YAML frontmatter（`name`、`description`）以及：

- 触发条件与不适用条件；
- 输入契约；
- 工具调用流程；
- 输出契约；
- 失败/终止条件；
- 证据与安全要求。

实现入口：

- `recommendation/skills/registry.py`：最小显式 `SkillRegistry`，不动态加载、不训练 Router；
- `recommendation/skills/shopping.py`：比较 Skill；
- `recommendation/skills/clarification.py`：澄清 Skill；
- `ElectronicsPlannerExecutor.invoke_skill(registry, name, **arguments)`：现有 Agent executor 的薄入口，默认不存在 Skill 调用时行为不变。

没有新增第二套 Search Backend、Product Store、Memory Store、Eligibility、RAG validator 或 Planner Runtime。

## 2. Shopping Comparison Skill

### 输入与验证

- 2–5 个显式真实商品引用（`parent_asin` 或 `namespace:parent_asin`）；
- 当前 query、当前用户合法 M9 history；
- Namespace、identity、重复 ID、Product Type/Role 全部校验；
- 无效身份/错误 Namespace/角色不符返回明确错误，要求用户提供合法主商品引用，不猜测。

### 流程

1. 解析真实商品 identity；
2. 调用现有 `catalogue.inspect`（Product Info）读取 title/price/brand/material/stock/warranty 等；
3. 校验 Namespace、Role；
4. 每件商品构造独立的 P15 Evidence Bundle（真实 raw field + 精确 span + UNKNOWN）；
5. 生成 `product_fact` / `unknown` claims，复用 P15 `validate_claims`；
6. 对共同证据字段做确定性横向比较；
7. 显示 active preference 的 catalogue-text observation（observed / not_evidenced），不伪造“用户喜欢”；
8. 提供条件性最佳匹配或明确说明无法可靠选择。

### 结果

| 指标 | 值 |
| --- | ---: |
| Comparison cases | 6 |
| 有效比较 | 4 |
| 预期拒绝（无效 identity / Namespace） | 2 |
| 比较真实商品 | 9 |
| 唯一商品 ID | 9 |
| 有效 claims | 72 |
| 拒绝 claims | **0** |
| UNKNOWN 字段数 | 54 |
| Qwen 调用 | **0** |
| 平均 Skill 延迟 | **1.49 ms** |
| Role mismatch 测试 | 拒绝（不把配件当主商品） |

示例：

- 两件键盘、无历史的比较：`B09828X5WY` vs `B00EWHGDVC`；
- 三件耳机、有历史偏好：`B0BLSQ5T75`（E7 Max，over-ear/noise cancelling）、`B08HM3MLJF`、`B0BQ38B3NC`；
- 负偏好场景：`B004K6R43G` vs `B00008VF63`（earbuds），over-ear/wireless/water resistance 关系不虚报；
- 所有有效比较均输出 price/brand/stock/material/size/warranty **UNKNOWN**，不编造质量、库存或材质事实；
- 条件性选择只依据 active preference 与 catalogue text 的 observed 关系，明确声明“不是客观质量排名”。

## 3. Preference Clarification Skill

### Ask

- 优先复用现有 Decision Gate 的 `ask_question`；
- 无 Gate question 时，用 active conflict 或缺失关键需求选择一个具体问题；
- 返回 `awaiting_clarification`、`selected_product_id=null`、`finalize_allowed=false`；
- 未回复时绝不 FINALIZE、绝不推荐。

### Resume（真实用户回复）

- 必须有真实非空 user message 与稳定 `turn_id`；
- 写入：现有 `PreferenceMemoryService.process_turn`（M9 合法 user-event writer）；
- 重新读取 `HistoryStateReader`，保留 active/inactive audit；
- 更新/否定/撤回由现有 memory 语义处理；
- 相同 `turn_id` 重复提交为 idempotent，不重复写入；
- 读取授权用户使用 `conversation_id == user_key`；跨用户请求被现有 reader 拒绝。

### 结果

| 案例 | 结果 |
| --- | --- |
| `clr_missing_need` | 缺失需求 → ASK；真实回复 “noise cancelling + wireless” → 新增 `wireless option`，恢复 Rule 流程 FINALIZE |
| `clr_conflict` | wired/wireless 冲突 → ASK；真实回复 “Forget my wired preference. I prefer wireless.” → 移除 wired、新增 wireless，恢复 Rule FINALIZE |
| `clr_withdrawal` | “Forget my wireless preference.” → wireless 从 active 移除，inactive audit 保留 |
| `clr_unanswered` | 保持 `awaiting_clarification`；selected=null；finalize=false |
| `clr_duplicate_reply` | 第一次 added=1；第二次 `already_processed=true`，active 数量不变 |
| `clr_cross_user` | 跨用户 conversation read 被 `HistoryStateReader` 拒绝（PermissionError） |

用户回复均为显式标注的测试输入 `explicit_test_user_reply`，不是真实线上用户行为，也不是 evaluator answer。

## 4. E2E 与现有 Agent 集成

12 条 E2E（6 comparison + 6 clarification），均保存完整轨迹：

- `runs/p16/e2e/*.json`：输入、真实工具调用、Product Info/Evidence Bundle、claims validation、内存更新、Gate 状态；
- `runs/p16/e2e_trajectories.json`、`runs/p16/summary.json`、`runs/p16/metrics.json`、`runs/p16/per_case.jsonl`。

真实调用链：

```text
Existing Agent executor / Rule planner
  -> invoke_skill("shopping-comparison")
     -> catalogue identity/Product Info
     -> P15 Evidence Bundle + Claim Validation
  -> invoke_skill("preference-clarification", operation=ask)
     -> existing Gate ASK
  -> real user reply
     -> invoke_skill("preference-clarification", operation=resume)
        -> M9 process_turn + HistoryStateReader
  -> resume existing Agent shopping flow under existing Gate
```

- 所有 12 条轨迹使用 **BGE `bge_dense`** 检索后端；
- 没有 evaluator-only 输入进入 Skill 或最终用户回答；
- 没有修改 Serving V1/旧 Planner 默认行为；
- 不运行 Reserved Test。

## 5. Claim 与证据

- Comparison 的 72 个 claims 全部通过 P15 自动结构/词面验证，0 个 rejected；
- 每个 product_fact 引用真实 product evidence ID，UNKNOWN 只以 unknown claim 出现；
- `automated_validation=true`、`semantic_truth_verified=false`；
- 没有人工语义审核，不能声称“零语义幻觉”；
- 推荐结论明确是**条件性最匹配**，不是商品客观质量或用户满意度证明；
- 没有为了虚增 Claim 合格率删除所有有意义理由：比较表保留了 title/category evidence、active preference observation 和 UNKNOWN 限制。

## 6. 成本与性能

- Comparison：无 Qwen 调用；4 个有效比较平均 Skill 延迟 **1.49 ms**（真实工具 Product Info + Evidence Bundle 总计）；工具调用为 `catalogue_identity_resolution`、`product_info`。
- Clarification：无 Qwen 调用；使用现有 M9 事件写入；延迟记录见 E2E。
- P15 的解释生成仍是真实 Qwen `/explain`（平均 11.6 s，不同任务），本轮**不能**用不同任务直接声称优化百分比；仅报告 P16 同任务的结构化比较延迟。
- 未训练/微调；未重建 BGE；未增加大型依赖/新服务。

## 7. 失败与限制

- 2 个 comparison 案例为预期失败：无效 product ID、错误 Namespace；都明确要求合法引用，未猜测。
- Role mismatch 测试被拒绝：配件/类型不符不能作为主商品强制比较。
- 用户回复全部为测试输入；没有真实线上多轮用户数据。
- Memory 写路径使用的是现有 M9 `process_turn`；原始 M5 session audit 保留，未新写第二套事件存储。
- 自动 Claim 验证不等于语义真实性；human review 未执行。
- 未实现 P17 MCP，未新增第三个 Skill。

## 8. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `SHOPPING_COMPARISON` | **PASS** | 2–5 真实商品、Product Info、P15 Evidence、Claim validation、UNKNOWN、无效身份拒绝、条件性结论 |
| `PREFERENCE_CLARIFICATION` | **PASS** | Gate ASK、真实回复更新、撤回、idempotency、未回复不 FINALIZE、跨用户隔离 |
| `SKILL_RUNTIME_INTEGRATION` | **PASS** | 最小 registry + 现有 executor `invoke_skill`；12 条真实 E2E；无第二套 Runtime |
| `EVIDENCE_GROUNDING` | **PASS（结构化/自动验证范围）** | 72/72 claims 通过自动验证；但 semantic truth 未人工核验 |
| `LEGACY_REGRESSION` | **PASS** | P16 + P15/P14-C1/D2/C0-R/Eval100 定向回归 43 passed |
| **`P16`** | **GO** | 两个 Skills 已是可复用真实执行工作流；不代表未经人工审核的解释语义质量 |

**进入 P17 MCP 的条件：** Skills 的输入/输出契约、registry、现有 executor 调用入口和 E2E 轨迹已具备；可以进入 P17 MCP 封装。但 P17 不应把 `automated_validation` 当成语义真实结论，仍需要独立语义审核。

## 9. 交付

- `P16_REPORT.md`、`P16_SOURCE_DIFF.txt`
- `skills/shopping-comparison/SKILL.md`、`skills/preference-clarification/SKILL.md`
- `recommendation/skills/{registry,shopping,clarification}.py`
- `experiments/p16/{run_e2e,metrics}.py`
- `runs/p16/e2e/`、`e2e_trajectories.json`、`summary.json`、`metrics.json`、`per_case.jsonl`、`decision.json`
- `tests/test_p16_skills.py`：7 passed；P16 + P15/P14-C1/D2/C0-R/Eval100 定向回归 43 passed
- 修改：`experiments/p14c0/electronics_executor.py`（新增 `invoke_skill` 薄入口；不改变默认执行路径）
