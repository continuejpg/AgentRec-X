# P14-E — Shopping Preference Benchmark Pilot 报告

> 结果文件：`data/p14e/benchmark/`（40 条任务 + evaluator-only + history/persona + schema/manifest/review）；
> 运行证据：`runs/p14e/validation/summary.json`、`runs/p14e/smoke/`；
> 规格：[`P14E_BENCHMARK_SPEC.md`](P14E_BENCHMARK_SPEC.md)。
> 本轮只做 pilot：未训练模型、未实现新评分器、未运行正式 Agent Benchmark、未 commit。
> 环境：Windows 11 + WSL2 CPU-only；数据 76,918 商品读取、BM25 索引 2.4 s、smoke 峰值 RSS 829 MB。

## 1. 实际构建结果

| 项 | 值 |
| --- | --- |
| 任务数 | **40**（`p14e_pilot_0001`–`p14e_pilot_0040`） |
| Split | `pilot_dev`（仅此一个；未冻结、未改名） |
| 任务类型分布 | regular_history **12**；same_query_persona_pair **10**（5 对）；preference_update **3** + preference_withdrawal **2** + preference_conflict **3**（动态共 **8**）；insufficient_information **5**；no_verified_match **3** + unverifiable_constraint **2**（合理拒绝/证据不足共 **5**） |
| 期望决策分布 | recommend **26**；clarification_required **9**；**no_verified_match 3**；insufficient_evidence **2** |
| Persona / 历史事件 | 40 个合成 persona；**149** 个冻结历史事件（109 个带偏好候选；40/40 任务有可用偏好历史） |
| 数据文件 | `agent_tasks.jsonl` 58,811 B；`evaluator_only.jsonl` 290,217 B；`history_events.jsonl` 87,373 B；`personas.jsonl` 17,912 B；`manifest.json`；`review_checklist.json` 15 条；`schema_v0.1.json` |
| 生成器 | `p14e-0.1.0`；labeler `p14e-labeling-0.1.0`；seed **20261108**；全部可重生成（确定性） |

## 2. 使用的商品目录与 Grounding

- 目录：`data/p13r/electronics/catalog_meta.jsonl`（P13-R，Electronics C4 sampled metadata），namespace **`amazon_electronics_c4`**。
- 一致性检查（本轮实测）：**76,918 行**，item_id 全部为 10 位 ASIN 形态且**唯一**；quarantine **655** 条（482 条 book/media 标题、173 条空/过短文本）；**accepted 76,263**；其中可识别商品类型 **11,092**（headphones/earbuds/keyboard/mouse/speaker），其余 65,171 条 untyped 仍进入检索索引但不是 benchmark 候选池。
- 商品属性**只**来自 `text`（标题+描述）的显式词证据：17 个属性（wireless/wired/noise_cancelling/over_ear/on_ear/in_ear/microphone/gaming/waterproof/foldable/mechanical/rgb/rechargeable/multidevice/usb/silent/ergonomic），每条附 evidence span；**price / brand / size / material / stock 一律 UNKNOWN**（源文件没有这些字段；未用 LLM 补全）。
- 局限：文本证据稀疏（**92.75%** 的候选池商品至少缺一个被请求属性证据）；目录是 C4 sampled，不是完整 Electronics；`no_verified_match` 只相对于本目录/证据。

## 3. 构造顺序、来源与随机种子

严格按 **persona/偏好 → 历史事件 → 冻结 → 当前 query → 评价规则** 构建：

1. 依据商品类型/场景定义可被真实文本证据支持的正/负偏好（没有反选 target 商品）；
2. 用模板生成 2–5 个带时间顺序的用户表达事件，标注 `assert / withdraw / supersede` 语义与 `preference_candidates`；
3. 冻结 `history_events.jsonl`；agent-visible `history_sessions` 与冻结文本逐字一致（验证 40/40）；
4. 生成自然当前 query（不重复历史软偏好；硬约束在 query 中显式说明）；
5. 独立 labeler 在真实目录上重新计算 expected decision/候选标签（生成器不算标签）。

来源字段全部写入任务/事件：`history_source=constructed_template`、`profile_source=synthetic_scenario_profile`、`construction_method=template_generation_v1`、`generator_version=p14e-0.1.0`、`random_seed=20261108`、`review_status=automated_checked|pending_human_review`。**未声称 human_verified**（无人类核查参与）。

## 4. 质量检查（D4 的 12 项）

| # | 检查 | 结果 |
| --- | --- | --- |
| 1 | 任务数量/类型分布 | 40 条，12/10/8/5/5（允许的微调：动态 3+2+3 合并为 8，见 manifest） |
| 2 | 有效历史事件覆盖率 | **109/149** 事件带偏好候选；**40/40** 任务有可用偏好历史 |
| 3 | 历史事件与偏好证据一致性 | event replay → evaluator effective/withdrawn：**40/40 一致** |
| 4 | 商品 ID / Namespace 有效性 | namespace 全部 `amazon_electronics_c4`；validated samples **700/700** 在 accepted catalogue 中；无非法 ID |
| 5 | 硬约束可验证比例 | **38/40** 任务的每个硬约束在当前池内至少有一个产品证据（2 个 unverifiable 任务有意为 UNKNOWN） |
| 6 | UNKNOWN 属性比例 | **92.75%**（候选池中至少缺一个请求属性证据的比例；作为已知 grounding 限制报告） |
| 7 | Agent-visible 标准答案泄漏 | 禁止键/子串扫描 **0 命中**；evaluator-only 字段不在 agent 文件；测试覆盖 |
| 8 | 同 Query 不同 Persona 差异 | **5/5** 对：effective preferences 不同；verified candidate sets 的对称差 410–2,048 个商品 |
| 9 | 任务近重复 | **0** 近重复组（query+history 去重后无重复） |
| 10 | 需要人工复核任务数 | **15/40** 进入 `review_checklist.json`（含 same-query 对照、no_verified_match、unverifiable、少量候选任务） |
| 11 | 当前目录候选可达性（独立 BM25 探针） | recommend 任务 26 条：**hit@10 = 22/26（84.6%）**，hit@50 = **23/26（88.5%）**；no_verified_match 任务不强制有命中 |
| 12 | Smoke Test 结果 | 见 §6：8 条代表性任务全部执行成功 |

`runs/p14e/validation/summary.json`：`pass=true`，failures=[], warnings=[]。

## 5. 实际可验证的商品属性（示例）

- 可判定：form factor（over-ear / on-ear / in-ear，互斥）、wireless/wired 证据、noise cancelling、microphone、waterproof、foldable、mechanical、RGB、rechargeable、usb、silent、ergonomic、gaming、multi-device。
- 不可判定（所有任务显式标 UNKNOWN，不当作满足/违反）：price、brand、size、material、stock、warranty。
- 候选可接受性定义：硬约束全部有 verified 证据 + 无负偏好 violation + 至少一个正偏好 satisfied（labeler 确定性计算，非 LLM）。

## 6. Smoke Test（真实工具链，8 条代表性任务）

任务：`0013/0014`（同 Query 两 persona）、`0001`（regular）、`0029`（withdrawal）、`0023`（update）、`0026`（conflict）、`0031`（ASK）、`0036`（no_verified_match）。
工具与结果（详见 `runs/p14e/smoke/summary.json` 与每条 trajectory）：

| 工具 | 是否真实执行 | 结果 |
| --- | --- | --- |
| Memory（P14-A M9+M5，`source_kind=synthetic_fixture`） | 是（8/8） | history-probe 恢复：**recall 1.0**，**F1 0.933**；current-query 直接检索召回低（M9 lexical read 语义，单独记录） |
| BM25 Search（accepted `CatalogSearchSource`） | 是（8/8） | recommend 任务 **hit@10 = 100%**（8/8 探针），真实 catalogue 76,263 条 |
| Product Info（Electronics thin adapter） | 是（8/8） | facts + evidence span；brand/price 明确 UNKNOWN，不补全 |
| CandidateLedger | 是（8/8） | `record_from_source(CATALOG_SEARCH)` 记录 provenance，verified 候选 10/10 入账 |
| Eligibility | 是（8/8） | CATEGORY=Electronics verified；FEATURE=wireless 全部 **UNRESOLVED**（文本证据不进入 ProductMetadata.features，符合“不伪造”预期） |
| 资源 | — | smoke wall **11.78 s**，峰值 RSS **829 MB**，catalogue load 8.24 s，BM25 index 2.44 s |

发现并记录的集成问题（不改旧代码、不掩盖）：

1. M9 的读取是 lexical：用当前需求 query 往往召回不到历史属性词；smoke 额外用“agent-visible 历史拼接 probe”并单独报告（evaluator 标签不参与生成 query）。
2. M9 对 `over-ear`/`in-ear` 这类 form-factor 的 **撤回/更新未被 extractor 识别**（update 任务出现 1 个 stale active preference，F1=0.933）；因此本 benchmark 的“偏好更新”评价不能直接依赖当前 M9，需要后续 memory adapter/能力改进（不在本轮范围）。
3. Eligibility 的 FEATURE 约束因 P13-R metadata 无结构化 features 而 UNKNOWN；CATEGORY 可 verified。该结果已作为 grounding 限制上报，而非把 UNKNOWN 改成通过。

## 7. 未通过 / 需人工复核

- 15/40 任务 `pending_human_review`：same-query 对照 10 条、no_verified_match 3 条、unverifiable 2 条（原因：语义对照/目录覆盖/不可验证硬约束需人工确认）。
- 属性抽取为文本正则，可能误匹配 accessory 标题（review 队列标记，未宣称已确认）。
- 本报告没有任何“已人工确认”的标签；全部标签状态为 `automated_checked` 或 `pending_human_review`。

## 8. 是否能扩展到 200–500 条

**可以，但目前只能作为 pilot_dev 扩展，需先完成人工审查与两项工程前置：**

- 数量：当前模板族/属性词表可生成更多任务，但必须保证 persona/user 级隔离、同 Query group 不拆、近重复检测、历史引用不跨 split（已在 spec §7 冻结）；
- 人工审查：15 条 review queue 必须先审；建议扩到 200–500 时至少审 20–30% 语义标签；
- grounding：文本证据稀疏（92.75% UNKNOWN）导致候选池偏小；扩展前应决定是否接受 lexical-only grounding，或补充结构化 metadata（合法来源）；
- Memory 能力：M9 当前无法可靠处理 form-factor 更新/撤回；P14-B/P14-C 前需要 memory adapter 或评测只报告“当前 M9 能力”；
- 冻结 reserved-test：独立生成并封存，不与 pilot_dev 混用。

## 9. 最终判定

| 判定 | 结果 | 理由 |
| --- | --- | --- |
| `DATA_CONSTRUCTION` | **PASS** | 40 条任务实际生成，schema/文件/manifest/review 完整，可确定性重生成 |
| `HISTORY_PROVENANCE` | **PASS** | 事件/来源/seed/版本齐全；replay 40/40 一致；撤回/更新有来源映射（但无人类核查） |
| `PRODUCT_GROUNDING` | **LIMITED** | 商品身份/namespace 真实有效；但只有文本证据属性，price/brand/material/stock UNKNOWN，unknown 比例 92.75% |
| `LABEL_QUALITY` | **LIMITED** | 标签确定性、独立重算、same-query 5/5 差异；但 15/40 待人工复核，正则 grounding 有误匹配风险 |
| `LEAKAGE_ISOLATION` | **PASS** | 物理文件分离；禁止键/子串 0 命中；labeler 独立；无 target 注入 |
| `ADAPTER_SMOKE` | **LIMITED** | Memory/Search/Info/Ledger/Eligibility 均真实执行并落盘；但 M9 读取 lexical、更新/撤回不完整，Eligibility 的 FEATURE 全 UNKNOWN |
| `LEGACY_REGRESSION` | **PASS** | 未改旧代码；P14-A 74 项 + 旧核心 142 项 = **216 passed**（41.3 s）；P14-E 测试 **8 passed** |
| **P14-E 总判断** | **LIMITED**（可继续扩展，但需先人工审查 + grounding/memory 前置；不是 NO-GO） | Pilot 的数据、隔离、schema 与工具链真实成立；语义标签和 M9 记忆语义尚未达到“可宣称质量”的程度 |

> 下一步（不在本轮执行）：完成 15 条 review queue → 决定 lexical-only grounding 是否可接受 → 按 spec §7 扩到 200–500 并冻结 reserved-test → 再进入 P14-B 的 preference-aware 训练/评测；不得把 pilot_dev 改名成 frozen test。

## 10. 文件与复现

- 生成：`python -m experiments.p14e.build_pilot`
- 校验：`python -m experiments.p14e.validate` → `runs/p14e/validation/summary.json`
- Smoke：`python -m experiments.p14e.smoke`（WSL `.venv`，CPU）→ `runs/p14e/smoke/`
- 测试：`pytest tests/test_p14e_benchmark.py`（8 passed）+ 既有回归（216 passed）
- 数据：`data/p14e/benchmark/`（agent-visible 与 evaluator-only 物理分离；`schema_v0.1.json`、`manifest.json`、`review_checklist.json`）
- 源码差异：`P14E_SOURCE_DIFF.txt`
