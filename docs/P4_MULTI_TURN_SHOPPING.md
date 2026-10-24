# Phase 4 — 多轮购物意图与自适应推荐

> 在现有 AgentRec-X 架构上闭环：结构化当前购物意图 → 基于意图检索 → 冲突/信息不足时澄清并可继续 →
> 基于最终列表与可验证事实给出理由。
>
> **工程验证，不是线上用户效果。** 36 个构造场景、76 轮、12 段真实历史；所有标签由手写规则与
> 独立目录谓词核验。

## 0. TL;DR

| 指标 | 单轮基线 | Phase 4 |
|---|---|---|
| 当前意图状态准确率 | 21.1% | **100%** |
| 过时约束残留轮次 | 0 / 36（无意图可残留） | **0 / 36** |
| 硬约束违反率 | **43 / 380 = 11.3%** | **0 / 319 = 0%** |
| 未决事实被展示 | 118 | **0** |
| 冲突处理正确率 | 0 / 4 | **4 / 4** |
| 澄清闭环成功率 | 0 / 8 | **8 / 8** |
| 检索适应率 | 52 / 52 | **52 / 52** |
| 推荐理由可追溯率 | 0 / 0 | **561 / 561 = 100%** |
| 端到端任务完成率 | 0 / 36 | **36 / 36** |
| p50 / p95 | 1489 / 1902 ms | **1473 / 1929 ms** |

- **生产改动**：新增 `recommendation/demo/shopping_intent.py`；扩展 policy、service、schemas；
  修正“澄清时仍渲染候选列表”的展示顺序。RRF、候选深度、排序、完成守卫、冻结评测资产未动。
- **没有第二次记忆存储**：意图是 `PreferenceMemorySnapshot` 上的只读投影；本轮提取是预览，
  推荐完成后仍由 loop 只写一次。
- **澄清可继续**：`run_shopping_turn` 返回 `clarification`，下一轮带 `clarification_id` 回答；
  意图更新后重新推荐。未验证商品不展示，硬约束不放宽。

---

## 1. 当前意图状态与更新规则

### 1.1 数据来源

```
PreferenceMemoryService（唯一存储）
  └─ get_active_preferences(user_key)      # ACTIVE entries，带 source_turn_id/source_text
  └─ extractor.extract(current_message)    # 本轮预览，不写库
        ↓
shopping_intent.derive_intent(snapshot, extraction)
  └─ hard / soft / conflicts / turn_ids / active_preference_count
```

- 轮内预览应用与 memory 相同的语义：`REMOVE` 先墓碑化匹配项，`ADD` 去重追加，
  `REPLACE` 先撤销同 kind/polarity（或 `replaces` 指定值）再追加。
- 推荐完成时 loop 仍执行 `persist_memory_stage`，**本轮只写一次**；澄清轮由 service 补写一次
  （`_commit_clarify_turn`），否则用户回答后冲突/约束会丢失。
- 支持：新增、替换、撤销、放宽（用更大的上限 `REPLACE` 旧值）、硬/软区分、来源轮次、显式冲突。

### 1.2 硬/软规则（保守）

| 类型 | 判为 hard 的条件 | 否则 |
|---|---|---|
| `price_max` / `price_min` | 原文含 under / budget / at most / at least / maximum / minimum 等 | soft |
| `category` / `brand` / `color` / `material` / `feature` | 原文含 must / only / exactly / required | soft |
| 请求显式 constraints | 始终 hard（`forced_hard=True`，不依赖措辞） | — |
| `free_form_constraint` | 仅当显式 constraints | 默认 soft，不参与 eligibility |

模糊表达默认 soft：soft 只影响排序/理由，**不能排除商品**。

### 1.3 冲突

- `price_min > price_max` → `price_range`；
- 同一品牌同时被 hard prefer / hard avoid → `brand_contradiction`；
- 冲突以 `Conflict(code, detail, question, options)` 显式记录，并直接生成澄清问题；
- 不做“猜一个能兼容的值”。

---

## 2. 检索适应

`IntentQueryPolicy` 包装现有 plan policy，只重建 `SEARCH_CATALOG` 的参数：

```
hard category → hard brand → soft category → soft brand → soft feature → 用户消息词
去重后最多 8 个（SearchCatalogArguments 自己的上限）
```

- 被撤销/替换的值不在意图里，因此**不会进入查询**；
- 价格不作为词项：仍由确定性 eligibility 检查 `price_max/price_min`；
- 不使用目标商品、未来交互或目录答案构造查询；
- 原有 RRF、候选深度、融合与最终排序不变（只换 lexical terms）。

---

## 3. 澄清与继续执行

### 3.1 触发

`ClarificationPolicy`（包装 plan policy）：

1. **预检索冲突**：第一步直接 `ASK_CLARIFICATION`，不跑任何来源；
2. **无可验证候选**：计划本来要 FINISH 时，若 `verified_eligible_count == 0 and unresolved_count > 0`，
   问“无法核实 … 共 N 个候选；能否给出具体值或放宽？”——不声称这些候选不合格；
3. 每次 run 最多问一次；只有 controller 提供该 action 时才问。

### 3.2 终态与继续

- 澄清是 **suspension**：`status=waiting_for_user / termination_reason=awaiting_user`，
  列表为空，`recommendations` 为空；不是成功、不是失败；
- 响应新增 `clarification`（id/question/kind/blocks）与 `intent`（hard/soft/conflicts）；
- 下一轮请求带 `clarification_id` + 回答文本；service 校验 session、取出 pending，回答进入
  意图预览并重新推荐；
- pending 只存 question/session/kind（进程内），**不是第二份偏好存储**；重启丢失，响应中也带问题文本；
- 同一 session 最多一次澄清；第二次仍不可判定时走既有 `failed/no_verified_candidate`，不循环；
- 完成守卫、`succeeded=false`、未验证商品不展示：全部保持。

### 3.3 展示顺序修复

`LoopController._finalize_response` 现在**只要本次 run 有 clarification 就渲染问题**（此前条件是
“没有产生候选”）。原逻辑在“检索到了候选但都不可验证”时会把空结果解释当成回答，用户看不到问题；
修复后问题是消息正文，且不展示任何商品。

---

## 4. 基于证据的推荐理由

`AgentRecommendationView.reason`（新增可选字段）由 `shopping_intent.build_reason(record, intent)`
生成，逐商品、按身份绑定：

- hard price：`within your $30.00 budget (catalogue price $8.99)`；
- hard price_min：`above your $80.00 minimum (catalogue price $84.99)`；
- hard category/brand：只有目录确实匹配才写 `matches required …`；
- soft category/brand：只有匹配才写 `matches your interest/preferred brand …`；
- colour：只有目录 `Color` 存在且匹配才写；
- 无匹配时退回 `catalogue facts: brand …, category …, price $…`，字段缺失则省略；
- 没有可用事实时明确写 `no catalogue fact was available…`。

没有 LLM 评分器，没有模型生成商品事实；理由永远来自该商品的目录记录。最终列表变化时理由随
`recommendations` 一起重建（Phase 3.1 的一致性机制仍生效）。

---

## 5. 生产代码差异

```
 recommendation/control/loop.py       |  12 +-
 recommendation/demo/agent_policy.py  |  94 ++++++++++++
 recommendation/demo/agent_schemas.py |  62 +++++++++
 recommendation/demo/agent_service.py | 212 +++++++++++++++++++++++++++++++-
 4 files changed, 372 insertions(+), 8 deletions(-)
新增:
 recommendation/demo/shopping_intent.py
 experiments/p4_multi_turn_eval.py
 tests/test_p4_shopping_intent.py
 tests/test_p4_clarification_policy.py
```

- 新增 API 字段（全部可选、追加）：`clarification_id`（请求）、`clarification` / `intent`（响应）、
  `reason`（推荐项）；旧请求/响应不受影响。
- 未改：RRF、Learned Ranker、候选深度、来源计划、冻结 evaluator / 预注册 / 检查点。

---

## 6. 自动化测试

```bash
pytest -q tests/test_p4_shopping_intent.py tests/test_p4_clarification_policy.py \
         tests/test_demo_agent_api.py tests/test_agent_chat_adapter.py \
         tests/test_agent_control_plane.py tests/test_p3_constraint_empty_message.py \
         tests/test_p3_final_list_consistency.py tests/test_phase32_terminal_semantics.py
# 135 passed
```

覆盖：硬/软判定、替换、撤销、放宽、冲突、强制 hard 的显式约束、查询词优先级与去重、
理由身份绑定、未决问题不声称不存在、policy 只在冲突/未决时提问、每次 run 只问一次、
wire 字段可选性。

---

## 7. 多轮场景评测

### 7.1 场景集

- 12 段真实历史（index 12000–12011）；**36 个场景 / 76 轮**，每个 2–3 轮；
- 类型：价格新增→替换（12）、品牌新增→撤销（8）、类别新增→撤销（8）、
  价格冲突→澄清→继续（4）、不可验证约束→澄清→继续（4）；
- 每轮标签手写：期望 hard 集合、soft 集合、是否应澄清及类型、是否应有推荐、应消失的旧值；
- 独立核验：商品价格/品牌/类别来自目录记录；理由逐条声明与记录比对；
- 基线：同场景、同一 runtime，改用单轮 `run`（无意图、无澄清、无理由）。

### 7.2 结果

| 指标 | 基线 | Phase 4 |
|---|---|---|
| 意图准确率 | 21.1% | **100%**（76/76） |
| 过时约束残留轮次 | 0/36（无意图） | **0/36** |
| 硬违反 / 展示商品 | 43 / 380（11.3%） | **0 / 319** |
| 未决事实被展示 | 118 | **0** |
| 冲突处理 | 0/4 | **4/4** |
| 澄清闭环 | 0/8 | **8/8** |
| 检索适应 | 52/52 | **52/52** |
| 理由声明可追溯 | 0/0 | **561/561（100%）** |
| 场景完成 | 0/36 | **36/36** |
| p50 / p95 | 1489 / 1902 ms | **1473 / 1929 ms** |

基线任务完成率 0 是标签口径（含意图/澄清期望），不代表基线不返回商品；它的 43 个硬违反和
118 个未决展示才是关键差异。

### 7.3 产物

| 文件 | SHA-256 |
|---|---|
| `runs/p4_multi_turn_eval_v3/config.json` | `7ed5de730597480afe245865c090b8520f9ce9293072323e673e905630dd1ea8` |
| `runs/p4_multi_turn_eval_v3/samples.jsonl` | `d54b0e32f4a2a136ae28f63a576563f8ac545d8b3d3dd8a4bc9a56544d77129f` |
| `runs/p4_multi_turn_eval_v3/summary.json` | `9502ea63d18b6818af6e55b64180442029a18c12265abe05568de42a6969c5bf` |
| `runs/p4_multi_turn_eval_baseline/summary.json` | `fda27de958d8b41a4b70a8078c009adab59501c841706f23f3c5b8641e1a9d3c` |
| `experiments/p4_multi_turn_eval.py` | `a9dadf599d3735c1546a6ee331e3c88a282c8cb587363796735e45d6efc9de04` |
| `tests/test_p4_shopping_intent.py` | `651e64f67f54cc3e57e7f7e9f330175d8f1b1bfe8fa7e3b3180d101fceb7af0a` |
| `tests/test_p4_clarification_policy.py` | `747cb740ce37435e5f560719c0b366dcdeff83d9d18625fd5f2b24e5d4ca8a46` |

`runs/` 由 `.gitignore` 忽略，未提交。

### 7.4 复现

```bash
# Phase 4（36 场景）
.venv/bin/python -m experiments.p4_multi_turn_eval --out runs/p4_multi_turn_eval_v3

# 同一场景的单轮基线
.venv/bin/python -m experiments.p4_multi_turn_eval --baseline --out runs/p4_multi_turn_eval_baseline
```

---

## 8. 技术决策记录

1. **不新建记忆存储**：意图是现有 snapshot 的投影；本轮预览避免“先写再读”的双写。
2. **硬/软由措辞判定且默认 soft**：模糊表达不能排除商品；显式 constraints 强制 hard。
3. **价格不进 lexical query**：价格是确定性 eligibility 维度，不是检索词。
4. **澄清是 suspension**：`waiting_for_user`，不是成功/失败；问题由 policy 生成、loop 渲染。
5. **pending 存 question 而非偏好**：继续交互所需的最小状态；真正的偏好在唯一 memory store。
6. **理由逐商品、逐事实生成**：没有 LLM 评分器，理由不创造目录没有的事实。
7. **一次澄清上限**：避免问题循环；第二次不可判定回到既有终态语义。

---

## 9. 尚未解决

1. **HTTP 端点**：Phase 4.1 已接入
   `POST /v1/demo/sessions/{session_id}/agent/shopping`（见
   [`docs/P4_1_SHOPPING_API.md`](P4_1_SHOPPING_API.md)）。
2. **pending 是进程内状态**：重启/多 worker 丢失；响应带问题文本可让客户端重试，但严格可恢复
   需要持久化（Phase 4.1 已明确记录该限制，未新建数据库）。
3. **提取器词表限制**：自然语言 hard brand（“must have brand X”）不解析；类别替换（“make it
   camping instead”）不解析，因此场景改用“撤销”验证；hard brand 只能走显式 constraints。
4. **一次澄清后仍不可判定**会落到 `failed/no_verified_candidate`，不会再次提问。
5. **软偏好仍是排序目标**：检索适应率用“Top-5 是否出现当前意图词”衡量，不是查询字符串审计。
6. **36 个构造场景**不是线上分布；基线是“功能缺失”对照，不是调优后的系统。
7. 未跑全量测试；未提交 Git。
