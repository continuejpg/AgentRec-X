# Phase 3 — 购物场景评测：约束、可信度与一次最小修复

> 范围：48 个可人工核验的购物场景，覆盖硬约束、软偏好、组合条件、不可满足与不可验证请求；
> 用确定性的目录事实独立核验结果；只实施了一个有证据支持的最小修复（空结果降级说明）。
>
> **不是线上效果评估**：场景是构造的，历史来自离线序列，软偏好只是排序目标而非保证。

> **Phase 3.1 更新**：本文 §3.1/§4.2 记录的“解释列表 ≠ 最终列表”已在 Phase 3.1 修复：
> 非空一致率 **17/38 → 38/38**，事实 **200/200 → 354/354**，非消息输出零变化。
> 见 [`docs/P3_1_FINAL_LIST_CONSISTENCY.md`](P3_1_FINAL_LIST_CONSISTENCY.md)。

## 0. TL;DR

- **硬约束 0 违反**：40 个硬约束场景、59 个被展示商品，独立核验 **0 违反、0 未决被展示**；
  商品身份合法率 **100%**，无重复、无 PAD。
- **发现并修复一个明确缺陷**：约束导致无可行商品时，旧文本说“没有未看过的商品了”（把约束
  结果谎报成历史耗尽）；10/10 个空约束场景都是这个错误解释。修复后 **10/10** 明确说明
  “没有商品满足当前约束（kind=value）”及核验计数。
- **修复只改消息层**：3 个生产文件 +55/−1 行；48 个场景中除这 10 条消息外，**其它字段逐场景完全
  一致**（0 个非消息差异）。
- **残留缺陷（Phase 3.1 已修复）**：非空结果的渲染文本仍只描述 history 源列表，**21/38 个场景的
  消息商品集合 ≠ 最终推荐集合**（平均覆盖率 65.5%）；修复后 **38/38**，见
  [`docs/P3_1_FINAL_LIST_CONSISTENCY.md`](P3_1_FINAL_LIST_CONSISTENCY.md)。
- **软偏好**：8 个软偏好场景，Top-5 命中率均值 **17.5%**（0–60%）；这是排序目标，不是保证。
- 延迟 p50 **1544 ms** / p95 **1887 ms**（单进程 CPU，含两源检索与渲染）。

---

## 1. A — Phase 2C 收口状态

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/agent-source-limit-ablation` / `9eabbfe` |
| 2B 提交 | `a728203` → `9c2df95` → `f215183` → `9eabbfe`（4 个本地提交，未 push） |
| 2C 未提交文件 | `experiments/r_lr05_independent_eval.py`、`r_lr05_baseline_rootcause.py`、`r_lr05_feature_parity.py`、`docs/P2C_LEARNED_RANKER_EVAL.md` |
| 2C 定向测试 | `tests/test_v2_learned_ranker.py` → **13 passed** |
| 2C 脚本可编译 | 三个 `r_lr05_*` 脚本 `py_compile` 通过；报告中的命令与实际产物一致 |

按本轮限制（"不自动提交 Git"），2C 文件保持未跟踪；2B 的四个提交已把依赖顺序固定：
2B.4/2B.5 import 2B.3，2B.6 import 2B.5，2C 驱动独立 import 冻结组件。

---

## 2. B — 场景集与标签规则

### 2.1 场景集

- **8 个真实历史**：离线序列 artifact 的 index `12000–12007`（`test_history = items[:-1]`）。
- **6 类场景 / 8 个历史 = 48 个场景**：

| 类型 | 数量 | 构造 | 期望 |
|---|---|---|---|
| `hard_price` | 8 | 阈值取该历史未约束候选池价格中位数 | 只展示 `price ≤ 阈值` 的商品 |
| `hard_category` | 8 | 取池内某商品的叶子 `categories` 文本 | 只展示类别串包含该词的商品 |
| `hard_brand` | 8 | 取池内某商品的 `Brand Name` / `Manufacturer` | 只展示品牌串包含该值的商品 |
| `hard_combo` | 8 | 价格中位数 + 叶子类别（两条件） | 两个条件都满足 |
| `soft` | 8 | 消息里说“I'd prefer <brand/category> if possible”，无硬约束 | 软目标：Top-5 中匹配占比（不保证） |
| `impossible` | 8 | price_max 0.01 / 不存在品牌 / 不存在类别 / `free_form_constraint`（目录不可判定） | 期望空结果，且说明约束原因 |

- 硬约束场景与对照运行使用**同一条消息、同一 plan**，因此候选池相同；约束是唯一变量。
- 软偏好场景的消息不同（会改变 catalog 查询），所以只测软偏好满足率，不参与硬约束对照。

### 2.2 标签与核验规则（确定性、独立于系统自评）

所有标签都由**我们自己的目录谓词**计算，不读取系统 `eligibility` 视图来判分：

| 约束 | 独立谓词 |
|---|---|
| `price_max` | `parse_price(record.price_text) <= parse_price(value)`；无价格 → `unknown` |
| `category` | `value.casefold() in c.casefold()` for c in `categories + [main_category]`；无类别 → `unknown` |
| `brand` | `value.casefold() in brand.casefold()`，brand 取 `Brand Name`/`Brand`/`Manufacturer`；无品牌 → `unknown` |
| 其它（`free_form_constraint`） | 目录不可判定 → `unknown` |

- **期望非空**：对照池里存在一个独立谓词全部 `satisfied` 的商品；否则期望空。
- **硬违反**：展示商品存在谓词 `violated`；**未决被展示**：谓词 `unknown` 却被展示（违反 fail-closed）。
- **事实核验**：解析消息里的 `Main category/Category/Title/Store/Feature/Detail` 行，与对应
  商品的目录记录比对；允许渲染器的省略号截断。
- **解释一致性**：消息里出现的商品 ID 序列 vs 最终 `recommendations` 的 ID 序列（顺序与集合）。

---

## 3. C — 评测结果

### 3.1 汇总（`runs/p3_scenario_eval_fixed_v2/summary.json`）

| 指标 | 结果 |
|---|---|
| 场景 / 成功 / 失败 | 48 / 48 / 0 |
| 硬约束场景 | 40 |
| 被展示商品数（硬约束场景） | 59 |
| **硬约束违反率** | **0 / 59 = 0.0%** |
| 未决事实被展示 | **0 / 59 = 0.0%** |
| 商品身份合法率 | **100%**（目录存在 + `item_id` 与映射一致） |
| 重复商品 / PAD | 0 / 0 |
| 期望结果命中率 | **40/40 = 100%**（该空的空，该非空的非空） |
| 空结果（有约束） | 10 |
| **空结果给出约束原因** | **10/10（修复前 0/10）** |
| 消息与最终列表一致（非空场景） | **17/38 = 44.7%**（Phase 3.1 已修复为 38/38） |
| 消息对最终列表的平均覆盖 | **65.5%** |
| 目录事实可追溯率 | **200/200 = 100%** |
| 软偏好 Top-5 满足率（8 场景） | 均值 **17.5%**（0.0–0.6） |
| p50 / p95 延迟 | **1544 ms / 1887 ms** |

### 3.2 硬约束：0 违反，且失败关闭

- 价格、类别、品牌、两条件组合：59 个展示商品全部满足独立谓词；
- 无可行商品的 10 个场景全部返回空（8 个 impossible + 2 个 combo），没有“退而求其次”；
- 没有任何“目录无法判定却被展示”的商品（`unknown` 被正确排除）。

这验证了当前代码**已经保证**的约束：硬约束在最终展示集合上执行，且失败关闭。

### 3.3 软偏好：只是排序目标

8 个软偏好场景的 Top-5 匹配占比为 0.2 / 0.2 / 0.0 / 0.0 / 0.0 / 0.6 / 0.2 / 0.2，均值 **17.5%**。
系统不承诺软偏好满足；本阶段也不把它当硬指标。请求声明路径不使用存储记忆的重排（那是
chat 路径的 M10 能力），因此这里的软偏好主要通过消息/查询产生影响——这是设计边界，不是缺陷。

---

## 4. D — 主要失败模式与唯一修复

### 4.1 已修复：空约束结果的解释是假的

**证据**：修复前 10 个“约束导致空结果”的场景全部返回
`No unseen product is left to recommend from this interaction history...`，而实际是
`price_max=0.01` / 不存在的品牌 / 不可验证约束把候选全部排除了。这不是措辞问题，是**错误归因**：
把“没有合规商品”说成“历史耗尽”。

**修复**（3 个文件，+55/−1）：

| 文件 | 改动 |
|---|---|
| `recommendation/agent/rendering.py` | 新增 `build_constraint_empty_response(view)`：只读已计算的 eligibility view，输出约束 `kind=value`、评估数、verified/ineligible/unresolved 计数，以及“不展示不合规或未验证商品” |
| `recommendation/agent/pipeline.py` | `finalize_stage` 在 `constraint_projection.is_empty` 时优先渲染上述文本（不再落到“历史耗尽”） |
| `recommendation/demo/agent_service.py` | 完成守卫**拒绝完成**（无文本）时，若投影为空则用同一文本填充 `message`，避免空白回答 |

**回归测试**：`tests/test_p3_constraint_empty_message.py`（3 项）：
文本包含约束与计数、`finalize_stage` 优先约束原因、无约束时仍保留原“历史耗尽”文本。

**修复前后**（同 48 场景）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 空约束场景给出正确原因 | 0/10 | **10/10** |
| 硬违反 / 未决展示 | 0 / 0 | 0 / 0 |
| 期望命中 | 40/40 | 40/40 |
| 事实可追溯 | 200/200 | 200/200 |
| 非消息字段差异 | — | **0 个场景**（逐场景比较 presented/violations/facts/consistency） |

修复只改变这 10 条消息；非空回答、排序、候选集、硬约束结果全部不变。

### 4.2 残留：解释列表 ≠ 最终推荐列表（Phase 3.1 已修复）

**证据**：38 个非空场景中，只有 17 个的消息商品序列与最终 `recommendations` 完全一致；
平均覆盖 65.5%。原因是 `finalize_stage` 渲染的是 history capability 的 `reranking`/`enrichment`
（5 条历史源候选），而最终答案是多源融合、约束投影后的集合（可能含 catalog 商品，且数量不同）。
消息里的目录事实 100% 可追溯，但它描述的是**另一组商品**。

**Phase 3.1 已修复**：新增 `build_final_list_response`，并在最终列表构建之后用
`_consistent_message` 以最终列表为唯一商品来源重渲染；48 场景一致率 17/38 → 38/38，
非消息输出零变化（见 [`docs/P3_1_FINAL_LIST_CONSISTENCY.md`](P3_1_FINAL_LIST_CONSISTENCY.md)）。
Phase 3 当时的根因分析保留如下：渲染输入是 history capability 的 `reranking`/`enrichment`，
而最终答案在 `_execute_plan` 里由融合后的 presentable 集合决定；正确做法是让渲染输入
成为“最终列表 + 按身份索引的目录事实”。

### 4.3 残留：完成守卫把“无可行商品”判为 failed

10 个空约束场景中 6 个 `finished/completed`，4 个 `failed/completion_refused`（价格 0.01 与
不可验证约束，因为仍有 unresolved 候选）。修复后这 4 个也有解释文本，但 run status 仍是 failed。
建议下一步评估：无任何 verified 候选时，是否应接受 completion 并返回“空 + 约束原因”，
而不是让守卫拒绝完成；这会改动控制面策略，本轮不做。

---

## 5. E — 验收与复现

### 5.1 测试

```bash
# 新增回归测试 + 受影响的控制面测试（本轮 111 passed）
.venv/bin/python -m pytest -q tests/test_p3_constraint_empty_message.py \
    tests/test_agent_graph.py tests/test_agent_control_plane.py

# agent_service 改动影响 Demo Agent API（本轮 38 passed）
.venv/bin/python -m pytest -q tests/test_demo_agent_api.py

# Phase 2C 定向测试（13 passed）
.venv/bin/python -m pytest -q tests/test_v2_learned_ranker.py
```

定向测试合计 **111 + 38 + 13**；未运行全量测试。

### 5.2 运行场景评测

```bash
# 修复后（本报告主结果）
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_fixed_v2

# 修复前基线：同一驱动、同一场景集，在应用修复前的代码上运行
# （已归档在 runs/p3_scenario_eval_baseline_v2/，不覆盖）
```

驱动会先跑 8 个对照（无约束）识别可用历史与候选池，再跑 48 个场景；约 4 分钟（单进程 CPU）。

### 5.3 产物与校验和

| 文件 | SHA-256 |
|---|---|
| `runs/p3_scenario_eval_baseline_v2/config.json` | `e18ac39af1d8450de60779cc154f1b2cd04387fe2e522a7bd91d5b73b2b08014` |
| `runs/p3_scenario_eval_baseline_v2/cases.json` | `b3bf3fae8ef7c3fb93f3e0f6dfb0c9636f18c564908dfc5283bb03a593338fb5` |
| `runs/p3_scenario_eval_baseline_v2/samples.jsonl` | `2a20bcc04dcbe83aeebf48208e932fdc828c6d69b1627f6f499048622877d035` |
| `runs/p3_scenario_eval_baseline_v2/summary.json` | `f18dd0529b218f2c580c34aa9693beb26253c67745f02ac763cd0320a006a839` |
| `runs/p3_scenario_eval_fixed_v2/config.json` | `051f76e2829b6de65224f80844eb4dd9d70f79565129ef62ffd7a63628470fde` |
| `runs/p3_scenario_eval_fixed_v2/cases.json` | `b3bf3fae8ef7c3fb93f3e0f6dfb0c9636f18c564908dfc5283bb03a593338fb5` |
| `runs/p3_scenario_eval_fixed_v2/samples.jsonl` | `6a81ba34de8b7816850c2b1dfe39a53d200a55acf98be88d4569bc68e8695033` |
| `runs/p3_scenario_eval_fixed_v2/summary.json` | `19f6aa438352af64d449320de706a5ec526ae57e118a8f464be7d263d506d38b` |
| `experiments/p3_scenario_eval.py` | `414c1a1ee781e5a02e235f490d095441cb28c4f3c8acc9375435a9939cf2dcb3` |
| `tests/test_p3_constraint_empty_message.py` | `c18cf7fc432d78825455551f1ce38e9eefb5ff07640979c85924d9eb3b2e54c2` |

两个运行目录各自带 `SHA256SUMS.json`；`runs/` 被 gitignore 忽略。

### 5.4 代码改动（未提交，按本轮限制）

```
 recommendation/agent/pipeline.py     | 10 ++++++++++
 recommendation/agent/rendering.py    | 32 ++++++++++++++++++++++++++++++++
 recommendation/demo/agent_service.py | 14 +++++++++++++-
 3 files changed, 55 insertions(+), 1 deletion(-)
新增: experiments/p3_scenario_eval.py
      tests/test_p3_constraint_empty_message.py
      docs/P3_SHOPPING_SCENARIO_EVAL.md
```

没有修改：默认 RRF 排序、Learned Ranker、生产来源计划、冻结 evaluator / 预注册 / 报告 / 检查点。
没有跑全量测试，没有自动提交 Git。

---

## 6. 面试叙事（3 分钟版）

> 我给 Agent 做了一套**可复现的购物场景评测**：48 个场景、8 段真实历史，覆盖价格/类别/品牌/
> 组合/不可满足/不可验证。所有标签都用**独立的目录谓词**核验，不让系统自己判自己。
>
> 结果分两层：工程上，硬约束执行得很干净——59 个展示商品 0 违反、0 未决展示、身份 100% 合法；
> 这证明约束投影和 fail-closed 是有效的。可信度上，我发现一个真问题：约束把候选全部排除时，
> 回答却说“没有未看过的商品了”，把**约束结果**谎报成**历史耗尽**。我用 55 行、3 个文件改掉了
> 这个降级说明，并加了 3 个回归测试；修复前后对比，10 个空约束场景从 0/10 变成 10/10 正确解释，
> 其它 38 个场景的所有结构化结果**逐场景不变**。
>
> 我也没有掩盖剩下的问题：多源回答的文本仍只描述 history 源候选，21/38 个场景的消息商品集合
> 不等于最终推荐集合。这需要一个“融合最终集”渲染器，属于下一步，不是本轮的最小改动。
>
> 整个评测是确定性的、离线的、约 4 分钟；不训练、不改排序、不碰冻结资产。

**一句话总结**：硬约束可信，降级说明已修；解释与最终答案的一致性仍是下一道坎。

---

## 7. 限制与不夸大

1. **构造场景，不是线上用户**：8 段历史 × 6 类场景；约束值从候选池推导，用于覆盖行为，不代表
   真实购物分布。
2. **软偏好 ≠ 保证**：17.5% 的 Top-5 命中只是排序结果；请求路径不启用存储记忆重排，不能据此
   声称“偏好理解有效”。
3. **解释一致性**：Phase 3 时 44.7% 一致率/65.5% 覆盖率；Phase 3.1 已修复为 38/38、100% 覆盖，
   但保留两种消息格式（原本一致的 17 个场景仍用 M10 文本）。
4. **4 个空约束 run 仍为 failed**（完成守卫拒绝），只是现在有正确文本；Phase 3.2 已为它们增加
   专属终态原因 `no_verified_candidate`（与执行失败区分，状态仍 failed），见
   [`docs/P3_2_TERMINAL_SEMANTICS.md`](P3_2_TERMINAL_SEMANTICS.md)。
5. **延迟是本机单进程**（p50 1.5 s），不是生产 SLA。
6. **没有 LLM 判分**：所有核验是确定性代码；这避免了“同模型生成标签又判分”，但也意味着
   语义质量（相关性、说服力）没有被评估。
