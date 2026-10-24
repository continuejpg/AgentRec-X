# Phase 3.1 — 最终推荐列表与解释一致性

> 目标：多来源推荐时，文字解释必须只描述**最终展示的商品**，且事实与商品身份一一对应。
>
> 结果：非空场景一致率 **17/38 → 38/38**；事实 **354/354** 可追溯；48 个场景的候选列表、
> 硬约束、身份合法率、软偏好与空结果全部不变。只改渲染输入与调用顺序。

## 0. TL;DR

- **根因**：最终列表在 `agent_service._execute_plan` 里由 `build_recommendations_view(...)`
  从**融合后的 presentable 集合**（含 catalog 商品、硬约束投影、可选记忆重排）构建；
  而消息在此之前由 loop 的 `finalize_stage` 渲染，输入只有 **history capability** 的
  `reranking`/`enrichment`/`tool_result`。catalog 候选在 finalize 之后才融合，所以消息
  天然只描述 history 那 5 条。
- **修复**：新增 `build_final_list_response(...)`，并在 `_execute_plan` 里加入
  `_consistent_message(...)`——**以最终列表为唯一商品来源**；消息已与最终列表逐位一致时保持
  原样（保留 M10 偏好证据文本），不一致时按最终列表 + 按 `parent_asin` 查目录事实重渲染。
- **效果**：48 场景中 21 个不一致全部修正，17 个原本一致的不变；非空一致率 **38/38**，
  事实 **354/354**（原 200/200），空约束解释仍 **10/10**。
- **非消息输出零变化**：48 行逐场景比较，presented / 硬违反 / 身份 / 重复 / 期望命中 / 软偏好
  **0 差异**。

---

## 1. A — 保护当前成果

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/agent-source-limit-ablation` / `9eabbfe` |
| 已存在未提交改动 | Phase 3 的 `pipeline.py`、`rendering.py`、`agent_service.py`；2C 的 3 个脚本 + 报告 |
| 本轮新增 | `rendering.py` / `agent_service.py` 的一致性修复；`tests/test_p3_final_list_consistency.py`；本文档 |
| 未做 | 没有覆盖/删除/提交任何现有文件；没有 push/merge；没有改冻结资产 |

`git status` 中 2B 的四个提交仍在；2C/P3/3.1 的文件保持未跟踪/未提交（按"不自动提交 Git"）。

---

## 2. B — 分歧的准确位置

调用顺序（一次 `DemoAgentService.run`）：

```
LoopController._finalize_response()
  └─ finalize_stage(state)                      # 渲染发生在这里
       ├─ state["reranking"]     ← history capability 的 M10B 输出（5 条 history 候选）
       ├─ state["enrichment"]    ← history capability 的 M8 输出
       └─ state["tool_result"]   ← history capability 的 Tool 结果
  └─ result.final_response                       # 文本只覆盖 history 候选

DemoAgentService._execute_plan()
  ├─ presentable_candidates(state)               # 融合：history + catalog_search + ...
  ├─ 硬约束投影 / 可选记忆重排
  ├─ recommendations = build_recommendations_view(..., limit=payload.k)   # 最终列表在这里确定
  └─ AgentRecommendResponse(message=result.final_response, recommendations=...)
```

**为什么消息只描述 history**：

1. `finalize_stage` 的输入是 capability 阶段的结构化产物；在 `history + catalog_search` 计划里，
   这个 capability 只负责 history 源。catalog 源由 `CandidatePlane` 执行后写入 ledger，**在
   finalize 之前没有进入 `state["reranking"]`/`enrichment`**。
2. 最终列表是 loop 结束之后，在 `_execute_plan` 里对 `presentable_candidates` 做的融合视图；
   它与 finalize 使用的 5 条 history 候选可以不同（数量、身份、顺序都可能不同）。
3. 17/38 之所以原本一致，是这些场景的融合结果恰好等于 history envelope；21/38 则不是。

Phase 3 的空约束修复只处理"最终列表为空"的分支，没有改变"非空时消息来自 history"。

---

## 3. C — 最小修复

### 3.1 新渲染器：只接受最终列表

`recommendation/agent/rendering.py`

```python
def build_final_list_response(recommendations, facts_by_asin) -> str:
    """Render exactly the final recommendation list, in its presented order."""
```

- 商品来源 = 传入的 `recommendations`（就是响应将要发布的列表），不重新推导候选；
- 顺序 = 传入顺序（修复后列表的顺序，包含记忆重排结果）；
- 事实 = `facts_by_asin[parent_asin]`，按身份查，不按位置；
- 只打印目录确实有的字段（Title / Main category / Category / Store），缺失即省略；
- 文本只含这些商品 ID，不会出现列表外的商品。

### 3.2 调用顺序：在最终列表之后

`recommendation/demo/agent_service.py`

```python
message = _consistent_message(message, recommendations, self._metadata)
```

`_consistent_message`：

- 若 `message` 里的 ASIN 序列 **逐位等于** 最终列表 → 原样保留（保留已有的 M10 偏好证据文本，
  它本来就只描述这些商品）；
- 否则 → `build_final_list_response(recommendations, _facts_for(metadata, final_ids))`；
- 空列表 → 原样返回（继续走 Phase 3 的约束解释，不会退回"历史耗尽"）。

调用点位于 `build_recommendations_view(...)` **之后**、`AgentRecommendResponse(...)` **之前**，
即融合、约束投影、可选记忆重排都已完成之后。

### 3.3 明确没有改的东西

- 候选生成、来源计划、RRF、Learned Ranker、重排、硬约束、`k` 截断、最终列表：**零改动**；
- 空约束解释（Phase 3）：保留并继续生效；
- `completion_refused` 内部状态：按范围外要求不动。

---

## 4. D — 自动化测试

新增 `tests/test_p3_final_list_consistency.py`（6 项）：

| 测试 | 覆盖点 |
|---|---|
| `test_a_mismatched_text_is_replaced_by_the_final_list` | 旧文本提到列表外商品 → 重渲染，只含最终 ID |
| `test_a_text_that_already_matches_is_kept_unchanged` | 已一致文本不重写 |
| `test_facts_follow_identity_not_position` | 两个商品互换位置，Title/Category/Store 仍各归其主 |
| `test_a_reordered_final_list_reorders_the_explanation` | 偏好重排改变顺序 → 解释顺序跟随最终列表 |
| `test_an_empty_final_list_keeps_the_existing_constraint_message` | 空约束文本不被覆盖 |
| `test_missing_catalogue_fields_are_omitted_not_invented` | 缺字段不补、不编 |

定向测试：

```bash
.venv/bin/python -m pytest -q tests/test_p3_final_list_consistency.py \
    tests/test_p3_constraint_empty_message.py tests/test_demo_agent_api.py
# 47 passed
```

48 场景回归（同一驱动、同一场景集）：

```bash
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_v31
```

---

## 5. E — 修复前后（48 场景）

| 指标 | Phase 3 基线（`fixed_v2`） | Phase 3.1（`v31`） |
|---|---|---|
| 场景 / 成功 | 48 / 48 | 48 / 48 |
| **消息与最终列表一致（非空）** | **17/38 = 44.7%** | **38/38 = 100%** |
| 消息集合一致 | 44.7% | **100%** |
| 消息对最终列表覆盖 | 65.5% | **100%** |
| 目录事实可追溯 | 200/200 | **354/354** |
| 硬约束违反 | 0/59 | **0/59** |
| 未决事实被展示 | 0/59 | **0/59** |
| 商品身份合法率 | 100% | **100%** |
| 期望结果命中 | 40/40 | **40/40** |
| 空约束解释正确 | 10/10 | **10/10** |
| 软偏好 Top-5 满足率 | 17.5% | **17.5%**（未动） |
| p50 / p95 | 1544 / 1887 ms | 1567 / 1913 ms |

**逐场景比较**：48 行中

- presented 列表：0 差异；
- 硬违反 / 身份 / 重复 / 期望命中 / 软偏好：0 差异；
- 仅 **21 条消息**变化，正好是修复前不一致的那 21 个；17 条原本一致的文本保持不变；
- 修复后仍不一致的场景：**0**。

---

## 6. 代码差异

```
 recommendation/agent/rendering.py    | 72 +++++++++++++++++++++++++++++
 recommendation/demo/agent_service.py | 85 +++++++++++++++++++++++++++++-
```

（上面的统计含 Phase 3 已存在的改动；3.1 新增的是 `build_final_list_response` 与
`_consistent_message`/`_facts_for`/`_product_ids` 及其调用点。）

新增文件：

```
tests/test_p3_final_list_consistency.py
docs/P3_1_FINAL_LIST_CONSISTENCY.md
```

没有修改：`pipeline.py`（Phase 3 的约束分支保留）、RRF、Learned Ranker、查询策略、候选深度、
冻结 evaluator / 预注册 / 历史报告 / 检查点。

---

## 7. 残留与未验证事项

1. **两种消息格式并存**：原本一致的 17 个场景保留 M10 偏好证据文本，21 个不一致场景使用新的
   “Final recommendation list”格式。两者都只描述最终列表，但版式不同；统一版式需要评估是否会
   丢失偏好证据展示，本轮不做。
2. **4/10 空约束 run 的终态原因**：Phase 3.2 已从 `completion_refused` 细化为
   `no_verified_candidate`（状态仍 `failed`，仍不展示任何未验证商品）；见
   [`docs/P3_2_TERMINAL_SEMANTICS.md`](P3_2_TERMINAL_SEMANTICS.md)。
3. **事实覆盖取决于目录**：Title/Category/Store 缺失时不打印，所以“每个商品都有完整属性”不成立；
   本阶段只保证“打印出来的属性可追溯且归属正确”。
4. **价格未进入解释**：目录价格字段存在，但本轮只打印 Title/Main category/Category/Store；
   避免把未经本阶段核验的字段当作解释事实。
5. **只有 48 个构造场景的证据**：38/38 是这批场景上的确定性结果，不代表线上用户效果。
6. **未运行全量测试**：只跑了新增测试 + Phase 3 回归 + Demo Agent API（47 passed）。

---

## 8. 复现

```bash
# 48 场景（修复后）
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_v31

# 定向测试
.venv/bin/python -m pytest -q tests/test_p3_final_list_consistency.py \
    tests/test_p3_constraint_empty_message.py tests/test_demo_agent_api.py
```

产物：`runs/p3_scenario_eval_v31/`（SHA-256 见其 `SHA256SUMS.json`）；
对照：`runs/p3_scenario_eval_fixed_v2/`（Phase 3 基线）。

| 文件 | SHA-256 |
|---|---|
| `runs/p3_scenario_eval_v31/config.json` | `fc8a85d0314df38cdc56093400df75d0ab3a86a1739a1b375c09ee1ecb55a95b` |
| `runs/p3_scenario_eval_v31/cases.json` | `b3bf3fae8ef7c3fb93f3e0f6dfb0c9636f18c564908dfc5283bb03a593338fb5` |
| `runs/p3_scenario_eval_v31/samples.jsonl` | `798cf66344139ca087ac05d736a6f5a4a8457e0bd9e14c2f13e87b87403d31f8` |
| `runs/p3_scenario_eval_v31/summary.json` | `48e1fe2416aaa06bf8c5e350a16ae7782a05213f49f26dee1bdf84676e794011` |
| `experiments/p3_scenario_eval.py` | `414c1a1ee781e5a02e235f490d095441cb28c4f3c8acc9375435a9939cf2dcb3` |
| `tests/test_p3_final_list_consistency.py` | `7afc6d5630f3cc0951cc315cf88ec721e45da51d1f90388fb951ed1e68e400b3` |
