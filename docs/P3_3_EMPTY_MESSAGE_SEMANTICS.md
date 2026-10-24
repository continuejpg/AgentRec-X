# Phase 3.3 — 空结果解释与验证状态对齐

> 目标：空结果消息必须区分“已确认没有符合约束的商品”和“没有商品能被验证为符合约束”。
>
> 结果：**10 个空约束场景 = 6 个 confirmed_no_match + 4 个 unverified**，0 个错位；
> 4 个 `no_verified_candidate` 场景改用“无法核实”语义；终态、列表、硬约束、身份、事实、
> 最终列表一致性全部不变。

## 0. TL;DR

- **修复前**：所有空约束消息都以 `No candidate satisfies the active constraints (...)` 开头。
  6 个“全部证伪”场景这样说是正确的；4 个 `no_verified_candidate` 场景（8 个 ineligible +
  2 个 unresolved、或 10 个 unresolved）这样说就是**把未决当成了证伪**——消息自己下一行还写着
  `unresolved: 2/10`，自相矛盾。
- **修复**：`build_constraint_empty_response(view)` 按 eligibility view 分三支：
  1. `assessed == 0`：没有候选可检查 → “No candidate was available to evaluate…”；
  2. `unresolved > 0`：有未决候选 → “No candidate could be verified as satisfying…”，
     并说明未决数量、缺失事实不按违反处理、这不是“确认不存在”；
  3. `unresolved == 0`：全部证伪 → 保留原文 “No candidate satisfies…”。
- **没有改**：完成守卫、`no_verified_candidate` / `completion_refused`、terminal status、
  候选、约束、排序、最终列表。调用方 `finalize_stage` / `agent_service` 原样复用。
- **48 场景回归**：10/10 空约束解释仍正确；6 个 confirmed、4 个 unverified、0 个 other；
  非空消息/列表/事实/终态原因 **0 变化**。

---

## 1. 代码差异

`recommendation/agent/rendering.py` — `build_constraint_empty_response`（唯一生产代码改动）：

```text
修复前:
  No candidate satisfies the active constraints (<req>).
  Candidates assessed: ...; ineligible: ...; unresolved (...): ...
  Nothing is shown rather than a non-compliant or unverified product.
  This is a constraint outcome, not an exhausted interaction history.

修复后（三支）:
  assessed == 0:
    No candidate was available to evaluate against the active constraints (<req>).
    ... Nothing is shown because no candidate was retrieved to verify ...
    This is a constraint outcome, not a confirmed absence of a matching product.

  unresolved > 0:
    No candidate could be verified as satisfying the active constraints (<req>).
    ... Nothing is shown rather than an unverified product: N candidate(s) remain
    undecided, and missing facts are not treated as violations.
    This is a verification outcome, not a confirmed absence of a matching product,
    and not an exhausted interaction history.

  unresolved == 0:
    No candidate satisfies the active constraints (<req>).   # 原文不变
    ... Nothing is shown rather than a non-compliant or unverified product.
    This is a constraint outcome, not an exhausted interaction history.
```

- 只读现有 `requirements` / `assessed_identities` / `verified_eligible_count` /
  `ineligible_count` / `unresolved_count`；
- 不重新评估约束，不推断缺失字段为违反，不展示任何商品；
- `pipeline.finalize_stage`（completed 空结果）与 `agent_service`（refused 空结果兜底）
  的调用点不变。

---

## 2. 消息前后对比（实际运行）

### 2.1 未决候选（4 个 `no_verified_candidate` 场景）

`h12000`，`price_max=0.01`（8 个证伪 + 2 个未决）：

```text
修复前:
No candidate satisfies the active constraints (price_max=0.01).
Candidates assessed: 10; verified eligible: 0; ineligible: 8; unresolved (catalogue cannot decide): 2.
Nothing is shown rather than a non-compliant or unverified product.
This is a constraint outcome, not an exhausted interaction history.

修复后:
No candidate could be verified as satisfying the active constraints (price_max=0.01).

Candidates assessed: 10; verified eligible: 0; ineligible: 8; unresolved (catalogue cannot decide): 2.
Nothing is shown rather than an unverified product: 2 candidate(s) remain undecided, and missing facts are not treated as violations.
This is a verification outcome, not a confirmed absence of a matching product, and not an exhausted interaction history.
```

`h12002`，`free_form_constraint=extra-large`（10 个未决）：

```text
No candidate could be verified as satisfying the active constraints (unverifiable=extra-large).

Candidates assessed: 10; verified eligible: 0; ineligible: 0; unresolved (catalogue cannot decide): 10.
Nothing is shown rather than an unverified product: 10 candidate(s) remain undecided, and missing facts are not treated as violations.
This is a verification outcome, not a confirmed absence of a matching product, and not an exhausted interaction history.
```

终态仍是 `failed / no_verified_candidate / succeeded=false`，列表为空——本阶段没碰终态语义。

### 2.2 已确认无匹配（6 个场景，文案不变）

```text
No candidate satisfies the active constraints (brand=zzzz-no-such-brand).

Candidates assessed: 10; verified eligible: 0; ineligible: 10; unresolved (catalogue cannot decide): 0.
Nothing is shown rather than a non-compliant or unverified product.
This is a constraint outcome, not an exhausted interaction history.
```

### 2.3 空候选列表（新增分支）

```text
No candidate was available to evaluate against the active constraints (price_max=20).

Candidates assessed: 0; verified eligible: 0; ineligible: 0; unresolved (catalogue cannot decide): 0.
Nothing is shown because no candidate was retrieved to verify against the constraints.
This is a constraint outcome, not a confirmed absence of a matching product.
```

该分支在 48 个场景中未被触发（没有“约束生效但零候选”的场景），只由单测覆盖。

---

## 3. 定向测试

`tests/test_p3_constraint_empty_message.py` 已更新为 6 项：

| 测试 | 断言 |
|---|---|
| `test_confirmed_no_match_states_that_no_candidate_satisfies` | 全证伪 → “No candidate satisfies…”，计数、原文结尾 |
| `test_unresolved_candidates_are_not_reported_as_confirmed_absence` | 未决 → “No candidate could be verified…”，且 **不含** “No candidate satisfies” |
| `test_an_empty_candidate_list_says_nothing_was_available_to_check` | 空列表 → “No candidate was available…”，不含 “No candidate satisfies” |
| `test_finalize_renders_the_confirmed_reason` | `finalize_stage` 走 confirmed 分支 |
| `test_finalize_keeps_the_unverified_semantics_when_candidates_are_undecided` | `finalize_stage` 走 unverified 分支 |
| `test_finalize_keeps_the_exhaustion_text_when_no_constraint_is_active` | 无约束时仍是原文 |

```
pytest -q tests/test_p3_constraint_empty_message.py \
         tests/test_p3_final_list_consistency.py \
         tests/test_phase32_terminal_semantics.py \
         tests/test_demo_agent_api.py
# 57 passed
```

---

## 4. 48 场景回归（v32 → v33）

```bash
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_v33
```

| 指标 | v32 | v33 |
|---|---|---|
| 空约束解释正确（含约束 token） | 10/10 | **10/10** |
| `empty_reason_kind` = confirmed_no_match | —（无此字段） | **6** |
| `empty_reason_kind` = unverified | — | **4** |
| `empty_reason_kind` = no_candidate_available | — | 0 |
| `empty_reason_kind` = other | — | **0** |
| 硬约束违反 | 0/59 | **0/59** |
| 未决事实被展示 | 0/59 | **0/59** |
| 商品身份合法率 | 100% | **100%** |
| 消息与最终列表一致 | 38/38 | **38/38** |
| 事实可追溯 | 354/354 | **354/354** |
| 期望命中 | 40/40 | **40/40** |
| 软偏好 Top-5 | 17.5% | **17.5%** |
| 终态原因 | 44 completed + 4 no_verified | **44 + 4（不变）** |

逐场景比较：48 行中 presented / terminal reason / fact claims **0 差异**；
只有 4 条 unverified 消息文本变化，6 条 confirmed 消息按代码分支保持原文。

---

## 5. 剩余问题

1. **`free_form_constraint` 在消息里显示为 `unverifiable=extra-large`**：这是 reasoner 的内部
   维度名，不是用户词；可读性可后续改进（不在本阶段）。
2. **未决场景仍是 `failed / no_verified_candidate`**：这是 3.2 的刻意语义（不能核实 ≠ 成功），
   本阶段未改。
3. **空候选列表分支未被 48 场景覆盖**：仅单测；若线上出现“约束生效但零候选”，需要真实回归。
4. **没有“近似匹配”提示**：空结果不展示任何被排除商品，符合“不展示未验证/不合规商品”的约束，
   但也意味着用户看不到差一点的条件；这是产品取舍，不是 bug。
5. 仅 48 个构造场景，不是线上效果；未跑全量测试，未提交 Git。

---

## 6. 产物与复现

| 文件 | SHA-256 |
|---|---|
| `runs/p3_scenario_eval_v33/config.json` | `b5474453eadfc821533786b87c511813cab83ebf6d61595e50240fee42cf6387` |
| `runs/p3_scenario_eval_v33/cases.json` | `b3bf3fae8ef7c3fb93f3e0f6dfb0c9636f18c564908dfc5283bb03a593338fb5` |
| `runs/p3_scenario_eval_v33/samples.jsonl` | `f2c474d3865228f0c09e72c1755a334e82bf878e738b00009b8e360a60655016` |
| `runs/p3_scenario_eval_v33/summary.json` | `5e4079629dab04443ad62a7fcabf04391e25a5202919a2c1d89287dde1cd8b65` |
| `experiments/p3_scenario_eval.py` | `157742e4d4593d4ea7e43e8537cf6a6f365f02b5d1cb59b6854c8f400f24dcd9` |
| `tests/test_p3_constraint_empty_message.py` | `6e43e5f56750016eea34ebcbc5b63b9b0e2dec42b84ca239d0522d3fd3f0da2f` |

对照：`runs/p3_scenario_eval_v32/`（3.2 基线）。驱动新增 `empty_reason_kind` 分类字段，
其余逻辑未变。

```bash
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_v33
.venv/bin/python -m pytest -q tests/test_p3_constraint_empty_message.py \
    tests/test_p3_final_list_consistency.py tests/test_phase32_terminal_semantics.py \
    tests/test_demo_agent_api.py
```
