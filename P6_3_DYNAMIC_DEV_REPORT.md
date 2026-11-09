# P6.3 Dynamic Dev 真实执行与评分报告

## 1. 范围与状态

本轮只运行 `data/p6_benchmark_v0/dynamic_dev.jsonl` 的 6 个 Dev 任务；**未读取/运行 4 个 Holdout 任务**，未启动 Qwen、未训练、未修改 Serving V1 / Agent / Eligibility / 角色规则、未 commit。

实际结果：**PASS 0 / FAIL 5 / UNKNOWN 1**。  
关键结论：执行闭环已打通；当前动态任务的 turn 文本是“操作指令”而非真实用户语言，Rule Agent 不能可靠解析其中数条；同时现有 HTTP 多轮路径没有把已记录的任务角色目标应用到后续候选过滤。因此 **当前 6 个任务不足以直接作为 Qwen 对照的质量基础**。

## 2. 真实执行配置

```text
policy            deterministic（Rule Agent；未启用 Qwen/LLM provider）
engine            fusion_rrf，sources = sequential + two_tower + catalog_lexical
                  RRF K=60，source head=100
SASRec            runs/sasrec_canonical_2026/best.pt
Two-Tower         runs/twotower_public_2026_full/checkpoint.pt（Serving 默认 `_full`）
catalogue         156,746 items；data/processed/Sports_and_Outdoors_products.jsonl
state mode        apply_constraints（M5 当前有效状态参与）
k / sources       5；history + catalog_search
profile           demo-user-1；trusted history 长度 5（固定、所有任务相同）
session           每个任务独立 session / 独立 M5 + memory 命名空间
turn 输入         dynamic_dev.jsonl 的 turn message 原样发送
限制              max_steps=6，max_tool_calls=4（Agent 默认边界）
```

运行产物：`runs/p6_3_dynamic/dev_20261107_135456/`（`summary.json`、`tasks.jsonl`、`turns.jsonl`）。
Runner：`experiments/p6_benchmark/run_dynamic_dev.py`（只读执行/记录/评分，不修改任何 Agent 逻辑）。

## 3. 6 个任务的最终实际展示结果

以下均来自最终一次 `response.recommendations`（promotion 之后的最终展示列表），不是 promotion 前候选数。

| task | 结果 | 最终展示 IDs | 最终有效约束 | 角色 MATCH/MISMATCH/UNKNOWN | PreferenceUpdateConsistent | tool calls |
|---|---|---|---:|---:|---:|---:|
| budget_update | **FAIL** | B09WD13RLR, B0BX5QFWQN, B0BJH866HJ, B0BPH4BMBX, B09LW2KHPM | 无（remove budget 生效；但 update→60 未生效） | 1 / 4 / 0 | false（turn 1 预算 60 未应用） | 8 |
| preference_retract | **FAIL** | B00RU3RBNE, B0BX5QFWQN, B094ZG2YK5, B0BJH866HJ, B07HF7M273 | 无 | 3 / 2 / 0 | false（“lighter” preference 仍在 memory 中，未撤回） | 6 |
| goal_switch | **FAIL** | B078X1D4YF, B08D1PYPYS | 无 | 0 / 2 / 0 | false / evidence 缺失（目标是否切到 water bottle 无可验证状态） | 4 |
| role_constraint | **FAIL** | B07CRN6L4Q, B0BX5QFWQN, B00T75JUBY, B0BJH866HJ, B01N9SE2T9 | 无 | 1 / 4 / 0 | false / evidence 缺失（accessory role 未应用到最终过滤） | 4 |
| normal_termination | **FAIL** | B01N3OBRWC, B0BX5QFWQN, B07G116VL2, B09LW2KHPM, B016HM7QRE | price_max=50 | 0 / 3 / 2 | true（价格约束保持） | 4 |
| insufficient_candidates | **UNKNOWN** | （空列表） | price_max=1 | 0 / 0 / 0（无展示项） | true（硬约束保持） | 4 |

说明：

- `budget_update`：最终无价格约束（remove budget 生效），但“update price_max to 60”没有更新 M5；最终列表混入 sleeping pad、lantern 等非 water-filter 商品。
- `preference_retract`：memory 中仍存在 `feature = "prefer lighter options"`（active）；“retract lighter preference”未被规则抽取器识别为撤回。
- `goal_switch`：最终列表是 2 个 bottle/filter 商品，但 0 个通过独立 filtered-bottle 角色检查（第一项是 replacement filters，第二项被判定为 accessory/consumable）。
- `role_constraint`：只有 Katadyn replacement cartridge 通过 accessory 角色检查；LifeStraw 主产品、Ultra Flow cartridge、sleeping pad 等不通过。
- `normal_termination`：price_max=50 被正确保留，但最终列表混入 mouth guards、dance soles、water filter、两个 lantern；0 个 cookware MATCH。
- `insufficient_candidates`：最终列表为空，最新硬约束 price_max=1 保留；Agent 文本明确说明 “No candidate could be verified…”，但 catalogue 中存在 213 个 role-match 商品、其中 78 个价格未知，因此**无法独立证明“目录中不存在 ≤$1 的合格商品”**，按任务要求标 UNKNOWN，而不是 PASS。

## 4. P6 动态指标（现有 `dynamic_metrics` 计算）

```text
tasks                        = 6
VerifiedTaskSuccess          = 0.000000
HardConstraintViolationRate  = 0.000000   # 最终展示项没有可证实的硬约束违反
ProductRoleCorrectness       = 0.227273   # 5 / 22 个最终展示项通过角色检查
PreferenceUpdateConsistency  = 0.333333   # normal_termination、insufficient_candidates
EligibleCandidateCoverage    = 0.500000   # 3/6 任务最终展示了至少一个独立验证合格商品
ToolCallsPerTask             = 5.000000
UnknownRate                  = 0.166667   # 1/6
```

注意：`HardConstraintViolationRate=0` **不等于**约束更新正确。`budget_update` 的“update to 60”没有应用，属于 PreferenceUpdateConsistency 失败，而不是最终展示项违反价格上限。

## 5. 主要失败原因与 Evaluator/Agent 兼容性

### 5.1 Turn 语言不是 Rule Agent 可解析的自然语言

`dynamic_dev.jsonl` 的后续 turn 是 operator-style 指令：

```text
update price_max to 60
remove budget
retract lighter preference
switch goal to water bottle
request replacement accessory role
finalize if eligible candidates exist
do not loosen hard constraint
```

Rule Agent 的 rule extractor 只能可靠解析其中的自然语言/关键词模式。实测：

- turn 1 “update price_max to 60” 后，`constraint_trace.final_constraints` 仍为 `[["price_max","40"]]`；
- “retract lighter preference” 后 memory 仍有 `feature: prefer lighter options`（active）；
- “switch goal to water bottle” 没有形成可验证的目标切换状态。

因此 5 个 FAIL 中有大量失败首先来自**任务输入格式**，而不是 Agent 的排序质量。

### 5.2 多轮 HTTP 等价路径下“角色目标”没有应用到推荐

实测每轮的 `role_diagnostics` 都是 `null`，`candidate_filter.role_target` 也是 `null`。源码上：

- `run_shopping_turn` 把角色目标写入 `_role_current_targets[task_key]`（`task_key = task_id`，本路径为 `session.session_id`）；
- `_execute_plan` 的规则路径通过 `_role_target_for(payload.user_key)` 读取（本路径为 `session.user_key`，与 task_id 不同）；
- 只有 LLM proposal 分支显式把 task_key 的目标镜像到 `payload.user_key`（`agent_service.py` 约 1094 行），规则路径没有这一步。

结果是：初始 turn 有类型词时 `promote_explicit_type_matches` 还能过滤；后续 turn（如 “remove budget”“finalize…”）没有类型词，角色目标又未被读取，最终列表就会出现 sleeping pad、lantern、mouth guards、dance soles 等明显不相关商品。这是**已定位的真实集成问题**，不是任务答案或 Eligibility 规则问题；本轮未修改 Agent。

### 5.3 其他兼容性问题

- P6 dynamic task 文件没有初始 history、初始 M5 状态、期望最终约束/偏好/角色、或“目录中是否存在合格商品”的显式字段；本轮 Runner 只能用任务文本独立推导期望，并在 catalogue 上独立扫描商品证据。
- `insufficient_candidates` 无法用当前 catalogue 证明“不存在 ≤$1 的 water filter”：有 78 个角色匹配商品价格缺失，所以按规则必须 UNKNOWN；不能因 Agent 未找到就断言任务不可满足。
- 没有把 static Gold 当作真实成功：每个任务的最终状态、候选、tool 调用、M5 约束都来自真实响应/诊断；期望条件与 catalogue 证据是独立于 Agent 的静态检查，两者在报告中分开记录。

## 6. 是否足以作为 Qwen 对照的执行基础

**当前不足以作为 Qwen 对照的质量级执行基础，只能作为执行链路 smoke。**

进入 Qwen Base/SFT 对照前至少需要：

1. 把 turn 改成真实用户语言，或提供结构化 turn action（与 Planner Contract 对齐），并固定期望状态；
2. 修复多轮路径的角色目标 key/应用问题（或在对照中显式禁用角色条件并标注）；
3. 给每个任务补上初始 history/M5、期望最终约束/偏好/角色、以及“catalogue 是否存在合格商品”的可验证字段；
4. 明确 UNKNOWN 的判定规则（尤其 catalogue 价格缺失）；
5. 保持 6 个 Dev / 4 个 Holdout 的隔离；Holdout 仍未运行。

在这些条件满足前，Qwen 对照会同时受“任务输入不可解析”和“规则路径角色目标丢失”影响，无法公平归因模型能力。

## 7. 交付

- `experiments/p6_benchmark/run_dynamic_dev.py`（新增最小只读 Runner）
- `P6_3_DYNAMIC_DEV_REPORT.md`
- `P6_3_DYNAMIC_DEV_SOURCE_DIFF.txt`
- `runs/p6_3_dynamic/dev_20261107_135456/{summary.json,tasks.jsonl,turns.jsonl}`

未修改 Agent / Serving V1 / Eligibility / 角色规则 / task 数据；未运行 Holdout、Qwen、Fusion-20k、训练；未 commit/push/deploy。
