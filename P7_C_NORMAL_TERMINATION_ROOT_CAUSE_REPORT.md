# P7-C normal_termination 共同失败归因报告

## 0. 结论

**主要失败点是检索 Query / 召回可达性，不是 role 过滤；Qwen 额外叠加了 Planner 在空候选 observation 上 FINALIZE 的契约错误。**

- 目录独立扫描确认：`cookware/main + price_max<=50` 有 **258** 个合格商品。
- 在所有可恢复的真实 Agent 候选池中，258 个里只有 **1 个**出现：`B085RYZ82R`。
- 这个唯一合格商品在 Rule turn0 被召回、通过 role gate，并进入 final candidates；**没有被错误过滤**。
- Rule turn1 重新检索时，Query 变成 “Those look fine if you have eligible options...” 这类不含 `cookware` 的 follow-up 文本，候选池里 **0 个 cookware 合格商品**；role gate 因“无 MATCH”清空，最终展示为空。
- Qwen Base/SFT 的 `FINALIZE is not allowed now` 发生在 `candidate_ids` 为空的 observation 上；没有证据表明当时已有可合法展示候选被 Planner 忽略。SFT turn0 曾展示 4 个候选（含唯一合格商品），但第二 turn 的候选状态未持久化，**UNVERIFIED**。

本轮只做轨迹复用和只读目录扫描，**未修改代码、未重跑 Benchmark、未启动 Qwen/GPU**。

## 1. 任务与状态

`normal_termination`：

- turn0：`Show me camping cookware under $50.`
- turn1：`Those look fine if you have eligible options; show me the best ones.`
- 目标：`goal_type=cookware`, `role=main`, `price_max=50`
- Rule 两 turn 的真实 M5 状态：`price_max=[50]`、`role_target={cookware,main}`，均正确；hard constraints 未丢失。

## 2. 目录与候选池计数

独立目录扫描（只读 `assess_candidate` + price）：

| 阶段 | 数量 |
|---|---:|
| catalogue size | 156,746 |
| role match | 647 |
| role unknown | 147,127 |
| role-match 且 price 未知 | 292 |
| **role-match 且 price<=50（合格）** | **258** |

`runs/p7_c/normal_eligible_asins.json` 保存了 258 个合格 ASIN；示例：

`B07D8CTXS7`, `B07CCPCXN8`, `B078PDB8MC`

从真实 Rule 轨迹可恢复的 Agent 候选 ID 共 **19** 个；与 258 的交集：

```text
eligible ∩ known_agent_candidates = {B085RYZ82R}
```

**257 个合格商品从未出现在可恢复的真实候选池中。**

> 说明：`candidate_filter.stage_note` 明确记录 `raw_recall_count` 是 grounded ledger identity 数，source-level raw recall 没有持久化。因此 SASRec / Two-Tower / BM25 的**单源原始 top-N 与来源级计数标记 UNVERIFIED**；这里使用的是实际落盘的 grounded/RRF/role/eligibility 计数。

## 3. Rule 真实轨迹

### turn0

Query terms：

`Show / me / camping / cookware / under / $50.`

| 阶段 | 数量/结果 |
|---|---|
| source calls | history=1, catalog_search=1；两者均 `candidates` |
| raw grounded ledger | 10 |
| RRF candidates | 8 |
| role gate | MATCH=1, MISMATCH=4, UNKNOWN=3 |
| eligibility | 15 assessed；verified eligible=8，ineligible=2，unresolved=5 |
| final candidates | 4：`B085RYZ82R`, `B091VYKC96`, `B003IUTHFI`, `B000P9H0BI` |
| eligible in final | `B085RYZ82R`（唯一合格 cookware） |

turn0 的关键事实：唯一合格商品**已经被召回并保留**；此时不是 role filter 或 final promotion 出错。

### turn1

Query terms：

`Those / look / fine / if / you / have / eligible / options;`

| 阶段 | 数量/结果 |
|---|---|
| source calls | history=1, catalog_search=1；两者均 `candidates` |
| raw grounded ledger | 10 |
| RRF candidates | 7 |
| role gate | **MATCH=0**, MISMATCH=3, UNKNOWN=4 |
| eligibility | verified eligible=7，ineligible=5，unresolved=3 |
| final candidates | **0** |
| eligible in pool | **0** |

turn1 的 7 个候选全部是 mismatch/unknown；role gate 按 `select_grounded_recommendations` 的规则：

```python
# product_role.py:390-392
selected = [*matched, *unknown] if matched else []
```

无 MATCH 时清空。**但清空之前，258 个合格 cookware 没有任何一个进入候选池；不是“召回了却被 role gate 错误排除”。**

### 第一处丢失点

- turn0：`B085RYZ82R` 已被召回并展示；不是丢失点。
- turn1：follow-up Query 丢失了 `cookware` 类型词；第一处丢失在 **检索 Query 构造**，而不是候选过滤。
- 相关源码：
  - `agent_service.py:1330`：`terms = query_terms(intent, payload.message)`，turn1 直接使用 follow-up 文本；
  - `agent_service.py:1450-1455`：`state_constraints_changed` 强制 `RETRIEVE`；
  - `agent_service.py:1527-1553`：anchor 复用只覆盖“纯撤回/无 previous + more options”，没有覆盖普通 follow-up；
  - `agent_service.py:1575`：`intent_terms=r1_run_terms` 把该 Query 送入真实检索。

因此 Rule 的失败链是：

```text
follow-up 不含 cookware
→ 重新 RETRIEVE，Query 使用 follow-up 文本
→ 召回池没有 cookware
→ role gate MATCH=0
→ final candidates=0
```

不是 Planner 决策（Rule 无 Planner）。

## 4. Qwen Base / SFT-v0

### 4.1 可复用证据

- SFT turn0：Action `RETRIEVE, RERANK, FINALIZE`；最终 ID：
  `B085RYZ82R, B091VYKC96, B003IUTHFI, B000P9H0BI`；
  与 Rule turn0 相同，含唯一合格商品 `B085RYZ82R`。
  SFT turn0 的 `retrieval_feedback`：raw=10，RRF=8，role MATCH=1，MISMATCH=4，UNKNOWN=3。
- Base normal：没有 turn record；`tools=0`；失败于执行早期，具体候选阶段无法从轨迹恢复。
- SFT normal：turn0 成功；第二 turn 在 Planner 执行阶段失败，没有 turn record；因此第二 turn 的候选池和 Query **UNVERIFIED**。

### 4.2 提前 FINALIZE 的候选状态

真实错误为：

```text
PlannerContractError: FINALIZE is not allowed now
```

`run_planner` 在 `planner_contract.py:171-178` 先计算 `allowed_actions`；当 `candidate_ids` 为空时，allowed actions 只有 ASK / RETRIEVE，FINALIZE 直接拒绝。因此：

- **在该次 FINALIZE 的 observation 中，候选集合为空**；没有可合法展示的候选被忽略。
- 但不能证明“如果不 FINALIZE、改 RETRIEVE，就一定能召回合格商品”；第二 turn 没有 RETRIEVE 轨迹，具体原因标记 **UNVERIFIED**。
- SFT turn0 虽然有 4 个展示候选（含 `B085RYZ82R`），但第二 turn 是新的 Planner observation，不能把上一 turn 的候选当作当前 FINALIZE 的合法候选。
- Base 没有 turn record，精确 stage 同样 **UNVERIFIED**。

结论：Qwen 的 FINALIZE 是 **Planner 决策/契约错误**（在空候选观察上选择 FINALIZE，而不是 RETRIEVE/ASK），但底层候选为空的原因与 Rule 一样受**检索可达性**限制；不能把这次失败说成“已有合格候选但 Planner 没展示”。

## 5. 回答 P7-C 的五个问题

### 1) 主要失败点在召回、过滤还是 Planner？

**首先是召回 Query 可达性**：

- 257/258 合格商品从未进入可恢复的真实候选池；
- 唯一被召回的 `B085RYZ82R` 在 turn0 通过 role/final；
- turn1 follow-up Query 丢失 `cookware` 后，候选池 0 个合资格 cookware。

role 过滤是**结果为空的下游原因**：当候选池无 MATCH 时，`select_grounded_recommendations` 正确地不展示 mismatch/unknown，而不是错误地排除了合格商品。  
Planner 是 Qwen 的**额外失败层**：空候选时提前 FINALIZE，而不是 RETRIEVE/ASK。

### 2) 是否存在已经被召回却错误排除的合格商品？

**没有。**

- 258 中唯一进入候选池的是 `B085RYZ82R`；
- 它在 Rule turn0 是 MATCH，且进入 final IDs；
- 未发现它在 role gate 或 final promotion 被错误排除；
- turn1 没有合格商品进入候选池，因此不存在“召回后被过滤”的合格商品。

### 3) Rule、Base、SFT-v0 是否受同一底层限制？

**是，至少 Rule 与 SFT 的 turn0 证据一致：**

- 两者都只召回了唯一合格商品 `B085RYZ82R`；
- 两者都出现同一个低召回候选池（raw=10 / RRF=8 / MATCH=1）；
- SFT 第二 turn 与 Base 的 FINALIZE 错误都是在“当前候选为空”的 observation 上发生的，说明 Qwen 也受**同一 Agent 检索/候选链**限制，只是多了一层 Planner 错误。

Base 的 n=1 轨迹缺失，精确的候选池只能标记 UNVERIFIED；但不能假设 Base 有独立于该工具链的召回。

### 4) 需要修复现有工具，还是靠更好的 Planner 决策？

**两者都需要，但优先级不同。**

- **主要应修工具/查询链**：普通 follow-up 不应丢掉显式产品类型；当有效目标未变、只是用户说“show me the best ones”时，应复用/锚定 `cookware` 查询，或至少避免无类型词的重新检索。
- **Planner 改进是第二优先**：Qwen 应学会“候选为空先 RETRIEVE/ASK，不能 FINALIZE”；这能避免契约失败，但不能让 257 个未召回的合格商品凭空出现。
- 不建议通过放宽 role gate 来“修复”展示：没有 MATCH 时展示 UNKNOWN 会违反现有 role 证据契约，也会把工具限制伪装成成功。

### 5) 下一轮优先优化什么？

1. **Query anchor / follow-up retrieval**：在 goal/constraints 未改变时保留 `cookware` 类型锚点，避免用 “Those look fine...” 直接覆盖检索 Query；
2. **召回覆盖**：评估 258 个合格 cookware 中为什么只有 1 个进入 top candidate pool；扩大/校准 lexical retrieval 的类别覆盖（不改 Serving V1）；
3. **Planner 契约保护**：空候选时禁止 FINALIZE，强制 RETRIEVE 或 ASK；
4. 再讨论 SFT-v1：FINALIZE-on-empty 轨迹可作为 Planner 契约失败案例，但 257/258 未召回的根因是工具/查询，不应作为 Planner SFT 正/负标签直接使用。

## 6. 交付物与状态

- 本报告：`P7_C_NORMAL_TERMINATION_ROOT_CAUSE_REPORT.md`
- 258 个合格 ASIN：`runs/p7_c/normal_eligible_asins.json`
- 阶段计数与候选池证据：`runs/p7_c/evidence.json`
- 复用轨迹：`runs/p6_4_dynamic/dev_20261107_160922/`（Rule）、
  `runs/p6_4_dynamic/base_20261107_183429/`、`runs/p6_4_dynamic/sft_20261107_182912/`

本轮未修改 Serving V1 / Gold / Evaluator，未训练，未扩充任务，未重跑完整 P7，未 commit/push/deploy，未操作 GPU 实例。  
仍需明确的残余 UNVERIFIED：单源 SASRec/Two-Tower/BM25 原始 top-N 与来源级计数；Qwen 第二 turn 的精确候选池/提前 FINALIZE 前是否执行过 RETRIEVE。
