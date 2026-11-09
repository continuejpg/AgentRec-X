# P1B-2：有效需求更新与 R1 目标变化修复

## 0. 结果与边界

本阶段只修复：

1. 当前有效购物需求 / price constraint lifecycle；
2. R1 对目标变化的决策与旧条件污染；
3. 无 MATCH 候选时不再用 UNKNOWN 填充 Top-K；
4. 诊断与轨迹记录。

未修改：

- frozen tasks / Gold / manifest hash / frozen protocol；
- Evaluator 的原有成功判定；
- Serving V1 默认路径；
- SASRec / Two-Tower / BM25 / BGE / ItemCF；
- LLM API；
- Soft Memory → M6.1。

版本化运行：

```text
runs/agent_benchmark/dev_c_p1b2_shadow_v2/
runs/agent_benchmark/dev_c_p1b2_applied_v2/
```

结果：

| 指标 | P1B-0 | P1B-1 | P1B-2 |
|---|---:|---:|---:|
| C_shadow | 4/20 | 7/20 | **8/20** |
| C_applied | 4/20 | 7/20 | **8/20** |
| 原 P1B-1 通过回归 | — | — | 无 |

新增通过：`dev_05_budget_conflict`。  
P1B-1 的 7 个通过任务全部保持通过。

---

## 1. dev_05：旧价格和旧 source_text 在哪里产生

原来的逐轮链路：

```text
turn 0: 30 -> active price_max=30
turn 1: 10 -> active price_max=10
turn 2: 用户说 under 50
        derive_intent 仍从 snapshot 中带出旧 10/30
        intent.hard 取最严格价格 -> 仍是 10
        M5 user_constraint_stated 的 source_text 仍来自旧 under-10 语句
        turn 3 “Show more options” 又回到旧 10
```

根因不在单个 M5 event 写入，而在两层：

1. **legacy intent projection**：`derive_intent` 以 snapshot 的 active entries 为起点，只追加当前价格，没有对同 kind 的旧 price_max 做 effective replacement；
2. **M5 state**：`user_constraint_stated` 是 upsert by exact value，30、10、50 可以同时 ACTIVE；旧 value 仍进入 `authorized_hard_constraints`。

因此：

- 旧 source_text 被继续写入 event；
- R1 query 仍带 `under 30 dollars` / `under 10 dollars` 旧短语；
- 最终候选被旧硬约束限制，water filter 被挤掉，返回 food/UNKNOWN。

### 修复

#### A. `derive_intent` 的 current-turn 价格替换

`recommendation/demo/shopping_intent.py`：

- 当本轮 extraction 或显式约束包含 `price_max`/`price_min` 时，先移除 snapshot 中同 kind 的 active price entry，再应用本轮 entry；
- 新 entry 的 `source_text` 来自**当前 utterance**；
- 其他 kind 不受影响。

#### B. task-scoped latest price projection

`recommendation/demo/agent_service.py`：

- `_task_price_constraints` 保存当前 task 最新有效 price bound；
- 后续没有价格表达的回合（如 “Show more options”）继续使用该最新值；
- 不会因为旧 snapshot 重新拾回旧 10/30。

#### C. M5 effective lifecycle

`recommendation/state/belief_updater.py`：

- 同一 scope、同一 `price_max`/`price_min` 的新显式约束会把旧 ACTIVE 值标为 `SUPERSEDED`；
- 事件历史不删除，仍可审计；
- `authorized_hard_constraints` 只投影最新 ACTIVE 值；
- 撤回当前预算后不会自动恢复旧 10/30；
- 不同 scope（task vs long_term）不互相覆盖。

#### D. Query 清洁

`_sanitize_price_query_terms` 移除 R1 query 中的旧价格短语：

```text
"water filter under 30 dollars" -> "water filter"
```

修复后的 dev_05：

```text
turn 0: price_max=30, RETRIEVE
turn 1: price_max=10, RETRIEVE
turn 2: price_max=50, RETRIEVE
        query = water filter | Actually | make | the | budget | under | 50 | dollars.
turn 3: price_max=50, RERANK
        query = water filter | Show | more | options.
final: 5 个完整净水器候选，含 MATCH 商品
```

**dev_05 已 pass。**

---

## 2. M5 是否正确表达最新价格条件

修复后：

- Shadow：R1 `r1_constraints` 在 dev_05 最后一轮为 `price_max=50`；
- Applied：M5 authorized / R1 也是 `price_max=50`；
- 旧 30/10 只作为历史 status=`superseded` 保留；
- 当前 `user_constraint_stated` 的 `source_text` 为 `Actually make the budget under 50 dollars.`；
- 撤回当前 50 后 active price 为空，旧 10/30 不恢复。

对应测试：

- `test_same_scope_price_corrections_keep_only_the_latest_hard_value`
- `test_price_correction_supersedes_old_budget_and_retraction_does_not_restore_it`
- `test_replayed_budget_corrections_project_only_the_latest_effective_value`
- 修正后的 `test_apply_enforces_seeded_constraint_and_shows_true_origin`

---

## 3. dev_09：原因为何没有 RETRIEVE

实际轨迹：

```text
turn 0: “I do not know which camping gear I need.” -> waiting_for_user
turn 1: “I mainly need camping cookware.” -> waiting_for_user
        role target = cookware/main
        role match = 10, mismatch = 2, unknown = 3
turn 2: “No additional preference.” -> finished
        role target 仍为 cookware/main
        final = 5 个 cookware
        retrieval_actions = []
```

因此：

- 原先**没有**发生实际 R1 RERANK；
- `actions=[]` 是因为最后一轮走的是 M7 preference-answer 路径，用户的 “No additional preference” 直接结束澄清，没有新的 R1 retrieval/rerank；
- 这不是“R1 偷选了 RERANK”，也不是漏打印一个已执行动作；
- 最终候选由澄清回答路径的 base response + grounded role filter 产生；
- evaluator 的 `expected action any=[RETRIEVE, RERANK]` 与这条交互路径不匹配。

### 新的通用规则

R1 现在在以下情况强制 `RETRIEVE`：

- task-scoped role target 明确变化；
- 该变化由显式用户表达引起，而不是 “show more options”；
- 即使本轮 extraction 是 soft，也不会复用旧候选窗口。

对应实现：

```text
_update_role_target(...) returns changed
-> r1_plan_action = RETRIEVE
-> reason = grounded_role_target_changed
```

但 dev_09 的最后一轮没有进入这条 R1 分支，因此 P1B-2 不伪造一个未执行的 RETRIEVE，也不修改 evaluator 的 action 断言。

---

## 4. dev_17：真实动作错误还是诊断错误

dev_17 最终：

- role target = water_filter/main；
- final 5 个商品 role match；
- gold identity 正确；
- `retrieval_actions = []`。

原因与 dev_09 相同：

```text
turn 0/1 waiting_for_user
turn 2 “No additional preference” -> finished via M7 answer path
```

**没有实际执行 R1 action，也不是诊断记录漏写。**  
实际动作不存在，因此不应补造一个日志条目。当前失败是 evaluator 的 action/metadata 断言与 M7 澄清回答路径不一致，不是 R1 的隐藏决策错误。

---

## 5. R1 目标变化与 Query 污染

本轮修复后的 R1 输入/输出：

- 当前用户显式目标优先；
- task-scoped role target 跨 follow-up 保持；
- 旧商品角色撤回时回退到最近有效 main target；
- 当前有效价格使用 latest task projection；
- 旧价格短语在 query 构造前被清除；
- RERANK 只在候选仍满足当前目标、无类型/角色变化、无约束放宽需要时发生；
- 明确类型/角色变化会触发 `grounded_role_target_changed -> RETRIEVE`；
- “Show more options” 不当作商品类型变化。

仍需明确：

- 旧目标不会通过 role target 或 query anchor 进入新 query；
- 旧价格不会通过 intent price 或 R1 anchor 进入新 query；
- dev_05 已验证 query 中不再有 `under 30 dollars` 旧短语。

---

## 6. 无可信候选时不输出明显错误角色

P1B-2 将角色选择器规则收紧：

```text
MATCH exists
  -> MATCH + UNKNOWN（UNKNOWN 只作诊断/后备，不算 MATCH）
no MATCH
  -> 不展示 MISMATCH，也不把 UNKNOWN 当作满足目标
```

dev_12 在 P1B-2 中最终为空而不是错误角色填充，即是此规则的可观察结果；这是“候选不足”而不是 K 补全。

---

## 7. 20 任务结果

| task | P1B-1 | P1B-2 | 原因 |
|---|---:|---:|---|
| dev_01 | pass | pass | role target / withdrawal |
| dev_02 | fail | fail | Gold 标题 “Bottles” forbidden 误报 |
| dev_03 | pass | pass | accessory 目标保持 |
| dev_04 | pass | pass | 类型切换 |
| dev_05 | fail | **pass** | price lifecycle + query 清洁 |
| dev_06 | fail | fail | material memory/pref |
| dev_07 | fail | fail | weight memory/pref + action |
| dev_08 | pass | pass | withdrawal |
| dev_09 | fail | fail | M7 answer 路径无 R1 action |
| dev_10 | pass | pass | role target |
| dev_11 | fail | fail | Gold 标题 forbidden 误报 |
| dev_12 | fail | fail | 新目标无 MATCH 候选，返回不足 |
| dev_13 | fail | fail | retrieval/role |
| dev_14 | fail | fail | preference memory |
| dev_15 | fail | fail | retrieval |
| dev_16 | fail | fail | role-correct identity miss |
| dev_17 | fail | fail | M7 answer 路径无 R1 action |
| dev_18 | fail | fail | role-correct identity miss |
| dev_19 | pass | pass | cookware set |
| dev_20 | pass | pass | broad -> filter |

Shadow 与 Applied 都是 `8/20`，差异仅来自 M5 state mode 的常规约束投影；P1B-2 的 role/query 修复不依赖 M5 通道。

---

## 8. 受控测试

新增/更新：

1. **预算 30→10→50**：intent 只保留 50；state active 只保留 50；
2. **撤回预算**：旧 10/30 不恢复；
3. **user/task scope**：已有 scope 测试保持；
4. **角色目标变化**：`_update_role_target` 返回 changed，并支持 accessory 撤回回退；
5. **soft preference + 可比较候选**：既有 M7/R1 测试保持 RERANK；
6. **预算放宽**：dev_05 实测 `10 -> 50` 重新 RETRIEVE 并恢复 filters；
7. **无关反馈**：既有 no-action/RERANK 测试；
8. **Query 无旧约束**：`_sanitize_price_query_terms` 测试；
9. **no MATCH**：角色选择器返回空，不输出 MISMATCH/UNKNOWN-as-match；
10. **SQLite replay**：replay 后 latest price 仍是唯一 effective，旧值 status=`superseded`。

测试结果：

```text
tests/test_product_role.py
tests/test_p4_shopping_intent.py
tests/test_state_beliefs.py
tests/test_m5_1_state_adapter.py
tests/test_m5_1_acceptance.py
tests/test_m5_2_integration.py
tests/test_m5_r1_channel.py
tests/test_m7_grounded_actions.py
tests/test_demo_agent_api.py
tests/test_p4_shopping_api.py
-> 221 passed
```

未修改 Evaluator 成功判定；P1B-1 的 MATCH/MISMATCH/UNKNOWN 行为继续通过测试。

---

## 9. 直接回答

1. **旧价格和旧 source_text 在哪里产生并保留？**
   - 在 `derive_intent` 从 legacy snapshot 取 active price entries 时保留；
   - M5 `user_constraint_stated` 以 exact value upsert，旧 30/10 可同时 ACTIVE；
   - `intent.hard` 取最严格价格，导致旧 10 继续覆盖新 50；
   - 旧 source_text 随旧 intent entry 继续写入 event。

2. **修复后 M5 是否表达最新价格？**
   - 是。Shadow/Applied 的 effective price 均为最新 50；
   - 旧值保留历史但 status=`superseded`；
   - source_text 为当前 under-50 utterance。

3. **dev_09 原先为什么没有 RETRIEVE？**
   - 最后一步是 M7 “No additional preference” 回答，没有进入 R1 正常分支；
   - 没有实际 RERANK，也没有实际 RETRIEVE；
   - `actions=[]` 是真实无动作，不是漏记。

4. **Planner 是否仍可能把已撤回目标带入 Query？**
   - role target 撤回回退到 main；旧价格短语在 query 前被清洗；
   - dev_05 已显示 query 不含旧 `under 30 dollars`；
   - 没有任务专用关键词/ASIN 规则。

5. **dev_17 是真实动作错误还是诊断错误？**
   - 真实是 M7 answer 路径没有执行 R1 action；
   - 不是诊断漏记，也不是实际 RERANK 被错误读取；
   - 不补造动作；当前失败属于 evaluator action 断言与交互路径不一致。

6. **新逻辑是否引入无意义检索？**
   - 20 任务只新增 dev_05 的预算放宽 RETRIEVE；
   - 没有无限重试；
   - `Show more options` 没有无条件触发类型变化；
   - Shadow/Applied 均 8/20，无新增回归。

7. **通过率变化由哪些修复引起？**
   - 新增 dev_05：latest price projection + M5 supersede + query 清洗 + 无 MATCH 不填充；
   - P1B-1 的 role filter 仍保留 dev_01/04/10 等收益；
   - 没有修改 Gold/Evaluator 或使用 expected ASIN。

---

## 10. 交付与遗留

报告文件：

- `benchmarks/agent/reports/P1B2_EFFECTIVE_INTENT_PLANNER.md`
- `benchmarks/agent/reports/P1B2_TURN_COMPARISON.csv`

轨迹：

- `runs/agent_benchmark/dev_c_p1b2_shadow_v2/trajectories.jsonl`
- `runs/agent_benchmark/dev_c_p1b2_applied_v2/trajectories.jsonl`

遗留：

- dev_09/dev_17 的 action 断言与 M7 answer 路径不一致；属于 evaluator/task expectation，不是 P1B-2 可伪造的动作；
- dev_02/dev_11 的 Gold forbidden title keyword 误报；
- dev_16/dev_18 role-correct identity miss；
- dev_06/07/14/15 等 memory/retrieval 问题不在本阶段范围。

本阶段结束，不继续 P1B-3/P1C，不 commit/push/deploy。
