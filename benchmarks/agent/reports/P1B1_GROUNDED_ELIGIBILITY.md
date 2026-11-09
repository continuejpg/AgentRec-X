# P1B-1：Grounded Product Eligibility 与最终候选排序修复

## 0. 边界与结果

本阶段只修改 Agent Demo 的最终商品资格层，不修改：

- SASRec / Two-Tower / BM25 / BGE / ItemCF / RRF；
- Serving V1 默认路径；
- frozen tasks、Gold、manifest hash、冻结协议；
- 商品目录、模型参数或 LLM API。

新增版本化运行：

```text
runs/agent_benchmark/dev_c_p1b1_final/
```

C 组结果：

| 指标 | P1B-0 | P1B-1 |
|---|---:|---:|
| runnable | 20 | 20 |
| completed | 20 | 20 |
| deterministic pass | 4/20 | **7/20** |
| 原通过回归 | — | 无 |

新增通过：`dev_01`、`dev_04`、`dev_10`。  
原有 `dev_03`、`dev_08`、`dev_19`、`dev_20` 保持通过。

---

## 1. 候选链路结论

真实链路：

```text
SASRec / Two-Tower / BM25
 -> CandidateLedger / RRF fusion
 -> presentable candidates
 -> promote_explicit_type_matches（原有宽松类型/角色检查）
 -> NEW: product_role grounded MATCH / MISMATCH / UNKNOWN
 -> limit k
 -> final Top-K / FINALIZE
```

关键修复：**角色资格在 `limit=k` 之前作用于整个 presentable candidate set**，不是在已经截断的 Top-K 上事后过滤。因此：

- 正确角色商品即使基础融合排名较低，也能被保留；
- MISMATCH 不会因为需要凑满 k 被补入；
- 没有 MATCH 时不会伪造一个 MATCH；
- UNKNOWN 独立计数，不自动升级为 MATCH。

当前新增文件/修改：

- `recommendation/demo/product_role.py`：通用 type/role 判断；
- `recommendation/demo/agent_trajectory.py`：`build_recommendations_view(..., role_target, role_records)` 在 limit 前过滤；
- `recommendation/demo/agent_service.py`：task-scoped type/role target、role diagnostics、最终结果约束；
- `tests/test_product_role.py`：6 个受控角色测试。

---

## 2. 七个 eligibility/ranking 任务

| task | 目标类型/角色 | 原始候选有正确角色？ | 修复后最终 | 结果 | 根因/证据 |
|---|---|---|---|---|---|
| dev_02 | water_filter / main | 是（Water Filters） | B09CJN4XGM, B00A9A2HKM, B08XJQ9HPX | 仍 fail | 3 个 main filter，identity hit=True；但 `must_exclude_title_terms=bottle` 命中 B09CJN 标题里的 “2 Collapsible Water Bottles”，属于标题关键词/Gold 误报，不是角色 MISMATCH |
| dev_04 | water_filter / main | 是 | B09Y53NYDK, B00QFXGSIY, B08ZYMXLMH | **pass** | MISMATCH cookware/part/bottle 被移除；MATCH filter 保留 |
| dev_05 | water_filter / main | 最后候选池无 filter | B084J5D7Z4（UNKNOWN food） | 仍 fail | 初始候选有 filter，但 budget conflict 的后续检索返回了 food/UNKNOWN；不是角色层丢失，而是 retrieval/planner 链路丢失 |
| dev_11 | water_filter / main | 是 | B09CJN4XGM, B00A9A2HKM, B08XJQ9HPX, B09Y53NYDK | 仍 fail | 4 个 main filter，identity hit=True；标题 “Bottles” 触发 forbidden 关键词误报 |
| dev_16 | stove / main | 是 | B0B7R6DL1P, B08T5M5FZT | 仍 fail | 2 个真实 stove，role 正确；但 expected Gold identity 是 B01985/B07F6，role-correct identity miss |
| dev_17 | water_filter / main | 是 | B0BX5QFWQN, B08ZYMXLMH, B08XJQ9HPX, B09XVPFBFZ, B0041HS4WO | 仍 fail | role/identity 正确；R1 action 断言观测为空，属于 action/evaluator 问题，不是角色过滤 |
| dev_18 | water_filter / main | 是 | B08XJQ9HPX, B008JX0QP4 | 仍 fail | 2 个真实 main filter；Gold expected identity 未命中，role-correct |

### 直接回答

1. **正确角色商品是否进入原始候选池？**
   - dev_02/04/11/16/17/18：是；
   - dev_05：初始是，但 budget conflict 后续检索最终池中没有，属于 retrieval/planner 丢失。

2. **如果进入，在哪个具体阶段丢失？**
   - dev_02/11：没有被角色层丢失；最终是 role-correct filters，失败来自 evaluator 的 forbidden title keyword；
   - dev_16/18：没有被角色层丢失；最终 role-correct，但 Gold identity 不在候选/期望列表；
   - dev_17：最终 role/identity 正确，失败在 R1 action 诊断；
   - dev_04：原先部分被角色过滤丢失，P1B-1 修复恢复；
   - dev_05：最终候选池无正确类型，发生在角色层之前。

3. **不符合需求的商品为什么通过过滤？**
   - 修复前，`promote_explicit_type_matches` 只在最终 Top-K 上做部分词/分类过滤；错误商品如 cookware、filtered bottle、fuel tank 已经占据了列表。
   - 修复后，presentable pool 在 limit 前按 type/role 过滤，MISMATCH 不再进入最终列表。
   - UNKNOWN 不会被当作 MATCH；但按设计，若无 MATCH，UNKNOWN 会保留在诊断/最终候选中，必须独立统计。

4. **是否存在强行补满 Top-K 的逻辑？**
   - 角色层不补满 K；少于 K 允许；
   - 观察到的 `dev_05` 仅有 1 个 UNKNOWN，不是由角色层补入的 MATCH；
   - 未新增任何 target/ASIN 置顶逻辑。

5. **是否存在标题关键词角色误判？**
   - 是，仍未完全解决：
     - B09CJN4XGM 是主 water filter pump，但标题含 “2 Collapsible Water Bottles”，被 evaluator 的 forbidden title term 命中；
     - 这不是角色 MISMATCH；当前角色层正确判为 main filter。
   - 这属于 **Gold/evaluator 的标题关键词粒度问题**，不通过改 Gold 来掩盖。

---

## 3. 通用角色判断

`product_role.py` 基于可信 title/category 输出：

```text
MATCH    -- 可信证据支持目标类型和角色
MISMATCH -- 有可信证据表明类型或角色冲突
UNKNOWN  -- 证据不足，不升级为 MATCH
```

支持的类型/角色包括：

- main product；
- replacement/accessory（mouthpiece、cartridge、spare part 等）；
- filtered bottle 与 complete filter 区分；
- cookware 与 stove；
- lantern、sleeping pad、water treatment 等基础类型。

不使用：

- benchmark expected ASIN；
- dev task id；
- 人工标签；
- 或 LLM 推测。

### 受控测试

`tests/test_product_role.py`：

1. complete filter + replacement mouthpiece + filtered bottle + unrelated；
2. similar filter/bottle keyword but different type；
3. correct main product below base rank；
4. all visible candidates lack a match（UNKNOWN 保留但不算 MATCH；已知 MISMATCH 被移除）；
5. accessory request accepts compatible part, rejects main/bottle part；
6. broad `camping gear` leaves all candidates untouched。

---

## 4. Price constraint 生命周期

dev_05 的事件：

```text
turn 0: price_max=30
turn 1: price_max=10
turn 2: 用户说 under 50
       但 M5 记录的 user_constraint_stated 仍是旧 value=10，
       source_text 仍是 “Can you find one under 10 dollars?”
```

结论：

- 两者是**同一 task scope 的 price_max，不是不同 scope**；
- 当前是 append-only history 保留了旧 30/10；`intent` 的 trusted constraint 仍取旧 10；
- applied R1 constraints 曾出现 `10` 与 `30` 并存；
- **P1B-1 未完成 price lifecycle 修复**。这不是避免简单删除全部旧价格；需要后续把“当前用户价格修正”投影为 effective replacement，而不是只保留历史。此问题已记录为 P1B-2 候选，不在 P1B-1 做 Memory 重构。

---

## 5. 修复后的结果与回归

| 指标 | 修复前 P1B-0 | P1B-1 |
|---|---:|---:|
| pass | 4/20 | 7/20 |
| new pass | — | dev_01, dev_04, dev_10 |
| failed | 16 | 13 |
| 原有 pass 回归 | — | 无 |
| Applied arm | 4/20 | 7/20（角色层独立于 M5 state mode） |

### 改变的任务

- **dev_01**：角色目标从“accessory after withdrawal”恢复到 main water filter；最终 filters；pass。
- **dev_04**：cookware/bottle/part 被移除，保留 filter；pass。
- **dev_10**：broad cookware role-correct 商品不再被 identity 之外的噪声影响；pass。
- **dev_02/11**：最终从混合/bottle 变为 main filters，role 正确；仍因 forbidden 标题词失败（Gold/evaluator issue）。
- **dev_16/18**：最终 role-correct，但 Gold identity 未命中；失败保留。
- **dev_17**：最终 role/identity 正确，仍因 R1 action 断言失败；不属于角色层。
- **dev_05**：最终仍无正确 filter；retrieval/planner 问题，不由角色层伪造。

### 是否错误过滤

- 受控测试中，MISMATCH 过滤是正向的（cookware vs filter、bottle vs standalone filter、main vs accessory）；
- 20 任务未见原有通过回归；
- 没有把 expected ASIN 作为执行输入；
- **dev_02/11 的 forbidden 命中是 title keyword Gold 误报**，不是角色层错误过滤；本阶段不修改 Gold。
- **price lifecycle 仍有残余过度约束**。

---

## 6. 指标

新增 [P1B1_ROLE_METRICS.csv](P1B1_ROLE_METRICS.csv)，包含：

- `retrieved_role_match_count`
- `eligible_role_match_count`
- `final_role_match_count`
- `final_role_mismatch_count`
- `final_role_unknown_count`
- `gold_identity_hit`
- `forbidden_role_hit`
- `hard_constraint_violation_count`
- `candidate_rejection_reasons`
- 最终 IDs 和 first failure

说明：

- role match 来自可信 title/category 判断，**不使用 Gold**；
- `gold_identity_hit` 仅用于报告，不参与执行过滤；
- hard_constraint_violation_count 当前为 0（没有观察到最终商品与可判定约束冲突；UNKNOWN 不计为 violation）；这只是观测结果，不是人工 relevance。

逐轮连接轨迹在：

```text
runs/agent_benchmark/dev_c_p1b1_final/trajectories.jsonl
```

---

## 7. 测试

新增：

```text
tests/test_product_role.py       6 passed
```

相关回归：

```text
tests/test_m5_r1_channel.py
tests/test_m7_grounded_actions.py
tests/test_demo_agent_api.py
tests/test_p4_shopping_api.py
tests/test_preference_construction.py
tests/test_m5_1_acceptance.py
tests/test_m5_2_acceptance.py
-> 142 passed
```

原有历史运行未覆盖：

- `dev_c_final2`；
- `dev_c_p1a_final`；
- `dev_c_p1b_*`；
- 新增 `dev_c_p1b1_final` 与 `dev_c_p1b1_applied`。

---

## 8. 遗留

1. **price_max 10/30 lifecycle 未修**：同 task scope append-only + 旧 intent 保留；需要定向 M5/task-intent 修复。
2. **Gold 标题关键词误报**：B09CJN “2 Collapsible Water Bottles” 触发 bottle forbidden；不能靠删除 Gold 或放宽断言修复。
3. **dev_05** 的正确商品在最终 retrieval 池中缺失；不是角色层。
4. **dev_16/18** role-correct identity miss；需更多 Gold 粒度/角色断言策略，不能改写 expected ASIN。
5. **dev_17** R1 action/诊断缺失，非角色层。

---

## 9. 直接回答

1. **7 个 eligibility/ranking 任务各自的根因**
   - dev_02：角色已是 main filter，最终失败是 forbidden title keyword（“Bottles”）误报；
   - dev_04：角色过滤修复，已 pass；
   - dev_05：最终候选池没有 filter，根因在角色层之前的 retrieval/planner；
   - dev_11：角色已是 main filter，最终失败是 forbidden title keyword；
   - dev_16：角色已是 stove，Gold identity 未命中；
   - dev_17：角色/identity 正确，R1 action 诊断缺失；
   - dev_18：角色已是 filter，Gold identity 未命中。

2. **候选池有正确商品但最终筛选错误的**
   - dev_02、dev_04、dev_11、dev_16、dev_17、dev_18；
   - dev_04 已修复；其余部分被 Gold/evaluator 或 role-correct identity 粒度限制，或 action 断言限制。

3. **目录证据不足**
   - dev_05 最终 B084J5D7Z4 为 UNKNOWN food；角色层没有把它算 MATCH；
   - `P1B1_ROLE_METRICS` 明确给出 UNKNOWN 数；没有把 UNKNOWN 升格为 MATCH。

4. **修复后的最终商品变化/错误过滤**
   - dev_01：B001 gas-tank vent → 4 个 main filters（pass）；
   - dev_04：混合 cookware/bottle → 3 个 filters（pass）；
   - dev_10：broad cookware → role-correct cookware（pass）；
   - dev_02/11：混合/bottle → main filters（role correct，仍因 forbidden keyword fail）；
   - dev_16/18：role-correct 商品替换了旧混排，但 Gold identity 未命中；
   - 没有原通过任务回归；没有 MISMATCH 被用来补满 K。

5. **Price Constraint 生命周期**
   - 确认 dev_05 是同一 task scope 的第二次 price_max 修正；
   - 当前 M5 event/intent 仍保留旧 10（并曾保留 30），当前用户 under-50 没有被投影为新的 effective hard constraint；
   - **未在 P1B-1 修复**；这是后续定向 M5/task-intent 生命周期工作，不是简单删除旧价格。

6. **20 个开发任务通过率**
   - 4/20 → **7/20**；
   - 新增：dev_01、dev_04、dev_10；
   - 无原通过回归；
   - Applied arm 同样 7/20，说明角色资格层独立于 M5 state mode。

7. **是否独立于 M5 实验输入通道**
   - 是。角色目标与 catalog title/category 判断独立于 M5 Shadow/Applied；P1B-1 的 shadow 与 applied 结果一致；
   - M5 的 soft/hard-state 结论不应由 P1B-1 角色提升来推断。

---

## 10. 结论

- 7 个 eligibility/ranking 任务中，角色层修复显著改变了候选：dev_04/10 以及 dev_01 变为 pass；
- dev_02/11 已从错误角色商品变为 role-correct filters，但 Gold 标题关键词仍导致 fail；
- dev_16/18 已是 role-correct，但 expected identity 未命中；
- dev_17 角色/identity 正确，失败在 R1 action 断言；
- dev_05 的正确商品在最终候选池中不存在，不能靠角色过滤修复；
- 角色过滤独立于 M5 Shadow/Applied 通道，shadow 与 applied 都是 7/20；
- **没有回归，也没有用 Gold 置顶或补足 K。**

本阶段结束，不继续 P1B-2/P1B-3，不 commit/push/deploy。
