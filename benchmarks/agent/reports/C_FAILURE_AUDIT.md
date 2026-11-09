# C 组 20 个开发任务失败归因与可信度审计

## 审计边界

- 数据集：`benchmarks/agent/tasks/dev.json`，20 个任务；
- 原始运行：`runs/agent_benchmark/dev_c_final2/`（保留 2/20 原结果）；
- 审计轨迹：`runs/agent_benchmark/dev_c_audit/`，只额外记录每轮 grounded candidate pool，不改变策略、M5/R1 或任务；
- 冻结任务、冻结哈希、Gold、Serving V1：本阶段均未修改；
- 本报告不是用户满意度或推荐质量结论。

## 总览

- 任务数：**20**
- 执行状态：**{"finished": 19, "waiting_for_user": 1}**
- 原始 deterministic pass：**2/20**（dev_03、dev_19）
- A/B/D：当前环境无 LLM provider，未执行；未生成模拟结果；
- 人工相关性标签：0。

根因计数（每任务只计一个最早根因）：

| 根因 | 数量 |
|---|---:|
| `eligibility_or_ranking_error` | 7 |
| `memory_state_error` | 4 |
| `evaluator_or_gold_issue` | 3 |
| `none` | 2 |
| `retrieval_candidate_miss` | 2 |
| `clarification_state_error` | 1 |
| `planner_decision_error` | 1 |

## 逐任务诊断：执行与状态

| task_id | 任务类型 | 轮数 | 完成状态 | 第一处失败断言 | Gold 有效需求 | 实际 M5 effective preferences |
| --- | --- | --- | --- | --- | --- | --- |
| dev_01_goal_correction_withdraw | goal_correction_withdraw | 3 | finished | none of the expected product identities were presented | complete water filter; replacement mouthpiece has been withdrawn | feature |
| dev_02_accessory_then_restore | accessory_then_restore | 3 | finished | a forbidden product role appeared in the final list | complete water filter; no cartridge/mouthpiece/bottle | feature |
| dev_03_accessory_target | accessory_target | 3 | finished | （通过） | replacement mouthpiece for a water filter | feature |
| dev_04_goal_change_to_filter | goal_change_to_filter | 3 | finished | a forbidden product role appeared in the final list | water filter; cookware request withdrawn | category\|feature |
| dev_05_budget_conflict | budget_conflict | 4 | finished | none of the expected product identities were presented | water filter under $50; no sleeping pad/lantern | feature\|price_max |
| dev_06_cookware_material | cookware_material | 3 | finished | none of the expected product identities were presented | camping cookware with material=titanium | category\|feature |
| dev_07_cookware_lightweight_pref | cookware_lightweight_pref | 3 | finished | none of the expected product identities were presented | lightweight camping cookware; no water filter/stove | category\|feature |
| dev_08_filter_lightweight_withdraw | filter_lightweight_withdraw | 4 | waiting_for_user | conversation did not finish (last status='waiting_for_user') | water filter; weight preference withdrawn | category |
| dev_09_ambiguous_to_cookware | ambiguous_to_cookware | 3 | finished | none of the expected product identities were presented | camping cookware after ambiguity | category |
| dev_10_cross_turn_reference | cross_turn_reference | 3 | finished | none of the expected product identities were presented | camping cookware; second-option reference | （空） |
| dev_11_filter_bottle_role | filter_bottle_role | 3 | finished | none of the expected product identities were presented | complete water filter; no bottle | feature |
| dev_12_goal_change_sleeping_to_lantern | goal_change_sleeping_to_lantern | 3 | finished | none of the expected product identities were presented | camping lantern; no sleeping pad | category |
| dev_13_goal_change_tablets_to_filter | goal_change_tablets_to_filter | 3 | finished | none of the expected product identities were presented | water filter; no purification tablets | feature |
| dev_14_gravity_filter_pref | gravity_filter_pref | 3 | finished | expected active preferences missing: ['weight'] | gravity water filter + lighter weight | feature |
| dev_15_filter_straw_pref | filter_straw_pref | 3 | finished | none of the expected product identities were presented | water filter straw + lighter weight | category\|weight |
| dev_16_cookware_to_stove | cookware_to_stove | 3 | finished | none of the expected product identities were presented | camping stove; no cookware | category |
| dev_17_filter_budget_pref | filter_budget_pref | 3 | finished | none of the expected product identities were presented | water filter under $50 + cheaper; no bottle | feature\|price_max |
| dev_18_filter_color_correction | filter_color_correction | 3 | finished | none of the expected product identities were presented | water filter; any color; no bottle | feature |
| dev_19_cookware_set_reference | cookware_set_reference | 4 | finished | （通过） | camping cookware set; no water filter | category |
| dev_20_broad_gear_to_filter | broad_gear_to_filter | 4 | finished | none of the expected product identities were presented | camping gear -> water filter + lighter weight | category\|feature |

## 逐任务诊断：M5 / R1 / 商品角色

| task_id | M5 关键事件 | 撤回后 effective | R1 预期 | R1 实际 | 候选池有角色 | Gold ID 在池 | Top-K 有角色 | Gold ID 在 Top-K | 最早根因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev_01_goal_correction_withdraw | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced -> user_constraint_stated -> preference_retracted | retraction event present; active projection still contains feature | RETRIEVE (goal correction/retraction) | RETRIEVE | True | True | False | False | memory_state_error |
| dev_02_accessory_then_restore | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE then RERANK | RETRIEVE\|RERANK | True | True | True | True | eligibility_or_ranking_error |
| dev_03_accessory_target | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE/RERANK | RETRIEVE\|RERANK | True | True | True | True | none |
| dev_04_goal_change_to_filter | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE (goal change) | RETRIEVE\|RERANK | True | True | True | True | eligibility_or_ranking_error |
| dev_05_budget_conflict | task_context_updated -> user_preference_stated -> user_constraint_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after budget update | RETRIEVE\|RERANK | True | True | False | False | eligibility_or_ranking_error |
| dev_06_cookware_material | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after material preference | RETRIEVE\|RERANK | True | False | True | False | memory_state_error |
| dev_07_cookware_lightweight_pref | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after weight preference | RETRIEVE\|RERANK | True | True | True | False | memory_state_error |
| dev_08_filter_lightweight_withdraw | task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced | no preference_retracted event; active category remains; conversation waiting | RERANK after withdrawal | RERANK | True | True | False | False | clarification_state_error |
| dev_09_ambiguous_to_cookware | task_context_updated -> product_comparison_feedback -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE for changed goal | RERANK | True | False | False | False | planner_decision_error |
| dev_10_cross_turn_reference | task_context_updated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after reference | RETRIEVE\|RERANK | True | True | True | False | evaluator_or_gold_issue |
| dev_11_filter_bottle_role | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE after role correction | RETRIEVE\|RERANK | True | True | False | False | eligibility_or_ranking_error |
| dev_12_goal_change_sleeping_to_lantern | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE for lantern goal | RETRIEVE | False | False | False | False | retrieval_candidate_miss |
| dev_13_goal_change_tablets_to_filter | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE for filter goal | RETRIEVE | False | False | False | False | retrieval_candidate_miss |
| dev_14_gravity_filter_pref | task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after weight preference | RERANK | True | True | True | True | memory_state_error |
| dev_15_filter_straw_pref | task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after weight preference | RERANK | True | True | True | False | evaluator_or_gold_issue |
| dev_16_cookware_to_stove | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE for stove goal | RETRIEVE | True | False | False | False | eligibility_or_ranking_error |
| dev_17_filter_budget_pref | task_context_updated -> user_preference_stated -> user_constraint_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after price preference | RERANK | True | True | True | False | eligibility_or_ranking_error |
| dev_18_filter_color_correction | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE/RERANK after color correction | RETRIEVE\|RERANK | True | False | True | False | eligibility_or_ranking_error |
| dev_19_cookware_set_reference | task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RERANK after reference | RETRIEVE\|RERANK | True | True | True | True | none |
| dev_20_broad_gear_to_filter | task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced | n/a | RETRIEVE for filter goal | RERANK | True | False | True | False | evaluator_or_gold_issue |

## 逐任务证据与二次问题

### dev_01_goal_correction_withdraw

- 类型：`goal_correction_withdraw`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：complete water filter; replacement mouthpiece has been withdrawn
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced -> user_constraint_stated -> preference_retracted`
- 撤回：`replacement mouthpiece for a water filter`；撤回后：`retraction event present; active projection still contains feature`
- R1：预期 `RETRIEVE (goal correction/retraction)`；实际 `RETRIEVE`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B08QYL2SD1`
- 根因：`memory_state_error`；二次类别：eligibility_or_ranking_error；Gold 状态：valid。
- 证据：preference_retracted(value=replacement mouthpiece...) is present, but the effective active projection still lists feature; final B08QYL2SD1 is a filtered bottle, not a trusted complete filter.

### dev_02_accessory_then_restore

- 类型：`accessory_then_restore`；状态：`finished`；第一失败：`a forbidden product role appeared in the final list`。
- Gold：complete water filter; no cartridge/mouthpiece/bottle
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE then RERANK`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`True`。
- 最终：`B001HBUW1I|B0BG9ZDBQ8|B08QYL2SD1|B0B42LS5L3|B09CJN4XGM`
- 根因：`eligibility_or_ranking_error`；二次类别：retrieval_candidate_miss (unrelated identities remain in the presented pool)；Gold 状态：valid。
- 证据：final contains B09CJN4XGM and B0BG9ZDBQ8 filters, but also B001HBUW1I gas-tank vent, B08QYL2SD1 filtered bottle and B0B42LS5L3 fishing line.

### dev_03_accessory_target

- 类型：`accessory_target`；状态：`finished`；第一失败：`none`。
- Gold：replacement mouthpiece for a water filter
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE/RERANK`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`True`。
- 最终：`B00NE7NZCG`
- 根因：`none`；二次类别：none；Gold 状态：valid。
- 证据：final is B00NE7NZCG; deterministic pass.

### dev_04_goal_change_to_filter

- 类型：`goal_change_to_filter`；状态：`finished`；第一失败：`a forbidden product role appeared in the final list`。
- Gold：water filter; cookware request withdrawn
- 实际 active：`category|feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE (goal change)`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`True`。
- 最终：`B000PGT9T2|B07GXT1T2L|B08QYL2SD1|B09Y53NYDK|B00QFXGSIY`
- 根因：`eligibility_or_ranking_error`；二次类别：memory_state_error (category/feature projection)；Gold 状态：valid。
- 证据：final includes role-correct filters B09Y53NYDK and B00QFXGSIY but retains forbidden cookware B000PGT9T2 and B07GXT1T2L, plus B08QYL2SD1 filtered bottle.

### dev_05_budget_conflict

- 类型：`budget_conflict`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：water filter under $50; no sleeping pad/lantern
- 实际 active：`feature|price_max`；事件：`task_context_updated -> user_preference_stated -> user_constraint_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after budget update`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B084J5D7Z4`
- 根因：`eligibility_or_ranking_error`；二次类别：planner_decision_error (RETRIEVE also observed)；Gold 状态：valid。
- 证据：active price_max is present and candidate pool contains water-filter roles; final is B084J5D7Z4 freeze-dried food.

### dev_06_cookware_material

- 类型：`cookware_material`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping cookware with material=titanium
- 实际 active：`category|feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after material preference`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`False`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B07CZ9Y7CR|B086948LMN|B07JFWYH1P|B08NTPL2XK|B018XJXYHA`
- 根因：`memory_state_error`；二次类别：evaluator_or_gold_issue (expected ASIN list narrower than role)；Gold 状态：role correct, identity set too narrow。
- 证据：material is absent from active_attributes (actual category|feature); final is titanium cookware, role-correct but none of the expected ASINs.

### dev_07_cookware_lightweight_pref

- 类型：`cookware_lightweight_pref`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：lightweight camping cookware; no water filter/stove
- 实际 active：`category|feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after weight preference`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B06WGLP747|B08HWP19XK|B00F6K91YA|B0BKFGDBBM|B00S8NHNHE`
- 根因：`memory_state_error`；二次类别：eligibility_or_ranking_error; gold exact list narrower than role；Gold 状态：role correct, identity set too narrow。
- 证据：active lacks weight; final has cookware but also B08HWP19XK water filter; expected IDs absent.

### dev_08_filter_lightweight_withdraw

- 类型：`filter_lightweight_withdraw`；状态：`waiting_for_user`；第一失败：`conversation did not finish (last status='waiting_for_user')`。
- Gold：water filter; weight preference withdrawn
- 实际 active：`category`；事件：`task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`no preference_retracted event; active category remains; conversation waiting`
- R1：预期 `RERANK after withdrawal`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`(empty)`
- 根因：`clarification_state_error`；二次类别：memory_state_error (no preference_retracted; active category)；Gold 状态：valid。
- 证据：last status waiting_for_user; event log has no preference_retracted for the final turn; active category remains; no final candidates.

### dev_09_ambiguous_to_cookware

- 类型：`ambiguous_to_cookware`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping cookware after ambiguity
- 实际 active：`category`；事件：`task_context_updated -> product_comparison_feedback -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE for changed goal`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`False`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B0721H8FS9|B08HWP19XK|B002ARUSDU|B0BX5QFWQN|B0067AN0VI`
- 根因：`planner_decision_error`；二次类别：eligibility_or_ranking_error；Gold 状态：valid。
- 证据：R1 only RERANK, no RETRIEVE; last pool has cookware role but no gold ID; final contains paracord strap, water filter, hat, novel.

### dev_10_cross_turn_reference

- 类型：`cross_turn_reference`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping cookware; second-option reference
- 实际 active：`(empty)`；事件：`task_context_updated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after reference`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B08HWP19XK|B08P5FVV7R|B09MPTB9TW|B0BKFGDBBM|B09MQ8J9CM`
- 根因：`evaluator_or_gold_issue`；二次类别：eligibility_or_ranking_error (filters/stoves remain)；Gold 状态：role correct, expected identity set too narrow。
- 证据：final includes cookware B08P5FVV7R/B0BKFGDBBM (role correct) but not expected ASINs; also water filter B08HWP19XK and stoves.

### dev_11_filter_bottle_role

- 类型：`filter_bottle_role`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：complete water filter; no bottle
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE after role correction`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B08QYL2SD1`
- 根因：`eligibility_or_ranking_error`；二次类别：retrieval_candidate_miss (gold ID not final)；Gold 状态：valid。
- 证据：last pool has filter role and expected identity; final is only B08QYL2SD1 filtered bottle.

### dev_12_goal_change_sleeping_to_lantern

- 类型：`goal_change_sleeping_to_lantern`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping lantern; no sleeping pad
- 实际 active：`category`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE for lantern goal`；实际 `RETRIEVE`
- 候选池：角色匹配=`False`，Gold ID=`False`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B08P2MC2G8|B08VLWYTD9|B08P2MMQ2J`
- 根因：`retrieval_candidate_miss`；二次类别：planner_decision_error (retrieval query did not bring lantern)；Gold 状态：valid。
- 证据：R1 RETRIEVE observed, but last candidate pool has no lantern and no expected ID; final is three sleeping pads.

### dev_13_goal_change_tablets_to_filter

- 类型：`goal_change_tablets_to_filter`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：water filter; no purification tablets
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE for filter goal`；实际 `RETRIEVE`
- 候选池：角色匹配=`False`，Gold ID=`False`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B09VF61RDX|B0BVQVNJNG`
- 根因：`retrieval_candidate_miss`；二次类别：planner_decision_error；Gold 状态：valid。
- 证据：R1 RETRIEVE observed, but last pool has no filter role and no expected ID; final is two Potable Aqua tablet products.

### dev_14_gravity_filter_pref

- 类型：`gravity_filter_pref`；状态：`finished`；第一失败：`expected active preferences missing: ['weight']`。
- Gold：gravity water filter + lighter weight
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after weight preference`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`True`。
- 最终：`B0B1HNTZCQ|B0BX5QFWQN|B08HWP19XK|B0B6Z7PTZ8|B07Y4XX65M`
- 根因：`memory_state_error`；二次类别：gold active expectation valid；Gold 状态：valid。
- 证据：gold filter identity present, but active_attributes lacks weight; M5 only feature.

### dev_15_filter_straw_pref

- 类型：`filter_straw_pref`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：water filter straw + lighter weight
- 实际 active：`category|weight`；事件：`task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after weight preference`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B0B7P4YS2G|B09Y53NYDK`
- 根因：`evaluator_or_gold_issue`；二次类别：identity set too narrow; forbidden bottle not present；Gold 状态：role correct, identity set too narrow。
- 证据：final B0B7P4YS2G/B09Y53NYDK are straw/filter roles and active includes weight, but neither is in expected ASIN list.

### dev_16_cookware_to_stove

- 类型：`cookware_to_stove`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping stove; no cookware
- 实际 active：`category`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE for stove goal`；实际 `RETRIEVE`
- 候选池：角色匹配=`True`，Gold ID=`False`；Top-K：角色匹配=`False`，Gold ID=`False`。
- 最终：`B004Z10XIO|B00XQF9RB4|B00SLIWVGE|B013PQZYLK|B08CVBJ6FR`
- 根因：`eligibility_or_ranking_error`；二次类别：planner/retrieval candidate quality；Gold 状态：valid。
- 证据：R1 RETRIEVE observed, but final is all cookware and no stove; last pool did not contain an expected stove identity.

### dev_17_filter_budget_pref

- 类型：`filter_budget_pref`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：water filter under $50 + cheaper; no bottle
- 实际 active：`feature|price_max`；事件：`task_context_updated -> user_preference_stated -> user_constraint_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after price preference`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B08QYL2SD1|B010MFZTSU`
- 根因：`eligibility_or_ranking_error`；二次类别：gold identity mismatch；Gold 状态：valid。
- 证据：active price_max and expected filter identity are in pool, but final is two filtered drinking bottles.

### dev_18_filter_color_correction

- 类型：`filter_color_correction`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：water filter; any color; no bottle
- 实际 active：`feature`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE/RERANK after color correction`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`False`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B08QYL2SD1|B0132J8QV0`
- 根因：`eligibility_or_ranking_error`；二次类别：retrieval candidate miss；Gold 状态：valid。
- 证据：final B08QYL2SD1 and B0132J8QV0 are filtered bottles; expected full-filter identity is absent from the last pool.

### dev_19_cookware_set_reference

- 类型：`cookware_set_reference`；状态：`finished`；第一失败：`none`。
- Gold：camping cookware set; no water filter
- 实际 active：`category`；事件：`task_context_updated -> user_preference_stated -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RERANK after reference`；实际 `RETRIEVE|RERANK`
- 候选池：角色匹配=`True`，Gold ID=`True`；Top-K：角色匹配=`True`，Gold ID=`True`。
- 最终：`B00XQF9RB4|B013PQZYLK|B00SLIWVGE`
- 根因：`none`；二次类别：none；Gold 状态：valid。
- 证据：final B00XQF9RB4/B013PQZYLK/B00SLIWVGE are cookware; deterministic pass.

### dev_20_broad_gear_to_filter

- 类型：`broad_gear_to_filter`；状态：`finished`；第一失败：`none of the expected product identities were presented`。
- Gold：camping gear -> water filter + lighter weight
- 实际 active：`category|feature`；事件：`task_context_updated -> user_preference_stated -> feature_explanation_shown -> candidate_pool_referenced -> ranking_snapshot_referenced`
- 撤回：`none`；撤回后：`n/a`
- R1：预期 `RETRIEVE for filter goal`；实际 `RERANK`
- 候选池：角色匹配=`True`，Gold ID=`False`；Top-K：角色匹配=`True`，Gold ID=`False`。
- 最终：`B08HWP19XK|B09KRPFMM7|B00B1OSU4W|B01M2B68C4|B00GGMPFQ0`
- 根因：`evaluator_or_gold_issue`；二次类别：eligibility_or_ranking_error (survival kit/tent/first-aid remain)；Gold 状态：role correct, identity set too narrow。
- 证据：final includes B08HWP19XK and B00B1OSU4W water filters (role correct), but expected identity absent; also survival kit/tent/first-aid.

## 撤回专项审计

### dev_01_goal_correction_withdraw

- Event log 有 `preference_retracted(value=replacement mouthpiece for a water filter)`。
- 但 task-scoped effective projection 仍出现 `feature`；`memory_events` 中 M5 `preference_count` 在最后轮为 0，而 memory active projection 仍含 feature，说明“历史事件已撤回”与“当前有效 memory projection”不一致。
- 该状态没有阻止后续返回瓶装滤水商品 B08QYL2SD1，说明旧/残留 feature 可能继续影响当前决策，至少不是一个干净的撤回状态。

### dev_08_filter_lightweight_withdraw

- 最后一轮 `forget the weight preference` 没有产生 `preference_retracted` 事件；只产生 `task_context_updated`。
- 最后状态为 `waiting_for_user`，没有最终列表；active projection 仍为 `category`。
- 这同时是 clarification/round-alignment 失败与撤回事件缺失；最早可定位问题是 pending clarification 没有被有效回答/清理。

其余任务没有有效撤回语义，不能把普通 `user_preference_stated` 当作撤回事件。

## 商品角色与 Gold 审计

- 商品角色正确且 Gold ID 不匹配的任务：`dev_06`、`dev_07`、`dev_10`、`dev_15`、`dev_20`。这些最终商品属于任务要求的主要角色（cookware/filter/straw），但不在约 7–9 个 expected ASIN 列表中。
- 这不自动等于推荐正确：如 `dev_07` 同时含 forbidden water filter，`dev_20` 同时含 survival kit/tent/first-aid；但审计必须把“角色匹配”与“Gold 精确身份不足”分开。
- `dev_09`、`dev_12`、`dev_13`、`dev_16` 的候选池/最终角色缺失更接近真实检索/规划问题。
- `dev_18` 最终含 bottle/filter bottle，属于 forbidden bottle 角色；角色词命中只是标题字符串，不应视为 Gold 通过。

当前判定：**Gold 列表本身可保留为开发集断言，但其中若干任务的精确身份集合不足以单独证明角色失败。** 本阶段不修改 Gold，只记录建议：后续报告应增加 `role_ok` 与 `identity_ok` 两个独立指标，或扩展开发集 expected 集合；冻结集不改。

## 结论

当前 C 组 2/20 是真实结果且已保留。失败并非单一原因（每任务只计最早根因）：

- 1 个 `clarification_state_error`（dev_08）；
- 4 个 `memory_state_error`（dev_01、dev_06、dev_07、dev_14）；
- 1 个 `planner_decision_error`（dev_09）；
- 2 个 `retrieval_candidate_miss`（dev_12、dev_13）；
- 7 个 `eligibility_or_ranking_error`（dev_02、dev_04、dev_05、dev_11、dev_16、dev_17、dev_18）；
- 3 个 `evaluator_or_gold_issue`（dev_10、dev_15、dev_20）；
- 2 个 `none`（dev_03、dev_19 通过）。

在这些问题分离开前，不能把 C 的低分解释为 Memory/Planner 无效，也不能把 A/B/D 与 C 的差异解释为策略收益。

## 按优先级排列的修复建议（最多 5 项）

1. **修复 M5 task-scoped retraction 的 effective projection（最高优先级）**
   - 模块：`recommendation/demo/state_integration.py`、`recommendation/state/shopping_adapter.py`、benchmark trace。
   - 证据：dev_01 有 `preference_retracted` 但 active projection 仍含 `feature`；dev_08 最后一轮没有 retraction event。
   - 影响：撤回任务无法证明“撤回后不再影响当前决策”；也是当前 2/20 中较容易首先定位的状态错误。
   - 回归：增加“有 retraction event 且 active projection 不含对应值”的 SQLite replay 测试，以及两轮撤回后 R1 不再使用该值。

2. **在 Benchmark 中显式记录并评估 candidate-pool 与 final-list 的角色分层**
   - 模块：`benchmarks/agent/adapters.py`、`evaluator.py`、报告。
   - 证据：dev_06/07/10/15/20 的最终角色正确但 Gold identity 不匹配；dev_02/04 候选池有正确角色但最终混入 forbidden role。
   - 影响：当前 `final_any` 精确身份会把角色正确误判为失败，也不能定位 ranking vs retrieval。
   - 回归：让报告同时输出 `role_ok`、`identity_ok`、`forbidden_role_ok`，不修改 Gold 或通过阈值。

3. **修复 pending clarification 的轮次对齐/撤回处理**
   - 模块：`recommendation/demo/agent_service.py` 的多轮 pending 分支；benchmark 用户模拟器。
   - 证据：dev_08 最后 `waiting_for_user`；benchmark adapter 在无法匹配选项时忽略 pending，导致用户新目标与问题状态错位。
   - 影响：M5/R1 轨迹可能因未回答问题而缺少事件，不能被解释为策略失败。
   - 回归：新增“pending 问题被新目标覆盖/撤回”时的 deterministic resume 测试。

4. **为 R1 增加“任务类型/目标变化”动作断言与 query 证据**
   - 模块：`recommendation/demo/retrieval_plan.py`、benchmark trace。
   - 证据：dev_09 新目标 camp cookware 只有 RERANK、无 RETRIEVE；dev_12/13/16 虽触发 RETRIEVE，但 candidate pool 没有目标角色。
   - 影响：无法区分错误动作（应 retrieve 却 rerank）与正确动作但 query/召回失败。
   - 回归：记录 R1 query terms/source/target role 并在 dev_09、dev_12、dev_13、dev_16 上做离线断言。

5. **在 C 组最终呈现前加入角色/资格过滤的可观测诊断（非策略修复）**
   - 模块：`recommendation/demo/agent_service.py` 的 presentable/role filter 与 benchmark evaluator。
   - 证据：dev_02/04/11/17/18 的 candidate pool 有正确角色/Gold ID，最终 Top-K 仍出现 gas-tank vent、bottle、cookware 等 forbidden 角色。
   - 影响：这是当前最明确的 eligibility/ranking 类问题；修复前不应将低分归因于 Memory 或 Planner 收益。
   - 回归：对受控 candidate pool 加 forbidden-role 过滤测试；只验证 eligibility，不改变默认 Serving V1。
