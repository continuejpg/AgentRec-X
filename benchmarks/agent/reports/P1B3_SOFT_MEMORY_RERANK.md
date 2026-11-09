# P1B-3：M5 Soft Preference → M6.1 Grounded Reranking

## 0. 范围与结论

本阶段只接通：

```text
M5 active / explicit soft preference
        -> bounded adapter
        -> M6.1 rank_with_task_preferences
        -> unchanged candidate identity set
        -> final Top-K
```

没有：

- 修改 SASRec / Two-Tower / BM25 / BGE / ItemCF 的分数；
- 修改默认 Serving V1；
- 修改 frozen tasks、hash、协议或 Gold；
- 把软偏好升级成硬约束；
- 使用 expected ASIN 或 LLM；
- 让 inferred hypothesis 自动变成 confirmed preference。

版本化运行：

```text
runs/agent_benchmark/dev_c_p1b3_soft_off_v2/
runs/agent_benchmark/dev_c_p1b3_soft_on_v2/
```

20 个开发任务结果：

| arm | pass |
|---|---:|
| soft_off | 8/20 |
| soft_on | 8/20 |

P1B-2 的 8 个通过任务全部保持通过，没有回归。

---

## 1. 实际执行链路审计

### M5 → for_ranker

`recommendation/state/state_projector.py::for_ranker` 输出：

- `features`: ACTIVE preference beliefs；
- `kind` / `value` / `polarity` / `direction` / `scope` / `lifecycle` / `strength_rank` / `is_soft`；
- retracted / superseded 不出现在 features；
- inferred hypothesis 存在，但 `is_soft=True`；
- hard constraints 不进入 ranker projection。

### 新增的消费通道

新增：

```text
recommendation/demo/soft_preference_ranking.py
```

它只把 **M6.1 真实可评估** 的字段映射为 `TaskPreference`：

- `feature` + `value=weight` + `direction=minimize` → weight / lower；
- `feature` + `value=weight` + `direction=maximize` → weight / higher；
- `color` / `material` + `prefer` → 对应 categorical attribute；
- inferred、retracted、unsupported free text（例如 “lighter option” 没有方向）不进入 M6.1。

M6.1 本身复用：

```python
preference_construction.rank_with_task_preferences(...)
```

没有新建排序模型，也没有新增属性规则。

### 执行位置

`_execute_plan`：

```text
hard constraint / role eligibility filtering
-> presentable candidate set
-> M5 soft preference adapter
-> M6.1 rerank
-> role eligibility + limit k
-> final Top-K
```

M6.1 只重排，不新增、不删除候选。被 hard constraint 排除的商品不会因软偏好重新进入。

### M7 answer / R1 RERANK

审计结果：

- 正常 `run_shopping_turn` 最终路径进入 `_execute_plan`，可以消费 soft preference；
- M7 `option.action in {"none","skip"}` 原先直接返回 base response，完全绕过 M6.1；
- R1 `RERANK` 分支原先只复用 `r1_previous`，没有 soft rerank；
- P1B-3 在两个路径都补上了：

```text
M7 none/skip:
    base candidate snapshot -> M6.1 soft rerank -> final answer (no tool call)

R1 RERANK:
    previous candidate window -> M6.1 soft rerank -> final answer
```

没有伪造 R1 action；没有为满足 Gold 强制重新检索。

---

## 2. M5 哪些字段实际进入 M6.1

实际消费的字段：

- `preference_id`（provenance）；
- `kind`；
- `value`；
- `direction`；
- `scope`；
- `lifecycle`；
- `is_soft`（必须为 false）；
- `polarity`。

不消费：

- inferred hypothesis（`is_soft=True`）；
- retracted / superseded；
- 没有可评估属性的自由文本 feature；
- category 短语；
- hard constraint。

Legacy Memory 与 M5 去重：

- `merge_soft_preferences` 以 `(attribute, action, value)` 去重；
- legacy / M7 confirmed task preference 优先；
- M5 同 key 不重复追加，不会双倍权重。

---

## 3. 与 Legacy Memory 的区别

P1B-3 消费的 M5 额外信息：

- **direction**：例如 `weight/minimize`；
- **scope**：task / long_term；
- **lifecycle**：explicit / confirmed；
- **provenance**：preference_id；
- **retraction 生命周期**：retracted feature 从 for_ranker 消失。

Legacy Memory 的 M9 active entries 仍作为原有输入存在；P1B-3 不是第二个 memory：

- 只读取 `for_ranker` 已有 active projection；
- 排序层没有自己的偏好状态；
- 没有把 M5 feature 写回 legacy memory；
- 没有对同一 weight preference 重复计权；
- 没有把 inferred preference 升级为 confirmed。

真实开发任务中：

- dev_08、dev_15 的 M5 weight 偏好被 M6.1 实际消费；
- dev_07、dev_14 的 “lighter option” 没有 direction，被明确拒绝消费；
- dev_06 的 “titanium cookware” 没有可用的 M5 属性映射，不猜测。

---

## 4. Soft_off / Soft_on 对照

两组使用相同：

- catalogue / history；
- SASRec / Two-Tower / BM25 / RRF；
- state mode（shadow）；
- role / hard constraint 规则；
- 同一 M5 event replay；
- 同一 k=5；
- 同一候选集合。

soft_off 保留原有执行路径，不注入 M5 soft preference；soft_on 只增加 M5 → M6.1 通道。

逐轮结果：

- 20 任务 pass：8 vs 8；
- `candidate_identity_changed` 全为 false；
- 无 fetch / retrieval 次数增加；
- 无 rerank 异常；
- M7 none/skip 与 R1 RERANK 未重复执行 tool。

### 实际消费情况

| task | active soft | consumed | rerank applied | ranking changed | final order changed |
|---|---:|---:|---:|---:|---:|
| dev_08 | 2 | 1（weight/lower） | true | false | false |
| dev_15 | 2 | 1（weight/lower） | true | false | false |
| 其余 dev 任务 | 0-3 | 0 | false | false | false |

说明：

- dev_08 / dev_15 确实建立了 M5 weight preference，并进入 M6.1；
- 但候选商品的重量证据相同、缺失或基础顺序主导，因此最终顺序没有变化；
- 没有为了制造排名变化修改权重或候选；
- 候选身份集合保持不变。

---

## 5. 撤回偏好

dev_08：

```text
turn 0: 建立 water filter 目标
turn 1: 确认 weight/lower
turn 2: M6.1 consumed weight，rerank_applied=true，same order
turn 3: “Forget the weight preference”
        active soft weight 消失
        consumed_soft_preference_count=0
        retraction_effective=true
        其他 category preference 保留
```

最终 pass 保持。

说明：

- 撤回后 M6.1 不再获得 weight feature；
- 不自动恢复更早的 weight 偏好；
- 其他偏好（category）继续存在；
- 如果候选集合/基础排序已变化，不承诺与历史顺序完全一致。

---

## 6. UNKNOWN 与 Hard Constraint

- 重量缺失的商品不会获得轻量优势；
- `rank_with_task_preferences` 把未知证据排在已知证据之后，未知数由 `unknown_evidence_count` 记录；
- 软偏好不改变 hard constraint；硬约束排除的候选不会因 soft rerank 进入；
- 没有把 UNKNOWN 当成 MATCH 或可信排序依据。

对应测试：

- `test_missing_weight_evidence_gets_no_fabricated_advantage`
- `test_soft_ranking_never_adds_a_candidate_removed_by_hard_constraints`

---

## 7. M7 answer / R1 action 审计结论

对 dev_09、dev_17：

- 最终路径是 M7 “No additional preference” 回答；
- 没有实际执行 RETRIEVE/RERANK；
- 因此 `actions=[]` 是真实无动作，不是漏记；
- P1B-3 不补造 action；
- 如果该路径上有有效 soft preference，现在已经可以应用 M6.1（见 dev_15）；
- 没有证据表明 R1 偷选了 RERANK。

分类：

1. M7 使用已有效候选快照完成回复：合理；
2. 新目标/偏好使候选失效却绕过执行：本阶段通过 `_execute_plan`、M7 none/skip、R1 RERANK 的 M6.1 补充解决 soft 排序；type/hard 变化仍走既有 R1 规则；
3. 仅 action 断言不匹配：dev_09/dev_17 属于这一类，未伪造。

---

## 8. 诊断字段

每轮新增：

- `active_soft_preference_count`
- `consumed_soft_preference_count`
- `soft_preference_source`
- `rerank_applied`
- `ranking_changed`
- `candidate_identity_changed`
- `unknown_evidence_count`
- `retraction_effective`
- `ranking_reason`

外加：

- `consumed_preferences`（preference_id / attribute / action / value / scope）
- `candidate_count`

控制逻辑：

- `candidate_identity_changed` 始终通过候选集合身份比较得出，不把重新召回产生的新商品归因于排序；
- `ranking_changed` 只来自 M6.1 的 `changed_order`；
- `retraction_effective` 基于上一轮 consumed count 降为 0；
- `soft_preference_source` 只在真实消费 M5 feature 时为 `m5_for_ranker`。

逐轮 CSV：

`benchmarks/agent/reports/P1B3_SOFT_OFF_ON_COMPARISON.csv`

---

## 9. 20 任务变化

| task | soft_off | soft_on | 说明 |
|---|---:|---:|---|
| P1B-2 8 个通过任务 | pass | pass | 无回归 |
| dev_08 | pass | pass | weight consumed，retraction effective |
| dev_15 | fail | fail | weight consumed，无最终顺序变化 |
| dev_07/14 | fail | fail | “lighter option” 无 direction，未消费 |
| dev_06 | fail | fail | 非可映射 feature，不猜测 |
| dev_09/17 | fail | fail | M7 answer 无 R1 action，非 soft 层 |

通过率未变化：8/20。没有为了提升开发集通过率修改任何排序权重、候选或 Gold。

---

## 10. 受控测试

新增 `tests/test_p1b3_soft_rerank.py`：

1. inferred / unsupported preference 不进入 M6.1；
2. confirmed weight preference 被消费并改变相同候选池顺序；
3. 缺失重量不伪造优势；
4. retracted preference 不再产生 soft evidence；
5. Legacy + M5 同 weight 不重复计权；
6. material 偏好映射到 M6.1 支持属性；
7. soft ranking 不增加 hard-filtered 集合之外的候选。

此外保留 P1B-2 / M5 / M7 回归。

最终相关测试结果：

```text
228 passed
```

---

## 11. 直接回答

1. **M5 哪些字段实际进入 M6.1？**
   - active、explicit/confirmed 的 feature direction/weight；
   - color/material prefer；
   - 保留 preference_id、scope、lifecycle、polarity；
   - inferred、retracted/superseded、无方向自由文本不进入。

2. **与 Legacy Memory 相比新增什么？**
   - M5 提供 direction、scope、lifecycle、retraction 生命周期和 provenance；
   - 但真实 dev 任务中只有 dev_08/dev_15 的 weight 被实际消费；
   - 没有重复计权。

3. **是否存在真实候选池内排序变化？**
   - 控制测试中存在（synthetic real catalogue facts）；
   - 20 个开发任务中 dev_08/dev_15 的 M6.1 被执行，但实际最终顺序未变化；
   - 原因记录为 same_order，未伪造变化。

4. **撤回后排序影响是否消失？**
   - 是。dev_08 撤回后 active weight 消失，consumed=0，`retraction_effective=true`，最终仍 pass；
   - 其他 category 偏好保留。

5. **M7 answer 路径是否可能绕过必要排序？**
   - 原先会；P1B-3 已在 none/skip 路径加入 M6.1 soft rerank，不调用工具；
   - dev_15 证明该路径可消费 M5 weight；
   - R1 action 仍不伪造。

6. **是否出现重复计权、错误属性、UNKNOWN 误判？**
   - 去重逻辑避免 Legacy/M5 重复；
   - 只消费可映射属性；
   - UNKNOWN 不获得优势，unknown 独立计数；
   - 没有 inferred → confirmed 自动升级。

7. **20 个开发任务有无可解释变化？**
   - soft_off 与 soft_on 均 8/20，无回归；
   - dev_08/dev_15 有真实 M5 消费但最终顺序未变；
   - 其余没有可消费的 confirmed M5 soft feature；
   - 未修改冻结数据或评测成功判定。

---

## 12. 交付与遗留

交付：

- `P1B3_SOFT_MEMORY_RERANK.md`
- `P1B3_SOFT_OFF_ON_COMPARISON.csv`
- `runs/agent_benchmark/dev_c_p1b3_soft_off_v2/`
- `runs/agent_benchmark/dev_c_p1b3_soft_on_v2/`
- `recommendation/demo/soft_preference_ranking.py`
- `tests/test_p1b3_soft_rerank.py`

遗留：

- 20 个开发任务缺少足够明确、可比较的 M5 soft weight 场景，因此没有开发集最终顺序变化；这是数据场景限制，不是伪造排序变化；
- dev_07/dev_14 的 “lighter option” 没有 M5 direction，仍不消费；
- dev_09/dev_17 的 R1 action 断言问题仍是 M7 交互路径/evaluator 问题，不在 P1B-3 伪造。

本阶段结束，不继续 P1C/P2，不 commit/push/deploy。
