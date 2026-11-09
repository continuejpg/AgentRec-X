# P1A：M5 有效偏好投影与多轮澄清状态修复

## 范围

本阶段只修改：

- M5 `PREFERENCE_RETRACTED` 对同一目标的 hard-constraint projection；
- value-only retraction 对 constraint-only survivor 的唯一解析；
- pending preference clarification 的新目标/撤回/skip 分流；
- benchmark 的 pending option simulator 与诊断字段；
- 回归测试。

未修改：

- `frozen.json`、冻结哈希、冻结评测协议；
- Gold、expected ASIN、评测断言；
- Serving V1、R1 检索策略、M6.1、BM25、SASRec、Two-Tower；
- A/B/D 或 LLM API。

原始 C 结果保留：`runs/agent_benchmark/dev_c_final2/`（2/20）。
新结果：`runs/agent_benchmark/dev_c_p1a_final/`（4/20）。

---

## 1. M5 修复

### 根因

`recommendation/state/belief_updater.py::_apply_preference_retracted` 原先只把匹配的
`PreferenceBelief` 标为 RETRACTED，没有处理同一 scope/kind/value/polarity 的
`DecisionConstraint`。

dev_01 的正确事件序列：

```text
task_context_updated
user_preference_stated (feature=replacement mouthpiece...)
user_constraint_stated (feature=replacement mouthpiece...)
preference_retracted (feature=replacement mouthpiece...)
```

修复前 withdrawal 后：

```text
M5 preference_count = 0
M5 hard_constraint_count = 1        <-- 错误仍活动
```

修复后：

```text
M5 preference_count = 0
M5 hard_constraint_count = 0
M5 preferences = []
M5 constraints = []
M5 ranker features = []
constraint record status = superseded
```

修复只匹配同一 scope + kind + value + polarity；不同 feature 值和其他 kind 不会被清空。
新增测试：

- targeted retraction 清除目标 belief 和 constraint；
- 另一个 feature 保持 active；
- task-scope retraction 不删除 long-term constraint；
- constraint-only survivor 可被唯一 retraction 清除；
- SQLite reopen/replay 后结果一致。

注意：修复前 benchmark 报告中的 `active_attributes=feature` 还来自 benchmark adapter 把
legacy M9 memory 和 task preferences 合并；它不是 M5 effective preference。M5 的实际
projection 修复后为空。该 reporting 混淆已在 P1A 报告中明确，不再把 legacy memory 当作
M5 active state。

---

## 2. dev_08 的真实原因

dev_08 的关键不是“pending 永远卡住”这么简单。逐轮 trace：

```text
turn 0: waiting_for_user
        M7 实际问的是 brand（KOR/Stanley/Membrane Solutions/...），不是 weight
        event: user_preference_stated(water filter for hiking)

turn 1: "I prefer the lighter option."
        不匹配 brand 选项
        旧 adapter fallback 错误选中 pref:none，导致没有记录 weight

turn 3: "Forget the weight preference."
        之前从未产生 weight preference
        因而 resolve_retraction 没有可撤回的目标
```

旧行为在 turn 3 继续停在 waiting；P1A 修复后：

- adapter 不再在无法匹配时自动选择 `pref:none`；
- service 会把明确的 withdrawal/new goal 从 pending preference 中分流，作为普通用户轮；
- withdrawal 轮不再立刻追问无关的 M7 preference；
- dev_08 现在 **finished**，最终 3 个 water filter；
- 但仍然 **没有 `preference_retracted` 事件**，因为没有 weight preference 可撤回。

因此 dev_08 的 deterministic pass 不能证明“weight 撤回事件”发生；它只证明 pending
不再永久阻塞。这是 Gold/task 与 M7 实际提问之间的语义错位，属于本阶段遗留项。

---

## 3. Pending 修复行为

`recommendation/demo/agent_service.py` 现在区分：

- 带 `clarification_option`：正常回答；
- 无 option、且消息是明确撤回/终止：丢弃 pending，按新用户轮处理；
- 无 option、且消息是明确新目标：丢弃 orphaned pending，按新用户轮处理；
- `No additional preference` / `skip`：仍作为合法 decline 处理；
- 普通自由文本偏好回答：不强行当作新目标。

Benchmark adapter 也同步移除了“匹配不到选项就选 none”的兜底；这是一个 benchmark
用户模拟器 bug，不能让它把撤回文本变成一次伪造的 decline。

新增测试覆盖：

- `forget ...` 不永久阻塞；
- 新目标不永久阻塞；
- `skip` / `no additional preference` 不被误判为新目标；
- pending 被消费后不再 replay 旧请求。

---

## 4. R1 实际读取状态

P1A 的 M5 修复改变了 SQLite event log、`for_agent`、`for_constraints`、`for_ranker`
的 effective state。

但当前 C benchmark profile 仍是 `AGENTRECX_STATE_MODE=shadow`：

- R1 normal-turn plan 仍读取原有 trusted intent/legacy memory；
- R1 **没有**直接读取 M5 `for_constraints`；
- dev_01 修复后 R1 仍为 `RETRIEVE`，query terms 仍为 `complete water filter`，并没有
  `replacement mouthpiece` 旧词；最终仍是 B08QYL2SD1；
- 因此本阶段没有观察到“M5 修复直接改变 R1 决策”的效果。

这不是把 R1 改坏了，而是 shadow 模式的设计边界：M5 记录有效状态，但不在 shadow 中
供给 R1。若要让 R1 真正消费 M5 effective state，需要 P1B 将 integration 切到
`apply_constraints` 或增加受控的 effective-state input；本阶段没有做，因为这会改变
推荐路径语义。

**结论：M5 effective state 已修复；但在当前 shadow benchmark 中，R1 仍未读取它。**

---

## 5. C 组修复前后结果

| 指标 | 修复前 dev_c_final2 | 修复后 dev_c_p1a_final |
|---|---:|---:|
| runnable | 20 | 20 |
| completed | 19 | 20 |
| deterministic pass | 2/20 | 4/20 |
| human labels | 0 | 0 |

### 新增通过

- `dev_08_filter_lightweight_withdraw`
- `dev_20_broad_gear_to_filter`

### 原有通过无回归

- `dev_03_accessory_target`
- `dev_19_cookware_set_reference`

### 仍然失败且保留

- `dev_01`：最终仍是 B08QYL2SD1，M5 已清空但最终角色不匹配；剩余问题是 retrieval/ranking/legacy context。
- `dev_06`、`dev_07`、`dev_14`：M5 仍没有 expected `material`/`weight`，这是 rule extraction/M7 提问与 Gold 语义不一致，不是本阶段撤回投影问题。
- `dev_08`：通过 deterministic task，但没有 weight `preference_retracted` 事件（见上）。
- 其余失败仍归因于候选召回、角色过滤、R1 action 或 Gold 粒度；没有做跨模块临时修补。

### 失败类型变化（按修复前审计口径）

| 根因 | 修复前 | 修复后 |
|---|---:|---:|
| `none`（通过） | 2 | 4 |
| `memory_state_error` | 4 | 4 |
| `eligibility_or_ranking_error` | 7 | 7 |
| `evaluator_or_gold_issue` | 3 | 2 |
| `retrieval_candidate_miss` | 2 | 2 |
| `planner_decision_error` | 1 | 1 |
| `clarification_state_error` | 1 | 0 |

说明：dev_08 从 clarification 失败变为 deterministic pass；dev_20 从 identity/Gold 失败变为 pass；
dev_01 的失败仍保留为 memory/eligibility 混合问题。该表不是重新跑审计器后的正式标签，
而是按原审计根因与修复后通过集合对账的 P1A 结果。

逐任务前后表：[P1A_STATE_COMPARISON.csv](P1A_STATE_COMPARISON.csv)。

---

## 6. 新增/既有测试

P1A 相关：

```text
tests/test_state_beliefs.py
tests/test_m5_2_state_adapter.py
tests/test_m5_2_acceptance.py
tests/test_m5_2_integration.py
tests/test_m5_1_state_adapter.py
tests/test_m5_1_acceptance.py
tests/test_m5_state_acceptance.py
tests/test_m7_grounded_actions.py
tests/test_agent_benchmark.py
```

最近一次相关回归：

```text
220 passed
```

更广的相关回归（M5/M7/R1/benchmark/demo/P4）：

```text
442 passed
```

另外新增：

- M5 targeted retraction/constraint scope/replay tests；
- pending withdrawal integration test；
- benchmark pending resolver test；
- D proposal 仍仅 rewrite 的审计测试保持通过。

`dev_c_final2` 原始 2/20 未覆盖或删除。

---

## 7. 遗留风险

1. **R1 尚未读取 M5 effective state（shadow）**：需要 P1B 决定是否切换 integration mode。
2. **dev_08 没有实际 weight retraction event**：原因是此前从未记录 weight；需要修 Gold/task-M7 对齐或提取语义，P1A 未改。
3. **dev_06/07/14 仍缺少 material/weight**：提取器没有把 “titanium cookware”/“lighter option”映射为对应属性；不是 M5 projection 撤回 bug。
4. **benchmark active_attributes 仍混合 legacy memory**：P1A 报告已区分；若要自动报表，应拆成 `m5_active` 与 `legacy_memory_active`。
5. **任务/会话隔离**：C 使用唯一 session/task key，未见跨任务污染；但 runtime 仍共享 SQLite 文件，尚未做改变任务顺序的完整实验。
6. 冻结集未运行，未修改；人工标签仍为 0。

---

## 8. 直接回答

1. **dev_01 的 active feature 是不是撤回投影 Bug？**
   - 真正存在的是 **M5 hard constraint 未随 targeted preference retraction 撤销**的 Projection Bug；已修复，hard_constraint_count 1→0，constraint → superseded。
   - benchmark 报告的 `active_attributes=feature` 不是 M5 effective preference，而是 legacy M9 memory 与 task preference 的合并展示；不能再把它当作 M5 active preference。

2. **dev_08 为什么没有进入正确撤回分支？**
   - turn 0 实际问的是 brand，不是 weight；
   - “lighter option” 不是 brand 选项，旧 simulator 又错误 fallback 到 `pref:none`；
   - 因而 weight preference 从未建立，turn 3 的 `forget weight` 没有可撤回目标；
   - P1A 修复了 pending 阻塞和 fallback，dev_08 现在 finished/pass，但**没有实际 weight retraction event**。

3. **修复是否真正改变了下游 R1 使用的状态？**
   - M5 effective state 改变了；但在当前 `shadow` C profile 中 R1 不读取 M5 projection，因此没有观察到 R1 行为改变。
   - 这是 P1B/后续集成问题，不是本阶段可宣称的收益。

4. **修复前后 C 组通过率和失败类型有什么变化？**
   - 2/20 → 4/20；
   - completed 19 → 20；
   - 新增 pass：dev_08、dev_20；
   - 原 pass 无回归；
   - dev_08 的 clarification 失败消失，但语义上的 weight retraction 事件仍缺失；
   - 其余失败仍保留。

5. **是否存在跨任务/跨 session 污染风险？**
   - 当前未见；C 使用唯一 session/task key，pending 清理按 task 进行。
   - 共享 runtime/SQLite 与 legacy memory 合并仍是审计风险；未做运行顺序重排实验。
