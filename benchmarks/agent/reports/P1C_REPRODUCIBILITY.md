# P1C_REPRODUCIBILITY — 状态隔离、顺序与重复运行验证

## 0. 运行配置

同一配置：

```text
split=dev
policy=C
state_mode=shadow
soft_m5=off
fixed benchmark profile (SASRec/Two-Tower/BM25/RRF; BGE/ItemCF off)
```

运行：

| arm | output |
|---|---|
| forward | `runs/agent_benchmark/dev_c_p1b3_soft_off_v2/` |
| repeat (forward) | `runs/agent_benchmark/dev_c_p1c_repeat/` |
| reverse (20 tasks reversed) | `runs/agent_benchmark/dev_c_p1c_reverse/` |

三次均：

```text
runnable=20 completed=20 passed=8/20
```

每个 run 使用独立 `--out` 目录：

```text
decision_state.sqlite3
checkpoint.sqlite3
preference_memory.sqlite3
```

没有复用同一 session/task 的共享 state DB 来制造“隔离已通过”的表象。

---

## 1. 比较结果

逐任务比较：

- final product identities；
- retrieval actions；
- candidate pool snapshots；
- R1 reason/query terms/constraints；
- M5 semantic state summary:
  - `preference_count`
  - `active_attributes`
  - `soft_hypothesis_count`
  - `hard_constraint_count`
- legacy pass。

结果：

```text
20/20 tasks: final identities equal
20/20 tasks: retrieval actions equal
20/20 tasks: candidate pools equal
20/20 tasks: R1 query/constraints equal
20/20 tasks: semantic state summary equal
20/20 tasks: legacy pass equal
```

forward/repeat/reverse 都是 8/20。

---

## 2. raw state_digest 的 caveat

原始 `state_digest` 在三次运行中不同。原因：

- 每次 run 的 session/task identity、turn key、event id 与时间/事件 envelope 不同；
- digest 覆盖事件级 identity/provenance，不等于 effective semantic state。

因此：

- **不能用 raw digest 不等直接宣告状态泄漏**；
- P1C 比较的是 `preference_count`、effective `active_attributes`、
  `soft_hypothesis_count`、`hard_constraint_count`、R1 effective constraints 等
  语义字段；
- 语义字段全部一致。

---

## 3. 顺序影响

`--task-order reverse` 将 20 个任务顺序完全反转：

- final/action/candidate/R1/state 与 forward 一致；
- 没有观察到跨任务 session/state 泄漏；
- 没有观察到 reverse 造成的通过率变化。

这不证明任意并发隔离；它证明当前顺序化 benchmark 的 task/session 状态按任务隔离。

---

## 4. Soft_off / Soft_on 交叉检查

P1B-3 的独立运行：

```text
runs/agent_benchmark/dev_c_p1b3_soft_off_v2/
runs/agent_benchmark/dev_c_p1b3_soft_on_v2/
```

- 两组 20/20 completed，均 8/20；
- dev_08/dev_15 的 M5 weight preference 在 soft_on 中 consumed，但最终 final order
  未变化；
- soft_off 不注入 M5 soft；
- 候选身份集合在两组中保持不变；
- 无重复检索/重复工具调用增加。

逐轮数据：

`benchmarks/agent/reports/P1B3_SOFT_OFF_ON_COMPARISON.csv`

---

## 5. 独立 SQLite 与重复运行

- forward/repeat/reverse 各自独立目录与独立 SQLite；
- 同一 config 的 forward/repeat 语义一致；
- 20 个任务的重复运行没有出现不同 final/action；
- 重启/replay 的 evaluator V2 与 legacy 结果均可从保存轨迹重算（legacy CSV 与
  V2 CSV 来自同一批轨迹）。

---

## 6. 结论

- **task order 不改变当前 C 结果**；
- **repeat 不改变当前 C 结果**；
- **无证据表明共享 SQLite 或 session 唯一性掩盖了状态泄漏**；
- raw event digest 不适合作为语义一致性指标，已改用语义 state summary；
- 当前 benchmark 对顺序化、单进程、独立 `--out` 运行具备可复现性；
- 并发/多进程、真实 LLM 多 arm、同一 DB 多 policy 并发仍未验证，不能声称已验证。
