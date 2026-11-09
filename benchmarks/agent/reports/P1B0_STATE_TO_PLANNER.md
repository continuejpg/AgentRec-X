# P1B-0：M5 Effective State → R1 Retrieval Planner 真实链路验证

## 0. 边界

本阶段只验证/补齐 M5 → R1 输入链路；没有：

- 修改冻结任务、Gold、冻结哈希或评测协议；
- 接入 LLM API；
- 优化 BM25、BGE、ItemCF、Fine Ranker；
- 修改默认 Serving V1 或模型参数；
- 根据 Shadow/Applied 开发集差异宣称泛化收益；
- 覆盖历史 `dev_c_final2` / `dev_c_p1a_final`。

新增运行：

```text
runs/agent_benchmark/dev_c_p1b_shadow_final/
runs/agent_benchmark/dev_c_p1b_applied_final5/
```

---

## 1. 修改前的真实数据流

### Shadow

```text
user message
  -> M9 extraction + legacy snapshot
  -> derive_intent()/query_terms()                 (legacy intent/query)
  -> integrate_turn()                              (records M5 events)
  -> _integrate_state_turn()
       mode=shadow: return (trusted_constraints, intent) unchanged
  -> R1 normal-turn action: based on r1_previous + extraction
  -> run(payload.constraints=trusted_constraints)
  -> eligibility/ranking
```

Shadow 中：

- `for_agent`、`for_constraints`、`for_ranker` 只被构造用于 diagnostics；
- R1 不读取这些 projection；
- M5 对推荐结果没有执行链影响。

### APPLY_CONSTRAINTS（原实现）

```text
integrate_turn()
  -> outcome.authorized_constraints = for_constraints(state)
  -> merge_constraint_pairs(trusted_constraints, authorized)
  -> effective_intent = _effective_intent(intent, merged)
  -> run(payload.constraints=merged)
```

原实现已经把**用户显式 hard constraint**接入 eligibility；但：

- R1 的 `RERANK` 分支会直接返回旧候选窗口，不读取新 constraints；
- R1 的 RETRIEVE query terms 来自 legacy intent，不读取 M5 新授权项；
- M5 soft preferences 完全未进入 R1；
- 没有跨轮 M5 hard set 变化检测。

---

## 2. 本阶段的安全 M5 → R1 通道

### 2.1 输入字段

只使用可信、当前有效字段：

| 来源 | 字段 | 进入 R1 的方式 |
|---|---|---|
| `for_constraints(state)` | active, user-explicit, task/long-term authorized | 加入 `payload.constraints`，可能改变 eligibility |
| `for_agent(state).preferences` | active soft/explicit beliefs | 只进入 bounded state diagnostic；本阶段**不**追加到 effective intent/query，不成为 hard constraint |
| `outcome.authorized_constraints` | 当前授权 hard set | 用于跨轮变化检测 |
| `state_projection` | scope/kind/value/is_soft/status | 诊断记录；scope/status 不丢失 |

没有使用 inferred hypothesis 作为 hard constraint；`for_constraints` 本身只输出
`USER_EXPLICIT` 且 active 的约束。

### 2.2 行为变化

1. **当前显式输入优先**
   - `merge_constraint_pairs` 支持 current-turn explicit kind 优先；当前请求显式 constraint 与旧 M5 同 kind 不同 value 时，不再追加旧值。
   - 已知限制：dev_05 的旧 `price_max=30` 与 `price_max=10` 都是 M5 历史 hard constraints，最后 `show more options` 没有当前 explicit constraint，因此 applied R1 仍然同时收到二者；没有 final 差异，但保留为残余过度约束。

2. **M5 hard set 变化触发 RETRIEVE**
   - 服务维护每 task 的 `_m5_authorized_constraints`；
   - 若当前授权 hard set 相比上一轮变化，R1 action 强制为 `RETRIEVE`，
     reason=`m5_effective_constraints_changed`；
   - 避免 `RERANK` 复用旧约束下产生的候选窗口。

3. **Effective query terms**
   - APPLY 模式下从 effective intent 重新计算 catalogue terms；
   - 只有 M5 新授权的**hard** query-safe 项会进入 query；
   - soft preference 不进入 query/constraints，只保留在 diagnostic。

4. **Soft 保持 soft**
   - M5 soft beliefs 只进入 bounded state diagnostic / `for_agent` projection；
   - 本阶段不追加到 effective intent，不进入 query/constraints；
   - 不产生 hard filter。

## 3. 受控链路测试

新增 [tests/test_m5_r1_channel.py](tests/test_m5_r1_channel.py)。

### 测试 1：M5-only hard constraint

直接在 M5 event store 中 seed `color=red`，然后运行普通 shopping turn：

```text
shadow:
  R1 consumed constraints = []
  final candidates > 1

applied:
  R1 consumed constraints = [{"kind":"color","value":"red"}]
  final candidates = ["cand-red"]
```

这证明 APPLY 的真实 hard constraint 从 M5 进入 R1 并改变 eligible Top-K。

### 测试 2：soft preference 不升级

M5 active feature `red`：

```text
applied:
  R1 constraints = []
  m5_soft_preferences 包含 red
  final candidates > 1
```

这证明 soft preference 可见但未成为 hard filter。

### 测试 3：撤回

先 seed hard `color=red`，完成一轮后撤回：

```text
before: R1 constraints = [color=red]
after : R1 constraints = []
        R1 action = RETRIEVE
        reason = m5_effective_constraints_changed
        final candidates 恢复 > 1
```

### 测试 4：未建立的偏好

对不存在的 `weight` 发起 value-only retraction：

- `retraction_match.resolved == 0`；
- 不产生 `preference_retracted` 事件；
- R1 constraints 为空。

已有测试：

- [tests/test_state_beliefs.py](tests/test_state_beliefs.py)：定向撤回、scope 隔离、constraint-only 撤回；
- [tests/test_m5_2_state_adapter.py](tests/test_m5_2_state_adapter.py)：SQLite reopen/replay；
- M7 测试：确认重量偏好后，现有候选按可信属性重排。

## 4. 20 个开发任务 Shadow vs Applied

| 指标 | C_shadow | C_applied |
|---|---:|---:|
| runnable | 20 | 20 |
| completed | 20 | 20 |
| deterministic pass | 4/20 | 4/20 |
| final IDs differences | 0 | 0 |
| R1 action differences | 0 | 0 |
| R1 consumed hard-constraint differences | 0 | 1（dev_05 历史 ceiling 重复） |
| R1 query-term differences | — | 0 |
| M5 soft diagnostic differences | — | 15 |

详细逐任务对比：[P1B0_SHADOW_APPLIED_COMPARISON.csv](P1B0_SHADOW_APPLIED_COMPARISON.csv)。  
逐轮结构化轨迹：[P1B0_TURN_TRACES.jsonl](P1B0_TURN_TRACES.jsonl)。

### 为什么大多数任务结果相同？

这是**行为本应相同**与**状态尚未改变执行输入**的混合，不是单纯“集成失败”：

- dev_01：M5 撤回后 authorized hard set 已为空；最终 B08QYL2SD1 的失败来自检索/角色/排序，不是旧 constraint 继续生效。
- dev_05：M5 历史中同时保留 `price_max=10` 和旧 `price_max=30`，applied 的 R1 constraints 仍然包含二者；没有 final/action 差异。这是 append-only state 的重复 ceiling 现象，不是新的误召回；本报告保留为残余过度约束。
- dev_06/07/14：effective state 是 feature/category，不是 task expected 的 material/weight；R1 soft 不 hard-filter，且这些属性不进入当前 R1 query/ranking。
- dev_08：没有建立 weight preference，因而没有 retraction event；shadow/applied 都没有可撤回 hard constraint。
- dev_09：商品目标变化仍是原有 planner 行为；M5 没有产生新的 authorized hard constraint，因此 applied 不改变 action。
- dev_20：M5 soft `price` 只出现在 bounded state diagnostic 中，没有追加到 effective intent；因此也没有 query/final 差异。这证明“状态可见”不等于“state 已经影响 planner”。

---

## 5. 重复与顺序

反向顺序子集运行：

```text
dev_20, dev_08, dev_06, dev_05, dev_01
```

结果与正向 applied 运行逐任务完全一致：

- 所有 5 个任务 final IDs 相同；
- 所有 5 个任务 R1 action 相同；
- `dev_20` 在 shadow、applied、reverse 三次运行中均为 pass；
- 没有观察到跨任务 pending/constraint 残留。

限制：

- 只对 5 个任务做了 reverse 顺序实验，不是 20 个任务全量顺序矩阵；
- 仍共享同一个 runtime/SQLite 文件，隔离依赖 session/task key；
- 因此可报告“已观测顺序稳定”，不能报告“已证明完全顺序不敏感”。

---

## 6. Applied 风险

- **误召回**：controlled soft/hard tests 未产生新候选；20 个开发任务 final IDs 无差异。
- **错误过滤**：controlled hard test 正确过滤为 `cand-red`；没有观察到无关 hard constraint 误加。
- **过度约束**：dev_05 的历史 `price_max=10` 与旧 `price_max=30` 同时进入 applied R1 constraints；没有 final/action 差异，但这是一个真实的残余过度约束，后续需要 state lifecycle 处理；同 kind 的**当前轮**输入优先已实现并有单测。
- **隐式升级**：soft 状态没有进入 constraints；inferred 由 M5 projection 排除；测试覆盖。
- **R1 RERANK 旧窗口**：通过 M5 hard set 变化检测强制 RETRIEVE；controlled retraction test 验证。

---

## 7. 直接回答

1. **M5 哪个具体字段真正进入 R1？**
   - `for_constraints(state).constraints`（active、user-explicit、scope-aware）经 `merge_constraint_pairs` 后进入 `payload.constraints`，参与 eligibility；
   - `for_agent(state).preferences` 进入 bounded state diagnostic；本阶段不把它追加到 effective intent/query，也不升级为 hard constraint；
   - `_m5_authorized_constraints` 用于检测跨轮 hard set 变化，触发必要 RETRIEVE；
   - `for_ranker` 不是 R1 的直接输入（只在 M7/M6.1 实验分支使用）。

2. **哪些任务出现可解释的 RETRIEVE/RERANK 或查询变化？**
   - 20 个开发任务没有出现新的 action/query/final 差异；
   - controlled hard/retraction tests：M5 hard constraint 改变 eligibility；hard set 变化时 R1 action 变为 RETRIEVE；
   - soft preference 只出现在 state diagnostic，未进入 R1 query/constraints。

3. **哪些任务只改变状态却没有改变推荐结果？为什么？**
   - dev_01、dev_05、dev_06、dev_07、dev_14、dev_20 等；
   - 原因分别是旧 constraint 已正确消除、state 只是同一 legacy 条件、soft preference 仅进入 diagnostic 而未进入 R1 query/constraints、或当前候选池/排序未受新增 hard constraint 影响。

4. **撤回后的旧硬约束是否彻底失效？**
   - M5 projection 中：是。dev_01 修复后 hard constraint count 1→0，constraint=superseded；controlled retraction test R1 constraints 从 `[color=red]` 变为 `[]`；
   - 但 legacy memory 中仍可能有 `feature` 展示项，benchmark 的 `active_attributes` 字段混合了 legacy memory，不能把它当作 M5 constraint。

5. **Applied 是否引入新的误召回/错误过滤/过度约束？**
   - 20 个开发任务 final/action 无差异；
   - 没有观察到新的误召回或错误过滤；
   - dev_05 仍显示历史相同 kind 的重复 price ceiling（10 与 30 同时进入 R1 constraints），这是残余过度约束，但没有改变最终列表；
   - soft 不升级、hard-only 过滤、撤回后恢复均通过 controlled tests；
   - 没有泛化收益或质量问题声明。

6. **当前 C 组能否称为 Memory-Guided Retrieval？**
   - **不能泛化地这样称呼。**
   - 当前 C_shadow 不读取 M5 projection；
   - C_applied 已有一条可验证的 hard-constraint + soft-diagnostic 通道，但 20 个开发任务没有产生 final/action 改变，且 soft 偏好尚未用于 R1 ranking；
   - 更准确名称是：**M5-aware experimental input channel under apply_constraints**，不是已证明的 Memory-Guided Retrieval 产品能力。

---

## 8. 测试与遗留

新增/相关测试：

```text
tests/test_m5_r1_channel.py       5 passed
tests/test_m5_1_state_adapter.py  39 passed
tests/test_m5_2_state_adapter.py
tests/test_state_beliefs.py
tests/test_m7_grounded_actions.py
```

完整 P1B/M5 相关回归：`240 passed`。新增测试覆盖：

- M5-only hard constraint → R1 constraints 与 final 变化；
- shadow 不消费 M5 constraint；
- soft preference 不升级；
- retraction 后 hard set 清空并触发 RETRIEVE；
- unmatched retraction 不产生伪造事件；
- current-turn explicit kind 优先；
- 既有 SQLite replay/scope 测试保持通过。

本阶段完成后先停止，不自动执行 P1B-1/P1C；无 commit、push 或 deploy。

后续若继续：

- 需要把 soft M5 preferences 接入 M6.1 rerank（不是 hard filter）；
- 需要把 R1 action/query evidence 进一步结构化；
- 需要做全 20 任务顺序矩阵和独立 state DB 隔离实验；
- dev_08 的 Gold/M7 属性错位仍需要单独处理，不能靠 M5 通道假装撤回发生。
