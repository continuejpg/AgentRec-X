# P3C-3C 根因定位报告

## 1. 根因

第一处错误不是 M5 写入，也不是最终 `run_shopping_turn` 的 constraints；而是：

`recommendation/state/shopping_adapter.py::merge_constraint_pairs`（884-917）

第二轮：

```text
trusted_pairs = []
proposal_constraints = [price_max=50]
authorized_constraints = [price_max=50]
current_turn_kinds = ['price_max']       ← 错误来源
merged = []                              ← 正确 50 在这里被丢弃
```

`current_turn_kinds` 包含 `price_max`，`merge_constraint_pairs` 因此认为“当前 turn 已提供
price_max”，跳过 M5 authorized 的 50。之后 P3C-3 的防御性注入把 50 补回：

```text
constraint_trace.final_constraints = [price_max=50]
```

所以第二轮的实际最终 constraints 已是 50；但 Stub 轨迹中的 `r1_diagnostics.constraints`
仍是第一轮的 30，这是 **诊断未刷新/提前返回造成的 stale diagnostic**，不是 R1 实际输入。

证据：

```text
turn 2 constraint_trace:
  trusted_pairs=[]; proposal=[price_max=50]; authorized=[price_max=50]
  current_turn_kinds=['price_max']; merged=[]
  final_constraints=[price_max=50]

turn 2 r1_diagnostics.constraints=[price_max=30]   ← stale
```

第二轮最终为空不是价格过滤直接造成：候选层记录为 4 个候选（hard eligible=6, unresolved=4），
但 API 最终列表为空；空发生在 R1 之后的显式类型/最终收敛路径，不是 `price_max` 本身。

---

## 2. 四轮 Stub API 结果

Stub 明确标记 `STUB_TEST_DOUBLE_NOT_REAL_QWEN`，不是 Qwen。

| Turn | M5 active price_max | constraint_trace final | R1 diagnostics | Eligibility | Final |
|---:|---|---|---|---|---|
| 1 | 30 | [price_max=30] | 30 | eligible=10/ineligible=1/unresolved=4 | 3 IDs |
| 2 | 50 | [price_max=50] | **30 stale** | eligible=6/ineligible=0/unresolved=4 | **[]** |
| 3 | None | [] | [] | not evaluated | 3 IDs |
| 4 | None | [] | [] | not evaluated | 3 IDs |

- Turn 1：$30 进入 eligibility；
- Turn 2：最终 constraints 已是 50（不是旧 30）；R1 诊断 stale；
- Turn 3：撤回首轮/二轮预算，M5 hard count=0，不再价格过滤；
- Turn 4：旧预算未复活；
- Turn 2 final 为空是后续显式类型/最终收敛路径，不是预算过滤；该路径尚未单点修复。

详细：`P3C3C_FOUR_TURN_TRACE.csv`

---

## 3. 修复状态

### 已成立

- 第一轮 $30 进入 Eligibility（P3C-3 修复）；
- 第二轮最终 constraints 为 $50（P3C-3 防御性注入），不是旧 $30；
- 第三、四轮 M5 hard count=0，旧预算未复活（Stub）；
- 不新建第二套价格状态；复用 M5/ConstraintEvaluator。

### 未解决

1. `merge_constraint_pairs` 仍会因为 `current_turn_kinds` 丢掉 authorized 约束，依赖防御性
   注入才恢复；应进一步把 `current_turn_kinds` 的来源修正为真正的 current-turn trusted
   constraints，而不是 `_task_price_constraints` 的旧值。
2. 第二轮 final 为空，发生在 R1 之后的显式类型/最终收敛路径；尚未定位到唯一函数。
3. SQLite replay、同轮 API 幂等、其他 scope 隔离没有新增测试。
4. 真实 Qwen 未验证（无 GPU）。

---

## 4. 回答

1. **旧 $30 在哪个函数重新进入？**
   - 不是进入最终 constraints；`constraint_trace.final_constraints=50`。
   - 出现在 `merge_constraint_pairs` 的 `current_turn_kinds=['price_max']` 导致的 merged 空值，
     以及 `r1_diagnostics` 的 stale 记录（30）。
2. **修复后是否真正进入最终资格检查？**
   - 第二轮最终 constraints 是 50，hard constraint evaluated=true；价格更新已进入最终
     constraints。但 API final 为空，空发生在后续类型/最终收敛，不是价格过滤。
3. **预算更新/撤回是否正确？**
   - 更新：最终 constraints 正确为 50；R1 诊断 stale，需修。
   - 撤回：Turn3/4 Stub 中 M5 hard=0、无价格过滤，旧预算未复活；通过。
4. **是否错误接受 UNKNOWN？**
   - eligibility 有 unresolved 计数；没有证据把价格 UNKNOWN 当 MATCH。
5. **默认 Serving V1 是否不变？**
   - 是；未修改 `/v1/recommend`。

---

## 5. 交付

- `P3C3C_ROOT_CAUSE.md`
- `P3C3C_FOUR_TURN_TRACE.csv`
- `P3C3C_SOURCE_DIFF.txt`
- Stub 轨迹：`P3C3B_API_TRAJECTORY.jsonl` / `P3C2B_STUB_API_TRAJECTORY.jsonl`

未启动 GPU、未训练、未运行完整 Benchmark、未改 Serving V1/Gold/Evaluator、未 RL、
未 commit/push/deploy。
