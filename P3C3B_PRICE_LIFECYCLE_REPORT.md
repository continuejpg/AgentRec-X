# P3C-3B 价格硬约束生命周期报告

## 1. 结论

本阶段**未完成价格更新/撤回的完整闭环**，但确认并修复了第一轮预算进入 Eligibility 的问题。

| 场景 | 状态 |
|---|---|
| 第一轮 $30 进入硬约束 eligibility | **PASS（Stub）** |
| 第二轮 $30 → $50 更新 | **FAIL / UNRESOLVED** |
| 第三轮撤回 $50 | **未执行** |
| 第四轮 show more 无旧预算 | **未执行** |
| 默认 Serving V1 | 未修改 |

Stub 不是真实 Qwen，不能冒充模型结果。

---

## 2. 根因定位（已确认部分）

真实 Stub Shopping API 第一轮：

```text
proposal update_constraint(price_max=30)
M5 hard_constraint_count=1
BEFORE: hard_constraint_evaluated=false, R1 constraints=[]
AFTER : hard_constraint_evaluated=true
        eligible=10, ineligible=1, unresolved=4
        R1 constraints=[price_max=30]
        final=3 MATCH known-price products
```

根因是 proposal 当前硬约束在进入 `_integrate_state_turn`/R1 合并时被当成
“current-turn trusted constraint 已提供”，导致 M5 authorized constraint 被 `current_turn_kinds`
跳过，最终 eligibility 未执行。

`recommendation/demo/agent_service.py::run_shopping_turn` 增加了防御性注入：

- 只对有效 proposal constraints；
- 按 kind 替换，而不是 min/max 合并；
- 复用既有 `AgentConstraintItem` / `CandidateEligibilityEvaluator`。

因此第一轮修复成立。

---

## 3. 第二轮 FAIL

第二轮 Stub：

```text
turn 1: under $30
turn 2: make budget $50 + forget lighter preference
```

M5 第二轮状态 hard count=1；但实测 R1 仍报告：

```text
R1 constraints = [{'kind':'price_max','value':'30'}]
```

而不是 $50；最终候选为空。

这说明旧 $30 仍在某个合并/查询路径重新进入 R1，可能来自：

- `_task_price_constraints` 的旧值；
- legacy intent hard constraints；
- `merge_constraint_pairs` 的 current-turn 语义；
- R1 的 persisted query anchor。

目前尚未定位到唯一函数；没有新增价格存储，也没有采用 min/max 合并。

`P3C3B_PRICE_STATE_TRACE.csv` 明确记录：

```text
round 2: effective_price_max=50 expected but not confirmed
         r1_constraints=[price_max=30] stale
         final=[]
```

---

## 4. 未执行项

- 三轮撤回 $50；
- 四轮 show more；
- $30 不恢复的测试；
- weight/material 不受影响的测试；
- 其他 task/user scope 隔离；
- 固定商品 A/B/C/D 的价格 eligibility 单测；
- SQLite replay/幂等 API 测试。

这些未完成，不填估计值。

---

## 5. 回答

1. **价格约束之前在哪个环节丢失？**
   - proposal current-turn kind 导致 M5 authorized constraint 被 `current_turn_kinds` 跳过，
     Eligibility 未执行。
2. **修复后是否真正进入最终资格检查？**
   - 第一轮是（`hard_constraint_evaluated=true`）；
   - 第二轮否（R1 仍是旧 $30）。
3. **预算更新和撤回是否正确？**
   - 更新：**未正确**；
   - 撤回：**未执行/未验证**。
4. **是否错误接受价格 UNKNOWN？**
   - 第一轮没有：UNKNOWN 计为 unresolved，未进入 3 个 known-price MATCH 商品。
5. **默认 Serving V1 是否完全不变？**
   - 是；未修改 `/v1/recommend`。

---

## 6. 交付

- `P3C3B_PRICE_LIFECYCLE_REPORT.md`
- `P3C3B_PRICE_STATE_TRACE.csv`
- `P3C3B_API_TRAJECTORY.jsonl`（STUB，两轮 + 失败路径；不是四轮完成轨迹）
- `P3C3B_SOURCE_DIFF.txt`

未启动 GPU、未训练、未运行完整 Benchmark、未修改 Serving V1/Gold/Evaluator、未 RL、
未 commit/push/deploy。

**本阶段不能宣称价格生命周期闭环完成。**
