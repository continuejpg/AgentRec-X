# P3C-3 M5 Hard Constraint 修复报告

## 1. 修复内容

问题：M5 `hard_constraint_count=1 (price_max)`，但候选 trace 的
`hard_constraint_evaluated=false`、R1 constraints=[]。

### 源码修改

`recommendation/demo/agent_service.py::run_shopping_turn`

在 `_integrate_state_turn` 之后增加 proposal 当前硬约束的防御性注入：

- 只对有效的 proposal constraints；
- 按 kind 替换旧值，不追加两个 price_max；
- 复用现有 `AgentConstraintItem` / `CandidateEligibilityEvaluator`，没有新建价格状态系统；
- 不影响默认路径（无 proposal 时分支不执行）。

相关修复：
- 移除 proposal 的 constraint kind 被误当作“当前 trusted constraint 已提供”而跳过 M5
  authorized constraint 的逻辑；
- 当前 proposal 的 price_max 覆盖旧的同 kind 值。

---

## 2. 修复效果

Stub API 第一轮（测试替身，不是 Qwen）：

```text
proposal: set_goal + update_constraint(30) + add_preference(weight)
M5: hard_constraint_count=1
before: hard_constraint_evaluated=false
after : hard_constraint_evaluated=true
        eligible=10, ineligible=1, unresolved=4
R1 constraints=[price_max=30]
final API = 3 MATCH 商品
```

因此价格硬约束已经真正进入最终 eligibility 检查；价格未知候选被计入 **UNRESOLVED**，没有被
标记为 MATCH。

商品级证据：

| parent_asin | item_id | price | price_max | price eligibility | role |
|---|---:|---:|---:|---|---|
| B00CBR1UI6 | 38577 | 16.95 | 30 | MATCH | MATCH |
| B08XJQ9HPX | 123918 | 29.89 | 30 | MATCH | MATCH |
| B09Y53NYDK | 136697 | 19.99 | 30 | MATCH | MATCH |
| B00LCWB5UY | 48441 | unknown | 30 | **UNKNOWN (not eligible)** | UNKNOWN |

详细：`P3C3_PRICE_ELIGIBILITY_BEFORE_AFTER.csv`。

---

## 3. 仍未解决：预算更新

Stub 第二轮 proposal：

```text
update_constraint price_max=50
retract_preference weight
```

M5 状态是 hard count=1，但当前 R1 diagnostics 仍显示：

```text
R1 constraints = [price_max=30]
final_ids = []
```

第二轮候选 trace 的 hard eligibility 不再是 false，但 R1 的当前约束仍是旧 $30，且最终为空。
因此：

- **第一轮 $30 约束已修复并生效**；
- **第二轮 $50 更新没有正确进入 R1 当前约束**；
- 预算撤回未在本轮 API 中测试；
- 旧 superseded 数值是否复活未验证；
- 最终空列表的原因仍包含目标/角色或旧约束，未完全定位。

不能声称预算更新/撤回已经正确。

---

## 4. 测试

本阶段使用真实 Shopping API + Stub HTTP server 验收（不是 Qwen）：

- 第一轮：HTTP 200，hard filter evaluated=true；
- 模型不可用：HTTP 503，M5 event delta=0；
- 未运行完整 Benchmark。

未新增大规模 smoke；没有发现会错误接受 price UNKNOWN：候选 trace 中 price unknown 计为
UNRESOLVED，未出现在最终 API 的 3 个 MATCH 商品中。

---

## 5. 回答

1. **价格约束之前丢在哪？**
   - 在 proposal 当前约束进入 `_integrate_state_turn`/R1 的合并阶段：proposal 的 constraint
     kind 被当作“当前 turn 已提供”，导致 M5 authorized constraint 被跳过，最终 eligibility
     未评估。
2. **修复后是否进入最终资格检查？**
   - 第一轮是：`hard_constraint_evaluated=true`、eligible/ineligible/unresolved 有实际计数。
   - 第二轮仍否：R1 constraints 仍为旧 30，最终为空。
3. **预算更新/撤回是否正确？**
   - 更新：**未完全正确**（第二轮 50 未进入 R1 当前约束）。
   - 撤回：本阶段 Stub API 未覆盖，仍为 UNVERIFIED。
4. **是否错误接受价格 UNKNOWN？**
   - 第一轮没有：UNKNOWN 计入 unresolved，最终 API 只返回 3 个 known-price MATCH 商品。
5. **默认 Serving V1 是否完全不变？**
   - 是；仅修改了 proposal/experimental Agent 分支，未改 `/v1/recommend`。

---

## 6. 交付

- `P3C3_HARD_CONSTRAINT_FIX_REPORT.md`
- `P3C3_PRICE_ELIGIBILITY_BEFORE_AFTER.csv`
- `P3C3_SOURCE_DIFF.txt`
- Stub 轨迹：`P3C2B_AGENT_API_TRAJECTORY.jsonl`（STUB，非真实 Qwen）

未启动 Qwen GPU、未重新训练、未运行完整 Benchmark、未修改 Serving V1/Gold/Evaluator、
未开始 RL、未 commit/push/deploy。
