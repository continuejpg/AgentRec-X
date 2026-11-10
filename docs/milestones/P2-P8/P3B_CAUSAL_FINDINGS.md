# P3B 因果发现

## 0. 结论摘要

- 默认 Agent **M5/R1/soft-M5 关闭**；这些模块只有在显式环境配置下才真实运行。
- 受控 6 场景显示：预算更新、LLM 撤回、LLM 角色切换可以被 M5/R1 正确消费；但规则撤回、规则角色切换复用候选、配件→主商品空候选、M6.1 未接收软偏好仍存在。
- 现有 20 dev 任务的 V2 对 LLM 增益不敏感：D1 有效 proposal 52 个、M5 events 38 个，但 C vs D1 只有 3/20 最终商品集合变化，V2 PASS 集合完全相同。

---

## 1. 52 Accepted → 38 M5 Events 的原因

D1 的 52 个 accepted operations 中：

```text
noop               13
clarify             1
set_goal           25
update_goal         9
update_constraint   2
add_preference      1
retract_preference  1
```

`noop` 和 `clarify` 是控制操作，不产生 M5 状态事件；其余 38 个 operation
对应 38 个 M5 events。**这不是事件丢失，而是 14 个 control operations 本来就不写入状态。**

“38 events”仍只是**事件写入数**，不等于 38 次有效状态变化；事件可能被 reducer supersede 或
被后续目标覆盖。现有 20 任务轨迹没有逐事件保存 `state_digest` 前后值，因此**有效状态变化
的精确次数 = UNKNOWN**，不能把 38 当作有效变更数。

---

## 2. 受控 6 场景结果

完整数据：`P3B_CONTROLLED_INTERVENTIONS.csv`。

| 场景 | RULE_APPLIED | LLM_APPLIED | 判断 |
|---|---|---|---|
| 预算 30→50 | RETRIEVE，hard constraint 保留 1，最终变化 | RETRIEVE，最终变化 | 状态被消费；PASS |
| 撤回预算 | **retraction_match resolved=0，hard constraint 仍为 1** | resolved=1，hard 0，最终变化 | 规则撤回阻断；LLM 路径正常 |
| 主→配件 | **R1=RERANK，最终仍为主商品** | RETRIEVE，最终切换为配件 | 规则路径候选复用未失效；LLM 正常 |
| 配件→主 | 最终为空 | 最终为空 | 状态/R1 变化，但候选缺失；不是状态未消费 |
| 确认轻量偏好 | 最终变化；soft diagnostics `None` | 最终变化；soft diagnostics `None` | M6.1 soft 未实际消费 |
| 撤回轻量偏好 | retraction_match resolved=0，feature 仍存在 | resolved=1，feature=0，最终变化 | 规则撤回阻断；LLM 正常 |

### 2.1 规则撤回预算

RULE_APPLIED `retract_budget`：

```text
retraction_match.requested = 1
retraction_match.resolved = 0
unmatched_value_present = True
hard_constraint_count = 1
```

因此规则路径的 budget retraction **没有改变 effective hard constraint**；最终为空不是
“正确消费后的合理结果”，而是状态仍保留了旧约束。

### 2.2 规则 main→accessory

RULE_APPLIED：

```text
r1.action = RERANK
r1.reason = previous_candidates_still_satisfy_request
final_ids = 与 initial 相同（主商品）
```

role 目标/状态变化后，R1 仍复用旧主商品候选窗口，没有让候选失效。LLM_APPLIED 同场景
走 RETRIEVE 并返回配件，说明该阻断至少部分是规则路径的候选复用问题。

### 2.3 配件→主商品空候选

RULE 和 LLM 都返回 `final_ids=[]`；R1 有 `grounded_role_target_changed`，但没有足够
可用的主商品候选。这是**候选/目录覆盖不足**，不是 M5 状态没有写入。

### 2.4 轻量偏好与 M6.1

`confirm_weight` 中 R1 发生 RETRIEVE，最终商品变化，但
`soft_rerank_diagnostics=None`（没有调用 M6.1 soft rerank）或 `rerank_applied=False`。
说明“M5 有软偏好 + soft 开关打开”仍不足以证明 M6.1 真正消费了该软偏好。

---

## 3. 现有 20 任务为什么对 LLM 增益不敏感

使用 `P3_TASK_CAUSAL_TRACE.csv` / `P3_BOTTLENECK_METRICS.json`：

- D1 52 accepted，38 M5 events；D1 所有 20 个任务都有 proposal events；
- C vs D0 最终商品变化 2/20；
- C vs D1 最终商品变化 3/20；
- D0 vs D1 1/20；
- D1 vs D1-v2 1/20；
- 四组 V2 PASS 集合完全相同（12/5/3）；
- 失败任务集合相同，且 lifecycle/role/retrieval component 也相同。

解释：

1. 默认 M5/R1/soft 关闭，所以“默认 Agent”对 proposal 不敏感；
2. 即使 benchmark 开启 M5，规则 extractor/intent 仍在 R1 中占主导；
3. 状态变化较弱：controlled 显示只有部分场景被 R1/M6.1 消费；
4. 失败主要落在候选缺失、角色/证据、M5 lifecycle，而不是 Parser；
5. V2 是角色/lifecycle/evidence 的离散 rubric，对小幅商品身份变化不敏感。

不能把“同分”解释为“LLM 无效”；更准确是：当前 dev 任务与状态消费链路没有给
Parser 增益留下可观测空间。

---

## 4. 是否适合接入可选 Qwen Agent API

**暂不适合直接接入生产默认 API。**

原因（代码事实）：

- 默认 `/sessions/.../agent/shopping` 不传 `proposal_operations`；
- 默认 M5/R1/soft 均关闭；
- 规则 extractor 仍负责 intent/R1；
- R1 的旧候选复用和生命周期失败尚存在；
- D0 真实无效 JSON 必须保留为失败，不能静默 fallback。

可以做一个**实验性 route/flag**，但前提是：显式 profile、proposal 仅作为 M5 输入、
规则解析来源分离、失败可见、soft-M5 消费验证。当前代码没有该安全入口。

## 5. 是否具备学习型 Planner 的价值空间

**有限，且不应先做 RL。**

- controlled 预算/撤回/主→配件场景中，LLM 路径的 R1 RETRIEVE 确实改变了状态/候选；
- 但配件→主为空、规则撤回失败、soft-M6.1 未消费说明当前瓶颈在 **候选/证据/状态消费**；
- 在这些问题修复前，Planner 的替代动作难以带来可信 reward；
- 没有发现“同一状态/预算下，多步动作选择”是当前主要增益来源。

因此：

- 不建议把当前结果解释为 Planner 需要 RL；
- 应先修通用状态/候选消费，再做学习型 Planner 评估；
- Agent RL 仍只有 toy contract，不是真实训练。

---

## 6. 直接回答

1. **默认关闭的模块在实验配置中是否真实启用？**
   - 是；controlled 中 DEFAULT off，RULE_APPLIED/LLM_APPLIED 的 M5/R1/soft 有真实诊断。
2. **M5 有效状态在哪些路径被消费？**
   - controlled 的预算更新、LLM 撤回、LLM 主→配件、确认/撤回软偏好有 R1/最终商品变化；
   - M6.1 没有稳定消费 soft preference。
3. **哪些状态变化没有正确传递？**
   - 规则 budget retraction（resolved=0）；
   - 规则 main→accessory（R1 RERANK 复用旧候选）；
   - M6.1 soft weight preference；
   - accessory→main 为空候选（状态已变，候选不足）。
4. **哪些最终推荐不变化是合理的？**
   - 撤回/拒绝不需要改变商品的场景（例如 noop/decline）；
   - 状态变化但当前候选仍满足需求，R1 合理 RERANK；
   - 但 RULE main→accessory 保持主商品不是合理的，因为目标角色已变。
5. **20 任务为什么不敏感？**
   - 四组 pass 集合相同、最终身份仅 1–3 个任务变化；失败由 role/lifecycle/candidate 主导；
   - 状态消费路径弱 + 候选/证据限制，不是 Parser 分数问题。
6. **是否适合接入可选 Qwen Agent API？**
   - 暂不适合默认 API；可做显式实验 profile，但需修 proposal 传递/状态消费/可见失败。
7. **Planner 是否有学习/RL 价值空间？**
   - 当前证据不足；主要瓶颈在状态/候选/证据，不应先用 RL 优化 Planner。
