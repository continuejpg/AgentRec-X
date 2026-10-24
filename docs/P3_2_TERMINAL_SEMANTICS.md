# Phase 3.2 — 无可行商品场景的终态语义

> 目标：在没有可验证、可推荐商品时，让内部状态、API 状态与用户消息采用一致且可区分的语义，
> 既不把业务性无结果误报为执行失败，也不把"未能核实"伪装成成功。
>
> 结论：**需要一处最小修改**——完成守卫的 `no_verified_compliant_candidate` 拒绝改用专属
> 终态原因 `no_verified_candidate`；状态仍为 `failed`、`succeeded=false`、不展示任何未验证商品，
> 守卫本身、候选、约束、最终列表均未改动。

## 0. TL;DR

- **原语义**：`completion_refused` 同时覆盖了三类完全不同的情况：
  1. 无 verified 候选但仍有 unresolved 候选（"无法核实"）；
  2. 候选集未通过 grounded 校验 / 验证被拒（完整性失败）；
  3. 最近一次执行失败（`last_execution_failed`）。
  只有 `termination_detail` 的前缀能区分，`termination_reason` 这个稳定字段不能。
- **修复**：新增 `TerminationReason.NO_VERIFIED_CANDIDATE = "no_verified_candidate"`，仅在守卫
  代码为 `no_verified_compliant_candidate` 时使用；其余拒绝仍为 `completion_refused`。
- **没有放宽任何东西**：守卫仍拒绝完成，状态仍是 `failed`，`succeeded=false`，
  `feasible=0` 时依然不展示任何候选；没有伪造 verified 候选、没有放宽约束、没有强制完成。
- **48 场景**：44 个 `completed`（38 个非空 + 6 个全不符合的空结果），4 个
  `no_verified_candidate`（原为 `completion_refused`）；**其余字段逐场景 0 变化**，
  空约束解释仍 10/10，硬违反 0/59，身份 100%，消息一致率 38/38，事实 354/354。
- 定向测试 **157 passed**（含 7 个新增终态语义测试）。

---

## 1. A — 保护现状

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/agent-source-limit-ablation` / `9eabbfe` |
| 已存在改动 | Phase 3 / 3.1 的 `pipeline.py`、`rendering.py`、`agent_service.py`；2C 与 P3 的全部未提交文件 |
| 本轮新增改动 | `control/schemas.py`、`control/completion.py`、`control/loop.py`、`demo/observation.py`、`demo/agent_chat_adapter.py`、`evaluation/agent/runner.py`；`tests/test_phase32_terminal_semantics.py`；本文档 |
| 未做 | 没有覆盖/删除/提交任何现有文件；没有 push/merge；没有改冻结评测资产 |

---

## 2. B — 当前状态语义与调用链

### 2.1 调用链

```
policy 提出 FINISH
  └─ CompletionGuard.check(...)                       # 唯一允许结束的闸门
       ├─ execution_failed            → 拒绝 last_execution_failed
       ├─ produced & !grounded        → 拒绝 candidates_not_grounded
       ├─ produced & grounded & 约束生效 & feasible==0 & unresolved>0
       │                               → 拒绝 no_verified_compliant_candidate
       └─ 其余                        → 接受 completion_accepted
  └─ LoopController.abort_completion()
       └─ _terminate(status, termination_reason, "code: detail")
  └─ build_terminal_view()  → API terminal.status / termination_reason / detail / succeeded
  └─ agent_outcome()        → chat 层 outcome（recommended / refused / failed / ...）
  └─ observation.decide_replan() → BLOCKING_TERMINATIONS
  └─ evaluation.runner._terminal_outcome() → TerminalOutcome
```

### 2.2 四个问题

**1. `completion_refused` 当前表示什么？**
"完成守卫拒绝了结束提议，且没有可重试的预算/状态"。它是一个**拒绝族**，不是单一原因：
`no_verified_compliant_candidate`、`candidates_not_grounded`、`last_verification_refused`、
`last_execution_failed` 都会落到这里。

**2. 无 verified 但存在 unresolved 时为什么拒绝？**
守卫的判据是 `feasible == 0 and unresolved > 0`。一个"未决"候选的缺失事实一旦被读到，仍可能
变成合格；现在结束就等于**放弃核实**，把"还不知道"当成"没有"。这是刻意 fail-closed，不是 bug。

**3. 正确的空约束解释在哪生成？**
- 全部候选**已被证伪**（`unresolved == 0`）：守卫接受完成，`pipeline.finalize_stage` 在
  `constraint_projection.is_empty` 时渲染约束解释 → `status=finished, succeeded=true`。
- 存在 unresolved（守卫拒绝）：`agent_service._execute_plan` 在 `message` 为空且投影为空时，
  用同一个 `build_constraint_empty_response` 填充 `message` → `status=failed`。
  用户看到的文字相同，但终态不同。

**4. 内部/API/消息是否一致？**
消息一致（都说明约束），但 `termination_reason` 不一致：第 3 类（无法核实）与第 4 类
（完整性/执行失败）同为 `completion_refused`，只能靠 `termination_detail` 的字符串前缀区分。

### 2.3 修改前的四类结果

| 类别 | status | reason | succeeded | 列表 |
|---|---|---|---|---|
| 1. 成功推荐 | `finished` | `completed` | true | 非空 |
| 2. 已处理、无匹配 | `finished` | `completed` | true | 空 |
| 3. 需要进一步核实 | `failed` | `completion_refused` | false | 空 |
| 4. 执行失败 / 安全拒绝 | `failed` | `execution_failed` / `invalid_action` / `completion_refused`（完整性） | false | 空 |

类别 2 与 4 可区分；类别 3 与 4 在 `reason` 字段上不可区分。

---

## 3. C — 最小修复与兼容性

### 3.1 改动

| 文件 | 改动 |
|---|---|
| `recommendation/control/schemas.py` | 新增 `TerminationReason.NO_VERIFIED_CANDIDATE = "no_verified_candidate"` |
| `recommendation/control/completion.py` | 新增 `CompletionGuard.termination_reason(refusal)`：仅 `no_verified_compliant_candidate` → 新原因，其余 → `COMPLETION_REFUSED`；`terminal_status` 仍为 `FAILED` |
| `recommendation/control/loop.py` | `abort_completion` 使用守卫给出的 reason（不再硬编码 `COMPLETION_REFUSED`） |
| `recommendation/demo/observation.py` | `BLOCKING_TERMINATIONS` 加入新原因（replan 仍不可绕过） |
| `recommendation/demo/agent_chat_adapter.py` | 新原因在 chat 层仍映射为 `refused`（wire outcome 词表不变） |
| `recommendation/evaluation/agent/runner.py` | 新原因在评测词表仍映射为 `COMPLETION_REFUSED`（评测粒度不变） |

守卫的判定逻辑、`terminal_status`、状态枚举 `RunStatus`、`succeeded` 语义都未动；没有新增成功
路径，也没有新增候选或放宽约束。

### 3.2 兼容性影响

- `terminal.termination_reason` 在 API 里是字符串；新值是**追加**，旧值不变。严格按枚举白名单
  校验的客户端需要接受新值；本仓库内的消费方已全部更新。
- chat 的 `outcome` 词表不变（仍为 `refused`），所以前端行为不变。
- 评测 `TerminalOutcome` 词表不变（仍为 `COMPLETION_REFUSED`），历史用例不受影响。
- replan 门仍把新原因视为阻塞；不存在"通过 replan 绕过核实"的路径。

### 3.3 为什么不是"直接改成成功"

类别 3 的语义是"还没有证明任何候选合格"，不是"已证明没有合格候选"。把它改成
`finished/succeeded=true` 会把未核实包装成成功，违反 fail-closed；因此保持 `failed`，
只增加专属原因。类别 2（全部已证伪）本来就是 `finished/completed`，保持不变。

---

## 4. D — 回归验证

### 4.1 定向测试

新增 `tests/test_phase32_terminal_semantics.py`（7 项）：原因映射、其它拒绝仍为
`completion_refused`、新原因仍 `FAILED`、与成功/执行失败不同、chat 仍 `refused`、replan 仍阻塞、
评测词表仍为 `COMPLETION_REFUSED`。

```
pytest -q tests/test_phase32_terminal_semantics.py \
         tests/test_constraint_eligibility.py \
         tests/test_agent_chat_adapter.py \
         tests/test_agent_control_plane.py \
         tests/test_v2_observation_replan_gate.py \
         tests/test_demo_agent_api.py
# 157 passed
```

### 4.2 48 场景前后对比

| 指标 | v31（3.1 后） | v32（3.2 后） |
|---|---|---|
| `completed` | 44 | 44 |
| `no_verified_candidate` | 0 | **4** |
| `completion_refused` | 4 | 0 |
| 空约束解释正确 | 10/10 | **10/10** |
| 消息与最终列表一致 | 38/38 | **38/38** |
| 事实可追溯 | 354/354 | **354/354** |
| 硬约束违反 | 0/59 | **0/59** |
| 未决事实被展示 | 0/59 | **0/59** |
| 身份合法率 | 100% | **100%** |
| 期望结果命中 | 40/40 | **40/40** |
| 软偏好 Top-5 | 17.5% | **17.5%** |

逐场景比较：**4 行**的 `termination_reason` 从 `completion_refused` 变为
`no_verified_candidate`（h12000/h12004 价格 0.01，h12002/h12006 不可验证约束）；
presented / 消息 / 事实 / 硬约束 / 身份 / 软偏好 **0 变化**。

4 个新原因场景仍然是：`status=failed`、`succeeded=false`、空列表、约束解释文本；
它们现在与"执行失败"在 `termination_reason` 上可区分。

### 4.3 真正的失败与安全拒绝

48 个场景中没有执行失败或安全拒绝，因此这部分用定向测试覆盖：
`last_execution_failed` / `candidates_not_grounded` / `last_verification_refused` 仍是
`completion_refused`；`execution_failed` 与 `invalid_action` 的路径未改。

---

## 5. 尚未解决 / 下一步

1. **未决场景的消息措辞**：Phase 3.3 已修复——4 条消息改为
   "No candidate could be verified as satisfying the active constraints ..."，不再声称
   "No candidate satisfies"；见 [`docs/P3_3_EMPTY_MESSAGE_SEMANTICS.md`](P3_3_EMPTY_MESSAGE_SEMANTICS.md)。
2. **`completion_refused` 仍是一个拒绝族**：`candidates_not_grounded` / `last_verification_refused` /
   `last_execution_failed` 仍是同一原因，靠 detail 前缀区分。若未来需要更细的稳定枚举，可按
   guard code 再拆；本轮只拆了与"无法核实"最相关的一类。
3. **不变的成功/失败语义**：类别 2（全部证伪）仍是 `finished/completed`；类别 3 仍是 `failed`。
   没有新增"业务性无结果"的成功状态——这是刻意的。
4. **仅 48 个构造场景**，不是线上用户效果；未跑全量测试。
5. **未改动** RRF、Learned Ranker、候选深度、查询构造、来源计划、偏好排序、消息格式统一、
   价格解释、场景集扩展。

---

## 6. 复现与产物

```bash
# 48 场景（3.2 后）
.venv/bin/python -m experiments.p3_scenario_eval --out runs/p3_scenario_eval_v32

# 终态语义定向测试
.venv/bin/python -m pytest -q tests/test_phase32_terminal_semantics.py \
    tests/test_constraint_eligibility.py tests/test_agent_chat_adapter.py \
    tests/test_agent_control_plane.py tests/test_v2_observation_replan_gate.py \
    tests/test_demo_agent_api.py
```

| 文件 | SHA-256 |
|---|---|
| `runs/p3_scenario_eval_v32/config.json` | `f25c6c2f676df1a3097f66771c11242b016d2a7d54669e4b1591f4d6e6092ae2` |
| `runs/p3_scenario_eval_v32/cases.json` | `b3bf3fae8ef7c3fb93f3e0f6dfb0c9636f18c564908dfc5283bb03a593338fb5` |
| `runs/p3_scenario_eval_v32/samples.jsonl` | `ed9d65430a6db6b56b7a3105113dc641cfe8affefde14d7e348c625772395215` |
| `runs/p3_scenario_eval_v32/summary.json` | `412701f501c1d7e7f1100d5a13ffbedb27e3af4e4e855dace520bf13dd09274a` |
| `tests/test_phase32_terminal_semantics.py` | `c06918f191e0bbe27ec733724bb9a99894e7642645c7e4c71c6a577194a65dc5` |

对照：`runs/p3_scenario_eval_v31/`（3.1 基线）；驱动未变（`414c1a1e…`），
所以 v32 的 `samples.jsonl` 逐行带有新 `termination_reason`，其它字段与 v31 相同。
