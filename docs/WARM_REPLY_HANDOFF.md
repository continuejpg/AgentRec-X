# Warm / Cold Reply 闭环交接单（WARM_REPLY_HANDOFF）

> 用途：M1-C 持久化 Web Session 的 **Warm Clarification Reply 闭环** + M1-D S6 的
> **CPU/BM25 Cold Resume 闭环**均已实现并验证；新会话从这里接手。
> 最后更新：2026-11-21（M1-D S6 结束时）

## 0. 环境与仓库

| 项 | 值 |
|---|---|
| 仓库根（Windows） | `D:\IT\CODE\PROJECTION\agentrec\X` |
| 仓库根（WSL） | `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| Python | 项目 `.venv`（Linux/WSL）`.venv/bin/python`，Python 3.10 |
| 测试前缀 | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest ... -q -p no:cacheprovider` |
| Git | 在 Windows PowerShell 操作同一目录；**未 Push** |

## 1. Git 基线

- M1-C 冲刺**接手时** HEAD：`cf3a465`（`refactor: separate recommendation snapshot construction from session writes`）。
  用户提示的 `762e636` 是它的父提交（`cf3a465 → 762e636 → 32f7770`），仓库实际多 2 个提交。
- M1-C 结束时：本地 Commit `3643ffc`（`feat: complete durable warm clarification reply`），未 Push。
- M1-D S6 接手时 HEAD：`3643ffc`（与任务提示一致）；M1-D 结束本地 Commit
  `b1165b5`（`feat: restore durable clarification after MCP restart`），未 Push。
- M1-E 接手时 HEAD：`b1165b5`（与任务提示一致）；本轮结束会再产生一个本地 Commit
  （`feat: merge task requirements and enforce price budgets`），仍未 Push。
- M1-C 提交前的 Coordinator 原型（`state_store.py` / `reply_executor.py` /
  `test_web_reply_executor.py`）已随 `3643ffc` 入库，没有被删除或改写成简化版。

## 2. 已具备且已验证的能力

### 2.1 Web State（SQLite v6）

`web_demo/state_store.py`：`meta/web_session/web_identity/web_task/web_pending/web_turn/web_request`；
v6 只新增 `web_pending.continuation_json TEXT`（Cold Resume 快照；旧行为 NULL，仍按 409 处理）。
Request Ledger：`claim_reply_request`、`renew_request_lease`、`mark_request_external_started`、
`checkpoint_reply_result`、`mark_request_ambiguous`、`finalize_reply_request`、`get_request`；
执行阶段 `CLAIMED → EXTERNAL_STARTED → RESULT_DURABLE → COMPLETED`，`AMBIGUOUS` fail-closed。

本冲刺**唯一新增**的 store API：

```python
get_pending_for_clarification(session_id, clarification_id) -> PendingRecord | None
```

只读；让回复路由在 Pending 仍 OPEN 或已被 CONSUMED（重放）时都能拿到 `pending_id` / `reply_turn_id`。
所有 claim/lease/fencing/reset-generation 规则保持原样。

### 2.2 M9 作用域（已修复，未改）

`recommendation/skills/clarification.py::resume_with_user_reply(..., user_key=...)`；
`recommendation/mcp/server.py::submit_clarification_reply` 显式校验 Session Key 归属。
真实 MCP 测试：`tests/test_mcp_session_memory_scope.py`(3)。

### 2.3 真实推荐契约（`web_demo/reply_result_contract.py`）

schema `agentrecx.web.reply_turn.v1`。本次做一个**向后兼容的兼容性修正**：
商品 item 的标识接受 `product_id`（BFF 卡片真实字段）或 `item_id`（MCP selected_product 字段）；
`_selected_card` 同时带两者，保证从 SQLite 恢复时 behavior snapshot 能重建。
原有“必须有非空标识”的约束不变（`tests/test_web_real_recommendation_contract.py`(7) 全绿）。

### 2.4 Handle 生命周期证明（轮 A，`web_demo/handle_lifecycle.py`）

- `HandleRef(profile_id, session_id, clarification_id)`；
- `HandleTicket`：某子进程签发 Handle 的凭证（manager 实例 ID + 子进程实例 ID + 签发/绝对过期时刻 + 工具）；
- `ProcessIdentity`：当前服务该 profile 的子进程身份；
- `verify_handle(ref, ticket, identity, now=...)`：固定顺序
  `handle_missing → session_mismatch → process_unavailable → stale_process → process_not_ready → expired`；
- `issued_handles_in_response(...)`：从真实 MCP 结果提取 `(clarification_id, expires_in_seconds)`，
  覆盖 `request_preference_clarification` 顶层字段与 `run_agent_turn` 的嵌套 `clarification`；
  **没有正数 TTL 的 Handle 一律不记录**。

`web_demo/mcp_manager.py`：`MCPManager.instance_id`、每个 `ServerRuntime.process_instance_id`/`generation`
（每次 start/restart 重新生成）；`call_tool` 成功返回后记录 Handle，绑定**调用开始前**捕获的进程 ID；
绝对过期从请求开始计时；`check_handle(ref)` 是纯状态检查（零 MCP/零 SQLite/零 M9）。
未新增 MCP 工具，工具表未改动。

### 2.5 Reply Assembler（轮 B，`web_demo/reply_assembly.py`）

纯函数 `assemble_reply_presentation(request, *, catalogue_size)`：

- 输入真实 MCP 载荷：`recommend_products`、`get_product_info`、`submit_clarification_reply`、
  `get_active_preferences`；
- 输出 `products`（含 `recommendation_id` / `request_id` / `ranker_version`）、`narrative`、
  `preferences`、`memory_update`；
- 只调用现有 BFF 视图函数（`vm.product_card` / `vm.products_payload` / `vm.preference_state` /
  `vm.memory_update_payload` / `vm.narrative_memory`），不写 Session、不调 MCP、不改输入；
- 选中商品缺失/不完整、非 `finalized` 推荐 → `ReplyAssemblyError`，绝不产生空推荐；
- `new_recommendation_id()` 沿用 `rec-{uuid12}` 规则，由调用方签发一次并持久化。

`tests/test_web_reply_assembly.py` 里有一条**逐字段等价**测试：同一批真实形状载荷下，
Assembler 输出与内存模式 `POST /api/clarification/reply` 的商品/文案/`memory_update`/`preferences`
逐字段相等（仅规范化随机 ID 与时间戳）。

### 2.6 持久化 Warm Reply 执行链（轮 C）

`web_demo/demo_flow.py::clarification_reply` 现在分流：

- **内存模式**：原路径完全不变；
- **持久化模式** → `_persistent_clarification_reply`：
  按 `clarification_id` 从 SQLite 找 Pending（OPEN 或 CONSUMED）→ 只支持 `mode="cpu_live"`
  （qwen_live 仍 409）→ `WarmReplyExecutor`。

`web_demo/reply_executor.py` 顺序：

```
Ledger first: COMPLETED -> replay | RESULT_DURABLE -> checkpoint finalize |
              EXTERNAL_STARTED/AMBIGUOUS -> busy/ambiguous (never re-execute)
otherwise: resolve handle (warm proof, else Cold Resume) -> claim -> heartbeat
-> EXTERNAL_STARTED -> submit_clarification_reply -> get_active_preferences
-> recommend_products -> get_product_info(selected) -> assemble -> build_reply_payload
-> checkpoint(RESULT_DURABLE) -> finalize(COMMIT) -> HTTP 200
```

- Ledger 状态优先：COMPLETED 重放、RESULT_DURABLE 只读检查点 Finalize、
  EXTERNAL_STARTED/AMBIGUOUS 直接 busy/ambiguous —— 这几种都**不需要** Handle 或 Cold Resume；
- 只有 “尚无 Request Ledger 行” 的新尝试才解析 Handle；
- `handle_check`、`cold_resume`、`assemble` 都是构造注入，缺 checker 时 fail-closed，缺 assembler 直接 TypeError；
- `EXTERNAL_STARTED` 之后的超时/异常 → `AMBIGUOUS`（503），永不自动重跑；
- Finalize 单事务写 Task/Pending/Request/Reply Turn；**提交后**才镜像到内存 Session 并返回 200；
- Reply Turn 使用 ASK 时签发的 `pending.reply_turn_id`；
- 持久化结果形状：`{reply: <contract payload>, preferences, memory_update, steps}`；
  `turn_from_result` 会把 `memory_update`/`tool_activity` 一并投影进 `web_turn`，
  重启后 GET 的回复 Turn 与在线响应一致。

### 2.7 HTTP 行为与错误码

成功/重放 200（响应 = 持久化结果重建，完全一致）；其余一律非 200：

| outcome | HTTP | code |
|---|---|---|
| Handle 不成立（含 Cold Resume 不可用） / 非 cpu_live | 409 | `clarification_resume_not_ready` |
| 另一执行者持有 lease | 409 | `clarification_reply_in_progress` |
| 同 request 不同内容 | 409 | `clarification_reply_conflict` |
| Reset/任务版本/租约 fencing | 409 | `clarification_reply_fenced` |
| `EXTERNAL_STARTED` 后不确定（超时/读写失败） | 503 | `clarification_reply_ambiguous` |
| 检查点不可持久化 | 503 | `clarification_result_not_durable` |
| 无匹配 Pending / CANCELLED / EXPIRED | 409 | `clarification_not_pending` |

`MCPManagerError` 的现有处理器负责状态码映射；`app.py` 未改动。
Cold Resume 的失败细节在 `detail.handle`：`continuation_missing` / `continuation_expired` /
`continuation_unsupported_*` / `continuation_session_mismatch` / `continuation_task_mismatch` /
`continuation_generation_mismatch` / `continuation_malformed` / `cold_resume_failed`。

### 2.8 Cold Resume（M1-D S6，CPU/BM25 单轮）

**持久化快照**（`web_demo/continuation.py`，`agentrecx.web.clarification_continuation.v1`）：

```
schema/version/mode/profile_id/session_id/web_task_id/pending_id/clarification_id/
reset_generation/query/question/reason/expires_at
mcp: {schema: agentrecx.mcp.clarification_continuation.v1, version, mode, user_key,
      session_id, clarification_id, question, reason, current_query,
      task: <MCP 进程 ASK 时使用的完整 task dict>, task_constraints, created_at, expires_at}
```

- ASK 响应新增 `continuation` 块（JSON-only；**不含** handle 对象、进程对象、闭包或内存地址）；
- `commit_ask(..., continuation=...)` 在同一个事务写入 `web_pending.continuation_json`，
  并用已提交的行值**强制重写** Web 绑定字段（session/profile/task/pending/clarification/mode/
  generation/expires_at），调用方无法伪造绑定；
- 旧行（v5 及更早 / 无快照）→ `continuation=None` → 一律 409，不猜测。

**新进程恢复**（`recommendation/mcp/server.py::restore_clarification_handle`）：

- 只读恢复：校验 schema/version/mode → 本进程 trusted user → session_id → created/expires →
  快照 task **必须等于**本进程 `ctx.current_task(current_query)`（fixture 漂移即拒绝）→
  重建 `ClarificationState` 并签发**新** `clarification_id`；
- 不调用 ask skill、不写 M9、不推荐；旧 handle id 永不复用；
- 返回 `clarification_id` + `expires_in_seconds`，父进程经理据此记录新 Handle 票据。

**BFF 顺序**（`web_demo/reply_executor.py::_resolve_handle` + `demo_flow._cold_resume_handle`）：

1. 先做 warm `check_handle`；通过则照旧；
2. 仅当失败码 ∈ {`handle_missing`, `stale_process`, `expired`} 且存在快照时才尝试 Cold Resume；
3. 先做 BFF 侧快照校验（session/task/pending/clarification/generation/expiry/query/user），
   **校验与 restore 都在 claim 之前**；restore 无业务副作用，失败时零 claim、零 M9、Pending 不变；
4. restore 返回的新 handle 再用 `manager.check_handle` 证明，然后走原 Claim/Lease/Heartbeat/
   EXTERNAL_STARTED/真实执行链，`submit_clarification_reply` 使用**新** handle id；
5. 同一 Pending 的并发 Cold Reply 各自 restore 一个 handle，但只有一个能 Claim 成功；
   失败方的孤儿 handle 随 TTL 失效（无业务效果）。

### 2.9 Task Requirements 与预算硬约束（M1-E）

`recommendation/preference/task_requirements.py`（纯函数）：

- `extract_task_requirements(text)`：中/英文显式购物表述 → `budget`（max/min、currency、source_text）、
  `category`、`use_cases`；无匹配不猜。中文预算必须带 `元/块/人民币/RMB/CNY`（或英文 `$`）；
  `预算改成/改为/调整到` 作为“最新一次声明”；币种只从字面量取，绝不从无关符号推断。
- `TaskRequirements.merged(update)`：新 budget/category 覆盖旧的，`use_cases` 取并集，保留两条 source_text。
- `augment_query`：只追加英文 category/use-case（如 `laptop gaming`），**绝不追加预算数字**
  （BM25 会把 `8000` 当型号并污染检索）。
- `evaluate_price_budget(price_fact, budget)`：`within / over / unknown / currency_mismatch /
  not_applicable`；`historical_metadata` 价格在币种一致时可用于历史预算核对；
  缺价、缺币种、跨币种一律**不视为合规**。边界为闭区间：7999/8000 在 8000 内，8001 超。

持久化/恢复链路：

- ASK 的 continuation 新增 `mcp.task_requirements`（原查询的 5000 CNY / laptop）；
- `restore_clarification_handle` 恢复它；`submit_clarification_reply` 把新回复合并进去
  （8000 CNY / gaming，保留 laptop）并在响应返回 `task_requirements`；
- Coordinator 把合并结果作为 `recommend_products(task_requirements=...)` 参数传入；
  `recommend_products` 再从 query 提取一次并合并（直接调用也成立），用 `augment_query` 检索；
- executor 对每个候选执行 `evaluate_price_budget`：超预算候选排除；未验证候选在 Gate 之后仍被拒绝；
- `budget_unverifiable` / `budget_exceeded` 是“已知拒绝”，不是 AMBIGUOUS：Coordinator 用
  `build_refusal_payload` 持久化一条**无商品**的拒绝回复（HTTP 200、`recommendation: null`、
  `refusal: {code, reason, budget, task_requirements}`），Replay/重启恢复一致；`web_turn.items` 为空。

真实数据缺口（本部署）：

- 默认 C4 Electronics 目录只有 `item_id/category/text`；`catalogue.inspect(price)` 返回
  `unknown`（reason: “catalog_meta.jsonl has no price field”）→ 20/20 候选无价 → `budget_unverifiable`。
- 可选 Raw Meta sidecar（`data/enrichment/electronics_metadata_sidecar.jsonl`）确有数值 price，
  但 `inspect` 标为 `historical_metadata`、`currency: "USD"`、`price_kind: historical_crawl_time`、
  source `amazon_reviews_2023:raw_meta_Electronics`；用户预算 8000 CNY，仓库无汇率来源
  → `currency_mismatch`。因此本部署下不存在“8000 元预算的真实可核实商品”；系统明确拒绝展示
  未验证商品，绝不伪造推荐。

## 3. 测试状态（真实数字）

- **M1-E 全量回归（29 个文件，最终代码）：`345 passed in 781.11s`，exit code 0。**
  命令含 `tests/test_task_requirements.py`、`tests/test_budget_enforcement.py`、
  `tests/test_web_requirement_update_e2e.py`，以及 M1-C/M1-D 的全部 22 个文件。
- 预算边界/需求合并纯函数：`16 passed`（7999/8000/8001 闭区间、币种不匹配、缺价、合并覆盖）。
- executor 预算执行（含 Gate 集成，使用明确标注的测试商品价格）：
  `4 passed`（超预算排除、全超预算拒绝、跨币种拒绝、无预算行为不变）。
- M9 共享库并发建 schema 回归：`test_two_stores_can_create_the_same_fresh_schema_concurrently`
  通过（修复前两个 MCP 子进程同时建库会 `UNIQUE constraint failed: meta.key`）。
- 真实需求更新 Cold Resume E2E（`tests/test_web_requirement_update_e2e.py`）：
  在本轮最终代码上连续 2 次单跑通过，并包含在 345 中；断言 ASK 快照 5000 CNY/laptop →
  restart → 合并 8000 CNY/gaming/laptop → 推荐工具收到合并约束 → `budget_unverifiable`
  拒绝 → Replay/重启一致。
- **M1-D S6 全量回归（22 个文件）：`294 passed in 629.92s`，exit code 0。**
- 真实跨 OS 进程 Cold Resume（`tests/test_web_cold_resume_e2e.py` 的 `real_cold` 两条）：
  `2 passed in 157.78s`（PID A→B、旧 handle stale、restore 新 handle、M9、推荐、200、Replay、双重启）。
- 该文件完整跑：`12 passed`（2 真实 + 10 确定性注入：损坏/过期快照、restore 失败、
  并发、外部执行后崩溃、RESULT_DURABLE、Reset/跨 Session）。
- 持久化快照单测（`tests/test_web_continuation.py`）：`38 passed`。
- opt-in 真实 MCP 集成（`AGENTRECX_RUN_MCP_INTEGRATION=1`）：`1 passed in 32.75s`（9 工具断言）。
- M1-C 结束基线：`244 passed`；M1-D 22 文件 `294 passed`；M1-E 命令扩大到 29 文件
  （新增需求/预算/M9 竞态测试，并纳入 4 个既有相关文件），最终 `345 passed`。
- 真实 MCP 单测（`tests/test_web_handle_lifecycle.py` 的 restart/expiry 场景）：约 67s。

全量命令：

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_web_real_recommendation_contract.py \
  tests/test_web_reply_executor.py \
  tests/test_web_reply_assembly.py \
  tests/test_web_warm_reply_e2e.py \
  tests/test_web_cold_resume_e2e.py \
  tests/test_web_continuation.py \
  tests/test_web_handle_lifecycle.py \
  tests/test_web_recommendation_metadata.py \
  tests/test_web_recommendation_snapshot.py \
  tests/test_mcp_session_memory_scope.py \
  tests/test_web_reply_execution_checkpoint.py \
  tests/test_web_reply_finalize_store.py \
  tests/test_web_request_store.py \
  tests/test_web_m9_durability.py \
  tests/test_web_reply_safety.py \
  tests/test_web_ask_persistence.py \
  tests/test_web_bff_persistence.py \
  tests/test_web_task_store.py \
  tests/test_web_state_store.py \
  tests/test_web_session_identity.py \
  tests/test_web_demo_unit.py \
  tests/test_memory_store.py \
  -q -p no:cacheprovider --basetemp=/tmp/m1d_cold
```

## 4. 剩余工作与真实阻断

1. **Cold Resume 仅覆盖 CPU/BM25 单轮**（本轮范围）：`pending.mode == "cpu_live"` 且
   `recommend_products` 一次能给出结果。以下仍是明确的 409：
   - **qwen_live 持久化 Pending**：Qwen 的 handle 依赖子进程内的 planner/executor continuation，
     不是 JSON 可序列化状态；需要事件化/重放式设计，不在本轮。
   - **多轮跟随 ASK**：回复后 `recommend_products` 若再次 `awaiting_clarification`
     → 503 `clarification_reply_ambiguous`（M9 已写、结果不可最终展示、禁止自动重试）；
     需要新的 ASK/Pending + continuation 轮次。
2. **价格数据缺口（本轮真实结论）**：默认 C4 目录没有 price；可选 sidecar 的价格是 USD 历史
   抓取值；仓库没有 CNY/USD 汇率来源。因此“8000 元预算 + 真实可核实价格”在本部署中不存在，
   系统只能给出 `budget_unverifiable` 的明确拒绝。要完成真实商品推荐，需要：
   - 有 CNY 价格（或有来源的汇率转换）的目录；或
   - 用户以 USD 预算提问（且明确接受 historical crawl-time price 语义）。
3. **laptop 不在 Product Role 契约的类型表**（`recommendation/preference/product_role.py` 只有
   headphones/earbuds/keyboard/mouse/speaker）。中文查询经 `augment_query` 变成
   `... laptop gaming` 后可以检索到真实笔记本/配件，但没有 laptop 角色过滤；在预算不可验证时
   不会 finalize 任何商品，所以本轮不受影响。若未来要推荐笔记本，需要扩展类型表与测试。
4. **中文回复仍不进 M9**：M9 抽取器是英文规则；本轮的“游戏/8000”是 **task-scoped**
   `task_requirements`，不是永久偏好。这是有意的边界：购买任务约束不应写进长期偏好。
5. **直接从 MCP 客户端冷恢复未开放**：`restore_clarification_handle` 需要可信的 Web 快照；
   旧 handle 在子进程重启后对直接客户端仍是 `invalid_or_expired_clarification_id`。
6. **不承诺 exactly-once**：Web SQLite 与 M9 SQLite 是两库；M9 写成功后 Web finalize 失败 →
   AMBIGUOUS，绝不返回成功。
7. **并发 Cold Reply 的孤儿 handle**：失败方 restore 出的新 handle 没有业务副作用，
   会随子进程 TTL 失效；不做跨进程回收（避免引入第二套生命周期）。
8. **opt-in 集成测试断言已按真实工具注册表更新**：
   `tests/test_web_demo_integration.py` 的 `EXPECTED_TOOLS` 现为真实 9 工具
   （`remember_preference` 自 `d66f95a` 起无条件注册、权限在调用时校验；
   `restore_clarification_handle` 为 M1-D 新增）。已用
   `AGENTRECX_RUN_MCP_INTEGRATION=1` 实测通过（见第 3 节），未禁用测试。
9. 未 Push；未使用 `git add -A`。

## 5. 不可违反的安全规则

1. 裸 `warm_available=True` **不存在**；Handle 必须由 `MCPManager.check_handle(ref)` 证明。
2. 外部调用前必须已持久化 `EXTERNAL_STARTED`。
3. `EXTERNAL_STARTED`/`AMBIGUOUS` 永不自动重跑 MCP。
4. `RESULT_DURABLE` 只读检查点 Finalize，不再推荐；COMPLETED 只重放。两者都不需要 Handle。
5. Finalize COMMIT 前不得更新权威 Session、不得返回 200。
6. Reply Turn 只用已签发的 `pending.reply_turn_id`。
7. M9 与 Web State 不是同一事务域；不宣称 exactly-once。
8. 不修改 Planner/Ranker/Gate/M9 事件语义、不改前端契约（回复响应字段与内存模式一致）。
9. Cold Resume 的 restore 是**无业务副作用**的进程内状态重建，必须在 Claim 之前；
   真正的业务执行只有在 `EXTERNAL_STARTED` 之后才开始。
10. 旧 Handle id 永不复用；新进程签发新 Handle；快照缺失/损坏/过期/版本不符/绑定不符 → 409。
11. **Task budget 是硬约束**：`within` 才允许 finalize；`over` 直接排除；`unknown`/
    `currency_mismatch` 永不视为合规。预算边界为闭区间（8000 ≤ 8000），金额比较不做四舍五入。
12. **拒绝必须是已知结果**：只有真实推荐返回了 `budget_unverifiable`/`budget_exceeded` 才走
    refusal；`awaiting_clarification`/`error`/通用 `no_verified_candidate_in_scope` 仍按原语义
    处理（后者仍是 503，避免改变既有 Gate 边界）。refusal 在 finalize COMMIT 后才返回 200，且
    **没有商品**（`recommendation: null`），Replay/重启必须返回同一 refusal。
13. 预算数字**不得**进入 BM25 查询文本；只追加 category/use-case 英文词。
14. M9 与 task requirements 是两个域：M9 是永久用户偏好，task requirements 是本次购买约束；
    不得把任务预算写成永久 M9 偏好来“证明”覆盖。
15. 不使用 `git add -A`；不 Push。

## 6. 已知陷阱

- 编辑 Python 文件**不要用 PowerShell `-replace` 写回**；用 `edit` 工具或脚本，写完先 `python -c "import ..."`。
- 真实 MCP 测试必须**在同一个 `asyncio.run` / TestClient lifespan 内** start/call/stop；
  不要在测试线程里用 `asyncio.run(manager.call_tool(...))`（会跨 event loop，阻塞到 tool timeout）。
- 真实 FastAPI 测试先轮询 `GET /api/health` 到 `ready`，再发 chat/reply。
- 真实子进程 PID 通过 `/proc` 枚举：按 `PPid == os.getpid()` + cmdline 含
  `recommendation.mcp.server` + `--user-key <profile>` 过滤。
- `web_turn.turn_seq` 是插入序（MAX+1）；Reply Turn 用 `-rN`。
- `recommend_products` 不返回 `products`/`narrative`/`recommendation_id`；展示必须由 BFF 生成。
- 真实 MCP 测试较慢（启动约 24–32s/进程）；Cold Resume 全量约 11–12 分钟。
- HTTP 并发回复在 `EXTERNAL_STARTED` 期间重复到达时，store 的既定语义是
  `RequestAmbiguous`（503），不是 BUSY；这是 fail-closed 设计，不要改成"自动重跑"。
- v6 迁移给 `web_pending` 加列；旧行是 NULL。若用 SQL 手工构造已过期 pending，
  必须先改 `created_at` 再改 `expires_at`（CHECK `expires_at > created_at`）。
- 测试替身 `FakeManager.call_tool` 必须接受真管理器的 `timeout_s` 等 kwargs。
- **BM25 数字污染**：`gaming laptop under 8000` 会命中型号 `8000`（适配器/电池），检索结果与
  预算无关。预算只作为硬约束，不进入查询；查询增强只用 `laptop gaming`。
- **sidecar 价格是 historical USD**：`inspect` 返回 `historical_metadata` + `currency: "USD"`，
  且仍把 `price` 放进 `unknown_fields`（不是 live price）；跨币种比较必须拒绝。
- **refusal 响应形状**：`recommendation: null`，`refusal: {code, reason, detail, budget,
  task_requirements}`；前端用 `response.recommendation ?? null`，不会渲染商品卡。
- **共享 M9 DB 的建库竞态**：两个 MCP 子进程会同时打开同一个 `m9.sqlite3`；schema 初始化必须
  幂等（`INSERT OR IGNORE` + 重读），否则子进程启动即崩、`/api/health` 永不 ready。
  已加回归 `test_two_stores_can_create_the_same_fresh_schema_concurrently`。
- 真实 E2E 的 `_wait_ready` 超时统一用 300s；机器负载高时 240s 会误报启动失败。

## 7. 交接检查清单

- [x] HEAD/工作区核对（M1-E 接手 `b1165b5`，与任务提示一致）
- [x] 中文原查询 5000 CNY / laptop 进入 ASK continuation；restore 后仍保留
- [x] 中文回复“游戏，预算改成 8000 元”合并为 8000 CNY / gaming / laptop（不是并存）
- [x] `recommend_products` 实际收到合并后的 `task_requirements`，并对 20 个候选执行价格核对
- [x] 默认目录无价 → `budget_unverifiable`；HTTP 200 的可重放 refusal，`recommendation: null`
- [x] 预算边界 7999/8000/8001（闭区间）与跨币种/缺价拒绝有单测和 executor 集成测试
- [x] Warm Reply / Cold Reply / 旧 Handle 拒绝 / M9 / Replay / RESULT_DURABLE / AMBIGUOUS /
      Reset Fencing 仍全绿（`345 passed`）
- [x] opt-in 真实 MCP 集成通过（`1 passed`）
- [x] 修复 M9 共享库并发建 schema 竞态并加回归
- [x] 本地 Commit，未 Push
- [ ] 真实 CNY 价格（或有来源汇率）/ laptop Product Role / qwen_live 持久化回复 /
      多轮跟随 ASK（明确未实现，见第 4 节）

## 8. 上一冲刺执行记录（M1-D S6）

| 步骤 | 结果 |
|---|---|
| 接手核对 | `git rev-parse HEAD` = `3643ffc`；工作区 clean |
| 实现迭代（真实 probe） | ASK → child A(591) → restart → child B(602) → 新 handle → 200 / 真实商品 `B00LJX4KZ8` |
| Cold Resume 真实 E2E | `2 passed in 157.78s` |
| Cold Resume 完整文件 | `12 passed` |
| Continuation 单测 | `38 passed` |
| opt-in 集成测试 | `1 passed in 37.83s` |
| **最终全量（22 文件）** | **`294 passed in 629.92s`，exit code 0** |
| `git diff --check` | 无输出 |
| Commit / Push | 1 个本地提交（`feat: restore durable clarification after MCP restart`）/ 未 Push |

真实端到端证据（`tests/test_web_cold_resume_e2e.py::test_real_cold_resume_after_mcp_child_restart`）：

1. 真实 BFF 启动真实 BM25 MCP 子进程 A，`/proc` 记录 PID A；创建 Web Session 并发起真实购物请求；
2. 真实 `request_preference_clarification` 生成 ASK；Task/Pending/Continuation 一次性落库
   （`continuation.expires_at == pending.expires_at`，MCP task_id = `p14e2_lite_0057`）；
3. `POST /api/demo/restart` 只重启 profile_0057 子进程 → PID B（与 A 不相交），
   `manager.check_handle(old)` = `stale_process`；
4. 原 Session + 原 Pending + 新回复 → 工具顺序严格为
   `restore_clarification_handle → submit_clarification_reply → get_active_preferences →
   recommend_products → get_product_info`；submit 使用**新** handle id；
5. M9 真实写入并读回；真实商品卡 + Narrative + `rec-…` ID 持久化；
6. HTTP 200；Task/Pending/Request/Reply Turn 状态正确；
7. 同请求 replay：响应完全一致、零 MCP、M9 行数不变；
8. 重启整个 BFF：GET 恢复同一 Reply Turn/商品/Recommendation ID，replay 仍 200 且零 MCP。

另一条真实测试 `test_real_cold_resume_after_bff_and_mcp_restart` 覆盖 BFF 与 MCP 同时重启。

## 9. 本次冲刺执行记录（M1-E）

| 步骤 | 结果 |
|---|---|
| 接手核对 | `git rev-parse HEAD` = `b1165b5`；工作区 clean |
| 新纯函数模块 | `recommendation/preference/task_requirements.py` |
| 真实 probe（MCP 直连） | ASK 快照 `5000 CNY / laptop`；restore 后不变；回复合并为 `8000 CNY / gaming / laptop`；推荐 20/20 候选 `price UNKNOWN` → `budget_unverifiable` |
| 需求/价格单测 | `16 passed`（含 7999/8000/8001、跨币种、缺价） |
| executor/Gate 预算集成 | `4 passed`（测试商品价格，明确非 MCP E2E） |
| 真实需求更新 Cold Resume E2E | 连续 2 次单跑通过（含 PID A→B、旧 handle stale、合并约束、refusal、Replay、重启） |
| M9 竞态修复 | `INSERT OR IGNORE` + 重读；回归测试通过 |
| **最终全量（29 文件）** | **`345 passed in 781.11s`，exit code 0** |
| opt-in 真实 MCP 集成 | `1 passed in 32.75s` |
| `git diff --check` | 无输出 |
| Commit / Push | 1 个本地提交（`feat: merge task requirements and enforce price budgets`）/ 未 Push |

真实端到端证据（`tests/test_web_requirement_update_e2e.py`）：

1. 真实 BFF + BM25 MCP 子进程 A：中文查询“推荐一台 5000 元的笔记本” → 真实 Gate
   `NEED_USER_CLARIFICATION`（缺 use case）→ ASK；continuation 的
   `mcp.task_requirements` = `{budget: 5000 CNY, category: laptop}`；
2. `POST /api/demo/restart` → PID B（与 A 不相交），旧 handle = `stale_process`；
3. 回复“游戏，预算改成 8000 元” → 工具严格为
   `restore_clarification_handle → submit_clarification_reply → get_active_preferences →
   recommend_products`；restore 参数带 5000，推荐参数带 **8000 CNY / gaming / laptop**；
4. 推荐真实执行：`effective_query = 推荐一台 5000 元的笔记本 laptop gaming`，20 个候选全部
   `price UNKNOWN`（`catalog_meta.jsonl has no price field`）→ Gate 的 ALLOW 被预算硬约束覆盖为
   `budget_unverifiable`，`selected_product_id = null`；
5. Coordinator 持久化 refusal（HTTP 200、`recommendation: null`），message 明确写出
   `8000 CNY`、`price UNKNOWN`、`No product is shown ... not a recommendation`；
6. DB：Request `COMPLETED`、Pending `CONSUMED`、Reply Turn 商品 items 为空、refusal 落库；
   M9 中文回复不产生永久偏好（`active == []`）；
7. Replay 完全一致且零 MCP；BFF 重启后恢复同一 refusal Turn，Replay 仍 200 且零 MCP。

价格数据证据：

- 默认 `catalog_meta.jsonl`：`inspect(price)` → `status=unknown`，reason
  “catalog_meta.jsonl has no price field”；
- 真实 sidecar 行（`amazon_reviews_2023:raw_meta_Electronics`）：数值 price、`currency: "USD"`、
  `price_kind: historical_crawl_time`；CNY 预算 → `currency_mismatch`（单测）；
- 因此没有把 USD 历史价格冒充为 8000 元内的真实推荐；没有为通过测试而放宽币种。
