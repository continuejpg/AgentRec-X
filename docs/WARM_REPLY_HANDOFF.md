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
- M1-D S6 接手时 HEAD：`3643ffc`（与任务提示一致）；本轮结束会再产生一个本地 Commit
  （`feat: restore durable clarification after MCP restart`），仍未 Push。
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

## 3. 测试状态（真实数字）

- **M1-D S6 全量回归（22 个文件，最终代码）：`294 passed in 629.92s`，exit code 0。**
- 真实跨 OS 进程 Cold Resume（`tests/test_web_cold_resume_e2e.py` 的 `real_cold` 两条）：
  `2 passed in 157.78s`（PID A→B、旧 handle stale、restore 新 handle、M9、推荐、200、Replay、双重启）。
- 该文件完整跑：`12 passed`（2 真实 + 10 确定性注入：损坏/过期快照、restore 失败、
  并发、外部执行后崩溃、RESULT_DURABLE、Reset/跨 Session）。
- 持久化快照单测（`tests/test_web_continuation.py`）：`38 passed`。
- opt-in 真实 MCP 集成（`AGENTRECX_RUN_MCP_INTEGRATION=1`）：`1 passed in 37.83s`（9 工具断言已更新）。
- M1-C 结束基线：`244 passed`；本轮增量 38（continuation）+ 12（cold e2e）= 50 → 294。
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
   `recommend_products` 一次能 finalize。以下仍是明确的 409：
   - **qwen_live 持久化 Pending**：Qwen 的 handle 依赖子进程内的 planner/executor continuation，
     不是 JSON 可序列化状态；需要事件化/重放式设计，不在本轮。
   - **多轮跟随 ASK**：回复后 `recommend_products` 若再次 `awaiting_clarification`
     → 503 `clarification_reply_ambiguous`（M9 已写、结果不可最终展示、禁止自动重试）；
     需要新的 ASK/Pending + continuation 轮次。
   - **中文预算约束**：本轮未新增任何中文约束解析；沿用现有 M9 抽取能力。
2. **直接从 MCP 客户端冷恢复未开放**：`restore_clarification_handle` 需要可信的 Web 快照；
   旧 handle 在子进程重启后对直接客户端仍是 `invalid_or_expired_clarification_id`。
3. **不承诺 exactly-once**：Web SQLite 与 M9 SQLite 是两库；M9 写成功后 Web finalize 失败 →
   AMBIGUOUS，绝不返回成功。
4. **并发 Cold Reply 的孤儿 handle**：失败方 restore 出的新 handle 没有业务副作用，
   会随子进程 TTL 失效；不做跨进程回收（避免引入第二套生命周期）。
5. **opt-in 集成测试断言已按真实工具注册表更新（本轮修复）**：
   `tests/test_web_demo_integration.py` 的 `EXPECTED_TOOLS` 现为真实 9 工具
   （`remember_preference` 自 `d66f95a` 起无条件注册、权限在调用时校验；
   `restore_clarification_handle` 为本轮新增）。已用
   `AGENTRECX_RUN_MCP_INTEGRATION=1` 实测通过（见第 8 节），未禁用测试。
6. 未 Push；未使用 `git add -A`。

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
11. 不使用 `git add -A`；不 Push。

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

## 7. 交接检查清单

- [x] HEAD/工作区核对（M1-D 接手 `3643ffc`，与任务提示一致）
- [x] Cold Resume 真实跨进程 E2E（PID A/B、旧 handle 拒绝、新 handle、M9、推荐、200、Replay、双重启）
- [x] Warm Reply 仍正常（`test_web_warm_reply_e2e.py` 全绿；FakeManager 无快照时仍 409）
- [x] Replay / RESULT_DURABLE / AMBIGUOUS / Reset / 并发 / 跨 Session 仍 fail-closed
- [x] 全量回归通过：`294 passed`（真实数字见第 3 节）
- [x] opt-in 集成测试断言按真实 9 工具更新并实测通过
- [x] 本地 Commit，未 Push
- [ ] qwen_live 持久化回复 / 多轮跟随 ASK / 中文预算约束（明确未实现，见第 4 节）

## 8. 本次冲刺执行记录（M1-D S6）

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
