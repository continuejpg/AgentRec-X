# Warm Reply 闭环交接单（WARM_REPLY_HANDOFF）

> 用途：M1-C 持久化 Web Session 的 **Warm Clarification Reply 闭环**已实现并验证；新会话从这里接手。
> 最后更新：2026-11-21（本次冲刺结束时）

## 0. 环境与仓库

| 项 | 值 |
|---|---|
| 仓库根（Windows） | `D:\IT\CODE\PROJECTION\agentrec\X` |
| 仓库根（WSL） | `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| Python | 项目 `.venv`（Linux/WSL）`.venv/bin/python`，Python 3.10 |
| 测试前缀 | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest ... -q -p no:cacheprovider` |
| Git | 在 Windows PowerShell 操作同一目录；**未 Push** |

## 1. Git 基线

- 本次冲刺**接手时** HEAD：`cf3a465`（`refactor: separate recommendation snapshot construction from session writes`）。
  用户提示的 `762e636` 是它的父提交（`cf3a465 → 762e636 → 32f7770`），仓库实际多 2 个提交。
- 本冲刺结束时：**一个本地 Commit**（`feat: complete durable warm clarification reply`，hash 见 `git log -1`），
  未 Push；提交前工作区包含的 Coordinator 原型（`state_store.py` / `reply_executor.py` /
  `test_web_reply_executor.py`）已随本提交一起入库，没有被删除或改写成简化版。
- 接手时的未提交修改（`web_demo/state_store.py` 的 `execution_stage='COMPLETED'`）也属于本提交。

## 2. 已具备且已验证的能力

### 2.1 Web State（SQLite v5，未改 schema）

`web_demo/state_store.py`：`meta/web_session/web_identity/web_task/web_pending/web_turn/web_request`；
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
  （Cold Resume / qwen_live 暂不支持，返回 409）→ `WarmReplyExecutor`。

`web_demo/reply_executor.py` 顺序：

```
replay? → prove live handle → claim → heartbeat → EXTERNAL_STARTED
→ submit_clarification_reply → get_active_preferences → recommend_products
→ get_product_info(selected) → assemble → build_reply_payload
→ checkpoint(RESULT_DURABLE) → finalize(COMMIT) → HTTP 200
```

- `handle_check` 与 `assemble` 都是构造注入，缺 checker 时 fail-closed，缺 assembler 直接 TypeError；
- `EXTERNAL_STARTED` 之后的超时/异常 → `AMBIGUOUS`（503），永不自动重跑；
- `RESULT_DURABLE` 只读检查点 Finalize，不再调用 MCP；
- Finalize 单事务写 Task/Pending/Request/Reply Turn；**提交后**才镜像到内存 Session 并返回 200；
- Reply Turn 使用 ASK 时签发的 `pending.reply_turn_id`；
- 持久化结果形状：`{reply: <contract payload>, preferences, memory_update, steps}`；
  `turn_from_result` 会把 `memory_update`/`tool_activity` 一并投影进 `web_turn`，
  重启后 GET 的回复 Turn 与在线响应一致。

### 2.7 HTTP 行为与错误码

成功/重放 200（响应 = 持久化结果重建，完全一致）；其余一律非 200：

| outcome | HTTP | code |
|---|---|---|
| Handle 不成立 / restored / 非 cpu_live | 409 | `clarification_resume_not_ready` |
| 另一执行者持有 lease | 409 | `clarification_reply_in_progress` |
| 同 request 不同内容 | 409 | `clarification_reply_conflict` |
| Reset/任务版本/租约 fencing | 409 | `clarification_reply_fenced` |
| `EXTERNAL_STARTED` 后不确定（超时/读写失败） | 503 | `clarification_reply_ambiguous` |
| 检查点不可持久化 | 503 | `clarification_result_not_durable` |
| 无匹配 Pending / CANCELLED / EXPIRED | 409 | `clarification_not_pending` |

`MCPManagerError` 的现有处理器负责状态码映射；`app.py` 未改动。

## 3. 测试状态（真实数字）

- **全量回归（20 个文件，最终代码）：`244 passed in 462.07s`，exit code 0。**
- 真实 MCP + FastAPI 端到端文件（`tests/test_web_warm_reply_e2e.py`）：`6 passed in 129.09s`
  （2 条真实 MCP 子进程 + 4 条确定性 fail-closed）。
- 真实 MCP 单测（`tests/test_web_handle_lifecycle.py` 中的 restart/expiry 场景）：约 67s。
- 本轮增量：`reply_executor` 8→12、`reply_safety` 6→10、新增 `reply_assembly`(7)、`warm_reply_e2e`(6)，
  合计 224 → 244（轮 A 基线 224）。

全量命令：

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_web_real_recommendation_contract.py \
  tests/test_web_reply_executor.py \
  tests/test_web_reply_assembly.py \
  tests/test_web_warm_reply_e2e.py \
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
  -q -p no:cacheprovider --basetemp=/tmp/m1c_warm
```

## 4. 剩余工作与真实阻断

1. **Cold Resume 未实现**：进程/子进程重启后旧 Handle 一律拒绝（`handle_missing` / `stale_process`），
   这是刻意的安全行为；跨进程恢复需要 MCP 侧持久化 continuation，不在本次范围。
2. **持久化 Warm Reply 仅支持 CPU rule planner**（`pending.mode == "cpu_live"`）。
   qwen_live 持久化 Pending → 409 `clarification_resume_not_ready`。
3. **多轮跟随 ASK 未支持**：回复后 `recommend_products` 若再次 `awaiting_clarification`
   → 503 `clarification_reply_ambiguous`（M9 已写，结果不可最终展示，禁止自动重试）。
   需要新的 ASK/Pending 流程。
4. **不承诺 exactly-once**：Web SQLite 与 M9 SQLite 是两库；M9 写成功后 Web finalize 失败 →
   AMBIGUOUS，绝不返回成功。
5. **opt-in 集成测试既有失败（与本次无关，未修）**：
   `tests/test_web_demo_integration.py:140-143` 仍假设 `remember_preference` 只出现在可写 profile；
   自 `d66f95a` 起服务端无条件注册（每台 8 个工具）。临时改成
   `set(EXPECTED_TOOLS) | {"remember_preference"}` 后整套真实流程通过（34.99s），断言随后已还原。
   建议单独一行修复。
6. 未 Push；未使用 `git add -A`。

## 5. 不可违反的安全规则

1. 裸 `warm_available=True` **不存在**；Handle 必须由 `MCPManager.check_handle(ref)` 证明。
2. 外部调用前必须已持久化 `EXTERNAL_STARTED`。
3. `EXTERNAL_STARTED`/`AMBIGUOUS` 永不自动重跑 MCP。
4. `RESULT_DURABLE` 只读检查点 Finalize，不再推荐。
5. Finalize COMMIT 前不得更新权威 Session、不得返回 200。
6. Reply Turn 只用已签发的 `pending.reply_turn_id`。
7. M9 与 Web State 不是同一事务域；不宣称 exactly-once。
8. 不修改 Planner/Ranker/Gate/M9 事件语义、不改前端契约（回复响应字段与内存模式一致）。
9. 不使用 `git add -A`；不 Push。

## 6. 已知陷阱

- 编辑 Python 文件**不要用 PowerShell `-replace` 写回**；用 `edit` 工具或脚本，写完先 `python -c "import ..."`。
- 真实 MCP 测试必须**在同一个 `asyncio.run` / TestClient lifespan 内** start/call/stop。
- 真实 FastAPI 测试先轮询 `GET /api/health` 到 `ready`，再发 chat/reply。
- `web_turn.turn_seq` 是插入序（MAX+1）；Reply Turn 用 `-rN`。
- `recommend_products` 不返回 `products`/`narrative`/`recommendation_id`；展示必须由 BFF 生成。
- 真实 MCP 测试较慢（启动约 24s/进程），全量约 7–8 分钟。
- HTTP 并发回复在 `EXTERNAL_STARTED` 期间重复到达时，store 的既定语义是
  `RequestAmbiguous`（503），不是 BUSY；这是 fail-closed 设计，不要改成"自动重跑"。

## 7. 交接检查清单

- [x] HEAD/工作区核对（接手 `cf3a465`，与交接单一致；用户提示的 `762e636` 是父提交）
- [x] 全量回归通过：`244 passed`（真实数字见第 3 节）
- [x] 真实 MCP + FastAPI Warm Reply 成功、Replay、重启恢复
- [x] 失败/恢复场景：Handle 不存在/过期/跨 Session/子进程重启、超时、Lease 失效、Reset fencing
- [x] 本地 Commit，未 Push
- [ ] Cold Resume / qwen_live 持久化回复 / 多轮跟随 ASK（明确未实现）

## 8. 本次冲刺执行记录

| 步骤 | 结果 |
|---|---|
| 接手基线（轮 A 前，17 文件） | `202 passed in 299.11s` |
| 轮 A（Handle 生命周期）后 | `224 passed in 372.73s` |
| 轮 B/C 真实 MCP e2e 单独跑 | `6 passed in 129.09s` |
| **最终全量（20 文件）** | **`244 passed in 462.07s`，exit code 0** |
| `git diff --check` | 无输出 |
| Commit / Push | 1 个本地提交 / 未 Push |

真实端到端覆盖（`tests/test_web_warm_reply_e2e.py`）：
真实 BM25 MCP 子进程 → ASK → `/api/clarification/reply` → M9 写入与偏好读取 →
完整商品卡片 + Narrative + 推荐 ID → SQLite Task/Pending/Request/Reply Turn 原子完成 →
相同请求 Replay（零 MCP、零新 Turn）→ BFF 重启后 GET 恢复 + Replay；另有一条
“重启后 restored Pending 必须 409、零 MCP、数据不变”的真实测试。
