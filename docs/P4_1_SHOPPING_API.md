# Phase 4.1 — 多轮购物 HTTP API 集成

> 将 Phase 4 的 `DemoAgentService.run_shopping_turn` 接入 FastAPI：创建 demo session 后，
> 通过 HTTP 完成“请求 → 澄清 → 回答 → 推荐”的完整闭环。
>
> 不重写购物意图、检索、澄清或理由逻辑；路由只做 session 查找、server-owned history 装配和委托。

## 0. TL;DR

- 新路由：`POST /v1/demo/sessions/{session_id}/agent/shopping`。
- 新请求模型 `AgentShoppingRequest`：`message`、`k`（1–100）、`sources`、`constraints`（≤8）、
  `clarification_id`（可选）；**不接收** `history`/`user_key`，二者由 session 拥有。
- 响应复用 `AgentRecommendResponse`：`clarification`、`intent`、推荐项 `reason` 全部可用。
- 澄清往返：冲突/不可验证 → `clarification` + `waiting_for_user`；下一轮带
  `clarification_id` 回答 → 意图更新并重新推荐。
- 会话隔离、未知/跨会话 id、重复回答、过期（TTL 30 分钟）均有明确 HTTP 状态。
- 定向测试 **13 个新 HTTP 测试**，受影响套件 **148 passed**；原有 `/agent/recommend` 不回归。
- **持久性限制**：pending 与 session 都是进程内状态，重启/多 worker 不保证恢复；本阶段不新建
  数据库，也不宣称持久化。

---

## 1. 路由与 schema

### 1.1 创建会话（复用现有接口）

```http
POST /v1/demo/sessions
Content-Type: application/json

{"profile_id": "<profile id from GET /v1/demo/profiles>"}
```

返回 `201 {"session_id": "...", ...}`。`session_id` 是后续购物路径的会话标识；
session 拥有 trusted history 与独立的 preference-memory `user_key`。

### 1.2 购物回合

```http
POST /v1/demo/sessions/{session_id}/agent/shopping
Content-Type: application/json

{
  "message": "I need a yoga mat, budget $30",
  "k": 5,
  "sources": ["history", "catalog_search"],
  "constraints": [],
  "clarification_id": null
}
```

`AgentShoppingRequest`（`extra="forbid"`、frozen）：

| 字段 | 类型/范围 | 说明 |
|---|---|---|
| `message` | 非空文本（`Message`） | 本轮用户请求；不可信文本 |
| `k` | 严格整数 1–100 | 最终展示条数 |
| `sources` | `tuple[str, ...]`，默认 `history + catalog_search` | 现有来源计划，每源都执行并融合 |
| `constraints` | `tuple[AgentConstraintItem, ...]`，最多 8 | 本轮硬约束；同时投影进购物意图 |
| `clarification_id` | 可选 opaque id | 回答上一轮 pending 问题时必填 |

请求体**没有** `history` 和 `user_key`：路由用 `session.trusted_user_history` 和
`session.user_key` 构造 `AgentRecommendRequest`，客户端无法把请求指向别的会话/命名空间。

### 1.3 响应

复用 `AgentRecommendResponse`：

- `recommendations[]`：最终列表，每项含 `reason`（证据绑定）；
- `intent`：本轮 hard/soft/conflicts/turn_ids；
- `clarification`：有 pending 问题时存在，含 `clarification_id`、`question`、`kind`
  （`conflict` / `unverified`）、`blocks`；
- `terminal`：`waiting_for_user/awaiting_user` 或 `finished/completed`；
- `message`：最终列表文本（Phase 3.1 一致性）或澄清问题。

---

## 2. 调用链

```
POST /v1/demo/sessions/{id}/agent/shopping
  └─ runtime.sessions.turn(session_id)          # 查找 + 会话锁 + 独立 turn 序列
       └─ session.trusted_user_history          # server-owned
       └─ session.user_key                      # 独立 memory 命名空间
  └─ AgentRecommendRequest(history=..., user_key=..., message/k/sources/constraints/clarification_id)
  └─ DemoAgentService.run_shopping_turn(payload, session_id=session.user_key)
       ├─ pending 校验（session/TTL/一次性）
       ├─ memory snapshot + 本轮 extraction 预览 → ShoppingIntent
       ├─ 冲突？`ClarificationPolicy` 在第一步提问
       ├─ 否则：IntentQueryPolicy 重建 catalog terms；run loop
       │     └─ 无 verified + 有 unresolved → `ClarificationPolicy` 在 FINISH 前提问
       └─ 附加 `reason` / `intent` / `clarification`
```

没有新策略、没有新排序、没有第二套记忆；路由约 40 行，其余是 Phase 4 服务。

---

## 3. 澄清往返与失败处理

| 场景 | HTTP | 响应 |
|---|---|---|
| 正常请求 | 200 | `route=recommend`，推荐 + `intent` + `reason`，`clarification=null` |
| 硬约束冲突 | 200 | `clarification.kind=conflict`，`terminal=waiting_for_user`，`recommendations=[]` |
| 全部不可验证 | 200 | `clarification.kind=unverified`，列表为空 |
| 正确回答 | 200 | 意图更新，返回新推荐；`clarification=null` |
| 未知 id | 404 | `{"error":"unknown_clarification", ...}` |
| 跨会话 id | 404 | 同上；**不会消费**对方 pending 问题 |
| 重复回答 | 404 | 第一次回答已消费该 id |
| 过期 id | 410 | `{"error":"clarification_expired", ...}` |

- pending 记录：`{session_id, question, kind, created_at}`；TTL **30 分钟**。
- 同一 session 最多一次澄清；第二次仍不可判定时走既有 `failed/no_verified_candidate`。
- 回答成功后 pending 被消费；重复/并发回答只有一个成功（锁内 get→validate→pop）。
- 错误体沿用项目 `{"error", "detail"}` 契约。

---

## 4. 自动化测试

`tests/test_p4_shopping_api.py`（13 项）：

1. OpenAPI 发布路由；
2. `k=101`/额外字段（`history`、`user_key`）→ 422；
3. 正常回合：最终列表、intent、每项 reason、message 与列表一致；
4. 冲突 → 单个 pending 问题 + `waiting_for_user`；
5. 用 `clarification_id` 回答 → 意图更新 + 新推荐；
6. 不可验证约束 → `unverified` 问题 → 回答后完成；
7. 未知 id → 404；
8. 跨会话回答 → 404 且原 session 的 pending 仍可用；
9. 重复回答 → 404；
10. 过期（直接老化 created_at）→ 410，之后 404；
11. 硬预算：所有展示商品目录价格 ≤ 上限，reason 含价格事实；
12. 每项 reason 使用该商品自己的价格；
13. 原 `/v1/demo/agent/recommend` 仍返回推荐（不回归）。

```
pytest -q tests/test_p4_shopping_api.py tests/test_p4_shopping_intent.py \
         tests/test_p4_clarification_policy.py tests/test_demo_agent_api.py \
         tests/test_agent_chat_adapter.py tests/test_agent_control_plane.py \
         tests/test_p3_constraint_empty_message.py tests/test_p3_final_list_consistency.py \
         tests/test_phase32_terminal_semantics.py
# 148 passed
```

测试使用现有 `tests/agent_reranking_fixture.py` 的合成目录/引擎与 `tests/demo_fixture.py`
的 session/profile 机制，不重复建立评测机制。

---

## 5. 可复现请求示例

```bash
BASE=http://127.0.0.1:8000

# 1) 选一个 profile 并创建会话
curl -s $BASE/v1/demo/profiles | head
SID=$(curl -s -X POST $BASE/v1/demo/sessions \
  -H 'content-type: application/json' \
  -d '{"profile_id":"<profile-id>"}' | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# 2) 第一轮：预算与下限冲突 -> 澄清
curl -s -X POST $BASE/v1/demo/sessions/$SID/agent/shopping \
  -H 'content-type: application/json' \
  -d '{"message":"Recommend some hiking gear.","k":5,
       "constraints":[{"kind":"price_max","value":"20"},
                       {"kind":"price_min","value":"30"}]}'
# => {"clarification":{"clarification_id":"...","kind":"conflict",...},
#     "terminal":{"status":"waiting_for_user","termination_reason":"awaiting_user"}}

# 3) 回答并重新推荐
curl -s -X POST $BASE/v1/demo/sessions/$SID/agent/shopping \
  -H 'content-type: application/json' \
  -d '{"message":"make it under $30 instead",
       "clarification_id":"<clarification_id>"}'
# => 200，recommendations[].reason 有值，clarification=null
```

错误示例：

```bash
curl -i -X POST $BASE/v1/demo/sessions/$SID/agent/shopping \
  -H 'content-type: application/json' \
  -d '{"message":"make it under $30 instead","clarification_id":"unknown"}'
# HTTP/1.1 404，{"error":"unknown_clarification","detail":"..."}
```

---

## 6. 状态持久化限制（明确声明）

- **pending clarification 是进程内字典**（`DemoAgentService._pending_clarifications`），
  不写入 SQLite、不跨进程共享；TTL 30 分钟。重启或多 worker 部署时，回答可能落到没有该
  pending 的进程，得到 404。
- **session 注册表本身也是进程内的**（现有 `DemoSessionManager` 语义）：重启后 `session_id`
  失效，因此即使 preference memory 的 SQLite 文件还在，也无法通过原 session 继续。
- 现有存储只有 **preference memory**（用户偏好/来源轮次），没有“pending 问题”模型。
  以最小改动支持恢复需要新增表/字段或引入共享缓存；本阶段**不新建数据库**，也不宣称具备
  跨重启恢复。
- 多 worker 部署需要 sticky session（同一 `session_id` 固定到同一进程），否则澄清往返
  不保证；这是部署约束，不是本阶段代码行为。

---

## 7. 代码差异

```
 recommendation/api/demo_routes.py    |  56 +++++++++
 recommendation/demo/__init__.py      |   2 +
 recommendation/demo/agent_schemas.py |  +48（AgentShoppingRequest + 导出）
 recommendation/demo/agent_service.py |  +20（pending TTL/隔离校验）
新增: tests/test_p4_shopping_api.py
```

（`git diff --stat` 的累计值还包含 Phase 4 未提交的意图/policy/service 改动。）

未改：RRF、Learned Ranker、候选深度、来源计划、完成守卫语义、冻结评测资产。

---

## 8. 尚未解决

1. **无持久化 pending / 无跨 worker 共享**：见 §6；下一步可评估在现有 store 增加
   clarification 表，或使用共享缓存。
2. **TTL 固定 30 分钟**，不可配置；过期后只能开启新回合。
3. **重复回答返回 404**：一次性语义清晰，但网络重试的客户端无法区分“已成功但丢响应”；
   后续可加幂等键。
4. **没有 create-and-shop 便捷端点**：客户端必须先 `POST /sessions`。
5. **demo 无鉴权**：session id 即能力凭证，仅用于本地演示。
6. **未运行全量测试**；未自动提交 Git。
