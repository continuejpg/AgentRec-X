# P17 — MCP Tool Server Integration

> 范围：P17 only。使用官方 MCP Python SDK v2 的 `MCPServer`，实现本地 stdio MCP Server，把现有 AgentRec-X 的 Search、Recommendation、Memory、Evidence 和两个 P16 Skills 暴露给独立 MCP Client。
> **不需要 GPU**：本轮 MCP Server 与独立客户端都在 CPU/本地运行；BGE query encoding 本地 CPU，Qwen 未被调用。没有连接/重启 AutoDL GPU 服务。

## 0. 结论

- **MCP Server 真实运行**：官方 SDK `mcp==2.3.0`，协议 `2026-07-28`，stdio transport，两个独立 server 进程（BGE + BM25 配置）。
- **独立 MCP Client 真实发现并调用 7 个 Tools**：`search_products`、`get_product_info`、`recommend_products`、`get_active_preferences`、`compare_products`、`request_preference_clarification`、`submit_clarification_reply`。
- **直接复用核心**：Catalogue / BGE / P15 Evidence Bundle / P16 Skill Registry / Rule Planner+Executor / V0/Role/Eligibility/Gate / M9 Memory，没有第二套 Runtime。
- **实际 E2E**：2 个 stdio server、26 次 client 调用、16 次成功、10 个预期的失败/边界错误；写入、未确认、过期、跨用户与非法身份均返回明确错误。
- **安全边界**：本地单用户可信上下文，MCP Tool 不接受 `user_id`；写入默认关闭，测试 server 仅显式 `--allow-memory-writes` 并接受明确的 `explicit_test_user_reply`。没有生产认证，**MEMORY_ACCESS_CONTROL = LIMITED**。
- **P17 = GO（本地 MCP 可复现）**，不代表公网部署已具备认证/安全条件。

## 1. SDK / 协议 Manifest

`runs/p17/sdk_manifest.json`：

- SDK：官方 `mcp==2.3.0`（v2 stable line，`MCPServer`，不是 v1 `FastMCP`）；
- Python：3.10.12；
- protocol：`2026-07-28`；
- transport：**stdio only**；
- server class：`mcp.server.MCPServer`；
- client class：`mcp.Client`；
- 依赖：`mcp-types==2.3.0`、`httpx2==2.14.0`、`pydantic==2.13.5`；
- 不实现 Streamable HTTP；不开放未认证 Memory。

日志全部写 **stderr**，不污染 MCP stdio 消息通道。

## 2. Server 实现

- `recommendation/mcp/server.py`、`recommendation/mcp/__init__.py`；
- 启动时加载现有 `ElectronicsCatalogue`，按 `--search-mode bge|bm25` 使用现有 Search Backend；
- 通过 `--user-key` + `--memory-fixture` 绑定本地可信用户上下文；
- MCP Tools 是现有工具/证据/Skill 的薄适配层，不复制业务实现；
- 写入默认关闭；只有 `--allow-memory-writes` 的受控本地测试 server 才允许写入。

### 2.1 本地启动

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
.venv/bin/python -m recommendation.mcp.server \
  --user-key p14e2_lite_0049 \
  --memory-fixture data/p14e2/benchmark/dev.agent.jsonl \
  --search-mode bge \
  --allow-memory-writes
```

实际 stdout 由 MCP stdio 协议占用；manifest/日志写 stderr。

### 2.2 Tools

| MCP Tool | 真实实现 |
|---|---|
| `search_products` | `CatalogSearchSource`（BGE 或 BM25），真实 76,263 件 Electronics catalogue |
| `get_product_info` | `catalogue.inspect` + P15 `build_product_evidence`，保留 raw field/span/未知字段 |
| `recommend_products` | 现有 Rule Planner + `ElectronicsPlannerExecutor` + V0/Role/Eligibility/Gate |
| `get_active_preferences` | `HistoryStateReader` 读取启动绑定的授权用户 M9 active/inactive |
| `compare_products` | P16 `shopping-comparison` Skill |
| `request_preference_clarification` | P16 Clarification Ask，返回绑定用户的 handle |
| `submit_clarification_reply` | P16 Clarification Resume + 现有 M9 `process_turn`，要求 explicit local confirmation |

输入 Schema 由 SDK 从 type hints 生成；`top_k` 有 1–50 上限，`product_refs` 有 2–5 限制；**没有 `user_id` 参数**，没有文件路径、SQL 或代码执行参数。

## 3. 用户身份与 Memory 权限

- Server 启动配置固定 `user_key`；调用方不能传任意 `user_id` 改变读取/写入对象。
- `get_active_preferences` 只返回已绑定用户的 M9 状态；inactive/removed 单独返回，不混入 active。
- `request_preference_clarification` 生成 `clarification_id`，绑定 user key、query、创建/过期时间。
- `submit_clarification_reply` 检查：handle 存在、绑定 user、未过期、写入开关、`confirmed_by_user` 和 `confirmation_source=explicit_test_user_reply`。
- 跨用户 handle、无效 handle、过期 handle、未确认写入、只读 server 均返回 ToolError，而不是静默成功。
- **生产认证未实现**；本地测试使用显式标注的测试用户回复，不冒充真实线上用户；因此 `MEMORY_ACCESS_CONTROL = LIMITED`。

## 4. 真实 MCP Client E2E

`runs/p17/client_transcript.json`：

- 使用官方 `mcp.Client` + `StdioServerParameters` 启动独立子进程；
- 两个 server：
  - A：user `p14e2_lite_0049`，BGE，`--allow-memory-writes`；
  - B：user `p14e2_lite_0056`，BM25，写入关闭，短 TTL 用于过期测试。
- 26 次 client 调用；16 成功，10 个预期错误/边界（invalid ID、namespace、>5 products、duplicate refs、`top_k` 超限、unknown tool、unconfirmed write、write-disabled、expired handle、cross-user handle）。
- Discovery：7/7 tools；protocol `2026-07-28`。
- 主要成功结果：
  - search BGE：`score_kind=bge_dense`，真实 catalogue；
  - product info：真实 evidence span + UNKNOWN；
  - recommendation：`finalized`，Gate `ALLOW_FINALIZE`，selected `B004K6R43G`；
  - compare：P16 comparison；无效/role 场景返回明确状态；
  - clarification：awaiting handle；真实测试回复 `I prefer noise cancelling.` → M9 新增 active；同一 turn 重试 `already_processed=true`；
  - server B 的 write-disabled、过期、跨用户 handle 均被拒绝；
  - `get_active_preferences` 只显示各自 server 绑定的用户。

### 4.1 性能（同轮本地 CPU）

| 项 | 值 |
|---|---:|
| Server A startup + connect | 35.8 s |
| Tool discovery | 2.33 ms |
| BGE first search | 18.2 s（本地 CPU encoder；首发加载/编码） |
| BM25 search | 2.9 s |
| Recommendation | 148 ms |
| Comparison | 4.4 ms |
| Product Info | ~2 ms |
| Client calls | 26 |
| 预期错误 | 10 |

**没有 GPU/Qwen 调用**。BGE 首次 query 在 CPU 上较慢，本轮只报告实测值，不做无依据优化百分比。

## 5. 失败与边界案例

- `get_product_info`：
  - 不存在 ID → `product_not_found`；
  - 错误 Namespace → `namespace mismatch`；
  - 重复/缺字段 → SDK schema 校验。
- `compare_products`：
  - >5 件 → SDK `maxItems` 校验错误；
  - 重复商品 → `invalid_product_reference`；
  - 主商品类型不符 → `role_mismatch`，不把配件当主商品。
- `request_preference_clarification`：
  - 未回复保持 `awaiting_clarification`；
  - 无效/过期 handle 被拒绝；
  - 不同 server 的 handle 不能跨用户使用。
- `submit_clarification_reply`：
  - 只读 server → `memory_writes_disabled`；
  - 未确认 → `user_confirmation_required`；
  - 重复同一 turn → idempotent/no duplicate entry；
  - 模型生成的文本不会被当作真实用户回复。
- 客户端未知 tool / 非法参数返回明确 `is_error`，不静默成功。
- 商品文本没有进入任何 Qwen 生成路径；MCP layer 只返回/使用真实 catalogue 字段与 P15 证据。

## 6. 报告问题回答

1. **MCP Server 是否真正运行？** 是：两个独立 stdio server 进程，`MCPServer`，protocol 2026-07-28。
2. **是否有独立客户端成功发现和调用 Tools？** 是：官方 `mcp.Client(StdioServerParameters)`，7/7 tools，26 calls。
3. **是否直接复用原 AgentRec-X Core？** 是：现有 Catalogue/Search/P15/P16/Executor/M9，没有第二套 Runtime。
4. **商品与 Memory 来源是否仍然可追溯？** 是：product info 有 raw field/span/source；prefs 有 memory/source_turn/source_text，active/inactive 分离。
5. **两个 Skill 是否能通过 MCP 调用？** 是：`compare_products` 与 clarification 两工具均实际调用 P16。
6. **用户身份与写入权限是否得到验证？** 本地绑定验证和写入确认测试通过；但无生产身份认证，因此仅 **LIMITED**。
7. **哪些功能只适合本地可信环境？** 全部：stdio-only、固定 user key、memory fixture、test-only write enable；远程 HTTP/公开 Memory 未实现。
8. **是否具备后续演示或部署条件？** 本地 demo GO；公网部署前必须补充真实认证、授权、传输安全、限流/会话安全和 TLS，当前不满足。

## 7. 判定

| 判定 | 结果 | 说明 |
|---|---|---|
| `MCP_PROTOCOL` | **PASS** | 官方 SDK 2.3.0，MCPServer，protocol 2026-07-28，stdio，stderr 日志 |
| `TOOL_DISCOVERY` | **PASS** | 独立 client 发现 7/7 tools |
| `SEARCH_RECOMMEND_TOOLS` | **PASS** | 真实 BGE/BM25、Product Info、P15 Evidence、Rule/Executor/V0/Gate |
| `SKILLS_EXPOSURE` | **PASS** | P16 comparison/clarification 通过 MCP 实际调用 |
| `MEMORY_ACCESS_CONTROL` | **LIMITED** | 本地可信单用户 + 默认关闭写入 + 显式测试确认；无生产认证 |
| `END_TO_END_CLIENT` | **PASS** | 两个 server、26 calls、7 tools、完整错误/写入/过期/跨用户测试 |
| `LEGACY_REGRESSION` | **PASS** | P17 + P16/P15/P14-C1/D2/C0-R/Eval100 定向回归 49 passed |
| **`P17`** | **GO** | 本地 MCP 接口与独立客户端调用可复现；不代表公网部署安全条件 |

## 8. 交付

- `P17_REPORT.md`、`P17_SOURCE_DIFF.txt`
- `recommendation/mcp/server.py`
- `runs/p17/sdk_manifest.json`、`tool_schemas.json`
- `runs/p17/client_transcript.json`
- `runs/p17/metrics.json`、`decision.json`
- `tests/test_p17_mcp.py`：6 passed；P17 + P16/P15/P14-C1/D2/C0-R/Eval100 定向回归 49 passed
- 未训练模型、未重建 BGE、未运行 Reserved Test、未 commit/push；未操作 AutoDL 实例。

**GPU 结论：P17 不需要 GPU，也没有重新连接 GPU。** 后续只有需要真实 Qwen `/plan` 或 `/explain` 时才需要恢复 GPU/tunnel；MCP Server 本体的 stdio、Catalogue、BGE CPU、Memory 和两个 Skills 均可在当前 CPU 环境运行。
