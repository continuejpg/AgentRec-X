# AgentRec-X M1 Engineering Baseline

> **Status: M1-C..M1-F baseline, recorded at `af07cc5` (2026-11-21).**
> This page is documentation only. M1-G added no feature, no frontend change and no demo
> script. It describes what the code and the committed tests actually support; it is not a
> production-readiness or recommendation-quality claim.

## 1. 当前版本与运行环境

| 项 | 实际值 |
|---|---|
| 仓库（Windows） | `D:\IT\CODE\PROJECTION\agentrec\X` |
| 仓库（WSL） | `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| 分支 / HEAD | `master` / `af07cc5411471f1810a25bfca320bb3060726f9c` |
| Python | 3.10.12（WSL `.venv`） |
| Web 后端 | FastAPI 0.141.1、Pydantic 2.13.5 |
| MCP | `mcp` 2.3.0，stdio 子进程 |
| 测试 | pytest 9.1.1 |
| 计算设备 | CPU；M1 流程不使用 GPU |
| LLM | 默认不启用 Qwen；M1 流程不依赖外部模型 |
| 前端 | `web/` 的 Next.js 应用是可选展示层，不在 M1 后端测试路径内 |

M1 的验收路径是 **CPU/BM25 + 本地 SQLite + 真实 C4 目录 + Raw Meta 历史价格 sidecar**，
不是线上服务，也没有实时价格、库存或可用性数据。

## 2. 主要模块与职责

| 路径 | 职责 |
|---|---|
| `web_demo/app.py` | BFF FastAPI 路由：health、session、chat SSE、clarification reply、replay、memory、behavior |
| `web_demo/state_store.py` | Durable Web SQLite（schema v6）：Session/Task/Pending/Turn/Request、ID 计数、claim/fencing |
| `web_demo/continuation.py` | Cold Resume 快照的构造与校验（外层 Web + 内层 MCP，v1） |
| `web_demo/reply_executor.py` | Reply Coordinator：handle 证明、claim、heartbeat、外部执行、checkpoint、finalize、replay、refusal |
| `web_demo/reply_assembly.py` | 从真实工具 payload 组装商品卡/narrative；不自行生成商品或价格 |
| `web_demo/viewmodels.py` | 前端 view model；`_price_view` 区分 UNKNOWN、live `present` 和 demo historical FX |
| `web_demo/mcp_manager.py` | 每个 profile 一个 MCP stdio 子进程；注入 `--price-mode` 等参数；启动/停止/健康检查 |
| `web_demo/config.py` | `Settings`、固定 profile、环境变量解析、Demo 价格模式开关 |
| `recommendation/mcp/server.py` | MCP 工具注册（9 个工具）、目录/记忆上下文、restore、reply、recommend |
| `recommendation/memory/` | M9 偏好抽取、process_turn、SQLite/Memory store（schema v1） |
| `recommendation/preference/` | History reader、Decision Gate、产品类型/角色、task requirements、价格预算判断 |
| `recommendation/catalog/` | C4/Raw Meta enrichment、`demo_price.py`、产品 metadata schema |
| `recommendation/control/` | BM25 `CatalogSearchSource`、P15 evidence、RAG evidence |
| `experiments/p14e/catalogue.py` | C4 `catalog_meta.jsonl` 适配、`search`、`inspect`、laptop/属性抽取 |
| `experiments/p14c0/electronics_executor.py` | Planner/Executor：检索、评分、角色/预算过滤、Gate、finalize/refusal |
| `recommendation/shopping_companion/planner_bridge.py` | 规则 Planner 与 Executor 的 action/observation 契约 |
| `recommendation/demo/planner_contract.py` | PlannerAction/Observation/Run 契约与错误边界 |
| `web/` | Next.js 展示层；消费 `/api/*`，不参与后端正确性判定 |

## 3. Session、Task、Pending、Request 的持久化关系

```
web_session 1 ── N web_task        (同一 session 至多一个 active task)
web_task    1 ── N web_pending     (同一 task 至多一个 OPEN pending)
web_pending 1 ── 1 web_request     (reply turn 的 claim/result ledger)
web_session 1 ── N web_turn        (turn_seq 单调；reply 行挂 task)
web_session 1 ── N web_identity    (turn/reply id 的序列，永不复用)
```

关键表（`web_demo/state_store.py`，`WEB_STATE_SCHEMA_VERSION = 6`）：

| 表 | 关键字段/约束 |
|---|---|
| `web_session` | `session_id` PK、profile、`turn_counter`、`reply_counter`、`reset_generation`、`retired` |
| `web_task` | `task_id` PK、`session_id` FK、status、`task_version`、`reset_generation`、`current_requirements_json`、`requirements_version` |
| `web_pending` | `pending_id` PK、`task_id` FK、`clarification_id`、`query`、`turn_id`、`reply_turn_id` UNIQUE、mode（`cpu_live`/`qwen_live`）、payload、status、expiry、`continuation_json`（v6） |
| `web_turn` | `(session_id, turn_seq)` PK、`turn_id` UNIQUE、`task_id`、payload envelope |
| `web_request` | `(session_id, request_id)` PK、pending/task、fingerprint、claim token、lease、`task_version_at_claim`、`reset_generation_at_claim`、`result_json`、`execution_stage` |
| `web_identity` | `(session_id, kind, value)` PK、`(session_id, kind, seq)` UNIQUE；turn/reply 计数 |

关键不变量：

- `ux_web_task_one_active`：一个 Web session 同时只能有一个 `RUNNING`/`WAITING_FOR_USER` task。
- `ux_web_pending_one_open`：一个 task 同时只能有一个 `OPEN` pending。
- `web_pending.reply_turn_id` 是 M9 reply 写入的幂等身份，唯一且由 ASK 时签发。
- `web_session.reset_generation` 在 reset 时递增；Task/Pending/Request 的 generation 不匹配即被 fencing。
- `web_request` 的 claim 记录 task version/reset generation；旧 claim 不能提交新 generation 的结果。

## 4. M9 偏好持久化与作用域

- 存储：`recommendation/memory/store.py` 的 `SQLitePreferenceStore`；`MEMORY_SCHEMA_VERSION = 1`。
- 表：`meta` + `preference_memory`；主键 `(user_key, memory_id)`，因此不同 user_key 的条目不可能碰撞。
- `user_key` 作用域：
  - Web 请求带 `session_id` 时，MCP `AppContext.memory_for(session_id)` 以 **Web session_id** 作为 M9 user key，保证不同 Session 的偏好隔离。
  - 无 session 的 benchmark/fixture 路径使用固定 `p14e2_lite_*` user key，并保留独立的内存 fixture store，避免污染 Web M9 DB。
- 幂等：`memory_id` 由 `(user_key, source_turn_id, kind, value, polarity)` 确定性派生；
  `process_turn` 依据 `source_turn_id` 做 already_processed 判定，重复 reply 不会重复写入。
- 语义：`REPLACE` 只在显式修正语句（如 `instead`/`make that`/`I meant`）下 supersede；普通
  `ADD` 不覆盖旧偏好。中文回复不进入 M9（M9 抽取器是英文规则）；中文“游戏/8000”属于
  task-scoped requirements，见第 7 节。
- 两个 MCP 子进程共享一个 M9 SQLite 文件；schema 初始化已改为幂等
  `INSERT OR IGNORE` + 重读，并有并发回归
  `test_two_stores_can_create_the_same_fresh_schema_concurrently`。
- M9 与 Web State **不是同一事务域**；不承诺跨库 exactly-once。

## 5. Warm Reply 与 Cold Resume

### 5.1 Warm Reply（进程仍活着）

`WarmReplyExecutor` 的顺序：

1. `MCPManager.check_handle(ref)` 证明 handle 仍属于当前 profile/session/process；
2. `claim_reply_request`（payload fingerprint + task version/reset generation）；
3. heartbeat lease；
4. `mark_request_external_started`；
5. `submit_clarification_reply`（真实 M9 写入，`reply_turn_id` 幂等）；
6. `get_active_preferences`；
7. `recommend_products`；
8. 成功时 `get_product_info` + BFF assembler；
9. `checkpoint_reply_result`（`RESULT_DURABLE`）；
10. `finalize_reply_request`（写 Turn、更新 Session/Task/Pending，之后才 HTTP 200）。

安全规则：没有裸 `warm_available=True`；handle 证明先于 claim；外部调用前必须已持久化
`EXTERNAL_STARTED`；`AMBIGUOUS` 不自动重跑；COMPLETED 只 replay，不重新调用 MCP。

### 5.2 Cold Resume（MCP 子进程已重启）

- ASK 时 `web_pending.continuation_json` 保存版本化快照：
  - 外层 `agentrecx.web.clarification_continuation.v1`：session/task/pending/clarification、
    generation、绝对过期时间；
  - 内层 `agentrecx.mcp.clarification_continuation.v1`：MCP task dict、question、query、expiry。
- 快照只包含 JSON，不含 handle id、进程对象、闭包或内存地址；旧 handle 永不复用。
- `restore_clarification_handle` 在**新进程**内重建无业务副作用的状态，并签发新 handle；
  校验必须在 claim 之前完成。
- 校验失败/缺失/过期/版本不符/绑定不符/generation 不符 → **HTTP 409
  `clarification_resume_not_ready`**，不猜测、不恢复。
- 范围限制：
  - 只支持 `pending.mode == "cpu_live"` 的 resume；
  - `qwen_live` 的持久化 pending 仍是 409；
  - 不承诺 Web SQLite 与 M9 的跨库 exactly-once。

> **M4 update (2026-11-22):** the M1 line `reply 后若再次需要 ASK -> 503` is superseded.
> A complete CPU/BM25 next ASK is now committed atomically as a new pending and returns HTTP
> 200 `awaiting_clarification`; the genuine uncertain-execution path still returns 503
> `AMBIGUOUS`. See [`M4_MULTITURN_ASK.md`](M4_MULTITURN_ASK.md). The historical M1 statement
> above is retained as the state at `af07cc5`.

## 6. Request Ledger、Checkpoint、Finalize、Replay

执行阶段（`web_request.execution_stage`）：

```
CLAIMED -> EXTERNAL_STARTED -> RESULT_DURABLE -> COMPLETED
                    \-> AMBIGUOUS (fail-closed; no automatic retry)
```

- **Claim**：`claim_reply_request` 根据 `(session_id, request_id)` 和 payload fingerprint 返回
  `claimed`/`reclaimed`/`replay`；claim token + lease 防止两个执行器同时写。
- **Fencing**：claim 时记录 task version/reset generation；提交时若不一致 → `RequestFenced`，
  请求失败，不写错误结果。
- **External started**：MCP 外部调用前写入；任何后续进程崩溃都只能进入 `AMBIGUOUS`。
- **Checkpoint**：`checkpoint_reply_result` 写 `result_json` 并置 `RESULT_DURABLE`；此时结果已可
  恢复，但尚未对用户返回 200。
- **Finalize**：`finalize_reply_request` 在 Turn/Session/Task/Pending 的权威更新完成后把请求置
  `COMPLETED`；HTTP 200 只在此之后返回。
- **Replay**：COMPLETED 请求直接返回已存结果，零 MCP；RESULT_DURABLE 只做 finalize，不再调用
  推荐工具；两者都不需要 handle。
- **AMBIGUOUS**：不确定外部是否已完成；返回 503，不自动重跑，避免重复 M9 写入。
- **Recommendation ID**：在 checkpoint 前只生成一次并写入 durable payload；Replay 复用同一个 id。

## 7. 用户需求合并与预算 Gate

### 7.1 Task requirements

`recommendation/preference/task_requirements.py`：

- 从用户原文/回复中只提取**显式**表述：`budget`（max/min、currency、source_text）、
  `category`、`use_cases`；无法匹配就不提取。
- 中文预算要求带 `元/块/人民币/RMB/CNY`；英文复用 budget/under/at least 等词；币种只从字面量
  读取，不推断。
- ASK 时从原 query 生成初始 requirements；reply 时把新 reply 合并到旧 requirements：
  **新 budget/category 覆盖旧的，use_cases 取并集**，原始 source_text 保留用于审计。
- 原始 5000 元文本保留在 `pending.query`/durable query 中；BM25 检索使用
  `strip_budget_expressions` 删除预算短语后，再追加 `laptop`/`gaming` 等有效英文词。
  因此检索文本是 `推荐一台 笔记本 laptop gaming`，不含 5000/8000；预算数字不进入 BM25。
- 商品型号中的数字（如 `Dell 5000`，没有 `元` 或 budget/price 关键词）不会被该规则删除。

### 7.2 预算 Gate

`experiments/p14c0/electronics_executor.py` + `evaluate_price_budget`：

- 每个候选的 price fact 被分类为：
  - `within`：币种一致且金额 ≤ max（或 ≥ min，闭区间：8000 ≤ 8000）；
  - `over`：超过预算；
  - `unknown`：无价格、无法匹配 ID、非法价格或无明确币种；
  - `currency_mismatch`：预算币种与价格币种不同。
- 只有 `within` 的候选可以成为最终商品；`unknown`/`currency_mismatch` 永远不会被当作合规。
- 如果 Gate 首选没有可验证价格或超预算，executor 在 role-compatible 候选里回退到 rank 最高的
  **verified within-budget** 候选，并写 `budget_verification.selection_note`；没有 verified
  候选时返回 `budget_unverifiable`/`budget_exceeded`。
- 这两种预算拒绝是**已知结果**：Coordinator 返回 HTTP 200、`recommendation: null`、
  `refusal` payload（无商品）、durable/replayable；不伪装成成功推荐。
- Gate/Planner 的既有语义未为了预算而改写；预算只是 finalization 的硬约束。

## 8. CPU/BM25 商品推荐与演示价格换算

### 8.1 真实目录与检索

- C4 accepted catalogue：**76,263** 个 item；原始 `catalog_meta.jsonl` 76,918 行中 655 行被
  loader quarantine。
- BM25 来源是 `CatalogSearchSource`（`recommendation/control/catalog_search.py`），候选池是有界
  top-K，不是全目录穷举；不把“有界候选池无结果”说成“全目录无商品”。
- 真实 BM25 `laptop gaming` top 20：**9 个**有历史 USD 价格；其中只有 **1 个主商品笔记本**有价
  且在预算内（MSI GF65，`B0BY2Z27LC`）；其余为键盘膜/充电器/电池/支架等配件，被 laptop
  role 过滤。

### 8.2 Demo historical FX（仅演示）

- 默认 `AGENTRECX_PRICE_MODE=strict`：C4 无 price → UNKNOWN；预算 Gate 拒绝无法验证的价格。
- `AGENTRECX_PRICE_MODE=demo_historical_fx` 才启用：
  - 通过真实 C4 `parent_asin`/`item_id` 关联 Raw Meta sidecar；
  - **25,736** 个商品有可解析历史 USD price（占 accepted catalogue 33.75%）；
  - `Decimal` + 固定 `1 USD = 7.00 CNY` + `ROUND_HALF_UP`，两位小数；
  - 换算后 ≤ ¥8,000 的商品 **25,441** 个；50,527 条 matched record 无价格，保持 UNKNOWN；
  - MSI GF65：`$899.00 -> ¥6,293.00`。
- 保存字段：`item_id`、原始 USD、换算 CNY、固定汇率、来源、`dataset_version`、
  `history_year=2023`（从 `amazon_reviews_2023:...` 来源确认）、`price_kind=historical_demo_converted`、
  `source_price_kind=historical_crawl_time`、`historical_time=None`（sidecar 没有 per-item 报价日期，
  不编造）和 `historical_time_note`。
- 商品卡/推荐说明的精确文案：
  **“演示参考价：历史美元价格按固定汇率换算，非当前市场售价。”**
- Strict 模式不加载该 index；BFF 把 `--price-mode` 显式传给子进程（strict 也显式传入），避免
  继承环境变量造成泄漏。
- 已提交的结果保存换算价格和汇率；用 strict 配置重启后 Replay 仍返回原 CNY/汇率，零 MCP，
  不重新计算。

## 9. 关键环境变量、数据库与测试入口

### 9.1 Web BFF

| 变量 | 默认/作用 |
|---|---|
| `AGENTRECX_WEB_DEMO_PORT` | 8787；host 固定 `127.0.0.1` |
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | `bge`（默认）或 `bm25`（CPU/测试用） |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | 默认 0；`1` 允许 profile_0057 的 M9 写入 |
| `AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S` | 180s |
| `AGENTRECX_WEB_STATE_DB` | Web State SQLite 路径；不设置则进程内 store |
| `AGENTRECX_MEMORY_DB` | M9 SQLite 路径（共享给两个 MCP 子进程） |
| `AGENTRECX_BEHAVIOR_*` | 可选 behavior event store/窗口参数 |
| `AGENTRECX_PRICE_MODE` | `strict`（默认）或 `demo_historical_fx` |
| `AGENTRECX_DEMO_PRICE_SIDECAR` | demo 模式使用的 Raw Meta sidecar 路径 |
| `AGENTRECX_METADATA_MODE` / `AGENTRECX_METADATA_SIDECAR` | 可选 Raw Meta enrichment；与 price mode 分开 |

### 9.2 MCP 子进程关键参数

`--user-key`、`--memory-fixture`、`--search-mode`、`--clarification-ttl-seconds`、
`--metadata-mode`、`--metadata-sidecar`、`--price-mode`、`--demo-price-sidecar`、
`--allow-memory-writes`、`--corrective-rag`、`--warmup`、`--evidence-verifier`。

### 9.3 SQLite / 日志

- Web State：`agentrecx.web.*` schema **v6**，表见第 3 节。
- M9：`preference_memory` schema **v1**。
- Behavior（可选）：本地 events SQLite；不属于 M9。
- 日志：`runs/web_demo/logs/bff.log`、`runs/web_demo/logs/mcp-profile_*.log`（路径由 log dir 配置）。
- `.sqlite`/`.sqlite3`、`__pycache__`、`.env*`（除 `.env.example`）、`runs/` 均在 `.gitignore` 中。

### 9.4 测试入口

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X

# 本轮 M1-G smoke（实际执行）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_task_requirements.py tests/test_budget_enforcement.py \
  tests/test_web_reply_executor.py tests/test_web_continuation.py \
  tests/test_demo_price.py::test_fixed_fx_conversion_is_decimal_with_two_decimals \
  tests/test_demo_price.py::test_invalid_or_missing_prices_stay_unknown \
  tests/test_demo_price.py::test_price_mode_is_strict_by_default_and_demo_only_when_explicit \
  tests/test_demo_price.py::test_strict_catalogue_does_not_expose_a_demo_price \
  -q -p no:cacheprovider

# 可选真实 MCP 集成（不在普通 smoke 中）
AGENTRECX_RUN_MCP_INTEGRATION=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_web_demo_integration.py -q -p no:cacheprovider
```

## 10. 最新已验证测试结果与实际覆盖范围

### 10.1 完整 M1 回归（历史证据，M1-G **未重跑**）

- 命令：33 个 M1 相关测试文件的 pytest 命令（具体文件列表见
  `docs/WARM_REPLY_HANDOFF.md` 第 3 节；包含 warm/cold resume、requirement update、
  demo price、continuation、M9、request ledger、replay、reset、product role 等）。
- 结果：**`369 passed in 857.14s`，exit code 0**。
- 运行对象：M1-F 收尾代码；该代码内容即当前 `af07cc5` 的功能代码（M1-G 只新增文档）。
- 重要：这是 **33 个 M1 相关文件**的结果，不是整个仓库 261 个 `tests/test_*.py` 的全绿声明。

### 10.2 本轮 M1-G 实际 Smoke（**重新运行**）

- 文件：`test_task_requirements.py`、`test_budget_enforcement.py`、
  `test_web_reply_executor.py`、`test_web_continuation.py`，以及 demo price 的 4 个纯函数测试。
- 结果：**`75 passed in 24.33s`，exit code 0**。
- 覆盖：需求合并/预算边界、executor 预算回退、reply coordinator 状态、Cold Resume snapshot、
  demo FX 换算/缺价/严格模式隔离。
- 未运行：真实 MCP E2E（`test_web_demo_price_e2e.py`、warm/cold real E2E）本轮没有重跑；
  上次完整 M1 回归的结果见 10.1。

### 10.3 历史 opt-in 结果

- `AGENTRECX_RUN_MCP_INTEGRATION=1 tests/test_web_demo_integration.py`：
  **`1 passed in 33.33s`**（M1-F 阶段，本轮未重跑）。
- Qwen live 相关 opt-in 不在 M1 范围；本轮未配置、未启用。

## 11. 代码与证据入口

- Warm/Cold、M1-C..M1-F 的详细证据：`docs/WARM_REPLY_HANDOFF.md`
- 当前工作区总览：`docs/ONBOARDING.md`（注意其坐标可能落后于 M1 分支/HEAD，以 `git log -1` 为准）
- 工程实践：`docs/ENGINEERING_PRACTICE.md`
- 测试：`tests/`（261 个 test 文件；M1 相关 33 个见 10.1）
- 关键代码：`web_demo/state_store.py`、`web_demo/reply_executor.py`、
  `web_demo/continuation.py`、`recommendation/mcp/server.py`、
  `recommendation/preference/task_requirements.py`、
  `experiments/p14c0/electronics_executor.py`、`recommendation/catalog/demo_price.py`。
