# WEB_DEMO_REPORT — AgentRec-X Web Shopping Assistant

> **判定口径（遵循 `AGENTS.md` 分层报告）**：
> - **工程正确性（engineering）**：本报告所有可跑项均在真实模型/真实 MCP/真实浏览器上验证；
> - **效果/质量（quality）**：**NOT ESTABLISHED**。本 Demo 不新增离线质量结论，不声称优于任何
>   基线，也没有线上 A/B、点击、转化或用户满意度证据；
> - **部署/安全**：仅 localhost 本地演示；**无认证/TLS/限流**，不构成公网生产就绪；
> - 本轮没有训练/微调、没有重建 BGE/索引、没有新 Agent Runtime、没有跑 Reserved Test、没有 commit/push。

## 0. 一句话总结

基于既有 AgentRec-X 能力，交付了一个**可本地一条命令启动**、真实调用 P17 MCP stdio Server
（7 个工具）、真实 BGE/BM25 + Rule Planner + Decision Gate + P15/P16 + M9 Memory 的
**Next.js 16 / React 19 / Tailwind 4 + FastAPI BFF** 购物助手：用户可用自然语言搜索、查看真实
商品与证据、选择 2–5 件比较、接收澄清问题、用真实输入触发 M9 偏好写入并继续推荐；Qwen 仅以
历史 Replay 出现且全程醒目标注。

验收命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
# 浏览器 http://127.0.0.1:3000（BFF http://127.0.0.1:8787/api/health）
```

## 1. 交付物 / 修改范围

### 1.1 新增

| 路径 | 内容 |
|---|---|
| `web/` | Next.js 16 前端（App Router，React 19，TypeScript strict，Tailwind 4），三栏桌面/单栏移动、SSE、商品卡、证据抽屉、比较、澄清、Memory/偏好、Replay、readiness；42 个文件 |
| `web_demo/` | FastAPI BFF：两个长生命周期官方 `mcp.Client`（stdio）、会话、SSE、view model、错误映射、health/重启/优雅关闭；9 个文件 |
| `tests/test_web_demo_unit.py` | BFF fake-manager 单测（29 tests） |
| `tests/test_web_demo_integration.py` | 真实 MCP 集成测试（BM25，env-gated） |
| `e2e/` | 独立 Playwright 项目（系统 Chrome，6 个测试文件 / 7 个测试） |
| `scripts/run_web_demo.ps1`、`scripts/stop_web_demo.ps1` | 一条命令启动/停止（WSL BFF + Windows 前端） |
| `scripts/run_web_demo_bff.sh` | WSL 内 BFF 启动（BGE/writes=1/PID/tee） |
| `scripts/run_web_demo_e2e.ps1` | 一条命令 E2E（启动→断言→停止→汇总） |
| `scripts/web_demo_source_diff.py` | 生成本报告配套 `WEB_DEMO_SOURCE_DIFF.txt` |
| `third_party/` | 上游 license 副本、复用指南、依赖 manifest、header 审计工具与结果 |
| `THIRD_PARTY_NOTICES.md` | Apache-2.0 归属与修改声明、19 个适配文件表、资产政策 |
| `docs/WEB_DEMO_CONTRACT.md` | 前后端共享契约（内部构建文档） |
| `WEB_DEMO_GUIDE.md` | 本地运行、三场景、测试、故障排查、限制 |
| `runs/web_demo/` | BFF smoke/集成转录、日志、E2E summary/REPORT、Desktop/Mobile 截图、trace |

### 1.2 既有文件唯一改动

`recommendation/mcp/server.py`（该文件在 worktree 中本就是 P17 未提交文件）：
- `confirmation_source` 的 `Literal` 增加 `"web_ui_explicit_user_reply"`，保留
  `explicit_test_user_reply` / `trusted_local_client`；
- 确认校验由 `== explicit_test_user_reply` 改为 `not in {显式测试回复, Web UI 回复}`；
- 仅 docstring/notes + `Literal`/确认校验集合的最小新增（不是重写）；**P17 旧测试 6 passed**
  （行为兼容）。
除此之外没有修改推荐/检索/Gate/Memory/Skill 任何核心文件。

### 1.3 未提交 / 基线

- 本轮开始前 worktree 已有 **332 个 untracked + 大量 modified**（前序里程碑），已在
  `runs/web_demo/baseline/` 记录 HEAD/分支/status/diffstat；本轮未 revert、未 stage、未 commit。
- 本轮结束 `git log` 仍为 `4dec6d2`，无新提交；`git status` 新增项全部属于上表 scope。
- 完整字节/SHA/唯一既有文件 diff：见 `WEB_DEMO_SOURCE_DIFF.txt`。

## 2. 架构与真实调用链

```text
Browser (Windows, Chrome/Edge)
  → Next.js 16 dev server 127.0.0.1:3000
      → HTTP + SSE (fetch + ReadableStream, keep-alive, AbortController)
  → FastAPI BFF (WSL Ubuntu-22.04, .venv/bin/python, 127.0.0.1:8787)
      → 2 × long-lived official mcp.Client (stdio)
          profile_0049 → recommendation.mcp.server --user-key p14e2_lite_0049 (read-only)
          profile_0057 → recommendation.mcp.server --user-key p14e2_lite_0057 (writes gated)
              → 真实 Electronics catalogue 76,263 + BGE dense / BM25
              → Rule Planner + V0 + Product Role + Eligibility + Decision Gate
              → P15 Evidence Bundle / P16 Comparison & Clarification / M9 Memory
```

- MCP Server 是**唯一工具边界**；BFF 不 import `recommendation.*` 核心模块，不读 catalogue，
  不重排、不补造、不填 UNKNOWN。
- 两个 MCP server 为**进程级长连接**，BFF 进程内复用；启动、健康、超时、异常、优雅关闭、
  日志（stderr 落盘）、`/api/demo/restart` 均在 `web_demo/mcp_manager.py` 实现。
- 前端只消费契约里的结构化事件/view model，不解析 P17 原始 JSON。

## 3. 问题回答（用户报告 K 的 11 问）

### Q1. 实际复用了哪些开源代码？

**一级复用（Apache-2.0，已复制/适配）**：`anthropics/commerce-agents`
（commit `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`）。
- 来源：`examples/web-shared/`（MessageBubble、Transcript、Composer、Suggestions、Inspector、
  icons、base.css 等）与 `examples/retail/storefront-web/`（ProductTile/ProductImage、ComparisonGrid、
  ProductCarousel、globals tokens、layout 组织方式）。
- 结果：**19 个适配文件**保留 `Copyright 2026 Anthropic PBC` + `SPDX-License-Identifier:
  Apache-2.0` + `Modified for AgentRec-X Web Demo (2026-11)` 说明；文件清单与修改摘要见
  `THIRD_PARTY_NOTICES.md` §3.1。
- **未复制**整个仓库、Claude Messages API Runtime、Claude Agent SDK、Anthropic 商品目录、
  Anthropic 模拟用户/Memory Store、商品图/品牌资产。

**仅设计参考（未复制代码）**：ShopGym（MIT，storefront 布局/响应式）、PAHF（MIT，偏好澄清/
Memory 交互）、TechJam（无 LICENSE，Fast/LLM path 与进度反馈）、Coveo（无 LICENSE，
Carousel/Comparison/NextActions/Tool Progress/事件分离）。逐仓分析见
`third_party/UPSTREAM_REUSE_GUIDE.md`。**无许可证仓库零复制。**

### Q2. 哪些代码是新写的？

`web_demo/`（FastAPI BFF、MCP 长连接管理、会话/SSE/view-model/demo flow）、`web/` 中 15 个
纯新模块（DemoApp、Header、Sidebar、RightPanel、ProductImage 中性占位、EvidenceList、
ProductDetails、ClarificationCard、ReadinessBanner、ReplayPanel、ComparisonBar、types/api/chat/
format/scroll/useMediaQuery 等，`THIRD_PARTY_NOTICES.md` §3.2）、`tests/test_web_demo_*`、
`e2e/`、启动/停止/E2E 脚本、`scripts/web_demo_source_diff.py`、`third_party/` 审计工具与
`WEB_DEMO_*` 文档。

### Q3. 第三方 License 是否正确保留？

**是，无保留项**（`ref-license-dev` 独立审计 + 复跑）：
- Apache-2.0 六项：逐字 LICENSE 副本（SHA-256 `3ddf9be5…` 与上游一致）、保留版权头、注明修改、
  上游无 NOTICE（已记录）、无商标/品牌素材使用、无附加限制；
- 未授权仓库零复制；
- `python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md`
  结果 **PASS (0 problems)**（报告 `third_party/header-audit.json`）；
- 依赖：`third_party/DEPENDENCY_MANIFEST.json` 枚举 npm 3 direct + 6 dev 与 Python 14 个运行
  依赖（mcp MIT、fastapi MIT、uvicorn BSD-3、pydantic MIT、Next/React/Tailwind MIT、
  TypeScript Apache-2.0 等）。License 审计仅静态检查，不判定代码同源性/视觉相似度。

### Q4. 是否真正调用了现有 MCP Server？

**是，真实 stdio、官方 `mcp==2.3.0`、protocol 2026-07-28，7/7 tools。**
- BFF 证据：`runs/web_demo/bff_smoke_20261109T104600Z.json`、`mcp_integration_transcript.json`、
  `runs/web_demo/logs/mcp-profile_*.log`（真实 manifest 与启动日志）。
- 工具发现：`search_products, get_product_info, recommend_products, get_active_preferences,
  compare_products, request_preference_clarification, submit_clarification_reply`。
- 每个 profile 首查 BGE **18–83s**（encoder 冷启动，E2E 最终实测 83.2s/57.0s），热查
  **0.15–2.1s**；recommend ~0.08–0.32s；compare ~4–5ms；product info ~2–24ms（实测随并发波动）。
- 未启动任何新 Runtime、未直接 import core。

### Q5. 是否能完成真实搜索与推荐？

**是。** 场景 1（profile_0049，BGE）：
- `search_products` 返回真实 catalogue 76,263，`score_kind=bge_dense`；10 张真实卡片；
- `recommend_products` 走既有 Rule Planner + Gate：实测 `status=finalized`、
  `gate.action=ALLOW_FINALIZE`、`reason_code=evidence_sufficient`、selected 真实商品
  （E2E 最终轮 `B004K6R43G`，对应 P18 Demo-1；隔离 smoke 曾为 `B08D32JM5F`，BM25 集成曾为
  `B00NB304UO`——结果随搜索模式/query 真实变化，不是 mock 固定值）；
- 卡片展示真实 ASIN/namespace、product type/form factor、retrieval source、Gate/role、
  可验证属性、UNKNOWN、`Price unavailable / UNKNOWN`。

### Q6. 是否能查看真实商品证据？

**是。** `get_product_info` 返回真实 P15 `product_evidence` span（raw field/span/span text/
source record）与 `unknown_fields`。实测：
- 6 条真实 evidence span（title/description/categories/…，`lexical_span_exact`）；
- `unknown_fields = price, brand, stock, material, size, warranty`，各带 reason；
- `semantic_truth_verified=false`，UI 明确标注「P15 自动校验 ≠ 人工语义审核」；
- 商品事实与用户偏好分开显示；无真实图片 URL → 自绘中性占位（不 hotlink 上游图片）。

### Q7. 是否能比较商品？

**是。** 场景 2（P18 已验证 pair `B004K6R43G` vs `B00008VF63`）：
- BFF → MCP `compare_products` → P16 Shopping Comparison：`status=comparison_ready`；
- 实测 `claims_valid=16 / claims_rejected=0 / unknown_count=12`；真实 attributes/UNKNOWN/
  preference relation/evidence source/conditional recommendation/final_answer 均渲染；
- 非法 ref（如 `B000NOTREAL`）：MCP 返回真实 `product_not_found`，BFF 映射 HTTP 400
  `invalid_product_reference`，UI 展示真实错误，不伪造比较结果。

### Q8. 是否能通过真实用户回复更新 Memory？

**是，且不是前端改标签。** 场景 3（profile_0057，写开启）：
- 模糊需求 → `recommend_products` 真实返回 `awaiting_clarification` /
  `gate.action=NEED_USER_CLARIFICATION` / `reason_code=insufficient_decision_criteria`；
- 浏览器输入 `I prefer noise cancelling and wireless.` → BFF POST 收到真实用户文本 →
  `submit_clarification_reply(confirmed_by_user=true, confirmation_source=
  "web_ui_explicit_user_reply", 幂等 turn_id)`；
- 真实 M9 结果：`update.added=2 / superseded=0 / removed=0 / skipped_duplicates=0`；
- 随后 `get_active_preferences` 真实刷新出 `wireless`、`noise cancelling` 两条 active；
  `recommend_products` 续跑真实返回 finalized；
- 幂等重试实测 `already_processed=true`；profile_0049 只读时真实拒绝
  `memory_writes_disabled`（UI 显示真实错误）。
- 边界：M9 为进程内 fixture-backed；BFF 重启即重置；无生产认证/多用户持久化。

### Q9. 是否区分 CPU Live 和 Qwen Replay？

**是。**
- 默认 **CPU Live Mode**（真实 BGE/BM25 + Rule Planner + V0 + Role/Eligibility/Gate +
  P15/P16 + P17 MCP，无 Qwen/GPU）；
- **Qwen Replay Mode**：只读取 `runs/p18/qwen_replay.json` 的既有真实轨迹
  （`replay=true, live_call=false`），UI 固定醒目徽标
  `Historical Replay — Not a Live Model Call`；不冒充实时推理；
- **Qwen Live Mode**：当前无可用服务，UI 显示 disabled + reason；不自动启 GPU、不占 GPU。

### Q10. 是否保留原有 Agent Core 和 MCP 安全边界？

**是。**
- BFF 只经 MCP 工具取数/写 memory；MCP server 内部仍是既有 Catalogue/Search/Planner/
  Gate/P15/P16/M9；
- 身份：前端只发 `profile_id`（服务端 allowlist），**不传 user_id/user_key**；写入确认只由
  BFF 在真实 POST 后构造，前端不可传 `confirmed_by_user`/`confirmation_source`；
- 跨 profile 隔离：两个独立 MCP server/两个 user key，测试覆盖 session 路由与
  `memory_writes_disabled` 拒绝；
- 失败不静默：MCP `is_error`/ToolError/readiness/非法 ref 全部映射为真实错误；
- 无新 Runtime、无 core 语义改动（唯一既有文件改动见 §1.2），无公网 Memory 暴露。

### Q11. 用户能否在本地一条命令启动整个 Demo？

**能，已实测。** `powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1`：
- BFF 两 profile 并行启动，实测 ready ~32–48s（BGE），前端 3000 ready，自动打开浏览器；
- 打印每 profile `startup_ms/search_mode/writes_enabled`；
- 停止：`scripts/stop_web_demo.ps1`（实测释放 3000/8787，无 WSL 残留）；
- E2E：`scripts/run_web_demo_e2e.ps1`（启动→真实浏览器→停止→summary）。

## 4. 测试与验证结果

> 只跑与 Web Demo/P17/P16/P15 相关的定向测试；**未跑全仓库 pytest、未跑 Reserved Test**。

| 验证 | 命令 | 结果 |
|---|---|---|
| BFF 单测 | `.venv/bin/python -m pytest -q tests/test_web_demo_unit.py` | **29 passed**（fake manager：session/profile 隔离、SSE 顺序、取消不写脏状态、422 注入拒绝、writes-disabled、reply 构造、replay） |
| 真实 MCP 集成 | `AGENTRECX_RUN_MCP_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_web_demo_integration.py` | **1 passed**（真实 BM25；两 server ready ~25.6s；7/7 tools；catalogue 76263；clarification added=2；幂等；只读拒绝） |
| P17 旧合同 | `.venv/bin/python -m pytest -q tests/test_p17_mcp.py` | **6 passed** |
| 前端类型 | `cmd /c "pnpm --dir web exec tsc --noEmit"` | 0 errors |
| 前端构建 | `cmd /c "pnpm --dir web build"` | 0 errors（Next 16.4.5 / React 19.3.1 / Tailwind 4.3.5） |
| License/header 审计 | `python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md` | **PASS (0 problems)** |
| 启动器 | `scripts/run_web_demo.ps1 -NoBrowser` | BFF ready 32.1s / 32.1s；前端 200；真实 SSE 10 卡 + Gate + 6 evidence + UNKNOWN；stop 后端口释放 |
| E2E（真实浏览器） | `scripts/run_web_demo_e2e.ps1` | **PASS：7/7 tests passed、64 项 evidence checks 全过、8/8 截图尺寸正确、cleanup 后 3000/8787 无监听**；Playwright 157.8s；详见 `runs/web_demo/e2e/REPORT.md` |

**E2E 最终实测（真实栈，零 mock）**
- Demo 启动到 `/api/health` ready：**46.7s**（两个 MCP server 分别 **35.6s / 35.5s** startup，catalogue 76,263，7/7 tools）。
- 冷查询（每个 profile 首次 BGE encoder 懒加载）：**83.2s / 57.0s**；热查询 0.19–2.1s；比较 HTTP round-trip 155ms；澄清回复 280ms。
- Scenario 1：SSE 200；10 张真实卡（含 `B004K6R43G`），每卡 `Price unavailable / UNKNOWN`（10/10），Gate `ALLOW_FINALIZE/evidence_sufficient`，selected `B004K6R43G`；evidence 抽屉 4 条真实 P15 span（`pev_01` raw_field title 0–52 + 原文 + source_record catalog_meta.jsonl + UNKNOWN + `semantic_truth_verified=false`）。
- Scenario 2：`comparison_ready`；dimensions=`title, price, brand, material, stock, warranty`；真实 claims valid=16/rejected=0/unknown=12；conditional=null 时 UI 显示 "NOT STATED / no text"；非法 ref → 真实 HTTP 400 `invalid_product_reference`。
- Scenario 3：真实 clarification → reply → M9 `added=2` / `web_ui_explicit_user_reply`；preferences 面板与 `GET /api/preferences` 均出现 wireless + noise cancelling；续推荐真实卡 `B07HKQLKGS`。
- Mobile：390×844 单栏、Drawer（≤340px）、comparison sheet（≤430px）、composer 固定底部。
- Modes：CPU Live 默认 `aria-pressed`；Qwen Replay 醒目徽标 + `live_call=false` + 8 条真实轨迹（含 final-answer-only 不再崩溃）；Qwen Live disabled + 真实 disabled_reason。
- 真实失败：无 pending handle 的 clarification reply → 409 `clarification_not_pending`；注入 `confirmed_by_user/user_id` → 422（extra=forbid）；非法 compare ref → 400；只读 turn 不渲染任何 memory 成功块。

**集成期发现并修复的真实缺陷（都已复验）**
1. **Next.js 16 dev 源限制**：页面经 `127.0.0.1:3000` 打开时，dev 资源（HMR/覆盖层）被 Next 拒绝，页面只有 HTML 不 hydrate，浏览器侧零 BFF 调用。修复：`web/next.config.ts` 增加 `allowedDevOrigins: ["127.0.0.1", "localhost"]`；修复后 Playwright 实测 hydrate 并真实请求 health/config/sessions/preferences。
2. **ReplayPanel 空 `terminal` 崩溃**：`runs/p18/qwen_replay.json` 的 8 条真实轨迹中 3 条是 final-answer-only（无 terminal/gate）；旧类型把 terminal 当必填导致切换 Replay 整页崩。修复：`web/lib/types.ts` 将 terminal/gate 改为可选可空、补 `final_answer`；`web/components/ReplayPanel.tsx` 对缺失 terminal 显示 "final-answer-only"/"terminal: not recorded"，并原样展示真实 final_answer，不伪造 selected/ask；E2E 复验 8 条轨迹均可浏览且不崩。
3. E2E harness 侧修正（非产品代码）：`activity-line` 是持久工具历史而非 streaming 指示，改用 composer 重新 enabled + running tool row 作为完成信号；比较面板表头被 CSS 大写，断言改为大小写不敏感并限定 `comparison-view`；profile_0049 fixture 自带 3 条 prefs，模糊 query 实测全部 finalize，故写禁用的 *clarification UI* 路径不可达，改在 BFF 边界断言真实 409/422/400。

E2E 断言范围（不靠截图宣告通过）：真实 SSE 事件序列、真实 cards/ASIN/namespace、每卡
`Price unavailable / UNKNOWN`、details/evidence 真实 span/raw field/source record/UNKNOWN、
P16 compare 200 与 DOM 数据交叉验证、非法 ref 真实 400、profile_0057 clarification→
`web_ui_explicit_user_reply`→added=2→preferences 面板刷新→续推荐、mobile 单栏/Drawer、
Qwen Replay 徽标/Qwen Live disabled、真实失败不伪造成功。

## 5. 性能实测（CPU，无 GPU；E2E 终值）

| 指标 | 实测 |
|---|---|
| MCP server cold start（并行两 profile，E2E 最终轮） | **35.6s / 35.5s**（更早隔离轮 26.4–37.8s；P18 基线 34.2–40.4s） |
| Demo 启动到 `/api/health` ready | **46.7s**（含前端启动与两 server 并行） |
| BFF `/api/health` 首查可用 | 启动后 ~2s 即返回 `starting`（页面先渲染） |
| 首个 BGE search（每 profile） | **57.0–83.2s**（encoder + 首次 encode 冷启动；随机器负载/并发波动） |
| 热 BGE search | 0.19–2.1s（P18 基线 0.149s） |
| recommend_products | 0.08–0.32s |
| get_product_info（单条） | ~2–24ms |
| compare_products（2 件） | ~4–5ms（页面 HTTP round-trip 155ms） |
| clarification reply（含 M9 写入） | ~280ms |

UI 用真实 tool timeline + SSE `: keep-alive`（5s）展示长任务，不把冷启动伪装成失败/瞬时成功；
未做无依据的百分比优化声明，未重建索引/模型。

## 6. 截图与轨迹

- Desktop（1440×900）：
  - [desktop-search.png](runs/web_demo/screenshots/desktop-search.png)（真实 10 卡 + Gate + UNKNOWN）
  - [desktop-evidence.png](runs/web_demo/screenshots/desktop-evidence.png)（P15 原文 span + UNKNOWN）
  - [desktop-comparison.png](runs/web_demo/screenshots/desktop-comparison.png)（P16 dimensions/claims）
  - [desktop-clarification-memory.png](runs/web_demo/screenshots/desktop-clarification-memory.png)（Memory Update Added=2）
- Mobile（390×844）：
  - [mobile-search.png](runs/web_demo/screenshots/mobile-search.png)
  - [mobile-drawer.png](runs/web_demo/screenshots/mobile-drawer.png)
  - [mobile-comparison-sheet.png](runs/web_demo/screenshots/mobile-comparison-sheet.png)
- 模式：[replay-mode.png](runs/web_demo/screenshots/replay-mode.png)（Historical Replay 徽标）
- 轨迹：`runs/web_demo/traces/`（retain-on-failure；全 PASS 故无 trace.zip，符合设计）；
  E2E 汇总：[summary.json](runs/web_demo/e2e/summary.json)、
  [REPORT.md](runs/web_demo/e2e/REPORT.md)、`evidence.jsonl`、`payloads/`。
- 真实后端/工具证据：`runs/web_demo/bff_smoke_*.json`、`mcp_integration_transcript.json`、
  `runs/web_demo/logs/mcp-profile_*.log`。

## 7. 已知限制（如实）

1. **质量/线上效果 NOT ESTABLISHED**：没有真实用户 A/B、点击、转化、满意度或偏好条件相关性
   标签；构造场景/本地 Demo 不等于线上购物成功率。
2. **商品证据边界**：catalogue snapshot 无可信价格/库存/评分/图片；UNKNOWN 不填；无图片 URL →
   中性占位；P15 校验为 automatic validation，不是人工语义审核。
3. **首查延迟**：CPU BGE encoder 冷启动导致每个 profile 首次搜索 18–45s；Demo 用真实状态展示，
   不假装优化；无 GPU/Qwen 实时链路。
4. **Memory 为本地进程内存**：fixture-backed M9，BFF 重启即重置；无生产认证、无多用户持久化、
   无 TLS/限流；仅 127.0.0.1。
5. **Qwen Replay 只是历史轨迹**：不是实时模型；Qwen Live 未实现/未启用。`runs/p18/qwen_replay.json`
   中有 3 条 final-answer-only 轨迹没有 `terminal` 字段（只有 `final_answer`），UI 以
   "terminal not recorded" 诚实展示，不伪造状态；这是真实历史数据形状，不是模型输出。
6. **无购物车/结算/订单/支付/库存**；没有实现公网部署。
7. **前端为本地 Demo**：不声明生产级可访问性/国际化/埋点/SEO；移动端 Sheet/Replay 的最终
   浏览器证据以后端/前端构建 + E2E 报告为准。
8. **License 审计为静态 header/清单核对**，不判定代码同源或视觉相似度；若 `web/` 适配文件再
   变更需重跑审计。
9. **一个已知真实并发边界（E2E 集成期发现，未在正常串行 UI 触发）**：MCP server 的本地 BGE
   encoder 是首次查询时懒加载的，若首个 `search_products` 在 encoder 加载过程中被客户端中止，
   紧接着同一 server 再收到 search，`experiments/p14c1/search_backend.py:68` 存在
   `meta device tensor → numpy` 竞态。正常串行 Web UI 不会触发；BFF 未修改该 P18 核心代码，
   仅记录为已知限制（不在本轮范围内修核心）。
10. **只读 profile 的 clarification UI 路径不可达**：`profile_0049` fixture 自带 3 条 active
    preferences，实测 7 条模糊 query 全部返回 `ALLOW_FINALIZE`，因此无法在当前 demo 中通过 UI
    触发 write-disabled 的 clarification 回复。写禁用边界已在 MCP/集成测试与 BFF 层验证
    （真实 `memory_writes_disabled`、409 `clarification_not_pending`、422 身份注入拒绝），
    UI 层的只读 refusal 路径作为已知限制记录，不伪造。

## 8. 判定

| 层 | 判定 | 说明 |
|---|---|---|
| 工程正确性（本地） | **PASS** | 真实 MCP 7/7、真实 BGE、三场景、SSE/错误/身份边界、单测+集成+前端 build+E2E |
| 开源复用/License | **PASS（无保留项）** | Apache-2.0 义务满足；无证仓库零复制；依赖 manifest 完整 |
| 安全/隔离 | **LIMITED** | 本地单机可信上下文；无认证/TLS/限流；写入只对显式 POST + 受控 profile 开启 |
| 效果/线上质量 | **NOT ESTABLISHED** | 无 A/B/点击/转化/满意度证据；不声称优于基线 |
| 公网生产就绪 | **NO** | 仅 localhost 本地演示 |

**结论：WEB DEMO 本地可用（engineering PASS），但质量与生产就绪度未建立；所有“真实”声明都以
真实 MCP 调用、真实浏览器 E2E 和可复现脚本为依据，不以 mock 或截图冒充。**
