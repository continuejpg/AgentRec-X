# Internship Demo Guide — 体系 A

> **目标：** 在一台普通 Windows + WSL2 开发机上，用真实 catalogue、真实 FastAPI BFF、真实
> MCP stdio、真实 M9/Behavior 和真实浏览器，完成一段可解释、可恢复的 Shopping Agent 演示。
>
> 当前正式展示只包含 **体系 A**。不启动 AutoDL、V2 Runtime、SASRec/Two-Tower/RRF Serving
> 或前端之外的新服务。两个 P1 开关默认 OFF，关闭时保持旧版兼容。

## 1. 真实启动顺序

### 1.1 推荐：一键启动（BGE，默认）

前置：仓库根目录 WSL `.venv`、Node.js + pnpm、Chrome/Edge。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
```

- Next.js: <http://127.0.0.1:3000>
- BFF: <http://127.0.0.1:8787/api/health>
- 停止: `powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1`

BGE 冷启动通常需要等待 `/api/health` 的 `ready=true`；两个 profile 会各自加载 catalogue/index。
不要因为首屏出现 `starting` 就重复启动第二个 BFF。

### 1.2 最小手动启动（BM25，适合快速演示）

```bash
# WSL / Linux，仓库根目录；MCP 由 BFF 子进程自动拉起
AGENTRECX_WEB_DEMO_SEARCH_MODE=bm25 \
AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1 \
AGENTRECX_PRICE_MODE=demo_historical_fx \
AGENTRECX_MEMORY_AWARE_RETRIEVAL=1 \
AGENTRECX_BEHAVIOR_AWARE_RERANK=1 \
.venv/bin/python -m web_demo
```

```powershell
# Windows，另一个终端
cmd /c "pnpm --dir web install"
cmd /c "pnpm --dir web dev"
```

Open <http://127.0.0.1:3000>. MCP 不是一个需要单独启动的 HTTP 端口；它由 BFF 按 profile spawn。

## 2. 本次演示配置

| 变量 | 值 | 说明 |
|---|---|---|
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | `bm25` 或 `bge` | 快速演示用 BM25；默认启动脚本用 BGE |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | `1` | 只对 write-enabled Demo B 生效 |
| `AGENTRECX_PRICE_MODE` | `demo_historical_fx` | **仅**用于笔记本预算演示；不是实时价格 |
| `AGENTRECX_MEMORY_AWARE_RETRIEVAL` | `1` | 打开 M9 辅助召回；默认 `0` |
| `AGENTRECX_BEHAVIOR_AWARE_RERANK` | `1` | 打开正式 Behavior 软重排；默认 `0` |
| `NEXT_PUBLIC_BFF_URL` | 可选 | 默认 `http://127.0.0.1:8787` |

`/api/health` 和 `/api/demo/config` 会显示 `price_mode` 与两个 `features` 布尔值，演示前先核对。
没有前端运行时开关；修改这些值后必须重启 BFF/MCP。客户端不能绕过服务端配置。

## 3. 三个主要场景（已验证输入）

### 场景 1 — 基础 Agent + M4 多轮 ASK + Budget Gate

**Profile：** `Demo B · Memory write`（`profile_0057`）
**价格模式：** `demo_historical_fx`

| 步骤 | 输入 |
|---|---|
| 1 | `推荐一台笔记本` |
| 2 | 选择“游戏”或输入 `游戏` |
| 3 | 输入 `8000 元` |

预期：

- 两次真实 clarification；第二次回复后走真实 `recommend_products`；
- 商品卡显示 `¥6,293.00 CNY`，并显示“演示参考价：历史美元价格按固定汇率换算，非当前市场售价”；
- Gate `ALLOW_FINALIZE`；Task Requirements 中包含 `use_cases=["gaming"]` 和 `budget=8000 CNY`；
- 不会把 `8000` 拼进 retrieval query，也不会把历史美元价当实时价格。

### 场景 2 — Memory-aware Retrieval

**Profile：** `Demo B · Memory write`
**开关：** `AGENTRECX_MEMORY_AWARE_RETRIEVAL=1`

1. 在左侧 **Memory lab** 输入 `I prefer USB.`，点击 **Remember**；
2. 等待 `M9 update: added ...`，确认 Active preferences 出现 `usb`；
3. 发送 `studio headphones`；
4. 查看商品区 **Personalization evidence** 的 `Memory retrieval` 条。

已验证的真实结果（BM25）：

```text
base=20, aux=10, merged=22
new_item_ids=2
extra_latency≈44 ms
```

BGE smoke：

```text
search_mode=bge
base=20, aux=10, merged=23
new_item_ids=3
extra_latency≈105 ms
```

不要求 Top-1 改变；展示新增 candidate IDs / 元数据即可。不要通过修改推荐文案声称 Memory 影响了召回。

### 场景 3 — Behavior-aware Rerank

**Profile：** `Demo B · Memory write`
**开关：** `AGENTRECX_BEHAVIOR_AWARE_RERANK=1`

1. 发送 `wireless studio headphones`，等待商品卡；
2. 在某张商品卡点击 **Like**（不要只打开 Behavior Shadow）；
3. 等待 Behavior Inspector / `/api/behavior/profile` 出现该商品；
4. 再次发送 `wireless studio headphones`；
5. 查看 `Personalization evidence` 的 `Behavior rerank` 条。

已验证的真实行为：

```text
active_product_signals=1
active_attribute_signals=1
candidates_with_behavior_signal=20
top_behavior_score≈0.233333
```

这是正式 `recommend_products` 的 Scorer 贡献；不是仅 Shadow Preview。**Top-1 可能不变**，
此时应如实说“分数发生变化，当前 Query 的 Top-1 未变”，不能伪造排序变化。

### 场景 4 — 安全与恢复（进阶）

- 当前任务硬约束优先于 M9/Behavior；`hard_unknown` 候选不进入推荐。
- 预算无法验证时 Budget Gate 拒绝，不用历史价格猜通过。
- 已提交 Reply 的 Replay 从 durable result 返回，不重新执行 MCP。
- M4 的 ASK 在两轮之间可 Cold Resume；测试可参考 `tests/test_web_multiturn_ask_e2e.py`。
- “杀进程”不是普通用户按钮；只作为进阶故障恢复演示。

## 4. 浏览器验收

### 4.1 已通过的真实浏览器路径

本轮实际运行：

```text
e2e/tests/09-internship-demo.spec.ts
  laptop ASK x2 + demo FX + Memory evidence + Like + Behavior evidence
  -> 1 passed
```

运行环境：真实 Windows Chrome（Playwright `channel="chrome"`）、真实 Next.js、真实 FastAPI BFF、
真实 MCP/BM25、真实 M9/Behavior；不是 mock。

同时运行的 CPU 浏览器场景：

- `02-comparison`：通过
- `03-clarification-memory`：通过
- `04-mobile`（2 个）：通过
- `05-modes`：通过

旧 `01`/`06` 浏览器断言仍假设旧 query 会直接 `ALLOW_FINALIZE`，与当前真实的
`NEED_USER_CLARIFICATION` 行为不一致；`07`/`08` 需要真实 Qwen GPU 服务，当前未启动。
因此不要声称整个旧 E2E suite 全部通过；以本轮通过的 CPU 场景和 09 为准。

### 4.2 如何运行 09

先按第 1.2 节启动带 flags + demo FX 的 BFF 和前端，然后：

```powershell
cmd /c "cd e2e && node_modules\.bin\playwright test 09-internship-demo.spec.ts --reporter=list"
```

若 `09` 被 skip，依次检查：

```powershell
curl http://127.0.0.1:8787/api/health
```

要求：

```json
"price_mode": "demo_historical_fx",
"features": {"memory_aware_retrieval": true, "behavior_aware_rerank": true}
```

### 4.3 截图

测试通过后，截图在：

```text
runs/web_demo/screenshots/desktop-internship-personalization.png
```

如果浏览器自动化不可用，只能报告 HTTP/页面/API 范围，不能声称完成真实浏览器操作。

## 5. Phoenix（可选）

Phoenix 不是启动前置条件。需要 trace 时：

1. 启动本地 Phoenix（默认 <http://127.0.0.1:6006>）；
2. 设置 `AGENTRECX_OTEL_ENABLED=1` 与 `AGENTRECX_OTEL_ENDPOINT`（仅本地演示）；
3. 重启 BFF；
4. 在 Phoenix 查看 `agentrecx.web_demo` / MCP / reply span。

没有 collector 时 `/api/health.observability.enabled=false`，UI 明确显示未启用，不伪造 trace。

## 6. 已知限制与排错

| 现象 | 处理 |
|---|---|
| `/api/health` 长时间 `starting` | 等 BGE 加载完成；不要重复启动；查看 `runs/web_demo/logs/` |
| 预算问题被拒绝 | 确认 `AGENTRECX_PRICE_MODE=demo_historical_fx`；strict 模式下 UNKNOWN 价格不会通过 |
| Memory evidence 没有出现 | 确认 Memory flag=1，BFF/MCP 已重启；M9 中是否有 ACTIVE 正向偏好 |
| Behavior 没有贡献 | 确认 Behavior flag=1；确认 Like 真正写入 `/api/behavior/events`，不是只点开 Shadow |
| Demo B 不能写 Memory | 确认 BFF 启动带 `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1` |
| 换了 profile/重启后旧 Memory 消失 | 默认 in-memory Session/M9 生命周期；使用 durable 路径的测试另行验证 |
| 旧 01/06 浏览器测试失败 | 旧断言已过期；以 09 + 02–05 为准，不要改成伪造 ALLOW |
| 07/08 Qwen 失败 | 未启动真实 GPU/Qwen；演示说明中明确标为可选/历史，不静默回退 |

## 7. 不要做的事

- 不要声称历史美元价是实时价格；
- 不要声称 Behavior 每次都改变 Top-1；
- 不要用修改文案冒充 Memory 召回；
- 不要为了演示打开客户端可控的 flag；
- 不要删除 `docs/development/PROJECT_SOURCE_AUDIT.md`；
- 不要启动或宣传 V2 Runtime 为当前能力；
- 不要为了提高演示效果伪造商品、价格、评分、Memory 或评测数字。
