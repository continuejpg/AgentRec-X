# Phase 4.3 — 端到端验收与版本收口

> 用真实浏览器（headless Chrome + CDP）对 `/demo/` 的多轮购物面板做端到端验收；
> 将 Phase 4 / 4.1 / 4.2 按功能边界整理为本地提交。
>
> **浏览器验收是实际执行的**；未执行的项目在 §5 明确标为未验证。

## 0. 结论

- 真实服务 + 真实浏览器通过 9 项检查：创建会话、首轮推荐、预算替换、冲突澄清、
  回答后重推、无法核实提示、过期/重复 id 的 404 处理、reset 清空、控制台/网络审计。
- 控制台 **0 个未处理异常**；购物期间 **0 次 `/chat` 请求**；只出现预期的
  `/agent/shopping` 调用和一次被页面处理的 404。
- 定向测试 **151 passed**；`node --check app.js` 通过。
- 4 个本地提交（3 个功能 + 本报告），工作树干净；未 push/merge。

---

## 1. 验收环境

| 项 | 值 |
|---|---|
| 服务 | `./scripts/start_demo.sh --port 8011`（真实 runtime：SASRec checkpoint + 完整目录） |
| 健康检查 | `/v1/demo/health` → 200 |
| 浏览器 | Google Chrome 155.0.8059.86，`--headless=new`，CDP `127.0.0.1:9223` |
| 驱动 | 仓库 `.venv` 的 Python + `websockets` 直连 CDP，无新增依赖 |
| 页面 | `http://127.0.0.1:8011/demo/` |
| 产物 | `runs/p4_3_browser_acceptance/report.json` + 8 张截图 |

驱动脚本不在仓库内（临时验收脚本）；截图和 JSON 报告在 `runs/`（gitignored）。

---

## 2. 真实浏览器结果

| # | 检查 | 实际结果 |
|---|---|---|
| 1 | 创建 session 并提交首次请求 | 页面自动创建 session；`I need a yoga mat, budget $30` → 2 条推荐 |
| 2 | 卡片顺序与理由归属 | 页面卡片 ASIN/顺序与 API `recommendations` 完全一致；每条 `.shop-reason` 文本与 API `reason` 逐字一致 |
| 3 | 修改预算后意图更新 | `make it under $50 instead` → intent 只剩 `price_max: 50`，`price_max: 30` 消失 |
| 4 | 硬约束冲突 | max price 50 + “my budget is at least $80” → 页面显示冲突问题，`#shop-cards` 中 0 张卡片，状态 “Waiting for your answer” |
| 5 | 回答后重推 | 同一 session 回答 `make it under $200 instead` → 问题消失，返回 2 条推荐，intent 含 max 200/min 80 |
| 6 | 无法核实 | 新 session + max price 0.01 + “I need a yoga mat” → `unverified` 问题，页面文本含 “could not verify”，无商品 |
| 7 | 重复/失效 id | 先用外部 HTTP 消费 pending，再由页面用旧 id 回答 → API 404 `unknown_clarification`；页面显示 “Clarification unavailable”，不自动重发，无卡片 |
| 8 | reset 清空 | 点击 Reset → 新 session；shop transcript 子节点 0、卡片 0、clarification 隐藏、intent 恢复占位、状态 “Ready” |
| 9 | 控制台/网络 | `Runtime.exceptionThrown` 0；console error 0（已过滤本次预期 404 资源日志）；`/chat` 请求 0；无意外请求 |

### 浏览器截图

```
runs/p4_3_browser_acceptance/01_ready.png
runs/p4_3_browser_acceptance/02_first_recommendation.png
runs/p4_3_browser_acceptance/03_budget_replaced.png
runs/p4_3_browser_acceptance/04_conflict_question.png
runs/p4_3_browser_acceptance/05_after_answer.png
runs/p4_3_browser_acceptance/06_unverified.png
runs/p4_3_browser_acceptance/07_stale_clarification.png
runs/p4_3_browser_acceptance/08_after_reset.png
```

截图已人工查看：02 显示推荐卡片/理由/intent；04 显示冲突问题且无卡片；06 显示
“could not verify” 文案；08 显示 reset 后面板清空。

### 浏览器验收中发现并修复的问题

1. **输入框 placeholder 与内容混淆**：placeholder 恰为示例句，截图检查时确认是
   占位符而非残留值；为避免浏览器自动填充把历史值写回，移除了 `name` 并加
   `autocomplete="off"`，reset 时显式清空 message/max price。
2. 未发现逻辑缺陷；其余检查一次通过。

---

## 3. 定向测试（实际执行）

```bash
pytest -q tests/test_p4_shopping_web.py tests/test_p4_shopping_api.py \
         tests/test_p4_shopping_intent.py tests/test_p4_clarification_policy.py \
         tests/test_demo_agent_api.py tests/test_agent_chat_adapter.py \
         tests/test_agent_control_plane.py tests/test_p3_constraint_empty_message.py \
         tests/test_p3_final_list_consistency.py tests/test_phase32_terminal_semantics.py
# 151 passed in 30.67s

node --check recommendation/web/app.js
# exit 0
```

覆盖：Phase 4 意图/帮助策略、Phase 4.1 购物 API（含 404/410/跨会话/重复）、
Phase 4.2 Web 契约与完整 HTTP 流程、Phase 3/3.1/3.2 的硬约束、最终列表一致性与终态语义。
未运行全量测试。

---

## 4. 本地提交

| Commit | 内容 | 文件 |
|---|---|---|
| `11d17b7` | Phase 4：意图、检索适应、澄清、逐商品理由 | `shopping_intent.py`、`agent_policy.py`、`agent_service.py`、`agent_schemas.py`、`loop.py`、P4 报告、评测驱动、2 个测试 |
| `89d0d41` | Phase 4.1：HTTP API | `api/demo_routes.py`、`demo/__init__.py`、API 测试、P4.1 报告 |
| `8866399` | Phase 4.2：Web 购物面板 | 3 个 web 文件、Web 测试、P4.2 报告 |
| 本文件所在 commit | Phase 4.3：验收报告 | `docs/P4_3_ACCEPTANCE.md` |

- 未 push / merge / rebase；未修改 Phase 2B/2C/3.x 的历史提交与 `runs/` 产物。
- `runs/` 全部 gitignored；`.venv`、`.venv-ranker`、Chrome profile 未入库。
- 提交后 `git status --porcelain` 为空（见最终报告）。

---

## 5. 未验证 / 明确限制

1. **410 过期未在浏览器中验证**：pending TTL 为 30 分钟，验收不等待；410 由
   `tests/test_p4_shopping_api.py::test_an_expired_clarification_is_410` 覆盖。
   浏览器只验证了 404（失效/重复）路径。
2. **“已完成但空结果”未走浏览器**：无法核实（unverified）已走浏览器；completed + 空列表的
   约束解释由 Phase 3 测试与场景集覆盖。
3. **无真实人工视觉验收**：headless Chrome + 截图检查，不是人眼在桌面浏览器中的完整交互；
   但 DOM、顺序、文本、网络、控制台均由真实浏览器执行验证。
4. **session / pending 是进程内状态**：多 worker 需 sticky session，重启不恢复；未新增数据库。
5. **无鉴权**：demo 不应直接暴露到不可信公网；页面内已提示。
6. **未跑全量测试**；未做跨重启/多 worker 测试。
7. 前端只提供 max price 输入；min/category/brand 依赖自然语言。

---

## 6. 可复现 Demo 步骤

```bash
# 1) 启动（真实 demo runtime）
./scripts/start_demo.sh --port 8000

# 2) 打开
#    http://127.0.0.1:8000/demo/
```

页面自动创建 session，在 **Multi-turn shopping** 面板：

1. `I need a yoga mat, budget $30` → Ask：卡片 + 理由 + intent（price_max 30）。
2. `make it under $50 instead` → intent 更新为 50，旧值消失。
3. Max price 填 `50`，`my budget is at least $80` → 冲突问题，无商品。
4. 清空 Max price，`make it under $200 instead` → 同 session 重新推荐。
5. Max price 填 `0.01`，`I need a yoga mat` → “could not verify” 澄清；清空 max 后回答
   `make it under $30 instead` → 推荐。
6. 重复用同一 clarification_id（或外部先消费）→ 页面提示 “Clarification unavailable”，
   不自动重发。
7. Reset session → 购物面板清空，新 session 就绪。

（过期路径需等待 30 分钟或依赖 API 测试，见 §5。）

---

## 7. 最终工作树

提交完成后：`git status --porcelain` 为空；分支 `feat/agent-source-limit-ablation`；
HEAD 为本报告 commit。`runs/p4_3_browser_acceptance/` 与其余 `runs/` 保留在本地、未入库。
