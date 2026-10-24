# Phase 4.2 — 多轮购物 Demo 前端集成

> 在现有 `/demo/` 页面加入 Multi-turn shopping 面板，复用同一个 demo session 与
> Phase 4.1 的 `POST /v1/demo/sessions/{session_id}/agent/shopping`。
>
> 前端只渲染后端返回的最终列表、理由、意图和澄清问题；不保存权威购物意图，不补造商品属性。

## 0. TL;DR

- 新增“Multi-turn shopping”面板：消息输入、可选 max price、澄清问题、当前意图、最终推荐卡片。
- 正常推荐、等待回答（冲突/无法核实）、无匹配、澄清过期、真正失败在状态条与卡片区分。
- 请求期间 `state.shop.sending` + 按钮禁用防重复提交；404/410 不清空后自动重试，只提示并允许新回合。
- 商品卡片严格按 API 顺序，理由逐商品显示；空结果显示后端约束解释；不使用 `innerHTML`。
- 前端变更 3 个文件（+376/−4）；新增 3 个静态/流程测试；受影响套件 **151 passed**；
  `node --check` 通过。Phase 4.3 已用真实 headless Chrome 完成浏览器验收（见 `docs/P4_3_ACCEPTANCE.md`）。

---

## 1. 前端改动

### 1.1 页面结构（`recommendation/web/index.html`）

在 Active preferences 与 Conversation 之间新增：

```html
<section class="panel shopping-panel" aria-label="Multi-turn shopping">
  ...
  <div id="shop-status" role="status"></div>
  <div id="shop-transcript" aria-live="polite"></div>
  <form id="shop-form">
    <input id="shop-message" ... />
    <label>Max price <input id="shop-max-price" type="number" min="0" step="0.01" /></label>
    <button id="shop-send" type="submit">Ask</button>
  </form>
  <div id="shop-clarification" role="status" hidden></div>
  <h3>Current intent (from the server)</h3>
  <div id="shop-intent"></div>
  <h3>Final recommendations (server order)</h3>
  <div id="shop-cards"></div>
</section>
```

面板内明确写明：session 与 pending 问题都在单个服务进程内、无鉴权、不可暴露到不可信网络。

### 1.2 样式（`recommendation/web/styles.css`）

新增 `.shop-*`、`.intent-chip.hard/soft/avoid`、`.shop-reason` 等；沿用既有 `.turn`、`.card`、
`.composer` 样式，支持深浅色。

### 1.3 客户端逻辑（`recommendation/web/app.js`）

新增状态（**只保存 UI 所需的最小状态**）：

```js
state.shop = { sending: false, pendingId: null, turns: 0 };
```

- `resetShopView()`：新 session / reset 时清空 transcript、cards、intent、pending；
- `setShopBusy()`：请求中禁用输入/按钮，并同步禁用 New/Reset session；
- `shopSend()`：
  - 读取 message、可选 max price（转为 `{kind:"price_max", value}`）；
  - 若有 `pendingId`，请求体带 `clarification_id`；
  - POST `/sessions/{id}/agent/shopping`；
  - 成功：渲染 `intent`、`clarification`、`recommendations[].reason` 与 `message`；
  - 失败：按 `session_not_found` / `unknown_clarification` / `clarification_expired` / 其它区分。

---

## 2. API 集成

```js
POST /v1/demo/sessions                       // 页面启动时已有，复用
POST /v1/demo/sessions/{id}/agent/shopping   // 新增调用
body = { message, k: 5, sources: ["history","catalog_search"], constraints, clarification_id? }
```

- 会话由页面顶部的 New session 创建；购物面板使用**同一个** `state.sessionId`。
- 前端不发送 `history` 或 `user_key`：服务器从 session 读取。
- `clarification_id` 只来自上一响应；回答成功/失败后按响应更新或清空，不手工拼接。
- 约束输入只有 max price；类别/品牌/价格等仍可由自然语言表达，由后端意图投影处理。

---

## 3. 状态区分

| UI 状态 | 触发 | 展示 |
|---|---|---|
| success | 200，`recommendations` 非空 | 卡片 + 理由；状态 “Recommended N product(s)” |
| waiting (conflict) | 200，`clarification.kind=conflict` | 黄色问题框；不展示商品；保留 `clarification_id` |
| waiting (unverified) | 200，`clarification.kind=unverified` | “catalogue could not verify a requirement” |
| empty | 200，列表空、`terminal.status=finished` | 卡片区显示后端约束解释；状态 “No product satisfied…” |
| failed | 200，`terminal.status=failed` | 状态条显示 `termination_reason`；不补造结果 |
| expired question | 410 `clarification_expired` | 提示并清空 pending；**不自动重发** |
| unknown/double answer | 404 `unknown_clarification` | 提示并清空 pending；不自动重发 |
| session expired | 404 `session_not_found` | 禁用输入，提示新建 session |
| other API failure | 4xx/5xx | 通用失败提示；无部分结果 |

- 请求期间 `sending=true` 且按钮/输入禁用，防重复提交。
- 404/410 只清空**本地** pending；不重复消费同一问题。
- 网络失败时不清 pending（问题可能仍有效），用户可重试；服务端一次性消费语义不变。

---

## 4. 展示与事实一致性

- `recommendations` 按 API 数组顺序逐项渲染，不排序、不筛选、不查目录补充字段。
- 每张卡片只显示 API 的 `rank`、`title`、`parent_asin`、`sources`、`reason`。
- `reason` 是后端按商品身份生成的证据文本；前端不做拼接、推断或改写。
- 空结果不显示“推荐卡片”，只显示后端 `message`（约束解释）。
- 意图 chips 只显示 `intent.hard/soft/conflicts`，不显示原始用户文本或未返回的字段。
- 全部动态文本用 `textContent`/`createElement`；`app.js` 中无 `.innerHTML`。

---

## 5. 部署与状态边界

- **同一服务进程**：session 注册表、pending clarification、购物意图都在进程内；
  多 worker 或重启后不保证可恢复（Phase 4.1 已记录 TTL=30 分钟、无持久化）。
- 页面提示中写明：**demo 无鉴权，不要直接暴露到不可信公网**；公开部署需另加访问控制。
- 本阶段未引入数据库，也未重构 session 存储。

---

## 6. 验证

### 6.1 前端检查

```bash
node --check recommendation/web/app.js   # exit 0
```

### 6.2 测试

新增 `tests/test_p4_shopping_web.py`（3 项）：

1. `/demo/` 包含全部购物控件 id；
2. `/demo/app.js` 调用 `/agent/shopping`、携带 `clarification_id`、处理 404/410、无 `.innerHTML`；
3. 完整 HTTP 流程：创建 session → 冲突澄清 → 回答 → 推荐 + reasons + intent（模拟页面调用序列）。

```bash
pytest -q tests/test_p4_shopping_web.py tests/test_p4_shopping_api.py \
         tests/test_p4_shopping_intent.py tests/test_p4_clarification_policy.py \
         tests/test_demo_agent_api.py tests/test_agent_chat_adapter.py \
         tests/test_agent_control_plane.py tests/test_p3_constraint_empty_message.py \
         tests/test_p3_final_list_consistency.py tests/test_phase32_terminal_semantics.py
# 151 passed
```

**浏览器验收由 Phase 4.3 完成**：headless Chrome + CDP 对 `/demo/` 执行了真实点击流程，
覆盖推荐、预算替换、冲突澄清、回答后重推、无法核实、失效 id 与 reset；结果见
[`docs/P4_3_ACCEPTANCE.md`](P4_3_ACCEPTANCE.md)。本文件记录的是 4.2 交付时的静态/HTTP 测试。

---

## 7. 可重复演示步骤

```bash
# 1) 启动本地 demo（需要已准备 .venv 与 runs/demo_catalog）
./scripts/start_demo.sh --port 8000
# 2) 浏览器打开
#    http://127.0.0.1:8000/demo/
```

页面会自动创建 session。然后在 **Multi-turn shopping** 面板：

1. **正常推荐**：输入 `I need a yoga mat, budget $30`，Max price 留空，点 Ask。
   - 期望：状态 `Recommended N product(s)`；卡片按 API 顺序出现；每条 reason 含
     `within your $30.00 budget (catalogue price $…)`；Current intent 显示 hard `price_max: 30`
     与 soft `category: yoga mat`。
2. **替换约束**：输入 `make it under $50 instead`，点 Ask。
   - 期望：intent 只剩 `price_max: 50`；旧值 30 不再出现；卡片理由随新预算变化。
3. **冲突澄清**：Max price 填 `50`，输入 `my budget is at least $80`，点 Ask。
   - 期望：黄色冲突问题；不展示商品；状态 “Waiting for your answer: the requirements conflict.”
4. **回答并继续**：清空 Max price，输入 `make it under $90 instead`，点 Ask。
   - 期望：问题消失；返回最终卡片；intent 同时含 max 90 与 min 80；理由同时说明
     `within your $90.00 budget` / `above your $80.00 minimum`（若适用）。
5. **无法核实**：Max price 填 `0.01`，输入 `I need a yoga mat`，点 Ask。
   - 期望：kind `unverified` 的澄清问题（“catalogue could not verify…”），无商品。
   回答 `make it under $30 instead`（清空 max price）→ 返回卡片。
6. **过期/重复回答**：同一 `clarification_id` 已回答一次后再次点 Ask → 红色提示
   “Clarification unavailable”；等待超过 30 分钟 → “Clarification expired”。两者都不会自动重发。
7. **无匹配**：Max price 填 `1`，输入 `I need a yoga mat`，点 Ask（若后端返回 completed 空结果）。
   - 期望：状态 “No product satisfied…”，卡片区显示后端约束解释，不显示伪造商品。

---

## 8. 完整演示叙事（一段）

> 打开 `/demo/`，选择一个 profile 并创建 session。我在 Multi-turn shopping 面板输入
> “I need a yoga mat, budget $30”。Agent 返回 5 个商品，每张卡片都带自己的目录理由，
> 当前意图显示价格上限 30 和软类别 yoga mat。接着我说“make it under $50 instead”，
> 意图立刻替换为 50，旧值没有残留，列表随预算变化。
>
> 随后我把 max price 设为 50，并说“my budget is at least $80”。Agent 没有硬凑结果，而是
> 返回一个冲突问题：最低价高于最高价。系统不展示任何商品。我回答“make it under $90 instead”，
> 同一个 `clarification_id` 被消费，意图更新为 80–90，重新返回商品和理由。
>
> 如果我把预算设成 0.01，Agent 会说“catalogue could not verify a requirement”，而不是说
> “没有这种商品”；这是 Phase 3.3 的“无法核实”语义。若我重复回答同一个问题，页面会提示
> 该澄清已不可用，并允许我重新发起，而不是自动重复消费。

---

## 9. 代码差异

```
 recommendation/web/app.js     | 275 +++++++++++++++++++++++++++++++++++++-
 recommendation/web/index.html |  45 +++++++++
 recommendation/web/styles.css |  60 +++++++++
 3 files changed, 376 insertions(+), 4 deletions(-)
新增: tests/test_p4_shopping_web.py
```

（`git diff --stat` 的累计值还包含 Phase 4 / 4.1 未提交的 API 与购物逻辑改动。）

---

## 10. 剩余问题

1. **浏览器验收**：Phase 4.3 已用 headless Chrome 验证；仍非人工桌面浏览器视觉验收。
2. **前端只提供 max price 输入**：min price、category/brand 等只能靠自然语言；约束冲突演示需要
   “max price + 文本 min”组合。
3. **UI 不持久化 transcript**：刷新页面会重建 session 视图；session 本身也随重启失效。
4. **pending 与 session 均为进程内**：多 worker 需 sticky session，重启不恢复（4.1 已声明）。
5. **无鉴权**：session id 即凭证，仅本地演示。
6. **404/410 后输入内容不自动重发**：用户需重新输入；这是刻意的“不自动重复消费”行为。
7. 未运行全量测试；未提交 Git。
