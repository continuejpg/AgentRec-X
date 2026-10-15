# AgentRec-X V2 / E-02C-NARRATION-FIX — 最终 Chat 叙述与权威推荐卡片一致性修复

> 前序：[`docs/V2_E02C_REPORT.md`](V2_E02C_REPORT.md)（§10 记录了这个用户可见问题：`message` 的编号列表
> 来自 controller 运行内部、在累计候选/记忆重排/k 截断之前生成，可能与最终卡片不一致）。
> 结论：**PASS** —— `llm_agent` 推荐回合的 `ChatResponse.message` 现在只从该响应自己的最终卡片渲染，
> 商品集合、展示数量与顺序与 `recommendations` 完全一致，不再引用被 `k` 截断的商品，也不再复述
> controller 的中间排名；非推荐分支（direct / 无候选 / refused / failed）与全部旧模式保持不变。

---

## 0. 一句话结论

```text
修复前（E-02C 实测，多来源 + prefer black + k=4）：
  cards      : [j-black, j-red, j-blue, j-plain]
  message 列表: 1. j-red  2. j-blue  3. j-black  4. j-plain     ← controller 内部顺序
修复后（同一 turn）：
  cards      : [j-black, j-red, j-blue, j-plain]
  message 列表: 1. j-black  2. j-red  3. j-blue  4. j-plain     ← 与卡片逐项一致
k=2 时：修复前 message 仍列出被截断的 j-blue；修复后 message 只含 j-red / j-black（连 item id 702 都不出现）
```

## 1. HEAD 和初始工作树

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall   →  26 modified + 44 untracked
```

与 E-02C 报告结束状态一致。未 reset/checkout/restore/clean/stash，未删未跟踪文件，未装依赖，
未训练/加载模型，未调用在线 LLM，未改 Fast Serving `/v1/recommend`。

## 2. 旧 message 的准确来源与错误发生位置

| 事实 | 位置 |
|---|---|
| 叙述的**产生** | `AgentRecommendResponse.message` = `result.final_response`，由 accepted renderer 在 **controller 运行内部**（graph finalize）生成，依据是运行**当时**的候选/富化状态 |
| 旧文本进入最终响应的唯一位置 | `recommendation/demo/agent_chat_adapter.py:369`（修复前）：`text = message if message is not None else response.message` —— `llm_agent` 推荐分支从不传 `message=`，因此运行内部文本被直接采用 |
| 与之冲突的权威数据 | 同一次调用随后构造的 `cards`（`:365-367`），其内容来自 E-02A-V 的累计候选 → RRF → E-02A 的 Memory 重排 → `limit=payload.k` |
| 为什么单来源路径看不出问题 | 单来源时"运行内候选集合/顺序"与"最终卡片"相同；一旦发生多来源累计、偏好重排或 k 截断，两者分离 ⇒ 缺陷只在 `llm_agent` 的推荐分支出现 |

核查过旧叙述的实际语义（未假设它"只是编号"）：它包含——头部（"Top N candidate(s) from the sequential
recommender, ordered by …"）、逐候选行（`parent_asin`、`original SASRec rank`、ranking score、偏好匹配
计数、M8 富化出的目录事实行）与尾部免责声明。因此**不能**用正则替换编号修复（会同时改错商品名次、
遗留被截断商品、并把运行内目录事实重新带进响应），必须整段改为基于最终卡片的确定性渲染。

## 3. 修复策略、文件及 file:line

**策略**：在最终 ChatResponse 组装边界（adapter）做**唯一一次**确定性渲染，输入只有 `cards`。

| 文件:行 | 内容 |
|---|---|
| `recommendation/demo/agent_chat_adapter.py:90` | 新增 `NO_VERIFIED_REASON = "no verified catalogue reason for this request"`：未获验证 reason 的商品如实标注，绝不补写 |
| `recommendation/demo/agent_chat_adapter.py:93-146` | 新增 `recommendation_narration(cards)`：头部（数量 + "in the order this response presents them"）+ 逐卡一行（`序号. parent_asin (item item_id) - reason`，reason 为空则 `NO_VERIFIED_REASON`）+ 事实尾行（ranking score 与 `score_kind`、`fusion_sources`、M10A 三类计数）+ 仅当存在已验证 reason 时追加"理由已按该商品自身目录证据验证"说明。文档字符串明确写出：**不再复用运行内叙述**及其原因 |
| `recommendation/demo/agent_chat_adapter.py:426-437` | 组装分支：`message` 显式参数 > **有卡片 → `recommendation_narration(cards)`** > 无卡片 → `response.message`（空则用既有 `OUTCOME_MESSAGES[outcome]`） |
| `recommendation/demo/agent_chat_adapter.py:517,524` | 导出 `NO_VERIFIED_REASON` / `recommendation_narration` |
| `docs/USAGE.md`（llm_agent 段） | 记录新的叙述契约 |
| `tests/test_v2_narration_consistency.py`（**新增，12 项**） | §5 A–H 全部场景，均为 API 级执行测试 |

**未改动**：Controller/编排、RRF 与累计候选、M10A/M10B、RAG 检索与 Explanation Validator、
`ChatResponse` Schema（无新增字段）、deterministic/llm Chat、Agent Recommend、Serving。

## 4. Memory 排序前后 cards 与 message 对照（实测）

```text
同一 fixture / 同一请求（"RedWidget BlackWidget"），两个真实来源：
无 Memory      cards=[j-red, j-black, j-blue, j-plain]   message 列表=[j-red, j-black, j-blue, j-plain] ✔ 一致
prefer black   cards=[j-black, j-red, j-blue, j-plain]   message 列表=[j-black, j-red, j-blue, j-plain] ✔ 一致
               （修复前该轮 message 列表为 [j-red, j-blue, j-black, j-plain]，即 controller 内部顺序）
REMOVE         cards 回到 [j-red, j-black, j-blue, j-plain]，message 列表随之回到同一顺序 ✔
```

补充：`k` 与卡片字段不受影响 —— j-black 仍为 `original_rank=2 → reranked_rank=1`、
`sasrec_score=1.137496`、`score_kind="bm25"`、`fusion_sources=["catalog_search"]`，
`item_id`/偏好证据/知识证据/已验证 reason 全部逐字段不变（测试逐项断言）。

## 5. k 截断前后 cards 与 message 对照（实测）

```text
累计 grounded=3（history: [j-red, j-blue]，catalog_search: [j-black, j-red]），k=2：
修复前  cards=[j-red, j-black]   message 仍列出 "2. j-blue (…)"  ← 引用了被截断商品
修复后  cards=[j-red, j-black]   message 列表=[j-red, j-black]
        message 中 "j-blue" 不出现；"702"（被截断商品的 item_id）亦不出现
```

## 6. RAG 成功、失败与无证据情况（实测）

| 情形 | message 行为 |
|---|---|
| RAG 成功（已验证 reason） | 对应商品行直接引用该 reason（逐字，如 `Recommended because its catalogue details.Color is "black", which matches a preference you stated.`），并在尾部说明理由已按目录证据验证 |
| RAG 被拒（幻觉引用/跨商品/错值） | 所有商品行显示 `no verified catalogue reason for this request`；断言 message 中不出现 `GhostWidget`/`j-ghost`/`weight`/`material`/`premium` 等任何未经证实属性 |
| Provider 超时 | 同上（`NO_VERIFIED_REASON`），推荐卡片与顺序不变 |
| 商品无 Evidence Pack（如 j-blue/j-plain） | 该商品行显示 `NO_VERIFIED_REASON`，但仍按最终顺序列出，不失联、不补写 |
| RAG 关闭 | 不出现任何 `Recommended because` 文本；同时断言**旧叙述的特征串**（"from the sequential recommender"、"Original SASRec ranks are shown"、"Facts above are quoted from Amazon Reviews 2023"）均不出现，即"关闭 RAG"不会让 controller 的旧商品属性断言重新生效 |

## 7. direct / refused / failed 分支回归（实测）

```text
direct（"hello there"，direct plan）      recommendations=[]  outcome=direct   message 不以 "Top " 开头，非空 ✔
refused（lexical 计划零匹配）             recommendations=[]  outcome=refused  message 保持运行自带文本，非空 ✔
failed（engine 抛错 → 执行失败被拒）       recommendations=[]  outcome=refused  message 非空，candidate_count=0 ✔
非法计划（未授权来源）                     仍为 HTTP 502 llm_invalid_plan（无 ChatResponse 叙述可改）✔
```

## 8. 四种 Feature Flag 组合（实测）

| Memory Ranking | RAG | cards 集合 | message 列表 | 首位 | 偏好计数 | 商品知识证据 | 解释状态 |
|---|---|---|---|---|---|---|---|
| 关 | 关 | 4 件（同） | 与 cards 一致 | j-red | match=0 | 无 | 全部 `NO_VERIFIED_REASON` |
| 开 | 关 | 4 件（同） | 与 cards 一致 | j-black | match=1 | 无 | 全部 `NO_VERIFIED_REASON` |
| 关 | 开 | 4 件（同） | 与 cards 一致 | j-red | match=0 | 有 | 有 reason 的行引用之，其余 `NO_VERIFIED_REASON` |
| 开 | 开 | 4 件（同） | 与 cards 一致 | j-black | match=1 | 有 | 同上 |

四种组合下 `sasrec_score`/`score_kind`/`fusion_sources` 恒定（j-black 恒为
`1.137496 / bm25 / ["catalog_search"]`）；关闭 RAG 时四种组合的 message 均不含任何商品属性断言。

## 9. 所有回归命令及通过/失败数量

```bash
.venv/bin/python -m pytest tests/test_v2_narration_consistency.py \
  tests/test_v2_phase4_joint_acceptance.py tests/test_product_knowledge_rag.py \
  tests/test_demo_memory_ranking.py tests/test_multi_source_fusion_presentation.py \
  tests/test_demo_llm_agent_memory_acceptance.py tests/test_demo_llm_agent_memory.py \
  tests/test_agent_chat_adapter.py tests/test_demo_llm_agent_mode.py tests/test_agent_state_isolation.py \
  tests/test_agent_control_plane.py tests/test_llm_tool_selection.py tests/test_llm_selected_controller.py \
  tests/test_demo_agent_api.py tests/test_demo_api.py tests/test_demo_multiturn.py \
  tests/test_demo_llm_mode.py tests/test_demo_preference_pipeline_fixes.py \
  tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_provider_record_replay.py tests/test_demo_control_plane.py \
  tests/test_demo_recommendation_trace.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_control_stage2_candidates.py tests/test_constraint_eligibility.py \
  tests/test_candidate_reference_boundary.py tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
```

```text
995 passed in 83.16s        （36 个文件，0 failed；本轮新增 12 项一致性测试）
```

**三处既有断言按契约变更更新（不是为掩盖失败）**，每处都写明原断言固化了什么：

1. `tests/test_demo_llm_agent_mode.py:445` 原断言 `chat["message"] == reference["message"]`
   （"答案是 controller 的渲染文本，不是第二次渲染"）——**这正是缺陷本身**：该断言要求 chat 复用运行内
   叙述。已替换为更强断言：chat 的叙述逐项等于其**自己的卡片**、且不含 controller 中间叙述的特征串；
   同时保留卡片字段与 accepted endpoint 的逐项相等，并保留 accepted endpoint 继续使用运行内文本的事实。
2. `tests/test_agent_chat_adapter.py:155` 原断言 `chat.message == "ok"`（stub 的运行内文本直通）——
   属同一契约；已替换为"有卡片即渲染卡片"+ 显式 `message=` 覆盖仍生效（覆盖通道被保留并断言）。
3. `tests/test_v2_phase4_joint_acceptance.py:447` 原断言 `"j-blue" in body["message"]` —— 这是 E-02C
   **故意记录边界**的断言；本轮正是修复该边界，已翻转为 `not in` 并说明。

## 10. 最终 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall
开始：26 modified + 44 untracked
结束：26 modified + 46 untracked
```

本轮新增 2 个未跟踪文件：`tests/test_v2_narration_consistency.py`、`docs/V2_E02C_NARRATION_FIX_REPORT.md`。
本轮唯一改动的生产文件是 `recommendation/demo/agent_chat_adapter.py`（本就在 modified 集合中，故计数不变），
外加文档 `docs/USAGE.md`；其余为测试文件更新。前序 TIGER / Demo / M9 / M10 / E-01 / E-02 成果与 12 份文档
逐条确认仍在，无删除、无回退。

## 11. 最终结论

**PASS。** 完成标准逐项由真实 API 级执行测试支持：

```text
ChatResponse.message 与最终权威推荐卡片
  ├─ 商品集合一致（解析 message 的编号行 → (parent_asin, item_id) 集合 == 卡片集合）
  ├─ 展示数量一致（行数 == len(recommendations)，编号为 1..n 连续）
  ├─ 顺序一致（逐项相等，且等于 reranked_order）
  ├─ 不引用被 k 截断的商品（连 item_id 都不出现）
  ├─ 不引用未执行来源的商品（message 中每个身份都来自本轮呈现集合）
  └─ 不引入未经 RAG 验证的商品属性（无 reason 即 `NO_VERIFIED_REASON`，且不含旧叙述的目录事实文本）
```

同时：direct / 无候选 / refused / failed 分支的 message 与状态契约不变；四种开关组合下 cards 与 message
均一致；deterministic / llm Chat、Legacy Agent Recommend、Serving `/v1/recommend`、provider replay
全部不回归（995 passed / 0 failed）。本轮到此停止，不进入第 5 大阶段。
