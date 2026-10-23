> **审计快照（2026-10-16）。** 调用链按当时 HEAD 记录，**不是当前架构说明**；后续交付未回填。
> 当前调用链以 [`ARCHITECTURE.md`](ARCHITECTURE.md) 与工作树代码为准。

# 三条真实调用链（只读源码核验；中断处明确标注）

## 链 A：用户输入 → Memory 更新 → 下一轮读取
```text
POST /v1/demo/sessions/{sid}/chat
  → recommendation/api/demo_routes.py:chat
  → recommendation/demo/runtime.py（会话/运行时；默认 store = artifacts/demo/preference_memory.sqlite3，见 runtime.py:93,132-138）
  → recommendation/memory/service.py:231 PreferenceMemoryService.process_turn(...)
        · 抽取：memory/extraction.py（:158 :167 :172 :395 的引用/身份/指代守卫，避免把指代当取值）
        · 归一：memory/service.py:158 _normalise_span；:120 _supersedes（旧条目置 superseded）
        · 写入：PreferenceStore（SQLite，按 user_key/session 作用域）
  → 返回 TurnMemoryResult（service.py:96）/ MemoryUpdateSummary（:67）
  → 下一轮：demo/agent_service.py:553 project_active_preferences(memory/query_adapter.py)
        → 只暴露**价值级受限投影** {kind, polarity, value}（不含 memory_id/来源原文/其他用户）
```
**状态**：**贯通** ✓（`test_memory_service.py`、`test_memory_store.py`、`test_demo_llm_agent_memory.py`、`test_demo_llm_agent_memory_acceptance.py`）。
**关键点**：第 t 轮改变偏好后，第 t+1 轮读到的是**更新后的 ACTIVE 状态** ✓，而不是仅把话堆进对话历史 ✓（投影来自 store 的 active 条目 ✓）。

## 链 B：Memory → 商品检索 / 候选 / 排序
```text
1) 词法改写：agent_policy.py:156,166（use_memory）→ memory/query_adapter.py::project_active_preferences
             → control/catalog_search.py（BM25，真实目录 156,746 条）        ← L3，test_memory_bm25_retrieval.py
2) 候选过滤/重排：demo/agent_service.py:512-525 注入 preference_matcher / preference_reranker
             → memory_ranking（demo/memory_ranking.py）+ preference_matching/matcher.py:458（PRODUCT 身份）
                                                                             ← L3/L4，test_demo_memory_ranking.py
3) 属性证据：demo/product_knowledge.py + control/grounding.py:63,67,164,219（identity_map 契约）
             → KnowledgeEvidenceView{parent_asin,item_id,field_path,value,source='catalog_metadata'}
                                                                             ← 证据来自可信目录 ✓，非模型自述 ✓
```
**状态**：**贯通** ✓（检索改写 L3 ✓；排序影响 **L4** ✓，含 E-04A 真实 Provider 运行中 `k-black` 升至第 1 位 ✓）。

## 链 C：Memory → Agent Policy / Controller → 工具决策
```text
demo/runtime.py（llm_agent_turn）
  → demo/tool_selection.py:214（LLM 提议来源计划；真实 Provider 在 E-04A 验证 1 次 ✓）
  → ToolSelectionPlanner 校验（SourceSelection）→ demo/agent_service.py.run_selection
  → control/loop.py（LoopController：预算/终止；_plane_candidate_actions）
      → control/candidate_plane.py（CandidatePlane + CandidateLedger；agent_service.py:51-52,527 接入）
      → demo/observation.py（受限 Observation）→ demo/replan_policy.py（Gate + 至多一次追加）
  → Validator 允许后才执行 → 轨迹与指标（evaluation/agent/metrics.py:64,79-100,160）
```
**状态**：**贯通** ✓，但**链 C 中 memory 的接入点是受限投影**，且**动作选择不以 memory 为条件** ✗：
`agent_policy` 的动作集合由「请求声明的来源 + 预算 + 观测（候选是否充足/是否越界）」决定 ✓；
`use_memory` 只改变**词法查询** ✓，**不改变选哪个工具** ✗ ⇒ **"Memory → Agent 动作变化" 中断于策略层** ✗（无成对对照证据 = EVIDENCE_MISSING ✗）。
