> **审计快照（2026-10-16）。** 结论来自源码与当时的执行证据，**不是当前架构说明**；后续交付未回填。
> 当前入口见 [`ONBOARDING.md`](ONBOARDING.md)。

# 简历声明真实性判断（仅依据源码与执行证据）

## A. "构建结构化用户记忆与多轮偏好更新机制。"
**判定：SUPPORTED** ✓
依据：偏好本体（`memory/schemas.py:78,105,112,134`）+ 服务与持久化（`memory/service.py:168,231`；SQLite `preference_memory.sqlite3`，路径可由 `AGENTRECX_MEMORY_DB` 配置）+ supersede/撤销（`service.py:120`、`PreferenceMode.remove`）+ 来源证据与敏感信息拦截（`schemas.py:171,201,211`）+ 跨用户隔离（`tests/test_agent_state_isolation.py`）+ 测试（`test_memory_service.py`、`test_memory_store.py`、`test_demo_llm_agent_memory*.py`）。
**限定**：抽取为**确定性规则 + 守卫**（`memory/extraction.py`），不依赖在线 LLM ✓；"多轮"的端到端证据来自受控/离线用例 ✓。
**建议表述**：*"设计并实现结构化用户偏好记忆（多种偏好类型、极性、add/replace/remove 与 supersede 状态机），以 SQLite 持久化并按用户/会话隔离，保留来源片段并拦截敏感信息。"*

## B. "将用户记忆用于商品检索与候选排序。"
**判定：SUPPORTED** ✓（检索 L3、排序 L4）
依据：`memory/query_adapter.py::project_active_preferences` → 词法查询（`agent_policy.py:156,166 use_memory`）✓，测试 `test_memory_bm25_retrieval.py`、`test_memory_query_adapter.py`；排序侧 `demo/agent_service.py:206,246-249,512-525` + `demo/memory_ranking.py`，测试 `test_demo_memory_ranking.py`、`test_demo_llm_agent_memory_acceptance.py`；**并有真实模型 + 真实 Provider 的一次端到端证据**：生效偏好投影 `[{color: black}]` 到达规划上下文，最终排序中 `k-black` 由第 2 位升至第 1 位 ✓。
**限定**：验证环境为**真实模型 + 1 次真实 Provider 调用**（额度已用尽 ✗）；**无**线上流量/规模证据 ✗。
**建议表述**：*"把生效偏好投影折入 BM25 词法查询与候选重排（同候选池对照下排序发生预期变化），并在一次真实 Provider 端到端运行中验证偏好商品升至首位。"*

## C. "基于用户记忆和工具预算实现受约束的 Agent 动态规划。"
**判定：PARTIALLY_SUPPORTED** ◐ —— **"工具预算 + 受约束"成立 ✓；"基于用户记忆"不成立 ✗**
成立部分：来源计划由 LLM 提议后经 `ActionValidator` 校验 ✓（E-04A 真实 1 次：`["catalog_search","history"]` 通过并执行 ✓）；预算与终止由 `control/loop.py` 强制执行 ✓（P0：18 例 100% 合规、0 约束违反 ✓）；有界重规划**至多一次** ✓（`observation.py`+`replan_policy.py`，16+24 项测试 ✓；E-04A-R1 真实 1 次：追加来源执行、工具 2/4、步数 3/6 同一预算 ✓）。
不成立部分 ✗：策略层的**动作选择不以 memory 为条件**（`use_memory` 仅影响词法查询 ✗）；**缺少成对对照证据** ⇒ "Memory → Agent 动作变化" 记为 **EVIDENCE_MISSING** ✗。
**建议替代表述（准确且不失分量）**：
*"在显式工具/步骤预算与动作合法性校验下实现 Agent 工具选择与**至多一次**的有界重规划（工具提议—校验—执行—验证四段分离，含真实 Provider 一次性验证）；用户记忆用于检索与重排，而非直接改写规划。"*

## 简历红线（不得写入）
* "用户记忆驱动 Agent 规划/工具选择" ✗（EVIDENCE_MISSING）
* "真实 LLM 持续在线闭环" ✗（额度 4/4 用尽，仅一次性验证）
* "线上/大规模推荐质量提升" ✗（P3 = 固定前缀 n=1,000，exploratory）
* "TIGER 生成式推荐质量通过/已融合" ✗（Gate B-3 PARTIAL，不准入）
* "融合算法带来显著收益" ✗（`serving_v1` 含 lexical，无法归因；多重比较未校正）
