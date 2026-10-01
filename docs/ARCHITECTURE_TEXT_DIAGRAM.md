# AgentRec-X — 纯文字架构图

本文件是 AgentRec-X 的**纯文字（ASCII / text-only）架构图集**：没有图片、没有 Mermaid、没有外部
渲染依赖，只需要等宽字体。

它回答的第一件事不是"系统有多大"，而是 **哪些是产品、哪些是研究**。每张图都带下面三个标签之一：

```text
[SERVING]        当前产品真实执行的路径（/demo/）
[RESEARCH ONLY]  离线实验与评测，不在浏览器里执行
[SHARED]         两条轨道真正共用的东西（数据、协议、评测器）
```

权威描述顺序仍然是：代码 → [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) 的编号章节 → 本文件。
本文件写于检视提交 `b14398c` 之后；状态与数值取自 [`docs/PROJECT_STATE.md`](PROJECT_STATE.md)、
[`docs/M4_RESULT.md`](M4_RESULT.md) 与 [`docs/M4_EXECUTION_MANIFEST.md`](M4_EXECUTION_MANIFEST.md)。

三个术语，全篇统一：

```text
BM25 / catalogue evidence  = 对候选商品自身目录元数据的词法取证（ProductEnricher）；
                             全域词法检索（CatalogSearchSource）属于研究侧
LLM                        = proposal 层——产品里是 route + 偏好值，研究侧是动作选择；
                             ≠ 推荐真相：不能给出 item id / 分数 / 证据 / 排名 / 指标
M3 / M4 control plane      = 离线 agent policy 评测栈（recommendation/control/）；
                             ≠ serving 架构，≠ 产品功能
```

图目录：

```text
图 1   PRODUCT（SERVING）总图            <- 先看这个
图 2   OFFLINE LAB（RESEARCH ONLY）总图
图 3   晋升闸门：什么可以进入产品
图 4   产品服务路径（细节）
图 5   权威边界（TRUST BOUNDARY）
图 6   AgentGraph（产品的默认编排）
图 7   控制面回路                        [RESEARCH ONLY]
图 8   候选平面                          [RESEARCH ONLY]
图 9   离线评估流水线                    [RESEARCH ONLY]
图 10  外部后端边界                      [RESEARCH ONLY]
图 11  状态归属                          [SHARED]
图 12  里程碑与证据链
附录   仓库地图（找代码用；原 L0–L8 分层图）
```

---

## 图 1 — PRODUCT（SERVING）总图

这是 `/demo/` 今天真实执行的路径。**只有一个候选生成器：SASRec。**

```text
                              User
                               │
                               ▼
                    FastAPI app + browser UI
                               │
                               ▼
                     DemoSessionManager
              per-session memory namespace · decision mode
                               │
                               ▼
                          AgentGraph                    [SERVING]
                  conversational orchestration only
                               │
              ┌────────────────┴────────────────┐
              │                                 │
   Deterministic decision            Optional LLM Agent
   (default; needs no key)           plan = route + preference values only
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                  Preference Memory (SQLite)      trusted state
                               │
                               ▼
                     RecommendationTool
                               │
                               ▼
            Serving V1: SASRec + Two-Tower + catalogue lexical
            reciprocal_rank_fusion (RRF_K = 60)  -> fused candidates
                  accepted checkpoint
                               │
                               ▼
                     Trusted candidate set
              parent_asin <-> item_id · PAD = 0 is never a candidate
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   MetadataIndex + ProductEnricher     PreferenceMemoryService
   catalogue facts · CANDIDATE-SCOPED  per user_key
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                PreferenceCandidateMatcher
                MATCH / VIOLATION / UNKNOWN
                               │
                               ▼
                     PreferenceReranker
        deterministic lexicographic policy:
        fewer violations -> more matches -> original SASRec order
                               │
                               ▼
               Result + Recommendation Trace
               Trace is a read-only view; NEVER an input to ranking
```

**不在这张图上的东西**（都在图 2）：Two-Tower、TIGER / Semantic-ID、multi-source fusion、
candidate plane、candidate ledger、M3/M4 的 adaptive policy、外部 TIGER backend。

---

## 图 2 — OFFLINE LAB（RESEARCH ONLY）总图

`RESEARCH ONLY`：不在浏览器里执行。唯一例外是 `AGENTRECX_CONTROL_PLANE=loop` 这个 opt-in
alpha（默认关闭）；即便如此，demo 也只注册一个候选源。

```text
  Frozen dataset + protocol                                             [SHARED]
  Amazon Reviews 2023 · Sports & Outdoors · chronological temporal leave-two-out
  20,000-user cohort · full catalogue · seen-item masking · agentrecx.eval_protocol.v1
                    │
      ┌─────────────┼──────────────┬──────────────────┐
      ▼             ▼              ▼                  ▼
   SASRec      Two-Tower      TIGER-FP32        popularity / metadata
   (also the   (dual encoder) (Semantic-ID      retrieval
    serving     RESEARCH ONLY   generative)      RESEARCH ONLY
    model)                     RESEARCH ONLY
      └─────────────┴──────────────┴──────────────────┘
                    │
                    ▼
        Offline arms: single-source · fixed fusion · ledger RRF
                    │
                    ▼
        Shared evaluator — ONE implementation, also the product's metric source   [SHARED]
                    │
                    ▼
        Recall / HR / NDCG @ K = 5 / 10 / 20
                    │
                    ▼
        Paired uncertainty (exact McNemar · paired user bootstrap)
                    │
                    ▼
        Frozen reports + hash manifest  ->  M3 / M4 conclusions
                    │
                    ▼
        Bounded LLM control plane (M3 / M4)  <- the only research stack the demo can opt into
```

两条已执行的对照，方向一致：

```text
M3  真实有界 LLM policy      vs  fixed-fusion comparator        -> 模型驱动策略更差
M4  LLM adaptive 源选择      vs  fixed four-source traversal     -> 自适应更差（工具调用更少）

两者都只描述"一个系统配置 + 一个冻结协议"，不是"LLM 做推荐不行"的一般结论。
```

---

## 图 3 — 晋升闸门：什么可以进入产品（PROMOTION GATE）

```text
   new model / source / policy
              │
              ▼
   OFFLINE LAB: shared evaluator + frozen protocol + preregistration
              │
      evidence sufficient?
        ┌─────┴─────┐
       yes          no
        │            │
        ▼            ▼
   promotion      stays research only
   candidate

   当前裁决（逐项）:
     SASRec ....................... [SERVING]        已接受；产品唯一的候选生成器
     Two-Tower .................... [RESEARCH ONLY]  未接线；晋升需要 serving-candidate audit
     TIGER / Semantic-ID .......... [RESEARCH ONLY]  已 qualified，但明确未在 runtime 启用
     multi-source fusion .......... [RESEARCH ONLY]  另有实现缺口：preference reranker 只在
                                                     history 候选上运行，"对融合候选重排"未实现
     LLM adaptive source selection  [RESEARCH ONLY]  未晋升：M4 中输给固定多源遍历
     bounded control-plane loop ... [OPT-IN ALPHA]   默认关闭（AGENTRECX_CONTROL_PLANE=loop）

   注意：fusion 那一行不是"证据否决了 fusion"。离线融合结果回答的是另一个问题，
   横在中间的是"实现 + 审计"；本文件按事实这样写，而不包装成一个决策。
```

---

## 图 4 — 产品服务路径（SERVING，细节）

浏览器 demo 实际走的路径。这条路径上**只有一个候选生成器**（SASRec），
**没有** candidate plane、ledger、control plane、Two-Tower、TIGER 或 fusion。

```text
User / Browser
      |  same-origin HTML/CSS/JS · no build step · no CDN
      |  default decision mode: offline, deterministic, NO API key
      |  optional decision mode: LLM Agent, opt-in, provider-configured
      v
FastAPI app  (recommendation/api/app.py)      +   browser UI (recommendation/web)
      |
      v
DemoSessionManager  (recommendation/demo/sessions.py)
      |  session_id · user_key · per-session preference-memory namespace · decision mode
      v
AgentGraph  (recommendation/agent/graph.py)
      |  conversational orchestration only
      |  deterministic decision seam by default
      |  LLM Agent mode: ONE validated turn plan (route + preference values), extra="forbid"
      v
RecommendationTool  (recommendation/tools/recommendation.py)
      |
      v
SASRecInferenceEngine  (recommendation/inference/sasrec.py)
      |  Serving V1: three sources fused by RRF_K = 60 (SASRec alone when disabled/absent)
      v
Trusted candidate set
      |  identity: parent_asin <-> item_id · PAD = 0 is NEVER a candidate
      |
      +--------------------------------+--------------------------------+
      |                                                                 |
      v                                                                 v
MetadataIndex + ProductEnricher                        PreferenceMemoryService
(recommendation/catalog, recommendation/rag)           (recommendation/memory)
catalogue facts · CANDIDATE-SCOPED evidence            SQLite, per user_key
                                                       explicit prefs: ADD / REPLACE / REMOVE
      |                                                                 |
      +--------------------------------+--------------------------------+
                                       v
                    PreferenceCandidateMatcher  (recommendation/preference_matching)
                    per-preference evidence: MATCH / VIOLATION / UNKNOWN
                                       |
                                       v
                    PreferenceReranker  (recommendation/reranking)
                    deterministic lexicographic policy:
                    fewer VIOLATIONs -> more MATCHes -> original SASRec order
                                       |
                                       v
                    Grounded, preference-aware recommendation
                    original SASRec rank preserved NEXT TO the final rank
                                       |
                                       v
                    Response serialization  (recommendation/demo/serialization.py)
                                       |
                    +------------------+------------------+
                    v                                     v
             ChatResponse (JSON)              Recommendation Trace
                                              read-only view over the response's own state
                                              NEVER an input to ranking
```

失败语义：任何一级失败都**显式抛出**，不倒退回"看起来合理"的输出；
候选耗尽（`returned_k == 0`）是正常结果，渲染为中性空状态，**没有** popularity / random 兜底。

---

## 图 5 — 权威边界（TRUST / AUTHORITY BOUNDARY · SERVING + RESEARCH）

这是全项目的第一安全属性：**对话层不能改写行为数据**，且由类型而不是约定强制。

```text
=============================== TRUST BOUNDARY ===============================

[ CONVERSATIONAL / DECISION SIDE ]        may influence ONLY:
      which supported preferences are stored
      whether the recommendation flow runs (route)
      presentation order, through supported preferences

      two interchangeable decision policies on this side:
        deterministic   accepted keyword model + rule-based extractor  (default)
        LLM Agent       ONE validated turn plan (extra="forbid")

      the LLM plan schema has NO field for a product id, score, evidence,
      rank or mutation result => the model cannot express truth even if it tries

      can write to NOTHING on the trusted side below
------------------------------------------------------------------------------
[ TRUSTED CODE ]                          owns and decides:
      behavioural history .............. application-owned, read-only to the agent
      canonical identity ............... parent_asin <-> item_id, PAD = 0 excluded
      catalogue facts .................. MetadataIndex
      model output ..................... SASRec scores
      evidence truth ................... PreferenceCandidateMatcher
      final preference-aware order ..... PreferenceReranker
      evaluation metrics ............... shared evaluator

==============================================================================
```

结构性封堵（不是约定）：

```text
  chat text  --X-->  trusted interaction history    (decision fn takes ONE string;
                                                     AgentDecision forbids extra fields;
                                                     history channel has no reducer)
  chat text  ----->  preference memory              (only explicit statements; each
                                                     entry carries source span + turn)
  preference memory --X--> interaction history      (no shared type exists)
  RAG  ----------->  candidate set                  (can only annotate, never widen)
  reranker  ------>  permutation of its input       (renderer re-verifies by identity)
  browser  --X---->  any trusted field              (request schemas extra="forbid")
```

---

## 图 6 — AgentGraph（SERVING：产品的默认编排，DAG v2）

`AgentGraph` 只做编排；每个协作者都是**注入的可选 Protocol**，图从不构造默认实现。
`AGENT_GRAPH_VERSION = 2`。

```text
load_memory -> decide -> { direct ---> finalize
                         | recommend -> enrich -> match_preferences -> rerank -> finalize }
            -> persist_memory

注入项与路由（节点只在协作者被注入时才存在）:
  tool                              decide -> {finalize | recommend -> finalize}   M7B/M7C
  + product_enricher                adds enrich                                     M8
  + memory_service                  adds load_memory, persist_memory                 M9
  + preference_matcher + reranker   adds match_preferences, rerank                  M10D

构造期校验（半配置状态必须响亮失败）:
  matcher 与 reranker 必须成对出现
  matcher 需要 enricher（它读取 M8 附加的 metadata）
  每个协作者必须暴露自己的那一个方法
  图从不构造默认 matcher / reranker / metadata store / memory store

finalize 的渲染分支由"哪个结构化阶段实际运行过"决定：
  RerankingReport -> 否则 enrichment 结果 -> 否则原始 Tool 结果
  绝不通过检查字符串来决定
```

---

## 图 7 — 控制面回路（RESEARCH ONLY · 2.0-alpha · opt-in）

`AGENTRECX_CONTROL_PLANE=graph`（默认，M7B–M10D DAG）或
`AGENTRECX_CONTROL_PLANE=loop`（有界 agent loop）。
它改变的只是**谁决定下一步**，不改变**推荐怎么做**。

```text
PolicyContext --(AgentPolicy.choose)--> ActionProposal            [untrusted]
                                              |
                                      ActionValidator              [controller-owned]
                                              |
                                        ValidatedAction            stamped metadata
                                              |
                                  Trusted ActionExecutor           accepted pipeline
                                              |
                                        DomainResult               NOT policy-visible
                                              |
                                       ResultVerifier              identity / counts / ranks
                                              |
                                       ObservationAdapter          minimised
                                              |
                                        Observation                what the policy may see
                                              |
                                        LoopController             owns the loop
```

回路拓扑（`recommendation/control/topology.py`）：

```text
initialize -> check_limits -> policy -> validate_action -> dispatch
                   ^                                         |-- execute -> verify
                   |                                         |   -> observe
                   |                                         |   -> update_state
                   +-----------------------------------------+--------+
                   `-- complete -> finalize / refuse / abort -> END

节点用 LangGraph Command(goto=...) 提交路由：
条件边在节点运行"之前"求值，看不到该节点内部产生的裁决。
```

权威切分（类型级，不是约定级）：

```text
  ActionProposal != ValidatedAction       PolicyContext != AgentGraphState
  DomainResult   != Observation           FINISH        != finished

  AgentPolicy      choose() -> ActionProposal   不得执行工具/写状态/数步数/给候选 id/
                                                自校验/宣告结束
  LoopController   生命周期·计数器·可用动作·调度·预算·终止·轨迹   不得打分/排序/检索/
                                                解释偏好/提交记忆/渲染
  ActionValidator  可用性·参数复校验·盖 action_id/step_index/run_id/turn_id
                                                不得把提案"修"成另一个动作
  ResultVerifier   工具接地·数量与排名完整性·候选身份·阶段对齐    不得改写或编造结果
  CompletionGuard  FINISH 是否是合法终点          不得为失败或未接地的运行背书

预算 max_steps / max_tool_calls / max_retries 由控制器配置，策略既读不到也抬不高。
两个驱动器一个引擎：driver="graph" 与 driver="direct" 跑同一套引擎方法。
```

---

## 图 8 — 候选平面（RESEARCH ONLY · Stage 2）

```text
AgentPolicy                     chooses WHICH source — and nothing else
    |
    v
CandidatePlane                  executes that source · grounds it · records it
    |-- HISTORY        -> accepted RecommendationTool -> SASRecInferenceEngine  (SASRec logits)
    |-- CATALOG_SEARCH -> CatalogSearchSource over the WHOLE catalogue          (BM25)
    |-- SIMILAR_ITEM   -> registered item-item tool, or ABSENT                  (similarity)
    |-- TWO_TOWER      -> TwoTowerSource   [RESEARCH, opt-in]                   (learned cosine)
    `-- TIGER          -> TigerSource      [QUALIFIED, NOT enabled in runtime]  (SID log-probs)
    |
    |   ONE ACTION CONSULTS ONE SOURCE  (adaptive retrieval is the point;
    |   no "call every source" method exists, so fixed fan-out is not the easy path)
    v
GroundingVerifier               is this identity one the trusted side knows?
    |                           FAILS CLOSED: unconfirmable => UNGROUNDED
    |                           retained ONLY as audit evidence, never presented
    v
CandidateLedger                 the run's record of candidate identity + provenance
    |  dedup IDENTITY, not evidence: one entry per parent_asin;
    |    a later sighting APPENDS CandidateProvenance, never replaces
    |  NEVER fuse heterogeneous raw scores: each kept under its own score_kind
    |  cross-source order = RECIPROCAL RANK FUSION over each source's OWN ranking
    |  original source ranks stay recoverable
    |  only record_from_source() may add — a policy is nowhere in that call path
    v
  ranking / presentation

  Enum 成员身份 != 运行时可可用性：CandidateSource 只声明"仓库里有这个工具"，
  真正能否被选中由控制器算出的 available_sources 决定，未注册的源在可信边界被拒绝。
  M4 实验用的是一个"冻结的四源子集"：history / catalog_search / similar_item / two_tower。
```

`CatalogSearchSource` 与 M8 的候选域检索器**刻意分开**：后者按设计只在 SASRec 候选内工作，
这正是 RAG 不能扩大候选集的原因；扩大范围是策略可以主动选择的能力，只发生在这里。
两者共享 `recommendation.rag.retrieval` 的 tokenizer / BM25 常量 / idf，因而不会漂移。
零命中就返回零命中（`matched=False`），**绝不**退化成任意目录切片。

---

## 图 9 — 离线评估流水线（RESEARCH ONLY）

本图中的任何东西都**不在**浏览器 demo 内运行。

```text
Frozen dataset + protocol
  Amazon Reviews 2023 · Sports & Outdoors · chronological temporal leave-two-out
  frozen 20,000-user cohort (deterministic seed) · full catalogue (156,746 items)
  seen-item masking · tie-breaking deterministic
                              |
      +--------------+--------+---------+-------------------+
      |              |                  |                   |
      v              v                  v                   v
   SASRec       Two-Tower           TIGER-FP32        popularity / metadata
   (also the    (dual encoder)      (Semantic-ID      retrieval
    serving     RESEARCH ONLY        generative)       RESEARCH-ONLY ARMS
    model)                         RESEARCH ONLY
      +--------------+------------------+-------------------+
                     |
        Offline benchmark arms  (experiments/, recommendation/evaluation)
                     |
           +---------+---------+
           v                   v
     Single-source         Fixed fusion / ledger RRF
     evaluation            (rank fusion over the four sources)
           +---------+---------+
                     v
        Shared evaluator — ONE implementation for every arm
                     |
        Recall / HR / NDCG at K = 5 / 10 / 20
                     |
                     v
        Paired uncertainty (exact McNemar + paired user bootstrap,
                            10,000 resamples, seed 20260201, unit = user)
                     |
                     v
        Research conclusions + frozen reports (docs/reports/*.txt, hash-bound)
```

两条已经执行完毕的离线对照（同一冻结队列、同一评测器、同一协议）：

```text
  M3   existing real Agent system (LLMAgentPolicy + LoopController, bounded budget)
         vs   fixed-fusion comparator
       => 主终点 ΔNDCG@10 = -0.0028794446, 95% CI [-0.0036454951, -0.0021236081]
          区间整体位于 0 以下：真实 Agent 系统"可检测地"弱于固定融合
          行为原因：20,000 个轨迹里 19,999 个是 recommend_from_history -> finish，
          从未调用 catalog_search；SELECT_SOURCE 在该冻结配置下不可达

  M4   LLM adaptive source selection   vs   fixed four-source traversal
       （同一四源宇宙：history / catalog_search / similar_item / two_tower）
       => fixed NDCG@10 = 0.0115174, adaptive NDCG@10 = 0.0085684
          ΔNDCG@10 = -0.0029490, 95% CI [-0.0041647, -0.0017272], McNemar p = 0.0001288
          预注册判定 SUPPORTED（统计上成立的"负结果"：让模型自己挑源更差）

  两条都是"一个系统配置在冻结协议下的一个结果"，
  不是"LLM agent 做推荐不行"的一般性结论。
```

---

## 图 10 — 外部后端边界（RESEARCH ONLY）

规范：[`docs/TIGER_BACKEND.md`](TIGER_BACKEND.md)；规则：[`AGENTS.md`](../AGENTS.md) 第 19 节。
后端**只提供原始分数**，从不掩码、排序、解析身份、挑选候选或报告指标。

```text
      AgentRec-X 进程  (CPU PyTorch 环境)          ||    后端 venv (CUDA PyTorch 环境)
 ==================================================||==========================================
   materialiser: item_id + text, 身份在外层重新附着 ||
   recommendation/backends/tiger_backend.py        ||
     = the ONLY bridge, and deliberately:          ||
        - NO ML import, NO backend import          ||
        - NEVER reads/accepts/stores/returns parent_asin
        - required_frontier = K_max + |seen|  computed HERE,
          writes ONLY the integer — never seen-item identities
                                                   ||
                    |                              ||
                    |   subprocess spawn + filesystem contract (JSONL / .npy / .npz / .json)
                    |   catalogue_handoff  : item_id + content text      (no identity, no target)
                    |   train_exposure     : train_history ONLY
                    |   eval_cohort        : case_id, history, required_frontier (integer)
                    v                              vv
                                                   tiger_public CLI
                                                     build-features -> frozen sentence encoder
                                                     fit-sid       -> RQ-VAE + 3-level SIDs
                                                                      + dedup digit + audit
                                                     train         -> T5 enc-dec over SID layout
                                                     score         -> constrained decode over
                                                                      catalogue prefix trie
                    ^                              ||
                    |   score_batch / scores.npy : item_id -> RAW SCORE only
                    |                              ||
   evaluate_batched (UNCHANGED, shared evaluator) <++

  PAD = 0 由负哨兵与任何 token 空间隔离；PAD 行永不被 tokenise 或解析。
  适配器不修改根 requirements.txt；共享评测器、冻结协议、队列、身份映射与
  CandidateSource 枚举都不为任何后端而改。
  pytest.ini: norecursedirs = backends  =>  AgentRec-X 测试套件从不收集后端测试。
```

---

## 图 11 — 状态归属与数据流（SHARED）

```text
  +---------------------------------+----------------------+-------------------------------+
  | STATE                           | OWNER                | WRITABLE BY                   |
  +---------------------------------+----------------------+-------------------------------+
  | trusted interaction history     | application/session  | application only              |
  | preference memory               | PreferenceMemory     | the user's own explicit       |
  |                                 | Service + store      | statements                    |
  | browser transcript              | the page             | the page                      |
  | tool result / enrichment /      | graph run (derived)  | the stage that produced it    |
  | evidence / reranking            |                      |                               |
  | live session registry           | DemoSessionManager   | the session layer             |
  | accepted artifacts              | repository           | explicit retraining, to a new |
  |                                 |                      | run directory                 |
  +---------------------------------+----------------------+-------------------------------+

  两条推论：
    1) 原始顺序永不被覆盖 —— tool_result / enrichment 保持 SASRec 顺序，
       其 rank 就是权威 original_rank；reranking 报告是"追加的派生状态"。
    2) 每轮一个一致的内存快照 —— 快照在轮次开始读取一次，供整轮使用，
       因此轮内的写入不可能影响本轮。
```

---

## 图 12 — 里程碑与证据链（MILESTONE → ARTIFACT → HASH）

```text
  Phase 1  Recommendation Baseline
    preprocessing -> real-data integration -> unified evaluation protocol
    -> ItemCF [PARTIAL] -> SASRec -> offline comparison -> inference -> FastAPI service
         |
         v
  Phase 2  Agent system      AgentGraph · RecommendationTool · RAG(enrichment)
                             · Preference Memory · grounded explanation
         |
         v
  Phase 3  Agent enhancement deterministic reranking · preference evidence
                             · multi-turn · Web Demo
         |
         v
  Phase 4  Generative recommendation       semantic_id/  [FROZEN GenRec v0, historical]
         |
         v
  Step 2.x TIGER backend behind an explicit boundary
           features -> SID (RQ-VAE) -> T5 -> constrained retrieval -> raw scores
         |
         v
  M1 persistence ............ COMPLETE  (canonical remote, fresh clone, manifest 28/28)
  M2 paired uncertainty ..... COMPLETE  (4 audited claims; 4/4 DIRECTIONAL ONLY or
                                        NO DETECTABLE DIFFERENCE)
  M3 Agent vs fixed fusion .. COMPLETE  (statistically supported NEGATIVE result)
  M4 adaptive vs fixed ...... COMPLETE  (statistically supported NEGATIVE result)

  Every accepted result is bound to: executed code commit · seal ref (Git tag M4-sealed)
  · preregistration SHA256 · cohort identity · per-file SHA256 of frozen modules
  · result artifact SHA256.  "SUPPORTED" is a preregistered classification LABEL,
  not a claim about direction.
```

---

## 当前状态速查（STATUS AT A GLANCE）

```text
  [SERVING]        preprocessing（离线产物）· SASRec 训练/推理 · RecommendationTool
                   catalogue metadata · candidate-scoped RAG（BM25 取证）
                   preference memory (SQLite) · preference evidence · deterministic reranking
                   AgentGraph DAG v2 · FastAPI + browser demo
                   optional LLM Agent decision mode（proposal only）
  [SHARED]         dataset · split（temporal leave-two-out）· shared evaluator · metric primitives
  [RESEARCH ONLY]  Two-Tower · TIGER-FP32 · popularity/metadata arms · fusion experiments
                   bounded control plane（2.0-alpha）· candidate plane + ledger + grounding
                   Stage 3 interaction plane · Stage 4 reasoning plane · provider record/replay
                   external TIGER backend（backends/tiger_public，独立 venv）
  [FROZEN]         recommendation/semantic_id/（GenRec v0）· eval protocol v1 · M3/M4 results
  [OPT-IN ALPHA]   AGENTRECX_CONTROL_PLANE=loop（默认关闭；开启后仍只有一个候选源）
  [DEFERRED / NOT IMPLEMENTED]
                   ItemCF under the unified protocol · weighted/calibrated/learned fusion
                   reranking over the fused ledger · SIMILAR_ITEM by default（需要索引）
                   LangGraph naming alignment（实现是自定义 controller）
                   external durable hosting for the 11 large artifacts（non-blocking）
```

**权威来源**：[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) · [`docs/PROJECT_STATE.md`](PROJECT_STATE.md)
· [`docs/M4_RESULT.md`](M4_RESULT.md) · [`README.md`](../README.md)


---

## 附录 — 仓库地图（REPOSITORY MAP，找代码用）

这不是门面图。产品的形状看图 1，研究的形状看图 2；这一张只回答"哪个包在哪一层"。

```text
==========================================================================================
  L0  EXTERNAL WORLD
==========================================================================================
  Amazon Reviews 2023 · Sports & Outdoors   DeepSeek API (deepseek-flash)   Browser
  (raw, read-only, never mutated)           (optional decision policy)      (untrusted)
        |                                          |
        |  raw data                                |  optional policy calls
        v                                          |  (both enter at L5)
==========================================================================================
  L1  DATA & PREPROCESSING                                     [CPU]
==========================================================================================
  recommendation/preprocess.py · recommendation/io_utils.py
    validate -> chronological order -> ITERATIVE k-core (fixed point)
    -> deterministic user/item mapping -> PAD = 0 reserved -> parent_asin canonical
        |
        +--> processed artifacts (reproducible: raw + config + code)
        +--> recommendation/catalog/metadata.py   MetadataIndex (catalogue facts)
        +--> recommendation/datasets/  sasrec · twotower · window  (temporal splits)
        |
        v
==========================================================================================
  L2  MODELS & TRAINING                                        [GPU for training]
==========================================================================================
  models/sasrec.py          <-- THE serving model (only learned component online)
  models/twotower.py        [RESEARCH ONLY] dual encoder
  training/  sasrec · twotower · losses · checkpoint  (validation-only selection)
  inference/ sasrec (SASRecInferenceEngine) · twotower · ranking
  semantic_id/              [FROZEN historical GenRec v0; not extended]
  backends/tiger_public/    [RESEARCH] RQ-VAE + Semantic IDs + T5  -- see 图 10
        |
        v
==========================================================================================
  L3  EVALUATION                                               [frozen protocol]
==========================================================================================
  evaluation/split.py     temporal_leave_two_out   (agentrecx.eval_protocol.v1)
  evaluation/metrics.py   HR@K · Recall@K · NDCG@K  (single positive => HR == Recall)
  evaluation/evaluator.py · evaluation/batched.py   ONE evaluator for EVERY arm
  evaluation/agent/       trajectory · cases · metrics · attribution · runner
  baselines/itemcf.py     [PARTIAL] engineering baseline
        |
        v
==========================================================================================
  L4  TRUSTED SERVING CORE  (owns truth)
==========================================================================================
  tools/recommendation.py        RecommendationTool  -> SASRecInferenceEngine
  memory/                        PreferenceMemoryService (SQLite, per user_key)
  rag/retrieval.py               ProductEnricher  (CANDIDATE-SCOPED only)
  preference_matching/           PreferenceCandidateMatcher  MATCH/VIOLATION/UNKNOWN
  reranking/                     PreferenceReranker  deterministic lexicographic
        |
        v
==========================================================================================
  L5  CONVERSATION & SERVING SURFACE          <-- Browser 与 DeepSeek 在此进入
==========================================================================================
  agent/graph.py       AgentGraph (DAG v2) — default orchestration   见 图 6
  agent/               decision · state · rendering · pipeline
  api/                 FastAPI app · demo_routes · schemas · error mapping
  demo/                DemoSessionManager · DemoRuntime · serialization
  demo/llm_policy.py   [OPT-IN] LLM Agent decision mode (route + preference values only)
  web/                 plain HTML/CSS/JS, no build step, same-origin only
        |
        +-------------------------------+
        |                               |
        v                               v
==========================================================================================
  L6  CONTROL PLANE 2.0-alpha (OPT-IN) + CANDIDATE / STAGE 3 / STAGE 4 PLANES   见 图 7、8
==========================================================================================
  control/loop.py        LoopController — bounded agent loop
  control/policy.py      AgentPolicy        (untrusted proposer)
  control/validation.py  ActionValidator    control/completion.py  CompletionGuard
  control/verification.py ResultVerifier    control/topology.py    loop topology
  control/candidate_plane.py · candidate_ledger.py · grounding.py
  control/catalog_search.py · similar_item.py · tiger_source.py · two_tower_source.py
  control/task_state.py · arguments.py · constraint_eligibility.py · memory_proposal.py
  control/evidence.py · evidence_executor.py · reasoning_executor.py · grounded_reasoning.py
  control/model_client.py · model_policy.py · provider_adapter.py · model_recorder.py
        |
        v
==========================================================================================
  L7  EXTERNAL BACKEND BOUNDARY   (separate venv, separate process)   见 图 10
==========================================================================================
  recommendation/backends/tiger_backend.py   adapter: no ML import, no backend import,
                                             never sees parent_asin, scores only
        ~~ subprocess + filesystem (JSONL / .npy / .npz / .json) ~~
  backends/tiger_public/  cli · features · quantizer (RQ-VAE) · tiger (T5) · trie ·
                          retrieve · scoring · dedup · contracts · io
        |
        v
==========================================================================================
  L8  EVIDENCE & PROVENANCE
==========================================================================================
  tests/            pytest suite (pytest.ini: norecursedirs = backends)
  experiments/      offline arms, smokes, benchmarks, M3/M4 harnesses
  runs/             run outputs (gitignored; selected evidence force-added)
  docs/reports/     frozen reports, hash-bound
  docs/*_PREREGISTRATION.md · M4_EXECUTION_MANIFEST.json · Git tag M4-sealed
  scripts/ · config/ · Dockerfile · docker-compose.yml · .github/
```
