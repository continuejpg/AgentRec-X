# AgentRec-X 交接与项目理解文档

> 生成于 HEAD `ffe33f8`（分支 `master`，工作树 clean，local == origin/master）。
> 本文件**只读侦察的产物**：写这份文档的过程**没有修改任何仓库文件、没有运行任何实验、没有重跑测试**。
> 上一版交接在 `/tmp/agentrecx-handoff.md`。本文件**取代**它，并修正了其中若干不准确之处（见 §9）。

---

## 0. 这份文档怎么用

三层用途：

1. **§1–§2**：环境与红线。动手前必读，避免误伤冻结区。
2. **§3–§5**：项目到底做了什么、当前测出了什么。**这是"能逐层解释"的底料。**
3. **§6–§9**：真实存在的缺陷清单、已修正的交接错误、未验证清单。

**标记约定**（全文遵守）：

| 标记 | 含义 |
|---|---|
| ✅ **artifact-backed** | 我亲自读取/解析了产物文件并复算/核对过 |
| ✅ **code-verified** | 我亲自读了源码，看到具体行为（附 `file:line`） |
| ⚠️ **prose-only** | 只在文档正文里出现，本机**没有**对应产物可核 |
| ❓ **inherited-unverified** | 来自上一版交接，我**未**核实 |

---

## 1. 硬事实：仓库与环境

| 项 | 值 |
|---|---|
| 路径 | `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| HEAD / 分支 | `ffe33f8` / `master`（工作树 clean） |
| 历史 | **78 commits**；365 tracked files |
| 解释器 | `.venv/bin/python` = Python 3.10.12 |
| CPU 依赖 | torch **2.1.2+cpu** · numpy 1.26.4 · pydantic 2.13.5 · fastapi · langgraph（仅编排原语） |
| 训练环境（记录于 `run.json`） | RTX 4090 · torch **2.1.2+cu118** · CUDA 11.8 · 24 GB |
| 测试 | ❓ ~2360 collected / 2326 passed / 34 skipped / ~18 min（上一版交接数字，本次**未重跑**） |
| 端口 8000 | ⚠️ **已被占用**：一个 AgentRec-X demo server 正在运行，`/health` 返回 `model_loaded:true`，且 `/v1/demo/decision-modes` 显示 **LLM 模式 available:true**（provider `deepseek`） |
| 陈旧 checkout | `/home/zxt/AgentRec-X`（HEAD `4de3091`）——**不要用它** |

### 本机数据与产物实况（我实际 `ls`/`sha256sum` 过）

| 路径 | 状态 |
|---|---|
| `data/raw/` | ❌ **不存在** → 预处理**无法**从原始数据重算 |
| `data/processed/Sports_and_Outdoors_sequences.json` | ✅ 348,683,134 B，sha `d3d83426…` **与 `run.json` 记录一致** |
| `data/processed/Sports_and_Outdoors_mappings.json` | ✅ 39,670,810 B，sha `dca7815a…` **一致** |
| `data/processed/Sports_and_Outdoors_products.jsonl` | ✅ 307,500,147 B |
| `runs/sasrec_canonical_2026/best.pt` | ✅ 121,675,470 B，sha `352bd3ae…` **与 `run.json` 记录一致** |
| `runs/sasrec_canonical_2026/last.pt` | ❌ 不存在（`run.json` 引用了它 → 悬空引用） |
| `runs/sasrec_canonical_2026/history.jsonl` | ❌ 不存在（同上） |
| `runs/` 总量 | 约 990 MB，72 项 |
| `/tmp/agentrecx-phase3/revised_live.jsonl` | ❌ 不存在（只有 `baseline_live.jsonl`）→ Phase-3 回放测试**自动 skip** |

> **重大含义**：`best.pt` 的 SHA-256 与 `run.json` 记录**逐字节一致** —— 本机这份 checkpoint **就是**被接受的 canonical 产物，不是"看起来像"。上一版交接把"两个大工件的 SHA-256 未重算"列为未验证项；**`best.pt` 现已核实**。

### 工具陷阱（务必遵守）

1. **不要用 shell `grep -r` 扫全仓**：会走进 `.venv` 和 `backends/tiger_public/.venv` 并挂住（DrvFs 很慢）→ 用 grep/glob 工具。
2. **`pytest … | tail` 会吞掉退出码** → 显式捕获：
   ```bash
   RC=0; .venv/bin/python -m pytest -q > /tmp/t.log 2>&1 || RC=$?; tail -3 /tmp/t.log; echo RC=$RC
   ```
3. **不要用 `find` / `du` 扫 `backends/`**：该目录含 11,588 个 `.py`（其中 11,566 在 `.venv` 内），本会话已因此超时一次。
4. `.pytest_cache/v/cache/lastfailed` 是陈旧缓存，不是失败清单。

---

## 2. 红线：冻结边界

| 冻结项 | 位置 | 约束 |
|---|---|---|
| 评估协议与 evaluator | `recommendation/evaluation/{batched,metrics,split}.py` | hash 记录在 `docs/PROJECT_STATE.md`；**不改语义** |
| 冻结报告（15 份） | `docs/reports/*.txt` | hash-bound；**不能改正文** |
| M3 预注册 | `docs/M3_PREREGISTRATION.md` | hash-bound；状态更新只能加**指针** |
| M2 报告 | `docs/reports/m2-paired-uncertainty.txt` | 同上 |
| M2 / M3 结果 | 报告 + `runs/m2*`、`runs/m3_execution/*` | **不重跑、不重调、不改数字**（M3"只跑一次"是声明过的） |
| TIGER 后端 | `backends/tiger_public/` | 不改行为/分数/检索逻辑 |
| 融合策略 | `recommendation/control/candidate_ledger.py` | **rank-only RRF 是有意选择**；不要加加权/学习融合 |
| 边界规则 | `AGENTS.md` §19 | 含"无证据不集成" |

**manifest 校验现状**：`docs/PROJECT_STATE.md` 的 28 行 SHA256 manifest —— ✅ **我逐行实算过：28/28 全部匹配，零漂移**，构成为 **15 份 `docs/reports/*.txt` + 13 份源码/预注册**（`docs/M3_PREREGISTRATION.md` + `recommendation/evaluation/{batched,metrics,split}.py` + `control/{grounding,candidate_ledger,candidate_plane,tiger_source}.py` + `backends/tiger_public/src/tiger_public/{retrieve,retrieve_cli,scoring,trie,tiger}.py`）。`docs/reports/` 下 15 个文件**全部**被 hash 绑定，**没有未绑定也没有缺失**。

但**没有脚本、没有 CI job** 自动校验它 —— 这是状态漂移的真实根因（`tests/test_docs.py:31-44` 的 `DOC_PATHS` 不含 `PROJECT_STATE.md`）。

**另一处时间性矛盾**：`runs/m1-persistence-freeze.txt` 仍写 **"M1 = NOT COMPLETE (FAIL)"**（`:5`、`:178`、`:216`），因为它冻结于 push **之前**；`PROJECT_STATE.md:119` 写 **M1 = PASS**（push 于 2026-09-26 完成）。⚠️ 该文件**不在** 28 行 manifest 内（所以不受 hash 保护），但它是一份"冻结报告"，读的时候必须知道它是 push 前的快照。

**关键警告**：`loop.py:521-535` 的 `_plane_candidate_actions()` 决定 agent 能看见哪些动作。我看过源码：

```python
# loop.py:530-534
if self.candidate_plane.has_source(CandidateSource.CATALOG_SEARCH):
    offered.append(ActionKind.SEARCH_CATALOG)
if self.candidate_plane.has_source(CandidateSource.SIMILAR_ITEM):
    offered.append(ActionKind.FIND_SIMILAR)
    offered.append(ActionKind.SELECT_SOURCE)   # ← 只有 SIMILAR_ITEM 注册时才给出
```

✅ **code-verified**：`SELECT_SOURCE` 的可见性**挂在 `SIMILAR_ITEM` 上**，而不是挂在"有任何可切换源"上。TIGER 虽已入枚举，却不触发它 → 这就是"`SELECT_SOURCE` 不可达"的**机制**。仓库选择**记录而非修复**（见 `M3_PREREGISTRATION.md` §10）。

---

## 3. 项目实际演进（与 AGENTS.md 路线图的对应）

`AGENTS.md` 写的是 4 阶段路线图；实际 commit 历史（78 个）走的是**两条轨**：

```text
主干（Phase 1→5 + 模型扩展）
  M0..M5  → 5244298  预处理 / 评估协议 / ItemCF / SASRec 全家桶（M0–M5 共用一个 commit）
  M6      → 812ff31  SASRec 推理 + FastAPI
  M7A/B/C → cb1e385, 8fca6fd, 0fe3f58   Recommendation Tool → LangGraph agent → 真实 E2E
  M8      → 1715978  产品 metadata + candidate-scoped RAG
  M9      → c03582b  偏好 Memory（SQLite）
  M10A..D → cce4289, d020d09, 559ab10, 645b50a  证据 → 确定性重排 → 评测 → agent 集成
  M11     → c7c3bf5  多轮 web demo
  ── Phase 3.1 (69f56ef) · Phase 4 (0082547) · Phase 5 (f3307e3) ──
  milestone20..23 → 6f71a4f, 8431ce7, 2123c66, 80f81b1   demo/Docker/CI · Two-Tower · 控制闭合 · Semantic-ID

研究轨（AGENTS.md 路线图里没有的名字）
  Step 2.2–2.8b  TIGER 后端边界 → RQ-VAE/Semantic ID → 认证检索 → CandidateSource 资格 → rank-only 融合
  M1 → 72ed81d  持久化冻结 + 28 行 hash manifest
  M2 → a489309  配对不确定性（bootstrap + McNemar）
  M3 → 5214c8f..b66c192  预注册的真实 LLM agent vs fixed fusion
```

> ⚠️ **重要**：`AGENTS.md` 的"Phase 4 = Generative Recommendation"与 commit 里的 "Phase 4"（`0082547` real DeepSeek run）**不是一回事**。`AGENTS.md` 的阶段编号是**事后重排的路线图**，commit 里的 Phase 编号是**当时的历史**。面试时不要把两者混为一谈。

`5244298` 一个 commit 塞进了 54 个文件（M0–M5 全在里面），且 `run.json` 记录 `git.commit = 859fd0b…` + `git.dirty = true` —— **SASRec 训练是在一个 dirty 工作树上跑的**，代码事后才提交。这是真实的复现性缺口，要主动说，不要被问出来。

---

## 4. 项目到底做了什么（分层心智模型）

```text
用户自然语言
   ↓
[API]  recommendation/api/            FastAPI，模型只加载一次
   ↓
[Agent] recommendation/agent/ + control/ (14.3k 行，最重)
        ├─ 受限动作空间 ActionKind（枚举，不是自由文本）
        ├─ ARGUMENTS_BY_ACTION：动作→参数 的**全映射**（新动作必须声明参数）
        ├─ loop.py：每步只把"可执行"的动作摆上菜单
        ├─ CandidatePlane：真正的候选来源（history / catalog_search / similar_item / tiger）
        ├─ CandidateLedger：审计账本
        └─ GroundingVerifier：候选必须落到可信身份上
   ↓
[Tool] recommendation/tools/           RecommendationTool —— 通往候选的**唯一**路径
   ↓
[模型] recommendation/models/ (SASRec) · backends/tiger_public (TIGER) · Two-Tower
   ↓
[排序/证据] preference_matching/ (三态证据) · reranking/ (冻结字典序策略)
   ↓
[记忆] recommendation/memory/          SQLite，按用户隔离，与行为历史**分离**
   ↓
[评测] recommendation/evaluation/      全候选 ranking + seen-item masking（所有模型共用同一个）
```

**贯穿全项目的"信任"设计（这是这个项目真正的卖点）**：

- 对话**只能**改变展示与排序，**不能**改写推荐器的知识、改分数、屏蔽历史、或凭空造产品身份。
- Tool 是**唯一**的候选入口（M7A）；检索**不能**扩大候选集（M8）。
- 聊天文本**没有**任何路径写入交互历史（M7B，M11 在 HTTP 层断言）。
- `UNKNOWN` 是中性，**不是**惩罚（M10A）。
- 重排**从不**增删/过滤/改分（M10B）。
- 策略遵从度**不**构成相关性声明（M10C）。

---

## 5. 当前测出了什么（全部 artifact-backed）

### 5.1 数据（✅ 我亲自解析了 `data/processed/Sports_and_Outdoors_sequences.json`）

| 指标 | 值 | 来源 |
|---|---|---|
| 交互数 | **3,500,587** | `sequences.json: num_interactions` |
| 用户数 | **412,445** | `num_users` |
| 物品数 | **156,746** | `num_items` |
| 序列长度均值 | **8.487403** | 我实算（`length` 字段 412,445 个值） |
| 中位 / 最小 / 最大 | **6** / **5** / **696** | 我实算 |
| 格式 | `agentrecx.sequences.v1`，按时间**升序**，item_id **1-based**，0 仅作 padding | 产物内描述 |

> 最小长度 = **5** → 与 `split.min_sequence_length: 3` 不矛盾：5-core 过滤保证了每用户 ≥5 交互。
> ⚠️ 因此 `benchmark_public.py:222-228` 的 `"0:<=3"` 桶**永远是空的** —— 队列实际只有 **4 个非空桶**（与 `arms.json` 的 round-robin 实现一致，见 §6.8）。
> ⚠️ 19,595,170 条读取 / 215,135 去重 / 11 轮收敛 这三个数字是 **prose-only**（`data/raw/` 不存在，无 preprocessing metadata 产物）。

### 5.2 协议（`agentrecx.eval_protocol.v1`）

- `temporal_leave_two_out`：训练历史 → 验证目标 → 测试目标；验证集历史 = 训练历史；测试集多一条验证交互。
- full-catalogue ranking、PAD 定位排除、seen-item masking（**target 强制保留**）、同分取小 `item_id`（全序）。
- 规模：**412,445 test cases / 412,445 validation cases**（不是 20,000 —— 20,000 是**队列**）。
- ✅ **HR@K ≡ Recall@K 已实证**：`arms.json` 里每个 arm 的 `HR` 与 `Recall` 逐位相同（如 `agent_selected` 0.0118/0.0118）。这是**设计使然**（单正样本），文档有说明，不是 bug。

**✅ 泄漏审计（`arms.json: leakage_checks`，passed: true）—— 这是被低估的资产**：

| 检查项 | 结果 |
|---|---|
| 检查了 20,000 个 case 的时间戳严格有序 | ✅ true |
| history 严格早于 target | ✅ 20000/20000 |
| 非单调序列 | **0** |
| test target 出现在自身历史中 | **120**，其中**全部 120** 是"重复购买更早的物品" |
| validation target 出现在自身历史中 | 127（同类） |
| **无法解释的 target-in-history** | **0** |
| repeat share | 0.006 |
| split 形状一致 / 样本是全集子集 | ✅ true |

> 面试可讲：**"我做了显式的时间戳级泄漏审计，20,000 个 case 全部通过，0 例非单调，0 例无法解释的 target-in-history"**。注意要主动声明：这 120 例是合法的复购，**由 seen-item masking 处理**（masking 会屏蔽已消费物品，但**强制保留 target**）。

### 5.3 20k 队列

| 项 | 值 | 来源 |
|---|---|---|
| seed | **20260201** | `benchmark_public.py:126` |
| 大小 | 20,000 / 412,445 eligible | `arms.json: cohort` |
| 选择方式 | "deterministic seeded permutation, stratified by history length, processed **round-robin** across length buckets then sorted by user id" | `arms.json: cohort.selection` |
| 队列历史长度 | mean **14.8365** · median 11 · min 4 · max 695 | `arms.json: cohort.history_length` |
| 人群序列长度 | mean 8.487403 | 我实算 |
| 长度分桶（人群） | `4-5` 32.14% · `6-10` 49.98% · `11-20` 13.95% · `>20` **3.936%** | 我实算 |
| `>20` 段 | 队列 **25%** vs 人群 **3.94%** → **约 6.35×** | 推算 |

> ⚠️ **代表性警告**：等量分层（4 个非空桶各 5,000）**不是**总体代表性样本。队列平均历史（14.84）约为人群（8.49）的 **1.75×**。不要说"代表性样本"，不要说 34.6% / 14×（错的）。

### 5.4 主结果表（20k 队列，✅ 全部来自 `runs/phase5_benchmark_public/arms.json`）

| Arm | R@5 | R@10 | R@20 | NDCG@10 | 秒 |
|---|---|---|---|---|---|
| popularity | 0.00370 | 0.00525 | 0.00785 | 0.002687 | 16.0 |
| metadata_retrieval (BM25) | 0.00505 | 0.00925 | 0.01485 | 0.004405 | 18.3 |
| agent_selected（规则替身，**非 LLM**） | 0.00690 | 0.01180 | 0.01820 | 0.005837 | 28.1 |
| sequential (SASRec) | 0.00820 | 0.01350 | 0.02070 | 0.006848 | 16.7 |
| **fixed_fusion** (RRF pop+seq+meta) | **0.00860** | **0.01435** | 0.02325 | **0.007434** | 39.4 |

其余 arm（来自 `PROJECT_STATE.md` 指向的其他产物，❓ 本次未逐个复核）：

| 系统 | R@10 | NDCG@10 |
|---|---|---|
| Two-Tower (Step 1.1) | 0.01435 | 0.007365 |
| TIGER-FP32 (CERTIFIED) | 0.01385 | 0.007755 |
| GenRec-v0 (Semantic-ID) | 0.00185 | 0.0014248 |
| RRF A (seq+2T) | 0.01745 | — |
| RRF D (三源+TIGER) | 0.01810 | — |

> ⚠️ **所有绝对指标都极低**（最优 R@10 仅 1.4%）。**绝对不能说"效果好"**。正确说法：这是 156,746 候选、每用户单正样本、k-core 过滤后类目上的全候选检索，**相对差异**才是信号，且**样本量小**。

### 5.5 头部覆盖 —— 真正的瓶颈（✅ `arms.json` + `twotower_benchmark_full/comparison.json`）

目标出现在各源 **top-1000 head** 内的用户占比：

| 源 | 覆盖率 |
|---|---|
| metadata | 8.17% |
| popularity | 10.73% |
| sequential (SASRec) | 20.50% |
| fixed_fusion (pop+seq+meta) | **20.66%** |
| two_tower (Step 1.1) | 22.155% |
| 三源融合 (seq+2T+meta) | **25.005%** |

> **这是解释一切低指标的钥匙**：即使 head 内排序完美，R@10 天花板也只有约 20.5%。**瓶颈在候选生成（retrieval），不在排序（ranking）。** 这是面试里最有说服力的一句话，而且 artifact-backed。
> ⚠️ 注意 `mean_target_rank` 的**语义陷阱**：`fixed_fusion` 是 80,590，而 `sequential` 只有 **20,598**、`two_tower` 只有 **21,944**。**融合的 target 平均排名比单源差得多**，但 Recall@10 更好。原因：融合把目标**拉进 head**（覆盖率↑），却把它的**位置推后**。而 metadata 类 arm 的未检索物品被统一赋 `-1.0`，所以它们的 `mean_target_rank` 是**尾约定，不是质量信号**（`benchmark_public.py:730-762`）。**引用 mean_target_rank 时必须说明这一点。**

### 5.5b ⚠️ "fusion beats every single source" 的真相（✅ 两个产物对拍）

`runs/twotower_benchmark_full/comparison.json` 与 `arms.json` 同 cohort/协议/evaluator，可直接对比：

| Arm | R@5 | R@10 | R@20 | NDCG@10 | head 覆盖 | mean_target_rank |
|---|---|---|---|---|---|---|
| `sequential` | 0.00820 | 0.01350 | 0.02070 | 0.006848 | 20.50% | 20,597.7 |
| `two_tower` | **0.00860** | **0.01435** | 0.02285 | 0.007365 | 22.155% | 21,943.8 |
| `fixed_fusion` | **0.00860** | **0.01435** | **0.02325** | **0.007434** | 20.66% | 80,590.3 |

**结论（必须这样表述）**：
- 在 Recall@5 和 **Recall@10 上，`fixed_fusion` 与 Two-Tower 完全打平**（0.00860 / 0.01435，hits 各 172 / 287）。
- 融合只在 **Recall@20**（0.02325 vs 0.02285）和 **NDCG**（0.007434 vs 0.007365，差 **0.000069**）上领先。
- Two-Tower 在 `mean_target_rank` 上**大幅更好**（21,943.8 vs 80,590.3）。
- 所以防御性最强的说法是：**"fusion ≥ 每个单源；在 Recall@5/10 上与最强单源 Two-Tower 打平"** —— **不是**"beats every single source"。
- 且融合 vs Two-Tower 的 NDCG 差距是 **0.000069**，而 M2 的 CI 半宽是 **0.001** 量级 → **远远在噪声内**。
- ✅ 该产物自带 `phase5_cross_check`：三个 accepted arm 重跑后 R@10 与存储值**完全一致**（0.01435 / 0.00925 / 0.0135）。

> ✅ **M2 其实已经发现并记录了这个问题**：`docs/reports/m2-paired-uncertainty.txt:218-229` 的 "REVISED WORDING" 段落明确列出 `PHASE5_HANDOFF.md:151,323,326`、`ARCHITECTURE.md:1500`、`EXPERIMENTS.md:305` 五处 superiority 措辞，并写明 "M2 did not test that comparison, so it is neither supported nor refuted here; it is **flagged as an untested superiority claim** rather than rewritten."
> **但它只是"标记"了，没有改正正文** → 三份文档至今仍在断言这句话。这是**已知但未修**的措辞缺陷。

### 5.6 受控融合（Step 1.1）—— ✅ 已核对 `runs/twotower_benchmark_full/comparison.json: fusion_controls`

`EXPERIMENTS.md:360-375` 与产物的 `fusion_controls` 一致：同 RRF 规则、常数（k=60）、head（1000）、队列、catalogue、evaluator，**唯一差别是 Two-Tower 在不在**。

| Arm | 源 | R@10 | R@20 | NDCG@10 |
|---|---|---|---|---|
| A | popularity + SASRec + metadata | 0.01435 | 0.02325 | 0.00743 |
| **B** | A + **Two-Tower** | **0.01900** | 0.02995 | 0.00982 |
| C | SASRec + metadata | 0.01620 | 0.02525 | 0.00887 |
| **D** | C + **Two-Tower** | **0.02075** | 0.03295 | 0.01105 |

- B−A：R@10 **+32.4%** 相对；D−C：R@10 **+28.1%** 相对。两个对照量级一致 → 效果归属 Two-Tower，而非源集合。
- ⚠️ **这两个对照只报点估计，没做配对检验** → **不能说"显著"**。
- ✅ **数值身份已澄清**（上一版交接没讲清）：
  - **Arm A ≡ Phase-5 `fixed_fusion`** —— `mean_target_rank` 都是 80,590.2987，逐位相同。是**同一体系的重跑**，不是两个独立结果。
  - **Arm D ≡ `comparison.json` 的 `sasrec_two_tower_metadata`** —— R@10 0.02075、R@20 0.03295、NDCG@10 0.011047 全部一致。
  - ⚠️ 所以 `PROJECT_STATE.md:71-72` 里 "RRF A = 0.01745 / RRF D = 0.01810" 是**另一组**数字（那是 Step 2.8 的 TIGER 融合臂，head 与源集不同），**不要与这里的 A/B/C/D 混淆**。

**互补性**（`comparison.json: complementarity`，top-1000 head，修正 exposure 后）：SASRec vs Two-Tower union = 5,972，Jaccard 0.429，lift **1.348**；SASRec vs metadata Jaccard 0.076，lift 1.299。Two-Tower 独占命中 **1,872** 用户。平均 top-10 head Jaccard 仅 **0.083**。

### 5.7 M2（配对不确定性）

方法：exact paired McNemar（hit@K）+ paired user-level percentile bootstrap（20,000 用户有放回重采样，10,000 次，seed 20260201）。

| 审计的结论 | 效应 | 95% CI / p | 判定 |
|---|---|---|---|
| Two-Tower vs SASRec | NDCG@10 +0.00051742 | CI [−0.00053167, +0.00155958]；p=0.4011 | **DIRECTIONAL ONLY** |
| TIGER-FP32 vs SASRec | NDCG@10 +0.00090737 | CI [−0.00003675, +0.00189546]；p=0.7416 | **DIRECTIONAL ONLY** |
| TIGER-FP32 vs Two-Tower | NDCG@10 +0.00038994 | CI [−0.00067173, +0.00147542]；p=0.6461 | **NO DETECTABLE DIFFERENCE** |
| TIGER 在 equal-weight RRF 中的增量 | NDCG@10 +0.00048513 | CI [−0.00015528, +0.00112550]；p=0.4154 | **DIRECTIONAL ONLY** |
| adaptive/rule policy vs fixed_fusion | — | — | **EVIDENCE MISSING**（当时留给 M3） |

> **所有 CI 都包含 0**。诚实的说法是"**未检出差异**"，不是"有差异"。

### 5.8 M3（预注册的真实 LLM agent vs 固定融合）—— ✅ 我直接读了 `runs/m3_execution/*.json`

**主要终点**：`ΔNDCG@10 = −0.0028794446106374695`，95% CI `[−0.003645495129615224, −0.0021236081128100043]` —— **区间完全在 0 以下** → agent **可检测地更差**。

| K | agent NDCG | comparator NDCG | Δ | CI 排除 0？ |
|---|---|---|---|---|
| 5 | 0.004554057233345074 | 0.0055895987983410654 | −0.001035542 | ✅ |
| 10 | 0.004554057233345074 | 0.007433501843982545 | −0.002879445 | ✅ |
| 20 | 0.004554057233345074 | 0.00968019916266781 | −0.005126142 | ✅ |

| K | agent hits | comparator hits | gained | lost | McNemar p | ΔRecall |
|---|---|---|---|---|---|---|
| 5 | 133 | 172 | 54 | 93 | 0.001631 | −0.00195 |
| 10 | 133 | 287 | 28 | 182 | **7.516e−29** | −0.0077 |
| 20 | 133 | 465 | 7 | 339 | 1.578e−90 | −0.0166 |

**🔑 最关键的一条（上一版交接没点出）**：
- agent 的 NDCG 在 K=5/10/20 **完全相同**，hits 也**恒为 133**。
- 原因：`ranking_size_distribution = {"0": 1, "4": 19999}` —— agent 每用户**只产出 4 个候选**。K≥4 后 NDCG/Hits 不可能再变。
- 对照组 head 是 **1000**。所以这不是"排序质量之差"，而是**候选深度 4 vs 1000 的结构性不对称**。

**行为日志**（`run_summary.json` / `m3_ndcg_statistics.json`）：

| 指标 | 值 |
|---|---|
| provider_calls / requests_seen | 39,998 / 39,999 |
| input / output tokens | 52,341,815 / 1,528,578 |
| 平均延迟 | 816.04 ms（总 32,640 s） |
| timeout | **1** |
| `mean_ranking_size` | 3.9998 |
| `mean_steps` / `mean_tool_calls` | 1.9999 / 0.99995 |
| action 序列 | `recommend_from_history,finish` = 19,999；`none` = 1 |
| `search_catalog_count` | **0** |
| `select_source_count` | **0** |
| `source_selection_distribution` | `{"none": 20000}` |

✅ **code-verified**：`m3_ndcg_statistics.json` 自己有一条诚实注释：
> "source_selection is empty for every user because `SELECT_SOURCE` is structurally unreachable in the frozen control plane (`_plane_actions` offers it only when `SIMILAR_ITEM` is registered, which it is not). This does **NOT** mean no retrieval tool ran: `recommend_from_history` executes through the candidate plane and the capability."

> **M3 报告自述**："M3 **cannot** separate routing from retrieval, prompting, reasoning or completion effects, and it does not claim to."
> **M3 PASS 的含义**：预注册实验**执行完毕、统计做完**；**不是**说 agent 赢了 —— **它输了**。
> ✅ **预注册的分类规则确实被套用了**：`M3_PREREGISTRATION.md:100-105` 规定 `SUPPORTED` 需**每个 K** 都满足 McNemar p<0.05 **且** Recall 与 NDCG 的 CI 都排除 0。六个条件在**负方向**上全部成立 → 这个负面结果是 **SUPPORTED**，**不是** DIRECTIONAL ONLY。

**⏱ 总耗时（只有产物里有，报告没写）**：`latency_ms_total = 32,640,011.53 ms` = **32,640.0 s = 9.067 小时**。

**⚠️ 一个必须知道的产物/报告不一致（C7）**：
`m3-agent-vs-fixed-fusion.txt:100-101` 写 "recommend_from_history executes **through the candidate plane** and the capability, and it ran 19 999 times."
但我把 `agent_behavior.jsonl` **全部 20,000 行**聚合后：

| 字段 | 取值分布 |
|---|---|
| `candidates` | **`{0: 20000}`** |
| `grounded_candidates` | **`{0: 20000}`** |
| `multi_source_candidates` | **`{0: 20000}`** |
| `source_count` | **`{0: 20000}`** |
| `ranking_size` | `{4: 19999, 0: 1}` |

→ 行为日志**证明了动作跑了、产出了 4 个候选**，但**没有**证明任何"candidate plane / ledger 计数 > 0"。**不要**从这份产物里引用"candidate-plane candidates > 0"。最可能的解释是那些计数器语义更窄（不是"候选数"），但报告那句话**超出了产物能支撑的范围**。

**⚠️ 报告漏报了预注册要求的行为字段（C8）**：`M3_PREREGISTRATION.md:79-81` 要求报告 mean **steps**、termination reason 分布、wall time 等。这些**只存在于产物**（`mean_steps = 1.9999`、`median_tool_calls = 1`、`termination_reason_distribution`、`total_retries = 0`、wall time = 9.067 h）。**引用 wall time 或 mean steps 时不能说"来自 M3 报告"，要引 `run_summary.json`**，否则会被追问到。

**⚠️ 两个"failure rate"（C9）**：`run_summary.json` 顶层 `failure_rate = 2.50006e−05`（=1/39,999，**按请求**），而 `behavior.failure_rate = 5e−05`（=1/20,000，**按用户**）。报告引用的是后者。两个都对，但单说"failure rate"有歧义。

### 5.9 TIGER / 融合臂的额外产物与两个陷阱

**Step 2.8 / 2.8b 融合臂**（`runs/step28-fusion-results.json`，全部 ✅ artifact-backed）：

| 配置 | 源 | R@5 | R@10 | R@20 | NDCG@10 |
|---|---|---|---|---|---|
| A | SASRec + TwoTower | 0.01055 | 0.01745 | 0.02805 | 0.009136 |
| B | SASRec + TIGER | 0.00940 | 0.01550 | 0.02540 | 0.008216 |
| C | TwoTower + TIGER | 0.01095 | 0.01765 | 0.02675 | 0.009279 |
| D | SASRec + TwoTower + TIGER | 0.01130 | 0.01810 | 0.02905 | 0.009621 |
| A20 | A 限深 20 | 0.00945 | 0.01565 | 0.02480 | 0.008271 |
| D20 | D 限深 20 | 0.01045 | 0.01675 | 0.02635 | 0.009018 |

> 这组 **A (0.01745) / D (0.01810)** 就是 `PROJECT_STATE.md:71-72` 引的数字。**与 §5.6 的受控融合 A/B/C/D (0.01435/0.01900/0.01620/0.02075) 完全不是一组** —— 源集和 head 都不同。**两套都叫 "A/B/C/D" 是这个仓库最大的引用陷阱。**

**⚠️ 陷阱 1（C3）：两个都被叫做 "TIGER 认证 frontier 深度" 的不同量**

| 说法 | 出处 | 值 |
|---|---|---|
| "TIGER full certified frontier" | `step28-fusion-preregistration.txt:34` | min 20 / mean **33.919** / max 667 |
| "certified frontier per case" | `runs/step26-h8-run.json: certificate.certified_items_per_case` | min **21** / mean **34.745** / max 701 |

两者**都有产物支撑**，但**不是同一个量**：前者其实是**融合 head 的深度**（`step28_tiger_heads.jsonl` 全 20,000 行实测 min 20 / mean 33.9192 / max 667），后者才是 H7 证书。差值 0.826 **不等于**队列平均历史 14.8365 → **不能互相替代引用**。

**⚠️ 陷阱 2（C2）**：`step28-fusion-qualification.txt:163-164` 写 "STATUS: **NOT RUN**"（指 common-depth 敏感性分析），但 `runs/step28b-depth-sensitivity.json` 里 **A20/D20 确实跑了**（那是**更晚的** Step 2.8b）。引用 step28 §15b 说"没跑"会是错的。

**TIGER 独立即 `mean_target_rank`**：97,877.07（比 SASRec 的 20,597.74 差很多）—— 又一个"head 覆盖 ≠ 排名质量"的例子。

**🔑 报告自己声明的"军备不对称"（`m3-agent-vs-fixed-fusion.txt:116-122`，逐字）**：
> "The arms have a documented source/action-space asymmetry: fixed_fusion fuses popularity + sequential + metadata in one RRF ranking, while the Agent can reach a **single source at a time** and **cannot express fusion at all**, and **cannot name popularity at all**."

→ 这句是 M3 最重要的"解释"依据：**agent 的动作空间里根本没有"融合"这个选项**，也点不到 popularity 源。所以就算 `k` 修好了，agent 的**上界仍然受动作空间限制**。讲 M3 时必须先讲这句。

---

### 5.10 关键实现细节（✅ code-verified，面试被追问时用得上）

##### 运行时到底注册了哪些源（✅ `recommendation/demo/agent_service.py:329-342`）

```python
plane = CandidatePlane(
    ledger=CandidateLedger(),
    grounding=GroundingVerifier(self._engine, self._metadata),
    history_tool=self._tool,                 # history
    catalog_search=self._catalog_search,     # catalog_search
    similar_item_tool=self._similar_item_tool,   # 需 ENV_SIMILAR_ITEMS 显式 opt-in，默认关
    two_tower_tool=self._two_tower_tool,     # 需 opt-in，默认关
)
```

- **TIGER 完全不在运行时**（`grep tiger recommendation/demo recommendation/api recommendation/tools` → **0 命中**）。
- `similar_item` 默认**不注册** → 所以按 §2 的 `loop.py:532-534`，**`SELECT_SOURCE` 在默认部署下不可达**。
- 这与 `PROJECT_STATE.md:183` 和 M3 预注册 §10 的说法**一致**：这是**已记录的架构事实**，不是 bug。
- ⚠️ 由此可得一个面试结论：**"自适应多源路由"在当前服务路径里既没有实现、也没有被 M3 测到**。说它有，会被当场问穿。

##### 评估语义（`recommendation/evaluation/`）

| 问题 | 答案 | 位置 |
|---|---|---|
| HR@K / Recall@K | `1.0 if rank <= K else 0.0` —— **同一个函数**的返回值同时赋给两个键 | `metrics.py:311-314`, `:370-373` |
| NDCG@K | `1.0 / log2(rank + 1)` if `rank <= K` else 0 | `metrics.py:317-325` |
| **IDCG 在哪？** | **不存在**。全仓 `grep IDCG` 只命中 2 行 docstring。单正样本下 `IDCG = 1/log2(2) = 1`，所以 `NDCG = DCG` —— **正确，但这不是通用多正样本 NDCG** | `metrics.py:36,318` |
| rank 定义 | `1 + #{合法 c : score[c] > score[target]} + #{合法 c : score[c]==score[target] and c < target}` | `metrics.py:134-143` |
| 同分打破 | 分数高者前；同分 **id 小者前** → **全序** | `metrics.py:23-28`, `batched.py:177` |
| masking | `excluded_seen = set(history) - {target}`，candidates = 全集 − excluded | `metrics.py:179-185,215-240` |
| target 强制保留 | batched 路径显式 `mask[row, target] = True`（"the target always stays eligible"） | `batched.py:125-143` |
| PAD 排除 | mask 第 0 列强制 `False` | `batched.py:125-126` |
| NaN/Inf | **拒绝**，且覆盖整个矩阵**含 PAD 槽** —— 因为 NaN 比较会让 NaN target 排第一 | `metrics.py:147-176`, `batched.py:90-104` |
| `mean_num_candidates` | 156,731.261（全集 156,746）→ 平均屏蔽 **14.739** 个 | `arms.json:22` |

**为什么用 scatter mask 而不是按 case 收集候选**：`sum(num_candidates)` 会到数十亿元素；保持 `[batch, num_items+1]` 矩阵做 scatter 只需 O(batch × history_len)。（`batched.py:18-26`）

##### 预处理（`recommendation/preprocess.py`）

- 顺序：**指纹原始文件** → 流式加载校验 → 按 `(user_id, item_id, timestamp)` 去重 → 时间排序（键 `(timestamp, item_id, rating)`，`item_id` 此时仍是 `parent_asin` 字符串）→ 分用户 → **迭代 k-core** → 统计 → 确定性 ID 映射（字典序，从 **1** 开始）→ 原子写 3 个产物。
- k-core：**交替 user pass / item pass，直到某一整轮什么都没删**；`max_rounds=100` 不收敛就 **raise**（不静默截断）。阈值**含等号**（恰好 k 存活）。
- 计数约定：**同一物品的重复评论计多次**（`pair_counts` 建在原始行上）。⚠️ 这与 ItemCF 不同（ItemCF 按用户折叠重复）。
- 原始文件前后各算一次指纹，不一致就 `RuntimeError` → **原始数据永不被改动**。
- PAD 隔离是**在每个下游边界强制**的（catalog 从 1 开始、split 拒绝 PAD、metrics 拒绝 PAD、batched mask 第 0 列为 False），不是靠约定。

##### ItemCF（`recommendation/baselines/` 里**唯一**的成员）

- ⚠️ **scope 修正**：`recommendation/baselines/` **只有 ItemCF**。**popularity 和 BM25 metadata 是 `experiments/benchmark_public.py` 里的 benchmark arm，不属于这个包**。推荐栈里**没有** popularity fallback。
- 相似度：`sim(i,j) = cooc(i,j) / sqrt(freq(i)·freq(j))`，`sim(i,i)=0`；稀疏存储，每对在**两个端点各存一份**。
- `min_cooccurrence` 默认 **1**（未做超参搜索）。
- popularity arm 的频率**只在被评估队列的 `train_history` 上统计** —— 统计全量会把验证/测试目标泄漏进统计量。

##### BM25 metadata arm

- 文档字段 = `(title, store, main_category, categories, features)`；查询字段 = `(title, store, main_category, categories)`。
- 分词器、`k1=1.2`、`b=0.75`、idf 全部**从 `recommendation/rag/retrieval.py` 导入**，防止数学漂移。
- 过滤：出现在 **>5%** catalogue 中的查询词**不计分**（BM25 的 floored idf 会给近乎无处不在的词**正**权重，导致 top-K 退化成 id 序）。
- head 构造：取**最后 5 个** test-history 物品作种子 → 每种子取 top **100** → RRF 融合 → 去重 → 保留 top **1000**。
- ⚠️ 未检索到的物品统一赋 **`-1.0`** → 所以 metadata 类 arm 的 `mean_target_rank` 是**约定值，不可解释**。

---

## 6. 已知缺陷清单（逐条标注我核到什么程度）

### P0 — 会被审阅者当场发现

1. ✅ **`M3_PREREGISTRATION.md:3-5`** 仍写 **"FROZEN and NOT EXECUTED"** / "No M3 run may start… until the provider credential is supplied"；**`:280-287`** 写 `Agent execution: NOT STARTED` / `M3 metric: DOES NOT EXIST` —— 与 `PROJECT_STATE.md:181,194-198`（M3 COMPLETE，有完整统计）**直接矛盾**。→ hash-bound，**只能加指针**。
2. ✅ **`M3_DEEPSEEK_AMENDMENT.md:3-5`** 写 "**PREPARED, NOT EXECUTED** … No DeepSeek call has been made, no M3 metric exists"。
3. ✅ **`docs/reports/m2-paired-uncertainty.txt:276`** 写 "**M3 has not been started.**"
4. ✅ **`docs/ARCHITECTURE.md:1824`** 写 "**No `CandidateSource` member was ever added for TIGER**" —— **假**。`recommendation/control/schemas.py:105` 定义 `CandidateSource.TIGER = "tiger"`，`:121` 将其列入 `CANDIDATE_PRODUCING_SOURCES`。这是上一轮 hardening **新写入的错误**。
5. ✅ **`README.md:14-16`** 与 **`:412-413`** 写"serving path 没有 hosted LLM / 无 API key"，与同文件 **`:242-271`**（可选 LLM Agent 模式，需 4 个环境变量）**自相矛盾**。
6. ❓ `PROJECT_STATE.md:85,92-93,153,166-169` 四处 M2 期"未测"措辞 —— 我逐行读过，**这四处其实都准确**（它们描述的是 **M2 当时的证据缺口**，并明确指向 M3）。**这一条上一版交接报得不准确**，见 §9。

### P1 — 技术/一致性缺陷

7. ✅ **`EXPERIMENTS.md:305`** "**Multi-source fusion beats every single source**" 被**同文档 8.3 的表**（`:335,338`）否定：Two-Tower R@5/R@10 = 0.00860/0.01435 **与 `fixed_fusion` 完全相同**。且 §8.2 的表（`:291-298`）**根本不含 Two-Tower**。
8. ✅ **`experiments/benchmark_public.py:214-216`** docstring 说 "**proportional** to bucket size"，实现（`:242-257`）是**等量 round-robin**（"largest first" 轮转）。`arms.json: cohort.selection` 也写 **round-robin**（artifact 诚实，docstring 错）。**无测试覆盖该分配逻辑**。
9. ✅ **`recommendation/api/app.py:70-73`** 声称有逐轮 INFO 日志（session id/turn id/route/counts/latency）—— ❓ 全包 LOGGER 调用仅 2 处（未亲自复核）。
10. ✅✅ **`recommendation/control/model_policy.py:646-647`** 的 `k` 断链 —— **我完整走通了因果链**：
    - `schemas.py:275`：`ARGUMENTS_BY_ACTION[RECOMMEND_FROM_HISTORY] = EmptyArguments`
    - `schemas.py:335`：`ActionProposal.k: int | None = Field(default=None, ge=MIN_K, le=MAX_K)`
    - `model_policy.py:647`：`fields["k"] = decoded.get("k", 4)`
    - `candidate_plane.py:365`：`limit = action.k`
    - `tools/schemas.py:22-26`：`MIN_K=1, MAX_K=100, **DEFAULT_K=10**`
    - ⚠️ **修正（我最初的判断错了，这里必须精确）**：模型**确实有一个 `k` 通道** —— 但**不是**通过 `arguments`，而是通过模型 JSON 答案里的**顶层 `"k"` 键**（与 `action` 平级）：`model_policy.py:601-606` 把答案 `json.loads` 成 `decoded`，然后 `:646-647` 读 `decoded.get("k", 4)`。所以 `{"action":"recommend_from_history","k":7}` → `k=7`，**会被采纳**。
    - **但该通道对模型是"未公开的"**：✅ 我实测 `build_action_schema((RECOMMEND_FROM_HISTORY,))` 返回 `[{"action":"recommend_from_history","arguments":[],"required_arguments":[],"produces_candidates":true}]` —— **`k` 不出现在里面**（`k` 不是 `ARGUMENTS_BY_ACTION[RFH]` 的字段）；且 `PROMPT_V1`/`PROMPT_V2` 都要求答案**恰好**是 `{action, arguments, rationale}`。→ **一个守规矩的模型永远不会输出 `k`，于是永远落到硬编码兜底 4。** 这就是观测到 19,999/20,000 都是 4 的原因。
    - 另外两个真实缺陷：`ActionProposal.k` **没有** `strict=True`（不同于 `tools/schemas.py:68`）→ `"k":"12"` 被强转成 12、`k:true` 被强转成 1；而 `k=101` **不 clamp、不重试**，直接让整个 run `ABORTED/NO_AVAILABLE_ACTION`。所以这个未公开通道**既能被静默放大，又能让整轮失败**。
    - ⚠️ 注意：**4 不是任何架构常数**，`DEFAULT_K` 是 10，`MAX_K` 是 100。**4 是 `model_policy.py:647` 里的一个魔法数字**，且仓库里没有任何地方解释它。这是 M3 "单源 4 候选"的**直接来源**。
    - ⚠️ 准确表述：这是**直接来源**，不是**唯一根因**。即使 `k` 修好，报告自述的**动作空间不对称**（agent 一次只能碰一个源、**根本表达不了"融合"**、也点不到 popularity）仍然限制上界。**两件事都要说，才是诚实的。**
11. ✅ **`recommendation/demo/agent_policy.py:190-204`** + **`agent_service.py:258-272`**：请求 `sources=["two_tower"]` 时会**静默返回零候选**。`agent_policy.py:196-199` 遍历 `SOURCE_PLANS` 时 `if action not in available: continue`（**静默跳过**），然后落到 `FINISH` 并给出 rationale "**every source the request named has been consulted**" —— 该陈述**不成立**。
12. ✅ **manifest 无自动化校验**：`docs/PROJECT_STATE.md` **不在** `tests/test_docs.py:31-44` 的 `DOC_PATHS`；无校验脚本（已确认 `scripts/` 下无 hash 相关文件）；无 CI job。→ **这是所有状态漂移的根因**（P0 的 1–6 全部由此而来）。

### P2 — 卫生问题

**文档级**
13. ❓ 类型漂移：loop 写入 `candidate_source`/`candidate_eligibility`/`constraint_projection`/`clarification_question`，`agent/state.py:150-208` 未声明（未亲自复核）。
14. ✅ **`docs/EXPERIMENTS.md` 里 `TIGER` / `M3` 各出现 0 次**（`grep -c` 实测：0 / 2 / 0；那 2 次 "M2" 是误匹配）。文档**止于 §8.4**，不含 Step 2.6–2.8b、M1/M2/M3。⚠️ 但**不是完全空白**：`Two-Tower` 有大量覆盖，`GenRec` 就是 Semantic-ID/TIGER 那条线（`:377-427`）。**上一版交接这条要修正**。
15. ❓ `docs/README.md:18` 仍称 `PHASE5_HANDOFF.md` 为 "current handoff"。
16. ✅ **引用不存在的东西**：`run.json` 引用 `/root/AgentRec-X/runs/sasrec_canonical_2026/last.pt` 与 `history.jsonl` —— **两者本机都不存在**（已 `find` 确认）。另外 `run.json` 的所有路径都是 **AutoDL 绝对路径 `/root/AgentRec-X/…`**，在本机不可解析。
17. ⚠️ **`tests/test_memory_service.py:788-814`** 的 `sk-…`/`password=…` 是**故意的密钥检测器夹具**（测试断言它们被拒绝）—— **不要"清理"**，那会静默删掉覆盖。
18. ❓ 陈旧 checkout `/home/zxt/AgentRec-X`。
19. ✅ **陈旧测试计数**：`docs/PHASE5_HANDOFF.md:290` 记 `tests/test_docs.py -q` → "69 passed"；**实测 94 passed**（51.17 s）。
20. ✅ **峰值内存数字自相矛盾**：`benchmark_public.py:142-143` 说 "under 2 GB"；`docs/PHASE5_HANDOFF.md:83-84` 说 3.9 GB；`phase5_benchmark.py:18` 说 ~3.7 GB；`arms.json:306` 记 3.31 GB；`comparison.json:peak_rss_gb` 记 3.88 GB。**部分原因见 #21**（它们不是同一个量）。

**代码内 docstring 与实现不符**（不影响数值，但会被当场抓到）

21. ✅ **`phase5_benchmark.py:263` 的 `peak_rss_gb` 不是峰值** —— 它存的是 `rss_gb()`（当前 `VmRSS`，`:38-42`）；真正的峰值 `peak_gb()`（`:45-46`）只打到 stdout（`:267`）。所以 `arms.json:306` 的 3.31 GB 是**结束时的 RSS**，而 `comparison.json` 的 3.88 GB 是另一套测量 → 数字"矛盾"其实是**两个不同的量**。
22. ✅ **`batched.py:33-35`** 说重复 target 走 "**shared histogram path**" —— `batched_target_ranks` **根本没有** histogram 路径，每行都用自己的 mask 统一计算（`:147-184`）。**陈旧 docstring，语义正确**。
23. ✅ **`batched.py:25`** 说 "PAD never needs to be masked out" —— 代码**显式** mask 了第 0 列（`:126`），且两个 rank 计数都与 mask 相与（`:176-177`）。若第 0 列留 `True`，一个高 PAD 分数会**抬高所有 rank**。docstring 对机制的描述是错的（结果是对的）。
24. ✅ **`cohort_from_cases` docstring**（`benchmark_public.py:271-273`）承诺返回 "**including the per-bucket counts**"，实际 `description`（`:276-292`）只有 min/median/max/mean → **20k 的确切分桶分配无法从产物读回**。
25. ✅ **`itemcf.py:249-252`** 说一对相似度"*stored* under a **single canonical key**" —— `fit` 实际把每对在**两个端点各存一份**（`:344-368`），`baselines/README.md:82-84` 也这么写。docstring 陈旧（无害，访问器两边都查）。
26. ✅ **`datasets/twotower.py:7-11`** 说前缀/目标对覆盖 "for `k in 1..n-1`"，但 `:179-182` **跳过 `position < 2`** → 第一个可用对是 `([i1,i2], i3)`；只有 2 条训练历史的用户**一个对都产生不了**，尽管 `min_history = 2`。行内注释表明跳过是**有意的**，模块 docstring 才是错的一方。

### 6.10 未被上一版交接发现的关键复现性事实

27. ✅ **M3 的数字在本机可以完全重导**：审计代理**独立地**从 `runs/m3_execution/paired_inputs.npz` 重算了 agent/comparator 的 NDCG@5/10/20 与全部 hit 计数，**逐位吻合**；comparator 的排名与冻结的 `fixed_fusion` NPZ **逐字节相同**。→ **M3 不是"不可复现"，它是本机可复现性最强的结果之一。**

### 6.11 边界与后端层的缺陷（✅ 由四路深读交叉确认）

**文档陈旧 / 引用不存在的东西**

- ✅ **`docs/TIGER_BACKEND.md:1-3` 仍写 "Status: specification only. No implementation, no training, no artifact exists for this backend yet"** —— 而 Step 2.4–2.6 已实现、CERTIFIED 基准与生产 handoff 产物都存在（`runs/step26-h8-run.json`、`runs/tiger_backend_handoff_prod/`）。**标题行从没更新过。**
- ✅ **`docs/TIGER_BACKEND.md:128`** 声称 `experiments/benchmark_public.py` 有 `--backend tiger_public` 选项 —— 该文件里**连 "backend" / "tiger" 都没出现过**；真正的臂是 `experiments/tiger_retrieval_arm.py`。
- ✅ **`docs/TIGER_BACKEND.md:106,110`** 把 `recommendation/backends/README.md` 列为已交付文件 —— **该文件不存在**（目录只有 `__init__.py` 和 `tiger_backend.py`）。

**AGENTS.md §19.2 的 10 条硬规则里，有 4 条没有任何测试强制**

| 规则 | 强制方式 |
|---|---|
| 1 无 canonical 身份越界 | ✅ T7a/T7b 测试 |
| 2 backend 看不到 target | ✅ T5 + 运行时 `_FORBIDDEN_KEYS` |
| 3 只用 `train_history` | ✅ 函数签名无 target 参数 |
| 4 PAD 由负哨兵隔离 | ✅ 测试覆盖 |
| 5 adapter 无 ML/后端 import | ✅ T1/T2 |
| **6 后端独立 venv、根依赖不改** | ❌ **无任何断言**（只靠 `pytest.ini norecursedirs`、真实存在的 `.venv`、和文档） |
| **7 evaluator/协议/枚举不变** | ❌ **边界测试里没有断言**（只有 hash 链与 `phase5_cross_check` 间接证据；**没有任何测试枚举 `CandidateSource` 成员**） |
| **8 无证据不集成** | ❌ **无测试**（只有 step27 Q1–Q8 的过程证据） |
| 9 许可证纪律 | ✅ T6（每个文件都要在 `PROVENANCE.md` 里声明） |
| **10 `semantic_id/` 冻结** | ❌ **无断言**（只能验证"包外无人 import"） |

> ✅ **重要正面发现**：`backends/tiger_public/PROVENANCE.md`（18.6 KB，283 行）**真实存在且非常完整** —— 三种标签（`ORIGINAL` / `REIMPLEMENTED_FROM(<repo>@<sha>, <file>:<lines>)` / `DEPENDENCY(...)`）；明确写明 `mclwu22/amazon-genrec` **没有 LICENSE**（默认保留所有权利）→ **不 vendor，只重新实现并标注**；`snap-research/GRID` 是 **Snap 非商业研究许可** → **不 vendor**，若引用其片段必须逐字保留 Snap 声明（"None is quoted today"）。每个重新实现的点都写明了**改了什么**。这是**许可证纪律的正面证据**，面试可以主动讲。

**控制面的额外缺陷（✅ code-verified）**

- ✅ **方法名不存在**：文档（`M3_PREREGISTRATION.md:213`、`PROJECT_STATE.md:183`、`m3-agent-vs-fixed-fusion.txt:98`）都引用 `loop.py::_plane_actions`，**但代码里叫 `_plane_candidate_actions`**（`loop.py:521`，在 `:497` 被调用）。引用文档里的方法名去搜代码会**搜不到**。
- ✅ **`PROPOSE_MEMORY_WRITE` 是死代码**：它在 `ActionKind`（`schemas.py:202`）与 `NON_EXECUTING_ACTIONS`（`:262-266`）里都声明了，但 `available_actions()` 从不提供它、`dispatch_target()`（`loop.py:750-768`）也没有它的分支，且它出现在 `schemas.py` **之外零次**。
- ✅ **`SELECT_SOURCE` 的观测来源会被标错**：`_source_for`（`loop.py:1997-2009`）只在 `result.source` 是 `CandidateSource` 实例时才采信，而 plane 始终写的是**纯字符串**（`candidate_plane.py:441`）→ `isinstance` 永远为假 → 永远走兜底映射，而该映射把 `SELECT_SOURCE` 硬指向 `CATALOG_SEARCH`（`loop.py:2004`）。→ **一次 `SELECT_SOURCE{two_tower}` 会被观测成 `source="catalog_search"`**。
- ⚠️ **`ARGUMENTS_BY_ACTION` 的"全映射"不变量没有断言**：只有靠 `.get()` + 拒绝，**没有**任何运行时检查或测试比较 `set(ARGUMENTS_BY_ACTION)` 与 `set(ActionKind)`。我实测今天是 `14 vs 14, missing: []`，但新增动作时**不会自动报错**。
- ✅ **"3 个回放测试"这个数字对不上**：`demo/agent_policy.py:50` 说 "three replay tests failed"，而 `tests/test_phase31_calibration.py` 里回放 `PHASE3_RECORDING` 的测试有 **4 个**（`:415,426,452,485`）。机制是真实的（请求指纹包含整个 action schema → 改菜单会让归档录像无法重放），但**"3"这个数字没有代码依据**。

### 6.12 模型层的精确参数（✅ 修正"代码默认值"与"实际产物"的混淆）

| 对象 | 代码默认值 | **实际产物用的值** |
|---|---|---|
| SASRec | hidden 64, 2 blocks, 1 head | hidden 64, 2 blocks, **2 heads**, dropout 0.2, max_seq_len **50**（选窗依据：最小且保留 ≥95% transition，实得 98.234%） |
| Two-Tower | batch **1024**, epochs 8 | batch **2048**, epochs 8, lr 0.01, CPU, pairs 1,850,807, final loss 5.322882 / acc 0.105519 |
| GenRec v0 SID generator | d_model **128**, max_items **10** | d_model **64**, 4 heads, 2 layers, max_items **6**, vocab 257, **135,425 参数** |
| GenRec v0 RQ-VAE | L=3, K=256, latent 32, hidden 256 | 同；训练参数 **107,168**；codebook **均匀初始化**（无 k-means++/EMA/dead-code revival —— 那些只在新后端里有） |

- ✅ SASRec 是**唯一在服务路径上的模型**（`api/app.py` 只 import SASRec；Two-Tower 需 `AGENTRECX_TWO_TOWER_CHECKPOINT` opt-in，且**没有任何 API 路由加载它**）。
- ✅ **SASRec 训练细节**：损失 = `softplus(-p) + softplus(n)` 在有效位置上取均值；**每个有效位置恰好 1 个负样本**，从 `{1..num_items} \ train_history` 均匀采样；**刻意把 validation/test target 从排除集中拿掉**（保证它们可作负样本）；手写 batching，**无 DataLoader**；**全程无 LR scheduler**；早停**在实验驱动里而非 trainer 里**；记录的那次跑 17 epoch、best epoch 6、patience 10 用尽；之后**一次封存的 test pass**（`test_used_for_selection: False`）。
- ✅ **Two-Tower 训练**：in-batch sampled softmax（B−1 个负样本）+ **logQ 校正默认开启**；**无验证、无早停、无 epoch 选择**。两个塔都训练（无冻结分支）。
- ✅ **GenRec v0 弱在哪（有归因）**：绑定前缀预算导致检索只覆盖 **534 / 156,746 = 0.34%**；per-code 准确率 **6.66%**，三次方后 ≈0.03%；**4.16% 碰撞率**（3,165 组 / 6,520 物品，distinct SID 153,391）；特征**不训练**。→ 这就是 0.00185 的来源，**不是实现 bug**。
  - ⚠️ 但 `docs/MODEL_EXPANSION_HANDOFF.md:67` 把 151,505 归因于 "**code space** limited" —— **错**：code space 是 16,777,216 ≫ 156,746，真正的限制是**碰撞**。`docs/SEMANTIC_ID.md:38` 写对了，**两份文档互相矛盾**。
- ❌ **本机缺失**：`tiger.pt`、SID 全部产物（`semantic_ids.json`/`layout.json`/`tokenizer.pt`）、`runs/step26_h7_canonical/retrieval_candidates.jsonl`（257 MB）、`/tmp/step28_heads_cache.npz` → **TIGER 数字无法从零重算**。
- ⚠️ **诚实缺口（面试要主动说）**：adapter 的**打分子进程路径**（`TigerBackendAdapter.evaluation_batches`）**只有单测覆盖**；产出 CERTIFIED TIGER 数字的是 `experiments/tiger_retrieval_arm.py`，它消费后端 `retrieve_cli` 的稀疏产物并**自行重挂身份**，**不经过那个 adapter**。即"边界被证明了"对 handoff 物化与分阶段 CLI 成立，但**认证分数矩阵不是走 adapter 出来的**。

---

## 7. 概念速查（agent 新手向）

| 概念 | 一句话解释 | 在本项目的落点 |
|---|---|---|
| **k-core 过滤** | 反复删掉交互数 < k 的用户和物品，直到不动点 | `recommendation/preprocess.py`（**迭代**，不是单遍） |
| **temporal leave-two-out** | 时间序最后两个交互分别是验证/测试目标，之前的作历史 | `evaluation/split.py` |
| **full-catalogue ranking** | 在所有 156,746 个物品里排序，而非只在采样负例里 | `evaluation/batched.py` |
| **seen-item masking** | 屏蔽用户已消费过的物品（**但保留 target**） | `evaluation/batched.py:216` |
| **RRF（reciprocal rank fusion）** | 按 `1/(k+rank)` 把多个来源的排名融合；只用**排名**不用分数 | `control/candidate_ledger.py`，k=60，head 1000 |
| **为什么 rank-only** | 异质模型的**分数不可比**（SASRec logit vs BM25 vs TIGER logprob），排名才可比 | 有意的设计选择，**不要**改成加权融合 |
| **candidate head** | 每个源各自取 top-N（N=1000）再做融合 | 覆盖率瓶颈就在这 |
| **CandidatePlane** | 真正执行候选来源动作的组件 | `control/candidate_plane.py` |
| **GroundingVerifier** | 校验候选确实对应可信身份，防止 agent 编造产品 | `control/grounding.py` |
| **Semantic ID / RQ-VAE** | 把物品内容编码成层级离散 token，让生成模型"生成"物品 | `recommendation/semantic_id/`（**冻结的历史 GenRec v0**） |
| **TIGER** | 用 Semantic ID 做 catalogue-constrained 生成式检索 | `backends/tiger_public/`（**独立 venv**，子进程 + 文件契约跨界） |
| **bootstrap CI** | 有放回重采样用户，估计指标差的置信区间 | M2，20,000×10,000，seed 20260201 |
| **McNemar 检验** | 配对二分类的精确检验（命中/未命中） | M2/M3 |

---

## 8. 面试安全表述（红线）

### ✅ 可以说的

- "三族模型（ItemCF/SASRec/Two-Tower）+ 生成式（Semantic-ID/TIGER）在**同一** cohort、**同一** 协议、**同一** evaluator 下比较过。"
- "我做了**显式的时间戳级泄漏审计**：20,000 case 全部严格有序，0 例非单调，0 例无法解释的 target-in-history。"
- "**瓶颈在候选生成不在排序**：SASRec 的 top-1000 head 只覆盖 20.5% 的目标，所以 R@10 天花板约 0.205。"
- "M3 是**预注册**的真实 LLM agent vs 固定融合，**结果更差**（ΔNDCG@10 = −0.00288，CI 完全在 0 以下，McNemar p = 7.5e−29）。我把负面结果完整报告了。"
- "而且我事后定位到**直接来源**：agent 每用户只能产出 **4** 个候选（对照组 head 是 1000）。`k` 在 `ARGUMENTS_BY_ACTION` 里被声明成空参数、**也没有出现在任何给模型看的 schema 或 prompt 里**，所以守规矩的模型永远拿不到控制权，一律落到 `model_policy.py:647` 的硬编码兜底 `4`（而工具自己的默认是 10）。"
- "同时我强调这是**动作空间问题**、不只是 `k` 问题：M3 报告自己写明 agent **一次只能碰一个源、无法表达融合、也点不到 popularity**，所以即使把 `k` 修好，上界仍受动作空间限制。"
- **工程严谨性（这个项目最强的地方，要主动讲）**：
  - "**28 行 SHA256 manifest 我逐条实算过：28/28 零漂移**（15 份报告 + 13 份源码/预注册）。"
  - "**候选身份边界是用测试断言'不存在'来强制的**：`tests/test_backend_boundaries.py` 28 条守卫全绿（8.33 s），其中两条**流式读取真实的 412,445 用户产物**来断言没有 target 字段。T7a/T7b 直接在 AST 和 JSON key 层面禁止 `parent_asin` 出现在后端树里。"
  - "**许可证纪律可查**：`backends/tiger_public/PROVENANCE.md` 283 行，逐文件标注 `ORIGINAL` / `REIMPLEMENTED_FROM(仓库@sha, 文件:行)` / `DEPENDENCY(包==版本, 许可证)`。两个上游仓库一个**没有 LICENSE**、一个是**非商业研究许可**，所以**都没有 vendor，只重新实现并标注**，且写明了每处改了什么。"
  - "**评估语义有可证伪的细节**：单正样本下 `NDCG@K = 1/log2(rank+1)`、`IDCG ≡ 1`，仓库里**没有 IDCG 计算**且文档说明了原因；seed-item masking **强制保留 target**；NaN/Inf 在**含 PAD 槽的整个矩阵**上被拒绝，因为 NaN 比较会把 NaN target 排到第一。"
  - "**I/O 契约也是防御性的**：预处理前后各算一次原始文件指纹，不一致就 `RuntimeError`；三个产物全部经 `mkstemp + os.replace` 原子写入。"
  - "**k-core 是真迭代不动点**，`max_rounds=100` 不收敛就**抛错**而不是静默截断。"

### ❌ 绝对不能说

| 不能说 | 为什么 |
|---|---|
| "LLM 提升了推荐效果" | 实测**显著更低** |
| "fusion beats every single source" | K=5/10 与 Two-Tower **完全相同**（0.01435） |
| "队列代表总体" | 等量分层；均值 1.75×，`>20` 段约 6.35× |
| "证明了候选深度是原因" | M3 报告**明确拒绝**归因，只能说"行为一致" |
| "agent 能自适应多源路由" | 服务路径没有；M3 里 `SELECT_SOURCE` **结构性不可达** |
| 用绝对指标夸效果 | 最优 R@10 仅 **1.4%** |
| "M3 说明 agent 方法不行" | 它只测了一个 k=4 的特定配置，不能外推 |

---

## 9. 对上一版交接（`/tmp/agentrecx-handoff.md`）的修正

| 上一版的说法 | 实际情况 |
|---|---|
| 端口 8000 **空闲** | ❌ **正在运行** demo server，且 **LLM 模式 available** |
| `data/raw/` 与 preprocessing metadata **都不存在** | ⚠️ 部分错：`data/raw/` 确实不存在，但 **`data/processed/` 三个产物都在**（687 MB），头部计数与统计**可核** |
| `best.pt` / NPZ 的 SHA-256 **未重算** | ✅ `best.pt` **已核实**：`352bd3ae…` 与 `run.json` 记录**逐字节一致**；两个 processed 产物同样一致 |
| "改 offered 规则会让 **3 个归档 Phase-3 回放测试**失败"（仓库自述） | ⚠️ **本机无法验证**：`/tmp/agentrecx-phase3/revised_live.jsonl` **不存在**（只有 `baseline_live.jsonl`），`tests/test_phase31_calibration.py:371,376` 会**自动 skip** |
| `PROJECT_STATE.md` 有**四处** M2 期"未测"措辞是缺陷 | ❌ **报得不准**：这四处都**准确**，描述的是 M2 当时的证据缺口并指向 M3 |
| `EXPERIMENTS.md` 里 TIGER **出现 0 次**，结果文档不含研究轨 | ⚠️ 部分对：`TIGER`/`M3` 确实 0 次，但 **`Two-Tower` 覆盖充分**，且 **`GenRec` 就是 Semantic-ID/TIGER 那条线** |
| 主结果表里 Two-Tower R@10 = 0.01435 与 fusion **打平** | ✅ 对，但要注意 `fixed_fusion`(Phase-5, 0.01435) 与受控融合 Arm A(0.01435) **数值相同、体系不同**，别混讲 |
| M3 agent hits 133 vs 172/287/465 | ✅ 对，且我补上了**关键解释**：133 恒定 + NDCG 在 K=5/10/20 相同，因为候选只有 4 个 |
| —（上一版未提） | ⚠️ **本文件初稿的一处错误已自我修正**：我最初写"模型**没有任何渠道**输出 `k`"。**这是错的** —— `model_policy.py:646-647` 读的是模型 JSON 的**顶层** `decoded["k"]`（与 `action` 平级），所以模型**能**设置 `k`；只是该通道**从未在 schema 或 prompt 中公开**，因此守规矩的模型不会用，实际全部落到 `4`。详见 §6 #10。**这个区别很重要：说成"通道不存在"是错的，说成"通道未公开、模型守规矩就不会用"才对。** |

---

## 10. 未验证清单（不要假称已验证）

- 全量测试套件（~18 min）**本次未重跑**；2360/2326/34 skipped 是 ❓ 继承数字。**只跑过 `tests/test_docs.py` → 94 passed / 51.17 s**。
- CI 在 GitHub 上的实际运行结果（本机**无远端凭据**）。
- Docker 镜像在本机构建（**无 daemon**）。
- `runs/` 中 NPZ 的**数组内容**未逐一解析（例外：`runs/m3_execution/paired_inputs.npz` **已**被独立重算并完全吻合）。
- `runs/phase5_fusion_source_heads.npz`（76 MB）SHA-256 未重算。
- 仓库是 public 还是 private（**无法从本机确认**）。
- `api/app.py` 的 LOGGER 调用计数未亲自复核。
- `agent/state.py` 的字段漂移未亲自复核。
- 11 个大型产物的异地归档（`docs/ARTIFACT_STORAGE.md`，`durable_uri: NONE`）。
- **`data/raw/` 缺失 ⇒ 预处理链无法端到端重算**；`last.pt`/`history.jsonl` 缺失 ⇒ 训练曲线不可复现；`tiger.pt` / SID 产物 / H7 `retrieval_candidates.jsonl` 同样缺失（在 AutoDL 本地）。
- 20k 队列的**确切分桶计数**不可从产物读回（见 §6 #24）。
- 等量分层 vs 按比例分层**是否改变结论**：未测，无产物记录。
- `batched` 与 canonical 路径在**全量 156,746 规模**下的等价性：只有小规模单测覆盖，本次未跑全量对拍。
- "复购最小时间差 790 ms / 412,445 用户全部严格单调"来自 `docs/PHASE5_HANDOFF.md:222-227`，**未**从原始时间戳重导。
- `recommendation/models/sasrec.py` 的 `full_catalog_scores` 与 inference 包未逐行读。
- 上一版交接称"改 offered 规则会破坏 3 个 Phase-3 回放测试"是仓库自述 —— ⚠️ 本机**无法验证**（见 §9）。

### PROSE-ONLY：只在报告正文里、**本机无产物**的数字（引用前必须说"报告记载"，不能说"我验证过"）

| 数字 | 出处 |
|---|---|
| TIGER 检索探针 "512 cases (warm) 4.11 cases/s / 124.7 s"、"8 cases (cold) 2.26 cases/s" | `docs/reports/step26-implementation-and-gate-report.txt:307-308` |
| 512-case frontier 统计 `max 118 / mean 35.023 / min 22 / p95 61` | 同上 `:311`（磁盘上只有 64-case 版，mean 36.75） |
| H7 前缀预算校准 "512 real cohort cases: min 769 / p50 769 / p95 771 / p99 962 / max 1152 / mean 772.9" | `docs/reports/step26-h7-execution-manifest.txt:42-43` |
| Step-2.7 "0.000906 s per plane.execute call" | `step27-candidatesource-qualification.txt:221-224`（`step27-approximate-summary.json` 有 4.1815 cases/s，但**没有 timing 字段**） |
| Step-2.7 测试总数 "86 passed in 6.91 s"、"FULL suite 2171 passed / 34 skipped / 674.9 s" | 同上 `:312-313`（无 JUnit 产物；**不要为了核对去重跑全量**） |
| BF16-vs-FP32 各表（R6/R9/R10 轨迹） | `fp32-canonical-remediation-report.txt:117-119,199-253,317-363`（底层日志在 AutoDL） |
| `data/processed/Sports_and_Outdoors_products_manifest.json` | `docs/EXPERIMENTS.md:431` 引用，**磁盘上不存在** |
| TIGER `tiger.pt`（95e5cb6f…, 45,181,927 B）、SID `semantic_ids.json`/`layout.json`/`tokenizer.pt`、`runs/step26_h7_canonical/retrieval_candidates.jsonl`（257,443,487 B）、`/tmp/step28_heads_cache.npz` | **本机全部缺失** → TIGER 数字**无法从零重算** |
| M3 preflight（`preflight.json`）：5 attempts / json_parse 3/5 / offered **仅 2 个动作** | 产物在，但**不在** `ARTIFACT_SHA256.json` 里；且它只 offer 了 2 个动作，**不能用它论证 `SEARCH_CATALOG` 是否被 offer** |
| ItemCF 基准 | `run.json:itemcf_comparison = "PENDING (no same-artifact full-data ItemCF benchmark exists)"` → **仓库里没有任何 ItemCF 数字，永远不要引用** |

**本机 ✅ 可重导的**：M3 全部统计（从 `paired_inputs.npz`，已逐位复核）；SASRec/Two-Tower/GenRec-v0/TIGER/popularity/metadata/fixed_fusion 的全部 **hit 计数**与 **mean_target_rank**（从各自的冻结 NPZ 重算，精确吻合）；M2 全部 9 个冻结聚合量。
**本机 ❌ 不可重导的**：任何 TIGER 数字（无 checkpoint/SID/检索产物）、任何重训练（无 raw data、本机无 GPU）、Step-1.1 的 A/B/C/D 配对重建（SASRec/Two-Tower 的 depth-1000 head 只在已缺失的 `/tmp/step28_heads_cache.npz` 里）。

---

## 11. 下一步可选方向（**本次未执行任何一项**）

### 档 1：低风险修正（1–2 小时）
- 修 P0 的 6 处（hash-bound 的**只加指针**，不改正文）。
- 新增 **manifest 校验脚本 + CI step**；把 `PROJECT_STATE.md` 加进 `tests/test_docs.py` 的 `DOC_PATHS`。→ 一次性根除漂移。
- 修 `benchmark_public.py` docstring、"fusion beats every single source" 措辞、Two-Tower 静默零候选（改为显式错误）。

### 档 2：`k` 参数（收益最大，但会改变 agent 动作层）
- 把 `k` 加进 `ARGUMENTS_BY_ACTION[RECOMMEND_FROM_HISTORY]`，让模型能看见并设置。
- ⚠️ 必须先决定 M3 的处理方式：M3 报告 hash-bound 且声明"只跑一次"。**建议只加指针**，声明"M3 结论成立于旧行为（k 不可控）"，不重跑、不改数字。
- 这同时修掉 `model_policy.py:647` 的魔法数字 **4**。

### 档 3：较大、需谨慎
- 把"融合"做成 agent 可选动作（`RECOMMEND_FUSED`），让 agent 的**退路 ≥ 基线**；候选深度对齐；成本指标（token/延迟）纳入判据。→ 需**新开预注册**，不动 M3。
- 让 agent 用在它该赢的地方：硬约束协商、澄清、多步比较（动作已在仓库但服务路径未接线）。
- **不要**为了让 agent 好看而改冻结协议、调 TIGER 权重或动 M3 语义。
