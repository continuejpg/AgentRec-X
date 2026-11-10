# P14-B0 — Memory Repair & Preference-Aware Recommendation Baseline

> 范围：仅 P14-B0。完成 M9 更新/撤回语义最小修复、合法 history-state 读取、Preference-Aware Scorer V0、40 条 Pilot Dev 固定候选池公平对照、消融与逐任务分析。
> 未训练任何模型，未改 P14-E 标签，未改 Serving/Planner/Ledger，未 commit。
> 证据：`runs/p14b0/`；测试：P14-B0 13 项 + 相关回归共 **199 passed**（详见 §11）。

## 0. 一句话结论

M9 的 stale preference 根因已定位并修复（撤回目标解析 + 跨 kind 值匹配）；新增的合法 history-state 读取在 40 条任务上恢复出有效偏好的任务数从 lexical read 的 **11/40** 提升到 **34/40**；在**完全相同**的 BM25 候选池上，Preference-Aware Scorer 把可推荐任务的 Hit@1 从 **0.3462** 提升到 **0.6923**、MRR@10 从 **0.5190** 提升到 **0.7388**，No-History 与 Shuffled-History 均低于 True-History。结论仍是 **pilot-dev provisional**：15 条待人工审核、缺少 ASK/拒绝决策门、标签与评分器存在弱监督同源风险，因此 P14-B0 总判断为 **LIMITED**。

## 1. M9 stale preference 的根因

P14-E 更新的用例：`I prefer over-ear.` → `Forget my earlier preference for over-ear.` → `I now prefer in-ear.` 之后，`over-ear` 仍保持 active。

根因（两处叠加，均与具体任务无关的通用缺陷）：

1. **撤回正则不识别含连字符的取值**：`recommendation/memory/extraction.py::_RETRACTION` 的 leading/trailing 捕获字符集为 `[a-z0-9 ]`，`over-ear`/`in-ear` 在 `-` 处截断，正则整体不匹配 ⇒ 没有移除指令。`noise cancelling`（空格）可以匹配，所以 P14-E 只暴露出 form-factor 这一类。
2. **撤回目标未去除话语/指代词**：即便匹配成功，target 是 `earlier preference for over-ear`；`_retraction_target` 只去掉 `preference` 等少数 filler，保留 `earlier`/`for`，生成的 value 与存储值 `over-ear` 不相等。
3. **写入侧值比较是严格 casefold 相等**：`service.py::_apply_locked` 对 `kind=None` 的撤回用 `entry.value.casefold() == removal.value.casefold()`，因此上述任何规范化差异都不会命中。

修复前复现（真实 extractor 输出）：`I prefer over-ear.` → candidate kind=`free_form_constraint`, value=`over-ear`；`Forget my earlier preference for over-ear.` → **removals=[]**（根因 1）；`Forget my earlier preference for noise cancelling.` → removal 存在但 target 含 `earlier/for`（根因 2）。

## 2. 更新/撤回语义的修复

最小修复（只改 M9 两个文件，不改变 M5/M9 核心契约，不新增 store）：

| 文件 | 修改 |
| --- | --- |
| `recommendation/memory/extraction.py` | `_RETRACTION` 的 leading/trailing 字符集加入 `'` 与 `-`；`_retraction_target` 增加确定性停用词剔除（earlier/previous/old/for/about/the/my/…），空目标返回 `(None,None)`，不猜测 |
| `recommendation/memory/service.py` | `_apply_locked` 的 value-only 撤回匹配改为规范化比较（`re.sub(r"[^a-z0-9]+","",value.casefold())`），使 `over ear`/`over-ear`/`OVER-EAR` 命中同一存储值；kind 撤回语义不变 |

语义结果（已测试）：

- 同一属性新值：显式撤回后新偏好 active，旧值 REMOVED（tombstone 保留在 history）；**普通新增偏好仍不会隐式覆盖**（M9 原有契约 `REPLACE` 才 supersede，保留）；
- 显式撤回：旧值 REMOVED，不再出现在 active 读取/HistoryStateReader；仍出现在 audit trail；
- 显式否定：`avoid` 条目，与撤回区分（否定是 active 的负约束）；
- 冲突：不同 polarity 可同时 active（由上层/决策策略处理），不擅自合并；
- 顺序：`logical_seq` 与 turn 顺序保持确定；M5 event 契约未改；
- 多用户：store 以 `user_key` 隔离；
- 证据：`source_turn_id/source_text` 保留（测试覆盖）。

修复后端到端复现（真实 M9 service）：`prefer over-ear → prefer wireless → forget over-ear → prefer in-ear` 的 active = `wireless, in-ear`；history = `over-ear REMOVED, wireless ACTIVE, in-ear ACTIVE`。

## 3. 合法 history-state 读取优于旧 lexical-only 路径吗？

**是，在本 Pilot Dev 的任务中明显更好，但不是“语义完美”。**

旧路径（`ShoppingCompanionMemory.search(current_query)`，P14-A 的 M9 lexical read）：40 条任务中只有 **11/40** 能返回任意 active preference；因为当前购物需求通常不重复历史属性词（M9 读取按 value token 与 query 交集过滤）。

新合法路径（`recommendation/preference/history.py::HistoryStateReader`）：
- 只读 `PreferenceMemoryService.get_active_preferences(user_key)` / `get_memory_history(user_key)`，即 **M9 已应用的当前状态**，不重建 store、不读 evaluator-only、不用标签构造 query；
- 要求 `conversation_id == user_key`，拒绝跨 persona 读取（测试覆盖）；
- 预算 `max_preferences=10`；active 按 `(query token overlap DESC, logical_seq DESC, memory_id ASC)` 排序；
- 保留 `memory_id / source_turn_id / source_text / logical_seq`，同时返回 REMOVED/SUPERSEDED 的 audit trail。

实测：40/40 任务中 **34/40** 读到了 active preference（lexical 为 11/40），5 条任务读到 inactive entries（撤回/更新可见），13/40 任务的 active 偏好与当前 query 有 token overlap。两者对比如下：

| 读取路径 | 有 active 偏好的任务 | 说明 |
| --- | ---: | --- |
| lexical read（current query） | **11/40** | 依赖当前 query 与历史词重合 |
| history-state read（本修复） | **34/40** | query-independent 重建，再做预算内相关性排序 |
| 有 inactive entries | 5/40 | 撤回/更新可审计 |

限制：history-state read 返回的是 **canonical M9 值**（如 `over-ear`/`noise cancelling`），部分自然语言偏好（如 “comfort”）无法 canonicalize 到 P13-R 文本属性，被 harness 明确丢弃并计入 unknown；它不是“全知偏好”，只是合法的当前状态读取。

## 4. Preference-Aware Scorer 使用的特征与证据

模块：`recommendation/preference/scorer.py`（`PreferenceAwareScorer`，`SCORER_VERSION="preference_scorer_v0.1"`），输入/输出严格遵守 C4：

输入：`preferences`（仅来自上述合法读取；active）、`current_query`、**固定候选集**（含 `product_ref{namespace,item_id}`、`query_rank`、`query_score`、商品属性证据、form_factor）、`hard_requirements`（从 **agent-visible query** 解析，非标签）、`namespace`。

输出（每个候选）：`product_ref`、`final_score`、`query_relevance_score`（池内 rank 归一化 `(n-rank)/(n-1)`）、`preference_match_score`、`preference_conflict_score`、`matched_preferences`、`conflicting_preferences`、`unknown_preferences`、`hard_status`/`hard_tier`、`evidence_refs`（catalogue evidence + memory_id/source_turn_id）、`presentable`、`rank`、`scoring_version`。

评分（权重在运行前固定，未在 40 条上调参）：

- Query relevance：`0.50 × rank_norm`；
- 正偏好匹配：每个 **有文本证据** 的匹配属性 `+0.20`（最多 3 项）；
- 负偏好冲突：每个商品明确具备的 avoid 属性 `-0.35`；
- 硬约束：分三档 **verified(0) > unknown(1) > violated(2)**；violated 候选 `final_score = -1 - rank/n`，排在所有非 violated 之后且 `presentable=False`；unknown 档只加 `-0.10` 惩罚，**不当作满足**；
- 属性 UNKNOWN：0 正分，明确进入 `unknown_preferences`；
- 稳定 tie-break：`hard_tier ASC, final_score DESC, query_rank ASC, namespace, item_id`；相同输入输出逐字节稳定（测试覆盖）；
- 不搜索：scorer 只接收候选，不调用任何 catalogue/search（测试与代码审查）；
- namespace：候选 namespace 与声明不一致时直接 `ValueError`（测试覆盖）。

## 5. Search 与 Recommendation 的职责区分

| 层 | 职责 | 本轮的实现 |
| --- | --- | --- |
| Search | 召回固定候选池：query→BM25 top-K，附 rank/score/score_kind | 复用 accepted `CatalogSearchSource`（P13-R，catalogue namespace `amazon_electronics_c4`）；40 条任务的候选池已冻结落盘 `runs/p14b0/candidate_pools.jsonl` |
| Recommendation | 在 **同一候选池** 内用长期偏好与硬约束重排，解释匹配/冲突/UNKNOWN | 新 `PreferenceAwareScorer`；不增删候选、不二次检索、不使用目标标签 |
| 决策（本轮的缺口） | ASK / 拒绝 / 继续检索 | 尚未在 P14-B0 实现；scorer 只输出 `presentable` 标记 |

硬约束仍由“商品证据 + form-factor 互斥”判定；Scorer 的 hard tier 是独立实现，未改动 `CandidateEligibilityEvaluator`，两者都遵守“UNKNOWN 不等于满足”。

## 6. 相同候选池：Search Only vs Search + Preferences 的实际指标

固定配置：BM25 top-K=20，score_kind=bm25，同一 query/目录/候选/标签/任务；26 条可推荐任务（其中 22 条有 verified candidate 在池内；4 条 retrieval unreachable）。逐任务明细见 `runs/p14b0/*.jsonl`。

| 方法 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | top1 matched | top1 conflicts | top1 hard violations | top1 hard verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A. Search Only** | 0.3462 | 0.6923 | 0.8462 | 0.5190 | 14 | 3 | 0 | 24/26 |
| **B. Search + Preferences（True History）** | **0.6923** | **0.8077** | 0.8462 | **0.7388** | **32** | **0** | 0 | **26/26** |
| C. No History | 0.3846 | 0.7308 | 0.8462 | 0.5468 | 16 | 2 | 0 | 26/26 |
| D. Shuffled History（诊断负对照） | 0.3077 | 0.6923 | 0.7692 | 0.4763 | 16 | 4 | 0 | 25/26 |

- Hit@10 三者相同（0.8462），因为 22 条可达任务的 verified 商品本来就在 BM25 top-10 内；提升集中在 **排序（Hit@1/MRR）**，不是召回；
- B 相对 A：Hit@1 **+0.3461（+34.6pp）**、Hit@5 **+11.5pp**、MRR@10 **+0.2198**；top1 匹配偏好 14→32，明确冲突 3→0；
- 逐任务类别（26 条）：`both_right=9, b_only=9, both_wrong=4, retrieval_unreachable=4, a_only=0`；即 **B 修复了 9 条、没有把 A 的正确答案改错**；
- 4 条 `retrieval_unreachable` 不计入 Ranker 错误（verified total>0 但 top-20 无 verified candidate），已在 `error_analysis.jsonl` 记录。

## 7. True History / No History / Shuffled History 的差异

同一固定候选池与同一 scorer 配置：

- **True History**：Hit@1 0.6923 / MRR 0.7388，top1 匹配偏好 32；
- **No History**：Hit@1 0.3846 / MRR 0.5468，top1 匹配偏好 16 —— 只靠 query 与 hard-tier 重排，明显低于 True History；说明提升主要来自**恢复出的历史偏好**，而不是 hard-tier 排序副作用；
- **Shuffled History**（每任务使用另一合法 persona 的历史读取结果；不是 evaluator 标签）：Hit@1 0.3077 / MRR 0.4763，**低于 No History 与 Search Only**；说明无关历史会产生干扰，History 的效果依赖“读对用户的历史”，不是任意历史文本都加分。

以上均为 Pilot Dev 诊断，不是泛化结论。

## 8. Same-Query Cross-Persona 是否体现可解释的偏好差异？

5 组相同 query / 不同 persona（固定候选池必然相同，因为 query 相同）：

| 组 | 任务 | top1 不同 | verified 最佳排名（左/右） | top1 命中偏好（左/右） | 解释 |
| --- | --- | --- | --- | --- | --- |
| headphones commuting | 0013 / 0014 | 是 | 1 / 1 | 1 / 2 | 两组偏好不同，均命中 verified；排序差异由 over-ear/foldable 等证据产生 |
| headphones work calls | 0015 / 0016 | 是 | 1 / 1 | 2 / 2 | 同 query 下不同偏好选择不同商品 |
| keyboard office | 0017 / 0018 | **否** | None / None | 0 / 0 | 该 query 的 top-20 池内没有 verified candidate（retrieval coverage 限制），无法观察偏好差异 |
| mouse office | 0019 / 0020 | 是 | 8 / 1 | 0 / 1 | 偏好把右侧用户的 verified 商品排到第 1，左侧仍在第 8 |
| speaker living room | 0021 / 0022 | 是 | 1 / None | 1 / 0 | 左侧命中 verified；右侧池内无 verified，差异为 preference evidence 驱动但受 coverage 限制 |

结论：**4/5 组**在同一候选池上产生与当前有效偏好一致、可追溯到证据的排名差异；1/5 因候选覆盖不足无法评价。排名变化≠改善，本表同时给出 verified 排名与偏好命中；没有为了制造差异而修改 query 或标签。

## 9. 哪些商品属性因 UNKNOWN 不能参与评分？

P13-R `catalog_meta.jsonl` 只有 `item_id/category/text`；因此：

- **完全 UNKNOWN**：price、brand、size、material、stock、warranty（没有对应字段，绝不补全）；
- 可参与评分的是 17 个 **文本证据属性**（wireless/wired/noise_cancelling/over_ear/on_ear/in_ear/microphone/gaming/waterproof/foldable/mechanical/rgb/rechargeable/multidevice/usb/silent/ergonomic）；
- 当前候选池中 **92.75%** 的商品至少缺一个被请求属性证据（P14-E 统计），所以评分中 `unknown_preferences` 数量高（A:54 → B:39，仍高）；
- 不可验证的硬约束（price/warranty）在 harness 中标记为 `unverifiable`，scorer 的对应 hard tier 为 unknown：**不会被高 BM25 分数绕过**；
- 自然语言偏好若无法 canonicalize 到上述词表（如 “comfort”），读取后明确丢弃并计入 unknown，不猜测。

## 10. 15 条待人工审核任务是否影响结论？

影响明显且已如实报告，未通过筛选任务美化结果：

- 15/40 pending（`review_status=pending_human_review`，原因：same-query 对照、no_verified_match、unverifiable、少量候选）；
- 其中 **10/26 条可推荐任务** pending。全量 vs 排除 pending 的敏感性：

| 子集 | A Hit@1 / MRR | B Hit@1 / MRR |
| --- | --- | --- |
| 全量 26 条 | 0.3462 / 0.5190 | 0.6923 / 0.7388 |
| 排除 15 条 pending（16 条） | 0.4375 / 0.6236 | **0.7500 / 0.8177** |

方向一致（B 仍优于 A），但绝对数值变化，说明结论依赖待审标签质量；因此报告标注 **provisional / pilot-dev**，不得对外宣称已证明个性化推荐有效。审核队列仍保留在 `data/p14e/benchmark/review_checklist.json`，未自动升级为 human_verified。

## 11. 是否引入新旧代码污染？

- 只修改了 **2 个共享文件**：`recommendation/memory/extraction.py`、`recommendation/memory/service.py`（M9 输入解析与写入匹配的最小修复，理由见 §1–2）；
- 新增 5 个源码文件：`recommendation/preference/{__init__,history,scorer}.py`、`experiments/p14b0/{__init__,evaluate}.py` + 1 个测试文件；
- **未修改**：Serving V1、SASRec/Two-Tower、P13-R 数据、P14-E 标签/任务（哈希测试 `test_p14e_benchmark_files_unchanged` 通过）、Planner/规则、CandidateLedger/Eligibility、M5 事件契约；
- 工作区原有的 29 个未提交修改保持不变（包括原有的 `recommendation/preference_matching/schemas.py` 修改，本轮未触碰）。

回归测试（本机 CPU / WSL）：

| 批次 | 结果 |
| --- | --- |
| P14-B0 新测试 `tests/test_p14b0_memory_scorer.py` | **13 passed** |
| 相关回归：memory service/store/query_adapter/bm25、P14-A memory、P14-E benchmark、constraint eligibility、serving fusion | 合计 **199 passed**（含 13 新测试，59.3 s） |
| 修复前后 M9 快速回归（test_memory_service 等 134 项） | 修复前 134 passed；修复后合并批次全绿 |

资源（本机 CPU）：evaluate wall **96.33 s**；catalogue load 42.56 s；BM25 index 19.32 s；单 Query 搜索 median **818.71 ms** / max 1530.45 ms；单 Query 评分 median **3.32 ms** / max 7.16 ms；未用 GPU；未运行全量 pytest。

## 12. 是否具备扩大 Benchmark 与进入 P14-B 的条件？

**有条件具备（P14-B0 = LIMITED），尚不能直接进入训练/正式评测。**

前置条件（按顺序）：

1. **完成 15 条 pending 人工审核**（尤其 10 条可推荐任务）；未完成前所有数字只能 provisional；
2. **加入决策门**：本轮的 scorer 只重排；14 条非推荐任务中 **9 条**会被 scorer 给出 top1（conflict 3、withdrawal 1、insufficient 5），因为当前没有 ASK/拒绝策略。no_verified_match/unverifiable 5 条因 hard unknown 未被呈现（符合预期）。P14-B/C 前必须把 `presentable` 接入 Planner 的 ASK/FINALIZE 语义，而不是把 top1 直接当最终答案；
3. **改善候选覆盖**：26 条可推荐任务中 4 条 retrieval unreachable、4 条 both wrong；P13-R 是 C4 sampled 目录，扩展前要决定是否补充更多真实 metadata/更完整的候选；
4. **降低弱监督同源性**：P14-E 标签与 scorer 都使用文本属性规则，B 的提升可能有“自一致性”成分；扩展时必须引入独立标注/人工审核或另一套属性来源，不能把本报告数字当作真实用户收益；
5. **冻结 reserved-test**：pilot_dev 不可改名成 frozen test；扩展 200–500 时按 P14-E spec §7 做 persona/模板/同 query 组隔离。

### 最终判定

| 判定 | 结果 | 理由 |
| --- | --- | --- |
| `MEMORY_FIX` | **PASS** | stale 根因定位+通用修复；13 项新测试+134 项 M9 回归通过；M5/M9 契约未改 |
| `HISTORY_RETRIEVAL` | **PASS** | 合法 history-state read：34/40 vs lexical 11/40；授权/预算/inactive 审计/无 oracle；但需承认 canonicalize 丢弃部分自然语言偏好 |
| `PREFERENCE_SCORER` | **PASS** | 确定性、可解释、UNKNOWN 不冒充满足、hard violation 不靠相关度通过、namespace 拒绝、无二次检索（测试覆盖） |
| `FAIR_COMPARISON` | **PASS** | 同一候选池/query/标签/任务；A/B/C/D 全量逐任务落盘；无 oracle 注入；允许并如实报告 retrieval unreachable |
| `LABEL_RELIABILITY` | **LIMITED** | 15/40 pending（10/26 recommend）；标签与评分器同源弱监督；无人类核查 |
| `LEGACY_REGRESSION` | **PASS** | 199 项相关测试通过；P14-E 数据哈希未变；Serving/Planner/Ledger/SASRec 未触碰 |
| **P14-B0** | **LIMITED**（provisional pilot-dev；Memory 修复与基线可用，但标签、决策门与覆盖未达 GO 条件） | 具备扩大 Benchmark 的基础，不具备宣称个性化收益或直接进入 P14-B 训练的条件 |

## 附：产物与复现

- 评估：`python -m experiments.p14b0.evaluate`（WSL CPU）→ `runs/p14b0/`
  - `candidate_pools.jsonl`（固定候选池）、`search_only.jsonl`、`search_preferences.jsonl`、`no_history.jsonl`、`shuffled_history.jsonl`、`memory_reads.jsonl`、`same_query_cross_persona.jsonl`、`error_analysis.jsonl`、`non_recommend_audit.jsonl`、`timings.jsonl`、`summary.json`、`manifest.json`
- 源码差异：[`P14B0_SOURCE_DIFF.txt`](P14B0_SOURCE_DIFF.txt)
- 未改动：`data/p14e/benchmark/*`（冻结；哈希测试）、Serving/Planner/Ledger/Eligibility、旧模型；未 commit
