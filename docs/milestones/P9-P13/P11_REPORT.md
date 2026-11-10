# P11 — Grounded Personalized Shopping Benchmark 报告

## 0. 结论

**已建立 164 个可执行、可追踪的 Dev 购物任务，覆盖 86 个真实用户。但本地 Amazon 处理产物没有 review 文本，Level A（明确历史偏好陈述）为 0；历史个性化只能达到 Level B（重复行为倾向）。**

| 维度 | 判断 |
|---|---|
| 真实历史/商品空间 | **VERIFIED** |
| 任务 Schema / 可执行性 | **VERIFIED** |
| 历史个性化证据 | **LIMITED：Level A = 0；Level B = 100 tasks** |
| 相同 Query 不同历史 | **VERIFIED：69 pairs / 138 task instances** |
| BM25 / BGE / Hybrid 候选覆盖 | **LIMITED** |
| History 来源候选覆盖 | **LIMITED：17/164** |
| 最小真实 Rule-Agent 执行 | **LIMITED：strict success 2/10** |
| P13 Planner 决策实验 | **暂不具备完整条件** |
| **P11 总判断** | **LIMITED** |

## 1. 数据核验与证据等级

本地可用：

- `Sports_and_Outdoors_sequences.json`：412,445 用户、156,746 商品、3,500,587 交互；
- 交互有 rating 和 timestamp；
- **没有 review 文本**；
- 商品 metadata 有 title/categories/features/description/price_text；
- P8-D 300 Dev cases、mapping、BGE index 可用。

因此：

- **Level A = 0**：没有“用户明确正向评论某项属性”的证据；
- **Level B = 100 tasks**：重复历史行为/类别倾向；
- **Level C**：任务当前声明约束；
- **UNKNOWN = 64 task instances**：无足够历史/属性证据。

rating 不能自动解释为某商品属性的明确偏好。

## 2. Benchmark 规模

Machine-readable：

- `data/p11_benchmark/tasks.jsonl`
- `data/p11_benchmark/schema.json`
- `data/p11_benchmark/manifest.json`
- `data/p11_benchmark/query_pools.json`
- `P11_BENCHMARK_SCHEMA.md`

| 项 | 值 |
|---|---:|
| Task instances | **164** |
| Unique real users | **86** |
| Historical personalization | 138 |
| Explicit intent + history | 15 |
| Multi-turn state change | 3 |
| Clarification required | 4 |
| Unsatisfiable / unknown | 4 |
| Same-query pairs | **69**（138 tasks） |
| Level A | 0 |
| Level B | 100 |
| UNKNOWN | 64 |
| Tasks with verified candidate pool | 156 |
| Tasks without verified pool | 8 |
| Need human preference audit | 100 |
| Machine-verified catalog constraints | 56 |
| Need human (unknown/clarification) | 8 |

全部任务为 `split=dev`；冻结 Holdout 未使用；没有 reserved evaluation 任务。

## 3. 任务与 Gold 定义

### Level B 历史证据

只从真实 `train_history` 中提取匹配商品类型的历史 item，记录 ASIN/title/category/rating/timestamp。至少两个 matching prior interactions 才标 Level B。

**Level B 只是行为倾向，不是“用户明确喜欢某属性”。** 相同 Query 的 evidence/no-evidence 用户对用于检查系统是否能区分历史信号，但不能制造两个虚假偏好。

### Level C 当前请求

商品类型、预算、目标由任务声明；当前硬约束优先于历史兴趣；Level C 不算历史发现。

### Verified candidate pool

Gold 不是 next-item target，也不是 BM25/BGE top result。候选 pool 由：

1. catalog metadata 的 title/category/features/description 匹配；
2. Product Role `MATCH`；
3. `price_text` 可验证预算；

确定性构造，按 ASIN 排序，最多 100 个。BM25/BGE 排名不参与 Gold。它是 **bounded verified candidate pool**，不是全 156,746 目录 oracle。

UNKNOWN/missing price/未知角色不补齐、不伪造；不能自动算 MATCH。

## 4. 检索覆盖

164 tasks，其中 156 个有 verified pool：

| Source | Reachable@10 | Reachable@50 | mean found@10 |
|---|---:|---:|---:|
| BM25 | **36 / 156** | **140 / 156** | 0.776 |
| BGE | 14 / 156 | 102 / 156 | 0.256 |
| Hybrid | 13 / 156 | 86 / 156 | 0.500 |
| History source（P8-D cache Top-50） | — | **17 / 164** | — |

解读：

- BM25 Top-50 覆盖较强，但 Top-10 只有 36/156；候选广而头部排序有限；
- BGE/Hybrid Top-10 覆盖低，无法证明优于 BM25；
- History 来源仅 17/164 可达，这是历史个性化最明显的限制；
- Hybrid Top-50 低于 BM25，说明“放入更多来源”不等于质量提升。

历史 affinity proxy（仅 Level B，不是人类确认满意度）：BM25 61/100，BGE 61/100，Hybrid 59/100。

## 5. 最小真实 Agent 执行

Rule Agent + 真实 Hybrid/BGE + 真实 user history，执行 10 个分层任务（不是只挑容易任务）：

| 指标 | 结果 |
|---|---:|
| Tasks | 10 |
| Strict verified success（final 全在 bounded pool） | **2 / 10** |
| Catalog-validity success（role/price 合法，pool 外也算） | **4 / 10** |
| Invalid finalize | **5 / 10** |
| HardConstraintViolation | **3 / 10** |
| Execution errors | 0 |

失败分类：

- 无 evidence 的 D1 出现 role UNKNOWN；
- multi-turn goal switch 后 final role MISMATCH；
- clarification tasks 被直接 FINALIZE，而不是 ASK/UNKNOWN；
- unknown-role 任务 finalize UNKNOWN 角色商品；
- `under $1` unsatisfiable 无 final（正确没有乱给候选）。

部分 final 商品 role/price 合法但不在 bounded pool 中，只能算 **catalog-valid but unverified**，不能算 strict success，也不能自动算错误。

执行使用 Rule Planner，state mode 为 `off`；Qwen 未启动，GPU 未用于训练；多轮 M5 有效状态语义尚未完整验证。

## 6. 十个问题的回答

1. **是否存在足够可靠的个性化偏好证据？**
   **没有 Level A。** 本地无 review 文本；只有 Level B 行为重复。不能宣称用户明确喜欢某属性。

2. **成功构建了多少真正可验证任务？**
   164 个 task 中 156 个有 catalog/role/price verified pool；100 个 Level B 的 preference claim 仍需人工审核；56 个当前意图/约束任务可由 catalog 验证；8 个 UNKNOWN 无 pool。

3. **多少任务只能验证搜索，不能验证历史个性化？**
   15 个 explicit intent 任务可验证当前商品和约束；8 个 UNKNOWN 任务不可验证；其余 138 个 D1 有 Level B 历史信号，但个性化收益没有 Level A/人工确认，不能算已完成 personalization Gold。

4. **相同 Query、不同真实历史能否可靠构造？**
   可以构造 **69 pairs / 138 tasks**，用户历史真实；但只有 Level B 行为差异，不能宣称明确的不同偏好。

5. **哪些商品属性适合自动验证？**
   Product Role、title/category、`price_text` 与预算、可明确出现的 category/feature 词。无法可靠自动验证：兼容性、lightweight、saltwater、主观质量/舒适度。

6. **多少 Gold 需要人工审核？**
   100 个 Level B preference claim；8 个 UNKNOWN/clarification；56 个 catalog/constraints 可机器验证。

7. **现有 SASRec/Two-Tower 是否能提供有效历史候选？**
   **LIMITED。** History source 仅 17/164 可达。

8. **BM25+BGE 可验证候选覆盖如何？**
   Top-50：BM25 140/156，BGE 102/156，Hybrid 86/156；Top-10：36/14/13。BM25 覆盖较好但头部有限，BGE/Hybrid 未证明更优。

9. **候选覆盖是否足以支持 P13 Planner 决策实验？**
   **LIMITED。** 可做搜索/检索子集实验；不足以支持历史个性化策略选择实验。

10. **下一阶段继续 P12 还是重新评估数据来源？**
   **LIMITED：继续 P12 的搜索/候选覆盖子集；不应进入纯个性化 P13。** 若要可靠的历史个性化，需要补充 review 文本或独立偏好/搜索意图标签。

## 7. 最终建议

- P11 任务数据：**GO（可执行 Dev artifact）**
- 历史个性化证据：**LIMITED**
- 搜索 / Agent 验证：**LIMITED**
- P13 个性化 Planner 实验：**暂不 GO**
- SFT-v1：**NO-GO**：Level A 为零、Level B 需人工审核、Agent 失败尚未隔离。

## 8. 交付物

- `P11_REPORT.md`
- `P11_SOURCE_DIFF.txt`
- `P11_BENCHMARK_SCHEMA.md`
- `data/p11_benchmark/tasks.jsonl`
- `data/p11_benchmark/schema.json`
- `data/p11_benchmark/manifest.json`
- `data/p11_benchmark/query_pools.json`
- `experiments/p11_build_benchmark.py`
- `experiments/p11_evaluate.py`
- `experiments/p11_agent_exec.py`
- `tests/test_p11_benchmark.py`
- `runs/p11/eval/summary.json`
- `runs/p11/eval/per_task.jsonl`
- `runs/p11/agent/summary.json`
- `runs/p11/agent/results.jsonl`

## 9. 限制

- 本地无 review 文本，Level A 无法建立；
- Level B 不是明确偏好，不能自动转成 SFT 标签；
- candidate pool 是 bounded sample，不是完整 catalogue oracle；
- 164 tasks / 86 users，达到建议的 50–100 用户范围，但不是大规模数据；
- Agent 仅 10 个 Rule 任务，state mode off，未完整验证 M5 multi-turn；
- 未训练模型、未运行 Holdout、未修改 Serving V1/M5/Planner/Role/Eligibility；
- 未 commit/push/deploy；GPU 实例未关闭。
