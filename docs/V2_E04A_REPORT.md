# AgentRec-X V2 / E-04A — Real Provider Smoke Test（配置准备阶段报告）

> 本轮按用户指令执行**方案 3**：先给出真实 Provider 的配置与执行说明，**本轮不发送任何在线请求**；
> 检查点按**方案 2**：只做只读审计，**不创建分支、不 `git add`、不 `git commit`**。
> 结论：**PASS（真实联调已执行，5 项判据全部由真实测量通过；累计 4 / 4 次授权调用用尽）**
> —— 见 §18 首轮真实联调结果与 §19 的 E-04A-R1 Replanning 定向验证。
>
> **本轮确认的四条边界（不得被后续报告弱化）：**
> 1. **真实 Provider 联调已通过**：4 次真实 DeepSeek 调用全部成功（tool_selection、product_explanation、
>    replan_source_choice，以及首次因**实验脚本**缺陷中断的那一次），零重试、零探测、零越权；
> 2. **E-04A-R1 的初始来源计划是确定性的**（脚本构造 `SourceSelection(["history"])` 交给真实
>    `run_selection`），真实 DeepSeek **只负责"追加哪一个来源"这一次决策** ✗ 它不是"模型自主生成的
>    完整工具计划"；
> 3. **本轮没有证明任何线上推荐质量提升，也没有证明生产级稳定性** ✗ —— 只证明了接口层与行为层的
>    基础可用性（§7 的三层区分保持不变）；真实推荐资产与质量指标属于 **E-04B**；
> 4. **不再进行任何在线 Provider 调用**：4/4 授权额度已用尽，后续任务（含 E-04B）**不调用**
>    DeepSeek 在线 API ✓。

---

## 0. 一句话结论

```text
授权状态：未获得真实 Provider 调用授权（用户选择「先配置、后授权」）          ⇒ 未调用任何在线模型
配置状态：AGENTRECX_LLM_BASE_URL / MODEL / API_KEY 全部未设置；
          provider_configured() == False；无 key、无 base_url、无 profile         ⇒ 无法调用
已交付：  ① 配置与执行指南（含协议、/v1 规则、安全注入、脚本行为审计、预算与
             费用上限的诚实边界、各能力所需调用次数、脱敏自检）
          ② 只读检查点审计（92 个文件 A–E 分类 + 精确 git add 白名单 + 风险 + 备份方案）
          ③ 受限 smoke 脚本（默认 dry run、硬调用上限、失败不重试、不打印任何密钥）
          ④ 离线回归 1070 项 0 失败（真实调用路径本身未被修改）
```

## 1. HEAD、初始工作树

```text
$ git rev-parse HEAD                    → faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall              → 32 modified + 58 untracked（本轮开始时）
$ git diff --stat                       → 32 files changed, 3384 insertions(+), 226 deletions(-)
```

未执行 reset / checkout / restore / clean / stash / commit / push，未删除任何未跟踪文件。

## 2. 待保存代码、测试、文档与不应提交文件的检查点清单

完整清单见 [`docs/V2_E04A_CHECKPOINT_PLAN.md`](V2_E04A_CHECKPOINT_PLAN.md)（本轮新增，只读审计产物）：

| 类别 | 数量 | 处理建议 |
|---|---|---|
| **A. 生产代码** | 35 | 纳入检查点（含 `recommendation/backends/tiger_engine.py`，标注为 TIGER 实验路径） |
| **B. 回归测试** | 32 | 纳入检查点 |
| **C. 技术文档** | 18 | 纳入检查点（13 份阶段报告 + USAGE + 本轮的 Provider 指南/清单/报告） |
| **D. 不应提交** | **0** | 未发现应排除的候选文件（数据库、checkpoint、缓存、临时文件均未出现在候选清单 ✓） |
| **E. 需人工确认** | 7 | TIGER/外部 backend 实验文件（`experiments/t03lite_*.py`、`backends/tiger_public/.../serve_worker.py`、`docs/TIGER_SERVING.md`、`recommendation/backends/tiger_engine.py` 等） |

文档给出了**逐文件的 `git add` 命令**（每条 ≤5 个路径，**不使用 `git add -A`**），A+B+C+E = 92 = 全部候选 ✓。

## 3. 是否发现潜在凭据或大型工件入库风险

| 检查 | 结果 |
|---|---|
| 凭据字面量扫描（92 个候选文件） | **0 命中** ✓ |
| `.env*` / `*.key` / `*.pem` / `.m4_credential` | 候选清单中无；`.gitignore` 已覆盖 ✓ |
| 数据库（`*.db`/`*.sqlite*`）、模型工件（`*.pt`/`*.safetensors`/`data/`/`artifacts/`） | 候选清单中无匹配；`.gitignore` 覆盖 ✓ |
| 大型二进制 | 候选最大 48 KB（测试文件）；无大工件 ✓ |
| 未跟踪且**未**被忽略的敏感文件 | 未发现 ✓ |

## 4. 真实 Provider 配置状态（不打印任何密钥）

```text
AGENTRECX_LLM_BASE_URL         absent
AGENTRECX_LLM_MODEL            absent
AGENTRECX_LLM_API_KEY          absent      （credential_present=False）
AGENTRECX_LLM_PROFILE/TIMEOUT/JSON_MODE/THINKING/INPUT_PRICE/OUTPUT_PRICE   absent（均取默认）
AGENTRECX_AGENT_REPLANNING / EXPLANATIONS / MEMORY_RANKING / ...            absent（均取默认/关闭）
provider_configured()          False
可用的 Provider profile        deepseek（默认匹配）、openai_compatible
```

协议：OpenAI 兼容 `POST {BASE_URL}/chat/completions`（`provider_adapter.py`）；`BASE_URL` **需含 `/v1`**
（适配器只追加 `/chat/completions`）。详见 [Provider 指南](V2_E04A_PROVIDER_GUIDE.md) §1–§3。

## 5. 是否具备用户调用授权

**否。** 用户明确选择「先完成配置准备与安全核验，本轮不发送在线请求，待确认配置与预算并再次授权后
才执行」。因此本报告不含任何真实模型输出 ✓，也不把 fake 结果冒充真实联调 ✓（§7 已按此区分）。

## 6. 实际调用的 Provider、模型及次数

**零次调用** ✓。未构造网络客户端、未发送 HTTP 请求；`experiments/e04a_provider_smoke.py` 仅执行过
dry run（dry run 会在 `--run` 缺失时直接退出 ✓）。

## 7. Tool Selection 真实响应与执行结果

**未执行**（无授权/无配置）。已交付可运行的受限脚本与判定口径：

* 判定分三层（脚本执行后据此报告）：**接口层**（结构化计划是否通过 Schema 校验）、
  **行为层**（来源选择是否合理、是否触达合法工具）、**推荐效果**（本轮不作结论 ✓）；
* 记录字段：`planned_sources`（是否属于部署允许来源）、`action_counts`、`terminal`
  （status/termination_reason/steps/tool_calls/limits）、最终合法商品、`ranked_with_preferences`；
* 重要约定：真实模型选出的来源若与 fake 测试不同，**先判是否违反接口或安全契约**，
  不因“与 fake 不同”而判失败 ✓。

## 8. Memory 跨轮真实上下文

**未执行**。脚本设计为同一 Demo Session 内两轮：先 `I prefer black.`（M9 真实写入，读取
`memory_update.changed` 与 `active_preferences`），再发推荐请求并核对
`audit.ranked_with_preferences`（**pre-write** 快照）与 `active_preferences`（**post-write** 面板）
的差异，以及「当前请求」与「历史偏好」在模型上下文里始终是分离字段 ✓（M9 写入与投影本身不调用
模型 ✓，因此这部分离线已由 `test_demo_llm_agent_memory_acceptance.py` 等 26 项测试覆盖 ✓）。

## 9. Replanning 实际行为

**未执行**。脚本在 `--replanning` 下会检查：`audit.replan.status` / `reason_code` /
`chosen_source`、`terminal.steps/tool_calls` 与 `limits`（**同一 Run、原总预算**）、
以及 `replan_execution`（若有）的初始/最终观察。门控契约（合法来源、预算、至多一次）由
E-03A/E-03B 的 39 项离线测试锁定 ✓；若真实模型选择**不追加**，脚本只记录决策合法性与「本轮是否
覆盖了实际追加执行路径」，不强迫模型输出指定答案 ✓。

## 10. Product RAG 结构化 Claim 验证结果

**未执行**。脚本在 `--explanations` 下会记录每张卡的 `reason_status` 与
`knowledge_evidence[].field_path`、以及 `audit.rag`（enabled/model_called/accepted/rejected/
abstained/failure）。验证口径不变：`evidence_id` 必须存在、`field_path`/`value` 必须与目录单元完全
一致，**验证器不会为了让真实模型输出通过而放宽** ✓；无证据时输出 `insufficient_evidence` 而非编造理由 ✓。

## 11. 真实模型调用耗时、用量与异常

**无真实调用**，因此无耗时/用量数据 ✓。脚本已具备遥测能力：每次调用记录用途（tool_selection /
replan_source_choice / product_explanation）、成功与否、耗时、响应字符数、是否合法 JSON、
provider 返回的 token 数与（在设置了单价时）费用估算；**不含 prompt 原文、不含响应原文、不含密钥** ✓。

## 12. 与 Fake 测试相比发现的实际差异

**无法比较**（无真实调用）✓。本轮确认的、与 fake 场景**结构性相关**的差异点已在脚本与指南中明确：
① 真实模型可能选择与脚本化 fake 不同的来源（判定口径见 §7）；
② planner 默认 `max_attempts=2` 可能产生**一次额外计费调用**，脚本已改为 `max_attempts=1` 消除 ✓；
③ fake 客户端不返回 usage/费用，真实 provider 的 token 与费用只有真实调用才能记录 ✓；
④ fake 无法暴露真实 provider 的限流/超时/内容过滤行为 ✓（离线异常路径另有 63 项 replay 测试覆盖 ✓）。

## 13. 生产代码修改文件与 file:line

**本轮未修改任何生产代码** ✓（未改动推荐算法、控制器、Memory、排序、解释、旧模式）。

新增（非生产路径，均未执行在线调用）：

| 文件 | 说明 |
|---|---|
| `experiments/e04a_provider_smoke.py`（**新增**） | 受限真实 Provider smoke 脚本：`_CappedClient:71`（硬调用上限，超限抛错不发请求）、`use_llm_client(capped, max_attempts=1)`（`main` 内，消除重试）、`_purpose_of:113`（按 payload 形状判定调用用途）、`_estimate_cost:126`（仅在设置了单价时给费用估算）、`_provider_report:158`（脱敏配置自检）、`main:236`（默认 dry run，`--run` 才联网；无配置时拒绝运行） |
| `docs/V2_E04A_PROVIDER_GUIDE.md`（**新增**） | 配置与执行指南（协议、`/v1` 规则、安全注入、脚本行为审计、预算与费用边界、调用次数、脱敏自检） |
| `docs/V2_E04A_CHECKPOINT_PLAN.md`（**新增**） | 只读检查点审计（92 文件 A–E 分类、逐文件 `git add` 白名单、风险与备份方案） |

## 14. 回归测试数量和结果

```bash
.venv/bin/python -m pytest tests/test_v2_schema_integrity.py tests/test_v2_product_feedback.py \
  tests/test_v2_bounded_replanning.py tests/test_v2_observation_replan_gate.py \
  tests/test_v2_narration_consistency.py tests/test_v2_phase4_joint_acceptance.py \
  tests/test_multi_source_fusion_presentation.py tests/test_demo_memory_ranking.py \
  tests/test_product_knowledge_rag.py tests/test_demo_llm_agent_memory_acceptance.py \
  tests/test_agent_state_isolation.py tests/test_agent_control_plane.py \
  tests/test_llm_selected_controller.py tests/test_demo_llm_agent_mode.py \
  tests/test_provider_record_replay.py tests/test_demo_llm_agent_memory.py \
  tests/test_agent_chat_adapter.py tests/test_llm_tool_selection.py tests/test_demo_agent_api.py \
  tests/test_demo_api.py tests/test_demo_multiturn.py tests/test_demo_llm_mode.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_demo_control_plane.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_sessions.py tests/test_demo_web.py tests/test_demo_fusion_view.py \
  tests/test_llm_agent_availability_guard.py tests/test_control_stage2_candidates.py \
  tests/test_constraint_eligibility.py tests/test_candidate_reference_boundary.py \
  tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
# → 1070 passed in 90.61s（0 failed）
```

覆盖：Chat `llm_agent`、M9 Memory、Tool Selection、Bounded Replanning、Memory Ranking、
多来源融合、Product RAG、商品反馈、Schema 完整性、Provider Record/Replay、Agent Control Plane、
Legacy Chat（deterministic / llm）、Fast Serving `/v1/recommend` ✓。**未修改任何推荐逻辑** ✓。

## 15. 尚未验证的真实模型资产与质量问题

1. **真实 Provider 可用性**：未验证网络连通、账号额度、限流与内容过滤策略 ✓（属 E-04B/E-04C 范围）。
2. **真实推荐资产**：未加载 349 MB SASRec checkpoint / 300 MB 元数据 / 6.21 GB item-item / TIGER
   后端 ✓（E-04B）。
3. **推荐效果**：**未做任何质量结论** —— 本轮（乃至授权后的 smoke）只能证明接口层与行为层可用性，
   不能宣称推荐效果提升 ✓（§7 已明确三层区分）。
4. **费用硬上限**：脚本只有**调用次数**上限，**没有** token/费用硬上限 ✓（诚实边界见指南 §5；
   如需硬约束，应在 Provider 侧设置密钥/账户级配额）。
5. **多轮记忆的长期行为、并发真实 provider 行为**未验证 ✓。

## 16. 最终 HEAD、git status

```text
$ git rev-parse HEAD                    → faeff0b6f790d8c9d077d7d07ee26998aeecad2a（未提交、未 reset）
$ git status --short -uall              → 32 modified + 62 untracked
```

本轮新增未跟踪文件：`experiments/e04a_provider_smoke.py`、`docs/V2_E04A_PROVIDER_GUIDE.md`、
`docs/V2_E04A_CHECKPOINT_PLAN.md`、`docs/V2_E04A_REPORT.md`（58 → 62 ✓）。
**未执行任何 Git 写操作** ✓。

## 17. 最终结论

**PARTIAL（首轮结论，已在 E-04A-R1 后升级为 PASS —— 保留原文以保持记录连续）。**

```text
已完成（可复核）：
  ① 真实 Provider 接线审计（协议、profile、路径、超时/重试、日志风险、record/replay 覆盖）
  ② 配置与执行指南（含 /v1 规则、安全注入、脱敏自检、脚本行为审计、预算与费用边界、逐能力调用次数）
  ③ 受限 smoke 脚本（dry run 默认、硬调用上限、失败不重试、零隐式调用、无密钥输出）
  ④ 检查点只读审计（92 文件 A–E 分类、逐文件 git add 白名单、E 类 7 文件待确认、备份方案）
  ⑤ 离线回归 1070 项 0 失败；生产代码零改动

缺失条件（阻塞真实联调的唯一原因）：
  A. 用户对真实付费/外部 Provider 调用的明确授权（当前：未授权）
  B. Provider 配置：AGENTRECX_LLM_BASE_URL（含 /v1）、AGENTRECX_LLM_MODEL、AGENTRECX_LLM_API_KEY
     （当前：三者均未设置；provider_configured() == False）
  C. 调用预算确认：--max-calls（建议 4 或 6）与可接受的费用量级（脚本仅能计数，不能限制费用）
```

满足 A/B/C 后，按指南 §8 的两条命令即可完成真实联调（Tool Selection → Memory → 可选
Replanning → 可选 Product RAG，全程 ≤6 次调用），随后即可把本报告升级为 E-04A 的完整 PASS 证据。

本轮到此停止：**不自行进入 E-04B、不执行任何 Git 写操作、不调用在线 Provider** ✓。


---

# 18. 真实联调结果（2026-10-15 授权执行，累计 3 / 4 次真实调用）

## 18.1 环境与执行方式

```text
环境：WSL2（Linux 6.18.33.2-microsoft-standard-WSL2）
目录：/mnt/d/IT/CODE/PROJECTION/agentrec/X     解释器：<repo>/.venv/bin/python
配置：复用 scripts/start_demo.sh 的 load_dotenv_into_environment（加载 .env：8 set / 0 preserved）
      → provider_configured() == True，profile=deepseek，model=deepseek-flash，
        请求 URL = https://api.deepseek.com/v1/chat/completions，timeout 30s，json_mode=true
凭据：仅以进程环境注入；报告与日志中从未出现 Key（脚本只输出 credential_present/length）
限额：--max-calls 3（本次），脚本硬上限；max_attempts=1（无重试）；无探测请求
```

## 18.2 每次真实调用（用途 / 成功 / 耗时 / Token / 结构化校验）

| # | 用途 | 结果 | 耗时 | 响应 | 合法 JSON | Token（in/out/total） | provider request id |
|---|---|---|---|---|---|---|---|
| 1 | `tool_selection` | ✓ 成功 | 1.048 s | 178 字符 | ✓ | 592 / 44 / 636 | `9aff01af-2892-4b95-8d8f-f1e30fd20c12` |
| 2 | `product_explanation` | ✓ 成功 | 1.267 s | 663 字符 | ✓ | 1227 / 171 / 1398 | `5e6f0ae5-8fca-49f2-b495-30865c3e0619` |

* 两次调用均通过现有 `ModelClient`/适配器接口（`provider_adapter.py`）✓，无协议/鉴权异常 ✓。
* 无第 3 次调用：终端 Gate 判定候选充分，**Replanning 未被询问** ✓（预算未被消耗 ✓）。
* 费用：未设置单价 ⇒ `estimated_cost=null` ✓（**不作费用结论**；token 数为 provider 自报 ✓）。
* **另需如实披露**：首次运行（`--max-calls 4`）在**第 1 次调用返回之后**因我脚本的
  `response.usage` 误当字典使用而抛错（`AttributeError`），该次调用已计入授权额度 ✗；
  这是**本次实验脚本的缺陷**（非 provider/鉴权/模型/协议问题），已修复（`usage()` 为方法；
  另新增 `--preseed-preference` 以便在剩余预算内覆盖四项能力）。总计消耗 **3 / 4** 次 ✓。

## 18.3 A. Tool Selection（真实响应与执行结果）

```text
planned_sources  = ["catalog_search", "history"]     ← 真实模型选择，二者均为部署允许来源 ✓
合法性与执行      : ActionValidator + CandidatePlane 通过；action_counts =
                    {search_catalog:1, recommend_from_history:1, finish:1}
terminal         : status=finished, termination_reason=completed, steps=3/6, tool_calls=2/4, retries=0
最终合法商品      : [k-black, k-hist, k-blue, k-red]（全部 grounded，k=4）
接口层           : ✓（结构化计划通过现有 Schema 校验并实际执行）
行为层           : ✓（在允许来源集合内选择；未触达任何未授权工具；预算内完成）
```

## 18.4 B. Memory 跨轮真实上下文

```text
M9 写入（确定性、不调用模型）：user_key 为本 session；active = [color/prefer/black] ✓
真实 planner 上下文          ：ranked_with_preferences = [{kind: color, value: black}] ✓
                                （受限价值级投影，未含 user_key/source_text/其他 session ✓）
最终排序效果                  ：k-black 由第 2 位升到第 1 位 ✓（M10A 命中 + M10B 生效）
```

## 18.5 C. Replanning 实际行为（**部分覆盖**）

```text
audit.replan = { status: "not_triggered", reason_code: "candidates_sufficient",
                 chosen_source: null }
⇒ 同一 Controller Run 内、k=4 已满足 ⇒ 确定性 Gate 关闭，模型**未被询问**，
  预算与来源安全契约未被触碰 ✓；但"真实模型主动选择一个合法追加来源并被执行"的路径
  在本次预算内**未覆盖** ✗（这正是 §5.C 要求分开记录的两件事）
```

## 18.6 D. Product RAG 结构化 Claim 验证结果

```text
audit.rag = { enabled: true, model_called: true, accepted: 4, rejected: 0, abstained: 0, failure: null }
每张卡：reason_status = "grounded"
  k-black → 引用 ["title", "details.Color"] ✓  ← 真实模型引用了**目录字段**（且与偏好同字段）
  k-hist / k-blue / k-red → 引用 ["title"] ✓
⇒ 真实模型输出的结构化 claim 全部通过**现有确定性验证器**（未放宽任何验证规则 ✓）；
  无证据时的降级路径由离线测试覆盖（rejected/insufficient_evidence ✓）
```

## 18.7 与 Fake 测试的差异（实测）

1. 真实模型选择 `["catalog_search","history"]`（fake 常返回单一来源）⇒ 判定口径按"是否违反契约"，
   实测**合法** ✓；
2. 真实 provider 提供了 token 与 request id（fake 无）✓；
3. 真实模型的 claim 覆盖 `details.Color` 这类目录字段，验证器照常逐字段校验 ✓；
4. 真实延迟 ~1.0–1.3 s/次（离线 fake 为 0）✓。

## 18.8 结论与剩余缺口

```text
PASS 判据逐项：
  ① 真实模型经现有 ModelClient 接口工作                       ✓ 实测
  ② Tool Selection 结构化结果经验证并实际执行                  ✓ 实测
  ③ 多轮 Memory 受限投影正确传递                              ✓ 实测
  ④ Replanning 模型决策符合来源与预算安全契约                  ◐ 部分（Gate 关闭路径实测 ✓；
                                                             模型选择追加来源路径未覆盖 ✗）
  ⑤ 商品 RAG 真实 Claim 可被确定性验证、非法输出可安全拒绝      ✓ 实测（接受 4/4；拒绝路径离线覆盖）
  ⑥ 控制器/Memory/排序/解释/旧模式不回归                       ✓ 1070 项离线回归 0 失败

最终（首轮）：**PARTIAL** —— 唯一剩余缺口是 ④ 的"追加来源"路径。
**后续（E-04A-R1，见 §19）：该缺口已由 1 次真实的 `replan_source_choice` 调用补齐（案例 A），
本报告结论因此升级为 PASS ✓。**
补齐条件（二选一，均需新的授权）：
  (a) 再授权 2 次真实调用：一次用于选出"首次候选不足"的单来源计划，一次用于模型的追加来源决策；
  (b) 接受"仅 Gate 关闭路径"的证据，并把该项标注为规则级覆盖（离线 39 项 E-03A/E-03B 测试已覆盖）。
本轮未修改 `.env`、未改业务代码、未改测试、未触碰 Git 状态 ✓（唯一改动是实验脚本
`experiments/e04a_provider_smoke.py` 的自身缺陷修复与一个可选参数）。


---

# 19. E-04A-R1：剩余一次授权的 Replanning 定向验证（真实 DeepSeek）

## 19.1 能否用一条真实调用完成该场景

**能。** 关键设计：初始计划**不由模型生成**，而是由脚本构造 `SourceSelection(route="recommend",
selected_sources=("history",))` 直接交给**真实** `DemoAgentService.run_selection` ✓ —— 因此
Tool Selection 不产生调用 ✓；Product Explanation 关闭 ✓ —— 于是模型路径**只剩** Replan 决策 ✓。
离线自检（`--replan-only --self-check`，使用计数用离线替身、**不联网**）已证明：
`model_calls = 1`、`capped_calls = 1`，且该次调用的 `purpose = "replan_source_choice"` ✓✓。

## 19.2 Smoke 脚本的修改（仅实验脚本，未改业务代码）

| 修改 | 内容 |
|---|---|
| `--replan-only` | 新场景模式：确定性计划 `[history]` → 真实 `run_selection` → 单次 Replan 决策 |
| `--self-check` | 该模式下用计数用离线替身作答，**不发送任何请求**，用于在花钱前证明调用次数与执行轨迹 |
| `_history_limited_engine()` | 修正夹具：顺序来源只返回 1 件（`k-hist`），身份映射仍覆盖全目录（4 件）——与 E-03B 测试替身同构，使"首次候选不足"成为**真实 observation** 而非伪造 |
| `_action_counts()` | 从 run 自身的步骤记录统计动作（原先误用聊天契约的字段） |
| `usage()` 修复 | `ModelResponse.usage` 是方法；现读取 `input_tokens/output_tokens/total_tokens/latency_ms/estimated_cost/request_id` |

离线自检中修掉的 3 个缺陷（均在**未消耗任何真实调用**时暴露 ✓）：`action_counts` 字段来源错误、
`AgentRecommendResponse` 无 `audit` 字段、夹具让 history 返回全目录导致 Gate 关闭。

## 19.3 首次真实工具 Observation（来自运行自身的 `replan_execution.initial`）

```json
{"candidates": 1, "sources_consulted": ["history"],
 "source_outcomes": {"history": "candidates"},
 "remaining_tool_calls": 3, "remaining_steps": 5}
```

## 19.4 实际传给 DeepSeek 的受限字段清单（`build_replan_request_payload`，无身份/无原文）

```text
user_request, stated_preferences（价值级 ACTIVE 投影，本次为空）,
observation{requested_k=4, verified_candidates=1, consulted_sources=["history"],
            source_outcomes={"history":"candidates"}, remaining_tool_calls=3,
            remaining_steps=5, step_index=1},
eligible_sources[{name:"catalog_search", purpose:"lexical full-catalogue search ..."}],
instructions（至多一个来源、不得改预算/工具/候选/查询）
```
✓ 未包含 `user_key`、`source_text`、商品标题/分数、其他 session 的 memory、store 句柄。

## 19.5 实际 DeepSeek 结构化决策（真实调用 1 次）

```text
purpose=replan_source_choice  ok=true  1.066 s  245 字符  合法 JSON ✓
tokens: input 458 / output 53 / total 511   provider request id: 1d016628-07eb-494e-8d7f-7a39e77fd726
→ 决策: {"replan": true, "source": "catalog_search", ...}   （经 parse_replan_answer 校验：名字在 eligible 内 ✓）
→ 记录: status="appended_candidates", reason_code="appended_source_proposed",
        model_called=true, chosen_source="catalog_search", decision_step=1
⇒ **案例 A：模型返回合法的追加规划，且工具实际执行** ✓
```

## 19.6 ActionValidator 与 Controller 执行轨迹（真实、单次 Run）

```text
step0 recommend_from_history  obs=recommendation cand=1  validated ✓   ← 真实 capability
      └ 首次 Observation（§19.3）→ 确定性 Gate 允许（1 < k=4，未试合法来源 catalog_search，
        预算 3/5 ≥ 2 步，无拒绝，未用过唯一一次机会）
      └ DeepSeek 决策 → catalog_search（唯一 eligible）
step1 search_catalog          obs=candidate_set cand=4 validated ✓   ← 追加来源真实执行（同一 Run）
step2 finish                  validated ✓
action_counts = {recommend_from_history: 1, search_catalog: 1, finish: 1}
terminal = finished / completed，steps 3/6，tool_calls 2/4，retries 0
```

## 19.7 预算消耗

| 项 | 值 |
|---|---|
| 本次新增 Provider 调用 | **1**（`capped_calls = 1`；cap=1，若有第二次会**抛错而不发送** ✓） |
| 单次 Run 的工具预算 | `tool_calls 2 / 4` ✓（首次 1 + 追加 1，**同一份预算** ✓，无外层重启 ✓） |
| 单次 Run 的步骤预算 | `steps 3 / 6` ✓ |
| Replan 次数 | **1**（每请求至多一次 ✓，`decision_step=1` ✓） |
| 追加来源数 | **1**（`catalog_search`，唯一合法候选 ✓） |

## 19.8 初次与最终候选商品 ID

```text
初次（history）：["k-hist"]                       ← 1 件，k=4 不足 ⇒ Gate 打开
最终（累计）：   ["k-hist", "k-black", "k-blue", "k-red"]   ← 4 件，k=4 生效
⇒ 首轮候选未丢失、追加来源候选并入同一融合池、无新增未验证商品、未超 k ✓
```

## 19.9 实际新增 Provider 调用次数 / 19.10 E-04A 累计

| 阶段 | 调用 | 说明 |
|---|---|---|
| E-04A 首轮（含脚本缺陷中断的 1 次） | 3 | tool selection ✓ + product explanation ✓ + 1 次因 `usage` 误用而中断（调用已发生 ✓） |
| **E-04A-R1（本轮）** | **1** | replan_source_choice ✓ |
| **累计** | **4 / 4** | **未超出授权上限** ✓；后续未经新授权不再发起任何调用 ✓ |

## 19.11 结论

**PASS。** §11 的五项判据全部由真实测量支持：

```text
① 真实模型经现有 ModelClient 接口工作                 ✓（4 次调用全部成功，零重试）
② Tool Selection 结构化结果经验证并实际执行            ✓（E-04A §18）
③ 多轮 Memory 受限投影正确传递                        ✓（E-04A §18）
④ Replanning 模型决策符合来源与预算安全契约，且
   追加来源被真实执行                                 ✓（本 §19，案例 A）
⑤ 商品 RAG 真实 Claim 可被确定性验证、非法输出可安全拒绝 ✓（E-04A §18，接受 4/4；拒绝路径离线覆盖）
⑥ 控制器/Memory/排序/解释/旧模式不回归                 ✓（离线 1070 项 0 失败；本轮仅改实验脚本）
```

本轮未修改 `.env`、业务代码、测试与 Git 状态 ✓；未加载大型资产 ✓；未做推荐质量结论 ✓。
