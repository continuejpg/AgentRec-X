# AgentRec-X 下一会话交接（2026-11-07）

## 0. 一句话状态

P2/P3/P4 的推荐底座、M5、R1、Planner Contract 和 Planner SFT-v0 已完成；P4C 发现
SFT-v0 没有可确认的任务级收益；P5 确认个性化信号真实进入 Agent，但归因仍不完整；
P6 已完成 **Benchmark 协议/时间切分/指标/样本任务**，**尚未运行完整模型评测**。

当前不应声称：
- “已证明 Planner SFT 提升推荐质量”；
- “P6 已完成质量评测”；
- “真实 Qwen 目前可调用”——GPU 实例已关闭。

---

## 1. 工作区、环境、GPU

### 本地工作区

```text
D:\IT\CODE\PROJECTION\agentrec\X
```

### 本地可用 CPU 测试环境

```text
C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env
```

它包含：pydantic 2.10.3、pytest、fastapi、anyio、httpx、langgraph。
项目内 `.venv` / `.venv-ranker` 是 Linux 布局，不适用于当前 Windows 本地执行。

### 运行命令

```powershell
$venv='C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONIOENCODING='utf-8'
& "$venv\Scripts\python.exe" -m pytest -q tests/test_p6_benchmark.py --no-header -p no:cacheprovider
```

### GPU / 远端

- AutoDL 实例目前 **已关闭**，SSH `connect.cqa1.seetacloud.com:29824` 不可用；
- 只要实例是“关机”而非“释放”，`/root/autodl-tmp` 数据应保留；
- 远端模型：`/root/autodl-tmp/models/Qwen3.5-9B-ms`
- 远端 Python：`/root/autodl-tmp/agentrecx-sft/bin/python`
- 远端仓库只有 `training/` 和 `data/`，**没有 recommendation/ 包**；远端只能跑
  model_server/training，不能跑完整本地 Agent 服务。

### 重启 Planner 服务

```bash
# Base Planner，不能加载 Preference Parser adapter
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --port 8008
```

本地隧道：

```powershell
ssh -N -L 18008:127.0.0.1:8008 -p 29824 `
  -i C:\Users\zxt\.ssh\agentrecx_autodl `
  -o IdentitiesOnly=yes root@connect.cqa1.seetacloud.com
```

P4B Adapter 路径（需开机确认仍在）：

```text
/root/autodl-tmp/agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter/
```

**重要风险：** Planner SFT-v0 adapter 当前只在远端，没有本地备份。若实例被释放，
adapter 可能丢失；下次开机先检查该目录，必要时备份。

---

## 2. 项目硬约束（每次都要遵守）

- 默认 Serving V1 `/v1/recommend` 行为不变；
- 不训练 SASRec/Two-Tower，除非明确要求；
- 不训练 Planner SFT / RL，除非明确要求；
- 不启动 GPU，除非用户明确开启且需要；
- 不修改 Gold/Evaluator/冻结任务；
- 不集成本轮未要求的 MCP/Skills/新 Agent 框架；
- 不 commit/push/deploy，除非明确要求；
- `data/processed` 是真实大数据，不删除、不提交；
- 评测结果不得用模型 rationale 替代任务成功；
- 任何 “工具调用成功” 不等于 “任务成功”。

---

## 3. 推荐底座现状

### 默认 Serving V1

- 默认 profile：`serving_v1`
- 组件：SASRec + Two-Tower + catalogue lexical/BM25 + RRF
- 默认入口：`POST /v1/recommend` → `FusionInferenceEngine(serving_v1)`
- 该行为保持不变。

### Agent Shopping 实际链路

```text
ServicePlannerExecutor.execute(RETRIEVE)
  → DemoAgentService.run_shopping_turn
    → self.run
      → _execute_plan
        → LoopController / CandidatePlane
          → history: RecommendationTool.run
            → FusionInferenceEngine.recommend(history, k)
              → sequential: SASRecInferenceEngine
              → two_tower: TwoTowerInferenceEngine
              → catalog_lexical: BM25
              → RRF(K=60)
          → catalog_search: BM25
          → CandidateLedger RRF
        → role / eligibility / final explicit-type promotion
  → response.recommendations
```

关键文件：

- `recommendation/demo/agent_service.py`
- `recommendation/control/candidate_plane.py`
- `recommendation/control/capability.py`
- `recommendation/control/candidate_ledger.py`
- `recommendation/tools/recommendation.py`
- `recommendation/inference/fusion.py`
- `recommendation/evaluation/split.py`
- `recommendation/evaluation/metrics.py`

### P5 已确认的事实

- `history` 源不是纯 SASRec，而是 **FusionInferenceEngine serving_v1**；
- Fusion 内部真实包含 SASRec、Two-Tower、BM25；
- Agent 的 `_two_tower_tool` 是 None，但 Two-Tower 仍通过 Fusion 参与；
- 实际 profile `demo-user-1` 有 5 条 history；
- 历史 5 条传入 `FusionInferenceEngine.recommend`；
- Agent 最终 5 个推荐中 3 个来自 history 源；
- Agent 层再做一次 CandidateLedger RRF。

### P5 未验证

- 每个内部 source 的独立候选数和指标贡献；
- 没有实现 source-level ablation；
- 空 history 被 `AgentRecommendRequest` min_length=1 拒绝，无 fallback 路径；
- 还不能做严格 SASRec/TwoTower 归因；
- P5 最终结论：**PARTIAL**。

---

## 4. P4B Planner SFT-v0（已训练，但不可当作泛化结论）

### 训练

- 数据：`data/planner_sft_v0/`
  - train 267 / val 53
  - actions：RETRIEVE 100 / ASK 50 / RERANK 90 / FINALIZE 80
  - 数据来源：合成请求 + 真实 Amazon catalogue metadata
  - 标签：规则/工具语义，**不是 Qwen 输出**
- Base：Qwen3.5-9B Base，未加载 Preference Parser SFT-v2
- LoRA：r=16, alpha=32, dropout=0.05, all-linear
- 2 epochs，34 steps，516 秒
- train loss 0.2635，eval loss 0.000986
- remote adapter：
  `/root/autodl-tmp/agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter/`
- 本地配置/脚本：
  - `training/planner_sft/config_lora.yaml`
  - `training/planner_sft/train_lora.py`
  - `training/planner_sft/evaluate_val.py`

### 验证结果

- 53/53 Action exact / schema valid / arguments exact；
- **但这是高度同分布的开发集，模型明显记住训练 rationale/template；不能声称泛化。**
- P4C 真实对照：
  - 4 个新场景 × Base/SFT = 8 条轨迹；
  - 所有进入 RETRIEVE 的轨迹都执行了 RERANK；
  - M6.1 全部 `rerank_applied=false`（无支持证据）；
  - 没有可确认的任务级收益；
  - SFT 在 ASK 场景复述训练模板；
  - 结论：无改善、无明确退步、主要是无差异。

### P4C/P4D 相关产物

- `P4C_BASE_VS_SFT_REPORT.md`
- `P4C_BASE_VS_SFT_TRACES.jsonl`
- `P4C_BASE_VS_SFT_SUMMARY.json`
- `P4D_ADAPTIVE_RETRIEVAL_REPORT.md`
- `P4D_ADAPTIVE_RETRIEVAL_TRACES.jsonl`
- `P4D_SFT_V1_CANDIDATE_SAMPLES.jsonl`
- `P4B_PLANNER_SFT_V0_TRAINING_REPORT.md`
- `P4B_VAL_EVAL.json`
- `P4B_VAL_PREDICTIONS.jsonl`

---

## 5. Planner Contract / HTTP / Tool 现状

### 文件

- `recommendation/demo/planner_contract.py`
  - Action：ASK / RETRIEVE / RERANK / FINALIZE
  - Observation：query、candidates、candidate_evidence、m5_state、allowed_actions、
    last_tool_observation、remaining_steps、last_retrieve_query、retrieval_feedback、
    role_target、eligibility
  - Validator、max_steps、run_planner
- `recommendation/demo/planner_service_executor.py`
  - RETRIEVE → `run_shopping_turn`
  - RERANK → `_m5_soft_rerank`
  - FINALIZE → final IDs
  - ASK → terminal awaiting user
- `recommendation/demo/planner_http_proposer.py`
  - `/plan` HTTP client
  - 显式 opt-in：`AGENTRECX_QWEN_PLANNER=on` + `AGENTRECX_QWEN_PLANNER_URL`
  - 解析 envelope，拒绝 Preference Operations，无规则 fallback
- `training/agent_sft/planner_prompt.py`
  - 独立 Planner Prompt
- `training/agent_sft/model_server.py`
  - `/propose`（Preference Parser）
  - `/plan`（Planner Action）
  - **Base Planner 启动时不要传 `--adapter`**

### P4D 真实现象

- Base 模型在 `retry_after_role_feedback` 场景连续重复同一 RETRIEVE query 3 次，
  然后无候选 FINALIZE，被 Validator 拒绝；
- 其他场景 RETRIEVE→RERANK→FINALIZE；
- 严格预算场景模型 ASK，未绕过约束；
- 说明“模型会选 RERANK/ASK”不等于“自适应检索成功”。

### 重要修复记录

- `merge_constraint_pairs` 修复：只有 actual trusted kind 存在时才抑制 authorized；
- `_integrate_state_turn`：Proposal 模式 `current_turn_kinds=()`，Rule 模式只用 trusted_pairs；
- 防御注入修复：原 `existing` 在过滤前计算，导致约束被清空；已改；
- `ServicePlannerExecutor` 的候选反馈字段已区分：
  - `post_role_eligibility_candidate_count`
  - `presented_candidate_count`
  避免把 promotion 前 3 个候选误当最终候选。

P3C 跨层预算回归：

- `tests/test_p3c_budget_cross_layer.py`
- `tests/test_p3c_final_merge.py`
- 4 passed
- 四轮预算：30 → 50 → None → None。

---

## 6. P4D 候选反馈的确定结论

P4D 曾出现：

```text
grounded_candidate_count=3
hard_eligible=3
final_candidate_count=3
candidate_ids=[]
```

根因不是 CandidateLedger 丢失：

1. `_execute_plan` 的 `build_recommendations_view` 先生成 3 个候选；
2. `candidate_filter_diagnostics` 记录这 3 个为 `final_candidate_count`；
3. `run_shopping_turn` 随后执行
   `promote_explicit_type_matches`（`agent_service.py:1529`），
   replacement cartridge 的 accessory gate 将 3 个不匹配候选过滤为 0；
4. `response.recommendations=[]`，所以 Planner 的 `candidate_ids=[]` 是真实的；
5. `final_candidate_count=3` 是 promotion 前诊断值，不是最终展示数。

已做最小修复：只把 Observation 字段命名为
`post_role_eligibility_candidate_count` / `presented_candidate_count`，
**没有修改推荐/角色/Eligibility 逻辑**。

---

## 7. P6 当前交付（最重要）

### 已存在

```text
P6_BENCHMARK_V0_REPORT.md
P6_BENCHMARK_V0_SOURCE_DIFF.txt
data/p6_benchmark_v0/
  classical_dev.jsonl       160
  classical_holdout.jsonl    40
  dynamic_dev.jsonl           6
  dynamic_holdout.jsonl       4
  protocol.json
  leakage_report.json
experiments/p6_benchmark/
  build_benchmark.py
  metrics.py
  __init__.py
tests/test_p6_benchmark.py  4 passed
```

### 协议

classical：

```text
history = items[:-2]
validation target = items[-2]
test target = items[-1]
user-disjoint dev/holdout
full catalogue ranking, seen mask, target re-admitted
```

样本检查：

```text
users total = 412,445
items = 156,746
dev = 160, holdout = 40
user_overlap = 0
dev test targets = 160
holdout test targets = 40
test target overlap = 2
history-target overlap cases = 3
```

指标：

```text
Recall@10, Recall@50, NDCG@10, MRR@10, CatalogueCoverage@10
```

dynamic：

- dev/holdout template_id 不重叠；
- 6 dev / 4 holdout 合成购物任务；
- 真实 catalogue ASIN seed；
- 指标：VerifiedTaskSuccess、HardConstraintViolationRate、ProductRoleCorrectness、
  PreferenceUpdateConsistency、EligibleCandidateCoverage、ToolCallsPerTask、UnknownRate；
- 尚未运行完整 dynamic Agent 评测。

### P6 未完成

- 没有在完整 160/40 上跑 SASRec/TwoTower/Fusion 的真实 full-ranking；
- 没有 full-catalogue scorer 适配器；Fusion 当前只返回 fused top-k；
- 没有跑四组公平 arm；
- 没有最终质量结论；
- 没有 item-disjoint holdout（当前 item target overlap=2，已如实记录）；
- 没有 bootstrap 置信区间。

### P6 关键判断

**PROTOCOL READY，FULL EVAL PENDING。**

---

## 8. 测试状态

当前应保留的测试：

```text
tests/test_planner_contract.py          11 passed
tests/test_planner_sft_v0.py             3 passed
tests/test_p6_benchmark.py               4 passed
tests/test_p3c_final_merge.py            3 passed
tests/test_p3c_budget_cross_layer.py     1 passed
```

不要因为清理而删除这些关键测试。临时 Stub/debug 脚本在系统 `%TEMP%`，不是仓库核心；
不要让根目录继续堆叠重复报告。

---

## 9. 下一步建议（按优先级）

### A. 先做 P6 的实际 classical eval

需要：

1. 为 SASRec/TwoTower/Fusion 提供 `scorer(history, target) -> full catalogue scores`
   的只读适配器；
2. 对 160 dev / 40 holdout 生成 ranked lists；
3. 用 `experiments/p6_benchmark/metrics.py` 计算 Recall/NDCG/MRR/Coverage；
4. 先 dev，再一次性 holdout；不得用 holdout 调参；
5. 记录实际命令、模型 checkpoint、耗时。

### B. 再做 dynamic task eval

1. Traditional Serving V1；
2. Rule Agent；
3. Qwen Base Planner（需要 GPU）；
4. Planner SFT-v0（需要 GPU）。
固定 M5 初始状态、catalogue、sources、k、max_steps、eligibility。

### C. 如果继续研究 Planner

优先解决 P4C/P4D 暴露的两个问题：

1. **何时不应 RERANK**：无 M6.1 支持证据时标签应是 FINALIZE；
2. **何时应改 query**：当 role/eligibility 不合格或 `candidate_ids` 为空时，
   训练可区分的 RETRIEVE 改写/终止行为。

不要直接进入 P7 或 RL；用户未要求。

---

## 10. 新会话开场建议

可以直接说：

```text
工作区 D:\IT\CODE\PROJECTION\agentrec\X。
先阅读 NEXT_SESSION_HANDOFF.md 和 P6_BENCHMARK_V0_REPORT.md。
GPU 当前关闭，不要启动。
本轮只做 P6 的实际 classical 评测：先实现只读 full-ranking scorer 适配器，
在 data/p6_benchmark_v0 的 dev 上运行 SASRec/TwoTower/Fusion，再一次性跑 holdout。
不要修改默认 Serving V1/Gold/Evaluator，不训练，不 commit。
```

如果新任务与 P6 无关，也应先读本交接和 `P6_BENCHMARK_V0_REPORT.md`，避免重复已完成的
调试。
