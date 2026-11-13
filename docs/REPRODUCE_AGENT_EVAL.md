# 复现真实 Qwen Agent 评测（P20）

本文对应：

- `experiments/p20_agent_eval/` — 30 条自然语言任务、Baseline/Corrective/Complete Shadow；
- `experiments/p20_metadata/run_verifier_pilot.py` — 49 条 Claim–Evidence GPU Pilot；
- `recommendation/control/evidence_verifier.py` — Shadow Verifier。

评测结果与限制见 `docs/AGENT_QUALITY_BASELINE.md`、`docs/EVIDENCE_VERIFIER.md`。

## 1. 依赖

Python 基线：**3.10.12**（3.10.x 可复现）。

```bash
.venv/bin/python -m pip install -r requirements-eval.txt
.venv/bin/python -m pip check
```

`requirements-eval.txt` 组合了：

- `requirements.txt`：现有 Agent/服务依赖（NumPy、Pydantic、LangGraph 等）；
- `requirements-mcp.txt`：`mcp==2.3.0`，并显式固定 `anyio==4.15.1`；
- `requirements-dev.txt`：pytest/httpx 测试依赖；
- `requirements-cpu.txt`：验收过的 `torch==2.1.2+cpu`。

注意：

- 这是 **CPU client 环境** 的依赖组合。
- 不要把它安装到 AutoDL 的 Qwen CUDA 服务环境；那里已有的 CUDA/PyTorch 栈不能替换。
- Qwen 推理服务是外部 GPU 服务，不是 pip 依赖。它的模型和启动环境不属于本仓库 Python 环境。
- 评测脚本不会下载 Qwen 模型，也不会自行启动 GPU 服务。

当前实测组合：

```text
Python 3.10.12
mcp 2.3.0
anyio 4.15.1
numpy 1.26.4
pydantic 2.13.5
fastapi 0.141.1
pytest 9.1.1
torch 2.1.2+cpu
```

## 2. 外部数据资产

以下路径在仓库中按现有规则被 `.gitignore` 忽略，不会随 GitHub 仓库分发：

```text
data/p13r/electronics/catalog_meta.jsonl
data/enrichment/electronics_metadata_sidecar.jsonl
data/p14e2/benchmark/dev.agent.jsonl
data/p14e2/benchmark/dev.evaluator.jsonl
data/p14b3_eval100/eval100.agent.jsonl
data/p14b3_eval100/eval100.evaluator.jsonl
```

复现前必须按项目已有的数据/资产流程准备这些文件。任务和评测器不会从网络自行下载 Amazon 原始数据或 metadata sidecar。

数据用途：

- `catalog_meta.jsonl`：76,263 个真实电子产品 catalogue；
- `electronics_metadata_sidecar.jsonl`：Metadata Enrichment 输入；
- `dev.*`：20 条 P14-E2 Dev 任务；
- `eval100.*`：100 条 P14-B3 Eval-100 任务，其中 10 条用于本轮自然任务评测。

任务集只使用已有行，不使用 P18 Reserved 任务，也不修改 P18 Frozen Manifest。

## 3. Qwen GPU 服务

评测使用现有 **Qwen3.5-9B Base** checkpoint，不做训练、不做额外模型下载。

已验证环境：

```text
GPU: AutoDL RTX 4090 24GB
Model: /root/autodl-tmp/models/Qwen3.5-9B-ms
Python runtime: /root/autodl-tmp/agentrecx-sft/bin/python
```

服务启动命令（服务端）：

```bash
cd /root/autodl-tmp/agentrecx
/root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --port 8009 \
  --max-new-tokens 512
```

本地评测端通过 SSH tunnel 暴露为 `http://127.0.0.1:18029`：

```bash
ssh -N -L 18029:127.0.0.1:8009 <autodl-host> -p <port> -i <key>
```

该服务同时服务 `/plan` 和 `/explain`。实测 peak VRAM 为 **23,709 MiB**，剩余约
**384 MiB**；显存余量很小，必须顺序执行，不能并发加载第二个 9B 模型。

Qwen 服务不在本仓库的 `requirements-eval.txt` 中；它属于外部环境资产。

## 4. 运行 A/B：30 条自然语言任务

设置真实 Qwen Planner 环境：

```bash
export AGENTRECX_QWEN_PLANNER=1
export AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029
export AGENTRECX_QWEN_PLANNER_MODEL=Qwen3.5-9B-Base
export AGENTRECX_QWEN_PLANNER_TIMEOUT_S=300
```

生成任务集：

```bash
PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/task_suite.py
```

运行 Baseline 和 Corrective（同一进程，两个 policy）：

```bash
PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/run_task_arms.py \
  --policies baseline corrective
```

输出：

```text
runs/p20_agent_eval/task_suite.json
runs/p20_agent_eval/task_arm_baseline.json
runs/p20_agent_eval/task_arm_corrective.json
runs/p20_agent_eval/task_arms.json
```

## 5. 运行 C：Complete Shadow Agent

C 使用 10 条代表性子集，通过现有 MCP server 逐任务启动，以保证每个任务使用自己的
Memory fixture 和 user key。

```bash
export AGENTRECX_QWEN_PLANNER=1
export AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029
export AGENTRECX_QWEN_PLANNER_MODEL=Qwen3.5-9B-Base
export AGENTRECX_QWEN_EXPLAIN=1

PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/run_complete_shadow.py
```

该脚本会：

- 启动现有 `recommendation.mcp.server`；
- 打开 `--corrective-rag`；
- 打开 `--evidence-verifier`（Shadow only）；
- 设置 `--metadata-mode enriched`；
- 记录 final answer、P15、Verifier、Gate、Planner、tokens、latency。

输出：

```text
runs/p20_agent_eval/complete_shadow.json
```

Shadow 不会改变 Gate、Memory、选品或最终回答；当前不启用任何自动拦截/改写。
C 的 MCP 冷启动约 20–31 秒/任务，E2E latency 明显高于 A/B direct harness。

## 6. 生成 Claim Review Sheet

```bash
PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/build_review_sheet.py
```

输出：

```text
runs/p20_agent_eval/claim_review_sheet.json
runs/p20_agent_eval/claim_review_sheet.csv
runs/p20_agent_eval/claim_review_sheet.md
```

所有行默认：

```text
human_review_status = pending_independent_human_review
```

Harness 不会把任何行升级为 human gold。P15 的 `automated_validated` 只是结构校验，
不是语义真实性。

## 7. 49 条 Verifier GPU Pilot（可选复跑）

```bash
export AGENTRECX_QWEN_EXPLAIN=1
PYTHONPATH=. .venv/bin/python experiments/p20_metadata/build_verifier_dataset.py
PYTHONPATH=. .venv/bin/python experiments/p20_metadata/run_verifier_pilot.py
PYTHONPATH=. .venv/bin/python experiments/p20_metadata/run_verifier_qwen_only.py
```

输出均在 `runs/p20_metadata/`，属于生成型实验产物，按仓库规则不提交 Git。

## 8. 回归

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_evidence_verifier.py \
  tests/test_corrective_rag.py \
  tests/test_planner_reliability.py \
  tests/test_p14c0_contract.py \
  tests/test_p15_rag.py \
  tests/test_p16_skills.py \
  tests/test_p17_mcp.py \
  tests/test_qwen_live_mcp_integration.py \
  tests/test_electronics_metadata_enrichment.py \
  tests/test_docs.py
```

已验证结果：

```text
149 passed, 1 skipped
```

P18 已知状态仍为：

```text
1 failed, 5 passed
```

原因是既有 `recommendation/control/catalog_search.py` freeze hash 漂移；本轮没有修改
P18 Frozen Manifest，也没有使用 P18 Reserved 调参。

## 9. 结果口径

- A/B 是 30 条任务；
- C 是 10 条代表性子集，不是全量；
- A/B direct harness 与 C MCP harness 的 Memory 装载方式不完全一致；
- 不能在未标明 harness 的情况下直接比较三者 task success；
- 所有 Dev/Eval-100 标签都是 provisional；
- 自然 Claim 的 human review 仍为 pending；
- 真实 Qwen 任务成功率显著低于受控单功能测试，不能把受控测试结果当生产准确率。

完整结果与失败分析见 `docs/AGENT_QUALITY_BASELINE.md`。
