# P2-SFT：Qwen3.5-9B 训练、Base/SFT Parser 与 AutoDL 环境报告

## 0. 结论

Qwen3.5-9B 的 G0/G1/G2 已在真实 AutoDL GPU 上执行完成，并产出了实际 LoRA adapter 与
Base/SFT Parser 对照结果。

| 项目 | 状态 |
|---|---|
| Qwen3.5-9B 加载 | PASS |
| G0 兼容性/模板/mask/JSON/LoRA | PASS |
| G1 少量 LoRA steps + adapter reload | PASS |
| G2 正式 LoRA-SFT | PASS（2 epochs，300 steps） |
| Base vs SFT Parser 指标 | PASS，见下表 |
| Agent D0/D1 端到端 | NOT IMPLEMENTED |
| 人工 20 条 test 核验 | PENDING，未冒充已审核 |

数据版本存在已知不一致，**本轮训练使用当前锁定版本 `p2sft.data.v1.1`**；不再宣称
原始 a031 数据未改变。完整审计见 `training/agent_sft/reports/DATA_VERSION_AUDIT.md`。

---

## 1. 实际环境

| 项目 | 实测值 |
|---|---|
| 实例 | AutoDL `autodl-container-8t4t7dlvfu-c4d01ee1` |
| GPU | NVIDIA RTX 6000D |
| 显存 | 85651 MiB ≈ 83.0 GiB usable |
| Compute capability | (12, 0) / SM120 |
| Driver / CUDA | 595.71.05 / 13.2 driver, torch CUDA 12.8 |
| Python | 3.11.18 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.22.1 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |
| 模型 | `Qwen/Qwen3.5-9B`（ModelScope 镜像） |
| 模型类 | `Qwen3_5ForCausalLM` |
| model_type | `qwen3_5_text` |
| 参数量 | 8,953,803,264 |
| 训练 dtype | BF16 |
| 数据 | 2400 train / 240 val / 200 test，`p2sft.data.v1.1` |

Qwen3.5 是 hybrid-attention（24 linear + 8 full attention）模型；本机没有
`causal_conv1d` / `flash-linear-attention`，G0/G1/G2 使用 **PyTorch fallback**，
正确但较慢。

---

## 2. G0：加载与兼容性预检

命令：

```bash
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD /root/autodl-tmp/agentrecx-sft/bin/python \
  -m training.agent_sft.gpu_preflight \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/reports/g0_preflight.json
```

结果：**PASS，proceed_to_G1**

- tokenizer `Qwen3_5Tokenizer`；
- chat template 存在，`enable_thinking=False` 可用；
- completion-only mask：PASS；
- 32 条样本最大 total tokens 337，无截断；
- 一次 greedy JSON 生成和 schema 校验成功（修复了 prompt 的 schema 字段说明）；
- LoRA `all-linear`：2232 个模块，43,278,336 trainable parameters；
- 模块分布包含 `q/k/v/o_proj`、`gate/up/down_proj`、`in_proj_qkv/z/b/a`、`out_proj`；
- 加载后显存 16.68 GiB，LoRA 后 16.85 GiB，峰值 16.85 GiB；
- GPU 可用，BF16 支持。

G0 报告：

`training/agent_sft/reports/g0_preflight.json`

---

## 3. G1：少量 LoRA steps

命令：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/agentrecx-sft/bin/python \
  -m training.agent_sft.smoke_train \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/output/qwen3_5_9b_smoke --steps 2
```

结果：**PASS**

- losses：`[0.4634, 0.1930]`，finite，无 NaN/OOM；
- trainable params：43,278,336；
- peak allocated：23.85 GiB；
- adapter 保存成功；
- reload adapter 后生成 schema-valid JSON；
- kernel 仍为 PyTorch fallback。

G1 报告：

`training/agent_sft/reports/g1_smoke_report.json`

---

## 4. G2：正式 LoRA-SFT

命令：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/agentrecx-sft/bin/python \
  -m training.agent_sft.train \
  --config training/agent_sft/configs/qwen3_5_9b_lora.yaml \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --data-dir training/agent_sft/data \
  --output-dir training/agent_sft/output/qwen3_5_9b_lora
```

实测训练结果：

```text
steps: 300
epochs: 2.0
train_runtime: 3307.24 s (~55.1 min)
train_loss: 0.0145605
final eval_loss: 0.1361951
train_samples_per_second: 1.451
trainable_parameters: 43,278,336
adapter size: ~166 MB
```

这些是训练/验证 loss，**不是推荐质量指标**；最终效果以 Parser 评测为准。

Adapter：

```text
/root/autodl-tmp/agentrecx/training/agent_sft/output/qwen3_5_9b_lora/adapter/
本地备份：training/agent_sft/output/qwen3_5_9b_lora_adapter.tar.gz
远程备份：/root/autodl-tmp/qwen3_5_9b_lora_adapter.tar.gz
```

配置、数据 hash、环境、seed、loss history：

`training/agent_sft/output/qwen3_5_9b_lora/training_report.json`

---

## 5. Base vs SFT Parser 评测

相同 Parser Test、Prompt Schema、`max_new_tokens=256`、greedy decoding、Qwen3.5
non-thinking 模板。

| metric | Rule (historical) | Base Qwen3.5-9B | SFT Qwen3.5-9B |
|---|---:|---:|---:|
| JSON Schema Valid Rate | 1.00 | 0.33 | **0.97** |
| Operation Exact Match | 0.82 | 0.095 | **0.945** |
| Operation Macro-F1 | 0.750 | 0.254 | 0.703 |
| Operation Micro-F1 | 1.00* | 0.481 | 0.959 |
| Slot Accuracy | 1.00 | 0.12 | **0.945** |
| Retraction Accuracy | 1.00 | 1.00 | 1.00 |
| Unsupported Proposal Rate | 0.00 | 0.67 | **0.03** |
| Failure/Abstention Rate | 0.00 | 0.67 | **0.03** |
| p50 latency ms | ~0.00003 | 2095.9 | 2972.9 |
| p95 latency ms | ~0.00003 | 3046.0 | 4955.1 |

说明：

- SFT 相对 Base 在 schema valid、exact match、slot accuracy、unsupported rate 上明显改善；
- **Operation Macro-F1 = 0.703，低于 Rule 的 0.750**，不能宣称所有指标都优于规则基线；
- latency 是单请求 greedy 测量，Rule 是纯 CPU 正则，Base/SFT 是 GPU fallback 模型，
  不可把 Rule 的微秒级延迟与模型延迟直接比较；
- Base/SFT latency 差异可能来自生成长度和适配器路径，不是质量指标。

逐样本预测：

```text
training/agent_sft/eval/base/predictions.jsonl
training/agent_sft/eval/sft/predictions.jsonl
```

汇总 CSV：

`training/agent_sft/eval/qwen3_5_9b_parser_comparison.csv`

---

## 6. 数据版本问题（必须保留）

用户给出的原始版本：

```text
a031... / 47d3... / d790...
```

当前训练版本：

```text
p2sft.data.v1.1
fa3b... / c831... / e85f...
```

原始 `.jsonl` 快照已不在工作区，无法做逐字节 diff；代码变更历史确认存在
price_update Gold 修复、catalogue context 对齐、material_pref state 修复，所以
**不能继续宣称数据未改变**。当前训练使用的是锁定版本，训练报告记录了该版本 hash。

详见：

- `training/agent_sft/reports/DATA_VERSION_AUDIT.md`
- `training/agent_sft/data/LOCKED_VERSION.json`

20 条 pending 人工抽检：尚未完成人工审核，`human_verified_count=0`。

---

## 7. 未完成

- Agent D0/D1 端到端接入：未完成，本报告只报告 Parser 指标；
- 20 条人工 test 核验：pending；
- 未运行 frozen Agent Benchmark；
- 未修改 Serving V1、SASRec、Two-Tower、BM25、ItemCF；
- 未开始 Agent RL/DPO/PPO/GRPO；
- 未在 Qwen3.5 上安装 fast kernels；当前是 fallback 路径。

## 8. 复现命令摘要

```bash
# 环境
/root/miniconda3/bin/conda create -p /root/autodl-tmp/agentrecx-sft python=3.11 -y
/root/autodl-tmp/agentrecx-sft/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
/root/autodl-tmp/agentrecx-sft/bin/pip install transformers==5.22.1 peft accelerate \
  pyyaml safetensors sentencepiece protobuf

# 模型（ModelScope，已下载）
# /root/autodl-tmp/models/Qwen3.5-9B-ms

# G0/G1/G2 与评测命令见 training/agent_sft/README.md
```
