# P2-SFT：Qwen3.5-9B Preference Operation LoRA

本目录是 **本地准备 + AutoDL G0/G1/G2** 的独立模块。目标模型只有：

```text
Qwen/Qwen3.5-9B
```

不再维护 Qwen3-8B 训练路线。**数据版本需特别注意**：原始 `a031/47d/d790`
快照已不在工作区；当前锁定版本为 `p2sft.data.v1.1`（`fa3b/c831/e85f`）。
完整差异见 `reports/DATA_VERSION_AUDIT.md`，不得再声称数据未改变。

官方兼容性依据：

- [Transformers Qwen3.5 文档](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/docs/source/en/model_doc/qwen3_5.md)
- 官方 checkpoint：`Qwen/Qwen3.5-9B`
- 文本路径类：`Qwen3_5ForCausalLM`
- Qwen3.5 是 hybrid-attention、natively multimodal 家族；本任务只加载 **text-only
  CausalLM**，不加载 vision tower，也不做多模态训练。

## 0. 运行包

- `AUTODL_RUN_PACKAGE.md` — AutoDL 运行命令与备份
- `AUTODL_RUN_PACK_MANIFEST.json` — 文件/数据 hash 清单
- `data/LOCKED_VERSION.json` — 锁定数据版本 `p2sft.data.v1.1`
- `reports/DATA_VERSION_AUDIT.md` — 原始/当前数据不一致审计

## 1. 当前阶段

| 项目 | 状态 |
|---|---|
| Qwen3.5-9B 训练/推理代码迁移 | READY |
| 数据版本核验 / 锁定 AutoDL 版本 | `p2sft.data.v1.1`，原始 a031 快照缺失 |
| G0 模型加载/chat template/JSON/mask | **PASS（AutoDL RTX 6000D 83 GiB）** |
| G1 少量 LoRA step | **PASS（finite loss、adapter reload）** |
| G2 正式 SFT | **PASS（2 epochs，300 steps）** |
| Base vs SFT Parser 指标 | **SFT schema 0.97 / exact 0.945，见报告** |

本仓库已有实际 G0/G1/G2 结果与 adapter：

- 运行环境：RTX 6000D 83 GiB、Python 3.11、torch 2.11+cu128、transformers 5.22.1；
- adapter：`output/qwen3_5_9b_lora/adapter/`，本地备份 `output/qwen3_5_9b_lora_adapter.tar.gz`；
- 报告：`../P2_SFT_EXPERIMENT_REPORT.md` 与 `reports/g0_preflight.json`、
  `reports/g1_smoke_report.json`、`output/qwen3_5_9b_lora/training_report.json`。

## 2. AutoDL 环境

AutoDL 镜像建议 CUDA 12.x + Python 3.10/3.11 + 预装 PyTorch；不要为了本任务重装
CUDA/PyTorch。

```bash
conda create -n agentrecx-sft python=3.11 -y
conda activate agentrecx-sft
pip install -r training/agent_sft/requirements-autodl.txt
```

`requirements-autodl.txt` 要求支持 Qwen3.5 的 Transformers（>=5.9.0）、PEFT 和
Accelerate。若 G0 报 `Qwen3_5ForCausalLM` 不存在，说明环境版本不匹配；停止训练并按
官方 Qwen3.5 文档安装对应 Transformers，不要降级到 Qwen3-8B。

模型下载：

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B
```

可选：Qwen3.5 Gated DeltaNet 的 fast kernels（A100 上初始 LoRA 可用 PyTorch fallback）：

```bash
# 仅在 G0 确认基础环境正常后尝试
# pip install -r training/agent_sft/requirements-kernels.txt
```

## 3. G0：GPU 预检

```bash
python -m training.agent_sft.gpu_preflight \
  --model /root/models/Qwen3.5-9B \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/reports/g0_preflight.json
```

检查：

- 实际加载类 / `model_type` / 参数规模 / dtype；
- tokenizer、chat template、`enable_thinking=False`；
- 真实样本长度、截断；
- completion-only mask；
- 一次 JSON 生成与 schema 解析；
- Qwen3.5 hybrid-attention 的 `layer_types`、`causal_conv1d`/`fla` 可用性、
  kernel fallback 模式；
- GPU 型号/显存、加载后显存；
- PEFT `all-linear` 实际 LoRA 模块数量/分布、trainable 参数。

任何一项失败都必须停止 G1/G2，不生成模拟日志。

## 4. G1：少量训练 Step

```bash
python -m training.agent_sft.smoke_train \
  --model /root/models/Qwen3.5-9B \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/output/qwen3_5_9b_smoke \
  --steps 2
```

检查：

- loss 有限、无 NaN/OOM；
- backward/optimizer step 正常；
- Qwen3.5 kernel 模式、peak VRAM；
- adapter 保存；
- reload adapter 后生成一次 JSON，schema valid。

## 5. G2：正式 LoRA-SFT

BF16 LoRA（A100 80GB）：

```bash
python -m training.agent_sft.train \
  --config training/agent_sft/configs/qwen3_5_9b_lora.yaml \
  --data-dir training/agent_sft/data \
  --output-dir training/agent_sft/output/qwen3_5_9b_lora
```

QLoRA 兜底：

```bash
pip install -r training/agent_sft/requirements-qlora.txt
python -m training.agent_sft.train \
  --config training/agent_sft/configs/qwen3_5_9b_qlora.yaml \
  --data-dir training/agent_sft/data \
  --output-dir training/agent_sft/output/qwen3_5_9b_qlora
```

起始参数：

```text
LoRA r=16 alpha=32 dropout=0.05
learning_rate=1e-4
max_seq_length=2048
per_device_train_batch_size=2
gradient_accumulation_steps=8 -> effective batch 16
epochs=2
bf16=true
completion-only loss
seed=20261105
```

`training_report.json` 保存配置、数据 hash、模型类/参数、环境、GPU、seed、loss history。

## 6. Base vs SFT Parser 评测

Rule baseline（CPU，历史参考，不是模型结果）：

```bash
python -m training.agent_sft.evaluate_parser \
  --data training/agent_sft/data/test.jsonl --provider rule \
  --out-dir training/agent_sft/eval/rule
```

Base Qwen3.5-9B：

```bash
python -m training.agent_sft.evaluate_parser \
  --data training/agent_sft/data/test.jsonl --provider base \
  --model /root/models/Qwen3.5-9B \
  --out-dir training/agent_sft/eval/base
```

SFT adapter：

```bash
python -m training.agent_sft.evaluate_parser \
  --data training/agent_sft/data/test.jsonl --provider sft \
  --model /root/models/Qwen3.5-9B \
  --adapter training/agent_sft/output/qwen3_5_9b_lora/adapter \
  --out-dir training/agent_sft/eval/sft
```

三者使用同一 Parser Test、Prompt Schema、`max_new_tokens=256`、greedy decoding 和
Qwen3.5 non-thinking 模板。输出逐样本预测、错误、per-op F1 与 p50/p95 latency。

合并：

```bash
python -m training.agent_sft.compare_parser_results \
  --rule-dir training/agent_sft/eval/rule \
  --base-dir training/agent_sft/eval/base \
  --sft-dir training/agent_sft/eval/sft \
  --out training/agent_sft/eval/qwen3_5_9b_parser_comparison.csv
```

## 7. 导出

```bash
tar -czf qwen3_5_9b_lora_adapter.tar.gz \
  -C training/agent_sft/output/qwen3_5_9b_lora adapter training_report.json training_log_history.json
```

## 8. 当前未完成

- G0/G1/G2 均未在 AutoDL 执行；
- 无 adapter、无 Base/SFT metrics；
- 未接入 AgentRec-X M5/R1/M6.1 执行闭环；
- 本阶段不产生 Agent 指标或训练效果声明。
