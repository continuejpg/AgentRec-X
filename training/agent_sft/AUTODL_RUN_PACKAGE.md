# AutoDL Qwen3.5-9B 运行包（P2-SFT）

## 0. 运行包状态

本包是 **待 GPU 执行包**：

- G0/G1/G2 尚未执行；
- 没有 adapter；
- 没有 Base/SFT metrics；
- 不允许用本文件声称训练已完成。

镜像内容与数据版本以 `run_pack_manifest.json` 的 SHA-256 为准。

## 1. 唯一模型

```text
Qwen/Qwen3.5-9B
```

文本路径：`Qwen3_5ForCausalLM`  
不维护 Qwen3-8B 训练/回退路线。

## 2. 数据版本

当前锁定版本：

```text
version: p2sft.data.v1.1
train: fa3b1c828bdc9d8cbaca51e7eddd656fb495fbf15bc4cb3fe6751fa08bc8df97
val:   c8317ad3d9d8822096bf557559407d82f10af18fcaa3738751b413758f95dd05
test:  e85f85767508d0a2533134e031dd78e46553c93360b3866566532bd6b6807b02
```

不要使用旧报告里的 `a031/47d/d790` 作为训练数据 hash。原始快照已缺失，详见
`reports/DATA_VERSION_AUDIT.md`。

## 3. 依赖安装

适用 GPU：A100 80GB、H100 80GB、**RTX PRO 6000 96GB（Blackwell，SM12x）**均可用于
BF16 LoRA。96GB 显存足够 9B BF16 + LoRA；RTX PRO 6000 的实际 compute capability
必须由 G0 打印（SM120/SM122 取决于具体卡与驱动），不能只按商品名判断。

RTX PRO 6000 注意：

- 使用镜像内支持 Blackwell 的 CUDA/PyTorch（CUDA 12.8+，`torch.cuda.is_bf16_supported()`
  必须为 True）；
- `causal-conv1d` / `fla` 的 fast kernel 可能没有对应 SM12x 预编译 wheel；
  初始训练使用 `use_kernels=false` 的 PyTorch fallback 即可；
- 不要一开始强装 FlashAttention/`fla`；先跑 G0，确认 import 和 fallback 正常；
- 96GB 下初始 batch 2/grad accumulation 8 很保守，可先用它跑通 G1/G2，再考虑增大
  micro-batch；
- QLoRA/bitsandbytes 在 Blackwell 上依赖版本更敏感，优先 BF16 LoRA，不把 QLoRA
  作为首跑路线。

```bash
conda create -n agentrecx-sft python=3.11 -y
conda activate agentrecx-sft
pip install -r training/agent_sft/requirements-autodl.txt
```

G0 若报告 `Qwen3_5ForCausalLM` 不存在，停止；按官方 Qwen3.5 文档修正 Transformers
版本，不降级到 Qwen3-8B。

建议环境变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/autodl-tmp/hf_cache
```

先做 GPU/驱动自检（不加载模型）：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("bf16", torch.cuda.is_bf16_supported())
print("mem_gib", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

期望：`bf16=True`；RTX PRO 6000 的实际 compute capability 由这里读取并记录到 G0。

## 4. 模型下载

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B
```

## 5. G0 预检

```bash
python -m training.agent_sft.gpu_preflight \
  --model /root/models/Qwen3.5-9B \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/reports/g0_preflight.json
```

输出必须包含：

- 实际模型类/`model_type`/参数规模；
- hybrid-attention `layer_types` 与 linear/full 层数；
- `causal_conv1d`/`fla` 可用性，fallback/fast kernel 模式；
- GPU 型号、总显存、加载后/峰值显存；
- tokenizer、chat template、`enable_thinking=False`；
- 样本长度/截断；
- completion-only mask；
- JSON 生成/schema 解析；
- LoRA 模块数量、分布、trainable 参数。

G0 未通过时不得执行 G1/G2。

## 6. G1 少量 LoRA Step

```bash
python -m training.agent_sft.smoke_train \
  --model /root/models/Qwen3.5-9B \
  --data-dir training/agent_sft/data \
  --out training/agent_sft/output/qwen3_5_9b_smoke \
  --steps 2
```

输出必须包含：

- finite loss；
- backward/step 成功；
- peak VRAM；
- kernel 模式；
- adapter 保存；
- reload adapter 后一次 schema-valid JSON 生成。

## 7. G2 正式训练

```bash
python -m training.agent_sft.train \
  --config training/agent_sft/configs/qwen3_5_9b_lora.yaml \
  --data-dir training/agent_sft/data \
  --output-dir training/agent_sft/output/qwen3_5_9b_lora
```

保存：

```text
output/qwen3_5_9b_lora/adapter/
output/qwen3_5_9b_lora/checkpoints/
output/qwen3_5_9b_lora/training_report.json
output/qwen3_5_9b_lora/training_log_history.json
output/qwen3_5_9b_lora/trainer_metrics.json
```

## 8. Base / SFT Parser

```bash
python -m training.agent_sft.evaluate_parser \
  --data training/agent_sft/data/test.jsonl --provider base \
  --model /root/models/Qwen3.5-9B \
  --out-dir training/agent_sft/eval/base

python -m training.agent_sft.evaluate_parser \
  --data training/agent_sft/data/test.jsonl --provider sft \
  --model /root/models/Qwen3.5-9B \
  --adapter training/agent_sft/output/qwen3_5_9b_lora/adapter \
  --out-dir training/agent_sft/eval/sft

python -m training.agent_sft.compare_parser_results \
  --rule-dir training/agent_sft/eval/rule \
  --base-dir training/agent_sft/eval/base \
  --sft-dir training/agent_sft/eval/sft \
  --out training/agent_sft/eval/qwen3_5_9b_parser_comparison.csv
```

## 9. 备份

```bash
RUN_ID=20261105_qwen35_9b_lora
tar -czf ${RUN_ID}_adapter.tar.gz \
  -C training/agent_sft/output/qwen3_5_9b_lora \
  adapter training_report.json training_log_history.json trainer_metrics.json
tar -czf ${RUN_ID}_eval.tar.gz \
  -C training/agent_sft/eval \
  rule base sft qwen3_5_9b_parser_comparison.csv
```

回传时必须同时保留：

- adapter 目录与 `chat_template.jinja`；
- `schema_version.txt`；
- `training_report.json`；
- 数据 hash；
- G0/G1/G2 报告；
- Base/SFT `parser_metrics.csv` 与 `predictions.jsonl`。

## 10. 禁止

- 不得用 Qwen3-8B 训练结果替代；
- 不得伪造 G0/G1/G2 或 metrics；
- 不得修改数据/manifest 覆盖锁定版本；
- 不得运行 frozen Agent Benchmark；
- 不得开始 Agent RL；
- 不 commit/push/deploy。
