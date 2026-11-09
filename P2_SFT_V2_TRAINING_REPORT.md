# P2-SFT-v2 增量训练报告

## 1. 是否实际训练

**是。** 从现有 `qwen3_5_9b_lora` adapter 初始化，训练一轮增量 SFT，未覆盖旧 adapter。

## 2. 数据

- 版本：`p2sft.data.v1.2`；
- 新样本：600 条（480 train / 120 val）；
- 旧 train 2400 + 新 480 = 2880 train；
- 旧 val 240 + 新 120 = 360 val；
- 新样本来自 6 个独立模板族：多条件更新、clarify/noop、复杂撤回、跨轮指代、accessory/main role、预算反复；
- 不使用 Parser Test、100 条困难测试、frozen task 作为训练；
- 每条经 Schema、evidence substring、字段语义校验；
- 新数据 hash：`fcc0417c6fcf0bf2d2b50feec31cf7cb2f4fbf8691192d10365cfa5e950079b3`；
- 混合后 train hash：`e91911a5a61ef01cc1f805c01ec2a3a30cc113b570ef582efe6def40d57f197e`；
- manifest：`training/agent_sft/data_v2/manifest.json`。

## 3. 训练参数

```text
init adapter: training/agent_sft/output/qwen3_5_9b_lora/adapter
output:       training/agent_sft/output/qwen3_5_9b_lora_v2
steps:        180
epochs:       1.0
lr:           5e-5
LoRA:         r=16, alpha=32, dropout=0.05, all-linear
batch:        2 x 8 = 16
dtype:        BF16
seed:         20261106
runtime:      1828.63 s
train_loss:   0.0049319
eval_loss:    0.0497856
adapter:      ~166 MB
```

报告：`training/agent_sft/output/qwen3_5_9b_lora_v2/training_report.json`

## 4. Parser 最终诊断

SFT-v2 在**已冻结的独立困难测试**上最终诊断一次（此前未用于训练）：

| metric | SFT-v1 hard | SFT-v2 hard |
|---|---:|---:|
| JSON Valid | 0.93 | **0.99** |
| Exact Match | 0.66 | **0.92** |
| Macro-F1 | 0.635 | 0.702 |
| Slot Accuracy | 0.535 | 0.951 |
| Retraction Accuracy | 1.00 | 1.00 |
| Unsupported Rate | 0.07 | 0.01 |

说明：困难集 100 条仍无人工审核，该结果只能作为 final diagnostic，不能当作正式人工 Gold。

原始 200 条 Parser Test 最终诊断：Schema 1.00、Exact 1.00、Slot 1.00、
Retraction 1.00、Unsupported 0.00。该结果接近饱和，可能部分来自新数据与模板语义相近，
不能单独作为泛化结论。

## 5. Agent 端到端

D1-v2（SFT-v2）真实运行 20 个 dev 任务：

| 指标 | C | D0 | D1-v1 | D1-v2 |
|---|---:|---:|---:|---:|
| V2 PASS | 12 | 12 | 12 | 12 |
| FAIL | 5 | 5 | 5 | 5 |
| UNKNOWN | 3 | 3 | 3 | 3 |
| legacy pass | 8 | 8 | 7 | 7 |
| p50 / p95 ms | 9173 / 20167 | 12139 / 19755 | 17823 / 33086 | 15341 / 20625 |
| tokens | 0 | 25,973 | 25,760 | 25,721 |

**结果：SFT-v2 明显改善了 Parser 困难集 exact/slot，但没有改善 V2 Agent 任务成功率。**

## 6. 结论

- 增量 SFT 真实发生并产出新 adapter；
- 解析器在困难表达上改善明显；
- Agent 层多个失败任务与 C/D0/D1-v1 完全相同（dev_05/06/07/12/13/14/17/18），说明剩余瓶颈主要是 M5 lifecycle、goal/role 语义与候选召回/证据，不是单纯 Parser；
- 不把 Parser 提升冒充 Agent 提升。
