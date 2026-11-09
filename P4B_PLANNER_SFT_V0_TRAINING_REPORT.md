# P4B Planner SFT-v0 首次 LoRA 训练报告

## 1. 训练结果

| 项目 | 实际值 |
|---|---|
| Base model | `/root/autodl-tmp/models/Qwen3.5-9B-ms` |
| Model class | `Qwen3_5ForCausalLM` |
| GPU | NVIDIA RTX 6000D |
| 初始 Adapter | 无（Base，不加载 Preference Parser SFT-v2） |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear |
| Train / Val | 267 / 53 |
| Epochs | 2 |
| 实际 steps | 34 |
| Train runtime | 516 s（8 分 36 秒） |
| Train loss | 0.2635 |
| Eval loss | 0.0009858 |
| Trainable params | 43,278,336 |

训练命令：

```bash
cd /root/autodl-tmp/agentrecx
PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m training.planner_sft.train_lora \
  --config training/planner_sft/config_lora.yaml \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms \
  --data-dir data/planner_sft_v0 \
  --output-dir training/planner_sft/output/qwen3_5_9b_planner_lora_v0
```

Adapter 保存位置（远端）：

```text
/root/autodl-tmp/agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter/
```

包含 `adapter_model.safetensors`、`adapter_config.json`、tokenizer、chat_template。

## 2. Assistant-only loss 检查

训练前 preflight：

```text
train_samples=267
val_samples=53
max_total_tokens=1526
prompt_tokens_min=447
completion_tokens_min=35
masked_prompt_tokens=233435
```

单样本编码检查：

```text
prompt_tokens=457
completion_tokens=47
masked=True
```

即 prompt（system + user）全部为 `-100`，只有 assistant Planner Action token 参与 loss。

## 3. Val 评测

用重新加载的 Planner Adapter 启动现有 `/plan`：

```text
proposal server on 127.0.0.1:8008 adapter=.../qwen3_5_9b_planner_lora_v0/adapter
```

53 条 val 真实模型推理：

```text
samples=53
action_exact=53/53 (1.000)
schema_valid=53/53 (1.000)
argument_exact=53/53 (1.000)
errors=[]
```

分类：

```text
RETRIEVE: 16/16
ASK:        8/8
RERANK:    15/15
FINALIZE:  14/14
```

**重要限制：** 该 val 与训练场景模板高度同分布，模型明显记住了 label rationale；
100% 只能证明训练/格式/加载链路可通过，**不能作为泛化能力或最终决策质量结论**。

## 4. 真实推理案例

### RETRIEVE

```json
{
  "action": "RETRIEVE",
  "arguments": {"query": "I need a water bottle under $60 for tailgating."},
  "rationale": "No current candidates and the request is sufficiently specified; call the retrieval tool."
}
```

latency ≈ 4168 ms。

### ASK

```json
{
  "action": "ASK",
  "arguments": {"question": "Which product type, trip style, and budget should I use?"},
  "rationale": "No candidates and a missing decision attribute; ask before retrieval."
}
```

latency ≈ 2915 ms。

### RERANK

```json
{
  "action": "RERANK",
  "arguments": {},
  "rationale": "Candidates exist, have not been reranked, and current catalogue evidence supports the soft preference; consult M6.1 before finalizing."
}
```

latency ≈ 3184 ms。

### FINALIZE

```json
{
  "action": "FINALIZE",
  "arguments": {},
  "rationale": "Candidates already passed M6.1 reranking and are validated; finalize without a new retrieval."
}
```

latency ≈ 1981 ms。

## 5. 验收状态

| 验收项 | 状态 |
|---|---|
| GPU LoRA 训练完成 | **PASS** |
| Assistant-only loss | **PASS**（mask 检查通过） |
| 独立 Adapter 保存 | **PASS** |
| 重新加载 Adapter | **PASS** |
| Val 真实预测 | **PASS**（53/53） |
| `/plan` 使用新 Adapter | **PASS**（真实 HTTP 推理） |
| 影响 Preference Parser | **未触碰**；新 Adapter 路径独立 |
| 决策质量泛化 | **UNVERIFIED / 不可由本 val 证明** |
| 最终任务成功 | **UNVERIFIED**（未跑 Agent Benchmark） |

## 6. 真实失败与弱点

本轮 val 无失败，但这是开发集同分布结果，暴露的实际弱点是：

1. **模板记忆而非决策泛化**：模型复述训练 rationale；val query/场景模式与 train 相同。
2. **未验证 RERANK 的实际工具效果**：标签是 RERANK，但真实 M6.1 是否改变排序未测。
3. **未验证真实 Observation 变化**：val 是静态观测，不是多步交互中的新 Observation。
4. **未验证工具执行后的状态**：只评 Action，不评 Tool 结果、Eligibility 或 final answer。
5. **数据规模小（320）且合成**：不能把 val 100% 当作真实线上准确率。
6. **未做 Base 同分布对照**：Base 的先前真实轨迹只有 P3D 两条定性轨迹；
   本轮没有量化 Base-vs-Adapter 对照。

## 7. 训练产物

- `P4B_TRAINING_REPORT.json`
- `P4B_TRAINING_LOG_HISTORY.json`
- `P4B_VAL_EVAL.json`
- `P4B_VAL_PREDICTIONS.jsonl`
- `P4B_PLANNER_SFT_V0_SOURCE_DIFF.txt`
- 远端 Adapter：`training/planner_sft/output/qwen3_5_9b_planner_lora_v0/adapter/`

## 8. 结论

- **训练成功**；Adapter 可被现有 `/plan` 加载；
- **Action JSON 格式合法**；在开发集上 Action/Arguments 全匹配；
- **但 100% val 是模板化开发集结果，不是泛化结论**；
- 未运行完整 Agent Benchmark，未验证最终任务成功；
- Preference Parser SFT-v2 Adapter 未被加载或覆盖；
- 未修改 Gold/Evaluator/Serving V1，未 commit/push/deploy。
