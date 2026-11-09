# P14-B3 — 两种个性化 Reranker 直接实验（Qwen3-Reranker-4B + Qwen3.5-9B Setwise）

> 范围：仅 P14-B3。真实加载并推理 `Qwen/Qwen3-Reranker-4B`（4-bit）与 Qwen3.5-9B Base Setwise；同一固定候选池对照 BM25/V0。
> 未训练/微调，未改 V0/Gate/Memory/Planner/标签，未运行 Reserved Test，未 commit。
> **Setwise 因 GPU 资源冲突被阻塞，未产生任何结果，也没有伪造结果**（依据 C7 明确记录阻塞；详见 §5）。

## 0. 结论

- **Qwen3-Reranker-4B + Instruction-Aware Preferences 在探索性 Dev 20 上超过 V0：Top-1 12/15 vs 11/15，MRR@10 0.856 vs 0.833；Top-5 都是 14/15。** 优势仅 1 条任务，样本小，不能视为稳健结论。
- **明确偏好指令有独立收益**：Query Only 8/15 → Query+Prefs 10/15 → Query+Prefs+Instruction 12/15；指令版本相比 Query+Prefs 的 MRR 从 0.769 升到 0.856，5 个任务 Top-1 改变，+2 正确、0 丢失；负偏好冲突 Top-1 从 3 降到 1。
- **Setwise 未运行**：在 Qwen3.5-9B 服务持续占用 17.6 GiB、仅剩约 6.18 GiB 显存的 4090D 上，9B 4-bit 直接加载需约 5.94 GiB 权重并触发 OOM；CPU 端 9B 一个 6 候选组超过 13 分钟未产生输出。按任务要求不得停止 Qwen 服务/实例，故 **Setwise 无结果、无指标、无伪造**。
- **不替换 V0 默认实现**：Qwen3-Reranker 可作为可选候选继续验证；V0 仍是最低成本、可解释、已验证的默认推荐底座。

## 1. 模型与运行 Manifest

### Qwen3-Reranker-4B（已完成真实推理）

| 项 | 值 |
| --- | --- |
| Model | `Qwen/Qwen3-Reranker-4B` |
| Revision | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` |
| License | Apache-2.0 |
| 架构 / 规模 | `Qwen3ForCausalLM`，4B |
| 下载来源 | Hugging Face via `hf-mirror.com` → `/root/autodl-tmp/models/Qwen3-Reranker-4B` |
| 量化 | bitsandbytes NF4 4-bit（double quant），fp16 compute |
| 官方打分 | yes/no token logits → `softmax(false,true)[1]`；不是普通生成概率 |
| 权重 sha256 | shard1 `cf2e87cbf71fa628961532232e04dd6c19702a0a057f5e2aff95ea1aca4fd488`；shard2 `78946d22b7f6456ea7a5358dbdf3982de36c5bac1f166a5fd58e18e31db8048a` |
| 资源（Dev 333 pairs） | 加载 5.15 s；三组共 8.58 s；峰值额外显存 **3,520 MiB**；运行后释放；Qwen 服务未停 |

### Qwen3.5-9B Base Setwise（被阻塞）

- checkpoint：`/root/autodl-tmp/models/Qwen3.5-9B-ms`（Base，服务已加载）；
- 计划：固定 5 商品分组、BM25 顺序分组、组内排序、组冠军跨组比较，只报告真实 Top-1；
- 实际：**未产生可见输出**，状态 `BLOCKED_RESOURCE`，见 `runs/p14b3/setwise_status.json`。

## 2. 固定候选池与输入

复用 P14-B2 的 Dev 20 固定池（`runs/p14b3/pools/dev.jsonl`，来自 `runs/p14b2/pools`）：

- 同 Query、BM25 Top-20、Product Type/Role 过滤后 **111 个合法候选**（20 任务）；
- 所有已完成方法使用完全相同的候选集合、相同真实 catalogue title/text、相同 tie-break、相同评价标签（仅评估使用）；
- 类型违规 0，Hard violation 0；候选不可达 1（0046，所有方法都无法补救）。

Qwen3-Reranker 输入：

| 变体 | 指令 | Query |
| --- | --- | --- |
| A `query_only` | 官方默认 retrieval instruction | current query |
| B `query_prefs` | 官方默认 instruction | query + 合法 active preferences 摘要 |
| C `prefs_instruction` | 固定偏好重排指令（query 必须满足；正偏好提升；明确负偏好降低；UNKNOWN 不得视为满足；不推断未提供事实；撤回偏好不提供） | 同 B |

偏好摘要来自 M9/M5 合法读取；无历史任务不注入伪造偏好；撤回偏好不进输入（Train Smoke 0031 的 `noise_cancelling` 已撤回，active prefs 仅剩 wireless）。

## 3. 结果（Dev 20，recommend n=15）

| 方法 | Top-1 | Top-5 | MRR@10 | 偏好匹配/冲突/未知 Top-1 | Hard 违规 | 类型违规 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 6/15 | 14/15 | 0.611 | 5 / 2 / 17 | 0 | 0 |
| V0 Scorer | 11/15 | 14/15 | 0.833 | 9 / 1 / 14 | 0 | 0 |
| Qwen3-Reranker Query Only | 8/15 | 13/15 | 0.689 | 5 / 3 / 17 | 0 | 0 |
| Qwen3-Reranker + Preferences | 10/15 | 14/15 | 0.769 | 10 / 3 / 12 | 0 | 0 |
| **Qwen3-Reranker + Preference Instruction** | **12/15** | **14/15** | **0.856** | **12 / 1 / 11** | 0 | 0 |

- Qwen3 指令版相对 V0：**gained 0047、0054；lost 0053**，净 +1 Top-1；
- 约束遵从：两版 Qwen3 都未越过类型/硬约束（均已由 role-filtered pool 保证；Qwen 评分不是商品事实证据）；
- 候选不可达 0046：所有方法失败，单独报告为检索/候选可达性问题，不归因于 Reranker。

### 偏好消融（Qwen3-Reranker-4B）

| 对照 | Top-1 | MRR@10 | Top-1 改变 | 新正确 | 丢失正确 |
| --- | ---: | ---: | ---: | --- | --- |
| Query Only → +Preferences | 8 → 10 | 0.689 → 0.769 | 4 | 0047、0048 | — |
| +Preferences → +Preference Instruction | 10 → **12** | 0.769 → **0.856** | 5 | 0045、0049 | — |

**明确指令带来的独立收益是真实的**（非仅 Query 改写）：输入偏好相同，仅固定指令变化；0045 的 `in_ear` 负偏好从被违反（前两版）变为正确避开，0049 也恢复正确。与此同时 0053 仍失败：模型仍选择 wired gaming keyboard，尽管存在 `wired` 负偏好——说明指令是有效但不完全的偏好控制，不是可靠的硬约束。

## 4. 典型错误与 Same-Query Cross-Persona

- **0045**（正：over_ear/noise_cancelling；负：in_ear）：Query Only / +Prefs 把入耳/骨传导商品排前并违反负偏好；Instruction 版选 AIWA over-ear，Top-1 正确、正偏好 2、冲突 0。
- **0049**（正 wireless；负 over_ear）：Instruction 版选 Targus earbud，恢复正确。
- **0047**（wireless/ergonomic 鼠标）：+Prefs 起正确，Instruction 保持。
- **0048**（microphone/wireless speaker）：+Prefs 起正确。
- **0053**（负 wired 的 keyboard persona）：V0 正确；Qwen3 三组均将 wired gaming keyboard 排第 1/2，未遵守 `wired` 负偏好 → **指令仍会被通用相关性压过**。
- **0054**（偏好更新 in_ear/wireless）：Qwen3 三组 Top-1 正确；V0 失败。
- **0051**（same-query earbuds persona，正 rechargeable/microphone，负 over_ear）：所有方法都未排到 verified Top-1，属于排序难点，非类型/Gate 问题。
- **0046**：候选池仅 1 个且无 verified，四种方法全部失败，**候选不可达**。
- Same-Query Cross-Persona（0050-0053）：共享同一 query、不同 persona 的 active preferences 确实改变了 Qwen3 的排序（0053 等），但 0053 的负偏好仍未被完全遵守；0045/0049 说明偏好输入+指令的组合可以改变结果。
- 撤回偏好未用于输入（0031 验证）。

## 5. Setwise 真实阻塞记录（无结果、无伪造）

`runs/p14b3/setwise_status.json`：

- GPU：RTX 4090D 24,564 MiB；Qwen3.5-9B 服务占用 18,026 MiB；剩余 **6,178 MiB**；
- Qwen3.5-9B 4-bit 直接加载：allocator warmup 尝试 7.01 GiB → OOM；关闭 warmup 后权重阶段进程约 5.94 GiB，最后 96 MiB 分配失败（仅 94.25 MiB 空闲）；
- 4-bit CPU offload：accelerate/bitsandbytes dispatch 在 meta tensor 上失败；
- CPU bf16 9B 单任务（6 候选）：加载成功、16.8 GiB RSS、15 核，但 **约 13 分钟未产生一次生成**，不可用于 20 个任务；
- 未停止/重启 Qwen 服务或 AutoDL 实例（里程碑禁止）；没有 Setwise Top-1/Top-5/MRR/偏好指标，**不伪造**；
- 解除阻塞需要：额外 ≥8 GiB 空闲 GPU（含 activation/KV）、独立 GPU/实例，或用户明确授权临时卸载 Qwen 服务并恢复。

## 6. 成本

| | V0 | Qwen3-Reranker-4B |
| --- | --- | --- |
| 设备 | CPU | GPU（与 Qwen 服务共存） |
| 111 候选 | ~0.0013 s | 333 对 = 3 变体 × 111；8.58 s + 加载 5.15 s |
| 峰值额外显存 | 0 | 3,520 MiB |
| 模型 | 确定性特征加权 | 4B，4-bit NFM |
| 质量收益 | 11/15 | **12/15（+1）** |

Qwen3 的 +1 Top-1 收益伴随 4B 模型加载、量化与 GPU 共存成本；当前不足以替换 V0 默认。Setwise 的潜在收益无法评估，且资源需求更高（9B 4-bit 权重本身约 5.9 GiB，加上 KV/activation 已超过当前 6.18 GiB 余量）。

## 7. 报告问题回答

1. **Qwen3-Reranker-4B 是否超过 V0？** 在本次探索性 Dev 20 上**是**：12/15 vs 11/15 Top-1，MRR 0.856 vs 0.833；但仅 +1 条、样本小，不能视为稳定优势。
2. **偏好指令有没有独立收益？** **有**：Query+Prefs → +Instruction 提升 10→12 Top-1、MRR 0.769→0.856；5 个 Top-1 改变，+2 正确、0 丢失；负偏好冲突 3→1。它仍不能保证所有负偏好（0053 失败）。
3. **Setwise 是否比独立打分更好？** **无法回答**：Setwise 被 GPU 资源阻塞，没有真实结果；不做推断。
4. **哪种方案对负偏好最可靠？** V0 与 Qwen3 Instruction 的 Top-1 负偏好冲突均为 1；Qwen3 Query/ Prefs 为 3；Instruction 修复了 0045/0049，但仍输 0053。当前 **V0 更稳定、成本最低**；Instruction 是最有前景的神经方案但仍非硬约束。
5. **哪种方案成本更合理？** V0 显然最合理；Qwen3-4B 4-bit 约 3.5 GiB 额外显存、5 s 加载、数百 ms/任务，换来 +1 Top-1；Setwise 需要更多资源且无结果。
6. **是否应该替换 V0？** **否（当前）**。保留 V0 默认；Qwen3-Reranker + Instruction 作为可选候选，需要独立 Dev/更大标注集验证收益稳定性与负偏好可靠性。
7. **下一阶段是否值得偏好专项微调？** 暂不。偏好指令已有增益，但 0053/0046（不可达）显示瓶颈还包括负偏好覆盖、检索可达性和样本量；在获得独立验证集与更可靠正负监督前不做微调/训练。

## 8. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `QWEN3_RERANKER` | **PASS** | 真实 Qwen3-Reranker-4B 4-bit 推理完成；三变体真实分数；指令版 12/15 > V0 11/15；无伪造 |
| `SETWISE_RERANKER` | **FAIL** | 资源阻塞（9B 与 Qwen 服务共存显存不足，CPU 不可行）；无 Setwise 结果，不伪造 |
| `PREFERENCE_GAIN` | **PASS** | 相同输入下仅指令变化：Top-1 10→12、MRR 0.769→0.856；负偏好冲突 3→1 |
| `FAIR_COMPARISON` | **PASS**（已完成方法） | BM25/V0/Qwen3 三组同 Query/池/类型约束/证据/TopK/tie-break；Setwise 未运行，不比较 |
| **P14-B3** | **LIMITED** | 4B 实验和指令消融真实完成；Setwise 被真实 GPU 资源阻塞；尚不足以替换 V0，但 Qwen3 Instruction 值得作为下一轮可选候选验证 |

> Dev 20 已参与 P14-C0-R 开发，本结果只是探索性开发集诊断；Reserved Test 未运行；未训练/SFT/LTR；未修改 V0/Gate/Planner/Memory/标签；未 commit。

## 9. 产物

- `runs/p14b3/pools/{dev,train_smoke}.jsonl`（固定输入，无标签）
- `runs/p14b3/scores_qwen3reranker_{train_smoke,dev}.json`
- `runs/p14b3/setwise_status.json`（阻塞证据，无伪造）
- `runs/p14b3/train_smoke_summary.json`
- `runs/p14b3/comparison.json`、`per_task.jsonl`
- `runs/p14b3/model_manifest.json`（模型/权重哈希/资源/阻塞）
- `experiments/p14b3/{prepare_inputs,evaluate}.py`；`tests/test_p14b3_rerankers.py`
- 报告：`P14B3_REPORT.md`、`P14B3_SOURCE_DIFF.txt`
