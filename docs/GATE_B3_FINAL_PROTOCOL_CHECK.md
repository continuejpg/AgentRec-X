# Gate B-3 最终协议核对（训练 vs 推理）

> 冻结配置（跑分前固定、未调参 ✓，`runs/e04b_gate_b3_final_diag/final_diag_summary.json:frozen` ✓）：
> `mode=CERTIFIED`、`allow_mode_downgrade=false`、`beam=512`、`prefix_budget=200000`、`max_hist_items=20`、
> `k_max=20`、`k_requested=20`、`device=cpu`、同一 100 例同一顺序 ✓。

| 项 | 来源（file:line / 工件） | 实际值 | 判定 |
|---|---|---|---|
| Item ID ↔ SID 版本 | `sid/semantic_ids.json:checkpoint_sha256` ↔ checkpoint 记录 | 绑定一致（`d514aafcddf0edb9…` 前缀 ✓） | **MATCH** ✓ |
| SID 编码/解码 | `checkpoint/generator_layout.json`（code_space 1024、codebook 256、3 levels、`level_offsets [0,256,512,768]`） | 与 runtime SID 一致（`sid/layout.json` ✓） | **MATCH** ✓ |
| 特殊 token / EOS | `generator_layout.json:special_tokens`（pad 1024 / bos 1025 / eos 1026 / sep 1027）；`score_rule.json:eos_in_score` | 推理前缀 BOS+history+SEP ✓ | **MATCH**（配置级 ✓） |
| 推理前缀结构 | `serve_worker.py:41`、`generator_layout.json:derivation` | BOS + 历史 SID + SEP ✓ | **MATCH** ✓ |
| 历史排列顺序 | `evaluation/split.py`（chronological ✓）、`sequences.json` 自述 oldest-first ✓ | 升序 ✓ | **MATCH** ✓ |
| Seen Mask 规则 | `evaluation/metrics.py:179 excluded_seen`（正例从掩码中移除 ✓） | case 29 实测：`target_in_mask_used_for_ranking=false` ✓ | **MATCH** ✓ |
| 评分与 Rank 定义 | `tiger_engine.py:811-816,848`（1-based、唯一、按 rank 排序 ✓） | 100 例零契约错误 ✓ | **MATCH** ✓ |
| **训练历史窗口长度** | 训练源码/配置**不在本仓** ✗（`checkpoint/tiger.json` 无序列长度键 ✗） | `max_hist_items=20` 为原生默认 ✓ | **UNKNOWN** ✗ |
| **训练预测目标/偏移** | 同上 ✗ | 未证实 ✗ | **UNKNOWN** ✗ |
| 训练截断方向 | 同上 ✗（推理侧为"取最新 N 项"✓） | 未证实 ✗ | **UNKNOWN** ✗ |
| train/val/test 与 checkpoint 匹配 | `run.json`（cohort 412,445 / `sampled:false` ✓）与协议同源 ✓ | 与冻结协议一致 ✓ | **MATCH**（记录级 ✓） |

**结论上限**：推理侧契约（SID/前缀/顺序/掩码/排名）已逐项核对为 **MATCH** ✓；
**训练侧三项为 UNKNOWN** ✗ ⇒ **不得**声称"训练与推理协议一致" ✗（这正是本轮结论被封顶的原因之一 ✓）。
