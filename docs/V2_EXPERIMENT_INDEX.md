# V2 实验资产索引（冻结 2026-10-16 07:15）

| 类别 | 路径 | 说明 |
|---|---|---|
| 资产审计 | docs/V2_E04B_REAL_ASSET_AUDIT.md | E-04B-1 四后端状态矩阵 |
| 协议冻结 | docs/V2_E04B_DUAL_EVALUATION_PROTOCOL.md + experiments/e04b_protocol_v2.py | 双层评价协议（机读+文档） |
| Gate A | docs/V2_E04B2_GATE_A_REPORT.md + experiments/e04b2_gate_a.py + runs/e04b2_gate_a/ | 真实 BM25 与协议自检 |
| P0 | docs/V2_E04B_P0_REPORT.md + experiments/e04b_p0_rule_baseline.py + runs/e04b_p0* | B2 规则行为基线 |
| Gate B-1 | docs/V2_E04B_GATE_B1_SASREC_REPORT.md + experiments/e04b_gate_b1_sasrec.py + runs/e04b_gate_b1_sasrec/ | SASRec 离线 100 例 |
| Gate B-2 | docs/V2_E04B_GATE_B2_TWOTOWER_REPORT.md + experiments/e04b_gate_b2_twotower.py + runs/e04b_gate_b2_twotower/ | Two-Tower 离线 100 例 |
| Gate B-3 | docs/V2_E04B_GATE_B3_TIGER_REPORT.md、GATE_B3_*.md + experiments/e04b_gate_b3_tiger.py、e04b_gate_b3_final_diag.py + runs/e04b_gate_b3_* | TIGER 真实生成 + 结项（PARTIAL） |
| Gate C-1/C-2 | docs/GATE_C1_SASREC_API_REPORT.md、GATE_C2_TWO_TOWER_API_REPORT.md、GATE_C12_R2_ACCEPTANCE.md、GATE_C12_SERVICE_COMPARISON.md、GATE_C12_FINAL_DECISION.md + experiments/gate_c12_backend_acceptance.py | 单后端真实 API 验收（R2 PASS） |
| 任务 03–05 | docs/V2_BACKEND_ADMISSION.md、V2_ROADMAP_ALIGNMENT.md、V2_MINIMAL_SERVICE_ACCEPTANCE.md、V2_RESOURCE_RISKS.md、V2_TASK_03_05_DECISION.md、V2_TASK05_ERROR_MAPPING_REPORT.md | 准入冻结 / 链路确认 / 边界验收 / 错误映射修复 |
| P3 | docs/P3_EVALUATION_PROTOCOL.md、P3_BACKEND_COMPLEMENTARITY.md、P3_FINAL_DECISION.md + runs/p3_unified*/ + runs/p3_unified_analysis/ | 三路径统一质量评测（n=1,000） |

## 关键归档大小与哈希（真实核对）

| 文件 | 字节 | SHA256（前 16 位） |
|---|---|---|
| runs/p3_unified/sasrec/responses_run1.jsonl | 1,176,474 | 见 v2_freeze_facts.json |
| runs/p3_unified_r2/two_tower/responses_run1.jsonl | 1,174,533 | 见 v2_freeze_facts.json |
| runs/p3_unified_fusion/fusion/responses_run1.jsonl | 1,186,813 | 见 v2_freeze_facts.json |
| runs/gate_c12_service_validation_r2/sasrec_summary.json | 见 manifest | 见 manifest |

全部逐例记录的 SHA256 计算依据存放于 `runs/p3_unified_analysis/v2_freeze_facts.json`。
