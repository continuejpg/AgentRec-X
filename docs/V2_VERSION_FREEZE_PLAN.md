# V2 版本冻结与提交方案（**待授权执行**，冻结 2026-10-16 07:15）

## 1. 当前 Git 事实

```text
branch : v2/e04a-pass-checkpoint
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54
已跟踪修改（3）: `docs/ARCHITECTURE.md`, `recommendation/api/app.py`, `recommendation/inference/fusion.py`
未跟踪文件: 44 个（docs/experiments/tests 新增）
被忽略     : runs/ 下 495 个文件（实验原始响应与归档）
```

## 2. 建议提交范围（白名单，禁止 `git add -A`）

**必须纳入**：
* 生产源码（本轮获批的最小改动）：`recommendation/inference/fusion.py`（+2 个单源 profile 与 `UnknownCatalogueIdentity`）、
  `recommendation/api/app.py`（profile 选择 + 422 映射 + 模块级 import）；
* 文档：`docs/ARCHITECTURE.md`（状态标记修正）+ 本批次全部 `docs/V2_*.md`、`docs/P3_*.md`、`docs/GATE_*.md`、`docs/V2_E04B*.md`；
* 测试：`tests/test_v2_*.py`、`tests/test_v2_task05_service_boundaries.py`；
* 实验驱动：`experiments/e04b_*.py`、`experiments/gate_c12_backend_acceptance.py`。

**不得纳入**：`runs/**`（模型权重、1,000 例原始响应、Gate 归档）—— 由 `.gitignore` 排除，须**独立持久化**保存（见 §4）。

## 3. 提交顺序建议（4 个原子提交，均需授权）

1. `feat(serving): single-source profiles + unknown-identity 422 mapping`（两个源码文件 + profile/边界测试）
2. `test(v2): C-1/C-2 service acceptance drivers and boundary suite`（experiments + tests）
3. `docs(v2): gate C, task 03-05 and P3 evaluation records`（全部文档）
4. `chore(docs): architecture plane status markers`（`docs/ARCHITECTURE.md` 单独提交，便于回溯）

提交信息须写明：**工程验收 PASS 但模型质量仍为 exploratory**、TIGER 不准入、两项既有失败与 5xx 覆盖缺口。

## 4. 大资产独立持久化（不入 Git）

```bash
# 例：仓库外归档（路径需按你的持久化策略确认；本轮未执行）
tar -czf /path/to/persistent/agentrecx_v2_evidence_$(date +%Y%m%d).tar.gz \
  runs/p3_unified runs/p3_unified_r2 runs/p3_unified_fusion runs/p3_unified_analysis \
  runs/gate_c12_service_validation_r2 runs/e04b_gate_b1_sasrec runs/e04b_gate_b2_twotower runs/e04b_gate_b3_*
```
模型权重（`best.pt` 121 MB、`checkpoint.pt` 162 MB、TIGER 45 MB + 特征 481 MB）**保持原位、不得删除或移动**；
其哈希已记录于 `V2_ASSET_MANIFEST.json`，可据此校验未变。

## 5. 校验建议

```bash
sha256sum runs/sasrec_canonical_2026/best.pt runs/twotower_public_2026_full/checkpoint.pt
python -c "import json;print(json.load(open('docs/V2_ASSET_MANIFEST.json'))['frozen_at'])"
```

## 6. 明确不做

* 不执行 `git commit/reset/clean`（**未获授权**）；不删除历史模型、Checkpoint、Gate 报告或原始记录；
* 不把 `runs/**` 提交进 Git（避免仓库膨胀与凭据/大数据风险）。
