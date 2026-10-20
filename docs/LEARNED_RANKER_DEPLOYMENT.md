# Learned Ranker V1 — 部署与运维说明（实验 profile）

> 状态：**实验路径**（`serving_v1_learned_ranker`）。默认 `serving_v1` 不使用本模块，也不需要安装 LightGBM。
> 算法质量：**NOT ESTABLISHED**（A6 post-hoc 对照中 RRF 与 LightGBM 无稳定优劣）。本文件不构成上线批准。

## 1. 组件
| 组件 | 路径 | SHA256 |
|---|---|---|
| 模型 | `runs/lr_v1_formal/model.txt` (42010 B) | `c1b37d4a2afe9187000c73bc7cb7bb22cc811dda3a2fe9bfe2118fa513db6eb9` |
| 特征 schema | `runs/lr_v1_formal/feature_schema.json` | `ad1d0be168dc9f183c45683c26e24a0086e882632570e2feffa8f9971598a422` |
| 训练清单 | `runs/lr_v1_formal/train_manifest.json` | 记录 C0 参数、best_iteration=22、22 棵树 |
| 训练数据（JSONL） | `runs/lr_v1_r2/rlr01_groups.jsonl` (5,000 组 / 1,366,080 行) | `371a549c2246d7616d2ac0eb2abc4dada94c31274e8e2a0531f7b1591b01ca55` |

## 2. 依赖（仅该 profile 需要）
```bash
python3.10 -m venv .venv-ranker
.venv-ranker/bin/pip install -r requirements-serving-ranker.txt   # lightgbm==4.7.0, numpy==1.26.4
```
已在全新 venv 中实测通过（Python 3.10.12 / NumPy 1.26.4 / LightGBM 4.7.0，模型加载 + 15 维预测成功）。

## 3. 配置
| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `AGENTRECX_FUSION_PROFILE` | `serving_v1` | 设为 `serving_v1_learned_ranker` 启用精排 |
| `AGENTRECX_RANKER_MODEL` | `runs/lr_v1_formal/model.txt` | 模型文件路径 |
| `AGENTRECX_RANKER_SCHEMA` | `runs/lr_v1_formal/feature_schema.json` | 特征顺序/量化协议 |

## 4. 启动
```bash
AGENTRECX_FUSION_PROFILE=serving_v1_learned_ranker \
AGENTRECX_RANKER_MODEL=/path/to/model.txt \
AGENTRECX_RANKER_SCHEMA=/path/to/feature_schema.json \
uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8000
```
默认（无 LightGBM）：
```bash
uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8000    # serving_v1，不需要 lightgbm
```

## 5. 校验与恢复
```bash
sha256sum model.txt feature_schema.json     # 必须与第 1 节一致
```
`runs/` 被 `.gitignore` 忽略，因此 **clone 仓库不会带来模型**：权重必须随发布包单独分发（或对象存储），
并用上面的 SHA256 校验。缺失/损坏时的行为：`LearnedRanker.load()` 记录 `load_error`，
每个请求走 **full-precision RRF 回退** 并累加 `fallback_count`（不回退绕过身份校验）。

## 6. 运行特征与验收
- 特征构造：`recommendation/inference/learned_ranker.py::build_feature_rows`（离线/在线**同一实现**）；
- 排序：`ranker_score DESC → rrf_score(6dp) DESC → item_id ASC`；
- 实测：15/15 特征与冻结训练数据逐值一致（20 例 / 5,420 行）；真实 API 200×20、零违规、`fallback_count=0`；
- 默认 profile 100 例与捕获态 RRF oracle **100/100** 有序一致，且默认路径**未导入** lightgbm。

## 7. 独立模型归档（2026-10-20）

| 项 | 值 |
|---|---|
| 归档文件 | `/mnt/d/IT/CODE/PROJECTION/agentrec/backups/lr_v1_learned_ranker_20261020.tar.gz`（21928 B） |
| 归档 SHA256 | `c751c641fafef7737d9b9a9df90861359ce0d5cfc7e31a91668be994ca68ee43` |
| 内容（6 成员） | `model.txt` · `feature_schema.json` · `train_manifest.json` · `metrics.json` · `A6_authoritative.json` · `SHA256SUMS` |
| 恢复 | `tar -xzf lr_v1_learned_ranker_20261020.tar.gz -C <target> && cd <target> && sha256sum -c SHA256SUMS` |
| 完整性实测 | 解压成功 ✓；内部 5 个文件 `sha256sum -c` 全部 OK ✓；归档 sha256 已记录 ✓ |
| 持久化等级 | ⚠️ 位于**仓库之外**（未入 Git ✓）但**与仓库同一文件系统** ⇒ 仅“本地归档” ✗；尚无独立外部存储，**不宣称异地可靠交付** ✗ |
