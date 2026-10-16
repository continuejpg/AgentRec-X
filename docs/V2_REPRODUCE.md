# V2 最小可复现系统说明（冻结 2026-10-16 07:15）

## 环境（已实测）

```text
Python 3.10.12 · torch 2.1.2+cpu（CUDA 不可用） · FastAPI/Starlette + httpx TestClient
解释器：/mnt/d/IT/CODE/PROJECTION/agentrec/X/.venv/bin/python
工作目录：/mnt/d/IT/CODE/PROJECTION/agentrec/X
```

## 模型资产与哈希

| 资产 | 字节 | SHA256（前 16 位） |
|---|---|---|
| runs/sasrec_canonical_2026/best.pt | 121,675,470 | 352bd3ae7ebc5e20 |
| runs/twotower_public_2026_full/checkpoint.pt | 162,221,222 | 0e56be01f872ed0c |
| runs/twotower_public_2026_full/metadata_categories.json | 2,741,318 | 46469955655b8c55 |
| data/processed/Sports_and_Outdoors_mappings.json | 见 manifest | 见 V2_ASSET_MANIFEST.json |
| data/processed/Sports_and_Outdoors_sequences.json | 见 manifest | 见 V2_ASSET_MANIFEST.json |

## API 启动与 Profile 选择

```bash
export AGENTRECX_FUSION_PROFILE=serving_v1_sequential_only   # 或 serving_v1_two_tower_only / 缺省 serving_v1
python -m uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8000
```

未设置该变量时为 `serving_v1`（三源融合）；未知名会在启动时**直接失败**（fail-closed）。

## 请求 / 响应示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/recommend -H 'content-type: application/json' \
  -d '{"history": ["B001U9YK8I", "B07GYZSV2"], "k": 10}'
```

```json
{"recommendations": [{"rank": 1, "item_id": 12345, "parent_asin": "B0...", "score": 1.23}],
 "requested_k": 10, "returned_k": 10, "history_length": 2}
```

未知商品身份 → **422** `{"error": "unknown_item", "detail": "..."}`；非法 K / 空历史 / 类型错误 → **422**。

## 回归测试（已实跑）

```bash
python -m pytest tests/test_v2_task05_service_boundaries.py -q      # 17 passed
python -m pytest tests/test_serving_fusion.py tests/test_api.py -q  # 默认行为未回归
python -m pytest tests/test_v2_gate_c12_profiles.py -q             # 4 passed（profile 隔离）
```

## 复算已归档 P3 指标

```bash
# 逐例原始响应在 runs/p3_unified*/*/responses_run{1,2}.jsonl（各 1,000 行，两轮独立）
python -m experiments.gate_c12_backend_acceptance --backend sasrec --cases 1000 --out runs/p3_recompute
```
冻结统计文件：`runs/p3_unified_analysis/P3_UNIFIED_METRICS.json`、`P3_PAIRED_BOOTSTRAP.json`、`P3_COMPLEMENTARITY.json`。

⚠️ **未验证项（不虚构）**：`uvicorn` 真实监听端口的启动命令本轮**未实跑**（验收使用进程内 ASGI TestClient）；
并发压测、生产内存门槛、`AGENTRECX_FUSION_PROFILE` 在多 worker 部署下的行为均**未验证**。
