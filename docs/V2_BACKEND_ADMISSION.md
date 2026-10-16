# V2 后端准入冻结（任务 03）

## 1. 准入结论
| 后端 | 工程服务验收 | 依据（冻结产物 ✓） | 总体推荐质量 | TIGER 融合资格 |
|---|---|---|---|---|
| **SASRec** | **PASS** ✓ | Gate C-1：真实 `/v1/recommend` 100/100 成功 + Schema 100/100 + 非法/PAD/重复/历史回显 **0** + **API↔离线 Top20 parity 100/100** + **两轮重复 100/100** ✓ | **待确认** ◐（exploratory n=100，无 sealed 复用 ✗） | — |
| **Two-Tower** | **PASS** ✓ | Gate C-2：同上全部 100/100 ✓（profile `serving_v1_two_tower_only`） | **待确认** ◐（同上；**无 sealed 全队列对照** ✗） | — |
| **TIGER** | **不准入** ✗ | Gate B-3 **PARTIAL 结项**：冻结 HR@20 = **0/100** ✓、复测 1/100（唯一命中 case 29 合法历史重复 ✓）、训练侧协议 **UNKNOWN** ✗、跨运行一致性未建立 ✗ | 不成立 ✗ | **禁止启用** ✗（不得加入融合/补位/降级 ✗） |

## 2. 冻结清单（版本哈希与配置）
```text
SASRec        : runs/sasrec_canonical_2026/best.pt  sha256 352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912 ✓
                hidden 64 / 2 blocks / max_seq_len 50 / params 10,135,104 ✓（与 run.json 一致 ✓）
Two-Tower     : runs/twotower_public_2026_full/checkpoint.pt  sha256 0e56be01…（完整值见 runs/gate_c12_service_validation_r2/manifest.json ✓）
                GRU / emb 128 / hidden 128 / max_seq_len 50 / params 40,551,297 ✓（含 metadata_categories.json ✓）
身份映射      : data/processed/Sports_and_Outdoors_mappings.json（1-based / PAD=0 / num_items 156,746 ✓）
评测队列      : data/processed/Sports_and_Outdoors_sequences.json（Gate B-1/B-2/C-1/C-2 同一 100 例同序 ✓）
协议          : agentrecx.eval_protocol.v1 / temporal_leave_two_out / seed 20261015 ✓
运行 Profile  : 缺省 serving_v1（未变 ✓）；新增只读 serving_v1_sequential_only / serving_v1_two_tower_only ✓
环境          : CPU-only（torch 2.1.2+cpu）、FastAPI/Starlette/httpx 见 manifest ✓
```

## 3. `serving_v1` 默认配置未变化的证据 ✓
* `FUSION_PROFILES["serving_v1"] == ("sequential","two_tower","catalog_lexical")` ✓（新 profile 为**纯新增** ✓）。
* 未设置 `AGENTRECX_FUSION_PROFILE` 时 ⇒ `FusionInferenceConfig.for_profile("serving_v1")` ✓（与旧默认构造等价 ✓）。
* 回归：`tests/test_serving_fusion.py` + `tests/test_api.py` → **36 passed** ✓；新 profile 隔离测试 → **4 passed** ✓。

## 4. 历史保留（未覆盖 ✓）
* Gate C-1/C-2 **PARTIAL 报告**（`docs/GATE_C1_SASREC_API_REPORT.md`、`GATE_C2_TWO_TOWER_API_REPORT.md`）与归档
  `runs/gate_c12_service_validation_r3/` **原样保留** ✓。
* **PASS 结果**（`docs/GATE_C12_R2_ACCEPTANCE.md` + `runs/gate_c12_service_validation_r2/`）**新增** ✓。

## 5. 版本管理清单（**未提交、未清理** ✓）
```text
已跟踪修改（3）: recommendation/api/app.py、recommendation/inference/fusion.py（本轮获批配置层补丁 ✓）、
                 docs/ARCHITECTURE.md（P0-CLOSE 授权文档修正 ✓）
新增未跟踪（本轮）: docs/V2_BACKEND_ADMISSION.md、V2_ROADMAP_ALIGNMENT.md、V2_MINIMAL_SERVICE_ACCEPTANCE.md、
                 V2_RESOURCE_RISKS.md、V2_TASK_03_05_DECISION.md、docs/GATE_C12_R2_ACCEPTANCE.md、
                 experiments/gate_c12_backend_acceptance.py、tests/test_v2_gate_c12_profiles.py、
                 tests/test_v2_task05_service_boundaries.py
（`runs/**` 全部由 .gitignore 排除 ✓，作为本地证据归档保留 ✓）
```
