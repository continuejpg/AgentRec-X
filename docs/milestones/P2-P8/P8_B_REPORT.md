# P8-B 真实 BGE-base 语义检索与 BM25 Hybrid 报告

## 0. 结论

**真实 BGE-base-en-v1.5 商品索引已在 RTX 4090D 上构建完成，并完成 BM25-only / BGE-only / Hybrid 三模式与真实 Agent opt-in 验证。**

| 项 | 状态 |
|---|---|
| 真实 BGE-base 模型 | **VERIFIED**：revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| 156,746 商品 Embedding | **VERIFIED**：156,746 × 768 float32，全部归一化 |
| 商品/索引/catalogue 对齐 | **VERIFIED**：严格 identity + catalogue fingerprint 校验 |
| BM25-only | **VERIFIED**：默认 Agent 路径，未加载 BGE |
| BGE-only | **VERIFIED**：真实 BGE 查询返回真实 catalogue ASIN |
| Hybrid | **VERIFIED**：真实 BM25 + BGE RRF，matched_fields 含两来源 |
| 真实 Agent opt-in Hybrid | **VERIFIED**：BGE 被实际调用，final provenance `score_kind=hybrid_bm25_bge`，最终商品 role 均 MATCH |
| CPU 性能/生产化 | **LIMITED**：冷启动与首次查询延迟较高；NumPy exact retrieval 可运行但不适合低延迟生产 |
| Source Policy 开发条件 | **LIMITED / 可进入受控开发**：BGE/Hybrid 通道可用，但 latency/环境封装需后续处理 |

本轮没有使用 fake-dense、hashed fallback 或仅源码描述冒充 BGE。

## 1. 旧 BGE 清理

| 文件/assets | 处理 |
|---|---|
| `recommendation/inference/dense_catalog.py` 旧草稿 | **替换**为 BGE-base-only v3 实现 |
| 旧 hashed TF-IDF fallback | **删除** |
| 旧 BGE-small 384 维假设 | **删除** |
| `tests/test_dense_catalog.py` 旧 hashed/fake-dense 测试 | **删除** |
| `recommendation/control/catalog_search.py` 未完成 Hybrid | **保留核心 RRF 并重构**为显式 bm25/bge/hybrid 三模式 |
| `docs/DEMO_ACCEPTANCE.md` BGE 文档 | 更新为 BGE-base 768 / v3 / 模式开关 |
| `experiments/build_bge_index.py` | **新增**真实索引构建脚本 |

未删除 SASRec、Two-Tower、Qwen、用户历史、catalogue、现有实验产物或其他模块的未提交改动。

## 2. 真实模型与构建环境

| 项 | 值 |
|---|---|
| Model ID | `BAAI/bge-base-en-v1.5` |
| Revision | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| Embedding dim | 768 |
| Query instruction | `Represent this sentence for searching relevant passages: ` |
| Document instruction | 无 |
| Similarity | normalized dot product / cosine |
| GPU | RTX 4090D, 24GB |
| 远端构建环境 | `/root/autodl-tmp/bge_env`（独立 venv；sentence-transformers 6.2.0；torch 2.11.0+cu128） |
| 本地验证环境 | 项目 Python + `runs/p8_bge/py_deps` 依赖 overlay；torch 2.5.1 / transformers 4.38.2 / sentence-transformers 2.7.0 |
| 模型本地路径 | `models/bge-base-en-v1.5` |

模型从官方 mirror `hf-mirror.com` 下载；直连 HuggingFace 超时。模型文件与 revision 已固定。

## 3. 商品文本与索引

- 商品文本来自当前 catalogue 的 `title + store + main_category + categories + features + description`；
- 无用户历史、目标标签、测试答案、价格或 role 判断；
- 去空值/重复字段，截断 2048 字符；BGE `max_seq_length=512`；
- 156,746 个 `parent_asin` 全部有有效文本，`empty_text=0`；
- 无重复 identity；
- 向量 float32、768 维、`normalize_embeddings=True`；
- 加载时校验每个向量 finite 且 L2 norm≈1；
- 156,746 × 768 × 4 = **481,523,712 bytes = 459.2 MiB** 纯向量；
- 实际 NPZ 文件 **844,552,356 bytes ≈ 805.4 MiB**（含 titles/identities/zip 开销）。

索引 metadata：

```text
index_format=agentrecx.dense_catalog.v3
model_id=BAAI/bge-base-en-v1.5
model_revision=a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
embedding_dim=768
normalize_embeddings=True
similarity=cosine_dot_product
text_build_version=agentrecx.catalog_text.v1
item_count=156746
text_max_chars=2048
max_seq_length=512
catalogue_fingerprint=7104ef901d8eb4ab799a5f419902ff7dad0ce7ec2aac8a86bdd2d07ddc85b9af
identity_fingerprint=9d7c73e12d40541bc0d6ba962464ba88014d7e0bfe5d7a3c100ae38fa0c92470
mapping_fingerprint=dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912
built_at=2026-11-07T14:04:32.885072+00:00
```

**索引文件**：`runs/p8_bge/bge_base_dense.npz`  
**sidecar**：`runs/p8_bge/bge_base_dense.npz.json`

构建命令（远端）：

```bash
/root/autodl-tmp/bge_env/bin/python build_bge_index.py \
  --catalogue .../Sports_and_Outdoors_products.jsonl \
  --mapping .../Sports_and_Outdoors_mappings.json \
  --model /root/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --model-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --out .../bge_base_dense.npz \
  --device cuda --batch-size 128
```

构建结果：156,746 records, empty_text=0, GPU 编码成功，无半成品被当作有效索引。

## 4. catalogue/索引严格对齐

加载时必须同时通过：

- 索引内部 identity/vector 对齐、维度、唯一性、有限值、归一化；
- 当前 catalogue 的 `parent_asin` 集合与索引 identities **完全一致**；
- 当前 catalogue 的 **text fingerprint** 与索引 metadata 完全一致；
- 若有当前 mapping，mapping SHA 与索引 metadata 一致。

验证：

- 真实当前 catalogue 加载成功；
- 删除/替换一个 identity 的 stale catalogue 校验抛出 `ValueError`；
- BGE build 缺 model_path 抛 `ValueError`，不允许 hashed fallback；
- `DenseCatalogIndex.load` 在模型缺失时 fail-fast。

## 5. 三模式真实检索结果

真实索引上执行 4 个 query：

- `camping cookware`
- `water filter`
- `filtered water bottle`
- `hiking gear`

完整候选/标题/来源/耗时见 `runs/p8_bge/validation.json`。代表性 top-5：

| Query | BM25-only | BGE-only | Hybrid |
|---|---|---|---|
| camping cookware | `B01985FDA4, B00WDQIL7I, B06WP1W3TQ...` | `B08R5XNLML, B07S8GNB4T, B0BQQ3Q88M...` | `B07ZQ65TTM, B00YA2HRAK, B085WZ3BKT...` |
| water filter | `B0B6Z7PTZ8, B00STSIHXC, B09CJN4XGM...` | `B08XX786P1, B0BC89C4QS, B00FUANH0U...` | `B08CX86STF, B099Z9641J, B09GK41P6M...` |
| filtered water bottle | `B07BSY9YYX, B08D1PYPYS, B0BNL69LSS...` | `B0987D9PKM, B0BFR4VJTW, B00FUANH0U...` | `B08D1PYPYS, B0BFR4VJTW, B0987D9PKM...` |
| hiking gear | `B006J3EFM6, B07FK7WYR5, B01FZBZWEC...` | `B09NBWD57V, B0879HB6ZR, B0C57XGJP5...` | `B09NBWD57V, B006J3EFM6, B0879HB6ZR...` |

- 所有模式返回的 ASIN 均属于当前 catalogue；
- BGE-only hit 的 `matched_fields=("bge_dense",)`；
- Hybrid hit 的 `matched_fields` 为 `("bm25","bge_dense")` 或单来源，证明实际融合；
- Hybrid 使用 rank-based reciprocal-rank fusion，不直接混合未校准 BM25/cosine 分数；
- source-level raw counts 已在三模式输出中分别记录。

### 结果质量说明

- `camping cookware` 的 BGE-only top-5 全部 role-MATCH；
- `water filter` / `filtered water bottle` 有 MATCH/MISMATCH/UNKNOWN 混合，符合 role gate 仍需执行的预期；
- `hiking gear` 的 role 目标本身不明确，BGE 返回 UNKNOWN 属于合理证据不足；
- 不把“更多 MATCH”或更高 cosine 直接声明为相关性提升；这里只证明通道可用。

## 6. CPU 性能与内存

| 指标 | 实测 |
|---|---:|
| 索引加载 | **41.0 s**（本地 CPU，含模型加载） |
| 进程 RSS 增量 | **+1073.7 MiB**（含模型 + 索引；纯向量 459.2 MiB） |
| BGE-only 冷查询 | 20.7 s |
| BGE-only 热查询 | 1.17–2.80 s |
| BM25-only 查询 | 1.48–8.45 s |
| Hybrid 查询 | 1.85–14.76 s（首次较慢，热查询约 1.85–4.03 s） |
| 真实 Agent Hybrid 请求 | 113.6 s（含整个 Agent 运行，不只是 BGE） |

结论：**可运行但 CPU latency 是明显限制**；NumPy exact retrieval 内存/正确性可接受，冷启动和热查询延迟不足以直接称为生产级低延迟。未加 FAISS，因为当前 NumPy 已能完成真实验证；生产化前再根据 SLA 评估 ANN。

## 7. 真实 Agent opt-in 验证

配置：

```text
AGENTRECX_CATALOG_SEARCH_MODE=hybrid
AGENTRECX_DENSE_INDEX=runs/p8_bge/bge_base_dense.npz
AGENTRECX_BGE_MODEL_PATH=models/bge-base-en-v1.5
AGENTRECX_MAPPINGS_PATH=data/processed/Sports_and_Outdoors_mappings.json
```

真实 Agent request：

```text
I need a water filter under $40 for camping.
```

结果：

- `catalog_search.mode = hybrid`
- `catalog_search.score_kind = hybrid_bm25_bge`
- Agent source calls：`history=1, catalog_search=1`
- BGE 被实际调用（dense wrapper 计数；验证脚本的 `dense_calls` 包含后续 cross-check）
- grounded candidates 15 个，provenance `score_kind=hybrid_bm25_bge`
- source query：`water filter under I need a water filter under $40`
- 最终推荐：`B08XJQ9HPX`, `B07L47J44Q`
- 两个最终商品均为 role-MATCH，`water_filter/main`
- BGE-only top-10 与该 Agent 的 grounded 池有 10 个交集，证明 Hybrid 的 BGE 候选实际进入 CandidateLedger 可见候选
- final recommendations 与 catalogue identity 一致

默认模式验证：

```text
catalog_score_kind = bm25
catalog_search.mode = bm25
dense_index = None
```

即默认 Agent 不加载 BGE 模型或索引。

## 8. 最小 fail-fast / 回归

已验证：

- 缺少 BGE 模型时 build 抛 `ValueError`，不会 hashed fallback；
- 当前 catalogue 与索引 identity/text 不匹配时 load 抛 `ValueError`；
- Hybrid 中未知 ASIN 不进入合法候选；
- BM25-only 默认路径不加载 dense；
- Hybrid 实际合并 BM25 + BGE（matched_fields 双来源）。

补充：`tests/test_control_stage2_candidates.py` 25 passed。旧 `tests/test_dense_catalog.py` 已删除（其 hashed/fake-dense 验收不再适用）。

## 9. 五个交付问题的直接回答

### 1) 旧 BGE 文件如何处理？

- 旧 `dense_catalog.py` 草稿：**替换**为真实 BGE-base v3；
- hashed fallback / BGE-small 384：**删除**；
- 旧 `tests/test_dense_catalog.py`：**删除**；
- `catalog_search.py` Hybrid：保留 RRF 思路，**重构**成显式三模式；
- 其他模块/未提交改动：保留。

### 2) 最终模型、revision、环境？

`BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`；  
远端 RTX 4090D + 独立 `/root/autodl-tmp/bge_env`（ST 6.2.0 / torch 2.11 CUDA）；本地验证依赖 overlay。

### 3) Embedding 与索引是否完整？

**是。** 156,746/156,746 商品，768 维，float32，normalized；无空文本、重复 identity、NaN/Inf；GPU 构建成功；索引 805.4 MiB。

### 4) 索引是否与当前 catalogue 严格匹配？

**是。** 加载时校验 identity 集合、catalogue text fingerprint、mapping SHA；stale 测试抛错。

### 5) 三模式是否真实可用？

- BM25-only：**VERIFIED**
- BGE-only：**VERIFIED**
- Hybrid：**VERIFIED**
  
真实 Agent：**Hybrid opt-in VERIFIED**；BGE 实际调用，provenance 可区分。

### 6) 真实 Agent 是否确实调用 BGE？

**是。** `source_calls.catalog_search=1`、`dense_calls>0`、grounded provenance `score_kind=hybrid_bm25_bge`、BGE top-10 与 grounded 池有 10 个交集；最终两个推荐均 role-MATCH。

### 7) CPU 延迟和内存是否可接受？

- 正确性/内存：可接受；纯向量 459.2 MiB，进程 RSS 增量约 1.07 GiB；
- CPU 延迟：**LIMITED**；索引加载 41s，BGE 冷查询 20.7s、热查询约 1.2–2.8s；Hybrid 热查询约 1.9–4.0s。
- 不足以宣称生产低延迟；需要时再评估 FAISS/常驻服务/GPU query encoder。

### 8) 真实阻塞/质量限制？

- CPU 冷启动与热查询延迟；
- 本地验证依赖 overlay，不是干净正式 venv；远端已有独立 BGE env；
- 未做 BGE relevance/满意度的离线或人工质量评测；只证明真实检索与身份/角色链路可用；
- 未修改 Planner Source Policy，因此还没有“动态来源选择”的行为结论。

### 9) 是否具备下一步 Source Policy 条件？

**LIMITED GO：可以进入受控的 P8 Source Policy 开发**，因为真实 BGE/BGE-only/Hybrid 和 Agent opt-in 已打通；  
但必须保留 opt-in、先接受当前 CPU latency 限制，并只做来源选择/融合策略，不重新审计 BGE 资产。

## 10. 关键文件

- 源码 diff：`P8_B_SOURCE_DIFF.txt`
- 索引构建：`experiments/build_bge_index.py`
- 索引：`runs/p8_bge/bge_base_dense.npz`
- 索引元数据：`runs/p8_bge/bge_base_dense.npz.json`
- 验证结果：`runs/p8_bge/validation.json`
- 模型：`models/bge-base-en-v1.5`
- 本地验证依赖 overlay：`runs/p8_bge/py_deps`（不提交 Git）

本轮未训练、未运行 Holdout、未修改 Gold/Evaluator、Serving V1 默认行为、M5/Eligibility/Role Gate、Planner Contract；未 commit/push/deploy；GPU 实例未关闭（已停止 Qwen planner server 释放显存，未操作实例）。
