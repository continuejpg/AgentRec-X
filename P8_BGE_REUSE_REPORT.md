# P8 BGE 资产复核与最小复用报告

## 0. 结论

**旧 BGE 只有未提交的源码/测试草稿；没有模型权重、没有商品 Embedding、没有 BGE 索引，也没有真实 Agent 调用。本次实际完成了 1 个关键安全修复：禁止 BGE 请求静默退化为 hashed lexical fallback。BGE 本身仍未可用，P8 Source Policy 尚不具备条件。**

| 项 | 状态 |
|---|---|
| 旧 BGE 源码 | 存在，但为 untracked WIP |
| Hybrid BM25+dense 源码 | 存在，但为 uncommitted WIP |
| BGE 模型权重 | **缺失 / UNVERIFIED** |
| 商品 Embedding / BGE index | **缺失 / UNVERIFIED** |
| `sentence-transformers` | 项目相关环境均未安装 |
| 当前 Agent 实际调用 BGE | **否，VERIFIED** |
| BM25 | VERIFIED 默认可用 |
| BGE | UNVERIFIED / 不可用 |
| Hybrid | 代码存在，LIMITED；真实 BGE 端到端未验证 |

## 1. 找到的旧 BGE 资产

### 1.1 源码

| 文件 | 内容 | 状态 |
|---|---|---|
| `recommendation/inference/dense_catalog.py` | `DenseCatalogIndex`：catalogue 文本 dense 向量、BGE-small-en-v1.5 路径、hashed fallback、v2 `.npz` 格式 | **untracked** |
| `recommendation/control/catalog_search.py` | `CatalogSearchSource` 的 opt-in dense/hybrid 融合 | **modified, uncommitted** |
| `tests/test_dense_catalog.py` | hashed index + fake dense hybrid 单元测试 | **untracked** |
| `docs/DEMO_ACCEPTANCE.md` | 预期可选 BGE artifact：`agentrecx.dense_catalog.v2`，156,746 × 384 | 文档存在 |

`git status` 证据：

```text
M recommendation/control/catalog_search.py
?? recommendation/inference/dense_catalog.py
?? tests/test_dense_catalog.py
```

`git log -- recommendation/inference/dense_catalog.py ...` 没有 BGE/dense 的提交历史；只能确认它是工作区里的实现，不能确认过去曾经作为 accepted 功能运行过。

### 1.2 用途

`dense_catalog.py` 是 **Query-to-Product catalogue dense retrieval**：

```text
catalogue title/categories/features/description/store
    -> document embedding
user query
    -> query embedding
cosine dot product -> parent_asin ranks
```

它不是 Item-to-Item 内容检索、不是推荐模型文本特征、不是 reranker。索引是 **精确 NumPy**，不是 FAISS/ANN；`CatalogSearchSource` 在 opt-in 时把 BM25 与 dense 用 RRF（K=60）融合。

### 1.3 实际资产缺失

检查本地、项目目录、远程 GPU 和 HuggingFace cache：

- `models--BAAI--bge-small-en-v1.5` 只有 `refs/main`，**没有 snapshots / blobs / 权重文件**；
- 没有 `bge_dense.npz`、dense embedding、FAISS index 或等价向量文件；
- `sentence-transformers` 在项目可用环境、Conda 环境、远端 GPU 环境中均不可 import；
- 没有 BGE index 构建脚本或已保存的构建产物；
- 远端 `/root/autodl-tmp` 也没有 BGE/dense 资产。

因此当前无法执行真实 BGE 推理。

## 2. 旧实现的关键问题

### 2.1 已确认并修复：BGE 请求静默退化为 hashed fallback

原 `DenseCatalogIndex.build`：

```python
if encoder_kind == "bge" and model_path:
    ... build BGE ...
# 否则继续
vectors = np.zeros(...)
...
return cls(..., encoder_kind="hashed_tfidf")
```

调用 `build(..., encoder_kind="bge", model_path=None)` 不会报错，而是静默生成 **hashed TF-IDF** 索引。虽然 artifact 的 `encoder_kind` 会标成 `hashed_tfidf`，但对于 BGE 请求这是不可靠的配置行为，容易被误当成 BGE 资产。

已修复为：

```python
if encoder_kind == "bge":
    if not model_path:
        raise ValueError(
            "building a BGE index requires an explicit local model_path; "
            "refusing to substitute the hashed fallback"
        )
```

对应回归测试：

```text
tests/test_dense_catalog.py::test_bge_build_does_not_silently_fall_back_to_hashed_encoder
```

结果：`4 passed`。

### 2.2 尚未解决的可靠性风险

1. **模型/索引完全缺失**：无法验证 BGE-small-en-v1.5 的 pooling、normalization、维度与 query/document encoding；当前只能代码审查，运行时 **UNVERIFIED**。
2. **Query instruction 缺失**：BGE 英文检索通常建议查询侧 instruction；当前 `encode(query)` 直接编码 query，属于质量限制，不是索引错误，但会降低检索质量。
3. **索引与当前 catalogue 的强对齐未验证**：`DenseCatalogIndex.load` 只校验 artifact 内部 identities 自洽，未把 identities 与当前 `MetadataIndex.records`/mapping 做交叉校验。换 catalogue/mapping 时可能无法拒绝同形 stale index。
4. **Hybrid source 对 dense-only 未知 ID 无前置丢弃**：`CatalogSearchSource` 融合时会加入 dense rank 中的 ID；如果 stale index 含当前 catalogue 没有的 ASIN，需要依赖后续 grounding 才能拦截。建议正式接入前增加 catalogue identity 校验。
5. **无真实 BGE 运行**：没有一条真实 query 经 BGE 返回当前 catalogue ASIN；因此无法声称 BGE 检索质量或可用性。

## 3. 当前 Agent 是否真的调用 BGE？

**没有。**

代码路径：

- `DemoAgentService` 只在 `AGENTRECX_DENSE_RETRIEVAL` 为 on 且 `AGENTRECX_DENSE_INDEX` 有路径时加载 `DenseCatalogIndex`；
- 默认 `AGENTRECX_DENSE_RETRIEVAL` 不走该分支；
- P6/P7/P8 Runner 未设置该 env；
- `benchmarks/agent/runner.py` 明确 `AGENTRECX_DENSE_RETRIEVAL=off`。

实测默认 demo service：

```text
catalog_score_kind = bm25
dense_index = None
```

所以当前真实 Agent 的 Catalog Search 是 **BM25 only**。

## 4. 哪些代码/数据可复用

可复用：

- `DenseCatalogIndex` 的 artifact v2、identity/vector 对齐、BGE encoder 分支、hashed fallback 分离；
- `CatalogSearchSource` 的 opt-in hybrid RRF 融合实现；
- `tests/test_dense_catalog.py` 的 hashed/fake-dense 测试骨架；
- 当前 catalogue 与 ASIN 身份数据（`parent_asin`）；
- 现有 CandidateLedger、Role Gate、Eligibility、CandidatePlane，可在 dense 接入后继续兜底。

不可复用：

- 不存在可验证的 BGE 模型权重；
- 不存在商品 Embedding/index；
- 当前环境没有 sentence-transformers；
- 没有真实 BGE 构建/运行记录。

## 5. 本次实际完成

1. 复核源码、测试、文档、本地/远端 cache、环境依赖；
2. 修复 BGE 无模型时静默降级为 hashed 的关键问题；
3. 增加 1 个针对性回归；
4. 验证默认 Agent 仍为 BM25-only，未修改默认 Serving V1；
5. 未伪造 BGE 资产、未下载新模型、未接入无效 dense source。

验证命令结果：

```text
tests/test_dense_catalog.py    4 passed
default demo: catalog_score_kind=bm25, dense_index=None
```

## 6. BM25 / BGE / Hybrid 当前状态

| 通道 | 状态 |
|---|---|
| BM25 `CatalogSearchSource` | **VERIFIED**：默认可用，输出真实 catalogue ASIN |
| BGE exact dense | **UNVERIFIED / unavailable**：源码存在，模型/index/依赖缺失 |
| Hybrid BM25+dense | **LIMITED**：RRF 融合代码和 fake-dense 单测存在；真实 BGE index 未构建，未端到端验证 |
| Agent 默认调用 | BM25 only |

## 7. 最小转换方案（后续执行，不在本轮实现）

1. 获得并落盘 `BAAI/bge-small-en-v1.5` 的完整权重（当前 cache 只有 ref，无 blobs；不要用 hashed fallback 冒充）。
2. 在专用环境安装 `sentence-transformers`（不改变项目默认依赖/根环境；遵守依赖策略）。
3. 用当前 `Sports_and_Outdoors_products.jsonl` 的 156,746 个 `parent_asin` 构建 `agentrecx.dense_catalog.v2` 索引：
   - `encoder_kind="bge"`, dim=384, normalize_embeddings=True；
   - 记录 model path、model revision、catalogue/mapping hash；
   - 校验 index identities 与当前 catalogue/mapping 完全对齐。
4. 在正式接入前增加 catalogue identity 校验，并检查 stale index / unknown ASIN。
5. 设置：
   ```text
   AGENTRECX_DENSE_RETRIEVAL=on
   AGENTRECX_DENSE_INDEX=<bge_dense.npz>
   ```
   只做小样本真实 query 测试，确认 BGE-only 和 Hybrid 都返回当前 catalogue ASIN。
6. 再评估 query instruction / 文本截断 / hybrid 权重；未通过前不报告检索质量提升。

## 8. 是否具备继续开发 P8 Source Policy 的条件？

**不具备。**

- 当前没有可执行的 BGE 语义通道；
- Hybrid 只有 fake-dense 单测，没有真实 BGE index；
- 没有真实 query 的 BGE 候选证据；
- 不能把 P8 Source Policy 建在未验证的 dense source 上。

需要在模型、索引、依赖、catalogue 对齐和真实检索验证完成后，再进入 Source Policy。

## 9. 交付

- 报告：`P8_BGE_REUSE_REPORT.md`
- 源码 diff：`P8_BGE_SOURCE_DIFF.txt`
- 默认 BM25 验证：demo harness 输出 `catalog_score_kind=bm25`, `dense_index=None`
- 单测：`tests/test_dense_catalog.py` 4 passed

本轮未训练、未运行 Holdout、未修改 Serving V1/Gold/Evaluator/Eligibility、未新增 Cross-Encoder/MCP/Skills/Multi-Agent/RL、未 commit/push/deploy；未启动或停止 GPU 实例。
