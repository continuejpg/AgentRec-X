# P12 — Review Evidence Recovery & Retrieval Diagnosis 报告

## 0. 结论

**原始 Amazon Sports & Outdoors 评论可以可靠关联到 P11 的真实训练历史；但恢复文本只能形成未经人工确认的 Level A 候选。检索诊断表明 Hybrid 落后 BM25 不是 BGE 模型/Index 错误，而是固定 Top-K 融合预算问题；P11 bounded pool 也造成了评测漏标。**

| 维度 | 判断 |
|---|---|
| 原始评论可取得 | **VERIFIED** |
| user+item+timestamp 匹配 | **VERIFIED：801/801；0 ambiguous** |
| Level A 确认 | **NO：0 confirmed；56 个 heuristic candidate 待审核** |
| Hybrid 根因诊断 | **完成；非模型/Index 错误** |
| P11 bounded-pool 误差修正 | **完成指标拆分** |
| P13 个性化 Agent 实验 | **LIMITED，仍不 GO** |
| **P12 总判断** | **LIMITED** |

## 1. 原始评论恢复

本地没有原始 review 文件。按官方 Amazon Reviews 2023 数据源获取：

```text
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz
```

只流式匹配 P11 的真实用户和 train-history：

- `user_id + parent_asin + timestamp_ms` 精确匹配；
- timestamp 统一为毫秒；
- rating 仅辅助核对；
- 未匹配 validation/test target、未来评论或冻结 Holdout；
- 未构建完整评论数据库。

原始文件 fingerprint：

```text
sha256 = 8f6ddb51c2674d048387533f1c58e22840e5eee639a59737eae979d66a87d642
size   = 2,634,864,204 bytes
raw records = 19,595,170
```

匹配结果：

| 指标 | 值 |
|---|---:|
| P11 users | 86 |
| train-history keys | 801 |
| raw records scanned | 19,595,170 |
| user review records | 1,172 |
| exact key matches | **801** |
| unmatched user records（非 history） | 371 |
| ambiguous keys | **0** |
| rating mismatches | **0** |

结论：评论文本可与训练历史可靠关联，未用同名商品/相同 rating/猜测强制关联。

## 2. Level A 偏好证据

P11 的 100 个 Level B claims：

- 100/100 有匹配 review；
- heuristic `explicit_positive_attribute_candidate`：**56 claims**；
- explicit negative candidate：**11 claims**；
- 无 review 文本：0；
- **confirmed Level A = 0**。

56 个只是规则候选，不是确认偏好。典型风险：

- “lightweight / perfect / easy” 评价的是单个商品，不能自动推广为稳定用户偏好；
- 睡眠垫任务的历史评论可能实际讲睡袋、帐篷或水瓶架；
- rating=5 不能解释为“喜欢某属性”；
- 同一商品评论不能确认跨任务偏好。

输出仍为：

```text
level_a_status = level_a_candidate_unreviewed
confirmed_level_a = false
human_review_required = true
```

原 P11 任务不变；增强副本：

- `data/p11_benchmark/tasks_p12_evidence.jsonl`

原 `tasks.jsonl` 未覆盖。

69 个 same-query pairs 中，44 对至少一侧存在 review candidate；但仍不能自动升级为 Level A。

## 3. 检索诊断

### 3.1 Query/模型/Index

- BM25/BGE/Hybrid 使用同一 P11 task query；
- BGE：`BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，768 维，156,746 条 index；
- query instruction 正确存在；
- 未发现 model/index/query-instruction 路径错误。

### 3.2 覆盖

156 个有 verified pool 的任务：

| Source | Reachable@10 | Reachable@50 | mean found@10 |
|---|---:|---:|---:|
| BM25 | 36 | **140** | 0.776 |
| BGE | 14 | 102 | 0.256 |
| Hybrid legacy | 13 | 86 | 0.500 |
| Hybrid bounded BM25 head=50 | 13 | 87 | 0.500 |

### 3.3 Hybrid 根因

代码路径：

- `CatalogSearchSource.search(hybrid)` 把 BM25 与 dense 放入同一 RRF；
- dense 只取 `limit`；
- 最终只输出 `limit` 个融合候选；
- dense-only 候选占用融合 Top-K 槽位，挤掉 BM25 head 中的合法候选。

因此：

- **不是 BGE 模型错误**；
- **不是 index 错误**；
- 是 fixed Top-K fusion budget / source ranking tradeoff；
- 把 BM25 截断为同样深度只从 86→87，不能解决；
- 未调默认 RRF；仅保留 opt-in `hybrid_bm25_head` 对照。

### 3.4 History 缓存公平性

- 163/163 任务历史与 P8-D cache 完全一致；
- History 结果是真实 P8-D 模型输出，不是输入错位；
- 但 History Tool **query-independent**，不接收当前 task query；
- 因此 17/164 可达是该工具契约的结果，不是 query-aware retriever 的公平比较；
- Serving V1 History 内部已含 `catalog_lexical` BM25；外部 `catalog_search` 再引入 BM25，存在重复信号。

## 4. 评测口径修正

P11 candidate pool 按 ASIN 排序截断到 100，不是完整 oracle。

| 指标 | 结果 |
|---|---|
| Bounded-Pool Reachability | BM25 140/156@50；BGE 102；Hybrid 86 |
| Catalog-Verified Validity | Agent 子集 4/10 |
| Strict Pool Success | 2/10 |
| Preference Satisfaction | 未确认（Level A=0；Level B 需人工） |
| Planner Decision Validity | invalid finalize 5/10；hard violation 3/10 |

最小 Agent 子集（Rule Agent + 真实 Hybrid/BGE + history）：

| 指标 | 值 |
|---|---:|
| strict verified success | 2/10 |
| catalog-validity success | 4/10 |
| invalid finalize | 5/10 |
| HardConstraintViolation | 3/10 |
| execution errors | 0 |

- 2 个 final catalog-valid 但不在 bounded pool 内；
- 5 个 invalid finalize 主要来自 clarification/unknown 情境；
- 3 个 hard violation 来自 role UNKNOWN/MISMATCH；
- unsatisfiable 任务没有 final（行为正确）；
- strict 与 catalog-validity 必须分开报告。

## 5. 十个问题的回答

1. **能否可靠关联评论？**
   **能。** 801/801 keys 精确匹配；0 ambiguous；0 rating mismatch；只用 train history。

2. **100 个 Level B 多少得到文字支持？**
   100/100 有 review；56 个 heuristic positive candidate；11 个负向候选；**0 confirmed Level A**；56 个需人工审核。

3. **能否形成真正不同历史偏好证据的同 Query 配对？**
   69 pairs 存在，44 对至少一侧有 review candidate；但尚未人工确认 Level A。

4. **Hybrid 落后 BM25 的主要原因？**
   不是 BGE/index 错误，而是固定 Top-K RRF 中 dense-only 候选挤占 BM25 合法候选槽位；bounded BM25 head 86→87，无实质修复。

5. **History 17/164 旧缓存是否公平？**
   输入完全一致（163/163），但 History query-independent；不能作为 query-aware retriever 比较，且与外部 BM25 有重复信号。

6. **失败来源？**
   检索：BM25 Top-10 弱、BGE/Hybrid 更弱、History 仅 17；评测：bounded pool 导致 2 个 catalog-valid 被严格漏标；Agent：5 个 invalid finalize；不能归为单一原因。

7. **是否足以支持 P13？**
   **LIMITED，暂不 GO。** Review 可恢复，但 Level A 为零；56 个候选待人工审核；候选覆盖/Agent 决策仍有缺口。

8. **仍缺少什么？**
   人工确认的 Level A；query-conditioned history；更完整 Gold/候选覆盖；真实搜索意图标签；未暴露的 reserved evaluation split。

## 6. 任务隔离

- P11 的 164 个任务继续全部为 Dev；
- 不重命名为 Test；
- P12 只增加 review evidence enriched copy，不覆盖原任务；
- 未建立 reserved evaluation split；P13/P14 不得使用本轮已暴露任务作为冻结测试；
- P6/P8/P9 Holdout 未触碰。

## 7. 代码与产物

新增：

- `experiments/p12_match_reviews.py`
- `experiments/p12_audit_evidence.py`
- `experiments/p12_diagnose.py`
- `experiments/p12_hybrid_diagnosis.py`
- `tests/test_p12_evidence.py`
- `data/p12/review_keys.jsonl`
- `data/p11_benchmark/tasks_p12_evidence.jsonl`
- `runs/p12/reviews/reviews_matched.jsonl`
- `runs/p12/reviews/match_report.json`
- `runs/p12/evidence/summary.json`
- `runs/p12/evidence/claims.jsonl`
- `runs/p12/diagnosis/diagnosis.json`
- `runs/p12/hybrid/summary.json`

修改：

- `recommendation/control/catalog_search.py`
  - 仅新增 opt-in `hybrid_bm25_head`；默认 `None` 保持原 hybrid 行为；
  - 不修改 Serving V1 默认路径。

未修改：

- SASRec / Two-Tower / BGE 参数；
- Planner、M5 核心语义、Role Gate、Eligibility；
- 正式 Source Policy、Gold/Evaluator、Holdout。

## 8. 最终判断

| 项 | 状态 |
|---|---|
| 原始评论恢复 | **GO** |
| 训练历史关联 | **GO** |
| Level A confirmed | **NO** |
| Hybrid/检索诊断 | **完成，LIMITED** |
| 评测口径修正 | **完成** |
| P13 数据准备 | **LIMITED** |
| SFT-v1 | **NO-GO** |
| **P12 总判断** | **LIMITED** |

下一步优先：

1. 人工审核 56 个 review candidates；
2. 明确 History（query-independent）与 Catalog（query-aware）分工；
3. 在固定预算下重新评估融合，无新证据不继续调参；
4. 用未暴露用户建立 reserved split；
5. 再判断 P13 是否具备可靠数据基础。

未训练/微调模型，未运行 SFT-v1，未进入 P13，未运行 Holdout，未 commit/push/deploy；GPU 实例未关闭。
