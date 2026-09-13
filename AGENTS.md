# AgentRec-X Development Instructions

## 1. Project Overview

AgentRec-X is an intelligent e-commerce recommendation system combining:

* traditional and sequential recommendation;
* LLM Agent orchestration;
* Retrieval-Augmented Generation (RAG);
* long-term user memory;
* ranking and critique;
* Semantic-ID-based generative recommendation.

The final system should support:

user natural-language request
→ intent understanding
→ user preference retrieval
→ product knowledge retrieval
→ recommendation model invocation
→ candidate filtering/ranking
→ constraint checking
→ recommendation explanation
→ multi-turn interaction.

The project is developed incrementally.

Do not implement later-stage components before the current milestone is stable and verified.

---

## 2. Development Roadmap

The project follows this order:

### Phase 1 — Recommendation Baseline

1. Amazon Reviews 2023 preprocessing
2. Real-data integration verification
3. Unified evaluation protocol
4. ItemCF baseline
5. SASRec dataset
6. SASRec model
7. SASRec smoke test
8. Full SASRec training
9. Offline model comparison
10. Recommendation inference
11. FastAPI recommendation service

### Phase 2 — Agent System

1. LangGraph workflow
2. Planner / intent understanding
3. Recommendation Tool
4. RAG product knowledge retrieval
5. User Memory
6. Recommendation explanation

### Phase 3 — Agent Enhancement

1. Ranking
2. Critic / constraint checking
3. Multi-turn interaction
4. Web Demo

### Phase 4 — Generative Recommendation

1. Product embeddings
2. RQ-VAE
3. Semantic IDs
4. SID Transformer
5. SASRec vs Semantic-ID recommendation comparison

Do not prematurely implement a later phase.

---

## 3. Current Project Status

Completed:

* AutoDL development environment
* NVIDIA RTX 4090 GPU verification
* PyTorch / CUDA verification
* DeepSeek Harness GPU access verification
* preprocessing synthetic-data milestone

The preprocessing pipeline currently provides:

* input validation;
* chronological ordering;
* iterative k-core filtering;
* deterministic user/item mappings;
* item ID 0 reserved for padding;
* reproducible processed artifacts;
* synthetic regression tests.

Current priority:

Recommendation Baseline.

The next recommendation milestones must preserve compatibility with the completed preprocessing pipeline.

---

## 4. Engineering Principles

For every task:

1. Inspect the existing repository before modifying files.
2. Reuse existing working components whenever appropriate.
3. Make the smallest justified change.
4. Do not rewrite stable code without a concrete reason.
5. Keep responsibilities separated between:

   * preprocessing;
   * datasets;
   * models;
   * evaluation;
   * training;
   * inference;
   * APIs;
   * experiments.
6. Prefer small, verified milestones over large speculative implementations.
7. Add regression tests when fixing bugs.
8. Run relevant tests after changes.
9. Execute smoke tests before expensive operations.
10. Report exactly what was changed and what was actually verified.

Never claim code works unless an appropriate verification command has actually been executed.

Do not silently change the project architecture, data protocol, or evaluation methodology.

---

## 5. Recommendation-System Correctness

Recommendation experiments must avoid temporal leakage.

User interactions must remain chronologically ordered.

For sequential recommendation:

* never use random train/test splitting;
* prefer temporal splitting or leave-one-out / leave-two-out evaluation;
* training data must contain only events that happened before validation/test targets.

A typical sequential split may follow:

history:
[i1, i2, i3, i4, i5, i6]

train:
[i1, i2, i3, i4]

validation target:
i5

test target:
i6

The exact protocol must be documented and consistently reused.

---

## 6. Unified Evaluation Protocol

All recommendation models that are compared must use the same evaluation protocol.

This applies to:

* ItemCF;
* SASRec;
* future Semantic-ID / SID models.

Primary metrics:

* HR@K;
* Recall@K;
* NDCG@K.

The evaluation implementation must explicitly state whether it uses:

* full-ranking evaluation; or
* sampled-negative evaluation.

Never compare model results produced under different candidate-generation or evaluation protocols without clearly labeling the difference.

For single-positive leave-one-out evaluation, remember that:

HR@K and Recall@K are numerically equivalent.

This should be documented rather than treated as an implementation bug.

Candidate masking must prevent already-consumed historical items from being recommended unless the experimental protocol explicitly requires otherwise.

Evaluation code should use deterministic tie-breaking where applicable.

---

## 7. Preprocessing Invariants

The existing preprocessing pipeline establishes several invariants that later modules must respect.

### Chronology

User histories represent:

past → future.

Do not reorder them arbitrarily.

### Item IDs

Item integer ID:

0

is reserved for padding.

No real item may use ID 0.

Real item IDs begin from 1.

### User IDs

Existing mappings should remain deterministic and compatible with saved artifacts.

### Canonical Item Identity

The preprocessing pipeline currently uses Amazon Reviews 2023 `parent_asin` as the canonical product identifier.

Do not switch to `asin` or another identifier without an explicit experiment or migration plan.

### Raw Data

Never mutate raw Amazon data in place.

Processed datasets must be reproducible from:

* raw input;
* preprocessing configuration;
* code version.

### k-core Filtering

Frequency filtering must preserve the existing iterative fixed-point semantics.

Do not replace true iterative k-core filtering with a single users-then-items filtering pass.

---

## 8. Data Safety and Storage

Large data should not be committed to Git.

Do not commit:

* raw Amazon datasets;
* large processed datasets;
* model checkpoints;
* embeddings;
* caches;
* virtual environments;
* experiment outputs;
* API keys;
* credentials;
* secrets.

Raw datasets and generated artifacts should remain reproducible and externally stored where appropriate.

Never hard-code credentials.

Never modify raw datasets in place.

Before destructive operations, confirm the target path and ensure it is not raw source data.

---

## 9. Synthetic and Smoke Testing

Before any expensive or large-scale computation:

first verify the complete pipeline on a small dataset.

For recommendation preprocessing:

synthetic data
→ tests
→ small real-data sample
→ full real dataset.

For SASRec:

tiny dataset
→ forward pass
→ loss computation
→ tiny overfit test
→ evaluation sanity check
→ GPU full training.

For RQ-VAE / Semantic ID:

small item subset
→ reconstruction/codebook sanity check
→ Semantic-ID generation verification
→ larger training.

Expensive computation must not be the first validation step.

---

## 10. GPU Usage

The project runs on an AutoDL RTX 4090 environment.

GPU access has already been verified.

Do not reinstall or upgrade:

* NVIDIA drivers;
* CUDA;
* PyTorch;
* DeepSeek Harness

unless a future dependency clearly requires it and the change is explicitly approved.

Use GPU only when it materially benefits the task.

CPU is preferred for:

* preprocessing;
* synthetic tests;
* ItemCF;
* most evaluation-code development;
* lightweight data inspection;
* unit tests.

GPU should primarily be used for:

* SASRec training;
* embedding inference at scale;
* RQ-VAE training;
* Semantic-ID models;
* SID Transformer training.

---

## 11. Dependency Policy

Do not introduce unnecessary frameworks.

Prefer the standard library or existing project dependencies when sufficient.

If a new dependency is required:

1. explain why it is necessary;
2. keep it minimal;
3. document it;
4. avoid modifying the base CUDA/PyTorch environment unnecessarily.

Development dependencies should preferably live in the project virtual environment rather than altering unrelated system packages.

---

## 12. Repository Structure

Keep the codebase modular.

Recommended top-level responsibilities include:

* `recommendation/`
* `agent/`
* `rag/`
* `memory/`
* `semantic_id/`
* `api/`
* `frontend/`
* `experiments/`
* `tests/`

Do not create these modules before they are actually needed.

Within recommendation code, keep separate concerns for:

* preprocessing;
* datasets;
* models;
* training;
* evaluation;
* inference.

Do not place the entire recommendation system in one large Python file.

---

## 13. Testing Requirements

Important behavior must have automated tests.

Tests should cover, where applicable:

* chronological ordering;
* temporal splitting;
* filtering behavior;
* mapping invariants;
* padding invariants;
* deterministic behavior;
* candidate masking;
* metric correctness;
* model tensor shapes;
* loss computation;
* checkpoint save/load;
* inference consistency.

When a bug is discovered:

1. reproduce it;
2. add a regression test;
3. fix it;
4. rerun the relevant test suite.

Do not fix a reproducible bug without preserving a test for it when practical.

---

## 14. Reproducibility

Experimental results should be reproducible.

Where relevant, control:

* random seeds;
* dataset version;
* preprocessing thresholds;
* model hyperparameters;
* evaluation settings;
* candidate protocol;
* checkpoint path;
* software versions.

Configuration should not be scattered across unrelated source files.

Important experimental parameters must be stored or logged.

---

## 15. Git Discipline

Use Git as the project history and safety boundary.

Before substantial changes:

inspect:

* repository status;
* relevant files;
* existing tests.

After each milestone:

* run tests;
* inspect `git diff`;
* inspect `git status`;
* summarize modifications.

Do not silently overwrite unrelated user work.

Do not commit generated large artifacts.

A milestone should be committed only after its acceptance criteria pass.

---

## 16. DeepSeek Harness Execution Rules

DeepSeek Harness acts as the development and ML-engineering executor for AgentRec-X.

It may:

* inspect the repository;
* write code;
* modify tests;
* execute shell commands;
* process data;
* run tests;
* train models when requested;
* inspect logs and metrics;
* fix verified bugs;
* update documentation.

It must not interpret a broad project goal as permission to implement all remaining phases.

Every task should remain within the explicitly requested milestone.

When ambiguity does not affect experimental correctness:

make a reasonable engineering decision, document it, and continue.

When ambiguity materially affects:

* dataset semantics;
* temporal correctness;
* evaluation fairness;
* irreversible data operations;
* expensive training;

surface the issue before committing to the operation.

---

## 17. Milestone Completion Standard

A milestone is not complete because source code exists.

A milestone is complete only when:

1. required files exist;
2. implementation is internally consistent;
3. automated tests pass;
4. a relevant smoke test passes;
5. outputs/artifacts are inspected;
6. important invariants are verified;
7. documentation reflects the implementation;
8. Git status/diff is reviewed;
9. unresolved issues are explicitly reported.

Each completion report should include:

* files created;
* files modified;
* architectural decisions;
* commands executed;
* tests executed;
* test results;
* smoke-test results;
* generated artifacts;
* unresolved issues;
* recommended next milestone.

---

## 18. Current Priority Rule

Until the Recommendation Baseline is stable, do not implement:

* LangGraph Agent workflows;
* RAG;
* Memory;
* Critic Agent;
* Ranking Agent;
* Semantic ID;
* RQ-VAE;
* SID Transformer.

The current required progression is:

preprocessing
→ real-data integration
→ unified evaluation protocol
→ ItemCF
→ SASRec
→ offline comparison
→ inference
→ recommendation API.

Only after this recommendation backbone is stable should the project move to the Agent layer.

At the beginning of every new milestone or substantial task, read this AGENTS.md before making repository changes.
