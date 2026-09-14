# AgentRec-X — Documentation Index

Start with the [project README](../README.md) for the overview; use the documents below for
depth.

| Document | Read it when you want to… |
| --- | --- |
| [README](../README.md) | understand what the project is, what is learned vs rule-based, the accepted benchmark, and how to get started |
| [ARCHITECTURE.md](ARCHITECTURE.md) | know what each component owns, where the trust boundaries are, how failures behave, and how heavy objects are reused |
| [EXPERIMENTS.md](EXPERIMENTS.md) | check the dataset, the evaluation protocol, the accepted metrics, the policy diagnostics, and threats to validity |
| [USAGE.md](USAGE.md) | actually run the server, the demo, the API, the tests and the smokes — including troubleshooting |
| [PROJECT_HISTORY.md](PROJECT_HISTORY.md) | see the milestone sequence, its commits, and the invariants that held across them |

Development instructions and engineering rules live in [AGENTS.md](../AGENTS.md).

## Where things live

* **Per-package contracts** — one README per package under `recommendation/`, e.g.
  [`agent/`](../recommendation/agent/README.md),
  [`memory/`](../recommendation/memory/README.md),
  [`preference_matching/`](../recommendation/preference_matching/README.md),
  [`reranking/`](../recommendation/reranking/README.md),
  [`catalog/`](../recommendation/catalog/README.md),
  [`rag/`](../recommendation/rag/README.md),
  [`tools/`](../recommendation/tools/README.md),
  [`evaluation/`](../recommendation/evaluation/README.md),
  [`demo/`](../recommendation/demo/README.md),
  [`api/`](../recommendation/api/README.md).
* **Evidence** — accepted metrics live in `runs/sasrec_canonical_2026/run.json`; catalogue
  statistics in `data/processed/Sports_and_Outdoors_products_manifest.json`; policy
  diagnostics are reproducible via `experiments.reranking_evaluation_smoke`.
* **Verification** — the pytest suites under `tests/` and the PASS/FAIL smoke scripts under
  `experiments/`.

## Documentation conventions

Numbers quoted in these documents come from a committed artifact or an executed smoke run,
and each is attributed. Where a value is not recorded, the text says so rather than
estimating it. Two distinctions are maintained throughout:

* **result categories** — the recommendation benchmark, offline policy diagnostics, and
  engineering measurements are reported separately and never merged;
* **validity level** — no relevance, satisfaction or conversion claim is attached to
  preference-aware reranking, because no preference-conditioned relevance labels exist.
