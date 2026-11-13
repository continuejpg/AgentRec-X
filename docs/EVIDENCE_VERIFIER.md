# Independent Evidence Verifier — GPU Pilot

## 1. Existing Evidence Audit

P15 already has:

- `EvidenceBundle` with product spans and preference provenance;
- `validate_claims`, which checks:
  - valid claim types;
  - product/user ownership of references;
  - exact spans;
  - inactive/retracted preferences;
  - UNKNOWN promotion;
  - unsupported recommendation refs;
- `render_final_answer`, which deterministically renders only validated claims.

It is explicitly **structural/lexical validation**, not semantic entailment
(`semantic_truth_verified: false`).  The missing layer is:

> Does the cited evidence actually mean the claim?

The independent verifier is a second Qwen role over the existing `/explain` endpoint.
It receives only product id, claims, evidence and unknown fields; it does not receive P15
validation, Planner hidden state, Gate decision or the final answer rationale.

## 2. Open-Source Reuse Decision

| Project | URL | Licence | Decision |
|---|---|---|---|
| RAGAS Faithfulness | https://github.com/vibrantlabsai/ragas · https://docs.ragas.io | Apache-2.0 (GitHub API) | Borrow the decompose/verify **pattern**; no framework dependency |
| DeepEval Faithfulness / Hallucination | https://github.com/confident-ai/deepeval | Apache-2.0 (GitHub API) | Not adopted: judge-framework dependency and different evaluation contract |
| TruLens groundedness | https://github.com/truera/trulens | MIT (GitHub API) | Not adopted: observability framework is broader than this pilot |
| HF NLI/entailment model | e.g. https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | not adopted | Would add another model/VRAM; existing Qwen role is sufficient for the pilot |

No third-party code was copied. No new Python dependency was added. The implementation is
a small Qwen prompt/client and strict JSON parser in the existing project.

## 3. Evaluation Dataset

Built from real P15 bundles / real catalogue products:

- 49 cases;
- 4 products: `B004K6R43G`, `B07WZFL74B`, `B09H2JQD48`, `B00005USA4`;
- 41 dev / 8 holdout;
- labels: 19 SUPPORTED, 11 CONTRADICTED, 19 INSUFFICIENT_EVIDENCE;
- label source: **provisional_rule_authored_not_human_verified**;
- categories include direct support, absent facts, wrong product evidence,
  negative form factor, Mechanical Feeling vs mechanical switches, historical price,
  C4/metadata form-factor conflict, missing preference refs, wrong preference polarity.

Artifacts are git-ignored under `runs/p20_metadata/claim_evidence_*`.

## 4. Independent Verifier

Implementation: `recommendation/control/evidence_verifier.py`.

Input:

- product id;
- claim list (claim id/text/type/attribute/cited refs);
- evidence items with product id, raw field, span/excerpt and attribute;
- unknown fields.

Output:

```json
{"verdicts": [
  {"claim_id": "...", "verdict": "SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE",
   "reason": "...", "evidence_refs": ["..."], "missing_support": ["..."]}
]}
```

Rules:

- same Qwen checkpoint and `/explain` service;
- batch size 2 in the pilot; bounded output;
- strict verdict enum and claim-id validation;
- parser records positional claim-id alignment when the model prefixes/renames ids;
- invalid JSON/verdict is `invalid_model_payload`, never counted as success;
- no Gate/Memory/selection changes;
- shadow-only in MCP (`--evidence-verifier` / `AGENTRECX_EVIDENCE_VERIFIER=1`).

## 5. GPU Execution Evidence

- GPU: AutoDL RTX 4090 24 GB
- model: existing Qwen3.5-9B-ms (`/plan` + `/explain`)
- service baseline VRAM: 17,549 MiB
- observed verifier peak: **22,437 MiB** used, 1,656 MiB free, 24,564 MiB total
- pilot: 25 verifier calls, 70,166 input tokens, 3,278 output tokens
- verifier call latency: P50 **4,266 ms**, P95 **5,497 ms**; total call time 112,959 ms
- invalid JSON/verdict: **0** after parser repair; 25 batched calls used positional id
  alignment because Qwen prefixed claim ids
- no OOM; margin is tight (about 1.6 GiB free)

## 6. Baseline vs Verifier

P15 baseline: structural/lexical validator.
Verifier: independent Qwen semantic verdict.

| Metric | P15 baseline | Qwen verifier |
|---|---:|---:|
| Unsupported claim detection recall | 2/30 = **6.7%** | 26/30 = **86.7%** |
| Unsupported claim precision | — | **100%** (26/26) |
| Supported claim accuracy | — | **19/19 = 100%** |
| False rejection rate (supported) | 0% | **0%** |
| INSUFFICIENT_EVIDENCE correct | — | 16/19 |
| CONTRADICTED correct | — | 8/11 |
| Baseline accept rate | **95.9%** | — |
| Invalid JSON/verdict | — | **0** |

Split:

- dev: unsupported recall 21/25 = 84%, supported 16/16 = 100%;
- holdout: unsupported recall 5/5 = 100%, supported 3/3 = 100% (only 8 cases).

### Semantic gap examples

- `c03`: claim `"The keyboard has mechanical switches."` cites a product whose title says
  `"Mechanical Feeling"`.  P15 accepts it structurally; verifier returns
  `INSUFFICIENT_EVIDENCE` because a subjective phrase does not establish switch hardware.
- `a06`: claim `"active noise cancellation"` cites only a product title; P15 accepts;
  verifier returns INSUFFICIENT.
- `p04`: claim `"You prefer over-ear"` cites an active `avoid over-ear` preference;
  P15 accepts active provenance; verifier returns CONTRADICTED.

### Actual false negatives

- `d05`: C4 metadata categories say Over-Ear while metadata details say On Ear; verifier
  returned SUPPORTED (it did not resolve the cross-source conflict).
- `p06`, `p07`: claim is true in the bundle but the **cited preference ref is the wrong
  source**; verifier returned SUPPORTED and did not enforce citation provenance.
- `p08`: preference claim has no cited evidence; verifier returned SUPPORTED, while the
  expected label is INSUFFICIENT_EVIDENCE.

These are real limitations, not hidden by label changes.

## 7. Real Agent Shadow E2E

Six real Qwen Agent tasks (`0041`, `0042`, `0043`, `0044`, `0049`, `0055`) through MCP with
`--evidence-verifier`:

- all 6 finalized and Gate `ALLOW_FINALIZE`;
- 23 P15 claims;
- verifier verdicts: **23/23 SUPPORTED**;
- verifier did not change Gate, final answer, Memory or Planner actions;
- verifier calls: 20,630 input / 1,596 output tokens, total ~54.3 s across six calls;
- E2E total: 259.0 s, including main Qwen Planner/Explain and verifier shadow.

No unsupported semantic claim occurred in these six final answers, so shadow mode found
nothing to intercept.  The positive detection evidence comes from the 49-case GPU pilot.

## 8. Limitations

- No human gold labels; all semantic labels are provisional and were authored as pilot
  cases.  The 8-case holdout was not used for prompt/label iteration, but remains provisional.
- Not a second model architecture: both roles share Qwen3.5-9B, so correlated model errors
  remain possible.
- The verifier is not yet an interceptor; no claim is blocked or regenerated automatically.
- Four pilot false negatives remain: C4/metadata conflict and citation-provenance errors.
- The verifier currently accepts some claims using surrounding evidence despite the
  citation-specific instruction; stricter citation/provenance enforcement is future work.
- Peak VRAM is high (22.4 GiB / 24 GiB); larger batches or longer contexts risk OOM.
- P18 Frozen Manifest was not modified; the known pre-existing hash failure remains.
