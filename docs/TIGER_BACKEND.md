# AgentRec-X — Public TIGER Backend Adapter Specification (Step 2.2)

> **Status: specification only.** No implementation, no training, no artifact exists for this
> backend yet. This document is the contract that Step 2.3 (skeleton) and Steps 2.4–2.7
> implement against. It adds no Agent capability, no fusion change and no `CandidateSource`.

**What this replaces.** `docs/SEMANTIC_ID.md` describes **custom GenRec v0** (commit `80f81b1`):
a custom RQ-VAE-style tokenizer over *deterministically seeded, untrained* item features, a
custom tiny Transformer, a bounded prefix search, and a measured Recall@10 of `0.00185`. That
implementation is **frozen as a historical baseline** and is not extended any further. Its
weakness is measured and attributed in [`MODEL_EXPANSION_HANDOFF.md`](MODEL_EXPANSION_HANDOFF.md)
§6: untrained item features, a binding prefix-search budget (0.34 % catalogue coverage) and
per-code accuracy that compounds.

**What this specifies.** A **separate public-TIGER backend** — a real item-content encoder, a
real RQ-VAE, and a T5-style generative recommender — that owns only *learning and scoring*,
while AgentRec-X keeps ownership of identity, split, exposure, cohort, ranking and evaluation.

Lineage: Step 2.1 audited `mclwu22/amazon-genrec` and `snap-research/GRID` and concluded
**adapt, do not vendor**. Both licences constrain what may be copied: `amazon-genrec` ships **no
LICENSE file at all**, and `GRID` is **Snap Inc., non-commercial research only** with a
mandatory notice-retention clause. Every reimplementation is therefore labelled in
`backends/tiger_public/PROVENANCE.md`. See §13.

Cross-references: [AGENTS.md](../AGENTS.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
[MODEL_EXPANSION_HANDOFF.md](MODEL_EXPANSION_HANDOFF.md) · [PHASE5_HANDOFF.md](PHASE5_HANDOFF.md) ·
[SEMANTIC_ID.md](SEMANTIC_ID.md) · [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 0. Governing invariant

```text
AgentRec-X owns:  parent_asin <-> item_id - PAD = 0 - temporal_leave_two_out -
                  agentrecx.eval_protocol.v1 - train-history exposure - 20k cohort -
                  full catalogue - seen-item masking - tie-breaking - ranking -
                  Recall / NDCG / HR - GroundingVerifier - CandidateLedger -
                  Agent runtime - CandidatePlane - fusion - SASRec - Two-Tower

TigerBackend owns: item_id <-> backend_row - item content encoding - RQ-VAE /
                   Semantic IDs - TIGER - catalogue-constrained and certified
                   retrieval - raw item scores
```

**What the backend receives, stated precisely.** The backend **is** given `item_id` histories as
model inputs and produces scores keyed by `item_id`. It does not own their identity *semantics*:
it has no `parent_asin`, no user identity, no split, no target, no masking decision, no ranking
authority, no metric, and no evaluation. Receiving an opaque `item_id` is not holding authority
over it.

**Two absolute absences that shape the whole document:**

1. **`parent_asin` never crosses the backend boundary — and never enters the adapter either.**
2. **No target is visible to the backend.** Not gated, not flagged — absent from the contract.
   There is no `grade_only` switch to misuse, because no target field exists.

`recommendation/semantic_id/` remains read-only historical GenRec v0. It receives no content
embeddings, no k-means++, no dedup digit, no architecture change — ever, beyond an optional
compatibility reader for an existing v0 artifact.

---

## 1. Lifecycle

Four ordered stages, each a CLI subcommand, each writing a hashed artifact:

```text
STAGE 1  build_features(catalogue_text)                    -> ItemFeatureArtifact
STAGE 2  fit_semantic_ids(item_features)                   -> SemanticIdArtifact
STAGE 3  train_generator(train_exposure, semantic_ids)     -> TigerCheckpoint
STAGE 4  score(history_item_ids, required_frontier, semantic_ids, checkpoint)
                                                           -> Iterator[ScoreBatch]
```

| Stage | Inputs | Output | Depends on |
|---|---|---|---|
| 1 | `products_text.jsonl` + `catalogue.json` | `item_features/` | frozen catalogue only |
| 2 | `item_features/` | `quantizer/` + `semantic_ids.json` | stage 1 |
| 3 | `train_exposure.jsonl` + `semantic_ids.json` | `generator/` | stages 1–2 |
| 4 | `eval_cohort.jsonl` (histories + `required_frontier`) + `semantic_ids.json` + `generator/` | `scores/` shards or stream | stages 1–3 |

Stage 4 is the **only** stage that reads `eval_cohort.jsonl`, which contains **no target** (§8).
Stages 1–3 never read any cohort file. There is no `build_catalogue` method and no operation
producing SIDs directly from the catalogue.

Production crossing is always filesystem + subprocess:

```text
recommendation/backends/tiger_backend.py  --subprocess-->  <backend venv>/bin/python -m tiger_public.cli <stage>
```

The `TigerBackend` "interface" is a **contract** (schemas + semantics), expressed in
`contracts.py` as frozen dataclasses plus one `Protocol` used for documentation, structural
typing in the backend's own tests, and a single in-process test double — **not** a production
object AgentRec-X imports.

---

## 2. Directory placement

```text
X/
|- recommendation/                      # FROZEN (semantic_id/ = GenRec v0 historical)
|  `- backends/                         # NEW - the ONLY bridge; stdlib + numpy only; item_id-only
|     |- __init__.py
|     |- README.md
|     `- tiger_backend.py               # TigerBackendAdapter - SIDResolver - schema mirrors
|- backends/                            # NEW tree, own venv, never imported by X, item_id-only
|  `- tiger_public/
|     |- README.md
|     |- PROVENANCE.md
|     |- pyproject.toml
|     |- src/tiger_public/
|     |  |- __init__.py
|     |  |- contracts.py                # frozen dataclasses + Protocol
|     |  |- io.py                       # artifact read/write + hash verification
|     |  |- features.py                 # stage 1
|     |  |- quantizer.py                # stage 2
|     |  |- dedup.py                    # stage 2
|     |  |- scoring.py                  # stage 4 - the frozen item-score rule (section 5)
|     |  |- tiger.py                    # stage 3
|     |  |- trie.py                     # stage 3/4 - search constraint ONLY
|     |  |- retrieve.py                 # stage 4 - APPROXIMATE | CERTIFIED (section 15)
|     |  `- cli.py                      # build-features | fit-sid | train | score
|     `- tests/
|- experiments/
|  |- materialize_tiger_backend.py      # NEW - AgentRec-X-side; composes text, computes required_frontier
|  `- benchmark_public.py               # + `--backend tiger_public` selects the adapter
|- tests/
|  |- test_backend_boundaries.py        # NEW - AST/schema/provenance guards (T1-T8)
|  `- test_tiger_backend_adapter.py     # NEW - adapter behaviour
`- docs/
   `- TIGER_BACKEND.md                  # THIS SPECIFICATION
```

`scoring.py` is deliberately its own module: the frozen item-score rule (§5) must be one
implementation shared by beam search, branch-and-bound and exhaustive scoring, so it cannot
drift between retrieval algorithms.

`AGENTS.md` §12 does not list `backends/`; this adds one top-level responsibility for the reason
the file itself gives — separation of concerns — recorded as an explicit architectural decision.

---

## 3. Layering: canonical identity vs adapter vs backend

Four layers, strictly ordered, each with one job. **`parent_asin` exists only in the outer two.**

```text
+--------------------------------------------------------------------------+
| L0  AgentRec-X canonical identity layer      (existing, unchanged)       |
|     parent_asin <-> item_id - PAD = 0 - the accepted mappings artifact   |
+--------------------------------------------------------------------------+
                                  |  item_id only
                                  v
+--------------------------------------------------------------------------+
| L1  Materialisation layer   experiments/materialize_tiger_backend.py     |
|     MAY read parent_asin, for exactly three purposes:                    |
|       (a) composing products_text.jsonl text from the catalog artifact   |
|       (b) computing required_frontier per case (needs the seen set)      |
|       (c) provenance/hash bookkeeping                                    |
|     It writes ONLY item_id into every handoff artifact.                  |
+--------------------------------------------------------------------------+
                                  |  item_id only
                                  v
+--------------------------------------------------------------------------+
| L2  TigerBackendAdapter   recommendation/backends/tiger_backend.py       |
|     NEVER reads, accepts, stores or returns parent_asin.                 |
|     Accepts:  item_id histories + required_frontier                      |
|     Returns:  raw scores keyed by opaque item_id, in the evaluator's     |
|               own [batch, num_items + 1] item_id-column convention       |
|     Contains: no ML import, no backend import, no identity semantics     |
+--------------------------------------------------------------------------+
                                  |  item_id only
                                  v
+--------------------------------------------------------------------------+
| L3  TigerBackend   backends/tiger_public/                                |
|     item_id <-> backend_row <-> SID - content embedding - RQ-VAE - TIGER |
|     retrieval - raw scores                                               |
|     NEVER sees parent_asin, a target, a mask decision, or a metric.      |
+--------------------------------------------------------------------------+
```

The required flow:

```text
TigerBackend
  -> raw scores keyed by opaque item_id
TigerBackendAdapter
  -> raw scores keyed by opaque item_id          (no translation, no identity)
AgentRec-X canonical identity layer
  -> parent_asin                                 (only when something needs it)
```

**Two consequences:**

1. **The adapter performs no identity translation at all.** The evaluator's score matrix is
   `[batch, num_items + 1]` indexed by *item-id column*, with column 0 the contract-only PAD
   column. That is exactly the backend's convention, so the adapter's conversion (§10) is a
   *column alignment and fill*, not a `row -> parent_asin` mapping.
2. **Text composition happens in L1, before the handoff.** The adapter never sees the
   catalogue's text fields, never sees `parent_asin`, and never composes anything.

### 3.1 Identity spaces and the complete mapping

```text
  parent_asin (opaque str)                     AUTHORITY: L0 - mappings artifact
        |   item2id / id2item                              id2item[0] is None
        |   *** THIS EDGE NEVER CROSSES L1 -> L2 ***
        v
  AgentRec-X item_id   1 .. 156746             PAD = 0 is never a real item
        |   bridge: backend_row = item_id - 1            AUTHORITY: CatalogueHandoff
        v
  backend_row   0 .. 156745                    dense, 0-based, backend-private
        |   RQ-VAE assignment                            AUTHORITY: SemanticIdArtifact
        v
  SID = (c1..cL [, dd])                        c in [0,K);  dd = collision ordinal in [0,K)
        |   dedup_levels == 1  =>  SID is UNIQUE per backend_row
        v
  token block [off_0+c1, ..., off_{L-1}+cL, off_L+dd]      AUTHORITY: TokenLayout
```

| Rule | Owner | Meaning |
|---|---|---|
| `item_id -> parent_asin` | **L0/L1 only** | never performed by L2 or L3 |
| `SID -> item_id` | **L3** (against its own `assignment`) | total iff `dedup_levels == 1` **and** the SID is a catalogue SID; otherwise unresolved |
| `SID -> parent_asin` | **L0 only** | composition `SID -> item_id -> parent_asin`, outside L2 and L3 |

`item_id -> SID` is total; `SID -> item_id` **fails closed** — unknown SID means *nothing*, never
a nearest neighbour, partial match or default. `COLLISION_RULE = "eligible items first, then
ascending item_id"` is retained as defence in depth; with `dedup_levels == 1` it is provably a
no-op, and a test asserts `collision_size == 1` for every real row.

---

## 4. Precise Python / CLI boundary

```python
CONTRACT_VERSION = "agentrecx.tiger_backend.v3"
PAD_SENTINEL = -1                                  # no real digit is negative
```

```python
# =========================== AgentRec-X -> backend =========================== #

@dataclass(frozen=True)
class CatalogueHandoff:
    """Materialised from frozen AgentRec-X artifacts. NO parent_asin anywhere."""
    format: str                     # "agentrecx.tiger.catalogue.v3"
    num_items: int                  # 156746
    num_users: int                  # 412445
    pad_id: int                     # always 0
    first_real_id: int              # always 1
    mappings_sha256: str
    sequences_sha256: str
    products_sha256: str
    #: backend_row (0..num_items-1) -> opaque item_id (1..num_items).
    #: Explicit table; NO position is ever inferred; NO parent_asin is carried.
    item_ids: tuple[int, ...]

@dataclass(frozen=True)
class TrainExposureHandoff:
    """train_history ONLY. Structurally cannot carry a target."""
    format: str                     # "agentrecx.tiger.train_exposure.v3"
    protocol: str                   # "temporal_leave_two_out"
    protocol_version: str           # "agentrecx.eval_protocol.v1"
    field_source: str               # "EvaluationCase.train_history"
    users: int
    examples: int
    examples_sha256: str
    rows: tuple[tuple[int, ...], ...]

@dataclass(frozen=True)
class EvalCohortHandoff:
    """Histories + an integer frontier request ONLY. No target field, for any stage."""
    format: str                     # "agentrecx.tiger.eval_cohort.v3"
    cohort_seed: int                # 20260201
    cohort_size: int                # 20000
    protocol_version: str
    k_values: tuple[int, ...]       # (5, 10, 20) - the frozen cut-offs
    case_ids: tuple[int, ...]       # opaque, stable, == row order
    test_histories: tuple[tuple[int, ...], ...]
    #: AgentRec-X-computed integer per case; see section 15.3. The backend receives
    #: the COUNT only and never the seen-item identities.
    required_frontier: tuple[int, ...]
    required_frontier_stats: dict   # {"min":.., "max":.., "mean":.., "k_max":20, "cases":20000}
    #: NOTE: no validation_target, no test_target, no grade_only switch, no seen set.

# =========================== backend -> AgentRec-X =========================== #

@dataclass(frozen=True)
class ItemFeatureArtifact:
    format: str                     # "agentrecx.tiger.item_features.v3"
    num_items: int
    dim: int
    path: str
    sha256: str
    encoder: dict
    empty_text_items: int
    truncated_items: int
    catalogue_sha256: str

@dataclass(frozen=True)
class SemanticIdArtifact:
    format: str                     # "agentrecx.tiger.semantic_ids.v3"
    num_items: int
    levels: int                     # L codebook digits
    dedup_levels: int               # 0 or 1
    codebook_size: int              # K
    dedup_vocab_size: int           # the dedup level's own range
    #: item_id-indexed, length num_items + 1.
    #: assignment[0] == (PAD_SENTINEL,)*levels  -- NOT a valid-looking SID (section 12).
    assignment: tuple[tuple[int, ...], ...]
    pad_row: tuple[int, ...]
    collision_audit: dict           # pre_dedup + post_dedup blocks (section 13)
    utilisation: tuple[dict, ...]   # per-level reporting block (section 14)
    content_features: dict
    checkpoint_sha256: str

@dataclass(frozen=True)
class TigerCheckpoint:
    format: str                     # "agentrecx.tiger.checkpoint.v3"
    layout: dict                    # the TokenLayout, verbatim
    score_rule: dict                # the frozen item-score definition (section 5), by value
    model: dict
    training: dict
    exposure_sha256: str
    semantic_ids_sha256: str
    determinism: dict

@dataclass(frozen=True)
class ScoreBatch:
    """Raw scores for ONE batch, in request order. Follows the frozen arm contract."""
    format: str                     # "agentrecx.tiger.scores.v3"
    case_ids: tuple[int, ...]       # receipt: row i corresponds to case_ids[i]
    scores: "np.ndarray"            # float32 [batch, num_items + 1]
    reachable: "np.ndarray"         # bool    [batch, num_items + 1]
    reachable_real_count: "np.ndarray"   # int32 [batch] - real items reachable per row
    status: str                     # "APPROXIMATE" | "CERTIFIED"
    certification: dict             # section 15; present-but-empty when APPROXIMATE
    retrieval: dict                 # scored_share, expanded_nodes, beam, prefix_budget
    generation: dict                # generated top-1 SIDs, invalid count
```

**Stage CLI contract:**

```bash
python -m tiger_public.cli build-features --catalogue DIR --out DIR [--encoder ...] [--limit N] [--device ...]
python -m tiger_public.cli fit-sid        --catalogue DIR --features DIR --out DIR [--levels 3] [--dedup]
python -m tiger_public.cli train          --exposure FILE --sid DIR --out DIR [--epochs ...] [--bf16]
python -m tiger_public.cli score          --cohort FILE --sid DIR --ckpt DIR --out DIR --status {approximate,certified}
```

Each subcommand validates its inputs against `format` + `CONTRACT_VERSION`, writes only inside
`--out`, emits **one JSON status object on stdout**, and exits non-zero on any validation
failure. The adapter attaches the backend's stderr verbatim to `TigerBackendError` — never a
silently empty score matrix.

**Documentation-only `Protocol`** (backend tests + one in-process double; not a production import
path):

```python
class TigerBackend(Protocol):
    def build_features(self, catalogue: CatalogueHandoff, out_dir: Path,
                       config: Mapping[str, Any]) -> ItemFeatureArtifact: ...
    def fit_semantic_ids(self, catalogue: CatalogueHandoff, features: ItemFeatureArtifact,
                         out_dir: Path, config: Mapping[str, Any]) -> SemanticIdArtifact: ...
    def train_generator(self, exposure: TrainExposureHandoff, sid: SemanticIdArtifact,
                        out_dir: Path, config: Mapping[str, Any]) -> TigerCheckpoint: ...
    def score(self, cohort: EvalCohortHandoff, sid: SemanticIdArtifact,
              checkpoint: TigerCheckpoint, config: Mapping[str, Any]) -> Iterator[ScoreBatch]: ...
```

**Boundary rules the shape enforces:**

| Rule | Mechanism |
|---|---|
| Backend cannot learn canonical identity | no `parent_asin` parameter, field, or file on any L2/L3 path |
| **Adapter cannot learn canonical identity** | `parent_asin` is not a parameter, field, local, or return value anywhere in `recommendation/backends/`; asserted by `T7a` |
| Backend cannot see a target | `EvalCohortHandoff` has no target field; `score()` has no target parameter |
| Backend cannot see the seen set | it receives the integer `required_frontier`, never identities |
| Backend cannot mask | returns `reachable` as *data*; never applies `seen` |
| Backend cannot decide the catalogue | `CatalogueHandoff.item_ids` is authority; a mismatch is a refusal |
| Backend cannot rank or select for the evaluator | `ScoreBatch` carries scores, never a ranked list |
| Backend cannot claim a certified result cheaply | `status` is declared and its `certification` block is verified by the adapter (§15.6) |
| AgentRec-X cannot reach into the backend | the adapter holds no ML import and no backend import |

---

## 5. The frozen canonical TIGER item-score definition

This is fixed **before** Steps 2.5/2.6 and is a contract, not an implementation detail. It lives
in `backends/tiger_public/src/tiger_public/scoring.py` and is recorded by value in every
`TigerCheckpoint.score_rule`.

### 5.1 The definition

For a catalogue item `i` with Semantic-ID token path `t(i) = (t_1, ..., t_M)` where
`M = per_item_tokens`:

```text
score(i | history)
    =  SUM over l=1..M  of  log P_model( t_l | history, t_<l )
```

where

```text
P_model( . | history, t_<l )  =  softmax over the model's FULL token vocabulary
                                 of the decoder logits at step l
logits                        =  the model's raw output distribution, unmodified
```

**EOS: excluded from the sum.** The score covers the item's complete Semantic-ID token path only.
`EOS` is used by the *search* to recognise a completed path, and is **not** a term in the score.
This decision is frozen; changing it changes every number and requires a new `score_rule` version
plus a full re-run.

### 5.2 Model scoring semantics are not the search algorithm

```text
model scoring semantics   = the formula in 5.1, evaluated against the FULL-vocabulary log-softmax
                            -> defines the item ORDER
search algorithm          = how much of the catalogue is visited
                            (exhaustive | beam | branch-and-bound | bounded trie expansion)
                            -> defines the item COVERAGE
```

The catalogue trie / constrained decoder is a **SEARCH constraint only**. It restricts which
token paths are *considered*. It **must not** silently renormalise probabilities over only the
currently valid child tokens, because that would make an item's score depend on which other items
happened to be in the search frontier — i.e. on the search budget.

**Beam search, branch-and-bound and exhaustive scoring must all rank items under the identical
item-score definition of §5.1.** They may differ only in coverage, and that difference is reported
as `status` and `scored_share`, never as a different ranking rule.

```json
"score_rule": {
  "version": 1,
  "eos_in_score": false,
  "softmax": "full_vocabulary",
  "child_renormalisation": false,
  "terms": "sum of log P over the item's complete SID token path",
  "note": "changing any field requires a new version and a full re-benchmark"
}
```

A renormalised-over-valid-children variant is a **documented but unselected** alternative. If it
is ever selected it requires its own `score_rule.version`, a pre-registered experiment, and a
complete re-benchmark: the two rules are not comparable.

### 5.3 Admissibility restated against this frozen definition

For a prefix `p = (t_1, ..., t_m)` with `m < M`:

```text
S(p) = SUM over l=1..m of log P_model( t_l | history, t_<l )
```

Each term is a log-probability from a full-vocabulary softmax, hence **<= 0**, and **independent
of `m`** — for a fixed prefix, every extension `q >= p` keeps the same first `m` terms unchanged.
Therefore:

```text
S(q) = S(p) + SUM over l=m+1..M of log P_model(.)   <=   S(p)      for every q >= p
```

so `S(p)` is an **admissible upper bound** on every item in the subtree rooted at `p`, under the
exact rule of §5.1. The branch-and-bound maintains a max-heap keyed by `S(p)`, a running
threshold `T` = the current `required_frontier`-th best **completed** item score, and terminates a
branch exactly when `S(p) < T`; global termination when the best remaining prefix has `S(p) < T`.
Since every popped bound is `>=` every completion beneath it, no unvisited item can enter the
certified frontier.

---

## 6. AgentRec-X → backend input schemas

```text
runs/tiger_backend_handoff/
|- catalogue.json          # CatalogueHandoff
|- catalogue_items.jsonl   # {"backend_row": int, "item_id": int}        <- NO parent_asin
|- products_text.jsonl     # {"item_id": int, "text": str}               <- text pre-composed by L1
|- train_exposure.jsonl    # {"case_id": int, "items": [item_id, ...]}   <- train_history ONLY
|- eval_cohort.jsonl       # {"case_id": int, "history": [...], "required_frontier": int}
|- score_request.json      # {"status": "approximate|certified", "batch_size": int, ...}
`- manifest.json           # sha256 of every file above + the three source artifact hashes
```

```json
// catalogue.json
{ "format": "agentrecx.tiger.catalogue.v3", "contract_version": "agentrecx.tiger_backend.v3",
  "num_items": 156746, "num_users": 412445, "pad_id": 0, "first_real_id": 1,
  "item_id_span": [1, 156746],
  "mappings_sha256": "dca7815a...", "sequences_sha256": "d3d83426...", "products_sha256": "..." }
```

```json
// catalogue_items.jsonl
{"backend_row": 0, "item_id": 1}
{"backend_row": 1, "item_id": 2}
```

```json
// eval_cohort.jsonl   (targets and seen sets absent)
{"case_id": 0,     "history": [1042, 8831, 55, 90114], "required_frontier": 12}
```

**Rules:**

1. `parent_asin` appears in **no** file in this directory. L1 consumes it internally for text
   composition and frontier computation; L2 and L3 never see it.
2. `products_text.jsonl` carries the pre-composed text. Composition is L1-owned and auditable in
   one place, using the amazon-genrec template: `title . categories . store . features .
   description`, `max_chars` 1000.
3. `catalogue_items.jsonl` is the **explicit** row-id table; the backend must not infer
   `item_id = row + 1`, and refuses any inconsistency (missing row, duplicate, out-of-span id,
   non-ascending order).
4. `train_exposure.jsonl` is built from `case.train_history` **only**, by a function whose
   parameters cannot name a target (§8).
5. `eval_cohort.jsonl` carries `case_id`, `history`, and the integer `required_frontier` —
   nothing else.
6. Every file is hashed in `manifest.json` alongside the three *source* artifact hashes, so a
   reviewer traces the backend's inputs to the frozen artifacts without trusting the backend's
   own report.

---

## 7. Backend-internal artifact schemas

### 7.1 `item_features/` (stage 1)

```text
item_features.npy     # float32 [num_items, dim]; row r = item_id r+1  (NO PAD row)
item_features.json    # ItemFeatureArtifact
manifest.json
```

```json
{ "format": "agentrecx.tiger.item_features.v3",
  "num_items": 156746, "dim": 768, "dtype": "float32", "order": "C", "endianness": "little",
  "encoder": { "id": "sentence-transformers/sentence-t5-base", "revision": "<hf-sha>",
               "pooling": "mean", "normalize": false, "frozen": true },
  "text_template": "title | categories | store | features | description",
  "max_chars": 1000, "empty_text_items": 0, "truncated_items": 0,
  "nan_inf_scan": "clean",
  "catalogue_sha256": "...", "products_text_sha256": "...",
  "device": "cuda:0", "seconds": 0.0, "seed": 2026, "pad_row_present": false }
```

Items whose composed text is blank are **embedded, not dropped** — coverage stays total, which is
the property the v0 tokenizer obtains by construction and must not be lost. `nan_inf_scan` must
read `clean` before the file is accepted.

### 7.2 `semantic_ids.json` (stage 2)

```json
{ "format": "agentrecx.tiger.semantic_ids.v3",
  "contract_version": "agentrecx.tiger_backend.v3",
  "num_items": 156746, "levels": 3, "dedup_levels": 1,
  "codebook_size": 256, "dedup_vocab_size": 256,
  "code_space": 16777216, "code_space_with_dedup": 4294967296,
  "pad_row": [-1, -1, -1, -1],
  "assignment": ["row 0 = [-1,-1,-1,-1] sentinel", "... 156746 real rows"],
  "collision_audit": { "...": "see section 13" },
  "utilisation": [ "... see section 14" ],
  "content_features": { "path": "item_features/item_features.npy", "sha256": "...",
                        "encoder": "sentence-transformers/sentence-t5-base", "dim": 768 },
  "quantizer": { "init": "kmeans++", "normalize_input": true, "revive_dead": true, "beta": 0.25,
                 "encoder_dims": [768, 256, 128], "latent": 64,
                 "epochs": 50, "batch_size": 4096, "lr": 1e-3 },
  "checkpoint_sha256": "..." }
```

### 7.3 `quantizer/` (stage 2)

```text
tokenizer.pt     # {"state_dict", "config"} - weights_only=True loadable; no pickled classes
tokenizer.json   # architecture + config + normalisation stats + item_features sha256
```

`tokenizer.json` records `mean`/`std` (or `BatchNorm` running stats) and the
**`item_features.npy` sha256**, so a tokenizer cannot be silently applied to a different feature
matrix.

### 7.4 `generator/` (stage 3)

```text
tiger.pt - tiger.json - layout.json - score_rule.json
```

```json
// layout.json - the token-space contract, mirrored in the adapter
{ "format": "agentrecx.tiger.token_layout.v3",
  "levels": 3, "dedup_levels": 1, "codebook_size": 256, "dedup_vocab_size": 256,
  "per_item_tokens": 4,
  "level_offsets": [0, 256, 512, 768],
  "special": { "pad": 1024, "bos": 1025, "eos": 1026 },
  "vocab_size": 1027,
  "sid_to_token": "token = level_offsets[l] + code_l",
  "item_block": "[t0,t1,t2,t3]",
  "sentinel_tokenisable": false }
```

```json
// tiger.json
{ "format": "agentrecx.tiger.checkpoint.v3",
  "layout_sha256": "...", "score_rule_sha256": "...",
  "model": { "family": "T5ForConditionalGeneration", "d_model": 256, "num_layers": 6,
             "num_heads": 4, "d_ff": 1024, "dropout": 0.1, "params": 0, "init": "random" },
  "training": { "objective": "seq2seq cross-entropy over 4 tokens + EOS",
                "optimizer": "AdamW", "lr": 5e-4, "scheduler": "OneCycleLR",
                "batch_size": 512, "epochs": 20, "bf16": true,
                "max_hist_items": 20, "in_len": 81, "tgt_len": 5,
                "examples": 0, "steps": 0, "seconds": 0.0,
                "labels_from": "agentrecx.tiger.train_exposure.v3 (train_history only)",
                "validation_used": false },
  "exposure_sha256": "...", "semantic_ids_sha256": "...",
  "determinism": { "seed": 2026, "torch": "...", "cuda": "...", "backend_git": "..." } }
```

`vocab_size = levels*K + dedup_levels*dedup_vocab_size + 3` is **derived**, never hardcoded.
`validation_used: false` is an assertion: no validation split exists inside the backend (§8), so
checkpoint selection is by a fixed epoch budget or by a criterion the backend cannot compute.

---

## 8. Train-history-only exposure, and the total absence of targets

Six layers, strongest first.

1. **Targets are not in the contract.** `EvalCohortHandoff` has no `test_targets` and no
   `validation_target`. There is no `grade_only` switch, because a switch implies a protected
   field exists; the field does not exist.
2. **The materialiser cannot name a target.** `materialize_train_exposure(cases)` reads
   `case.train_history`; the writer receives `tuple[tuple[int, ...]]` and has no `EvaluationCase`
   in scope. `validation_target` / `test_target` are unreachable by construction — the technique
   `semantic_id/dataset.py` already uses in GenRec v0.
3. **Stage isolation.** `train_exposure.jsonl` and `eval_cohort.jsonl` are separate files; the
   `train` subcommand takes only `--exposure`. Stages 1–3 never read a cohort file; only `score`
   reads `eval_cohort.jsonl`, and it receives histories plus an integer frontier count.
4. **Arithmetic leak detection.** A parity test extends the accepted `tests/test_training_parity.py`:
   the materialised exposure count must equal the accepted `train_history`-derived count **and** be
   strictly less than the count obtainable if `validation_target` were included — which differs by
   exactly 412 445. A leaked target changes the count, so a leak is detectable by arithmetic, not
   by inspection.
5. **`field_source` attestation + hashes.** The exposure file declares
   `field_source: "EvaluationCase.train_history"` and `protocol_version`; `tiger.json` stores
   `exposure_sha256`, so a checkpoint names the exact file it learned from.
6. **Adapter-side refusal.** Before scoring, the adapter verifies per row that
   `history` is a subset of `train_history + (validation_target,)` for that user — i.e. the
   test-time history is legitimate and contains no `test_target` — and refuses the run on any
   unexplained membership. The `seen` sets computed here are used **only** to compute
   `required_frontier` in L1; they are never sent to the backend as identities.

**Checkpoint selection without a backend-side target.** The backend implements **no**
Recall/NDCG, **no** validation loop and **no** validation split, so `validation_used: false` is
guaranteed true rather than merely asserted.

§8 originally described a per-epoch cadence in which AgentRec-X would evaluate `generator/epoch-N/`
externally with `validation_target` Recall@10 and pick the winner.  **That is superseded by the
Gate-B.1 registration in §17.4**, for two reasons: it selects a *production* checkpoint with a
recommendation metric, and it contradicts H5's own "no cohort metric is asserted" clause.  The
backend writes one checkpoint at the end of the configured epoch budget and the registered
selection rule is simply *that* checkpoint.  H5's "resumable checkpoint every epoch" clause is
therefore an open item (§17.4, discrepancy 1), not something the current trainer implements.

---

## 9. Keeping the external repositories' split / evaluator out

**Structural:** the backend cannot express an evaluator (it is never given targets) and cannot
express a split (it is never given raw sequences).

| External component | Why it cannot enter |
|---|---|
| amazon-genrec `spark/60_split.py`, `30_sessionize_skew.py` | the backend receives an **already-split** exposure file; no raw ingestion path exists under `backends/` |
| amazon-genrec `tiger/40_make_tokens.py` valid/test writers | the backend receives one exposure file, not per-user sequences to slice; `n_train = n - 2` is never reimplemented |
| amazon-genrec `70_eval.py`, `60_baseline.py` | beam-limited candidate-set metrics are a different protocol; `score()` returns a full-catalogue matrix with nothing for them to attach to |
| GRID `src/components/eval_metrics.py` | same, plus excluded by the file allow-list; it scores collisions as hits with no seen-masking |
| GRID `src/data/loading/*` (TFRecord, pre-split dirs) | replaced by the handoff JSONL; TensorFlow is not a backend dependency |
| GRID `src/train.py` / `inference.py` (Hydra/Lightning/DDP/restart) | not ported; the backend CLI is four subcommands |

**Test-enforced** (`tests/test_backend_boundaries.py`, in the style of the existing AST guards
that keep `recommendation/agent` free of `rag`/`memory`/`reranking`):

```text
T1  no file under backends/ imports recommendation.*
T2  no file under recommendation/backends/ imports torch, transformers,
    sentence_transformers, lightning, hydra, tensorflow, sklearn
T3  no file under recommendation/backends/ imports recommendation.evaluation.*
    or recommendation.semantic_id.*        (the adapter must not reuse GenRec v0)
T4  file allow-list: backends/ contains no path matching */eval*|*/metric*|*/split*
    other than the named retrieve.py
T5  no handoff artifact declares a field named target/label/valid/test, and
    no seen-item identity appears in eval_cohort.jsonl
T6  every file under backends/tiger_public/ appears in PROVENANCE.md
T7a no occurrence of "parent_asin" anywhere under recommendation/backends/
T7b no occurrence of "parent_asin" anywhere under backends/
T8  the score rule is single-sourced: backends/.../scoring.py is the only
    module computing item scores, and no module renormalises over valid children
```

`T7a`/`T7b` resolve the strongest form of the boundary: the adapter never touches
`parent_asin`, and `parent_asin` is permitted only in L0 and L1 — which are outside both guarded
trees. `T8` keeps the frozen score rule single-sourced so beam search, branch-and-bound and
exhaustive scoring cannot diverge in ranking semantics.

---

## 10. Batched, streamed, sharded output

No monolithic `[20000, 156747]` array is required or produced. The frozen evaluator already
consumes batches (`evaluate_batched`, `DEFAULT_BATCH_SIZE = 256`), so the backend streams
`ScoreBatch` objects and the adapter feeds them straight through.

```text
(a) STREAMING (preferred)
    adapter spawns `tiger_public.cli score --stream`
    -> length-prefixed JSON headers + raw float32 blocks on stdout
    -> adapter reads one ScoreBatch, converts it, yields to evaluate_batched
    Peak memory: O(batch x num_items), the same as today's evaluator.

(b) SHARDED ARTIFACT (resumability / audit)
    scores/part-00000.npz ... part-000NN.npz
      scores              float32 [batch_i, num_items + 1]
      reachable           bool    [batch_i, num_items + 1]
      reachable_real_count int32  [batch_i]
      case_ids            int64   [batch_i]
    scores/index.json
      {"format": "agentrecx.tiger.scores_index.v3", "num_items": 156746,
       "cohort_size": 20000, "batch_size": 256, "shards": [...],
       "row_offsets": [...], "total_rows": 20000,
       "status": "APPROXIMATE|CERTIFIED", "certification": {...}}
```

`case_ids` is the **receipt**: row *i* of shard *s* corresponds to `case_ids[i]`, and the adapter
asserts the concatenated `case_ids` equals the frozen cohort order exactly. A mismatch is a hard
refusal, not a warning.

Adapter conversion rules:

| Step | Rule |
|---|---|
| **S1** | `scores[:, 0] = 0.0`, `reachable[:, 0] = False`. The PAD column exists only because the evaluator's contract is `[batch, num_items + 1]`; it is masked positionally and its value is never read. |
| **S2** | Re-index from backend row `r` to item-id column `item_id` using `CatalogueHandoff.item_ids` — **not** by assuming contiguity. A permuted `item_ids` is caught here. No `parent_asin` is consulted; this is a column alignment, not an identity translation. |
| **S3** | **Refuse on an empty reachable frontier:** if any row has `reachable_real_count == 0`, the run is refused (§16). No tail score is constructed for such a row. |
| **S4** | Otherwise, unreachable items are filled with the documented tail score `min(scores[:, 1:][reachable]) - 1.0`, which exists because S3 guaranteed at least one reachable real item. Strictly below every reachable item — the **same** convention Phase 5's retrieval arms and Step 2's GenRec arm already use. |
| **S5** | `reachable` is diagnostics + certification input only; it is **never** handed to the evaluator as a mask. |
| **S6** | `case_ids` must equal the frozen cohort ordering; otherwise refuse. |
| **S7** | The adapter never sorts, never slices a top-k, never applies `seen`, and never returns `parent_asin`. The frozen arm calls `evaluate_batched` unchanged; seen masking, tie-breaking, ranking and metrics remain entirely evaluator-owned. |

Reported block (diagnostics only; `ArmResult` / `arms.json` shape unchanged):

```json
"backend": { "kind": "tiger_public", "contract_version": "agentrecx.tiger_backend.v3",
             "exposure_sha256": "...", "semantic_ids_sha256": "...",
             "layout": { "levels": 3, "dedup_levels": 1, "vocab_size": 1027 },
             "score_rule": { "version": 1, "eos_in_score": false,
                             "softmax": "full_vocabulary", "child_renormalisation": false },
             "retrieval_status": "APPROXIMATE",
             "scored_share": 0.0042, "expanded_nodes": 0, "beam": 10, "prefix_budget": 512,
             "reachable_real_count": { "min": 1, "max": 0, "mean": 0.0 },
             "invalid_generation_rate": 0.0 },
"protocol": { "evaluator": "recommendation.evaluation.batched",
              "masking": "evaluator-owned", "tie_break": "score desc, item_id asc",
              "cohort_seed": 20260201, "cohort_size": 20000 }
```

---

## 11. Milestones

```text
Step 2.2  TIGER Backend Adapter Specification          design only
Step 2.3  Backend skeleton + handoff bridge            contracts / schemas / process isolation / tests - NO ML
Step 2.4  Content embedding + RQ-VAE + collision-free Semantic IDs
                                                       Sentence-T5 - GRID-inspired quantizer - dedup - SID audits - NO TIGER
Step 2.5  TIGER generator + trie-constrained generation
                                                       HF T5-style - smoke - full GPU training - NO canonical benchmark claim
Step 2.6  Correctness-bounded catalogue retrieval + canonical benchmark
                                                       APPROXIMATE and CERTIFIED modes - shared evaluator only - 20k frozen cohort
Step 2.7  TigerCandidateSource / Agent integration     only after 2.6 establishes value
```

Catalogue retrieval cannot precede TIGER's existence; `retrieve.py` belongs to 2.6, after a
generator exists to retrieve from. Attribution:

| Step | Tests exactly | Does not test |
|---|---|---|
| 2.4 | representation + tokenization quality (coverage, dedup, dead codes, `id_entropy`) | generator training, retrieval, benchmark |
| 2.5 | generator training (loss curve, code accuracy, invalid-generation rate, smoke retrieval) | retrieval completeness, canonical metrics |
| 2.6 | retrieval correctness (APPROXIMATE vs CERTIFIED) + the canonical benchmark | representation or generator architecture |

| Milestone | Deliverables | Gate | Non-goals |
|---|---|---|---|
| **2.2** (this doc) | `docs/TIGER_BACKEND.md` + AGENTS/ARCHITECTURE/handoff cross-references | document review | no code, no training |
| **2.3** (done) | `contracts.py`, `io.py`, `cli.py`, `recommendation/backends/tiger_backend.py`, `materialize_tiger_backend.py`, boundary + adapter tests | all 15 smoke checks; 25 boundary guards; 42 adapter tests | no ML |
| **2.3** | `contracts.py`, `io.py`, `recommendation/backends/tiger_backend.py`, `materialize_tiger_backend.py`, `test_backend_boundaries.py` (T1–T8), `test_tiger_backend_adapter.py`; **all ML stages stubbed** | **G1, G2, G3, G6, G8, G9, G10, G12**; the full suite stays green | no features, no quantizer, no TIGER, no retrieve |
| **2.4** (implemented; full run deferred) | `features.py`, `quantizer.py`, `dedup.py` | **G4, G5, G5b, G11** pass; **H2** pass on 2 000 real items; **H3** blocked at full scale by one frozen design choice (see the note below) | no TIGER training; no evaluator/fusion change; no ANN |
| **2.4F** (done) | `experiments/audit_tiger_sid.py`, `backends/tiger_public/REPRODUCIBILITY.md`, `requirements-ml.txt`, transactional `fit-sid`, encoder-revision pinning | read-only audit on the 2,000-item artifact; all suites green | no algorithm, schema, protocol or evaluator change |
| **2.5** | `scoring.py`, `tiger.py`, `trie.py` | **G5b, G7**, then **H5, H6** | no canonical benchmark claim; no cohort metric asserted |
| **2.6** | `retrieve.py` (APPROXIMATE + CERTIFIED), benchmark wiring | **G13, G14, G15, G16**, then **H7, H8** | no representation/generator change; no re-tuning against the cohort |
| **2.7** | `TigerCandidateSource`, `CandidateSource` enum, fusion experiment | separate pre-registered controlled experiment | no fusion claim without holding every other source constant |

---

## 12. PAD is structurally impossible as a SID

```text
PAD_SENTINEL  = -1
levels        = 3, dedup_levels = 1
pad_row       = (-1, -1, -1, -1)
assignment[0] = (-1, -1, -1, -1)          # NOT a tokenisable SID
real item SIDs = assignment[1:]           # every digit in [0, K)
```

| # | Asserted property |
|---|---|
| **P1** | `assignment[0] == pad_row == (PAD_SENTINEL,)*levels`; every entry negative |
| **P2** | for every real row `r >= 1`, every digit is in `[0, codebook_size)` |
| **P3** | **the PAD row cannot be tokenised**: `TokenLayout.tokenise(pad_row)` raises rather than producing `[-1,...]` or any token; the tokeniser rejects negative digits as a precondition |
| **P4** | the resolver **always ignores `assignment[0]`** — its lookup table is built from rows `1..num_items` only, so `resolve((-1,-1,-1,-1))` returns `item_id=None, invalid=True`, a fail-closed miss and not a PAD hit |
| **P5** | no history contains `0` (validated on materialisation and on read); a `0` is a hard refusal naming the user |
| **P6** | the evaluator's PAD column is contract-only (`scores[:,0] = 0.0`) and PAD never appears in any case's top-k |

Five independent isolations:

| # | Isolation | Where |
|---|---|---|
| **I1** | **Two vocabularies.** Item vocabulary: `0 = PAD`, real `1..N`. Token vocabulary: codes occupy `[0, levels*K)`, the dedup level `[levels*K, levels*K + dedup_vocab_size)`, specials strictly **above** the whole code space (`pad = 1024, bos = 1025, eos = 1026` for L=3, K=256). Token `0` is the legal code `(level 0, code 0)`. | `TokenLayout` |
| **I2** | Reserved row 0 is a **sentinel**, excluded from every statistic. | `SemanticIdArtifact` reader |
| **I3** | **No PAD row in the features.** `item_features.npy` has exactly `num_items` rows; `pad_row_present: false` asserted on read. | stage 1 |
| **I4** | PAD column is contract-only in scores. | adapter S1 |
| **I5** | Histories can never contain `0`. | materialiser + `io.py` |

### 12.1 Dedup token-space overflow

```text
pre_dedup.largest_group <= dedup_vocab_size
```

`dedup_vocab_size` is declared in the artifact and in `layout.json`. Stage 2 computes
`pre_dedup.largest_group` **before** assigning dedup digits, then either fits it or refuses:

```text
if pre_dedup.largest_group > dedup_vocab_size:
      REFUSE the build, exit non-zero,
      message = "largest pre-dedup collision group N exceeds dedup vocab K;
                 re-fit with a larger codebook, more levels, or a wider dedup level"
```

No silent wrap, no modulo, no truncation, and no fallback to a shared SID. A 256-way dedup level
genuinely cannot disambiguate a 300-member group, and pretending otherwise would recreate the v0
defect with extra steps. The build **is** authorised to expand the dedup level's range (e.g.
`dedup_vocab_size = 1024`) and must say so in `layout.json` and `collision_audit`; the choice is
made from the data, recorded, and never tuned against the evaluation cohort.

---

## 13. Collision audit: pre-dedup and post-dedup both preserved

Collapsing the audit to only the final collision-free assignment would discard the Semantic-ID
quality signal. Both blocks are retained:

```json
"collision_audit": {
  "pre_dedup": {
    "collision_groups": 0,
    "items_in_collision": 0,
    "collision_rate": 0.0,
    "largest_group": 1,
    "distinct_sids": 0
  },
  "post_dedup": {
    "collision_groups": 0,
    "items_in_collision": 0,
    "collision_rate": 0.0,
    "largest_group": 1,
    "distinct_sids": 156746
  },
  "dedup_vocab_size": 256,
  "dedup_scheme": "collision_ordinal",
  "first_member_zero": true,
  "largest_group_le_dedup_vocab": true,
  "overflow_refusal_triggered": false
}
```

| Concern | Block used | Rationale |
|---|---|---|
| **Dedup overflow check** (§12.1) | `pre_dedup.largest_group` | the overflow is a property of the quantizer's code space, *before* the dedup digit masks it |
| **Semantic-ID quality diagnostic** | `pre_dedup.*` | how well the learned codes separate items — the quantity generative retrieval quality actually depends on |
| **Final identity uniqueness** | `post_dedup.*` | the guarantee the resolver and the evaluator rely on |

**Acceptance:**

```text
post_dedup.collision_groups == 0
post_dedup.distinct_sids     == num_items
```

so final uniqueness is still guaranteed; the pre-dedup block only makes the *quality cost* of
achieving it visible. A run whose `pre_dedup.collision_rate` is very high is not rejected by this
gate — it is **reported**.

---

## 14. Utilisation reporting and the dead-code hard stop

An earlier draft asserted that a `>20 % dead-code` gate was "the objective test the GenRec v0
tokenizer's 33/10/0 dead codes fail". **That was arithmetically false** — `33/256 = 12.9 %`,
`10/256 = 3.9 %`, `0/256 = 0 %` — and the claim is withdrawn. The threshold is **not** moved to
force the historical baseline to fail, and nothing in this specification claims that GenRec v0
fails the gate.

The gate is a *catastrophic-collapse* detector, not a quality-superiority criterion:

```text
dead_codes > 20% of a level
    -> catastrophic-collapse HARD STOP (refuse the build)
```

Reported, **without** becoming an arbitrary superiority criterion:

| Field | Meaning |
|---|---|
| `used_codes` | distinct codes selected at this level |
| `dead_codes` | `codebook_size - used_codes` |
| `frac_layer_coverage` | `used_codes / codebook_size`, as GRID reports it |
| `id_entropy` | `Categorical(probs = per-code frequencies).entropy()` |

```json
"utilisation": [
  { "level": 0, "codebook_size": 256, "used_codes": 0, "dead_codes": 0,
    "frac_layer_coverage": 0.0, "id_entropy": 0.0, "collapsed": false },
  { "level": 1, "...": "..." },
  { "level": 2, "...": "..." }
]
```

A level is flagged `collapsed: true` **only** on the `>20 %` rule. Every other reading of these
numbers is reported as an observation in `run.json`, not as a pass/fail.

**Any stricter threshold must be justified and pre-registered independently of the observed
AgentRec-X benchmark.** A stricter dead-code or entropy bound may be added to H3 only as a
separate, dated, pre-registered amendment that names the criterion before any Step-2.6 benchmark
number exists, and never by back-fitting a threshold to v0's or TIGER's observed counts.

---

## 15. Retrieval statuses: APPROXIMATE and CERTIFIED

A bounded prefix budget followed by a tail score **mixes generator quality with search
truncation** — the methodological limitation that froze GenRec v0. The two modes are named,
reported, and never conflated.

### 15.1 APPROXIMATE — diagnostic only

```text
constrained beam(k) + bounded trie expansion (prefix_budget B) + tail score
```

Must report `scored_share`, `expanded_nodes`, `beam_width`, `prefix_budget`. Its
Recall/NDCG/HR are **diagnostic only** and are labelled as such in `run.json` and in every table
they appear in. Step 2.5's smoke uses this mode.

**Empty-frontier refusal.** The tail score `min(reachable_scores) - 1` is defined only if at least
one real item is reachable. Therefore:

```text
reachable_real_count == 0 for any row
    -> HARD REFUSAL for that case and for the run
    -> no tail score is constructed, no NaN/Inf fallback, no filled sentinel
    -> the run fails and reports the offending case_ids
```

The backend emits `reachable_real_count` per row explicitly so this is checkable without
inference.

### 15.2 CERTIFIED — required for the canonical comparison

Required before any TIGER number is compared with SASRec or Two-Tower. It must prove that **the
items needed for the evaluator's top-K are complete**, by one of:

```text
(a) full-catalogue exact scoring
(b) exact branch-and-bound over the semantic-ID trie with an admissible upper bound
```

Both rank items under the **identical frozen item-score rule of §5.1**. The bound is admissible
against that rule (§5.3). The search maintains a max-heap keyed by `S(p)` descending and a running
threshold `T` = the current `required_frontier`-th best **completed** item score, terminates a
branch exactly when `S(p) < T`, and terminates globally when the best remaining prefix has
`S(p) < T`.

```json
{ "status": "CERTIFIED",
  "method": "branch_and_bound",
  "score_rule_version": 1,
  "admissible_bound": "full-vocabulary log-probability prefix sum <= 0 for all continuations",
  "k_prime_field": "required_frontier",
  "threshold_final": 0.0,
  "prefixes_expanded": 0,
  "nodes_pruned_by_bound": 0,
  "certified_items_per_case": { "min": 0, "max": 0, "mean": 0.0 },
  "certificate_holds_for_all_cases": true,
  "budget_exhausted_cases": 0 }
```

`budget_exhausted_cases > 0` means the adapter **automatically downgrades** the run to
APPROXIMATE, rewrites `retrieval_status` in `run.json`, and the report says so. A run that cannot
certify within its resource budget is reported as approximate rather than presented as the
definitive TIGER Recall@10.

### 15.3 The per-case certification frontier request

`required_frontier` is an **AgentRec-X-owned integer**, one per case, computed in L1 as:

```text
required_frontier(case) = K_max + |seen(case)|
```

where `K_max = max(k_values) = 20` for this protocol and `seen(case)` is the evaluator's own seen
set for that case (`EvaluationCase.test_seen`). It is the worst case in which every already-seen
item outranks the entire frontier.

Ownership split:

```text
AgentRec-X owns:        seen set - masking - final ranking - K
TigerBackend receives:  required_frontier = K_max + |seen|   (an integer, never identities)
TigerBackend proves:    the raw top-required_frontier frontier is COMPLETE
```

The backend never learns which items are seen, cannot reconstruct the set from the count, and
makes no claim about the post-masking order. The evaluator applies the mask and ranks exactly as
it does today.

**`required_frontier` is not assumed small.** The evaluator's `seen` set comes from the full
`test_history`, which may be far longer than the backend's truncated model-input history
(`max_hist_items = 20`). A user with 60 interactions has `|seen| = 59` regardless of how much of
that history the model reads. `required_frontier` is therefore bounded by `K_max + |test_history|`,
with no claim of a small constant.

Because of that, the frontier request is **measured before the full CERTIFIED run**:

```text
H7 pre-step (REQUIRED, cheap, AgentRec-X-side, no backend call):
    compute required_frontier for all 20 000 cohort cases
    record:  min_required_frontier
             max_required_frontier
             mean_required_frontier
             p95 / p99
             histogram by bucket
    persist into eval_cohort.jsonl.required_frontier_stats and run.json

if max_required_frontier or mean_required_frontier makes certification
   infeasible within the resource budget:
        report the run as APPROXIMATE
        do NOT present it as the definitive TIGER Recall@10
```

### 15.4 Reported statuses, never conflated

```text
run.json:
  "retrieval_status": "APPROXIMATE" | "CERTIFIED"
  "comparison_eligible": true | false        # true only when CERTIFIED
```

`comparison_eligible: false` for every APPROXIMATE run, and no table may place an APPROXIMATE
TIGER row beside SASRec / Two-Tower without that label.

---

## 16. Gates

### 16.1 Smoke gate (no GPU; before any GPU job)

| Gate | Condition | Step |
|---|---|---|
| **G1** | the full suite is unchanged with the backend venv absent; `test_backend_boundaries.py` (T1–T8) green | 2.3 |
| **G2** | `materialize_tiger_backend.py --limit 2000` writes all files; `item_id` span `1..2000`; all exposure rows satisfy `1 <= id <= 2000`; PAD never appears; **no `parent_asin` in any handoff artifact** | 2.3 |
| **G3** | negative adapter tests: misaligned `item_ids`, wrong `num_items`, a `0` in a history, missing `sha256`, `CONTRACT_VERSION` mismatch, `pad_row_present: true`, a `scores` shape different from `[batch, num_items+1]`, and a `parent_asin` argument rejected at the adapter boundary | 2.3 |
| **G6** | shard/stream `case_ids` receipt equals the cohort order; `row_offsets` sum to `total_rows`; shape-only pass with stubbed scores | 2.3 |
| **G8** | the adapter's stream/shard feeds `evaluate_batched` on 32 cases producing finite Recall@10 / NDCG@10 / HR@10 **with no evaluator change** | 2.3 |
| **G9** | round-trip `item_id -> row -> SID -> token -> SID -> row -> item_id` is the identity for all 2 000 items; a **shuffled** `item_ids` table is detected, not accepted | 2.3 |
| **G10** | P1–P6 of §12 in one test | 2.3 |
| **G12** | `T6` green: every backend file is in `PROVENANCE.md` | 2.3 |
| **G4** | 2 000 items embed on **CPU** in < 2 min; `empty_text_items` / `truncated_items` reported; `nan_inf_scan: clean` | 2.4 |
| **G5** | RQ-VAE on 2 000 items CPU in < 1 min; `levels=3, dedup=1`; `post_dedup.collision_groups == 0`; `post_dedup.distinct_sids == 2000`; **`pre_dedup` block present and non-degenerate**; PAD sentinel intact; `largest_group_le_dedup_vocab` true | 2.4 |
| **G5b** | the synthetic overflow case: a hand-built assignment whose `pre_dedup.largest_group` exceeds `dedup_vocab_size` **refuses the build** with the documented message, and does not wrap or share a SID | 2.4 |
| **G11** | the smoke run repeats bit-identically on a second invocation (same seeds, same hashes) | 2.4 |
| **G13** | a synthetic hand-computable SID tree where the true top-K is known: branch-and-bound returns exactly that top-K; the bound prunes at least one node; forcing `prefix_budget=1` downgrades the run to APPROXIMATE automatically | 2.6 |
| **G14** | `comparison_eligible` is `false` for every APPROXIMATE run and `true` only for a CERTIFIED run whose `certificate_holds_for_all_cases` is true | 2.6 |
| **G15** | **scoring-rule consistency:** beam search, branch-and-bound and exhaustive scoring produce the *same relative order* on a small catalogue where all three are computable; and a test asserts no module renormalises over valid children (`T8`) | 2.6 |
| **G16** | **empty-frontier refusal:** a synthetic case with zero reachable real items fails the run with the offending `case_id`; no NaN, no Inf, and no fabricated tail score appears anywhere in the output | 2.6 |

### 16.2 Full-GPU gate (AutoDL RTX 4090; not the CPU-only 7 GB development host)

| Gate | Condition | Step | Estimate |
|---|---|---|---|
| **H1** | frozen inputs: `mappings_sha256 == dca7815a...`, `sequences_sha256 == d3d83426...`, catalogue 156 746, users 412 445, `field_source == EvaluationCase.train_history`, exposure arithmetic parity (§8.4) | 2.4 | — |
| **H2** | 156 746 x 768 float32 is about **481 MB**; `nan_inf_scan: clean`; `empty_text_items` / `truncated_items` recorded | 2.4 | ~4–10 min |
| **H3** | `post_dedup.collision_groups == 0`; `post_dedup.distinct_sids == 156 746`; coverage `1.000`; `largest_group_le_dedup_vocab` true; **`pre_dedup` block recorded in full**; per-level utilisation recorded; **hard stop only on `dead_codes > 20 %` of a level** — no comparative threshold applied to any other reading | 2.4 | ~5–15 min |
| **H4** | `layout.json` + `score_rule.json` written, hashed, and re-tested against the adapter **before** TIGER training starts, so neither the token space nor the scoring rule can drift mid-project | 2.4 | — |
| **H5** | TIGER training, `d_model=256, layers=6`, batch 512, bf16, `epochs=20`, `lr=5e-4`, seed 2026, **2 263 252 examples** (Gate-B.1 registration; §17.4); `validation_used: false`. The "1 850 807 examples" and "resumable checkpoint every epoch" clauses of the original H5 text are superseded/parked — see §17.4 | 2.5 | ~3–6 h |
| **H6** | `invalid_generation_rate` and APPROXIMATE `scored_share` measured and labelled diagnostic | 2.5 | — |
| **H7** | **pre-step:** `min/max/mean/p95/p99_required_frontier` computed and recorded **before** any CERTIFIED attempt; then CERTIFIED retrieval over the frozen 20 k cohort with `certificate_holds_for_all_cases` true within budget, otherwise automatic APPROXIMATE downgrade | 2.6 | diagnostics: minutes; certification: hours, budget-dependent |
| **H8** | canonical benchmark via the unchanged evaluator, `--cohort 20000`; `run.json` carrying the whole hash chain plus `retrieval_status` and `comparison_eligible` | 2.6 | ~22 min |

**Hard stops:** refuse if the dedup-overflow bound fails; refuse if the exposure hash differs from
the smoke definition; refuse on any zero-reachable row; refuse if `run.json` would lack
`exposure_sha256` + `semantic_ids_sha256` + `layout` + `score_rule` + `retrieval_status`. No
threshold, epoch, beam, budget or frontier K is ever tuned against the 20 000-user cohort.

---

## 17. Target architecture

```text
AgentRec-X frozen artifacts
  owns: parent_asin <-> item_id - PAD = 0 - split - targets - cohort - evaluator
        |
        |   L1 materialisation (experiments/materialize_tiger_backend.py)
        |   reads parent_asin ONLY here, for text composition + required_frontier
        |   writes: item_ids - pre-composed text - train_history - cohort histories
        |            + required_frontier (an integer per case)
        v
TigerBackendAdapter                              [ item_id ONLY - no parent_asin, ever ]
  recommendation/backends/tiger_backend.py
  stdlib + numpy - subprocess + filesystem - no ML import - no identity semantics
        |
        v
content embedding        Sentence-T5-base, frozen                      stage 1
        v
RQ-VAE                   GRID-inspired: normalise, k-means++, beta 0.25  stage 2
        v
collision-free Semantic IDs   dedup digit + overflow refusal
                              pre_dedup + post_dedup audit retained      stage 2
        v
TIGER                    random-init HF T5 seq2seq                      stage 3
        v
catalogue-constrained / certified retrieval                              stage 4
  scoring.py defines the ONE frozen item-score rule:
      score(i | history) = SUM_l log P_full-vocab(t_l | history, t_<l),  EOS excluded
  trie.py is a SEARCH constraint only; it never renormalises probabilities
  APPROXIMATE (beam + bounded expansion + tail)  -> diagnostic only
  CERTIFIED     (full exact | branch-and-bound with admissible prefix bound)
                -> proves the raw top-required_frontier frontier is complete
        v
raw item scores          streamed/sharded ScoreBatch, keyed by opaque item_id
                         zero-reachable row -> hard refusal, no NaN/Inf fallback
        v
TigerBackendAdapter      [ item_id ONLY - alignment + fill, no ranking, no masking ]
        v
AgentRec-X shared evaluator
  owns: seen masking - tie-break - full-catalogue ranking - Recall / NDCG / HR
  parent_asin is re-attached HERE, outside the adapter, only if a consumer needs it
```

---

## 17.1 Step-2.4 implementation note: the learning rate

Step 2.4 froze `learning_rate = 1e-3` (the published GRID value). Measured on real
`sentence-t5-base` embeddings, that setting **collapses the later residual levels**:

| setting | used codes per level (K = 256) | verdict |
|---|---|---|
| `lr = 1e-3` (as specified) | 256 / 20 / 16 | levels 1 and 2 are 92 % and 94 % dead |
| `lr = 3e-4` | 256 / 79 / 60 | levels 1 and 2 are 69 % and 77 % dead |
| `lr = 1e-4` | 256 / 100 / 68 | 1 dead code at level 0 |

Every setting was measured on the same 2 000-item real-embedding artifact, 60 epochs,
batch 512, seed 2026, with k-means++ seeding and dead-code revival enabled. The measurement is
reported rather than tuned away: `--learning-rate` is an explicit CLI argument, the frozen
value remains the default, and the value actually used is recorded in
`semantic_ids.json.quantizer`.

**The consequence for H3.** H3 requires no level to exceed 20 % dead codes. No measured setting
satisfied that at the smoke scale, and the closest (`lr = 3e-4`) still leaves levels 1 and 2 at
69 % and 77 % dead. Two distinct questions are therefore open and are **not** resolved by
tuning here:

1. whether residual levels 1–2 are useful at all on this embedding distribution, given that
   level 0 already reconstructs to MSE < 0.001; and
2. whether H3's 20 % threshold is the right health criterion for a *hierarchical* code, where
   coarse-to-fine levels are expected to differ in occupancy.

Both are pre-registered questions for the full-catalogue run, not adjustments to make a gate
pass. The `--allow-dead-codes` flag exists so a small-scale mechanism check can proceed while
recording the waiver in the artifact; it must never be used for a reported result.

## 17.2 Production full-run configuration (PRE-REGISTERED, Step 2.4F)

Two configurations exist and must not be confused. The smoke configuration exists to prove the
mechanism on a CPU host; the production configuration is what a 156,746-item artifact is built
with, and it is frozen here **before** any recommendation-quality benchmark exists.

### Production configuration — 156,746 items

```text
encoder           sentence-transformers/sentence-t5-base
encoder_revision  fc5d4628481afbbaaacd7af6bb07cf9d3865f781   (immutable snapshot SHA-1)

levels            3
codebook_size     256
latent_dim        64
encoder_dims      768,256,128
beta              0.25

epochs            50
batch_size        4096
learning_rate     3e-4

normalize_input   true
revive_dead       true
dedup_vocab_size  256
seed              2026
allow_dead_codes  false
```

### Smoke configuration — 2,000 items (NOT the production configuration)

```text
epochs            60
batch_size        512
learning_rate     3e-4
encoder           smoke  (or sentence-t5-base for the real-embedding smoke)
```

The `epochs = 60` / `batch_size = 512` values were the **2,000-item CPU smoke** settings. They
are recorded here so a later reader cannot mistake the smoke run for the production run, and so
the production numbers are not quoted from a configuration that never produced them.

### The learning-rate change, and its justification

The production learning rate is **changed from the earlier specification of `1e-3` to `3e-4`**.
The change is made **before any recommendation-quality benchmark**, and rests only on
Step-2.4 tokenizer-health diagnostics — never on a Recall/NDCG number, because no such
measurement exists for this backend yet:

| learning rate | codes used per level (K = 256, 2,000 real items) |
|---|---|
| `1e-3` (earlier specification) | 256 / 20 / 16 |
| `3e-4` (production) | 256 / 79 / 60 |
| `1e-4` | 256 / 100 / 68 |

`1e-3` leaves levels 1 and 2 at 92 % and 94 % dead. `3e-4` is the value that keeps all three
levels functional without any search beyond these three measured points.

**No further hyperparameter search is performed.** `epochs`, `batch_size`, `latent_dim`,
`codebook_size`, `beta` and `dedup_vocab_size` are frozen above and are not to be tuned against
the evaluation cohort or against any health metric.

### What the Step-2.4F audit established about levels 1 and 2

The read-only audit (`experiments/audit_tiger_sid.py`) measured how much each residual level
actually contributes, which the single aggregate `reconstruction_loss` cannot show. On the
2,000-item real-embedding artifact (`lr = 3e-4`, 60 epochs):

| reconstruction | MSE against the normalised target |
|---|---|
| level 0 only | 0.000 751 |
| levels 0 + 1 | 0.000 710 |
| levels 0 + 1 + 2 | **0.000 694** |
| (zero prediction, for scale) | 0.001 302 |

Levels 1 and 2 together improve the reconstruction by **7.6 %** over level 0 alone. Residual
norms fall 0.2319 → 0.0644 → 0.0527 → 0.0461 across the three levels, so most of the latent
signal is consumed by level 0. Occupancy is 256 / 79 / 60 codes with entropy at 0.95 / 0.77 /
0.73 of the per-level maximum.

This is the measurement behind the two open questions in §17.1, and it is why the production
configuration above is pre-registered rather than tuned: whether levels 1-2 are worth their
depth is a question for the full-catalogue run to answer, not a reason to adjust a gate now.

## 17.3 Training exposure vs evaluation cohort (the two populations)

Two populations are materialised for the backend and they are **deliberately not the same**:

```text
training exposure   every eligible user's train_history   -> the ordinary training corpus
evaluation cohort   a deterministic --cohort subset       -> what the evaluator scores
```

They were coupled: `--cohort` sized *both*, so `--cohort 2000` (the default) shipped a
**2,000-user** training corpus for a **412,445-user** catalogue. That is invisible in a metrics
table and visible only in the handoff populations, which is why the manifest now records them.

### Production path — `--limit` absent

```text
all eligible cases --+--> case.train_history --> train_exposure   (412,445 users)
                     |
                     `--> deterministic --cohort selection --> eval_cohort  (20,000 cases)
```

`--cohort` sizes **only** the evaluation cohort. Training exposure is every eligible user and
does not depend on it.

### Smoke path — `--limit N`

The full histories reference item ids outside a reduced smoke catalogue, so a full-history
artifact would be incoherent. The remap is therefore applied to the evaluation selection, and
**both** the exposure and the cohort come from the same remapped, usable cases. In smoke mode the
cohort also defines the source cases used for item remapping.

### Verifying a production handoff

The manifest carries an additive `populations` block, so the distinction is checkable by
inspection without changing any artifact schema:

```text
eligible_users        412445
train_exposure_users  412445
eval_cohort_cases      20000
catalogue_items       156746
exposure_field_source EvaluationCase.train_history
smoke_remapped        false
```

### What the materializer can and cannot leak

Every exposure row is **byte-identical to that case's `train_history`**, so the materializer
cannot introduce a validation or test target — it passes the history through unchanged. A target
item may still *recur* inside a history, because a repeat purchase is a legal event and is
documented in `PHASE5_HANDOFF.md` §8; that recurrence is not materializer leakage.

The cohort artifact carries `case_id`, `history` and `required_frontier` only. No target key and
no canonical identity appears in any handoff artifact.

## 17.4 Step 2.5 Gate B.1: the frozen generator-example count, and the registered configuration

**Registered and frozen: 2 263 252 generator examples.**

The Gate-B completion report transposed two of the three length sums. This section records the
correction, the measurement that establishes it, and the guard that keeps it from recurring.

### The three sums, kept distinct

Measured by streaming the accepted production exposure
(`runs/tiger_backend_handoff_prod/train_exposure.jsonl`, 412 445 rows, 32 MB) — one row in memory
at a time, no example list built, no training:

| quantity | expression | value |
|---|---|---|
| train-history item **occurrences** | `sum(len(row))` | 2 675 697 |
| **next-item transitions = generator examples** | `sum(max(0, len(row) - 1))` | **2 263 252** |
| GenRec-v0 / Two-Tower pairs | `sum(max(0, len(row) - 2))` | 1 850 807 |

`2 675 697` is **not** an example count and must never be reported as one. It is the number of
train-history item occurrences, i.e. how many items appear across the 412 445 histories.

### Why 2 263 252 is the generator's count

`build_examples` (Gate B) emits **one example for every adjacent target position** of a history of
`n` items, i.e. `n - 1` examples, each with source
`train_history[max(0, k - max_hist_items):k]` and target `train_history[k]`. Summed over the
accepted exposure that is exactly 2 263 252, and it is also the transition count AgentRec-X's own
parity test derives (`tests/test_training_parity.py`: `2 263 252 - 412 445 = 1 850 807`). The
exposure artifact's own record agrees: `train_exposure.json` declares `"examples": 2263252`, and
the handoff manifest declares `train_exposure_examples: 2263252`.

H5's `1 850 807` is the GenRec-v0 convention (`semantic_id/dataset.py` iterates from position 2)
and is the Two-Tower pair count from `docs/MODEL_EXPANSION_HANDOFF.md` §3. It is a *different*
objective, not a different reading of the same one: it drops the first transition of every user.
Gate B implements the next-item objective, so H5's figure is superseded by the registration here.

**Guards.** `tests/test_backend_boundaries.py` pins all three sums, the users count and the
catalogue size against the artifact (skipping with a stated reason when the handoff is not
materialised), and `experiments/smoke_tiger_step25.py --report-production` streams and prints the
same three sums, refusing if the transition count differs from 2 263 252.

### Registered production configuration (Gate C input)

```text
example semantics          train_history only; one next-item example per adjacent target
                           position; expected production examples = 2 263 252
max_hist_items             20
d_model                    256
num_layers                 6
num_heads                  4
d_ff                       1024
dropout                    0.1
batch_size                 512
bf16                       true  (CUDA autocast; a CPU run reports fp32 regardless)
epochs                     20
learning_rate              5e-4
seed                       2026
validation_used            false
optimizer                  AdamW            (recorded as "AdamW"; torch.optim.AdamW, defaults)
scheduler                  OneCycleLR       (max_lr = learning_rate,
                                             pct_start = warmup_fraction = 0.05,
                                             total_steps = epoch_budget *
                                             ceil(examples / batch_size))
weight_decay               0.0
gradient clipping          clip_grad_norm_(max_norm = 1.0); disabled only if max_grad_norm <= 0
decoder_start_token_id     layout["special"]["bos"], read from the accepted layout (1025 today)
label padding              none: every target is exactly per_item_tokens + 1 long, so no label is
                           padded. The HF loss is CrossEntropyLoss(ignore_index=-100), so if a
                           padded label ever appeared it would be ignored rather than learned as
                           a token - but no code path produces one.
checkpoint cadence         1 checkpoint, written once at the end of the configured epoch budget
final-checkpoint selection the final checkpoint. No validation or test recommendation metric is
                           consulted, because the backend computes none.
```

**Two discrepancies are reported here rather than silently resolved, and neither is a
metric-driven tuning decision.** Both must be settled before Gate D.

1. **H5's "resumable checkpoint every epoch" is not implemented.** §8 describes the backend
   writing `generator/epoch-0007/` per epoch for external selection; the implementation writes one
   final checkpoint (`tiger.pt`, `tiger.json`, `layout.json`, `score_rule.json`). The trainer
   *accepts* `resume_state` (`optimizer`, `scheduler`, `epoch`, `history`) but the CLI has no
   `--resume` and no per-epoch cadence. Selecting a checkpoint by validation Recall@10 (§8)
   contradicts "no validation metric may select a production checkpoint", and the backend has no
   evaluator by construction. The registered rule above is therefore **final checkpoint, no metric
   selection**; H5's cadence clause is either waived in the Gate-C registration or implemented as
   in-training `val_loss`-free resumability.
2. **The accepted Step-2.4F `layout.json` predates SEP.** It declares
   `special = {pad: 1024, bos: 1025, eos: 1026}` and `vocab_size = 1027`; Gate B registers a fourth
   special (`sep = 1027`) and `vocab_size = 1028`, and `build_examples` refuses a layout without
   `sep` because deriving a token id at use time could alias a real code. Training against the
   frozen archive as-is therefore **hard-stops**. The fix is a metadata-only re-derivation of
   `sid/layout.json` (add `sep`, set `vocab_size` 1028, keep the format tag or bump it to v4); the
   accepted `semantic_ids.json` assignment, `tokenizer.pt`, `item_features.npy` and every Semantic
   ID are unchanged, so no RQ-VAE refit and no re-encode is required. This is registered as
   Gate C's first action, not as a silent edit to a frozen artifact.

## 18. Provenance and licence constraints

Both audited repositories are legally *read-only design sources*; neither may be vendored.

| Repo | Licence found | Permitted | Forbidden |
|---|---|---|---|
| `mclwu22/amazon-genrec` | **no LICENSE file** — no licence granted (default all-rights-reserved) | reading it, learning the design, **reimplementing** the algorithms, citing it | copying files or long verbatim excerpts into AgentRec-X; redistributing its code |
| `snap-research/GRID` | **Snap Inc., non-commercial research purposes only**; redistribution must retain the notice | reading it, reimplementing, citing; retaining the notice in `PROVENANCE.md` if any excerpt is quoted | **vendoring** its files; commercial use; redistribution without the notice |

What is absorbed, and from where:

* **amazon-genrec** — the frozen Sentence-T5 text-embedding stage, the per-level token-offset
  layout, the GPU-resident batch-slicing training pattern, and the one *working*
  catalogue-constrained prefix trie (`prefix_allowed_tokens_fn`). Its Spark pipeline, its
  split, its beam-limited evaluator and its absent seeding are **not** absorbed.
* **GRID** — the RQ-VAE encoder/decoder shape, input normalisation, squared-L2 distance with
  `beta = 0.25`, k-means++ codebook initialisation, the dedup-digit *algorithm*, and the
  per-level coverage / `id_entropy` diagnostics. Its Lightning/Hydra/TFRecord scaffolding, its
  disabled prefix check, and its `eval_step` defect are **not** absorbed.

**Conflict rule:** GRID wins the quantizer, amazon-genrec wins the generator and decoder. The
only shared claim is the dedup digit, where GRID's *placement* (a separate pass, verifiable
against the assignment) is adopted — plus the overflow refusal of §12.1, which **neither repo
has** and both need.

`PROVENANCE.md` is mandatory and labels every file `ORIGINAL`,
`REIMPLEMENTED_FROM(<repo>@<sha>, <file>:<lines>)`, or `DEPENDENCY(<pkg>==<ver>, <licence>)`.
Every reimplementation carries an inline citation naming the source revision and what changed.
`T6` asserts every backend file appears in `PROVENANCE.md`, so a pasted file cannot go unnoticed.

---

## 19. Dependency isolation

| Layer | Venv | Contents | Import rule |
|---|---|---|---|
| **AgentRec-X** | `X/.venv` | unchanged (CPU torch, NumPy, FastAPI, LangGraph) | must not import `backends/` |
| **Adapter** | `X/.venv` | `recommendation/backends/tiger_backend.py` — stdlib + NumPy only | subprocess + filesystem only; no torch, no transformers |
| **Backend** | `backends/tiger_public/.venv` | torch, transformers, sentence-transformers, numpy<2, pyarrow | must not import `recommendation.*`; reads JSONL/NPY only |

The root `requirements.txt` is **not touched**. `AGENTS.md` §10 forbids disturbing the CUDA/PyTorch
stack and §11 says new dependencies live in the project virtual environment; a nested venv under
`backends/tiger_public/` satisfies both, so the backend's CUDA torch never coexists with
AgentRec-X's CPU torch. The exchange format is deliberately dumb (JSONL, `.npy`, `.npz`, `.json`):
no pickle, no shared class, no shared torch across the boundary. Backend tests live in
`backends/tiger_public/tests/` and are **skipped**, not failed, when the backend venv is absent,
so the full suite and the packaging remain valid.
