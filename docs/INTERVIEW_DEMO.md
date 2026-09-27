# AgentRec-X — Interview Demo Script

An operator script for a **3–5 minute** live demonstration. It says exactly what to type, what to
point at, what to say, and what not to claim. Everything here was verified against the running
system; nothing is aspirational.

---

## 1. Demo purpose

The demo shows one path end to end:

```text
trusted behavioural recommendation
      -> explicit conversational preference memory
      -> catalogue-grounded evidence
      -> deterministic reranking
      -> auditable Recommendation Trace
```

The message to land:

> **Conversation can change what the user sees without giving the conversational layer authority
> to rewrite behavioural history, product identity, catalogue facts or recommendation-model
> scores.**

This is **not** a demo of autonomous multi-source recommendation. The browser path has one
candidate generator (SASRec) and does not route between retrieval models.

---

## 2. Pre-demo checklist

```bash
git rev-parse HEAD          # expect the commit you intend to present
./scripts/start_demo.sh     # foreground; Ctrl+C to stop
```

Then confirm, in order:

| Check | Expect |
| --- | --- |
| `http://127.0.0.1:8000/demo/` | the browser demo loads |
| `GET /v1/demo/health` | `status=ok`, `model_loaded=true`, `metadata_loaded=true`, `demo_ready=true` |

Presenter checklist:

- [ ] **Click "New session".** A new session gets a brand-new, empty preference namespace, so no
      earlier demonstration can leak preferences into this one.
- [ ] Confirm **Active preferences** shows *No active preferences yet.* before Turn 1.
- [ ] Confirm the **Recommendation Trace** block opens and is readable (it is collapsed by
      default under each agent turn).
- [ ] Do **not** start DeepSeek, vLLM, Ollama or any hosted provider. The serving path needs no
      API key and no network.
- [ ] Leave the browser at a normal width so the trace block is comfortably readable.

Startup loads the checkpoint and catalogue once (roughly 20 s). If an artifact is missing the
process exits with an explicit error rather than starting a broken server.

---

## 3. The 3–5 minute demo

Five turns. Type them in order, in the same session.

---

### Turn 1 — establish the recommender

**TYPE**

```text
Recommend some useful gear for a weekend camping trip.
```

**EXPECT TO SEE**

- five real catalogue candidates, each with an original SASRec rank
- final rank equal to original rank (nothing has moved yet)
- **Active preferences:** none
- **Recommendation Trace:** Route = RECOMMENDATION, Candidates = 5, no evidence section,
  no memory changes

**SAY**

> "The candidate pool comes from the sequential recommender over trusted user history — not from a
> keyword search over the catalogue. I keep the original SASRec rank visible so later
> conversational changes stay auditable."

**POINT AT**

- the five cards and their **original rank** labels
- the Recommendation Trace: **Route**, **Candidates**

*Do not say* that the word "camping" made the system search the catalogue for camping products.

---

### Turn 2 — establish memory

**TYPE**

```text
I prefer lightweight and durable products.
```

**EXPECT TO SEE**

- **Active preferences** gains two separate entries: `lightweight`, `durable`
- **Recommendation Trace → Memory changes:** `added: lightweight, durable`
- the Turn-1 ranking does not change (preferences written this turn apply from the next turn)

**SAY**

> "That statement is stored as user-owned conversational state, in two independent parts. It does
> not touch behavioural history, and it takes effect from the following turn."

**POINT AT**

- the **Active preferences** panel
- **Recommendation Trace → Memory changes**

*Do not* explain parser internals. One sentence is enough here.

---

### Turn 3 — the main demo moment

**TYPE**

```text
Recommend the camping gear again using my preferences.
```

**EXPECT TO SEE**

- **Active preferences:** `lightweight`, `durable`
- **Recommendation Trace → Evidence:** `MATCH 3 candidates`, `VIOLATION 0 candidates`,
  `UNKNOWN 5 candidates`
- **Recommendation Trace → Reranking:** the movement below, with titles
- the cards' own evidence lines agree with the trace counts

Verified example movement (profile-dependent):

```text
Original SASRec -> Final
#1 -> #1
#3 -> #2
#4 -> #3
#2 -> #4
#5 -> #5
```

**SAY**

> "The recommender itself has not changed. The system evaluates catalogue metadata against the
> stored preferences and applies a deterministic reranker. You can see the original SASRec rank
> beside the final rank."

Then add, while pointing at the `lightweight` line:

> "Matching is deliberately conservative and exact-term: `ultralight` does not automatically count
> as `lightweight`. Unsupported inference stays `UNKNOWN` rather than being guessed — that is a
> grounding choice, not a language-understanding feature."

**POINT AT**

- **Memory changes** (no change this turn) and **Active preferences**
- **Evidence**: the MATCH count, and explicitly the `lightweight` = `UNKNOWN` rows
- **Reranking**: `#3 → #2 ↑`, `#4 → #3 ↑`, `#2 → #4 ↓` — and that `#1` and `#5` did not move

This is the turn that shows evidence, policy and auditability together. Do not rush it.

---

### Turn 4 — establish mutable memory

**TYPE**

```text
Remove my lightweight preference.
```

**EXPECT TO SEE**

- **Route: DIRECT** (no recommendation this turn)
- the reply: `Got it. I removed your lightweight preference. Active preference: durable.`
- **Recommendation Trace → Memory changes:** `removed: lightweight`
- **Active preferences:** `durable` only

**SAY**

> "This acknowledgement is generated from the persisted mutation result — not by trusting the
> user's sentence and pretending the update succeeded. A removal that matched nothing would keep
> the generic reply instead."

**POINT AT**

- the acknowledgement text
- **Recommendation Trace → Memory changes: removed**, and the remaining active preference

---

### Turn 5 — close the loop

**TYPE**

```text
Recommend the camping gear again.
```

**EXPECT TO SEE**

- **Active preferences:** `durable` only
- **Recommendation Trace → Evidence:** `MATCH 3 candidates`, `VIOLATION 0`,
  `UNKNOWN 2 candidates` (no `lightweight` rows at all)
- the same ranking movement as Turn 3, and the trace agreeing with the visible cards

**SAY**

> "The next recommendation reads the updated memory: `lightweight` is gone, so only durability
> contributes evidence. The state transition is fully visible in the trace."

**POINT AT**

- **Evidence** now counting only `durable`
- **Reranking** still agreeing card-for-card with the trace

**Stop the live demo here.** See §10.

---

## 4. What the interviewer should have learned

```text
SASRec owns behavioural candidate generation.

Conversation owns explicit preferences.

Catalogue metadata supplies evidence.

Deterministic code owns evidence and reranking.

Recommendation Trace makes the resulting state transition auditable.
```

Which is the project's architecture statement in one line:

> **Decision authority is narrower than truth authority.**

---

## 5. Safe claims vs unsafe claims

| Safe to say | Do NOT say |
| --- | --- |
| The serving demo uses SASRec as its candidate generator | The browser Agent dynamically chooses among SASRec, Two-Tower and TIGER |
| Explicit preferences persist across turns | The browser path performs adaptive multi-source fusion |
| Catalogue metadata is used as evidence for stored preferences | The natural-language request directly retrieves matching catalogue products |
| Reranking is deterministic and inspectable | Evidence uses general semantic understanding |
| Original SASRec ranks remain visible and auditable | This is production-ready |
| Recommendation Trace reflects authoritative response state | The live Agent improves recommendation accuracy |
| The serving path needs no hosted LLM API key and no network | The LLM policy beat the fixed-fusion baseline |
| Two-Tower and TIGER belong to the offline research track | Any claim that the browser path is "intelligent" or "autonomous" |

---

## 6. Expected variability — do not panic

These differences do **not** mean the demo failed:

- **Which catalogue fields appear** in an evidence snippet can vary by product.
- **Titles may be missing** for some candidates; the trace then shows the `parent_asin` instead.
- **Evidence counts and rank movement depend on the stored preferences and the profile's
  history.** A different profile, or a different product set, legitimately moves different cards.
- **The exact candidate products** can differ from the example above; only the trace/card
  agreement is required.
- **Trace panels are collapsed by default** and may need a click; expansion state is cosmetic.

Genuine failures — stop and say so rather than working around them:

- a stated preference does not appear in **Active preferences** afterwards;
- the removal acknowledgement appears but `lightweight` is still active;
- **Recommendation Trace** disagrees with the cards (evidence counts or original/final ranks);
- evidence shows a `MATCH` or `VIOLATION` the card does not show;
- `model_loaded` or `metadata_loaded` is false in `/v1/demo/health`;
- any server error / 500.

---

## 7. If the interviewer asks "Where is the LLM?"

> The interview serving path deliberately uses the original deterministic, offline decision seam —
> there is no hosted model in it and no API key. The repository also contains a later bounded
> LLM control-plane research track, kept separate from the demo. We did evaluate a live
> LLM policy there, once, on the frozen 20,000-user cohort; in that specific configuration it
> underperformed the fixed-fusion comparator on the primary endpoint. So: an LLM exists in the
> repository, it is not what serves this demo, and it did not win when we measured it.

Keep it to those sentences. Do not open the M3 report during the demo.

---

## 8. If the interviewer asks "Why call this an Agent?"

> Because the conversational layer is a bounded orchestration layer, not a chatbot: it holds
> multi-turn session state, persists explicit user preferences, and acts through an explicit
> tool/action boundary — it decides *whether* to invoke the recommendation path and *what
> constraint* the user has stated. What defines it is that its decision authority is deliberately
> narrower than the truth authority of the trusted code underneath: it can influence what is
> shown, and it cannot rewrite behavioural history, product identity or model scores.
> The determinism of the current decision seam is a design choice; the seam is injectable, which
> is exactly why a model-driven policy could be evaluated later behind it.

*Do not* define "agent" as "it uses an LLM" — the browser serving path does not.

---

## 9. If the interviewer asks "Why are some recommendations not obviously camping products?"

The verified demo profile can return items such as a water filter, fishing line, ear protection,
an umbrella and a fitness band. Answer honestly:

> Candidate generation here is behavioural and sequential — it comes from the trusted history of
> this profile, not from the words in my request. The request does not replace the candidate
> generator with catalogue search; the conversational preferences rerank a trusted candidate pool.
> Lexical catalogue retrieval exists elsewhere in the repository — in the research and
> control-plane work — but it is not the main browser serving path.

That is an intentional boundary plus a current limitation: the demo shows *trust, memory,
evidence and auditability*, not open-ended product search. Do not present it as a feature.

---

## 10. Demo stop rule

**After Turn 5, stop the prepared demo.**

Do not, during a normal interview:

- switch providers or start a hosted model;
- open the M3 evaluation;
- launch TIGER or any retrieval backend;
- run benchmarks or show artifact manifests;
- improvise new preference syntaxes the parser does not support;
- demonstrate semantic synonyms the matcher does not implement;
- attempt multi-source routing.

Those are available for a deeper technical discussion **only if the interviewer explicitly asks**.
If they do ask, answer from `README.md`, `docs/ARCHITECTURE.md` and the research reports — do not
try to demonstrate them live.
