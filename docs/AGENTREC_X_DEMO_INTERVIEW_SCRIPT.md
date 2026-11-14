# AgentRec-X Final Demo V2 — Interview Script

## 90-second version

**0–10 s — Frame it**

> “This is AgentRec-X: a shopping agent with real catalogue retrieval, long-term preference memory,
> and an evidence-aware Gate. The UI is only a thin layer over the existing Agent.”

**10–30 s — Shopping + evidence**

1. Open http://127.0.0.1:3000.
2. Click **Search with evidence** or type:
   `I need earbuds I can use while working out. It needs to be wireless.`
3. Point out:
   - real BGE product cards;
   - UNKNOWN fields and evidence;
   - the Gate decision and tool timeline.

**30–50 s — Long-term memory**

1. Click **A · Remember my preference**.
2. Submit the clarification reply: `I prefer a microphone.`
3. Point to the **Active Memory** and **Memory timeline**:
   - the new M9 event;
   - its logical sequence and source turn.
4. Send `Recommend again using my updated preferences.`

**50–70 s — Withdraw / update**

1. In **Memory Lab**, click `+ microphone` or type a preference.
2. Click `withdraw microphone` (or submit `Forget my microphone preference.`).
3. Show the timeline: the event remains with status **WITHDRAWN**, while Active Memory no longer
   contains it.
4. Send `Recommend again using my updated preferences.`

**70–90 s — Replay rankers + close**

1. Open right panel → **Rankers**.
2. Show the explicit **Recorded Experiment Replay** badge and Memory OFF/ON rank movement.
3. Close:

> “This is real M9 memory and real ranking output. The ranker comparison is clearly replayed from
> P100 GPU experiments; we do not pretend it is live. The production default remains the Existing
> Scorer because P100 did not prove a replacement.”

---

## 5-minute version

### 1. Architecture (30 s)

- Next.js UI → FastAPI BFF → MCP stdio → existing AgentRec-X.
- Show the tool timeline: `search_products`, `get_product_info`, `recommend_products`, Gate.
- State the boundary: no second Agent, no new memory store, no reranking.

### 2. Shopping + Top-K (45 s)

- Run the workout-earbuds query.
- Open product cards: title/ID, verified attributes, UNKNOWN, evidence provenance, no fabricated
  price/rating.
- Show a clarification case and the ASK card.

### 3. Memory lifecycle (90 s)

- Scenario A:
  - vague query → real clarification;
  - reply `I prefer a microphone.`;
  - Memory timeline event; Active Memory;
  - recommend again; product cards come from the real Agent.
- Scenario B:
  - use Memory Lab `avoid over-ear` plus `+ microphone`;
  - show both active preferences;
  - recommend again.
- Scenario C:
  - `withdraw microphone`;
  - show the removed/inactive event and the active state change;
  - recommend again.

### 4. Event-sourced explanation (45 s)

- Point out that withdrawal did not delete the historical event.
- Explain:
  `Event written → active state changed → ranking recomputed`.
- Do not claim the top-1 must always change; report the real result.

### 5. Ranker / Memory OFF–ON replay (60 s)

- Open **Rankers**.
- Select a P100 task.
- Toggle Memory OFF/ON and show recorded top-5/rank movement.
- Point out:
  - Experimental / Shadow labels;
  - P100 ranker replay is **not** a live inference;
  - production default remains Existing.

### 6. Evaluation honesty (30 s)

> “P100 has 1,971 public ESCI judged products, but 99.3% of Agent candidates are unjudged. The
> public judged track shows Original BGE stronger than ShopRanker Query Only on nDCG, so we did not
> replace the production Scorer. HUMAN memory preference Gold is still 0.”

### 7. Close

- Reproducible: commit, manifests, run logs.
- No fake data; UNKNOWN remains UNKNOWN.
- Next step: true per-session memory isolation and human preference review.

---

## Demo checks to say out loud

- [ ] Live CPU Agent path actually ran (tool rows visible).
- [ ] M9 write actually returned added/superseded/removed counts.
- [ ] Active Memory and timeline reflect the write.
- [ ] Replay badge visible on ranker panel.
- [ ] No price/rating/stock/image invented.
- [ ] Qwen Live unavailable is shown honestly; no silent fallback.
- [ ] “Clear session” does not claim to erase M9.
