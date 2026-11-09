```yaml
name: preference-clarification
description: Ask one concrete clarification question when the current shopping request lacks a key need or has conflicting active preferences, then persist the user's real reply through the existing M9 memory service. Use only when clarification is required; never fabricate a reply or FINALIZE while waiting.
```

# Preference Clarification Skill

## When to use

Use when:

- the current request lacks a key need for a recommendation;
- active M9 preferences conflict;
- the existing Decision Gate returns `NEED_USER_CLARIFICATION`;
- the user explicitly updates or retracts a preference and the Agent must update memory.

Do not ASK on every recommendation. Do not use an evaluator answer or an expected clarification as a
user reply. Do not turn unrelated history into a current hard constraint.

## Required input

- current user/conversation key and current query;
- the current authorised M5/M9 history read;
- the existing Gate decision (if available);
- for a follow-up: an **actual user-authored reply** and a stable turn id.

## Flow

### Ask

1. Reuse the existing Gate `ask_question` when the Gate supplied one.
2. Otherwise choose one concrete question from active conflict or missing criteria.
3. Return `status = awaiting_clarification`, `selected_product_id = null`,
   `finalize_allowed = false`.
4. Do not produce a product recommendation or a final answer that implies one.

### Resume after a reply

1. Require a non-empty real user reply and a stable `turn_id`.
2. Call the existing `PreferenceMemoryService.process_turn` writer (same M9 semantics).
3. Re-read via `HistoryStateReader`; return active and inactive entries with their audit trail.
4. Same `turn_id` submitted twice must be idempotent (`already_processed=true`, no new entry).
5. Resume the normal existing Agent shopping flow with the updated preferences; the existing Gate
   still decides FINALIZE/ASK.
6. A retraction (“Forget my wireless preference.”) must remove the matching active entry while
   retaining the removed history entry.

## Output contract

Ask:

```json
{"schema":"agentrecx.p16.preference_clarification.v1","skill":"preference-clarification",
 "status":"awaiting_clarification","question":"...","selected_product_id":null,
 "finalize_allowed":false,"requires_user_reply":true}
```

Resume:

```json
{"schema":"agentrecx.p16.preference_clarification.v1","status":"memory_updated",
 "user_key":"...","turn_id":"...","already_processed":false,
 "update":{"added":1,"superseded":0,"removed":0,"already_processed":false},
 "active_preferences":[],"inactive_preferences":[]}
```

## Failure / termination

- Empty/fabricated reply: refuse; never synthesize a message.
- No reply: remain `awaiting_clarification`; do not FINALIZE or choose a product.
- Duplicate reply: idempotent, no duplicate user event/entry.
- Cross-user conversation read: rejected by the existing history reader.
- Unauthorised/evaluator-derived text: never accepted as a user reply.
- The Skill does not bypass Eligibility or the Decision Gate.

## Evidence and safety

The existing event-sourced memory owns lifecycle, supersession, removal, and audit. The Skill only
orchestrates the accepted reader/writer; it does not implement a second memory store. Product facts
are not questions for the user to guess — only genuine user needs/preferences may be clarified.
