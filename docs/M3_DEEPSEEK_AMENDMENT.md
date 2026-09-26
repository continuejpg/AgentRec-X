# M3 DEEPSEEK COMPATIBILITY AMENDMENT

**Status: PREPARED, NOT EXECUTED.** This amendment fills the single OPEN item of
`docs/M3_PREREGISTRATION.md` §9. No DeepSeek call has been made, no M3 metric exists, and no
evaluation has been run.

This document changes **nothing** in §2–§6 of the preregistration. In particular it does not
change the cohort, the evaluator, the K values, the comparator, the endpoints, the statistics, the
stop rule, the Agent's routing, its tool budget, or any decision criterion.

---

## 1. Frozen provider identity (§9 fields)

| field | env var | frozen value |
|---|---|---|
| base URL | `AGENTRECX_LLM_BASE_URL` | `https://api.deepseek.com` |
| model | `AGENTRECX_LLM_MODEL` | `deepseek-flash` |
| credential | `AGENTRECX_LLM_API_KEY` | supplied out-of-band; read from the environment at call time; never printed, logged, committed or serialised |
| provider profile | `AGENTRECX_LLM_PROFILE` | `deepseek` (the repository's existing profile) |
| JSON mode | `AGENTRECX_LLM_JSON_MODE` | `1` (adapter default) |
| thinking | `AGENTRECX_LLM_THINKING` | unset / false (adapter default) → thinking **disabled** |
| transport | — | standard-library `urllib_transport` |
| timeout | `AGENTRECX_LLM_TIMEOUT` | adapter default, 30.0 s |

Resolved endpoint: `POST https://api.deepseek.com/chat/completions`.

**Proposed and frozen `max_tokens`: 512.**

Reasoning: the response is one small JSON action object (`{"action": ..., "arguments": {...}}`),
which measures in the tens of tokens; 512 leaves generous headroom against the mid-string
truncation DeepSeek warns about in JSON Output mode while keeping the output side of the cost
negligible. `build_provider_client` defaults `max_tokens` to `None` — i.e. **the field is not sent
at all** — and no environment variable binds it, so the harness must pass it explicitly when it
constructs the client. This is a harness-level argument, not a policy change.

Sampling parameters sent: `temperature = 0.0`, `max_tokens = 512`. `top_p` is never sent, which is
consistent with DeepSeek fixing `top_p` at 1.0 in non-thinking mode. `reasoning_effort` is never
sent. Because thinking is disabled, `temperature` is honoured rather than ignored (DeepSeek
ignores it in thinking mode), so the value is meaningful.

## 2. Verified compatibility (documentation-verified, no live call)

| question | answer |
|---|---|
| API shape | **Chat Completions** — `POST {base_url}/chat/completions`. The adapter does not use DeepSeek's Responses API, although DeepSeek offers one. |
| JSON / tool mechanism | **JSON Output** via `response_format={"type": "json_object"}`. **No tool calling**: the adapter never emits a `tools` parameter. |
| `tool_choice` | **Not sent at all** — neither `required` nor named. No tool-calling path exists to name. |
| thinking-mode requirement | **Required and shape-verified.** DeepSeek enables thinking by default, so it must be disabled explicitly, and the OpenAI-format control is exactly `{"thinking": {"type": "enabled/disabled"}}` — byte-identical to what the `deepseek` profile emits. |
| encoding note | DeepSeek documents that the OpenAI *SDK* needs `extra_body={"thinking": ...}`. That is an SDK kwarg restriction; the adapter builds the JSON body itself, so the top-level `thinking` key is correct and no `extra_body` wrapper applies. |
| `reasoning_content` | Not applicable: thinking is disabled and no `tools` parameter is sent, so DeepSeek's "pass `reasoning_content` back or get HTTP 400" obligation never arises. |

Verified against DeepSeek's official documentation: [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
and [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/). `deepseek-flash` is
confirmed as a valid model id — it appears in DeepSeek's own sample code.

## 3. JSON output format / schema example (the frozen addition)

DeepSeek's JSON Output mode has two stated requirements that the existing request does **not**
satisfy: the literal word `json` must appear in the system or user prompt, and an example of the
desired shape must be given. The policy's system prompt says "JSON" (uppercase), and the user turn
is a `json.dumps` of context/actions whose keys never contain the word. Neither message carries a
format example.

The missing presentation is therefore supplied by one frozen string, applied at the transport seam
by `experiments/m3_agent_arm.py::JsonFormatClient` — a decorator that appends it to the request's
system prompt and touches nothing else:

```text
Output format:
Answer with exactly one json object and nothing else — no prose, no code fence.
The object must be valid json with this shape:
{"action": "<one offered action name>", "arguments": {}}
Put any action arguments in the arguments object; omit them for actions that take none.
```

This adds a **format instruction only**. It does not name or favour an action, does not describe
when to choose one, does not alter the offered action list, the context payload, the retry
correction, the step budget or the tool-call budget. The policy remains exactly the accepted
`LLMAgentPolicy`; the decorator is removable in one line, and is visible in one place.

Known, accepted residual risk: DeepSeek documents that JSON Output "may occasionally return empty
content". The adapter raises `ModelCallError(code="empty_response")` for that case, so it surfaces
as a **retryable failure** inside the frozen `max_retries = 1` rather than as a silent bad answer.
Whatever rate this reaches is a behavioural M3 result to be reported, not a defect to be tuned
away after the fact.

## 4. Known limitation — source-access asymmetry (recorded, NOT fixed)

The accepted `fixed_fusion` comparator fuses `popularity` + `sequential` + `metadata`. The frozen
`CandidateSource` enum contains `HISTORY`, `CATALOG_SEARCH`, `SIMILAR_ITEM`, `TWO_TOWER` and
`TIGER`, so the correspondence an Agent can actually name is:

| comparator source | Agent-namable equivalent |
|---|---|
| `sequential` | `CandidateSource.HISTORY` |
| `metadata` | `CandidateSource.CATALOG_SEARCH` |
| `popularity` | **none — no enum member exists** |

The Agent therefore cannot reach `popularity`, while the comparator fuses it. This is recorded as a
limitation and is **deliberately not repaired**: extending the enum would edit frozen protocol
(AGENTS.md §19.2 rule 7), and restricting the comparator would change the accepted baseline the
preregistration pins. It is exposed in code as
`experiments/m3_agent_arm.py::UNREACHABLE_SOURCES` and covered by a test, so the asymmetry cannot
be forgotten when the M3 result is worded.

## 5. Amendment checklist (all must hold before the run)

1. `max_tokens=512` is passed explicitly when the Agent harness builds the provider client
   (the adapter default of `None` does not send the field).
2. `JsonFormatClient` wraps the provider client, so the request carries the `json` word and the
   shape example.
3. Provider, model and sampling parameters are exactly as §1; the credential is read from the
   environment and never written anywhere.
4. §2–§6 of `docs/M3_PREREGISTRATION.md` are unchanged.
5. No result has been seen before any of the above is recorded.

An amendment that alters §2–§6 after a result has been seen invalidates M3.
