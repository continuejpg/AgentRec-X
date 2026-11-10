// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

/**
 * Real planner metadata → the per-turn planner badge (docs/QWEN_LIVE_CONTRACT.md §3).
 *
 * Sources, in the order the BFF may send them:
 *  - SSE `done`: `planner: "qwen_base"|"rule"`, `planner_calls`, `planner_latency_ms`
 *  - SSE `products` / persisted turn views: an optional planner block
 *  - the P17 `run_agent_turn` result's `planner` object (kind/model/calls/latency_ms array)
 *
 * Nothing is invented: a turn without backend planner metadata renders no planner badge, and the
 * label for a known kind comes only from the backend's own `kind` value.
 */

import type {
  ClarificationReplyResponse,
  DonePayload,
  PlannerMeta,
  PlannerMetaWire,
  ProductsPayload,
  TurnView,
} from "./types";

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function latencyValue(value: unknown): number | number[] | null {
  const scalar = finiteNumber(value);
  if (scalar != null) return scalar;
  if (!Array.isArray(value)) return null;
  const values = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
  return values.length ? values : null;
}

/**
 * Normalises one planner value from the wire. Returns null when the backend did not actually name
 * a planner (`kind` missing): the UI must not guess "rule" or "qwen" from surrounding context.
 */
export function normalisePlanner(value: PlannerMetaWire): PlannerMeta | null {
  if (!value) return null;
  if (typeof value === "string") {
    const kind = nonEmptyString(value);
    return kind ? { kind } : null;
  }
  const object = record(value);
  if (!object) return null;
  const kind = nonEmptyString(object.kind ?? object.planner_kind ?? object.planner);
  if (!kind) return null;
  const parserErrors = Array.isArray(object.parser_errors)
    ? object.parser_errors.filter((item): item is string => typeof item === "string")
    : null;
  return {
    kind,
    model: nonEmptyString(object.model),
    calls: finiteNumber(object.calls ?? object.planner_calls),
    latency_ms: latencyValue(object.latency_ms ?? object.planner_latency_ms),
    actions: Array.isArray(object.actions) ? object.actions : null,
    parser_errors: parserErrors,
  };
}

function withOptionalMeta(
  base: PlannerMeta,
  calls: unknown,
  latencyMs: unknown,
  model: unknown,
): PlannerMeta {
  return {
    ...base,
    model: base.model ?? nonEmptyString(model),
    calls: base.calls ?? finiteNumber(calls),
    latency_ms: base.latency_ms ?? latencyValue(latencyMs),
  };
}

/** Planner metadata from the SSE `done` event (flat fields per the contract). */
export function plannerFromDone(done: DonePayload): PlannerMeta | null {
  const base = normalisePlanner(done.planner);
  if (!base) return null;
  return withOptionalMeta(base, done.planner_calls, done.planner_latency_ms, done.planner_model);
}

/** Planner metadata from a `products` SSE event, if the backend attached it. */
export function plannerFromProducts(payload: ProductsPayload | null | undefined): PlannerMeta | null {
  if (!payload) return null;
  const base = normalisePlanner(payload.planner ?? payload.planner_kind ?? null);
  if (!base) return null;
  return withOptionalMeta(base, payload.planner_calls, payload.planner_latency_ms, payload.planner_model);
}

/** Planner metadata from a persisted TurnView (session restore). */
export function plannerFromTurnView(view: TurnView): PlannerMeta | null {
  const base = normalisePlanner(view.planner ?? view.planner_kind ?? null);
  if (!base) return null;
  return withOptionalMeta(base, view.planner_calls, view.planner_latency_ms, view.planner_model);
}

/** Planner metadata from POST /api/clarification/reply when the resumed turn was Qwen. */
export function plannerFromReplyResponse(
  response: ClarificationReplyResponse | null | undefined,
): PlannerMeta | null {
  if (!response) return null;
  const base = normalisePlanner(response.planner ?? null);
  if (!base) return null;
  return withOptionalMeta(
    base,
    response.planner_calls,
    response.planner_latency_ms,
    response.planner_model,
  );
}

/** Later backend reports win, but a missing field never erases an earlier real value. */
export function mergePlanner(
  current: PlannerMeta | null,
  incoming: PlannerMeta | null,
): PlannerMeta | null {
  if (!incoming) return current;
  if (!current) return incoming;
  return {
    kind: incoming.kind || current.kind,
    model: incoming.model ?? current.model ?? null,
    calls: incoming.calls ?? current.calls ?? null,
    latency_ms: incoming.latency_ms ?? current.latency_ms ?? null,
    actions: incoming.actions ?? current.actions ?? null,
    parser_errors: incoming.parser_errors ?? current.parser_errors ?? null,
  };
}

/** Human label for the backend's planner kind only; unknown kinds are shown raw, not prettified. */
export function plannerLabel(planner: PlannerMeta): string {
  switch (planner.kind) {
    case "qwen_base":
      return "Qwen Base Live";
    case "rule":
      return "Rule Planner";
    case "none":
      // The backend explicitly reports that no planner drove this turn (e.g. a Skill/comparison
      // call); "No planner used" states that without claiming a model.
      return "No planner used";
    default:
      return planner.kind;
  }
}

/** e.g. "Qwen Base Live · 2 planner calls" — the example copy from the contract. */
export function plannerSummary(planner: PlannerMeta): string {
  const parts = [plannerLabel(planner)];
  if (planner.calls != null) parts.push(`${planner.calls} planner call${planner.calls === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

/** Total planner latency: scalar from `done`, or the sum of the MCP per-call array. */
export function plannerLatencyTotal(planner: PlannerMeta): number | null {
  const value = planner.latency_ms;
  if (typeof value === "number") return value;
  if (Array.isArray(value) && value.length) return value.reduce((sum, item) => sum + item, 0);
  return null;
}
