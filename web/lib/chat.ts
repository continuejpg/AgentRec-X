// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

/** Client-side turn model: the BFF TurnView plus streaming/progress state the browser owns. */

import type {
  ClarificationPayload,
  ComparisonPayload,
  ErrorPayload,
  MemoryUpdatePayload,
  PlannerMeta,
  PreferenceState,
  ProductsPayload,
  ToolStatus,
  TurnView,
} from "./types";
import { plannerFromTurnView } from "./planner";

export interface LocalToolStatus extends ToolStatus {
  /** performance.now() when the running row arrived; used for the live elapsed counter. */
  receivedAt?: number;
}

export interface ClarificationReplyState {
  text: string;
  status: "sending" | "done" | "error";
  /** The handle this reply was written against; a new follow-up handle re-opens the UI. */
  clarification_id?: string | null;
  steps: ToolStatus[];
  memory_update: MemoryUpdatePayload | null;
  preferences: PreferenceState | null;
  recommendation: ProductsPayload | null;
  error: ErrorPayload | null;
  /** Real planner metadata when the resumed turn was a Qwen Live turn. */
  planner: PlannerMeta | null;
}

export type TurnStatus = "streaming" | "complete" | "interrupted" | "error";

export interface LocalTurn {
  key: string;
  turn_id: string | null;
  user_text: string;
  assistant_text: string;
  tool_activity: LocalToolStatus[];
  products: ProductsPayload | null;
  comparison: ComparisonPayload | null;
  clarification: ClarificationPayload | null;
  memory_update: MemoryUpdatePayload | null;
  error: ErrorPayload | null;
  created_at: string | null;
  pending: boolean;
  status: TurnStatus;
  stop_reason: string | null;
  total_ms: number | null;
  /** Real planner metadata from the backend (Qwen Base Live / Rule); null when unreported. */
  planner: PlannerMeta | null;
  startedAt: number;
  /** Set when the BFF marked the narrative final. */
  text_final: boolean;
  /** Reply state for a pending clarification that was answered from this UI. */
  reply: ClarificationReplyState | null;
}

let localCounter = 0;

export function newTurnKey(): string {
  localCounter += 1;
  return `local-${Date.now()}-${localCounter}`;
}

/** Maps a persisted TurnView (session restore) into the browser turn model. */
export function turnFromView(view: TurnView, index: number): LocalTurn {
  return {
    key: view.turn_id || `restored-${index}`,
    turn_id: view.turn_id || null,
    user_text: view.user_text ?? "",
    assistant_text: view.assistant_text ?? "",
    // A persisted turn is finished; a `running` row in it can never still be running.
    tool_activity: finaliseStaleToolRows((view.tool_activity ?? []).map((row) => ({ ...row }))),
    products: view.products ?? null,
    comparison: view.comparison ?? null,
    clarification: view.clarification ?? null,
    memory_update: view.memory_update ?? null,
    error: view.error ?? null,
    created_at: view.created_at ?? null,
    pending: false,
    status: view.error ? "error" : "complete",
    stop_reason: null,
    total_ms: null,
    planner: plannerFromTurnView(view),
    startedAt: 0,
    text_final: true,
    reply: null,
  };
}

/** True when the latest turn still has an open clarification (the user has not replied). */
export function hasOpenClarification(turns: LocalTurn[]): boolean {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (turn.clarification) {
      // A follow-up clarification is a new handle even if the previous reply is done.
      if (turn.reply?.clarification_id && turn.reply.clarification_id !== turn.clarification.clarification_id) {
        return true;
      }
      return turn.reply == null || turn.reply.status === "error";
    }
  }
  return false;
}

function replaceRow(
  list: LocalToolStatus[],
  index: number,
  incoming: Partial<LocalToolStatus>,
): LocalToolStatus[] {
  const next = [...list];
  next[index] = { ...list[index], ...incoming };
  return next;
}

/**
 * Merges one real `tool_status` event into the timeline.
 *
 * `running` opens a row (a repeated identical notification is not duplicated). A terminal status
 * closes the newest matching running row; rows are matched on tool *and* label first because a
 * Qwen turn legitimately repeats the same tool with a different label per planner action
 * (`planner_action` → "Planner Action: RETRIEVE", then "Planner Action: RERANK", …). If the
 * backend only ever sent a terminal status, it is appended as its own completed row — never
 * dropped, never turned into a fake animation.
 */
export function mergeToolStatus(list: LocalToolStatus[], incoming: ToolStatus): LocalToolStatus[] {
  if (incoming.status === "running") {
    const last = list[list.length - 1];
    if (last && last.status === "running" && last.tool === incoming.tool && last.label === incoming.label) {
      // MCP progress notifications may repeat; refresh the open row instead of duplicating it.
      return replaceRow(list, list.length - 1, { ...incoming, receivedAt: performance.now() });
    }
    return [...list, { ...incoming, receivedAt: performance.now() }];
  }
  let fallbackIndex = -1;
  for (let index = list.length - 1; index >= 0; index -= 1) {
    const row = list[index];
    if (row.tool !== incoming.tool || row.status !== "running") continue;
    if (row.label === incoming.label) return replaceRow(list, index, incoming);
    if (fallbackIndex < 0) fallbackIndex = index;
  }
  if (fallbackIndex >= 0) return replaceRow(list, fallbackIndex, incoming);
  return [...list, { ...incoming }];
}

/**
 * Closes progress rows whose completion the backend never reported.
 *
 * A finished turn (SSE `done`, an aborted stream, or a persisted turn restored from the session)
 * cannot still have a running step. Such a row is shown as browser-local `unknown` with an
 * explicit note — never as an endless spinner (fake animation) and never as a fabricated "ok".
 */
export function finaliseStaleToolRows(list: LocalToolStatus[]): LocalToolStatus[] {
  if (!list.some((row) => row.status === "running")) return list;
  return list.map((row) =>
    row.status === "running"
      ? {
          ...row,
          status: "unknown",
          detail: [row.detail, "Completion was not reported by the backend before the turn ended."]
            .filter(Boolean)
            .join(" · "),
        }
      : row,
  );
}
