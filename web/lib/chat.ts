// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

/** Client-side turn model: the BFF TurnView plus streaming/progress state the browser owns. */

import type {
  ClarificationPayload,
  ComparisonPayload,
  ErrorPayload,
  MemoryUpdatePayload,
  PreferenceState,
  ProductsPayload,
  ToolStatus,
  TurnView,
} from "./types";

export interface LocalToolStatus extends ToolStatus {
  /** performance.now() when the running row arrived; used for the live elapsed counter. */
  receivedAt?: number;
}

export interface ClarificationReplyState {
  text: string;
  status: "sending" | "done" | "error";
  steps: ToolStatus[];
  memory_update: MemoryUpdatePayload | null;
  preferences: PreferenceState | null;
  recommendation: ProductsPayload | null;
  error: ErrorPayload | null;
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
    tool_activity: (view.tool_activity ?? []).map((row) => ({ ...row })),
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
    startedAt: 0,
    text_final: true,
    reply: null,
  };
}

/** True when the latest turn still has an open clarification (the user has not replied). */
export function hasOpenClarification(turns: LocalTurn[]): boolean {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (turn.clarification) return turn.reply == null || turn.reply.status === "error";
  }
  return false;
}
