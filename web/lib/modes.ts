// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

/**
 * Mode resolution against the real BFF (docs/QWEN_LIVE_CONTRACT.md §2).
 *
 * The UI never decides that Qwen Base Live is available by itself: availability is true only when
 * the backend explicitly reports `modes.qwen_live.available === true` (GET /api/health first, then
 * GET /api/demo/config for an older/startup snapshot). Everything else is Unavailable and shows
 * the backend's own `disabled_reason`.
 */

import type { DemoConfig, Health } from "./types";

/**
 * Fallback shown only when a BFF has not supplied any reason yet (e.g. the first health response
 * of an older build). It keeps the honest "no live service" wording, and is replaced as soon as
 * the backend sends a real `disabled_reason` / `qwen_live.detail`.
 */
export const QWEN_DISABLED_FALLBACK = "No live Qwen service detected (waiting for /api/health)";

export interface QwenLiveState {
  /** True only when the backend explicitly reports the mode as available. */
  available: boolean;
  label: string;
  badge: string | null;
  /** The backend's real reason while unavailable; shown verbatim (no client invention). */
  disabledReason: string | null;
  detail: string | null;
  status: string | null;
  model: string | null;
  checkedAt: string | null;
  enabled: boolean | null;
  configured: boolean | null;
  latencyMs: number | null;
}

/** Real availability of the Qwen Live mode, exactly as the backend reported it. */
export function resolveQwenLive(health: Health | null, config: DemoConfig | null): QwenLiveState {
  const block = health?.qwen_live ?? null;
  const healthMode = health?.modes?.qwen_live ?? null;
  const configMode = config?.modes?.qwen_live ?? null;
  // Selectable only when the live backend explicitly reports modes.qwen_live.available === true.
  // /api/health carries the current probe; /api/demo/config is the startup advertisement. Order:
  // health modes block → health qwen_live snapshot → config modes → "not available".
  const available =
    typeof healthMode?.available === "boolean"
      ? healthMode.available
      : typeof block?.available === "boolean"
        ? block.available
        : configMode?.available === true;

  const meta = configMode ?? healthMode;
  const disabledReason = available
    ? null
    : (meta?.disabled_reason ?? healthMode?.disabled_reason ?? block?.detail ?? QWEN_DISABLED_FALLBACK);
  const badge = available ? (meta?.badge ?? healthMode?.badge ?? null) : null;

  return {
    available,
    label: meta?.label ?? "Qwen Live",
    badge,
    disabledReason,
    detail: meta?.detail ?? null,
    status: block?.status ?? null,
    model: block?.model ?? null,
    checkedAt: block?.checked_at ?? null,
    enabled: typeof block?.enabled === "boolean" ? block.enabled : null,
    configured: typeof block?.configured === "boolean" ? block.configured : null,
    latencyMs: block?.latency_ms ?? null,
  };
}
