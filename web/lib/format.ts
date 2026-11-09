// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): reduced to the formatting helpers this demo needs,
// added BFF view-model helpers (latency, tool phases, namespaces, relative time) and removed the
// commerce money/order helpers that do not apply to a catalogue-only demo.

const numberFormat = new Intl.NumberFormat("en-US");

export function formatNumber(value: number): string {
  return numberFormat.format(value);
}

/** "1 product", "3 products". */
export function plural(count: number, one: string, many = `${one}s`): string {
  return `${formatNumber(count)} ${count === 1 ? one : many}`;
}

/**
 * Durations from BFF payloads are milliseconds. Tool calls are usually sub-second while a cold
 * BGE search is seconds, so both are shown side by side.
 */
export function formatLatency(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

/** Total turn time from the `done` event, e.g. "2.4 s". */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function parseDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** "Nov 9, 2026, 18:15" — shown with a `title` for the raw ISO timestamp. */
export function formatDateTime(value: string | null | undefined): string {
  const date = parseDate(value);
  if (!date) return value ?? "";
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** "just now", "3 min ago", "Nov 9". */
export function formatRelativeTime(value: string | null | undefined, now = Date.now()): string {
  const date = parseDate(value);
  if (!date) return value ?? "";
  const delta = now - date.getTime();
  if (delta < 45_000) return "just now";
  if (delta < 3_600_000) return `${Math.max(1, Math.round(delta / 60_000))} min ago`;
  if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)} h ago`;
  if (delta < 7 * 86_400_000) return `${Math.round(delta / 86_400_000)} d ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** "01:23" style elapsed clock for the readiness timer. */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

/** `amazon_electronics_c4:B004K6R43G` → `amazon_electronics_c4`. */
export function namespaceOf(namespacedId: string | null | undefined): string {
  if (!namespacedId) return "";
  const index = namespacedId.indexOf(":");
  return index > 0 ? namespacedId.slice(0, index) : "";
}

/** "wireless_earbuds" / "in-ear" → "Wireless earbuds" / "In ear". */
export function humanize(value: string | null | undefined): string {
  if (!value) return "";
  const spaced = value.replaceAll("_", " ").replaceAll("-", " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** "ALLOW_FINALIZE" → "Allow finalize". */
export function humanizeEnum(value: string | null | undefined): string {
  if (!value) return "";
  const spaced = value.replaceAll("_", " ").toLowerCase().trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Understandable phase copy for real tool-status rows. The BFF sends a short noun label
 * ("Search", "Product Info", "Clarification", …); when it sends a longer descriptive label this
 * demo keeps the backend's own words. The real MCP tool name is always shown next to it.
 */
const TOOL_PHASES: Record<string, string> = {
  search_products: "Searching Products",
  recommend_products: "Validating Constraints",
  get_product_info: "Loading Product Details",
  compare_products: "Building Comparison",
  request_preference_clarification: "Waiting for Clarification",
  submit_clarification_reply: "Updating Preferences",
  get_active_preferences: "Updating Preferences",
  get_product_evidence: "Preparing Evidence",
  product_evidence: "Preparing Evidence",
  check_product_type: "Checking Product Type",
};

/** Short noun labels the BFF uses; these are replaced by the phase copy above. */
const SHORT_BFF_LABELS = new Set([
  "search",
  "product info",
  "recommend",
  "recommendation",
  "compare",
  "comparison",
  "clarification",
  "memory",
  "evidence",
]);

export function toolPhaseLabel(tool: string, label?: string | null): string {
  const trimmed = label?.trim();
  if (trimmed) {
    const normalized = trimmed.toLowerCase().replace(/[._-]+/g, " ").trim();
    const isShortNoun = SHORT_BFF_LABELS.has(normalized) || trimmed.split(/\s+/).length === 1;
    if (!isShortNoun) return trimmed;
  }
  return TOOL_PHASES[tool] ?? (trimmed ? trimmed : humanize(tool));
}

/** The human label for a raw relation / match / verification status (unknown values pass through). */
export function statusLabel(value: string | null | undefined): string {
  if (!value) return "UNKNOWN";
  return humanizeEnum(value);
}

/** Catalogue attributes are rendered as "In ear" chips rather than raw snake_case. */
export function attributeLabel(value: string | null | undefined): string {
  return humanize(value);
}

/** Trims a long source record for one-line display while keeping the full value in `title`. */
export function shorten(value: string, max = 64): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}
