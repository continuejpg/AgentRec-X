// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): rewritten as a typed client for the AgentRec-X
// BFF contract (docs/WEB_DEMO_CONTRACT.md §2), including a hand-written fetch + ReadableStream
// SSE reader (POST /api/chat/stream cannot use EventSource).

/**
 * Typed HTTP/SSE client for the local AgentRec-X BFF.
 *
 * The BFF is the only source of product, evidence, preference and gate data. This client never
 * injects identities (no user_id / user_key / confirmed_by_user); it sends only a session id and
 * opaque product references.
 */

import type {
  ClarificationReplyResponse,
  ComparisonPayload,
  DemoConfig,
  Health,
  ProductDetailPayload,
  ProductsPayload,
  ReplayResponse,
  ServerEvent,
  SessionSummary,
  SessionView,
  PreferenceState,
} from "./types";

/** BFF base URL; override with NEXT_PUBLIC_BFF_URL for a non-default port. */
export const BFF_BASE = (process.env.NEXT_PUBLIC_BFF_URL ?? "http://127.0.0.1:8787").replace(
  /\/+$/,
  "",
);

export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError")
  );
}

export function isTimeoutError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "TimeoutError") ||
    (error instanceof Error && error.name === "TimeoutError")
  );
}

/** A real BFF error, carrying the contract's `{code, message, detail}` when one was returned. */
export class BffError extends Error {
  readonly code: string;
  readonly status: number | null;
  readonly detail: unknown;
  readonly tool: string | null;

  constructor(
    code: string,
    message: string,
    options: { status?: number | null; detail?: unknown; tool?: string | null } = {},
  ) {
    super(message);
    this.name = "BffError";
    this.code = code;
    this.status = options.status ?? null;
    this.detail = options.detail;
    this.tool = options.tool ?? null;
  }
}

export function errorPayload(error: unknown): { code: string; message: string; detail?: unknown; tool?: string | null } {
  if (error instanceof BffError) {
    return { code: error.code, message: error.message, detail: error.detail, tool: error.tool };
  }
  if (error instanceof Error) {
    if (isAbortError(error)) return { code: "aborted", message: "The request was stopped." };
    if (isTimeoutError(error)) return { code: "timeout", message: "The request timed out waiting for the BFF." };
    return { code: "client_error", message: error.message };
  }
  return { code: "client_error", message: String(error) };
}

interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const JSON_HEADERS = { "Content-Type": "application/json", Accept: "application/json" } as const;
const DEFAULT_TIMEOUT_MS = 20_000;
/** MCP calls are allowed up to 180 s by the BFF; keep a slightly larger client ceiling. */
const TOOL_TIMEOUT_MS = 190_000;

async function toBffError(response: Response): Promise<BffError> {
  let code = `http_${response.status}`;
  let message = `${response.status} ${response.statusText || "request failed"}`;
  let detail: unknown;
  let tool: string | null = null;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object") {
      const record = body as Record<string, unknown>;
      if (typeof record.code === "string" && record.code) code = record.code;
      if (typeof record.message === "string" && record.message) message = record.message;
      if ("detail" in record) detail = record.detail;
      if (typeof record.tool === "string") tool = record.tool;
    }
  } catch {
    // Non-JSON error body: keep the status line.
  }
  return new BffError(code, message, { status: response.status, detail, tool });
}

async function request<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new DOMException("The request timed out.", "TimeoutError"));
  }, timeoutMs);
  const forwardAbort = () => controller.abort(options.signal?.reason);
  if (options.signal) {
    if (options.signal.aborted) forwardAbort();
    else options.signal.addEventListener("abort", forwardAbort, { once: true });
  }
  try {
    const response = await fetch(`${BFF_BASE}${path}`, {
      method,
      headers: body === undefined ? { Accept: "application/json" } : JSON_HEADERS,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });
    if (!response.ok) throw await toBffError(response);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof BffError) throw error;
    if (isAbortError(error) && !timedOut) throw error;
    if (timedOut) throw new DOMException("The request timed out.", "TimeoutError");
    throw new BffError(
      "bff_unreachable",
      `Cannot reach the local BFF at ${BFF_BASE}. Is it running? (This page retries automatically.)`,
      { detail: error instanceof Error ? error.message : String(error) },
    );
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", forwardAbort);
  }
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

interface StreamOptions {
  signal?: AbortSignal;
  /** Called on every `: keep-alive` comment so the UI can show the stream is alive. */
  onHeartbeat?: () => void;
}

/**
 * Parses an `text/event-stream` body into typed events.
 *
 * Contract frames are `event: <type>\ndata: <json>\n\n` with `: keep-alive` comments. Multi-line
 * data is joined with newlines (per the SSE spec) though the BFF sends one JSON line per frame.
 */
async function* readEventStream(
  body: ReadableStream<Uint8Array>,
  options: StreamOptions,
): AsyncGenerator<ServerEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType: string | null = null;
  let dataLines: string[] = [];

  const dispatch = (): ServerEvent | null => {
    const frameType = eventType;
    const lines = dataLines;
    eventType = null;
    dataLines = [];
    if (lines.length === 0) return null;
    const raw = lines.join("\n");
    let data: unknown;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new BffError(
        "sse_parse_error",
        `Malformed SSE payload from the BFF for event "${frameType ?? "message"}".`,
        { detail: raw.slice(0, 400) },
      );
    }
    return { type: frameType ?? "message", data } as ServerEvent;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        let line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        if (line === "") {
          const event = dispatch();
          if (event) yield event;
        } else if (line.startsWith(":")) {
          options.onHeartbeat?.();
        } else if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^ /, ""));
        }
        // Other SSE fields (id:, retry:) are not used by the contract.
      }
    }
    // A final frame without a trailing blank line still counts.
    const tail = buffer.trim();
    if (tail.length > 0) {
      for (const line of tail.split("\n")) {
        const clean = line.endsWith("\r") ? line.slice(0, -1) : line;
        if (clean.startsWith("event:")) eventType = clean.slice(6).trim();
        else if (clean.startsWith("data:")) dataLines.push(clean.slice(5).replace(/^ /, ""));
      }
    }
    const last = dispatch();
    if (last) yield last;
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class BffClient {
  readonly base = BFF_BASE;

  health(signal?: AbortSignal): Promise<Health> {
    return request<Health>("GET", "/api/health", undefined, { signal, timeoutMs: 8_000 });
  }

  config(signal?: AbortSignal): Promise<DemoConfig> {
    return request<DemoConfig>("GET", "/api/demo/config", undefined, { signal, timeoutMs: 8_000 });
  }

  createSession(profileId: string): Promise<SessionView> {
    return request<SessionView>("POST", "/api/sessions", { profile_id: profileId });
  }

  listSessions(limit = 20, signal?: AbortSignal): Promise<SessionSummary[]> {
    return request<{ sessions: SessionSummary[] }>("GET", `/api/sessions?limit=${limit}`, undefined, {
      signal,
      timeoutMs: 10_000,
    }).then((data) => data.sessions ?? []);
  }

  getSession(sessionId: string, signal?: AbortSignal): Promise<SessionView> {
    return request<SessionView>("GET", `/api/sessions/${encodeURIComponent(sessionId)}`, undefined, {
      signal,
      timeoutMs: 10_000,
    });
  }

  resetSession(sessionId: string): Promise<SessionView> {
    // The BFF endpoint takes no body; only the path identifies the session.
    return request<SessionView>("POST", `/api/sessions/${encodeURIComponent(sessionId)}/reset`);
  }

  preferences(sessionId: string, signal?: AbortSignal): Promise<PreferenceState> {
    return request<PreferenceState>(
      "GET",
      `/api/preferences?session_id=${encodeURIComponent(sessionId)}`,
      undefined,
      { signal, timeoutMs: 20_000 },
    );
  }

  async *chatStream(
    body: { session_id: string; message: string; selected_product_ids?: string[] },
    options: StreamOptions = {},
  ): AsyncGenerator<ServerEvent> {
    const response = await fetch(`${BFF_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { ...JSON_HEADERS, Accept: "text/event-stream" },
      body: JSON.stringify({
        session_id: body.session_id,
        message: body.message,
        ...(body.selected_product_ids && body.selected_product_ids.length
          ? { selected_product_ids: body.selected_product_ids }
          : {}),
      }),
      signal: options.signal,
      cache: "no-store",
    });
    if (!response.ok) throw await toBffError(response);
    if (!response.body) {
      throw new BffError("bff_no_stream", "The BFF returned no SSE body for the chat stream.");
    }
    yield* readEventStream(response.body, options);
  }

  productDetails(sessionId: string, productRef: string, signal?: AbortSignal): Promise<ProductDetailPayload> {
    return request<ProductDetailPayload>(
      "POST",
      "/api/products/details",
      { session_id: sessionId, product_ref: productRef },
      { signal, timeoutMs: TOOL_TIMEOUT_MS },
    );
  }

  compare(sessionId: string, productRefs: string[], signal?: AbortSignal): Promise<ComparisonPayload> {
    return request<ComparisonPayload>(
      "POST",
      "/api/products/compare",
      { session_id: sessionId, product_refs: productRefs },
      { signal, timeoutMs: TOOL_TIMEOUT_MS },
    );
  }

  clarificationReply(
    sessionId: string,
    clarificationId: string,
    message: string,
    signal?: AbortSignal,
  ): Promise<ClarificationReplyResponse> {
    return request<ClarificationReplyResponse>(
      "POST",
      "/api/clarification/reply",
      { session_id: sessionId, clarification_id: clarificationId, message },
      { signal, timeoutMs: TOOL_TIMEOUT_MS },
    );
  }

  replay(signal?: AbortSignal): Promise<ReplayResponse> {
    return request<ReplayResponse>("GET", "/api/replay", undefined, { signal, timeoutMs: 10_000 });
  }
}

export const api = new BffClient();
export type { ProductsPayload };
