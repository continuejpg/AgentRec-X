// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): the Markdown renderer was removed because the BFF
// narrative is plain text (docs/WEB_DEMO_CONTRACT.md §2); added an ErrorBubble that shows the
// real BFF error code/message and a streaming caret retained from the shared component.

"use client";

import type { ErrorPayload } from "@/lib/types";

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end" data-testid="user-bubble">
      <div className="user-bubble max-w-[86%] whitespace-pre-wrap px-3.5 py-2 text-[14.5px] leading-normal sm:max-w-[78%]">
        {text}
      </div>
    </div>
  );
}

/** Assistant prose is rendered as inert text (no HTML / markdown injection from payloads). */
export function AssistantText({ text, streaming }: { text: string; streaming?: boolean }) {
  if (!text && !streaming) return null;
  return (
    <div
      data-testid="assistant-text"
      className={`whitespace-pre-wrap text-[14.5px] leading-relaxed text-(--ink) ${streaming ? "streaming-caret" : ""}`}
    >
      {text}
    </div>
  );
}

export function ErrorBubble({ error }: { error: ErrorPayload }) {
  return (
    <div
      role="alert"
      data-testid="error-bubble"
      data-error-code={error.code}
      className="rounded-xl border border-(--danger)/40 bg-(--danger-soft) px-3 py-2.5 text-[13px] leading-relaxed text-(--danger)"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em]">Error</span>
        <span className="rounded bg-(--danger)/10 px-1.5 py-px font-mono text-[11px]">{error.code}</span>
        {error.tool ? <span className="font-mono text-[11px] opacity-80">tool: {error.tool}</span> : null}
      </div>
      <p className="mt-1 font-medium">{error.message}</p>
      {error.detail != null ? (
        <pre className="panel-scroll mt-1.5 max-h-32 overflow-auto whitespace-pre-wrap break-words rounded bg--card/60 p-1.5 font-mono text-[11px] font-normal text-(--danger)">
          {typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
