// AgentRec-X Final Demo V2 - explicit, real M9 write control for live scenarios.
// This calls the BFF -> MCP remember_preference adapter; it is not a new memory store.
"use client";

import { useState } from "react";
import { api, errorPayload } from "@/lib/api";
import type { MemoryRememberResponse, PreferenceState } from "@/lib/types";
import { Button, Pill } from "./ui";

export default function MemoryLab({
  sessionId,
  writeEnabled,
  onUpdated,
}: {
  sessionId: string | null;
  writeEnabled: boolean;
  onUpdated: (preferences: PreferenceState) => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<MemoryRememberResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (message: string) => {
    if (!sessionId || !message.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.rememberMemory(sessionId, message.trim());
      setResult(response);
      onUpdated(response.preferences);
      setText("");
    } catch (caught) {
      setError(errorPayload(caught).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="border-t border-(--line) pt-3" data-testid="memory-lab">
      <div className="mb-1.5 flex items-center gap-2">
        <h2 className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Memory lab</h2>
        <Pill tone={writeEnabled ? "ok" : "muted"}>{writeEnabled ? "M9 writes on" : "read-only"}</Pill>
      </div>
      <p className="mb-1.5 text-[10px] leading-snug text-(--ink-faint)">
        Explicit user-authored text through the same M9 writer and audit trail as clarification replies.
        It does not rerank by itself.
      </p>
      <div className="flex gap-1">
        <input
          value={text}
          disabled={!sessionId || !writeEnabled || busy}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void submit(text);
          }}
          placeholder="e.g. I prefer a microphone."
          className="min-w-0 flex-1 rounded-lg border border-(--line) bg-(--card) px-2 py-1.5 text-[11px] text-(--ink) outline-none focus:border-(--accent)"
          data-testid="memory-lab-input"
        />
        <Button
          size="sm"
          variant="secondary"
          disabled={!sessionId || !writeEnabled || busy || !text.trim()}
          onClick={() => void submit(text)}
          data-testid="memory-lab-save"
        >
          {busy ? "Saving..." : "Remember"}
        </Button>
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        <button
          type="button"
          disabled={!sessionId || !writeEnabled || busy}
          onClick={() => void submit("I prefer a microphone.")}
          className="rounded-md border border-(--line) px-1.5 py-0.5 text-[9.5px] text-(--ink-soft) hover:border-(--accent)"
        >
          + microphone
        </button>
        <button
          type="button"
          disabled={!sessionId || !writeEnabled || busy}
          onClick={() => void submit("I do not want over-ear headphones.")}
          className="rounded-md border border-(--line) px-1.5 py-0.5 text-[9.5px] text-(--ink-soft) hover:border-(--accent)"
        >
          avoid over-ear
        </button>
        <button
          type="button"
          disabled={!sessionId || !writeEnabled || busy}
          onClick={() => void submit("Forget my microphone preference.")}
          className="rounded-md border border-(--line) px-1.5 py-0.5 text-[9.5px] text-(--ink-soft) hover:border-(--warn)"
        >
          withdraw microphone
        </button>
      </div>
      {result ? (
        <p className="mt-1.5 rounded-lg bg-(--ok-soft)/50 px-2 py-1 text-[10px] leading-snug text-(--ok)" data-testid="memory-lab-result">
          M9 update: added {String((result.update as Record<string, unknown>)?.added ?? 0)}, removed{" "}
          {String((result.update as Record<string, unknown>)?.removed ?? 0)}, superseded{" "}
          {String((result.update as Record<string, unknown>)?.superseded ?? 0)}. Recommendation is not rerun automatically.
        </p>
      ) : null}
      {error ? <p className="mt-1 text-[10px] leading-snug text-(--danger)">{error}</p> : null}
    </section>
  );
}
