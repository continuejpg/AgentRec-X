// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): kept the docked single-field composer and added a
// real Stop control (AbortController), an explicit disabled reason while the BFF is starting, and
// data-testid hooks for the Playwright owner. Cart/bag vocabulary removed.

"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "./icons";

export interface Prefill {
  text: string;
  /** Changes on every request so the same text can be offered twice. */
  nonce: number;
}

export function Composer({
  send,
  stop,
  ready,
  busy,
  label = "Message the AgentRec-X shopping assistant",
  placeholder = "Ask for a product, a comparison, or share a preference…",
  disabledHint,
  prefill,
  className = "",
}: {
  send: (text: string) => void;
  stop?: () => void;
  ready: boolean;
  busy: boolean;
  label?: string;
  placeholder?: string;
  disabledHint?: string;
  prefill?: Prefill | null;
  className?: string;
}) {
  const [draft, setDraft] = useState("");
  const boxRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!prefill) return;
    setDraft(prefill.text);
    boxRef.current?.focus();
  }, [prefill]);

  // Grow with the draft up to the CSS max-height, then scroll.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, 160)}px`;
  }, [draft]);

  const submit = () => {
    if (!draft.trim() || busy || !ready) return;
    send(draft);
    setDraft("");
  };

  const disabled = !ready || busy;
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <form
        data-testid="composer"
        className="flex items-center gap-2 rounded-[16px] border border-(--line-strong) bg-(--card) py-1.5 pl-4 pr-1.5 shadow-(--shadow) transition-colors focus-within:border-(--accent)"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={boxRef}
          data-testid="composer-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
          rows={1}
          aria-label={label}
          disabled={!ready}
          placeholder={busy ? "Working…" : ready ? placeholder : (disabledHint ?? "Waiting for the local BFF…")}
          className="max-h-40 min-w-0 flex-1 resize-none bg-transparent py-1.5 text-[16px] text-(--ink) outline-none transition placeholder:text-(--ink-soft)/70 disabled:cursor-not-allowed"
        />
        {busy && stop ? (
          <button
            type="button"
            data-testid="composer-stop"
            onClick={stop}
            aria-label="Stop the current response"
            title="Stop the current response (AbortController)"
            className="grid h-9 w-9 shrink-0 place-items-center rounded-[11px] border border-(--line-strong) bg-(--card) text-(--ink-2) transition hover:border-(--danger) hover:text-(--danger)"
          >
            <Icon name="stop" size={14} />
          </button>
        ) : null}
        <button
          type="submit"
          data-testid="composer-send"
          disabled={disabled || !draft.trim()}
          aria-label="Send message"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-[11px] bg-(--ink) text-(--surface) transition hover:brightness-110 disabled:opacity-35"
        >
          <Icon name="arrow-up" size={16} />
        </button>
      </form>
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 px-1 text-[10.5px] text-(--ink-faint)">
        <span>{ready ? "Catalogue evidence only · no price, stock or rating is invented" : (disabledHint ?? "Waiting for the local demo services…")}</span>
        <span className="hidden sm:inline">Enter sends · Shift+Enter for a new line</span>
      </div>
    </div>
  );
}
