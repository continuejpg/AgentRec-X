// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import type { QwenLiveState } from "@/lib/modes";
import { formatLatency } from "@/lib/format";
import { Icon } from "./icons";
import { Button, Pill } from "./ui";
import type { UiMode } from "./Header";

/**
 * Explicit Qwen Base Live availability, from the real BFF health/config.
 *
 * - Unavailable: the backend's own `disabled_reason` is shown verbatim; the mode cannot be
 *   selected and there is no fallback.
 * - Unavailable *while selected*: an additional danger notice states that nothing is being run
 *   and the user must switch back (never a silent CPU fallback).
 * - Available: nothing is rendered here; the header carries the live badge.
 */
export default function QwenLiveNotice({
  qwenLive,
  mode,
  onSwitchToCpu,
}: {
  qwenLive: QwenLiveState;
  mode: UiMode;
  onSwitchToCpu: () => void;
}) {
  if (qwenLive.available) return null;
  const selected = mode === "qwen_live";
  const reason = qwenLive.disabledReason ?? "No live Qwen service detected";
  const meta: string[] = [];
  if (qwenLive.status) meta.push(`status=${qwenLive.status}`);
  if (qwenLive.enabled != null) meta.push(`enabled=${qwenLive.enabled}`);
  if (qwenLive.configured != null) meta.push(`configured=${qwenLive.configured}`);
  if (qwenLive.model) meta.push(`model=${qwenLive.model}`);
  if (qwenLive.latencyMs != null) meta.push(`probe=${formatLatency(qwenLive.latencyMs)}`);

  return (
    <section
      data-testid="qwen-unavailable-notice"
      data-state={selected ? "selected-unavailable" : "unavailable"}
      role={selected ? "alert" : "status"}
      aria-live="polite"
      className={`mt-4 rounded-xl border p-3 ${
        selected ? "border-(--danger)/50 bg-(--danger-soft)/40" : "border-(--line) bg-(--card)"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Icon
          name="alert"
          size={14}
          className={selected ? "text-(--danger)" : "text-(--ink-faint)"}
        />
        <span className="text-[12px] font-bold uppercase tracking-[0.04em] text-(--ink)">
          {selected ? "Qwen Live is unavailable — this turn was not run" : "Qwen Live — Unavailable"}
        </span>
        <Pill tone={selected ? "danger" : "muted"} className="ml-auto">
          {qwenLive.status ?? "unavailable"}
        </Pill>
      </div>
      <p
        className={`mt-1 whitespace-pre-wrap break-words text-[12px] leading-relaxed ${
          selected ? "text-(--danger)" : "text-(--ink-2)"
        }`}
      >
        {reason}
      </p>
      {selected ? (
        <p className="mt-1 text-[11.5px] leading-snug text-(--ink-2)">
          The backend no longer reports Qwen Base Live as available, so nothing is executed and no
          CPU/Replay fallback is substituted. Switch back to CPU Live to continue this session.
        </p>
      ) : (
        <p className="mt-1 text-[11.5px] leading-snug text-(--ink-soft)">
          Qwen Live can only be selected when the backend health probe reports it available. CPU
          Live remains the default live mode and Qwen Replay remains a read-only historical replay.
        </p>
      )}
      {meta.length ? (
        <p className="mt-1 break-words font-mono text-[10px] text-(--ink-faint)">{meta.join(" · ")}</p>
      ) : null}
      {selected ? (
        <Button size="sm" variant="secondary" icon="refresh" className="mt-2" onClick={onSwitchToCpu}>
          Switch back to CPU Live
        </Button>
      ) : null}
    </section>
  );
}
