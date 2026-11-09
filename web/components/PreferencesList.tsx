// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): stores the Inspector memory list as a preference
// panel over the BFF PreferenceState (active vs inactive/audit, source text, provenance),
// with explicit M9 provenance and a real refresh action.

"use client";

import { useState } from "react";
import type { ErrorPayload, PreferenceItem, PreferenceState } from "@/lib/types";
import { formatDateTime, formatRelativeTime, plural, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { IconButton, Pill, Skeleton } from "./ui";

function polarityTone(item: PreferenceItem): "ok" | "danger" | "muted" {
  if (item.polarity === "avoid") return "danger";
  if (item.polarity === "prefer") return "ok";
  return "muted";
}

function PreferenceRow({ item, inactive = false }: { item: PreferenceItem; inactive?: boolean }) {
  return (
    <li
      data-testid={inactive ? "preference-inactive-item" : "preference-active-item"}
      data-memory-id={item.memory_id}
      className={`rounded-lg border px-2.5 py-2 ${
        inactive ? "border-(--line) bg-(--well)/40 opacity-80" : "border-(--line) bg-(--card)"
      }`}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <Pill tone={polarityTone(item)}>{statusLabel(item.polarity)}</Pill>
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-(--ink)" title={item.value ?? undefined}>
          {item.value}
        </span>
        <Pill tone="muted">{statusLabel(item.kind)}</Pill>
        <Pill tone={inactive ? "muted" : "ok"} dot>
          {statusLabel(item.status)}
        </Pill>
      </div>
      {item.source_text ? (
        <p className="mt-1 line-clamp-2 text-[11.5px] italic leading-snug text-(--ink-soft)" title={item.source_text ?? undefined}>
          “{item.source_text}”
        </p>
      ) : null}
      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[10px] text-(--ink-faint)">
        <span title={item.memory_id}>{item.memory_id}</span>
        {item.source_turn_id ? <span>turn {item.source_turn_id}</span> : null}
        <span title={formatDateTime(item.last_updated)}>{formatRelativeTime(item.last_updated)}</span>
      </div>
    </li>
  );
}

export default function PreferencesList({
  state,
  loading,
  error,
  onRefresh,
  className = "",
}: {
  state: PreferenceState | null;
  loading: boolean;
  error: ErrorPayload | null;
  onRefresh: () => void;
  className?: string;
}) {
  const [showInactive, setShowInactive] = useState(false);
  const active = state?.active ?? [];
  const inactive = state?.inactive ?? [];
  const activeOnly = state?.history_status === "active_preferences";

  return (
    <section className={className} data-testid="preferences-panel" aria-labelledby="preferences-heading">
      <div className="flex items-center gap-2">
        <h2 id="preferences-heading" className="text-[12px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">
          Active preferences
        </h2>
        {state ? <span className="text-[11px] font-medium text-(--ink-soft)">{active.length}</span> : null}
        <IconButton icon="refresh" label="Refresh active preferences" onClick={onRefresh} size={26} className="ml-auto" />
      </div>

      {loading && !state ? (
        <div className="mt-2 space-y-1.5">
          <Skeleton className="h-14 w-full" />
          <Skeleton className="h-14 w-full" />
        </div>
      ) : error ? (
        <div className="mt-2 rounded-lg border border-(--danger)/40 bg-(--danger-soft) px-2.5 py-2 text-[11.5px] text-(--danger)" role="alert">
          <div className="font-semibold">Could not load preferences</div>
          <div className="mt-0.5 break-words">{error.message}</div>
          {error.code ? <div className="mt-0.5 font-mono text-[10.5px] opacity-80">{error.code}</div> : null}
          <button type="button" onClick={onRefresh} className="mt-1 font-semibold underline underline-offset-2">
            Retry
          </button>
        </div>
      ) : !state ? (
        <p className="mt-2 text-[11.5px] text-(--ink-soft)">Preferences load once the session and MCP memory are ready.</p>
      ) : (
        <div className="mt-2">
          {!activeOnly ? (
            <div className="mb-2 rounded-lg border border-(--warn)/40 bg-(--warn-soft)/50 px-2.5 py-1.5 text-[11px] text-(--warn)">
              M9 history status is <span className="font-mono">{state.history_status}</span>; only
              <span className="font-mono"> active_preferences</span> entries may be shown as active.
            </div>
          ) : null}
          {active.length ? (
            <ul className="space-y-1.5">
              {active.map((item) => (
                <PreferenceRow key={item.memory_id} item={item} />
              ))}
            </ul>
          ) : (
            <p className="rounded-lg border border-dashed border-(--line-strong) px-2.5 py-2 text-[11.5px] leading-relaxed text-(--ink-soft)">
              No active preferences returned by M9 for this profile yet. They appear here only after a real memory
              write.
            </p>
          )}

          <div className="mt-2">
            <button
              type="button"
              onClick={() => setShowInactive((value) => !value)}
              aria-expanded={showInactive}
              data-testid="preference-inactive-toggle"
              className="flex w-full items-center gap-1.5 rounded-lg px-1 py-1 text-left text-[11px] font-semibold text-(--ink-soft) hover:text-(--ink)"
            >
              <Icon name={showInactive ? "chevron-down" : "chevron-right"} size={12} />
              Inactive / audit history
              <span className="font-normal">({plural(inactive.length, "entry", "entries")})</span>
            </button>
            {showInactive ? (
              inactive.length ? (
                <ul className="mt-1 space-y-1.5" data-testid="preference-inactive-list">
                  {inactive.map((item) => (
                    <PreferenceRow key={item.memory_id} item={item} inactive />
                  ))}
                </ul>
              ) : (
                <p className="mt-1 px-1 text-[11px] text-(--ink-soft)">
                  No withdrawn or superseded preferences in the current M9 state.
                </p>
              )
            ) : null}
          </div>

          <div className="mt-2 border-t border-(--line) pt-1.5 font-mono text-[10px] leading-relaxed text-(--ink-faint)">
            <div className="truncate" title={state.audit_source ?? undefined}>
              {state.audit_source}
            </div>
            <div>
              fetched <span title={formatDateTime(state.fetched_at)}>{formatRelativeTime(state.fetched_at)}</span> · user{" "}
              {state.user_key}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
