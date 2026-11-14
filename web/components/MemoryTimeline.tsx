// AgentRec-X Final Demo V2 - Event-Sourced Memory timeline.
// Renders only entries returned by the real M9 preference store via the BFF view model.
"use client";

import { useMemo } from "react";
import type { PreferenceItem, PreferenceState } from "@/lib/types";
import { Icon } from "./icons";
import { Pill } from "./ui";

function eventLabel(item: PreferenceItem): string {
  const status = (item.status || "").toLowerCase();
  if (status === "active") return "ACTIVE";
  if (status === "superseded") return "SUPERSEDED";
  if (status === "removed") return "WITHDRAWN";
  return (item.status || "UNKNOWN").toUpperCase();
}

function toneFor(item: PreferenceItem): "ok" | "warn" | "muted" | "danger" {
  const status = (item.status || "").toLowerCase();
  if (status === "active") return (item.polarity || "").toLowerCase().includes("avoid") ? "warn" : "ok";
  if (status === "superseded") return "muted";
  if (status === "removed") return "danger";
  return "muted";
}

export default function MemoryTimeline({ state }: { state: PreferenceState | null }) {
  const events = useMemo(() => {
    if (!state) return [];
    const all = [...state.active, ...state.inactive];
    const seen = new Set<string>();
    return all
      .filter((item) => {
        const key = item.memory_id || `${item.value}-${item.source_turn_id}-${item.status}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .sort((left, right) => {
        const a = left.logical_seq ?? 1_000_000;
        const b = right.logical_seq ?? 1_000_000;
        if (a !== b) return a - b;
        return String(left.memory_id).localeCompare(String(right.memory_id));
      });
  }, [state]);

  return (
    <section className="border-t border-(--line) pt-3" data-testid="memory-timeline">
      <div className="mb-1.5 flex items-center gap-2">
        <h2 className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Memory timeline</h2>
        <span className="text-[10px] text-(--ink-faint)">{events.length} events</span>
        <Pill tone="muted">M9 audit</Pill>
      </div>
      {!state ? (
        <p className="text-[11px] text-(--ink-soft)">Active memory loads from the real M9 store.</p>
      ) : !events.length ? (
        <p className="rounded-lg border border-dashed border-(--line-strong) px-2 py-2 text-[11px] text-(--ink-soft)">
          No preference events for this demo user yet. Clarification replies can write real events.
        </p>
      ) : (
        <ol className="space-y-1.5" aria-label="Memory events, oldest first">
          {events.map((item, index) => (
            <li
              key={item.memory_id || `${item.value}-${index}`}
              className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5"
              data-testid="memory-event"
            >
              <div className="flex items-center gap-1.5">
                <span className="grid h-4 w-4 place-items-center rounded-full bg-(--well) font-mono text-[9px] text-(--ink-soft)">
                  {item.logical_seq ?? index + 1}
                </span>
                <span className="min-w-0 flex-1 truncate text-[11.5px] font-semibold text-(--ink)">
                  {item.value || "UNKNOWN value"}
                </span>
                <Pill tone={toneFor(item)}>{eventLabel(item)}</Pill>
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 pl-5 text-[9.5px] text-(--ink-faint)">
                <span className="font-mono">{item.kind || "preference"}</span>
                <span aria-hidden>路</span>
                <span>{item.polarity || "unknown polarity"}</span>
                {item.source_turn_id ? (
                  <>
                    <span aria-hidden>路</span>
                    <span className="font-mono">turn {item.source_turn_id}</span>
                  </>
                ) : null}
                {item.memory_id ? (
                  <>
                    <span aria-hidden>路</span>
                    <span className="font-mono" title={item.memory_id}>
                      id {item.memory_id.slice(0, 8)}...                    </span>
                  </>
                ) : null}
              </div>
              {item.source_text ? (
                <p className="mt-1 line-clamp-2 pl-5 text-[10px] italic leading-snug text-(--ink-soft)">
                  “{item.source_text}”
                </p>
              ) : null}
              {item.supersedes || item.superseded_by ? (
                <p className="mt-0.5 pl-5 font-mono text-[9px] text-(--ink-faint)">
                  {item.supersedes ? `supersedes ${String(item.supersedes).slice(0, 8)}` : ""}
                  {item.supersedes && item.superseded_by ? " 路 " : ""}
                  {item.superseded_by ? `superseded_by ${String(item.superseded_by).slice(0, 8)}` : ""}
                </p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
      <p className="mt-1.5 flex items-start gap-1 text-[9.5px] leading-snug text-(--ink-faint)">
        <Icon name="info" size={10} className="mt-0.5 shrink-0" />
        Withdrawal/supersession is shown as an event state; the original event is not deleted.
      </p>
    </section>
  );
}
