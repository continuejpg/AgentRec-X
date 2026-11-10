// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): Inspector's tool rows become an inline chat
// timeline driven by the BFF `tool_status` events; phase copy comes from real tool names, a live
// elapsed counter is shown for running calls, and no input/chain-of-thought is displayed.

"use client";

import { useEffect, useState } from "react";
import type { LocalToolStatus } from "@/lib/chat";
import { formatDuration, formatLatency, toolPhaseLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Pill } from "./ui";

/** Running rows tick locally; finished rows show the real `latency_ms` from the BFF. */
function RunningGlyph() {
  return (
    <span
      aria-hidden
      className="inline-block h-3 w-3 shrink-0 animate-spin rounded-full border-[1.5px] border-(--accent) border-r-transparent"
    />
  );
}

export default function ToolTimeline({
  activity,
  streaming,
  totalMs,
  stopReason,
  lastHeartbeatAt,
  defaultOpen,
}: {
  activity: LocalToolStatus[];
  streaming: boolean;
  totalMs: number | null;
  stopReason: string | null;
  lastHeartbeatAt?: number | null;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen ?? streaming);
  const [now, setNow] = useState(0);

  useEffect(() => {
    if (streaming) setOpen(true);
  }, [streaming]);

  useEffect(() => {
    if (!streaming) return;
    setNow(performance.now());
    const timer = window.setInterval(() => setNow(performance.now()), 500);
    return () => window.clearInterval(timer);
  }, [streaming]);

  if (!activity.length && !streaming) return null;

  const running = activity.filter((row) => row.status === "running").length;
  const failed = activity.filter((row) => row.status === "error").length;
  // Browser-local rows whose completion the backend never reported (see lib/chat.ts); shown
  // honestly as unreported, never as a fake spinner or a fabricated success.
  const unreported = activity.filter((row) => row.status === "unknown").length;
  const heartbeatAgo =
    streaming && lastHeartbeatAt != null && now > 0 ? Math.max(0, Math.round((now - lastHeartbeatAt) / 1000)) : null;

  return (
    <div className="overflow-hidden rounded-xl border border-(--line) bg-(--card)/70" data-testid="tool-timeline">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-(--ground)/60"
      >
        <span className={`shrink-0 ${running ? "text-(--accent)" : "text-(--ink-soft)"}`}>
          {running ? <RunningGlyph /> : <Icon name="clock" size={14} />}
        </span>
        <span className="min-w-0 flex-1 text-[12.5px] font-semibold text-(--ink)">
          Tool activity
          <span className="ml-1.5 font-normal text-(--ink-soft)">
            {activity.length} step{activity.length === 1 ? "" : "s"}
            {running ? " · running" : ""}
            {failed ? ` · ${failed} failed` : ""}
            {unreported ? ` · ${unreported} unreported` : ""}
          </span>
        </span>
        {totalMs != null ? (
          <span className="shrink-0 font-mono text-[11px] tabular-nums text-(--ink-soft)">{formatDuration(totalMs)}</span>
        ) : null}
        {heartbeatAgo != null && !running ? (
          <span className="hidden shrink-0 font-mono text-[10.5px] text-(--ink-faint) sm:inline">
            keep-alive {heartbeatAgo}s
          </span>
        ) : null}
        <Icon name={open ? "chevron-down" : "chevron-right"} size={14} className="shrink-0 text-(--ink-soft)" />
      </button>
      {open ? (
        <div className="border-t border-(--line) px-3 py-2">
          {activity.length ? (
            <ul className="space-y-1.5">
              {activity.map((row, index) => {
                const label = toolPhaseLabel(row.tool, row.label);
                const elapsed =
                  row.status === "running" && row.receivedAt != null && now > 0
                    ? now - row.receivedAt
                    : row.latency_ms;
                return (
                  <li
                    key={`${row.tool}-${index}`}
                    data-testid="tool-row"
                    data-tool={row.tool}
                    data-status={row.status}
                    className="flex items-start gap-2"
                  >
                    <span className="mt-px w-3.5 shrink-0 text-center leading-none">
                      {row.status === "running" ? (
                        <RunningGlyph />
                      ) : row.status === "ok" ? (
                        <Icon name="check" size={12} className="text-(--ok)" />
                      ) : row.status === "unknown" ? (
                        <Icon name="info" size={12} className="text-(--ink-faint)" />
                      ) : (
                        <Icon name="alert" size={12} className="text-(--danger)" />
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <span
                          className={`text-[12.5px] ${
                            row.status === "error" ? "font-semibold text-(--danger)" : "font-medium text-(--ink)"
                          }`}
                        >
                          {label}
                        </span>
                        <span className="font-mono text-[10.5px] text-(--ink-faint)" title="MCP tool name">
                          {row.tool}
                        </span>
                        <span className="ml-auto shrink-0 font-mono text-[10.5px] tabular-nums text-(--ink-soft)">
                          {row.status === "running"
                            ? elapsed != null
                              ? `${formatLatency(elapsed)}…`
                              : "running…"
                            : row.status === "unknown"
                              ? "unreported"
                              : formatLatency(elapsed)}
                        </span>
                      </div>
                      {row.detail ? (
                        <p
                          className={`mt-0.5 whitespace-pre-wrap break-words text-[11.5px] leading-snug ${
                            row.status === "error" ? "text-(--danger)" : "text-(--ink-soft)"
                          }`}
                        >
                          {row.detail}
                        </p>
                      ) : null}
                    </div>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-[12px] text-(--ink-soft)">Waiting for the first real tool call…</p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-(--line)/60 pt-1.5">
            <span className="text-[10.5px] text-(--ink-faint)">
              Real MCP tool calls only · no hidden reasoning is shown.
            </span>
            {stopReason ? (
              <Pill tone={stopReason === "finalized" ? "ok" : stopReason === "awaiting_clarification" ? "warn" : "muted"} className="ml-auto">
                {stopReason === "finalized"
                  ? "finalized"
                  : stopReason === "awaiting_clarification"
                    ? "awaiting clarification"
                    : stopReason === "comparison"
                      ? "comparison"
                      : stopReason}
              </Pill>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
