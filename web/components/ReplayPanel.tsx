// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useState } from "react";
import type { ErrorPayload, ReplayExample, ReplayResponse } from "@/lib/types";
import { formatLatency, humanize, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Button, EmptyState, Pill, Skeleton } from "./ui";
import { ErrorBubble } from "./MessageBubble";

function GateReplay({ example }: { example: ReplayExample }) {
  const gate = example.gate;
  if (!gate) return <p className="text-[11.5px] text-(--ink-soft)">No Gate block in this saved trajectory.</p>;
  return (
    <div className="rounded-lg border border-(--line) bg-(--card) p-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10.5px] font-bold uppercase tracking-wide text-(--ink-soft)">Gate</span>
        <Pill tone={gate.action === "ALLOW_FINALIZE" ? "ok" : gate.action ? "warn" : "muted"}>
          {gate.action ?? "no action"}
        </Pill>
        <span className="font-mono text-[10px] text-(--ink-faint)">{gate.reason_code ?? ""}</span>
      </div>
      {gate.reason ? <p className="mt-1 text-[11.5px] leading-snug text-(--ink-2)">{gate.reason}</p> : null}
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-(--ink-soft)">
        {gate.selected_candidate_id ? <span>selected {gate.selected_candidate_id}</span> : null}
        {gate.candidate_presentable_ids?.length ? (
          <span>presentable {gate.candidate_presentable_ids.length}</span>
        ) : null}
        {gate.product_role ? <span>role {gate.product_role}</span> : null}
      </div>
    </div>
  );
}

function ModelStats({ example }: { example: ReplayExample }) {
  const stats = example.model_stats;
  if (!stats) return null;
  const total = stats.latency_ms.reduce((sum, value) => sum + value, 0);
  return (
    <div className="rounded-lg border border-(--line) bg-(--card) p-2">
      <div className="text-[10.5px] font-bold uppercase tracking-wide text-(--ink-soft)">Saved model stats</div>
      <div className="mt-1 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        <div className="rounded bg-(--well)/60 px-1.5 py-1">
          <div className="text-[9.5px] text-(--ink-soft)">calls</div>
          <div className="font-mono text-[12.5px] font-semibold text-(--ink)">{stats.calls}</div>
        </div>
        <div className="rounded bg-(--well)/60 px-1.5 py-1">
          <div className="text-[9.5px] text-(--ink-soft)">sum latency</div>
          <div className="font-mono text-[12.5px] font-semibold text-(--ink)">{formatLatency(total)}</div>
        </div>
        <div className="rounded bg-(--well)/60 px-1.5 py-1">
          <div className="text-[9.5px] text-(--ink-soft)">input tokens</div>
          <div className="font-mono text-[12.5px] font-semibold text-(--ink)">{stats.input_tokens}</div>
        </div>
        <div className="rounded bg-(--well)/60 px-1.5 py-1">
          <div className="text-[9.5px] text-(--ink-soft)">output tokens</div>
          <div className="font-mono text-[12.5px] font-semibold text-(--ink)">{stats.output_tokens}</div>
        </div>
      </div>
      {stats.latency_ms.length ? (
        <div className="mt-1 font-mono text-[10px] text-(--ink-faint)">
          per-call: {stats.latency_ms.map((value) => formatLatency(value)).join(" · ")}
        </div>
      ) : null}
      {stats.parser_errors.length ? (
        <div className="mt-1 text-[10.5px] text-(--danger)">parser errors: {stats.parser_errors.join("; ")}</div>
      ) : null}
    </div>
  );
}

export default function ReplayPanel({
  replay,
  loading,
  error,
  onLoad,
}: {
  replay: ReplayResponse | null;
  loading: boolean;
  error: ErrorPayload | null;
  onLoad: () => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const examples = replay?.examples ?? [];
  const selected = examples[Math.min(selectedIndex, Math.max(0, examples.length - 1))] ?? null;
  const badge = replay?.badge ?? "Historical Replay — Not a Live Model Call";

  return (
    <div className="space-y-3" data-testid="replay-panel">
      <div className="rounded-2xl border-2 border-amber-500/70 bg-amber-50 p-3 text-amber-950">
        <div className="flex items-start gap-2">
          <Icon name="alert" size={18} className="mt-0.5 shrink-0 text-amber-600" />
          <div>
            <div className="text-[13px] font-bold uppercase tracking-[0.04em]">{badge}</div>
            <p className="mt-1 text-[12px] leading-relaxed">
              These are saved Qwen3.5-9B Base trajectories from earlier milestones (P14-C1 / P15), replayed from{" "}
              <span className="font-mono">runs/p18/qwen_replay.json</span>. No model request is made by this page.
            </p>
            <div className="mt-1 flex flex-wrap gap-1.5 font-mono text-[10.5px]">
              <span className="rounded bg-amber-600/15 px-1.5 py-px">live_call=false</span>
              <span className="rounded bg-amber-600/15 px-1.5 py-px">replay=true</span>
              {replay?.mode ? <span className="rounded bg-amber-600/15 px-1.5 py-px">mode={replay.mode}</span> : null}
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="space-y-1.5">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : error ? (
        <div className="space-y-2">
          <ErrorBubble error={error} />
          <Button size="sm" variant="secondary" icon="refresh" onClick={onLoad}>
            Retry GET /api/replay
          </Button>
        </div>
      ) : !examples.length ? (
        <EmptyState
          icon="play"
          title="No saved trajectories"
          hint="GET /api/replay returned no examples, so there is nothing to replay. Nothing is generated to fill the gap."
        >
          <Button size="sm" variant="secondary" icon="refresh" onClick={onLoad}>
            Reload replay
          </Button>
        </EmptyState>
      ) : (
        <>
          <div>
            <div className="mb-1 flex items-center gap-2">
              <h3 className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">
                Saved trajectories
              </h3>
              <span className="text-[10.5px] text-(--ink-faint)">{examples.length}</span>
            </div>
            <ul className="space-y-1">
              {examples.map((example, index) => (
                <li key={`${example.task_id}-${index}`}>
                  <button
                    type="button"
                    onClick={() => setSelectedIndex(index)}
                    aria-pressed={index === selectedIndex}
                    className={`w-full rounded-lg border px-2 py-1.5 text-left transition-colors ${
                      index === selectedIndex
                        ? "border-(--ink) bg-(--ink) text-(--surface)"
                        : "border-(--line) bg-(--card) text-(--ink) hover:border-(--accent)"
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] opacity-80">{example.task_id}</span>
                      <span
                        className={`ml-auto rounded-full px-1.5 py-px text-[9.5px] font-semibold ${
                          index === selectedIndex ? "bg-white/15" : "bg-(--well) text-(--ink-soft)"
                        }`}
                      >
                        {example.terminal ? humanize(example.terminal.status) : "final-answer only"}
                      </span>
                    </div>
                    <div className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug">{example.query}</div>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {selected ? (
            <div className="space-y-2" data-testid="replay-example" data-task-id={selected.task_id}>
              <div className="rounded-xl border border-(--line) bg-(--card) p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wide text-(--ink-soft)">Query (saved)</div>
                <p className="mt-0.5 text-[12.5px] leading-relaxed text-(--ink)">{selected.query}</p>
                <div className="mt-1 break-all font-mono text-[10px] text-(--ink-faint)">source: {selected.source}</div>
              </div>
              {selected.terminal ? (
                <div className="rounded-xl border border-(--line) bg-(--card) p-2.5" data-testid="replay-terminal">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[10.5px] font-bold uppercase tracking-wide text-(--ink-soft)">Terminal</span>
                    <Pill tone={selected.terminal.status === "finalized" ? "ok" : "warn"}>
                      {statusLabel(selected.terminal.status)}
                    </Pill>
                    {selected.terminal.selected_candidate_id ? (
                      <span className="font-mono text-[10.5px] text-(--ink-soft)">
                        selected {selected.terminal.selected_candidate_id}
                      </span>
                    ) : null}
                  </div>
                  {selected.terminal.ask_question ? (
                    <p className="mt-1 text-[11.5px] leading-snug text-(--ink-2)">
                      Asked: <span className="italic">{selected.terminal.ask_question}</span>
                    </p>
                  ) : null}
                </div>
              ) : (
                <div
                  className="rounded-xl border border-amber-500/50 bg-amber-50 p-2.5 text-amber-950"
                  data-testid="replay-terminal-missing"
                >
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[10.5px] font-bold uppercase tracking-wide">Terminal</span>
                    <Pill tone="warn">not recorded</Pill>
                  </div>
                  <p className="mt-1 text-[11.5px] leading-snug">
                    terminal: not recorded (final-answer-only trajectory). This saved trajectory has no terminal
                    event, so no selected candidate, Gate decision or clarification question is shown — those fields
                    are not fabricated.
                  </p>
                  {selected.final_answer ? (
                    <>
                      <div className="mt-2 text-[10px] font-bold uppercase tracking-wide">
                        Saved final answer (verbatim; not a terminal event)
                      </div>
                      <p className="mt-0.5 whitespace-pre-wrap text-[11.5px] leading-relaxed">
                        {selected.final_answer}
                      </p>
                    </>
                  ) : null}
                </div>
              )}
              <GateReplay example={selected} />
              <ModelStats example={selected} />
              <p className="text-[10.5px] leading-snug text-(--ink-faint)">
                Replay is not a live Qwen call and cannot be interacted with. Switch back to CPU Live to chat with the
                real local MCP tool chain.
              </p>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
