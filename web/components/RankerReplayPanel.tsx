// AgentRec-X Final Demo V2 - recorded P100 ranker comparison / memory OFF-ON replay.
// This panel is READ-ONLY: it renders saved Phase 5.1 GPU predictions and never calls a model.
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorPayload } from "@/lib/api";
import type { RankerReplayItem, RankerReplayResponse } from "@/lib/types";
import { humanize } from "@/lib/format";
import { Icon } from "./icons";
import { Pill } from "./ui";

const ARMS: { id: "existing" | "natural" | "hybrid"; label: string; note: string }[] = [
  { id: "existing", label: "Existing Scorer", note: "Production default" },
  { id: "natural", label: "ShopRanker Natural", note: "Experimental Shadow · Memory" },
  { id: "hybrid", label: "Fixed Hybrid", note: "Experimental Shadow · Structured memory" },
];

function RankList({ items, mode }: { items: RankerReplayItem[]; mode: "off" | "on" }) {
  if (!items.length) {
    return <p className="text-[11px] text-(--ink-soft)">No recorded items.</p>;
  }
  return (
    <ol className="space-y-1">
      {items.map((item) => (
        <li key={`${mode}-${item.product_id}`} className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5">
          <div className="flex items-start gap-1.5">
            <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-(--well) font-mono text-[9px] text-(--ink-soft)">
              {item.rank}
            </span>
            <div className="min-w-0 flex-1">
              <div className="line-clamp-2 text-[11.5px] font-semibold leading-snug text-(--ink)">{item.title}</div>
              <div className="mt-0.5 flex flex-wrap items-center gap-1 text-[9.5px] text-(--ink-faint)">
                <span className="font-mono">{item.product_id}</span>
                <span aria-hidden>路</span>
                <span className="font-mono">score {item.score.toFixed(4)}</span>
                {mode === "on" && item.off_rank != null ? (
                  <>
                    <span aria-hidden>路</span>
                    <span>
                      OFF rank {item.off_rank}
                      {item.rank_movement != null && item.rank_movement !== 0
                        ? ` (${item.rank_movement > 0 ? "+" : ""}${item.rank_movement})`
                        : " (no movement)"}
                    </span>
                  </>
                ) : null}
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {item.memory_matches.slice(0, 3).map((value) => (
                  <Pill key={`match-${item.product_id}-${value}`} tone="ok">
                    match {value}
                  </Pill>
                ))}
                {item.memory_conflicts.slice(0, 2).map((value) => (
                  <Pill key={`conflict-${item.product_id}-${value}`} tone="warn">
                    conflict {value}
                  </Pill>
                ))}
                {item.memory_unknown.slice(0, 2).map((value) => (
                  <Pill key={`unknown-${item.product_id}-${value}`} tone="muted">
                    unknown {value}
                  </Pill>
                ))}
                {!item.evidence_count ? <Pill tone="muted">no evidence refs</Pill> : null}
              </div>
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}

export default function RankerReplayPanel() {
  const [data, setData] = useState<RankerReplayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [arm, setArm] = useState<"existing" | "natural" | "hybrid">("existing");
  const [mode, setMode] = useState<"off" | "on">("on");

  const load = useCallback(async (nextTaskId: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const payload = await api.replayRankers(nextTaskId);
      if (!payload.available) {
        setError(payload.detail || "P100 ranker replay artifact is unavailable.");
        setData(null);
      } else {
        setData(payload);
        setTaskId(payload.task_id ?? null);
      }
    } catch (caught) {
      const parsed = errorPayload(caught);
      setError(parsed.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(null);
  }, [load]);

  const selected = useMemo(() => data?.arms?.[arm], [data, arm]);

  if (loading && !data) {
    return <p className="text-[12px] text-(--ink-soft)">Loading recorded P100 GPU predictions...</p>;
  }
  if (error || !data) {
    return (
      <div className="rounded-lg border border-(--warn)/40 bg-(--warn-soft)/40 p-2.5 text-[11.5px] leading-snug text-(--warn)">
        {error || "The ranker replay is unavailable."} No live ShopRanker/Hybrid fallback is fabricated.
        <button
          type="button"
          className="ml-1 underline underline-offset-2"
          onClick={() => void load(taskId)}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3" data-testid="ranker-replay-panel">
      <div className="rounded-xl border border-amber-500/50 bg-amber-50/80 p-2.5 text-[11px] leading-snug text-amber-950">
        <div className="flex items-center gap-1.5 font-bold uppercase tracking-wide">
          <Icon name="info" size={12} />
          {data.badge}
        </div>
        <p className="mt-1">
          Recorded P100 GPU scores over the fixed Original BGE Top-20. This is <b>not</b> a live model
          call and does not rerank anything. Experimental arms are Shadow only.
        </p>
      </div>

      <label className="block text-[10.5px] font-semibold text-(--ink-soft)">
        Recorded P100 task
        <select
          value={taskId || ""}
          onChange={(event) => void load(event.target.value)}
          className="mt-1 w-full rounded-lg border border-(--line) bg-(--card) px-2 py-1.5 text-[11.5px] text-(--ink)"
          data-testid="ranker-replay-task"
        >
          {data.tasks.map((task) => (
            <option key={task.task_id} value={task.task_id}>
              {task.task_id} · {task.query}
            </option>
          ))}
        </select>
      </label>

      <div className="rounded-lg border border-(--line) bg-(--well)/50 px-2.5 py-2 text-[11px] text-(--ink-2)">
        <p className="font-semibold text-(--ink)">{data.query}</p>
        <p className="mt-0.5 text-[10px] text-(--ink-soft)">
          {humanize(data.scenario_type || "no_memory")} · snapshot{" "}
          <span className="font-mono">{data.memory_snapshot_id || "UNKNOWN"}</span>
        </p>
        {data.active_preferences_on?.length ? (
          <div className="mt-1 flex flex-wrap gap-1">
            {data.active_preferences_on.map((pref) => (
              <Pill key={`${pref.memory_id}-${pref.value}`} tone="accent">
                {pref.polarity === "avoid" ? "avoid " : ""}
                {pref.value}
              </Pill>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-[10px] text-(--ink-faint)">No active memory in the recorded ON state.</p>
        )}
      </div>

      <div className="flex flex-wrap gap-1">
        {ARMS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => setArm(entry.id)}
            className={`rounded-lg border px-2 py-1 text-left text-[10.5px] ${
              arm === entry.id
                ? "border-(--accent) bg-(--accent-soft) text-(--accent-ink)"
                : "border-(--line) bg-(--card) text-(--ink-soft)"
            }`}
            data-testid={`ranker-arm-${entry.id}`}
          >
            <span className="block font-bold">{entry.label}</span>
            <span className="block text-[9px]">{entry.note}</span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => setMode("off")}
          className={`rounded-md border px-2 py-1 text-[10.5px] ${
            mode === "off" ? "border-(--ink) bg-(--ink) text-(--surface)" : "border-(--line) text-(--ink-soft)"
          }`}
          data-testid="ranker-mode-off"
        >
          Memory OFF
        </button>
        <button
          type="button"
          onClick={() => setMode("on")}
          className={`rounded-md border px-2 py-1 text-[10.5px] ${
            mode === "on" ? "border-(--ink) bg-(--ink) text-(--surface)" : "border-(--line) text-(--ink-soft)"
          }`}
          data-testid="ranker-mode-on"
        >
          Memory ON
        </button>
        <span className="ml-auto text-[9.5px] text-(--ink-faint)">
          {selected?.top1_changed ? "Top-1 changed OFF→ON" : "No Top-1 change under this memory state"}
        </span>
      </div>

      <RankList items={selected?.[mode] || []} mode={mode} />

      {mode === "on" ? (
        <details className="rounded-lg border border-(--line) bg-(--card) p-2 text-[10.5px] text-(--ink-soft)">
          <summary className="cursor-pointer font-semibold text-(--ink)">Show recorded Memory OFF top-5</summary>
          <div className="mt-2">
            <RankList items={selected?.off || []} mode="off" />
          </div>
        </details>
      ) : null}

      <p className="text-[9.5px] leading-snug text-(--ink-faint)">{data.notes?.join(" ")}</p>
    </div>
  );
}
