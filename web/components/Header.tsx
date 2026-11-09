// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import type { DemoConfig, Health, ModeKey } from "@/lib/types";
import { formatElapsed } from "@/lib/format";
import { Icon } from "./icons";
import { IconButton, Pill, StatusDot, type Tone } from "./ui";

export type UiMode = "cpu_live" | "qwen_replay";

function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span
      aria-hidden
      className={`grid shrink-0 place-items-center rounded-lg bg-(--ink) text-(--surface) ${
        small ? "h-7 w-7" : "h-8 w-8"
      }`}
    >
      <Icon name="spark" size={small ? 15 : 17} className="text-(--accent)" />
    </span>
  );
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <BrandMark small={compact} />
      <span className="min-w-0 leading-tight">
        <span className={`block font-bold tracking-[-0.02em] text-(--ink) ${compact ? "text-[14px]" : "text-[15.5px]"}`}>
          AgentRec-X
        </span>
        {compact ? null : (
          <span className="block text-[10.5px] font-medium uppercase tracking-[0.12em] text-(--ink-soft)">
            Shopping Assistant
          </span>
        )}
      </span>
    </span>
  );
}

function readiness(health: Health | null, healthError: string | null): { tone: Tone; label: string; title: string } {
  if (!health) {
    return {
      tone: "danger",
      label: "BFF offline",
      title: healthError ?? "The local BFF is not reachable; this page retries automatically.",
    };
  }
  if (health.ready) {
    return {
      tone: "ok",
      label: "Ready",
      title: `Both MCP profiles ready · catalogue ${health.catalogue_size ?? "?"} · uptime ${Math.round(health.uptime_s ?? 0)} s`,
    };
  }
  if (health.status === "error") {
    return { tone: "danger", label: "Service error", title: health.servers.find((server) => server.error)?.error ?? "A local MCP server reported an error." };
  }
  if (health.status === "degraded") {
    return { tone: "warn", label: "Degraded", title: "At least one local service is degraded." };
  }
  return {
    tone: "warn",
    label: `Starting ${formatElapsed(health.uptime_s ?? 0)}`,
    title: "MCP cold start loads the catalogue and BGE index; typically 35–45 s.",
  };
}

function ModeControl({
  modes,
  mode,
  onChange,
  disabled,
}: {
  modes: Partial<Record<ModeKey, ModeInfoUnavailable>>;
  mode: UiMode;
  onChange: (mode: UiMode) => void;
  disabled: boolean;
}) {
  const replay = modes.qwen_replay;
  const live = modes.qwen_live;
  const cpu = modes.cpu_live;
  const items: { key: UiMode | "qwen_live"; label: string; title: string; available: boolean; active: boolean }[] = [
    {
      key: "cpu_live",
      label: "CPU Live",
      title: cpu?.detail ?? "Live local CPU MCP pipeline (default).",
      // CPU Live is the default live mode; readiness is reported separately by the readiness chip.
      available: true,
      active: mode === "cpu_live",
    },
    {
      key: "qwen_replay",
      label: "Qwen Replay",
      title: replay?.badge ?? "Historical Replay — Not a Live Model Call",
      available: replay?.available !== false,
      active: mode === "qwen_replay",
    },
    {
      key: "qwen_live",
      label: "Qwen Live",
      title: live?.disabled_reason ?? "No live Qwen service detected (P18).",
      available: false,
      active: false,
    },
  ];
  return (
    <div role="group" aria-label="Assistant mode" className="flex items-center gap-0.5 rounded-lg bg-(--ground) p-[3px]">
      {items.map((item) => {
        const isDisabled = disabled || !item.available;
        return (
          <button
            key={item.key}
            type="button"
            aria-pressed={item.active}
            disabled={isDisabled}
            title={item.title}
            onClick={() => {
              if (item.key === "qwen_live") return;
              onChange(item.key);
            }}
            className={`flex items-center gap-1 whitespace-nowrap rounded-md px-2 py-1 text-[11.5px] transition-colors ${
              item.active
                ? "bg-(--card) font-semibold text-(--ink) shadow-(--shadow-sm)"
                : item.available
                  ? "font-medium text-(--ink-soft) hover:text-(--ink)"
                  : "cursor-not-allowed font-medium text-(--ink-faint) line-through decoration-1"
            }`}
          >
            {item.key === "qwen_replay" ? <Icon name="play" size={10} /> : null}
            {item.key === "qwen_live" ? <Icon name="alert" size={10} className="text-(--ink-faint)" /> : null}
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

type ModeInfoUnavailable = {
  available: boolean;
  label?: string;
  detail?: string | null;
  badge?: string | null;
  disabled_reason?: string | null;
};

export default function Header({
  health,
  healthError,
  config,
  mode,
  onModeChange,
  onOpenSidebar,
  onOpenPanel,
  onTogglePanel,
  panelOpen,
  onOpenReplay,
  onRetryHealth,
  sessionTitle,
  profileLabel,
  streaming,
}: {
  health: Health | null;
  healthError: string | null;
  config: DemoConfig | null;
  mode: UiMode;
  onModeChange: (mode: UiMode) => void;
  onOpenSidebar: () => void;
  onOpenPanel: () => void;
  onTogglePanel: () => void;
  panelOpen: boolean;
  onOpenReplay: () => void;
  onRetryHealth: () => void;
  sessionTitle: string | null;
  profileLabel: string | null;
  streaming: boolean;
}) {
  const status = readiness(health, healthError);
  const modes = (config?.modes ?? health?.modes ?? {}) as Partial<Record<ModeKey, ModeInfoUnavailable>>;
  return (
    <header className="z-30 flex h-14 shrink-0 items-center gap-2 border-b border-(--line) bg-(--chrome)/95 px-2.5 backdrop-blur sm:px-3.5">
      <IconButton icon="menu" label="Open conversations and preferences" onClick={onOpenSidebar} className="xl:hidden" />
      <span className="hidden xl:block">
        <Wordmark compact />
      </span>
      <span className="min-w-0 xl:hidden">
        <Wordmark compact />
      </span>
      <span className="mx-1 hidden h-5 w-px bg-(--line) xl:block" aria-hidden />
      <div className="hidden min-w-0 flex-1 xl:block">
        <div className="truncate text-[12.5px] font-semibold text-(--ink)" title={sessionTitle ?? undefined}>
          {sessionTitle ?? "New shopping session"}
        </div>
        <div className="flex items-center gap-1.5 text-[10.5px] text-(--ink-soft)">
          {profileLabel ? (
            <>
              <Icon name="user" size={10} />
              <span className="truncate">{profileLabel}</span>
            </>
          ) : (
            <span>Loading demo profile…</span>
          )}
          {streaming ? <span className="text-(--accent)">· streaming</span> : null}
        </div>
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        {profileLabel ? (
          <button
            type="button"
            onClick={onOpenSidebar}
            title={`Current demo profile: ${profileLabel}`}
            className="flex max-w-[150px] items-center gap-1 rounded-lg border border-(--line) bg-(--card) px-1.5 py-1 text-[10.5px] font-semibold text-(--ink-2) transition-colors hover:border-(--accent) xl:hidden"
          >
            <Icon name="user" size={10} className="shrink-0" />
            <span className="truncate">{profileLabel}</span>
          </button>
        ) : null}
        <div className="hidden md:block">
          <ModeControl modes={modes} mode={mode} onChange={onModeChange} disabled={streaming} />
        </div>
        <button
          type="button"
          onClick={onRetryHealth}
          data-testid="readiness-chip"
          data-state={health?.ready ? "ready" : health ? health.status : "offline"}
          title={status.title}
          className="flex items-center gap-1.5 rounded-lg border border-(--line) bg-(--card) px-2 py-1 text-[11px] font-semibold text-(--ink-2) transition-colors hover:border-(--accent)"
        >
          <StatusDot tone={status.tone} label={status.label} />
          <span className="hidden sm:inline">{status.label}</span>
        </button>
        <IconButton icon="play" label="Open Qwen Replay (historical trajectories)" onClick={onOpenReplay} />
        <button
          type="button"
          onClick={onOpenPanel}
          aria-label="Open details, evidence and comparison panel"
          title="Details / Evidence / Comparison"
          className="grid h-[30px] w-[30px] place-items-center rounded-lg text-(--ink-soft) transition-colors hover:bg-(--ground) hover:text-(--ink) xl:hidden"
        >
          <Icon name="panel-right" size={16} />
        </button>
        <IconButton
          icon={panelOpen ? "panel-right" : "compare"}
          label={panelOpen ? "Hide context panel" : "Show context panel"}
          onClick={onTogglePanel}
          className="hidden xl:grid"
        />
        {mode === "qwen_replay" ? (
          <Pill tone="warn" className="hidden sm:inline-flex">
            Historical Replay — not live
          </Pill>
        ) : null}
      </div>
    </header>
  );
}
