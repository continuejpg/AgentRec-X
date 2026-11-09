// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import type { DemoConfig, Health, ModeInfo } from "@/lib/types";
import { formatElapsed, plural } from "@/lib/format";
import { Icon } from "./icons";
import { Button, Pill } from "./ui";

function serverLabel(profileId: string, config: DemoConfig | null): string {
  const profile = config?.profiles.find((entry) => entry.profile_id === profileId);
  return profile ? profile.label : profileId;
}

function StepMark({ status }: { status: "done" | "running" | "error" | "pending" }) {
  if (status === "done") return <Icon name="check" size={13} className="text-(--ok)" />;
  if (status === "error") return <Icon name="alert" size={13} className="text-(--danger)" />;
  if (status === "running") {
    return (
      <span
        aria-hidden
        className="inline-block h-3 w-3 animate-spin rounded-full border-[1.5px] border-(--accent) border-r-transparent"
      />
    );
  }
  return <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full border border-(--line-strong)" />;
}

export default function ReadinessBanner({
  health,
  healthError,
  config,
  bffBase,
  elapsedSeconds,
  onRetry,
}: {
  health: Health | null;
  healthError: string | null;
  config: DemoConfig | null;
  bffBase: string;
  elapsedSeconds: number;
  onRetry: () => void;
}) {
  const servers = health?.servers ?? [];
  const readyServers = servers.filter((server) => server.status === "ready").length;
  const totalServers = servers.length || 2;
  const allReady = health?.ready === true;
  const failed = health?.status === "error" || servers.some((server) => server.status === "error");
  const degraded = health?.status === "degraded" || servers.some((server) => server.status === "degraded");

  if (allReady && !degraded && !failed) return null;

  if (!health) {
    return (
      <section
        className="rounded-2xl border border-(--danger)/35 bg-(--danger-soft)/40 p-3.5 ac-reveal"
        data-testid="readiness-banner"
        data-state="offline"
        aria-live="polite"
      >
        <div className="flex flex-wrap items-center gap-2">
          <Icon name="alert" size={15} className="text-(--danger)" />
          <h2 className="text-[13.5px] font-semibold text-(--ink)">Local BFF is not reachable yet</h2>
          <Pill tone="danger" dot className="ml-auto">
            offline
          </Pill>
        </div>
        <p className="mt-1.5 text-[12.5px] leading-relaxed text-(--ink-2)">
          The page is rendered and will connect automatically. It is trying{" "}
          <span className="font-mono text-[11.5px]">{bffBase}</span> — start the local BFF (FastAPI, WSL
          <span className="font-mono"> .venv/bin/python -m web_demo</span>) and this banner clears once its health
          endpoint answers. Cold start of the two MCP servers takes about 35–45 s after that.
        </p>
        {healthError ? (
          <p className="mt-1.5 break-words rounded-lg bg-(--card)/70 px-2 py-1 font-mono text-[10.5px] text-(--danger)">
            {healthError}
          </p>
        ) : null}
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="secondary" icon="refresh" onClick={onRetry}>
            Retry now
          </Button>
          <span className="font-mono text-[10.5px] text-(--ink-faint)">waiting {formatElapsed(elapsedSeconds)}</span>
        </div>
      </section>
    );
  }

  const stepStatus = (server: (typeof servers)[number]): "done" | "running" | "error" | "pending" => {
    if (server.status === "ready") return "done";
    if (server.status === "error") return "error";
    if (server.status === "starting") return "running";
    return "pending";
  };

  return (
    <section
      className="rounded-2xl border border-(--line) bg-(--card) p-3.5 shadow-(--shadow-sm) ac-reveal"
      data-testid="readiness-banner"
      data-state={failed ? "error" : degraded ? "degraded" : "starting"}
      aria-live="polite"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-(--accent-soft) text-(--accent-ink)">
          <Icon name="refresh" size={14} />
        </span>
        <h2 className="text-[13.5px] font-semibold text-(--ink)">
          {failed
            ? "Local demo services report an error"
            : degraded
              ? "Local demo services are degraded"
              : "Initializing local demo services"}
        </h2>
        <Pill tone={failed ? "danger" : degraded ? "warn" : "accent"} dot className="ml-auto">
          {health.status}
        </Pill>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-(--ink-2)">
        The BFF is running. Two long-lived MCP servers load the 76,263-item Electronics catalogue and the BGE index;
        cold start is ~35–45 s. This page stays interactive and the composer unlocks when both profiles are ready.
      </p>

      <div className="mt-2.5 space-y-1.5">
        <div className="flex items-center gap-2 text-[12px]">
          <StepMark status="done" />
          <span className="font-medium text-(--ink)">BFF health endpoint</span>
          <span className="ml-auto font-mono text-[10.5px] text-(--ink-soft)">
            uptime {health.uptime_s != null ? `${Math.round(health.uptime_s)} s` : "—"}
          </span>
        </div>
        {servers.map((server) => (
          <div key={server.profile_id} className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px]">
            <StepMark status={stepStatus(server)} />
            <span className="font-medium text-(--ink)">{serverLabel(server.profile_id, config)}</span>
            <span className="font-mono text-[10.5px] text-(--ink-soft)">
              {server.profile_id} · {server.search_mode}
            </span>
            <span className="ml-auto font-mono text-[10.5px] text-(--ink-soft)">
              {server.status === "ready"
                ? `${server.tools?.length ?? 0} tools${server.startup_ms != null ? ` · ${(server.startup_ms / 1000).toFixed(1)} s` : ""}`
                : server.status}
            </span>
            {server.error ? (
              <div className="w-full break-words rounded bg-(--danger-soft) px-2 py-1 font-mono text-[10.5px] text-(--danger)">
                {server.error}
              </div>
            ) : null}
          </div>
        ))}
        <div className="flex items-center gap-2 text-[12px]">
          <StepMark status={health.catalogue_size != null ? "done" : readyServers === totalServers ? "done" : "pending"} />
          <span className="font-medium text-(--ink)">Catalogue loaded</span>
          <span className="ml-auto font-mono text-[10.5px] text-(--ink-soft)">
            {health.catalogue_size != null ? plural(health.catalogue_size, "item") : "pending"}
          </span>
        </div>
      </div>

      <div className="mt-2.5 h-1 w-full overflow-hidden rounded-full bg-(--well)" aria-hidden>
        <div
          className="h-full rounded-full bg-(--accent) transition-[width] duration-500"
          style={{ width: `${Math.round((readyServers / Math.max(1, totalServers)) * 100)}%` }}
        />
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10.5px] text-(--ink-soft)">
        <span>
          {readyServers}/{totalServers} MCP servers ready
        </span>
        <span aria-hidden>·</span>
        <span className="font-mono">elapsed {formatElapsed(elapsedSeconds)}</span>
        {health.memory_writes_enabled != null ? (
          <>
            <span aria-hidden>·</span>
            <span>memory writes {health.memory_writes_enabled ? "enabled" : "disabled"}</span>
          </>
        ) : null}
        <Button size="sm" variant="ghost" icon="refresh" className="ml-auto" onClick={onRetry}>
          Refresh status
        </Button>
      </div>
      {failed || degraded ? (
        <p className="mt-1.5 rounded-lg bg-(--warn-soft)/60 px-2 py-1.5 text-[11.5px] leading-snug text-(--warn)">
          {failed
            ? "A server reported an error. Check the BFF logs in runs/web_demo/logs/; the demo will not fabricate results while a server is down."
            : "Some functionality may be limited until every profile reports ready."}
        </p>
      ) : null}
    </section>
  );
}

/** Re-exported so pages can build mode notices without importing types everywhere. */
export function modeNotice(mode: ModeInfo | undefined): string | null {
  if (!mode) return null;
  if (!mode.available) return mode.disabled_reason ?? `${mode.label} is unavailable.`;
  return mode.detail ?? mode.badge ?? null;
}
