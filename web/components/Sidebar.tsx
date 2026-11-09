// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useState } from "react";
import type {
  DemoConfig,
  ErrorPayload,
  PreferenceState,
  SessionSummary,
  SessionView,
} from "@/lib/types";
import { BFF_BASE } from "@/lib/api";
import { formatRelativeTime, plural, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Wordmark } from "./Header";
import PreferencesList from "./PreferencesList";
import { Button, IconButton, Pill, Skeleton } from "./ui";

export default function Sidebar({
  config,
  session,
  recent,
  recentLoading,
  preferences,
  prefsLoading,
  prefsError,
  onRefreshPreferences,
  onNewConversation,
  onSelectSession,
  onSelectProfile,
  onResetSession,
  busy,
  variant,
  onClose,
}: {
  config: DemoConfig | null;
  session: SessionView | null;
  recent: SessionSummary[];
  recentLoading: boolean;
  preferences: PreferenceState | null;
  prefsLoading: boolean;
  prefsError: ErrorPayload | null;
  onRefreshPreferences: () => void;
  onNewConversation: () => void;
  onSelectSession: (sessionId: string) => void;
  onSelectProfile: (profileId: string) => void;
  onResetSession: () => void;
  busy: boolean;
  variant: "column" | "drawer";
  onClose?: () => void;
}) {
  const [confirmReset, setConfirmReset] = useState(false);
  const activeProfileId = session?.profile_id ?? config?.default_profile_id ?? null;
  const profiles = config?.profiles ?? [];
  const notices = config?.notices ?? [];
  const profileLabel = (profileId: string) =>
    profiles.find((profile) => profile.profile_id === profileId)?.label ?? profileId;

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="sidebar">
      <div className="flex items-center gap-2 border-b border-(--line) px-3 py-3">
        <Wordmark />
        {variant === "drawer" ? (
          <IconButton icon="x" label="Close sidebar" onClick={() => onClose?.()} size={28} className="ml-auto" />
        ) : null}
      </div>

      <div className="panel-scroll min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-3">
        <Button
          variant="primary"
          icon="spark"
          data-testid="new-conversation"
          className="w-full"
          disabled={busy}
          onClick={() => {
            onNewConversation();
            onClose?.();
          }}
        >
          New conversation
        </Button>

        <section aria-labelledby="recent-heading">
          <div className="mb-1.5 flex items-center gap-2">
            <h2 id="recent-heading" className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">
              Recent conversations
            </h2>
            {recent.length ? <span className="text-[10.5px] text-(--ink-faint)">{recent.length}</span> : null}
          </div>
          {recentLoading && !recent.length ? (
            <div className="space-y-1.5">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : recent.length ? (
            <ul className="space-y-1" data-testid="recent-sessions">
              {recent.map((entry) => {
                const active = entry.session_id === session?.session_id;
                return (
                  <li key={entry.session_id}>
                    <button
                      type="button"
                      disabled={busy && !active}
                      onClick={() => {
                        onSelectSession(entry.session_id);
                        onClose?.();
                      }}
                      aria-current={active ? "true" : undefined}
                      data-testid="recent-session"
                      data-session-id={entry.session_id}
                      className={`w-full rounded-lg border px-2 py-1.5 text-left transition-colors ${
                        active
                          ? "border-(--ink)/30 bg-(--ink) text-(--surface)"
                          : "border-transparent text-(--ink) hover:border-(--line) hover:bg-(--ground)"
                      } disabled:cursor-not-allowed disabled:opacity-50`}
                    >
                      <div className="truncate text-[12px] font-semibold">{entry.title || "Untitled session"}</div>
                      <div className={`mt-0.5 flex items-center gap-1.5 text-[10px] ${active ? "text-white/70" : "text-(--ink-soft)"}`}>
                        <span className="truncate">{profileLabel(entry.profile_id)}</span>
                        <span aria-hidden>·</span>
                        <span>{plural(entry.turn_count, "turn")}</span>
                        <span aria-hidden>·</span>
                        <span title={entry.updated_at}>{formatRelativeTime(entry.updated_at)}</span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="rounded-lg border border-dashed border-(--line-strong) px-2 py-2 text-[11.5px] leading-snug text-(--ink-soft)">
              No conversations returned by the BFF yet. New conversations are created on this machine and reset when
              the BFF restarts.
            </p>
          )}
        </section>

        <section aria-labelledby="profile-heading">
          <h2 id="profile-heading" className="mb-1.5 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">
            Demo profile
          </h2>
          {profiles.length ? (
            <div className="space-y-1" data-testid="profile-selector">
              {profiles.map((profile) => {
                const active = profile.profile_id === activeProfileId;
                return (
                  <button
                    key={profile.profile_id}
                    type="button"
                    disabled={busy}
                    aria-pressed={active}
                    onClick={() => {
                      onSelectProfile(profile.profile_id);
                      onClose?.();
                    }}
                    className={`w-full rounded-lg border px-2 py-1.5 text-left transition-colors ${
                      active
                        ? "border-(--accent) bg-(--accent-soft)/60"
                        : "border-(--line) bg-(--card) hover:border-(--line-strong)"
                    } disabled:cursor-not-allowed disabled:opacity-50`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className={`text-[12px] font-semibold ${active ? "text-(--accent-ink)" : "text-(--ink)"}`}>
                        {profile.label}
                      </span>
                      {active ? <Icon name="check" size={11} className="text-(--accent-ink)" /> : null}
                    </div>
                    <p className="mt-0.5 line-clamp-2 text-[10.5px] leading-snug text-(--ink-soft)">{profile.description}</p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      <Pill tone={profile.memory_writes ? "warn" : "muted"}>
                        memory {profile.memory_writes ? "writes on" : "read-only"}
                      </Pill>
                      <Pill tone="muted">{profile.search_mode}</Pill>
                      <span className="font-mono text-[9.5px] text-(--ink-faint)">{profile.user_key}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <p className="text-[11.5px] text-(--ink-soft)">Profiles load from GET /api/demo/config…</p>
          )}
          {session ? (
            <p className="mt-1 font-mono text-[9.5px] text-(--ink-faint)">
              session {session.session_id} · {statusLabel(session.profile_id)}
            </p>
          ) : null}
        </section>

        <PreferencesList
          state={preferences}
          loading={prefsLoading}
          error={prefsError}
          onRefresh={onRefreshPreferences}
          className="border-t border-(--line) pt-3"
        />

        <section className="border-t border-(--line) pt-3">
          <h2 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Session</h2>
          <Button
            variant={confirmReset ? "accent" : "secondary"}
            size="sm"
            icon="refresh"
            data-testid="reset-session"
            disabled={!session || busy}
            className="w-full"
            onClick={() => {
              if (!confirmReset) {
                setConfirmReset(true);
                window.setTimeout(() => setConfirmReset(false), 4000);
                return;
              }
              setConfirmReset(false);
              onResetSession();
              onClose?.();
            }}
          >
            {confirmReset ? "Confirm: clear this session" : "Clear session"}
          </Button>
          <p className="mt-1 text-[10px] leading-snug text-(--ink-faint)">
            Clears this session&apos;s turns and pending clarification. M9 long-term memory is not cleared; restarting
            the BFF resets it to its fixture.
          </p>
        </section>

        {notices.length ? (
          <section className="rounded-lg bg-(--well)/60 p-2">
            <ul className="list-disc space-y-0.5 pl-3 text-[10px] leading-snug text-(--ink-soft)">
              {notices.map((notice) => (
                <li key={notice}>{notice}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>

      <div className="border-t border-(--line) px-3 py-2">
        <div className="flex items-center gap-1.5 text-[10px] text-(--ink-faint)">
          <Icon name="info" size={10} />
          <span className="truncate" title={BFF_BASE}>
            BFF {BFF_BASE}
          </span>
        </div>
        <p className="mt-0.5 text-[9.5px] leading-snug text-(--ink-faint)">
          Local demo only · catalogue facts · no cart, checkout or payment.
        </p>
      </div>
    </div>
  );
}
