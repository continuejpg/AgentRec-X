// AgentRec-X Web Demo (2026-11) - Behavior Inspector for Behavior Feedback V1.
//
// This panel is the only place that shows the first-party behavior data.  It never writes
// M9 automatically: a behavior-derived suggestion is shown as an inference and the existing
// M9 writer is called only after the user explicitly confirms it.
//
// The Behavior OFF/ON lists are a *shadow preview* over the last issued recommendation.
// The production product list and the Decision Gate are never re-ordered by this panel.

"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, errorPayload } from "@/lib/api";
import { behaviorEventLabel, behaviorTracker } from "@/lib/behavior";
import { formatRelativeTime, humanize, plural } from "@/lib/format";
import type {
  BehaviorProfile,
  BehaviorShadow,
  BehaviorSuggestion,
  ErrorPayload,
  MemoryRememberResponse,
} from "@/lib/types";
import { Icon } from "./icons";
import { Button, EmptyState, Pill, Skeleton } from "./ui";

function ToggleButton({
  enabled,
  busy,
  onClick,
}: {
  enabled: boolean;
  busy: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label="Behavior personalization"
      data-testid="behavior-toggle"
      disabled={busy}
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[11.5px] font-semibold transition-colors disabled:opacity-45 ${
        enabled
          ? "border-(--ok)/40 bg-(--ok-soft) text-(--ok)"
          : "border-(--line-strong) bg-(--card) text-(--ink-soft)"
      }`}
    >
      <span
        aria-hidden
        className={`relative h-3.5 w-6 rounded-full transition-colors ${enabled ? "bg-(--ok)" : "bg-(--line-strong)"}`}
      >
        <span
          className={`absolute top-0.5 h-2.5 w-2.5 rounded-full bg-white transition-[left] ${enabled ? "left-3" : "left-0.5"}`}
        />
      </span>
      {enabled ? "Personalization ON" : "Personalization OFF"}
    </button>
  );
}

function Section({
  title,
  aside,
  children,
  open = false,
}: {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
  open?: boolean;
}) {
  return (
    <details open={open} className="group rounded-xl border border-(--line) bg-(--card)/70">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-2 text-[12px] font-semibold text-(--ink) [&::-webkit-details-marker]:hidden">
        <Icon name="chevron-down" size={12} className="text-(--ink-soft) group-open:rotate-0 -rotate-90 transition-transform" />
        <span>{title}</span>
        {aside ? <span className="ml-auto text-[10.5px] font-normal text-(--ink-soft)">{aside}</span> : null}
      </summary>
      <div className="border-t border-(--line) px-2.5 py-2">{children}</div>
    </details>
  );
}

function Counts({ profile }: { profile: BehaviorProfile }) {
  const counts = profile.counts;
  const items = [
    { label: "Impressions", value: counts.product_impression },
    { label: "Clicks", value: counts.product_click },
    { label: "Compares", value: counts.product_compare },
    { label: "Like / dislike", value: counts.explicit_feedback },
  ];
  return (
    <div className="grid grid-cols-4 gap-1" data-testid="behavior-counts">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-(--line) bg-(--ground)/60 px-1.5 py-1 text-center">
          <div className="text-[15px] font-bold tabular-nums text-(--ink)">{item.value}</div>
          <div className="text-[9.5px] leading-tight text-(--ink-soft)">{item.label}</div>
        </div>
      ))}
      {counts.product_compare ? (
        <p className="col-span-4 text-[10px] leading-snug text-(--ink-soft)">
          Compare lifecycle: {counts.product_compare_attempted} attempted ·{" "}
          {counts.product_compare_completed} completed · {counts.product_compare_failed} failed
        </p>
      ) : null}
      {counts.click_after_impression ? (
        <p className="col-span-4 text-[10px] leading-snug text-(--ink-faint)">
          {counts.click_after_impression} click(s) had a matching visible impression; impressions stay
          exposure context only.
        </p>
      ) : null}
      {counts.recorded_replay || counts.synthetic_test ? (
        <p className="col-span-4 text-[10px] leading-snug text-(--ink-faint)">
          Excluded from profile: {counts.recorded_replay} recorded replay · {counts.synthetic_test} synthetic test.
        </p>
      ) : null}
    </div>
  );
}

function InterestList({ profile }: { profile: BehaviorProfile }) {
  if (!profile.attribute_interests.length) {
    return (
      <p className="text-[11.5px] leading-snug text-(--ink-soft)">
        No verified attribute interest yet. An impression is exposure context and contributes zero
        interest; a weak interest requires a click, and an explicit item feedback requires Like/Dislike.
        Comparisons are operation provenance, not preferences.
      </p>
    );
  }
  return (
    <ul className="space-y-1" data-testid="behavior-interests">
      {profile.attribute_interests.map((interest) => (
        <li key={interest.attribute} className="rounded-lg border border-(--line) bg-(--ground)/50 px-2 py-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-[12px] font-semibold text-(--ink)">{humanize(interest.label)}</span>
            <Pill tone="info">score {interest.score.toFixed(2)}</Pill>
            <span className="ml-auto text-[10px] text-(--ink-faint)">
              {plural(interest.distinct_products, "product")} · {plural(interest.evidence_count, "event")}
            </span>
          </div>
          {interest.evidence.length ? (
            <div className="mt-0.5 truncate font-mono text-[9.5px] text-(--ink-faint)" title={interest.evidence.map((entry) => `${entry.event_id} → ${entry.product_id}`).join("\n")}>
              evidence: {interest.evidence.slice(0, 3).map((entry) => entry.product_id).join(", ")}
            </div>
          ) : null}
          <div className="mt-0.5 text-[9.5px] leading-snug text-(--ink-faint)">
            source: {(interest.source_event_types ?? []).map(humanize).join(", ") || "—"} · decay
            half-life {Math.round(profile.half_life_seconds / 60)} min
          </div>
        </li>
      ))}
    </ul>
  );
}

function EventList({ profile }: { profile: BehaviorProfile }) {
  if (!profile.recent_events.length) {
    return <p className="text-[11.5px] text-(--ink-soft)">No browser events recorded for this session yet.</p>;
  }
  return (
    <ul className="space-y-1" data-testid="behavior-events">
      {profile.recent_events.slice(0, 10).map((event) => (
        <li key={event.event_id} className="rounded-lg border border-(--line) bg-(--ground)/40 px-2 py-1">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-semibold text-(--ink-2)">
              {behaviorEventLabel(event.event_type, event.metadata.action_status)}
            </span>
            <span className="truncate font-mono text-[10px] text-(--ink-soft)" title={event.product_id}>
              {event.product_id}
            </span>
            <span className="ml-auto text-[9.5px] text-(--ink-faint)">pos {event.displayed_position}</span>
          </div>
          <div className="flex flex-wrap items-center gap-1 text-[9.5px] text-(--ink-faint)">
            <span className="truncate">rec {event.recommendation_id}</span>
            <span aria-hidden>·</span>
            <span>{formatRelativeTime(event.occurred_at)}</span>
            {event.event_source !== "LIVE_USER_INTERACTION" ? (
              <Pill tone="warn">{event.event_source}</Pill>
            ) : null}
            {event.metadata.feedback ? <Pill tone="muted">{event.metadata.feedback}</Pill> : null}
            {event.metadata.action_status ? (
              <Pill
                tone={
                  event.metadata.action_status === "succeeded"
                    ? "ok"
                    : event.metadata.action_status === "failed"
                      ? "danger"
                      : "warn"
                }
              >
                {event.metadata.action_status}
              </Pill>
            ) : null}
            {event.metadata.compared_product_ids?.length ? (
              <Pill tone="muted">{event.metadata.compared_product_ids.length} compared</Pill>
            ) : null}
            {event.metadata.failure_code ? (
              <span className="text-(--danger)" title={event.metadata.failure_message ?? undefined}>
                {event.metadata.failure_code}
              </span>
            ) : null}
            {event.event_type === "product_impression" ? (
              <span className="text-(--ink-faint)">exposure only · no interest</span>
            ) : null}
          </div>
        </li>
      ))}
    </ul>
  );
}

function RankingColumn({
  title,
  tone,
  rows,
  emptyHint,
}: {
  title: string;
  tone: "muted" | "accent";
  rows: BehaviorShadow["eligible_base_ranking"];
  emptyHint: string;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-(--line) bg-(--ground)/40 p-1.5">
      <div className="mb-1 flex items-center gap-1">
        <Pill tone={tone}>{title}</Pill>
        <span className="ml-auto text-[9.5px] text-(--ink-faint)">top {rows.length}</span>
      </div>
      {rows.length ? (
        <ol className="space-y-0.5">
          {rows.map((row) => (
            <li key={`${title}-${row.product_id}`} className="flex items-center gap-1 text-[10.5px]">
              <span className="w-3 shrink-0 text-right font-mono text-(--ink-faint)">{row.rank}</span>
              <span className="min-w-0 truncate text-(--ink-2)" title={row.product_title ?? row.product_id}>
                {row.product_title ?? row.product_id}
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-[10px] leading-snug text-(--ink-soft)">{emptyHint}</p>
      )}
    </div>
  );
}

function DiagnosticRanking({ shadow }: { shadow: BehaviorShadow }) {
  return (
    <div className="rounded-lg border border-dashed border-(--line-strong) bg-(--ground)/40 p-2" data-testid="behavior-diagnostic-ranking">
      <div className="flex flex-wrap items-center gap-1">
        <Pill tone="warn">Diagnostic Ranking — Not Final Recommendations</Pill>
        <span className="ml-auto text-[9.5px] text-(--ink-faint)">
          {shadow.diagnostic_ranking.length} shown · all displayed candidates
        </span>
      </div>
      <p className="mt-1 text-[9.5px] leading-snug text-(--ink-soft)">
        This is a developer diagnostic over every displayed candidate. Cards marked “ineligible” can
        show a hypothetical behavior contribution but can never enter the eligible Top-K.
      </p>
      <ol className="mt-1.5 space-y-1">
        {shadow.diagnostic_ranking.map((row) => (
          <li key={row.product_id} className="flex items-center gap-1.5 rounded-md border border-(--line) bg-(--card) px-1.5 py-1">
            <span className="w-4 shrink-0 text-right font-mono text-[10px] text-(--ink-faint)">{row.rank}</span>
            <span className="min-w-0 flex-1 truncate text-[10.5px] text-(--ink-2)" title={row.product_title ?? row.product_id}>
              {row.product_title ?? row.product_id}
            </span>
            <Pill tone={row.gate_eligible ? "ok" : "warn"}>
              {row.gate_eligible ? "eligible" : "ineligible"}
            </Pill>
            <span className="shrink-0 font-mono text-[9.5px] text-(--ink-faint)">
              #{row.diagnostic_base_rank}→#{row.rank}
            </span>
            <span className={`shrink-0 font-mono text-[9.5px] ${row.behavior_contribution ? "text-(--accent-ink)" : "text-(--ink-faint)"}`}>
              {row.behavior_contribution >= 0 ? "+" : ""}
              {row.behavior_contribution.toFixed(3)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ShadowBlock({
  shadow,
  loading,
  error,
  hasRecommendation,
}: {
  shadow: BehaviorShadow | null;
  loading: boolean;
  error: ErrorPayload | null;
  hasRecommendation: boolean;
}) {
  if (loading) {
    return (
      <div className="space-y-1">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }
  if (!hasRecommendation) {
    return (
      <p className="text-[11.5px] leading-snug text-(--ink-soft)">
        No issued recommendation yet. Send a shopping query; the next Behavior Shadow will use only
        behavior events recorded before that recommendation was issued.
      </p>
    );
  }
  if (error) {
    return (
      <div className="rounded-lg border border-(--danger)/30 bg-(--danger-soft)/40 px-2 py-1.5 text-[11px] text-(--danger)">
        {error.message}
      </div>
    );
  }
  if (!shadow) return <p className="text-[11.5px] text-(--ink-soft)">Shadow not loaded yet.</p>;
  const gate = shadow.gate_context;
  const hasEligible = shadow.eligible_candidate_count > 0;
  return (
    <div className="space-y-2" data-testid="behavior-shadow">
      <div className="flex flex-wrap items-center gap-1">
        <Pill tone={shadow.eligible_behavior_applied ? "accent" : "muted"} dot>
          {shadow.eligible_behavior_applied
            ? "Eligible behavior applied"
            : hasEligible
              ? "No eligible behavior adjustment"
              : "No eligible ranking"}
        </Pill>
        <Pill tone={shadow.eligible_ranking_changed ? "violet" : "muted"}>
          {shadow.eligible_ranking_changed ? "eligible order changed" : "eligible order same"}
        </Pill>
        {shadow.diagnostic_behavior_applied ? (
          <Pill tone="warn">diagnostic signal</Pill>
        ) : null}
        <span className="ml-auto font-mono text-[9.5px] text-(--ink-faint)" title={shadow.recommendation_id}>
          {shadow.recommendation_id}
        </span>
      </div>

      <div className="rounded-md border border-(--line) bg-(--ground)/50 px-1.5 py-1 text-[9.5px] leading-snug text-(--ink-soft)">
        <div>
          Gate: <span className="font-mono text-(--ink-2)">{gate.action ?? "—"}</span> · allow_finalize{" "}
          <span className="font-mono text-(--ink-2)">{String(gate.allow_finalize)}</span> · selected{" "}
          <span className="font-mono text-(--ink-2)">{gate.selected_candidate_id ?? "none"}</span>
        </div>
        <div>
          Eligible candidates: <span className="font-mono text-(--ink-2)">{gate.eligible_candidate_count}</span> ·
          source <span className="font-mono text-(--ink-2)">{gate.eligibility_source}</span>
        </div>
        <div className="mt-0.5 text-(--ink-faint)">{gate.interpretation}</div>
        {gate.no_eligible_reason ? (
          <div className="mt-0.5 text-(--warn)">No eligible reason: {humanize(gate.no_eligible_reason)}</div>
        ) : null}
      </div>

      {hasEligible ? (
        <>
          <div className="grid grid-cols-2 gap-1.5">
            <RankingColumn
              title="Eligible OFF"
              tone="muted"
              rows={shadow.eligible_base_ranking}
              emptyHint="No eligible candidates."
            />
            <RankingColumn
              title="Eligible ON"
              tone="accent"
              rows={shadow.eligible_behavior_ranking}
              emptyHint="No eligible candidates."
            />
          </div>
          {shadow.eligible_changes.length ? (
            <div className="space-y-1">
              {shadow.eligible_changes.slice(0, 5).map((change) => (
                <div key={change.product_id} className="rounded-lg border border-(--line) bg-(--card) px-2 py-1">
                  <div className="flex items-center gap-1.5 text-[11px]">
                    <span className="font-mono text-(--ink-faint)">
                      #{change.base_rank} → #{change.shadow_rank}
                    </span>
                    <span className="min-w-0 truncate font-semibold text-(--ink-2)" title={change.product_title ?? change.product_id}>
                      {change.product_title ?? change.product_id}
                    </span>
                    <span className="ml-auto shrink-0 font-mono text-[10px] text-(--accent-ink)">
                      {change.behavior_contribution >= 0 ? "+" : ""}
                      {change.behavior_contribution.toFixed(3)}
                    </span>
                  </div>
                  {change.matched_attribute_interests.length ? (
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      {change.matched_attribute_interests.map((match) => (
                        <Pill key={match.attribute} tone="info" title={`verified evidence: ${match.evidence.map((entry) => entry.evidence_id).join(", ")}`}>
                          {humanize(match.attribute)} · {match.score.toFixed(2)}
                        </Pill>
                      ))}
                    </div>
                  ) : null}
                  {change.suppressed_reason ? (
                    <div className="mt-0.5 text-[10px] text-(--warn)">
                      suppressed: {humanize(change.suppressed_reason)}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[11px] leading-snug text-(--ink-soft)">
              No eligible ranking change. {shadow.eligible_behavior_applied
                ? "Behavior signal changed scores but not the eligible order."
                : "The current behavior profile has no admissible signal for eligible candidates."}
            </p>
          )}
        </>
      ) : (
        <div className="rounded-lg border border-(--warn)/30 bg-(--warn-soft)/40 px-2 py-1.5 text-[11px] leading-snug text-(--warn)">
          No eligible recommendation ranking exists for this Gate state. The diagnostic list below is
          not a final recommendation list; the original Gate/ASK outcome remains authoritative.
        </div>
      )}

      <DiagnosticRanking shadow={shadow} />

      {shadow.ineligible_candidates_with_behavior_signal.length ? (
        <p className="text-[9.5px] leading-snug text-(--warn)">
          {shadow.ineligible_candidates_with_behavior_signal.length} ineligible candidate(s) show a
          diagnostic behavior contribution and are excluded from the eligible ranking by construction.
        </p>
      ) : null}
      <p className="text-[9.5px] leading-snug text-(--ink-faint)">
        Formal comparison is eligible-only: fixed query, candidates, metadata, positions, Active M9
        memory, eligibility and Gate. Diagnostic rows never become recommendations, and the normal
        product list above is not re-ranked.
      </p>
    </div>
  );
}

function SuggestionCard({
  suggestion,
  onAccept,
  onDismiss,
  busy,
  result,
  writesEnabled,
}: {
  suggestion: BehaviorSuggestion;
  onAccept: () => void;
  onDismiss: () => void;
  busy: boolean;
  result: { written: boolean; message: string } | null;
  writesEnabled: boolean;
}) {
  return (
    <div className="rounded-xl border border-(--accent)/40 bg-(--accent-soft)/40 p-2.5" data-testid="behavior-suggestion">
      <div className="flex items-center gap-1.5">
        <Icon name="spark" size={14} className="text-(--accent-ink)" />
        <span className="text-[12px] font-semibold text-(--accent-ink)">Behavior-inferred preference suggestion</span>
      </div>
      <p className="mt-1 text-[11.5px] leading-snug text-(--ink-2)">
        You interacted with {plural(suggestion.distinct_products, "product")} twice or more whose catalogue
        evidence contains <span className="font-semibold">{humanize(suggestion.label)}</span>. This is a weak
        short-term signal, not an explicit preference. Remember it as a long-term M9 preference?
      </p>
      <p className="mt-0.5 text-[10px] leading-snug text-(--ink-soft)">
        If confirmed, the existing M9 writer receives exactly: “{suggestion.message}”
      </p>
      {result ? (
        <p className={`mt-1 text-[11px] font-semibold ${result.written ? "text-(--ok)" : "text-(--danger)"}`}>
          {result.message}
        </p>
      ) : (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <Button
            size="sm"
            variant="primary"
            disabled={busy || !writesEnabled}
            title={writesEnabled ? undefined : "This demo profile is read-only; switch to Demo B to persist M9 preferences."}
            onClick={onAccept}
          >
            {busy ? "Writing…" : "Remember this preference"}
          </Button>
          <Button size="sm" variant="ghost" disabled={busy} onClick={onDismiss}>
            Not now
          </Button>
          <span className="ml-auto text-[9.5px] text-(--ink-faint)">
            {writesEnabled ? "no auto-write" : "read-only profile"}
          </span>
        </div>
      )}
    </div>
  );
}

export default function BehaviorInspector({
  sessionId,
  latestRecommendationId,
  memoryWritesEnabled,
  onBehaviorEnabledChange,
  onMemoryWritten,
}: {
  sessionId: string | null;
  latestRecommendationId: string | null;
  memoryWritesEnabled: boolean;
  onBehaviorEnabledChange: (enabled: boolean) => void;
  onMemoryWritten: () => void;
}) {
  const [profile, setProfile] = useState<BehaviorProfile | null>(null);
  const [shadow, setShadow] = useState<BehaviorShadow | null>(null);
  const [loading, setLoading] = useState(false);
  const [shadowLoading, setShadowLoading] = useState(false);
  const [error, setError] = useState<ErrorPayload | null>(null);
  const [shadowError, setShadowError] = useState<ErrorPayload | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteMessage, setDeleteMessage] = useState<string | null>(null);
  const [suggestionBusy, setSuggestionBusy] = useState(false);
  const [suggestionResult, setSuggestionResult] = useState<{ written: boolean; message: string } | null>(null);
  const [dismissedSuggestion, setDismissedSuggestion] = useState<string | null>(null);

  const refresh = useCallback(() => setRefreshKey((value) => value + 1), []);

  useEffect(() => {
    // Never keep the previous session's profile visible while the new one loads.
    setProfile(null);
    setShadow(null);
    setError(null);
    setShadowError(null);
    setSuggestionResult(null);
    setDismissedSuggestion(null);
    setDeleteMessage(null);
  }, [sessionId]);

  useEffect(() => {
    const unsubscribe = behaviorTracker.subscribe((info) => {
      if (info.accepted || info.rejected) refresh();
    });
    return unsubscribe;
  }, [refresh]);

  useEffect(() => {
    if (!sessionId) {
      setProfile(null);
      setShadow(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSuggestionResult(null);
    api
      .behaviorProfile(sessionId)
      .then((next) => {
        if (cancelled) return;
        setProfile(next);
        setEnabled(next.enabled);
        onBehaviorEnabledChange(next.enabled);
      })
      .catch((err) => {
        if (!cancelled) setError(errorPayload(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // onBehaviorEnabledChange is stable in DemoApp (useCallback with no deps).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, refreshKey]);

  useEffect(() => {
    if (!sessionId || !latestRecommendationId || !enabled) {
      setShadow(null);
      return;
    }
    let cancelled = false;
    setShadowLoading(true);
    setShadowError(null);
    api
      .behaviorShadow(sessionId, latestRecommendationId, 10)
      .then((next) => {
        if (!cancelled) setShadow(next);
      })
      .catch((err) => {
        if (!cancelled) setShadowError(errorPayload(err));
      })
      .finally(() => {
        if (!cancelled) setShadowLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, latestRecommendationId, enabled, refreshKey]);

  const toggle = useCallback(async () => {
    if (!sessionId || busy) return;
    setBusy(true);
    setDeleteMessage(null);
    try {
      const next = !enabled;
      const response = await api.setBehaviorEnabled(sessionId, next);
      setEnabled(response.enabled);
      onBehaviorEnabledChange(response.enabled);
      if (!next) {
        behaviorTracker.clearLocal();
        setShadow(null);
      }
      refresh();
    } catch (err) {
      setError(errorPayload(err));
    } finally {
      setBusy(false);
    }
  }, [busy, enabled, onBehaviorEnabledChange, refresh, sessionId]);

  const remove = useCallback(async () => {
    if (!sessionId || busy) return;
    setBusy(true);
    setDeleteMessage(null);
    try {
      const response = await api.deleteBehavior(sessionId);
      behaviorTracker.clearLocal();
      setConfirmDelete(false);
      setShadow(null);
      setDeleteMessage(response.message);
      refresh();
    } catch (err) {
      setError(errorPayload(err));
    } finally {
      setBusy(false);
    }
  }, [busy, refresh, sessionId]);

  const acceptSuggestion = useCallback(async () => {
    if (!sessionId || !profile?.suggestion || suggestionBusy) return;
    setSuggestionBusy(true);
    try {
      const response: MemoryRememberResponse = await api.rememberMemory(sessionId, profile.suggestion.message);
      const written = !response.error;
      setSuggestionResult({
        written,
        message: written
          ? `M9 update: +${response.update?.added ?? 0} active preference(s); this suggestion will disappear after refresh.`
          : response.error?.message ?? "The M9 write failed.",
      });
      if (written) {
        onMemoryWritten();
        setDismissedSuggestion(profile.suggestion.suggestion_id);
        refresh();
      }
    } catch (err) {
      const payload = errorPayload(err);
      setSuggestionResult({ written: false, message: payload.message });
    } finally {
      setSuggestionBusy(false);
    }
  }, [onMemoryWritten, profile?.suggestion, refresh, sessionId, suggestionBusy]);

  const visibleSuggestion = useMemo(() => {
    const suggestion = profile?.suggestion ?? null;
    if (!suggestion) return null;
    if (suggestion.suggestion_id === dismissedSuggestion) return null;
    return suggestion;
  }, [dismissedSuggestion, profile?.suggestion]);

  if (!sessionId) {
    return (
      <EmptyState
        icon="info"
        compact
        title="No active session"
        hint="Create a shopping session to collect real browser behavior events."
      />
    );
  }

  return (
    <div className="space-y-2.5" data-testid="behavior-inspector">
      <div className="flex items-center gap-1.5">
        <h3 className="text-[12.5px] font-semibold text-(--ink)">Behavior Inspector</h3>
        <Pill tone={enabled ? "ok" : "muted"} dot>
          {enabled ? "ON" : "OFF"}
        </Pill>
        <button
          type="button"
          onClick={refresh}
          className="ml-auto grid h-6 w-6 place-items-center rounded-md text-(--ink-soft) hover:bg-(--ground) hover:text-(--ink)"
          aria-label="Refresh behavior profile"
          title="Refresh behavior profile"
        >
          <Icon name="refresh" size={12} />
        </button>
      </div>

      <p className="rounded-lg border border-(--line) bg-(--ground)/60 px-2 py-1.5 text-[10px] leading-snug text-(--ink-soft)">
        First-party local events only. An impression requires a card to be at least{" "}
        {profile?.behavior?.impression_visibility_threshold != null
          ? Math.round(profile.behavior.impression_visibility_threshold * 100)
          : 60}
        % visible for{" "}
        {profile?.behavior?.impression_min_visible_ms ?? 1000} ms. No external analytics, no raw chat or
        memory text.
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        <ToggleButton enabled={enabled} busy={busy} onClick={() => void toggle()} />
        {confirmDelete ? (
          <>
            <Button size="sm" variant="primary" disabled={busy} onClick={() => void remove()}>
              Confirm delete
            </Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
          </>
        ) : (
          <Button size="sm" variant="secondary" icon="x" disabled={busy} onClick={() => setConfirmDelete(true)}>
            Delete session events
          </Button>
        )}
      </div>
      {deleteMessage ? <p className="text-[10.5px] text-(--ok)">{deleteMessage}</p> : null}

      {loading && !profile ? (
        <div className="space-y-1.5">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-20 w-full" />
        </div>
      ) : null}
      {error ? (
        <div className="rounded-lg border border-(--danger)/30 bg-(--danger-soft)/40 px-2 py-1.5 text-[11px] text-(--danger)">
          {error.message}
        </div>
      ) : null}

      {profile ? (
        <>
          {!profile.enabled ? (
            <p className="rounded-lg border border-(--line) bg-(--ground)/60 px-2 py-1.5 text-[11px] leading-snug text-(--ink-soft)">
              Personalization is off. No new behavior events are collected for this session; the
              previously stored events are hidden from the profile and can be deleted above.
            </p>
          ) : (
            <>
              <Counts profile={profile} />
              <div className="grid grid-cols-2 gap-1.5 text-[10px] leading-snug">
                <div className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5">
                  <div className="font-semibold text-(--ink-2)">Explicit M9 memory</div>
                  <div className="text-(--ink-soft)">
                    {profile.explicit_memory?.active_count ?? 0} active · long-term · only user-confirmed writes
                  </div>
                </div>
                <div className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5">
                  <div className="font-semibold text-(--ink-2)">Short-term behavior</div>
                  <div className="text-(--ink-soft)">
                    {profile.counts.live_events} profile events · {Math.round(profile.window_seconds / 60)} min window
                    {profile.profile_provenance ? (
                      <span className="block">
                        {profile.profile_provenance.human_live_events} human ·{" "}
                        {profile.profile_provenance.automated_browser_e2e_events} automated browser
                      </span>
                    ) : null}
                  </div>
                </div>
              </div>
              <p className="text-[9.5px] leading-snug text-(--ink-faint)">
                Behavior interests are weak signals and are not permanent memory. Impressions are
                exposure context only; compare attempt/completed/failed is operation provenance, not a
                preference. Only a verified click or explicit Like/Dislike contributes short-term interest.
              </p>
              {visibleSuggestion ? (
                <SuggestionCard
                  suggestion={visibleSuggestion}
                  busy={suggestionBusy}
                  result={suggestionResult}
                  writesEnabled={memoryWritesEnabled}
                  onAccept={() => void acceptSuggestion()}
                  onDismiss={() => setDismissedSuggestion(visibleSuggestion.suggestion_id)}
                />
              ) : null}
              <Section title="Short-term interests" aside={`${profile.attribute_interests.length} attributes`} open>
                <InterestList profile={profile} />
              </Section>
              <Section title="Recent behavior events" aside={`${profile.counts.live_events} profile`}>
                <EventList profile={profile} />
              </Section>
              <Section title="Eligible OFF / ON + diagnostic" aside={shadow?.eligible_ranking_changed ? "eligible diff" : "no eligible diff"} open>
                <ShadowBlock
                  shadow={shadow}
                  loading={shadowLoading}
                  error={shadowError}
                  hasRecommendation={Boolean(latestRecommendationId)}
                />
              </Section>
              <p className="text-[9.5px] leading-snug text-(--ink-faint)">
                Behavior signals never write M9 automatically, never bypass the Gate and never change
                the production default ranking. Deleting events does not touch M9 memory or the
                conversation.
              </p>
            </>
          )}
        </>
      ) : !loading ? (
        <EmptyState icon="info" compact title="No behavior profile yet" hint="Send a shopping query and interact with the product cards." />
      ) : null}
    </div>
  );
}
