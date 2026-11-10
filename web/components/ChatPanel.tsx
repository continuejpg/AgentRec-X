// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): Transcript/ActivityLine adapted into the demo chat
// column — real SSE turn rendering (tool timeline, products, comparison, clarification, memory
// update, error/interrupted states), suggested real actions, readiness banner, fixed composer
// and the comparison bar. Markdown is dropped (plain-text narrative) and no retries are faked.

"use client";

import { useCallback, useMemo } from "react";
import type { LocalTurn } from "@/lib/chat";
import type { QwenLiveState } from "@/lib/modes";
import { plannerLatencyTotal, plannerSummary } from "@/lib/planner";
import type {
  DemoConfig,
  ErrorPayload,
  Health,
  PlannerMeta,
  ProductCard,
  SessionView,
} from "@/lib/types";
import { BFF_BASE } from "@/lib/api";
import { formatDuration, formatLatency, plural, statusLabel, toolPhaseLabel } from "@/lib/format";
import { useStickToBottom } from "@/lib/scroll";
import { Icon } from "./icons";
import { AssistantText, ErrorBubble, UserBubble } from "./MessageBubble";
import { Button, Pill, Skeleton } from "./ui";
import ClarificationCard, { MemoryUpdateStats } from "./ClarificationCard";
import ComparisonBar from "./ComparisonBar";
import { ComparisonSummaryCard } from "./ComparisonView";
import { Composer, type Prefill } from "./Composer";
import ProductsBlock from "./ProductsBlock";
import QwenLiveNotice from "./QwenLiveNotice";
import ReadinessBanner from "./ReadinessBanner";
import ToolTimeline from "./ToolTimeline";
import type { UiMode } from "./Header";

/** Latest real phase, from the tool rows themselves. */
function currentPhase(turn: LocalTurn): string {
  for (let index = turn.tool_activity.length - 1; index >= 0; index -= 1) {
    const row = turn.tool_activity[index];
    if (row.status === "running") return toolPhaseLabel(row.tool, row.label);
  }
  return turn.tool_activity.length ? "Preparing the response…" : "Contacting the local BFF…";
}

function preferredProduct(turn: LocalTurn): ProductCard | null {
  const payload = turn.products;
  if (!payload?.items.length) return null;
  return (
    payload.items.find((item) => item.product_id === payload.recommended_product_id) ??
    payload.items.find((item) => item.recommended) ??
    payload.items[0]
  );
}

/**
 * Real planner metadata for one assistant turn (docs/QWEN_LIVE_CONTRACT.md §3).
 *
 * Rendered only when the backend reported a planner `kind`; the label comes from that kind and
 * nothing is inferred when it is absent (an older BFF or a Rule turn without metadata).
 */
function PlannerBadge({ planner }: { planner: PlannerMeta }) {
  const latency = plannerLatencyTotal(planner);
  const title = [
    planner.model ? `model ${planner.model}` : null,
    latency != null ? `planner latency ${formatLatency(latency)}` : null,
    planner.actions?.length ? `${planner.actions.length} reported actions` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div
      className="flex flex-wrap items-center gap-x-2 gap-y-1"
      data-testid="turn-planner"
      data-planner-kind={planner.kind}
      title={title || undefined}
    >
      <Pill tone={planner.kind === "qwen_base" ? "accent" : "muted"} dot>
        <Icon name="spark" size={10} />
        {plannerSummary(planner)}
      </Pill>
      {latency != null ? (
        <span className="font-mono text-[10px] text-(--ink-faint)">planner {formatLatency(latency)}</span>
      ) : null}
      {planner.model ? <span className="font-mono text-[10px] text-(--ink-faint)">{planner.model}</span> : null}
    </div>
  );
}

function ActivityLine({ turn }: { turn: LocalTurn }) {
  return (
    <div role="status" className="flex items-center gap-2 text-[12.5px] text-(--ink-soft)" data-testid="activity-line">
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-(--accent)" />
      <span className="min-w-0 truncate">{currentPhase(turn)}</span>
    </div>
  );
}

function TurnMeta({ turn }: { turn: LocalTurn }) {
  if (turn.status === "streaming") return null;
  if (turn.status === "interrupted") {
    return (
      <p className="rounded-lg border border-(--warn)/40 bg-(--warn-soft)/50 px-2.5 py-1.5 text-[11.5px] leading-snug text-(--warn)" data-testid="interrupted-notice">
        Response stopped by you. The BFF may still be finishing this turn; reconnect the session to sync the persisted
        state, or retry the message. Nothing is filled in locally.
      </p>
    );
  }
  const bits: string[] = [];
  if (turn.total_ms != null) bits.push(`done in ${formatDuration(turn.total_ms)}`);
  if (turn.stop_reason) bits.push(statusLabel(turn.stop_reason));
  if (turn.turn_id) bits.push(`turn ${turn.turn_id}`);
  if (!bits.length && turn.status !== "error") return null;
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-(--ink-faint)" data-testid="turn-meta">
      {bits.map((bit, index) => (
        <span key={`${bit}-${index}`} className="font-mono">
          {bit}
          {index < bits.length - 1 ? <span className="ml-2 text-(--line-strong)">·</span> : null}
        </span>
      ))}
    </div>
  );
}

function TurnActions({
  turn,
  selection,
  compareMin,
  compareMax,
  onViewDetails,
  onViewEvidence,
  onCompare,
  onOpenComparison,
  onRetryMessage,
}: {
  turn: LocalTurn;
  selection: string[];
  compareMin: number;
  compareMax: number;
  onViewDetails: (product: ProductCard) => void;
  onViewEvidence: (product: ProductCard) => void;
  onCompare: (ids: string[]) => void;
  onOpenComparison: () => void;
  onRetryMessage: (text: string) => void;
}) {
  if (turn.status === "streaming") return null;
  const product = preferredProduct(turn);
  const canCompare = selection.length >= compareMin && selection.length <= compareMax;
  const hasAnything = product || turn.comparison || turn.error || turn.status === "interrupted" || turn.reply?.recommendation;
  if (!hasAnything) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5" data-testid="turn-actions">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-(--ink-faint)">Actions</span>
      {product ? (
        <>
          <Button size="sm" variant="secondary" onClick={() => onViewDetails(product)}>
            View details
          </Button>
          <Button size="sm" variant="secondary" onClick={() => onViewEvidence(product)}>
            View evidence
          </Button>
        </>
      ) : null}
      {canCompare ? (
        <Button size="sm" variant="secondary" icon="compare" onClick={() => onCompare(selection)}>
          Compare selected ({selection.length})
        </Button>
      ) : null}
      {turn.comparison ? (
        <Button size="sm" variant="secondary" icon="compare" onClick={onOpenComparison}>
          Open comparison
        </Button>
      ) : null}
      {turn.error || turn.status === "interrupted" ? (
        <Button size="sm" variant="secondary" icon="refresh" onClick={() => onRetryMessage(turn.user_text)}>
          Retry this message
        </Button>
      ) : null}
    </div>
  );
}

function TurnItem({
  turn,
  isLast,
  streaming,
  selection,
  compareMin,
  compareMax,
  lastHeartbeatAt,
  onToggleSelect,
  onViewDetails,
  onViewEvidence,
  onCompare,
  onOpenComparison,
  onReplyClarification,
  onRetryMessage,
  writeProfileId,
  writeProfileLabel,
  onSwitchProfileRetry,
}: {
  turn: LocalTurn;
  isLast: boolean;
  streaming: boolean;
  selection: string[];
  compareMin: number;
  compareMax: number;
  lastHeartbeatAt: number | null;
  onToggleSelect: (product: ProductCard) => void;
  onViewDetails: (product: ProductCard) => void;
  onViewEvidence: (product: ProductCard) => void;
  onCompare: (ids: string[]) => void;
  onOpenComparison: () => void;
  onReplyClarification: (turn: LocalTurn, text: string) => void;
  onRetryMessage: (text: string) => void;
  writeProfileId: string | null;
  writeProfileLabel: string | null;
  onSwitchProfileRetry: ((profileId: string, query: string) => void) | null;
}) {
  const showActivity =
    turn.status === "streaming" && !turn.assistant_text && !turn.products && !turn.clarification && !turn.error;
  return (
    <div className="space-y-2.5" data-testid="turn" data-turn-id={turn.turn_id ?? turn.key}>
      {turn.user_text ? <UserBubble text={turn.user_text} /> : null}
      <div className="flex items-start gap-2 sm:gap-2.5">
        <span
          aria-hidden
          className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-(--ink) text-(--accent)"
        >
          <Icon name="spark" size={13} />
        </span>
        <div className="min-w-0 flex-1 space-y-2">
          {turn.tool_activity.length || turn.status === "streaming" ? (
            <ToolTimeline
              activity={turn.tool_activity}
              streaming={turn.status === "streaming"}
              totalMs={turn.total_ms}
              stopReason={turn.stop_reason}
              lastHeartbeatAt={lastHeartbeatAt}
              defaultOpen={turn.status === "streaming" || isLast}
            />
          ) : null}

          {turn.assistant_text ? (
            <AssistantText text={turn.assistant_text} streaming={turn.status === "streaming" && !turn.text_final} />
          ) : null}

          {turn.products ? (
            <ProductsBlock
              payload={turn.products}
              selection={selection}
              compareMin={compareMin}
              compareMax={compareMax}
              onToggleSelect={onToggleSelect}
              onViewDetails={onViewDetails}
              onViewEvidence={onViewEvidence}
              onCompare={onCompare}
            />
          ) : null}

          {turn.comparison ? <ComparisonSummaryCard comparison={turn.comparison} onOpen={onOpenComparison} /> : null}

          {turn.memory_update && !turn.reply?.memory_update ? <MemoryUpdateStats update={turn.memory_update} /> : null}

          {turn.clarification ? (
            <ClarificationCard
              clarification={turn.clarification}
              reply={turn.reply}
              busy={streaming}
              retryQuery={turn.user_text}
              writeProfileId={writeProfileId}
              writeProfileLabel={writeProfileLabel}
              onSwitchProfileRetry={onSwitchProfileRetry}
              onReply={(text) => onReplyClarification(turn, text)}
            />
          ) : null}

          {turn.reply?.recommendation ? (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">
                <Icon name="check" size={12} className="text-(--ok)" />
                Continued recommendation after your reply
              </div>
              <ProductsBlock
                payload={turn.reply.recommendation}
                selection={selection}
                compareMin={compareMin}
                compareMax={compareMax}
                onToggleSelect={onToggleSelect}
                onViewDetails={onViewDetails}
                onViewEvidence={onViewEvidence}
                onCompare={onCompare}
              />
            </div>
          ) : null}

          {turn.error ? <ErrorBubble error={turn.error} /> : null}

          {turn.planner ? <PlannerBadge planner={turn.planner} /> : null}

          {showActivity ? <ActivityLine turn={turn} /> : null}

          <TurnMeta turn={turn} />
          {isLast ? (
            <TurnActions
              turn={turn}
              selection={selection}
              compareMin={compareMin}
              compareMax={compareMax}
              onViewDetails={onViewDetails}
              onViewEvidence={onViewEvidence}
              onCompare={onCompare}
              onOpenComparison={onOpenComparison}
              onRetryMessage={onRetryMessage}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function EmptyConversation({
  config,
  ready,
  mode,
  onSend,
  onOpenReplay,
}: {
  config: DemoConfig | null;
  ready: boolean;
  mode: UiMode;
  onSend: (text: string, selectedIds?: string[]) => void;
  onOpenReplay: () => void;
}) {
  const verified = config?.verified_comparison_ids ?? [];
  const prompts = useMemo(() => {
    const list: { key: string; label: string; text: string; hint: string; selectedIds?: string[] }[] = [
      {
        key: "search",
        label: "Search with evidence",
        text: "I need earbuds I can use while working out. It needs to be wireless.",
        hint: "Real BGE search → Gate → P15 evidence and UNKNOWN fields.",
      },
      {
        key: "clarify",
        label: "Vague request → clarification",
        text: "I need some new headphones. Not sure what matters yet.",
        hint: "Demo B (write profile) lets you answer the clarification and persist memory; Demo A offers to switch and retry.",
      },
    ];
    if (verified.length >= 2) {
      list.push({
        key: "compare",
        label: "Compare two products",
        text: `Compare ${verified.slice(0, 2).join(" and ")}.`,
        hint: "Pre-selects the verified comparison ids from GET /api/demo/config, then calls the real P16 compare.",
        selectedIds: verified.slice(0, 2),
      });
    }
    return list;
  }, [verified]);

  return (
    <div className="space-y-5 py-2" data-testid="empty-conversation">
      <div className="rounded-2xl border border-(--line) bg-(--card) p-4 shadow-(--shadow-sm)">
        <div className="flex items-center gap-2.5">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-(--ink) text-(--accent)">
            <Icon name="spark" size={20} />
          </span>
          <div>
            <h1 className="text-[16px] font-bold tracking-[-0.01em] text-(--ink)">Shop with a grounded assistant</h1>
            <p className="text-[11.5px] text-(--ink-soft)">
              AgentRec-X local demo · CPU Live is the default mode
            </p>
          </div>
          {mode === "qwen_replay" ? (
            <Pill tone="warn" className="ml-auto">
              Historical Replay
            </Pill>
          ) : null}
        </div>
        <p className="mt-2.5 text-[13px] leading-relaxed text-(--ink-2)">
          Ask natural-language shopping questions. The assistant searches the real 76,263-item Electronics catalogue
          through the P17 MCP tools, then shows catalogue evidence, Gate decisions and UNKNOWN fields. Nothing here is a
          mock product: no invented price, rating, stock or brand.
        </p>
        <ul className="mt-2 grid gap-1.5 text-[12px] text-(--ink-soft) sm:grid-cols-2">
          <li className="flex items-start gap-1.5">
            <Icon name="check" size={12} className="mt-0.5 text-(--ok)" /> Catalogue facts with raw evidence spans
          </li>
          <li className="flex items-start gap-1.5">
            <Icon name="check" size={12} className="mt-0.5 text-(--ok)" /> P16 comparisons of 2–5 products
          </li>
          <li className="flex items-start gap-1.5">
            <Icon name="check" size={12} className="mt-0.5 text-(--ok)" /> Real M9 preference memory on the write profile
          </li>
          <li className="flex items-start gap-1.5">
            <Icon name="alert" size={12} className="mt-0.5 text-(--warn)" /> No cart, checkout, price or stock claims
          </li>
        </ul>
      </div>

      <div>
        <div className="mb-1.5 flex items-center gap-2">
          <h2 className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Try a demo prompt</h2>
          <span className="text-[10.5px] text-(--ink-faint)">suggested inputs only</span>
        </div>
        <div className="grid gap-1.5 sm:grid-cols-3">
          {prompts.map((prompt) => (
            <button
              key={prompt.key}
              type="button"
              disabled={!ready || mode === "qwen_replay"}
              onClick={() => onSend(prompt.text, prompt.selectedIds)}
              data-testid={`demo-prompt-${prompt.key}`}
              className="flex flex-col gap-1 rounded-xl border border-(--line) bg-(--card) p-2.5 text-left transition-colors hover:border-(--accent) hover:bg-(--accent-soft)/40 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="text-[11px] font-bold uppercase tracking-wide text-(--ink-soft)">{prompt.label}</span>
              <span className="line-clamp-3 text-[12px] font-medium leading-snug text-(--ink)">“{prompt.text}”</span>
              <span className="mt-auto text-[10px] leading-snug text-(--ink-faint)">{prompt.hint}</span>
            </button>
          ))}
        </div>
      </div>

      {mode === "qwen_replay" ? (
        <div className="rounded-xl border border-amber-500/50 bg-amber-50 p-3 text-[12px] text-amber-950">
          <div className="font-bold uppercase tracking-wide">Historical Replay — Not a Live Model Call</div>
          <p className="mt-1">
            You are viewing saved Qwen trajectories. Switch to CPU Live to chat, or open the replay viewer to inspect a
            saved trajectory.
          </p>
          <Button size="sm" variant="secondary" icon="play" className="mt-2" onClick={onOpenReplay}>
            Open replay viewer
          </Button>
        </div>
      ) : null}

      {config?.notices.length ? (
        <ul className="list-disc space-y-0.5 pl-4 text-[10.5px] leading-snug text-(--ink-faint)">
          {config.notices.map((notice) => (
            <li key={notice}>{notice}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function ConversationSkeleton() {
  return (
    <div className="space-y-5 py-2" aria-hidden>
      <Skeleton className="h-24 w-full" />
      <div className="space-y-2">
        <Skeleton className="ml-auto h-10 w-2/3" />
        <Skeleton className="h-20 w-5/6" />
      </div>
    </div>
  );
}

export default function ChatPanel({
  health,
  healthError,
  config,
  session,
  sessionLoading,
  sessionError,
  turns,
  busy,
  mode,
  qwenLive,
  onSwitchToCpuLive,
  selection,
  compareMin,
  compareMax,
  selectionError,
  comparisonLoading,
  lastHeartbeatAt,
  readinessElapsed,
  prefill,
  onRetryHealth,
  onReconnect,
  onSend,
  onStop,
  onRetryMessage,
  onToggleSelect,
  onViewDetails,
  onViewEvidence,
  onCompare,
  onReplyClarification,
  onOpenComparison,
  onRemoveSelection,
  onClearSelection,
  onOpenReplay,
  onNewConversation,
  writeProfileId,
  writeProfileLabel,
  onSwitchProfileRetry,
}: {
  health: Health | null;
  healthError: string | null;
  config: DemoConfig | null;
  session: SessionView | null;
  sessionLoading: boolean;
  sessionError: ErrorPayload | null;
  turns: LocalTurn[];
  busy: boolean;
  mode: UiMode;
  qwenLive: QwenLiveState;
  onSwitchToCpuLive: () => void;
  selection: string[];
  compareMin: number;
  compareMax: number;
  selectionError: string | null;
  comparisonLoading: boolean;
  lastHeartbeatAt: number | null;
  readinessElapsed: number;
  prefill: Prefill | null;
  onRetryHealth: () => void;
  onReconnect: () => void;
  onSend: (text: string, selectedIds?: string[]) => void;
  onStop: () => void;
  onRetryMessage: (text: string) => void;
  onToggleSelect: (product: ProductCard) => void;
  onViewDetails: (product: ProductCard) => void;
  onViewEvidence: (product: ProductCard) => void;
  onCompare: (ids: string[]) => void;
  onReplyClarification: (turn: LocalTurn, text: string) => void;
  onOpenComparison: () => void;
  onRemoveSelection: (productId: string) => void;
  onClearSelection: () => void;
  onOpenReplay: () => void;
  onNewConversation: () => void;
  writeProfileId: string | null;
  writeProfileLabel: string | null;
  onSwitchProfileRetry: ((profileId: string, query: string) => void) | null;
}) {
  const { scrollRef, onScroll, showLatest, jumpToLatest } = useStickToBottom(turns, busy);
  const ready = health?.ready === true;
  const pausedForReplay = mode === "qwen_replay";
  // Qwen Live is a live mode only while the backend reports it available; while it is not, the
  // composer stays disabled and an explicit notice replaces any silent fallback.
  const qwenBlocked = mode === "qwen_live" && !qwenLive.available;
  const liveReady = ready && !pausedForReplay && !qwenBlocked;

  const jump = useCallback(() => jumpToLatest(), [jumpToLatest]);

  return (
    <div className="relative flex h-full min-h-0 flex-col">
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="panel-scroll chat-ground min-h-0 flex-1 overflow-y-auto"
        data-testid="transcript"
        aria-live="polite"
      >
        <div className="mx-auto w-full max-w-3xl px-3 py-3 sm:px-4 sm:py-4">
          <div className="space-y-4">
            <ReadinessBanner
              health={health}
              healthError={healthError}
              config={config}
              bffBase={BFF_BASE}
              elapsedSeconds={readinessElapsed}
              onRetry={onRetryHealth}
            />
          </div>
          <QwenLiveNotice qwenLive={qwenLive} mode={mode} onSwitchToCpu={onSwitchToCpuLive} />
          {sessionError ? (
            <div className="mt-4 rounded-xl border border-(--danger)/40 bg-(--danger-soft)/50 p-3" data-testid="session-error">
              <div className="flex items-center gap-1.5 text-[12.5px] font-semibold text-(--danger)">
                <Icon name="alert" size={13} /> Session error
              </div>
              <p className="mt-1 text-[12px] leading-snug text-(--ink-2)">
                {sessionError.message} <span className="font-mono text-[10.5px] text-(--ink-soft)">({sessionError.code})</span>
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" icon="refresh" onClick={onReconnect}>
                  Reconnect session
                </Button>
                <Button size="sm" variant="primary" icon="spark" onClick={onNewConversation}>
                  New conversation
                </Button>
              </div>
            </div>
          ) : null}

          {pausedForReplay ? (
            <div className="mt-4 rounded-xl border-2 border-amber-500/60 bg-amber-50 p-3 text-amber-950" data-testid="replay-mode-notice">
              <div className="flex items-center gap-2">
                <Icon name="alert" size={15} className="text-amber-600" />
                <span className="text-[12px] font-bold uppercase tracking-[0.04em]">
                  Historical Replay — Not a Live Model Call
                </span>
              </div>
              <p className="mt-1 text-[12px] leading-relaxed">
                The assistant is paused while you view saved Qwen trajectories. The composer is disabled in this mode;
                switch back to CPU Live to send a message.
              </p>
              <Button size="sm" variant="secondary" icon="play" className="mt-2" onClick={onOpenReplay}>
                Open replay viewer
              </Button>
            </div>
          ) : null}

          <div className="mt-4 space-y-5 pb-2">
            {sessionLoading ? <ConversationSkeleton /> : null}
            {!sessionLoading && !turns.length ? (
              <EmptyConversation config={config} ready={liveReady} mode={mode} onSend={onSend} onOpenReplay={onOpenReplay} />
            ) : null}
            {turns.map((turn, index) => (
              <TurnItem
                key={turn.key}
                turn={turn}
                isLast={index === turns.length - 1}
                streaming={busy}
                selection={selection}
                compareMin={compareMin}
                compareMax={compareMax}
                lastHeartbeatAt={lastHeartbeatAt}
                onToggleSelect={onToggleSelect}
                onViewDetails={onViewDetails}
                onViewEvidence={onViewEvidence}
                onCompare={onCompare}
                onOpenComparison={onOpenComparison}
                onReplyClarification={onReplyClarification}
                onRetryMessage={onRetryMessage}
                writeProfileId={writeProfileId}
                writeProfileLabel={writeProfileLabel}
                onSwitchProfileRetry={onSwitchProfileRetry}
              />
            ))}
          </div>
        </div>
      </div>

      {showLatest ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-28 z-10 flex justify-center">
          <button
            type="button"
            onClick={jump}
            className="pointer-events-auto rounded-full border border-(--line) bg-(--card) px-3.5 py-1.5 text-[12.5px] font-semibold text-(--ink) shadow-md transition hover:border-(--accent)"
          >
            ↓ Latest
          </button>
        </div>
      ) : null}

      <div className="ac-safe-bottom shrink-0 border-t border-(--line) bg-(--ground)/95 px-3 pb-2 pt-2 backdrop-blur sm:px-4">
        <div className="mx-auto w-full max-w-3xl">
          <ComparisonBar
            selection={selection}
            compareMin={compareMin}
            compareMax={compareMax}
            loading={comparisonLoading}
            selectionError={selectionError}
            onRemove={onRemoveSelection}
            onClear={onClearSelection}
            onCompare={() => onCompare(selection)}
          />
          <Composer
            send={onSend}
            stop={onStop}
            ready={liveReady}
            busy={busy}
            prefill={prefill}
            disabledHint={
              pausedForReplay
                ? "Qwen Replay is read-only historical trajectories — switch to CPU Live to chat."
                : qwenBlocked
                  ? "Qwen Live is unavailable — switch back to CPU Live to chat (no silent fallback)."
                  : !ready
                    ? "Waiting for the local BFF and MCP servers (cold start ~35–45 s)…"
                    : undefined
            }
            placeholder={
              session
                ? "Ask for a product, compare items, or share a preference…"
                : "Waiting for a session…"
            }
          />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5 text-[10px] text-(--ink-faint)">
            <span>
              {session ? `${session.turns.length ? plural(session.turns.length, "persisted turn") : "New session"} · ${session.session_id}` : "No session yet"}
            </span>
            <span className="flex items-center gap-1">
              <Icon name="info" size={10} />
              Catalogue evidence only; UNKNOWN stays UNKNOWN.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
