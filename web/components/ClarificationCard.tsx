// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useState } from "react";
import type { ClarificationReplyState } from "@/lib/chat";
import type { ClarificationPayload, MemoryUpdatePayload, PreferenceState } from "@/lib/types";
import { formatLatency, plural, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { ErrorBubble } from "./MessageBubble";
import Suggestions from "./Suggestions";
import { Button, Pill, SectionTitle } from "./ui";

export function MemoryUpdateStats({ update }: { update: MemoryUpdatePayload }) {
  const rows: { label: string; value: number; tone?: "ok" | "warn" }[] = [
    { label: "Added", value: update.added, tone: update.added ? "ok" : undefined },
    { label: "Superseded", value: update.superseded },
    { label: "Removed", value: update.removed },
    { label: "Skipped duplicates", value: update.skipped_duplicates },
  ];
  return (
    <div className="rounded-xl border border-(--line) bg-(--card)/70 p-3" data-testid="memory-update">
      <div className="flex flex-wrap items-center gap-2">
        <Icon name="check" size={13} className="text-(--ok)" />
        <span className="text-[12.5px] font-semibold text-(--ink)">Memory update</span>
        {update.already_processed ? (
          <Pill tone="info" title="Idempotent retry: this reply had already been processed">
            already processed
          </Pill>
        ) : (
          <Pill tone="ok">written</Pill>
        )}
        <span className="font-mono text-[10.5px] text-(--ink-faint)" title="confirmation_source">
          {update.confirmation_source}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {rows.map((row) => (
          <div key={row.label} className="rounded-lg bg-(--well)/70 px-2 py-1.5">
            <div className="text-[10.5px] font-medium leading-tight text-(--ink-soft)">{row.label}</div>
            <div className={`text-[16px] font-semibold tabular-nums ${row.tone === "ok" ? "text-(--ok)" : "text-(--ink)"}`}>
              {row.value}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-1.5 text-[10.5px] leading-snug text-(--ink-faint)">
        Real M9 result returned by the BFF after an explicit web-UI reply; this local demo writes to the labelled
        test profile only.
      </p>
    </div>
  );
}

function PreferencesDelta({ preferences }: { preferences: PreferenceState | null }) {
  if (!preferences) return null;
  return (
    <div className="rounded-xl border border-(--line) bg-(--info-soft)/40 p-3" data-testid="preferences-refreshed">
      <div className="flex items-center gap-2">
        <Icon name="refresh" size={13} className="text-(--info)" />
        <span className="text-[12.5px] font-semibold text-(--ink)">Active preferences refreshed</span>
        <span className="ml-auto text-[11px] text-(--ink-soft)">{plural(preferences.active.length, "active preference")}</span>
      </div>
      {preferences.active.length ? (
        <ul className="mt-1.5 space-y-1">
          {preferences.active.slice(0, 6).map((item) => (
            <li key={item.memory_id} className="flex flex-wrap items-baseline gap-1.5 text-[12px]">
              <Pill tone={item.polarity === "avoid" ? "danger" : "ok"}>{statusLabel(item.polarity)}</Pill>
              <span className="font-medium text-(--ink)">{item.value}</span>
              <span className="text-(--ink-soft)">· {statusLabel(item.kind)}</span>
              <span className="truncate text-[11px] italic text-(--ink-soft)" title={item.source_text ?? undefined}>
                “{item.source_text}”
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1 text-[12px] text-(--ink-soft)">The refreshed state returned no active preferences.</p>
      )}
      <p className="mt-1.5 font-mono text-[10px] text-(--ink-faint)" title={preferences.fetched_at}>
        source: {preferences.audit_source}
      </p>
    </div>
  );
}

/** The continuation steps the BFF reports for a clarification reply (real tool rows). */
function ReplySteps({ reply }: { reply: ClarificationReplyState }) {
  if (!reply.steps.length) return null;
  return (
    <div className="rounded-xl border border-(--line) bg-(--card)/60 p-3">
      <SectionTitle aside={`${plural(reply.steps.length, "step")}`}>Continuation tools</SectionTitle>
      <ul className="space-y-1">
        {reply.steps.map((step, index) => (
          <li key={`${step.tool}-${index}`} className="flex items-center gap-2 text-[12px]">
            <Icon
              name={step.status === "error" ? "alert" : "check"}
              size={12}
              className={step.status === "error" ? "text-(--danger)" : "text-(--ok)"}
            />
            <span className="font-mono text-[11px] text-(--ink-2)">{step.tool}</span>
            <span className="min-w-0 flex-1 truncate text-(--ink-soft)">{step.label}</span>
            <span className="font-mono text-[10.5px] tabular-nums text-(--ink-soft)">{formatLatency(step.latency_ms)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ClarificationCard({
  clarification,
  reply,
  busy,
  retryQuery,
  writeProfileId,
  writeProfileLabel,
  onReply,
  onSwitchProfileRetry,
}: {
  clarification: ClarificationPayload;
  reply: ClarificationReplyState | null;
  busy: boolean;
  /** The original user query, so a read-only profile can hand the same question to the write profile. */
  retryQuery: string;
  writeProfileId: string | null;
  writeProfileLabel: string | null;
  onReply: (text: string) => void;
  onSwitchProfileRetry: ((profileId: string, query: string) => void) | null;
}) {
  const [draft, setDraft] = useState("");
  const [validation, setValidation] = useState<string | null>(null);
  const pending = reply?.status === "sending";
  const done = reply?.status === "done";
  const writesOff = clarification.memory_write_enabled === false;
  const disabled = busy || pending || done || writesOff;

  const submit = () => {
    const text = draft.trim();
    if (!text) {
      setValidation("Write your preference reply first — the backend only stores what you actually type.");
      return;
    }
    setValidation(null);
    onReply(text);
  };

  return (
    <section
      className="rounded-2xl border border-(--warn)/40 bg-(--warn-soft)/35 p-3.5 shadow-(--shadow-sm) ac-reveal"
      data-testid="clarification-card"
      data-clarification-id={clarification.clarification_id}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-(--warn)/15 text-(--warn)">
          <Icon name="message" size={14} />
        </span>
        <h3 className="text-[13.5px] font-semibold text-(--ink)">The assistant needs one preference from you</h3>
        <Pill tone="warn" dot className="ml-auto">
          {statusLabel(clarification.reason)}
        </Pill>
      </div>
      <p className="mt-2 text-[13.5px] leading-relaxed text-(--ink)" data-testid="clarification-question">
        {clarification.question}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-(--ink-soft)">
        <span title={clarification.clarification_id}>handle {clarification.clarification_id}</span>
        <span>
          {clarification.expires_in_seconds != null
            ? `expires in ${Math.max(1, Math.round(clarification.expires_in_seconds / 60))} min`
            : "expiry not reported"}
        </span>
        <span>
          memory writes {clarification.memory_write_enabled ? "enabled" : "disabled"}
        </span>
      </div>

      {reply ? (
        <div className="mt-3 space-y-2.5">
          {reply.status !== "sending" ? (
            <div className="flex justify-end">
              <div className="user-bubble max-w-[86%] whitespace-pre-wrap px-3.5 py-2 text-[13.5px] leading-normal">
                {reply.text}
              </div>
            </div>
          ) : null}
          {pending ? (
            <div className="overflow-hidden rounded-xl border border-(--line) bg-(--card)" data-testid="memory-progress">
              <div className="h-1 w-full overflow-hidden bg-(--well)">
                <div className="ac-indeterminate h-full w-1/3 rounded-full bg-(--accent)" />
              </div>
              <div className="p-3">
                <div className="text-[12.5px] font-semibold text-(--ink)">Submitting your reply to the real MCP server…</div>
                <ul className="mt-1.5 space-y-1 text-[11.5px] text-(--ink-soft)">
                  <li>· Save the reply against the pending clarification</li>
                  <li>· Apply the M5/M9 memory update</li>
                  <li>· Re-read active preferences</li>
                  <li>· Continue the pending query and return the real recommendation</li>
                </ul>
                <p className="mt-1.5 text-[10.5px] text-(--ink-faint)">
                  These are the BFF request stages, not simulated progress; actual tool latencies appear when the
                  response arrives.
                </p>
              </div>
            </div>
          ) : null}
          {reply.status === "done" ? (
            <>
              {reply.memory_update ? <MemoryUpdateStats update={reply.memory_update} /> : null}
              <PreferencesDelta preferences={reply.preferences} />
              <ReplySteps reply={reply} />
              {reply.recommendation ? (
                <p className="text-[12px] text-(--ink-soft)">
                  The pending query continued and returned a real recommendation below ({reply.recommendation.items.length}{" "}
                  catalogue results).
                </p>
              ) : (
                <p className="text-[12px] text-(--ink-soft)">
                  The server returned no recommendation for the continued query.
                </p>
              )}
            </>
          ) : null}
          {reply.status === "error" && reply.error ? (
            <>
              <ErrorBubble error={reply.error} />
              <div className="text-[11.5px] text-(--ink-soft)">
                The clarification is still open; you can submit a new reply if the server allows it.
              </div>
            </>
          ) : null}
        </div>
      ) : null}

      {!done ? (
        <div className="mt-3">
          {writesOff ? (
            <div className="rounded-xl border border-(--warn)/40 bg-(--warn-soft)/60 px-3 py-2 text-[12px] text-(--warn)" data-testid="writes-disabled-notice">
              <p>
                Memory writes are disabled for this profile/server, so the BFF refuses clarification replies here.
                The reply is not faked; asking again on the write-enabled profile is the real path.
              </p>
              {writeProfileId && onSwitchProfileRetry ? (
                <button
                  type="button"
                  onClick={() => onSwitchProfileRetry(writeProfileId, retryQuery)}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-(--warn) px-2.5 py-1 text-[11.5px] font-semibold text-white transition hover:brightness-110"
                >
                  <Icon name="refresh" size={12} />
                  Switch to {writeProfileLabel ?? writeProfileId} and retry this query
                </button>
              ) : null}
              <p className="mt-1 text-[10.5px] text-(--warn)/90">
                A new session is created for the selected demo profile; the current read-only session stays in Recent
                conversations.
              </p>
            </div>
          ) : (
            <>
              {clarification.quick_options.length ? (
                <Suggestions
                  suggestions={clarification.quick_options}
                  disabled={disabled}
                  onPick={(option) => {
                    setDraft(option);
                    setValidation(null);
                  }}
                />
              ) : null}
              <label className="flex items-end gap-2 rounded-xl border border-(--line-strong) bg-(--card) p-1.5 pl-3 focus-within:border-(--accent)">
                <textarea
                  data-testid="clarification-input"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      if (!disabled) submit();
                    }
                  }}
                  rows={2}
                  disabled={disabled}
                  aria-label="Your reply to the clarification"
                  placeholder="Type your preference, e.g. “I prefer noise cancelling and wireless.”"
                  className="min-h-[42px] flex-1 resize-none bg-transparent px-1 py-1.5 text-[13.5px] text-(--ink) outline-none placeholder:text-(--ink-soft)/70 disabled:cursor-not-allowed"
                />
                <Button
                  variant="accent"
                  size="sm"
                  data-testid="clarification-submit"
                  disabled={disabled || !draft.trim()}
                  onClick={submit}
                >
                  {pending ? "Sending…" : "Send reply"}
                </Button>
              </label>
              <p className="mt-1 px-1 text-[10.5px] text-(--ink-faint)">
                Only typed input is submitted; the BFF sets the confirmation source because a real HTTP request
                arrived. Quick options appear only when the backend question provides fixed choices.
              </p>
            </>
          )}
          {validation ? <p className="mt-1 text-[11.5px] font-medium text-(--danger)">{validation}</p> : null}
        </div>
      ) : null}
    </section>
  );
}
