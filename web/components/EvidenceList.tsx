// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useState } from "react";
import type { EvidenceItem, UnknownField } from "@/lib/types";
import { attributeLabel, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Pill, type Tone } from "./ui";

/** P15 statuses are automatic checks; tone only highlights recognised outcomes. */
export function verificationTone(status: string | null | undefined): Tone {
  const value = (status ?? "").toLowerCase();
  if (!value) return "muted";
  if (value.includes("verified") || value === "ok" || value.includes("pass")) return "ok";
  if (value.includes("reject") || value.includes("fail") || value.includes("violation")) return "danger";
  if (value.includes("unknown") || value.includes("unsupported") || value.includes("pending")) return "warn";
  return "muted";
}

export function UnknownFieldsList({ fields }: { fields: UnknownField[] }) {
  if (!fields.length) {
    return (
      <p className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5 text-[11.5px] text-(--ink-soft)">
        No UNKNOWN fields reported for this payload.
      </p>
    );
  }
  return (
    <ul className="space-y-1" data-testid="unknown-fields">
      {fields.map((field) => (
        <li key={`${field.attribute}-${field.reason}`} className="flex flex-wrap items-start gap-1.5 text-[11.5px]">
          <span className="rounded-full bg-(--warn-soft) px-1.5 py-px text-[10.5px] font-semibold text-(--warn)">
            UNKNOWN · {attributeLabel(field.attribute)}
          </span>
          <span className="min-w-0 flex-1 leading-snug text-(--ink-soft)">{field.reason}</span>
        </li>
      ))}
    </ul>
  );
}

function EvidenceItemCard({ item }: { item: EvidenceItem }) {
  const [expanded, setExpanded] = useState(false);
  const excerpt = item.raw_excerpt ?? "";
  const longExcerpt = excerpt.length > 220;
  return (
    <li className="rounded-xl border border-(--line) bg-(--card) p-2.5" data-testid="evidence-item" data-evidence-id={item.evidence_id ?? ""}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-[10.5px] text-(--ink-2)">{item.evidence_id ?? "evidence"}</span>
        {item.attribute ? <Pill tone="info">{attributeLabel(item.attribute)}</Pill> : null}
        <Pill tone={verificationTone(item.verification_status)} title="P15 automated verification_status (not human review)">
          {statusLabel(item.verification_status)}
        </Pill>
        <span className="ml-auto whitespace-nowrap font-mono text-[10px] text-(--ink-faint)">
          {item.raw_field ?? "raw"}
          {item.span_start != null && item.span_end != null ? ` · ${item.span_start}–${item.span_end}` : ""}
        </span>
      </div>
      {item.span_text ? (
        <blockquote className="mt-1 border-l-2 border-(--line-strong) pl-2 text-[12px] leading-relaxed text-(--ink)">
          “{item.span_text}”
        </blockquote>
      ) : null}
      {excerpt && excerpt !== item.span_text ? (
        <div className="mt-1">
          <pre
            className={`panel-scroll whitespace-pre-wrap break-words rounded bg-(--well)/60 p-1.5 font-mono text-[10.5px] leading-relaxed text-(--ink-2) ${
              expanded ? "max-h-60 overflow-auto" : "max-h-16 overflow-hidden"
            }`}
          >
            {excerpt}
          </pre>
          {longExcerpt ? (
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              className="mt-0.5 text-[10.5px] font-semibold text-(--ink-soft) underline underline-offset-2 hover:text-(--ink)"
            >
              {expanded ? "Show less" : "Show raw excerpt"}
            </button>
          ) : null}
        </div>
      ) : null}
      <div className="mt-1 flex items-start gap-1 truncate font-mono text-[10px] text-(--ink-faint)" title={item.source_record ?? undefined}>
        <Icon name="box" size={10} className="mt-px shrink-0" />
        <span className="truncate">{item.source_record ?? ""}</span>
      </div>
    </li>
  );
}

export default function EvidenceList({
  evidence,
  unknownFields,
  sourceRecord,
  emptyHint = "No evidence spans were returned for this product.",
}: {
  evidence: EvidenceItem[];
  unknownFields: UnknownField[];
  sourceRecord: string | null;
  emptyHint?: string;
}) {
  return (
    <div className="space-y-2.5">
      <div>
        <div className="mb-1 flex items-center gap-2">
          <h4 className="text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Evidence spans</h4>
          <span className="text-[10.5px] text-(--ink-faint)">{evidence.length}</span>
        </div>
        {evidence.length ? (
          <ul className="space-y-1.5">
            {evidence.map((item, index) => (
              <EvidenceItemCard key={`${item.evidence_id ?? "evidence"}-${index}`} item={item} />
            ))}
          </ul>
        ) : (
          <p className="rounded-lg border border-dashed border-(--line-strong) px-2 py-2 text-[11.5px] text-(--ink-soft)">
            {emptyHint}
          </p>
        )}
      </div>
      <div>
        <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">UNKNOWN fields</h4>
        <UnknownFieldsList fields={unknownFields} />
      </div>
      {sourceRecord ? (
        <div className="border-t border-(--line) pt-1.5">
          <div className="text-[10.5px] font-semibold text-(--ink-soft)">Source record</div>
          <div className="break-all font-mono text-[10px] text-(--ink-2)">{sourceRecord}</div>
        </div>
      ) : null}
      <p className="rounded-lg bg-(--info-soft)/50 px-2 py-1.5 text-[10.5px] leading-snug text-(--info)">
        verification_status comes from the automated P15 evidence checks. There is no human semantic review in this
        demo (<span className="font-mono">semantic_truth_verified=false</span>).
      </p>
    </div>
  );
}
