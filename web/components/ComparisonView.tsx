// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): the storefront ComparisonGrid (pros/cons/price
// delta cards) is replaced by a renderer for the P16 ComparisonPayload contract —
// dimensions/cells, preference relations, conflicts, UNKNOWN fields, claims summary and the
// conditional recommendation, all passed through verbatim.

"use client";

import type {
  ComparisonCell,
  ComparisonConflict,
  ComparisonPayload,
  ComparisonProduct,
  PreferenceRelation,
} from "@/lib/types";
import { attributeLabel, formatLatency, humanize, plural, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Pill, TONE_SOFT, type Tone } from "./ui";
import { ProductImage } from "./ProductImage";

function roleTone(status: string | null | undefined): Tone {
  switch (status) {
    case "MATCH":
    case "match":
      return "ok";
    case "VIOLATION":
    case "violation":
    case "CONFLICT":
      return "danger";
    case "UNKNOWN":
    case "unknown":
      return "warn";
    default:
      return "muted";
  }
}

function relationTone(relation: string | null | undefined): Tone {
  switch (relation) {
    case "evidenced":
    case "supported":
    case "match":
      return "ok";
    case "conflict":
    case "violation":
      return "danger";
    case "not_evidenced":
      return "muted";
    default:
      return "info";
  }
}

function CellValue({ cell }: { cell: ComparisonCell }) {
  const tone = roleTone(cell.status);
  return (
    <div className="flex flex-col gap-0.5">
      <Pill tone={tone} className="w-fit">
        {statusLabel(cell.status)}
      </Pill>
      {cell.value != null && cell.value !== "" ? (
        <span className="whitespace-pre-wrap break-words text-[11.5px] text-(--ink)" title={cell.value ?? undefined}>
          {cell.value}
        </span>
      ) : (
        <span className="text-[10.5px] text-(--ink-faint)">no value</span>
      )}
      {cell.evidence_ids.length ? (
        <span className="font-mono text-[9.5px] text-(--ink-faint)" title={cell.evidence_ids.join(", ")}>
          {cell.evidence_ids.length} evidence ref{cell.evidence_ids.length === 1 ? "" : "s"}
        </span>
      ) : null}
    </div>
  );
}

function ConflictRow({ conflict, index }: { conflict: ComparisonConflict; index: number }) {
  const known = conflict.attribute || conflict.message || conflict.detail;
  return (
    <li className="rounded-lg border border-(--danger)/30 bg-(--danger-soft)/40 p-2 text-[12px] text-(--ink)">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-semibold">Conflict {index + 1}</span>
        {conflict.attribute ? <Pill tone="danger">{attributeLabel(conflict.attribute)}</Pill> : null}
        {conflict.products?.length ? (
          <span className="font-mono text-[10.5px] text-(--ink-soft)">{conflict.products.join(" · ")}</span>
        ) : null}
      </div>
      {known ? (
        <p className="mt-1 leading-snug">{conflict.message ?? conflict.detail ?? ""}</p>
      ) : (
        <pre className="panel-scroll mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words rounded bg-(--card)/70 p-1.5 font-mono text-[10.5px]">
          {JSON.stringify(conflict, null, 2)}
        </pre>
      )}
    </li>
  );
}

function PreferenceRelationRow({ relation }: { relation: PreferenceRelation }) {
  const perProduct = Object.entries(relation.per_product ?? {});
  return (
    <li className="rounded-lg border border-(--line) bg-(--card) p-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Pill tone={relation.polarity === "avoid" ? "danger" : "ok"}>{statusLabel(relation.polarity)}</Pill>
        <span className="text-[12.5px] font-semibold text-(--ink)">{relation.value}</span>
        <Pill tone={relationTone(relation.relation)}>{statusLabel(relation.relation)}</Pill>
        <span className="text-[10.5px] text-(--ink-soft)">
          {relation.observed_in_catalogue_text ? "observed in catalogue text" : "not observed in catalogue text"}
        </span>
      </div>
      {relation.source_text ? (
        <p className="mt-1 line-clamp-2 text-[11px] italic text-(--ink-soft)" title={relation.source_text ?? undefined}>
          “{relation.source_text}”
        </p>
      ) : null}
      {perProduct.length ? (
        <div className="mt-1 flex flex-wrap gap-1">
          {perProduct.map(([productId, value]) => (
            <span
              key={productId}
              title={`${productId}: ${value}`}
              className={`rounded-full px-1.5 py-px font-mono text-[10px] ${
                roleTone(value) === "ok"
                  ? TONE_SOFT.ok
                  : roleTone(value) === "danger"
                    ? TONE_SOFT.danger
                    : TONE_SOFT.muted
              }`}
            >
              {productId} · {statusLabel(value)}
            </span>
          ))}
        </div>
      ) : null}
    </li>
  );
}

export default function ComparisonView({
  comparison,
  latencyMs,
  onOpenDetails,
  compactProducts = false,
}: {
  comparison: ComparisonPayload;
  latencyMs?: number | null;
  onOpenDetails?: (productId: string) => void;
  compactProducts?: boolean;
}) {
  const dimensions = comparison.dimensions ?? [];
  const products = comparison.products ?? [];
  const ok = comparison.status === "comparison_ready";

  return (
    <div className="space-y-3" data-testid="comparison-view" data-status={comparison.status}>
      <div className="flex flex-wrap items-center gap-2">
        <Pill tone={ok ? "ok" : "warn"} dot>
          {statusLabel(comparison.status)}
        </Pill>
        <span className="text-[11.5px] text-(--ink-soft)">{plural(products.length, "product")} compared</span>
        {latencyMs != null ? (
          <span className="ml-auto font-mono text-[10.5px] text-(--ink-faint)" title="HTTP round-trip measured by this page">
            round-trip {formatLatency(latencyMs)}
          </span>
        ) : null}
      </div>

      {!compactProducts ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {products.map((product: ComparisonProduct) => (
            <div key={product.product_id} className="flex gap-2 rounded-xl border border-(--line) bg-(--card) p-2">
              <div className="h-14 w-14 shrink-0 overflow-hidden rounded-lg">
                <ProductImage productId={product.product_id} imageUrl={null} alt="" compact />
              </div>
              <div className="min-w-0 flex-1">
                <div className="line-clamp-2 text-[12.5px] font-medium leading-snug text-(--ink)" title={product.title ?? undefined}>
                  {product.title ?? product.product_id}
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-1">
                  <span className="font-mono text-[10.5px] text-(--ink-soft)">{product.product_id}</span>
                  <Pill tone={roleTone(product.role_status)}>{statusLabel(product.role_status)}</Pill>
                  {product.product_type ? <Pill tone="muted">{humanize(product.product_type)}</Pill> : null}
                </div>
                <div className="mt-0.5 flex items-center gap-2 text-[10.5px] text-(--ink-soft)">
                  {product.unknown_fields.length ? <span>{product.unknown_fields.length} UNKNOWN</span> : <span>no UNKNOWN</span>}
                  {onOpenDetails ? (
                    <button
                      type="button"
                      onClick={() => onOpenDetails(product.product_id)}
                      className="font-semibold text-(--ink-2) underline underline-offset-2 hover:text-(--accent-ink)"
                    >
                      details
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {dimensions.length ? (
        <div>
          <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Dimensions</h4>
          <div className="panel-scroll overflow-x-auto rounded-xl border border-(--line)">
            <table className="w-full min-w-[520px] border-collapse text-left">
              <thead>
                <tr className="border-b border-(--line) bg-(--well)/70">
                  <th className="px-2 py-1.5 text-[10.5px] font-bold uppercase tracking-wide text-(--ink-soft)">Attribute</th>
                  {products.map((product) => (
                    <th key={product.product_id} className="px-2 py-1.5 text-[10.5px] font-bold text-(--ink-soft)">
                      <span className="font-mono">{product.product_id}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dimensions.map((dimension) => (
                  <tr key={dimension.attribute} className="border-b border-(--line) last:border-b-0 align-top">
                    <th scope="row" className="whitespace-nowrap px-2 py-1.5 text-[11.5px] font-medium text-(--ink)">
                      {attributeLabel(dimension.attribute)}
                    </th>
                    {dimension.cells.map((cell) => (
                      <td key={`${dimension.attribute}-${cell.product_id}`} className="px-2 py-1.5">
                        <CellValue cell={cell} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p className="text-[12px] text-(--ink-soft)">P16 returned no comparison dimensions.</p>
      )}

      <div>
        <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Preference relations</h4>
        {comparison.preference_relations.length ? (
          <ul className="space-y-1.5">
            {comparison.preference_relations.map((relation) => (
              <PreferenceRelationRow key={`${relation.memory_id}-${relation.value}`} relation={relation} />
            ))}
          </ul>
        ) : (
          <p className="text-[12px] text-(--ink-soft)">No preference relations were returned.</p>
        )}
      </div>

      <div>
        <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Conflicts</h4>
        {comparison.conflicts.length ? (
          <ul className="space-y-1.5">
            {comparison.conflicts.map((conflict, index) => (
              <ConflictRow key={index} conflict={conflict} index={index} />
            ))}
          </ul>
        ) : (
          <p className="rounded-lg border border-(--line) bg-(--card) px-2 py-1.5 text-[12px] text-(--ink-soft)">
            No conflicts reported by P16.
          </p>
        )}
      </div>

      {comparison.unknown_fields.length ? (
        <div>
          <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">UNKNOWN fields</h4>
          <ul className="space-y-1">
            {comparison.unknown_fields.map((field) => (
              <li key={`${field.attribute}-${field.products.join(",")}`} className="flex flex-wrap items-center gap-1.5 text-[11.5px]">
                <span className="rounded-full bg-(--warn-soft) px-1.5 py-px text-[10.5px] font-medium text-(--warn)">
                  {attributeLabel(field.attribute)}
                </span>
                <span className="font-mono text-[10.5px] text-(--ink-soft)">{field.products.join(" · ")}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {comparison.claims_summary ? (
        <div>
          <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Claims summary</h4>
          <div className="grid grid-cols-3 gap-1.5">
            {[
              { label: "Valid", value: comparison.claims_summary.valid, tone: "text-(--ok)" },
              { label: "Rejected", value: comparison.claims_summary.rejected, tone: "text-(--danger)" },
              { label: "Unknown", value: comparison.claims_summary.unknown, tone: "text-(--warn)" },
            ].map((item) => (
              <div key={item.label} className="rounded-lg bg-(--well)/70 px-2 py-1.5 text-center">
                <div className="text-[10.5px] font-medium text-(--ink-soft)">{item.label}</div>
                <div className={`text-[16px] font-semibold tabular-nums ${item.tone}`}>
                  {item.value == null ? "—" : item.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div>
        <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Conditional recommendation</h4>
        <div
          className={`rounded-xl border p-2.5 text-[12.5px] leading-relaxed ${
            comparison.conditional_recommendation?.supported
              ? "border-(--ok)/40 bg-(--ok-soft)/50 text-(--ink)"
              : "border-(--line) bg-(--card) text-(--ink-soft)"
          }`}
          data-testid="conditional-recommendation"
        >
          <div className="mb-0.5 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-wide text-(--ink-soft)">
            <Icon name="info" size={11} />
            supported:{" "}
            {comparison.conditional_recommendation?.supported == null
              ? "not stated"
              : comparison.conditional_recommendation.supported
                ? "true"
                : "false"}
          </div>
          {comparison.conditional_recommendation?.text ? (
            <p className="whitespace-pre-wrap text-(--ink)">{comparison.conditional_recommendation.text}</p>
          ) : (
            <p>No conditional recommendation text was returned by P16.</p>
          )}
        </div>
      </div>

      {comparison.final_answer ? (
        <div>
          <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">P16 final answer</h4>
          <p className="whitespace-pre-wrap rounded-xl border border-(--line) bg-(--card) p-2.5 text-[12.5px] leading-relaxed text-(--ink)">
            {comparison.final_answer}
          </p>
        </div>
      ) : null}

      {comparison.evidence_sources.length ? (
        <div>
          <h4 className="mb-1 text-[11px] font-bold uppercase tracking-[0.08em] text-(--ink-soft)">Evidence sources</h4>
          <ul className="space-y-0.5">
            {comparison.evidence_sources.map((source) => (
              <li key={source} className="truncate font-mono text-[10.5px] text-(--ink-2)" title={source}>
                {source}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {comparison.notes.length ? (
        <ul className="list-disc space-y-0.5 pl-4 text-[10.5px] leading-snug text-(--ink-faint)">
          {comparison.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}

      <p className="border-t border-(--line) pt-1.5 text-[10.5px] leading-snug text-(--ink-faint)">
        P16 comparison output rendered verbatim; this UI does not recompute MATCH / VIOLATION / UNKNOWN, and makes no
        real-world quality claim beyond the conditional recommendation.
      </p>
    </div>
  );
}

/** A compact comparison card in the transcript; the full matrix lives in the right panel. */
export function ComparisonSummaryCard({
  comparison,
  onOpen,
}: {
  comparison: ComparisonPayload;
  onOpen: () => void;
}) {
  return (
    <div className="rounded-2xl border border-(--line) bg-(--card) p-3.5 shadow-(--shadow-sm)" data-testid="comparison-card">
      <div className="flex flex-wrap items-center gap-2">
        <Pill tone={comparison.status === "comparison_ready" ? "ok" : "warn"} dot>
          {statusLabel(comparison.status)}
        </Pill>
        <span className="text-[12.5px] font-semibold text-(--ink)">Comparison ready</span>
        <span className="text-[11.5px] text-(--ink-soft)">
          {comparison.products.map((product) => product.product_id).join(" vs ")}
        </span>
        <button
          type="button"
          onClick={onOpen}
          className="ml-auto inline-flex items-center gap-1 rounded-lg border border-(--line-strong) bg-(--card) px-2 py-[4px] text-[11.5px] font-semibold text-(--ink-2) transition-colors hover:border-(--accent) hover:bg-(--accent-soft) hover:text-(--accent-ink)"
        >
          Open comparison panel
          <Icon name="arrow-right" size={12} />
        </button>
      </div>
      {comparison.conditional_recommendation?.text ? (
        <p className="mt-1.5 line-clamp-3 whitespace-pre-wrap text-[12.5px] leading-relaxed text-(--ink)">
          {comparison.conditional_recommendation.text}
        </p>
      ) : null}
      <p className="mt-1.5 text-[10.5px] text-(--ink-faint)">
        Dimensions, preference relations, conflicts and UNKNOWN fields are in the comparison panel.
      </p>
    </div>
  );
}
