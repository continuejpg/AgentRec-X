// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): the storefront ProductCarousel / product grid is
// replaced by a ProductsBlock for the BFF ProductsPayload — real query metadata, verbatim Gate
// decision, catalogue cards and comparison selection; no carousel of mock products.

"use client";

import type { GateDecision, ProductCard as ProductCardModel, ProductsPayload } from "@/lib/types";
import { attributeLabel, formatNumber, humanize, plural, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Button, Pill, type Tone } from "./ui";
import ProductCard from "./ProductCard";

export function gateTone(action: string | null | undefined): Tone {
  switch (action) {
    case "ALLOW_FINALIZE":
      return "ok";
    case "NEED_USER_CLARIFICATION":
    case "ASK_CLARIFICATION":
      return "warn";
    case "REFUSE":
    case "BLOCK":
    case "NO_VERIFIED_CANDIDATE":
      return "danger";
    default:
      return "muted";
  }
}

function GateCard({ gate }: { gate: GateDecision }) {
  return (
    <div
      className="rounded-xl border border-(--line) bg-(--ground)/60 p-3"
      data-testid="gate-card"
      data-gate-action={gate.action}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-(--ink-soft)">Decision Gate</span>
        <Pill tone={gateTone(gate.action)} dot>
          {humanize(gate.action)}
        </Pill>
        <span className="font-mono text-[10.5px] text-(--ink-faint)" title="reason_code">
          {gate.reason_code}
        </span>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-(--ink-2)">{gate.reason ?? ""}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-(--ink-soft)">
        {gate.selected_candidate_id ? (
          <span>
            Selected: <span className="font-mono text-(--ink-2)">{gate.selected_candidate_id}</span>
          </span>
        ) : (
          <span>Selected: none</span>
        )}
        {gate.product_role ? <span>Role: {humanize(gate.product_role)}</span> : null}
        {gate.candidate_presentable_ids.length ? (
          <span title={gate.candidate_presentable_ids.join(", ")}>
            Presentable candidates: {formatNumber(gate.candidate_presentable_ids.length)}
          </span>
        ) : null}
      </div>
      {gate.unknown_fields.length ? (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          <span className="text-[11px] text-(--ink-soft)">Gate UNKNOWN:</span>
          {gate.unknown_fields.map((field) => (
            <span
              key={field}
              className="rounded-full bg-(--warn-soft) px-1.5 py-px text-[10.5px] font-medium text-(--warn)"
            >
              {attributeLabel(field)}
            </span>
          ))}
        </div>
      ) : null}
      <p className="mt-1.5 text-[10.5px] leading-snug text-(--ink-faint)">
        Passed through verbatim from the P16 Decision Gate; this UI does not recompute the action.
      </p>
    </div>
  );
}

export default function ProductsBlock({
  payload,
  selection,
  compareMin,
  compareMax,
  onToggleSelect,
  onViewDetails,
  onViewEvidence,
  onCompare,
}: {
  payload: ProductsPayload;
  selection: string[];
  compareMin: number;
  compareMax: number;
  onToggleSelect: (product: ProductCardModel) => void;
  onViewDetails: (product: ProductCardModel) => void;
  onViewEvidence: (product: ProductCardModel) => void;
  onCompare: (ids: string[]) => void;
}) {
  const recommended = payload.recommended_product_id
    ? payload.items.find((item) => item.product_id === payload.recommended_product_id) ?? null
    : null;
  const eligible = payload.gate?.candidate_presentable_ids ?? [];
  const selectedHere = payload.items.filter((item) => selection.includes(item.product_id));
  const canCompareEligible = eligible.length >= compareMin && eligible.length <= compareMax;
  const canCompareSelected = selectedHere.length >= compareMin;

  return (
    <section className="rounded-2xl border border-(--line) bg-(--card) p-3.5 shadow-(--shadow-sm) ac-reveal" data-testid="products-block">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-[14px] font-semibold text-(--ink)">
          {payload.items.length ? plural(payload.items.length, "catalogue result") : "No catalogue results"}
        </h3>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11.5px] text-(--ink-soft)">
          <span title="Search mode reported by the BFF">
            mode <span className="font-mono text-(--ink-2)">{payload.search_mode ?? "—"}</span>
          </span>
          <span aria-hidden>·</span>
          <span title="Score kind reported by the BFF">
            score <span className="font-mono text-(--ink-2)">{payload.score_kind ?? "—"}</span>
          </span>
          {payload.catalogue_size != null ? (
            <>
              <span aria-hidden>·</span>
              <span>catalogue {formatNumber(payload.catalogue_size)}</span>
            </>
          ) : null}
        </div>
      </div>
      <p className="mt-0.5 line-clamp-2 text-[12.5px] text-(--ink-2)" title={payload.query}>
        Query: <span className="italic">“{payload.query}”</span>
      </p>

      {payload.gate ? (
        <div className="mt-2.5">
          <GateCard gate={payload.gate} />
        </div>
      ) : null}

      {recommended ? (
        <div
          className="mt-2.5 flex flex-wrap items-center gap-2 rounded-xl border border-(--accent)/40 bg-(--accent-soft)/50 px-3 py-2"
          data-testid="gate-selected-item"
          data-product-id={recommended.product_id}
        >
          <Icon name="spark" size={15} className="text-(--accent-ink)" />
          <span className="text-[12px] font-semibold text-(--accent-ink)">Gate-selected item</span>
          <span className="min-w-0 flex-1 truncate text-[12px] text-(--ink-2)" title={recommended.title ?? undefined}>
            {recommended.title ?? recommended.product_id}
          </span>
          <button
            type="button"
            onClick={() => onViewEvidence(recommended)}
            className="shrink-0 rounded-lg border border-(--line-strong) bg-(--card) px-2 py-[4px] text-[11.5px] font-semibold text-(--ink-2) hover:border-(--accent) hover:text-(--accent-ink)"
          >
            View evidence
          </button>
        </div>
      ) : payload.gate?.selected_candidate_id ? (
        <p className="mt-2.5 rounded-lg bg-(--well)/70 px-3 py-2 text-[12px] text-(--ink-2)">
          The Gate selected <span className="font-mono">{payload.gate.selected_candidate_id}</span>, which is not in
          this enriched card list.
        </p>
      ) : null}

      {payload.items.length ? (
        <div className="mt-3 flex snap-x snap-mandatory gap-3 overflow-x-auto pb-1 sm:grid sm:snap-none sm:grid-cols-2 sm:overflow-visible">
          {payload.items.map((product) => {
            const selected = selection.includes(product.product_id);
            return (
              <div key={product.product_id} className="w-[286px] max-w-[85vw] shrink-0 snap-start sm:w-auto sm:max-w-none">
                <ProductCard
                  product={product}
                  selected={selected}
                  selectionOrder={selected ? selection.indexOf(product.product_id) + 1 : null}
                  selectedCount={selection.length}
                  compareMax={compareMax}
                  onToggleSelect={onToggleSelect}
                  onViewDetails={onViewDetails}
                  onViewEvidence={onViewEvidence}
                />
              </div>
            );
          })}
        </div>
      ) : (
        <p className="mt-3 rounded-xl border border-dashed border-(--line-strong) px-3 py-4 text-[12.5px] text-(--ink-soft)">
          The real catalogue search returned no items for this query. Nothing is fabricated to fill the space.
        </p>
      )}

      {payload.items.length ? (
        <div className="mt-2.5 flex flex-wrap items-center gap-2">
          {canCompareSelected ? (
            <Button size="sm" variant="primary" icon="compare" onClick={() => onCompare(selectedHere.map((item) => item.product_id))}>
              Compare selected ({selectedHere.length})
            </Button>
          ) : null}
          {canCompareEligible ? (
            <Button
              size="sm"
              variant="secondary"
              icon="compare"
              title="Uses gate.candidate_presentable_ids from the real payload"
              onClick={() => onCompare(eligible)}
            >
              Compare eligible candidates ({eligible.length})
            </Button>
          ) : null}
          {recommended ? (
            <Button size="sm" variant="secondary" onClick={() => onViewDetails(recommended)}>
              Open details
            </Button>
          ) : null}
          <span className="text-[10.5px] text-(--ink-faint)">
            Select {compareMin}–{compareMax} · selection stays in the comparison bar
          </span>
        </div>
      ) : null}

      <p className="mt-2 text-[10.5px] leading-snug text-(--ink-faint)">
        Backend order preserved; this UI does not re-rank. Catalogue facts only — UNKNOWN fields stay UNKNOWN and
        no price, stock or rating claim is made.
      </p>
    </section>
  );
}
