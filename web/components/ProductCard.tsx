// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): ProductTile reworked for the AgentRec-X catalogue
// view model — neutral placeholder art, catalogue facts / UNKNOWN fields, fixed
// "Price unavailable / UNKNOWN" text, evidence availability, namespaced id and comparison
// selection replace the retail price / rating / stock / add-to-cart controls.

"use client";

import type { ProductAttribute, ProductCard as ProductCardModel, UnknownField } from "@/lib/types";
import { attributeLabel, humanize, namespaceOf, shorten, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Pill, type Tone } from "./ui";
import { ProductImage } from "./ProductImage";

/** A trailing parenthetical such as "(48-Pack)" is kept unbreakable so the clamp cuts before it. */
export function ProductTitle({ title, className = "" }: { title: string; className?: string }) {
  const match = /^(.*\S)\s+(\([^()]+\))$/.exec(title);
  return (
    <div className={className} title={title}>
      {match ? (
        <>
          {match[1]} <span className="whitespace-nowrap">{match[2]}</span>
        </>
      ) : (
        title
      )}
    </div>
  );
}

export function matchTone(matchStatus: string | null | undefined): Tone {
  switch (matchStatus) {
    case "recommended":
      return "accent";
    case "eligible_candidate":
      return "info";
    case "search_result":
      return "muted";
    case "UNKNOWN":
      return "warn";
    default:
      return "muted";
  }
}

const SOURCE_LABEL: Record<string, string> = {
  search: "Search hit",
  recommendation: "Recommendation",
};

function AttributeLine({ attribute }: { attribute: ProductAttribute }) {
  const present = attribute.status === "present";
  const isPresenceFlag = attribute.value == null || attribute.value === "" || attribute.value === "true";
  return (
    <li className="flex min-w-0 items-baseline gap-1.5 text-[12px] leading-snug">
      <span className="min-w-0 truncate font-medium text-(--ink)" title={attribute.span_text ?? undefined}>
        {attributeLabel(attribute.attribute)}
      </span>
      {isPresenceFlag ? null : (
        <span className="min-w-0 truncate text-(--ink-soft)" title={attribute.value ?? undefined}>
          {attribute.value}
        </span>
      )}
      {attribute.evidence_id ? (
        <Icon name="check" size={11} className="shrink-0 text-(--ok)" />
      ) : present ? null : (
        <span className="shrink-0 text-[10.5px] text-(--ink-faint)">{statusLabel(attribute.status)}</span>
      )}
    </li>
  );
}

function UnknownChips({ unknown }: { unknown: UnknownField[] }) {
  if (!unknown.length) return null;
  const shown = unknown.slice(0, 3);
  const rest = unknown.length - shown.length;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((field) => (
        <span
          key={field.attribute}
          title={field.reason}
          className="whitespace-nowrap rounded-full bg-(--warn-soft) px-1.5 py-px text-[10.5px] font-medium text-(--warn)"
        >
          UNKNOWN · {attributeLabel(field.attribute)}
        </span>
      ))}
      {rest > 0 ? (
        <span
          title={unknown.map((field) => `${field.attribute}: ${field.reason}`).join("\n")}
          className="whitespace-nowrap rounded-full bg-(--warn-soft) px-1.5 py-px text-[10.5px] font-medium text-(--warn)"
        >
          +{rest}
        </span>
      ) : null}
    </div>
  );
}

export function PriceBlock({ price }: { price: ProductCardModel["price"] }) {
  const display = price?.display?.trim() || "Price unavailable / UNKNOWN";
  return (
    <div
      className="rounded-lg border border-dashed border-(--line-strong) bg-(--well)/60 px-2 py-1"
      title={price?.reason || "Catalogue payload reports no numeric price."}
      data-testid="price-unknown"
    >
      <div className="flex items-center gap-1.5 text-[11.5px] font-semibold text-(--ink-2)">
        <Icon name="tag" size={12} className="text-(--ink-soft)" />
        <span className="truncate">{display}</span>
      </div>
      {price?.reason ? <div className="mt-0.5 line-clamp-2 text-[10.5px] leading-snug text-(--ink-soft)">{price.reason}</div> : null}
    </div>
  );
}

export default function ProductCard({
  product,
  selected,
  selectionOrder,
  selectedCount,
  compareMax,
  onToggleSelect,
  onViewDetails,
  onViewEvidence,
}: {
  product: ProductCardModel;
  selected: boolean;
  selectionOrder: number | null;
  selectedCount: number;
  compareMax: number;
  onToggleSelect: (product: ProductCardModel) => void;
  onViewDetails: (product: ProductCardModel) => void;
  onViewEvidence: (product: ProductCardModel) => void;
}) {
  const namespace = namespaceOf(product.namespaced_id);
  const title = product.title ?? product.product_id;
  const selectBlocked = !selected && selectedCount >= compareMax;
  return (
    <article
      data-testid="product-card"
      data-product-id={product.product_id}
      data-selected={selected ? "true" : "false"}
      className={`relative flex w-full flex-col overflow-hidden rounded-xl border bg-(--card) shadow-(--shadow-sm) transition-[box-shadow,border-color] duration-200 hover:shadow-md ${
        selected ? "border-(--ink) ring-1 ring-(--ink)/30" : "border-(--line)"
      }`}
    >
      <div className="relative h-[118px] w-full shrink-0">
        <button
          type="button"
          onClick={() => onViewDetails(product)}
          aria-label={`View details for ${title}`}
          className="block h-full w-full"
        >
          <ProductImage productId={product.product_id} imageUrl={product.image_url} alt="" />
        </button>
        <div className="pointer-events-none absolute left-1.5 top-1.5 flex flex-wrap gap-1">
          {product.recommended ? (
            <Pill tone="accent" className="shadow-(--shadow-sm)" title="recommended=true in the backend payload; not a quality claim">
              Recommended
            </Pill>
          ) : (
            <Pill tone={matchTone(product.match_status)} className="shadow-(--shadow-sm)">
              {humanize(product.match_status)}
            </Pill>
          )}
        </div>
        <div className="pointer-events-none absolute right-1.5 top-1.5">
          {product.evidence_available ? (
            <Pill tone="info" className="shadow-(--shadow-sm)">
              Evidence
            </Pill>
          ) : (
            <Pill tone="muted" className="shadow-(--shadow-sm)">
              No evidence
            </Pill>
          )}
        </div>
        {selected ? (
          <span className="absolute bottom-1.5 right-1.5 grid h-6 min-w-6 place-items-center rounded-full bg-(--ink) px-1.5 text-[11px] font-bold text-(--surface) shadow-(--shadow-sm)">
            {selectionOrder}
          </span>
        ) : null}
      </div>

      <div className="flex flex-1 flex-col gap-1.5 p-2.5">
        <button
          type="button"
          onClick={() => onViewDetails(product)}
          className="text-left"
          aria-label={`Open details for ${title}`}
        >
          <ProductTitle title={title} className="line-clamp-2 text-[13px] font-semibold leading-snug text-(--ink)" />
        </button>

        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11px] text-(--ink-soft)">
          <span className="font-mono text-[11px] text-(--ink-2)" title={product.product_id}>
            {product.product_id}
          </span>
          {namespace ? (
            <span className="truncate font-mono text-[10.5px] text-(--ink-faint)" title={product.namespaced_id ?? undefined}>
              {namespace}
            </span>
          ) : null}
        </div>

        <div className="flex flex-wrap gap-1">
          {product.product_type ? (
            <Pill tone="muted" title="product_type from the catalogue record">
              {humanize(product.product_type)}
            </Pill>
          ) : null}
          {product.form_factor ? <Pill tone="muted">{humanize(product.form_factor)}</Pill> : null}
          {product.product_role ? <Pill tone="violet">{humanize(product.product_role)}</Pill> : null}
          <Pill tone="info" title={`source: ${product.source}`}>
            {SOURCE_LABEL[product.source] ?? humanize(product.source)}
          </Pill>
        </div>

        <div className="line-clamp-1 text-[11px] text-(--ink-soft)" title={product.retrieval_source}>
          {product.retrieval_source}
        </div>

        {product.attributes.length ? (
          <ul className="space-y-0.5">
            {product.attributes.slice(0, 4).map((attribute) => (
              <AttributeLine key={`${attribute.attribute}-${attribute.value ?? attribute.status}`} attribute={attribute} />
            ))}
          </ul>
        ) : (
          <div className="text-[11.5px] text-(--ink-soft)">No verifiable attributes in this payload.</div>
        )}

        <UnknownChips unknown={product.unknown_fields} />
        <PriceBlock price={product.price} />

        <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
          <button
            type="button"
            onClick={() => onViewDetails(product)}
            className="inline-flex items-center gap-1 rounded-lg border border-(--line-strong) bg-(--card) px-2 py-[5px] text-[11.5px] font-semibold text-(--ink-2) transition-colors hover:border-(--accent) hover:bg-(--accent-soft) hover:text-(--accent-ink)"
          >
            View details
          </button>
          <button
            type="button"
            onClick={() => onViewEvidence(product)}
            className="inline-flex items-center gap-1 rounded-lg border border-(--line-strong) bg-(--card) px-2 py-[5px] text-[11.5px] font-semibold text-(--ink-2) transition-colors hover:border-(--accent) hover:bg-(--accent-soft) hover:text-(--accent-ink)"
          >
            View evidence
          </button>
          <button
            type="button"
            onClick={() => onToggleSelect(product)}
            disabled={selectBlocked}
            aria-pressed={selected}
            title={selectBlocked ? `At most ${compareMax} products can be compared (BFF limit).` : "Select for comparison"}
            className={`ml-auto inline-flex items-center gap-1 rounded-lg px-2 py-[5px] text-[11.5px] font-semibold transition ${
              selected
                ? "bg-(--ink) text-(--surface) hover:brightness-110"
                : "border border-(--line-strong) bg-(--card) text-(--ink-2) hover:border-(--accent) hover:bg-(--accent-soft) hover:text-(--accent-ink) disabled:pointer-events-none disabled:opacity-45"
            }`}
          >
            {selected ? <><Icon name="check" size={12} /> Selected</> : "Select"}
          </button>
        </div>

        <div className="truncate font-mono text-[10px] text-(--ink-faint)" title={product.source_record ?? undefined}>
          {product.source_record ? shorten(product.source_record, 72) : ""}
        </div>
      </div>
    </article>
  );
}
