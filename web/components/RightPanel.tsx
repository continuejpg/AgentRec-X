// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import type {
  ComparisonPayload,
  ErrorPayload,
  EvidencePayload,
  PreferenceState,
  ProductCard,
  ProductDetailPayload,
} from "@/lib/types";
import { formatLatency, humanize, plural } from "@/lib/format";
import { Icon } from "./icons";
import { Button, EmptyState, IconButton, Pill, SectionTitle, type Tone } from "./ui";
import ComparisonView from "./ComparisonView";
import EvidenceList from "./EvidenceList";
import ProductDetails from "./ProductDetails";
import { ErrorBubble } from "./MessageBubble";
import { ProductImage } from "./ProductImage";

export type PanelTab = "details" | "evidence" | "comparison";

const TABS: { id: PanelTab; label: string; icon: "box" | "search" | "compare" }[] = [
  { id: "details", label: "Details", icon: "box" },
  { id: "evidence", label: "Evidence", icon: "search" },
  { id: "comparison", label: "Comparison", icon: "compare" },
];

function selectionTone(count: number, min: number, max: number): Tone {
  if (count < min) return "muted";
  if (count > max) return "danger";
  return "ok";
}

export interface RightPanelProps {
  tab: PanelTab;
  onTabChange: (tab: PanelTab) => void;
  variant: "column" | "sheet";
  onClose: () => void;

  selectedProduct: ProductCard | null;
  selectedProductId: string | null;
  detail: ProductDetailPayload | null;
  detailLoading: boolean;
  detailError: ErrorPayload | null;
  onLoadDetail: () => void;
  onRetryDetail: () => void;

  evidencePayload: EvidencePayload | null;

  comparison: ComparisonPayload | null;
  comparisonLoading: boolean;
  comparisonError: ErrorPayload | null;
  comparisonLatencyMs: number | null;
  onRetryCompare: () => void;

  selection: string[];
  compareMin: number;
  compareMax: number;
  selectionError: string | null;
  verifiedIds: string[];
  onRemoveSelection: (productId: string) => void;
  onClearSelection: () => void;
  onCompare: (ids: string[]) => void;
  onCompareVerified: () => void;

  preferences: PreferenceState | null;
  onOpenProductDetails: (productId: string) => void;
}

function CompareTab(props: RightPanelProps) {
  const {
    comparison,
    comparisonLoading,
    comparisonError,
    comparisonLatencyMs,
    onRetryCompare,
    selection,
    compareMin,
    compareMax,
    selectionError,
    verifiedIds,
    onRemoveSelection,
    onClearSelection,
    onCompare,
    onCompareVerified,
    onOpenProductDetails,
  } = props;
  const validSelection = selection.length >= compareMin && selection.length <= compareMax;
  const verifiedPair = verifiedIds.length >= compareMin ? verifiedIds.slice(0, Math.min(compareMax, verifiedIds.length)) : [];

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-(--line) bg-(--card)/70 p-2.5" data-testid="comparison-selection">
        <div className="flex items-center gap-2">
          <span className="text-[11.5px] font-semibold text-(--ink)">Comparison selection</span>
          <Pill tone={selectionTone(selection.length, compareMin, compareMax)}>
            {selection.length}/{compareMax}
          </Pill>
          {selection.length ? (
            <button
              type="button"
              onClick={onClearSelection}
              className="ml-auto text-[10.5px] font-semibold text-(--ink-soft) underline underline-offset-2 hover:text-(--ink)"
            >
              Clear
            </button>
          ) : null}
        </div>
        {selection.length ? (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {selection.map((productId, index) => (
              <span
                key={productId}
                className="inline-flex items-center gap-1 rounded-full border border-(--line-strong) bg-(--card) py-0.5 pl-2 pr-1 font-mono text-[10.5px] text-(--ink-2)"
              >
                <span className="text-(--ink-faint)">{index + 1}</span>
                {productId}
                <button
                  type="button"
                  onClick={() => onRemoveSelection(productId)}
                  aria-label={`Remove ${productId} from comparison`}
                  className="grid h-4 w-4 place-items-center rounded-full text-(--ink-soft) hover:bg-(--well) hover:text-(--ink)"
                >
                  <Icon name="x" size={10} />
                </button>
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-[11.5px] leading-snug text-(--ink-soft)">
            Select {compareMin}–{compareMax} products from the result cards (or use the verified demo pair below), then
            run the real P16 comparison.
          </p>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="primary"
            icon="compare"
            data-testid="compare-run"
            disabled={!validSelection || comparisonLoading}
            onClick={() => onCompare(selection)}
          >
            {comparisonLoading ? "Comparing…" : `Compare ${selection.length || ""}`.trim()}
          </Button>
          {verifiedPair.length ? (
            <Button size="sm" variant="secondary" icon="compare" data-testid="compare-verified" onClick={onCompareVerified} disabled={comparisonLoading}>
              Verified demo pair
            </Button>
          ) : null}
        </div>
        {verifiedPair.length ? (
          <p className="mt-1 font-mono text-[10px] text-(--ink-faint)">
            verified_comparison_ids: {verifiedPair.join(", ")} (from /api/demo/config)
          </p>
        ) : null}
        {!validSelection && selection.length ? (
          <p className="mt-1 text-[11px] font-medium text-(--warn)" data-testid="compare-selection-hint">
            Select {compareMin}–{compareMax} products; the BFF rejects any other count.
          </p>
        ) : null}
        {selectionError ? (
          <p className="mt-1 text-[11px] font-medium text-(--danger)" role="alert">
            {selectionError}
          </p>
        ) : null}
      </div>

      {comparisonLoading ? (
        <div className="overflow-hidden rounded-xl border border-(--line) bg-(--card)">
          <div className="h-1 w-full overflow-hidden bg-(--well)">
            <div className="ac-indeterminate h-full w-1/3 rounded-full bg-(--accent)" />
          </div>
          <div className="p-3">
            <div className="text-[12.5px] font-semibold text-(--ink)">Building the comparison…</div>
            <p className="mt-0.5 text-[11.5px] leading-relaxed text-(--ink-soft)">
              Calling the real <span className="font-mono">compare_products</span> MCP tool through the BFF. This can
              take a moment while the catalogue evidence is assembled.
            </p>
          </div>
        </div>
      ) : null}

      {comparisonError ? (
        <div className="space-y-2" data-testid="comparison-error">
          <ErrorBubble error={comparisonError} />
          <p className="text-[11px] leading-snug text-(--ink-soft)">
            The BFF returned this error for the requested product references; no fallback comparison is fabricated.
          </p>
          <Button size="sm" variant="secondary" icon="refresh" onClick={onRetryCompare}>
            Retry the same comparison
          </Button>
        </div>
      ) : null}

      {comparison ? (
        <ComparisonView comparison={comparison} latencyMs={comparisonLatencyMs} onOpenDetails={onOpenProductDetails} />
      ) : !comparisonLoading && !comparisonError ? (
        <EmptyState
          icon="compare"
          compact
          title="No comparison yet"
          hint={`Select ${compareMin}–${compareMax} products and run the comparison; results include dimensions, preference relations, conflicts, UNKNOWN fields and the conditional recommendation.`}
        />
      ) : null}
    </div>
  );
}

export default function RightPanel(props: RightPanelProps) {
  const {
    tab,
    onTabChange,
    variant,
    onClose,
    selectedProduct,
    detail,
    detailLoading,
    detailError,
    onLoadDetail,
    onRetryDetail,
    evidencePayload,
    preferences,
  } = props;

  const evidenceItems = evidencePayload?.evidence ?? detail?.evidence ?? [];
  const evidenceUnknown = evidencePayload?.unknown_fields ?? detail?.unknown_fields ?? [];
  const evidenceSource = evidencePayload?.source_record ?? detail?.source_record ?? null;
  const evidenceTitle = selectedProduct?.title ?? detail?.title ?? null;

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="right-panel">
      <div className="flex items-center gap-1 border-b border-(--line) px-2 py-1.5">
        <div role="tablist" aria-label="Contextual panel" className="flex min-w-0 gap-0.5 rounded-lg bg-(--ground) p-[3px]">
          {TABS.map((entry) => {
            const active = entry.id === tab;
            return (
              <button
                key={entry.id}
                type="button"
                role="tab"
                aria-selected={active}
                data-testid={`panel-tab-${entry.id}`}
                onClick={() => onTabChange(entry.id)}
                className={`flex items-center gap-1 whitespace-nowrap rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                  active ? "bg-(--card) font-semibold text-(--ink) shadow-(--shadow-sm)" : "font-medium text-(--ink-soft) hover:text-(--ink)"
                }`}
              >
                <Icon name={entry.icon} size={13} />
                {entry.label}
              </button>
            );
          })}
        </div>
        <IconButton
          icon={variant === "sheet" ? "x" : "panel-right"}
          label={variant === "sheet" ? "Close context panel" : "Hide context panel"}
          onClick={onClose}
          size={28}
          className="ml-auto"
        />
      </div>

      <div className="panel-scroll min-h-0 flex-1 overflow-y-auto p-3">
        {tab === "details" ? (
          <ProductDetails
            productId={props.selectedProductId}
            product={selectedProduct}
            detail={detail}
            loading={detailLoading}
            error={detailError}
            onLoad={onLoadDetail}
            onRetry={onRetryDetail}
            preferences={preferences}
          />
        ) : tab === "evidence" ? (
          evidenceItems.length || evidenceTitle ? (
            <div className="space-y-2" data-testid="evidence-panel">
              <div className="flex items-center gap-2">
                <h3 className="text-[12.5px] font-semibold text-(--ink)">Evidence for</h3>
                <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-(--ink-soft)" title={evidenceTitle ?? ""}>
                  {evidencePayload?.product_id ?? detail?.product_id ?? selectedProduct?.product_id ?? ""}
                </span>
              </div>
              {evidenceTitle ? (
                <p className="line-clamp-2 text-[11.5px] leading-snug text-(--ink-2)" title={evidenceTitle ?? undefined}>
                  {evidenceTitle}
                </p>
              ) : null}
              <EvidenceList evidence={evidenceItems} unknownFields={evidenceUnknown} sourceRecord={evidenceSource} />
            </div>
          ) : (
            <EmptyState
              icon="search"
              title="No evidence selected"
              hint="Evidence spans appear after a search/recommendation response or when you open “View evidence” on a product card."
            />
          )
        ) : (
          <CompareTab {...props} />
        )}
      </div>

      {tab === "comparison" && props.comparison ? (
        <div className="border-t border-(--line) px-3 py-1.5 text-[10px] text-(--ink-faint)">
          {plural(props.comparison.products.length, "product")} · P16 output ·{" "}
          {props.comparisonLatencyMs != null ? `round-trip ${formatLatency(props.comparisonLatencyMs)}` : humanize(props.comparison.status)}
        </div>
      ) : null}
    </div>
  );
}
