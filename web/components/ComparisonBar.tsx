// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { Icon } from "./icons";
import { Button, Pill } from "./ui";

export default function ComparisonBar({
  selection,
  compareMin,
  compareMax,
  loading,
  selectionError,
  onRemove,
  onClear,
  onCompare,
}: {
  selection: string[];
  compareMin: number;
  compareMax: number;
  loading: boolean;
  selectionError: string | null;
  onRemove: (productId: string) => void;
  onClear: () => void;
  onCompare: () => void;
}) {
  if (!selection.length) return null;
  const valid = selection.length >= compareMin && selection.length <= compareMax;
  return (
    <div
      className="mb-2 flex flex-wrap items-center gap-1.5 rounded-xl border border-(--line-strong) bg-(--card) px-2.5 py-1.5 shadow-(--shadow-sm) ac-reveal"
      data-testid="comparison-bar"
      aria-live="polite"
    >
      <span className="flex items-center gap-1 text-[11.5px] font-semibold text-(--ink)">
        <Icon name="compare" size={13} />
        Compare
      </span>
      <Pill tone={valid ? "ok" : "muted"}>
        {selection.length}/{compareMax}
      </Pill>
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
        {selection.map((productId) => (
          <span
            key={productId}
            className="inline-flex items-center gap-1 rounded-full border border-(--line-strong) bg-(--ground) py-0.5 pl-2 pr-1 font-mono text-[10.5px] text-(--ink-2)"
          >
            {productId}
            <button
              type="button"
              onClick={() => onRemove(productId)}
              aria-label={`Remove ${productId} from comparison`}
              className="grid h-4 w-4 place-items-center rounded-full text-(--ink-soft) hover:bg-(--well) hover:text-(--ink)"
            >
              <Icon name="x" size={10} />
            </button>
          </span>
        ))}
      </div>
      {!valid ? (
        <span className="text-[10.5px] font-medium text-(--warn)">
          Select {compareMin}–{compareMax}
        </span>
      ) : null}
      {selectionError ? (
        <span className="text-[10.5px] font-medium text-(--danger)" role="alert">
          {selectionError}
        </span>
      ) : null}
      <button
        type="button"
        onClick={onClear}
        className="text-[10.5px] font-semibold text-(--ink-soft) underline underline-offset-2 hover:text-(--ink)"
      >
        Clear
      </button>
      <Button size="sm" variant="primary" icon="compare" disabled={!valid || loading} onClick={onCompare}>
        {loading ? "Comparing…" : "Compare"}
      </Button>
    </div>
  );
}
