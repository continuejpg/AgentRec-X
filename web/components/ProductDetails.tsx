// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import type { ErrorPayload, PreferenceState, ProductCard, ProductDetailPayload } from "@/lib/types";
import { attributeLabel, humanize, namespaceOf, statusLabel } from "@/lib/format";
import { Icon } from "./icons";
import { Button, EmptyState, Pill, SectionTitle, Skeleton, type Tone } from "./ui";
import EvidenceList from "./EvidenceList";
import { ErrorBubble } from "./MessageBubble";
import { PriceBlock } from "./ProductCard";

function factTone(status: string | null | undefined): Tone {
  switch (status) {
    case "present":
      return "ok";
    case "unknown":
      return "warn";
    case "unsupported":
      return "muted";
    default:
      return "muted";
  }
}

function ActivePreferencesBlock({ preferences }: { preferences: PreferenceState | null }) {
  return (
    <section className="rounded-xl border border-(--info-soft)/80 bg-(--info-soft)/25 p-2.5" data-testid="details-active-preferences">
      <SectionTitle aside={<span className="font-mono text-[10px]">{preferences?.profile_id ?? ""}</span>}>
        Your active preferences
      </SectionTitle>
      {!preferences ? (
        <p className="text-[11.5px] text-(--ink-soft)">Preferences are not loaded yet.</p>
      ) : preferences.active.length ? (
        <ul className="space-y-1">
          {preferences.active.map((item) => (
            <li key={item.memory_id} className="flex flex-wrap items-baseline gap-1.5 text-[12px]">
              <Pill tone={item.polarity === "avoid" ? "danger" : "ok"}>{statusLabel(item.polarity)}</Pill>
              <span className="font-medium text-(--ink)">{item.value}</span>
              <span className="text-(--ink-soft)">· {statusLabel(item.kind)}</span>
              <span className="min-w-0 flex-1 truncate italic text-(--ink-soft)" title={item.source_text ?? undefined}>
                “{item.source_text}”
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[11.5px] text-(--ink-soft)">No active preferences returned by M9 for this profile.</p>
      )}
      <p className="mt-1.5 text-[10px] leading-snug text-(--ink-faint)">
        User memory from M9 — kept separate from catalogue product facts; it is not evidence about the product.
      </p>
    </section>
  );
}

function FactRow({ fact }: { fact: ProductDetailPayload["facts"][number] }) {
  return (
    <tr className="border-b border-(--line) align-top last:border-b-0">
      <th scope="row" className="whitespace-nowrap px-2 py-1.5 text-left text-[11.5px] font-medium text-(--ink)">
        {attributeLabel(fact.attribute)}
      </th>
      <td className="px-2 py-1.5">
        <Pill tone={factTone(fact.status)}>{statusLabel(fact.status)}</Pill>
      </td>
      <td className="px-2 py-1.5 text-[11.5px] leading-snug text-(--ink)">
        {fact.value != null && fact.value !== "" ? (
          <span className="whitespace-pre-wrap break-words">{fact.value}</span>
        ) : (
          <span className="text-(--ink-faint)">—</span>
        )}
        {fact.reason ? <div className="mt-0.5 text-[10.5px] leading-snug text-(--ink-soft)">{fact.reason}</div> : null}
      </td>
      <td className="px-2 py-1.5 font-mono text-[10px] leading-snug text-(--ink-soft)">
        {fact.source ? <div className="break-all">{fact.source}</div> : <span className="text-(--ink-faint)">—</span>}
        {fact.verification ? <div className="text-(--ink-faint)">{fact.verification}</div> : null}
      </td>
    </tr>
  );
}

function RawFieldSection({ field, values }: { field: string; values: string[] }) {
  return (
    <details className="group rounded-lg border border-(--line) bg-(--card)" open={field === "title"}>
      <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2 py-1.5 text-[11.5px] font-semibold text-(--ink-2) [&::-webkit-details-marker]:hidden">
        <Icon name="chevron-right" size={11} className="transition-transform group-open:rotate-90" />
        <span className="font-mono">{field}</span>
        <span className="font-normal text-(--ink-faint)">{values.length} record{values.length === 1 ? "" : "s"}</span>
      </summary>
      <ul className="space-y-1 border-t border-(--line) px-2 py-1.5">
        {values.map((value, index) => (
          <li key={`${field}-${index}`} className="panel-scroll max-h-28 overflow-auto whitespace-pre-wrap break-words rounded bg-(--well)/50 p-1.5 font-mono text-[10.5px] leading-relaxed text-(--ink-2)">
            {value}
          </li>
        ))}
      </ul>
    </details>
  );
}

export default function ProductDetails({
  productId: explicitProductId = null,
  product,
  detail,
  loading,
  error,
  onLoad,
  onRetry,
  preferences,
}: {
  productId?: string | null;
  product: ProductCard | null;
  detail: ProductDetailPayload | null;
  loading: boolean;
  error: ErrorPayload | null;
  onLoad: () => void;
  onRetry: () => void;
  preferences: PreferenceState | null;
}) {
  const productId = detail?.product_id ?? product?.product_id ?? explicitProductId ?? null;
  if (!productId) {
    return (
      <EmptyState
        icon="box"
        title="No product selected"
        hint="Open a product card with “View details” and its real catalogue record appears here — facts, raw fields, evidence and UNKNOWN, straight from the MCP get_product_info tool."
      />
    );
  }

  const title = detail?.title ?? product?.title ?? productId;
  const namespacedId = detail?.namespaced_id ?? product?.namespaced_id ?? "";
  const namespace = namespaceOf(namespacedId);
  const rawFields = detail?.raw_fields ?? {};
  const rawEntries = Object.entries(rawFields).filter((entry): entry is [string, string[]] => Array.isArray(entry[1]) && entry[1].length > 0);

  return (
    <div className="space-y-3" data-testid="product-details" data-product-id={productId}>
      <header className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          {detail?.product_type ?? product?.product_type ? (
            <Pill tone="muted">{humanize(detail?.product_type ?? product?.product_type ?? "")}</Pill>
          ) : null}
          {detail?.form_factor ?? product?.form_factor ? (
            <Pill tone="muted">{humanize(detail?.form_factor ?? product?.form_factor ?? "")}</Pill>
          ) : null}
          {detail?.product_role ?? product?.product_role ? (
            <Pill tone="violet">{humanize(detail?.product_role ?? product?.product_role ?? "")}</Pill>
          ) : null}
          <Pill tone={detail?.evidence_available ?? product?.evidence_available ? "info" : "muted"}>
            {detail?.evidence_available ?? product?.evidence_available ? "Evidence available" : "No evidence"}
          </Pill>
        </div>
        <h3 className="text-[14.5px] font-semibold leading-snug text-(--ink)" data-testid="details-title">
          {title}
        </h3>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10.5px] text-(--ink-soft)">
          <span>{productId}</span>
          {namespace ? <span className="text-(--ink-faint)">{namespace}</span> : null}
          {namespacedId ? (
            <button
              type="button"
              onClick={() => void navigator.clipboard?.writeText(namespacedId)}
              title={`Copy ${namespacedId}`}
              className="font-sans text-[10px] font-semibold text-(--ink-soft) underline underline-offset-2 hover:text-(--ink)"
            >
              copy namespaced id
            </button>
          ) : null}
        </div>
        {product ? <PriceBlock price={product.price} /> : null}
      </header>

      {loading ? (
        <div className="space-y-1.5">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : error ? (
        <div className="space-y-2">
          <ErrorBubble error={error} />
          <Button size="sm" variant="secondary" icon="refresh" onClick={onRetry}>
            Retry loading details
          </Button>
        </div>
      ) : !detail ? (
        <div className="space-y-2">
          <p className="rounded-lg border border-(--line) bg-(--card) px-2.5 py-2 text-[11.5px] leading-relaxed text-(--ink-soft)">
            Only the search card has been loaded so far. Load the full catalogue record from the real
            <span className="font-mono"> get_product_info </span> tool to see facts, raw fields and evidence spans.
          </p>
          <Button size="sm" variant="primary" icon="search" onClick={onLoad}>
            Load full catalogue record
          </Button>
        </div>
      ) : (
        <>
          <section>
            <SectionTitle aside="from get_product_info">Catalogue facts</SectionTitle>
            {detail.facts.length ? (
              <div className="panel-scroll overflow-x-auto rounded-xl border border-(--line)">
                <table className="w-full min-w-[430px] border-collapse text-left">
                  <thead>
                    <tr className="border-b border-(--line) bg-(--well)/70">
                      <th className="px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide text-(--ink-soft)">Attribute</th>
                      <th className="px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide text-(--ink-soft)">Status</th>
                      <th className="px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide text-(--ink-soft)">Value</th>
                      <th className="px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide text-(--ink-soft)">Source / verification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.facts.map((fact) => (
                      <FactRow key={fact.attribute} fact={fact} />
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-[11.5px] text-(--ink-soft)">The payload returned no facts.</p>
            )}
            {detail.notes.length ? (
              <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[10.5px] leading-snug text-(--ink-faint)">
                {detail.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
          </section>

          <section>
            <SectionTitle aside={`${rawEntries.length} raw fields`}>Raw catalogue fields</SectionTitle>
            {rawEntries.length ? (
              <div className="space-y-1">
                {rawEntries.map(([field, values]) => (
                  <RawFieldSection key={field} field={field} values={values} />
                ))}
              </div>
            ) : (
              <p className="text-[11.5px] text-(--ink-soft)">No raw fields were returned.</p>
            )}
          </section>

          <section>
            <SectionTitle>Evidence</SectionTitle>
            <EvidenceList
              evidence={detail.evidence}
              unknownFields={detail.unknown_fields}
              sourceRecord={detail.source_record}
              emptyHint="The catalogue record carried no evidence spans for this product."
            />
          </section>
        </>
      )}

      <ActivePreferencesBlock preferences={preferences} />

      {!detail && !loading && !error ? (
        <p className="font-mono text-[10px] text-(--ink-faint)" title={product?.source_record ?? undefined}>
          {product?.source_record ?? ""}
        </p>
      ) : null}
    </div>
  );
}
