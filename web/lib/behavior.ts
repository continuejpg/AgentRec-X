// AgentRec-X Web Demo (2026-11) - Behavior Feedback V1 client.
//
// First-party, whitelisted event tracking only:
//   * no PostHog/Snowplow SDK, no autocapture, no session recording;
//   * no raw chat text, no memory text, no user id;
//   * impressions require an IntersectionObserver visibility check, never the Top-K list;
//   * normal UI events are LIVE_USER_INTERACTION; a CDP-driven E2E browser can opt in to
//     AUTOMATED_BROWSER_E2E through window.__AGENTRECX_AUTOMATED_BROWSER_E2E__, so automated
//     mechanism evidence is never counted as human preference data.
//
// Events are batched and flushed to the local BFF.  A failed flush is retried; the BFF
// enforces event-id and recommendation+product impression idempotency server-side.

"use client";

import { useEffect, useRef, type RefObject } from "react";
import { api } from "./api";
import type { BehaviorEventInput, BehaviorEventSource, BehaviorEventType, ProductCard, ProductsPayload } from "./types";

const FLUSH_DELAY_MS = 450;
const FLUSH_BATCH_SIZE = 25;
const RETRY_DELAY_MS = 3_000;
const MAX_QUEUE = 120;
const CLICK_DEDUPE_MS = 1_200;
const COMPARE_DEDUPE_MS = 2_000;
const DEFAULT_IMPRESSION_MS = 1_000;
const DEFAULT_VISIBILITY_THRESHOLD = 0.6;

/** Only this source is produced by a normal human browser session. */
export const LIVE_EVENT_SOURCE = "LIVE_USER_INTERACTION" as const;

/** CDP E2E may set this flag before interacting; it is never set by the product UI. */
export function browserEventSource(): BehaviorEventSource {
  if (
    typeof window !== "undefined" &&
    (window as unknown as { __AGENTRECX_AUTOMATED_BROWSER_E2E__?: boolean })
      .__AGENTRECX_AUTOMATED_BROWSER_E2E__ === true
  ) {
    return "AUTOMATED_BROWSER_E2E";
  }
  return LIVE_EVENT_SOURCE;
}

export interface BehaviorFlushInfo {
  accepted: number;
  duplicates: number;
  rejected: number;
}

export interface CompareHandle {
  compare_id: string;
  recommendation_id: string;
  product_id: string;
  position: number;
  compared_product_ids: string[];
}

type BehaviorListener = (info: BehaviorFlushInfo) => void;

function newEventId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `evt-${crypto.randomUUID()}`;
  }
  return `evt-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function newCompareId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `cmp-${crypto.randomUUID()}`;
  }
  return `cmp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function eventBase(
  type: BehaviorEventType,
  recommendationId: string,
  productId: string,
  position: number,
): BehaviorEventInput {
  return {
    event_id: newEventId(),
    event_type: type,
    recommendation_id: recommendationId,
    product_id: productId,
    displayed_position: Math.max(1, Math.round(position)),
    event_source: browserEventSource(),
  };
}

class BehaviorTracker {
  private sessionId: string | null = null;
  private enabled = false;
  private minVisibleMs = DEFAULT_IMPRESSION_MS;
  private visibilityThreshold = DEFAULT_VISIBILITY_THRESHOLD;
  private queue: BehaviorEventInput[] = [];
  private timer: number | null = null;
  private flushing = false;
  private readonly seenImpressions = new Set<string>();
  private readonly lastClickAt = new Map<string, number>();
  private readonly lastCompareAt = new Map<string, number>();
  private readonly listeners = new Set<BehaviorListener>();
  private lastError: string | null = null;
  private lastFlushAt: string | null = null;

  constructor() {
    if (typeof window === "undefined") return;
    const flush = (keepalive: boolean) => {
      void this.flush(keepalive);
    };
    window.addEventListener("pagehide", () => flush(true));
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") flush(true);
    });
  }

  configure({
    sessionId,
    enabled,
    minVisibleMs,
    visibilityThreshold,
  }: {
    sessionId: string | null;
    enabled: boolean;
    minVisibleMs?: number;
    visibilityThreshold?: number;
  }): void {
    if (sessionId !== this.sessionId) {
      // Give the old session one last asynchronous chance to receive queued events.
      // flush() captured the old session id before this assignment; a failed post after
      // the switch is dropped rather than attributed to the new session.
      if (this.sessionId && this.enabled && this.queue.length) {
        void this.flush(false);
      }
      this.sessionId = sessionId;
      this.queue = [];
      this.seenImpressions.clear();
      this.lastClickAt.clear();
      this.lastCompareAt.clear();
      this.lastError = null;
      this.lastFlushAt = null;
      if (this.timer != null) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
    }
    this.enabled = Boolean(sessionId && enabled);
    if (typeof minVisibleMs === "number" && minVisibleMs > 0) this.minVisibleMs = minVisibleMs;
    if (typeof visibilityThreshold === "number" && visibilityThreshold > 0) {
      this.visibilityThreshold = Math.min(1, visibilityThreshold);
    }
    if (!this.enabled) {
      this.queue = [];
      if (this.timer != null) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = Boolean(this.sessionId && enabled);
    if (!this.enabled) {
      this.queue = [];
      if (this.timer != null) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
    }
  }

  get minimumVisibleMs(): number {
    return this.minVisibleMs;
  }

  get threshold(): number {
    return this.visibilityThreshold;
  }

  get pendingCount(): number {
    return this.queue.length;
  }

  get lastFlushError(): string | null {
    return this.lastError;
  }

  get lastFlushTimestamp(): string | null {
    return this.lastFlushAt;
  }

  subscribe(listener: BehaviorListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  clearLocal(): void {
    this.queue = [];
    this.seenImpressions.clear();
    this.lastClickAt.clear();
    this.lastCompareAt.clear();
    this.lastError = null;
    if (this.timer != null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  // -- public event helpers ------------------------------------------------
  impression(
    product: Pick<ProductCard, "product_id">,
    recommendationId: string | null | undefined,
    position: number | undefined,
  ): boolean {
    const rec = String(recommendationId ?? "").trim();
    if (!this.enabled || !rec || !product.product_id || !position) return false;
    const key = `${rec}|${product.product_id}`;
    if (this.seenImpressions.has(key)) return false;
    this.seenImpressions.add(key);
    return this.enqueue(
      eventBase("product_impression", rec, product.product_id, position),
    );
  }

  click(product: ProductCard, recommendationId: string | null | undefined, position: number | undefined): boolean {
    const rec = String(recommendationId ?? "").trim();
    if (!this.enabled || !rec || !product.product_id || !position) return false;
    const key = `${rec}|${product.product_id}`;
    const now = Date.now();
    if (now - (this.lastClickAt.get(key) ?? 0) < CLICK_DEDUPE_MS) return false;
    this.lastClickAt.set(key, now);
    return this.enqueue(eventBase("product_click", rec, product.product_id, position));
  }

  compareAttempt(
    payload: ProductsPayload,
    products: { product: ProductCard; position: number }[],
  ): CompareHandle | null {
    const rec = String(payload.recommendation_id ?? "").trim();
    if (!this.enabled || !rec || products.length < 2) return null;
    const ids = products.map((entry) => entry.product.product_id).filter(Boolean);
    if (new Set(ids).size < 2 || !products[0]?.position) return null;
    const key = `${rec}|${[...ids].sort().join("|")}`;
    const now = Date.now();
    if (now - (this.lastCompareAt.get(key) ?? 0) < COMPARE_DEDUPE_MS) return null;
    this.lastCompareAt.set(key, now);
    const compareId = newCompareId();
    const event = eventBase("product_compare", rec, products[0].product.product_id, products[0].position);
    event.metadata = {
      compared_product_ids: ids,
      action_status: "attempted",
      compare_id: compareId,
    };
    if (!this.enqueue(event)) return null;
    return {
      compare_id: compareId,
      recommendation_id: rec,
      product_id: products[0].product.product_id,
      position: products[0].position,
      compared_product_ids: ids,
    };
  }

  compareResult(
    handle: CompareHandle,
    status: "succeeded" | "failed",
    failure?: { code?: string | null; message?: string | null },
  ): boolean {
    if (!this.enabled || !this.sessionId) return false;
    const event = eventBase(
      "product_compare",
      handle.recommendation_id,
      handle.product_id,
      handle.position,
    );
    event.metadata = {
      compared_product_ids: handle.compared_product_ids,
      action_status: status,
      compare_id: handle.compare_id,
    };
    if (status === "failed") {
      event.metadata.failure_code = String(failure?.code || "compare_failed").slice(0, 120);
      event.metadata.failure_message = String(failure?.message || "comparison failed").slice(0, 400);
    }
    return this.enqueue(event);
  }

  feedback(
    product: ProductCard,
    recommendationId: string | null | undefined,
    position: number | undefined,
    feedback: "like" | "dislike",
  ): boolean {
    const rec = String(recommendationId ?? "").trim();
    if (!this.enabled || !rec || !product.product_id || !position) return false;
    const event = eventBase("explicit_item_feedback", rec, product.product_id, position);
    event.metadata = { feedback };
    return this.enqueue(event);
  }

  // -- queue/flush ---------------------------------------------------------
  private enqueue(event: BehaviorEventInput): boolean {
    if (!this.enabled || !this.sessionId) return false;
    if (this.queue.length >= MAX_QUEUE) this.queue.shift();
    this.queue.push(event);
    this.schedule(FLUSH_DELAY_MS);
    return true;
  }

  private schedule(delayMs: number): void {
    if (this.timer != null || !this.enabled) return;
    this.timer = window.setTimeout(() => {
      this.timer = null;
      void this.flush(false);
    }, Math.max(0, delayMs));
  }

  async flush(keepalive = false): Promise<void> {
    if (this.flushing || !this.enabled || !this.sessionId || this.queue.length === 0) return;
    this.flushing = true;
    const sessionId = this.sessionId;
    const batch = this.queue.splice(0, FLUSH_BATCH_SIZE);
    try {
      const response = await api.behaviorEvents(sessionId, batch, { keepalive });
      if (this.sessionId !== sessionId) return;
      this.lastError = null;
      this.lastFlushAt = response.accepted.length ? new Date().toISOString() : this.lastFlushAt;
      for (const listener of this.listeners) {
        try {
          listener({
            accepted: response.accepted.length,
            duplicates: response.duplicates.length,
            rejected: response.rejected.length,
          });
        } catch {
          // A UI listener must never break tracking.
        }
      }
    } catch (error) {
      if (this.sessionId !== sessionId) return;
      this.lastError = error instanceof Error ? error.message : String(error);
      // Put the batch back (bounded) and retry later; server-side idempotency makes this safe.
      this.queue = [...batch, ...this.queue].slice(-MAX_QUEUE);
    } finally {
      this.flushing = false;
      if (this.enabled && this.queue.length) {
        this.schedule(this.lastError ? RETRY_DELAY_MS : FLUSH_DELAY_MS);
      }
    }
  }
}

export const behaviorTracker = new BehaviorTracker();

/** Observe a product card's real browser visibility before recording an impression. */
export function useProductImpression({
  productId,
  recommendationId,
  position,
}: {
  productId: string;
  recommendationId: string | null | undefined;
  position: number | undefined;
}): RefObject<HTMLDivElement | null> {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = ref.current;
    const rec = String(recommendationId ?? "").trim();
    if (!node || !rec || !position) return;
    if (typeof IntersectionObserver === "undefined") return;
    let timer: number | null = null;
    let reported = false;
    const threshold = behaviorTracker.threshold;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const visible = entry.isIntersecting && entry.intersectionRatio >= threshold;
          if (visible && timer == null && !reported) {
            timer = window.setTimeout(() => {
              timer = null;
              if (reported) return;
              reported = true;
              behaviorTracker.impression({ product_id: productId }, rec, position);
            }, behaviorTracker.minimumVisibleMs);
          } else if (!visible && timer != null) {
            window.clearTimeout(timer);
            timer = null;
          }
        }
      },
      { threshold: [0, 0.25, threshold, 0.75, 1] },
    );
    observer.observe(node);
    return () => {
      observer.disconnect();
      if (timer != null) window.clearTimeout(timer);
    };
  }, [productId, recommendationId, position]);
  return ref;
}

export function behaviorEventLabel(
  type: BehaviorEventType,
  status?: "attempted" | "succeeded" | "failed" | null,
): string {
  switch (type) {
    case "product_impression":
      return "Impression";
    case "product_click":
      return "Click";
    case "product_compare":
      if (status === "attempted") return "Compare attempt";
      if (status === "succeeded") return "Compare completed";
      if (status === "failed") return "Compare failed";
      return "Compare";
    case "explicit_item_feedback":
      return "Like / dislike";
    default:
      return type;
  }
}
