// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, BFF_BASE, errorPayload, isAbortError } from "@/lib/api";
import {
  finaliseStaleToolRows,
  mergeToolStatus,
  newTurnKey,
  turnFromView,
  type ClarificationReplyState,
  type LocalTurn,
} from "@/lib/chat";
import { resolveQwenLive } from "@/lib/modes";
import {
  mergePlanner,
  plannerFromDone,
  plannerFromProducts,
  plannerFromReplyResponse,
} from "@/lib/planner";
import type {
  ClarificationPayload,
  ComparisonPayload,
  DemoConfig,
  ErrorPayload,
  EvidencePayload,
  Health,
  PreferenceState,
  ProductCard,
  ProductDetailPayload,
  ReplayResponse,
  SessionSummary,
  SessionView,
} from "@/lib/types";
import { useIsDesktop } from "@/lib/useMediaQuery";
import ChatPanel from "./ChatPanel";
import Header, { type UiMode } from "./Header";
import ReplayPanel from "./ReplayPanel";
import RightPanel, { type PanelTab, type RightPanelProps } from "./RightPanel";
import Sidebar from "./Sidebar";
import { Button, Sheet } from "./ui";

/** Only the session id is persisted in localStorage (docs/WEB_DEMO_CONTRACT.md §5). */
const SESSION_STORAGE_KEY = "agentrecx.web.session_id";
const RECENT_LIMIT = 20;
const HEALTH_POLL_MS = 1_500;
const HEALTH_READY_POLL_MS = 8_000;
const CONFIG_RETRY_MS = 2_000;

function readStoredSessionId(): string | null {
  try {
    return window.localStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(sessionId: string): void {
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } catch {
    // Storage can be unavailable in private contexts; the in-memory session still works.
  }
}

function clearStoredSessionId(): void {
  try {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Ignore.
  }
}

function emptyLocalTurn(partial: Partial<LocalTurn>): LocalTurn {
  return {
    key: partial.key ?? newTurnKey(),
    turn_id: partial.turn_id ?? null,
    user_text: partial.user_text ?? "",
    assistant_text: partial.assistant_text ?? "",
    tool_activity: partial.tool_activity ?? [],
    products: partial.products ?? null,
    comparison: partial.comparison ?? null,
    clarification: partial.clarification ?? null,
    memory_update: partial.memory_update ?? null,
    error: partial.error ?? null,
    created_at: partial.created_at ?? new Date().toISOString(),
    pending: partial.pending ?? false,
    status: partial.status ?? "complete",
    stop_reason: partial.stop_reason ?? null,
    total_ms: partial.total_ms ?? null,
    planner: partial.planner ?? null,
    startedAt: partial.startedAt ?? 0,
    text_final: partial.text_final ?? true,
    reply: partial.reply ?? null,
  };
}

/** SessionView → browser turns, attaching a persisted pending clarification to the last turn. */
function turnsFromSession(view: SessionView): LocalTurn[] {
  const mapped = (view.turns ?? []).map((turn, index) => turnFromView(turn, index));
  const pending = view.pending_clarification ?? null;
  if (!pending) return mapped;
  for (let index = mapped.length - 1; index >= 0; index -= 1) {
    if (!mapped[index].clarification) {
      mapped[index] = { ...mapped[index], clarification: pending };
      return mapped;
    }
    if (mapped[index].clarification?.clarification_id === pending.clarification_id) return mapped;
    break;
  }
  mapped.push(emptyLocalTurn({ key: `restored-clarification-${pending.clarification_id}`, clarification: pending }));
  return mapped;
}

/** Real error → the turn error view model. */
function toErrorPayload(error: unknown): ErrorPayload {
  return errorPayload(error);
}

export default function DemoApp() {
  const isDesktop = useIsDesktop();

  // ---- connectivity / config -------------------------------------------------
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [healthNonce, setHealthNonce] = useState(0);
  const [clock, setClock] = useState(0);
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const configRef = useRef<DemoConfig | null>(null);
  configRef.current = config;

  // ---- session / turns -------------------------------------------------------
  const [session, setSession] = useState<SessionView | null>(null);
  const sessionRef = useRef<SessionView | null>(null);
  sessionRef.current = session;
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState<ErrorPayload | null>(null);
  const [turns, setTurns] = useState<LocalTurn[]>([]);
  const [recent, setRecent] = useState<SessionSummary[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const bootRef = useRef(false);
  /** A query to replay once a session created by the write-profile switch is active. */
  const [autoQuery, setAutoQuery] = useState<{ sessionId: string; text: string } | null>(null);

  // ---- preferences ----------------------------------------------------------
  const [preferences, setPreferences] = useState<PreferenceState | null>(null);
  const [prefsLoading, setPrefsLoading] = useState(false);
  const [prefsError, setPrefsError] = useState<ErrorPayload | null>(null);

  // ---- streaming ------------------------------------------------------------
  const [streamingKey, setStreamingKey] = useState<string | null>(null);
  const streamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const [lastHeartbeatAt, setLastHeartbeatAt] = useState<number | null>(null);

  // ---- panel ----------------------------------------------------------------
  const [panelTab, setPanelTab] = useState<PanelTab>("details");
  const [panelOpen, setPanelOpen] = useState(true);
  const [panelSheetOpen, setPanelSheetOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // ---- details / evidence ---------------------------------------------------
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, ProductDetailPayload>>({});
  const detailsRef = useRef<Record<string, ProductDetailPayload>>({});
  detailsRef.current = details;
  const [detailsLoading, setDetailsLoading] = useState<string | null>(null);
  const detailsLoadingRef = useRef<string | null>(null);
  const [detailsError, setDetailsError] = useState<{ productId: string; error: ErrorPayload } | null>(null);
  const [evidenceByProduct, setEvidenceByProduct] = useState<Record<string, EvidencePayload>>({});
  const evidenceRef = useRef<Record<string, EvidencePayload>>({});
  evidenceRef.current = evidenceByProduct;

  // ---- comparison -----------------------------------------------------------
  const [selection, setSelection] = useState<string[]>([]);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonPayload | null>(null);
  const [comparisonError, setComparisonError] = useState<ErrorPayload | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonLatency, setComparisonLatency] = useState<number | null>(null);
  const compareBusyRef = useRef(false);

  // ---- clarification --------------------------------------------------------
  const [pendingClarification, setPendingClarification] = useState<ClarificationPayload | null>(null);
  const replyBusyRef = useRef(false);

  // ---- mode / replay --------------------------------------------------------
  const [mode, setMode] = useState<UiMode>("cpu_live");
  const [replayOpen, setReplayOpen] = useState(false);
  const [replay, setReplay] = useState<ReplayResponse | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState<ErrorPayload | null>(null);
  const replayBusyRef = useRef(false);

  const compareMin = config?.limits.compare_min ?? 2;
  const compareMax = config?.limits.compare_max ?? 5;
  const busy = streamingKey != null;

  // Real Qwen Base Live availability from GET /api/health + GET /api/demo/config. Never inferred.
  const qwenLive = useMemo(() => resolveQwenLive(health, config), [health, config]);
  const qwenLiveAvailable = qwenLive.available;
  const qwenLiveAvailableRef = useRef(qwenLiveAvailable);
  qwenLiveAvailableRef.current = qwenLiveAvailable;

  // ---------------------------------------------------------------------------
  // Health polling (real /api/health; keeps retrying while the BFF starts)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let lastReady = false;
    const tick = async () => {
      try {
        const next = await api.health();
        if (cancelled) return;
        lastReady = next.ready;
        setHealth(next);
        setHealthError(null);
      } catch (error) {
        if (cancelled) return;
        lastReady = false;
        setHealth(null);
        setHealthError(toErrorPayload(error).message);
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(tick, lastReady ? HEALTH_READY_POLL_MS : HEALTH_POLL_MS);
        }
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [healthNonce]);

  const retryHealth = useCallback(() => setHealthNonce((value) => value + 1), []);

  // A local elapsed clock for the readiness banner (shown next to the real uptime).
  useEffect(() => {
    if (health?.ready) return;
    const timer = window.setInterval(() => setClock((value) => value + 1), 1_000);
    return () => window.clearInterval(timer);
  }, [health?.ready]);

  // ---------------------------------------------------------------------------
  // Config (profiles, limits, verified ids, modes, notices)
  // ---------------------------------------------------------------------------
  const reachable = health != null;
  useEffect(() => {
    if (!reachable || config) return;
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const next = await api.config();
        if (!cancelled) setConfig(next);
      } catch {
        if (!cancelled) timer = window.setTimeout(tick, CONFIG_RETRY_MS);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [reachable, config]);

  // ---------------------------------------------------------------------------
  // Recent sessions
  // ---------------------------------------------------------------------------
  const refreshRecent = useCallback(async () => {
    try {
      const list = await api.listSessions(RECENT_LIMIT);
      setRecent(list);
      setRecentLoading(false);
    } catch {
      // The recent list is optional; a failure here never blocks the conversation.
      setRecentLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!session) return;
    setRecentLoading(true);
    void refreshRecent();
  }, [session?.session_id, refreshRecent]);

  // ---------------------------------------------------------------------------
  // Preferences (real MCP get_active_preferences through the BFF)
  // ---------------------------------------------------------------------------
  const refreshPreferences = useCallback(async () => {
    const sessionId = sessionRef.current?.session_id;
    if (!sessionId) return;
    setPrefsLoading(true);
    try {
      const state = await api.preferences(sessionId);
      if (sessionRef.current?.session_id !== sessionId) return;
      setPreferences(state);
      setPrefsError(null);
    } catch (error) {
      setPrefsError(toErrorPayload(error));
    } finally {
      setPrefsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (session && health?.ready) void refreshPreferences();
  }, [session?.session_id, health?.ready, refreshPreferences]);

  // ---------------------------------------------------------------------------
  // Session bootstrap / switching
  // ---------------------------------------------------------------------------
  const applySession = useCallback((view: SessionView) => {
    abortRef.current?.abort();
    streamingRef.current = false;
    setStreamingKey(null);
    setSession(view);
    storeSessionId(view.session_id);
    setTurns(turnsFromSession(view));
    setPendingClarification(view.pending_clarification ?? null);
    setSelectedProductId(null);
    detailsRef.current = {};
    setDetails({});
    detailsLoadingRef.current = null;
    setDetailsLoading(null);
    setDetailsError(null);
    evidenceRef.current = {};
    setEvidenceByProduct({});
    setSelection([]);
    setSelectionError(null);
    setComparison(null);
    setComparisonError(null);
    setComparisonLatency(null);
    compareBusyRef.current = false;
    setPreferences(null);
    setPrefsError(null);
  }, []);

  const bootSession = useCallback(async () => {
    const cfg = configRef.current;
    if (!cfg) return;
    setSessionLoading(true);
    setSessionError(null);
    try {
      const stored = readStoredSessionId();
      let view: SessionView | null = null;
      if (stored) {
        try {
          view = await api.getSession(stored);
        } catch (error) {
          const payload = toErrorPayload(error);
          if (payload.code === "bff_unreachable") throw error;
          // A stale session id from a previous BFF process: discard it and start a new session.
          clearStoredSessionId();
        }
      }
      if (!view) view = await api.createSession(cfg.default_profile_id);
      applySession(view);
    } catch (error) {
      setSessionError(toErrorPayload(error));
    } finally {
      setSessionLoading(false);
    }
  }, [applySession]);

  useEffect(() => {
    if (!config || session || bootRef.current) return;
    bootRef.current = true;
    void bootSession();
  }, [config, session, bootSession]);

  const startSessionForProfile = useCallback(
    async (profileId: string) => {
      if (streamingRef.current) return;
      setSessionLoading(true);
      setSessionError(null);
      try {
        const view = await api.createSession(profileId);
        applySession(view);
        void refreshRecent();
      } catch (error) {
        setSessionError(toErrorPayload(error));
      } finally {
        setSessionLoading(false);
      }
    },
    [applySession, refreshRecent],
  );

  const newConversation = useCallback(() => {
    const cfg = configRef.current;
    if (!cfg) return;
    const profileId = sessionRef.current?.profile_id ?? cfg.default_profile_id;
    void startSessionForProfile(profileId);
  }, [startSessionForProfile]);

  const selectProfile = useCallback(
    (profileId: string) => {
      if (sessionRef.current?.profile_id === profileId) return;
      void startSessionForProfile(profileId);
    },
    [startSessionForProfile],
  );

  const selectSession = useCallback(
    async (sessionId: string) => {
      if (streamingRef.current) return;
      if (sessionRef.current?.session_id === sessionId) {
        setSidebarOpen(false);
        return;
      }
      setSessionLoading(true);
      setSessionError(null);
      try {
        const view = await api.getSession(sessionId);
        applySession(view);
        void refreshRecent();
      } catch (error) {
        setSessionError(toErrorPayload(error));
      } finally {
        setSessionLoading(false);
      }
    },
    [applySession, refreshRecent],
  );

  const resetSession = useCallback(async () => {
    const current = sessionRef.current;
    if (!current || streamingRef.current) return;
    setSessionLoading(true);
    setSessionError(null);
    try {
      const view = await api.resetSession(current.session_id);
      applySession(view);
      void refreshRecent();
      void refreshPreferences();
    } catch (error) {
      setSessionError(toErrorPayload(error));
    } finally {
      setSessionLoading(false);
    }
  }, [applySession, refreshPreferences, refreshRecent]);

  const reconnect = useCallback(async () => {
    const sessionId = sessionRef.current?.session_id;
    setSessionLoading(true);
    setSessionError(null);
    try {
      if (sessionId) {
        const view = await api.getSession(sessionId);
        applySession(view);
        void refreshRecent();
      } else {
        bootRef.current = false;
        await bootSession();
      }
    } catch (error) {
      setSessionError(toErrorPayload(error));
    } finally {
      setSessionLoading(false);
    }
  }, [applySession, bootSession, refreshRecent]);

  // ---------------------------------------------------------------------------
  // Details & evidence
  // ---------------------------------------------------------------------------
  const loadDetailsById = useCallback(
    async (productId: string, force = false) => {
      const sessionId = sessionRef.current?.session_id;
      if (!sessionId) return;
      if (!force && (detailsRef.current[productId] || detailsLoadingRef.current === productId)) return;
      detailsLoadingRef.current = productId;
      setDetailsLoading(productId);
      setDetailsError(null);
      try {
        const next = await api.productDetails(sessionId, productId);
        if (sessionRef.current?.session_id !== sessionId) return;
        detailsRef.current = { ...detailsRef.current, [next.product_id]: next };
        setDetails(prev => ({ ...prev, [next.product_id]: next }));
      } catch (error) {
        setDetailsError({ productId, error: toErrorPayload(error) });
      } finally {
        if (detailsLoadingRef.current === productId) {
          detailsLoadingRef.current = null;
          setDetailsLoading(null);
        }
      }
    },
    [],
  );

  const openPanelTab = useCallback(
    (tab: PanelTab) => {
      setPanelTab(tab);
      if (isDesktop) setPanelOpen(true);
      else setPanelSheetOpen(true);
    },
    [isDesktop],
  );

  const openDetailsById = useCallback(
    (productId: string) => {
      setSelectedProductId(productId);
      openPanelTab("details");
      void loadDetailsById(productId);
    },
    [loadDetailsById, openPanelTab],
  );

  const openDetailsFor = useCallback((product: ProductCard) => openDetailsById(product.product_id), [openDetailsById]);

  const openEvidenceFor = useCallback(
    (product: ProductCard) => {
      setSelectedProductId(product.product_id);
      const hasEvidence = Boolean(evidenceRef.current[product.product_id]) || Boolean(detailsRef.current[product.product_id]);
      if (!hasEvidence) void loadDetailsById(product.product_id);
      openPanelTab("evidence");
    },
    [loadDetailsById, openPanelTab],
  );

  const openComparisonPanel = useCallback(() => openPanelTab("comparison"), [openPanelTab]);

  // ---------------------------------------------------------------------------
  // Comparison selection & request
  // ---------------------------------------------------------------------------
  const toggleSelect = useCallback(
    (product: ProductCard) => {
      const id = product.product_id;
      setSelectionError(null);
      if (selection.includes(id)) {
        setSelection(selection.filter((value) => value !== id));
        return;
      }
      if (selection.length >= compareMax) {
        setSelectionError(`At most ${compareMax} products can be compared (BFF comparison limit).`);
        return;
      }
      setSelection([...selection, id]);
    },
    [compareMax, selection],
  );

  const removeSelection = useCallback((productId: string) => {
    setSelectionError(null);
    setSelection(prev => prev.filter((value) => value !== productId));
  }, []);

  const clearSelection = useCallback(() => {
    setSelectionError(null);
    setSelection([]);
  }, []);

  const runCompare = useCallback(
    async (productIds: string[]) => {
      const sessionId = sessionRef.current?.session_id;
      if (!sessionId || compareBusyRef.current) return;
      if (productIds.length < compareMin || productIds.length > compareMax) {
        setSelectionError(`Select ${compareMin}–${compareMax} products to compare (BFF comparison limits).`);
        openPanelTab("comparison");
        return;
      }
      compareBusyRef.current = true;
      setComparisonLoading(true);
      setComparisonError(null);
      setComparison(null);
      setComparisonLatency(null);
      openPanelTab("comparison");
      const startedAt = performance.now();
      try {
        const payload = await api.compare(sessionId, productIds);
        if (sessionRef.current?.session_id !== sessionId) return;
        setComparison(payload);
        setComparisonLatency(performance.now() - startedAt);
      } catch (error) {
        setComparison(null);
        setComparisonError(toErrorPayload(error));
      } finally {
        setComparisonLoading(false);
        compareBusyRef.current = false;
      }
    },
    [compareMax, compareMin, openPanelTab],
  );

  const compareVerified = useCallback(() => {
    const ids = configRef.current?.verified_comparison_ids ?? [];
    if (ids.length < compareMin) return;
    void runCompare(ids.slice(0, compareMax));
  }, [compareMax, compareMin, runCompare]);

  // ---------------------------------------------------------------------------
  // Clarification reply (real HTTP POST; the BFF sets the confirmation source)
  // ---------------------------------------------------------------------------
  const replyClarification = useCallback(
    async (turn: LocalTurn, text: string) => {
      const sessionId = sessionRef.current?.session_id;
      const clarification = turn.clarification ?? pendingClarification;
      if (!sessionId || !clarification || replyBusyRef.current) return;
      replyBusyRef.current = true;
      const sending: ClarificationReplyState = {
        text,
        status: "sending",
        clarification_id: clarification.clarification_id,
        steps: [],
        memory_update: null,
        preferences: null,
        recommendation: null,
        error: null,
        planner: null,
      };
      setTurns(prev => prev.map(entry => (entry.key === turn.key ? { ...entry, reply: sending } : entry)));
      try {
        const response = await api.clarificationReply(sessionId, clarification.clarification_id, text);
        if (sessionRef.current?.session_id !== sessionId) return;
        const failed = Boolean(response.error);
        // A resumed Qwen Live turn reports its real planner metadata in the reply response.
        const replyPlanner = failed ? null : plannerFromReplyResponse(response);
        setTurns(prev =>
          prev.map(entry =>
            entry.key === turn.key
              ? {
                  ...entry,
                  clarification: response.clarification ?? entry.clarification,
                  planner: mergePlanner(entry.planner, replyPlanner),
                  reply: {
                    text,
                    status: failed ? "error" : "done",
                    clarification_id: clarification.clarification_id,
                    steps: response.steps ?? [],
                    memory_update: response.memory_update ?? null,
                    preferences: response.preferences ?? null,
                    recommendation: response.recommendation ?? null,
                    error: response.error ?? null,
                    planner: replyPlanner,
                  },
                }
              : entry,
          ),
        );
        if (!failed) {
          if (response.clarification) {
            // A real follow-up ASK (for example an unresolved M9 conflict) must stay pending;
            // never silently finalize and never drop the new server-issued handle.
            setPendingClarification(response.clarification);
            setSession(prev => (prev ? { ...prev, pending_clarification: response.clarification ?? null } : prev));
          } else {
            setPendingClarification(null);
            setSession(prev => (prev ? { ...prev, pending_clarification: null } : prev));
          }
          if (response.preferences) setPreferences(response.preferences);
          void refreshRecent();
          void refreshPreferences();
        }
      } catch (error) {
        const payload = toErrorPayload(error);
        setTurns(prev =>
          prev.map(entry =>
            entry.key === turn.key
              ? { ...entry, reply: { ...sending, status: "error", error: payload } }
              : entry,
          ),
        );
      } finally {
        replyBusyRef.current = false;
      }
    },
    [pendingClarification, refreshPreferences, refreshRecent],
  );

  // ---------------------------------------------------------------------------
  // Chat SSE (fetch + ReadableStream; POST cannot use EventSource)
  // ---------------------------------------------------------------------------
  const sendMessage = useCallback(
    async (raw: string, presetSelection?: string[]) => {
      const text = raw.trim();
      const sessionId = sessionRef.current?.session_id;
      if (!text || !sessionId || streamingRef.current) return;
      // Qwen Replay is historical/read-only; Qwen Live runs only while the backend reports it
      // available. Neither case ever quietly falls back to the CPU pipeline.
      if (mode === "qwen_replay") return;
      if (mode === "qwen_live" && !qwenLiveAvailable) return;
      if (health?.ready !== true) return;

      const key = newTurnKey();
      const selectedSnapshot = presetSelection ?? selection;
      const controller = new AbortController();
      abortRef.current = controller;
      streamingRef.current = true;
      setStreamingKey(key);
      setLastHeartbeatAt(null);
      setSelection([]);
      setSelectionError(null);

      const optimistic = emptyLocalTurn({
        key,
        user_text: text,
        pending: true,
        status: "streaming",
        startedAt: performance.now(),
        text_final: false,
        created_at: new Date().toISOString(),
      });
      setTurns(prev => [...prev, optimistic]);

      const patchTurn = (patch: (entry: LocalTurn) => LocalTurn) => {
        setTurns(prev => prev.map(entry => (entry.key === key ? patch(entry) : entry)));
      };

      try {
        const stream = api.chatStream(
          {
            session_id: sessionId,
            message: text,
            selected_product_ids: selectedSnapshot.length ? selectedSnapshot : undefined,
            // Sent only for the real live Qwen mode; the CPU request shape stays unchanged.
            mode: mode === "qwen_live" ? "qwen_live" : undefined,
          },
          {
            signal: controller.signal,
            onHeartbeat: () => setLastHeartbeatAt(performance.now()),
          },
        );
        for await (const event of stream) {
          if (sessionRef.current?.session_id !== sessionId) break;
          switch (event.type) {
            case "tool_status": {
              patchTurn(entry => ({ ...entry, tool_activity: mergeToolStatus(entry.tool_activity, event.data) }));
              break;
            }
            case "text": {
              const delta = event.data.text ?? "";
              const final = event.data.final === true;
              patchTurn(entry => {
                const nextText =
                  final && delta.startsWith(entry.assistant_text) ? delta : entry.assistant_text + delta;
                return { ...entry, assistant_text: nextText, text_final: final || entry.text_final };
              });
              break;
            }
            case "products": {
              const payload = event.data;
              patchTurn(entry => ({
                ...entry,
                products: payload,
                // Real planner metadata either on `products` or later on `done`; merged, not invented.
                planner: mergePlanner(entry.planner, plannerFromProducts(payload)),
              }));
              const candidate =
                payload.recommended_product_id ??
                payload.items.find(item => item.recommended)?.product_id ??
                payload.items[0]?.product_id ??
                null;
              if (candidate) {
                setSelectedProductId(prev => prev ?? candidate);
                void loadDetailsById(candidate);
              }
              break;
            }
            case "evidence": {
              const payload = event.data;
              const productId = payload.product_id;
              if (productId) {
                evidenceRef.current = { ...evidenceRef.current, [productId]: payload };
                setEvidenceByProduct(prev => ({ ...prev, [productId]: payload }));
                setSelectedProductId(prev => prev ?? productId);
              }
              break;
            }
            case "comparison": {
              const payload = event.data;
              patchTurn(entry => ({ ...entry, comparison: payload }));
              setComparison(payload);
              setComparisonError(null);
              setComparisonLatency(null);
              setPanelTab("comparison");
              if (isDesktop) setPanelOpen(true);
              break;
            }
            case "clarification": {
              const payload = event.data;
              patchTurn(entry => ({ ...entry, clarification: payload }));
              setPendingClarification(payload);
              break;
            }
            case "memory_update": {
              const payload = event.data;
              patchTurn(entry => ({ ...entry, memory_update: payload }));
              if (payload.preferences) setPreferences(payload.preferences);
              break;
            }
            case "done": {
              const payload = event.data;
              patchTurn(entry => ({
                ...entry,
                turn_id: payload.turn_id,
                stop_reason: payload.stop_reason,
                total_ms: payload.total_ms,
                planner: mergePlanner(entry.planner, plannerFromDone(payload)),
                pending: false,
                text_final: true,
                status: entry.error ? "error" : "complete",
              }));
              setPendingClarification(prev => (payload.stop_reason === "awaiting_clarification" ? prev : null));
              break;
            }
            case "error": {
              const payload = event.data;
              patchTurn(entry => ({
                ...entry,
                error: payload,
                pending: false,
                text_final: true,
                status: "error",
                stop_reason: "error",
              }));
              setPendingClarification(null);
              // A real Qwen planner failure can also make the live mode unavailable server-side.
              // Refresh health so the UI can show the backend's own reason; never retry as CPU.
              if (payload.code?.startsWith("qwen_")) setHealthNonce(value => value + 1);
              break;
            }
            default:
              break;
          }
        }
      } catch (error) {
        if (isAbortError(error)) {
          patchTurn(entry => ({ ...entry, pending: false, status: "interrupted", text_final: true }));
        } else {
          patchTurn(entry => ({
            ...entry,
            pending: false,
            text_final: true,
            status: "error",
            error: toErrorPayload(error),
          }));
        }
      } finally {
        // The stream is over: any progress row the backend never closed is shown as explicitly
        // unreported (browser-local) instead of a fake endless spinner or a fabricated success.
        patchTurn(entry => ({ ...entry, tool_activity: finaliseStaleToolRows(entry.tool_activity) }));
        abortRef.current = null;
        streamingRef.current = false;
        setStreamingKey(null);
        setLastHeartbeatAt(null);
        void refreshRecent();
        if (sessionRef.current?.session_id) void refreshPreferences();
      }
    },
    [health?.ready, isDesktop, loadDetailsById, mode, qwenLiveAvailable, refreshPreferences, refreshRecent, selection],
  );

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const retryMessage = useCallback(
    (text: string) => {
      if (streamingRef.current) return;
      void sendMessage(text);
    },
    [sendMessage],
  );

  // A read-only profile can hand the same question to the write-enabled demo profile; a new
  // session is created for that profile and the query is re-sent by the effect below.
  const switchProfileAndRetry = useCallback(
    async (profileId: string, query: string) => {
      if (streamingRef.current) return;
      setSessionLoading(true);
      setSessionError(null);
      try {
        const view = await api.createSession(profileId);
        applySession(view);
        setAutoQuery({ sessionId: view.session_id, text: query });
        void refreshRecent();
      } catch (error) {
        setSessionError(toErrorPayload(error));
      } finally {
        setSessionLoading(false);
      }
    },
    [applySession, refreshRecent],
  );

  useEffect(() => {
    if (!autoQuery) return;
    if (sessionRef.current?.session_id !== autoQuery.sessionId) return;
    if (health?.ready !== true || mode !== "cpu_live") return;
    setAutoQuery(null);
    void sendMessage(autoQuery.text);
  }, [autoQuery, health?.ready, mode, sendMessage]);

  // ---------------------------------------------------------------------------
  // Qwen Replay
  // ---------------------------------------------------------------------------
  const loadReplay = useCallback(async () => {
    if (replayBusyRef.current) return;
    replayBusyRef.current = true;
    setReplayLoading(true);
    setReplayError(null);
    try {
      const payload = await api.replay();
      setReplay(payload);
    } catch (error) {
      setReplayError(toErrorPayload(error));
    } finally {
      setReplayLoading(false);
      replayBusyRef.current = false;
    }
  }, []);

  const openReplay = useCallback(() => {
    setReplayOpen(true);
    if (!replay) void loadReplay();
  }, [loadReplay, replay]);

  const changeMode = useCallback(
    (next: UiMode) => {
      if (streamingRef.current) return;
      // Qwen Live is selectable only while the backend explicitly reports it available.
      if (next === "qwen_live" && !qwenLiveAvailableRef.current) return;
      setMode(next);
      if (next === "qwen_replay") {
        setReplayOpen(true);
        if (!replay) void loadReplay();
      }
      if (next === "qwen_live") {
        // Re-probe health on selection so a stale availability cannot be presented as live.
        retryHealth();
      }
    },
    [loadReplay, replay, retryHealth],
  );

  // Abort any in-flight stream when the page unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  // ---------------------------------------------------------------------------
  // Derived values
  // ---------------------------------------------------------------------------
  const selectedProduct: ProductCard | null = (() => {
    if (!selectedProductId) return null;
    for (const turn of turns) {
      for (const payload of [turn.products, turn.reply?.recommendation ?? null]) {
        const found = payload?.items.find(item => item.product_id === selectedProductId);
        if (found) return found;
      }
    }
    return null;
  })();
  const selectedDetail = selectedProductId ? details[selectedProductId] ?? null : null;
  const selectedDetailError =
    selectedProductId && detailsError?.productId === selectedProductId ? detailsError.error : null;
  const selectedEvidence = selectedProductId ? evidenceByProduct[selectedProductId] ?? null : null;
  const profileLabel =
    config?.profiles.find(profile => profile.profile_id === session?.profile_id)?.label ?? session?.profile_id ?? null;
  const writeProfile = config?.profiles.find(profile => profile.memory_writes) ?? null;
  const readinessElapsed = Math.max(health?.uptime_s ?? 0, clock);
  const streaming = streamingKey != null;

  const panelProps: Omit<RightPanelProps, "variant" | "onClose"> = {
    tab: panelTab,
    onTabChange: openPanelTab,
    selectedProduct,
    selectedProductId,
    detail: selectedDetail,
    detailLoading: Boolean(selectedProductId && detailsLoading === selectedProductId),
    detailError: selectedDetailError,
    onLoadDetail: () => {
      if (selectedProductId) void loadDetailsById(selectedProductId);
    },
    onRetryDetail: () => {
      if (selectedProductId) void loadDetailsById(selectedProductId, true);
    },
    evidencePayload: selectedEvidence,
    comparison,
    comparisonLoading,
    comparisonError,
    comparisonLatencyMs: comparisonLatency,
    onRetryCompare: () => void runCompare(selection),
    selection,
    compareMin,
    compareMax,
    selectionError,
    verifiedIds: config?.verified_comparison_ids ?? [],
    onRemoveSelection: removeSelection,
    onClearSelection: clearSelection,
    onCompare: (ids) => void runCompare(ids),
    onCompareVerified: compareVerified,
    preferences,
    onOpenProductDetails: openDetailsById,
  };

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-(--ground)" data-testid="app-shell">
      <Header
        health={health}
        healthError={healthError}
        config={config}
        mode={mode}
        onModeChange={changeMode}
        qwenLive={qwenLive}
        onOpenSidebar={() => setSidebarOpen(true)}
        onOpenPanel={() => setPanelSheetOpen(true)}
        onTogglePanel={() => setPanelOpen(value => !value)}
        panelOpen={panelOpen}
        onOpenReplay={openReplay}
        onRetryHealth={retryHealth}
        sessionTitle={session?.title ?? null}
        profileLabel={profileLabel}
        streaming={streaming}
      />

      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-[288px] shrink-0 border-r border-(--line) bg-(--card) xl:flex xl:min-h-0 xl:flex-col">
          <Sidebar
            config={config}
            session={session}
            recent={recent}
            recentLoading={recentLoading}
            preferences={preferences}
            prefsLoading={prefsLoading}
            prefsError={prefsError}
            onRefreshPreferences={() => void refreshPreferences()}
            onNewConversation={newConversation}
            onSelectSession={(sessionId) => void selectSession(sessionId)}
            onSelectProfile={selectProfile}
            onResetSession={() => void resetSession()}
            busy={streaming}
            variant="column"
          />
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <ChatPanel
            health={health}
            healthError={healthError}
            config={config}
            session={session}
            sessionLoading={sessionLoading}
            sessionError={sessionError}
            turns={turns}
            busy={streaming}
            mode={mode}
            qwenLive={qwenLive}
            onSwitchToCpuLive={() => changeMode("cpu_live")}
            selection={selection}
            compareMin={compareMin}
            compareMax={compareMax}
            selectionError={selectionError}
            comparisonLoading={comparisonLoading}
            lastHeartbeatAt={lastHeartbeatAt}
            readinessElapsed={readinessElapsed}
            prefill={null}
            onRetryHealth={retryHealth}
            onReconnect={() => void reconnect()}
            onSend={(text, selectedIds) => void sendMessage(text, selectedIds)}
            onStop={stopStream}
            onRetryMessage={retryMessage}
            onToggleSelect={toggleSelect}
            onViewDetails={openDetailsFor}
            onViewEvidence={openEvidenceFor}
            onCompare={(ids) => void runCompare(ids)}
            onReplyClarification={(turn, text) => void replyClarification(turn, text)}
            onOpenComparison={openComparisonPanel}
            onRemoveSelection={removeSelection}
            onClearSelection={clearSelection}
            onOpenReplay={openReplay}
            onNewConversation={newConversation}
            writeProfileId={writeProfile?.profile_id ?? null}
            writeProfileLabel={writeProfile?.label ?? null}
            onSwitchProfileRetry={(profileId, query) => void switchProfileAndRetry(profileId, query)}
          />
        </main>

        {panelOpen ? (
          <aside
            className="hidden w-[396px] shrink-0 border-l border-(--line) bg-(--card) xl:flex xl:min-h-0 xl:flex-col"
            data-testid="panel-column"
          >
            <RightPanel {...panelProps} variant="column" onClose={() => setPanelOpen(false)} />
          </aside>
        ) : null}
      </div>

      {/* Mobile / tablet: conversations & preferences drawer (left) */}
      {sidebarOpen && !isDesktop ? (
        <>
          <div className="fixed inset-0 z-40 bg-black/30" aria-hidden onClick={() => setSidebarOpen(false)} />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label="Conversations and preferences"
            className="ac-slide-in-left fixed inset-y-0 left-0 z-50 flex w-[min(92vw,340px)] flex-col border-r border-(--line) bg-(--card) shadow-(--shadow-lg)"
          >
            <Sidebar
              config={config}
              session={session}
              recent={recent}
              recentLoading={recentLoading}
              preferences={preferences}
              prefsLoading={prefsLoading}
              prefsError={prefsError}
              onRefreshPreferences={() => void refreshPreferences()}
              onNewConversation={newConversation}
              onSelectSession={(sessionId) => void selectSession(sessionId)}
              onSelectProfile={selectProfile}
              onResetSession={() => void resetSession()}
              busy={streaming}
              variant="drawer"
              onClose={() => setSidebarOpen(false)}
            />
          </aside>
        </>
      ) : null}

      {/* Mobile / tablet: details, evidence & comparison sheet (right) */}
      {panelSheetOpen && !isDesktop ? (
        <>
          <div className="fixed inset-0 z-40 bg-black/30" aria-hidden onClick={() => setPanelSheetOpen(false)} />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label="Details, evidence and comparison"
            className="ac-slide-in-right fixed inset-y-0 right-0 z-50 flex w-[min(100vw,430px)] flex-col bg-(--card) shadow-(--shadow-lg) sm:inset-y-2.5 sm:right-2.5 sm:rounded-[18px]"
            data-testid="panel-sheet"
          >
            <RightPanel {...panelProps} variant="sheet" onClose={() => setPanelSheetOpen(false)} />
          </aside>
        </>
      ) : null}

      {/* Qwen Replay: historical trajectories only */}
      {replayOpen ? (
        <Sheet
          title="Qwen Replay · historical trajectories"
          detail="not a live model call"
          onClose={() => setReplayOpen(false)}
          closeLabel="Close replay viewer"
          footer={
            mode === "qwen_replay" ? (
              <Button
                variant="primary"
                icon="spark"
                onClick={() => {
                  changeMode("cpu_live");
                  setReplayOpen(false);
                }}
              >
                Switch back to CPU Live
              </Button>
            ) : (
              <Button variant="accent" icon="play" onClick={() => changeMode("qwen_replay")}>
                Enter Qwen Replay mode
              </Button>
            )
          }
        >
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-(--ink-soft)">
            <span className="font-mono">GET {BFF_BASE}/api/replay</span>
            <button
              type="button"
              onClick={() => void loadReplay()}
              className="rounded-md border border-(--line-strong) bg-(--card) px-1.5 py-0.5 font-semibold text-(--ink-2) hover:border-(--accent)"
            >
              Reload
            </button>
          </div>
          <ReplayPanel replay={replay} loading={replayLoading} error={replayError} onLoad={() => void loadReplay()} />
        </Sheet>
      ) : null}
    </div>
  );
}
