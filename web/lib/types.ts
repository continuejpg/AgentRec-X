// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

/**
 * View-model types for the AgentRec-X Web Demo BFF.
 *
 * Source of truth: docs/WEB_DEMO_CONTRACT.md ("View models (all mapped from real MCP payloads)").
 * Every field below is a *view model* produced by the BFF from a real MCP tool result. The
 * frontend renders these verbatim and never re-derives, re-ranks, or fills in UNKNOWN values.
 */

// ---------------------------------------------------------------------------
// Health / config
// ---------------------------------------------------------------------------

export type ServiceStatus = "starting" | "ready" | "degraded" | "error";

export interface HealthServer {
  profile_id: string;
  user_key: string;
  status: ServiceStatus;
  search_mode: string;
  writes_enabled: boolean;
  startup_ms: number | null;
  error: string | null;
  tools: string[] | null;
}

export type ModeKey = "cpu_live" | "qwen_replay" | "qwen_live";

export interface ModeInfo {
  available: boolean;
  label: string;
  detail?: string | null;
  badge?: string | null;
  disabled_reason?: string | null;
}

/**
 * Optional Qwen Base Live health block from GET /api/health (docs/QWEN_LIVE_CONTRACT.md §2).
 *
 * Deliberately contains no endpoint or credential fields. All fields are optional so an older
 * BFF (no live mode yet) still type-checks and renders as "unavailable", never as fake live.
 */
export interface QwenLiveHealth {
  enabled?: boolean | null;
  configured?: boolean | null;
  status?: string | null;
  available?: boolean | null;
  model?: string | null;
  planner_action?: unknown;
  latency_ms?: number | null;
  checked_at?: string | null;
  detail?: string | null;
}

export interface Health {
  status: ServiceStatus;
  ready: boolean;
  started_at: string | null;
  uptime_s: number | null;
  catalogue_size: number | null;
  servers: HealthServer[];
  modes: Partial<Record<ModeKey, ModeInfo>>;
  memory_writes_enabled?: boolean;
  /** Real Qwen Live probe result; absent on an older BFF. Never inferred client-side. */
  qwen_live?: QwenLiveHealth | null;
}

export interface ProfileInfo {
  profile_id: string;
  label: string;
  description: string;
  user_key: string;
  memory_writes: boolean;
  search_mode: string;
}

export interface DemoLimits {
  search_top_k: number;
  compare_min: number;
  compare_max: number;
  clarification_ttl_s: number;
}

export interface DemoConfig {
  profiles: ProfileInfo[];
  default_profile_id: string;
  limits: DemoLimits;
  verified_comparison_ids: string[];
  modes: Partial<Record<ModeKey, ModeInfo>>;
  notices: string[];
  behavior?: BehaviorConfig | null;
}

/** First-party local behavior collector configuration (no external analytics SDK). */
export interface BehaviorConfig {
  enabled: boolean;
  master_enabled: boolean;
  semantics_version?: string;
  storage: string;
  impression_min_visible_ms: number;
  impression_visibility_threshold: number;
  window_seconds: number;
  half_life_seconds: number;
  event_types: string[];
  event_sources: string[];
  compare_action_statuses?: string[];
  live_source: string;
  profile_uses_live_only: boolean;
  impression_interest_weight?: number;
  compare_interest_weight?: number;
  retention_days: number;
  external_analytics: string | null;
  notes: string[];
}

// ---------------------------------------------------------------------------
// Planner metadata (real Qwen Base Live / Rule reports only)
// ---------------------------------------------------------------------------

/**
 * Real planner metadata as reported by the backend (docs/QWEN_LIVE_CONTRACT.md §3/§4).
 *
 * `kind` is the only planner label the UI renders: `"qwen_base"` or `"rule"` come from the
 * backend verbatim. The UI never substitutes a planner label of its own, and an absent planner
 * block renders nothing rather than an inferred "rule" claim.
 */
export interface PlannerMeta {
  kind: string;
  model?: string | null;
  calls?: number | null;
  /** `done` reports one scalar; the MCP planner object reports an array of per-call latencies. */
  latency_ms?: number | number[] | null;
  actions?: unknown[] | null;
  parser_errors?: string[] | null;
}

/** Wire shape: an object, a bare kind string, or absent (older BFF). */
export type PlannerMetaWire = PlannerMeta | string | null | undefined;

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export interface SessionSummary {
  session_id: string;
  profile_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  behavior_enabled?: boolean;
}

export interface SessionView {
  session_id: string;
  profile_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: TurnView[];
  pending_clarification: ClarificationPayload | null;
  behavior_enabled?: boolean;
  latest_recommendation_id?: string | null;
}

/** A persisted turn as returned by GET /api/sessions/{id} (and POST /api/sessions/{id}/reset). */
export interface TurnView {
  turn_id: string;
  user_text: string;
  assistant_text: string;
  tool_activity: ToolStatus[];
  products: ProductsPayload | null;
  comparison: ComparisonPayload | null;
  clarification: ClarificationPayload | null;
  memory_update: MemoryUpdatePayload | null;
  error: ErrorPayload | null;
  created_at: string;
  /** Optional real planner metadata added by the Qwen Live backend; never inferred here. */
  planner?: PlannerMetaWire;
  planner_model?: string | null;
  planner_kind?: string | null;
  planner_calls?: number | null;
  planner_latency_ms?: number | null;
  planner_actions?: unknown[] | null;
}

// ---------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------

export interface PreferenceItem {
  memory_id: string;
  value: string | null;
  polarity: string | null;
  kind: string | null;
  status: string | null;
  source_text: string | null;
  source_turn_id: string | null;
  last_updated: string | null;
  logical_seq?: number | null;
  supersedes?: string | null;
  superseded_by?: string | null;
  created_at?: number | null;
}

export interface PreferenceState {
  profile_id: string;
  user_key: string | null;
  history_status: string | null;
  active: PreferenceItem[];
  inactive: PreferenceItem[];
  audit_source: string | null;
  fetched_at: string;
}

export interface MemoryRememberResponse {
  turn_id: string;
  update: Record<string, unknown>;
  preferences: PreferenceState;
  message: string;
  error: ErrorPayload | null;
}

// ---------------------------------------------------------------------------
// Chat SSE view models
// ---------------------------------------------------------------------------

/**
 * Real row status from the BFF (`running|ok|error`). `unknown` is a browser-local finalisation
 * used only when the backend persisted/ended a turn without ever reporting completion for a row:
 * the UI states that honestly instead of leaving a fake endless spinner or claiming success.
 */
export type ToolRunStatus = "running" | "ok" | "error" | "unknown";

export interface ToolStatus {
  tool: string;
  label: string;
  status: ToolRunStatus;
  detail?: string | null;
  latency_ms?: number | null;
}

export interface UnknownField {
  attribute: string;
  reason: string;
}

export interface EvidenceItem {
  evidence_id: string | null;
  raw_field: string | null;
  span_start: number | null;
  span_end: number | null;
  span_text: string | null;
  raw_excerpt: string | null;
  attribute: string | null;
  verification_status: string | null;
  source_record: string | null;
}

export interface ProductAttribute {
  attribute: string;
  status: string;
  value: string | null;
  evidence_id: string | null;
  raw_field: string | null;
  span_text: string | null;
}

export interface ProductPrice {
  status: string;
  display: string;
  reason: string;
}

export interface ProductCard {
  product_id: string;
  namespaced_id: string | null;
  title: string | null;
  product_type: string | null;
  form_factor: string | null;
  product_role: string | null;
  source: string;
  retrieval_source: string;
  match_status: string;
  recommended: boolean;
  price: ProductPrice;
  attributes: ProductAttribute[];
  unknown_fields: UnknownField[];
  evidence_available: boolean;
  image_url: string | null;
  source_record: string | null;
  /** Server-assigned 1-based display position; never inferred by the browser. */
  displayed_position?: number;
}

export interface GateDecision {
  action: string | null;
  reason_code: string | null;
  reason: string | null;
  selected_candidate_id: string | null;
  unknown_fields: string[];
  product_role: string | null;
  candidate_presentable_ids: string[];
}

export interface ProductsPayload {
  query: string;
  search_mode: string | null;
  score_kind: string | null;
  catalogue_size: number | null;
  items: ProductCard[];
  recommended_product_id: string | null;
  gate: GateDecision | null;
  /** One new server-issued id for every displayed order; events are validated against it. */
  recommendation_id?: string | null;
  request_id?: string | null;
  ranker_version?: string | null;
  /** Optional real planner metadata; product/evidence shapes are otherwise unchanged. */
  planner?: PlannerMetaWire;
  planner_model?: string | null;
  planner_kind?: string | null;
  planner_calls?: number | null;
  planner_latency_ms?: number | null;
  planner_actions?: unknown[] | null;
}

export interface EvidencePayload {
  product_id: string | null;
  evidence: EvidenceItem[];
  unknown_fields: UnknownField[];
  source_record: string | null;
}

export interface DetailFact {
  attribute: string;
  status: string;
  value: string | null;
  reason: string | null;
  source: string | null;
  verification: string | null;
}

export interface RawFields {
  title?: string[];
  description?: string[];
  categories?: string[];
  features?: string[];
  [key: string]: string[] | undefined;
}

export interface ProductDetailPayload {
  product_id: string;
  namespaced_id: string | null;
  title: string | null;
  product_type: string | null;
  form_factor: string | null;
  product_role: string | null;
  facts: DetailFact[];
  evidence: EvidenceItem[];
  unknown_fields: UnknownField[];
  raw_fields: RawFields;
  source_record: string | null;
  evidence_available: boolean;
  notes: string[];
}

export type ComparisonCellStatus = "present" | "known" | "unknown" | "unsupported" | string;

export interface ComparisonCell {
  product_id: string;
  status: ComparisonCellStatus;
  value: string | null;
  evidence_ids: string[];
}

export interface ComparisonDimension {
  attribute: string;
  cells: ComparisonCell[];
}

export interface ComparisonProduct {
  product_id: string;
  title: string | null;
  product_type: string | null;
  role_status: string;
  facts: DetailFact[];
  unknown_fields: UnknownField[];
}

export interface PreferenceRelation {
  memory_id: string;
  value: string;
  polarity: string;
  relation: string;
  observed_in_catalogue_text: boolean;
  source_text: string;
  per_product: Record<string, string | null>;
}

/** P16 conflict rows are passed through verbatim; only the common keys are typed. */
export interface ComparisonConflict {
  attribute?: string | null;
  products?: string[] | null;
  message?: string | null;
  detail?: string | null;
  [key: string]: unknown;
}

export interface ComparisonUnknownField {
  attribute: string;
  products: string[];
}

export interface ClaimsSummary {
  valid: number | null;
  rejected: number | null;
  unknown: number | null;
}

export interface ConditionalRecommendation {
  supported: boolean | null;
  text: string | null;
}

export interface ComparisonPayload {
  status: string | null;
  products: ComparisonProduct[];
  dimensions: ComparisonDimension[];
  preference_relations: PreferenceRelation[];
  conflicts: ComparisonConflict[];
  unknown_fields: ComparisonUnknownField[];
  claims_summary: ClaimsSummary | null;
  conditional_recommendation: ConditionalRecommendation | null;
  final_answer: string | null;
  evidence_sources: string[];
  notes: string[];
}

export interface ClarificationPayload {
  clarification_id: string;
  question: string;
  reason: string | null;
  expires_in_seconds: number | null;
  memory_write_enabled: boolean;
  quick_options: string[];
}

export interface MemoryUpdatePayload {
  added: number;
  superseded: number;
  removed: number;
  skipped_duplicates: number;
  already_processed: boolean;
  confirmation_source: string;
  preferences?: PreferenceState | null;
}

export interface ErrorPayload {
  code: string;
  message: string;
  tool?: string | null;
  detail?: unknown;
}

export type StopReason = "finalized" | "awaiting_clarification" | "error" | "comparison" | string;

export interface DonePayload {
  turn_id: string;
  stop_reason: StopReason;
  total_ms: number;
  /** Optional real planner metadata added by the Qwen Live backend. */
  planner?: PlannerMetaWire;
  planner_model?: string | null;
  planner_calls?: number | null;
  planner_latency_ms?: number | null;
  planner_actions?: unknown[] | null;
}

export interface ClarificationReplyResponse {
  turn_id: string;
  memory_update: MemoryUpdatePayload;
  preferences: PreferenceState;
  recommendation: ProductsPayload | null;
  steps: ToolStatus[];
  clarification_id: string;
  /** A real follow-up Qwen question when M9 still has an unresolved conflict. */
  clarification?: ClarificationPayload | null;
  message: string;
  error: ErrorPayload | null;
  /** Optional real planner metadata when the resumed turn was a Qwen Live turn. */
  planner?: PlannerMetaWire;
  planner_model?: string | null;
  planner_calls?: number | null;
  planner_latency_ms?: number | null;
  planner_actions?: unknown[] | null;
  /** Current-task constraints and candidate re-evaluation metadata for a resumed Qwen turn. */
  task_constraints?: unknown;
  candidate_reevaluation?: unknown;
}

// ---------------------------------------------------------------------------
// Qwen Replay
// ---------------------------------------------------------------------------

export interface ReplayTerminal {
  status: string;
  selected_candidate_id: string | null;
  ask_question: string | null;
}

export interface ReplayGate {
  action?: string | null;
  reason_code?: string | null;
  reason?: string | null;
  selected_candidate_id?: string | null;
  candidate_presentable_ids?: string[] | null;
  product_role?: string | null;
  [key: string]: unknown;
}

export interface ReplayModelStats {
  calls: number;
  latency_ms: number[];
  input_tokens: number;
  output_tokens: number;
  parser_errors: string[];
}

export interface ReplayExample {
  mode: string;
  replay: boolean;
  live_call: boolean;
  source: string;
  task_id: string;
  query: string;
  /**
   * Optional: the saved final-answer-only trajectories (P14-C1 / P15, examples 5–7 in the real
   * runs/p18/qwen_replay.json) have no `terminal` block at all. Never fabricate one.
   */
  terminal?: ReplayTerminal | null;
  /** Optional for the same final-answer-only trajectories. */
  gate?: ReplayGate | null;
  model_stats?: ReplayModelStats | null;
  /** Present in final-answer-only trajectories instead of terminal/gate. */
  final_answer?: string | null;
}

export interface ReplayResponse {
  mode: string;
  replay: boolean;
  live_call: boolean;
  badge: string;
  source?: string;
  note?: string;
  examples: ReplayExample[];
}

export interface RankerReplayItem {
  product_id: string;
  title: string;
  rank: number;
  score: number;
  attributes: string[];
  evidence_count: number;
  memory_matches: string[];
  memory_conflicts: string[];
  memory_unknown: string[];
  off_rank?: number | null;
  rank_movement?: number | null;
}

export interface RankerReplayArm {
  off: RankerReplayItem[];
  on: RankerReplayItem[];
  top1_changed: boolean;
}

export interface RankerReplayTaskRef {
  task_id: string;
  query: string;
  scenario_type?: string | null;
}

export interface RankerReplayResponse {
  mode: string;
  replay: boolean;
  live_call: boolean;
  available: boolean;
  badge: string;
  source?: string;
  detail?: string;
  task_id?: string;
  query?: string;
  scenario_type?: string | null;
  memory_snapshot_id?: string | null;
  active_preferences_on?: PreferenceItem[];
  arms?: Record<"existing" | "natural" | "hybrid", RankerReplayArm>;
  tasks: RankerReplayTaskRef[];
  notes?: string[];
}

// ---------------------------------------------------------------------------
// Behavior Feedback V1 (first-party local events; no external analytics)
// ---------------------------------------------------------------------------

export type BehaviorEventType =
  | "product_impression"
  | "product_click"
  | "product_compare"
  | "explicit_item_feedback";

export type BehaviorEventSource =
  | "LIVE_USER_INTERACTION"
  | "AUTOMATED_BROWSER_E2E"
  | "RECORDED_REPLAY"
  | "SYNTHETIC_TEST";

export interface BehaviorEventInput {
  event_id: string;
  event_type: BehaviorEventType;
  recommendation_id: string;
  product_id: string;
  displayed_position: number;
  event_source: BehaviorEventSource;
  metadata?: {
    feedback?: "like" | "dislike";
    compared_product_ids?: string[];
    action_status?: "attempted" | "succeeded" | "failed";
    compare_id?: string;
    failure_code?: string;
    failure_message?: string;
  };
}

export interface BehaviorEventView {
  event_id: string;
  event_type: BehaviorEventType;
  event_source: BehaviorEventSource;
  recommendation_id: string;
  request_id: string | null;
  turn_id: string | null;
  product_id: string;
  product_title: string | null;
  displayed_position: number;
  ranker_version: string | null;
  occurred_at: string;
  metadata: {
    feedback?: "like" | "dislike";
    compared_product_ids?: string[];
    action_status?: "attempted" | "succeeded" | "failed";
    compare_id?: string;
    failure_code?: string;
    failure_message?: string;
  };
  enters_profile: boolean;
}

export interface BehaviorCounts {
  live_events: number;
  profile_events?: number;
  human_live_events?: number;
  automated_browser_e2e_events?: number;
  product_impression: number;
  product_click: number;
  product_compare: number;
  product_compare_events?: number;
  product_compare_attempted: number;
  product_compare_completed: number;
  product_compare_failed: number;
  explicit_feedback: number;
  explicit_like: number;
  explicit_dislike: number;
  click_after_impression: number;
  impressions_with_click: number;
  stored_total: number;
  recorded_replay: number;
  synthetic_test: number;
}

export interface BehaviorAttributeEvidence {
  event_id: string;
  product_id: string;
  evidence_id: string;
  occurred_at: string;
  event_type?: string;
}

export interface BehaviorAttributeInterest {
  attribute: string;
  label: string;
  score: number;
  distinct_products: number;
  evidence_count: number;
  source_event_types?: string[];
  evidence: BehaviorAttributeEvidence[];
}

export interface BehaviorProductInterest {
  product_id: string;
  product_title: string | null;
  score: number;
  positive_score: number;
  negative_score: number;
  counted_events: number;
  last_event_at: string | null;
}

export interface BehaviorRecentProduct {
  product_id: string;
  product_title: string | null;
  occurred_at: string;
  event_id: string;
}

export interface BehaviorComparedProduct {
  compare_id: string;
  product_id: string;
  product_title: string | null;
  compared_product_ids: string[];
  status: "attempted" | "succeeded" | "failed" | string;
  attempt_event_id: string | null;
  result_event_id: string | null;
  attempted_at: string | null;
  result_at: string | null;
  failure_code: string | null;
  failure_message: string | null;
  occurred_at: string;
}

export interface BehaviorCompareAttempt {
  compare_id: string;
  attempt_event_id: string | null;
  product_id: string;
  product_title: string | null;
  compared_product_ids: string[];
  status: string;
  attempted_at: string | null;
  failure_code: string | null;
}

export interface BehaviorRecentImpression {
  event_id: string;
  recommendation_id: string;
  product_id: string;
  product_title: string | null;
  displayed_position: number;
  occurred_at: string;
  clicked: boolean;
}

export interface BehaviorExplicitFeedback {
  event_id: string;
  product_id: string;
  product_title: string | null;
  feedback: "like" | "dislike" | string;
  occurred_at: string;
}

export interface BehaviorSuggestion {
  suggestion_id: string;
  attribute: string;
  label: string;
  score: number;
  distinct_products: number;
  evidence_count: number;
  evidence: BehaviorAttributeEvidence[];
  message: string;
  inferred_from_behavior: boolean;
  requires_confirmation: boolean;
  note: string;
}

export interface BehaviorProfile {
  schema: string;
  semantics_version?: string;
  session_id: string;
  enabled: boolean;
  generated_at: string;
  as_of?: string;
  window_seconds: number;
  half_life_seconds: number;
  counts: BehaviorCounts;
  source_counts: {
    stored_total: number;
    live_user_interaction: number;
    automated_browser_e2e?: number;
    recorded_replay: number;
    synthetic_test: number;
  };
  profile_sources?: string[];
  profile_provenance?: {
    human_live_events: number;
    automated_browser_e2e_events: number;
    human_preference_statistics: string;
  };
  signal_semantics?: Record<string, unknown>;
  recent_impressions: BehaviorRecentImpression[];
  recent_viewed: BehaviorRecentProduct[];
  recent_compared: BehaviorComparedProduct[];
  recent_compare_attempts: BehaviorCompareAttempt[];
  recent_explicit_feedback: BehaviorExplicitFeedback[];
  recent_categories: { category: string; score: number }[];
  attribute_interests: BehaviorAttributeInterest[];
  product_interests: BehaviorProductInterest[];
  recent_events: BehaviorEventView[];
  notes: string[];
  explicit_memory?: {
    status: "ok" | "disabled" | "unavailable" | string;
    active_count: number;
    inactive_count: number;
    error?: string | null;
    audit_source?: string | null;
  };
  suggestion?: BehaviorSuggestion | null;
  latest_recommendation_id?: string | null;
  behavior?: BehaviorConfig | null;
}

export interface BehaviorShadowMatchedAttribute {
  attribute: string;
  score: number;
  evidence: BehaviorAttributeEvidence[];
}

export interface BehaviorShadowRow {
  product_id: string;
  product_title: string | null;
  displayed_position: number;
  diagnostic_base_rank: number;
  diagnostic_base_score: number;
  diagnostic_rank: number;
  diagnostic_on_score: number;
  behavior_contribution: number;
  product_interest_signal: number;
  attribute_interest_signal: number;
  matched_attribute_interests: BehaviorShadowMatchedAttribute[];
  ignored_common_attributes?: { attribute: string; score: number; coverage: number }[];
  verified_attributes?: { attribute: string; value?: unknown; evidence_id?: string; span_text?: string | null }[];
  explicit_preference_conflict: boolean;
  suppressed_reason: string | null;
  recommended: boolean;
  gate_eligible: boolean;
  gate_selected: boolean;
  eligible_base_rank: number | null;
  eligible_base_score: number | null;
  eligible_behavior_contribution: number;
  eligible_on_score: number | null;
  eligible_rank: number | null;
}

export interface BehaviorShadowRankingRow {
  rank: number;
  product_id: string;
  product_title: string | null;
  base_score?: number;
  final_shadow_score?: number;
  behavior_contribution?: number;
  displayed_position?: number;
}

export interface BehaviorDiagnosticRankingRow {
  rank: number;
  diagnostic_base_rank: number;
  product_id: string;
  product_title: string | null;
  gate_eligible: boolean;
  diagnostic_base_score: number;
  behavior_contribution: number;
  diagnostic_on_score: number;
}

export interface BehaviorGateContext {
  action: string | null;
  allow_finalize: boolean;
  selected_candidate_id: string | null;
  candidate_presentable_ids: string[];
  eligible_candidate_ids: string[];
  eligible_candidate_count: number;
  eligibility_source: string;
  eligible_recommendation_available: boolean;
  no_eligible_reason: string | null;
  interpretation: string;
}

export interface BehaviorShadowChange {
  product_id: string;
  product_title: string | null;
  base_rank: number;
  shadow_rank: number;
  behavior_contribution: number;
  matched_attribute_interests: BehaviorShadowMatchedAttribute[];
  suppressed_reason: string | null;
  gate_eligible: boolean;
}

export interface BehaviorShadow {
  schema: string;
  semantics_version?: string;
  recommendation_id: string;
  request_id: string | null;
  query: string;
  created_at: string;
  generated_at: string;
  enabled: boolean;
  top_k: number;
  base_ranker_version: string;
  shadow_version: string;
  behavior_weight: number;
  explicit_preference_scale: number;
  gate: Partial<GateDecision> | null;
  gate_context: BehaviorGateContext;
  eligible_candidate_ids: string[];
  eligible_candidate_count: number;
  eligible_recommendation_available: boolean;
  no_eligible_reason: string | null;
  hard_constraint_violations: number;
  eligibility_unchanged: boolean;
  behavior_signal_present: boolean;
  eligible_behavior_applied: boolean;
  diagnostic_behavior_applied: boolean;
  applied: boolean;
  eligible_ranking_changed: boolean;
  diagnostic_ranking_changed: boolean;
  order_changed: boolean;
  eligible_base_ranking: BehaviorShadowRankingRow[];
  eligible_behavior_ranking: BehaviorShadowRankingRow[];
  diagnostic_ranking: BehaviorDiagnosticRankingRow[];
  diagnostic_candidates: BehaviorShadowRow[];
  eligible_changes: BehaviorShadowChange[];
  diagnostic_changes: BehaviorShadowChange[];
  ineligible_candidates_with_behavior_signal: BehaviorShadowChange[];
  eligible_ranking_ids: string[];
  eligible_base_ranking_ids: string[];
  diagnostic_ranking_ids: string[];
  active_preferences: { value: string; polarity: string; kind: string | null }[];
  inactive_preference_count: number;
  notes: string[];
  explicit_memory?: {
    status: "ok" | "disabled" | string;
    active_count: number;
    inactive_count: number;
  };
  behavior_profile?: {
    generated_at: string | null;
    as_of: string | null;
    counts: BehaviorCounts;
    attribute_interests: BehaviorAttributeInterest[];
    product_interests: BehaviorProductInterest[];
  };
  // V1 compatibility aliases; V1.1 makes these the eligible-only ranking.
  base_ranking: BehaviorShadowRankingRow[];
  behavior_ranking: BehaviorShadowRankingRow[];
  rows: BehaviorShadowRow[];
  changes: BehaviorShadowChange[];
}

export interface BehaviorEventsResponse {
  session_id: string;
  accepted: BehaviorEventView[];
  duplicates: string[];
  rejected: { event_id: string | null; code: string; message: string; detail?: unknown }[];
  accepted_count: number;
  duplicate_count: number;
  rejected_count: number;
}

export interface BehaviorEnabledResponse {
  session_id: string;
  enabled: boolean;
  behavior: BehaviorConfig;
}

export interface BehaviorDeleteResponse {
  session_id: string;
  deleted_events: number;
  enabled: boolean;
  message: string;
}

// ---------------------------------------------------------------------------
// SSE events
// ---------------------------------------------------------------------------

export type ServerEvent =
  | { type: "tool_status"; data: ToolStatus }
  | { type: "text"; data: { text: string; final?: boolean } }
  | { type: "products"; data: ProductsPayload }
  | { type: "evidence"; data: EvidencePayload }
  | { type: "comparison"; data: ComparisonPayload }
  | { type: "clarification"; data: ClarificationPayload }
  | { type: "memory_update"; data: MemoryUpdatePayload }
  | { type: "done"; data: DonePayload }
  | { type: "error"; data: ErrorPayload };

export interface ServerEventEnvelope {
  type: string;
  data: unknown;
}
