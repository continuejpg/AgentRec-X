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

export interface Health {
  status: ServiceStatus;
  ready: boolean;
  started_at: string | null;
  uptime_s: number | null;
  catalogue_size: number | null;
  servers: HealthServer[];
  modes: Partial<Record<ModeKey, ModeInfo>>;
  memory_writes_enabled?: boolean;
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
}

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
}

export interface SessionView {
  session_id: string;
  profile_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: TurnView[];
  pending_clarification: ClarificationPayload | null;
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

// ---------------------------------------------------------------------------
// Chat SSE view models
// ---------------------------------------------------------------------------

export type ToolRunStatus = "running" | "ok" | "error";

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
}

export interface ClarificationReplyResponse {
  turn_id: string;
  memory_update: MemoryUpdatePayload;
  preferences: PreferenceState;
  recommendation: ProductsPayload | null;
  steps: ToolStatus[];
  clarification_id: string;
  message: string;
  error: ErrorPayload | null;
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
