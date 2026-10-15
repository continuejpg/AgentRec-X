# AgentRec-X V2 / E-03C-SCHEMA-INTEGRITY — 公开 Schema 清单（自动生成）

> 生成方式：直接读取运行中的 pydantic 模型对象（`model_fields` / `model_config` / `__all__`），
> **不是**从源码文本抄写。生成命令见
> [`docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md`](V2_E03C_SCHEMA_INTEGRITY_REPORT.md) §9。
> 覆盖四个契约模块：`demo/schemas.py`（Chat 契约）、`demo/agent_schemas.py`（Agent/推荐契约）、
> `memory/schemas.py`、`preference_matching/schemas.py`。
>
> 说明：本文件证明**当前**公开面的形状（字段名/类型/required/default/约束），是本轮核验的原始记录；
> 它**不能**证明与事故前文件的字节级一致（见报告 §2/§4）。

```text


# demo.schemas: 28 models

## demo.schemas.CreateSessionRequest
   config: {'extra': 'forbid', 'frozen': True}
   - profile_id: <class 'str'> [default='demo-user-1'] min_length=1 max_length=200
   - decision_mode: Literal['deterministic', 'llm', 'llm_agent'] [default='deterministic']

## demo.schemas.ChatRequest
   config: {'extra': 'forbid', 'frozen': True}
   - message: <class 'str'> [required] min_length=1 max_length=2000
   - k: <class 'int'> [default=5] ge=1 le=100
   - decision_mode: Optional[Literal['deterministic', 'llm', 'llm_agent']] [default=None]

## demo.schemas.DemoProfileView
   config: {'extra': 'forbid', 'frozen': True}
   - profile_id: <class 'str'> [required]
   - display_name: <class 'str'> [required]
   - history_length: <class 'int'> [required] ge=0
   - history_distinct: <class 'int'> [required] ge=0

## demo.schemas.SessionResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - session_id: <class 'str'> [required]
   - profile: <class 'demo.schemas.DemoProfileView'> [required]
   - turn: <class 'int'> [required] ge=0
   - decision_mode: Literal['deterministic', 'llm', 'llm_agent'] [default='deterministic']

## demo.schemas.PreferenceMutationView
   config: {'extra': 'forbid', 'frozen': True}
   - kind: <class 'str'> [required]
   - polarity: <class 'str'> [required]
   - value: <class 'str'> [required]

## demo.schemas.MemoryUpdateView
   config: {'extra': 'forbid', 'frozen': True}
   - changed: <class 'bool'> [default=False]
   - added: tuple[demo.schemas.PreferenceMutationView, ...] [default=()]
   - superseded: tuple[demo.schemas.PreferenceMutationView, ...] [default=()]
   - removed: tuple[demo.schemas.PreferenceMutationView, ...] [default=()]
   - skipped_duplicates: <class 'int'> [default=0] ge=0
   - already_processed: <class 'bool'> [default=False]
   - removal_directives: <class 'int'> [default=0] ge=0

## demo.schemas.ActivePreferenceView
   config: {'extra': 'forbid', 'frozen': True}
   - kind: <class 'str'> [required]
   - polarity: <class 'str'> [required]
   - value: <class 'str'> [required]

## demo.schemas.EvidenceView
   config: {'extra': 'forbid', 'frozen': True}
   - kind: <class 'str'> [required]
   - polarity: <class 'str'> [required]
   - value: <class 'str'> [required]
   - status: Literal['match', 'violation', 'unknown'] [required]
   - metadata_field: str | None [default=None]
   - metadata_value: str | None [default=None]
   - detail: str | None [default=None]

## demo.schemas.ProductMetadataView
   config: {'extra': 'forbid', 'frozen': True}
   - title: str | None [default=None]
   - store: str | None [default=None]
   - main_category: str | None [default=None]
   - price_text: str | None [default=None]
   - categories: tuple[str, ...] [default=()]
   - details: tuple[tuple[str, str], ...] [default=()]

## demo.schemas.RecommendationCard
   config: {'extra': 'forbid', 'frozen': True}
   - reranked_rank: int | None [default=None] ge=1
   - original_rank: <class 'int'> [required] ge=1
   - parent_asin: <class 'str'> [required]
   - item_id: <class 'int'> [required]
   - sasrec_score: <class 'float'> [required]
   - score_kind: str | None [default=None]
   - fusion_sources: tuple[str, ...] [default=()]
   - fusion_source_ranks: dict[str, int] [default_factory=dict]
   - fused_rank: int | None [default=None]
   - metadata_status: Literal['found', 'missing'] [required]
   - metadata: demo.schemas.ProductMetadataView | None [default=None]
   - match_count: <class 'int'> [default=0] ge=0
   - violation_count: <class 'int'> [default=0] ge=0
   - unknown_count: <class 'int'> [default=0] ge=0
   - evidence: tuple[demo.schemas.EvidenceView, ...] [default=()]
   - fallback_reason: str | None [default=None]
   - movement_summary: str | None [default=None]
   - reason: str | None [default=None]
   - reason_status: <class 'str'> [default='not_attempted']
   - knowledge_evidence: ForwardRef('tuple[KnowledgeEvidenceView, ...]') [default=()]
   - feedback: str | None [default=None]

## demo.schemas.AuditView
   config: {'extra': 'forbid', 'frozen': True}
   - reranking_applied: <class 'bool'> [required]
   - candidate_count: <class 'int'> [required] ge=0
   - moved_count: <class 'int'> [required] ge=0
   - original_order: tuple[str, ...] [default=()]
   - reranked_order: tuple[str, ...] [default=()]
   - ranked_with_preference_count: <class 'int'> [default=0] ge=0
   - ranked_with_preferences: tuple[demo.schemas.ActivePreferenceView, ...] [default=()]
   - rag: ForwardRef('RagAuditView | None') [default=None]
   - replan: ForwardRef('ReplanAuditView | None') [default=None]
   - feedback: ForwardRef('ProductFeedbackView | None') [default=None]

## demo.schemas.KnowledgeEvidenceView
   config: {'extra': 'forbid', 'frozen': True}
   - evidence_id: <class 'str'> [required]
   - parent_asin: <class 'str'> [required]
   - item_id: <class 'int'> [required]
   - field_path: <class 'str'> [required]
   - value: <class 'str'> [required]
   - source: <class 'str'> [default='catalog_metadata']
   - retrieval_score: <class 'float'> [default=0.0]
   - truncated: <class 'bool'> [default=False]

## demo.schemas.RagAuditView
   config: {'extra': 'forbid', 'frozen': True}
   - enabled: <class 'bool'> [default=False]
   - model_called: <class 'bool'> [default=False]
   - products_examined: <class 'int'> [default=0] ge=0
   - products_with_evidence: <class 'int'> [default=0] ge=0
   - evidence_units: <class 'int'> [default=0] ge=0
   - explanations_accepted: <class 'int'> [default=0] ge=0
   - explanations_rejected: <class 'int'> [default=0] ge=0
   - explanations_abstained: <class 'int'> [default=0] ge=0
   - failure: str | None [default=None]
   - reason_source: str | None [default='template_from_validated_claims']

## demo.schemas.ReplanAuditView
   config: {'extra': 'forbid', 'frozen': True}
   - continue_recommended: <class 'bool'> [default=False]
   - reason_code: <class 'str'> [default='']
   - eligible_sources: tuple[str, ...] [default=()]
   - remaining_tool_budget: <class 'int'> [default=0] ge=0
   - remaining_step_budget: <class 'int'> [default=0] ge=0
   - decision_source: <class 'str'> [default='deterministic_rules']
   - status: str | None [default=None]
   - model_called: <class 'bool'> [default=False]
   - chosen_source: str | None [default=None]
   - initial_candidate_count: <class 'int'> [default=0] ge=0
   - final_candidate_count: <class 'int'> [default=0] ge=0

## demo.schemas.ProductFeedbackView
   config: {'extra': 'forbid', 'frozen': True}
   - status: <class 'str'> [required]
   - action: <class 'str'> [default='none']
   - reference: <class 'str'> [default='']
   - span: <class 'str'> [default='']
   - detail: <class 'str'> [default='']
   - parent_asin: str | None [default=None]
   - item_id: int | None [default=None]
   - final_position: int | None [default=None]
   - written: <class 'bool'> [default=False]
   - entry_status: str | None [default=None]
   - effect: str | None [default=None]
   - active_product_feedback: tuple[dict[str, str], ...] [default=()]
   - active_before: tuple[dict[str, str], ...] [default=()]

## demo.schemas.TraceMemoryChanges
   config: {'extra': 'forbid', 'frozen': True}
   - changed: <class 'bool'> [default=False]
   - added: tuple[str, ...] [default=()]
   - removed: tuple[str, ...] [default=()]
   - superseded: tuple[str, ...] [default=()]

## demo.schemas.TraceEvidenceSummary
   config: {'extra': 'forbid', 'frozen': True}
   - match_candidates: <class 'int'> [default=0] ge=0
   - violation_candidates: <class 'int'> [default=0] ge=0
   - unknown_candidates: <class 'int'> [default=0] ge=0

## demo.schemas.TraceRankChange
   config: {'extra': 'forbid', 'frozen': True}
   - parent_asin: <class 'str'> [required]
   - title: str | None [default=None]
   - original_rank: <class 'int'> [required] ge=1
   - final_rank: <class 'int'> [required] ge=1
   - moved: <class 'bool'> [default=False]

## demo.schemas.TraceGroundingView
   config: {'extra': 'forbid', 'frozen': True}
   - verified: <class 'int'> [default=0] ge=0
   - total: <class 'int'> [default=0] ge=0

## demo.schemas.TracePreferenceAction
   config: {'extra': 'forbid', 'frozen': True}
   - action: Literal['add', 'remove'] [required]
   - value: <class 'str'> [required]
   - kind: str | None [default=None]
   - applied: <class 'bool'> [default=False]

## demo.schemas.DecisionModeView
   config: {'extra': 'forbid', 'frozen': True}
   - mode: Literal['deterministic', 'llm', 'llm_agent'] [required]
   - label: <class 'str'> [required]
   - available: <class 'bool'> [required]
   - reason: str | None [default=None]
   - detail: str | None [default=None]
   - provider: str | None [default=None]
   - model: str | None [default=None]

## demo.schemas.DecisionModesResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - default: Literal['deterministic', 'llm', 'llm_agent'] [default='deterministic']
   - modes: tuple[demo.schemas.DecisionModeView, ...] [default=()]

## demo.schemas.RecommendationTrace
   config: {'extra': 'forbid', 'frozen': True}
   - route: Literal['direct', 'recommend'] [required]
   - candidate_count: int | None [default=None] ge=0
   - decision_mode: <class 'str'> [default='deterministic']
   - provider: str | None [default=None]
   - model: str | None [default=None]
   - proposed_route: str | None [default=None]
   - preference_actions: tuple[demo.schemas.TracePreferenceAction, ...] [default=()]
   - active_preferences: tuple[demo.schemas.ActivePreferenceView, ...] [default=()]
   - memory_changes: <class 'demo.schemas.TraceMemoryChanges'> [default_factory=TraceMemoryChanges]
   - evidence: demo.schemas.TraceEvidenceSummary | None [default=None]
   - ranking_changes: tuple[demo.schemas.TraceRankChange, ...] [default=()]
   - source: str | None [default=None]
   - grounding: demo.schemas.TraceGroundingView | None [default=None]

## demo.schemas.ChatResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - session_id: <class 'str'> [required]
   - turn_id: <class 'str'> [required]
   - turn: <class 'int'> [required] ge=1
   - route: Literal['direct', 'recommend'] [required]
   - message: <class 'str'> [required]
   - active_preferences: tuple[demo.schemas.ActivePreferenceView, ...] [default=()]
   - memory_update: <class 'demo.schemas.MemoryUpdateView'> [default_factory=MemoryUpdateView]
   - recommendations: tuple[demo.schemas.RecommendationCard, ...] [default=()]
   - agent: demo.agent_schemas.AgentExecutionView | None [default=None]
   - audit: <class 'demo.schemas.AuditView'> [required]
   - trace: demo.schemas.RecommendationTrace | None [default=None]

## demo.schemas.SessionStateResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - session_id: <class 'str'> [required]
   - profile: <class 'demo.schemas.DemoProfileView'> [required]
   - turn: <class 'int'> [required] ge=0
   - decision_mode: Literal['deterministic', 'llm', 'llm_agent'] [default='deterministic']
   - active_preferences: tuple[demo.schemas.ActivePreferenceView, ...] [default=()]
   - active_preference_count: <class 'int'> [default=0] ge=0

## demo.schemas.DemoHealthResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - status: <class 'str'> [required]
   - model_loaded: <class 'bool'> [required]
   - metadata_loaded: <class 'bool'> [required]
   - demo_ready: <class 'bool'> [required]
   - profiles: <class 'int'> [default=0] ge=0
   - active_sessions: <class 'int'> [default=0] ge=0
   - max_sessions: <class 'int'> [default=0] ge=0
   - detail: str | None [default=None]

## demo.schemas.ProfileListResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - profiles: tuple[demo.schemas.DemoProfileView, ...] [default=()]

## demo.schemas.ResetResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - session_id: <class 'str'> [required]
   - reset: <class 'bool'> [default=True]
   - detail: <class 'str'> [required]

# demo.agent_schemas: 16 models

## demo.agent_schemas.AgentConstraintItem
   config: {'extra': 'forbid', 'frozen': True}
   - kind: <class 'str'> [required]
   - value: <class 'str'> [required] min_length=1 max_length=120

## demo.agent_schemas.AgentRecommendRequest
   config: {'extra': 'forbid', 'frozen': True}
   - message: <class 'str'> [required] min_length=1 max_length=2000
   - history: list[Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[MinLen(min_length=1), MaxLen(max_length=200)])]] [required] min_length=1 max_length=500
   - k: <class 'int'> [default=5] ge=1 le=100
   - user_key: Optional[Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[MinLen(min_length=1), MaxLen(max_length=200)])]] [default=None]
   - sources: tuple[str, ...] [default=('history',)]
   - constraints: tuple[demo.agent_schemas.AgentConstraintItem, ...] [default=()] max_length=8

## demo.agent_schemas.AgentRecommendationView
   config: {'extra': 'forbid', 'frozen': True}
   - rank: <class 'int'> [required] ge=1
   - item_id: <class 'int'> [required] ge=1
   - parent_asin: <class 'str'> [required] min_length=1 max_length=200
   - score: <class 'float'> [required]
   - score_kind: str | None [default=None]
   - title: str | None [default=None]
   - sources: tuple[str, ...] [default=()]
   - reranked_rank: int | None [default=None] ge=1
   - match_count: <class 'int'> [default=0] ge=0
   - violation_count: <class 'int'> [default=0] ge=0
   - unknown_count: <class 'int'> [default=0] ge=0
   - evidence: tuple[preference_matching.schemas.PreferenceEvidence, ...] [default=()]
   - rerank_reason: str | None [default=None]

## demo.agent_schemas.AgentObservationView
   config: {'extra': 'forbid', 'frozen': True}
   - run_ref: <class 'str'> [required]
   - requested_k: <class 'int'> [required] ge=0
   - plan: tuple[str, ...] [default=()]
   - executed_sources: tuple[str, ...] [default=()]
   - unexecuted_sources: tuple[str, ...] [default=()]
   - source_calls: dict[str, int] [default_factory=dict]
   - source_outcomes: dict[str, str] [default_factory=dict]
   - failed_sources: tuple[str, ...] [default=()]
   - empty_sources: tuple[str, ...] [default=()]
   - grounded_count: <class 'int'> [default=0] ge=0
   - ungrounded_count: <class 'int'> [default=0] ge=0
   - multi_source_count: <class 'int'> [default=0] ge=0
   - presented_count: <class 'int'> [default=0] ge=0
   - candidates_sufficient: <class 'bool'> [default=False]
   - termination_status: <class 'str'> [default='']
   - termination_reason: str | None [default=None]
   - steps_used: <class 'int'> [default=0] ge=0
   - steps_limit: <class 'int'> [default=0] ge=0
   - tool_calls_used: <class 'int'> [default=0] ge=0
   - tool_calls_limit: <class 'int'> [default=0] ge=0
   - retries_used: <class 'int'> [default=0] ge=0
   - remaining_steps: <class 'int'> [default=0] ge=0
   - remaining_tool_calls: <class 'int'> [default=0] ge=0
   - has_further_budget: <class 'bool'> [default=False]
   - eligibility_evaluated: <class 'bool'> [default=False]
   - eligible_candidate_count: <class 'int'> [default=0] ge=0

## demo.agent_schemas.ReplanDecisionView
   config: {'extra': 'forbid', 'frozen': True}
   - continue_recommended: <class 'bool'> [required]
   - reason_code: <class 'str'> [required]
   - eligible_sources: tuple[str, ...] [default=()]
   - remaining_tool_budget: <class 'int'> [default=0] ge=0
   - remaining_step_budget: <class 'int'> [default=0] ge=0
   - decision_source: <class 'str'> [default='deterministic_rules']
   - explanation: str | None [default=None] max_length=280

## demo.agent_schemas.ReplanObservationView
   config: {'extra': 'forbid', 'frozen': True}
   - candidates: <class 'int'> [default=0] ge=0
   - sources_consulted: tuple[str, ...] [default=()]
   - source_outcomes: dict[str, str] [default_factory=dict]
   - remaining_tool_calls: <class 'int'> [default=0] ge=0
   - remaining_steps: <class 'int'> [default=0] ge=0

## demo.agent_schemas.AgentReplanView
   config: {'extra': 'forbid', 'frozen': True}
   - status: <class 'str'> [required]
   - reason_code: <class 'str'> [default='']
   - model_called: <class 'bool'> [default=False]
   - chosen_source: str | None [default=None]
   - eligible_sources: tuple[str, ...] [default=()]
   - decision_step: int | None [default=None]
   - initial: <class 'demo.agent_schemas.ReplanObservationView'> [default_factory=ReplanObservationView]
   - final: <class 'demo.agent_schemas.ReplanObservationView'> [default_factory=ReplanObservationView]

## demo.agent_schemas.AgentCandidateView
   config: {'extra': 'forbid', 'frozen': True}
   - parent_asin: <class 'str'> [required] min_length=1 max_length=200
   - grounded: <class 'bool'> [required]
   - verification_status: <enum 'CandidateVerificationStatus'> [required]
   - provenance: tuple[control.candidate_ledger.CandidateProvenance, ...] [default=()]

## demo.agent_schemas.GroundedCandidateView
   config: {'extra': 'forbid', 'frozen': True}
   - grounded_count: <class 'int'> [required] ge=0
   - ungrounded_count: <class 'int'> [required] ge=0
   - multi_source_count: <class 'int'> [required] ge=0
   - sources_present: tuple[str, ...] [default=()]
   - candidates: tuple[demo.agent_schemas.AgentCandidateView, ...] [default=()]

## demo.agent_schemas.AgentEligibilityView
   config: {'extra': 'forbid', 'frozen': True}
   - evaluated: <class 'bool'> [required]
   - requirements: tuple[dict[str, object], ...] [default=()]
   - verified_eligible_count: <class 'int'> [default=0] ge=0
   - ineligible_count: <class 'int'> [default=0] ge=0
   - unresolved_count: <class 'int'> [default=0] ge=0
   - assessments: tuple[dict[str, object], ...] [default=()]
   - projection: dict[str, object] | None [default=None]

## demo.agent_schemas.AgentTerminalView
   config: {'extra': 'forbid', 'frozen': True}
   - run_id: <class 'str'> [required] min_length=1 max_length=200
   - status: <class 'str'> [required]
   - termination_reason: str | None [default=None]
   - termination_detail: str | None [default=None] max_length=280
   - succeeded: <class 'bool'> [required]
   - steps: <class 'int'> [required] ge=0
   - tool_calls: <class 'int'> [required] ge=0
   - retries: <class 'int'> [required] ge=0
   - limits: dict[str, int] [default_factory=dict]

## demo.agent_schemas.TrajectoryStepView
   config: {'extra': 'forbid', 'frozen': True}
   - step: <class 'int'> [required] ge=0
   - action: str | None [default=None]
   - validated: <class 'bool'> [required]
   - observation_type: str | None [default=None]
   - candidate_count: <class 'int'> [default=0] ge=0
   - eligibility_state: str | None [default=None]
   - refusal_codes: tuple[str, ...] [default=()]
   - note: str | None [default=None] max_length=280

## demo.agent_schemas.TrajectorySummaryView
   config: {'extra': 'forbid', 'frozen': True}
   - step_count: <class 'int'> [required] ge=0
   - actions: tuple[str, ...] [default=()]
   - refusals: tuple[str, ...] [default=()]
   - steps: tuple[demo.agent_schemas.TrajectoryStepView, ...] [default=()]

## demo.agent_schemas.AgentTimingView
   config: {'extra': 'forbid', 'frozen': True}
   - total_ms: <class 'float'> [required] ge=0.0
   - agent_ms: <class 'float'> [required] ge=0.0
   - serialize_ms: <class 'float'> [required] ge=0.0

## demo.agent_schemas.AgentExecutionView
   config: {'extra': 'forbid', 'frozen': True}
   - outcome: Literal['direct', 'recommended', 'no_candidates', 'refused', 'failed'] [required]
   - planned_sources: tuple[str, ...] [default=()]
   - controller_executed: <class 'bool'> [required]
   - action_counts: dict[str, int] [default_factory=dict]
   - terminal: demo.agent_schemas.AgentTerminalView | None [default=None]
   - trajectory: demo.agent_schemas.TrajectorySummaryView | None [default=None]
   - grounded: demo.agent_schemas.GroundedCandidateView | None [default=None]
   - timing: demo.agent_schemas.AgentTimingView | None [default=None]

## demo.agent_schemas.AgentRecommendResponse
   config: {'extra': 'forbid', 'frozen': True}
   - api_version: <class 'str'> [default='1.0.0']
   - route: Literal['direct', 'recommend'] [required]
   - message: <class 'str'> [required]
   - control_plane: <class 'str'> [required]
   - recommendations: tuple[demo.agent_schemas.AgentRecommendationView, ...] [default=()]
   - grounded: <class 'demo.agent_schemas.GroundedCandidateView'> [required]
   - eligibility: <class 'demo.agent_schemas.AgentEligibilityView'> [required]
   - terminal: <class 'demo.agent_schemas.AgentTerminalView'> [required]
   - trajectory: <class 'demo.agent_schemas.TrajectorySummaryView'> [required]
   - observation: demo.agent_schemas.AgentObservationView | None [default=None]
   - replan: demo.agent_schemas.ReplanDecisionView | None [default=None]
   - replan_execution: demo.agent_schemas.AgentReplanView | None [default=None]
   - timing: <class 'demo.agent_schemas.AgentTimingView'> [default_factory=<lambda>]

# memory.schemas: 5 models

## memory.schemas.PreferenceCandidate
   config: {'extra': 'forbid', 'frozen': True}
   - kind: <enum 'PreferenceKind'> [required]
   - value: <class 'str'> [required] min_length=1
   - polarity: <enum 'PreferencePolarity'> [default=<PreferencePolarity.PREFER: 'prefer'>]
   - source_text: <class 'str'> [required] min_length=1
   - extractor: <class 'str'> [default='unspecified'] min_length=1
   - mode: <enum 'PreferenceMode'> [default=<PreferenceMode.ADD: 'add'>]
   - replaces: str | None [default=None]

## memory.schemas.PreferenceRemoval
   config: {'extra': 'forbid', 'frozen': True}
   - kind: memory.schemas.PreferenceKind | None [default=None]
   - value: str | None [default=None]
   - source_text: <class 'str'> [required] min_length=1
   - extractor: <class 'str'> [default='unspecified'] min_length=1
   - mode: <enum 'PreferenceMode'> [default=<PreferenceMode.REMOVE: 'remove'>]

## memory.schemas.PreferenceExtraction
   config: {'extra': 'forbid', 'frozen': True}
   - preferences: tuple[memory.schemas.PreferenceCandidate, ...] [default=()]
   - removals: tuple[memory.schemas.PreferenceRemoval, ...] [default=()]

## memory.schemas.PreferenceMemoryEntry
   config: {'extra': 'forbid', 'frozen': True}
   - memory_id: <class 'str'> [required] min_length=1
   - user_key: <class 'str'> [required] min_length=1
   - kind: <enum 'PreferenceKind'> [required]
   - value: <class 'str'> [required] min_length=1
   - polarity: <enum 'PreferencePolarity'> [default=<PreferencePolarity.PREFER: 'prefer'>]
   - source_text: <class 'str'> [required] min_length=1
   - source_turn_id: <class 'str'> [required] min_length=1
   - extractor: <class 'str'> [default='unspecified'] min_length=1
   - status: <enum 'PreferenceStatus'> [default=<PreferenceStatus.ACTIVE: 'active'>]
   - logical_seq: <class 'int'> [required] ge=1
   - created_at: <class 'float'> [required]
   - supersedes: str | None [default=None]
   - superseded_by: str | None [default=None]

## memory.schemas.PreferenceMemorySnapshot
   config: {'extra': 'forbid', 'frozen': True}
   - user_key: <class 'str'> [required] min_length=1
   - entries: tuple[memory.schemas.PreferenceMemoryEntry, ...] [default=()]
   - read_at: <class 'float'> [default_factory=time]

# preference_matching.schemas: 4 models

## preference_matching.schemas.PreferenceEvidence
   config: {'extra': 'forbid', 'frozen': True}
   - preference_id: <class 'str'> [required]
   - preference_kind: <enum 'PreferenceKind'> [required]
   - preference_polarity: <enum 'PreferencePolarity'> [required]
   - preference_value: <class 'str'> [required]
   - preference_source_text: <class 'str'> [required]
   - preference_source_turn_id: <class 'str'> [required]
   - preference_logical_seq: <class 'int'> [required] ge=1
   - status: <enum 'EvidenceStatus'> [required]
   - reason_code: <enum 'ReasonCode'> [required]
   - support: <enum 'MatchingSupport'> [required]
   - metadata_field: str | None [default=None]
   - metadata_value: str | None [default=None]
   - metadata_present: <class 'bool'> [default=False]
   - detail: str | None [default=None]

## preference_matching.schemas.CandidatePreferenceEvidence
   config: {'extra': 'forbid', 'frozen': True}
   - original_rank: <class 'int'> [required] ge=1
   - item_id: <class 'int'> [required]
   - parent_asin: <class 'str'> [required]
   - sasrec_score: <class 'float'> [required]
   - evidence: tuple[preference_matching.schemas.PreferenceEvidence, ...] [default=()]

## preference_matching.schemas.CandidateEvidenceCounts
   config: {'extra': 'forbid', 'frozen': True}
   - match_count: <class 'int'> [default=0] ge=0
   - violation_count: <class 'int'> [default=0] ge=0
   - unknown_count: <class 'int'> [default=0] ge=0

## preference_matching.schemas.PreferenceEvidenceReport
   config: {'extra': 'forbid', 'frozen': True}
   - candidates: tuple[preference_matching.schemas.CandidatePreferenceEvidence, ...] [default=()]
   - active_preference_count: <class 'int'> [default=0] ge=0
   - counts: <class 'preference_matching.schemas.CandidateEvidenceCounts'> [default_factory=CandidateEvidenceCounts]

# literals, enums and module constants (chat + agent)
   demo.schemas.DECISION_MODES = ('deterministic', 'llm', 'llm_agent')
   demo.schemas.DECISION_MODE_DETERMINISTIC = 'deterministic'
   demo.schemas.DECISION_MODE_LLM = 'llm'
   demo.schemas.DECISION_MODE_LLM_AGENT = 'llm_agent'
   demo.schemas.DEFAULT_DECISION_MODE = 'deterministic'
   demo.schemas.DEMO_API_VERSION = '1.0.0'
   demo.schemas.MAX_K = 100
   demo.schemas.MAX_MESSAGE_LENGTH = 2000
   demo.schemas.MIN_K = 1
   demo.schemas.DecisionMode = typing.Literal['deterministic', 'llm', 'llm_agent']
   demo.schemas.StrictK = typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Strict(strict=True), Ge(ge=1), Le(le=100)])]
   demo.schemas.Message = typing.Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[Strict(strict=True), MinLen(min_length=1), MaxLen(max_length=2000)])]
   demo.schemas.OpaqueId = typing.Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[Strict(strict=True), MinLen(min_length=1), MaxLen(max_length=200)])]
   demo.agent_schemas.AGENT_API_VERSION = '1.0.0'
   demo.agent_schemas.AGENT_OUTCOMES = ('direct', 'recommended', 'no_candidates', 'refused', 'failed')
   demo.agent_schemas.MAX_CONSTRAINTS = 8
   demo.agent_schemas.MAX_HISTORY_ITEMS = 500
   demo.agent_schemas.MAX_K = 100
   demo.agent_schemas.MIN_K = 1
   demo.agent_schemas.StrictK = typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Strict(strict=True), Ge(ge=1), Le(le=100)])]
   demo.agent_schemas.Message = typing.Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[Strict(strict=True), MinLen(min_length=1), MaxLen(max_length=2000)])]
   demo.agent_schemas.OpaqueId = typing.Annotated[str, FieldInfo(annotation=NoneType, required=True, metadata=[MinLen(min_length=1), MaxLen(max_length=200)])]

# __all__ sizes
   demo.schemas: 27 exported names
   demo.agent_schemas: 21 exported names
   memory.schemas: 15 exported names
   preference_matching.schemas: 16 exported names
```
