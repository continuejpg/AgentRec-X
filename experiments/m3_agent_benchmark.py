"""M3 execution runner — the formal entrypoint for the preregistered Agent evaluation.

    .venv/bin/python -m experiments.m3_agent_benchmark --preflight
    .venv/bin/python -m experiments.m3_agent_benchmark --execute

Two modes, and no third:

``--preflight``
    A provider smoke on **non-cohort** inputs. It never loads the 20 000-user cohort, never
    builds the Agent arm and never reaches the evaluator, so it cannot compute a
    recommendation metric. It exists to measure calls-per-user, latency, retry and failure
    rates against the frozen provider configuration before the one paid run.

``--execute``
    The single formal 20 000-user M3 evaluation. It refuses to start unless every hard check
    passes, runs the Agent arm once through the unmodified ``evaluate_batched``, and writes
    the paired evaluation inputs and sidecars the final statistics need.

Reuses, without modification: ``experiments/m3_agent_arm.py``, ``experiments/m3_agent_harness.py``,
the frozen 20 000-user cohort, ``runs/m3_evidence/phase5_fusion_source_heads.npz``, the accepted
``fixed_fusion`` comparator, and ``recommendation.evaluation.batched.evaluate_batched``. The policy,
the prompt, the cohort, the comparator, ``LoopLimits`` and prereg §2-§6 are untouched.

Fail-closed: any hard-check failure raises :class:`HardCheckFailure` and the process stops before
any model call and before any evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]

__all__ = [
    "HardCheckFailure",
    "EXPECTED",
    "FrozenInputs",
    "artifact_checks",
    "cohort_checks",
    "load_frozen_inputs",
    "run_preflight",
    "run_execute",
    "main",
]

# --------------------------------------------------------------------------- #
# the frozen configuration this runner asserts against
# --------------------------------------------------------------------------- #

EXPECTED: dict[str, Any] = {
    "cohort": 20_000,
    "catalogue": 156_746,
    "k_values": (5, 10, 20),
    "batch_size": 256,
    "limits": (6, 4, 1),  # max_steps, max_tool_calls, max_retries
    "fusion_sources": ("popularity", "sequential", "metadata"),
    "rrf_k": 60,
    "head": 1000,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 20260201,
    "artifacts": {
        "runs/m3_evidence/phase5_fusion_source_heads.npz":
            "c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4",
        "runs/m3_evidence/phase5_fixed_fusion_hits.npz":
            "62c2bc7c8013dfd651e4773e6dc328ae645e49f42af59d71fa421834a464b23f",
        "runs/m3_evidence/phase5_popularity_hits.npz":
            "f7719c1dc843472a8f66dd2eb788cc3dc155e638a56b522b425501be8e4ad44c",
        "runs/step26_baseline_evidence/phase5_sasrec_hits.npz":
            "3072c86055cd4d093147ef10e19ed8121fb70215d581198b3261380df83d0252",
    },
    "prereg": "docs/M3_PREREGISTRATION.md",
}

#: The provider configuration frozen by docs/M3_DEEPSEEK_AMENDMENT.md. The runner does not
#: inherit these from the environment: it FORCES them, and then hard-checks the payload that
#: would actually go out. The endpoint (base URL and model) is read from the environment
#: because the amendment fixes its value, not its storage.
FROZEN_PROVIDER: dict[str, Any] = {
    "profile": "deepseek",
    "thinking": False,
    "temperature": 0.0,
    "json_mode": True,
    "max_tokens": 512,
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
}

SEQUENCES = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO / "data/processed/Sports_and_Outdoors_products.jsonl"
PROJECT_STATE = REPO / "docs/PROJECT_STATE.md"

from recommendation.control.provider_adapter import provider_settings  # noqa: E402
DEFAULT_OUT = REPO / "runs" / "m3_execution"


class HardCheckFailure(RuntimeError):
    """A hard check failed. The run must stop before any model call or evaluation."""


def m3_provider_settings() -> Any:
    """The settings the M3 runner uses: endpoint from the environment, the rest FORCED.

    Forcing the profile matters: ``AGENTRECX_LLM_PROFILE`` unset resolves to
    ``openai_compatible``, which never sends ``thinking`` at all — and DeepSeek enables
    thinking by default, which would ignore the frozen ``temperature`` and return
    ``reasoning_content``. The amendment froze the ``deepseek`` profile, so the runner
    selects it regardless of the environment.
    """
    import dataclasses

    from recommendation.control.provider_adapter import PROVIDER_PROFILES, ProviderSettings

    profile = PROVIDER_PROFILES[FROZEN_PROVIDER["profile"]]
    try:
        base = provider_settings()
    except Exception:
        return ProviderSettings(
            base_url="",
            model="",
            profile=profile,
            timeout=30.0,
            json_mode=bool(FROZEN_PROVIDER["json_mode"]),
            thinking=bool(FROZEN_PROVIDER["thinking"]),
            api_key_present=False,
            input_price_per_million=None,
            output_price_per_million=None,
        )
    return dataclasses.replace(
        base,
        profile=profile,
        json_mode=bool(FROZEN_PROVIDER["json_mode"]),
        thinking=bool(FROZEN_PROVIDER["thinking"]),
    )


def build_m3_provider_client(*, transport: Any = None) -> Any:
    """Build the M3 provider client with the frozen profile and sampling configuration."""
    from recommendation.control.provider_adapter import build_provider_client

    settings = m3_provider_settings()
    return build_provider_client(
        settings=settings,
        transport=transport,
        temperature=FROZEN_PROVIDER["temperature"],
        max_tokens=FROZEN_PROVIDER["max_tokens"],
    )


def provider_config_checks(settings: Any = None) -> list[str]:
    """Verify the request the M3 client would actually send matches the frozen amendment.

    Checked against the real profile's ``build_payload``, not against the constants, so a
    profile that silently ignored a field would be caught.
    """
    resolved = settings if settings is not None else m3_provider_settings()
    problems: list[str] = []

    if resolved.profile.name != FROZEN_PROVIDER["profile"]:
        problems.append(
            f"provider profile is {resolved.profile.name!r}, frozen is "
            f"{FROZEN_PROVIDER['profile']!r}"
        )
    if bool(resolved.thinking) is not bool(FROZEN_PROVIDER["thinking"]):
        problems.append(f"thinking is {resolved.thinking}, frozen is {FROZEN_PROVIDER['thinking']}")
    if bool(resolved.json_mode) is not bool(FROZEN_PROVIDER["json_mode"]):
        problems.append(f"json_mode is {resolved.json_mode}, frozen is {FROZEN_PROVIDER['json_mode']}")

    payload = resolved.profile.build_payload(
        model=resolved.model or FROZEN_PROVIDER["model"],
        messages=[{"role": "user", "content": "x"}],
        temperature=FROZEN_PROVIDER["temperature"],
        thinking=FROZEN_PROVIDER["thinking"],
        json_mode=resolved.json_mode,
        max_tokens=FROZEN_PROVIDER["max_tokens"],
    )
    if payload.get("thinking") != {"type": "disabled"}:
        problems.append(f"payload thinking is {payload.get('thinking')!r}, frozen is disabled")
    if payload.get("temperature") != FROZEN_PROVIDER["temperature"]:
        problems.append(
            f"payload temperature is {payload.get('temperature')!r}, frozen is "
            f"{FROZEN_PROVIDER['temperature']}"
        )
    if payload.get("response_format") != {"type": "json_object"}:
        problems.append(f"payload response_format is {payload.get('response_format')!r}")
    if payload.get("max_tokens") != FROZEN_PROVIDER["max_tokens"]:
        problems.append(
            f"payload max_tokens is {payload.get('max_tokens')!r}, frozen is "
            f"{FROZEN_PROVIDER['max_tokens']}"
        )
    return problems


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# frozen inputs
# --------------------------------------------------------------------------- #


@dataclass
class FrozenInputs:
    """Everything the runner asserts about, loaded but not yet judged."""

    heads: list[list[int]]
    head_lengths: list[int]
    comparator_ranks: Any
    comparator_targets: Any
    popularity_targets: Any
    sasrec_top10: Any
    cohort: list[Any] = field(default_factory=list)
    catalogue_records: int = 0


def load_frozen_inputs(*, with_cohort: bool) -> FrozenInputs:
    """Load the frozen artifacts. Cohort loading is opt-in so preflight stays cohort-free."""
    import numpy as np

    heads_z = np.load(REPO / "runs/m3_evidence/phase5_fusion_source_heads.npz")
    fusion_z = np.load(REPO / "runs/m3_evidence/phase5_fixed_fusion_hits.npz")
    pop_z = np.load(REPO / "runs/m3_evidence/phase5_popularity_hits.npz")
    sas_z = np.load(REPO / "runs/step26_baseline_evidence/phase5_sasrec_hits.npz")

    matrix, lengths = heads_z["sequential"], heads_z["sequential_lengths"]
    heads = [matrix[i, : int(lengths[i])].tolist() for i in range(len(lengths))]

    frozen = FrozenInputs(
        heads=heads,
        head_lengths=[int(v) for v in lengths],
        comparator_ranks=fusion_z["target_rank"],
        comparator_targets=fusion_z["target_ids"],
        popularity_targets=pop_z["target_ids"],
        sasrec_top10=sas_z["top10"],
    )

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    frozen.catalogue_records = len(mappings["item2id"])

    if with_cohort:
        import experiments.benchmark_public as B
        from recommendation.evaluation.split import load_cohort_from_artifacts

        cases, report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
        selection = B.cohort_from_cases(cases, size=B.DEFAULT_COHORT_SIZE)
        frozen.cohort = list(selection["cases"])
        frozen.catalogue_records = report.catalog_size

    return frozen


# --------------------------------------------------------------------------- #
# hard checks (pure, so tests can drive them with synthetic inputs)
# --------------------------------------------------------------------------- #


def _prereg_manifest_sha() -> str | None:
    """The prereg SHA recorded in the project-state manifest, or None if absent."""
    import re

    text = PROJECT_STATE.read_text(encoding="utf-8")
    match = re.search(
        r"^\|\s*`(docs/M3_PREREGISTRATION\.md)`\s*\|\s*`([0-9a-f]{64})`\s*\|", text, re.M
    )
    return match.group(2) if match else None


def artifact_checks(frozen: FrozenInputs) -> list[str]:
    """Checks that need no cohort: artifact hashes, catalogue, prereg drift, frozen config."""
    import experiments.benchmark_public as B
    from recommendation.control.schemas import LoopLimits

    problems: list[str] = []

    for rel, want in EXPECTED["artifacts"].items():
        path = REPO / rel
        if not path.is_file():
            problems.append(f"missing frozen artifact {rel}")
            continue
        got = _sha256(path)
        if got != want:
            problems.append(f"artifact drift {rel}: {got} != {want}")

    if frozen.catalogue_records != EXPECTED["catalogue"]:
        problems.append(
            f"catalogue is {frozen.catalogue_records}, expected {EXPECTED['catalogue']}"
        )

    recorded = _prereg_manifest_sha()
    actual = _sha256(REPO / EXPECTED["prereg"])
    if recorded is None:
        problems.append("the preregistration is not bound in the PROJECT_STATE manifest")
    elif recorded != actual:
        problems.append(f"prereg drift: file {actual} != manifest {recorded}")

    limits = LoopLimits()
    observed = (limits.max_steps, limits.max_tool_calls, limits.max_retries)
    if observed != EXPECTED["limits"]:
        problems.append(f"LoopLimits drift: {observed} != {EXPECTED['limits']}")
    if tuple(B.FUSION_SOURCES) != EXPECTED["fusion_sources"]:
        problems.append(f"FUSION_SOURCES drift: {tuple(B.FUSION_SOURCES)}")
    if tuple(B.K_VALUES) != EXPECTED["k_values"]:
        problems.append(f"K_VALUES drift: {tuple(B.K_VALUES)}")
    if B.DEFAULT_BATCH_SIZE != EXPECTED["batch_size"]:
        problems.append(f"DEFAULT_BATCH_SIZE drift: {B.DEFAULT_BATCH_SIZE}")

    from recommendation.control.candidate_ledger import RRF_K

    if RRF_K != EXPECTED["rrf_k"]:
        problems.append(f"RRF_K drift: {RRF_K} != {EXPECTED['rrf_k']}")

    return problems


def cohort_checks(frozen: FrozenInputs) -> list[str]:
    """Checks that bind the cohort to the frozen per-user evidence."""
    import numpy as np

    problems: list[str] = []

    if not frozen.cohort:
        return ["the cohort was not loaded"]

    if len(frozen.cohort) != EXPECTED["cohort"]:
        problems.append(f"cohort is {len(frozen.cohort)}, expected {EXPECTED['cohort']}")

    if len(frozen.heads) != len(frozen.cohort):
        problems.append(
            f"head rows {len(frozen.heads)} != cohort {len(frozen.cohort)}: cannot align"
        )
        return problems

    targets = np.asarray([int(c.test_target) for c in frozen.cohort], dtype=np.int64)
    if not np.array_equal(targets, frozen.popularity_targets):
        problems.append("cohort target identities differ from the frozen comparator evidence")

    # the decisive alignment guard: the heads artifact must be the same per-user sequence the
    # frozen sequential evidence was produced from, position by position.
    heads_top10 = np.asarray(
        [(row[:10] + [-1] * 10)[:10] for row in frozen.heads], dtype=np.int64
    )
    sas_top10 = np.asarray(frozen.sasrec_top10, dtype=np.int64)
    if heads_top10.shape != sas_top10.shape:
        problems.append(f"head top-10 shape {heads_top10.shape} != frozen {sas_top10.shape}")
    elif not np.array_equal(heads_top10, sas_top10):
        differing = int((heads_top10 != sas_top10).any(axis=1).sum())
        problems.append(
            f"head rows are not aligned with the frozen sequential evidence: {differing} users differ"
        )

    return problems


def guard(*problems: Sequence[str]) -> None:
    """Fail closed on the first non-empty problem list."""
    flat = [p for group in problems for p in group]
    if flat:
        raise HardCheckFailure("hard check failed:\n  - " + "\n  - ".join(flat))


# --------------------------------------------------------------------------- #
# provider usage accounting
# --------------------------------------------------------------------------- #


class UsageRecordingClient:
    """Delegate to a provider client while accumulating usage and latency."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.failures = 0
        self.codes: dict[str, int] = {}
        self.finish_reasons: dict[str, int] = {}
        self.schema_actions: list[str] = []
        self.requests_seen = 0

    def complete(self, request: Any) -> Any:
        from recommendation.control.model_client import ModelCallError

        self.requests_seen += 1
        if not self.schema_actions:
            self.schema_actions = [
                str(entry.get("action"))
                for entry in (getattr(request, "action_schema", ()) or ())
                if isinstance(entry, dict) and entry.get("action") is not None
            ]
        try:
            response = self._inner.complete(request)
        except ModelCallError as exc:
            self.failures += 1
            code = getattr(exc, "code", "model_error")
            self.codes[str(code)] = self.codes.get(str(code), 0) + 1
            raise
        self.calls += 1
        reason = getattr(response, "finish_reason", None)
        if reason is not None:
            key = str(reason)
            self.finish_reasons[key] = self.finish_reasons.get(key, 0) + 1
        self.input_tokens += int(getattr(response, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(response, "output_tokens", 0) or 0)
        self.latency_ms += float(getattr(response, "latency_ms", 0.0) or 0.0)
        return response

    def summary(self) -> dict[str, Any]:
        total = self.calls + self.failures
        return {
            "provider_calls": self.calls,
            "provider_failures": self.failures,
            "attempts": total,
            "failure_rate": (self.failures / total) if total else 0.0,
            "failure_codes": dict(self.codes),
            "finish_reasons": dict(self.finish_reasons),
            "action_schema_offered": list(self.schema_actions),
            "schema_actions_validated": bool(self.schema_actions),
            "requests_seen": self.requests_seen,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "latency_ms_total": self.latency_ms,
            "latency_ms_mean": (self.latency_ms / self.calls) if self.calls else 0.0,
        }


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def _smoke_request() -> Any:
    """A synthetic, non-cohort policy request: fixed text, empty candidate state."""
    from recommendation.control.context import CandidateState, PolicyContext
    from recommendation.control.model_policy import (
        build_action_schema,
        build_policy_context_payload,
        build_policy_system_prompt,
    )
    from recommendation.control.model_client import ModelRequest

    from recommendation.control.schemas import ActionKind

    context = PolicyContext(
        user_request="Preflight smoke: no cohort user is involved.",
        available_actions=(ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.FINISH),
        has_trusted_history=True,
        candidate_state=CandidateState(),
        remaining_steps=5,
        remaining_tool_calls=3,
    )
    return ModelRequest(
        system_prompt=build_policy_system_prompt(),
        context_payload=build_policy_context_payload(context),
        action_schema=build_action_schema(tuple(context.available_actions)),
    )


def synthetic_case() -> Any:
    """A synthetic, non-cohort case used only to drive the control plane.

    Its ``test_target`` is never scored: preflight has no evaluator, so this case cannot
    produce a recommendation metric. The history uses ordinary catalogue item ids.
    """
    from recommendation.evaluation.split import EvaluationCase

    history = (1, 2, 3, 4)
    return EvaluationCase(
        user_id="preflight-synthetic",
        user_int_id=1,
        train_history=history,
        validation_target=5,
        test_target=5,
        sequence_length=len(history) + 1,
    )


def control_plane_probe(
    *,
    client: Any,
    harness_factory: Callable[[Any], Any] | None = None,
    item2id: Any = None,
) -> dict[str, Any]:
    """Drive the REAL policy through the REAL bounded control plane on a synthetic case.

    Real ``LLMAgentPolicy``, real ``LoopController``, real action validation — only the case and
    the catalogue binding are synthetic. This function never imports or calls ``evaluate_batched``
    and never loads the cohort, so it cannot compute a recommendation metric.
    """
    from experiments.m3_agent_arm import M3AgentRunner
    from recommendation.control.model_policy import LLMAgentPolicy

    if harness_factory is None:
        from experiments.m3_agent_harness import RealCatalogueHarnessFactory

        built = RealCatalogueHarnessFactory(
            policy_factory=lambda: LLMAgentPolicy(client, max_attempts=2),
            user_order=(1,),
        )
        harness_factory = built
        item2id = built.shared().item2id

    case = synthetic_case()
    runner = M3AgentRunner(harness_factory=harness_factory, item2id=item2id)
    outcome = runner.run(case)
    behavior = outcome.behavior
    return {
        "synthetic_case": {
            "user_int_id": int(case.user_int_id),
            "history_items": len(case.test_history),
            "cohort_member": False,
            "target_scored": False,
        },
        "action_sequence": behavior["action_sequence"],
        "termination_status": behavior["status"],
        "termination_reason": behavior["termination_reason"],
        "control_plane_completed": bool(behavior["succeeded"]),
        "steps": behavior["steps"],
        "tool_calls": behavior["tool_calls"],
        "retries": behavior["retries"],
        "ranking_size": behavior["ranking_size"],
        "fallback": bool(behavior["budget_exhausted"]),
        "failure": bool(behavior["failure"]),
        "candidates": behavior["candidates"],
        "evaluator_invoked": False,
        "recommendation_metrics_computed": False,
    }


def run_preflight(
    *,
    out: pathlib.Path,
    client_factory: Callable[[], Any] | None = None,
    calls: int = 3,
    harness_factory: Callable[[Any], Any] | None = None,
    item2id: Any = None,
) -> dict[str, Any]:
    """Provider smoke on non-cohort inputs. Structurally cannot reach the evaluator.

    This function does not import ``evaluate_batched``, does not load the cohort and does not
    build the Agent arm, so no recommendation metric can be produced here even by accident.
    """
    guard(artifact_checks(load_frozen_inputs(with_cohort=False)), provider_config_checks())

    if client_factory is None:
        client_factory = build_m3_provider_client
    recording = UsageRecordingClient(client_factory())
    request = _smoke_request()

    from recommendation.control.model_client import ModelCallError

    parsed = 0
    for _ in range(max(1, int(calls))):
        try:
            response = recording.complete(request)
        except ModelCallError:
            continue
        text = getattr(response, "text", "")
        try:
            json.loads(text)
            parsed += 1
        except (TypeError, ValueError):
            pass

    from recommendation.control.model_client import ModelCallError

    try:
        control = control_plane_probe(
            client=recording, harness_factory=harness_factory, item2id=item2id
        )
    except ModelCallError as exc:
        # reported, not raised: a preflight exists to surface this
        control = {
            "control_plane_error": str(getattr(exc, "code", "model_error")),
            "evaluator_invoked": False,
            "recommendation_metrics_computed": False,
        }

    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "preflight",
        "cohort_users": 0,
        "cohort_loaded": False,
        "evaluator_invoked": False,
        "recommendation_metrics_computed": False,
        "requested_calls": int(calls),
        "json_parse_successes": parsed,
        "json_parse_rate": (parsed / int(calls)) if int(calls) else 0.0,
        "provider_connected": recording.calls > 0,
        "control_plane": control,
        **recording.summary(),
    }
    (out / "preflight.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #


def agent_arm_ranks(
    *,
    cases: Sequence[Any],
    runner: Any,
    num_items: int,
    records: list[dict[str, Any]] | None = None,
    rankings: list[list[int]] | None = None,
) -> Any:
    """Run the Agent arm through the unmodified shared evaluator and return per-user ranks.

    This is the only place ``--execute`` touches the evaluator, and it is separated out so the
    plumbing can be tested without a 20 000-user cohort.
    """
    import numpy as np

    from experiments.m3_agent_arm import arm_agent_llm
    from recommendation.evaluation.batched import evaluate_batched

    batches = arm_agent_llm(
        num_items=num_items, runner=runner, records=records, rankings=rankings
    )
    result = evaluate_batched(
        num_items=num_items,
        score_batches=batches(cases, EXPECTED["batch_size"]),
        k_values=EXPECTED["k_values"],
        cohort="test",
    )
    return np.asarray(result.target_ranks, dtype=np.int64), result


def _mcnemar(b: int, c: int) -> float:
    import math

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def compute_statistics(agent_ranks: Any, comparator_ranks: Any) -> dict[str, Any]:
    """The preregistered §5 statistics: exact McNemar plus a paired user bootstrap."""
    import numpy as np

    n = len(agent_ranks)
    out: dict[str, Any] = {"users": int(n), "per_k": {}}
    rng = np.random.default_rng(EXPECTED["bootstrap_seed"])
    for k in EXPECTED["k_values"]:
        a = (agent_ranks <= k).astype(np.float64)
        c = (comparator_ranks <= k).astype(np.float64)
        b = int(((a > 0) & (c == 0)).sum())
        cc = int(((a == 0) & (c > 0)).sum())
        diff = a - c
        means = np.empty(EXPECTED["bootstrap_resamples"], dtype=np.float64)
        done = 0
        while done < len(means):
            take = min(500, len(means) - done)
            idx = rng.integers(0, n, size=(take, n))
            means[done : done + take] = diff[idx].mean(axis=1)
            done += take
        out["per_k"][f"K={k}"] = {
            "agent_hits": int(a.sum()),
            "comparator_hits": int(c.sum()),
            "agent_hit_comparator_miss": b,
            "agent_miss_comparator_hit": cc,
            "mcnemar_p": _mcnemar(b, cc),
            "recall_delta": float(diff.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
        }
    return out


def run_execute(
    *,
    out: pathlib.Path,
    client_factory: Callable[[], Any] | None = None,
    frozen: FrozenInputs | None = None,
) -> dict[str, Any]:
    """The single formal 20 000-user M3 evaluation."""
    import numpy as np

    from experiments.m3_agent_arm import M3AgentRunner, summarize_behavior
    from experiments.m3_agent_harness import RealCatalogueHarnessFactory

    frozen = frozen or load_frozen_inputs(with_cohort=True)
    guard(artifact_checks(frozen), cohort_checks(frozen), provider_config_checks())

    if client_factory is None:
        client_factory = build_m3_provider_client

    recording = UsageRecordingClient(client_factory())
    from recommendation.control.model_client import ModelCallError
    from recommendation.control.model_policy import LLMAgentPolicy

    def policy_factory() -> Any:
        return LLMAgentPolicy(recording, max_attempts=2 + EXPECTED["limits"][2])

    # provider failures must not silently become a ranking
    try:
        factory = RealCatalogueHarnessFactory(
            policy_factory=policy_factory,
            user_order=[int(c.user_int_id) for c in frozen.cohort],
        )
        runner = M3AgentRunner(harness_factory=factory, item2id=factory.shared().item2id)
        records: list[dict[str, Any]] = []
        rankings: list[list[int]] = []
        agent_ranks, _result = agent_arm_ranks(
            cases=frozen.cohort,
            runner=runner,
            num_items=frozen.catalogue_records,
            records=records,
            rankings=rankings,
        )
    except ModelCallError as exc:
        raise HardCheckFailure(f"provider call failed during --execute: {exc.code}") from exc

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "agent_ranks.npz",
        case_index=np.arange(len(agent_ranks), dtype=np.int64),
        target_rank=agent_ranks,
        target_ids=np.asarray(frozen.comparator_targets, dtype=np.int64),
    )
    with (out / "agent_behavior.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    with (out / "agent_rankings.jsonl").open("w", encoding="utf-8") as fh:
        for index, ranking in enumerate(rankings):
            fh.write(json.dumps({"index": index, "ranking": ranking}, sort_keys=True) + "\n")

    np.savez(
        out / "paired_inputs.npz",
        agent_rank=agent_ranks,
        comparator_rank=np.asarray(frozen.comparator_ranks, dtype=np.int64),
        target_ids=np.asarray(frozen.comparator_targets, dtype=np.int64),
    )
    statistics = compute_statistics(agent_ranks, np.asarray(frozen.comparator_ranks))
    (out / "m3_statistics.json").write_text(json.dumps(statistics, indent=2, sort_keys=True))

    summary = {
        "mode": "execute",
        "cohort_users": len(frozen.cohort),
        "catalogue_records": frozen.catalogue_records,
        "behavior": summarize_behavior(records),
        **recording.summary(),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    provenance = {
        "prereg_sha256": _sha256(REPO / EXPECTED["prereg"]),
        "artifacts": {rel: _sha256(REPO / rel) for rel in EXPECTED["artifacts"]},
        "expected": {k: v for k, v in EXPECTED.items() if k != "artifacts"},
        "sidecars": sorted(p.name for p in out.iterdir()),
    }
    (out / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    from recommendation.control.model_client import ModelCallError

    parser = argparse.ArgumentParser(
        prog="python -m experiments.m3_agent_benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="non-cohort provider smoke; never loads the cohort and never enters the evaluator",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="the single formal 20 000-user M3 evaluation",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="sidecar output directory")
    parser.add_argument("--calls", type=int, default=3, help="preflight provider calls")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    try:
        if args.preflight:
            summary = run_preflight(out=out, calls=args.calls)
        else:
            summary = run_execute(out=out)
    except HardCheckFailure as exc:
        print(f"STOP — {exc}", file=sys.stderr)
        return 3
    except ModelCallError as exc:
        # a missing or unusable provider is a stop, not a traceback
        print(f"STOP — provider is not usable: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
