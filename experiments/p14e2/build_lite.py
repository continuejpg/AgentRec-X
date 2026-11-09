"""P14-E2 Lite: add 80 tasks on top of the frozen P14-E pilot (train/dev/reserved_test).

Reuses the frozen P14-E construction primitives (history templates, persona model,
attribute-evidence labeler) without modifying the P14-E files, Scorer, Gate or labels.

Splits are decided per persona/family BEFORE tasks are built:
* train 40, dev 20, reserved_test 20 (original 40 stay `pilot_dev`);
* no persona / family / same-query pair crosses a split;
* reserved_test is built and schema/leakage-checked but never searched/scored/gated.

Run: ``python -m experiments.p14e2.build_lite``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from experiments.p14e.build_pilot import (
    BENCHMARK_VERSION as P14E_VERSION,
    FORM_ATTRS,
    GENERATOR_VERSION,
    NO_MATCH_HARD,
    PHRASES,
    QUERY_TEMPLATES,
    SCENARIO_PREFS,
    TYPE_WORD,
    UNVERIFIABLE_HARD,
    EventSpec,
    _event,
    _hard_constraint,
    _history_events,
    _history_for_ask,
    _history_for_conflict,
    _history_for_prefs,
    _history_for_update,
    _history_for_withdraw,
    _history_sessions,
    _phrase,
    _pref,
    _prefs_from_lists,
    _query_with_constraints,
    _unverifiable_constraint,
)
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import MIN_RECOMMENDABLE, active_prefs, label_task

E2_VERSION = "agentrecx.p14e2.shopping_preference.v0.1"
E2_GENERATOR = "p14e2-lite-0.1.0"
E2_SEED = 20261115
SPLITS = ("train", "dev", "reserved_test")
REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "data/p14e2/benchmark"
FORBIDDEN_AGENT_KEYS = {
    "effective_preferences", "withdrawn_preferences", "expected_decision", "decision_rationale",
    "labels", "verified_candidates", "verified_candidate_count", "hard_constraints", "persona",
}

#: New queries, deliberately different from the frozen P14-E query strings.
E2_QUERIES: dict[str, dict[str, str]] = {
    "headphones": {
        "commute": "Which headphones would work well for my commute?",
        "calls": "I take a lot of calls; which headphones should I get?",
        "travel": "I'm packing for a long trip and need a pair of headphones.",
        "gaming": "I want headphones for my gaming setup.",
    },
    "earbuds": {
        "workout": "I need earbuds I can use while working out.",
        "calls": "Which earbuds are good for phone calls?",
    },
    "keyboard": {
        "office": "I need a keyboard for working at my desk.",
        "gaming": "Which keyboard should I get for gaming?",
        "travel": "I want a keyboard I can carry when travelling.",
    },
    "mouse": {
        "office": "Which mouse is good for a desk setup?",
        "gaming": "I'm looking for a gaming mouse.",
        "travel": "I need a mouse that is easy to carry around.",
    },
    "speaker": {
        "home": "Which speaker should I get for my living room?",
        "outdoor": "I need a speaker for trips outside.",
    },
}
E2_SCENARIOS: dict[str, list[str]] = {
    "headphones": ["commute", "calls", "travel", "gaming"],
    "earbuds": ["workout", "calls"],
    "keyboard": ["office", "gaming", "travel"],
    "mouse": ["office", "gaming", "travel"],
    "speaker": ["home", "outdoor"],
}
#: Scenario prefs are the frozen P14-E ones, keyed by the scenario name it used.
E2_SCENARIO_KEY = {
    ("headphones", "commute"): ("headphones", "daily_commute"),
    ("headphones", "calls"): ("headphones", "work_calls"),
    ("headphones", "travel"): ("headphones", "long_flights"),
    ("headphones", "gaming"): ("headphones", "gaming"),
    ("earbuds", "workout"): ("earbuds", "workout"),
    ("earbuds", "calls"): ("earbuds", "calls"),
    ("keyboard", "office"): ("keyboard", "office"),
    ("keyboard", "gaming"): ("keyboard", "gaming"),
    ("keyboard", "travel"): ("keyboard", "travel"),
    ("mouse", "office"): ("mouse", "office"),
    ("mouse", "gaming"): ("mouse", "gaming"),
    ("mouse", "travel"): ("mouse", "travel"),
    ("speaker", "home"): ("speaker", "home"),
    ("speaker", "outdoor"): ("speaker", "outdoor"),
}

SAFE_HARD_ATTRS = ("wireless", "noise_cancelling", "over_ear", "microphone", "mechanical",
                   "rgb", "waterproof", "foldable", "ergonomic", "rechargeable")


def _context_clause(index: int) -> str:
    """A deterministic, natural, per-task context clause that keeps non-pair queries unique.

    It contains no product attribute names, so it never changes the decision criteria; it
    only ensures no two non-pair tasks share an exact query string across splits.
    """
    times = ("morning", "evening", "weekend", "daily")
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    rooms = ("living room", "home office", "bedroom", "kitchen", "study", "workshop", "garage", "garden")
    time_word = times[index % len(times)]
    weekday = weekdays[(index // len(times)) % len(weekdays)]
    room = rooms[(index // (len(times) * len(weekdays))) % len(rooms)]
    return f"I'll mostly use it in my {room} during {time_word} use on {weekday}s."


class LiteBuilder:
    def __init__(self, catalogue: ElectronicsCatalogue) -> None:
        self.catalogue = catalogue
        self.tasks: list[dict[str, Any]] = []
        self.evaluator: list[dict[str, Any]] = []
        self.personas: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.index = 0
        self.persona_index = 0

    def _next_persona(self, product_type: str, scenario: str, prefs: list[dict[str, Any]]) -> dict[str, Any]:
        self.persona_index += 1
        return {
            "persona_id": f"p14e2_persona_{self.persona_index:04d}",
            "user_id": f"p14e2_user_{self.persona_index:04d}",
            "product_type": product_type,
            "scenario": scenario,
            "preferences": list(prefs),
            "seed": E2_SEED + self.persona_index,
        }

    def _persona_spec(self, persona: dict[str, Any]):
        from experiments.p14e.build_pilot import PersonaSpec

        return PersonaSpec(
            persona_id=persona["persona_id"], user_id=persona["user_id"],
            product_type=persona["product_type"], scenario=persona["scenario"],
            preferences=list(persona["preferences"]), seed=int(persona["seed"]),
        )

    def add_task(
        self,
        *,
        family: str,
        family_id: str,
        split: str,
        product_type: str,
        scenario: str,
        query: str,
        persona: dict[str, Any],
        events: list[EventSpec],
        hard: list[dict[str, Any]],
        intended: set[str],
        intent_note: str,
        review_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        self.index += 1
        task_id = f"p14e2_lite_{self.index:04d}"
        base_date = date(2025, 9, 1) + timedelta(days=(self.index % 25) * 3)
        if family != "same_query_persona_pair":
            query = query + " " + _context_clause(self.index)
        persona_spec = self._persona_spec(persona)
        history_sessions = _history_sessions(events, task_id, persona["persona_id"], base_date)
        history_events = _history_events(events, task_id, persona["persona_id"], base_date)
        effective, withdrawn = active_prefs(history_events)
        pool = self.catalogue.by_type(product_type)
        labels = label_task({}, {"effective_preferences": effective, "hard_constraints": hard}, pool)
        if labels["expected_decision"] not in intended:
            raise ValueError(f"{family}: expected {labels['expected_decision']} not in {intended}")
        review_reasons = list(review_reasons or [])
        if family == "same_query_persona_pair":
            review_reasons.append("same-query contrast group requires human review of preference distinctness")
        if family in {"no_verified_match", "unverifiable_constraint"}:
            review_reasons.append("no-verified/unknown outcome is relative to the sampled catalogue and needs review")
        if labels["expected_decision"] == "recommend" and labels["verified_candidate_count"] < MIN_RECOMMENDABLE:
            review_reasons.append("few verified candidates; candidate coverage may be catalogue-limited")
        review_status = "pending_human_review" if review_reasons else "automated_checked"
        agent_task = {
            "schema": E2_VERSION,
            "task_id": task_id,
            "split": split,
            "user_id": persona["user_id"],
            "persona_id": persona["persona_id"],
            "current_query": query,
            "current_time": base_date.isoformat() + "T12:00:00",
            "history_sessions": history_sessions,
            "allowed_tools": ["mem_search", "mem_view", "mem_summarize_by_date", "product_search", "product_view"],
            "catalog_namespace": ELECTRONICS_NAMESPACE,
            "tool_budget": {"max_tool_calls": 12, "max_search_calls": 6, "max_product_views": 6},
            "public_constraints": [str(c["source_text"]) for c in hard],
            "construction_metadata": {
                "history_source": "constructed_template",
                "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1",
                "generator_version": E2_GENERATOR,
                "random_seed": E2_SEED,
                "review_status": review_status,
                "benchmark_split": split,
            },
        }
        evaluator_task = {
            "schema": E2_VERSION,
            "task_id": task_id,
            "task_type": family,
            "family_id": family_id,
            "split": split,
            "user_id": persona["user_id"],
            "persona_id": persona["persona_id"],
            "persona": {
                "persona_id": persona["persona_id"], "user_id": persona["user_id"],
                "product_type": product_type, "scenario": scenario,
                "profile_source": "synthetic_scenario_profile",
                "preferences": persona["preferences"],
            },
            "effective_preferences": effective,
            "withdrawn_preferences": withdrawn,
            "hard_constraints": hard,
            "catalog_namespace": ELECTRONICS_NAMESPACE,
            "product_type_pool": product_type,
            "expected_decision": labels["expected_decision"],
            "decision_rationale": labels["decision_rationale"],
            "labels": labels,
            "construction_metadata": {
                "history_source": "constructed_template",
                "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1",
                "generator_version": E2_GENERATOR,
                "random_seed": E2_SEED,
                "review_status": review_status,
                "review_reasons": review_reasons,
                "labeler_version": "p14e-labeling-0.1.0",
                "label_basis": "independent evidence-based labeler over the real catalogue; not copied from Scorer/Gate output",
                "intent_note": intent_note,
            },
        }
        self.tasks.append(agent_task)
        self.evaluator.append(evaluator_task)
        self.personas.append({
            "persona_id": persona["persona_id"], "user_id": persona["user_id"],
            "product_type": product_type, "scenario": scenario,
            "preferences": persona["preferences"], "profile_source": "synthetic_scenario_profile",
            "split": split, "family_id": family_id,
            "generation": {"generator_version": E2_GENERATOR, "random_seed": E2_SEED},
        })
        self.history_events.extend(history_events)
        return labels

    # -- families ---------------------------------------------------------- #
    def _scenario_parts(self, product_type: str, scenario: str):
        key = E2_SCENARIO_KEY[(product_type, scenario)]
        positives, negatives = SCENARIO_PREFS[key]
        return positives, negatives

    def build_no_history(self, split: str, count: int, family_prefix: str) -> None:
        types = ["headphones", "earbuds", "keyboard", "mouse", "speaker"]
        made = 0
        attempt = 0
        while made < count:
            product_type = types[(made + attempt) % len(types)]
            scenario = E2_SCENARIOS[product_type][attempt % len(E2_SCENARIOS[product_type])]
            query = E2_QUERIES[product_type][scenario]
            hard_name = SAFE_HARD_ATTRS[attempt % len(SAFE_HARD_ATTRS)]
            hard = [_hard_constraint(hard_name, f"hc_{split}_{made}_{hard_name}")]
            query_text = _query_with_constraints(query, hard)
            persona = self._next_persona(product_type, scenario, [])
            events = [_event("ctx", 0, [], f"I'm shopping around for {TYPE_WORD[product_type]}.")]
            try:
                self.add_task(
                    family="no_history_recommend", family_id=f"{family_prefix}_no_history_{made + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario, query=query_text,
                    persona=persona, events=events, hard=hard, intended={"recommend"},
                    intent_note="no recoverable history preference; the explicit hard constraint is the only decision criterion",
                )
                made += 1
            except ValueError:
                attempt += 1
                if attempt > 12:
                    self.warnings.append(f"{split}: no_history family short by {count - made}")
                    break
        return

    def build_history_influenced(self, split: str, count: int, family_prefix: str) -> None:
        types = ["headphones", "keyboard", "mouse", "speaker", "earbuds"]
        made = attempt = 0
        while made < count:
            product_type = types[(made + attempt) % len(types)]
            scenario = E2_SCENARIOS[product_type][attempt % len(E2_SCENARIOS[product_type])]
            positives, negatives = self._scenario_parts(product_type, scenario)
            prefs = _prefs_from_lists(positives, negatives)
            query = E2_QUERIES[product_type][scenario]
            persona = self._next_persona(product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(E2_SEED + made))
            try:
                self.add_task(
                    family="history_influenced", family_id=f"{family_prefix}_history_{made + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario, query=query,
                    persona=persona, events=events, hard=[], intended={"recommend"},
                    intent_note="current query omits the historical soft preferences; the legal history read should recover them",
                )
                made += 1
            except ValueError:
                attempt += 1
                if attempt > 12:
                    self.warnings.append(f"{split}: history family short by {count - made}")
                    break

    def build_pairs(self, split: str, pairs: int, family_prefix: str, pair_offset: int = 0) -> None:
        specs = [
            ("headphones", "commute", "Which headphones would you suggest for commuting?",
             (["noise_cancelling", "over_ear"], ["in_ear"]), (["foldable", "wireless"], ["over_ear"])),
            ("headphones", "calls", "Which headphones are good for calls?",
             (["microphone", "over_ear"], ["in_ear"]), (["microphone", "wireless"], ["over_ear"])),
            ("keyboard", "office", "Which keyboard fits an office desk?",
             (["silent", "wired"], ["rgb"]), (["rgb", "mechanical"], ["silent"])),
            ("mouse", "office", "Which mouse fits an office desk?",
             (["ergonomic", "wireless"], ["gaming"]), (["gaming", "rgb"], ["ergonomic"])),
            ("speaker", "home", "Which speaker fits a living room?",
             (["wireless", "microphone"], ["rechargeable"]), (["rechargeable", "waterproof"], ["microphone"])),
            ("earbuds", "workout", "Which earbuds work for workouts?",
             (["waterproof", "wireless"], ["over_ear"]), (["microphone", "rechargeable"], ["over_ear"])),
            ("keyboard", "gaming", "Which keyboard is good for gaming?",
             (["mechanical", "rgb"], ["silent"]), (["wireless", "rechargeable"], ["wired"])),
            ("mouse", "travel", "Which mouse is best for travel?",
             (["wireless", "rechargeable"], ["wired"]), (["ergonomic", "silent"], ["gaming"])),
        ]
        for pair_index in range(pairs):
            spec_index = pair_offset + pair_index
            product_type, scenario, query, a_spec, b_spec = specs[spec_index % len(specs)]
            pair_index = spec_index
            pair_query = query + " " + (
                "I'm weighing a few options this week, so a focused suggestion would help."
                if spec_index % 2 == 0
                else "I'd rather hear one clear suggestion than a long list."
            )
            for side, (positives, negatives) in (("a", a_spec), ("b", b_spec)):
                prefs = _prefs_from_lists(positives, negatives)
                persona = self._next_persona(product_type, scenario, prefs)
                events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(E2_SEED + 100 + pair_index))
                try:
                    self.add_task(
                        family="same_query_persona_pair", family_id=f"{family_prefix}_pair_{pair_index + 1:02d}",
                        split=split, product_type=product_type, scenario=scenario, query=pair_query,
                        persona=persona, events=events, hard=[], intended={"recommend"},
                        intent_note="same current query as the paired persona; different recoverable preferences",
                    )
                except ValueError:
                    self.warnings.append(f"{split}: pair {pair_index + 1}{side} skipped (labeler expected non-recommend)")

    def build_dynamics(self, split: str, family_prefix: str, updates: int, withdrawals: int, conflicts: int) -> None:
        update_specs = [
            ("headphones", "commute", "over_ear", "in_ear"),
            ("keyboard", "office", "wireless", "wired"),
            ("mouse", "travel", "wireless", "wired"),
            ("speaker", "home", "wireless", "wired"),
        ]
        conflict_specs = [
            ("headphones", "commute", "wireless", "wired"),
            ("keyboard", "office", "wireless", "wired"),
            ("mouse", "travel", "wireless", "wired"),
            ("speaker", "home", "wireless", "wired"),
        ]
        withdraw_specs = [
            ("headphones", "commute", "noise_cancelling"),
            ("keyboard", "office", "rgb"),
            ("mouse", "office", "ergonomic"),
            ("earbuds", "workout", "waterproof"),
        ]
        made = 0
        for i in range(updates):
            product_type, scenario, old, new = update_specs[(made + i) % len(update_specs)]
            prefs = [_pref("form_factor" if old in FORM_ATTRS else "attribute", old, "positive"),
                     _pref("attribute", "wireless", "positive")] if old != "wireless" else [
                _pref("attribute", "wireless", "positive")]
            persona = self._next_persona(product_type, scenario, prefs)
            events = _history_for_update(old, new, product_type)
            query = E2_QUERIES[product_type][scenario]
            try:
                self.add_task(
                    family="preference_update", family_id=f"{family_prefix}_update_{i + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario, query=query,
                    persona=persona, events=events, hard=[], intended={"recommend"},
                    intent_note=f"history explicitly retracts {old} and states {new}; the old value must not stay active",
                )
                made += 1
            except ValueError:
                self.warnings.append(f"{split}: update {i + 1} skipped")
        made = 0
        for i in range(withdrawals):
            product_type, scenario, name = withdraw_specs[(made + i) % len(withdraw_specs)]
            kind = "form_factor" if name in FORM_ATTRS else "attribute"
            prefs = [_pref(kind, name, "positive"), _pref("attribute", "wireless", "positive")] if name != "wireless" else [
                _pref("attribute", "wireless", "positive")]
            persona = self._next_persona(product_type, scenario, prefs)
            events = _history_for_withdraw(name, product_type)
            query = E2_QUERIES[product_type][scenario]
            hard = [_hard_constraint("wireless", f"hc_{split}_{i}_wireless")]
            try:
                self.add_task(
                    family="preference_withdrawal", family_id=f"{family_prefix}_withdraw_{i + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario,
                    query=_query_with_constraints(query, hard), persona=persona, events=events,
                    hard=hard, intended={"recommend"},
                    intent_note=f"history explicitly withdraws {name}; the remaining preference/hard constraint must drive the decision",
                )
                made += 1
            except ValueError:
                self.warnings.append(f"{split}: withdrawal {i + 1} skipped")
        for i in range(conflicts):
            product_type, scenario, left, right = conflict_specs[(made + i) % len(conflict_specs)]
            prefs = [_pref("attribute", left, "positive"), _pref("attribute", right, "positive")]
            persona = self._next_persona(product_type, scenario, prefs)
            events = _history_for_conflict(left, right, product_type)
            try:
                self.add_task(
                    family="preference_conflict", family_id=f"{family_prefix}_conflict_{i + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario,
                    query=E2_QUERIES[product_type][scenario], persona=persona, events=events,
                    hard=[], intended={"clarification_required"},
                    intent_note="contradictory active preferences; a genuine user decision is required",
                )
            except ValueError:
                self.warnings.append(f"{split}: conflict {i + 1} skipped")

    def build_ask(self, split: str, count: int, family_prefix: str) -> None:
        specs = [
            ("headphones", "commute", "I need some new headphones. Not sure what matters yet."),
            ("earbuds", "workout", "I want earbuds but haven't decided on anything specific."),
            ("keyboard", "office", "I need a keyboard for my desk. Any suggestions?"),
            ("mouse", "office", "I'm thinking about a new mouse for work."),
            ("speaker", "home", "I'd like a speaker for my home, not sure which kind."),
            ("headphones", "calls", "I need headphones for calls - no particular requirements."),
            ("keyboard", "gaming", "I want a keyboard for gaming, still open on the details."),
            ("mouse", "travel", "I need a mouse that I can take with me; haven't decided more."),
            ("speaker", "outdoor", "Looking for a speaker for outside, no fixed preferences."),
            ("earbuds", "calls", "I'm considering earbuds for calls but not sure what I need."),
        ]
        used = 0
        attempt = 0
        while used < count:
            product_type, scenario, query = specs[(used + attempt) % len(specs)]
            persona = self._next_persona(product_type, scenario, [])
            events = [_event("ctx", 0, [], f"I've only started looking at {TYPE_WORD[product_type]} options.")]
            try:
                self.add_task(
                    family="insufficient_information", family_id=f"{family_prefix}_ask_{used + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario, query=query,
                    persona=persona, events=events, hard=[], intended={"clarification_required"},
                    intent_note="no recoverable preference and no explicit hard requirement; the agent must ask what the user needs",
                )
                used += 1
            except ValueError:
                attempt += 1
                if attempt > 10:
                    self.warnings.append(f"{split}: ask family short by {count - used}")
                    break

    def build_evidence(self, split: str, count: int, family_prefix: str, unverifiable_count: int = 0) -> None:
        made = 0
        zero_pool = [
            ["over_ear", "mechanical", "waterproof"],
            ["over_ear", "rgb", "multidevice"],
            ["over_ear", "foldable", "multidevice"],
            ["over_ear", "mechanical", "multidevice"],
            ["over_ear", "waterproof", "rgb"],
            ["over_ear", "noise_cancelling", "foldable", "waterproof"],
        ]
        unver_pool = list(UNVERIFIABLE_HARD)
        zero_index = 0
        unver_index = 0
        while made < count:
            placed = False
            force_unverifiable = made < unverifiable_count or zero_index >= len(zero_pool)
            while zero_index < len(zero_pool) and not placed and not force_unverifiable:
                combo = zero_pool[zero_index]
                product_type, scenario = ("headphones", "commute")
                positives, _ = self._scenario_parts(product_type, scenario)
                prefs = _prefs_from_lists(positives[:1], [])
                hard = [_hard_constraint(name, f"hc_{split}_ev_{made}_{name}") for name in combo]
                persona = self._next_persona(product_type, scenario, prefs)
                events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(E2_SEED + 200 + made))
                try:
                    self.add_task(
                        family="no_verified_match", family_id=f"{family_prefix}_evidence_{made + 1:02d}",
                        split=split, product_type=product_type, scenario=scenario,
                        query=_query_with_constraints(E2_QUERIES[product_type][scenario], hard),
                        persona=persona, events=events, hard=hard, intended={"no_verified_match"},
                        intent_note="hard constraint combination has no verified candidate in the sampled catalogue",
                    )
                    zero_index += 1
                    made += 1
                    placed = True
                except ValueError:
                    zero_index += 1
            if placed:
                continue
            spec = unver_pool[unver_index % len(unver_pool)] if unver_pool else None
            unver_index += 1
            if spec is None:
                self.warnings.append(f"{split}: evidence family short by {count - made}")
                break
            product_type, scenario = ("headphones", "commute") if made % 2 == 0 else ("mouse", "office")
            positives, negatives = self._scenario_parts(product_type, scenario)
            prefs = _prefs_from_lists(positives, negatives)
            hard = [_unverifiable_constraint(spec, f"hc_{split}_ev_{made}")]
            persona = self._next_persona(product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(E2_SEED + 300 + made))
            try:
                self.add_task(
                    family="unverifiable_constraint", family_id=f"{family_prefix}_evidence_{made + 1:02d}",
                    split=split, product_type=product_type, scenario=scenario,
                    query=_query_with_constraints(E2_QUERIES[product_type][scenario], hard),
                    persona=persona, events=events, hard=hard, intended={"insufficient_evidence"},
                    intent_note="stated hard constraint has no verifiable field in catalog_meta.jsonl (price/warranty); UNKNOWN is preserved",
                )
                made += 1
            except ValueError:
                if unver_index > len(unver_pool) * 2:
                    self.warnings.append(f"{split}: evidence family short by {count - made}")
                    break

    # -- split plans -------------------------------------------------------- #
    def build_split(self, split: str, plan: dict[str, int]) -> None:
        prefix = split
        self.build_no_history(split, plan["no_history"], prefix)
        self.build_history_influenced(split, plan["history"], prefix)
        self.build_pairs(split, plan["pairs"], prefix, plan.get("pair_offset", 0))
        self.build_dynamics(split, prefix, plan["updates"], plan["withdrawals"], plan["conflicts"])
        self.build_ask(split, plan["ask"], prefix)
        self.build_evidence(split, plan["evidence"], prefix, plan.get("evidence_unver", 0))

    def build_all(self) -> None:
        self.build_split("train", {"no_history": 8, "history": 10, "pairs": 5, "pair_offset": 0, "updates": 2,
                                   "withdrawals": 2, "conflicts": 2, "ask": 4, "evidence": 2, "evidence_unver": 1})
        self.build_split("dev", {"no_history": 4, "history": 5, "pairs": 2, "pair_offset": 5, "updates": 1,
                                 "withdrawals": 1, "conflicts": 1, "ask": 3, "evidence": 1, "evidence_unver": 0})
        self.build_split("reserved_test", {"no_history": 3, "history": 5, "pairs": 1, "pair_offset": 7, "updates": 1,
                                           "withdrawals": 1, "conflicts": 1, "ask": 3, "evidence": 4, "evidence_unver": 2})


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path), "rows": len(rows)}


def write_outputs(builder: LiteBuilder, catalogue: ElectronicsCatalogue, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = {split: {"tasks": [], "evaluator": []} for split in SPLITS}
    for task, ev in zip(builder.tasks, builder.evaluator):
        by_split[task["split"]]["tasks"].append(task)
        by_split[task["split"]]["evaluator"].append(ev)
    files: dict[str, Any] = {}
    for split in SPLITS:
        files[f"{split}.agent.jsonl"] = _write_jsonl(out_dir / f"{split}.agent.jsonl", by_split[split]["tasks"])
        files[f"{split}.evaluator.jsonl"] = _write_jsonl(out_dir / f"{split}.evaluator.jsonl", by_split[split]["evaluator"])
    files["personas.jsonl"] = _write_jsonl(out_dir / "personas.jsonl", builder.personas)
    files["history_events.jsonl"] = _write_jsonl(out_dir / "history_events.jsonl", builder.history_events)

    # schema v0.2: same structure as P14-E v0.1, split enum extended; no frozen file is modified.
    base_schema = json.loads((REPO / "data/p14e/benchmark/schema_v0.1.json").read_text(encoding="utf-8"))
    base_schema["$id"] = E2_VERSION
    base_schema["title"] = "P14-E2 Lite Shopping Preference Benchmark Schema v0.2"
    base_schema["description"] = (
        "Same agent-visible/evaluator-only split as P14-E v0.1; splits are train/dev/reserved_test. "
        "pilot_dev (P14-E) remains a separate, exposed split."
    )
    for schema_key in ("agent_visible", "evaluator_only"):
        props = base_schema[schema_key]["properties"]
        if "split" in props:
            props["split"] = {"enum": ["pilot_dev", "train", "dev", "reserved_test"]}
        if schema_key == "agent_visible":
            props["schema"]["const"] = E2_VERSION
            props["task_id"]["pattern"] = "^p14e2_lite_[0-9]{4}$"
        else:
            props["schema"]["const"] = E2_VERSION
    files["schema_v0.2.json"] = {
        "bytes": (out_dir / "schema_v0.2.json").stat().st_size if (out_dir / "schema_v0.2.json").exists() else 0,
        "sha256": "",
    }
    (out_dir / "schema_v0.2.json").write_text(json.dumps(base_schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    files["schema_v0.2.json"] = {"bytes": (out_dir / "schema_v0.2.json").stat().st_size,
                                 "sha256": _sha256(out_dir / "schema_v0.2.json")}

    review = []
    for ev in builder.evaluator:
        meta = ev["construction_metadata"]
        if meta["review_status"] == "pending_human_review":
            review.append({
                "task_id": ev["task_id"], "split": ev["split"], "task_type": ev["task_type"],
                "reasons": meta.get("review_reasons") or [],
                "expected_decision": ev["expected_decision"],
                "evidence": {
                    "effective_preferences": [p.get("value") for p in ev["effective_preferences"]],
                    "hard_constraints": [c.get("value") for c in ev["hard_constraints"]],
                    "verified_candidate_count": ev["labels"]["verified_candidate_count"],
                },
            })
    (out_dir / "review_checklist.json").write_text(
        json.dumps({"schema": "agentrecx.p14e2.review_checklist.v1", "tasks": review}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    type_counts = {split: dict(sorted(Counter(e["task_type"] for e in by_split[split]["evaluator"]).items()))
                   for split in SPLITS}
    decision_counts = {split: dict(sorted(Counter(e["expected_decision"] for e in by_split[split]["evaluator"]).items()))
                       for split in SPLITS}
    manifest = {
        "schema": "agentrecx.p14e2.benchmark_manifest.v1",
        "benchmark_version": E2_VERSION,
        "generator_version": E2_GENERATOR,
        "random_seed": E2_SEED,
        "parent_benchmark": "data/p14e/benchmark (frozen pilot_dev; untouched)",
        "catalogue": {"namespace": ELECTRONICS_NAMESPACE, "path": "data/p13r/electronics/catalog_meta.jsonl",
                      "stats": catalogue.stats},
        "splits": {
            split: {
                "tasks": len(by_split[split]["tasks"]),
                "task_types": type_counts[split],
                "expected_decisions": decision_counts[split],
                "persona_ids": [t["persona_id"] for t in by_split[split]["tasks"]],
                "family_ids": sorted({e["family_id"] for e in by_split[split]["evaluator"]}),
            }
            for split in SPLITS
        },
        "total_tasks": len(builder.tasks),
        "review_required": len(review),
        "files": files,
        "construction_warnings": builder.warnings,
        "notes": [
            "constructed preference histories; not real purchase logs",
            "labels are generated by the frozen independent evidence labeler, not copied from Scorer/Gate output",
            "reserved_test must not be used for search/scorer/gate/planner runs before a freeze decision",
            "original P14-E 40 pilot_dev tasks and labels remain unchanged",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    builder = LiteBuilder(catalogue)
    builder.build_all()
    manifest = write_outputs(builder, catalogue, OUT_DEFAULT)
    print(json.dumps({"total": manifest["total_tasks"],
                      "splits": {s: manifest["splits"][s]["tasks"] for s in SPLITS},
                      "task_types": {s: manifest["splits"][s]["task_types"] for s in SPLITS},
                      "decisions": {s: manifest["splits"][s]["expected_decisions"] for s in SPLITS},
                      "review_required": manifest["review_required"],
                      "warnings": builder.warnings}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
