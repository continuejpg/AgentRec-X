"""P14-E Shopping Preference Benchmark pilot generator (deterministic, construction-first).

Construction order is fixed: **persona/preferences -> history events -> freeze ->
current query -> evaluation rules**.  The labeler (``labeling.py``) reads the frozen
assets and the real catalogue; it never receives a pre-selected target product.

Run: ``python -m experiments.p14e.build_pilot`` (add ``--out-dir`` to override).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import MIN_RECOMMENDABLE, active_prefs, detect_conflict, label_task

GENERATOR_VERSION = "p14e-0.1.0"
BENCHMARK_VERSION = "agentrecx.p14e.shopping_preference.v0.1"
SEED = 20261108
SPLIT = "pilot_dev"
DATASET_NAME = "P14-E Shopping Preference Benchmark Pilot"

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data/p14e/benchmark"

FORM_ATTRS = ("over_ear", "on_ear", "in_ear")
PHRASES: dict[str, str] = {
    "wireless": "wireless",
    "wired": "wired",
    "noise_cancelling": "noise cancelling",
    "over_ear": "over-ear",
    "on_ear": "on-ear",
    "in_ear": "in-ear",
    "microphone": "a built-in microphone",
    "gaming": "gaming features",
    "waterproof": "water resistance",
    "foldable": "a foldable design",
    "mechanical": "mechanical switches",
    "rgb": "RGB lighting",
    "rechargeable": "a rechargeable battery",
    "multidevice": "multi-device pairing",
    "usb": "USB connectivity",
    "silent": "quiet operation",
    "ergonomic": "an ergonomic shape",
}
TYPE_WORD = {
    "headphones": "headphones",
    "earbuds": "earbuds",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "speaker": "speaker",
}
QUERY_TEMPLATES = {
    "headphones": {
        "daily_commute": "I need a pair of headphones for my daily commute.",
        "work_calls": "Can you recommend headphones for work calls?",
        "long_flights": "I want headphones for long flights.",
        "gaming": "Looking for headphones for gaming.",
    },
    "earbuds": {
        "workout": "I need earbuds for workouts.",
        "calls": "I want earbuds for calls.",
    },
    "keyboard": {
        "office": "I need a keyboard for my office desk.",
        "gaming": "Looking for a mechanical keyboard for gaming.",
        "travel": "I want a compact keyboard for travel.",
    },
    "mouse": {
        "office": "I need a mouse for my office desk.",
        "gaming": "Looking for a gaming mouse.",
        "travel": "I want a small mouse for travel.",
    },
    "speaker": {
        "home": "I need a speaker for my living room.",
        "outdoor": "I want a speaker for outdoor trips.",
    },
}

SCENARIO_PREFS: dict[tuple[str, str], tuple[list[str], list[str]]] = {
    ("headphones", "daily_commute"): (["noise_cancelling", "over_ear"], ["in_ear"]),
    ("headphones", "work_calls"): (["microphone", "over_ear"], ["in_ear"]),
    ("headphones", "long_flights"): (["foldable", "wireless"], ["on_ear"]),
    ("headphones", "gaming"): (["gaming", "microphone"], ["in_ear"]),
    ("earbuds", "workout"): (["waterproof", "wireless"], ["over_ear"]),
    ("earbuds", "calls"): (["microphone", "wireless"], ["over_ear"]),
    ("keyboard", "office"): (["silent", "wireless"], ["rgb"]),
    ("keyboard", "gaming"): (["mechanical", "rgb"], ["silent"]),
    ("keyboard", "travel"): (["wireless", "foldable"], ["mechanical"]),
    ("mouse", "office"): (["ergonomic", "wireless"], ["gaming"]),
    ("mouse", "gaming"): (["gaming", "rgb"], ["silent"]),
    ("mouse", "travel"): (["wireless", "rechargeable"], ["wired"]),
    ("speaker", "home"): (["wireless", "microphone"], ["wired"]),
    ("speaker", "outdoor"): (["waterproof", "rechargeable"], ["wired"]),
}

#: Hard-constraint combinations that are rare/absent in the 76,918 sampled catalogue.
#: Each list starts with a mutually-exclusive form-factor constraint so that candidates
#: with the opposite form factor are *violated* (not merely UNKNOWN): that is what makes
#: the honest decision ``no_verified_match`` rather than ``insufficient_evidence``.
NO_MATCH_HARD: dict[str, list[list[str]]] = {
    "headphones": [
        ["over_ear", "waterproof", "foldable", "noise_cancelling"],
        ["over_ear", "mechanical", "noise_cancelling", "foldable"],
        ["over_ear", "rgb", "waterproof", "multidevice"],
    ],
    "earbuds": [
        ["in_ear", "waterproof", "noise_cancelling", "multidevice"],
        ["in_ear", "mechanical", "rgb", "foldable"],
    ],
}
UNVERIFIABLE_HARD = [
    {"name": "price_under_50", "text": "My budget is under $50."},
    {"name": "three_year_warranty", "text": "It must come with a three-year warranty."},
]


def _phrase(name: str, *, plural: bool = False) -> str:
    phrase = PHRASES.get(name, name.replace("_", " "))
    if plural and not phrase.startswith("a "):
        phrase = phrase + " headphones"
    return phrase


def _pref(kind: str, value: str, polarity: str = "positive") -> dict[str, Any]:
    if kind == "form_factor":
        return {"kind": "form_factor", "value": value, "polarity": polarity}
    return {"kind": "attribute", "name": value, "value": value, "polarity": polarity}


def _prefs_from_lists(positives: Sequence[str], negatives: Sequence[str]) -> list[dict[str, Any]]:
    prefs: list[dict[str, Any]] = []
    for name in positives:
        prefs.append(_pref("form_factor" if name in FORM_ATTRS else "attribute", name, "positive"))
    for name in negatives:
        prefs.append(_pref("form_factor" if name in FORM_ATTRS else "attribute", name, "negative"))
    return prefs


@dataclass
class EventSpec:
    raw_text: str
    candidates: list[dict[str, Any]]


@dataclass
class PersonaSpec:
    persona_id: str
    user_id: str
    product_type: str
    scenario: str
    preferences: list[dict[str, Any]]
    seed: int


@dataclass
class TaskSpec:
    task_type: str
    product_type: str
    scenario: str
    query: str
    personas: list[PersonaSpec]
    history_events: list[list[EventSpec]] = field(default_factory=list)  # per persona
    hard_constraints: list[dict[str, Any]] = field(default_factory=list)
    intent_note: str = ""
    fallback: bool = False


def _event(event_seed: str, event_count: int, candidates: list[dict[str, Any]], raw: str) -> EventSpec:
    return EventSpec(raw_text=raw, candidates=candidates)


def _history_for_prefs(prefs: Sequence[dict[str, Any]], *, product_type: str, rng: random.Random) -> list[EventSpec]:
    positives = [p for p in prefs if p["polarity"] == "positive"]
    negatives = [p for p in prefs if p["polarity"] == "negative"]
    events: list[EventSpec] = []
    if positives:
        first = positives[0]
        events.append(
            _event(
                "p1",
                0,
                [dict(first, status="assert")],
                f"I prefer {_phrase(str(first['value']))}.",
            )
        )
    if len(positives) > 1:
        second = positives[1]
        events.append(
            _event(
                "p2",
                1,
                [dict(second, status="assert")],
                f"I also prefer {_phrase(str(second['value']))}.",
            )
        )
    if negatives:
        negative = negatives[0]
        events.append(
            _event(
                "n1",
                2,
                [dict(negative, status="assert")],
                f"I don't like {_phrase(str(negative['value']))}.",
            )
        )
    events.append(
        _event(
            "ctx",
            3,
            [],
            f"I've been thinking about upgrading my {TYPE_WORD[product_type]} setup for a while.",
        )
    )
    return events


def _history_for_update(old: str, new: str, product_type: str) -> list[EventSpec]:
    old_kind = "form_factor" if old in FORM_ATTRS else "attribute"
    new_kind = "form_factor" if new in FORM_ATTRS else "attribute"
    return [
        _event(
            "old",
            0,
            [_pref(old_kind, old, "positive") | {"status": "assert"}],
            f"I prefer {_phrase(old)}.",
        ),
        _event(
            "extra",
            1,
            [_pref("attribute", "wireless", "positive") | {"status": "assert"}],
            f"I also prefer {_phrase('wireless')}.",
        ),
        _event(
            "remove_old",
            2,
            [_pref(old_kind, old, "positive") | {"status": "withdraw"}],
            f"Forget my earlier preference for {_phrase(old)}.",
        ),
        _event(
            "new",
            3,
            [_pref(new_kind, new, "positive") | {"status": "assert"}],
            f"I now prefer {_phrase(new)}.",
        ),
        _event(
            "ctx",
            4,
            [],
            f"I'm ready to replace my old {TYPE_WORD[product_type]}.",
        ),
    ]


def _history_for_withdraw(name: str, product_type: str) -> list[EventSpec]:
    kind = "form_factor" if name in FORM_ATTRS else "attribute"
    return [
        _event(
            "p1",
            0,
            [_pref(kind, name, "positive") | {"status": "assert"}],
            f"I prefer {_phrase(name)}.",
        ),
        _event(
            "p2",
            1,
            [_pref("attribute", "wireless", "positive") | {"status": "assert"}],
            f"I also prefer {_phrase('wireless')}.",
        ),
        _event(
            "withdraw",
            2,
            [_pref(kind, name, "positive") | {"status": "withdraw"}],
            f"Forget my earlier preference for {_phrase(name)}.",
        ),
        _event(
            "ctx",
            3,
            [],
            f"I'm just comparing {TYPE_WORD[product_type]} options now.",
        ),
    ]


def _history_for_conflict(left: str, right: str, product_type: str) -> list[EventSpec]:
    left_kind = "form_factor" if left in FORM_ATTRS else "attribute"
    right_kind = "form_factor" if right in FORM_ATTRS else "attribute"
    return [
        _event(
            "p1",
            0,
            [_pref(left_kind, left, "positive") | {"status": "assert"}],
            f"I want {_phrase(left)} for my {TYPE_WORD[product_type]}, that part is clear.",
        ),
        _event(
            "p2",
            1,
            [_pref(right_kind, right, "positive") | {"status": "assert"}],
            f"But I also really want {_phrase(right)}, so I'm not sure which way to go.",
        ),
        _event(
            "ctx",
            2,
            [],
            f"I keep going back and forth on what I want from a {TYPE_WORD[product_type]}.",
        ),
    ]


def _history_for_ask(product_type: str) -> list[EventSpec]:
    return [
        _event(
            "vague",
            0,
            [_pref("attribute", "comfort", "positive") | {"status": "assert"}],
            "Maybe I prefer something comfortable?",
        ),
        _event(
            "withdraw",
            1,
            [_pref("attribute", "comfort", "positive") | {"status": "withdraw"}],
            "Forget my earlier preference for comfort.",
        ),
        _event(
            "ctx",
            2,
            [],
            f"I'm only starting to look at {TYPE_WORD[product_type]} options.",
        ),
    ]


def _hard_constraint(name: str, constraint_id: str) -> dict[str, Any]:
    kind = "form_factor" if name in FORM_ATTRS else "attribute"
    text = f"It needs to be {_phrase(name)}."
    return {
        "constraint_id": constraint_id,
        "kind": kind,
        "value": name,
        "source_text": text,
        "attribute": name,
    }


def _unverifiable_constraint(spec: dict[str, Any], constraint_id: str) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "kind": "unverifiable",
        "value": spec["name"],
        "name": spec["name"],
        "source_text": spec["text"],
        "catalog_field": name_of_unverifiable(spec["name"]),
    }


def name_of_unverifiable(name: str) -> str:
    if "price" in name:
        return "price"
    if "warranty" in name:
        return "warranty"
    return name


def _query_with_constraints(query: str, constraints: Sequence[dict[str, Any]]) -> str:
    text = query
    for constraint in constraints:
        text = text + " " + str(constraint["source_text"])
    return text


def _history_sessions(events: Sequence[EventSpec], task_id: str, persona_id: str, base: date) -> list[dict[str, Any]]:
    sessions = []
    for index, event in enumerate(events):
        event_date = base + timedelta(days=index * 3)
        sessions.append(
            {
                "session_id": f"{task_id}_{persona_id}_s{index + 1}",
                "date": event_date.isoformat(),
                "turns": [{"role": "user", "content": event.raw_text}],
            }
        )
    return sessions


def _history_events(events: Sequence[EventSpec], task_id: str, persona_id: str, base: date) -> list[dict[str, Any]]:
    events_out = []
    for index, event in enumerate(events):
        event_date = base + timedelta(days=index * 3)
        events_out.append(
            {
                "event_id": f"{task_id}_{persona_id}_e{index + 1}",
                "session_id": f"{task_id}_{persona_id}_s{index + 1}",
                "timestamp": f"{event_date.isoformat()}T10:00:00",
                "sequence": index + 1,
                "speaker": "user",
                "raw_text": event.raw_text,
                "preference_candidates": event.candidates,
                "source_type": "constructed_template",
                "history_source": "constructed_template",
                "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1",
                "generator_version": GENERATOR_VERSION,
                "review_status": "automated_checked",
            }
        )
    return events_out


class Builder:
    def __init__(self, catalogue: ElectronicsCatalogue) -> None:
        self.catalogue = catalogue
        self.rng = random.Random(SEED)
        self.tasks: list[dict[str, Any]] = []
        self.evaluator: list[dict[str, Any]] = []
        self.personas: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.construction_warnings: list[str] = []

    # -- helpers ----------------------------------------------------------- #
    def _support_count(self, product_type: str, prefs: Sequence[dict[str, Any]], hard: Sequence[dict[str, Any]]) -> int:
        pool = self.catalogue.by_type(product_type)
        from experiments.p14e.labeling import assess_candidate

        count = 0
        for product in pool:
            assessment = assess_candidate(product, hard_constraints=hard, preferences=prefs)
            if assessment.acceptable:
                count += 1
        return count

    def _persona(self, index: int, product_type: str, scenario: str, prefs: Sequence[dict[str, Any]]) -> PersonaSpec:
        return PersonaSpec(
            persona_id=f"p14e_persona_{index:04d}",
            user_id=f"p14e_user_{index:04d}",
            product_type=product_type,
            scenario=scenario,
            preferences=list(prefs),
            seed=SEED + index,
        )

    def _base_task(self, task_type: str, task_index: int, product_type: str, scenario: str,
                   query: str, persona: PersonaSpec, events: Sequence[EventSpec],
                   hard: Sequence[dict[str, Any]], intent_note: str) -> None:
        task_id = f"p14e_pilot_{task_index:04d}"
        base_date = date(2025, 8, 1) + timedelta(days=(task_index % 20) * 4)
        history_sessions = _history_sessions(events, task_id, persona.persona_id, base_date)
        history_events = _history_events(events, task_id, persona.persona_id, base_date)
        agent_task = {
            "schema": BENCHMARK_VERSION,
            "task_id": task_id,
            "split": SPLIT,
            "user_id": persona.user_id,
            "persona_id": persona.persona_id,
            "current_query": _query_with_constraints(query, hard),
            "current_time": (base_date + timedelta(days=len(events) * 3)).isoformat() + "T12:00:00",
            "history_sessions": history_sessions,
            "allowed_tools": ["mem_search", "mem_view", "mem_summarize_by_date", "product_search", "product_view"],
            "catalog_namespace": ELECTRONICS_NAMESPACE,
            "tool_budget": {"max_tool_calls": 12, "max_search_calls": 6, "max_product_views": 6},
            "public_constraints": [str(c["source_text"]) for c in hard],
            "construction_metadata": {
                "history_source": "constructed_template",
                "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1",
                "generator_version": GENERATOR_VERSION,
                "random_seed": SEED,
                "review_status": "automated_checked",
                "benchmark_split": SPLIT,
            },
        }
        all_prefs: list[dict[str, Any]] = []
        all_events: list[dict[str, Any]] = []
        for index, event in enumerate(history_events):
            all_events.append(event)
        effective, withdrawn = active_prefs(history_events)
        pool = self.catalogue.by_type(product_type)
        task_prereq = {"task_id": task_id}
        labels = label_task(task_prereq, {"effective_preferences": effective, "hard_constraints": list(hard)}, pool)
        conflict = detect_conflict(effective)
        review_reasons: list[str] = []
        if task_type in {"same_query_persona_pair"}:
            review_reasons.append("same-query contrast group requires human review of preference distinctness")
        if task_type == "no_verified_match":
            review_reasons.append("no_verified_match is relative to the 76,918 sampled catalogue, not proof of nonexistence")
        if hard and any(c["kind"] == "unverifiable" for c in hard):
            review_reasons.append("hard constraint not verifiable from P13-R metadata (UNKNOWN, not fabricated)")
        if labels["expected_decision"] == "recommend" and labels["verified_candidate_count"] < MIN_RECOMMENDABLE:
            review_reasons.append("few verified candidates; candidate coverage may be catalogue-limited")
        evaluator_task = {
            "schema": BENCHMARK_VERSION,
            "task_id": task_id,
            "task_type": task_type,
            "split": SPLIT,
            "user_id": persona.user_id,
            "persona_id": persona.persona_id,
            "persona": {
                "persona_id": persona.persona_id,
                "user_id": persona.user_id,
                "product_type": product_type,
                "scenario": scenario,
                "profile_source": "synthetic_scenario_profile",
                "preferences": persona.preferences,
            },
            "effective_preferences": effective,
            "withdrawn_preferences": withdrawn,
            "hard_constraints": list(hard),
            "catalog_namespace": ELECTRONICS_NAMESPACE,
            "product_type_pool": product_type,
            "expected_decision": labels["expected_decision"],
            "decision_rationale": labels["decision_rationale"],
            "labels": labels,
            "construction_metadata": {
                "history_source": "constructed_template",
                "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1",
                "generator_version": GENERATOR_VERSION,
                "random_seed": SEED,
                "review_status": "pending_human_review" if review_reasons else "automated_checked",
                "review_reasons": review_reasons,
                "conflict": conflict,
                "labeler_version": "p14e-labeling-0.1.0",
            },
        }
        agent_task["construction_metadata"]["review_status"] = evaluator_task["construction_metadata"]["review_status"]
        self.tasks.append(agent_task)
        self.evaluator.append(evaluator_task)
        self.personas.append(
            {
                "persona_id": persona.persona_id,
                "user_id": persona.user_id,
                "product_type": product_type,
                "scenario": scenario,
                "preferences": persona.preferences,
                "profile_source": "synthetic_scenario_profile",
                "generation": {"generator_version": GENERATOR_VERSION, "random_seed": SEED},
            }
        )
        self.history_events.extend(history_events)

    # -- task families ------------------------------------------------------ #
    def build_regular(self, start_index: int) -> int:
        specs = [
            ("headphones", "daily_commute", None),
            ("headphones", "work_calls", "over_ear"),
            ("headphones", "long_flights", None),
            ("headphones", "gaming", None),
            ("keyboard", "office", "wireless"),
            ("keyboard", "gaming", None),
            ("keyboard", "travel", None),
            ("mouse", "office", None),
            ("mouse", "gaming", None),
            ("mouse", "travel", "wireless"),
            ("speaker", "home", None),
            ("earbuds", "workout", "waterproof"),
        ]
        index = start_index
        for offset, (product_type, scenario, hard_name) in enumerate(specs):
            positives, negatives = SCENARIO_PREFS[(product_type, scenario)]
            prefs = _prefs_from_lists(positives, negatives)
            hard = [_hard_constraint(hard_name, f"hc_{index}_{hard_name}")] if hard_name else []
            persona = self._persona(index, product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=self.rng)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("regular_history", index, product_type, scenario, query, persona, events, hard,
                            "current query omits the historical soft preferences")
            index += 1
        return index

    def build_pairs(self, start_index: int) -> int:
        pairs = [
            ("headphones", "daily_commute", "Which headphones would you suggest for commuting?",
             (["noise_cancelling", "over_ear"], ["in_ear"]),
             (["foldable", "wireless"], ["over_ear"])),
            ("headphones", "work_calls", "Which headphones are good for work calls?",
             (["microphone", "over_ear"], ["in_ear"]),
             (["microphone", "wireless"], ["over_ear"])),
            ("keyboard", "office", "Which keyboard fits an office desk setup?",
             (["silent", "wired"], ["rgb"]),
             (["rgb", "mechanical"], ["silent"])),
            ("mouse", "office", "Which mouse fits an office desk setup?",
             (["ergonomic", "wireless"], ["gaming"]),
             (["gaming", "rgb"], ["ergonomic"])),
            ("speaker", "home", "Which speaker fits a living room setup?",
             (["wireless", "microphone"], ["rechargeable"]),
             (["rechargeable", "waterproof"], ["microphone"])),
        ]
        index = start_index
        for product_type, scenario, query, a_spec, b_spec in pairs:
            for side, (positives, negatives) in (("a", a_spec), ("b", b_spec)):
                prefs = _prefs_from_lists(positives, negatives)
                persona = self._persona(index, product_type, scenario, prefs)
                events = _history_for_prefs(prefs, product_type=product_type, rng=self.rng)
                self._base_task(
                    "same_query_persona_pair", index, product_type, scenario, query, persona, events, [],
                    f"same current query as the paired user; history carries different preferences ({side})",
                )
                index += 1
        return index

    def build_dynamics(self, start_index: int) -> int:
        index = start_index
        updates = [
            ("headphones", "daily_commute", "over_ear", "in_ear"),
            ("keyboard", "office", "wireless", "wired"),
            ("mouse", "travel", "wireless", "wired"),
        ]
        for product_type, scenario, old, new in updates:
            persona = self._persona(
                index,
                product_type,
                scenario,
                [
                    _pref("form_factor" if old in FORM_ATTRS else "attribute", old, "positive"),
                    _pref("attribute", "wireless", "positive"),
                ]
                if old != "wireless"
                else [_pref("attribute", "wireless", "positive")],
            )
            events = _history_for_update(old, new, product_type)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("preference_update", index, product_type, scenario, query, persona, events, [],
                            f"history updates {old} -> {new}")
            index += 1
        conflicts = [
            ("headphones", "daily_commute", "wireless", "wired"),
            ("keyboard", "office", "wireless", "wired"),
            ("mouse", "travel", "wireless", "wired"),
        ]
        for product_type, scenario, left, right in conflicts:
            persona = self._persona(index, product_type, scenario,
                                    [_pref("attribute", left, "positive"), _pref("attribute", right, "positive")])
            events = _history_for_conflict(left, right, product_type)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("preference_conflict", index, product_type, scenario, query, persona, events, [],
                            "unresolved conflicting history preferences; a clarifying question is expected")
            index += 1
        return index

    def build_withdraw(self, start_index: int) -> int:
        index = start_index
        specs = [
            ("headphones", "daily_commute", "noise_cancelling"),
            ("keyboard", "office", "wireless"),
        ]
        for product_type, scenario, name in specs:
            kind = "form_factor" if name in FORM_ATTRS else "attribute"
            persona = self._persona(
                index, product_type, scenario,
                [_pref(kind, name, "positive"), _pref("attribute", "wireless", "positive")],
            )
            events = _history_for_withdraw(name, product_type)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("preference_withdrawal", index, product_type, scenario, query, persona, events, [],
                            f"history explicitly withdraws the earlier {name} preference")
            index += 1
        return index

    def build_ask(self, start_index: int) -> int:
        index = start_index
        specs = [
            ("headphones", "daily_commute", "I need some new headphones."),
            ("earbuds", "workout", "I need some new earbuds."),
            ("keyboard", "office", "I need some new keyboards."),
            ("mouse", "office", "I need some new mice."),
            ("speaker", "home", "I need some new speakers."),
        ]
        for product_type, scenario, query in specs:
            persona = self._persona(index, product_type, scenario, [])
            events = _history_for_ask(product_type)
            self._base_task("insufficient_information", index, product_type, scenario, query, persona, events, [],
                            "current request is too vague and history does not establish a usable preference; clarification expected")
            index += 1
        return index

    def build_no_match(self, start_index: int) -> int:
        index = start_index
        created = 0
        zero_match_specs = [
            ("headphones", "daily_commute"),
            ("earbuds", "workout"),
            ("headphones", "gaming"),
        ]
        for product_type, scenario in zero_match_specs:
            positives, negatives = SCENARIO_PREFS[(product_type, scenario)]
            prefs = _prefs_from_lists(positives[:1], [])
            pool = self.catalogue.by_type(product_type)
            chosen = None
            for combo in NO_MATCH_HARD[product_type]:
                hard = [_hard_constraint(name, f"hc_{index}_{name}") for name in combo]
                from experiments.p14e.labeling import assess_candidate

                verified = sum(1 for p in pool if assess_candidate(p, hard_constraints=hard, preferences=prefs).acceptable)
                if verified == 0:
                    chosen = combo
                    break
            if chosen is None:
                chosen = NO_MATCH_HARD[product_type][-1]
                self.construction_warnings.append(f"{product_type}: no zero-match combo found; used {chosen}")
            hard = [_hard_constraint(name, f"hc_{index}_{name}") for name in chosen]
            persona = self._persona(index, product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=self.rng)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("no_verified_match", index, product_type, scenario, query, persona,
                            events, hard, "hard constraint combination has zero verified candidates in the sampled catalogue")
            index += 1
            created += 1
        base_specs = [("headphones", "daily_commute"), ("mouse", "office")]
        for offset, (product_type, scenario) in enumerate(base_specs):
            positives, negatives = SCENARIO_PREFS[(product_type, scenario)]
            prefs = _prefs_from_lists(positives, negatives)
            unver = UNVERIFIABLE_HARD[offset]
            hard = [_unverifiable_constraint(unver, f"hc_{index}_unverifiable")]
            persona = self._persona(index, product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=self.rng)
            query = QUERY_TEMPLATES[product_type][scenario]
            self._base_task("unverifiable_constraint", index, product_type, scenario, query, persona, events, hard,
                            f"hard constraint {unver['name']} cannot be verified from catalog_meta.jsonl; UNKNOWN is expected")
            index += 1
        return index

    def build_all(self) -> None:
        index = 1
        index = self.build_regular(index)
        index = self.build_pairs(index)
        index = self.build_dynamics(index)
        index = self.build_withdraw(index)
        index = self.build_ask(index)
        index = self.build_no_match(index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(builder: Builder, catalogue: ElectronicsCatalogue, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "agent_tasks.jsonl": builder.tasks,
        "evaluator_only.jsonl": builder.evaluator,
        "personas.jsonl": builder.personas,
        "history_events.jsonl": builder.history_events,
    }
    hashes: dict[str, Any] = {}
    for name, rows in files.items():
        path = out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path), "rows": len(rows)}

    task_types: dict[str, int] = {}
    decisions: dict[str, int] = {}
    review_required = 0
    for item in builder.evaluator:
        task_types[item["task_type"]] = task_types.get(item["task_type"], 0) + 1
    for item in builder.evaluator:
        decisions[item["expected_decision"]] = decisions.get(item["expected_decision"], 0) + 1
        if item["construction_metadata"]["review_status"] == "pending_human_review":
            review_required += 1
    manifest = {
        "schema": "agentrecx.p14e.benchmark_manifest.v1",
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_name": DATASET_NAME,
        "split": SPLIT,
        "generator_version": GENERATOR_VERSION,
        "labeler_version": "p14e-labeling-0.1.0",
        "random_seed": SEED,
        "catalogue": {
            "namespace": ELECTRONICS_NAMESPACE,
            "path": "data/p13r/electronics/catalog_meta.jsonl",
            "stats": catalogue.stats,
        },
        "counts": {
            "tasks": len(builder.tasks),
            "personas": len(builder.personas),
            "history_events": len(builder.history_events),
            "task_types": dict(sorted(task_types.items())),
            "expected_decisions": dict(sorted(decisions.items())),
            "review_required": review_required,
        },
        "files": hashes,
        "construction_warnings": builder.construction_warnings,
        "notes": [
            "pilot_dev only; never rename this split into a frozen test split",
            "all preference histories are constructed templates, not real user purchase logs",
            "catalogue attributes are lexical evidence over real Amazon-C4 metadata; price/brand/material/stock are UNKNOWN",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P14-E shopping preference benchmark pilot")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    catalogue = ElectronicsCatalogue.load()
    builder = Builder(catalogue)
    builder.build_all()
    manifest = write_outputs(builder, catalogue, args.out_dir)
    print(json.dumps(manifest["counts"], indent=1, ensure_ascii=False))
    print("warnings:", builder.construction_warnings)


if __name__ == "__main__":
    main()
