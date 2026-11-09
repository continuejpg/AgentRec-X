"""Build 100 previously unseen P14-B3-Eval100 shopping tasks.

Reuses the frozen P14-E2 construction primitives and independent evidence labeler:
80 recommend tasks (20 no-history, 24 history, 20 same-query persona pairs, 8 update,
8 withdrawal) and 20 non-recommend tasks (8 clarification, 4 conflict, 4 no-verified,
4 unverifiable).  New persona/task/family IDs; no prior task/query/persona reuse.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from experiments.p14e.build_pilot import (
    FORM_ATTRS, SCENARIO_PREFS, TYPE_WORD, EventSpec, _event, _hard_constraint,
    _history_events, _history_for_conflict, _history_for_prefs, _history_for_update,
    _history_for_withdraw, _history_sessions, _pref, _prefs_from_lists,
    _query_with_constraints, _unverifiable_constraint, UNVERIFIABLE_HARD,
)
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import MIN_RECOMMENDABLE, active_prefs, label_task

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/p14b3_eval100"
VERSION = "agentrecx.p14b3eval100.shopping_preference.v0.1"
GENERATOR = "p14b3-eval100-0.1.0"
SEED = 20261109
TASK_PREFIX = "p14b3_eval100"

# New agent-visible query phrasings (not the frozen P14-E/E2 strings).
Q = {
    "headphones": [
        "I'm replacing my old headphones with a dependable new pair.",
        "Which headphones would you recommend for long work-from-home days?",
        "I want headphones for everyday listening and occasional travel.",
        "Can you help me pick a comfortable pair of headphones?",
        "I need new headphones for music and phone calls.",
    ],
    "earbuds": [
        "I'm looking for a reliable pair of earbuds for daily use.",
        "Which earbuds should I get for exercising and commuting?",
        "I need earbuds that are easy to carry and use for calls.",
        "Can you recommend earbuds for listening on the go?",
    ],
    "keyboard": [
        "I need a new keyboard for my home desk setup.",
        "Which keyboard would you recommend for writing and office work?",
        "I'm building a workspace and need a dependable keyboard.",
        "Can you help me choose a keyboard for daily typing?",
    ],
    "mouse": [
        "I need a new mouse for my computer desk.",
        "Which mouse would you recommend for everyday office work?",
        "I'm setting up a workspace and need a comfortable mouse.",
        "Can you help me pick a mouse for general computer use?",
    ],
    "speaker": [
        "I'm looking for a speaker for my living space.",
        "Which speaker would you recommend for home listening?",
        "I need a new speaker for music at home.",
        "Can you help me choose a speaker for a small room?",
    ],
}
SCENARIO_KEY = {
    "headphones": ("daily_commute", "work_calls", "gaming"),
    "earbuds": ("workout", "calls"),
    "keyboard": ("office", "gaming", "travel"),
    "mouse": ("office", "gaming", "travel"),
    "speaker": ("home",),
}
SAFE_HARD = ("wireless", "noise_cancelling", "over_ear", "microphone", "mechanical",
             "rgb", "waterproof", "foldable", "ergonomic", "rechargeable")
NO_MATCH_COMBOS = (["over_ear", "mechanical"], ["over_ear", "rgb"], ["over_ear", "foldable"],
                   ["over_ear", "waterproof"], ["over_ear", "noise_cancelling", "foldable"])


def _scenario_prefs(product_type: str, scenario: str):
    return SCENARIO_PREFS[(product_type, scenario)]


class Builder:
    def __init__(self, catalogue: ElectronicsCatalogue) -> None:
        self.catalogue = catalogue
        self.tasks: list[dict[str, Any]] = []
        self.evaluator: list[dict[str, Any]] = []
        self.personas: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.index = 0
        self.persona_index = 0
        self.used_queries: set[str] = set()

    def _persona(self, product_type: str, scenario: str, prefs: list[dict[str, Any]]) -> dict[str, Any]:
        self.persona_index += 1
        return {"persona_id": f"{TASK_PREFIX}_persona_{self.persona_index:04d}",
                "user_id": f"{TASK_PREFIX}_user_{self.persona_index:04d}",
                "product_type": product_type, "scenario": scenario,
                "preferences": list(prefs), "seed": SEED + self.persona_index}

    def _context(self, index: int) -> str:
        times = ("morning", "afternoon", "evening", "weekend")
        rooms = ("home office", "living room", "study", "bedroom", "workshop", "kitchen", "dining area", "den")
        days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        return (f"For this purchase I'll mainly use it in my {rooms[index % len(rooms)]} "
                f"during {times[(index // 2) % len(times)]} time on {days[index % len(days)]}s.")

    def add_task(self, *, family: str, family_id: str, product_type: str, scenario: str,
                 query: str, persona: dict[str, Any], events: list[EventSpec],
                 hard: list[dict[str, Any]], intended: set[str], intent_note: str,
                 allow_duplicate_query: bool = False) -> dict[str, Any]:
        self.index += 1
        task_id = f"{TASK_PREFIX}_{self.index:04d}"
        if not allow_duplicate_query:
            query = query + " " + self._context(self.index)
            if query in self.used_queries:
                query = query + f" (batch {self.index})"
            self.used_queries.add(query)
        base_date = date(2025, 11, 1) + timedelta(days=(self.index % 25) * 3)
        persona_spec = {
            "persona_id": persona["persona_id"], "user_id": persona["user_id"],
            "product_type": persona["product_type"], "scenario": persona["scenario"],
            "preferences": list(persona["preferences"]), "seed": int(persona["seed"]),
        }
        history_sessions = _history_sessions(events, task_id, persona["persona_id"], base_date)
        history_events = _history_events(events, task_id, persona["persona_id"], base_date)
        effective, withdrawn = active_prefs(history_events)
        pool = self.catalogue.by_type(product_type)
        labels = label_task({}, {"effective_preferences": effective, "hard_constraints": hard}, pool)
        if labels["expected_decision"] not in intended:
            raise ValueError(f"{family}: expected {labels['expected_decision']} not in {intended}")
        if labels["expected_decision"] == "recommend" and labels["verified_candidate_count"] < MIN_RECOMMENDABLE:
            raise ValueError(f"{family}: too few verified candidates")
        review_reasons = ["new Eval100 batch requires human review of task/evidence",
                          "same-query contrast requires review of preference distinctness"]
        if family == "same_query_persona_pair":
            review_reasons.append("pair preference distinctness review required")
        if family in {"no_verified_match", "unverifiable_constraint", "preference_conflict"}:
            review_reasons.append("non-recommend outcome relative to sampled catalogue needs review")
        agent = {
            "schema": VERSION, "task_id": task_id, "split": "eval100",
            "user_id": persona["user_id"], "persona_id": persona["persona_id"],
            "current_query": query, "current_time": base_date.isoformat() + "T12:00:00",
            "history_sessions": history_sessions,
            "allowed_tools": ["mem_search", "mem_view", "mem_summarize_by_date", "product_search", "product_view"],
            "catalog_namespace": ELECTRONICS_NAMESPACE,
            "tool_budget": {"max_tool_calls": 12, "max_search_calls": 6, "max_product_views": 6},
            "public_constraints": [str(c["source_text"]) for c in hard],
            "construction_metadata": {
                "history_source": "constructed_template", "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1", "generator_version": GENERATOR,
                "random_seed": SEED, "review_status": "pending_human_review",
                "benchmark_split": "eval100",
            },
        }
        evaluator = {
            "schema": VERSION, "task_id": task_id, "task_type": family, "family_id": family_id,
            "split": "eval100", "user_id": persona["user_id"], "persona_id": persona["persona_id"],
            "persona": {"persona_id": persona["persona_id"], "user_id": persona["user_id"],
                        "product_type": product_type, "scenario": scenario,
                        "profile_source": "synthetic_scenario_profile", "preferences": persona["preferences"]},
            "effective_preferences": effective, "withdrawn_preferences": withdrawn,
            "hard_constraints": hard, "catalog_namespace": ELECTRONICS_NAMESPACE,
            "product_type_pool": product_type, "expected_decision": labels["expected_decision"],
            "decision_rationale": labels["decision_rationale"], "labels": labels,
            "construction_metadata": {
                "history_source": "constructed_template", "profile_source": "synthetic_scenario_profile",
                "construction_method": "template_generation_v1", "generator_version": GENERATOR,
                "random_seed": SEED, "review_status": "pending_human_review",
                "review_reasons": review_reasons, "labeler_version": "p14e-labeling-0.1.0",
                "label_basis": "independent evidence-based labeler over real catalogue; no model ranking used",
                "intent_note": intent_note,
            },
        }
        self.tasks.append(agent)
        self.evaluator.append(evaluator)
        self.personas.append({"persona_id": persona["persona_id"], "user_id": persona["user_id"],
                              "product_type": product_type, "scenario": scenario,
                              "preferences": persona["preferences"],
                              "profile_source": "synthetic_scenario_profile", "split": "eval100",
                              "family_id": family_id,
                              "generation": {"generator_version": GENERATOR, "random_seed": SEED}})
        self.history_events.extend(history_events)
        return labels

    def _queries(self, product_type: str, scenario: str, index: int) -> str:
        variants = Q[product_type]
        return variants[index % len(variants)]

    def add_no_history(self, count: int) -> None:
        types = list(Q)
        made = attempt = 0
        while made < count:
            product_type = types[made % len(types)]
            scenario = SCENARIO_KEY[product_type][attempt % len(SCENARIO_KEY[product_type])]
            hard_name = SAFE_HARD[attempt % len(SAFE_HARD)]
            hard = [_hard_constraint(hard_name, f"hc_eval100_{made}_{hard_name}")]
            query = _query_with_constraints(self._queries(product_type, scenario, made), hard)
            persona = self._persona(product_type, scenario, [])
            events = [_event("ctx", 0, [], f"I'm shopping for {TYPE_WORD[product_type]}.")]
            try:
                self.add_task(family="no_history_recommend", family_id=f"eval100_nohist_{made + 1:02d}",
                              product_type=product_type, scenario=scenario, query=query, persona=persona,
                              events=events, hard=hard, intended={"recommend"},
                              intent_note="no history; hard constraint only")
                made += 1
            except ValueError:
                attempt += 1
                if attempt > 40:
                    raise RuntimeError(f"no_history short: {made}/{count}")

    def add_history(self, count: int) -> None:
        types = list(Q)
        made = attempt = 0
        while made < count:
            product_type = types[made % len(types)]
            scenario = SCENARIO_KEY[product_type][attempt % len(SCENARIO_KEY[product_type])]
            positives, negatives = _scenario_prefs(product_type, scenario)
            prefs = _prefs_from_lists(positives, negatives)
            query = self._queries(product_type, scenario, made)
            persona = self._persona(product_type, scenario, prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(SEED + made))
            try:
                self.add_task(family="history_influenced", family_id=f"eval100_hist_{made + 1:02d}",
                              product_type=product_type, scenario=scenario, query=query, persona=persona,
                              events=events, hard=[], intended={"recommend"},
                              intent_note="current query omits soft preferences; legal memory read recovers them")
                made += 1
            except ValueError:
                attempt += 1
                if attempt > 60:
                    raise RuntimeError(f"history short: {made}/{count}")

    def add_pairs(self, pairs: int) -> None:
        specs = [
            ("headphones", "Which headphones should I get for commuting?",
             (["noise_cancelling", "over_ear"], ["in_ear"]), (["foldable", "wireless"], ["over_ear"])),
            ("headphones", "Which headphones are best for calls?",
             (["microphone", "over_ear"], ["in_ear"]), (["microphone", "wireless"], ["over_ear"])),
            ("keyboard", "Which keyboard should I get for an office desk?",
             (["silent", "wired"], ["rgb"]), (["rgb", "mechanical"], ["silent"])),
            ("mouse", "Which mouse should I get for an office desk?",
             (["ergonomic", "wireless"], ["gaming"]), (["gaming", "rgb"], ["ergonomic"])),
            ("speaker", "Which speaker should I get for a living room?",
             (["wireless", "microphone"], ["rechargeable"]), (["rechargeable", "waterproof"], ["microphone"])),
            ("earbuds", "Which earbuds work best for workouts?",
             (["waterproof", "wireless"], ["over_ear"]), (["microphone", "rechargeable"], ["over_ear"])),
            ("keyboard", "Which keyboard should I get for gaming?",
             (["mechanical", "rgb"], ["silent"]), (["wireless", "rechargeable"], ["wired"])),
            ("mouse", "Which mouse is best for travel?",
             (["wireless", "rechargeable"], ["wired"]), (["ergonomic", "silent"], ["gaming"])),
            ("earbuds", "Which earbuds are good for commuting?",
             (["wireless", "microphone"], ["over_ear"]), (["waterproof", "rechargeable"], ["wired"])),
            ("headphones", "Which headphones are best for long trips?",
             (["wireless", "foldable"], ["in_ear"]), (["noise_cancelling", "over_ear"], ["foldable"])),
        ]
        made = 0
        for attempt, (product_type, base_query, spec_a, spec_b) in enumerate(specs * 3):
            if made >= pairs:
                break
            scenario = SCENARIO_KEY[product_type][attempt % len(SCENARIO_KEY[product_type])]
            query = base_query + f" I'm comparing a few options for an upcoming purchase ({attempt + 1})."
            snapshot = (len(self.tasks), len(self.evaluator), len(self.personas), len(self.history_events))
            ok = True
            for side, (positives, negatives) in (("a", spec_a), ("b", spec_b)):
                prefs = _prefs_from_lists(positives, negatives)
                persona = self._persona(product_type, scenario, prefs)
                events = _history_for_prefs(prefs, product_type=product_type,
                                            rng=random.Random(SEED + 100 + attempt))
                try:
                    self.add_task(family="same_query_persona_pair",
                                  family_id=f"eval100_pair_{made + 1:02d}",
                                  product_type=product_type, scenario=scenario, query=query,
                                  persona=persona, events=events, hard=[], intended={"recommend"},
                                  intent_note="same current query, different recoverable preferences",
                                  allow_duplicate_query=True)
                except ValueError:
                    ok = False
                    break
            if not ok:
                del self.tasks[snapshot[0]:]
                del self.evaluator[snapshot[1]:]
                del self.personas[snapshot[2]:]
                del self.history_events[snapshot[3]:]
                continue
            made += 1
        if made < pairs:
            raise RuntimeError(f"pairs short: {made}/{pairs}")

    def add_updates(self, count: int) -> None:
        specs = [("headphones", "over_ear", "in_ear"), ("keyboard", "wireless", "wired"),
                 ("mouse", "wireless", "wired"), ("speaker", "wireless", "wired"),
                 ("earbuds", "over_ear", "in_ear")]
        made = attempt = 0
        while made < count:
            product_type, old, new = specs[attempt % len(specs)]
            prefs = [_pref("form_factor" if old in FORM_ATTRS else "attribute", old, "positive")]
            persona = self._persona(product_type, SCENARIO_KEY[product_type][0], prefs)
            events = _history_for_update(old, new, product_type)
            query = self._queries(product_type, SCENARIO_KEY[product_type][0], attempt)
            try:
                self.add_task(family="preference_update", family_id=f"eval100_update_{made + 1:02d}",
                              product_type=product_type, scenario=SCENARIO_KEY[product_type][0],
                              query=query, persona=persona, events=events, hard=[], intended={"recommend"},
                              intent_note=f"history retracts {old} and states {new}; old value must not remain active")
                made += 1
            except ValueError:
                pass
            attempt += 1
            if attempt > 60:
                raise RuntimeError(f"updates short: {made}/{count}")

    def add_withdrawals(self, count: int) -> None:
        specs = [("headphones", "noise_cancelling"), ("keyboard", "rgb"), ("mouse", "ergonomic"),
                 ("earbuds", "waterproof"), ("headphones", "noise_cancelling")]
        made = attempt = 0
        while made < count:
            product_type, name = specs[attempt % len(specs)]
            kind = "form_factor" if name in FORM_ATTRS else "attribute"
            prefs = [_pref(kind, name, "positive")]
            persona = self._persona(product_type, SCENARIO_KEY[product_type][0], prefs)
            events = _history_for_withdraw(name, product_type)
            # Use the same descriptor as the initial preference so the independent Memory
            # extractor recognizes this retraction for every value (e.g. "ergonomic shape",
            # "rgb lighting"). Evaluator candidates/status are unchanged.
            initial_text = next(event.raw_text for event in events if event.candidates)
            descriptor = re.sub(r"^(i\s+prefer|i\s+like)\s+", "", initial_text.strip(), flags=re.IGNORECASE)
            descriptor = re.sub(r"^(an?|the)\s+", "", descriptor, flags=re.IGNORECASE).rstrip(".").strip()
            withdrawal_text = f"Forget my earlier preference for {descriptor}."
            events = [
                EventSpec(candidates=event.candidates, raw_text=withdrawal_text)
                if any(candidate.get("status") == "withdraw" for candidate in event.candidates)
                else event
                for event in events
            ]
            hard = [_hard_constraint("wireless", f"hc_eval100_withdraw_{made}_wireless")]
            query = _query_with_constraints(self._queries(product_type, SCENARIO_KEY[product_type][0], attempt), hard)
            try:
                self.add_task(family="preference_withdrawal", family_id=f"eval100_withdraw_{made + 1:02d}",
                              product_type=product_type, scenario=SCENARIO_KEY[product_type][0],
                              query=query, persona=persona, events=events, hard=hard, intended={"recommend"},
                              intent_note=f"history withdraws {name}; remaining hard constraint must drive decision")
                made += 1
            except ValueError:
                pass
            attempt += 1
            if attempt > 80:
                raise RuntimeError(f"withdrawals short: {made}/{count}")

    def add_nonrecommend(self) -> None:
        # 8 clarification, 4 conflict, 4 no-verified, 4 unverifiable = 20.
        ask_specs = [
            ("headphones", "I need new headphones and have not decided what matters yet."),
            ("earbuds", "I want earbuds but am still open on the details."),
            ("keyboard", "I need a keyboard for my desk; any suggestions?"),
            ("mouse", "I'm thinking about a new mouse for work."),
            ("speaker", "I'd like a speaker for my home, not sure which kind."),
            ("headphones", "I need headphones for calls, no particular requirements."),
            ("keyboard", "I want a keyboard for gaming, still open on the details."),
            ("mouse", "I need a mouse I can take with me; I haven't decided more."),
        ]
        for i, (product_type, query) in enumerate(ask_specs):
            persona = self._persona(product_type, SCENARIO_KEY[product_type][0], [])
            events = [_event("ctx", 0, [], f"I'm only starting to look at {TYPE_WORD[product_type]}.")]
            self.add_task(family="insufficient_information", family_id=f"eval100_ask_{i + 1:02d}",
                          product_type=product_type, scenario=SCENARIO_KEY[product_type][0], query=query,
                          persona=persona, events=events, hard=[], intended={"clarification_required"},
                          intent_note="no preference and no explicit hard requirement; agent must ask")
        conflict_specs = [("headphones", "wireless", "wired"), ("keyboard", "wireless", "wired"),
                          ("mouse", "wireless", "wired"), ("speaker", "wireless", "wired")]
        for i, (product_type, left, right) in enumerate(conflict_specs):
            prefs = [_pref("attribute", left, "positive"), _pref("attribute", right, "positive")]
            persona = self._persona(product_type, SCENARIO_KEY[product_type][0], prefs)
            events = _history_for_conflict(left, right, product_type)
            self.add_task(family="preference_conflict", family_id=f"eval100_conflict_{i + 1:02d}",
                          product_type=product_type, scenario=SCENARIO_KEY[product_type][0],
                          query=self._queries(product_type, SCENARIO_KEY[product_type][0], i),
                          persona=persona, events=events, hard=[], intended={"clarification_required"},
                          intent_note="contradictory active preferences require a genuine user decision")
        made = attempt = 0
        while made < 4:
            product_type = "headphones"
            combo = NO_MATCH_COMBOS[attempt % len(NO_MATCH_COMBOS)]
            positives, _ = _scenario_prefs("headphones", "daily_commute")
            prefs = _prefs_from_lists(positives[:1], [])
            hard = [_hard_constraint(name, f"hc_eval100_ev_{made}_{attempt}_{name}") for name in combo]
            persona = self._persona(product_type, "daily_commute", prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(SEED + 500 + attempt))
            try:
                self.add_task(family="no_verified_match", family_id=f"eval100_evidence_{made + 1:02d}",
                              product_type=product_type, scenario="daily_commute",
                              query=_query_with_constraints(Q["headphones"][0], hard),
                              persona=persona, events=events, hard=hard, intended={"no_verified_match"},
                              intent_note="hard constraint combination has no verified candidate in sampled catalogue")
                made += 1
            except ValueError:
                pass
            attempt += 1
            if attempt > 20:
                raise RuntimeError("no-verified evidence short")
        made = 0
        while made < 4:
            spec = list(UNVERIFIABLE_HARD)[made % len(UNVERIFIABLE_HARD)]
            product_type = "headphones" if made % 2 == 0 else "mouse"
            positives, negatives = _scenario_prefs(product_type, "daily_commute" if product_type == "headphones" else "office")
            prefs = _prefs_from_lists(positives, negatives)
            hard = [_unverifiable_constraint(spec, f"hc_eval100_unver_{made}")]
            persona = self._persona(product_type, "daily_commute" if product_type == "headphones" else "office", prefs)
            events = _history_for_prefs(prefs, product_type=product_type, rng=random.Random(SEED + 700 + made))
            self.add_task(family="unverifiable_constraint", family_id=f"eval100_unver_{made + 1:02d}",
                          product_type=product_type,
                          scenario="daily_commute" if product_type == "headphones" else "office",
                          query=_query_with_constraints(Q[product_type][0], hard), persona=persona,
                          events=events, hard=hard, intended={"insufficient_evidence"},
                          intent_note="hard constraint has no verifiable catalogue field; UNKNOWN preserved")
            made += 1


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")
    return {"bytes": path.stat().st_size, "rows": len(rows)}


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    builder = Builder(catalogue)
    builder.add_no_history(20)
    builder.add_history(24)
    builder.add_pairs(10)
    builder.add_updates(8)
    builder.add_withdrawals(8)
    builder.add_nonrecommend()
    assert len(builder.tasks) == 100, len(builder.tasks)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "eval100.agent.jsonl": _write_jsonl(OUT / "eval100.agent.jsonl", builder.tasks),
        "eval100.evaluator.jsonl": _write_jsonl(OUT / "eval100.evaluator.jsonl", builder.evaluator),
        "personas.jsonl": _write_jsonl(OUT / "personas.jsonl", builder.personas),
        "history_events.jsonl": _write_jsonl(OUT / "history_events.jsonl", builder.history_events),
    }
    base = json.loads((REPO / "data/p14e/benchmark/schema_v0.1.json").read_text(encoding="utf-8"))
    base["$id"] = VERSION
    base["title"] = "P14-B3-Eval100 shopping preference benchmark schema v0.1"
    for section in ("agent_visible", "evaluator_only"):
        props = base[section]["properties"]
        props["schema"] = {"const": VERSION}
        props["split"] = {"enum": ["eval100"]}
        if section == "agent_visible":
            props["task_id"] = {"type": "string", "pattern": "^p14b3_eval100_[0-9]{4}$"}
            props["construction_metadata"]["properties"]["benchmark_split"] = {"const": "eval100"}
            props["construction_metadata"]["properties"]["review_status"] = {"enum": ["pending_human_review", "human_verified"]}
    (OUT / "schema_v0.1.json").write_text(json.dumps(base, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    files["schema_v0.1.json"] = {"bytes": (OUT / "schema_v0.1.json").stat().st_size}
    review = []
    for evaluator in builder.evaluator:
        review.append({"task_id": evaluator["task_id"], "task_type": evaluator["task_type"],
                       "expected_decision": evaluator["expected_decision"],
                       "effective_preferences": [p.get("value") for p in evaluator["effective_preferences"]],
                       "withdrawn_preferences": [p.get("value") for p in evaluator["withdrawn_preferences"]],
                       "hard_constraints": [c.get("value") for c in evaluator["hard_constraints"]],
                       "verified_candidate_count": evaluator["labels"]["verified_candidate_count"],
                       "review_status": "pending_human_review",
                       "reasons": evaluator["construction_metadata"]["review_reasons"]})
    (OUT / "review_checklist_initial.json").write_text(
        json.dumps({"schema": "agentrecx.p14b3eval100.review.v1", "review_status": "pending_human_review",
                    "human_verified": False, "cases": review}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = Counter(e["task_type"] for e in builder.evaluator)
    decisions = Counter(e["expected_decision"] for e in builder.evaluator)
    manifest = {
        "schema": "agentrecx.p14b3eval100.manifest.v1", "benchmark_version": VERSION,
        "generator_version": GENERATOR, "random_seed": SEED, "total_tasks": len(builder.tasks),
        "split": "eval100", "task_types": dict(sorted(counts.items())),
        "expected_decisions": dict(sorted(decisions.items())),
        "personas": len(builder.personas),
        "catalogue": {"namespace": ELECTRONICS_NAMESPACE, "stats": catalogue.stats},
        "files": files,
        "review_status": "pending_human_review", "human_verified": False,
        "notes": ["new personas/families/IDs; no reuse of P14-E/P14-E2 tasks",
                  "labeler is independent evidence-based and does not use BM25/V0/Qwen ranking",
                  "no model result was used to choose answers",
                  "reserved P14-E2 test untouched; these tasks are evaluation-only, not training data"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(builder.tasks), "types": dict(counts), "decisions": dict(decisions),
                      "files": files}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
