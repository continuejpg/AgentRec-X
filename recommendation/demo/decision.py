"""Deterministic, offline route chooser for the Milestone 11 demo.

The accepted Milestone 7B trust boundary routes ``k`` through exactly one place: the
:class:`~recommendation.agent.decision.AgentDecision` produced by the injected decision
model.  There is no other channel, by design.  The demo therefore needs a real,
production-quality decision model -- not a test double -- and this is it.

What it is
----------
A small, explicit, **keyword rule** that chooses between the two accepted routes,
:attr:`AgentAction.RECOMMEND` and :attr:`AgentAction.DIRECT_RESPONSE`.  It reads only the
user message it is handed, and it passes ``k`` through so a chat request can choose how
many candidates to ask the Tool for.

What it is not
--------------
* **not** an LLM and **not** a provider client: no SDK is imported, no API key is read,
  no network call is made, and the demo is fully reproducible offline;
* **not** a trained intent classifier and **not** a planner -- it has two outcomes, both
  of which the accepted graph already supported, and no planning loop;
* **not** able to see trusted history, candidate ids, model scores, catalogue metadata
  or preference memory: the graph hands it ``build_decision_messages(user_message)``
  and nothing else, so there is no argument through which any of those could arrive.

The rule is intentionally legible: a message asks for recommendations when it contains
an explicit recommendation trigger, and is treated as a direct turn otherwise (a
greeting, a question about the demo, or a preference statement, which the memory nodes
handle independently of the route).
"""

from __future__ import annotations

import re
from typing import Sequence

from recommendation.agent import AgentDecision, DecisionMessage

__all__ = ["DEFAULT_DEMO_K", "RECOMMENDATION_TRIGGERS", "DemoDecisionModel", "looks_like_recommendation"]

#: Candidate count used when a caller does not choose one.
DEFAULT_DEMO_K = 5

#: Explicit recommendation triggers.  Deliberately lexical and narrow: the demo route
#: never guesses intent from sentiment, history or model output.
RECOMMENDATION_TRIGGERS: tuple[str, ...] = (
    r"\brecommend",
    r"\bsuggest",
    r"\bshow\s+me\b",
    r"\bgive\s+me\b",
    r"\bfind\s+(?:me|some|a|an|the)\b",
    r"\blooking\s+for\b",
    r"\bshopping\s+for\b",
    r"\bsearch(?:ing)?\s+for\b",
    r"\bwhat\s+(?:do\s+you\s+)?(?:have|recommend)",
    r"\bmore\s+(?:options|like|ideas)\b",
    r"\boptions\b",
    r"\bideas\b",
    r"\balternatives\b",
    r"\bbrowse\b",
    r"\brecommendation",
    r"\bproducts?\b",
    r"\brecommend\s+again\b",
    r"\bagain\b",
    r"\banything\s+else\b",
    r"\bwhat\s+else\b",
)

_TRIGGER = re.compile("|".join(RECOMMENDATION_TRIGGERS), re.IGNORECASE)

#: Honest direct-route text.  It states what the demo can do and makes no product,
#: quality or relevance claim.  Any preference actually written by this turn is reported
#: separately, from the accepted Milestone 9 write result, by the response serializer --
#: this text never claims a memory change it cannot see.
DIRECT_RESPONSE_TEXT = (
    "I'm a local AgentRec-X demo. I can recommend products from this demo profile's "
    "trusted history, and I can remember explicit preferences you state in your own "
    "words. Ask me to recommend something -- for example, \"Recommend some hiking "
    "gear.\" -- and any preference you state takes effect from the following turn."
)


def looks_like_recommendation(user_message: str) -> bool:
    """True when the message explicitly asks for product recommendations."""
    return bool(_TRIGGER.search(user_message))


class DemoDecisionModel:
    """Deterministic two-route decision model for the local demo.

    ``k`` is fixed per instance, which is why the demo runtime caches one compiled
    agent graph per requested ``k`` instead of mutating a shared decision model.
    """

    def __init__(self, k: int = DEFAULT_DEMO_K) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > 100:
            raise ValueError("k must be an integer in 1..100")
        self._k = k
        self.calls: list[tuple[DecisionMessage, ...]] = []

    @property
    def k(self) -> int:
        """The candidate count this model requests."""
        return self._k

    @property
    def call_count(self) -> int:
        """How many decisions were produced (diagnostics only)."""
        return len(self.calls)

    def decide(self, messages: Sequence[DecisionMessage]) -> AgentDecision:
        """Choose a route from the user message alone.

        Raises
        ------
        ValueError
            No user message is present.  The agent graph validates the produced
            decision, so a malformed prompt is loud rather than silently routed.
        """
        self.calls.append(tuple(messages))
        user_text = ""
        for message in messages:
            if getattr(message, "role", None) == "user":
                user_text = str(getattr(message, "content", ""))
        if not user_text.strip():
            raise ValueError("the decision prompt carries no user message")

        if looks_like_recommendation(user_text):
            return AgentDecision(action="recommend", k=self._k)
        return AgentDecision(action="direct_response", direct_response=DIRECT_RESPONSE_TEXT)
