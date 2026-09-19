"""The bounded loop as a LangGraph state graph with a real back-edge (2.0-alpha).

Topology::

    initialize -> check_limits -> policy -> validate_action -> dispatch
                       ^                                         |-- execute -> verify
                       |                                         |                |
                       |                                         |             observe
                       |                                         |                |
                       |                                         |          update_state
                       |                                         |                |
                       |                             (back-edge) |                |
                       +-----------------------------------------+----------------+
                       |
                       `-- complete -> finalize / refuse / abort -> END

The back-edge ``update_state -> check_limits`` (and ``refuse -> check_limits``) is the
structural change this milestone is about: after a non-terminal observation, control
genuinely returns toward the policy - through the budget check, so the controller's step
budget is re-tested on **every** iteration - and the policy sees the updated context.

Why the nodes return ``Command`` instead of using conditional edges
------------------------------------------------------------------
LangGraph resolves a node's outgoing conditional branches from the snapshot taken *before*
that node's body runs.  A verdict produced inside the node (a refused completion, or a
proposal the validator rejects) is therefore invisible to a routing function for that same
superstep, and the already-queued branch would still execute - which is how this loop first
deadlocked and then spun.  ``Command(goto=..., update=...)`` commits the decision and the
routing together, so a verdict and its edge can never disagree.  That is the mechanism this
topology uses everywhere, and it is why the routing logic lives inside the phases rather
than in separate functions.

Why the graph carries an engine object rather than the run's data
-----------------------------------------------------------------
The loop's data (trusted history, candidate artifacts) is trusted state.  Putting it in
LangGraph channels would create a channel a node could return into, which is exactly the
class of accident the accepted DAG avoids by never giving the decision node a
history-returning channel.  Instead one LangGraph channel holds the controller's
:class:`~recommendation.control.loop._LoopEngine`, and the nodes call its methods.  The
graph is therefore pure **control topology**: it decides *which phase runs next*, and it has
no channel in which a candidate, a score, a user key or a history entry could travel.

Termination is owned by the controller: ``check_limits`` runs before the policy on every
iteration and the engine sets a terminal status in exactly one place.  LangGraph's own
``recursion_limit`` is derived from the same budgets as a second, framework-level backstop
(see :func:`~recommendation.control.loop.recursion_limit`).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .loop import _LoopEngine

__all__ = [
    "LOOP_NODE_NAMES",
    "LOOP_PHASE_ORDER",
    "NODE_ABORT",
    "NODE_CHECK_LIMITS",
    "NODE_COMPLETE",
    "NODE_DISPATCH",
    "NODE_EXECUTE",
    "NODE_FINALIZE",
    "NODE_INITIALIZE",
    "NODE_OBSERVE",
    "NODE_POLICY",
    "NODE_REFUSE",
    "NODE_UPDATE_STATE",
    "NODE_VALIDATE_ACTION",
    "NODE_VERIFY",
    "build_loop_graph",
    "topology_mermaid",
]

#: The loop's phase names.  Exposed so tests and the smoke experiment can assert the
#: structure - including the back-edge - rather than trusting a picture in a docstring.
NODE_INITIALIZE = "initialize"
NODE_CHECK_LIMITS = "check_limits"
NODE_POLICY = "policy"
NODE_VALIDATE_ACTION = "validate_action"
NODE_DISPATCH = "dispatch"
NODE_EXECUTE = "execute"
NODE_VERIFY = "verify"
NODE_OBSERVE = "observe"
NODE_UPDATE_STATE = "update_state"
NODE_COMPLETE = "complete"
NODE_FINALIZE = "finalize"
NODE_REFUSE = "refuse"
NODE_ABORT = "abort"

#: Every declared node, sorted, for structural assertions.
LOOP_NODE_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            NODE_ABORT,
            NODE_CHECK_LIMITS,
            NODE_COMPLETE,
            NODE_DISPATCH,
            NODE_EXECUTE,
            NODE_FINALIZE,
            NODE_INITIALIZE,
            NODE_OBSERVE,
            NODE_POLICY,
            NODE_REFUSE,
            NODE_UPDATE_STATE,
            NODE_VALIDATE_ACTION,
            NODE_VERIFY,
        )
    )
)

#: The phase sequence of one loop iteration, in order.  ``check_limits`` heads it because
#: the back-edge returns there rather than straight to the policy, so the step budget is
#: re-tested before every decision.
LOOP_PHASE_ORDER: tuple[str, ...] = (
    NODE_INITIALIZE,
    NODE_CHECK_LIMITS,
    NODE_POLICY,
    NODE_VALIDATE_ACTION,
    NODE_DISPATCH,
    NODE_EXECUTE,
    NODE_VERIFY,
    NODE_OBSERVE,
    NODE_UPDATE_STATE,
)


class LoopGraphState(TypedDict, total=False):
    """The graph's only channels: the controller's engine, plus a visit counter.

    Deliberately minimal.  No trusted history, no candidates, no scores, no user key -
    there is no channel for them to travel in.
    """

    engine: _LoopEngine
    step: int


def _advance(state: LoopGraphState) -> dict[str, int]:
    """Return the visit-counter update for one phase."""
    return {"step": state.get("step", 0) + 1}


# --------------------------------------------------------------------------- #
# Nodes.  Each node is one phase, and each one commits its own routing.
# --------------------------------------------------------------------------- #


def _initialize(state: LoopGraphState) -> Command:
    """Read the turn's preference snapshot once, before any decision."""
    state["engine"].initialize()
    return Command(update=_advance(state), goto=NODE_CHECK_LIMITS)


def _check_limits(state: LoopGraphState) -> Command:
    """Enforce the step budget before the policy is consulted.

    This is the deterministic termination boundary.  Reaching it on the back-edge is what
    makes the budget apply to *every* iteration rather than only the first.
    """
    engine = state["engine"]
    engine.check_limits()
    if engine.control.is_terminal:
        return Command(update=_advance(state), goto=END)
    return Command(update=_advance(state), goto=NODE_POLICY)


def _policy(state: LoopGraphState) -> Command:
    """Ask the injected policy for exactly one action proposal.

    The phase always proceeds to validation: producing a proposal and judging it are
    different phases, and the validator - not the policy - owns the verdict.  If the policy
    failed to propose, the engine has already recorded a terminal state and the validator
    will report that instead of guessing.
    """
    state["engine"].choose()
    return Command(update=_advance(state), goto=NODE_VALIDATE_ACTION)


def _validate_action(state: LoopGraphState) -> Command:
    """Validate the proposal and stamp controller-owned execution metadata."""
    engine = state["engine"]
    engine.validate_action()
    if engine.control.is_terminal or not engine.has_current_action():
        # A refused proposal (or a missing one) is a protocol violation rather than a
        # recoverable condition; the engine has already recorded the reason.  No branch is
        # entered, because both of them would act on an action that does not exist.
        return Command(update=_advance(state), goto=END)
    return Command(update=_advance(state), goto=NODE_DISPATCH)


def _dispatch(state: LoopGraphState) -> Command:
    """Route the validated action to execution or to the completion branch.

    The routing is committed with the phase, so a FINISH proposal cannot leak into the
    execution branch and a recommendation action cannot leak into completion.
    """
    engine = state["engine"]
    if engine.dispatch_target() == "complete":
        return Command(update=_advance(state), goto=NODE_COMPLETE)
    return Command(update=_advance(state), goto=NODE_EXECUTE)


def _execute(state: LoopGraphState) -> Command:
    """Enforce the tool-call budget, then execute the capability."""
    engine = state["engine"]
    if not engine.check_tool_budget().continue_loop:
        # The budget stopped the step before anything ran; the engine recorded the
        # termination, and the phases after execution have nothing to verify.
        return Command(update=_advance(state), goto=END)
    engine.execute()
    return Command(update=_advance(state), goto=NODE_VERIFY)


def _verify(state: LoopGraphState) -> Command:
    """Verify the raw domain result against the run's trusted inputs."""
    state["engine"].verify()
    return Command(update=_advance(state), goto=NODE_OBSERVE)


def _observe(state: LoopGraphState) -> Command:
    """Adapt the verified result into the minimised, policy-visible observation."""
    state["engine"].observe()
    return Command(update=_advance(state), goto=NODE_UPDATE_STATE)


def _update_state(state: LoopGraphState) -> Command:
    """Adopt the verified candidate set and record the step.

    **The back-edge.**  From here control returns to ``check_limits`` and then to the
    policy, which sees the updated context: the new observation, a grounded candidate set,
    and a reduced budget.
    """
    state["engine"].update_state()
    return Command(update=_advance(state), goto=NODE_CHECK_LIMITS)


def _complete(state: LoopGraphState) -> Command:
    """Run the completion guard and commit the verdict's branch."""
    engine = state["engine"]
    engine.run_completion_guard()
    target = engine.completion_target()
    if target == "finalize":
        return Command(update=_advance(state), goto=NODE_FINALIZE)
    if target == "refuse":
        return Command(update=_advance(state), goto=NODE_REFUSE)
    return Command(update=_advance(state), goto=NODE_ABORT)


def _finalize(state: LoopGraphState) -> Command:
    """Render the accepted response, commit memory, and stop the run."""
    state["engine"].finalize()
    return Command(update=_advance(state), goto=END)


def _refuse(state: LoopGraphState) -> Command:
    """Record a retryable refusal and return control toward the policy."""
    state["engine"].refuse_completion()
    return Command(update=_advance(state), goto=NODE_CHECK_LIMITS)


def _abort(state: LoopGraphState) -> Command:
    """Record a refusal no further step can fix and stop the run."""
    state["engine"].abort_completion()
    return Command(update=_advance(state), goto=END)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_loop_graph() -> Any:
    """Compile the bounded loop topology.

    The caller supplies the engine at invocation time::

        graph.invoke({"engine": engine, "step": 0},
                     config={"recursion_limit": recursion_limit(limits)})

    so one compiled topology serves many runs.  Compiling is pure Python over the same
    process-scoped collaborators, exactly as the accepted DAG's graph cache works.
    """
    builder: StateGraph = StateGraph(LoopGraphState)

    # Every node declares its own successor through ``Command``, so each phase chooses the
    # next phase in exactly one place and no separate routing function can disagree with
    # the verdict that phase produced.
    for name, node in (
        (NODE_INITIALIZE, _initialize),
        (NODE_CHECK_LIMITS, _check_limits),
        (NODE_POLICY, _policy),
        (NODE_VALIDATE_ACTION, _validate_action),
        (NODE_DISPATCH, _dispatch),
        (NODE_EXECUTE, _execute),
        (NODE_VERIFY, _verify),
        (NODE_OBSERVE, _observe),
        (NODE_UPDATE_STATE, _update_state),
        (NODE_COMPLETE, _complete),
        (NODE_FINALIZE, _finalize),
        (NODE_REFUSE, _refuse),
        (NODE_ABORT, _abort),
    ):
        builder.add_node(name, node)

    builder.add_edge(START, NODE_INITIALIZE)
    return builder.compile()


#: The cycle's edges, declared explicitly.
#:
#: LangGraph cannot render edges that a ``Command`` chooses - they are not part of the
#: static graph - so the loop's cycle is declared here as data.  The structural tests assert
#: against this *and* against the live ``Command`` targets, so the declaration cannot drift
#: from the implementation.
DECLARED_CYCLE_EDGES: tuple[tuple[str, str], ...] = (
    (NODE_INITIALIZE, NODE_CHECK_LIMITS),
    (NODE_CHECK_LIMITS, NODE_POLICY),
    (NODE_POLICY, NODE_VALIDATE_ACTION),
    (NODE_VALIDATE_ACTION, NODE_DISPATCH),
    # The branch after dispatch.
    (NODE_DISPATCH, NODE_EXECUTE),
    (NODE_EXECUTE, NODE_VERIFY),
    (NODE_VERIFY, NODE_OBSERVE),
    (NODE_OBSERVE, NODE_UPDATE_STATE),
    # **The back-edge**: the observation is applied, then the budget is re-tested and
    # control returns to the policy.
    (NODE_UPDATE_STATE, NODE_CHECK_LIMITS),
    # The completion branch.
    (NODE_DISPATCH, NODE_COMPLETE),
    (NODE_COMPLETE, NODE_FINALIZE),
    (NODE_COMPLETE, NODE_REFUSE),
    (NODE_REFUSE, NODE_CHECK_LIMITS),
)


def topology_mermaid() -> str:
    """Return the loop topology as Mermaid text, including the declared cycle.

    LangGraph's renderer shows only the static edges (the entry point here) because every
    phase chooses its successor through ``Command``.  The declared cycle is appended, so
    the printed topology is the real one rather than a graph of disconnected nodes.
    """
    static = build_loop_graph().get_graph().draw_mermaid()
    lines = ["", "%% --- declared control cycle (Command-chosen edges) ---"]
    for source, target in DECLARED_CYCLE_EDGES:
        lines.append(f"\t{source} --> {target};")
    return "\n".join([static, *lines])
