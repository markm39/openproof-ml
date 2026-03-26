"""Best-first tactic search in Python.

Mirrors the Rust implementation in openproof's search.rs.
Used during expert iteration and evaluation.
"""

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256

from .pantograph_client import PantographClient


@dataclass(order=True)
class SearchNode:
    """A node in the proof search tree."""

    priority: float
    state_id: int = field(compare=False)
    tactics: list[str] = field(compare=False)
    goals: list[str] = field(compare=False)


@dataclass
class SearchResult:
    """Result of a proof search."""

    solved: bool
    tactics: list[str]
    remaining_goals: int
    expansions: int
    elapsed: float


def hash_goals(goals: list[str]) -> str:
    """Hash a set of goals for deduplication."""
    return sha256("|||".join(sorted(goals)).encode()).hexdigest()


def best_first_search(
    pantograph: PantographClient,
    propose_fn: Callable[[str], list[str]],
    type_expr: str,
    beam_width: int = 8,
    max_expansions: int = 200,
    timeout: float = 120.0,
    max_depth: int = 20,
    length_penalty: float = 0.1,
) -> SearchResult:
    """Run best-first tactic search on a goal.

    Args:
        pantograph: Running Pantograph client.
        propose_fn: Function that takes a goal string and returns candidate tactics.
        type_expr: The theorem type expression to prove.
        beam_width: Number of tactics to request per goal state.
        max_expansions: Maximum total tactic applications.
        timeout: Wall-clock timeout in seconds.
        max_depth: Maximum tactic sequence depth.
        length_penalty: Penalty per tactic depth in priority.

    Returns:
        SearchResult with solved status and tactic sequence.
    """
    start_time = time.monotonic()
    expansions = 0
    seen_states: set[str] = set()
    allocated_states: list[int] = []

    # Start proof goal
    state_id = pantograph.start_goal(type_expr)
    if state_id is None:
        return SearchResult(False, [], 0, 0, 0.0)
    allocated_states.append(state_id)

    initial_hash = sha256(type_expr.encode()).hexdigest()
    seen_states.add(initial_hash)

    # Priority queue (min-heap)
    frontier: list[SearchNode] = []
    initial_priority = 1.0  # one goal to prove
    heapq.heappush(
        frontier,
        SearchNode(priority=initial_priority, state_id=state_id, tactics=[], goals=[type_expr]),
    )

    while frontier:
        if time.monotonic() - start_time > timeout:
            break
        if expansions >= max_expansions:
            break

        node = heapq.heappop(frontier)

        if len(node.tactics) >= max_depth:
            continue

        # Get first goal for tactic proposal
        goal_text = node.goals[0] if node.goals else ""
        if not goal_text:
            continue

        # Get candidate tactics
        candidates = propose_fn(goal_text)
        if not candidates:
            continue

        for tactic in candidates[:beam_width]:
            result = pantograph.try_tactic(node.state_id, 0, tactic)
            expansions += 1

            if not result.success or result.new_state_id is None:
                continue

            allocated_states.append(result.new_state_id)

            # Proof complete
            if not result.remaining_goals:
                elapsed = time.monotonic() - start_time
                _cleanup(pantograph, allocated_states)
                return SearchResult(
                    solved=True,
                    tactics=node.tactics + [tactic],
                    remaining_goals=0,
                    expansions=expansions,
                    elapsed=elapsed,
                )

            # Dedup by goal hash
            goals_hash = hash_goals(result.remaining_goals)
            if goals_hash in seen_states:
                continue
            seen_states.add(goals_hash)

            new_tactics = node.tactics + [tactic]
            priority = len(result.remaining_goals) + length_penalty * len(new_tactics)
            heapq.heappush(
                frontier,
                SearchNode(
                    priority=priority,
                    state_id=result.new_state_id,
                    tactics=new_tactics,
                    goals=result.remaining_goals,
                ),
            )

    elapsed = time.monotonic() - start_time
    _cleanup(pantograph, allocated_states)
    return SearchResult(
        solved=False,
        tactics=[],
        remaining_goals=len(frontier[0].goals) if frontier else 0,
        expansions=expansions,
        elapsed=elapsed,
    )


def _cleanup(pantograph: PantographClient, states: list[int]):
    """Delete all allocated goal states."""
    for sid in states:
        try:
            pantograph.delete_goal(sid)
        except Exception:
            pass
