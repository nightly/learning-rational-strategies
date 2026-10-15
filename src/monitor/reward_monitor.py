import os
from collections import deque
from contextlib import contextmanager
from typing import Any, Mapping

import graphviz
from flloat.parser.ltlf import LTLfParser

# flloat assigns automaton state integers in an order that depends on Python's
# per-process hash seed, so ids sorted by those integers ("flloat") differ
# between processes for many formulas. The ids feed observations and tabular
# keys, so a checkpoint must be evaluated under the numbering it was trained
# with. "canonical" numbers states breadth-first from the initial state,
# visiting successors by their printed guard, which is hash-independent.
MONITOR_STATE_NUMBERINGS = ("canonical", "flloat")
_state_numbering = os.environ.get("PLTLF_MONITOR_STATE_NUMBERING", "canonical")


def _check_state_numbering(mode: Any) -> Any:
    # A mapping {formula: [id of each canonical state]} reproduces one process's
    # flloat numbering in any process.
    if isinstance(mode, Mapping):
        return {
            str(formula): [int(state_id) for state_id in ids]
            for formula, ids in mode.items()
        }
    mode = str(mode)
    if mode not in MONITOR_STATE_NUMBERINGS:
        raise ValueError(
            f"Unknown monitor state numbering {mode!r}; expected one of {MONITOR_STATE_NUMBERINGS}."
        )
    return mode


def monitor_state_numbering() -> Any:
    """Numbering applied to monitors constructed from now on."""
    return _check_state_numbering(_state_numbering)


@contextmanager
def use_monitor_state_numbering(mode: Any):
    """Construct monitors inside the block with ``mode`` numbering (a name or a permutation mapping)."""
    global _state_numbering
    previous = _state_numbering
    _state_numbering = _check_state_numbering(mode)
    try:
        yield
    finally:
        _state_numbering = previous


def _canonical_state_order(automaton) -> list:
    order = [automaton.initial_state]
    seen = {automaton.initial_state}
    queue = deque(order)
    while queue:
        state = queue.popleft()
        transitions = sorted(
            automaton.get_transitions_from(state), key=lambda item: str(item[1])
        )
        for _, _, destination in transitions:
            if destination not in seen:
                seen.add(destination)
                order.append(destination)
                queue.append(destination)
    # Unreachable states (none for flloat's minimal DFAs) keep a stable tail.
    order.extend(
        sorted((state for state in automaton.states if state not in seen), key=str)
    )
    return order


class BoolMonitor:
    """Finite-trace LTLf monitor backed by a deterministic automaton."""

    def __init__(self, formula_str: str, reward_value: float):
        self.reward_value = reward_value
        self.cache = {}

        self.formula_str = str(formula_str)
        self.formula = LTLfParser()(formula_str)
        self.automaton = self.formula.to_automaton()
        self.current_state = self.automaton.initial_state

        # Id mapping for the observation feature and tabular keys (see above).
        numbering = monitor_state_numbering()
        if isinstance(numbering, Mapping):
            if self.formula_str not in numbering:
                raise KeyError(
                    f"No state permutation given for formula {self.formula_str!r}."
                )
            canonical = _canonical_state_order(self.automaton)
            ids = numbering[self.formula_str]
            if sorted(ids) != list(range(len(canonical))):
                raise ValueError(
                    f"Invalid state permutation {ids} for formula {self.formula_str!r}."
                )
            self._states = [None] * len(canonical)
            for state, state_id in zip(canonical, ids):
                self._states[state_id] = state
            self.state_numbering = "permutation"
        elif numbering == "canonical":
            self._states = _canonical_state_order(self.automaton)
            self.state_numbering = "canonical"
        else:
            self._states = sorted(list(self.automaton.states), key=lambda s: str(s))
            self.state_numbering = "flloat"
        self._state_to_id = {s: i for i, s in enumerate(self._states)}
        self._accepting_state_ids = frozenset(
            self._state_to_id[state] for state in self.automaton.accepting_states
        )

    def _coerce_fluents(self, fluents):
        if isinstance(fluents, dict):
            return {str(key): bool(value) for key, value in fluents.items()}

        active = {str(item) for item in fluents}
        return {key: True for key in active}

    def step(self, fluents):
        key = (self.current_state, frozenset(k for k, v in fluents.items() if bool(v)))
        if key in self.cache:
            self.current_state = self.cache[key]
        else:
            self.current_state = self.automaton.get_successor(
                self.current_state, fluents
            )
            self.cache[key] = self.current_state

    def reset(self):
        self.cache = {}
        self.current_state = self.automaton.initial_state

    def reward(self):
        reward = 0
        if self.current_state in self.automaton.accepting_states:
            reward = self.reward_value
        else:
            reward = 0
        return reward

    # ---- helpers -----

    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def state_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_states))

    def state_id(self) -> int:
        return int(self._state_to_id[self.current_state])

    def initial_state_id(self) -> int:
        return int(self._state_to_id[self.automaton.initial_state])

    def is_accepting_state_id(self, state_id: int) -> bool:
        return int(state_id) in self._accepting_state_ids

    def numbering_fingerprint(self) -> dict:
        """JSON-safe signature of the id assignment: accepting ids and guarded transitions."""
        return {
            "numbering": self.state_numbering,
            "initial_id": self.initial_state_id(),
            "accepting_ids": sorted(self._accepting_state_ids),
            "transitions": [
                sorted(
                    [str(guard), int(self._state_to_id[destination])]
                    for _, guard, destination in self.automaton.get_transitions_from(
                        state
                    )
                )
                for state in self._states
            ],
        }

    def successor_state_id(self, state_id: int, fluents) -> int:
        current_state = self._states[int(state_id)]
        successor = self.automaton.get_successor(
            current_state,
            self._coerce_fluents(fluents),
        )
        return int(self._state_to_id[successor])

    # ---- export & visualisation ----
    def export(
        self, to_file=True, dir="exports", filename="bool_rm.dot"
    ) -> graphviz.Digraph:
        graph = self.automaton.to_graphviz()
        if to_file:
            graph.render(f"{dir}/{filename}")
        return graph

    def print_monitor(self, logger_obj=None):
        output = logger_obj.info if logger_obj else print
        output(f"Boolean monitor for formula: {self.formula}")
        output(f"Automaton's set of states: {self.automaton.states}")
        output(
            f"Automaton's set of accepting states states: {self.automaton.accepting_states}"
        )
        output(f"Automaton's transitions: {self.automaton.get_transitions()}")

    def __str__(self):
        from io import StringIO

        buffer = StringIO()
        self.print_monitor(
            logger_obj=type("Logger", (), {"info": lambda x: buffer.write(x + "\n")})
        )
        return buffer.getvalue().strip()
