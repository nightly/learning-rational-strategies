"""Per-agent probability thresholds of the curated benchmark objectives.

Agent ``i``'s objective is ``P_{>= q_i}[phi_i]``. Thresholds are part of each
task's specification and are fixed per agent, never per algorithm. They are
keyed by the task's export path relative to ``exports/`` (the suite directory
used by ``scripts/certification_summary.py``).

The values model what each role can plausibly demand, so some agents ask for
little and others over-reach, and the Pursuit tasks lean in different
directions: intruder-favoured, balanced and guard-favoured (see
``docs/AI/verification/objective-thresholds.md``).
Pursuit's ``player_0`` is the guard and the other players are intruders; guard
and intruder objectives exclude each other, so a profile can only be jointly
certified when ``q_guard + q_intruder`` leaves room for both sampling margins.
"""

from __future__ import annotations

from collections.abc import Mapping

CURATED_PROBABILITY_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    # The guard settles for occasionally denying the goal; the intruder expects
    # to reach it in most episodes.
    "gridworlds/pursuit/classic_2": {"player_0": 0.1, "player_1": 0.75},
    # Balanced: the guard wants captures in almost a third of episodes, the
    # intruder to evade in half.
    "gridworlds/pursuit/capture_2": {"player_0": 0.3, "player_1": 0.5},
    # The guard expects to hold the goal; the intruders settle for occasional
    # breaches. Trained intruders usually win, so the guard over-reaches.
    "gridworlds/pursuit/classic_3": {"player_0": 0.5, "player_1": 0.1, "player_2": 0.1},
    "gridworlds/pursuit/capture_3": {
        "player_0": 0.05,
        "player_1": 0.8,
        "player_2": 0.8,
    },
    # Symmetric coin race: each player asks for slightly less than an even share.
    "gridworlds/coins/own_coin_first/2": {"player_0": 0.4, "player_1": 0.4},
    # Both players insist on sustained mutual stag play.
    "matrix/stag_hunt/sustained_stag/2": {"player_0": 0.9, "player_1": 0.9},
    # Ambitions differ by player; players 2 and 3 rarely harvest safely in any
    # trained profile, so even their modest targets over-reach.
    "gridworlds/harvest/safe_harvest/4": {
        "player_0": 0.8,
        "player_1": 0.7,
        "player_2": 0.3,
        "player_3": 0.2,
    },
    # Winner-take-all race: each player wants to win a share of the races.
    "gridworlds/territory/role_claim/2": {"player_0": 0.25, "player_1": 0.2},
}


def curated_probability_thresholds(task: str) -> dict[str, tuple[str, float]]:
    """Return ``{agent: (">=", q_i)}`` for a curated task's export path."""
    try:
        thresholds = CURATED_PROBABILITY_THRESHOLDS[task]
    except KeyError as error:
        raise KeyError(
            f"No curated probability thresholds for task {task!r}."
        ) from error
    return {agent_id: (">=", float(value)) for agent_id, value in thresholds.items()}


__all__ = ["CURATED_PROBABILITY_THRESHOLDS", "curated_probability_thresholds"]
