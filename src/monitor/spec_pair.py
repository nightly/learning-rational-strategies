from dataclasses import dataclass


@dataclass(frozen=True)
class SpecPair:
    """Reward-monitor specification for one agent.

    If `agent_id` is None, the spec is treated as shared and applied to all
    agents in a parallel PettingZoo environment.
    """

    formula_str: str
    """Finite-trace LTL formula parsed by `flloat`."""

    reward_scalar: float = 1.0
    """Reward scalar retained for monitor construction; objective rewards are terminal 0/1 bits."""

    agent_id: str | None = None
    """Agent id this formula belongs to."""
