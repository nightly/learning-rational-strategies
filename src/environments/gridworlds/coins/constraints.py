from src.monitor.reward_wrapper import BoolRewardWrapper
from src.monitor.spec_pair import SpecPair

from .mix_in import CoinsLabelMixin


class _OwnCoinPreferenceHandcraftedReward:
    """Dense own-colour preference shaping without partner-harm credit."""

    def __call__(self, _wrapper, *, labels, current_agents, **_kwargs):
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        for agent_id in current_agents:
            if labels.get(f"{agent_id}_matched_coin_collected", False):
                rewards[agent_id] = 1.0
            elif labels.get(f"{agent_id}_mismatched_coin_collected", False):
                rewards[agent_id] = -1.0
        return rewards


def _agent_specs(env) -> list[SpecPair]:
    return [
        SpecPair(
            agent_id=agent_id,
            formula_str=(f"(!{agent_id}_mismatched_coin_collected) U {agent_id}_matched_coin_collected"),
            reward_scalar=1.0,
        )
        for agent_id in getattr(env, "possible_agents", [])
    ]


class CoinsOwnCoinFirstConstraint(CoinsLabelMixin, BoolRewardWrapper):
    """Collect one's matching coin before ever taking the partner's coin."""

    def __init__(
        self,
        env,
        max_steps_per_episode: int = 256,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        possible_agents = tuple(str(agent_id) for agent_id in getattr(env, "possible_agents", []))
        if possible_agents != ("player_0", "player_1"):
            raise ValueError("CoinsOwnCoinFirstConstraint requires exactly player_0 and player_1.")
        super().__init__(
            env,
            spec_rewards=_agent_specs(env),
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _OwnCoinPreferenceHandcraftedReward(),
        )


class CoinsProsocialConstraint(CoinsOwnCoinFirstConstraint):
    """Compatibility name for the selected own-coin-first objective."""


class CoinsClassicConstraint(CoinsOwnCoinFirstConstraint):
    pass


__all__ = [
    "CoinsClassicConstraint",
    "CoinsOwnCoinFirstConstraint",
    "CoinsProsocialConstraint",
]
