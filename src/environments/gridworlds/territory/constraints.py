from src.monitor.reward_wrapper import BoolRewardWrapper
from src.monitor.spec_pair import SpecPair

from .mix_in import TerritoryLabelMixin


class _PeacefulClaimHandcraftedReward:
    def __call__(self, _wrapper, *, native_rewards, labels, current_agents, **_kwargs):
        penalty = -1.0 if not bool(labels.get("all_zappers_ready", True)) else 0.0
        return {agent_id: float(native_rewards.get(agent_id, 0.0)) + penalty for agent_id in current_agents}


class _RoleClaimHandcraftedReward:
    def __init__(self) -> None:
        self.resolved = False

    def reset(self) -> None:
        self.resolved = False

    def __call__(self, _wrapper, *, labels, current_agents, **_kwargs):
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        if self.resolved:
            return rewards

        player_0_claimed = bool(labels.get("player_0_resource_claimed", False))
        player_1_claimed = bool(labels.get("player_1_resource_claimed", False))
        if not player_0_claimed and not player_1_claimed:
            return rewards

        self.resolved = True
        if player_0_claimed and not player_1_claimed:
            if "player_0" in rewards:
                rewards["player_0"] = 1.0
            if "player_1" in rewards:
                rewards["player_1"] = -1.0
        elif player_1_claimed and not player_0_claimed:
            if "player_0" in rewards:
                rewards["player_0"] = -1.0
            if "player_1" in rewards:
                rewards["player_1"] = 1.0

        return rewards


def _agent_specs(env, formula_template: str) -> list[SpecPair]:
    return [
        SpecPair(
            agent_id=agent_id,
            formula_str=formula_template.format(agent_id=agent_id),
            reward_scalar=1.0,
        )
        for agent_id in getattr(env, "possible_agents", [])
    ]


class TerritoryClaimConstraint(TerritoryLabelMixin, BoolRewardWrapper):
    """Objective wrapper for eventually claiming a resource per agent."""

    def __init__(
        self,
        env,
        max_steps_per_episode=None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        if max_steps_per_episode is None:
            max_steps_per_episode = int(getattr(env, "max_cycles", 1000))
        super().__init__(
            env,
            spec_rewards=_agent_specs(env, "F {agent_id}_resource_claimed"),
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn,
        )


class TerritoryPeacefulClaimConstraint(TerritoryLabelMixin, BoolRewardWrapper):
    """Objective wrapper for resource claims while all zappers stay ready."""

    def __init__(
        self,
        env,
        max_steps_per_episode=None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        if max_steps_per_episode is None:
            max_steps_per_episode = int(getattr(env, "max_cycles", 1000))
        super().__init__(
            env,
            spec_rewards=_agent_specs(
                env,
                "(G all_zappers_ready) & (F {agent_id}_resource_claimed)",
            ),
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _PeacefulClaimHandcraftedReward(),
        )


class TerritoryRoleClaimConstraint(TerritoryLabelMixin, BoolRewardWrapper):
    """Two-agent race in which each player tries to make the first sole claim."""

    def __init__(
        self,
        env,
        max_steps_per_episode=None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        possible_agents = tuple(str(agent_id) for agent_id in getattr(env, "possible_agents", []))
        expected_agents = ("player_0", "player_1")
        if possible_agents != expected_agents:
            raise ValueError(
                "TerritoryRoleClaimConstraint expects a 2-agent Territory env with "
                f"agents {expected_agents!r}; got {possible_agents!r}."
            )
        if max_steps_per_episode is None:
            max_steps_per_episode = int(getattr(env, "max_cycles", 1000))
        super().__init__(
            env,
            spec_rewards=[
                SpecPair(
                    agent_id="player_0",
                    formula_str=(
                        "(!player_1_resource_claimed) U (player_0_resource_claimed & !player_1_resource_claimed)"
                    ),
                    reward_scalar=1.0,
                ),
                SpecPair(
                    agent_id="player_1",
                    formula_str=(
                        "(!player_0_resource_claimed) U (player_1_resource_claimed & !player_0_resource_claimed)"
                    ),
                    reward_scalar=1.0,
                ),
            ],
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _RoleClaimHandcraftedReward(),
        )


__all__ = [
    "TerritoryClaimConstraint",
    "TerritoryPeacefulClaimConstraint",
    "TerritoryRoleClaimConstraint",
]
