from src.monitor.reward_wrapper import BoolRewardWrapper
from src.monitor.spec_pair import SpecPair

from .mix_in import HarvestLabelMixin


class _SafeHarvestHandcraftedReward:
    def __init__(self, agent_ids) -> None:
        self.agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
        self.reset()

    def reset(self) -> None:
        self.apple_paid = {agent_id: False for agent_id in self.agent_ids}
        self.zap_seen = {agent_id: False for agent_id in self.agent_ids}

    def __call__(self, _wrapper, *, labels, current_agents, **_kwargs):
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        for agent_id in current_agents:
            zapped_now = bool(labels.get(f"{agent_id}_zapped", False))
            apple_now = bool(labels.get(f"{agent_id}_apple_collected", False))

            if zapped_now and not self.zap_seen.get(agent_id, False):
                self.zap_seen[agent_id] = True
                rewards[agent_id] -= 1.0
                if self.apple_paid.get(agent_id, False):
                    self.apple_paid[agent_id] = False

            if apple_now and not self.zap_seen.get(agent_id, False) and not self.apple_paid.get(agent_id, False):
                rewards[agent_id] += 1.0
                self.apple_paid[agent_id] = True
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


class CHHarvestConstraint(HarvestLabelMixin, BoolRewardWrapper):
    """Objective wrapper for eventually collecting an apple per agent."""

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
            spec_rewards=_agent_specs(env, "F {agent_id}_apple_collected"),
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn,
        )


class CHSafeHarvestConstraint(HarvestLabelMixin, BoolRewardWrapper):
    """Objective wrapper for collecting an apple without being zapped per agent."""

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
                "(G !{agent_id}_zapped) & (F {agent_id}_apple_collected)",
            ),
            max_steps_per_episode=int(max_steps_per_episode),
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn
            or _SafeHarvestHandcraftedReward(getattr(env, "possible_agents", [])),
        )


class CHClassicConstraint(CHSafeHarvestConstraint):
    pass


__all__ = [
    "CHClassicConstraint",
    "CHHarvestConstraint",
    "CHSafeHarvestConstraint",
]
