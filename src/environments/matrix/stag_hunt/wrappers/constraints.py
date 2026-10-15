from __future__ import annotations

from src.monitor.reward_wrapper import BoolRewardWrapper
from src.monitor.spec_pair import SpecPair

from .mix_in import SHLabelMixin


class _HandshakeHandcraftedReward:
    def __init__(self) -> None:
        self.seen_hare = False
        self.paid_handshake = False

    def reset(self) -> None:
        self.seen_hare = False
        self.paid_handshake = False

    def __call__(self, _wrapper, *, native_rewards, labels, current_agents, **_kwargs):
        del native_rewards
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        if labels.get("unmirrored", False):
            return {agent_id: -1.0 for agent_id in current_agents}
        if labels.get("both_hare", False):
            self.seen_hare = True
            for agent_id in current_agents:
                rewards[agent_id] += 0.5
        if self.seen_hare and labels.get("both_stag", False) and not self.paid_handshake:
            self.paid_handshake = True
            for agent_id in current_agents:
                rewards[agent_id] += 1.0
        return rewards


class _BothModesHandcraftedReward:
    def __init__(self) -> None:
        self.seen_hare = False
        self.seen_stag = False

    def reset(self) -> None:
        self.seen_hare = False
        self.seen_stag = False

    def __call__(self, _wrapper, *, native_rewards, labels, current_agents, **_kwargs):
        del native_rewards
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        if labels.get("unmirrored", False):
            return {agent_id: -1.0 for agent_id in current_agents}
        if labels.get("both_hare", False) and not self.seen_hare:
            self.seen_hare = True
            for agent_id in current_agents:
                rewards[agent_id] += 1.0
        if labels.get("both_stag", False) and not self.seen_stag:
            self.seen_stag = True
            for agent_id in current_agents:
                rewards[agent_id] += 1.0
        return rewards


class SHHandshakeConstraint(SHLabelMixin, BoolRewardWrapper):
    """Objective wrapper for a hare-then-stag handshake without mismatches."""

    def __init__(
        self,
        env,
        max_steps_per_episode: int | None = None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        horizon = int(max_steps_per_episode or getattr(env, "max_steps", 20))
        specs = [
            SpecPair(agent_id=agent_id, formula_str="F (both_hare & X both_stag) & G !unmirrored")
            for agent_id in getattr(env, "possible_agents", [])
        ]
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=horizon,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _HandshakeHandcraftedReward(),
        )


class SHSustainedStagConstraint(SHLabelMixin, BoolRewardWrapper):
    """Objective wrapper for reaching sustained stag play after coordination."""

    def __init__(
        self,
        env,
        max_steps_per_episode: int | None = None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        horizon = int(max_steps_per_episode or getattr(env, "max_steps", 20))
        specs = [
            SpecPair(agent_id=agent_id, formula_str="F (both_stag & X (both_stag & G !unmirrored))")
            for agent_id in getattr(env, "possible_agents", [])
        ]
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=horizon,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn,
        )


class SHBothModesConstraint(SHLabelMixin, BoolRewardWrapper):
    """Objective wrapper requiring both hare and stag modes without mismatches."""

    def __init__(
        self,
        env,
        max_steps_per_episode: int | None = None,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        horizon = int(max_steps_per_episode or getattr(env, "max_steps", 20))
        specs = [
            SpecPair(agent_id=agent_id, formula_str="F both_hare & F both_stag & G !unmirrored")
            for agent_id in getattr(env, "possible_agents", [])
        ]
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=horizon,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _BothModesHandcraftedReward(),
        )
