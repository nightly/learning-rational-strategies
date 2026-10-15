from src.monitor.reward_wrapper import BoolRewardWrapper
from src.monitor.spec_pair import SpecPair

from .mix_in import PELabelMixin


class _PEClassicHandcraftedReward:
    def __call__(self, _wrapper, *, labels, done, current_agents, **_kwargs):
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        goal_reached = bool(labels.get("goal_reached", False))
        collision = bool(labels.get("collision", False))

        if "player_0" in rewards:
            if goal_reached:
                rewards["player_0"] -= 1.0
            elif done:
                rewards["player_0"] += 1.0

        for agent_id in rewards:
            if agent_id == "player_0":
                continue
            if goal_reached:
                rewards[agent_id] += 1.0
            if collision:
                rewards[agent_id] -= 1.0
        return rewards


class _PECaptureVsStealthHandcraftedReward:
    def __call__(self, _wrapper, *, labels, done, current_agents, **_kwargs):
        rewards = {agent_id: 0.0 for agent_id in current_agents}
        captured = bool(labels.get("guard_intruder_collision", False))
        collision_ever = bool(labels.get("collision_ever", False))

        if "player_0" in rewards:
            if captured:
                rewards["player_0"] += 1.0
            elif done:
                rewards["player_0"] -= 0.5

        for agent_id in rewards:
            if agent_id == "player_0":
                continue
            if captured or collision_ever:
                rewards[agent_id] -= 1.0
            elif done:
                rewards[agent_id] += 1.0
        return rewards


class PEClassicConstraint(PELabelMixin, BoolRewardWrapper):
    """Classic guard-vs-intruder pursuit objective wrapper."""

    def __init__(
        self,
        env,
        max_steps_per_episode=100,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        specs = [
            # Guard: prevent goal forever
            SpecPair(agent_id="player_0", formula_str="G !goal_reached", reward_scalar=1.0),

            # Intruders: eventually reach goal without ever colliding
            SpecPair(agent_id="player_1", formula_str="F goal_reached & G !collision_ever", reward_scalar=1.0),
        ]

        # add player_2 if present
        if "player_2" in getattr(env, "possible_agents", []):
            specs.append(
                SpecPair(
                    agent_id="player_2",
                    formula_str="F goal_reached & G !collision_ever",
                    reward_scalar=1.0,
                )
            )

        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=max_steps_per_episode,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _PEClassicHandcraftedReward(),
        )


class PECaptureVsStealth(PELabelMixin, BoolRewardWrapper):
    """Objective wrapper for guard capture versus intruder stealth."""

    def __init__(
        self,
        env,
        max_steps_per_episode=100,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        specs = [
            # Guard: eventually collide with an intruder (contact/capture)
            SpecPair(agent_id="player_0", formula_str="F guard_intruder_collision", reward_scalar=1.0),

            # Intruder: avoid guard contact and every other collision for the trace.
            SpecPair(
                agent_id="player_1",
                formula_str="G !guard_intruder_collision & G !collision_ever",
                reward_scalar=1.0,
            ),
        ]
        if "player_2" in getattr(env, "possible_agents", []):
            specs.append(
                SpecPair(
                    agent_id="player_2",
                    formula_str="G !guard_intruder_collision & G !collision_ever",
                    reward_scalar=1.0,
                )
            )
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=max_steps_per_episode,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn or _PECaptureVsStealthHandcraftedReward(),
        )


class PEIntruderCoordination(PELabelMixin, BoolRewardWrapper):
    """Objective wrapper for coordinated intruder goal reaching."""

    def __init__(
        self,
        env,
        max_steps_per_episode=100,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        specs = [
            SpecPair(agent_id="player_0", formula_str="G !goal_reached", reward_scalar=1.0),

            # Intruders: reach goal and avoid intruder-intruder collisions.
            SpecPair(
                agent_id="player_1",
                formula_str="F goal_reached & G !intruder_intruder_collision",
                reward_scalar=1.0,
            ),
        ]
        if "player_2" in getattr(env, "possible_agents", []):
            specs.append(
                SpecPair(
                    agent_id="player_2",
                    formula_str="F goal_reached & G !intruder_intruder_collision",
                    reward_scalar=1.0,
                )
            )
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=max_steps_per_episode,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn,
        )


class PEGapStrategy(PELabelMixin, BoolRewardWrapper):
    """Objective wrapper for reaching the gridworld gap without collisions."""

    def __init__(
        self,
        env,
        max_steps_per_episode=100,
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn=None,
    ):
        specs = [
            SpecPair(agent_id="player_0", formula_str="G !goal_reached", reward_scalar=1.0),
            SpecPair(agent_id="player_1", formula_str="F intruder_in_gap & G !collision_ever", reward_scalar=1.0),
        ]
        if "player_2" in getattr(env, "possible_agents", []):
            specs.append(
                SpecPair(
                    agent_id="player_2",
                    formula_str="F intruder_in_gap & G !collision_ever",
                    reward_scalar=1.0,
                )
            )
        super().__init__(
            env,
            spec_rewards=specs,
            max_steps_per_episode=max_steps_per_episode,
            reward_mode=reward_mode,
            handcrafted_reward_fn=handcrafted_reward_fn,
        )
