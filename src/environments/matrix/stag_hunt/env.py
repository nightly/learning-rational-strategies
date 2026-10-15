from __future__ import annotations

import functools

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from .common import Actions
from .renderer import StagHuntRenderer


class StagHuntMatrix(ParallelEnv):
    """
    Repeated 2-player stag hunt with global binary observations.

    Observations expose the last joint action and whether the last stage was
    unmirrored:
      ch0 = player_0 chose Hare
      ch1 = player_0 chose Stag
      ch2 = player_1 chose Hare
      ch3 = player_1 chose Stag
      ch4 = last outcome was off-diagonal
    """

    metadata = {"name": "stag_hunt_matrix_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_moves: int = 20,
        stag_payoff: float = 4.0,
        hare_payoff: float = 3.0,
        stag_fail_payoff: float = 0.0,
        flatten_observations: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        if int(num_agents) != 2:
            raise AssertionError("StagHuntMatrix currently supports exactly 2 agents.")

        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]
        self.max_moves = int(max_moves)
        self.max_steps = self.max_moves

        self.stag_payoff = float(stag_payoff)
        self.hare_payoff = float(hare_payoff)
        self.stag_fail_payoff = float(stag_fail_payoff)

        self.n_obs_types = 2 * self.n_agents + 1
        self.flatten_observations = bool(flatten_observations)
        self.rng = np.random.RandomState(0 if seed is None else seed)

        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}
        self._last_unmirrored = False
        self._cum_rewards: dict[str, float] = {}

        self.render_mode = render_mode
        self._renderer: StagHuntRenderer | None = None
        if self.render_mode is not None:
            self._renderer = StagHuntRenderer(self.render_mode)

        self.observation_spaces = {agent_id: self.observation_space(agent_id) for agent_id in self.possible_agents}
        self.action_spaces = {agent_id: self.action_space(agent_id) for agent_id in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        _ = agent
        channels = self.n_obs_types
        if self.flatten_observations:
            return Box(low=0, high=1, shape=(channels,), dtype=np.uint8)
        return Box(low=0, high=1, shape=(1, 1, channels), dtype=np.uint8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        _ = agent
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        _ = options
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {agent_id: None for agent_id in self.agents}
        self._last_unmirrored = False
        self._cum_rewards = {agent_id: 0.0 for agent_id in self.agents}

        obs = {agent_id: self._obs() for agent_id in self.agents}
        infos = {
            agent_id: {
                "round": 0,
                "last_actions": None,
                "outcome": None,
                "both_hare": False,
                "both_stag": False,
                "unmirrored": False,
            }
            for agent_id in self.agents
        }
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        player_0, player_1 = self.possible_agents
        act0 = int(actions.get(player_0, Actions.Hare))
        act1 = int(actions.get(player_1, Actions.Hare))

        both_hare = act0 == Actions.Hare and act1 == Actions.Hare
        both_stag = act0 == Actions.Stag and act1 == Actions.Stag
        unmirrored = act0 != act1

        if both_hare:
            reward_0, reward_1 = self.hare_payoff, self.hare_payoff
            outcome = "Hare-Hare"
        elif both_stag:
            reward_0, reward_1 = self.stag_payoff, self.stag_payoff
            outcome = "Stag-Stag"
        elif act0 == Actions.Stag:
            reward_0, reward_1 = self.stag_fail_payoff, self.hare_payoff
            outcome = "Stag-Hare"
        else:
            reward_0, reward_1 = self.hare_payoff, self.stag_fail_payoff
            outcome = "Hare-Stag"

        self._last_actions[player_0] = act0
        self._last_actions[player_1] = act1
        self._last_unmirrored = unmirrored
        self._cum_rewards[player_0] += reward_0
        self._cum_rewards[player_1] += reward_1
        self._round += 1

        rewards = {
            player_0: float(reward_0),
            player_1: float(reward_1),
        }
        terminations = {agent_id: False for agent_id in self.agents}
        env_truncated = self._round >= self.max_moves
        truncations = {agent_id: bool(env_truncated) for agent_id in self.agents}
        infos = {
            player_0: {
                "round": self._round,
                "last_actions": (act0, act1),
                "outcome": outcome,
                "both_hare": both_hare,
                "both_stag": both_stag,
                "unmirrored": unmirrored,
            },
            player_1: {
                "round": self._round,
                "last_actions": (act1, act0),
                "outcome": outcome,
                "both_hare": both_hare,
                "both_stag": both_stag,
                "unmirrored": unmirrored,
            },
        }
        obs = {agent_id: self._obs() for agent_id in self.agents}

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        if env_truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            raise ValueError("Renderer missing; create env with render_mode='human' or 'rgb_array'.")

        last = (
            self._last_actions.get("player_0"),
            self._last_actions.get("player_1"),
        )
        return self._renderer.render(
            round_no=self._round,
            last_actions=last,
            last_unmirrored=self._last_unmirrored,
            cum_rewards=self._cum_rewards,
            payoffs={
                "hare": self.hare_payoff,
                "stag": self.stag_payoff,
                "fail": self.stag_fail_payoff,
            },
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    def _obs(self):
        channels = np.zeros((self.n_obs_types,), dtype=np.uint8)

        def set_bits(agent_id: str, base_index: int) -> None:
            action = self._last_actions.get(agent_id)
            if action is None:
                return
            if action == Actions.Hare:
                channels[base_index] = 1
            elif action == Actions.Stag:
                channels[base_index + 1] = 1

        set_bits("player_0", 0)
        set_bits("player_1", 2)
        channels[2 * self.n_agents] = 1 if self._last_unmirrored else 0

        if self.flatten_observations:
            return channels
        return channels.reshape(1, 1, -1)

    @property
    def state_space(self):
        channels = self.n_obs_types
        if self.flatten_observations:
            return Box(low=0, high=1, shape=(channels,), dtype=np.uint8)
        return Box(low=0, high=1, shape=(1, 1, channels), dtype=np.uint8)

    def state(self):
        return self._obs()

    def num_cells(self) -> int:
        return 1

    def channel_names(self) -> list[str]:
        return [
            "player_0_hare",
            "player_0_stag",
            "player_1_hare",
            "player_1_stag",
            "unmirrored",
        ]

    def action_names(self, action: int) -> str:
        try:
            return Actions(int(action)).name.lower()
        except ValueError:
            return f"action_{action}"

    def get_state(self):
        player_0, player_1 = self.possible_agents
        return (
            tuple(self.agents),
            int(self._round),
            (
                self._last_actions.get(player_0),
                self._last_actions.get(player_1),
            ),
            bool(self._last_unmirrored),
            (
                float(self._cum_rewards.get(player_0, 0.0)),
                float(self._cum_rewards.get(player_1, 0.0)),
            ),
            self.rng.get_state(),
        )

    def set_state(self, state):
        if len(state) != 6:
            raise ValueError(f"StagHuntMatrix.set_state: expected 6 fields, got {len(state)}")

        agents, round_no, last_actions, last_unmirrored, cum_rewards, rng_state = state
        if len(last_actions) != 2:
            raise ValueError(f"StagHuntMatrix.set_state: last_actions must be len 2, got {len(last_actions)}")
        if len(cum_rewards) != 2:
            raise ValueError(f"StagHuntMatrix.set_state: cum_rewards must be len 2, got {len(cum_rewards)}")

        self.agents = list(agents)
        self._round = int(round_no)
        self._last_actions = {
            "player_0": None if last_actions[0] is None else int(last_actions[0]),
            "player_1": None if last_actions[1] is None else int(last_actions[1]),
        }
        self._last_unmirrored = bool(last_unmirrored)
        self._cum_rewards = {
            "player_0": float(cum_rewards[0]),
            "player_1": float(cum_rewards[1]),
        }
        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))
