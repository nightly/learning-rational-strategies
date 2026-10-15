"""Native PettingZoo implementation of Melting Pot's Coins substrate."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from src.environments.substrates._structured_observations import (
    object_observation,
    object_observation_ids,
    object_observation_spaces,
)
from src.environments.substrates._views import agent_window

Agent = str
Position = tuple[int, int]
ObservationMode = Literal["local", "global", "focal"]
RenderMode = Literal["human", "rgb_array"] | None

AGENTS: tuple[Agent, Agent] = ("player_0", "player_1")
ORIENTATIONS: tuple[str, ...] = ("N", "E", "S", "W")
ABS_DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RELATIVE_MOVE_TO_OFFSET: dict[int, int] = {
    1: 0,  # Forward.
    2: 1,  # Step right.
    3: 2,  # Backward.
    4: 3,  # Step left.
}
ACTION_TO_COMPONENTS: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 0),
    2: (3, 0),
    3: (4, 0),
    4: (2, 0),
    5: (0, -1),
    6: (0, 1),
}

COIN_COLORS: dict[str, tuple[int, int, int]] = {
    "coin_red": (238, 102, 119),
    "coin_blue": (68, 119, 170),
    "coin_yellow": (204, 187, 68),
    "coin_green": (34, 136, 51),
    "coin_purple": (170, 51, 119),
}
COIN_TYPES: tuple[str, ...] = tuple(COIN_COLORS)
PLAYER_COLORS = COIN_COLORS

FLOOR_COLOR = np.array((34, 45, 40), dtype=np.uint8)
WALL_COLOR = np.array((105, 108, 112), dtype=np.uint8)
WALL_LIGHT = np.array((145, 148, 152), dtype=np.uint8)
OUT_OF_BOUNDS_COLOR = np.array((0, 0, 0), dtype=np.uint8)
OUT_OF_VIEW_COLOR = np.array((80, 80, 80), dtype=np.uint8)
AVATAR_TRIM = np.array((245, 245, 235), dtype=np.uint8)


@dataclass(frozen=True)
class CoinsConfig:
    """Configuration for the native Coins substrate."""

    min_width: int = 10
    max_width: int = 15
    min_height: int = 10
    max_height: int = 15
    regrow_rate: float = 0.0005
    reward_self_for_match: float = 1.0
    reward_self_for_mismatch: float = 1.0
    reward_other_for_match: float = 0.0
    reward_other_for_mismatch: float = -2.0
    max_episode_steps: int = 5000
    stochastic_min_steps: int = 300
    stochastic_interval: int = 100
    stochastic_termination_probability: float = 0.05
    sprite_size: int = 8
    observation_mode: ObservationMode = "local"

    @property
    def world_rows(self) -> int:
        return self.max_height + 2

    @property
    def world_cols(self) -> int:
        return self.max_width + 2

    @property
    def world_rgb_shape(self) -> tuple[int, int, int]:
        return (
            self.world_rows * self.sprite_size,
            self.world_cols * self.sprite_size,
            3,
        )

    @property
    def local_rgb_shape(self) -> tuple[int, int, int]:
        rows, cols, _, _ = agent_window(self, 11, 11, 5, 9)
        return (rows * self.sprite_size, cols * self.sprite_size, 3)

    def validate(self) -> None:
        if self.min_width > self.max_width:
            raise ValueError("min_width must be <= max_width")
        if self.min_height > self.max_height:
            raise ValueError("min_height must be <= max_height")
        if not 0.0 <= self.regrow_rate <= 1.0:
            raise ValueError("regrow_rate must be in [0, 1]")
        if not 0.0 <= self.stochastic_termination_probability <= 1.0:
            raise ValueError("stochastic_termination_probability must be in [0, 1]")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.stochastic_interval <= 0:
            raise ValueError("stochastic_interval must be positive")
        if self.observation_mode not in {"local", "global", "focal"}:
            raise ValueError("observation_mode must be 'local', 'global' or 'focal'")


@dataclass
class Coin:
    position: Position
    state: str | None = None


@dataclass
class Avatar:
    agent: Agent
    index: int
    coin_type: str
    position: Position
    orientation: int
    reward: float = 0.0
    partner_mismatch: float = 0.0


class CoinsParallelEnv(ParallelEnv[Agent, dict[str, np.ndarray], int]):
    """PettingZoo ParallelEnv for the Coins substrate."""

    metadata = {
        "name": "coins_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    possible_agents = list(AGENTS)

    def __init__(self, config: CoinsConfig | None = None, render_mode: RenderMode = None):
        self.config = config or CoinsConfig()
        self.config.validate()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode

        self.agents: list[Agent] = []
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._rng = random.Random()
        self._ascii_map = ""
        self._grid: list[str] = []
        self._line_lengths: list[int] = []
        self._walls: set[Position] = set()
        self._spawn_points: list[Position] = []
        self._coins: dict[Position, Coin] = {}
        self._avatars: dict[Agent, Avatar] = {}
        self._coin_types: tuple[str, str] = ("coin_red", "coin_blue")
        self._step_count = 0
        self._last_events: list[dict[str, Any]] = []

        self._pygame = None
        self._screen = None
        self._clock = None

    @cached_property
    def _action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(7)

    @cached_property
    def _observation_space(self) -> gym.spaces.Dict:
        if self.render_mode is None:
            return gym.spaces.Dict(object_observation_spaces(self))
        rgb_shape = (
            self.config.local_rgb_shape if self.config.observation_mode != "global" else self.config.world_rgb_shape
        )
        return gym.spaces.Dict(
            {
                "RGB": gym.spaces.Box(0, 255, shape=rgb_shape, dtype=np.uint8),
                "MISMATCHED_COIN_COLLECTED_BY_PARTNER": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(),
                    dtype=np.float64,
                ),
            }
        )

    @property
    def object_observation_ids(self):
        return object_observation_ids(self)

    def observation_space(self, agent: Agent) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Agent) -> gym.spaces.Space:
        return self.action_spaces[agent]

    @property
    def ascii_map(self) -> str:
        """The upstream-shaped ASCII map generated for the current episode."""
        return self._ascii_map

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[Agent, dict[str, np.ndarray]], dict[Agent, dict[str, Any]]]:
        del options
        if seed is not None:
            self._rng.seed(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._last_events = []
        self._ascii_map = generate_ascii_map(
            self.config.min_width,
            self.config.max_width,
            self.config.min_height,
            self.config.max_height,
            self._rng,
        )
        self._load_ascii_map(self._ascii_map)
        self._coin_types = tuple(self._rng.sample(COIN_TYPES, k=2))  # type: ignore[assignment]

        spawn_points = list(self._spawn_points)
        self._rng.shuffle(spawn_points)
        if len(spawn_points) < len(self.possible_agents):
            raise RuntimeError("Coins requires at least two spawn points")

        self._avatars = {}
        for index, agent in enumerate(self.possible_agents):
            self._avatars[agent] = Avatar(
                agent=agent,
                index=index,
                coin_type=self._coin_types[index],
                position=spawn_points[index],
                orientation=self._rng.randrange(len(ORIENTATIONS)),
            )

        observations = self._observations(self.agents)
        infos = self._infos(self.agents)
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(
        self, actions: dict[Agent, int]
    ) -> tuple[
        dict[Agent, dict[str, np.ndarray]],
        dict[Agent, float],
        dict[Agent, bool],
        dict[Agent, bool],
        dict[Agent, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        live_agents = list(self.agents)
        self._last_events = []
        for avatar in self._avatars.values():
            avatar.reward = 0.0
            avatar.partner_mismatch = 0.0

        decoded = {agent: ACTION_TO_COMPONENTS.get(int(actions.get(agent, 0)), (0, 0)) for agent in live_agents}
        for agent in live_agents:
            _, turn = decoded[agent]
            if turn:
                self._avatars[agent].orientation = (self._avatars[agent].orientation + turn) % len(ORIENTATIONS)

        self._move_avatars(live_agents, decoded)
        for agent in live_agents:
            self._collect_coin(self._avatars[agent])
        self._regrow_coins()

        self._step_count += 1
        terminated_by_stochastic = self._should_stochastically_terminate()
        truncated_by_limit = self._step_count >= self.config.max_episode_steps and not terminated_by_stochastic

        observations = self._observations(live_agents)
        rewards = {agent: float(self._avatars[agent].reward) for agent in live_agents}
        terminations = {agent: terminated_by_stochastic for agent in live_agents}
        truncations = {agent: truncated_by_limit for agent in live_agents}
        infos = self._infos(live_agents)

        if terminated_by_stochastic or truncated_by_limit:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self._render_world()

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        frame = self.state()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode != "human":
            return None

        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.init()
            pygame.display.set_caption("meltingpot-pz: coins")
            height, width = frame.shape[:2]
            self._screen = pygame.display.set_mode((width, height))
            self._clock = pygame.time.Clock()

        assert self._screen is not None
        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(8)
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None

    def _load_ascii_map(self, ascii_map: str) -> None:
        raw_lines = ascii_map.split("\n")
        width = self.config.world_cols
        height = self.config.world_rows
        self._line_lengths = [min(len(line), width) for line in raw_lines[:height]]
        while len(self._line_lengths) < height:
            self._line_lengths.append(0)

        lines = [line[:width].ljust(width) for line in raw_lines[:height]]
        while len(lines) < height:
            lines.append(" " * width)
        self._grid = lines
        self._walls = set()
        self._spawn_points = []
        self._coins = {}
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if x >= self._line_lengths[y]:
                    continue
                position = (x, y)
                if char == "W":
                    self._walls.add(position)
                elif char == "_":
                    self._spawn_points.append(position)
                elif char == "C":
                    self._coins[position] = Coin(position=position, state=None)

    def _move_avatars(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int]],
    ) -> None:
        current_positions = {avatar.position: agent for agent, avatar in self._avatars.items()}
        desired: dict[Agent, Position] = {}
        for agent in live_agents:
            move, _ = decoded_actions[agent]
            avatar = self._avatars[agent]
            if move == 0:
                desired[agent] = avatar.position
                continue
            offset = RELATIVE_MOVE_TO_OFFSET[move]
            direction = (avatar.orientation + offset) % len(ORIENTATIONS)
            dx, dy = ABS_DELTAS[direction]
            target = (avatar.position[0] + dx, avatar.position[1] + dy)
            desired[agent] = avatar.position if self._is_blocked(target) else target

        target_counts = Counter(desired.values())
        for agent in live_agents:
            avatar = self._avatars[agent]
            target = desired[agent]
            if target == avatar.position:
                continue
            occupied_by = current_positions.get(target)
            conflict = target_counts[target] > 1 or occupied_by not in (None, agent)
            if not conflict:
                avatar.position = target

    def _collect_coin(self, avatar: Avatar) -> None:
        coin = self._coins.get(avatar.position)
        if coin is None or coin.state is None:
            return

        partner_agent = AGENTS[1 - avatar.index]
        matched = coin.state == avatar.coin_type
        if matched:
            avatar.reward += self.config.reward_self_for_match
            self._avatars[partner_agent].reward += self.config.reward_other_for_match
        else:
            avatar.reward += self.config.reward_self_for_mismatch
            self._avatars[partner_agent].reward += self.config.reward_other_for_mismatch
            self._avatars[partner_agent].partner_mismatch = 1.0

        self._last_events.append(
            {
                "event": "coin_consumed",
                "player_index": avatar.index + 1,
                "player_coin_type": avatar.coin_type,
                "coin_type": coin.state,
                "matched": matched,
            }
        )
        coin.state = None

    def _regrow_coins(self) -> None:
        for coin in self._coins.values():
            if coin.state is None and self._rng.random() < self.config.regrow_rate:
                coin.state = self._rng.choice(self._coin_types)

    def _should_stochastically_terminate(self) -> bool:
        if self._step_count < self.config.stochastic_min_steps:
            return False
        if self._step_count % self.config.stochastic_interval != 0:
            return False
        return self._rng.random() < self.config.stochastic_termination_probability

    def _is_blocked(self, position: Position) -> bool:
        x, y = position
        if y < 0 or y >= len(self._grid) or x < 0 or x >= self.config.world_cols:
            return True
        if x >= self._line_lengths[y]:
            return True
        return position in self._walls

    def _observations(self, agents: list[Agent]) -> dict[Agent, dict[str, np.ndarray]]:
        if self.render_mode is None:
            return {agent: object_observation(self, agent) for agent in agents}
        return {agent: self._observation(agent) for agent in agents}

    def _observation(self, agent: Agent) -> dict[str, np.ndarray]:
        rgb = self._render_agent(agent) if self.config.observation_mode != "global" else self._render_world()
        return {
            "RGB": rgb,
            "MISMATCHED_COIN_COLLECTED_BY_PARTNER": np.asarray(self._avatars[agent].partner_mismatch, dtype=np.float64),
        }

    def _infos(self, agents: list[Agent]) -> dict[Agent, dict[str, Any]]:
        infos = {}
        for agent in agents:
            avatar = self._avatars[agent]
            infos[agent] = {
                "meltingpot_player_index": avatar.index + 1,
                "coin_type": avatar.coin_type,
                "position": avatar.position,
                "orientation": ORIENTATIONS[avatar.orientation],
                "events": list(self._last_events),
            }
        return infos

    def _render_world(self) -> np.ndarray:
        if not self._grid:
            return np.zeros(self.config.world_rgb_shape, dtype=np.uint8)
        frame = np.zeros(self.config.world_rgb_shape, dtype=np.uint8)
        avatar_by_position = {avatar.position: avatar for avatar in self._avatars.values()}
        for y in range(self.config.world_rows):
            for x in range(self.config.world_cols):
                tile = self._render_tile((x, y), avatar_by_position)
                self._paint_tile(frame, x, y, tile)
        return frame

    def _render_agent(self, agent: Agent) -> np.ndarray:
        frame = np.zeros(self.config.local_rgb_shape, dtype=np.uint8)
        if not self._grid:
            return frame
        avatar = self._avatars[agent]
        avatar_by_position = {candidate.position: candidate for candidate in self._avatars.values()}
        rows, cols, left, forward = agent_window(self.config, 11, 11, 5, 9)
        for row in range(rows):
            for col in range(cols):
                rel_right = col - left
                rel_forward = forward - row
                world_pos = _relative_to_world(
                    avatar.position,
                    avatar.orientation,
                    rel_right,
                    rel_forward,
                )
                tile = self._render_tile(world_pos, avatar_by_position)
                self._paint_tile(frame, col, row, tile)
        return frame

    def _render_tile(self, position: Position, avatar_by_position: dict[Position, Avatar]) -> np.ndarray:
        size = self.config.sprite_size
        x, y = position
        if y < 0 or y >= len(self._grid) or x < 0 or x >= self.config.world_cols:
            return _solid_tile(OUT_OF_BOUNDS_COLOR, size)
        if x >= self._line_lengths[y]:
            return _solid_tile(OUT_OF_VIEW_COLOR, size)

        tile = _solid_tile(FLOOR_COLOR, size)
        if position in self._walls:
            tile = _wall_tile(size)

        coin = self._coins.get(position)
        if coin is not None and coin.state is not None:
            tile = _coin_tile(COIN_COLORS[coin.state], size)

        avatar = avatar_by_position.get(position)
        if avatar is not None:
            tile = _avatar_tile(PLAYER_COLORS[avatar.coin_type], avatar.orientation, size)
        return tile

    def _paint_tile(self, frame: np.ndarray, tile_x: int, tile_y: int, tile: np.ndarray) -> None:
        size = self.config.sprite_size
        y0 = tile_y * size
        x0 = tile_x * size
        frame[y0 : y0 + size, x0 : x0 + size] = tile


def generate_ascii_map(
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    rng: random.Random | None = None,
) -> str:
    """Procedurally generate the upstream-shaped Coins ASCII map."""
    if min_width > max_width:
        raise ValueError("min_width must be <= max_width")
    if min_height > max_height:
        raise ValueError("min_height must be <= max_height")
    rng = rng or random
    width = rng.randint(min_width, max_width)
    height = rng.randint(min_height, max_height)

    ascii_map = ["W"] * (width + 2) + [" "] * (max_width - width)
    for row in range(height):
        ascii_map += ["\nW"] + ["C"] * width + ["W"]
        if row == 1:
            ascii_map[-3] = "_"
        elif row == height - 2:
            ascii_map[-width] = "_"
        ascii_map += [" "] * (max_width - width)
    ascii_map += ["\n"] + ["W"] * (width + 2) + [" "] * (max_width - width)
    for _ in range(max_height - height):
        ascii_map += ["\n"] + [" "] * max_width
    return "".join(ascii_map)


def parallel_env(
    render_mode: RenderMode = None,
    config: CoinsConfig | None = None,
    **config_overrides: Any,
) -> CoinsParallelEnv:
    """Create the native Coins ParallelEnv."""
    if config is None:
        config = CoinsConfig(**config_overrides)
    elif config_overrides:
        config = replace(config, **config_overrides)
    return CoinsParallelEnv(config=config, render_mode=render_mode)


def env(render_mode: RenderMode = None, **kwargs: Any):
    """Create the native Coins AEC env."""
    return parallel_to_aec(parallel_env(render_mode=render_mode, **kwargs))


def _relative_to_world(
    origin: Position,
    orientation: int,
    rel_right: int,
    rel_forward: int,
) -> Position:
    if orientation == 0:
        dx, dy = rel_right, -rel_forward
    elif orientation == 1:
        dx, dy = rel_forward, rel_right
    elif orientation == 2:
        dx, dy = -rel_right, rel_forward
    else:
        dx, dy = -rel_forward, -rel_right
    return origin[0] + dx, origin[1] + dy


def _solid_tile(color: np.ndarray, size: int) -> np.ndarray:
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:] = color
    return tile


def _wall_tile(size: int) -> np.ndarray:
    tile = _solid_tile(WALL_COLOR, size)
    tile[0, :] = WALL_LIGHT
    tile[:, 0] = WALL_LIGHT
    return tile


def _coin_tile(color: tuple[int, int, int], size: int) -> np.ndarray:
    tile = _solid_tile(FLOOR_COLOR, size)
    rgb = np.asarray(color, dtype=np.uint8)
    center = (size - 1) / 2
    radius = max(size / 2 - 1, 1)
    for y in range(size):
        for x in range(size):
            if (x - center) ** 2 + (y - center) ** 2 <= radius**2:
                tile[y, x] = rgb
    tile[2 : size - 2, 2 : size - 2] = np.maximum(tile[2 : size - 2, 2 : size - 2], rgb)
    return tile


def _avatar_tile(
    color: tuple[int, int, int],
    orientation: int,
    size: int,
) -> np.ndarray:
    tile = _solid_tile(np.asarray(color, dtype=np.uint8), size)
    tile[0, :] = AVATAR_TRIM
    tile[-1, :] = AVATAR_TRIM
    tile[:, 0] = AVATAR_TRIM
    tile[:, -1] = AVATAR_TRIM
    mid = size // 2
    if orientation == 0:
        tile[1:4, mid] = AVATAR_TRIM
    elif orientation == 1:
        tile[mid, size - 4 : size - 1] = AVATAR_TRIM
    elif orientation == 2:
        tile[size - 4 : size - 1, mid] = AVATAR_TRIM
    else:
        tile[mid, 1:4] = AVATAR_TRIM
    return tile


__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "Avatar",
    "Coin",
    "COIN_TYPES",
    "CoinsConfig",
    "CoinsParallelEnv",
    "env",
    "generate_ascii_map",
    "parallel_env",
]
