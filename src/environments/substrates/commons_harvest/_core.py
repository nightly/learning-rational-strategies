"""Shared native implementation for Melting Pot Commons Harvest substrates."""

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

NUM_PLAYERS = 7
AGENTS: tuple[Agent, ...] = tuple(f"player_{index}" for index in range(NUM_PLAYERS))
ORIENTATIONS: tuple[str, ...] = ("N", "E", "S", "W")
ABS_DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RELATIVE_MOVE_TO_OFFSET: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}
ACTION_TO_COMPONENTS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (3, 0, 0),
    3: (4, 0, 0),
    4: (2, 0, 0),
    5: (0, -1, 0),
    6: (0, 1, 0),
    7: (0, 0, 1),
}

FLOOR_COLOR = np.array((212, 198, 175), dtype=np.uint8)
GRASS_COLOR = np.array((104, 154, 75), dtype=np.uint8)
DESSICATED_COLOR = np.array((172, 154, 128), dtype=np.uint8)
WALL_COLOR = np.array((92, 94, 98), dtype=np.uint8)
WALL_LIGHT = np.array((145, 148, 152), dtype=np.uint8)
APPLE_COLOR = np.array((214, 88, 88), dtype=np.uint8)
APPLE_DARK = np.array((155, 55, 55), dtype=np.uint8)
ZAP_COLOR = np.array((252, 252, 106), dtype=np.uint8)
OUT_OF_BOUNDS_COLOR = np.array((0, 0, 0), dtype=np.uint8)
AVATAR_TRIM = np.array((245, 245, 235), dtype=np.uint8)
PLAYER_COLORS: tuple[tuple[int, int, int], ...] = (
    (45, 110, 220),
    (125, 50, 200),
    (205, 5, 165),
    (245, 65, 65),
    (245, 130, 0),
    (195, 180, 0),
    (125, 185, 65),
)


@dataclass(frozen=True)
class CommonsHarvestConfig:
    """Configuration shared by Commons Harvest variants."""

    ascii_map: str = ""
    name: str = "open"
    upstream_name: str = "commons_harvest__open"
    num_players: int = NUM_PLAYERS
    regrowth_radius: float = 2.0
    regrowth_probabilities: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.025)
    max_episode_steps: int = 5000
    stochastic_min_steps: int = 1000
    stochastic_interval: int = 100
    stochastic_termination_probability: float = 0.15
    zap_cooldown: int = 2
    zap_beam_length: int = 3
    zap_beam_radius: int = 1
    frames_till_respawn: int = 4
    sprite_size: int = 8
    observation_mode: ObservationMode = "local"
    hidden_tile_rewards: tuple[tuple[str, float], ...] = (("putative_cooperator", -10.0),)

    @property
    def map_lines(self) -> tuple[str, ...]:
        return tuple(self.ascii_map.strip("\n").splitlines())

    @property
    def world_rows(self) -> int:
        return len(self.map_lines)

    @property
    def world_cols(self) -> int:
        lines = self.map_lines
        return len(lines[0]) if lines else 0

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
        lines = self.map_lines
        if not lines:
            raise ValueError("ascii_map must not be empty")
        if len({len(line) for line in lines}) != 1:
            raise ValueError(f"Commons Harvest {self.name} map must be rectangular")
        if self.num_players <= 0:
            raise ValueError("num_players must be positive")
        if self.regrowth_radius < 0:
            raise ValueError("regrowth_radius must be non-negative")
        if not 0.0 <= self.stochastic_termination_probability <= 1.0:
            raise ValueError("stochastic_termination_probability must be in [0, 1]")
        if not self.regrowth_probabilities:
            raise ValueError("regrowth_probabilities must not be empty")
        if any(prob < 0.0 or prob > 1.0 for prob in self.regrowth_probabilities):
            raise ValueError("regrowth probabilities must be in [0, 1]")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.stochastic_interval <= 0:
            raise ValueError("stochastic_interval must be positive")
        if self.zap_cooldown <= 0:
            raise ValueError("zap_cooldown must be positive")
        if self.zap_beam_length <= 0:
            raise ValueError("zap_beam_length must be positive")
        if self.zap_beam_radius < 0:
            raise ValueError("zap_beam_radius must be non-negative")
        if self.frames_till_respawn <= 0:
            raise ValueError("frames_till_respawn must be positive")
        if self.sprite_size <= 0:
            raise ValueError("sprite_size must be positive")
        if self.observation_mode not in {"local", "global", "focal"}:
            raise ValueError("observation_mode must be 'local', 'global' or 'focal'")


@dataclass
class Apple:
    position: Position
    live: bool = True


@dataclass
class Avatar:
    agent: Agent
    index: int
    position: Position
    orientation: int
    reward: float = 0.0
    active: bool = True
    respawn_timer: int = 0
    cooling_timer: int = 0
    role: str = "none"


class CommonsHarvestParallelEnv(ParallelEnv[Agent, dict[str, np.ndarray], int]):
    """PettingZoo ParallelEnv for a Commons Harvest variant."""

    metadata = {
        "name": "commons_harvest_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        config: CommonsHarvestConfig | None = None,
        render_mode: RenderMode = None,
    ):
        self.config = config or CommonsHarvestConfig()
        self.config.validate()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode

        self.possible_agents = [f"player_{index}" for index in range(self.config.num_players)]
        self.agents: list[Agent] = []
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._rng = random.Random()
        self._grid = list(self.config.map_lines)
        self._walls: set[Position] = set()
        self._floor: set[Position] = set()
        self._grass: set[Position] = set()
        self._hidden_reward_tiles: set[Position] = set()
        self._spawn_points: list[Position] = []
        self._inside_spawn_points: list[Position] = []
        self._apples: dict[Position, Apple] = {}
        self._avatars: dict[Agent, Avatar] = {}
        self._step_count = 0
        self._last_events: list[dict[str, Any]] = []
        self._last_zap_cells: set[Position] = set()

        self._pygame = None
        self._screen = None
        self._clock = None
        self._parse_ascii_map()

    @cached_property
    def _action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(8)

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
                "READY_TO_SHOOT": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(),
                    dtype=np.float64,
                ),
            }
        )

    @property
    def ascii_map(self) -> str:
        return self.config.ascii_map

    @property
    def object_observation_ids(self):
        return object_observation_ids(self)

    def observation_space(self, agent: Agent) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Agent) -> gym.spaces.Space:
        return self.action_spaces[agent]

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
        self._last_zap_cells = set()
        self._parse_ascii_map()

        inside_spawns = self._shuffled_positions(self._inside_spawn_points)
        normal_spawns = self._shuffled_positions(self._spawn_points)
        if len(inside_spawns) < 2:
            raise RuntimeError(f"Commons Harvest {self.config.name} requires two inside spawns")
        if len(normal_spawns) < self.config.num_players - 2:
            raise RuntimeError(
                f"Commons Harvest {self.config.name} requires {self.config.num_players - 2} normal spawns"
            )

        self._avatars = {}
        for index, agent in enumerate(self.possible_agents):
            if index < 2:
                position = inside_spawns[index]
            else:
                position = normal_spawns[index - 2]
            self._avatars[agent] = Avatar(
                agent=agent,
                index=index,
                position=position,
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
        self._last_zap_cells = set()
        for avatar in self._avatars.values():
            avatar.reward = 0.0

        self._advance_respawns()
        decoded = {agent: ACTION_TO_COMPONENTS.get(int(actions.get(agent, 0)), (0, 0, 0)) for agent in live_agents}

        for agent in live_agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            _, turn, _ = decoded[agent]
            if turn:
                avatar.orientation = (avatar.orientation + turn) % len(ORIENTATIONS)

        self._fire_zaps(live_agents, decoded)
        self._move_avatars(live_agents, decoded)
        for agent in live_agents:
            avatar = self._avatars[agent]
            if avatar.active:
                self._collect_apple(avatar)
                self._apply_hidden_tile_reward(avatar)
        self._regrow_apples()

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
            pygame.display.set_caption(f"meltingpot-pz: commons_harvest/{self.config.name}")
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

    def _parse_ascii_map(self) -> None:
        lines = list(self.config.map_lines)
        self._grid = lines
        self._walls = set()
        self._floor = set()
        self._grass = set()
        self._hidden_reward_tiles = set()
        self._spawn_points = []
        self._inside_spawn_points = []
        self._apples = {}
        for y, row in enumerate(lines):
            for x, char in enumerate(row):
                position = (x, y)
                if char == "W":
                    self._walls.add(position)
                    continue
                self._floor.add(position)
                if char == "A":
                    self._grass.add(position)
                    self._apples[position] = Apple(position=position, live=True)
                elif char == "P":
                    self._spawn_points.append(position)
                elif char == "Q":
                    self._inside_spawn_points.append(position)
                elif char == "I":
                    self._hidden_reward_tiles.add(position)

    def _shuffled_positions(self, positions: list[Position]) -> list[Position]:
        result = list(positions)
        self._rng.shuffle(result)
        return result

    def _advance_respawns(self) -> None:
        for avatar in self._avatars.values():
            if avatar.active:
                continue
            avatar.respawn_timer -= 1
            if avatar.respawn_timer <= 0:
                self._respawn_avatar(avatar)

    def _respawn_avatar(self, avatar: Avatar) -> None:
        occupied = {other.position for other in self._avatars.values() if other.active and other.agent != avatar.agent}
        candidates = [position for position in self._spawn_points if position not in occupied]
        if not candidates:
            candidates = list(self._spawn_points)
        avatar.position = self._rng.choice(candidates)
        avatar.orientation = self._rng.randrange(len(ORIENTATIONS))
        avatar.active = True
        avatar.respawn_timer = 0
        self._last_events.append({"event": "avatar_respawned", "player_index": avatar.index + 1})

    def _move_avatars(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int, int]],
    ) -> None:
        current_positions = {avatar.position: agent for agent, avatar in self._avatars.items() if avatar.active}
        desired: dict[Agent, Position] = {}
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            move, _, _ = decoded_actions[agent]
            if move == 0:
                desired[agent] = avatar.position
                continue
            offset = RELATIVE_MOVE_TO_OFFSET[move]
            direction = (avatar.orientation + offset) % len(ORIENTATIONS)
            dx, dy = ABS_DELTAS[direction]
            target = (avatar.position[0] + dx, avatar.position[1] + dy)
            desired[agent] = avatar.position if self._is_blocked(target) else target

        target_counts = Counter(desired.values())
        for agent, target in desired.items():
            avatar = self._avatars[agent]
            if target == avatar.position:
                continue
            occupied_by = current_positions.get(target)
            conflict = target_counts[target] > 1 or occupied_by not in (None, agent)
            if not conflict:
                avatar.position = target

    def _collect_apple(self, avatar: Avatar) -> None:
        apple = self._apples.get(avatar.position)
        if apple is None or not apple.live:
            return
        avatar.reward += 1.0
        apple.live = False
        self._last_events.append(
            {
                "event": "edible_consumed",
                "player_index": avatar.index + 1,
                "position": apple.position,
            }
        )

    def _apply_hidden_tile_reward(self, avatar: Avatar) -> None:
        if avatar.position not in self._hidden_reward_tiles:
            return
        reward = dict(self.config.hidden_tile_rewards).get(avatar.role, 0.0)
        if reward == 0.0:
            return
        avatar.reward += reward
        self._last_events.append(
            {
                "event": "hidden_role_based_reward_tile",
                "player_index": avatar.index + 1,
                "position": avatar.position,
                "role": avatar.role,
                "reward": reward,
            }
        )

    def _fire_zaps(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int, int]],
    ) -> None:
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            _, _, fire_zap = decoded_actions[agent]
            if avatar.cooling_timer > 0:
                avatar.cooling_timer -= 1
                continue
            if fire_zap != 1:
                continue
            avatar.cooling_timer = self.config.zap_cooldown
            self._zap_from(avatar)

    def _zap_ray_positions(self, avatar: Avatar) -> list[list[Position]]:
        rays: list[list[Position]] = []
        for lateral in range(
            -self.config.zap_beam_radius,
            self.config.zap_beam_radius + 1,
        ):
            length = self.config.zap_beam_length - abs(lateral)
            if length <= 0:
                continue
            rays.append(
                [
                    _relative_to_world(
                        avatar.position,
                        avatar.orientation,
                        lateral,
                        forward,
                    )
                    for forward in range(1, length + 1)
                ]
            )
        for lateral in (-self.config.zap_beam_radius, self.config.zap_beam_radius):
            if lateral:
                step = 1 if lateral > 0 else -1
                rays.append(
                    [
                        _relative_to_world(
                            avatar.position,
                            avatar.orientation,
                            lateral_step,
                            0,
                        )
                        for lateral_step in range(step, lateral + step, step)
                    ]
                )
        return rays

    def _zap_from(self, zapper: Avatar) -> None:
        hit_agents: set[Agent] = set()
        for ray in self._zap_ray_positions(zapper):
            for position in ray:
                if position in self._walls or not self._in_bounds(position):
                    break
                self._last_zap_cells.add(position)
                target = self._active_avatar_at(position, exclude=zapper.agent)
                if target is None:
                    continue
                if target.agent in hit_agents:
                    break
                hit_agents.add(target.agent)
                target.active = False
                target.respawn_timer = self.config.frames_till_respawn
                target.cooling_timer = 0
                self._last_events.append(
                    {
                        "event": "zap",
                        "source": zapper.index + 1,
                        "target": target.index + 1,
                    }
                )
                break

    def _regrow_apples(self) -> None:
        for apple in self._apples.values():
            if apple.live:
                continue
            num_neighbors = self._live_neighbor_count(apple.position)
            probability_index = min(
                num_neighbors,
                len(self.config.regrowth_probabilities) - 1,
            )
            probability = self.config.regrowth_probabilities[probability_index]
            if self._rng.random() < probability:
                apple.live = True

    def _live_neighbor_count(self, position: Position) -> int:
        px, py = position
        count = 0
        radius_sq = self.config.regrowth_radius**2
        for other in self._apples.values():
            if not other.live or other.position == position:
                continue
            ox, oy = other.position
            if (ox - px) ** 2 + (oy - py) ** 2 <= radius_sq:
                count += 1
        return count

    def _should_stochastically_terminate(self) -> bool:
        if self._step_count < self.config.stochastic_min_steps:
            return False
        if self._step_count % self.config.stochastic_interval != 0:
            return False
        return self._rng.random() < self.config.stochastic_termination_probability

    def _active_avatar_at(
        self,
        position: Position,
        exclude: Agent | None = None,
    ) -> Avatar | None:
        for avatar in self._avatars.values():
            if avatar.agent == exclude or not avatar.active:
                continue
            if avatar.position == position:
                return avatar
        return None

    def _is_blocked(self, position: Position) -> bool:
        return not self._in_bounds(position) or position in self._walls

    def _in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= y < self.config.world_rows and 0 <= x < self.config.world_cols

    def _ready_to_shoot(self, avatar: Avatar) -> float:
        if not avatar.active:
            return 0.0
        normalized = avatar.cooling_timer / self.config.zap_cooldown
        return float(max(1.0 - normalized, 0.0))

    def _observations(self, agents: list[Agent]) -> dict[Agent, dict[str, np.ndarray]]:
        if self.render_mode is None:
            return {agent: object_observation(self, agent) for agent in agents}
        return {agent: self._observation(agent) for agent in agents}

    def _observation(self, agent: Agent) -> dict[str, np.ndarray]:
        avatar = self._avatars[agent]
        if not avatar.active:
            rgb = np.zeros(
                self.config.local_rgb_shape
                if self.config.observation_mode != "global"
                else self.config.world_rgb_shape,
                dtype=np.uint8,
            )
        else:
            rgb = self._render_agent(agent) if self.config.observation_mode != "global" else self._render_world()
        return {
            "RGB": rgb,
            "READY_TO_SHOOT": np.asarray(
                self._ready_to_shoot(avatar),
                dtype=np.float64,
            ),
        }

    def _infos(self, agents: list[Agent]) -> dict[Agent, dict[str, Any]]:
        infos = {}
        for agent in agents:
            avatar = self._avatars[agent]
            infos[agent] = {
                "meltingpot_player_index": avatar.index + 1,
                "position": avatar.position,
                "orientation": ORIENTATIONS[avatar.orientation],
                "active": avatar.active,
                "respawn_timer": avatar.respawn_timer,
                "role": avatar.role,
                "hidden_reward_tiles": set(self._hidden_reward_tiles),
                "events": list(self._last_events),
            }
        return infos

    def _render_world(self) -> np.ndarray:
        frame = np.zeros(self.config.world_rgb_shape, dtype=np.uint8)
        avatar_by_position = {avatar.position: avatar for avatar in self._avatars.values() if avatar.active}
        for y in range(self.config.world_rows):
            for x in range(self.config.world_cols):
                tile = self._render_tile((x, y), avatar_by_position)
                self._paint_tile(frame, x, y, tile)
        return frame

    def _render_agent(self, agent: Agent) -> np.ndarray:
        frame = np.zeros(self.config.local_rgb_shape, dtype=np.uint8)
        avatar = self._avatars[agent]
        avatar_by_position = {candidate.position: candidate for candidate in self._avatars.values() if candidate.active}
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
        if not self._in_bounds(position):
            return _solid_tile(OUT_OF_BOUNDS_COLOR, size)
        if position in self._walls:
            return _wall_tile(size)

        if position in self._grass:
            apple = self._apples.get(position)
            if apple is not None and (apple.live or self._live_neighbor_count(position) > 0):
                tile = _grass_tile(size)
            else:
                tile = _solid_tile(DESSICATED_COLOR, size)
        else:
            tile = _floor_tile(size)

        apple = self._apples.get(position)
        if apple is not None and apple.live:
            tile = _apple_tile(size)

        if position in self._last_zap_cells:
            tile = _zap_tile(tile)

        avatar = avatar_by_position.get(position)
        if avatar is not None:
            tile = _avatar_tile(PLAYER_COLORS[avatar.index], avatar.orientation, size)
        return tile

    def _paint_tile(
        self,
        frame: np.ndarray,
        tile_x: int,
        tile_y: int,
        tile: np.ndarray,
    ) -> None:
        size = self.config.sprite_size
        y0 = tile_y * size
        x0 = tile_x * size
        frame[y0 : y0 + size, x0 : x0 + size] = tile


def make_parallel_env(
    *,
    config_cls: type[CommonsHarvestConfig],
    env_cls: type[CommonsHarvestParallelEnv],
    render_mode: RenderMode = None,
    config: CommonsHarvestConfig | None = None,
    **config_overrides: Any,
) -> CommonsHarvestParallelEnv:
    """Create a variant env while preserving config override ergonomics."""
    if config is None:
        config = config_cls(**config_overrides)
    elif config_overrides:
        config = replace(config, **config_overrides)
    return env_cls(config=config, render_mode=render_mode)


def make_aec_env(parallel_factory, render_mode: RenderMode = None, **kwargs: Any):
    """Create an AEC wrapper for a Commons Harvest parallel factory."""
    return parallel_to_aec(parallel_factory(render_mode=render_mode, **kwargs))


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


def _floor_tile(size: int) -> np.ndarray:
    tile = _solid_tile(FLOOR_COLOR, size)
    tile[::3, ::3] = np.maximum(tile[::3, ::3] - 8, 0)
    return tile


def _grass_tile(size: int) -> np.ndarray:
    tile = _solid_tile(GRASS_COLOR, size)
    tile[1::3, :] = np.maximum(tile[1::3, :] - 12, 0)
    return tile


def _wall_tile(size: int) -> np.ndarray:
    tile = _solid_tile(WALL_COLOR, size)
    tile[0, :] = WALL_LIGHT
    tile[:, 0] = WALL_LIGHT
    return tile


def _apple_tile(size: int) -> np.ndarray:
    tile = _grass_tile(size)
    center = (size - 1) / 2
    radius = max(size / 2 - 1, 1)
    for y in range(size):
        for x in range(size):
            if (x - center) ** 2 + (y - center) ** 2 <= radius**2:
                tile[y, x] = APPLE_COLOR
    tile[size // 2 :, size // 2] = APPLE_DARK
    return tile


def _zap_tile(tile: np.ndarray) -> np.ndarray:
    result = tile.copy()
    result[:, result.shape[1] // 2] = ZAP_COLOR
    result[result.shape[0] // 2, :] = ZAP_COLOR
    return result


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
    "Apple",
    "Avatar",
    "CommonsHarvestConfig",
    "CommonsHarvestParallelEnv",
    "NUM_PLAYERS",
    "ObservationMode",
    "RenderMode",
    "make_aec_env",
    "make_parallel_env",
]
