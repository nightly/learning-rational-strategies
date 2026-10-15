"""Shared native implementation for Melting Pot's Territory substrates."""

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
Topology = Literal["BOUNDED", "TORUS"]

ORIENTATIONS: tuple[str, ...] = ("N", "E", "S", "W")
ABS_DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RELATIVE_MOVE_TO_OFFSET: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}
ACTION_TO_COMPONENTS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0),
    2: (3, 0, 0, 0),
    3: (4, 0, 0, 0),
    4: (2, 0, 0, 0),
    5: (0, -1, 0, 0),
    6: (0, 1, 0, 0),
    7: (0, 0, 1, 0),
    8: (0, 0, 0, 1),
}

WALL_CHARS = {"F", "|", "=", "T", "J", "L", "W"}
FLOOR_CHARS = {","}
SPAWN_CHARS = {"P"}
RESOURCE_CHARS = {"R"}

FLOOR_COLOR = np.array((34, 30, 28), dtype=np.uint8)
FLOOR_LIGHT = np.array((48, 42, 38), dtype=np.uint8)
WALL_COLOR = np.array((66, 62, 60), dtype=np.uint8)
WALL_LIGHT = np.array((96, 91, 86), dtype=np.uint8)
UNCLAIMED_RESOURCE = np.array((83, 84, 84), dtype=np.uint8)
DAMAGED_COLOR = np.array((18, 18, 18), dtype=np.uint8)
DESTROYED_COLOR = np.array((70, 55, 48), dtype=np.uint8)
ZAP_COLOR = np.array((252, 252, 106), dtype=np.uint8)
CLAIM_COLOR = np.array((245, 245, 235), dtype=np.uint8)
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
    (50, 170, 160),
    (185, 115, 55),
)


@dataclass(frozen=True)
class RandomChoiceSpec:
    """Seeded ASCII expansion rule used by procedural Territory variants."""

    char: str
    a: Literal["spawn", "resource"]
    b: Literal["floor"]
    odds: tuple[int, int]


@dataclass(frozen=True)
class TerritoryConfig:
    """Configuration for a native Territory substrate variant."""

    ascii_map: str
    name: str
    num_players: int
    topology: Topology
    random_choices: tuple[RandomChoiceSpec, ...] = ()
    max_episode_steps: int = 5000
    stochastic_min_steps: int = 1000
    stochastic_interval: int = 100
    stochastic_termination_probability: float = 0.2
    zap_cooldown: int = 4
    zap_beam_length: int = 2
    zap_beam_radius: int = 1
    claim_beam_length: int = 2
    claim_beam_radius: int = 0
    resource_initial_health: int = 2
    resource_reward: float = 1.0
    resource_reward_rate: float = 0.01
    resource_reward_delay: int = 25
    resource_repair_delay: int = 15
    resource_repair_probability: float = 0.1
    sanction_freeze_frames: int = 25
    sanction_recovery_time: int = 50
    sprite_size: int = 8
    observation_mode: ObservationMode = "local"

    @property
    def map_lines(self) -> tuple[str, ...]:
        return tuple(self.ascii_map.strip("\n").splitlines())

    @property
    def world_rows(self) -> int:
        return len(self.map_lines)

    @property
    def world_cols(self) -> int:
        return len(self.map_lines[0])

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
        if not self.map_lines:
            raise ValueError("ascii_map must not be empty")
        if len({len(line) for line in self.map_lines}) != 1:
            raise ValueError("ascii_map must be rectangular")
        if self.num_players <= 0:
            raise ValueError("num_players must be positive")
        if self.topology not in {"BOUNDED", "TORUS"}:
            raise ValueError("topology must be 'BOUNDED' or 'TORUS'")
        probabilities = (
            self.stochastic_termination_probability,
            self.resource_reward_rate,
            self.resource_repair_probability,
        )
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("probabilities must be in [0, 1]")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.stochastic_interval <= 0:
            raise ValueError("stochastic_interval must be positive")
        if self.zap_cooldown <= 0:
            raise ValueError("zap_cooldown must be positive")
        if self.zap_beam_length <= 0 or self.claim_beam_length <= 0:
            raise ValueError("beam lengths must be positive")
        if self.zap_beam_radius < 0 or self.claim_beam_radius < 0:
            raise ValueError("beam radii must be non-negative")
        if self.resource_initial_health <= 0:
            raise ValueError("resource_initial_health must be positive")
        if self.resource_reward_delay < 0 or self.resource_repair_delay < 0:
            raise ValueError("resource delays must be non-negative")
        if self.sanction_freeze_frames < 0 or self.sanction_recovery_time < 0:
            raise ValueError("sanction timings must be non-negative")
        if self.observation_mode not in {"local", "global", "focal"}:
            raise ValueError("observation_mode must be 'local', 'global' or 'focal'")


@dataclass
class Resource:
    position: Position
    health: int
    owner: Agent | None = None
    frames_since_claimed: int = 0
    reward_active: bool = False
    destroyed: bool = False
    frames_since_zapped: int | None = None
    never_claimed: bool = True


@dataclass
class Avatar:
    agent: Agent
    index: int
    position: Position
    orientation: int
    reward: float = 0.0
    active: bool = True
    zap_cooling_timer: int = 0
    movement_freeze_timer: int = 0
    zapping_freeze_timer: int = 0
    removal_timer: int | None = None
    sanction_level: int = 1
    sanction_recovery_counter: int = 0


class TerritoryParallelEnv(ParallelEnv[Agent, dict[str, np.ndarray], int]):
    """PettingZoo ParallelEnv for a native Territory variant."""

    metadata = {
        "name": "territory_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, config: TerritoryConfig, render_mode: RenderMode = None):
        self.config = config
        self.config.validate()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode

        self.possible_agents = [f"player_{index}" for index in range(self.config.num_players)]
        self.agents: list[Agent] = []
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._rng = random.Random()
        self._grid = self.config.map_lines
        self._walls: set[Position] = set()
        self._floor: set[Position] = set()
        self._spawn_points: list[Position] = []
        self._resources: dict[Position, Resource] = {}
        self._avatars: dict[Agent, Avatar] = {}
        self._step_count = 0
        self._last_events: list[dict[str, Any]] = []
        self._last_zap_cells: set[Position] = set()
        self._last_claim_cells: set[Position] = set()

        self._pygame = None
        self._screen = None
        self._clock = None

    @cached_property
    def _action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(ACTION_TO_COMPONENTS))

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
        return "\n".join(self._grid)

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
        self._last_claim_cells = set()
        self._parse_ascii_map()

        spawn_points = self._shuffled_positions(self._spawn_points)
        if len(spawn_points) < len(self.possible_agents):
            raise RuntimeError(f"{self.config.name} requires enough spawn points")

        self._avatars = {}
        for index, agent in enumerate(self.possible_agents):
            self._avatars[agent] = Avatar(
                agent=agent,
                index=index,
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
        self._last_zap_cells = set()
        self._last_claim_cells = set()
        self._start_frame()
        self._advance_avatar_timers()
        self._release_claims_from_inactive_owners()

        decoded = {agent: ACTION_TO_COMPONENTS.get(int(actions.get(agent, 0)), (0, 0, 0, 0)) for agent in live_agents}

        for agent in live_agents:
            avatar = self._avatars[agent]
            if not self._can_move(avatar):
                continue
            _, turn, _, _ = decoded[agent]
            if turn:
                avatar.orientation = (avatar.orientation + turn) % len(ORIENTATIONS)

        self._fire_zaps(live_agents, decoded)
        self._move_avatars(live_agents, decoded)
        self._claim_with_paintbrushes(live_agents)
        self._fire_claim_beams(live_agents, decoded)
        self._update_resources()

        self._step_count += 1
        terminated = self._should_stochastically_terminate()
        truncated = self._step_count >= self.config.max_episode_steps and not terminated

        observations = self._observations(live_agents)
        rewards = {agent: float(self._avatars[agent].reward) for agent in live_agents}
        terminations = {agent: terminated for agent in live_agents}
        truncations = {agent: truncated for agent in live_agents}
        infos = self._infos(live_agents)

        if terminated or truncated:
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
            pygame.display.set_caption(f"meltingpot-pz: territory/{self.config.name}")
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
        self._grid = self.config.map_lines
        self._walls = set()
        self._floor = set()
        self._spawn_points = []
        self._resources = {}
        random_choice_by_char = {choice.char: choice for choice in self.config.random_choices}

        for y, line in enumerate(self._grid):
            for x, char in enumerate(line):
                category = self._category_for_char(char, random_choice_by_char)
                position = (x, y)
                if category == "wall":
                    self._walls.add(position)
                else:
                    self._floor.add(position)
                if category == "spawn":
                    self._spawn_points.append(position)
                elif category == "resource":
                    self._resources[position] = Resource(
                        position=position,
                        health=self.config.resource_initial_health,
                    )

    def _category_for_char(
        self,
        char: str,
        random_choice_by_char: dict[str, RandomChoiceSpec],
    ) -> Literal["floor", "wall", "spawn", "resource"]:
        if char in random_choice_by_char:
            choice = random_choice_by_char[char]
            a_odds, b_odds = choice.odds
            selected = self._rng.choice([choice.a] * a_odds + [choice.b] * b_odds)
            return selected
        if char in WALL_CHARS:
            return "wall"
        if char in SPAWN_CHARS:
            return "spawn"
        if char in RESOURCE_CHARS:
            return "resource"
        if char in FLOOR_CHARS:
            return "floor"
        raise ValueError(f"Unknown Territory map character {char!r}")

    def _shuffled_positions(self, positions: list[Position]) -> list[Position]:
        shuffled = list(positions)
        self._rng.shuffle(shuffled)
        return shuffled

    def _start_frame(self) -> None:
        for avatar in self._avatars.values():
            avatar.reward = 0.0

    def _advance_avatar_timers(self) -> None:
        for avatar in self._avatars.values():
            if not avatar.active:
                continue
            if avatar.removal_timer is not None:
                avatar.removal_timer -= 1
                if avatar.removal_timer <= 0:
                    self._remove_avatar(avatar)
                    continue
            if avatar.movement_freeze_timer > 0:
                avatar.movement_freeze_timer -= 1
            old_zapping_freeze = avatar.zapping_freeze_timer
            if avatar.zapping_freeze_timer > 0:
                avatar.zapping_freeze_timer -= 1
            if old_zapping_freeze == 1:
                avatar.zap_cooling_timer = max(avatar.zap_cooling_timer, self.config.zap_cooldown)
            if avatar.sanction_level != 1:
                avatar.sanction_recovery_counter += 1
                if (
                    self.config.sanction_recovery_time
                    and avatar.sanction_recovery_counter >= self.config.sanction_recovery_time
                ):
                    avatar.sanction_level = 1
                    avatar.sanction_recovery_counter = 0
                    self._last_events.append(
                        {
                            "event": "set_sanctioning_level",
                            "player_index": avatar.index + 1,
                            "level": 1,
                        }
                    )

    def _remove_avatar(self, avatar: Avatar) -> None:
        avatar.active = False
        avatar.removal_timer = None
        avatar.movement_freeze_timer = 0
        avatar.zapping_freeze_timer = 0
        avatar.zap_cooling_timer = 0
        self._last_events.append({"event": "avatar_removed", "player_index": avatar.index + 1})
        self._release_claims_from_inactive_owners()

    def _can_move(self, avatar: Avatar) -> bool:
        return avatar.active and avatar.movement_freeze_timer <= 0

    def _can_zap(self, avatar: Avatar) -> bool:
        return avatar.active and avatar.zapping_freeze_timer <= 0

    def _move_avatars(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        current_positions = {avatar.position: agent for agent, avatar in self._avatars.items() if avatar.active}
        desired: dict[Agent, Position] = {}
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not self._can_move(avatar):
                continue
            move, _, _, _ = decoded_actions[agent]
            if move == 0:
                desired[agent] = avatar.position
                continue
            offset = RELATIVE_MOVE_TO_OFFSET[move]
            direction = (avatar.orientation + offset) % len(ORIENTATIONS)
            dx, dy = ABS_DELTAS[direction]
            target = self._canonical_position((avatar.position[0] + dx, avatar.position[1] + dy))
            desired[agent] = avatar.position if target is None or self._is_blocked(target) else target

        target_counts = Counter(desired.values())
        for agent, target in desired.items():
            avatar = self._avatars[agent]
            if target == avatar.position:
                continue
            occupied_by = current_positions.get(target)
            conflict = target_counts[target] > 1 or occupied_by not in (None, agent)
            if not conflict:
                avatar.position = target

    def _fire_zaps(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not self._can_zap(avatar):
                continue
            _, _, fire_zap, _ = decoded_actions[agent]
            if avatar.zap_cooling_timer > 0:
                avatar.zap_cooling_timer -= 1
                continue
            if fire_zap != 1:
                continue
            avatar.zap_cooling_timer = self.config.zap_cooldown
            self._zap_from(avatar)

    def _zap_from(self, zapper: Avatar) -> None:
        hit_agents: set[Agent] = set()
        rays = self._beam_ray_positions(zapper, self.config.zap_beam_length, self.config.zap_beam_radius)
        for ray in rays:
            for raw_position in ray:
                position = self._canonical_position(raw_position)
                if position is None or position in self._walls:
                    break
                self._last_zap_cells.add(position)
                resource = self._resources.get(position)
                if resource is not None and not resource.destroyed:
                    blocked = self._zap_resource(zapper, resource)
                    if blocked:
                        break
                    continue
                target = self._active_avatar_at(position, exclude=zapper.agent)
                if target is None:
                    continue
                if target.agent in hit_agents:
                    break
                hit_agents.add(target.agent)
                self._zap_avatar(zapper, target)
                break

    def _zap_resource(self, zapper: Avatar, resource: Resource) -> bool:
        resource.health -= 1
        resource.frames_since_zapped = 0
        self._last_events.append(
            {
                "event": "resource_zapped",
                "player_index": zapper.index + 1,
                "position": resource.position,
                "health": max(resource.health, 0),
            }
        )
        if resource.health > 0:
            return True
        resource.destroyed = True
        resource.owner = None
        resource.reward_active = False
        resource.frames_since_claimed = 0
        resource.frames_since_zapped = None
        self._last_events.append(
            {
                "event": "destroyed_resource",
                "player_index": zapper.index + 1,
                "position": resource.position,
            }
        )
        return False

    def _zap_avatar(self, zapper: Avatar, target: Avatar) -> None:
        self._last_events.append(
            {
                "event": "zap",
                "source": zapper.index + 1,
                "target": target.index + 1,
            }
        )
        if target.sanction_level <= 1:
            target.sanction_level = 2
            target.movement_freeze_timer = max(target.movement_freeze_timer, self.config.sanction_freeze_frames)
            target.zapping_freeze_timer = max(target.zapping_freeze_timer, self.config.sanction_freeze_frames)
            self._last_events.append(
                {
                    "event": "set_sanctioning_level",
                    "player_index": target.index + 1,
                    "level": 2,
                }
            )
        else:
            target.sanction_level = 1
            target.removal_timer = 1
            target.movement_freeze_timer = max(target.movement_freeze_timer, 1)
            target.zapping_freeze_timer = max(target.zapping_freeze_timer, 1)
            self._last_events.append(
                {
                    "event": "set_sanctioning_level",
                    "player_index": target.index + 1,
                    "level": 1,
                }
            )
            self._last_events.append(
                {
                    "event": "removal_due_to_sanctioning",
                    "source": zapper.index + 1,
                    "target": target.index + 1,
                }
            )
        target.sanction_recovery_counter = 0
        self._last_events.append(
            {
                "event": "sanctioning",
                "source": zapper.index + 1,
                "target": target.index + 1,
            }
        )

    def _claim_with_paintbrushes(self, live_agents: list[Agent]) -> None:
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            raw_position = _relative_to_world(avatar.position, avatar.orientation, 0, 1)
            position = self._canonical_position(raw_position)
            if position is None:
                continue
            self._last_claim_cells.add(position)
            self._claim_resource(avatar, position)

    def _fire_claim_beams(
        self,
        live_agents: list[Agent],
        decoded_actions: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        for agent in live_agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            _, _, _, fire_claim = decoded_actions[agent]
            if fire_claim != 1:
                continue
            rays = self._beam_ray_positions(avatar, self.config.claim_beam_length, self.config.claim_beam_radius)
            for ray in rays:
                for raw_position in ray:
                    position = self._canonical_position(raw_position)
                    if position is None or position in self._walls:
                        break
                    self._last_claim_cells.add(position)
                    self._claim_resource(avatar, position)

    def _claim_resource(self, avatar: Avatar, position: Position) -> None:
        resource = self._resources.get(position)
        if resource is None or resource.destroyed:
            return
        if resource.owner == avatar.agent:
            return
        previous_owner = resource.owner
        resource.owner = avatar.agent
        resource.frames_since_claimed = 0
        resource.reward_active = False
        resource.never_claimed = False
        self._last_events.append(
            {
                "event": "claimed_resource",
                "player_index": avatar.index + 1,
                "position": position,
                "previous_owner": previous_owner,
            }
        )

    def _update_resources(self) -> None:
        for resource in self._resources.values():
            if resource.destroyed:
                continue
            if resource.owner is not None:
                owner = self._avatars[resource.owner]
                if not owner.active:
                    self._release_resource(resource)
                else:
                    resource.frames_since_claimed += 1
                    if resource.frames_since_claimed >= self.config.resource_reward_delay:
                        resource.reward_active = True
                        if self._rng.random() < self.config.resource_reward_rate:
                            owner.reward += self.config.resource_reward
                            self._last_events.append(
                                {
                                    "event": "resource_rewarded",
                                    "player_index": owner.index + 1,
                                    "position": resource.position,
                                }
                            )
            if resource.frames_since_zapped is not None and resource.health < self.config.resource_initial_health:
                if resource.frames_since_zapped >= self.config.resource_repair_delay:
                    if self._rng.random() < self.config.resource_repair_probability:
                        resource.health += 1
                        self._last_events.append(
                            {
                                "event": "resource_repaired",
                                "position": resource.position,
                                "health": resource.health,
                            }
                        )
                if resource.health >= self.config.resource_initial_health:
                    resource.frames_since_zapped = None
                else:
                    resource.frames_since_zapped += 1

    def _release_claims_from_inactive_owners(self) -> None:
        for resource in self._resources.values():
            if resource.owner is not None and not self._avatars[resource.owner].active:
                self._release_resource(resource)

    def _release_resource(self, resource: Resource) -> None:
        previous_owner = resource.owner
        resource.owner = None
        resource.frames_since_claimed = 0
        resource.reward_active = False
        self._last_events.append(
            {
                "event": "released_resource",
                "position": resource.position,
                "previous_owner": previous_owner,
            }
        )

    def _beam_ray_positions(
        self,
        avatar: Avatar,
        beam_length: int,
        beam_radius: int,
    ) -> list[list[Position]]:
        rays: list[list[Position]] = []
        for lateral in range(-beam_radius, beam_radius + 1):
            length = beam_length - abs(lateral)
            if length <= 0:
                continue
            rays.append(
                [
                    _relative_to_world(avatar.position, avatar.orientation, lateral, forward)
                    for forward in range(1, length + 1)
                ]
            )
        for lateral in (-beam_radius, beam_radius):
            if lateral:
                step = 1 if lateral > 0 else -1
                rays.append(
                    [
                        _relative_to_world(avatar.position, avatar.orientation, lateral_step, 0)
                        for lateral_step in range(step, lateral + step, step)
                    ]
                )
        return rays

    def _ready_to_shoot(self, avatar: Avatar) -> float:
        if not avatar.active or avatar.zapping_freeze_timer > 0:
            return 0.0
        normalized = avatar.zap_cooling_timer / self.config.zap_cooldown
        return max(1.0 - normalized, 0.0)

    def _active_avatar_at(self, position: Position, exclude: Agent | None = None) -> Avatar | None:
        for avatar in self._avatars.values():
            if avatar.agent == exclude or not avatar.active:
                continue
            if avatar.position == position:
                return avatar
        return None

    def _is_blocked(self, position: Position) -> bool:
        if position in self._walls:
            return True
        resource = self._resources.get(position)
        return resource is not None and not resource.destroyed

    def _canonical_position(self, position: Position) -> Position | None:
        x, y = position
        if self.config.topology == "TORUS":
            return (x % self.config.world_cols, y % self.config.world_rows)
        if 0 <= y < self.config.world_rows and 0 <= x < self.config.world_cols:
            return position
        return None

    def _should_stochastically_terminate(self) -> bool:
        if self._step_count < self.config.stochastic_min_steps:
            return False
        if self._step_count % self.config.stochastic_interval != 0:
            return False
        return self._rng.random() < self.config.stochastic_termination_probability

    def _observations(self, agents: list[Agent]) -> dict[Agent, dict[str, np.ndarray]]:
        if self.render_mode is None:
            return {agent: object_observation(self, agent) for agent in agents}
        return {
            agent: {
                "RGB": (
                    self._render_agent(agent) if self.config.observation_mode != "global" else self._render_world()
                ),
                "READY_TO_SHOOT": np.asarray(self._ready_to_shoot(self._avatars[agent]), dtype=np.float64),
            }
            for agent in agents
        }

    def _infos(self, agents: list[Agent]) -> dict[Agent, dict[str, Any]]:
        return {
            agent: {
                "meltingpot_player_index": self._avatars[agent].index + 1,
                "position": self._avatars[agent].position,
                "orientation": ORIENTATIONS[self._avatars[agent].orientation],
                "active": self._avatars[agent].active,
                "zap_cooldown": self._avatars[agent].zap_cooling_timer,
                "movement_freeze_timer": self._avatars[agent].movement_freeze_timer,
                "zapping_freeze_timer": self._avatars[agent].zapping_freeze_timer,
                "sanction_level": self._avatars[agent].sanction_level,
                "events": list(self._last_events),
            }
            for agent in agents
        }

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
                raw_world_pos = _relative_to_world(
                    avatar.position,
                    avatar.orientation,
                    rel_right,
                    rel_forward,
                )
                world_pos = self._canonical_position(raw_world_pos)
                tile = (
                    _solid_tile(OUT_OF_BOUNDS_COLOR, self.config.sprite_size)
                    if world_pos is None
                    else self._render_tile(world_pos, avatar_by_position)
                )
                self._paint_tile(frame, col, row, tile)
        return frame

    def _render_tile(
        self,
        position: Position,
        avatar_by_position: dict[Position, Avatar],
    ) -> np.ndarray:
        size = self.config.sprite_size
        tile = _floor_tile(size)
        if position in self._walls:
            tile = _wall_tile(size)
        resource = self._resources.get(position)
        if resource is not None:
            tile = _resource_tile(resource, size)
        if position in self._last_zap_cells:
            tile = _beam_tile(tile, ZAP_COLOR)
        elif position in self._last_claim_cells:
            tile = _beam_tile(tile, CLAIM_COLOR)
        avatar = avatar_by_position.get(position)
        if avatar is not None:
            tile = _avatar_tile(
                tile,
                np.array(PLAYER_COLORS[avatar.index % len(PLAYER_COLORS)], dtype=np.uint8),
                avatar.orientation,
                avatar.sanction_level,
            )
        return tile

    def _paint_tile(
        self,
        frame: np.ndarray,
        tile_col: int,
        tile_row: int,
        tile: np.ndarray,
    ) -> None:
        size = self.config.sprite_size
        y0 = tile_row * size
        x0 = tile_col * size
        frame[y0 : y0 + size, x0 : x0 + size] = tile


def make_parallel_env(
    *,
    config_cls: type[TerritoryConfig],
    env_cls: type[TerritoryParallelEnv],
    render_mode: RenderMode = None,
    config: TerritoryConfig | None = None,
    **config_overrides: Any,
) -> TerritoryParallelEnv:
    if config is None:
        config = config_cls(**config_overrides)
    elif config_overrides:
        config = replace(config, **config_overrides)
    return env_cls(config=config, render_mode=render_mode)


def make_aec_env(
    *,
    parallel_factory: Any,
    render_mode: RenderMode = None,
    **kwargs: Any,
):
    return parallel_to_aec(parallel_factory(render_mode=render_mode, **kwargs))


def _relative_to_world(
    origin: Position,
    orientation: int,
    rel_right: int,
    rel_forward: int,
) -> Position:
    dx_forward, dy_forward = ABS_DELTAS[orientation]
    dx_right, dy_right = ABS_DELTAS[(orientation + 1) % len(ABS_DELTAS)]
    return (
        origin[0] + dx_right * rel_right + dx_forward * rel_forward,
        origin[1] + dy_right * rel_right + dy_forward * rel_forward,
    )


def _solid_tile(color: np.ndarray, size: int) -> np.ndarray:
    return np.broadcast_to(color, (size, size, 3)).copy()


def _floor_tile(size: int) -> np.ndarray:
    tile = _solid_tile(FLOOR_COLOR, size)
    tile[::2, 1::2] = FLOOR_LIGHT
    return tile


def _wall_tile(size: int) -> np.ndarray:
    tile = _solid_tile(WALL_COLOR, size)
    tile[0, :] = WALL_LIGHT
    tile[:, 0] = WALL_LIGHT
    return tile


def _resource_tile(resource: Resource, size: int) -> np.ndarray:
    if resource.destroyed:
        tile = _floor_tile(size)
        tile[2:-2, 2:-2] = DESTROYED_COLOR
        return tile
    if resource.owner is None:
        tile = _solid_tile(UNCLAIMED_RESOURCE, size)
    else:
        owner_index = int(resource.owner.removeprefix("player_"))
        color = np.array(PLAYER_COLORS[owner_index % len(PLAYER_COLORS)], dtype=np.uint8)
        if resource.reward_active:
            tile = _solid_tile(color, size)
        else:
            wet = ((color.astype(np.uint16) + UNCLAIMED_RESOURCE.astype(np.uint16)) // 2).astype(np.uint8)
            tile = _solid_tile(wet, size)
    tile[0, :] = WALL_LIGHT
    tile[:, 0] = WALL_LIGHT
    if resource.health < 2:
        tile[2:6, 3:5] = DAMAGED_COLOR
        tile[4:6, 1:3] = DAMAGED_COLOR
    return tile


def _beam_tile(tile: np.ndarray, color: np.ndarray) -> np.ndarray:
    result = tile.copy()
    result[3:5, :] = color
    result[:, 3:5] = color
    return result


def _avatar_tile(
    background: np.ndarray,
    color: np.ndarray,
    orientation: int,
    sanction_level: int,
) -> np.ndarray:
    tile = background.copy()
    tile[1:-1, 1:-1] = color
    tile[2:-2, 2:-2] = ((color.astype(np.uint16) * 3 + 255) // 4).astype(np.uint8)
    if sanction_level > 1:
        tile[1, 1:-1] = DAMAGED_COLOR
        tile[-2, 1:-1] = DAMAGED_COLOR
    if orientation == 0:
        tile[0, 3:5] = AVATAR_TRIM
    elif orientation == 1:
        tile[3:5, -1] = AVATAR_TRIM
    elif orientation == 2:
        tile[-1, 3:5] = AVATAR_TRIM
    else:
        tile[3:5, 0] = AVATAR_TRIM
    return tile
