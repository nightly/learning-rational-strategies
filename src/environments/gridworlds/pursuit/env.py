import functools

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from .common import Actions, Coord, Tile
from .layout import Layout, default_layout
from .renderer import GuardGridworldRenderer


class PursuitEvasion(ParallelEnv):
    """
    Pursuit Evasion (Parallel PettingZoo)
    Also referred to as Guard-Gridworld.

    - Agents: 2 or 3 (0 is the Guard; 1 and 2 are normal).
    - Actions: {Left, Right, Up, Down, Stay}. Moves into walls/bounds are ignored.
    - Collision: any two agents share a cell after the simultaneous move.
    - Goal: true iff agent 0 OR 1 is in the goal region.

    Rewards default to 0; optional shaping provided via constructor.
    Observations (globalised): binary channels shared by all agents:
      [Wall, Goal, A0, A1, (A2)] with dtype=np.float32. If `flatten_observations`
      is True, this is flattened to (H*W*C,).
    """

    metadata = {"name": "guard_gridworld_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 3,
        layout: Layout | None = default_layout(),
        terminates_after_goal: bool = True,
        terminates_after_collision: bool = True,
        max_steps: int = 10000,
        reward_on_goal: float = 0,  # we derive rewards through EPLTL objectives
        reward_guard_failure: float = 0,  # we derive rewards through EPLTL objectives
        reward_normal_agent_collision: float = 0,  # we derive rewards through EPLTL objectives
        reward_guard_agent_collision: float = 0,  # we derive rewards through EPLTL objectives
        slip_intruder_prob: float = 0.1,
        slip_guard_prob: float = 0.1,
        flatten_observations: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        assert num_agents in (2, 3), "GuardGridworld supports 2 or 3 agents."
        self.n_agents = num_agents
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.render_mode = render_mode
        self._normal_agent_ids = [f"player_{i}" for i in range(1, num_agents)]  # non-guard agents
        self._guard_agent_ids = ["player_0"]

        self.layout = layout
        self.grid_size = self.layout.grid_size
        self.H, self.W = self.layout.grid_size

        self.start_positions = list(layout.start_positions) if layout.start_positions is not None else None
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # termination & rewards
        self.terminates_after_goal = bool(terminates_after_goal)
        self.terminates_after_collision = bool(terminates_after_collision)
        self.max_steps = int(max_steps)
        self.reward_on_goal = float(reward_on_goal)
        self.reward_guard_failure = float(reward_guard_failure)
        self.reward_normal_agent_collision = float(reward_normal_agent_collision)
        self.reward_guard_agent_collision = float(reward_guard_agent_collision)

        # slip probabilities
        self.slip_intruder_prob = float(slip_intruder_prob)
        self.slip_guard_prob = float(slip_guard_prob)

        # obs shape options
        self.flatten = bool(flatten_observations)

        # renderer (optional)
        self._renderer: GuardGridworldRenderer | None = None
        if self.render_mode is not None:
            self._renderer = GuardGridworldRenderer(self.render_mode)

        # runtime state
        self._t = 0
        self._collision_ever = False
        self._goal_last_step = False
        self._collision_last_step = False
        self._agent_pos: dict[str, Coord] = {}

        # observation channels: Wall, Goal, Agent_0, Agent_1, [Agent_2]
        self.n_agent_channels = self.n_agents
        self.n_obs_types = 2 + self.n_agent_channels

        self.observation_spaces = {a: self.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.action_space(a) for a in self.possible_agents}

    # ─────────────────────────── PettingZoo API ───────────────────────────
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        H, W, C = self.H, self.W, self.n_obs_types
        if self.flatten:
            return Box(low=0, high=1, shape=(H * W * C,), dtype=np.float32)
        return Box(low=0, high=1, shape=(H, W, C), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # Left, Right, Up, Down, Stay

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._t = 0
        self._collision_ever = False
        self._goal_last_step = False
        self._collision_last_step = False
        self._agent_pos = {}

        # set starting positions (distinct, non-wall)
        if self.start_positions:
            assert len(self.start_positions) >= self.n_agents
            for i, pid in enumerate(self.agents):
                r, c = self.start_positions[i]
                if (r, c) in self.layout.walls:
                    raise ValueError(f"start {r, c} is a wall")
                self._agent_pos[pid] = (int(r), int(c))
            if len(set(self._agent_pos.values())) != self.n_agents:
                raise ValueError("Duplicate start positions.")
        else:
            free = [(r, c) for r in range(self.H) for c in range(self.W) if (r, c) not in self.layout.walls]
            self.rng.shuffle(free)
            for i, pid in enumerate(self.agents):
                # avoid starting inside the goal to keep property meaningful
                while True:
                    rc = free.pop()
                    if rc not in self.layout.goal_region:
                        self._agent_pos[pid] = rc
                        break

        obs = {a: self._obs() for a in self.agents}
        infos = {
            a: {"t": self._t, "collision": False, "goal_reached": False, "property_satisfied": False}
            for a in self.agents
        }
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not actions or self.agents == []:
            self.agents = []
            return {}, {}, {}, {}, {}

        # simultaneous proposed moves
        proposed: dict[str, Coord] = {}
        for pid in self.agents:
            r, c = self._agent_pos[pid]
            act = int(actions[pid])

            # slip -> Stay (agent-specific probability)
            # if pid in self._guard_agent_ids:
            #     if self.slip_guard_prob > 0.0 and self.rng.rand() < self.slip_guard_prob:
            #         act = Actions.Stay
            # elif pid in self._normal_agent_ids:
            #     if self.slip_intruder_prob > 0.0 and self.rng.rand() < self.slip_intruder_prob:
            #         act = Actions.Stay

            # slip -> random legal cardinal move (agent-specific probability)
            slip_p = 0.0
            if pid in self._guard_agent_ids:
                slip_p = self.slip_guard_prob
            elif pid in self._normal_agent_ids:
                slip_p = self.slip_intruder_prob

            if slip_p > 0.0 and self.rng.rand() < slip_p:
                legal = []
                for cand in (Actions.Left, Actions.Right, Actions.Up, Actions.Down):
                    rr, cc = r, c
                    if cand == Actions.Left:
                        cc -= 1
                    elif cand == Actions.Right:
                        cc += 1
                    elif cand == Actions.Up:
                        rr -= 1
                    elif cand == Actions.Down:
                        rr += 1

                    if (0 <= rr < self.H and 0 <= cc < self.W) and ((rr, cc) not in self.layout.walls):
                        legal.append(int(cand))

                # if no legal cardinal move exists, fall back to staying
                act = int(self.rng.choice(legal)) if legal else Actions.Stay

            dr = dc = 0
            if act == Actions.Left:
                dc = -1
            elif act == Actions.Right:
                dc = 1
            elif act == Actions.Up:
                dr = -1
            elif act == Actions.Down:
                dr = 1
            elif act == Actions.Stay:
                pass

            nr, nc = r + dr, c + dc
            # blocked by wall or out-of-bounds → stay
            if not (0 <= nr < self.H and 0 <= nc < self.W) or (nr, nc) in self.layout.walls:
                nr, nc = r, c
            proposed[pid] = (nr, nc)

        # apply atomically
        self._agent_pos.update(proposed)
        self._t += 1

        # compute collision/goal
        positions = list(self._agent_pos.values())
        collision = len(positions) != len(set(positions))
        self._collision_ever = self._collision_ever or collision

        # ONLY non-guard agents can satisfy the goal now
        goal_reached = any(
            (pid in self._agent_pos) and (self._agent_pos[pid] in self.layout.goal_region)
            for pid in self._normal_agent_ids
        )
        property_satisfied = bool(goal_reached and not self._collision_ever)

        # rewards
        rewards = {a: 0.0 for a in self.agents}
        if goal_reached and self.reward_on_goal:
            # reward only the non-guard agent(s) actually in the goal
            for pid in self._normal_agent_ids:
                if pid in self._agent_pos and self._agent_pos[pid] in self.layout.goal_region:
                    rewards[pid] += self.reward_on_goal
            # Guards failed to prevent goal
            for pid in self._guard_agent_ids:
                rewards[pid] += self.reward_guard_failure

        # collision-shaped rewards
        if collision:
            # group agents by occupied cell to find all collision clusters
            cell_to_agents: dict[Coord, list[str]] = {}
            for pid, pos in self._agent_pos.items():
                cell_to_agents.setdefault(pos, []).append(pid)

            for occupants in cell_to_agents.values():
                if len(occupants) < 2:
                    continue

                guards_here = [pid for pid in occupants if pid in self._guard_agent_ids]
                normals_here = [pid for pid in occupants if pid in self._normal_agent_ids]

                if guards_here and normals_here:
                    # guard vs. normal(s): reward guard(s), penalize each normal
                    for g in guards_here:
                        # scale by number of intruders collided with
                        rewards[g] += self.reward_guard_agent_collision * len(normals_here)
                    for n in normals_here:
                        rewards[n] += self.reward_normal_agent_collision
                else:
                    # collisions among normals only: penalize each normal involved
                    for n in normals_here:
                        rewards[n] += self.reward_normal_agent_collision

        # terminations/truncations
        # terminate ONE step after the event occurred
        env_done = (self.terminates_after_goal and self._goal_last_step) or (
            self.terminates_after_collision and self._collision_last_step
        )
        env_trunc = self._t >= self.max_steps
        terms = {a: bool(env_done) for a in self.agents}
        truncs = {a: bool(env_trunc and not env_done) for a in self.agents}
        infos = {
            a: {
                "t": self._t,
                "collision": collision,
                "goal_reached": goal_reached,
                "property_satisfied": property_satisfied,
            }
            for a in self.agents
        }

        obs = {a: self._obs() for a in self.agents}

        # latch events for the *next* step's termination decision
        self._goal_last_step = bool(goal_reached)
        self._collision_last_step = bool(collision)

        if self.render_mode == "human":
            self.render()
        if env_done or env_trunc:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def render(self):
        if self.render_mode is None:
            return
        if not self._renderer:
            raise ValueError("Renderer missing; render_mode was None at __init__.")
        grid = self._grid_int()
        return self._renderer.render(grid, self._agent_pos)

    def close(self):
        if self._renderer:
            self._renderer.close()

    # ───────────────────────────── Internals ─────────────────────────────
    def _grid_int(self) -> np.ndarray:
        g = np.zeros((self.H, self.W), dtype=np.int32)
        for r, c in self.layout.walls:
            g[r, c] = Tile.Wall
        for r, c in self.layout.goal_region:
            g[r, c] = Tile.Goal if g[r, c] == Tile.Empty else g[r, c]
        return g

    def _global_map(self) -> np.ndarray:
        """
        Returns a (H, W, C) binary tensor (float32) of channels = [Wall, Goal, A0, A1, (A2)].
        """
        H, W, C = self.H, self.W, self.n_obs_types
        base = np.zeros((H, W, C), dtype=np.float32)

        # walls / goals
        for r, c in self.layout.walls:
            base[r, c, 0] = 1
        for r, c in self.layout.goal_region:
            base[r, c, 1] = 1

        # agents
        for i in range(self.n_agents):
            pid = f"player_{i}"
            r, c = self._agent_pos[pid]
            base[r, c, 2 + i] = 1

        return base

    def _obs(self):
        """Non-localised obs shared by all: [Wall, Goal, A0, A1, (A2)], dtype=float32."""
        base = self._global_map()  # (H, W, C), float32 {0,1}
        if self.flatten:
            return base.reshape(-1)  # (H*W*C,)
        return base

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                               Observation helpers                                #
    # ──────────────────────────────────────────────────────────────────────────────── #

    @property
    def state_space(self):
        """Describe the shape/dtype of state()."""
        H, W, C = self.H, self.W, self.n_obs_types
        if self.flatten:
            return Box(low=0, high=1, shape=(H * W * C,), dtype=np.float32)
        return Box(low=0, high=1, shape=(H, W, C), dtype=np.float32)

    def state(self):
        """
        Global, agent-agnostic state (same content as obs), matches state_space().
        """
        s = self._obs()
        return s

    def num_cells(self) -> int:
        """
        Number of spatial cells in the layered observation grid.
        """
        return self.grid_size[0] * self.grid_size[1]

    def channel_names(self) -> list[str]:
        """
        Human-friendly names for each observation channel index.

        Channels are:
          0: Wall
          1: Goal
          2+: Agents (0 = guard, others = normal agents)
        """
        names: list[str] = ["" for _ in range(self.n_obs_types)]

        # Static map layers
        if self.n_obs_types > 0:
            names[0] = "wall"
        if self.n_obs_types > 1:
            names[1] = "goal"

        # Agent position layers
        for i in range(self.n_agents):
            idx = 2 + i
            if idx >= self.n_obs_types:
                break
            if i == 0:
                names[idx] = "guard_0"
            else:
                names[idx] = f"agent_{i}"

        # Fallback for any extra/unnamed channels
        for i, name in enumerate(names):
            if not name:
                names[i] = f"channel_{i}"

        return names

    def action_names(self, action: int) -> str:
        """
        Human-friendly name for an action integer ID.
        """
        try:
            name = Actions(int(action)).name
        except ValueError:
            return f"action_{action}"
        return "".join(
            f"_{ch.lower()}" if ch.isupper() and i > 0 else ch.lower()
            for i, ch in enumerate(name)
        )

    # ────────────────────────────────────────────────────────────────────
    #               Lightweight state serialization (no deep clone)
    # ────────────────────────────────────────────────────────────────────
    def get_state(self):
        """
        Tuple layout:
          (
            agents: tuple[str, ...],
            t: int,
            collision_ever: bool,
            goal_last_step: bool,
            collision_last_step: bool,
            agent_pos: tuple[Coord, ...]     # ordered by possible_agents
            rng_state: object
          )
        """
        agents = tuple(self.agents) if getattr(self, "agents", None) is not None else tuple()
        t = int(getattr(self, "_t", 0))
        collision_ever = bool(getattr(self, "_collision_ever", False))
        goal_last_step = bool(getattr(self, "_goal_last_step", False))
        collision_last_step = bool(getattr(self, "_collision_last_step", False))

        # During normal operation, positions exist for all agents.
        # If called very early, fall back to None (but typical use is post-reset).
        pos = tuple(self._agent_pos.get(a, None) for a in self.possible_agents)

        rng_state = self.get_rng_state()
        return (agents, t, collision_ever, goal_last_step, collision_last_step, pos, rng_state)

    def set_state(self, state):
        (agents, t, collision_ever, goal_last_step, collision_last_step, pos, rng_state) = state

        self.agents = list(agents)
        self._t = int(t)
        self._collision_ever = bool(collision_ever)
        self._goal_last_step = bool(goal_last_step)
        self._collision_last_step = bool(collision_last_step)

        if len(pos) != self.n_agents:
            raise ValueError(f"GuardGridworld.set_state: agent_pos must be len {self.n_agents}, got {len(pos)}")

        # Validate positions (must be in-bounds and not a wall)
        new_pos = {}
        for i, pid in enumerate(self.possible_agents):
            rc = pos[i]
            if rc is None:
                raise ValueError(
                    "GuardGridworld.set_state: agent positions may not be None (expected post-reset state)."
                )
            r, c = map(int, rc)
            if not (0 <= r < self.H and 0 <= c < self.W):
                raise ValueError(f"GuardGridworld.set_state: out-of-bounds position {rc} for {pid}")
            if (r, c) in self.layout.walls:
                raise ValueError(f"GuardGridworld.set_state: position {rc} for {pid} is a wall")
            new_pos[pid] = (r, c)
        self._agent_pos = new_pos

        if rng_state is not None:
            self.set_rng_state(rng_state)

    def get_rng_state(self):
        # np.random.RandomState
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        import numpy as np

        self.rng = np.random.RandomState(int(seed))