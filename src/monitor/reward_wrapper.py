import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Union

import graphviz
import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import AgentID, ParallelEnv
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from src.monitor.config import RMConfig
from src.monitor.counterfactual import COUNTERFACTUAL_TRUE_PROPS_KEY
from src.monitor.episode_log import EpisodeLog
from src.monitor.reward_monitor import BoolMonitor
from src.monitor.spec_pair import SpecPair

Labels = Dict[str, bool]
# PerAgentLabels = Dict[AgentID, Labels]
LabelReturn = Union[Labels]  # Assume labels are shared
HandcraftedRewardFn = Callable[..., dict[AgentID, float]]


class BoolRewardWrapper(BaseParallelWrapper):
    """
    PettingZoo Parallel reward wrapper:
      - One BoolMonitor per agent (one LTLf spec per agent).
      - label() is shared (computed once per step).
      - Rewards are monitor-based.
      - Tracks EpisodeLog per agent across episodes.
    """

    # Set once this wrapper ends an episode (its own horizon or any agent done)
    # and cleared on reset, so ``agents`` reports the episode end even when the
    # wrapped env has a longer native horizon.
    _episode_over: bool = False

    def __init__(
        self,
        env,
        spec_rewards: list[SpecPair],
        max_steps_per_episode: int,
        config=RMConfig(),
        *,
        reward_mode: str = "objective",
        handcrafted_reward_fn: Optional[HandcraftedRewardFn] = None,
    ):
        assert isinstance(env, ParallelEnv)
        super().__init__(env)
        self.env = env

        self.agents_to_log = self.env.possible_agents
        self.max_steps_per_episode = max_steps_per_episode
        self.cache = {}  # Cache return results
        self.config = config
        self.reward_mode = self._coerce_reward_mode(reward_mode)
        self.handcrafted_reward_fn = handcrafted_reward_fn

        # Global bookkeeping
        self.n_episodes: int = 0
        self.n_timesteps: int = 0
        self.current_step: int = 0

        # Spec handling (exactly one spec selected per agent)
        self.spec_by_agent: Dict[AgentID, SpecPair] = {}
        for s in spec_rewards:
            if s.agent_id is None:
                raise ValueError("One spec per agent required: SpecPair.agent_id cannot be None.")
            if s.agent_id in self.spec_by_agent:
                raise ValueError(f"Duplicate spec for agent_id='{s.agent_id}'.")
            self.spec_by_agent[s.agent_id] = s

        missing = [a for a in self.agents_to_log if a not in self.spec_by_agent]
        if missing:
            raise ValueError(f"Missing SpecPair for agents: {missing}")

        self.rms_by_agent: Dict[AgentID, BoolMonitor] = {
            a: BoolMonitor(self.spec_by_agent[a].formula_str, self.spec_by_agent[a].reward_scalar)
            for a in self.agents_to_log
        }

        # EpisodeLog storage
        self.history_by_agent: dict[AgentID, list[EpisodeLog]] = defaultdict(list)
        self.last_labels: Labels = {}

        if self.config.temporally_extended:
            self._base_obs_space = {a: self.env.observation_space(a) for a in self.env.possible_agents}
            self._aug_obs_space = {}

            for a, base in self._base_obs_space.items():
                assert isinstance(base, gym.spaces.Box), "This augmentation assumes Box observations."
                n = self.rms_by_agent[a].n_states  # >= 2
                low_extra = np.array([0.0], dtype=np.float32)
                high_extra = np.array([float(n - 1)], dtype=np.float32)

                low = np.concatenate([base.low.astype(np.float32).reshape(-1), low_extra])
                high = np.concatenate([base.high.astype(np.float32).reshape(-1), high_extra])

                self._aug_obs_space[a] = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Per-episode counters
        self._reset_episode_counters()

    def label(self, infos: Optional[Dict[AgentID, Dict[str, Any]]] = None, **kwargs) -> LabelReturn:
        """
        Override in subclass. Should return dict[str,bool]  (shared labels, applied to every agent).
        Possibly extensible to agent-specific labels, with type: dict[agent_id, dict[str,bool]]
        """
        raise NotImplementedError

    @property
    def agents(self) -> list[AgentID]:
        """Active agents; empty after this wrapper has ended the episode, until reset."""
        if self._episode_over:
            return []
        return self.env.agents

    def monitor_numbering_fingerprints(self) -> dict[AgentID, dict[str, Any]]:
        return {agent: monitor.numbering_fingerprint() for agent, monitor in self.rms_by_agent.items()}

    def set_reward_mode(self, reward_mode: str) -> None:
        self.reward_mode = self._coerce_reward_mode(reward_mode)

    def _coerce_reward_mode(self, reward_mode: str) -> str:
        mode = str(reward_mode).lower()
        if mode not in {"objective", "handcrafted"}:
            raise ValueError("reward_mode must be one of {'objective', 'handcrafted'}.")
        return mode

    def _handcrafted_rewards(
        self,
        *,
        native_rewards: Dict[AgentID, float],
        labels: Labels,
        infos: Dict[AgentID, Dict[str, Any]],
        terms: Dict[AgentID, bool],
        truncs: Dict[AgentID, bool],
        done: bool,
        current_agents: Set[AgentID],
    ) -> Dict[AgentID, float]:
        if self.handcrafted_reward_fn is None:
            return {agent_id: float(native_rewards.get(agent_id, 0.0)) for agent_id in current_agents}

        rewards = self.handcrafted_reward_fn(
            self,
            native_rewards=dict(native_rewards),
            labels=dict(labels),
            infos=infos,
            terms=terms,
            truncs=truncs,
            done=done,
            current_agents=set(current_agents),
        )
        return {agent_id: float(rewards.get(agent_id, 0.0)) for agent_id in current_agents}

    # ----------------------------------------------------

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)

        self.n_timesteps += 1
        self.current_step += 1
        native_rewards = dict(rews) if rews is not None else {}
        infos = dict(infos) if infos is not None else {}
        terms = dict(terms) if terms is not None else {}
        truncs = dict(truncs) if truncs is not None else {}
        env_agents = getattr(self.env, "agents", None)
        current_agents = set(env_agents) if isinstance(env_agents, (list, tuple)) and env_agents else set(obs.keys())

        # 1) If max steps reached, force truncation for all active agents
        if self.max_steps_per_episode > 0 and self.current_step >= self.max_steps_per_episode:
            truncs = dict(truncs) if truncs is not None else {}
            for a in current_agents:
                truncs[a] = True

        # 2) Compute labels once, then apply per-agent
        labels_ret: LabelReturn = self.label(
            infos,
            rewards=native_rewards,
            observations=obs,
            terms=terms,
            truncs=truncs,
        )
        self.last_labels = dict(labels_ret)
        true_props = tuple(sorted(key for key, value in labels_ret.items() if bool(value)))

        # 3) Bookkeeping
        done = self._episode_done(current_agents, terms, truncs)
        self._episode_over = bool(done)
        # labels_ret["terminal"] = done

        # 4) Step monitors and assign objective rewards
        objective_rewards: Dict[AgentID, float] = {}
        for a in current_agents:
            rm = self.rms_by_agent[a]
            rm.step(labels_ret)

            # Terminal objective reward maximises probabilistic satisfaction.
            if done:
                r = 1.0 if rm.current_state in rm.automaton.accepting_states else 0.0
            else:
                r = 0.0
            objective_rewards[a] = r

        if self.reward_mode == "handcrafted":
            learner_rewards = self._handcrafted_rewards(
                native_rewards=native_rewards,
                labels=dict(labels_ret),
                infos=infos,
                terms=terms,
                truncs=truncs,
                done=done,
                current_agents=current_agents,
            )
        else:
            learner_rewards = dict(objective_rewards)

        # 5) Assign learner rewards and keep objective bookkeeping stable.
        rews: Dict[AgentID, float] = {}
        for a in current_agents:
            objective_reward = float(objective_rewards.get(a, 0.0))
            learner_reward = float(learner_rewards.get(a, 0.0))
            rews[a] = learner_reward
            self._cum_reward[a] += objective_reward
            self._learner_cum_reward[a] += learner_reward
            if self.reward_mode == "handcrafted":
                self._handcrafted_cum_reward[a] += learner_reward
            self._ep_len[a] += 1
            info = dict(infos.get(a, {}))
            info[COUNTERFACTUAL_TRUE_PROPS_KEY] = true_props
            info["objective_reward"] = objective_reward
            info["learner_reward"] = learner_reward
            if self.reward_mode == "handcrafted":
                info["handcrafted_reward"] = learner_reward
            infos[a] = info

        # 6) Bookkeeping on done
        if done:
            self.n_episodes += 1
            for a in self.agents_to_log:
                log = EpisodeLog(
                    episode=self.n_episodes,
                    t_end=self.n_timesteps,
                    ep_len=int(self._ep_len.get(a, 0)),
                    cum_reward=float(self._cum_reward.get(a, 0.0)),
                    start_step=self.n_timesteps - int(self._ep_len.get(a, 0)) + 1,
                    objective_cum_reward=float(self._cum_reward.get(a, 0.0)),
                    learner_cum_reward=float(self._learner_cum_reward.get(a, 0.0)),
                    handcrafted_cum_reward=(
                        float(self._handcrafted_cum_reward.get(a, 0.0))
                        if self.reward_mode == "handcrafted"
                        else None
                    ),
                )
                self.history_by_agent[a].append(log)
            # Counters reset on next reset()

        if self.config.temporally_extended:
            obs = {a: self._augment_obs(a, o) for a, o in obs.items()}

        return obs, rews, terms, truncs, infos

    def reset(self, *args, **kwargs):
        obs, infos = self.env.reset(*args, **kwargs)
        self._episode_over = False
        self.current_step = 0
        self.last_labels = {}
        for rm in self.rms_by_agent.values():
            rm.reset()
        reset_handcrafted = getattr(self.handcrafted_reward_fn, "reset", None)
        if callable(reset_handcrafted):
            reset_handcrafted()

        self._reset_episode_counters(initial_agents=set(obs.keys()))

        if self.config.temporally_extended:
            # attach initial monitor state
            obs = {a: self._augment_obs(a, o) for a, o in obs.items()}

        return obs, infos

    # ---------------------- Miscellaneous Helpers ----------------------
    def _reset_episode_counters(self, initial_agents: Optional[Set[AgentID]] = None):
        self._ep_len: defaultdict[AgentID, int] = defaultdict(int)
        self._cum_reward: defaultdict[AgentID, float] = defaultdict(float)
        self._learner_cum_reward: defaultdict[AgentID, float] = defaultdict(float)
        self._handcrafted_cum_reward: defaultdict[AgentID, float] = defaultdict(float)

    def _episode_done(
        self,
        current_agents: Set[AgentID],
        terms: Dict[AgentID, bool],
        truncs: Dict[AgentID, bool],
    ) -> bool:
        if not current_agents:
            return True  # many envs return empty dicts on terminal step
        dones = [bool(terms.get(a, False) or truncs.get(a, False)) for a in current_agents]
        return any(dones)

    # convenience
    def last(self, agent: Optional[AgentID] = None) -> Optional[EpisodeLog]:
        if agent is None:
            return None
        h = self.history_by_agent.get(agent, [])
        return h[-1] if h else None

    def to_dicts(self) -> Dict[AgentID, list[dict[str, Any]]]:
        return {a: [e.__dict__ for e in logs] for a, logs in self.history_by_agent.items()}

    def history_reset(self, *reset_args, **reset_kwargs):
        self.reset(*reset_args, **reset_kwargs)
        self.n_timesteps = 0
        self.n_episodes = 0
        self.current_step = 0
        self.history_by_agent = defaultdict(list)
        self.history = []

    # --- augment: expose spaces (PettingZoo calls observation_space(agent)) ---
    def observation_space(self, agent):
        if self.config.temporally_extended:
            return self._aug_obs_space[agent]
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def supports_counterfactual_replay(self) -> bool:
        return bool(self.config.temporally_extended)

    def counterfactual_monitor(self, agent: AgentID) -> BoolMonitor:
        return self.rms_by_agent[str(agent)]

    # --- helper to wrap observations ---
    def _augment_obs(self, agent, obs):
        if not self.config.temporally_extended:
            return obs
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        q = float(self.rms_by_agent[agent].state_id())
        return np.concatenate([o, np.array([q], dtype=np.float32)])

    # ---------------------- Export ----------------------
    def export(
        self,
        to_file: bool = True,
        dir: str = "exports",
        filename: str = "bool_rm.dot",
    ) -> Dict[AgentID, graphviz.Digraph]:
        """
        Export one automaton per agent monitor.

        - If filename contains "{agent}", it will be formatted with the agent id.
        Example: filename="rm_{agent}.dot"
        - Otherwise, we append the agent id before the extension:
        "bool_rm.dot" -> "bool_rm_player_0.dot"
        - If filename includes subdirectories (e.g. "graphs/rm.dot"), they are created too.
        """
        exports: Dict[AgentID, graphviz.Digraph] = {}

        export_dir = Path(dir).expanduser() if dir else Path(".")
        fname_path = Path(filename)

        # If filename includes a subdir, fold it into export_dir
        if fname_path.parent != Path("."):
            export_dir = export_dir / fname_path.parent
            filename = fname_path.name  # keep only the final filename part

        if to_file:
            export_dir.mkdir(parents=True, exist_ok=True)

        base = Path(filename)
        stem, suffix = base.stem, base.suffix  # suffix includes the dot, e.g. ".dot" or ""

        for agent, rm in self.rms_by_agent.items():
            # Make agent safe for filenames
            agent_token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(agent))

            if "{agent}" in filename:
                out_name = filename.format(agent=agent_token)
            else:
                out_name = f"{stem}_{agent_token}{suffix}" if suffix else f"{stem}_{agent_token}"

            exports[agent] = rm.export(
                to_file=to_file,
                dir=str(export_dir),
                filename=out_name,
            )  # ty:ignore[invalid-assignment]

        return exports
