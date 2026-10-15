from __future__ import annotations

from typing import Any, Dict, Optional

from pettingzoo.utils.env import AgentID

from src.environments.gridworlds.native import (
    events_from_infos,
    native_lifecycle,
    unwrap_native_env,
)


class HarvestLabelMixin:
    """
    Reward-derived labels for the native Harvest Partnership substrate.

    Notes:
      - Labels combine rewards, native infos, and wrapper state.
      - Apple collection yields a positive reward (+1 in the substrate config).
      - Punishment tiles yield a negative reward (-10 in the substrate config).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apple_collected_ever = False
        self._punishment_received_ever = False
        self._zapped_ever = False
        self._agent_apple_collected_ever = self._agent_flag_dict()
        self._agent_punishment_received_ever = self._agent_flag_dict()
        self._agent_zapped_ever = self._agent_flag_dict()
        self.last_labels: Dict[str, bool] = {}

    def reset(self, *args, **kwargs):
        self._apple_collected_ever = False
        self._punishment_received_ever = False
        self._zapped_ever = False
        self._agent_apple_collected_ever = self._agent_flag_dict()
        self._agent_punishment_received_ever = self._agent_flag_dict()
        self._agent_zapped_ever = self._agent_flag_dict()
        self.last_labels = {}
        return super().reset(*args, **kwargs)

    def label(
        self,
        infos: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        *,
        rewards: Optional[Dict[AgentID, float]] = None,
        terms: Optional[Dict[AgentID, bool]] = None,
        truncs: Optional[Dict[AgentID, bool]] = None,
        **_kwargs,
    ) -> Dict[str, bool]:
        fluents: Dict[str, bool] = {
            "timeout": False,
            "terminal": False,
            "stochastic_termination": False,
            "any_harvest_reward": False,
            "apple_collected": False,
            "punishment_received": False,
            "apple_collected_ever": False,
            "punishment_received_ever": False,
            "zapped": False,
            "zapped_ever": False,
        }
        for agent_id in self._agent_ids():
            fluents[f"{agent_id}_apple_collected"] = False
            fluents[f"{agent_id}_punishment_received"] = False
            fluents[f"{agent_id}_apple_collected_ever"] = bool(self._agent_apple_collected_ever.get(agent_id, False))
            fluents[f"{agent_id}_punishment_received_ever"] = bool(
                self._agent_punishment_received_ever.get(agent_id, False)
            )
            fluents[f"{agent_id}_zapped"] = False
            fluents[f"{agent_id}_zapped_ever"] = bool(self._agent_zapped_ever.get(agent_id, False))

        native_timeout, native_terminal = native_lifecycle(self.env)
        fluents["timeout"] = native_timeout
        fluents["terminal"] = native_terminal

        if terms:
            fluents["terminal"] = bool(fluents["terminal"] or any(bool(v) for v in terms.values()))
        if truncs:
            fluents["terminal"] = bool(fluents["terminal"] or any(bool(v) for v in truncs.values()))

        fluents["stochastic_termination"] = bool(fluents["terminal"] and not fluents["timeout"])

        if isinstance(rewards, dict) and rewards:
            values = [float(reward) for reward in rewards.values()]
            fluents["any_harvest_reward"] = any(value != 0.0 for value in values)

            for agent_id, reward in rewards.items():
                value = float(reward)
                apple_collected = value > 0.0
                punishment_received = value < 0.0
                fluents[f"{agent_id}_apple_collected"] = apple_collected
                fluents[f"{agent_id}_punishment_received"] = punishment_received
                self._agent_apple_collected_ever[agent_id] = (
                    bool(self._agent_apple_collected_ever.get(agent_id, False)) or apple_collected
                )
                self._agent_punishment_received_ever[agent_id] = (
                    bool(self._agent_punishment_received_ever.get(agent_id, False)) or punishment_received
                )

            fluents["apple_collected"] = any(
                bool(fluents.get(f"{agent_id}_apple_collected", False)) for agent_id in self._agent_ids()
            )
            fluents["punishment_received"] = any(
                bool(fluents.get(f"{agent_id}_punishment_received", False)) for agent_id in self._agent_ids()
            )
            self._apple_collected_ever = self._apple_collected_ever or bool(fluents["apple_collected"])
            self._punishment_received_ever = self._punishment_received_ever or bool(fluents["punishment_received"])

        agent_ids = self._agent_ids()
        agent_by_player_index = {player_index: agent_id for player_index, agent_id in enumerate(agent_ids, start=1)}
        for event_name, event in events_from_infos(infos):
            if event_name != "zap":
                continue
            try:
                target_index = int(event["target"])
            except (KeyError, TypeError, ValueError):
                continue
            agent_id = agent_by_player_index.get(target_index)
            if agent_id is None:
                continue
            fluents[f"{agent_id}_zapped"] = True
            self._agent_zapped_ever[agent_id] = True

        fluents["zapped"] = any(bool(fluents.get(f"{agent_id}_zapped", False)) for agent_id in agent_ids)
        self._zapped_ever = self._zapped_ever or bool(fluents["zapped"])

        fluents["apple_collected_ever"] = self._apple_collected_ever
        fluents["punishment_received_ever"] = self._punishment_received_ever
        fluents["zapped_ever"] = self._zapped_ever
        for agent_id in self._agent_ids():
            fluents[f"{agent_id}_apple_collected_ever"] = bool(self._agent_apple_collected_ever.get(agent_id, False))
            fluents[f"{agent_id}_punishment_received_ever"] = bool(
                self._agent_punishment_received_ever.get(agent_id, False)
            )
            fluents[f"{agent_id}_zapped_ever"] = bool(self._agent_zapped_ever.get(agent_id, False))

        self.last_labels = dict(fluents)
        return fluents

    def _agent_ids(self) -> tuple[str, ...]:
        return tuple(str(agent_id) for agent_id in getattr(self.env, "possible_agents", []))

    def _agent_flag_dict(self) -> Dict[str, bool]:
        return {agent_id: False for agent_id in self._agent_ids()}

    def _unwrap_to_native_env(self):
        return unwrap_native_env(getattr(self, "env", None))
