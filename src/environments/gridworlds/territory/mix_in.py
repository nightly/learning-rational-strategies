from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from pettingzoo.utils.env import AgentID

from src.environments.gridworlds.native import (
    events_from_infos,
    native_lifecycle,
    unwrap_native_env,
)

from .env import (
    TERRITORY_COLLECTIVE_REWARD_INFO_KEY,
    TERRITORY_READY_TO_SHOOT_INFO_KEY,
)


class TerritoryLabelMixin:
    READY_TO_SHOOT_THRESHOLD = 0.999

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resource_claimed_ever = False
        self._collective_resource_claimed_ever = False
        self._agent_resource_claimed_ever = self._agent_flag_dict()
        self._last_all_zappers_ready = True
        self.last_labels: Dict[str, bool] = {}

    def reset(self, *args, **kwargs):
        self._resource_claimed_ever = False
        self._collective_resource_claimed_ever = False
        self._agent_resource_claimed_ever = self._agent_flag_dict()
        self._last_all_zappers_ready = True
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
            "resource_claimed": False,
            "resource_claimed_ever": False,
            "collective_resource_claimed": False,
            "collective_resource_claimed_ever": False,
            "all_zappers_ready": self._last_all_zappers_ready,
            "any_zapper_cooling_down": not self._last_all_zappers_ready,
        }
        for agent_id in self._agent_ids():
            fluents[f"{agent_id}_resource_claimed"] = False
            fluents[f"{agent_id}_resource_claimed_ever"] = bool(self._agent_resource_claimed_ever.get(agent_id, False))

        native_timeout, native_terminal = native_lifecycle(self.env)
        fluents["timeout"] = native_timeout
        fluents["terminal"] = native_terminal

        if terms:
            fluents["terminal"] = bool(fluents["terminal"] or any(bool(v) for v in terms.values()))
        if truncs:
            fluents["terminal"] = bool(fluents["terminal"] or any(bool(v) for v in truncs.values()))

        fluents["stochastic_termination"] = bool(fluents["terminal"] and not fluents["timeout"])

        del rewards
        agent_ids = self._agent_ids()
        agent_by_player_index = {player_index: agent_id for player_index, agent_id in enumerate(agent_ids, start=1)}
        for event_name, event in events_from_infos(infos):
            if event_name != "claimed_resource":
                continue
            try:
                player_index = int(event["player_index"])
            except (KeyError, TypeError, ValueError):
                continue
            agent_id = agent_by_player_index.get(player_index)
            if agent_id is None:
                continue
            fluents[f"{agent_id}_resource_claimed"] = True
            self._agent_resource_claimed_ever[agent_id] = True

        fluents["resource_claimed"] = any(
            bool(fluents.get(f"{agent_id}_resource_claimed", False)) for agent_id in agent_ids
        )
        self._resource_claimed_ever = self._resource_claimed_ever or bool(fluents["resource_claimed"])

        collective_values: list[float] = []
        ready_values: list[float] = []
        if isinstance(infos, dict) and infos:
            for info in infos.values():
                if not isinstance(info, dict):
                    continue
                if TERRITORY_COLLECTIVE_REWARD_INFO_KEY in info:
                    collective_values.append(self._coerce_scalar(info[TERRITORY_COLLECTIVE_REWARD_INFO_KEY]))
                if TERRITORY_READY_TO_SHOOT_INFO_KEY in info:
                    ready_values.append(self._coerce_scalar(info[TERRITORY_READY_TO_SHOOT_INFO_KEY]))

        if collective_values:
            collective_resource_claimed = any(value > 0.0 for value in collective_values)
            fluents["collective_resource_claimed"] = collective_resource_claimed
            self._collective_resource_claimed_ever = (
                self._collective_resource_claimed_ever or collective_resource_claimed
            )

        if ready_values:
            self._last_all_zappers_ready = all(value >= self.READY_TO_SHOOT_THRESHOLD for value in ready_values)

        fluents["resource_claimed_ever"] = self._resource_claimed_ever
        fluents["collective_resource_claimed_ever"] = self._collective_resource_claimed_ever
        for agent_id in self._agent_ids():
            fluents[f"{agent_id}_resource_claimed_ever"] = bool(self._agent_resource_claimed_ever.get(agent_id, False))
        fluents["all_zappers_ready"] = self._last_all_zappers_ready
        fluents["any_zapper_cooling_down"] = not self._last_all_zappers_ready

        self.last_labels = dict(fluents)
        return fluents

    def _agent_ids(self) -> tuple[str, ...]:
        return tuple(str(agent_id) for agent_id in getattr(self.env, "possible_agents", []))

    def _agent_flag_dict(self) -> Dict[str, bool]:
        return {agent_id: False for agent_id in self._agent_ids()}

    def _coerce_scalar(self, value: Any) -> float:
        array = np.asarray(value)
        if array.shape == ():
            return float(array.item())
        return float(array.reshape(-1)[0])

    def _unwrap_to_native_env(self):
        return unwrap_native_env(getattr(self, "env", None))
