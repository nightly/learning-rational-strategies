from __future__ import annotations

from typing import Any, Dict, Optional

from pettingzoo.utils.env import AgentID

from src.environments.gridworlds.native import events_from_infos, native_lifecycle, unwrap_native_env


class CoinsLabelMixin:
    """Semantic labels for native two-player Coins events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_labels: Dict[str, bool] = {}

    def reset(self, *args, **kwargs):
        self.last_labels = {}
        return super().reset(*args, **kwargs)

    def label(
        self,
        infos: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        *,
        terms: Optional[Dict[AgentID, bool]] = None,
        truncs: Optional[Dict[AgentID, bool]] = None,
        **_kwargs,
    ) -> Dict[str, bool]:
        agent_ids = self._agent_ids()
        fluents: Dict[str, bool] = {
            "timeout": False,
            "terminal": False,
            "stochastic_termination": False,
            "coin_collected": False,
            "matched_coin_collected": False,
            "mismatched_coin_collected": False,
        }
        for agent_id in agent_ids:
            fluents[f"{agent_id}_coin_collected"] = False
            fluents[f"{agent_id}_matched_coin_collected"] = False
            fluents[f"{agent_id}_mismatched_coin_collected"] = False
            fluents[f"{agent_id}_harmed_by_partner"] = False

        native_timeout, native_terminal = native_lifecycle(self.env)
        fluents["timeout"] = native_timeout
        fluents["terminal"] = native_terminal
        if terms:
            fluents["terminal"] = bool(fluents["terminal"] or any(bool(value) for value in terms.values()))
        if truncs:
            any_truncation = any(bool(value) for value in truncs.values())
            fluents["timeout"] = bool(fluents["timeout"] or any_truncation)
            fluents["terminal"] = bool(fluents["terminal"] or any_truncation)
        fluents["stochastic_termination"] = bool(fluents["terminal"] and not fluents["timeout"])

        for name, payload in events_from_infos(infos):
            if name != "coin_consumed" or not isinstance(payload, dict):
                continue
            try:
                player_index = int(payload["player_index"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            collector = f"player_{player_index}"
            if collector not in agent_ids:
                continue
            matched = bool(payload.get("matched", False))
            fluents["coin_collected"] = True
            fluents[f"{collector}_coin_collected"] = True
            if matched:
                fluents["matched_coin_collected"] = True
                fluents[f"{collector}_matched_coin_collected"] = True
            else:
                fluents["mismatched_coin_collected"] = True
                fluents[f"{collector}_mismatched_coin_collected"] = True
                for agent_id in agent_ids:
                    if agent_id != collector:
                        fluents[f"{agent_id}_harmed_by_partner"] = True

        self.last_labels = dict(fluents)
        return fluents

    def _agent_ids(self) -> tuple[str, ...]:
        return tuple(str(agent_id) for agent_id in getattr(self.env, "possible_agents", []))

    def _unwrap_to_native_env(self):
        return unwrap_native_env(getattr(self, "env", None))


__all__ = ["CoinsLabelMixin"]
