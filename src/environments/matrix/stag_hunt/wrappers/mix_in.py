from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from pettingzoo.utils.env import AgentID

from ..common import Actions


def shared_stag_hunt_labels(
    last_actions: tuple[int | None, int | None] | None = None,
    info: Mapping[str, Any] | None = None,
) -> Dict[str, bool]:
    info_any = dict(info or {})

    player_0_action = None
    player_1_action = None
    if last_actions is not None:
        player_0_action, player_1_action = last_actions

    both_hare = bool(
        info_any.get(
            "both_hare",
            player_0_action == Actions.Hare and player_1_action == Actions.Hare,
        )
    )
    both_stag = bool(
        info_any.get(
            "both_stag",
            player_0_action == Actions.Stag and player_1_action == Actions.Stag,
        )
    )
    unmirrored = bool(
        info_any.get(
            "unmirrored",
            player_0_action is not None and player_1_action is not None and player_0_action != player_1_action,
        )
    )

    return {
        "both_hare": both_hare,
        "both_stag": both_stag,
        "unmirrored": unmirrored,
        "mirrored": bool(both_hare or both_stag),
        "player_0_hare": player_0_action == Actions.Hare,
        "player_0_stag": player_0_action == Actions.Stag,
        "player_1_hare": player_1_action == Actions.Hare,
        "player_1_stag": player_1_action == Actions.Stag,
    }


class SHLabelMixin:
    def label(self, infos: Optional[Dict[AgentID, Dict[str, Any]]] = None, **_kwargs) -> Dict[str, bool]:
        info_any: Dict[str, Any] = {}
        if infos:
            info_any = next(iter(infos.values()), {}) or {}

        player_0_action = self.env._last_actions.get("player_0")  # noqa: SLF001
        player_1_action = self.env._last_actions.get("player_1")  # noqa: SLF001

        return shared_stag_hunt_labels((player_0_action, player_1_action), info_any)
