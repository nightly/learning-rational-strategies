"""Agent-centred view windows, including the focal whole-map view.

Every spatial substrate supports three observation modes:

* ``"local"`` -- upstream's egocentric window (11x11 for most substrates);
* ``"global"`` -- the whole map in world coordinates, the same for everyone;
* ``"focal"`` -- egocentric like ``"local"``, but a square window centred on the
  agent and large enough to cover the whole map from any position in any
  orientation: full observability in the agent's own frame.

``"focal"`` exists because full information alone is not enough for learners
that are MLPs over a flattened grid: in a world-frame view an object in front of
the agent lands on different inputs at every position, so the relation has to
be learned again everywhere. A focal view keeps the agent at a fixed cell, as
``"local"`` does, while hiding nothing. The cost is size: a map of R x C cells
needs a window of side 2 * max(R, C) - 1.
"""

from __future__ import annotations

from typing import Any

FOCAL = "focal"


def is_focal(config: Any) -> bool:
    return getattr(config, "observation_mode", "local") == FOCAL


def focal_reach(config: Any) -> int:
    """Cells the focal window extends from the agent in every direction."""

    return max(int(config.world_rows), int(config.world_cols)) - 1


def agent_window(config: Any, rows: int, cols: int, left: int, forward: int) -> tuple[int, int, int, int]:
    """``(rows, cols, left, forward)`` of the agent-centred window.

    The arguments are the substrate's own local window -- its size, how many
    cells to the agent's left, and how many in front. In focal mode the window is
    a square centred on the agent that covers the whole map.
    """

    if is_focal(config):
        reach = focal_reach(config)
        side = 2 * reach + 1
        return side, side, reach, reach
    return rows, cols, left, forward


__all__ = ["FOCAL", "agent_window", "focal_reach", "is_focal"]
