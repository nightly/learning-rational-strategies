from collections import defaultdict
from typing import Any, Dict, Optional

from pettingzoo.utils.env import AgentID


class PELabelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Infer "gap cells" once (same heuristic you used)
        self._gap_cells = self._infer_gap_cells()

    def label(self, infos: Optional[Dict[AgentID, Dict[str, Any]]] = None, **_kwargs) -> Dict[str, bool]:
        fluents: Dict[str, bool] = {
            "goal_reached": False,
            "collision": False,
            "timeout": False,
            "collision_ever": False,
            "guard_intruder_collision": False,
            "intruder_intruder_collision": False,
            "intruder_in_gap": False,
        }

        info_any: Dict[str, Any] = {}
        if infos:
            info_any = next(iter(infos.values()), {}) or {}

        fluents["goal_reached"] = bool(info_any.get("goal_reached", False))
        fluents["collision"] = bool(info_any.get("collision", False))

        t = info_any.get("t", getattr(self.env, "_t", 0))
        max_steps = int(getattr(self.env, "max_steps", 0))
        fluents["timeout"] = int(t) >= max_steps if max_steps > 0 else False

        fluents["collision_ever"] = bool(getattr(self.env, "_collision_ever", False))

        guard_intruder_collision, intruder_intruder_collision = self._collision_types()
        fluents["guard_intruder_collision"] = bool(guard_intruder_collision)
        fluents["intruder_intruder_collision"] = bool(intruder_intruder_collision)

        fluents["intruder_in_gap"] = bool(self._intruder_in_gap())

        return fluents

    # --- helpers copied/adapted from your LabelledPursuitEvasion ---
    def _collision_types(self):
        pos = getattr(self.env, "_agent_pos", {})
        if not pos:
            return (False, False)

        guard_ids = set(getattr(self.env, "_guard_agent_ids", ["player_0"]))
        intruder_ids = set(getattr(self.env, "_normal_agent_ids", []))

        cell_to_agents = defaultdict(list)
        for pid, rc in pos.items():
            cell_to_agents[rc].append(pid)

        guard_intruder_collision = False
        intruder_intruder_collision = False
        for occupants in cell_to_agents.values():
            if len(occupants) < 2:
                continue
            guards_here = any(pid in guard_ids for pid in occupants)
            intruders_here = [pid for pid in occupants if pid in intruder_ids]

            if guards_here and len(intruders_here) >= 1:
                guard_intruder_collision = True
            if (not guards_here) and len(intruders_here) >= 2:
                intruder_intruder_collision = True

        return (guard_intruder_collision, intruder_intruder_collision)

    def _intruder_in_gap(self) -> bool:
        pos = getattr(self.env, "_agent_pos", {})
        if not pos or not self._gap_cells:
            return False
        intruder_ids = getattr(self.env, "_normal_agent_ids", [])
        return any(pos[pid] in self._gap_cells for pid in intruder_ids if pid in pos)

    def _infer_gap_cells(self):
        layout = getattr(self.env, "layout", None)
        if layout is None:
            return set()

        H, W = layout.grid_size
        walls = set(layout.walls)

        best_wall_count = -1
        best_gaps = set()

        for r in range(1, H - 1):
            internal_cols = range(1, W - 1)
            wall_count = sum((r, c) in walls for c in internal_cols)
            gaps = {(r, c) for c in internal_cols if (r, c) not in walls}
            if gaps and wall_count > best_wall_count:
                best_wall_count = wall_count
                best_gaps = gaps

        return best_gaps
