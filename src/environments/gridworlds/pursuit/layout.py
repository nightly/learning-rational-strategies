from dataclasses import dataclass
from .common import Coord
from typing import Sequence, Optional

@dataclass
class Layout:
    grid_size: tuple[int, int]          # (H, W)
    walls: set[Coord]                   # cells that are walls
    goal_region: set[Coord]             # cells that are goal
    start_positions: Optional[Sequence[Coord]] = None

def _rect(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    """Inclusive rectangle (r0..r1, c0..c1)."""
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

def default_layout() -> Layout:
    H, W = 9, 9

    # Outer border walls
    border = (
        _rect(0, 0, 0, W - 1)
        | _rect(H - 1, 0, H - 1, W - 1)
        | _rect(0, 0, H - 1, 0)
        | _rect(0, W - 1, H - 1, W - 1)
    )

    # Horizontal barrier directly above the goal band,
    # with TWO gaps (left & right) as in Fig. 3.
    barrier_row = 4
    gap_cols = (2, 6)                   # left/right gaps inside the border
    barrier = {(barrier_row, c) for c in range(1, W - 1) if c not in gap_cols}

    walls = border | barrier

    # Goal region: light-green band, three rows tall
    goal = _rect(5, 1, 7, W - 2)

    # Start positions to match Fig. 3 (paper’s “2” is our guard/player_0)
    # player_0 (G) starts in the green band; players 1 & 2 at the top row.
    start_positions = [
        (5, 4),  # player_0  (G)
        (1, 2),  # player_1
        (1, 6),  # player_2
    ]

    return Layout(grid_size=(H, W), walls=walls, goal_region=goal, start_positions=start_positions)
