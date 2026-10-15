from __future__ import annotations
from collections import defaultdict
import numpy as np

from .common import Tile
from src.environments.gridworlds.draw import (
    draw_walkable_tile,
    draw_colored_tile,
    draw_label_tile,
)

# palette (match style used across envs)
GREY   = (80, 80, 80)        # walls
GOAL   = (210, 255, 210)     # goal band
BLUE   = (65, 105, 225)      # guard (player_0)
ORANGE = (255, 165, 0)       # player_1
PURPLE = (148, 0, 211)       # player_2
GOLD   = (255, 200, 0)       # collision overlay

PLAYER_STYLE = {
    "player_0": {"colour": BLUE,   "label": "G"},  # Guard
    "player_1": {"colour": ORANGE, "label": "1"},
    "player_2": {"colour": PURPLE, "label": "2"},
}

BASE_TILE_COLORS = {
    int(Tile.Empty): None,     # walkable
    int(Tile.Wall):  GREY,     # solid
    int(Tile.Goal):  GOAL,     # soft green
}

class GuardGridworldRenderer:
    def __init__(self, render_mode: str, tile_size: int | tuple[int,int] = 80, fps: int = 10):
        if render_mode not in ("human", "rgb_array"):
            raise ValueError("GuardGridworldRenderer needs 'human' or 'rgb_array'")
        import pygame  # lazy import
        self.pg = pygame
        pygame.init(); pygame.font.init()
        self.render_mode = render_mode
        self.tile_w, self.tile_h = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
        self.fps = fps
        self._surface = self._display = self._clock = None

    def _init_surface(self, w: int, h: int):
        if self.render_mode == "human":
            self._display = self.pg.display.set_mode((w, h))
            self.pg.display.set_caption("Guard Gridworld")
            self._surface = self.pg.Surface((w, h))
            self._clock = self.pg.time.Clock()
        else:
            self._surface = self.pg.Surface((w, h))

    def render(self, grid: np.ndarray, agent_pos: dict[str, tuple[int,int]]):
        rows, cols = grid.shape
        W, H = cols * self.tile_w, rows * self.tile_h
        if self._surface is None:
            self._init_surface(W, H)

        # base map
        for r in range(rows):
            for c in range(cols):
                tile_id = int(grid[r, c])
                color = BASE_TILE_COLORS.get(tile_id)
                if color is None:
                    draw_walkable_tile(self._surface, c, r, self.tile_w, self.tile_h)
                else:
                    draw_colored_tile(self._surface, c, r, self.tile_w, self.tile_h, color)

        # agent overlays
        cells = defaultdict(list)
        for pid, rc in agent_pos.items():
            cells[rc].append(pid)

        for (r, c), pids in cells.items():
            if len(pids) == 1:
                pid = pids[0]
                style = PLAYER_STYLE.get(pid, {"colour": GOLD, "label": "?"})
                draw_label_tile(self._surface, c, r, self.tile_w, self.tile_h, style["label"], style["colour"])
            else:
                # collision (two+ in a cell)
                draw_label_tile(self._surface, c, r, self.tile_w, self.tile_h, "C", GOLD)

        if self.render_mode == "human":
            self._display.blit(self._surface, (0, 0))
            self.pg.display.flip()
            if self._clock: self._clock.tick(self.fps)
            return None

        data = self.pg.surfarray.array3d(self._surface)
        return np.transpose(data, (1, 0, 2)).copy()

    def close(self):
        if self.render_mode == "human" and self.pg:
            self.pg.quit()
            self._surface = self._display = self._clock = None
