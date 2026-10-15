from __future__ import annotations

import numpy as np

from src.environments.matrix.draw import (
    BAD,
    BG,
    NOTGOOD,
    OK,
    _rect,
    draw_bimatrix,
    draw_score_box,
    draw_title_bar,
)


class StagHuntRenderer:
    def __init__(self, render_mode: str, *, cell: int = 120, fps: int = 30):
        if render_mode not in ("human", "rgb_array"):
            raise ValueError("StagHuntRenderer needs 'human' or 'rgb_array'")
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.font.init()
        self.render_mode = render_mode
        self.cell = int(cell)
        self.fps = int(fps)
        self._surface = self._display = self._clock = None

    def _init(self, width: int, height: int) -> None:
        if self.render_mode == "human":
            self._display = self.pg.display.set_mode((width, height))
            self.pg.display.set_caption("Iterated Stag Hunt")
            self._surface = self.pg.Surface((width, height))
            self._clock = self.pg.time.Clock()
        else:
            self._surface = self.pg.Surface((width, height))

    def render(
        self,
        *,
        round_no: int,
        last_actions: tuple[int | None, int | None],
        last_unmirrored: bool,
        cum_rewards: dict[str, float],
        payoffs: dict[str, float],
    ):
        width = self.cell * 7
        height = self.cell * 4
        if self._surface is None:
            self._init(width, height)

        self._surface.fill(BG)

        title = _rect(16, 16, width - 32, 60)
        draw_title_bar(self._surface, title, f"Iterated Stag Hunt — Round {round_no}")

        matrix_rect = _rect(16, 96, self.cell * 4, self.cell * 3)
        hare = payoffs["hare"]
        stag = payoffs["stag"]
        fail = payoffs["fail"]
        bimatrix = (
            ((hare, hare), (hare, fail)),
            ((fail, hare), (stag, stag)),
        )

        def cell_bg(row_index: int, col_index: int, _payoff):
            if row_index != col_index:
                return BAD
            if row_index == 0:
                return NOTGOOD
            return OK

        highlight = None
        if last_actions[0] is not None and last_actions[1] is not None:
            highlight = (int(last_actions[0]), int(last_actions[1]))

        draw_bimatrix(
            self._surface,
            matrix_rect,
            row_actions=("Hare", "Stag"),
            col_actions=("Hare", "Stag"),
            payoffs=bimatrix,
            row_player_label="P0",
            col_player_label="P1",
            cell_bg_fn=cell_bg,
            highlight=highlight,
        )

        score_rect = _rect(matrix_rect.right + 16, matrix_rect.y, width - matrix_rect.right - 32, matrix_rect.height)
        draw_score_box(
            self._surface,
            score_rect,
            float(cum_rewards.get("player_0", 0.0)),
            float(cum_rewards.get("player_1", 0.0)),
        )

        if last_unmirrored:
            from src.environments.matrix.draw import BLACK, draw_badge

            draw_badge(
                self._surface,
                (matrix_rect.centerx, matrix_rect.bottom - 18),
                "RISK!",
                BAD,
                size=22,
                text_color=BLACK,
            )

        if self.render_mode == "human":
            self._display.blit(self._surface, (0, 0))
            self.pg.display.flip()
            if self._clock:
                self._clock.tick(self.fps)
            return None

        data = self.pg.surfarray.array3d(self._surface)
        return np.transpose(data, (1, 0, 2)).copy()

    def close(self) -> None:
        if self.render_mode == "human" and self.pg:
            self.pg.quit()
            self._surface = self._display = self._clock = None
