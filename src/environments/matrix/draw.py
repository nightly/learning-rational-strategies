from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import pygame

Color = Tuple[int, int, int]
CellPayoff = Tuple[Union[int, float], Union[int, float]]
PayoffMatrix = Sequence[Sequence[CellPayoff]]
CellBGFn = Callable[[int, int, CellPayoff], Optional[Color]]

FONT_SCALE: float = 1.25

BG: Color = (240, 240, 240)
PANEL: Color = (255, 255, 255)
BORDER: Color = (221, 221, 221)
TEXT: Color = (0, 0, 0)
BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
ACCENT: Color = (255, 200, 0)
OK: Color = (210, 255, 210)
BAD: Color = (255, 220, 220)
NOTGOOD: Color = (250, 213, 165)
P1_COLOR: Color = (65, 105, 225)
P2_COLOR: Color = (148, 0, 211)


def _rect(x: int, y: int, w: int, h: int) -> pygame.Rect:
    return pygame.Rect(x, y, w, h)


def fill_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    bg: Color = PANEL,
    border: Color = BORDER,
    width: int = 2,
) -> None:
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, border, rect, width=width)


def draw_text(
    surface: pygame.Surface,
    text: str,
    center: Tuple[int, int],
    *,
    size: int = 24,
    color: Color = TEXT,
    bold: bool = False,
) -> None:
    scaled_size = max(1, int(round(size * FONT_SCALE)))
    font = pygame.font.SysFont(None, scaled_size, bold=bold)
    image = font.render(text, True, color)
    surface.blit(image, image.get_rect(center=center))


def draw_badge(
    surface: pygame.Surface,
    center: Tuple[int, int],
    text: str,
    color: Color,
    *,
    size: int = 20,
    pad: int = 10,
    text_color: Color = WHITE,
) -> None:
    scaled_size = max(1, int(round(size * FONT_SCALE)))
    font = pygame.font.SysFont(None, scaled_size, bold=True)
    image = font.render(text, True, text_color)
    width, height = image.get_size()
    rect = pygame.Rect(0, 0, width + 2 * pad, height + 2 * pad)
    rect.center = center
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, BORDER, rect, width=2, border_radius=10)
    surface.blit(image, image.get_rect(center=rect.center))


def draw_title_bar(surface: pygame.Surface, rect: pygame.Rect, title: str) -> None:
    fill_panel(surface, rect)
    draw_text(surface, title, rect.center, size=28, bold=True)


def draw_score_box(
    surface: pygame.Surface,
    rect: pygame.Rect,
    p1: float,
    p2: float,
    *,
    title: str = "Cumulative reward",
) -> None:
    fill_panel(surface, rect)
    draw_text(surface, title, (rect.centerx, rect.y + 22), size=22, bold=True)
    row_y = rect.y + rect.height // 2
    draw_badge(surface, (rect.x + rect.width * 0.25, row_y), f"P0: {p1:.1f}", P1_COLOR)
    draw_badge(surface, (rect.x + rect.width * 0.75, row_y), f"P1: {p2:.1f}", P2_COLOR)


def _default_cell_formatter(payoff: CellPayoff, _i: int, _j: int) -> str:
    def _fmt(value: Union[int, float]) -> str:
        return f"{int(value)}" if float(value).is_integer() else f"{float(value):.1f}"

    return f"{_fmt(payoff[0])},{_fmt(payoff[1])}"


def draw_bimatrix(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    row_actions: Sequence[str],
    col_actions: Sequence[str],
    payoffs: PayoffMatrix,
    row_player_label: str = "Row",
    col_player_label: str = "Col",
    player_colors: Tuple[Color, Color] = (P1_COLOR, P2_COLOR),
    cell_formatter: Callable[[CellPayoff, int, int], str] = _default_cell_formatter,
    cell_bg_fn: Optional[CellBGFn] = None,
    highlight: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    grid_border_width: int = 2,
) -> None:
    rows, cols = len(row_actions), len(col_actions)
    if rows < 1 or cols < 1:
        raise ValueError("draw_bimatrix needs at least a 1x1 payoff grid.")

    highlights: set[Tuple[int, int]] = set()
    if highlight is not None:
        if isinstance(highlight, tuple):
            highlights.add((int(highlight[0]), int(highlight[1])))
        else:
            highlights = {(int(i), int(j)) for (i, j) in highlight}

    fill_panel(surface, rect)
    pad = 8
    tlx, tly = rect.x + pad, rect.y + pad
    width = rect.width - 2 * pad
    height = rect.height - 2 * pad

    label_height = max(28, int(height * 0.12))
    label_width = max(100, int(width * 0.22))
    cell_width = (width - label_width) // cols
    cell_height = (height - label_height) // rows

    draw_text(
        surface,
        col_player_label,
        (tlx + label_width + (width - label_width) // 2, tly + label_height // 2),
        size=22,
        bold=True,
    )
    draw_text(
        surface,
        row_player_label,
        (tlx + label_width // 2, tly + label_height + (height - label_height) // 2),
        size=22,
        bold=True,
    )

    for col_index, label in enumerate(col_actions):
        center_x = tlx + label_width + col_index * cell_width + cell_width // 2
        draw_badge(surface, (center_x, tly + label_height // 2), label, player_colors[1], size=18)

    for row_index, label in enumerate(row_actions):
        center_y = tly + label_height + row_index * cell_height + cell_height // 2
        draw_badge(surface, (tlx + label_width // 2, center_y), label, player_colors[0], size=18)

    for row_index in range(rows):
        for col_index in range(cols):
            cell_rect = pygame.Rect(
                tlx + label_width + col_index * cell_width,
                tly + label_height + row_index * cell_height,
                cell_width,
                cell_height,
            )
            payoff = payoffs[row_index][col_index]
            bg = PANEL
            if cell_bg_fn is not None:
                maybe_bg = cell_bg_fn(row_index, col_index, payoff)
                if maybe_bg is not None:
                    bg = maybe_bg
            fill_panel(surface, cell_rect, bg=bg, border=BORDER, width=grid_border_width)
            draw_text(surface, cell_formatter(payoff, row_index, col_index), cell_rect.center, size=28, bold=True)
            if (row_index, col_index) in highlights:
                pygame.draw.rect(surface, ACCENT, cell_rect, width=6)
