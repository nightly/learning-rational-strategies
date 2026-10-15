"""Upstream Melting Pot stamina bar, shared by the native RGB renderers.

Fruit Market, Factory Commons, and Predator Prey all build the same overlay in
``_create_stamina_overlay`` (``refs/meltingpot/meltingpot/configs/substrates/
{fruit_market,factory_commons,predator_prey}.py``) and drive it with the
``StaminaBar`` component (``refs/meltingpot/meltingpot/lua/levels/stamina/
shared_components.lua``):

* ``StaminaBar:setBarState`` puts the overlay in state ``level_<stamina>``,
  where ``<stamina>`` is the avatar's raw integer stamina, after every other
  updater has run. The sprite for a level depends only on that integer, not on
  the ``Stamina`` band amounts.
* Level ``i`` is an 8x8 sprite whose seven top rows are transparent and whose
  bottom row is ``x + G * max(i - 6, 0) + Y * (i if i < 7 else 12 - i) +
  R * max(6 - i, 0) + x``: six coloured pixels between two transparent ones.
  Levels ``i >= 13`` are fully transparent, so the bar disappears near full
  stamina (the ``invisible`` band of ``amountInvisible = 6`` values).
* The overlay sits on a per-player ``superOverlay_<i>`` layer at the end of the
  render order, one cell from the avatar. Of the four copies (N/E/S/W) only the
  one whose direction equals the viewer's orientation is drawn, so in every
  view -- the world view and every player's egocentric view, the owner's own
  included -- the bar is on the bottom row of the cell directly *above* the
  avatar on screen, painted over whatever occupies that cell. A dead avatar's
  bar switches to its wait state and is not drawn.

The native renderers reproduce that: after painting the tiles, they call
``paint_stamina_bar`` for every live avatar on the tile above it.
"""

from __future__ import annotations

import numpy as np

STAMINA_BAR_GREEN = np.array((62, 137, 72), dtype=np.uint8)
STAMINA_BAR_YELLOW = np.array((255, 216, 97), dtype=np.uint8)
STAMINA_BAR_RED = np.array((162, 38, 51), dtype=np.uint8)

#: Stamina values at or above this draw no bar (upstream ``if i >= 13``).
STAMINA_BAR_HIDDEN_FROM = 13
#: Coloured pixels in the bar (the 8-pixel row minus a transparent pixel at each end).
STAMINA_BAR_SEGMENTS = 6


def stamina_bar_level(stamina: int) -> int | None:
    """Return the visible upstream bar level for ``stamina``, or ``None``.

    The level is the raw stamina clamped at zero; ``None`` means the bar is not
    drawn (``stamina >= 13``). Levels ``0..12`` are thirteen distinct sprites.
    """
    level = int(stamina)
    if level >= STAMINA_BAR_HIDDEN_FROM:
        return None
    return max(level, 0)


def stamina_bar_colors(stamina: int) -> tuple[np.ndarray, ...] | None:
    """Return the six bar colours, left to right, or ``None`` when hidden.

    Level ``0..6``: ``level`` yellow then ``6 - level`` red. Level ``7..12``:
    ``level - 6`` green then ``12 - level`` yellow.
    """
    level = stamina_bar_level(stamina)
    if level is None:
        return None
    greens = max(level - 6, 0)
    yellows = level if level < 7 else 12 - level
    reds = max(6 - level, 0)
    return (STAMINA_BAR_GREEN,) * greens + (STAMINA_BAR_YELLOW,) * yellows + (STAMINA_BAR_RED,) * reds


def paint_stamina_bar(
    frame: np.ndarray,
    tile_x: int,
    tile_y: int,
    stamina: int,
    size: int,
) -> None:
    """Paint the bar for ``stamina`` on the bottom row of tile ``(tile_x, tile_y)``.

    Callers pass the tile directly above the avatar on screen. Pixel ``k`` of
    the upstream 8-pixel row maps to columns ``[k * size // 8, (k + 1) * size
    // 8)`` of the tile, so the default 8-pixel tiles match upstream exactly.
    """
    colors = stamina_bar_colors(stamina)
    if colors is None or tile_y < 0 or tile_x < 0:
        return
    y0 = tile_y * size
    x0 = tile_x * size
    if y0 + size > frame.shape[0] or x0 + size > frame.shape[1]:
        return
    height = max(1, size // 8)
    rows = slice(y0 + size - height, y0 + size)
    for segment, color in enumerate(colors, start=1):
        left = x0 + segment * size // 8
        right = x0 + (segment + 1) * size // 8
        frame[rows, left:right] = color


def paint_stamina_bars(
    frame: np.ndarray,
    placements: list[tuple[int, int, int]],
    size: int,
) -> None:
    """Paint bars for ``(avatar_tile_x, avatar_tile_y, stamina)`` placements.

    Each bar goes on the tile above its avatar; avatars in the top row of the
    frame have their bar outside the frame, as upstream.
    """
    for tile_x, tile_y, stamina in placements:
        paint_stamina_bar(frame, tile_x, tile_y - 1, stamina, size)


__all__ = [
    "STAMINA_BAR_GREEN",
    "STAMINA_BAR_HIDDEN_FROM",
    "STAMINA_BAR_RED",
    "STAMINA_BAR_SEGMENTS",
    "STAMINA_BAR_YELLOW",
    "paint_stamina_bar",
    "paint_stamina_bars",
    "stamina_bar_colors",
    "stamina_bar_level",
]
