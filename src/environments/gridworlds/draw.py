# rendering_primitives.py
import pygame

# ──────────────────────────────────────────────────────────────────────────────
# Palette (lean, gray-forward, tweak globally here)
# ──────────────────────────────────────────────────────────────────────────────
WALL_COLOR     = ( 90,  90,  90)   # dark gray
WALKABLE_COLOR = (190, 190, 190)   # light gray
TEXT_COLOR     = (  0,   0,   0)   # black
BORDER_COLOR   = (221, 221, 221)   # subtle border
BORDER_WIDTH   = 2

def _cell_rect(x: int, y: int, tile_w: int, tile_h: int) -> pygame.Rect:
    return pygame.Rect(x * tile_w, y * tile_h, tile_w, tile_h)

def draw_wall_tile(surface: pygame.Surface, x: int, y: int, tile_w: int, tile_h: int) -> None:
    rect = _cell_rect(x, y, tile_w, tile_h)
    pygame.draw.rect(surface, WALL_COLOR, rect)
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=BORDER_WIDTH)

def draw_walkable_tile(surface: pygame.Surface, x: int, y: int, tile_w: int, tile_h: int) -> None:
    rect = _cell_rect(x, y, tile_w, tile_h)
    pygame.draw.rect(surface, WALKABLE_COLOR, rect)
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=BORDER_WIDTH)

def draw_colored_tile(
    surface: pygame.Surface,
    x: int, y: int,
    tile_w: int, tile_h: int,
    fg_color: tuple[int, int, int],
) -> pygame.Rect:
    rect = _cell_rect(x, y, tile_w, tile_h)
    pygame.draw.rect(surface, fg_color, rect)
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=BORDER_WIDTH)
    return rect

def draw_label_tile(
    surface: pygame.Surface,
    x: int, y: int,
    tile_w: int, tile_h: int,
    label: str,
    fg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = TEXT_COLOR,
) -> None:
    rect = draw_colored_tile(surface, x, y, tile_w, tile_h, fg_color)
    font_size = int(min(tile_w, tile_h) * 0.5)
    font = pygame.font.SysFont(None, font_size)
    text = font.render(label, True, text_color)
    text_rect = text.get_rect(center=rect.center)
    surface.blit(text, text_rect)
