from enum import IntEnum

class Actions(IntEnum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
    Stay = 4

class Tile(IntEnum):
    Empty = 0
    Wall = 1
    Goal = 2

Coord = tuple[int, int]
