"""Compact semantic observations used when rendering is disabled.

The native substrates intentionally keep their simulation state independent of
pygame and pixels.  This module turns that state into fixed-size categorical
grids without calling any of the RGB rendering functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

from ._stamina_bar import stamina_bar_level
from ._views import is_focal

Position = tuple[int, int]
GridIds = Mapping[str, Mapping[str, int]]

ORIENTATION_IDS = {"north": 1, "east": 2, "south": 3, "west": 4}
# Upstream's stamina bar (see _stamina_bar): every live avatar's raw stamina
# 0..12, visible to all; hidden above. The red band drops an avatar out of
# Predator-Prey's defending group, and low stamina freezes movers everywhere.
_STAMINA_IDS = {f"level_{level}": level + 1 for level in range(13)}
# Coins colours are named as the environment names them; each episode draws two.
COIN_IDS = {
    "coin_red": 1,
    "coin_blue": 2,
    "coin_yellow": 3,
    "coin_green": 4,
    "coin_purple": 5,
}

_IMPLEMENTATION_CLASSES = {
    "AllelopathicHarvestOpenParallelEnv",
    "BoatRaceEightRacesParallelEnv",
    "ChemistryParallelEnv",
    "CleanUpParallelEnv",
    "CoinsParallelEnv",
    "CollaborativeCookingParallelEnv",
    "CommonsHarvestParallelEnv",
    "CoopMiningParallelEnv",
    "DaycareParallelEnv",
    "ExternalityMushroomsDenseParallelEnv",
    "FactoryCommonsEitherOrParallelEnv",
    "FruitMarketConcentricRiversParallelEnv",
    "GiftRefinementsParallelEnv",
    "HiddenAgendaParallelEnv",
    "PaintballParallelEnv",
    "PredatorPreyParallelEnv",
    "TerritoryParallelEnv",
    "MatrixGameParallelEnv",
}


def _kind(env: Any) -> str:
    for cls in type(env).__mro__:
        if cls.__name__ in _IMPLEMENTATION_CLASSES:
            return cls.__name__
    raise TypeError(f"unsupported structured-observation environment: {type(env)!r}")


def object_observation_ids(env: Any) -> GridIds:
    """Return immutable categorical ID metadata for an environment instance."""
    ids = _ids_for(env)
    ids["AVATARS"] = {agent: index + 1 for index, agent in enumerate(env.possible_agents)}
    ids["AVATAR_ORIENTATIONS"] = ORIENTATION_IDS
    if env.config.observation_mode == "global":
        # A global view is the same for every agent; this marks whose it is.
        # (A local view is centred on its owner, so needs no marker.)
        ids["SELF"] = {"self": 1}
    return MappingProxyType({key: MappingProxyType(dict(values)) for key, values in ids.items()})


def object_observation_spaces(env: Any) -> dict[str, gym.Space]:
    """Build the categorical grid spaces for ``env``."""
    shape = _grid_shape(env)
    result: dict[str, gym.Space] = {}
    for key, values in object_observation_ids(env).items():
        high = max(values.values(), default=0)
        dtype = np.uint8 if high <= np.iinfo(np.uint8).max else np.uint16
        result[key] = gym.spaces.Box(0, high, shape=shape, dtype=dtype)
    result.update(_scalar_spaces(env))
    return result


def _scalar_spaces(env: Any) -> dict[str, gym.Space]:
    name = _kind(env)

    def unit():
        return gym.spaces.Box(0.0, 1.0, shape=(), dtype=np.float64)

    if name == "AllelopathicHarvestOpenParallelEnv":
        return {"READY_TO_SHOOT": unit()}
    if name == "CleanUpParallelEnv":
        return {
            "READY_TO_SHOOT": unit(),
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP": gym.spaces.Box(
                0.0, float(len(env.possible_agents) - 1), shape=(), dtype=np.float64
            ),
        }
    if name == "CoinsParallelEnv":
        return {"MISMATCHED_COIN_COLLECTED_BY_PARTNER": gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float64)}
    if name in {
        "CommonsHarvestParallelEnv",
        "CoopMiningParallelEnv",
        "ExternalityMushroomsDenseParallelEnv",
        "TerritoryParallelEnv",
    }:
        return {"READY_TO_SHOOT": unit()}
    if name == "PaintballParallelEnv":
        return {"READY_TO_SHOOT": gym.spaces.Box(-np.inf, 1.0, shape=(), dtype=np.float64)}
    if name == "ChemistryParallelEnv":
        return {
            "VESICLE": gym.spaces.Box(0, len(env.config.compounds) - 1, shape=(), dtype=np.int64),
            "READY_TO_IO": unit(),
        }
    if name == "CollaborativeCookingParallelEnv":
        # An interact during the cooldown is silently ignored.
        return {"READY_TO_INTERACT": unit()}
    if name == "DaycareParallelEnv":
        # HOLDING: 0 nothing, 1 an apple, 2 a banana, as this agent sees kinds.
        return {
            "HOLDING": gym.spaces.Box(0.0, 2.0, shape=(), dtype=np.float64),
            "HUNGER": unit(),
        }
    if name == "FactoryCommonsEitherOrParallelEnv":
        return {"READY_TO_SHOOT": unit(), "STAMINA": unit()}
    if name == "FruitMarketConcentricRiversParallelEnv":
        fruit_count = 2
        return {
            "READY_TO_SHOOT": unit(),
            "STAMINA": unit(),
            "INVENTORY": gym.spaces.Box(0, np.iinfo(np.int64).max, shape=(fruit_count,), dtype=np.int64),
            "MY_OFFER": gym.spaces.Box(-3, 3, shape=(fruit_count,), dtype=np.int64),
            "OFFERS": gym.spaces.Box(-3, 3, shape=(102,), dtype=np.int64),
            "HUNGER": unit(),
        }
    if name == "GiftRefinementsParallelEnv":
        count = 3
        return {
            "READY_TO_SHOOT": unit(),
            "INVENTORY": gym.spaces.Box(
                0.0,
                float(env.config.capacity_per_type),
                shape=(count,),
                dtype=np.float64,
            ),
        }
    if name == "HiddenAgendaParallelEnv":
        players = len(env.possible_agents)
        return {
            "INVENTORY": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
            "READY_TO_SHOOT": unit(),
            "VOTING": gym.spaces.Box(0.0, 1.0, shape=(players, players + 2), dtype=np.float64),
        }
    if name == "PredatorPreyParallelEnv":
        return {"STAMINA": unit()}
    if name == "MatrixGameParallelEnv":
        return {
            "INVENTORY": gym.spaces.Box(0.0, np.inf, shape=(env.config.num_resources,), dtype=np.float64),
            "READY_TO_SHOOT": unit(),
            "INTERACTION_INVENTORIES": gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(2, env.config.num_resources),
                dtype=np.float64,
            ),
        }
    return {}


def object_observation(env: Any, agent: str) -> dict[str, np.ndarray]:
    """Build a structured observation without allocating an RGB frame."""
    ids = object_observation_ids(env)
    shape = _grid_shape(env)
    grids = {}
    for key, values in ids.items():
        high = max(values.values(), default=0)
        dtype = np.uint8 if high <= np.iinfo(np.uint8).max else np.uint16
        grids[key] = np.zeros(shape, dtype=dtype)
    avatar = env._avatars[agent]
    if not getattr(avatar, "active", True) and _kind(env) in {
        "AllelopathicHarvestOpenParallelEnv",
        "CleanUpParallelEnv",
        "CommonsHarvestParallelEnv",
        "DaycareParallelEnv",
        "ExternalityMushroomsDenseParallelEnv",
        "PaintballParallelEnv",
        "MatrixGameParallelEnv",
    }:
        grids.update(_scalar_values(env, agent))
        return grids

    if is_focal(env.config) and FOCAL_FAST_PATH:
        # Every map cell is in a focal window, usually many times over, so
        # build this viewer's world-frame grids once and gather the window
        # from them -- rows * cols cell evaluations instead of side * side.
        world = {
            key: np.zeros((env.config.world_rows, env.config.world_cols), dtype=grid.dtype)
            for key, grid in grids.items()
        }
        _fill_cells(env, agent, avatar, world, ids, _world_cells(env))
        index = _focal_index(env, avatar)
        inside = index >= 0
        safe = np.where(inside, index, 0)
        for key, grid in grids.items():
            grid[...] = np.where(inside, world[key].reshape(-1)[safe], 0)
    else:
        _fill_cells(env, agent, avatar, grids, ids, _view_cells(env, avatar))
    grids.update(_scalar_values(env, agent))
    return grids


# Gather focal windows from world-frame grids (exactly equivalent to evaluating
# each window cell; tests compare the two). Module-level so tests can disable it.
FOCAL_FAST_PATH = True


def _fill_cells(env, agent, avatar, grids, ids, cells) -> None:
    """Evaluate ``cells`` -- (row, col, world position) -- into ``grids``."""

    context = _context_for(env, agent)
    avatars = _visible_avatars(env)
    beams = _beam_map(env)
    for row, col, position in cells:
        values = _cell_values(env, position, agent, context)
        for key, value in values.items():
            if key in grids:
                grids[key][row, col] = value
        beam = beams.get(position)
        if beam is not None:
            grids["BEAMS"][row, col] = beam
        candidates = avatars.get(position, ())
        if candidates:
            visible = candidates[-1]
            grids["AVATARS"][row, col] = visible.index + 1
            grids["AVATAR_ORIENTATIONS"][row, col] = visible.orientation + 1
            _set_avatar_state(env, agent, visible, grids, row, col, ids)
            _set_avatar_extras(env, visible, grids, row, col, ids)
        if "SELF" in grids and position == avatar.position:
            grids["SELF"][row, col] = 1


def _world_cells(env: Any):
    for row in range(env.config.world_rows):
        for col in range(env.config.world_cols):
            yield row, col, (col, row)


class _SharedCache(dict):
    """A cache that environment copies share rather than duplicate.

    The focal index depends only on the (fixed) config, position and
    orientation, so a copy can reuse it; deep-copying it instead made every copy
    slower as the episode visited more cells (Daycare 6 ms -> 12 ms per copy
    after 600 steps), which matters to planners that copy the environment.
    """

    def __deepcopy__(self, memo: dict) -> _SharedCache:
        return self


def _focal_index(env: Any, avatar: Any) -> np.ndarray:
    """Flat world index of every focal window cell (-1 outside the map), cached
    per (position, orientation): it depends on nothing else."""

    cache = env.__dict__.get("_focal_index_cache")
    if cache is None:
        cache = env.__dict__["_focal_index_cache"] = _SharedCache()
    key = (tuple(avatar.position), int(avatar.orientation))
    index = cache.get(key)
    if index is None:
        rows, cols = _grid_shape(env)
        world_rows, world_cols = env.config.world_rows, env.config.world_cols
        index = np.full((rows, cols), -1, dtype=np.int64)
        for row, col, position in _view_cells(env, avatar):
            x, y = position
            if 0 <= x < world_cols and 0 <= y < world_rows and _in_bounds(env, position):
                index[row, col] = y * world_cols + x
        cache[key] = index
    return index


def _scalar_values(env: Any, agent: str) -> dict[str, np.ndarray]:
    """Return the existing nonvisual observation fields for one agent."""
    name = _kind(env)
    avatar = env._avatars[agent]
    if name == "AllelopathicHarvestOpenParallelEnv":
        return {"READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64)}
    if name == "CleanUpParallelEnv":
        return {
            "READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64),
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP": np.asarray(env._num_others_who_cleaned(agent), dtype=np.float64),
        }
    if name == "CoinsParallelEnv":
        return {"MISMATCHED_COIN_COLLECTED_BY_PARTNER": np.asarray(avatar.partner_mismatch, dtype=np.float64)}
    if name in {
        "CommonsHarvestParallelEnv",
        "CoopMiningParallelEnv",
        "ExternalityMushroomsDenseParallelEnv",
        "PaintballParallelEnv",
        "TerritoryParallelEnv",
    }:
        return {"READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64)}
    if name == "ChemistryParallelEnv":
        return {
            "VESICLE": np.asarray(env._compound_ids[avatar.vesicle], dtype=np.int64),
            "READY_TO_IO": np.asarray(env._ready_to_io(avatar), dtype=np.float64),
        }
    if name == "CollaborativeCookingParallelEnv":
        return {"READY_TO_INTERACT": np.asarray(float(avatar.interact_cooling_timer == 0), dtype=np.float64)}
    if name == "DaycareParallelEnv":
        return {
            "HOLDING": np.asarray(_daycare_holding(env, avatar), dtype=np.float64),
            "HUNGER": np.asarray(env._hunger_observation(avatar), dtype=np.float64),
        }
    if name == "FactoryCommonsEitherOrParallelEnv":
        return {
            "READY_TO_SHOOT": np.asarray(1.0 if avatar.grapple_cooldown <= 0 else 0.0, dtype=np.float64),
            "STAMINA": np.asarray(avatar.stamina / env.config.max_stamina, dtype=np.float64),
        }
    if name == "FruitMarketConcentricRiversParallelEnv":
        return {
            "READY_TO_SHOOT": np.asarray(1.0 if avatar.grapple_cooldown <= 0 else 0.0, dtype=np.float64),
            "STAMINA": np.asarray(avatar.stamina / env.config.max_stamina, dtype=np.float64),
            "INVENTORY": avatar.inventory.copy(),
            "MY_OFFER": env._public_offer(avatar),
            "OFFERS": env._nearby_offer_observation(avatar),
            "HUNGER": np.asarray(env._hunger(avatar), dtype=np.float64),
        }
    if name == "GiftRefinementsParallelEnv":
        return {
            "READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64),
            "INVENTORY": avatar.inventory.astype(np.float64, copy=True),
        }
    if name == "HiddenAgendaParallelEnv":
        return {
            "INVENTORY": np.asarray([avatar.inventory / env.config.inventory_capacity], dtype=np.float64),
            "READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64),
            "VOTING": env._voting_matrix.copy(),
        }
    if name == "PredatorPreyParallelEnv":
        return {"STAMINA": np.asarray(env._normalized_stamina(avatar), dtype=np.float64)}
    if name == "MatrixGameParallelEnv":
        assert avatar.latest_interaction_inventories is not None
        return {
            "INVENTORY": avatar.inventory.astype(np.float64, copy=True),
            "READY_TO_SHOOT": np.asarray(env._ready_to_shoot(avatar), dtype=np.float64),
            "INTERACTION_INVENTORIES": avatar.latest_interaction_inventories.astype(np.float64, copy=True),
        }
    return {}


def _grid_shape(env: Any) -> tuple[int, int]:
    config = env.config
    if config.observation_mode == "global":
        return config.world_rows, config.world_cols
    height, width, _ = config.local_rgb_shape
    return height // config.sprite_size, width // config.sprite_size


def _view_cells(env: Any, avatar: Any):
    rows, cols = _grid_shape(env)
    if env.config.observation_mode == "global":
        for row in range(rows):
            for col in range(cols):
                yield row, col, (col, row)
        return

    if is_focal(env.config):
        # Agent at the centre of a window covering the whole map.
        left, backward = cols // 2, rows // 2
    else:
        left = getattr(env.config, "local_view_left", cols // 2)
        backward = getattr(env.config, "local_view_backward", 1)
    forward = rows - backward - 1
    for row in range(rows):
        for col in range(cols):
            position = _relative_to_world(
                avatar.position,
                avatar.orientation,
                col - left,
                forward - row,
            )
            if hasattr(env, "_canonical_position"):
                position = env._canonical_position(position)
                if position is None:
                    yield row, col, (-1, -1)
                    continue
            elif hasattr(env, "_wrap"):
                position = env._wrap(position)
            yield row, col, position


def _relative_to_world(origin: Position, orientation: int, rel_right: int, rel_forward: int) -> Position:
    if orientation == 0:
        dx, dy = rel_right, -rel_forward
    elif orientation == 1:
        dx, dy = rel_forward, rel_right
    elif orientation == 2:
        dx, dy = -rel_right, rel_forward
    else:
        dx, dy = -rel_forward, -rel_right
    return origin[0] + dx, origin[1] + dy


def _visible_avatars(env: Any) -> dict[Position, list[Any]]:
    result: dict[Position, list[Any]] = {}
    for avatar in env._avatars.values():
        visible = getattr(avatar, "active", True)
        if _kind(env) == "HiddenAgendaParallelEnv":
            visible = visible or avatar.frozen or avatar.voted_out
        if _kind(env) == "BoatRaceEightRacesParallelEnv":
            visible = avatar.state != "playerWait"
        if visible:
            result.setdefault(avatar.position, []).append(avatar)
    return result


# One-step beam cells each renderer draws: (environment attribute, beam name).
# They show who just zapped, cleaned, mined, gifted, tagged, grappled or
# interacted -- and so who is now on cooldown. Later entries win a shared cell,
# matching the order the renderers draw them in.
_BEAM_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "CleanUpParallelEnv": (("_last_zap_cells", "zap"), ("_last_clean_cells", "clean")),
    "CollaborativeCookingParallelEnv": (("_last_interact_cells", "interact"),),
    "CommonsHarvestParallelEnv": (("_last_zap_cells", "zap"),),
    "CoopMiningParallelEnv": (("_last_mine_cells", "mine"),),
    # Daycare also marks the cell of every held fruit: upstream draws the grasp
    # beam for as long as an avatar holds something, which is how others see it.
    "DaycareParallelEnv": (("_last_grasp_cells", "grasp"), ("", "holding")),
    "ExternalityMushroomsDenseParallelEnv": (("_last_zap_cells", "zap"),),
    "FactoryCommonsEitherOrParallelEnv": (
        ("_last_grasp_cells", "grasp"),
        ("_last_grapple_cells", "grapple"),
    ),
    "FruitMarketConcentricRiversParallelEnv": (("_last_grapple_cells", "grapple"),),
    "GiftRefinementsParallelEnv": (("_last_gift_cells", "gift"),),
    "HiddenAgendaParallelEnv": (("_last_tag_cells", "tag"),),
    "PaintballParallelEnv": (("_last_zap_cells", "zap"),),
    "PredatorPreyParallelEnv": (("_last_interact_cells", "interact"),),
    "TerritoryParallelEnv": (("_last_claim_cells", "claim"), ("_last_zap_cells", "zap")),
    "MatrixGameParallelEnv": (("_last_beam_cells", "interact"),),
    # Allelopathic Harvest keeps one colour-keyed map; see _allelopathic_beams.
    "AllelopathicHarvestOpenParallelEnv": (
        ("", "zap"),
        ("", "plant_red"),
        ("", "plant_green"),
        ("", "plant_blue"),
    ),
}


def _beam_ids(name: str) -> dict[str, int]:
    return {beam: index + 1 for index, (_, beam) in enumerate(_BEAM_SOURCES.get(name, ()))}


def _allelopathic_beams(env: Any, ids: Mapping[str, int]) -> dict[Position, int]:
    # The zap beam is near-white; a planting beam takes its berry's colour, whose
    # dominant channel names it.
    beams = {}
    for position, color in env._last_beam_cells.items():
        if min(color) > 240:
            beams[position] = ids["zap"]
        else:
            channel = int(np.argmax(color))
            beams[position] = ids[("plant_red", "plant_green", "plant_blue")[channel]]
    return beams


def _beam_map(env: Any) -> dict[Position, int]:
    name = _kind(env)
    ids = _beam_ids(name)
    if not ids:
        return {}
    if name == "AllelopathicHarvestOpenParallelEnv":
        return _allelopathic_beams(env, ids)
    beams: dict[Position, int] = {}
    for attribute, beam in _BEAM_SOURCES[name]:
        if attribute:
            for position in getattr(env, attribute, ()):
                beams[position] = ids[beam]
    if name == "PredatorPreyParallelEnv":
        # Includes the beam a predator keeps showing while frozen after a kill.
        for position in env._beam_cells():
            beams[position] = ids["interact"]
    if name == "DaycareParallelEnv":
        for fruit in env._fruits.values():
            if fruit.holder is not None and fruit.position is not None:
                beams[fruit.position] = ids["holding"]
    return beams


def _ids_for(env: Any) -> dict[str, dict[str, int]]:
    name = _kind(env)
    base: dict[str, dict[str, int]] = {
        "TERRAIN": {"floor": 1, "wall": 2},
    }
    additions: dict[str, dict[str, dict[str, int]]] = {
        "AllelopathicHarvestOpenParallelEnv": {
            "TERRAIN": {"floor": 1, "soil": 2},
            "BERRIES": {
                "unripe_red": 1,
                "ripe_red": 2,
                "unripe_green": 3,
                "ripe_green": 4,
                "unripe_blue": 5,
                "ripe_blue": 6,
            },
            # A zap freezes for 25 steps but leaves the avatar sanctioned for 50;
            # a second zap while sanctioned removes it.
            "AVATAR_STATES": {
                "normal": 1,
                "frozen": 2,
                "sanctioned": 3,
                "sanctioned_frozen": 4,
            },
            # Planting colours the planter: the only record of who planted what.
            "AVATAR_COLORS": {"newborn": 1, "red": 2, "green": 3, "blue": 4},
        },
        "BoatRaceEightRacesParallelEnv": {
            # The semaphore colour is the race phase: rowing works only on green,
            # and anyone not landed when it turns red is disqualified.
            "TERRAIN": {
                "floor": 1,
                "wall": 2,
                "water": 3,
                "semaphore_red": 4,
                "semaphore_yellow": 5,
                "semaphore_green": 6,
            },
            "APPLES": {"active": 1, "paused": 2},
            "BARRIERS": {"off": 1, "on": 2},
            # The partner's last stroke: a boat moves only if both row together.
            "BOATS": {
                "hull": 1,
                "left_oar": 2,
                "right_oar": 3,
                "left_oar_row": 4,
                "left_oar_flail": 5,
                "right_oar_row": 6,
                "right_oar_flail": 7,
            },
            "AVATAR_STATES": {"player": 1, "rowing": 2, "landed": 3},
        },
        "ChemistryParallelEnv": {
            "TERRAIN": {"empty": 1},
            "COMPOUNDS": {},
            # What each avatar carries takes part in neighbouring reactions.
            "AVATAR_VESICLES": {},
        },
        "CleanUpParallelEnv": {
            "TERRAIN": {"sand": 1, "wall": 2, "river": 3, "grass": 4, "grass_edge": 5},
            "APPLES": {"live": 1},
            "DIRT": {"active": 1},
        },
        "CoinsParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2},
            # Colour is the whole dilemma: taking the partner's colour costs the
            # partner. These once compared against "red"/"blue" while the states
            # are "coin_red"..., so every coin read the same and no avatar had a
            # colour at all.
            "COINS": dict(COIN_IDS),
            "AVATAR_STATES": dict(COIN_IDS),
        },
        "CollaborativeCookingParallelEnv": {
            "TERRAIN": {
                "floor": 1,
                "counter": 2,
                "tomato_dispenser": 3,
                "dish_dispenser": 4,
                "delivery": 5,
                "pot": 6,
            },
            "ITEMS": {"tomato": 1, "dish": 2, "soup": 3},
            # The third tomato starts cooking, so one and two must differ.
            "POTS": {
                "empty": 1,
                "one_tomato": 2,
                "cooking": 3,
                "cooked": 4,
                "two_tomatoes": 5,
            },
            # When soup is ready, as upstream's loading bar shows it.
            "POT_PROGRESS": {f"progress_{level}": level for level in range(1, 11)},
            "AVATAR_ITEMS": {"tomato": 1, "dish": 2, "soup": 3},
        },
        "CommonsHarvestParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "grass": 3, "dessicated": 4},
            "APPLES": {"live": 1},
        },
        "CoopMiningParallelEnv": {
            "ORES": {"wait": 1, "iron": 2, "gold": 3, "gold_partial": 4},
        },
        "DaycareParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "tree_site": 3},
            # Sites that grow fruit, drawn whether or not a fruit hangs there:
            # fruit respawns only on trees, and nothing can be dropped on one.
            "TREES": {"tree": 1, "shrub": 2},
            # Attached fruit by kind and height; picked fruit -- held, or dropped
            # on the ground -- by kind only, as the RGB renderer draws it (a round
            # fruit rather than fruit on a tree). Without the picked codes a fruit
            # just grasped from the tree in front looks exactly like one still
            # hanging there, so whether eating would pay was invisible.
            "FRUITS": {
                "apple_tree": 1,
                "apple_shrub": 2,
                "banana_tree": 3,
                "banana_shrub": 4,
                "apple_picked": 5,
                "banana_picked": 6,
            },
        },
        "ExternalityMushroomsDenseParallelEnv": {
            "MUSHROOMS": {"red": 1, "green": 2, "blue": 3, "orange": 4},
            # A zap on a sanctioned avatar removes it instead of freezing it.
            "AVATAR_STATES": {
                "normal": 1,
                "frozen": 2,
                "sanctioned": 3,
                "sanctioned_frozen": 4,
            },
        },
        "FactoryCommonsEitherOrParallelEnv": {
            # The two dispensers decide what a hopper yields (a cube and an
            # apple, or two apples), the substrate's either-or choice.
            "TERRAIN": {
                "floor": 1,
                "wall": 2,
                "machine": 3,
                "dispenser_cube_apple": 4,
                "dispenser_two_apples": 5,
            },
            # A grasp leaves the object where it was, so held and free objects
            # must differ (as with Daycare's fruit); dispensed objects are drawn
            # in flight before they land.
            "FACTORY_OBJECTS": {
                "apple": 1,
                "blue_cube": 2,
                "held_apple": 3,
                "held_blue_cube": 4,
                "jumping_apple": 5,
                "jumping_blue_cube": 6,
            },
            "HOPPERS": {"empty": 1, "processing": 2, "occupied": 3},
            "AVATAR_STATES": {"normal": 1, "held": 2, "frozen": 3},
            "AVATAR_STAMINA": dict(_STAMINA_IDS),
        },
        "FruitMarketConcentricRiversParallelEnv": {
            "TERRAIN": {"ground": 1, "wall": 2, "river": 3},
            "TREES": {
                "empty": 1,
                "apple_unripe": 2,
                "apple_ripe": 3,
                "banana_unripe": 4,
                "banana_ripe": 5,
            },
            # A frozen avatar (grappled, grappling, or out of stamina) cannot move.
            "AVATAR_STATES": {
                "apple_farmer": 1,
                "banana_farmer": 2,
                "apple_farmer_frozen": 3,
                "banana_farmer_frozen": 4,
            },
            "AVATAR_STAMINA": dict(_STAMINA_IDS),
        },
        "GiftRefinementsParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "spawn": 3, "token_site": 4},
            "TOKENS": {"live": 1},
        },
        "HiddenAgendaParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "glass": 3, "fill": 4, "deposit": 5},
            "GEMS": {"live": 1},
            "AVATAR_STATES": {
                "active": 1,
                "frozen": 2,
                "voted_out": 3,
                "self_impostor": 4,
            },
        },
        "PaintballParallelEnv": {
            "TERRAIN": {
                "floor": 1,
                "wall": 2,
                "destroyable_wall": 3,
                "rubble": 4,
                "hill": 5,
                # Fewer zaps to break through.
                "damaged_destroyable_wall": 6,
                # Where each flag returns; capturing needs your own flag home.
                "red_home": 7,
                "blue_home": 8,
            },
            # Indicator tiles: which flags are home (capture the flag) or which
            # team controls the hill (king of the hill), which the paint alone
            # does not settle because control has hysteresis.
            "INDICATORS": {
                "red": 1,
                "blue": 2,
                "both": 3,
                "neither": 4,
                "uncontrolled": 5,
            },
            "PAINT": {"red": 1, "blue": 2},
            "FLAGS": {"red": 1, "blue": 2},
            "AVATAR_STATES": {
                "red": 1,
                "blue": 2,
                "red_flag_carrier": 3,
                "blue_flag_carrier": 4,
            },
            # How many more zaps an avatar can take.
            "AVATAR_HEALTH": {},
        },
        "PredatorPreyParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "safe_grass": 3, "fill": 4},
            "FOOD": {"apple": 1, "acorn": 2},
            # Eating shows upstream's five sprites; how far along a prey is says
            # how long it stays immobile and out of the defending group.
            "AVATAR_STATES": {
                "predator": 1,
                "prey": 2,
                "prey_alert": 3,
                "prey_with_acorn": 4,
                "prey_sit": 5,
                "prey_alert_with_acorn": 6,
                "prey_prep_to_eat": 7,
                "prey_first_bite": 8,
                "prey_second_bite": 9,
                "prey_last_bite": 10,
            },
            "AVATAR_STAMINA": dict(_STAMINA_IDS),
        },
        "TerritoryParallelEnv": {
            "RESOURCES": {"intact": 1, "damaged": 2, "destroyed": 3},
            "RESOURCE_OWNERS": {},
            # A fresh ("wet") claim cannot pay until it dries.
            "CLAIMS": {"wet": 1, "dry": 2},
            "AVATAR_STATES": {"normal": 1, "sanctioned": 2},
        },
        "MatrixGameParallelEnv": {
            "TERRAIN": {"floor": 1, "wall": 2, "spawn": 3},
            "RESOURCES": {},
            "AVATAR_STATES": {"unready": 1, "ready": 2},
            # After an interaction both avatars freeze and show a result colour
            # until removed; zapping them is wasted.
            "INTERACTION_RESULT": {},
            "AVATAR_ROLES": {},
        },
    }
    selected = additions[name]
    if name == "ChemistryParallelEnv":
        selected["COMPOUNDS"] = {compound.name: index + 1 for index, compound in enumerate(env.config.compounds)}
        selected["AVATAR_VESICLES"] = dict(selected["COMPOUNDS"])
    elif name == "PaintballParallelEnv":
        config = env.config
        top = max(
            config.initial_health,
            config.max_health_on_ground,
            config.max_health_on_own_color,
            config.max_health_on_enemy_color,
        )
        selected["AVATAR_HEALTH"] = {f"health_{h}": h for h in range(1, top + 1)}
    elif name == "TerritoryParallelEnv":
        selected["RESOURCE_OWNERS"] = {agent: index + 1 for index, agent in enumerate(env.possible_agents)}
    elif name == "MatrixGameParallelEnv":
        selected["RESOURCES"] = {f"class_{index}": index for index in range(1, env.config.num_resources + 1)}
        selected["INTERACTION_RESULT"] = {
            f"result_{index}": index + 1 for index in range(len(env.config.result_indicator_color_intervals))
        }
        roles = env.config.player_roles or ("default",)
        selected["AVATAR_ROLES"] = {role: index + 1 for index, role in enumerate(sorted(set(roles)))}
    base.update(selected)
    if name in _BEAM_SOURCES:
        base["BEAMS"] = _beam_ids(name)
    return base


def _context_for(env: Any, agent: str) -> dict[str, Any]:
    name = _kind(env)
    context: dict[str, Any] = {}
    if name == "BoatRaceEightRacesParallelEnv":
        boat_positions = {}
        for boat in env._boats:
            for position in boat.hull_positions():
                boat_positions[position] = 1
            strokes = boat.last_strokes or {}
            for side, position in boat.oar_positions().items():
                stroke = strokes.get(side)
                base = 2 if side == "L" else 3
                if stroke == "row":
                    base = 4 if side == "L" else 6
                elif stroke == "flail":
                    base = 5 if side == "L" else 7
                boat_positions[position] = base
        context["boats"] = boat_positions
    if name == "DaycareParallelEnv":
        # A held fruit can share a cell with one still hanging there (a parent
        # holding an apple in front of a banana tree). Picked fruit goes last so
        # it wins the cell, as RGB draws the loose fruit over the tree.
        fruits = sorted(
            (
                fruit
                for fruit in env._fruits.values()
                if fruit.position is not None and fruit.state in {"attached", "picked"}
            ),
            key=lambda fruit: fruit.state == "picked",
        )
        context["fruits"] = {fruit.position: fruit for fruit in fruits}
    if name == "FactoryCommonsEitherOrParallelEnv":
        # Objects in flight, then free, then held, so the most decisive state
        # wins a shared cell.
        def rank(obj):
            return 0 if obj.state == "jump" else 2 if obj.holder is not None else 1

        visible = [
            obj for obj in env._objects.values() if obj.position is not None and (obj.live or obj.state == "jump")
        ]
        context["objects"] = {obj.position: obj for obj in sorted(visible, key=rank)}
    return context


def _cell_values(env: Any, position: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    name = _kind(env)
    if not _in_bounds(env, position):
        return {}
    function: Callable[[Any, Position, str, dict[str, Any]], dict[str, int]] = _CELL_BUILDERS[name]
    return function(env, position, viewer, context)


def _in_bounds(env: Any, position: Position) -> bool:
    if hasattr(env, "_in_bounds"):
        return bool(env._in_bounds(position))
    x, y = position
    return 0 <= y < env.config.world_rows and 0 <= x < env.config.world_cols


def _base_terrain(env: Any, position: Position) -> int:
    return 2 if position in getattr(env, "_walls", set()) else 1


def _generic(env: Any, position: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    return {"TERRAIN": _base_terrain(env, position)}


def _allelopathic(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    if p not in env._floor:
        return {}
    result = {"TERRAIN": 2 if p in env._berries else 1}
    berry = env._berries.get(p)
    if berry is not None:
        result["BERRIES"] = (berry.color_id - 1) * 2 + (2 if berry.ripe else 1)
    return result


def _boat(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = (
        2
        if p in env._walls
        else 3
        if p in env._water
        else {"red": 4, "yellow": 5, "green": 6}[env._semaphore_color()]
        if p in env._semaphores
        else 1
    )
    result = {"TERRAIN": terrain}
    apple = env._apples.get(p)
    if apple is not None:
        result["APPLES"] = 1 if apple.active else 2 if apple.paused else 0
    barrier = env._barriers.get(p)
    if barrier is not None:
        result["BARRIERS"] = 2 if barrier.on else 1
    if p in context["boats"]:
        result["BOATS"] = context["boats"][p]
    return result


def _chemistry(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    result = {"TERRAIN": 1}
    cell = env._cells.get(p)
    if cell is not None:
        result["COMPOUNDS"] = env._compound_ids[cell.state] + 1
    return result


def _clean_up(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = (
        2 if p in env._walls else 3 if p in env._river else 4 if p in env._grass else 5 if p in env._grass_edges else 1
    )
    result = {"TERRAIN": terrain}
    if p in env._apples and env._apples[p].live:
        result["APPLES"] = 1
    if p in env._dirts and env._dirts[p].active:
        result["DIRT"] = 1
    return result


def _coins(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    x, y = p
    if y >= len(env._grid) or x >= env._line_lengths[y]:
        return {}
    result = {"TERRAIN": 2 if p in env._walls else 1}
    coin = env._coins.get(p)
    if coin is not None and coin.state is not None:
        result["COINS"] = COIN_IDS.get(coin.state, 0)
    return result


def _cooking(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    if p in env._void:
        return {}
    terrain = (
        2
        if p in env._counters
        else 3
        if p in env._tomato_dispensers
        else 4
        if p in env._dish_dispensers
        else 5
        if p in env._delivery_locations
        else 6
        if p in env._pots
        else 1
    )
    result = {"TERRAIN": terrain}
    item = env._counter_items.get(p, "empty")
    if item != "empty":
        result["ITEMS"] = {"tomato": 1, "dish": 2, "soup": 3}[item]
    pot = env._pots.get(p)
    if pot is not None:
        count = len(pot.contained_items)
        result["POTS"] = 4 if pot.cooked else 3 if pot.current_cooking_time else 5 if count >= 2 else 2 if count else 1
        if not pot.cooked and pot.current_cooking_time:
            total = max(env.config.cooking_time, 1)
            done = min(pot.current_cooking_time, total) / total
            result["POT_PROGRESS"] = max(1, min(10, int(np.ceil(done * 10))))
    return result


def _commons(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    if p in env._walls:
        return {"TERRAIN": 2}
    terrain = 1
    if p in env._grass:
        apple = env._apples.get(p)
        terrain = 3 if apple is not None and (apple.live or env._live_neighbor_count(p) > 0) else 4
    result = {"TERRAIN": terrain}
    if p in env._apples and env._apples[p].live:
        result["APPLES"] = 1
    return result


def _coop(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    result = {"TERRAIN": _base_terrain(env, p)}
    ore = env._ores.get(p)
    if ore is not None:
        result["ORES"] = {"wait": 1, "iron": 2, "gold": 3, "gold_partial": 4}[ore.state]
    return result


def _daycare(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = 2 if p in env._walls else 3 if p in env._tree_sites else 1
    result = {"TERRAIN": terrain}
    home = env._fruit_by_home.get(p)
    if home is not None:
        grown = env._fruits[home]
        if grown.kind is not None and grown.height is not None:
            child = env._avatars[viewer].role == "child"
            result["TREES"] = 2 if child or grown.height == "shrub" else 1
    fruit = context["fruits"].get(p)
    if fruit is not None and fruit.kind is not None:
        viewer_role = env._avatars[viewer].role
        visible_kind = _daycare_visible_kind(fruit.kind, viewer_role)
        if fruit.state == "picked":
            result["FRUITS"] = 5 if visible_kind == "apple" else 6
        elif fruit.height is not None:
            visible_height = "shrub" if viewer_role == "child" and fruit.height == "tree" else fruit.height
            base = 0 if visible_kind == "apple" else 2
            result["FRUITS"] = base + (1 if visible_height == "tree" else 2)
    return result


def _daycare_visible_kind(kind: str, viewer_role: str) -> str:
    # Parents cannot tell bananas from apples.
    return "apple" if viewer_role == "parent" and kind == "banana" else kind


def _daycare_holding(env: Any, avatar: Any) -> float:
    """What ``avatar`` holds: 0 nothing, 1 an apple, 2 a banana.

    The picked codes in ``FRUITS`` show a detached fruit but not whose it is --
    a fruit dropped in front of an agent looks like one it holds -- and the grasp
    action drops when holding, so the agent's own holding state decides what both
    grasp and eat do. Kinds are masked by role exactly as in the view.
    """

    if avatar.held_fruit is None:
        return 0.0
    fruit = env._fruits[avatar.held_fruit]
    kind = _daycare_visible_kind(fruit.kind or "apple", avatar.role)
    return 1.0 if kind == "apple" else 2.0


def _mushrooms(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    result = {"TERRAIN": _base_terrain(env, p)}
    mushroom = env._mushrooms.get(p)
    if mushroom is not None:
        result["MUSHROOMS"] = {"red": 1, "green": 2, "blue": 3, "orange": 4}.get(mushroom.state, 0)
    return result


def _factory(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    machine = env._machine_tiles.get(p)
    terrain = (
        2 if p in env._walls else 4 if machine == "s" else 5 if machine == "t" else 3 if machine is not None else 1
    )
    result = {"TERRAIN": terrain}
    obj = context["objects"].get(p)
    if obj is not None:
        base = {"apple": 1, "blue_cube": 2}[obj.kind]
        if obj.state == "jump":
            base += 4
        elif obj.holder is not None:
            base += 2
        result["FACTORY_OBJECTS"] = base
    hopper = env._hoppers.get(p)
    if hopper is not None:
        result["HOPPERS"] = 2 if hopper.processing_timer else 3 if hopper.has_cube else 1
    return result


def _fruit_market(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = 2 if p in env._walls else 3 if p in env._rivers else 1
    result = {"TERRAIN": terrain}
    tree = env._trees.get(p)
    if tree is not None:
        if tree.kind is None:
            result["TREES"] = 1
        else:
            base = 1 if tree.kind == "apple" else 3
            result["TREES"] = base + (2 if tree.ripe else 1)
    return result


def _gift(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = 2 if p in env._walls else 3 if p in env._spawn_points else 4 if p in env._tokens else 1
    result = {"TERRAIN": terrain}
    if p in env._tokens and env._tokens[p].live:
        result["TOKENS"] = 1
    return result


def _hidden(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = (
        2 if p in env._walls else 3 if p in env._glass else 4 if p in env._fill else 5 if p in env._deposit_tiles else 1
    )
    result = {"TERRAIN": terrain}
    gem_id = env._gem_by_home.get(p)
    if gem_id is not None and env._gems[gem_id].live:
        result["GEMS"] = 1
    return result


def _paintball(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    wall = env._destroyable_walls.get(p)
    homes = env._home_tiles
    terrain = (
        2
        if p in env._solid_walls
        else 4
        if wall is not None and wall.health <= 0
        else 6
        if wall is not None and wall.health <= env.config.damaged_wall_health_level
        else 3
        if wall is not None
        else 7
        if p == homes.get("red")
        else 8
        if p == homes.get("blue")
        else 5
        if p in env._hill_positions
        else 1
    )
    result = {"TERRAIN": terrain}
    if p in env._indicator_tiles:
        result["INDICATORS"] = {"red": 1, "blue": 2, "both": 3, "neither": 4, "uncontrolled": 5}[
            env._flag_control_state()
        ]
    paint = env._terrain.get(p)
    if paint in {"red", "blue"}:
        result["PAINT"] = 1 if paint == "red" else 2
    for flag in env._flags.values():
        if flag.carried_by is None and flag.position == p:
            result["FLAGS"] = 1 if flag.team == "red" else 2
    return result


def _predator(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = 4 if p in env._fill else 2 if p in env._walls else 3 if p in env._safe_grass else 1
    result = {"TERRAIN": terrain}
    if p in env._apples and env._apples[p].live:
        result["FOOD"] = 1
    elif p in env._acorns and env._acorns[p].live:
        result["FOOD"] = 2
    return result


def _territory(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    result = {"TERRAIN": _base_terrain(env, p)}
    resource = env._resources.get(p)
    if resource is not None:
        result["RESOURCES"] = (
            3 if resource.destroyed else 2 if resource.health < env.config.resource_initial_health else 1
        )
        if resource.owner is not None:
            result["RESOURCE_OWNERS"] = env.possible_agents.index(resource.owner) + 1
            result["CLAIMS"] = 2 if resource.reward_active else 1
    return result


def _matrix(env: Any, p: Position, viewer: str, context: dict[str, Any]) -> dict[str, int]:
    terrain = 2 if p in env._walls else 3 if p in env._spawn_points else 1
    result = {"TERRAIN": terrain}
    resource = env._resources.get(p)
    if resource is not None and resource.live:
        result["RESOURCES"] = resource.resource_class
    return result


_CELL_BUILDERS = {
    "AllelopathicHarvestOpenParallelEnv": _allelopathic,
    "BoatRaceEightRacesParallelEnv": _boat,
    "ChemistryParallelEnv": _chemistry,
    "CleanUpParallelEnv": _clean_up,
    "CoinsParallelEnv": _coins,
    "CollaborativeCookingParallelEnv": _cooking,
    "CommonsHarvestParallelEnv": _commons,
    "CoopMiningParallelEnv": _coop,
    "DaycareParallelEnv": _daycare,
    "ExternalityMushroomsDenseParallelEnv": _mushrooms,
    "FactoryCommonsEitherOrParallelEnv": _factory,
    "FruitMarketConcentricRiversParallelEnv": _fruit_market,
    "GiftRefinementsParallelEnv": _gift,
    "HiddenAgendaParallelEnv": _hidden,
    "PaintballParallelEnv": _paintball,
    "PredatorPreyParallelEnv": _predator,
    "TerritoryParallelEnv": _territory,
    "MatrixGameParallelEnv": _matrix,
}


def _set_avatar_state(
    env: Any,
    viewer: str,
    avatar: Any,
    grids: dict[str, np.ndarray],
    row: int,
    col: int,
    ids: GridIds,
) -> None:
    name = _kind(env)
    if "AVATAR_STATES" not in grids:
        if "AVATAR_ITEMS" in grids:
            item = getattr(avatar, "held_item", "empty")
            if item != "empty":
                grids["AVATAR_ITEMS"][row, col] = ids["AVATAR_ITEMS"].get(item, 0)
        return
    state = "normal"
    if name == "AllelopathicHarvestOpenParallelEnv" or name == "ExternalityMushroomsDenseParallelEnv":
        frozen = avatar.freeze_timer > 0
        if avatar.sanction_level > 1:
            state = "sanctioned_frozen" if frozen else "sanctioned"
        else:
            state = "frozen" if frozen else "normal"
    elif name == "BoatRaceEightRacesParallelEnv":
        state = avatar.state
    elif name == "CoinsParallelEnv":
        state = avatar.coin_type
    elif name == "FactoryCommonsEitherOrParallelEnv":
        state = "held" if avatar.held_by is not None else "frozen" if avatar.movement_freeze_timer > 0 else "normal"
    elif name == "FruitMarketConcentricRiversParallelEnv":
        frozen = avatar.movement_freeze_timer > 0 or avatar.held_by is not None
        state = f"{avatar.role}_frozen" if frozen else avatar.role
    elif name == "HiddenAgendaParallelEnv":
        state = (
            "self_impostor"
            if avatar.agent == viewer and avatar.role == "impostor"
            else "frozen"
            if avatar.frozen
            else "voted_out"
            if avatar.voted_out
            else "active"
        )
    elif name == "PaintballParallelEnv":
        state = f"{avatar.team}_flag_carrier" if avatar.carrying_flag is not None else avatar.team
    elif name == "PredatorPreyParallelEnv":
        if avatar.role == "predator":
            state = "predator"
        elif avatar.eating_timer > 0:
            state = f"prey_{env._eating_stage(avatar)}"
        elif avatar.held_item == "acorn":
            state = "prey_alert_with_acorn" if avatar.alert_timer > 0 else "prey_with_acorn"
        elif avatar.alert_timer > 0:
            state = "prey_alert"
        else:
            state = "prey"
    elif name == "TerritoryParallelEnv":
        state = "sanctioned" if avatar.sanction_level > 1 else "normal"
    elif name == "MatrixGameParallelEnv":
        state = "ready" if avatar.collected_at_least_one_resource else "unready"
    grids["AVATAR_STATES"][row, col] = ids["AVATAR_STATES"].get(state, 0)


def _set_avatar_extras(
    env: Any,
    avatar: Any,
    grids: dict[str, np.ndarray],
    row: int,
    col: int,
    ids: GridIds,
) -> None:
    """Per-avatar layers beyond the avatar state."""

    if "AVATAR_COLORS" in grids:
        grids["AVATAR_COLORS"][row, col] = avatar.color_id + 1
    if "AVATAR_VESICLES" in grids:
        grids["AVATAR_VESICLES"][row, col] = env._compound_ids[avatar.vesicle] + 1
    if "AVATAR_HEALTH" in grids:
        grids["AVATAR_HEALTH"][row, col] = max(0, min(avatar.health, len(ids["AVATAR_HEALTH"])))
    if "INTERACTION_RESULT" in grids:
        index = env._result_color_index(avatar.reward_to_determine_color)
        if index is not None:
            grids["INTERACTION_RESULT"][row, col] = index + 1
    if "AVATAR_ROLES" in grids:
        grids["AVATAR_ROLES"][row, col] = ids["AVATAR_ROLES"].get(avatar.role, 0)
    if "AVATAR_STAMINA" in grids and getattr(avatar, "active", True):
        level = stamina_bar_level(avatar.stamina)
        if level is not None:
            grids["AVATAR_STAMINA"][row, col] = level + 1


__all__ = [
    "object_observation",
    "object_observation_ids",
    "object_observation_spaces",
]
