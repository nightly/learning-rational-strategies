"""Variant maps and aliases for native Commons Harvest substrates."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from src.environments.substrates.commons_harvest._core import NUM_PLAYERS, CommonsHarvestConfig

OPEN_ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA    A      A    AAAW
WAA    AAA    AAA    AAW
WA    AAAAA  AAAAA    AW
W      AAA    AAA      W
W       A      A       W
W  A                A  W
W AAA  Q        Q  AAA W
WAAAAA            AAAAAW
W AAA              AAA W
W  A                A  W
W                      W
W                      W
W                      W
W  PPPPPPPPPPPPPPPPPP  W
W PPPPPPPPPPPPPPPPPPPP W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
""".strip("\n")

CLOSED_ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA    A  WW  A    AAAW
WAA    AAA WW AAA    AAW
WA    AAAAAWWAAAAA    AW
W      AAA WW AAA      W
W       A  WW  A       W
W  A       WW       A  W
W AAA  Q   WW   Q  AAA W
WAAAAA     WW     AAAAAW
W AAA  WWWWWWWWWW  AAA W
W  A       WW       A  W
WWWWWWWWWW WW WWWWWWWWWW
W          WW          W
W  WWWWWWWWWWWWWWWWWW  W
W  PPPPPPPPPPPPPPPPPP  W
W PPPPPPPPPPPPPPPPPPPP W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
""".strip("\n")

PARTNERSHIP_ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA    A  II  A    AAAW
WAA    AAA II AAA    AAW
WA    AAAAAIIAAAAA    AW
W      AAA II AAA      W
W       A  II  A       W
W  A       II       A  W
W AAA  Q   WW   Q  AAA W
WAAAAA     II     AAAAAW
W AAA  WWWWWWWWWW  AAA W
W  A       WW       A  W
WWWWWWWWWW WW WWWWWWWWWW
W          WW          W
W  WWWWWWWWWWWWWWWWWW  W
W  PPPPPPPPPPPPPPPPPP  W
W PPPPPPPPPPPPPPPPPPPP W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
""".strip("\n")

OPEN_REGROWTH_PROBABILITIES = (0.0, 0.0025, 0.005, 0.025)
CLOSED_REGROWTH_PROBABILITIES = (0.0, 0.001, 0.005, 0.025)


@dataclass(frozen=True)
class CommonsHarvestVariantSpec:
    name: str
    upstream_name: str
    ascii_map: str
    regrowth_probabilities: tuple[float, ...]
    num_players: int = NUM_PLAYERS


VARIANT_SPECS: dict[str, CommonsHarvestVariantSpec] = {
    "open": CommonsHarvestVariantSpec(
        name="open",
        upstream_name="commons_harvest__open",
        ascii_map=OPEN_ASCII_MAP,
        regrowth_probabilities=OPEN_REGROWTH_PROBABILITIES,
    ),
    "closed": CommonsHarvestVariantSpec(
        name="closed",
        upstream_name="commons_harvest__closed",
        ascii_map=CLOSED_ASCII_MAP,
        regrowth_probabilities=CLOSED_REGROWTH_PROBABILITIES,
    ),
    "partnership": CommonsHarvestVariantSpec(
        name="partnership",
        upstream_name="commons_harvest__partnership",
        ascii_map=PARTNERSHIP_ASCII_MAP,
        regrowth_probabilities=CLOSED_REGROWTH_PROBABILITIES,
    ),
}

ALIASES = {spec.upstream_name: name for name, spec in VARIANT_SPECS.items()}


def make_metadata(variant_name: str) -> dict[str, Any]:
    return {
        "name": f"commons_harvest_{variant_name}_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }


def make_config_class(
    class_name: str,
    variant_name: str,
) -> type[CommonsHarvestConfig]:
    """Build a dataclass config subclass for a public variant module."""
    spec = VARIANT_SPECS[variant_name]
    module_name = sys._getframe(1).f_globals.get("__name__", __name__)
    namespace = {
        "__doc__": f"Configuration for Commons Harvest: {variant_name}.",
        "__module__": module_name,
        "__annotations__": {
            "ascii_map": str,
            "name": str,
            "upstream_name": str,
            "num_players": int,
            "regrowth_probabilities": tuple[float, ...],
        },
        "ascii_map": spec.ascii_map,
        "name": spec.name,
        "upstream_name": spec.upstream_name,
        "num_players": spec.num_players,
        "regrowth_probabilities": spec.regrowth_probabilities,
    }
    return dataclass(frozen=True)(type(class_name, (CommonsHarvestConfig,), namespace))


__all__ = [
    "ALIASES",
    "CLOSED_ASCII_MAP",
    "CLOSED_REGROWTH_PROBABILITIES",
    "CommonsHarvestVariantSpec",
    "OPEN_ASCII_MAP",
    "OPEN_REGROWTH_PROBABILITIES",
    "PARTNERSHIP_ASCII_MAP",
    "VARIANT_SPECS",
    "make_config_class",
    "make_metadata",
]
