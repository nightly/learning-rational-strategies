"""Variant maps and aliases for native Territory substrates."""

from __future__ import annotations

from dataclasses import dataclass

from src.environments.substrates.territory._core import RandomChoiceSpec, Topology

OPEN_ASCII_MAP = """
F=====================================T
|,,RRRRR,,RR,,RR,,,,,,RR,,,,,,RR,,,,,,|
|,,,,,RR,,,,,,RR,,,,,,RR,,,,,,,,,,,,,,|
|,,,,,RR,,,,,,RR,,,,,,,,,,,,,,,,,,,,,,|
|,RR,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,RR,|
|,,,,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,,,,|
|,,,,,RR,,,,,,,,,,RRRR,,,,R,,,,,,,,,,,|
|,,RR,RR,,,,,,,,,,,,,,,,,,R,,,,,,,,,,,|
|,,,,,RR,,,,,,,RR,,,,,,,,,R,,,,,,,,,,,|
|,,,,,RRRR,,,,,,,,,,,,,,,,,,,,,RR,,,,,|
|,,,,,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
|,,RRRR,,,RRRRRR,,,,,,,,,,,RR,,,,R,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,R,,,,|
|,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,,P,,|
|,,,,RR,,,,,,,,,,,,,,,,RR,,,,,,,P,,,,,|
|,,,,,,,,,RR,,,,,,,,,,,,,,,,,,,,,P,,P,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,P,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,,P,,,|
|,,P,,,,P,,,P,,P,,,P,,,,P,P,,P,,P,,P,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
L=====================================J
""".strip("\n")

ROOMS_ASCII_MAP = """
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
""".strip("\n")

INSIDE_OUT_ASCII_MAP = """
F=====================T
|,,,,,,,,,,P,,,,,,,,,,|
|,P,,,,QQ,,,,,QQ,,,,P,|
|,,RRR,,,,RRR,,,,RRR,,|
|,,R,RAAAAR,RAAAAR,R,,|
|,,RRR,BB,RRR,BB,RRR,,|
|,,,A,,BB,,A,,BB,,A,,,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,,,A,,BB,,A,,BB,,A,,,|
|,,RRR,BB,RRR,BB,RRR,,|
|P,R,RAAAAR,RAAAAR,R,P|
|,,RRR,BB,RRR,BB,RRR,,|
|,,,A,,BB,,A,,BB,,A,,,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,,,A,,BB,,A,,BB,,A,,,|
|,,RRR,BB,RRR,BB,RRR,,|
|,,R,RAAAAR,RAAAAR,R,,|
|,,RRR,,,,RRR,,,,RRR,,|
|,P,,,,QQ,,,,,QQ,,,,P,|
|,,,,,,,,,,P,,,,,,,,,,|
L=====================J
""".strip("\n")

INSIDE_OUT_RANDOM_CHOICES = (
    RandomChoiceSpec("Q", "spawn", "floor", (1, 6)),
    RandomChoiceSpec("A", "resource", "floor", (2, 1)),
    RandomChoiceSpec("B", "resource", "floor", (1, 3)),
)


@dataclass(frozen=True)
class TerritoryVariantSpec:
    name: str
    upstream_name: str
    ascii_map: str
    num_players: int
    topology: Topology
    random_choices: tuple[RandomChoiceSpec, ...] = ()


VARIANT_SPECS: dict[str, TerritoryVariantSpec] = {
    "open": TerritoryVariantSpec(
        name="open",
        upstream_name="territory__open",
        ascii_map=OPEN_ASCII_MAP,
        num_players=9,
        topology="BOUNDED",
    ),
    "rooms": TerritoryVariantSpec(
        name="rooms",
        upstream_name="territory__rooms",
        ascii_map=ROOMS_ASCII_MAP,
        num_players=9,
        topology="TORUS",
    ),
    "inside_out": TerritoryVariantSpec(
        name="inside_out",
        upstream_name="territory__inside_out",
        ascii_map=INSIDE_OUT_ASCII_MAP,
        num_players=5,
        topology="BOUNDED",
        random_choices=INSIDE_OUT_RANDOM_CHOICES,
    ),
}

ALIASES = {spec.upstream_name: name for name, spec in VARIANT_SPECS.items()}

__all__ = [
    "ALIASES",
    "INSIDE_OUT_ASCII_MAP",
    "INSIDE_OUT_RANDOM_CHOICES",
    "OPEN_ASCII_MAP",
    "ROOMS_ASCII_MAP",
    "TerritoryVariantSpec",
    "VARIANT_SPECS",
]
