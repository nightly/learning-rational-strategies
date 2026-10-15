"""Territory substrate family entrypoints."""

from __future__ import annotations

from typing import Any

from pettingzoo.utils import parallel_to_aec

from src.environments.substrates.territory import inside_out, open, rooms

_SUB_SUBSTRATES = {
    "open": open,
    "territory__open": open,
    "rooms": rooms,
    "territory__rooms": rooms,
    "inside_out": inside_out,
    "territory__inside_out": inside_out,
}


def parallel_env(sub_substrate: str = "open", **kwargs: Any):
    """Create a native PettingZoo ParallelEnv for a Territory variant."""
    try:
        substrate_module = _SUB_SUBSTRATES[sub_substrate]
    except KeyError as exc:
        names = ", ".join(sorted(_SUB_SUBSTRATES))
        raise ValueError(f"Unknown territory sub-substrate {sub_substrate!r}. Available: {names}") from exc
    return substrate_module.parallel_env(**kwargs)


def env(sub_substrate: str = "open", **kwargs: Any):
    """Create a PettingZoo AEC environment for a Territory variant."""
    return parallel_to_aec(parallel_env(sub_substrate, **kwargs))


__all__ = [
    "env",
    "inside_out",
    "open",
    "parallel_env",
    "rooms",
]
