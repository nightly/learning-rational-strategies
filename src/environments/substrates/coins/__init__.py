"""Coins substrate entrypoints."""

from __future__ import annotations

from typing import Any

from pettingzoo.utils import parallel_to_aec

from src.environments.substrates.coins import coins

_SUB_SUBSTRATES = {
    "coins": coins,
}


def parallel_env(sub_substrate: str = "coins", **kwargs: Any):
    """Create a native PettingZoo ParallelEnv for a Coins sub-substrate."""
    try:
        substrate_module = _SUB_SUBSTRATES[sub_substrate]
    except KeyError as exc:
        names = ", ".join(sorted(_SUB_SUBSTRATES))
        raise ValueError(f"Unknown coins sub-substrate {sub_substrate!r}. Available: {names}") from exc
    return substrate_module.parallel_env(**kwargs)


def env(sub_substrate: str = "coins", **kwargs: Any):
    """Create a PettingZoo AEC environment for a Coins sub-substrate."""
    return parallel_to_aec(parallel_env(sub_substrate, **kwargs))


__all__ = ["coins", "env", "parallel_env"]
