"""Commons Harvest substrate family entrypoints."""

from __future__ import annotations

from typing import Any

from pettingzoo.utils import parallel_to_aec

from src.environments.substrates.commons_harvest import closed, open, partnership

_SUB_SUBSTRATES = {
    "open": open,
    "commons_harvest__open": open,
    "closed": closed,
    "commons_harvest__closed": closed,
    "partnership": partnership,
    "commons_harvest__partnership": partnership,
}


def parallel_env(sub_substrate: str = "open", **kwargs: Any):
    """Create a native PettingZoo ParallelEnv for a Commons Harvest variant."""
    try:
        substrate_module = _SUB_SUBSTRATES[sub_substrate]
    except KeyError as exc:
        names = ", ".join(sorted(_SUB_SUBSTRATES))
        raise ValueError(f"Unknown commons_harvest sub-substrate {sub_substrate!r}. Available: {names}") from exc
    return substrate_module.parallel_env(**kwargs)


def env(sub_substrate: str = "open", **kwargs: Any):
    """Create a PettingZoo AEC environment for a Commons Harvest variant."""
    return parallel_to_aec(parallel_env(sub_substrate, **kwargs))


__all__ = ["closed", "env", "open", "parallel_env", "partnership"]
