"""Small factory for thin Commons Harvest variant modules."""

from __future__ import annotations

import sys
from typing import Any

from src.environments.substrates.commons_harvest._core import (
    CommonsHarvestConfig,
    CommonsHarvestParallelEnv,
    RenderMode,
    make_aec_env,
    make_parallel_env,
)
from src.environments.substrates.commons_harvest._variants import make_config_class, make_metadata


def make_variant_api(
    *,
    variant_name: str,
    config_class_name: str,
    env_class_name: str,
) -> tuple[type[CommonsHarvestConfig], type[CommonsHarvestParallelEnv], Any, Any]:
    """Build the public config/env symbols used by a variant package."""
    module_name = sys._getframe(1).f_globals.get("__name__", __name__)
    config_cls = make_config_class(config_class_name, variant_name)
    env_cls = type(
        env_class_name,
        (CommonsHarvestParallelEnv,),
        {
            "__doc__": f"PettingZoo ParallelEnv for commons_harvest/{variant_name}.",
            "__module__": module_name,
            "metadata": make_metadata(variant_name),
        },
    )

    def parallel_env(
        render_mode: RenderMode = None,
        config: CommonsHarvestConfig | None = None,
        **config_overrides: Any,
    ) -> CommonsHarvestParallelEnv:
        return make_parallel_env(
            config_cls=config_cls,
            env_cls=env_cls,
            render_mode=render_mode,
            config=config,
            **config_overrides,
        )

    def env(render_mode: RenderMode = None, **kwargs: Any):
        return make_aec_env(
            parallel_factory=parallel_env,
            render_mode=render_mode,
            **kwargs,
        )

    parallel_env.__module__ = module_name
    env.__module__ = module_name
    return config_cls, env_cls, parallel_env, env


__all__ = ["make_variant_api"]
