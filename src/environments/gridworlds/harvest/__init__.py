from __future__ import annotations

__all__ = [
    "CHClassicConstraint",
    "CHHarvestConstraint",
    "CHSafeHarvestConstraint",
    "HARVEST_COLLECTIVE_REWARD_INFO_KEY",
    "HARVEST_READY_TO_SHOOT_INFO_KEY",
    "HarvestRGBObservationWrapper",
    "make_harvest_env",
]


def __getattr__(name: str):
    if name in {
        "HARVEST_COLLECTIVE_REWARD_INFO_KEY",
        "HARVEST_READY_TO_SHOOT_INFO_KEY",
        "HarvestRGBObservationWrapper",
        "make_harvest_env",
    }:
        from .env import (
            HARVEST_COLLECTIVE_REWARD_INFO_KEY,
            HARVEST_READY_TO_SHOOT_INFO_KEY,
            HarvestRGBObservationWrapper,
            make_harvest_env,
        )

        return {
            "HARVEST_COLLECTIVE_REWARD_INFO_KEY": HARVEST_COLLECTIVE_REWARD_INFO_KEY,
            "HARVEST_READY_TO_SHOOT_INFO_KEY": HARVEST_READY_TO_SHOOT_INFO_KEY,
            "HarvestRGBObservationWrapper": HarvestRGBObservationWrapper,
            "make_harvest_env": make_harvest_env,
        }[name]

    if name in {"CHClassicConstraint", "CHHarvestConstraint", "CHSafeHarvestConstraint"}:
        from .constraints import CHClassicConstraint, CHHarvestConstraint, CHSafeHarvestConstraint

        return {
            "CHClassicConstraint": CHClassicConstraint,
            "CHHarvestConstraint": CHHarvestConstraint,
            "CHSafeHarvestConstraint": CHSafeHarvestConstraint,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
