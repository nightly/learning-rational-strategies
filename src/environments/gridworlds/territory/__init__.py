from __future__ import annotations

__all__ = [
    "TERRITORY_COLLECTIVE_REWARD_INFO_KEY",
    "TERRITORY_READY_TO_SHOOT_INFO_KEY",
    "TerritoryClaimConstraint",
    "TerritoryPeacefulClaimConstraint",
    "TerritoryRGBObservationWrapper",
    "TerritoryRoleClaimConstraint",
    "make_territory_env",
]


def __getattr__(name: str):
    if name in {
        "TERRITORY_COLLECTIVE_REWARD_INFO_KEY",
        "TERRITORY_READY_TO_SHOOT_INFO_KEY",
        "TerritoryRGBObservationWrapper",
        "make_territory_env",
    }:
        from .env import (
            TERRITORY_COLLECTIVE_REWARD_INFO_KEY,
            TERRITORY_READY_TO_SHOOT_INFO_KEY,
            TerritoryRGBObservationWrapper,
            make_territory_env,
        )

        return {
            "TERRITORY_COLLECTIVE_REWARD_INFO_KEY": TERRITORY_COLLECTIVE_REWARD_INFO_KEY,
            "TERRITORY_READY_TO_SHOOT_INFO_KEY": TERRITORY_READY_TO_SHOOT_INFO_KEY,
            "TerritoryRGBObservationWrapper": TerritoryRGBObservationWrapper,
            "make_territory_env": make_territory_env,
        }[name]

    if name in {
        "TerritoryClaimConstraint",
        "TerritoryPeacefulClaimConstraint",
        "TerritoryRoleClaimConstraint",
    }:
        from .constraints import (
            TerritoryClaimConstraint,
            TerritoryPeacefulClaimConstraint,
            TerritoryRoleClaimConstraint,
        )

        return {
            "TerritoryClaimConstraint": TerritoryClaimConstraint,
            "TerritoryPeacefulClaimConstraint": TerritoryPeacefulClaimConstraint,
            "TerritoryRoleClaimConstraint": TerritoryRoleClaimConstraint,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
