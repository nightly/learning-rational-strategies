from __future__ import annotations

__all__ = [
    "COINS_COLLECTIVE_REWARD_INFO_KEY",
    "COINS_PARTNER_MISMATCH_INFO_KEY",
    "CoinsClassicConstraint",
    "CoinsOwnCoinFirstConstraint",
    "CoinsProsocialConstraint",
    "CoinsRGBObservationWrapper",
    "make_coins_env",
]


def __getattr__(name: str):
    if name in {
        "COINS_COLLECTIVE_REWARD_INFO_KEY",
        "COINS_PARTNER_MISMATCH_INFO_KEY",
        "CoinsRGBObservationWrapper",
        "make_coins_env",
    }:
        from .env import (
            COINS_COLLECTIVE_REWARD_INFO_KEY,
            COINS_PARTNER_MISMATCH_INFO_KEY,
            CoinsRGBObservationWrapper,
            make_coins_env,
        )

        return {
            "COINS_COLLECTIVE_REWARD_INFO_KEY": COINS_COLLECTIVE_REWARD_INFO_KEY,
            "COINS_PARTNER_MISMATCH_INFO_KEY": COINS_PARTNER_MISMATCH_INFO_KEY,
            "CoinsRGBObservationWrapper": CoinsRGBObservationWrapper,
            "make_coins_env": make_coins_env,
        }[name]

    if name in {
        "CoinsClassicConstraint",
        "CoinsOwnCoinFirstConstraint",
        "CoinsProsocialConstraint",
    }:
        from .constraints import (
            CoinsClassicConstraint,
            CoinsOwnCoinFirstConstraint,
            CoinsProsocialConstraint,
        )

        return {
            "CoinsClassicConstraint": CoinsClassicConstraint,
            "CoinsOwnCoinFirstConstraint": CoinsOwnCoinFirstConstraint,
            "CoinsProsocialConstraint": CoinsProsocialConstraint,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
