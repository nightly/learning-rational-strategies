from __future__ import annotations

__all__ = [
    "PursuitEvasion",
    "StagHuntMatrix",
]


def __getattr__(name: str):
    if name == "PursuitEvasion":
        from .gridworlds import PursuitEvasion

        return PursuitEvasion

    if name == "StagHuntMatrix":
        from .matrix import StagHuntMatrix

        return StagHuntMatrix

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
