from dataclasses import dataclass


@dataclass
class RMConfig:
    """Configuration for reward-monitor observation augmentation."""

    temporally_extended: bool = True
    """Append the current monitor-state id to each learner observation."""
