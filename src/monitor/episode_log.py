from dataclasses import dataclass


@dataclass
class EpisodeLog:
    """Per-agent episode summary stored by `BoolRewardWrapper`."""

    episode: int
    """Running episode index."""

    t_end: int
    """Global timestep at episode termination or truncation."""

    ep_len: int
    """Number of environment steps in the episode."""

    cum_reward: float
    """Objective cumulative reward kept for backward-compatible plotting."""

    start_step: int | None = None
    """Global timestep at the first step of the episode, when known."""

    objective_cum_reward: float | None = None
    """Cumulative terminal objective reward."""

    learner_cum_reward: float | None = None
    """Reward actually returned to the learner."""

    handcrafted_cum_reward: float | None = None
    """Learner reward for handcrafted baselines, when applicable."""
