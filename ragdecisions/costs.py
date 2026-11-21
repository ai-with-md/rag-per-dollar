"""Cost per 1,000 answers, and the cheapest configuration that meets an accuracy bar."""

import numpy as np


def cost_per_1000(input_tokens, output_tokens, seconds, input_price, output_price, hourly_rate):
    """Dollars per 1,000 answers.

    Tokens and seconds are per answer, prices are dollars per million tokens,
    and hourly_rate is dollars per hour of the machine doing search and reranking.
    """
    return 1000 * (input_tokens * input_price / 1e6 + output_tokens * output_price / 1e6
                   + seconds * hourly_rate / 3600)


def chance_of_meeting_bar(draws, best: int, bar_points: float) -> np.ndarray:
    """For each configuration, the share of bootstrap draws within bar_points of the best one.

    draws has shape (n_boot, configurations), holding accuracy as a share.
    """
    draws = np.asarray(draws, dtype=float)
    return (draws - draws[:, [best]] >= -bar_points / 100 - 1e-9).mean(axis=0)


def cheapest_meeting_bar(draws, costs, best: int, bar_points: float, chance: float = 0.8) -> int:
    """Index of the cheapest configuration with at least this chance of meeting the bar."""
    passing = np.flatnonzero(chance_of_meeting_bar(draws, best, bar_points) >= chance)
    return int(passing[np.argmin(np.asarray(costs)[passing])])
