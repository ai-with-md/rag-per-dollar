"""Bootstrap intervals over questions.

Every configuration answers the same questions, so one set of resampled
question indices is reused for all configurations. That keeps comparisons
between configurations paired.
"""

import warnings

import numpy as np


def resample_indices(n_questions: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """n_boot rows of question indices drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_questions, size=(n_boot, n_questions))


def bootstrap_means(values, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap means of a questions x configurations array, shape (n_boot, configurations).

    A 1-d array is treated as a single configuration. Missing values (nan) are
    ignored within each resample.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    idx = resample_indices(values.shape[0], n_boot, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a resample with only missing values gives nan
        return np.nanmean(values[idx], axis=1)


def interval(draws, level: float = 0.95):
    """Percentile interval of bootstrap draws along the first axis: (low, high)."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail], axis=0)
    return low, high
