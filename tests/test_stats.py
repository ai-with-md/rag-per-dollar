import numpy as np

from ragdecisions.stats import bootstrap_means, interval, resample_indices


def test_resampling_is_fixed_by_the_seed():
    assert (resample_indices(10, 5, seed=1) == resample_indices(10, 5, seed=1)).all()
    assert resample_indices(10, 5).shape == (5, 10)


def test_bootstrap_of_a_constant_has_no_spread():
    draws = bootstrap_means(np.ones(50), n_boot=200)
    assert draws.shape == (200, 1)
    assert np.allclose(draws, 1.0)


def test_interval_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(3)
    covered = 0
    for _ in range(200):
        sample = rng.random(200) < 0.6
        low, high = interval(bootstrap_means(sample, n_boot=500, seed=int(rng.integers(1e9))))
        covered += low[0] <= 0.6 <= high[0]
    assert 0.90 <= covered / 200 <= 0.99


def test_configurations_share_the_same_resamples():
    a = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    draws = bootstrap_means(np.column_stack([a, a]), n_boot=300)
    assert np.allclose(draws[:, 0], draws[:, 1])  # identical columns give identical draws


def test_missing_values_are_ignored():
    draws = bootstrap_means(np.array([1.0, np.nan, 1.0, np.nan]), n_boot=50)
    assert np.allclose(draws[~np.isnan(draws)], 1.0)
