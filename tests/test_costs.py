import numpy as np
import pytest

from ragdecisions.costs import chance_of_meeting_bar, cheapest_meeting_bar, cost_per_1000


def test_cost_adds_input_output_and_machine_time():
    assert cost_per_1000(1000, 10, 0, 0.5, 2.0, 0) == pytest.approx(0.52)
    assert cost_per_1000(0, 0, 3.6, 1, 1, 1) == pytest.approx(1.0)


def test_the_best_configuration_always_meets_its_own_bar():
    draws = np.array([[0.5, 0.7], [0.6, 0.6]])
    assert chance_of_meeting_bar(draws, best=1, bar_points=0)[1] == 1


def test_a_difference_exactly_at_the_bar_counts():
    draws = np.array([[0.65, 0.70]])
    assert chance_of_meeting_bar(draws, best=1, bar_points=5)[0] == 1


def test_chance_counts_the_share_of_draws():
    draws = np.array([[0.69, 0.70], [0.60, 0.70], [0.70, 0.70], [0.50, 0.70]])
    assert chance_of_meeting_bar(draws, best=1, bar_points=2)[0] == 0.5


def test_a_cheaper_configuration_wins_only_if_it_is_likely_to_meet_the_bar():
    draws = np.array([[0.69, 0.60, 0.70]] * 9 + [[0.60, 0.60, 0.70]])
    assert cheapest_meeting_bar(draws, [2.0, 1.0, 3.0], best=2, bar_points=2) == 0
    assert cheapest_meeting_bar(draws, [2.0, 1.0, 3.0], best=2, bar_points=2, chance=0.95) == 2
