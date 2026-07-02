import pytest

from budget_agent.hindsight import StepCost, calibration_pairs, remaining_costs
from budget_agent.parsing import Estimate


def test_remaining_costs_are_suffix_sums():
    costs = [StepCost(100, 1), StepCost(200, 0), StepCost(50, 1)]
    # w=500: 步消耗 = [600, 200, 550] → 后缀和 [1350, 750, 550]
    assert remaining_costs(costs, search_token_equiv=500) == [1350.0, 750.0, 550.0]


def test_remaining_costs_token_only():
    costs = [StepCost(100), StepCost(200)]
    assert remaining_costs(costs, search_token_equiv=0) == [300.0, 200.0]


def test_calibration_pairs_alignment_and_none_skipping():
    costs = [StepCost(100), StepCost(200), StepCost(50)]
    est = Estimate(low=100, high=400, p_success=0.5)
    pairs = calibration_pairs([est, None, est], costs, search_token_equiv=0)
    assert len(pairs) == 2
    assert pairs[0] == (est, 350.0)  # 第 1 步真值 = 100+200+50
    assert pairs[1] == (est, 50.0)   # 第 3 步真值 = 50


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        calibration_pairs([None], [StepCost(1), StepCost(2)])


def test_empty_trajectory():
    assert remaining_costs([]) == []
    assert calibration_pairs([], []) == []
