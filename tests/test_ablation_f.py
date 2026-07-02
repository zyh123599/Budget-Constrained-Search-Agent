"""消融 F 的机制验证:朴素覆盖惩罚可被"刷宽区间"hack,Winkler 不可。"""

from budget_agent.budget_state import BudgetSpec
from budget_agent.parsing import Estimate
from budget_agent.rewards import RewardConfig, compute_reward

import pytest

BUDGET = BudgetSpec(token_budget=2000, search_budget=3)
WINKLER = RewardConfig(calibration_rule="winkler")
NAIVE = RewardConfig(calibration_rule="coverage")


def _pairs(low, high, truth):
    return [(Estimate(low=low, high=high, p_success=0.5), truth)]


def test_naive_rule_is_hackable_winkler_is_not():
    sharp, wide = _pairs(500, 700, 600), _pairs(0, 10**6, 600)
    # 朴素覆盖:两者同为覆盖 → 惩罚同为 0,刷宽零代价
    naive_sharp = compute_reward(1.0, 500, 1, BUDGET, sharp, NAIVE)
    naive_wide = compute_reward(1.0, 500, 1, BUDGET, wide, NAIVE)
    assert naive_sharp.calibration_term == naive_wide.calibration_term == 0.0
    # Winkler:刷宽被宽度项惩罚
    w_sharp = compute_reward(1.0, 500, 1, BUDGET, sharp, WINKLER)
    w_wide = compute_reward(1.0, 500, 1, BUDGET, wide, WINKLER)
    assert w_sharp.calibration_term < w_wide.calibration_term


def test_naive_rule_still_penalizes_misses():
    missed = compute_reward(1.0, 500, 1, BUDGET, _pairs(100, 200, 600), NAIVE)
    assert missed.calibration_term == NAIVE.eta_calibration


def test_unknown_rule_raises():
    bad = RewardConfig(calibration_rule="crps")
    with pytest.raises(ValueError):
        compute_reward(1.0, 500, 1, BUDGET, _pairs(1, 2, 1.5), bad)
