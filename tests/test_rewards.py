import pytest

from budget_agent.budget_state import BudgetSpec
from budget_agent.parsing import Estimate
from budget_agent.rewards import (
    CurriculumSchedule,
    RewardConfig,
    compute_reward,
    total_cost,
)

BUDGET = BudgetSpec(token_budget=2000, search_budget=3, tier="2k×3")
CFG = RewardConfig(lambda_cost=0.2, mu_violation=0.5, eta_calibration=0.2)


def _pairs(low, high, truth):
    return [(Estimate(low=low, high=high, p_success=0.5), truth)]


def test_total_cost_weighting():
    assert total_cost(1000, 2, search_token_equiv=500) == 2000


def test_correct_cheap_calibrated_is_best():
    good = compute_reward(1.0, 500, 1, BUDGET, _pairs(400, 800, 600), CFG)
    assert good.total > 0.7
    assert good.violation_term == 0.0


def test_violation_penalized_on_either_dimension():
    ok = compute_reward(1.0, 1900, 3, BUDGET, _pairs(100, 200, 150), CFG)
    over_tok = compute_reward(1.0, 2100, 3, BUDGET, _pairs(100, 200, 150), CFG)
    over_search = compute_reward(1.0, 1900, 4, BUDGET, _pairs(100, 200, 150), CFG)
    assert ok.violation_term == 0.0
    assert over_tok.violation_term == over_search.violation_term == CFG.mu_violation
    assert over_tok.total < ok.total


def test_wide_interval_hack_loses_to_sharp_interval():
    """§4.3 防 hack:同样覆盖真值,极宽区间的 reward 必须低于窄区间。"""
    sharp = compute_reward(1.0, 500, 1, BUDGET, _pairs(500, 700, 600), CFG)
    hacked = compute_reward(1.0, 500, 1, BUDGET, _pairs(0, 50_000, 600), CFG)
    assert sharp.total > hacked.total


def test_no_estimate_gets_full_calibration_penalty():
    """不报区间不能逃掉校准约束:按截断上限满额罚。"""
    none = compute_reward(1.0, 500, 1, BUDGET, [], CFG)
    some = compute_reward(1.0, 500, 1, BUDGET, _pairs(400, 800, 600), CFG)
    assert none.calibration_term == CFG.eta_calibration * CFG.calibration_clip
    assert some.calibration_term < none.calibration_term


def test_calibration_term_is_clipped():
    """极端离谱的估计惩罚有上界,保证 RL 奖励尺度稳定。"""
    insane = compute_reward(1.0, 500, 1, BUDGET, _pairs(10**6, 10**7, 600), CFG)
    assert insane.calibration_term == CFG.eta_calibration * CFG.calibration_clip


def test_format_penalty_applied():
    bad = compute_reward(1.0, 500, 1, BUDGET, _pairs(400, 800, 600), CFG, format_ok=False)
    good = compute_reward(1.0, 500, 1, BUDGET, _pairs(400, 800, 600), CFG, format_ok=True)
    assert bad.total == pytest.approx(good.total - CFG.format_penalty)


def test_cost_term_normalized_by_budget_scalar():
    small = compute_reward(1.0, 500, 1, BUDGET, _pairs(400, 800, 600), CFG)
    big_budget = BudgetSpec(token_budget=10_000, search_budget=10)
    big = compute_reward(1.0, 500, 1, big_budget, _pairs(400, 800, 600), CFG)
    assert big.cost_term < small.cost_term  # 同样消耗,大预算下相对成本更低


class TestCurriculum:
    SCHED = CurriculumSchedule(stage1_end=0.2, stage2_end=0.6)

    def test_stage1_no_budget_pressure(self):
        w = self.SCHED.weights_at(0.1, CFG)
        assert w.lambda_cost == 0.0 and w.mu_violation == 0.0
        assert w.eta_calibration == CFG.eta_calibration  # 校准项全程在线

    def test_stage2_soft_penalty_ramps(self):
        w_early = self.SCHED.weights_at(0.3, CFG)
        w_late = self.SCHED.weights_at(0.55, CFG)
        assert 0 < w_early.lambda_cost < w_late.lambda_cost < CFG.lambda_cost
        assert w_early.mu_violation == w_late.mu_violation == 0.0

    def test_stage3_hard_constraint(self):
        w = self.SCHED.weights_at(0.8, CFG)
        assert w.lambda_cost == CFG.lambda_cost
        assert w.mu_violation == CFG.mu_violation

    def test_progress_clamped(self):
        assert self.SCHED.weights_at(1.5, CFG) == CFG
        assert self.SCHED.weights_at(-0.5, CFG).lambda_cost == 0.0

    def test_invalid_breakpoints_raise(self):
        with pytest.raises(ValueError):
            CurriculumSchedule(stage1_end=0.7, stage2_end=0.6)
