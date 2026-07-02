import pytest

from budget_agent.metrics import (
    TierResult,
    calibration_report,
    generalization_gap,
    pareto_auc,
    pareto_frontier,
    violation_rate,
)
from budget_agent.parsing import Estimate


def test_pareto_frontier_drops_dominated():
    points = [(100, 0.3), (200, 0.5), (150, 0.2), (300, 0.5), (400, 0.8)]
    # (150,0.2) 被 (100,0.3) 支配;(300,0.5) 被 (200,0.5) 支配
    assert pareto_frontier(points) == [(100, 0.3), (200, 0.5), (400, 0.8)]


def test_pareto_auc_hand_computed():
    points = [(0, 0.0), (100, 0.5), (200, 1.0)]
    # 阶梯:[0,100)→0, [100,200)→0.5, [200,400)→1.0;面积=(0+50+200)/400
    assert pareto_auc(points, cost_cap=400) == pytest.approx(0.625)


def test_pareto_auc_rewards_cheaper_same_success():
    cheap = pareto_auc([(100, 0.8)], cost_cap=1000)
    dear = pareto_auc([(500, 0.8)], cost_cap=1000)
    assert cheap > dear


def test_pareto_auc_edge_cases():
    assert pareto_auc([], cost_cap=100) == 0.0
    assert pareto_auc([(200, 0.9)], cost_cap=100) == 0.0  # 前沿全部超出成本轴
    with pytest.raises(ValueError):
        pareto_auc([(1, 1.0)], cost_cap=0)


def test_violation_rate():
    assert violation_rate([100, 300, 500], [200, 200, 600]) == pytest.approx(1 / 3)
    assert violation_rate([], []) == 0.0
    with pytest.raises(ValueError):
        violation_rate([1], [1, 2])


def test_calibration_report():
    pairs = [
        (Estimate(100, 300, 0.5), 200),  # 覆盖,W=200
        (Estimate(100, 300, 0.5), 400),  # 未覆盖,W=200+10*100=1200
    ]
    rep = calibration_report(pairs, alpha=0.2)
    assert rep.coverage == 0.5
    assert rep.mean_winkler == pytest.approx(700.0)
    assert rep.mean_width == pytest.approx(200.0)
    assert rep.n == 2


def test_calibration_report_empty():
    rep = calibration_report([])
    assert rep.n == 0 and rep.coverage == 0.0


def _tier(name, sr, vr, cov):
    return TierResult(
        tier=name, success_rate=sr, mean_cost=0, violation_rate=vr,
        coverage=cov, mean_winkler=0,
    )


def test_generalization_gap():
    train = [_tier("2k×3", 0.6, 0.10, 0.80)]
    test = [_tier("4k×6", 0.5, 0.25, 0.70)]
    gap = generalization_gap(train, test)
    assert gap["success_rate_gap"] == pytest.approx(0.1)
    assert gap["violation_rate_gap"] == pytest.approx(0.15)
    assert gap["coverage_gap"] == pytest.approx(0.1)
