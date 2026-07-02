import pytest

from budget_agent.winkler import interval_covers, normalized_winkler, winkler_score


def test_covering_interval_scores_width_only():
    assert winkler_score(100, 300, 200, alpha=0.2) == 200.0


def test_miss_below_penalized():
    # c < l:宽度 200 + (2/0.2)*(100-50) = 200 + 500
    assert winkler_score(100, 300, 50, alpha=0.2) == pytest.approx(700.0)


def test_miss_above_penalized():
    assert winkler_score(100, 300, 400, alpha=0.2) == pytest.approx(200 + 10 * 100)


def test_sharper_covering_interval_wins():
    sharp = winkler_score(180, 220, 200)
    wide = winkler_score(0, 1000, 200)
    assert sharp < wide


def test_infinite_width_hack_does_not_pay():
    """防 hack 核心性质(消融 F 的机制):把区间报得极宽虽然必然覆盖,
    但宽度项使其远差于"窄且准";朴素覆盖率指标下两者反而同分。"""
    honest = winkler_score(150, 350, 300)
    hacked = winkler_score(0, 10**6, 300)
    assert honest < hacked
    assert interval_covers(150, 350, 300) and interval_covers(0, 10**6, 300)


def test_degenerate_point_interval():
    # 点区间:命中零分,偏离按 2/alpha 罚
    assert winkler_score(200, 200, 200) == 0.0
    assert winkler_score(200, 200, 210) == pytest.approx(100.0)


def test_smaller_alpha_penalizes_misses_more():
    assert winkler_score(100, 300, 400, alpha=0.1) > winkler_score(100, 300, 400, alpha=0.5)


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        winkler_score(300, 100, 200)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        winkler_score(100, 300, 200, alpha=0.0)


def test_normalized_winkler():
    assert normalized_winkler(100, 300, 200, scale=1000) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        normalized_winkler(100, 300, 200, scale=0)
