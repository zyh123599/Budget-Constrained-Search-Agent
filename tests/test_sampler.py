import random

import pytest

from budget_agent.budget_sampler import (
    EXTRAP_TOKEN_TIERS,
    INTERP_TOKEN_TIERS,
    TRAIN_SEARCH_TIERS,
    TRAIN_TOKEN_TIERS,
    eval_budget_grid,
    sample_train_budget,
    tier_label,
)


def test_tier_label():
    assert tier_label(2000, 3) == "2k×3"
    assert tier_label(14_000, 6) == "14k×6"
    assert tier_label(2500, 3) == "2500×3"


def test_sample_stays_on_train_grid_and_is_seeded():
    rng = random.Random(42)
    for _ in range(50):
        spec = sample_train_budget(rng)
        assert spec.token_budget in TRAIN_TOKEN_TIERS
        assert spec.search_budget in TRAIN_SEARCH_TIERS
        assert spec.tier == tier_label(spec.token_budget, spec.search_budget)
    a = [sample_train_budget(random.Random(7)) for _ in range(10)]
    b = [sample_train_budget(random.Random(7)) for _ in range(10)]
    assert a == b


def test_train_grid_full_cross():
    grid = eval_budget_grid("train")
    assert len(grid) == len(TRAIN_TOKEN_TIERS) * len(TRAIN_SEARCH_TIERS)


def test_eval_grids_match_protocol():
    interp = eval_budget_grid("interp")
    extrap = eval_budget_grid("extrap")
    assert sorted({s.token_budget for s in interp}) == sorted(INTERP_TOKEN_TIERS)
    assert sorted({s.token_budget for s in extrap}) == sorted(EXTRAP_TOKEN_TIERS)
    # 测试档的 token 预算与训练档零交集(生死指标 #4 的前提)
    assert not set(INTERP_TOKEN_TIERS) & set(TRAIN_TOKEN_TIERS)
    assert not set(EXTRAP_TOKEN_TIERS) & set(TRAIN_TOKEN_TIERS)


def test_unknown_grid_kind_raises():
    with pytest.raises(ValueError):
        eval_budget_grid("ood")
