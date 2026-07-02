"""预算档采样与评测网格(研究计划 §4.4 / §5.5)。

训练档:token {2k, 6k, 10k} × 检索 {3, 6, 10},每回合随机采样;
内插测试档:token {4k, 8k};外推测试档:token {1k, 14k}。
生死指标 #4 要求同一 policy 在内插与外推档上不掉队。
"""

from __future__ import annotations

import random

from .budget_state import BudgetSpec

TRAIN_TOKEN_TIERS: tuple[int, ...] = (2_000, 6_000, 10_000)
TRAIN_SEARCH_TIERS: tuple[int, ...] = (3, 6, 10)
INTERP_TOKEN_TIERS: tuple[int, ...] = (4_000, 8_000)
EXTRAP_TOKEN_TIERS: tuple[int, ...] = (1_000, 14_000)
DEFAULT_EVAL_SEARCH_TIERS: tuple[int, ...] = (6,)  # 测试档默认固定中档检索预算


def tier_label(token_budget: int, search_budget: int) -> str:
    """可读档位标签,注入 prompt(如 "2k×3")。"""
    if token_budget % 1000 == 0:
        tok = f"{token_budget // 1000}k"
    else:
        tok = str(token_budget)
    return f"{tok}×{search_budget}"


def sample_train_budget(rng: random.Random) -> BudgetSpec:
    """训练时每回合从训练网格随机采样一个预算档(§4.4 步骤 2)。"""
    tokens = rng.choice(TRAIN_TOKEN_TIERS)
    searches = rng.choice(TRAIN_SEARCH_TIERS)
    return BudgetSpec(
        token_budget=tokens, search_budget=searches, tier=tier_label(tokens, searches)
    )


def eval_budget_grid(kind: str, search_tiers: tuple[int, ...] | None = None) -> list[BudgetSpec]:
    """评测网格:kind ∈ {train, interp, extrap}。

    interp/extrap 默认固定检索预算为中档 6,把泛化压力集中在 token 维度
    (与 §5.5 的 {4k, 8k} / {1k, 14k} 协议一致);需要全网格时显式传入。
    """
    if kind == "train":
        token_tiers = TRAIN_TOKEN_TIERS
        search_tiers = search_tiers or TRAIN_SEARCH_TIERS
    elif kind == "interp":
        token_tiers = INTERP_TOKEN_TIERS
        search_tiers = search_tiers or DEFAULT_EVAL_SEARCH_TIERS
    elif kind == "extrap":
        token_tiers = EXTRAP_TOKEN_TIERS
        search_tiers = search_tiers or DEFAULT_EVAL_SEARCH_TIERS
    else:
        raise ValueError(f"unknown grid kind: {kind!r} (expected train/interp/extrap)")
    return [
        BudgetSpec(token_budget=t, search_budget=s, tier=tier_label(t, s))
        for t in token_tiers
        for s in search_tiers
    ]
