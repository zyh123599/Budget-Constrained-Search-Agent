"""Winkler (interval) score —— 校准奖励的 proper scoring rule。

研究计划 §4.3 的核心防 hack 设计:校准项必须同时惩罚区间宽度与真值偏离。
若只用覆盖率,模型把区间报得无限宽即可刷满分;Winkler score 下,
无限宽区间的宽度项会无限增大,最优策略是"又窄又准"。

定义(Gneiting & Raftery, 2007;中心 (1-alpha) 预测区间 [lower, upper]):

    W_alpha(l, u; c) = (u - l)                            # 宽度项
                     + (2/alpha) * (l - c)   若 c < l     # 低估惩罚
                     + (2/alpha) * (c - u)   若 c > u     # 高估惩罚

分数越低越好,量纲与 c 相同(本课题中为 token 数)。
"""

from __future__ import annotations


def winkler_score(lower: float, upper: float, actual: float, alpha: float = 0.2) -> float:
    """单个区间估计的 Winkler score(越低越好)。

    Args:
        lower: 区间下界 ĉ_low。
        upper: 区间上界 ĉ_high,要求 upper >= lower(解析层保证;违反则抛错)。
        actual: hindsight 回填的真值 c_true(实际剩余消耗)。
        alpha: 置信水平参数,区间名义覆盖率为 1 - alpha(默认 80% 区间)。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if upper < lower:
        raise ValueError(f"invalid interval: upper={upper} < lower={lower}")
    score = upper - lower
    if actual < lower:
        score += (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        score += (2.0 / alpha) * (actual - upper)
    return score


def normalized_winkler(
    lower: float, upper: float, actual: float, scale: float, alpha: float = 0.2
) -> float:
    """按 scale(通常取预算标量)归一化的 Winkler score,便于放进 reward 保持量级稳定。"""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return winkler_score(lower, upper, actual, alpha) / scale


def interval_covers(lower: float, upper: float, actual: float) -> bool:
    """真值是否落在区间内(评测用覆盖率指标,不作训练信号)。"""
    return lower <= actual <= upper
