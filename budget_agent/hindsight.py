"""Hindsight labeling(研究计划 §4.3):rollout 结束后回填每步真值标签。

估计语义(与 BAGEN 的 rollout-replay 口径一致,保证任何轨迹都可回填标签):
第 t 步的区间预测的是"从本步起到轨迹**终止**(answer/ask/stop)实际还会
消耗的总成本",真值为后缀和:

    c_true_t = Σ_{i>=t} cost_i   (含第 t 步自身的消耗)

注意:这不是"完成任务还需的成本"——任务可行性由估计头的 p̂_success 承载。
如此定义后,止损步的正确估计是"小区间 + 低 p̂"(马上停,花不了多少;但成
不了),而非"大区间"。若用"完成任务所需成本"做真值,失败/止损轨迹将无法
回填标签,校准信号恰好在最需要止损的轨迹上消失。
"""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import Estimate


@dataclass(frozen=True)
class StepCost:
    """单步实际消耗:生成 token 数 + 该步发起的检索次数。"""

    tokens: int
    searches: int = 0

    def scalar(self, search_token_equiv: float = 500.0) -> float:
        return self.tokens + search_token_equiv * self.searches


def remaining_costs(
    costs: list[StepCost], search_token_equiv: float = 500.0
) -> list[float]:
    """每步的剩余成本真值(标量化后缀和),c_true_t = Σ_{i>=t} cost_i。"""
    out: list[float] = []
    acc = 0.0
    for c in reversed(costs):
        acc += c.scalar(search_token_equiv)
        out.append(acc)
    out.reverse()
    return out


def calibration_pairs(
    estimates: list[Estimate | None],
    costs: list[StepCost],
    search_token_equiv: float = 500.0,
) -> list[tuple[Estimate, float]]:
    """对齐每步的估计与 hindsight 真值,产出校准奖励的 (estimate, c_true) 对。

    estimates[t] 为 None(该步未给出有效估计)时跳过——格式惩罚在 reward
    层单独处理,不在此处混入。
    """
    if len(estimates) != len(costs):
        raise ValueError(
            f"length mismatch: {len(estimates)} estimates vs {len(costs)} costs"
        )
    truths = remaining_costs(costs, search_token_equiv)
    return [(e, t) for e, t in zip(estimates, truths) if e is not None]
