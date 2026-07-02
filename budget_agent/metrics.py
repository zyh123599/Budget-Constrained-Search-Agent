"""四个生死指标(研究计划 §5.5)。

1. Pareto AUC:成功率–成本曲线下面积(取非支配前沿后按成本轴积分并归一化);
2. 预算违约率:给定预算档下超支轨迹比例;
3. 校准质量:区间覆盖率 + 平均 Winkler 分数(必须显著超过 BAGEN 的 47%);
4. 未见预算档泛化:内插 {4k, 8k} 与外推 {1k, 14k} 上的同套指标。

全部纯 Python 实现,离线轨迹重放评估(§6)直接复用,不依赖训练栈。
"""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import Estimate
from .winkler import interval_covers, winkler_score


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(cost, success) 点集的非支配前沿:成本升序、成功率严格递增。"""
    frontier: list[tuple[float, float]] = []
    for cost, success in sorted(points):
        if not frontier or success > frontier[-1][1]:
            frontier.append((cost, success))
    return frontier


def pareto_auc(points: list[tuple[float, float]], cost_cap: float) -> float:
    """成功率–成本曲线下面积,阶梯积分至 cost_cap 并归一化到 [0, 1]。

    语义:任一成本水平 c 下前沿可达的最高成功率对 c 的平均。相同成功率下
    成本更低、或相同成本下成功率更高,AUC 均更大。
    """
    if cost_cap <= 0:
        raise ValueError(f"cost_cap must be positive, got {cost_cap}")
    frontier = [(c, s) for c, s in pareto_frontier(points) if c <= cost_cap]
    if not frontier:
        return 0.0
    area = 0.0
    for i, (cost, success) in enumerate(frontier):
        next_cost = frontier[i + 1][0] if i + 1 < len(frontier) else cost_cap
        area += success * (next_cost - cost)
    return area / cost_cap


def violation_rate(costs: list[float], budgets: list[float]) -> float:
    """预算违约率:cost > budget 的轨迹比例(生死指标 #2)。"""
    if len(costs) != len(budgets):
        raise ValueError("costs and budgets must align")
    if not costs:
        return 0.0
    return sum(1 for c, b in zip(costs, budgets) if c > b) / len(costs)


@dataclass(frozen=True)
class CalibrationReport:
    coverage: float       # 区间覆盖率(对标 BAGEN 47%)
    mean_winkler: float   # 平均 Winkler 分数(越低越好)
    mean_width: float     # 平均区间宽度(诊断"靠加宽刷覆盖率")
    n: int


def calibration_report(
    pairs: list[tuple[Estimate, float]], alpha: float = 0.2
) -> CalibrationReport:
    """校准质量报告(生死指标 #3)。覆盖率必须与宽度/Winkler 同时呈报,
    单独的覆盖率会掩盖"无限宽区间"这类退化解。"""
    if not pairs:
        return CalibrationReport(coverage=0.0, mean_winkler=float("inf"), mean_width=0.0, n=0)
    n = len(pairs)
    covered = sum(1 for e, c in pairs if interval_covers(e.low, e.high, c))
    total_w = sum(winkler_score(e.low, e.high, c, alpha) for e, c in pairs)
    total_width = sum(e.high - e.low for e, _ in pairs)
    return CalibrationReport(
        coverage=covered / n, mean_winkler=total_w / n, mean_width=total_width / n, n=n
    )


@dataclass(frozen=True)
class TierResult:
    """单一预算档上的汇总结果(离线重放评估的最小输出单元)。"""

    tier: str
    success_rate: float
    mean_cost: float
    violation_rate: float
    coverage: float
    mean_winkler: float


def generalization_gap(
    train_tiers: list[TierResult], test_tiers: list[TierResult]
) -> dict[str, float]:
    """未见预算档泛化(生死指标 #4):训练档与测试档(内插/外推)关键指标的
    均值差。gap 越小,预算条件化泛化越好。"""

    def _mean(rs: list[TierResult], attr: str) -> float:
        return sum(getattr(r, attr) for r in rs) / len(rs) if rs else 0.0

    return {
        "success_rate_gap": _mean(train_tiers, "success_rate") - _mean(test_tiers, "success_rate"),
        "violation_rate_gap": _mean(test_tiers, "violation_rate") - _mean(train_tiers, "violation_rate"),
        "coverage_gap": _mean(train_tiers, "coverage") - _mean(test_tiers, "coverage"),
    }
