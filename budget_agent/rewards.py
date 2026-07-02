"""奖励设计与课程调度(研究计划 §4.3 / §4.4)。

    R = R_answer − λ·C_norm − μ·1[C > B] − η·WinklerNorm(ĉ_low, ĉ_high; c_true)

各项均按预算标量归一化,保持不同预算档下 reward 量级可比:
- R_answer:答案 EM/F1 ∈ [0, 1](程序化验证,qa_metrics 提供);
- C_norm = (tokens + w·searches) / B_scalar,B_scalar = token 预算 + w·检索预算;
- 违约项:任一预算维度超支即扣 μ(硬预算,生死指标 #2 的训练信号);
- 校准项:逐步 Winkler score 归一化后取均值(proper scoring rule,防区间
  报得无限宽刷分,见 winkler.py 与消融 F);
- 另有轻量格式惩罚,约束结构化输出。

课程(§4.4):无预算 → 软惩罚(λ 线性爬升)→ 硬约束(μ 生效)。
η 全程恒定——估计头从第一步就要学。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .budget_state import BudgetSpec
from .parsing import Estimate
from .winkler import normalized_winkler


@dataclass(frozen=True)
class RewardConfig:
    lambda_cost: float = 0.2       # 成本惩罚权重 λ(作用于归一化成本)
    mu_violation: float = 0.5      # 硬预算违约惩罚 μ
    eta_calibration: float = 0.2   # 校准项权重 η
    alpha: float = 0.2             # 区间置信参数,名义覆盖率 1-α = 80%
    search_token_equiv: float = 500.0  # 一次检索折算的 token 等价成本 w
    format_penalty: float = 0.1    # 每步格式错误的轻惩罚(封顶一次)
    calibration_clip: float = 5.0  # 单步归一化 Winkler 的截断上限,保持
                                   # reward 尺度稳定,防校准项淹没答案奖励


@dataclass(frozen=True)
class RewardBreakdown:
    """奖励分解,便于训练日志与消融分析。"""

    answer: float
    cost_term: float
    violation_term: float
    calibration_term: float
    format_term: float

    @property
    def total(self) -> float:
        return (
            self.answer
            - self.cost_term
            - self.violation_term
            - self.calibration_term
            - self.format_term
        )


def total_cost(tokens: int, searches: int, search_token_equiv: float = 500.0) -> float:
    """实际总成本 C = token + 检索次数加权(§4.3)。"""
    return tokens + search_token_equiv * searches


def compute_reward(
    answer_score: float,
    used_tokens: int,
    used_searches: int,
    budget: BudgetSpec,
    calib_pairs: list[tuple[Estimate, float]],
    config: RewardConfig = RewardConfig(),
    format_ok: bool = True,
) -> RewardBreakdown:
    """整条轨迹的终局奖励分解。

    Args:
        answer_score: EM/F1 ∈ [0, 1]。
        used_tokens / used_searches: 轨迹实际总消耗。
        budget: 本回合预算合同 B。
        calib_pairs: hindsight.calibration_pairs 产出的 (estimate, c_true) 对。
        format_ok: 轨迹是否全程结构化输出合法。
    """
    b_scalar = budget.scalar(config.search_token_equiv)
    cost = total_cost(used_tokens, used_searches, config.search_token_equiv)

    cost_term = config.lambda_cost * (cost / b_scalar)

    violated = used_tokens > budget.token_budget or used_searches > budget.search_budget
    violation_term = config.mu_violation if violated else 0.0

    if calib_pairs:
        mean_w = sum(
            min(
                config.calibration_clip,
                normalized_winkler(e.low, e.high, c_true, scale=b_scalar, alpha=config.alpha),
            )
            for e, c_true in calib_pairs
        ) / len(calib_pairs)
        calibration_term = config.eta_calibration * mean_w
    else:
        # 全程无有效估计:按截断上限给满额惩罚,防止"干脆不报区间"逃掉校准约束
        calibration_term = config.eta_calibration * config.calibration_clip

    format_term = 0.0 if format_ok else config.format_penalty

    return RewardBreakdown(
        answer=answer_score,
        cost_term=cost_term,
        violation_term=violation_term,
        calibration_term=calibration_term,
        format_term=format_term,
    )


@dataclass(frozen=True)
class CurriculumSchedule:
    """三段课程:progress ∈ [0,1] 为训练进度(步数 / 总步数)。

    [0, stage1_end):        无预算(λ=0, μ=0),先学任务与估计格式
    [stage1_end, stage2_end): 软惩罚,λ 从 0 线性爬升到 λ_max,μ=0
    [stage2_end, 1]:         硬约束,λ=λ_max 且 μ 生效
    """

    stage1_end: float = 0.2
    stage2_end: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.stage1_end < self.stage2_end <= 1.0:
            raise ValueError(f"invalid curriculum breakpoints: {self}")

    def weights_at(self, progress: float, base: RewardConfig) -> RewardConfig:
        progress = min(1.0, max(0.0, progress))
        if progress < self.stage1_end:
            return replace(base, lambda_cost=0.0, mu_violation=0.0)
        if progress < self.stage2_end:
            frac = (progress - self.stage1_end) / (self.stage2_end - self.stage1_end)
            return replace(
                base, lambda_cost=base.lambda_cost * frac, mu_violation=0.0
            )
        return base
