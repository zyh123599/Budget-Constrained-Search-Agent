"""budget_agent:校准驱动的预算条件化搜索智能体策略学习——核心库。

纯 Python、零第三方依赖,训练栈(verl/vLLM)通过 verl_reward.compute_score
接入;离线轨迹重放评估直接复用同一套打分与指标。
"""

from .budget_state import BudgetSpec, BudgetState
from .budget_sampler import eval_budget_grid, sample_train_budget, tier_label
from .hindsight import StepCost, calibration_pairs, remaining_costs
from .metrics import (
    CalibrationReport,
    TierResult,
    calibration_report,
    generalization_gap,
    pareto_auc,
    pareto_frontier,
    violation_rate,
)
from .parsing import Action, Estimate, StepOutput, parse_estimate, parse_step
from .prompts import render_prompt
from .qa_metrics import cover_exact_match, exact_match, f1_score
from .rewards import (
    CurriculumSchedule,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    total_cost,
)
from .winkler import interval_covers, normalized_winkler, winkler_score

__version__ = "0.1.0"
