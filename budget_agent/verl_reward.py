"""verl 自定义奖励入口(§4.4 GRPO 主训练的接线点)。

verl 的 custom_reward_function 约定签名:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float

本模块把 parsing / hindsight / rewards 组装成该入口。多轮轨迹的逐步 token
消耗应由 rollout 侧记账后放进 extra_info(见下方约定);缺失时退化为对
solution_str 的空白分词近似,并在日志中可见(仅用于冒烟测试,正式训练
必须接真实 token 计数)。

extra_info 约定(prepare_data.py 与 rollout 环境共同维护):
    {
      "token_budget": int, "search_budget": int, "tier": str,
      "step_tokens": [int, ...],    # 每步生成 token 数(rollout 记账)
      "step_searches": [int, ...],  # 每步检索次数(0/1)
      "progress": float,            # 训练进度 ∈ [0,1],课程调度用(可选)
    }
"""

from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger(__name__)

from .budget_state import BudgetSpec
from .hindsight import StepCost, calibration_pairs
from .parsing import parse_step
from .qa_metrics import exact_match
from .rewards import CurriculumSchedule, RewardConfig, compute_reward

_DEFAULT_CONFIG = RewardConfig()
_DEFAULT_CURRICULUM = CurriculumSchedule()

# 多轮拼接的轨迹里,每一步以助手生成段为界。Search-R1 环境把检索结果包在
# <information>...</information> 中插回,以此切分助手步。
_INFO_SPLIT_RE = re.compile(r"<information>.*?</information>", re.DOTALL)


def _split_steps(solution_str: str) -> list[str]:
    """按 <information> 块把整条轨迹切成助手生成的各步。"""
    steps = [s for s in _INFO_SPLIT_RE.split(solution_str) if s.strip()]
    return steps or [solution_str]


def _approx_tokens(text: str) -> int:
    """空白分词近似 token 数——仅冒烟测试兜底,正式训练用 rollout 记账。"""
    return len(text.split())


def score_trajectory(
    solution_str: str,
    ground_truths: list[str],
    budget: BudgetSpec,
    step_tokens: list[int] | None = None,
    step_searches: list[int] | None = None,
    config: RewardConfig = _DEFAULT_CONFIG,
) -> float:
    """纯函数打分:离线轨迹重放评估(§6)与 verl 入口共用。"""
    step_texts = _split_steps(solution_str)
    parsed = [parse_step(t) for t in step_texts]

    if step_tokens is None:
        step_tokens = [_approx_tokens(t) for t in step_texts]
    if step_searches is None:
        step_searches = [
            1 if p.action is not None and p.action.value in ("search", "pivot") else 0
            for p in parsed
        ]
    n = min(len(parsed), len(step_tokens), len(step_searches))
    if not (len(parsed) == len(step_tokens) == len(step_searches)):
        _log.warning(
            "step count mismatch: parsed=%d, step_tokens=%d, step_searches=%d; truncating to %d",
            len(parsed), len(step_tokens), len(step_searches), n,
        )
    costs = [StepCost(tokens=step_tokens[i], searches=step_searches[i]) for i in range(n)]
    estimates = [parsed[i].estimate for i in range(n)]

    answer = ""
    for p in reversed(parsed):
        if p.action is not None and p.action.value == "answer":
            answer = p.content
            break
    answer_score = exact_match(answer, ground_truths) if answer else 0.0

    pairs = calibration_pairs(estimates, costs, config.search_token_equiv)
    breakdown = compute_reward(
        answer_score=answer_score,
        used_tokens=sum(c.tokens for c in costs),
        used_searches=sum(c.searches for c in costs),
        budget=budget,
        calib_pairs=pairs,
        config=config,
        format_ok=all(p.format_ok for p in parsed),
    )
    return breakdown.total


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """verl custom_reward_function 入口。"""
    extra_info = extra_info or {}
    budget = BudgetSpec(
        token_budget=int(extra_info.get("token_budget", 10_000)),
        search_budget=int(extra_info.get("search_budget", 10)),
        tier=str(extra_info.get("tier", "")),
    )
    if isinstance(ground_truth, dict):
        truths = ground_truth.get("target") or ground_truth.get("answers") or []
    elif isinstance(ground_truth, str):
        truths = [ground_truth]
    else:
        truths = list(ground_truth)

    config = _DEFAULT_CURRICULUM.weights_at(
        float(extra_info.get("progress", 1.0)), _DEFAULT_CONFIG
    )
    return score_trajectory(
        solution_str=solution_str,
        ground_truths=[str(t) for t in truths],
        budget=budget,
        step_tokens=extra_info.get("step_tokens"),
        step_searches=extra_info.get("step_searches"),
        config=config,
    )
