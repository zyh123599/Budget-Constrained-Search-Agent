#!/usr/bin/env python3
"""离线轨迹重放评估(§6 对策:预算 sweep 不重训,重放已有轨迹逐档打分)。

输入:JSONL,每行一条轨迹:
    {
      "solution_str": "<think>...</think><estimate>...</estimate><search>...",
      "ground_truths": ["Christopher Nolan"],
      "step_tokens": [400, 120],        # rollout 记账的每步生成 token 数
      "step_searches": [1, 0],
      "token_budget": 2000, "search_budget": 3, "tier": "2k×3"
    }

输出:每个预算档的 TierResult(成功率 / 均值成本 / 违约率 / 覆盖率 / Winkler)
+ 全局 Pareto AUC + 内插/外推泛化 gap(生死指标 #1–#4 一站式)。

用法:
    python scripts/eval_budget_sweep.py runs/qwen3_4b_step200.jsonl \
        --train-tiers 2k 6k 10k --alpha 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_agent.budget_state import BudgetSpec
from budget_agent.hindsight import StepCost, calibration_pairs
from budget_agent.metrics import (
    TierResult,
    calibration_report,
    generalization_gap,
    pareto_auc,
    violation_rate,
)
from budget_agent.parsing import parse_step
from budget_agent.qa_metrics import exact_match
from budget_agent.rewards import RewardConfig, total_cost
from budget_agent.verl_reward import _split_steps


def evaluate(path: Path, alpha: float, config: RewardConfig) -> tuple[dict[str, TierResult], list[tuple[float, float]]]:
    by_tier: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                by_tier[traj.get("tier", "unspecified")].append(traj)

    results: dict[str, TierResult] = {}
    pareto_points: list[tuple[float, float]] = []
    for tier, trajs in sorted(by_tier.items()):
        successes, costs, budgets, pairs = [], [], [], []
        for t in trajs:
            steps = _split_steps(t["solution_str"])
            parsed = [parse_step(s) for s in steps]
            n = min(len(parsed), len(t["step_tokens"]), len(t["step_searches"]))
            step_costs = [
                StepCost(t["step_tokens"][i], t["step_searches"][i]) for i in range(n)
            ]
            answer = next(
                (p.content for p in reversed(parsed) if p.action and p.action.value == "answer"),
                "",
            )
            successes.append(exact_match(answer, t["ground_truths"]) if answer else 0.0)
            cost = total_cost(
                sum(c.tokens for c in step_costs),
                sum(c.searches for c in step_costs),
                config.search_token_equiv,
            )
            costs.append(cost)
            budgets.append(
                BudgetSpec(t["token_budget"], t["search_budget"]).scalar(config.search_token_equiv)
            )
            pairs.extend(
                calibration_pairs([p.estimate for p in parsed[:n]], step_costs, config.search_token_equiv)
            )

        calib = calibration_report(pairs, alpha)
        sr = sum(successes) / len(successes)
        mc = sum(costs) / len(costs)
        results[tier] = TierResult(
            tier=tier,
            success_rate=sr,
            mean_cost=mc,
            violation_rate=violation_rate(costs, budgets),
            coverage=calib.coverage,
            mean_winkler=calib.mean_winkler,
        )
        pareto_points.append((mc, sr))
    return results, pareto_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--train-tiers", nargs="*", default=["2k", "6k", "10k"],
                        help="训练档 token 前缀,用于切分泛化 gap")
    parser.add_argument("--cost-cap", type=float, default=None,
                        help="Pareto AUC 的成本轴上限;缺省取最大档预算标量")
    args = parser.parse_args()

    config = RewardConfig(alpha=args.alpha)
    results, points = evaluate(args.trajectories, args.alpha, config)

    def fmt(x: float, width: int, prec: int) -> str:
        return f"{'n/a':>{width}}" if math.isnan(x) else f"{x:{width}.{prec}f}"

    header = f"{'tier':>10} {'succ':>6} {'cost':>9} {'viol':>6} {'cover':>6} {'winkler':>9}"
    print(header + "\n" + "-" * len(header))
    for r in results.values():
        print(
            f"{r.tier:>10} {r.success_rate:6.3f} {r.mean_cost:9.1f} "
            f"{r.violation_rate:6.3f} {fmt(r.coverage, 6, 3)} {fmt(r.mean_winkler, 9, 1)}"
        )

    cap = args.cost_cap or (max(c for c, _ in points) * 1.1 if points else 1.0)
    print(f"\nPareto AUC (cap={cap:.0f}): {pareto_auc(points, cap):.4f}")

    train = [r for r in results.values() if any(r.tier.startswith(p) for p in args.train_tiers)]
    test = [r for r in results.values() if r not in train]
    if train and test:
        print("Generalization gap (train -> unseen tiers):")
        for k, v in generalization_gap(train, test).items():
            print(f"  {k}: {v:+.4f}")


if __name__ == "__main__":
    main()
