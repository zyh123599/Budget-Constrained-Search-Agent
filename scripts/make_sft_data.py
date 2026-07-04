#!/usr/bin/env python3
"""SFT 热启动数据构造(§4.4 步骤 1):rollout 轨迹 → hindsight 重标注的多轮对话。

输入:rollout_eval.py 产出的轨迹 JSONL(来源可以是强模型 API 蒸馏的格式示范,
也可以是基座模型自身的 rollout——后者免 API 成本,格式正确率低一些但够用)。

处理:按 <information> 切步 → 后缀和回填每步真值 → sft.relabel_trajectory 把
<estimate> 替换为教学值(真值居中、宽度 w、p̂=平滑后的最终成败)→ 重建
user/assistant 多轮消息(env 轮 = <information> + 最新 <budget> 块)。

输出:messages 格式 JSONL(每行 {"messages": [{"role","content"},...]}),
transformers/TRL/LLaMA-Factory 均可直接消费。

用法:
    python scripts/make_sft_data.py runs/teacher_demos.jsonl \
        --out data/sft_train.jsonl --only-format-ok
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_agent.budget_state import BudgetSpec, BudgetState
from budget_agent.hindsight import StepCost
from budget_agent.parsing import parse_step
from budget_agent.prompts import render_prompt
from budget_agent.qa_metrics import exact_match
from budget_agent.sft import SFTConfig, relabel_trajectory
from budget_agent.verl_reward import _split_steps

_INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL)


def build_messages(traj: dict, config: SFTConfig, args) -> list[dict] | None:
    steps = _split_steps(traj["solution_str"])
    infos = _INFO_RE.findall(traj["solution_str"])
    n = min(len(steps), len(traj["step_tokens"]), len(traj["step_searches"]))
    if n == 0:
        return None
    steps = steps[:n]
    costs = [StepCost(traj["step_tokens"][i], traj["step_searches"][i]) for i in range(n)]

    if args.only_format_ok and not all(parse_step(s).format_ok for s in steps):
        return None

    success = bool(traj.get("answer")) and exact_match(traj["answer"], traj["ground_truths"]) > 0
    relabeled = relabel_trajectory(steps, costs, success, config)

    spec = BudgetSpec(traj["token_budget"], traj["search_budget"], traj.get("tier", ""))
    state = BudgetState(spec)
    messages = [{"role": "user", "content": render_prompt(
        traj["question"], state,
        include_budget=args.include_budget, estimate_mode=args.estimate_mode,
    )}]
    for i, step_text in enumerate(relabeled):
        messages.append({"role": "assistant", "content": step_text.strip()})
        state.charge(costs[i].tokens, costs[i].searches)
        if i < len(relabeled) - 1:  # 后面还有步 → 该步必然发起了检索
            info = infos[i] if i < len(infos) else "<information></information>"
            env = info + ("\n\n" + state.render() if args.include_budget else "")
            messages.append({"role": "user", "content": env})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, help="rollout_eval.py 产出的 JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=float, default=0.3, help="教学区间相对宽度 w")
    parser.add_argument("--p-smooth", type=float, default=0.1)
    parser.add_argument("--only-format-ok", action="store_true",
                        help="只保留全步格式正确的轨迹(教师数据建议开)")
    parser.add_argument("--include-budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimate-mode", choices=["before_action", "after_action", "none"],
                        default="before_action")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = SFTConfig(width=args.width, p_smooth=args.p_smooth)
    kept, skipped = 0, 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.trajectories.open(encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            if args.max_samples and kept >= args.max_samples:
                break
            msgs = build_messages(json.loads(line), config, args)
            if msgs is None:
                skipped += 1
                continue
            fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}, skipped {skipped} -> {out_path}")


if __name__ == "__main__":
    main()
