#!/usr/bin/env python3
"""构建带预算增广的训练/评测 parquet(§5.1 数据协议)。

数据源与 Search-R1 对齐:RUC-NLPIR/FlashRAG_datasets(HuggingFace)。
- 训练:nq + hotpotqa
- OOD 评测:2wikimultihopqa / musique / bamboogle

每条样本:随机采样(或固定)一个预算档 → 用 budget_agent.prompts 渲染带
<budget> 块的 prompt → ground truth 与预算合同写入 verl 的
reward_model / extra_info 字段(与 budget_agent.verl_reward.compute_score
的 extra_info 约定一致)。

用法(训练机上,需 pip install datasets pandas pyarrow):
    python scripts/prepare_data.py --datasets nq hotpotqa --split train \
        --budget-mode sample --out data/train.parquet
    python scripts/prepare_data.py --datasets bamboogle --split test \
        --budget-mode grid --grid interp --out data/eval_interp.parquet
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_agent.budget_sampler import eval_budget_grid, sample_train_budget
from budget_agent.budget_state import BudgetState
from budget_agent.prompts import render_prompt

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"


def build_row(question: str, answers: list[str], source: str, idx: int, spec, split: str) -> dict:
    prompt = render_prompt(question, BudgetState(spec))
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "token_budget": spec.token_budget,
            "search_budget": spec.search_budget,
            "tier": spec.tier,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="FlashRAG 子集名,如 nq hotpotqa 2wikimultihopqa musique bamboogle")
    parser.add_argument("--split", default="train")
    parser.add_argument("--budget-mode", choices=["sample", "grid"], default="sample",
                        help="sample=每样本随机训练档(训练);grid=按评测网格逐档展开(评测)")
    parser.add_argument("--grid", choices=["train", "interp", "extrap"], default="interp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import datasets  # 延迟导入:仅训练机需要
    import pandas as pd

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for name in args.datasets:
        ds = datasets.load_dataset(FLASHRAG_REPO, name, split=args.split)
        if args.max_per_dataset:
            ds = ds.select(range(min(args.max_per_dataset, len(ds))))
        for idx, ex in enumerate(ds):
            question = ex["question"]
            answers = [str(a) for a in ex["golden_answers"]]
            if args.budget_mode == "sample":
                specs = [sample_train_budget(rng)]
            else:
                specs = eval_budget_grid(args.grid)
            for spec in specs:
                rows.append(build_row(question, answers, name, idx, spec, args.split))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
