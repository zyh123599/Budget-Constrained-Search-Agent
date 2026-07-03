#!/usr/bin/env python3
"""检索侧隔离测试:answer hit rate,把"EM 偏低"定位到检索侧或生成侧。

协议:抽 N 条问题(默认 200),只跑检索不跑生成,统计 gold answer(规范化后)
以子串形式命中 top-k 段落的比例。判读(NQ + wiki-18 + e5 + top-3 配置下):

    hit rate ≥ 0.65   检索侧健康,EM 差异来自模型/检查点/解码侧
    0.60 – 0.65       灰区,对照上游复现记录再判
    < 0.60            检索侧有病:查 query 前缀、索引类型(Flat vs 近似)、语料版本

输入直接复用 prepare_data.py 产出的 parquet(question + golden answers 都在),
不重复依赖 datasets。

用法(任一装了 pandas+requests 的环境):
    python scripts/retrieval_probe.py --data data/eval_nq_test.parquet --limit 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_agent.qa_metrics import normalize_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="prepare_data.py 产出的 parquet")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--show-misses", type=int, default=5,
                        help="打印前 N 条未命中样本(人工 spot check)")
    args = parser.parse_args()

    import pandas as pd
    import requests

    df = pd.read_parquet(args.data)
    # 逐档展开的评测网格里同一问题会重复,按问题去重
    seen: set[str] = set()
    items: list[tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        q = str(row["extra_info"]["question"])
        if q in seen:
            continue
        seen.add(q)
        gt = row["reward_model"]["ground_truth"]["target"]
        answers = [str(x) for x in (gt.tolist() if hasattr(gt, "tolist") else gt)]
        items.append((q, answers))
        if len(items) >= args.limit:
            break

    hits, rank_hits, misses = 0, Counter(), []
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        resp = requests.post(args.retrieval_url, json={
            "queries": [q for q, _ in batch], "topk": args.topk, "return_scores": True,
        }, timeout=300)
        resp.raise_for_status()
        for (q, answers), docs in zip(batch, resp.json()["result"]):
            texts = [
                normalize_answer((d.get("document", d) or {}).get("contents", ""))
                for d in docs
            ]
            norm_answers = [normalize_answer(a) for a in answers]
            hit_rank = next(
                (i for i, t in enumerate(texts) if any(a and a in t for a in norm_answers)),
                None,
            )
            if hit_rank is not None:
                hits += 1
                rank_hits[hit_rank + 1] += 1
            elif len(misses) < args.show_misses:
                misses.append((q, answers))

    n = len(items)
    rate = hits / n if n else 0.0
    print(f"\nanswer hit rate @ top-{args.topk}: {rate:.3f}  ({hits}/{n})")
    for rank in sorted(rank_hits):
        print(f"  首次命中于 Doc {rank}: {rank_hits[rank]}")
    if misses:
        print(f"\n未命中样例(前 {len(misses)} 条,人工核对是否确实不该命中):")
        for q, answers in misses:
            print(f"  Q: {q}\n     gold: {answers}")

    print()
    if rate >= 0.65:
        print("判读:检索侧健康(≥0.65)。EM 差异来自模型/检查点/解码侧,可冻结 baseline。")
    elif rate >= 0.60:
        print("判读:灰区(0.60–0.65)。对照上游复现记录;优先查语料版本与 top-k。")
    else:
        print("判读:检索侧有病(<0.60)。按序查:query 前缀(e5 需 'query: ')、"
              "索引类型(必须 Flat 精确)、语料是否 wiki-18 标配 dump。")


if __name__ == "__main__":
    main()
