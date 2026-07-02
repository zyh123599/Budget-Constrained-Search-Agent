#!/usr/bin/env python3
"""服务器落地冒烟测试:不碰 GPU,秒级跑完,验证软件面一切就绪。

检查项:
1. budget_agent 可导入,关键防 hack 性质成立;
2. 一条样例轨迹能被解析并打分(reward 全链路);
3. 检索服务可达且返回格式正确(--retrieval-url,服务未启动则跳过);
4. 训练栈依赖(vllm/verl/datasets/transformers)可导入(缺失则提示)。

用法:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --retrieval-url http://127.0.0.1:8000/retrieve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL, SKIP = "[PASS]", "[FAIL]", "[SKIP]"
failures = 0


def check(name: str, fn) -> None:
    global failures
    try:
        detail = fn()
        print(f"{PASS} {name}" + (f" — {detail}" if detail else ""))
    except SkipCheck as e:
        print(f"{SKIP} {name} — {e}")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"{FAIL} {name} — {type(e).__name__}: {e}")


class SkipCheck(Exception):
    pass


def core_properties():
    from budget_agent.winkler import winkler_score

    sharp = winkler_score(500, 700, 600)
    wide = winkler_score(0, 10**6, 600)
    assert sharp < wide, "Winkler 防刷宽性质被破坏"
    return f"winkler(sharp)={sharp:.0f} < winkler(wide)={wide:.0f}"


def reward_pipeline():
    from budget_agent.budget_state import BudgetSpec
    from budget_agent.verl_reward import score_trajectory

    traj = (
        "<think>need the director</think>"
        "<estimate> low=400 high=900 p=0.8 </estimate>"
        "<search> Inception director </search>"
        "<information>Doc 1(Title: Inception) directed by Christopher Nolan</information>"
        "<think>got it</think>"
        "<estimate> low=30 high=120 p=0.95 </estimate>"
        "<answer> Christopher Nolan </answer>"
    )
    score = score_trajectory(
        traj, ["Christopher Nolan"], BudgetSpec(2000, 3, "2k×3"),
        step_tokens=[420, 60], step_searches=[1, 0],
    )
    assert score > 0, f"正确轨迹得分应为正,得到 {score}"
    return f"score={score:.3f}"


def retrieval_service(url: str | None):
    if not url:
        raise SkipCheck("未提供 --retrieval-url")
    import requests

    try:
        r = requests.post(url, json={"queries": ["capital of France"], "topk": 3,
                                     "return_scores": True}, timeout=30)
        r.raise_for_status()
    except requests.ConnectionError as e:
        raise SkipCheck(f"服务未启动({e.__class__.__name__});先跑 setup_retrieval.sh") from e
    docs = r.json()["result"][0]
    assert docs, "检索返回空结果"
    return f"top-1 字段: {list(docs[0].keys())}"


def training_stack():
    missing = []
    for mod in ("torch", "transformers", "datasets", "vllm", "verl"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SkipCheck(f"缺依赖 {missing}(核心库不受影响;训练前装齐)")
    import torch

    return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-url", default=None)
    args = parser.parse_args()

    check("核心库导入 + Winkler 防刷宽性质", core_properties)
    check("奖励全链路(解析→hindsight→打分)", reward_pipeline)
    check("检索服务连通性", lambda: retrieval_service(args.retrieval_url))
    check("训练栈依赖", training_stack)

    print(f"\n{'全部通过,可以开跑' if failures == 0 else f'{failures} 项失败,先修再跑'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
