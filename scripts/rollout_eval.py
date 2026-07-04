#!/usr/bin/env python3
"""多轮 rollout 轨迹生成器:vLLM + 本地检索服务,产出评估/SFT 用轨迹 JSONL。

这是管线中"生成轨迹"的一环(eval_budget_sweep.py 只重放不生成):

    prepare_data.py(评测网格 parquet)
        → rollout_eval.py(本脚本:批量多轮 rollout,逐步记账)
        → runs/*.jsonl
        → eval_budget_sweep.py(四大生死指标)/ make_sft_data.py(SFT 数据)

四种用途,由 --prompt-style / --intervene / --on-exhaust 切换:
1. 本方法评估(默认):预算增广 prompt,每轮 <information> 后重注 <budget> 块;
2. Search-R1 baseline 复现(--prompt-style searchr1):原版指令,巨额预算兜底,
   仅用于对齐上游 EM(Gate 1);
3. budget-clipped baseline(--on-exhaust force_answer):baseline 不感知预算,
   外部硬截断——token 耗尽或检索配额用完时拒绝新检索并强制作答,扫出它在各
   预算档的真实成功率–成本曲线(论文 Pareto 对比的公平基线);
4. 干预实验(§5.4,--intervene oracle|noise):两遍法——第一遍自然 rollout 记录
   实际后缀成本作参考真值,第二遍在 </estimate> 处截断、注入 oracle/加噪估计后
   续写动作。oracle 用第一遍的实现成本近似(干预后轨迹会分叉,这是文献通行近似,
   写作时需说明)。

用法(训练机,searchr1 环境):
    python scripts/rollout_eval.py \
        --model Qwen/Qwen3-4B --data data/eval_interp.parquet \
        --out runs/qwen3_4b_interp.jsonl --retrieval-url http://127.0.0.1:8000/retrieve

    # baseline 复现(Gate 1)
    python scripts/rollout_eval.py \
        --model PeterGriffinJin/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
        --data data/eval_nq_test.parquet --prompt-style searchr1 --no-chat \
        --out runs/searchr1_baseline_nq.jsonl --limit 500

    # 干预实验(冻结策略 = 训练后 checkpoint)
    python scripts/rollout_eval.py --model <ckpt> --data data/eval_interp.parquet \
        --intervene noise --sigma 0.5 --out runs/intervene_s05.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_agent.budget_state import BudgetSpec, BudgetState
from budget_agent.hindsight import StepCost, remaining_costs
from budget_agent.intervention import noisy_estimate, oracle_estimate, splice_estimate
from budget_agent.parsing import SEARCH_ACTIONS, TERMINAL_ACTIONS, parse_step
from budget_agent.prompts import render_prompt
from budget_agent.qa_metrics import exact_match

# Search-R1 原版指令(与其 infer.py 一致),baseline 复现时使用
_SEARCHR1_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by \
<search> query </search> and it will return the top searched results between \
<information> and </information>. You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer \
inside <answer> and </answer>, without detailed illustrations. For example, \
<answer> Beijing </answer>. Question: {question}\n"""

_STOP_STRINGS = ["</search>", "</pivot>", "</answer>", "</ask>", "<stop/>", "</stop>"]


@dataclass
class Traj:
    """一条进行中的轨迹:running 是喂给 vLLM 的全文,solution 是初始 prompt 之后的部分。"""

    question: str
    truths: list[str]
    spec: BudgetSpec
    running: str
    solution: str = ""
    step_tokens: list[int] = field(default_factory=list)
    step_searches: list[int] = field(default_factory=list)
    done: bool = False
    forcing: bool = False  # 预算耗尽,下一轮强制作答(--on-exhaust force_answer)
    final_action: str = ""
    answer: str = ""
    state: BudgetState = None  # type: ignore[assignment]
    ref_remaining: list[float] | None = None  # 干预模式:第一遍的后缀成本参考

    def __post_init__(self) -> None:
        self.state = BudgetState(self.spec)


def _passages_to_string(docs: list) -> str:
    """Search-R1 检索结果格式化:Doc i(Title: ...) 正文。对字段缺失鲁棒。"""
    out = []
    for i, item in enumerate(docs):
        content = item.get("document", item).get("contents", "") if isinstance(item, dict) else str(item)
        lines = content.split("\n")
        title, text = lines[0].strip('"'), "\n".join(lines[1:])
        out.append(f"Doc {i + 1}(Title: {title}) {text}")
    return "\n".join(out)


def _batch_retrieve(url: str, queries: list[str], topk: int) -> list[str]:
    import requests

    resp = requests.post(url, json={"queries": queries, "topk": topk, "return_scores": True}, timeout=120)
    resp.raise_for_status()
    return [_passages_to_string(r) for r in resp.json()["result"]]


def _initial_prompt(question: str, spec: BudgetSpec, args) -> str:
    if args.prompt_style == "searchr1":
        return _SEARCHR1_PROMPT.format(question=question)
    return render_prompt(
        question,
        BudgetState(spec),
        include_budget=args.include_budget,
        estimate_mode=args.estimate_mode,
    )


def _apply_chat(tokenizer, prompt: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:  # Qwen3:关闭原生思考模式,协议内的 <think> 由指令约定
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def _make_injector(args, rng: random.Random):
    """干预估计构造器:step_idx + 参考后缀成本 → 注入的 Estimate。"""

    def inject(traj: Traj, step_idx: int):
        ref = traj.ref_remaining or [float(traj.spec.token_budget)]
        c_true = ref[min(step_idx, len(ref) - 1)]
        if args.intervene == "oracle":
            return oracle_estimate(c_true)
        return noisy_estimate(c_true, args.sigma, rng)

    return inject


_FORCE_ANSWER_ENV = (
    "\n\n<information>Budget exhausted. You must stop searching and provide "
    "your best final answer now.</information>\n\n<answer>"
)


def _force_answers(llm, sampling_cls, trajs: list[Traj], args) -> None:
    """预算截断的强制作答轮:续写已预开的 <answer> 标签(公平 baseline 协议)。"""
    forcing = [t for t in trajs if t.forcing and not t.done]
    if not forcing:
        return
    sampling = sampling_cls(
        temperature=args.temperature, max_tokens=args.force_answer_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    )
    outs = llm.generate([t.running for t in forcing], sampling)
    for t, out in zip(forcing, outs):
        text, ntok = out.outputs[0].text, len(out.outputs[0].token_ids)
        t.running += text
        t.solution += text
        t.step_tokens.append(ntok)
        t.step_searches.append(0)
        t.state.charge(ntok, 0)
        t.answer = text.split("</answer>")[0].strip()
        t.done, t.final_action = True, "forced_answer"


def _rollout(llm, sampling_cls, trajs: list[Traj], args, inject=None) -> None:
    """批量多轮 rollout,就地推进 trajs。inject 非空时启用 §5.4 干预协议。"""
    normal = sampling_cls(
        temperature=args.temperature, max_tokens=args.max_tokens_per_turn,
        stop=_STOP_STRINGS, include_stop_str_in_output=True,
    )
    till_estimate = sampling_cls(
        temperature=args.temperature, max_tokens=args.max_tokens_per_turn,
        stop=["</estimate>"] + _STOP_STRINGS, include_stop_str_in_output=True,
    )

    for turn in range(args.max_turns):
        _force_answers(llm, sampling_cls, trajs, args)
        active = [t for t in trajs if not t.done and not t.forcing]
        if not active:
            if any(t.forcing and not t.done for t in trajs):
                continue
            break

        if inject is not None:
            # 干预:先生成到 </estimate>,清掉模型估计、注入干预值,再续写动作
            outs = llm.generate([t.running for t in active], till_estimate)
            pre_tokens = []
            for t, out in zip(active, outs):
                text, ntok = out.outputs[0].text, len(out.outputs[0].token_ids)
                spliced = splice_estimate(text, inject(t, turn))
                t.running += spliced
                t.solution += spliced
                pre_tokens.append(ntok)
            outs = llm.generate([t.running for t in active], normal)
            texts = [o.outputs[0].text for o in outs]
            ntoks = [p + len(o.outputs[0].token_ids) for p, o in zip(pre_tokens, outs)]
        else:
            outs = llm.generate([t.running for t in active], normal)
            texts = [o.outputs[0].text for o in outs]
            ntoks = [len(o.outputs[0].token_ids) for o in outs]

        pending: list[tuple[Traj, str]] = []  # (traj, query) 待检索
        for t, text, ntok in zip(active, texts, ntoks):
            t.running += text
            t.solution += text
            step = parse_step(t.solution.rsplit("</information>", 1)[-1])
            searched = int(step.action in SEARCH_ACTIONS and bool(step.content))
            # 预算截断协议:检索预算已尽时拒绝执行新检索(生成 token 照常记账)
            deny_search = (args.on_exhaust == "force_answer" and searched
                           and t.state.remaining_searches <= 0)
            if deny_search:
                searched = 0
            t.step_tokens.append(ntok)
            t.step_searches.append(searched)
            t.state.charge(ntok, searched)
            exhausted = (args.on_exhaust == "force_answer"
                         and (t.state.remaining_tokens <= 0 or deny_search))

            if step.action in TERMINAL_ACTIONS:
                t.done, t.final_action = True, step.action.value
                if step.action.value == "answer":
                    t.answer = step.content
            elif exhausted:  # 预算耗尽:下一轮强制作答(公平的 budget-clipped baseline)
                t.forcing = True
                t.running += _FORCE_ANSWER_ENV
                t.solution += _FORCE_ANSWER_ENV
            elif searched:
                t.final_action = step.action.value
                pending.append((t, step.content))
            else:  # 无动作/空 query:格式失败,终止
                t.done, t.final_action = True, "format_failure"

        if pending:
            infos = _batch_retrieve(args.retrieval_url, [q for _, q in pending], args.topk)
            for (t, _), info in zip(pending, infos):
                env = f"\n\n<information>{info}</information>\n\n"
                if args.prompt_style == "budget" and args.include_budget:
                    env += t.state.render() + "\n\n"
                t.running += env
                t.solution += env

    _force_answers(llm, sampling_cls, trajs, args)  # 兜底:最后一轮才耗尽预算的
    for t in trajs:  # 超轮数未终止的轨迹
        if not t.done:
            t.done, t.final_action = True, "max_turns"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="prepare_data.py 产出的 parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--prompt-style", choices=["budget", "searchr1"], default="budget")
    parser.add_argument("--include-budget", action=argparse.BooleanOptionalAction, default=True,
                        help="消融 C 用 --no-include-budget")
    parser.add_argument("--estimate-mode", choices=["before_action", "after_action", "none"],
                        default="before_action", help="消融 D/E 对应 after_action/none")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True,
                        help="套 chat template;Search-R1 基座检查点用 --no-chat")
    parser.add_argument("--intervene", choices=["none", "oracle", "noise"], default="none")
    parser.add_argument("--sigma", type=float, default=0.5, help="加噪干预的相对噪声幅度")
    parser.add_argument("--on-exhaust", choices=["none", "force_answer"], default="none",
                        help="force_answer=预算耗尽时强制作答(budget-clipped baseline 的 "
                             "Pareto sweep 用;本方法默认 none,自主止损是被评估的能力)")
    parser.add_argument("--force-answer-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens-per-turn", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条(冒烟/子采样)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                        help="检索索引 fp16 分片共卡时留 ~16GB 余量;索引在 CPU 可提到 0.85")
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)

    trajs: list[Traj] = []
    for _, row in df.iterrows():
        info = row["extra_info"]
        spec = BudgetSpec(int(info["token_budget"]), int(info["search_budget"]), str(info["tier"]))
        prompt = _initial_prompt(info["question"], spec, args)
        if args.chat:
            prompt = _apply_chat(tokenizer, prompt)
        gt = row["reward_model"]["ground_truth"]["target"]
        truths = [str(x) for x in (gt.tolist() if hasattr(gt, "tolist") else gt)]
        trajs.append(Traj(question=str(info["question"]), truths=truths, spec=spec, running=prompt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.intervene != "none":
        # 第一遍:自然 rollout(对照组),回填后缀成本作参考真值
        _rollout(llm, SamplingParams, trajs, args)
        control_path = out_path.with_suffix(".control.jsonl")
        _dump(trajs, control_path)
        print(f"control pass -> {control_path}")
        fresh: list[Traj] = []
        for t in trajs:
            costs = [StepCost(tok, s) for tok, s in zip(t.step_tokens, t.step_searches)]
            nt = Traj(question=t.question, truths=t.truths, spec=t.spec,
                      running=t.running[: len(t.running) - len(t.solution)])
            nt.ref_remaining = remaining_costs(costs) if costs else None
            fresh.append(nt)
        trajs = fresh
        _rollout(llm, SamplingParams, trajs, args, inject=_make_injector(args, random.Random(args.seed)))
    else:
        _rollout(llm, SamplingParams, trajs, args)

    _dump(trajs, out_path)

    n = len(trajs)
    em = sum(exact_match(t.answer, t.truths) if t.answer else 0.0 for t in trajs) / max(n, 1)
    viol = sum(t.state.violated for t in trajs) / max(n, 1)
    print(f"wrote {n} trajectories -> {out_path}")
    print(f"EM={em:.4f}  violation={viol:.4f}  "
          f"(完整四大指标: python scripts/eval_budget_sweep.py {out_path})")


def _dump(trajs: list[Traj], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps({
                "solution_str": t.solution,
                "ground_truths": t.truths,
                "step_tokens": t.step_tokens,
                "step_searches": t.step_searches,
                "token_budget": t.spec.token_budget,
                "search_budget": t.spec.search_budget,
                "tier": t.spec.tier,
                "question": t.question,
                "final_action": t.final_action,
                "answer": t.answer,
                "used_tokens": t.state.used_tokens,
                "used_searches": t.state.used_searches,
                "violated": t.state.violated,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
