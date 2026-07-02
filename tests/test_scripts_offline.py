"""脚本层离线契约测试:锁定 rollout_eval → make_sft_data / eval_budget_sweep
之间的 JSONL 数据协议,不碰 GPU / 网络 / vLLM。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from budget_agent.budget_state import BudgetSpec
from budget_agent.parsing import Action, parse_step

from make_sft_data import build_messages
from rollout_eval import Traj, _initial_prompt, _make_injector, _passages_to_string, _dump

SFT_ARGS = SimpleNamespace(
    only_format_ok=True, include_budget=True, estimate_mode="before_action"
)

TRAJ = {
    "solution_str": (
        "<think>need the director</think>"
        "<estimate> low=400 high=900 p=0.8 </estimate>"
        "<search> Inception director </search>"
        "\n\n<information>Doc 1(Title: Inception) directed by Christopher Nolan</information>\n\n"
        "<budget>\nRemaining token budget: 1580 / 2000\nRemaining search calls: 2 / 3\n"
        "Steps taken so far: 1\nBudget tier: 2k×3\n</budget>\n\n"
        "<think>got it</think>"
        "<estimate> low=30 high=120 p=0.95 </estimate>"
        "<answer> Christopher Nolan </answer>"
    ),
    "ground_truths": ["Christopher Nolan"],
    "step_tokens": [420, 60],
    "step_searches": [1, 0],
    "token_budget": 2000,
    "search_budget": 3,
    "tier": "2k×3",
    "question": "Who directed Inception?",
    "answer": "Christopher Nolan",
}


def test_sft_messages_roundtrip():
    msgs = build_messages(dict(TRAJ), _sft_config(), SFT_ARGS)
    assert msgs is not None
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant"]
    # 教学估计以 hindsight 后缀和为中心:第 1 步 c_true = 420+500+60 = 980
    step1 = parse_step(msgs[1]["content"])
    assert step1.format_ok and step1.action is Action.SEARCH
    assert step1.estimate.low <= 980 <= step1.estimate.high
    # 成功轨迹 → p̂ 高
    assert step1.estimate.p_success >= 0.85
    # env 轮带最新 <budget> 块
    assert "<information>" in msgs[2]["content"] and "<budget>" in msgs[2]["content"]


def test_sft_skips_malformed_when_filtering():
    bad = dict(TRAJ, solution_str="<answer>Nolan</answer>")  # 缺 <estimate>
    assert build_messages(bad, _sft_config(), SFT_ARGS) is None


def test_rollout_dump_matches_eval_contract(tmp_path):
    spec = BudgetSpec(2000, 3, "2k×3")
    t = Traj(question="q", truths=["a"], spec=spec, running="prompt")
    t.solution = TRAJ["solution_str"]
    t.step_tokens, t.step_searches = [420, 60], [1, 0]
    t.state.charge(420, 1)
    t.state.charge(60, 0)
    t.done, t.final_action, t.answer = True, "answer", "Christopher Nolan"

    out = tmp_path / "run.jsonl"
    _dump([t], out)
    row = json.loads(out.read_text())
    # eval_budget_sweep.py 必需字段
    for key in ("solution_str", "ground_truths", "step_tokens", "step_searches",
                "token_budget", "search_budget", "tier"):
        assert key in row, key
    assert row["used_tokens"] == 480 and row["violated"] is False


def test_initial_prompt_styles():
    spec = BudgetSpec(2000, 3, "2k×3")
    budget_args = SimpleNamespace(prompt_style="budget", include_budget=True,
                                  estimate_mode="before_action")
    p = _initial_prompt("Who?", spec, budget_args)
    assert "<budget>" in p and "<estimate>" in p
    searchr1_args = SimpleNamespace(prompt_style="searchr1")
    p = _initial_prompt("Who?", spec, searchr1_args)
    assert "<budget>" not in p and "<search>" in p and "Question: Who?" in p


def test_injector_oracle_and_noise():
    import random

    spec = BudgetSpec(2000, 3, "2k×3")
    t = Traj(question="q", truths=["a"], spec=spec, running="p")
    t.ref_remaining = [980.0, 560.0]

    oracle = _make_injector(SimpleNamespace(intervene="oracle", sigma=0.0), random.Random(0))
    e = oracle(t, 0)
    assert e.low <= 980 <= e.high
    e_last = oracle(t, 5)  # 越界步取参考序列末位
    assert e_last.low <= 560 <= e_last.high

    noise = _make_injector(SimpleNamespace(intervene="noise", sigma=1.0), random.Random(0))
    assert noise(t, 0) is not None


def test_passages_formatting_robust():
    docs = [
        {"document": {"contents": '"Inception"\nA 2010 film by Nolan.'}},
        {"contents": "Bare dict\nstill works"},
    ]
    s = _passages_to_string(docs)
    assert "Doc 1(Title: Inception)" in s and "Doc 2(Title: Bare dict)" in s


def _sft_config():
    from budget_agent.sft import SFTConfig

    return SFTConfig()
