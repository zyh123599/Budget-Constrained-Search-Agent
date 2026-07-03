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
from rollout_eval import Traj, _initial_prompt, _make_injector, _passages_to_string, _dump, _rollout


class _FakeLLM:
    """按脚本回放生成结果的假 vLLM:锁定 rollout 控制流,不碰 GPU。"""

    def __init__(self, script: dict[str, list[tuple[str, int]]]):
        self.script = script  # prompt 前缀无关,按调用顺序出队

    def generate(self, prompts, sampling):
        outs = []
        for _ in prompts:
            text, ntok = self.script["calls"].pop(0)
            out = SimpleNamespace(outputs=[SimpleNamespace(
                text=text, token_ids=list(range(ntok)))])
            outs.append(out)
        return outs


class _FakeSampling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _rollout_args(**over):
    base = dict(temperature=0.0, max_tokens_per_turn=512, max_turns=4,
                on_exhaust="force_answer", force_answer_tokens=64,
                prompt_style="searchr1", include_budget=True,
                retrieval_url="http://fake", topk=3)
    base.update(over)
    return SimpleNamespace(**base)

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


def test_on_exhaust_token_budget_forces_answer():
    """token 预算耗尽 → 跳过检索、下一轮强制作答(budget-clipped baseline 协议)。"""
    step = "<estimate>10 20 0.5</estimate><search> big question </search>"
    llm = _FakeLLM({"calls": [(step, 60), ("Paris </answer>", 5)]})  # 60 > 预算 50
    t = Traj(question="q", truths=["Paris"], spec=BudgetSpec(50, 5, "50×5"), running="p")
    _rollout(llm, _FakeSampling, [t], _rollout_args())
    assert t.done and t.final_action == "forced_answer"
    assert t.answer == "Paris"
    assert t.step_searches == [1, 0]  # 检索被记账但未执行(预算死于检索中途)
    assert t.state.violated  # token 超支如实反映


def test_on_exhaust_denies_search_over_quota(monkeypatch):
    """检索配额用完后再检索 → 拒绝执行并强制作答,配额不再累加。"""
    import rollout_eval as re_mod

    monkeypatch.setattr(re_mod, "_batch_retrieve", lambda url, qs, k: ["Doc 1(Title: t) x"] * len(qs))
    search_step = "<estimate>10 20 0.5</estimate><search> q </search>"
    llm = _FakeLLM({"calls": [
        (search_step, 5),          # 第 1 轮:合法检索,用掉唯一配额
        (search_step, 5),          # 第 2 轮:再检索 → 拒绝,转强制作答
        ("Paris </answer>", 3),    # 强制作答轮
    ]})
    t = Traj(question="q", truths=["Paris"], spec=BudgetSpec(1000, 1, "1k×1"), running="p")
    re_mod._rollout(llm, _FakeSampling, [t], _rollout_args())
    assert t.done and t.final_action == "forced_answer" and t.answer == "Paris"
    assert t.step_searches == [1, 0, 0]  # 第二次检索被拒,未计入
    assert t.state.used_searches == 1 and not t.state.violated


def test_on_exhaust_none_keeps_natural_termination(monkeypatch):
    """默认 none:超支照常发生(violation 是被测量的行为,不被外部拯救)。"""
    import rollout_eval as re_mod

    monkeypatch.setattr(re_mod, "_batch_retrieve", lambda url, qs, k: ["Doc 1(Title: t) x"] * len(qs))
    llm = _FakeLLM({"calls": [
        ("<estimate>10 20 0.5</estimate><search> q </search>", 60),
        ("<estimate>5 10 0.9</estimate><answer> Paris </answer>", 10),
    ]})
    t = Traj(question="q", truths=["Paris"], spec=BudgetSpec(50, 5, "50×5"), running="p")
    re_mod._rollout(llm, _FakeSampling, [t], _rollout_args(on_exhaust="none"))
    assert t.final_action == "answer" and t.answer == "Paris"
    assert t.state.violated  # 自然超支,如实记录


def _sft_config():
    from budget_agent.sft import SFTConfig

    return SFTConfig()
