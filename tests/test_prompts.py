import pytest

from budget_agent.budget_state import BudgetSpec, BudgetState
from budget_agent.prompts import render_prompt

STATE = BudgetState(BudgetSpec(token_budget=2000, search_budget=3, tier="2k×3"))


def test_full_method_prompt():
    p = render_prompt("Who directed Inception?", STATE)
    assert "<budget>" in p and "2k×3" in p
    assert p.index("<estimate>") < p.index("<search>")  # 估计先于动作说明
    assert "Question: Who directed Inception?" in p
    for tag in ("<search>", "<pivot>", "<answer>", "<ask>", "<stop/>"):
        assert tag in p


def test_ablation_c_no_budget_block():
    p = render_prompt("q", STATE, include_budget=False)
    assert "<budget>" not in p
    assert "<estimate>" in p  # 估计头保留


def test_ablation_d_estimate_after_action():
    p = render_prompt("q", STATE, estimate_mode="after_action")
    # 指令中动作规则先于估计规则出现 → 生成时动作无法条件于估计
    assert p.index("Take exactly one action") < p.index("<estimate>")


def test_ablation_e_no_estimate():
    p = render_prompt("q", STATE, estimate_mode="none")
    assert "<estimate>" not in p
    assert "<budget>" in p


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        render_prompt("q", STATE, estimate_mode="oracle")
