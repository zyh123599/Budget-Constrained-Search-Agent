import pytest

from budget_agent.budget_state import BudgetSpec, BudgetState


def test_spec_validation():
    with pytest.raises(ValueError):
        BudgetSpec(token_budget=0, search_budget=3)


def test_scalarization():
    spec = BudgetSpec(token_budget=2000, search_budget=3)
    assert spec.scalar(search_token_equiv=500) == 2000 + 1500


def test_charge_and_remaining():
    state = BudgetState(BudgetSpec(token_budget=1000, search_budget=2, tier="1k×2"))
    state.charge(tokens=300, searches=1)
    assert state.remaining_tokens == 700
    assert state.remaining_searches == 1
    assert state.step == 1
    assert not state.violated


def test_violation_on_either_dimension():
    s1 = BudgetState(BudgetSpec(token_budget=100, search_budget=5))
    s1.charge(tokens=150)
    assert s1.token_violated and s1.violated

    s2 = BudgetState(BudgetSpec(token_budget=10_000, search_budget=1))
    s2.charge(tokens=10, searches=2)
    assert s2.search_violated and s2.violated


def test_remaining_clamped_at_zero():
    state = BudgetState(BudgetSpec(token_budget=100, search_budget=1))
    state.charge(tokens=500, searches=3)
    assert state.remaining_tokens == 0
    assert state.remaining_searches == 0


def test_negative_cost_rejected():
    state = BudgetState(BudgetSpec(token_budget=100, search_budget=1))
    with pytest.raises(ValueError):
        state.charge(tokens=-1)


def test_render_contains_budget_facts():
    state = BudgetState(BudgetSpec(token_budget=2000, search_budget=3, tier="2k×3"))
    state.charge(tokens=500, searches=1)
    block = state.render()
    assert block.startswith("<budget>") and block.endswith("</budget>")
    assert "1500 / 2000" in block
    assert "2 / 3" in block
    assert "2k×3" in block
