import pytest

from budget_agent.hindsight import StepCost
from budget_agent.parsing import parse_step
from budget_agent.sft import SFTConfig, relabel_step, relabel_trajectory
from budget_agent.winkler import interval_covers


def test_relabel_replaces_existing_estimate():
    step = "<think>x</think>\n<estimate> low=1 high=2 p=0.5 </estimate>\n<search>q</search>"
    out = relabel_step(step, c_true=1000, success=True)
    parsed = parse_step(out)
    assert parsed.format_ok
    assert interval_covers(parsed.estimate.low, parsed.estimate.high, 1000)
    assert parsed.estimate.p_success == 0.9  # 成功轨迹,平滑后 1-0.1


def test_relabel_inserts_when_missing():
    step = "<think>x</think>\n<answer>Paris</answer>"
    out = relabel_step(step, c_true=200, success=False)
    parsed = parse_step(out)
    assert parsed.format_ok
    assert parsed.estimate.p_success == 0.1
    assert out.index("<estimate>") < out.index("<answer>")


def test_relabel_dedups_multiple_estimates():
    step = (
        "<estimate>1 2 0.5</estimate><think>revise</think>"
        "<estimate>3 4 0.5</estimate><answer>x</answer>"
    )
    out = relabel_step(step, c_true=100, success=True)
    assert out.count("<estimate>") == 1


def test_trajectory_relabel_uses_suffix_sums():
    steps = [
        "<estimate>1 2 0.5</estimate><search>a</search>",
        "<estimate>1 2 0.5</estimate><answer>b</answer>",
    ]
    costs = [StepCost(100, 1), StepCost(100, 0)]  # w=500 → 真值 [700, 100]
    out = relabel_trajectory(steps, costs, success=True)
    p0, p1 = parse_step(out[0]).estimate, parse_step(out[1]).estimate
    assert interval_covers(p0.low, p0.high, 700)
    assert interval_covers(p1.low, p1.high, 100)
    assert p0.low > p1.high  # 早期步的区间整体高于末步


def test_width_config():
    cfg = SFTConfig(width=0.5)
    out = relabel_step("<answer>x</answer>", c_true=1000, success=True, config=cfg)
    est = parse_step(out).estimate
    assert est.low == 750 and est.high == 1250


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        relabel_trajectory(["a"], [], success=True)
