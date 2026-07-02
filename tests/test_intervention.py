import random

from budget_agent.intervention import (
    noisy_estimate,
    oracle_estimate,
    render_estimate,
    splice_estimate,
)
from budget_agent.parsing import parse_estimate, parse_step
from budget_agent.winkler import interval_covers


def test_oracle_covers_truth():
    e = oracle_estimate(1000, width=0.2)
    assert interval_covers(e.low, e.high, 1000)
    assert e.high - e.low == 200


def test_zero_sigma_equals_oracle():
    e = noisy_estimate(1000, sigma=0.0, rng=random.Random(0))
    o = oracle_estimate(1000)
    assert (e.low, e.high) == (o.low, o.high)


def test_noise_degrades_coverage_monotonically():
    rng = random.Random(42)
    coverage = []
    for sigma in (0.0, 0.5, 2.0):
        hits = sum(
            interval_covers(*(lambda e: (e.low, e.high))(noisy_estimate(1000, sigma, rng)), 1000)
            for _ in range(500)
        )
        coverage.append(hits / 500)
    assert coverage[0] == 1.0
    assert coverage[0] > coverage[1] > coverage[2]


def test_rendered_estimate_roundtrips():
    e = oracle_estimate(1234, p_success=0.7)
    parsed = parse_estimate(render_estimate(e))
    assert parsed is not None and parsed.p_success == 0.7


def test_splice_replaces_model_estimate():
    partial = "<think>hmm</think>\n<estimate> low=1 high=2 p=0.9"  # 截断于闭合前
    spliced = splice_estimate(partial, oracle_estimate(800))
    full = spliced + "<search>q</search>"
    out = parse_step(full)
    assert out.format_ok
    assert interval_covers(out.estimate.low, out.estimate.high, 800)
    assert "low=1 high=2" not in spliced


def test_splice_removes_closed_estimates_too():
    partial = "<think>a</think><estimate>1 2 0.5</estimate><think>b</think>"
    spliced = splice_estimate(partial, oracle_estimate(500))
    assert spliced.count("<estimate>") == 1
    assert parse_estimate("<estimate>" + spliced.split("<estimate>")[1]) is not None
