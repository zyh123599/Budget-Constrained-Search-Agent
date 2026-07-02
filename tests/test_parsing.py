from budget_agent.parsing import Action, parse_estimate, parse_step


WELL_FORMED = """<think>I need the capital first.</think>
<estimate> low=200 high=800 p=0.7 </estimate>
<search> capital of France </search>"""


def test_well_formed_step():
    out = parse_step(WELL_FORMED)
    assert out.format_ok and not out.errors
    assert out.action is Action.SEARCH
    assert out.content == "capital of France"
    assert out.estimate.low == 200 and out.estimate.high == 800
    assert out.estimate.p_success == 0.7
    assert out.think == "I need the capital first."


def test_bare_triplet_estimate():
    est = parse_estimate("120, 400, 0.6")
    assert (est.low, est.high, est.p_success) == (120, 400, 0.6)


def test_named_estimate_out_of_order():
    est = parse_estimate("p=0.9 high=500 low=100")
    assert (est.low, est.high, est.p_success) == (100, 500, 0.9)


def test_invalid_estimates_rejected():
    assert parse_estimate("low=500 high=100 p=0.5") is None  # low > high
    assert parse_estimate("low=100 high=500 p=1.5") is None  # p 越界
    assert parse_estimate("low=-10 high=500 p=0.5") is None  # 负成本
    assert parse_estimate("about 300 tokens") is None        # 数字不足


def test_missing_estimate_flags_format_error():
    out = parse_step("<think>hm</think><answer>Paris</answer>")
    assert not out.format_ok
    assert "missing <estimate>" in out.errors
    assert out.action is Action.ANSWER  # 动作照常解析,惩罚在 reward 层


def test_missing_action_flags_format_error():
    out = parse_step("<think>hm</think><estimate>100 200 0.5</estimate>")
    assert not out.format_ok
    assert out.action is None


def test_all_action_tags():
    cases = {
        "<estimate>1 2 0.5</estimate><pivot>try composer instead</pivot>": Action.PIVOT,
        "<estimate>1 2 0.5</estimate><answer>Beijing</answer>": Action.ANSWER,
        "<estimate>1 2 0.5</estimate><ask>Which year?</ask>": Action.ASK,
        "<estimate>1 2 0.5</estimate><stop/>": Action.STOP,
        "<estimate>1 2 0.5</estimate><stop>budget too low</stop>": Action.STOP,
    }
    for text, expected in cases.items():
        out = parse_step(text)
        assert out.action is expected, text


def test_multiple_actions_takes_first_and_warns():
    out = parse_step(
        "<estimate>1 2 0.5</estimate><search>q1</search><answer>Paris</answer>"
    )
    assert out.action is Action.SEARCH
    assert any("multiple action tags" in e for e in out.errors)


def test_empty_search_query_is_format_error():
    out = parse_step("<estimate>1 2 0.5</estimate><search>  </search>")
    assert not out.format_ok
    assert any("empty <search>" in e for e in out.errors)


def test_last_estimate_wins():
    out = parse_step(
        "<estimate>1 2 0.5</estimate><think>revise</think>"
        "<estimate>100 300 0.8</estimate><answer>x</answer>"
    )
    assert out.estimate.low == 100 and out.estimate.p_success == 0.8


def test_action_inside_think_is_ignored():
    out = parse_step(
        "<think>I could try <search>capital of France</search> but I already know</think>"
        "<estimate> low=50 high=100 p=0.9 </estimate>"
        "<answer>Paris</answer>"
    )
    assert out.action is Action.ANSWER
    assert out.content == "Paris"


def test_scientific_notation_estimate():
    est = parse_estimate("low=1e3 high=5e3 p=0.7")
    assert est is not None
    assert est.low == 1000.0 and est.high == 5000.0 and est.p_success == 0.7


def test_scientific_notation_bare_triplet():
    est = parse_estimate("1.5e2 3e3 0.8")
    assert est is not None
    assert est.low == 150.0 and est.high == 3000.0 and est.p_success == 0.8
