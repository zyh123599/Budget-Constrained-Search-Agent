from budget_agent.budget_state import BudgetSpec
from budget_agent.verl_reward import compute_score, score_trajectory

TRAJ = """<think>Need the director.</think>
<estimate> low=300 high=900 p=0.8 </estimate>
<search> director of Inception </search>
<information> Inception was directed by Christopher Nolan. </information>
<think>Got it.</think>
<estimate> low=50 high=200 p=0.95 </estimate>
<answer> Christopher Nolan </answer>"""

EXTRA = {
    "token_budget": 2000,
    "search_budget": 3,
    "tier": "2k×3",
    "step_tokens": [400, 100],
    "step_searches": [1, 0],
    "progress": 1.0,
}


def test_correct_trajectory_scores_positive():
    score = compute_score("nq", TRAJ, {"target": ["Christopher Nolan"]}, EXTRA)
    assert score > 0.5


def test_wrong_answer_scores_low():
    wrong = TRAJ.replace("Christopher Nolan </answer>", "Steven Spielberg </answer>")
    assert compute_score("nq", wrong, {"target": ["Christopher Nolan"]}, EXTRA) < 0.2


def test_violation_reduces_score():
    over = dict(EXTRA, step_tokens=[1800, 1900])  # 3700 > 2000
    ok = compute_score("nq", TRAJ, {"target": ["Christopher Nolan"]}, EXTRA)
    violated = compute_score("nq", TRAJ, {"target": ["Christopher Nolan"]}, over)
    assert violated < ok


def test_ground_truth_formats():
    assert compute_score("nq", TRAJ, "Christopher Nolan", EXTRA) > 0.5
    assert compute_score("nq", TRAJ, ["Christopher Nolan"], EXTRA) > 0.5


def test_score_trajectory_without_accounting_falls_back():
    """缺 step_tokens 时用空白分词近似,仅冒烟测试路径,不应崩。"""
    budget = BudgetSpec(token_budget=2000, search_budget=3)
    s = score_trajectory(TRAJ, ["Christopher Nolan"], budget)
    assert isinstance(s, float)


def test_calibrated_stop_trajectory_not_catastrophic():
    """正确语义下的止损:小剩余消耗区间 + 低 p̂ + <stop/>。
    无答案分,但成本极低、无违约、校准良好,不应受灾难性惩罚。"""
    traj = (
        "<think>Budget too low to succeed.</think>"
        "<estimate> low=20 high=120 p=0.1 </estimate><stop/>"
    )
    extra = dict(EXTRA, step_tokens=[50], step_searches=[0])
    score = compute_score("nq", traj, {"target": ["whatever"]}, extra)
    assert -0.5 < score <= 0.0


def test_miscalibrated_stop_penalty_is_clipped():
    """把"完成任务还需的成本"错报进区间(5000-9000 vs 实际止损仅耗 50),
    校准惩罚生效但被 clip 截断,不会淹没整个 reward 尺度。"""
    traj = (
        "<think>Budget too low.</think>"
        "<estimate> low=5000 high=9000 p=0.1 </estimate><stop/>"
    )
    extra = dict(EXTRA, step_tokens=[50], step_searches=[0])
    score = compute_score("nq", traj, {"target": ["whatever"]}, extra)
    assert score < -0.5      # 确实被罚
    assert score > -1.5      # 但有下界(η·clip = 0.2×5 = 1.0 封顶)
