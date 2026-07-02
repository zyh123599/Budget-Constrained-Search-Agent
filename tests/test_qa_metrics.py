import pytest

from budget_agent.qa_metrics import cover_exact_match, exact_match, f1_score, normalize_answer


def test_normalize_answer():
    assert normalize_answer("The Eiffel  Tower!") == "eiffel tower"
    assert normalize_answer("A  cat") == "cat"


def test_exact_match_multi_reference():
    assert exact_match("Paris", ["paris", "Lyon"]) == 1.0
    assert exact_match("Paris, France", ["Paris"]) == 0.0


def test_cover_exact_match():
    assert cover_exact_match("the capital is Paris", ["Paris"]) == 1.0
    assert cover_exact_match("Lyon", ["Paris"]) == 0.0


def test_f1():
    assert f1_score("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)
    assert f1_score("", [""]) == 1.0
    assert f1_score("x", ["y"]) == 0.0
