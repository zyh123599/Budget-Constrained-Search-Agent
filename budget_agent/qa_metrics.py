"""QA 答案的程序化验证(R_answer):SQuAD 风格 EM / F1 + 子串 EM。

与 Search-R1 的评测口径对齐(其主指标为 Exact Match);多参考答案取最大值。
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """小写、去冠词、去标点、压空白(SQuAD 官方归一化)。"""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gt) for gt in ground_truths))


def cover_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """子串 EM:归一化后参考答案是否被预测覆盖(部分工作用作宽松指标)。"""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gt) in pred for gt in ground_truths if normalize_answer(gt)))


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    """token 级 F1,多参考取最大。"""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not pred_tokens or not gt_tokens:
            best = max(best, float(pred_tokens == gt_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
