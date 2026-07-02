"""动作与估计头的结构化解析(研究计划 §4.2)。

模型每步在同一段生成内输出两部分,用结构化标签区分:

    <think> ... </think>                          自由推理
    <estimate> low=120 high=400 p=0.7 </estimate>  剩余成本区间 + 成功概率
    动作标签(五选一):
      <search> query </search>       继续检索
      <pivot> new direction </pivot>  换检索方向 / 改写 query
      <answer> final answer </answer> 作答
      <ask> question </ask>           求助用户
      <stop/>                         止损放弃

解析必须对格式错误鲁棒(RL rollout 中不能抛异常):无效输出返回
format_ok=False 与错误列表,由 reward 层施加轻格式惩罚。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    SEARCH = "search"
    PIVOT = "pivot"
    ANSWER = "answer"
    ASK = "ask"
    STOP = "stop"


# 消耗检索次数的动作(pivot 本质是带方向重置的 search)
SEARCH_ACTIONS = frozenset({Action.SEARCH, Action.PIVOT})
# 终止动作
TERMINAL_ACTIONS = frozenset({Action.ANSWER, Action.ASK, Action.STOP})


@dataclass(frozen=True)
class Estimate:
    """预算估计头 ⟨ĉ_low, ĉ_high, p̂_success⟩:剩余成本区间 + 成功概率。"""

    low: float
    high: float
    p_success: float


@dataclass
class StepOutput:
    """一步生成的解析结果。"""

    action: Action | None = None
    content: str = ""  # 动作标签内的正文(query / 答案 / 问题)
    estimate: Estimate | None = None
    think: str = ""
    format_ok: bool = True
    errors: list[str] = field(default_factory=list)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ESTIMATE_RE = re.compile(r"<estimate>(.*?)</estimate>", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ACTION_RES: dict[Action, re.Pattern[str]] = {
    Action.SEARCH: re.compile(r"<search>(.*?)</search>", re.DOTALL),
    Action.PIVOT: re.compile(r"<pivot>(.*?)</pivot>", re.DOTALL),
    Action.ANSWER: re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
    Action.ASK: re.compile(r"<ask>(.*?)</ask>", re.DOTALL),
    Action.STOP: re.compile(r"<stop\s*/>|<stop>(.*?)</stop>", re.DOTALL),
}


def parse_estimate(text: str) -> Estimate | None:
    """从 <estimate> 正文解析 (low, high, p)。

    兼容两种写法:命名式 "low=120 high=400 p=0.7" 与裸三元 "120, 400, 0.7"。
    无效(数字不足 / low>high / p 越界 / 负成本)返回 None。
    """
    named = {}
    for key in ("low", "high", "p"):
        m = re.search(rf"\b{key}\s*[=::]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        if m:
            named[key] = float(m.group(1))
    if len(named) == 3:
        low, high, p = named["low"], named["high"], named["p"]
    else:
        numbers = _NUMBER_RE.findall(text)
        if len(numbers) < 3:
            return None
        low, high, p = (float(x) for x in numbers[:3])
    if low < 0 or high < low or not 0.0 <= p <= 1.0:
        return None
    return Estimate(low=low, high=high, p_success=p)


def parse_step(text: str) -> StepOutput:
    """解析一步生成:提取 think、估计头(取最后一次出现)与动作(取最先出现)。"""
    out = StepOutput()

    think_matches = _THINK_RE.findall(text)
    if think_matches:
        out.think = think_matches[-1].strip()

    est_matches = _ESTIMATE_RE.findall(text)
    if est_matches:
        out.estimate = parse_estimate(est_matches[-1])
        if out.estimate is None:
            out.format_ok = False
            out.errors.append("malformed <estimate> content")
    else:
        out.format_ok = False
        out.errors.append("missing <estimate>")

    # 先移除 <think> 块,防止模型在推理中"排练"的动作标签被误匹配
    text_no_think = _THINK_RE.sub("", text)

    # 按出现位置取最先的动作标签;多个动作标签视为格式警告但仍取第一个
    hits: list[tuple[int, Action, str]] = []
    for action, pattern in _ACTION_RES.items():
        m = pattern.search(text_no_think)
        if m:
            content = next((g for g in m.groups() if g is not None), "") if m.groups() else ""
            hits.append((m.start(), action, content.strip()))
    hits.sort(key=lambda h: h[0])

    if not hits:
        out.format_ok = False
        out.errors.append("missing action tag")
    else:
        _, out.action, out.content = hits[0]
        if len(hits) > 1:
            out.errors.append(f"multiple action tags, took first ({out.action.value})")
        if out.action in SEARCH_ACTIONS and not out.content:
            out.format_ok = False
            out.errors.append(f"empty <{out.action.value}> query")

    return out
