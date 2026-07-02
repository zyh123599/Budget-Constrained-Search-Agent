"""Prompt 模板:Search-R1 原版指令的预算增广扩展(§4.1 / §4.2)。

保持与 Search-R1 的 <think>/<search>/<information>/<answer> 协议兼容,
新增:<budget> 状态块(环境注入)、<estimate> 估计头(模型必须输出)、
<pivot>/<ask>/<stop> 扩展动作。

消融开关(§5.3,与 configs/ablations.md 对应):
- include_budget=False   → 消融 C:去掉预算状态输入 B_t
- estimate_mode="after_action" → 消融 D:估计头存在但后置于动作,
  自回归流中动作无法条件于估计(只估计不控制)
- estimate_mode="none"   → 消融 E:无估计头,纯 cost penalty(只控制不估计)
"""

from __future__ import annotations

from .budget_state import BudgetState

_HEADER_BUDGETED = """Answer the given question under a hard resource budget. \
The current budget state is shown between <budget> and </budget>; every token you generate \
and every search call you make consumes it. Exceeding the budget counts as a violation."""

_HEADER_PLAIN = """Answer the given question. Be efficient: every token you generate \
and every search call you make has a cost."""

_ESTIMATE_RULE = """Output a budget estimate inside <estimate> and </estimate>, in the form \
<estimate> low=L high=H p=P </estimate>, where [L, H] is your 80% interval for the total \
remaining cost you will actually consume from now until you terminate this episode (by \
answering, asking, or stopping), in tokens, counting each search call as \
{search_token_equiv} tokens, and P is your probability of eventually answering correctly. \
If success looks unreachable, the right signal is a LOW P with a small remaining-cost \
interval followed by <stop/> — not an inflated interval."""

_ACTION_RULE = """Take exactly one action:
   - <search> query </search> — call the search engine; results appear between \
<information> and </information>.
   - <pivot> new query </pivot> — abandon the current search direction and try a \
substantially different query.
   - <answer> final answer </answer> — give the final answer, without detailed \
illustrations. For example, <answer> Beijing </answer>.
   - <ask> question </ask> — ask the user for a missing critical detail.
   - <stop/> — give up now because the remaining budget is insufficient to succeed; \
stopping early is better than overspending on a doomed attempt."""


def render_prompt(
    question: str,
    state: BudgetState,
    search_token_equiv: float = 500.0,
    include_budget: bool = True,
    estimate_mode: str = "before_action",
) -> str:
    """渲染完整任务 prompt(首轮);后续轮由 rollout 环境在每次
    <information> 之后重新注入最新的 state.render() 块。

    estimate_mode ∈ {"before_action"(完整方法), "after_action"(消融 D),
    "none"(消融 E)}。
    """
    estimate_rule = _ESTIMATE_RULE.format(search_token_equiv=int(search_token_equiv))
    steps = ["Reason inside <think> and </think>."]
    if estimate_mode == "before_action":
        steps += [estimate_rule, _ACTION_RULE]
    elif estimate_mode == "after_action":
        steps += [_ACTION_RULE, "After your action tag, " + estimate_rule[0].lower() + estimate_rule[1:]]
    elif estimate_mode == "none":
        steps += [_ACTION_RULE]
    else:
        raise ValueError(f"unknown estimate_mode: {estimate_mode!r}")

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    header = _HEADER_BUDGETED if include_budget else _HEADER_PLAIN
    parts = [header, "", "At every turn you MUST, in order:", numbered, ""]
    if include_budget:
        parts += [state.render(), ""]
    parts.append(f"Question: {question.strip()}")
    return "\n".join(parts)
