"""Prompt 模板:Search-R1 原版指令的预算增广扩展(§4.1 / §4.2)。

保持与 Search-R1 的 <think>/<search>/<information>/<answer> 协议兼容,
新增:<budget> 状态块(环境注入)、<estimate> 估计头(模型必须输出)、
<pivot>/<ask>/<stop> 扩展动作。
"""

from __future__ import annotations

from .budget_state import BudgetState

BUDGET_SEARCH_TEMPLATE = """Answer the given question under a hard resource budget. \
The current budget state is shown between <budget> and </budget>; every token you generate \
and every search call you make consumes it. Exceeding the budget counts as a violation.

At every turn you MUST, in order:
1. Reason inside <think> and </think>.
2. Output a budget estimate inside <estimate> and </estimate>, in the form \
<estimate> low=L high=H p=P </estimate>, where [L, H] is your 80% interval for the total \
remaining cost you will actually consume from now until you terminate this episode (by \
answering, asking, or stopping), in tokens, counting each search call as \
{search_token_equiv} tokens, and P is your probability of eventually answering correctly. \
If success looks unreachable, the right signal is a LOW P with a small remaining-cost \
interval followed by <stop/> — not an inflated interval.
3. Take exactly one action:
   - <search> query </search> — call the search engine; results appear between \
<information> and </information>.
   - <pivot> new query </pivot> — abandon the current search direction and try a \
substantially different query.
   - <answer> final answer </answer> — give the final answer, without detailed \
illustrations. For example, <answer> Beijing </answer>.
   - <ask> question </ask> — ask the user for a missing critical detail.
   - <stop/> — give up now because the remaining budget is insufficient to succeed; \
stopping early is better than overspending on a doomed attempt.

{budget_block}

Question: {question}"""


def render_prompt(
    question: str, state: BudgetState, search_token_equiv: float = 500.0
) -> str:
    """渲染带预算状态的完整任务 prompt(首轮);后续轮由 rollout 环境在每次
    <information> 之后重新注入最新的 state.render() 块。"""
    return BUDGET_SEARCH_TEMPLATE.format(
        search_token_equiv=int(search_token_equiv),
        budget_block=state.render(),
        question=question.strip(),
    )
