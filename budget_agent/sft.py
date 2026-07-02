"""SFT 热启动数据构造(§4.4 步骤 1):hindsight 重标注轨迹估计头。

拿到任意来源的轨迹(强模型 API 蒸馏的格式示范、或 baseline rollout)后,
把每步 <estimate> 的区间与成功概率替换为 hindsight 真值的教学版本:

- 区间:以 c_true 为中心的相对宽度带 [c_true·(1−w/2), c_true·(1+w/2)],
  教"粗校准"——真值居中、宽度适度,后续精细校准交给 RL 的 Winkler 项;
- p̂:回填轨迹最终成败(EM),可选平滑避免教出 0/1 过自信。

产出的目标文本用于教会输出格式与粗校准,几百条即可(§6:蒸馏成本可忽略)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .hindsight import StepCost, remaining_costs
from .parsing import _ESTIMATE_RE

# 动作标签的起始位置(用于在缺失 <estimate> 时决定插入点)
_ACTION_OPEN_RE = re.compile(r"<(search|pivot|answer|ask|stop)\b")


@dataclass(frozen=True)
class SFTConfig:
    width: float = 0.3            # 教学区间的相对宽度 w
    p_smooth: float = 0.1         # 成功概率的平滑:p ∈ [p_smooth, 1-p_smooth]
    search_token_equiv: float = 500.0


def _render_estimate(c_true: float, success: bool, config: SFTConfig) -> str:
    low = max(0, int(c_true * (1 - config.width / 2)))
    high = max(low, int(round(c_true * (1 + config.width / 2))))
    p = 1.0 - config.p_smooth if success else config.p_smooth
    return f"<estimate> low={low} high={high} p={p:.2f} </estimate>"


def relabel_step(step_text: str, c_true: float, success: bool, config: SFTConfig = SFTConfig()) -> str:
    """把一步文本的 <estimate> 替换为 hindsight 教学值;缺失则插到动作标签前。"""
    target = _render_estimate(c_true, success, config)
    matches = list(_ESTIMATE_RE.finditer(step_text))
    if matches:
        # 只保留最后一个估计位(与解析层"最后一次出现生效"一致),其余剔除
        last = matches[-1]
        text = step_text[: last.start()] + target + step_text[last.end() :]
        for m in reversed(matches[:-1]):
            text = text[: m.start()] + text[m.end() :]
        return text
    action = _ACTION_OPEN_RE.search(step_text)
    if action:
        return step_text[: action.start()] + target + "\n" + step_text[action.start() :]
    return step_text + "\n" + target


def relabel_trajectory(
    step_texts: list[str],
    costs: list[StepCost],
    success: bool,
    config: SFTConfig = SFTConfig(),
) -> list[str]:
    """整条轨迹的 hindsight 重标注:第 t 步的教学区间以后缀和 c_true_t 为中心。"""
    if len(step_texts) != len(costs):
        raise ValueError(f"length mismatch: {len(step_texts)} texts vs {len(costs)} costs")
    truths = remaining_costs(costs, config.search_token_equiv)
    return [
        relabel_step(text, c_true, success, config)
        for text, c_true in zip(step_texts, truths)
    ]
