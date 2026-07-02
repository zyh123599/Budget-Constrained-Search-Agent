"""干预实验工具(§5.4):估计质量 → 策略收益的因果证据。

协议:同一冻结策略,rollout 时在 </estimate> 处截断生成,把模型自身估计
替换为(a)oracle 真值区间(rollout 回放回填)、(b)加噪估计(σ 递增)、
(c)模型原估计(对照),再继续生成动作。若"估计质量 → 任务收益/违约率"
曲线单调,则校准因果地改善策略成立(回应审稿攻击 2)。

本模块提供纯函数部分:干预估计的构造与文本拼接;截断-续写由 rollout 侧
(vLLM stop token = "</estimate>")驱动。
"""

from __future__ import annotations

import random

from .parsing import _ESTIMATE_RE, Estimate


def oracle_estimate(c_true: float, width: float = 0.2, p_success: float | None = None) -> Estimate:
    """oracle 干预:以真值为中心的窄区间(rollout 回放回填 c_true)。"""
    low = max(0.0, c_true * (1 - width / 2))
    high = c_true * (1 + width / 2)
    return Estimate(low=low, high=high, p_success=0.5 if p_success is None else p_success)


def noisy_estimate(
    c_true: float,
    sigma: float,
    rng: random.Random,
    width: float = 0.2,
    p_success: float | None = None,
) -> Estimate:
    """加噪干预:中心偏移 ~ N(0, (σ·c_true)²),σ 递增扫出质量-收益曲线。

    σ=0 退化为 oracle;区间相对宽度保持不变,只污染中心位置——干预变量
    单一(位置偏差),便于归因。
    """
    center = max(0.0, c_true + rng.gauss(0.0, sigma * c_true))
    low = max(0.0, center * (1 - width / 2))
    high = max(low, center * (1 + width / 2))
    return Estimate(low=low, high=high, p_success=0.5 if p_success is None else p_success)


def render_estimate(e: Estimate) -> str:
    return f"<estimate> low={e.low:.0f} high={e.high:.0f} p={e.p_success:.2f} </estimate>"


def splice_estimate(partial_generation: str, injected: Estimate) -> str:
    """把干预估计拼进截断的生成文本,返回续写前缀。

    rollout 以 "</estimate>" 为 stop 截断;不论截断文本里模型自己的
    <estimate> 写了什么/写没写完,一律清除并以注入值收尾,保证续写时
    动作只能条件于注入的估计。
    """
    text = _ESTIMATE_RE.sub("", partial_generation)  # 清掉已闭合的估计块
    open_idx = text.rfind("<estimate>")
    if open_idx != -1:
        text = text[:open_idx]  # 清掉未闭合的尾部估计块
    return text.rstrip() + "\n" + render_estimate(injected) + "\n"
