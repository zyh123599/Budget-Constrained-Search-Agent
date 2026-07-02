"""预算状态增广(研究计划 §4.1)。

每步把 B_t = (剩余 token 数, 剩余检索次数, 已用步数, 预算档标签) 文本化注入
prompt,不改模型架构。BudgetSpec 是一回合的预算合同,BudgetState 是执行中
的动态账本。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetSpec:
    """一回合的预算合同:token 预算 × 检索次数预算 + 粗档标签。"""

    token_budget: int
    search_budget: int
    tier: str = ""  # 例如 "low" / "mid" / "high",或 "2k×3" 细档

    def __post_init__(self) -> None:
        if self.token_budget <= 0 or self.search_budget <= 0:
            raise ValueError(f"budgets must be positive: {self}")

    def scalar(self, search_token_equiv: float = 500.0) -> float:
        """预算的标量化总量(token 等价),用于归一化 reward 各项。"""
        return self.token_budget + search_token_equiv * self.search_budget


@dataclass
class BudgetState:
    """执行中的预算账本:随 rollout 逐步 charge,支持随时渲染进 prompt。"""

    spec: BudgetSpec
    used_tokens: int = 0
    used_searches: int = 0
    step: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.spec.token_budget - self.used_tokens)

    @property
    def remaining_searches(self) -> int:
        return max(0, self.spec.search_budget - self.used_searches)

    @property
    def token_violated(self) -> bool:
        return self.used_tokens > self.spec.token_budget

    @property
    def search_violated(self) -> bool:
        return self.used_searches > self.spec.search_budget

    @property
    def violated(self) -> bool:
        """硬预算违约:任一维度超支即违约(生死指标 #2 的判据)。"""
        return self.token_violated or self.search_violated

    def charge(self, tokens: int = 0, searches: int = 0) -> "BudgetState":
        """记账一步消耗并推进步数计数(允许超支,超支由 violated 反映)。"""
        if tokens < 0 or searches < 0:
            raise ValueError("cost must be non-negative")
        self.used_tokens += tokens
        self.used_searches += searches
        self.step += 1
        return self

    def render(self) -> str:
        """渲染为注入 prompt 的 <budget> 块(§4.1 的文本化状态增广)。"""
        return (
            "<budget>\n"
            f"Remaining token budget: {self.remaining_tokens} / {self.spec.token_budget}\n"
            f"Remaining search calls: {self.remaining_searches} / {self.spec.search_budget}\n"
            f"Steps taken so far: {self.step}\n"
            f"Budget tier: {self.spec.tier or 'unspecified'}\n"
            "</budget>"
        )
