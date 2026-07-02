# 消融矩阵 A–F 的配置映射(研究计划 §5.3)

每个变体 = 数据渲染开关(`budget_agent.prompts.render_prompt`)× 奖励开关
(`budget_agent.rewards.RewardConfig`)。训练配置继承 `grpo_qwen3_4b.yaml`,
只改下表两组参数;评测统一用 `scripts/eval_budget_sweep.py`。

| # | 变体 | prompt 开关 | reward 开关 | 回答的问题 |
|---|---|---|---|---|
| A | 完整方法 | 默认(`include_budget=True, estimate_mode="before_action"`) | 默认(Winkler, η>0, λ>0, μ>0) | — |
| B | 去校准项 | 默认 | `eta_calibration=0` | 校准是否必要 |
| C | 去预算输入 | `include_budget=False` | 默认 | 条件化是否必要 |
| D | 只估计不控制 | `estimate_mode="after_action"`(估计后置,自回归流中动作无法条件于估计) | 默认 | 闭环是否必要 |
| E | 只控制不估计 | `estimate_mode="none"` | `eta_calibration=0`(无估计可校准) | 估计是否必要 |
| F | 朴素校准替换 | 默认 | `calibration_rule="coverage"`(1−覆盖率,无宽度项,可被刷宽 hack) | 防 hack 设计的价值 |

注意:
- D 的实现方式是**标签顺序**而非删信息:估计头仍训练、仍算校准奖励,但它
  出现在动作标签之后,生成动作时无法向前看到估计——这是"知行分离"的干净
  实现,回应审稿攻击 1(§11)。
- F 预期在训练中出现区间宽度爆炸(mean_width ↑、覆盖率虚高、Winkler 崩坏),
  用 `metrics.calibration_report` 的三元组直接展示。
- 干预实验(§5.4,oracle / 加噪估计喂入冻结策略)在 rollout 层实现,不在此表。
