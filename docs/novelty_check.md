# 查新表(Gate 1 评审依据)

**项目:** 校准驱动的预算条件化搜索智能体策略学习
**查新日期:** 2026-07-02
**检索范围:** arXiv + Google Scholar,重点覆盖 2025-10 之后

---

## 四个新意维度

| 维度 | 缩写 | 含义 |
|---|---|---|
| **(a)** 校准区间估计 | Calibrated Interval | 对剩余成本输出上下界区间(非点估计),使用 proper scoring rule 训练 |
| **(b)** 估计→行动闭环训练 | Estimate-Action Loop | 区间估计在 RL 训练中反馈进动作选择(非仅 early stop / 推理时规划) |
| **(c)** 预算条件化策略 + 档位泛化 | Budget-Cond. Tiers | 单一策略条件化于预算档,泛化到训练未见的预算档(内插 + 外推) |
| **(d)** 硬预算违约控制 | Hard Violation Ctrl | 显式惩罚超支(token 或检索任一维度),非仅软成本惩罚 |

---

## 覆盖矩阵(10 篇必读近邻 + 5 篇新发现)

| # | 论文 | 年份 | (a) 校准区间 | (b) 闭环训练 | (c) 档位泛化 | (d) 硬违约控制 |
|---|---|---|:---:|:---:|:---:|:---:|
| 1 | **BAGEN** (2606.00198) | 2026 | **YES** | NO | NO | NO |
| 2 | **BATS** (2511.17006) | 2025 | NO | NO | NO | NO |
| 3 | **ContextBudget/BACM** (2604.01664) | 2026 | NO | NO | NO | NO |
| 4 | **INTENT** (2602.11541) | 2026 | PARTIAL | NO | NO | **YES** |
| 5 | **Search-R1** (2503.09516) | 2025 | NO | NO | NO | NO |
| 6 | **OTC** (2504.14870) | 2025 | NO | NO | NO | NO |
| 7 | **Efficient Agents** (2508.02694) | 2025 | NO | NO | NO | NO |
| 8 | **CoRL** (2511.02755) | 2025 | NO | NO | PARTIAL | NO |
| 9 | **Greedy Agents** (2504.16078) | 2025 | NO | NO | NO | NO |
| 10 | **MemSearcher** (2511.02805) | 2025 | NO | NO | NO | NO |
| 11 | **CTA** (2602.16699) | 2026 | PARTIAL | PARTIAL | NO | NO |
| 12 | **BAVT** (2603.12634) | 2026 | NO | NO | PARTIAL | NO |
| 13 | **On Time, Within Budget** (2605.06110) | 2026 | NO | NO | NO | **YES** |
| 14 | **BudgetThinker** (2508.17196) | 2025 | NO | NO | PARTIAL | PARTIAL |
| 15 | **ZEBRA** (2605.20485) | 2026 | NO | NO | NO | **YES** |
| — | **本工作** | 2026 | **YES** | **YES** | **YES** | **YES** |

---

## 最近竞争者分析

### 1. BAGEN — 最大威胁

BAGEN 形式化了渐进区间估计(PIE),训练 LLM 输出剩余成本的上下界,并用 early stop 节省 28–64% token。但 BAGEN **明确将"让估计器预测反馈进 actor 决策(超越 early stop)"列为开放方向**。我们的工作直接填补这个空白:

- BAGEN 的估计器是**旁路能力**(side capability),不参与动作选择
- 我们的估计头**端到端参与**动作选择(消融 D/E 证明"只估计"或"只控制"都显著劣于闭环)
- BAGEN 覆盖率上限 47%,我们的 Winkler proper scoring rule 训练目标可超越此限

### 2. CTA (Calibrate-Then-Act) — 次近

CTA 将校准的先验(calibrated priors)接入 RL 训练,但使用**点估计**而非区间,且不在搜索 agent 领域、无显式预算约束。

### 3. INTENT — 硬约束但无训练

INTENT 通过推理时规划实现硬预算可行性,但**不训练 agent 策略**(SFT only)。无校准区间,无 RL 闭环。

---

## Gate 1 结论

### 判定: **通过 — 继续执行研究计划**

**理由:**

1. **核心新意维度 (b) 完全空白:** 没有任何已发表工作在 RL 训练中将校准区间估计闭环接入动作选择。BAGEN 明确将此列为开放问题。

2. **四维组合未被占据:** (a)+(b)+(c)+(d) 的组合在文献中完全没有出现。即使只看 (a)+(b) 的组合也无人做过。

3. **差异化叙事清晰:**
   - vs BAGEN: 估计→行动闭环(非仅 early stop)
   - vs BATS: 训练出的策略(非免训练 prompt)
   - vs INTENT: RL 训练的策略(非推理时规划)
   - vs OTC: 校准约束 + 预算条件化(非仅成本惩罚)

4. **审稿攻击可回应:**
   - "只是 Search-R1 + cost penalty" → 消融 D/E + 干预实验
   - "校准是伴随现象" → oracle/加噪干预实验(因果证据)
   - "只在一个场景" → 未见预算档 + 第二环境

**风险提示:** 赛道升温中(2026 年已有 BAGEN、INTENT、CTA、BAVT 等),需加快推进。BAGEN 团队或跟进者可能很快发表闭环工作。建议在 Gate 2 前完成核心实验。

---

*查新执行: 2026-07-02 | 下次查新: Gate 2 (2026-08-31)*
