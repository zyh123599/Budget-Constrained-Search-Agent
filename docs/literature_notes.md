# 文献笔记(逐篇)

**项目:** 校准驱动的预算条件化搜索智能体策略学习
**整理日期:** 2026-07-02

---

## 1. BAGEN — Are LLM Agents Budget-Aware? (arXiv 2606.00198, 2026)

**作者:** Yuxiang Lin, Zihan Wang, Mengyang Liu, Yuxuan Shan, Longju Bai, Junyao Zhang 等 (Northwestern, O2 Lab, Michigan, Cornell)

**它做了什么:**
- 形式化预算意识为渐进区间估计(Progressive Interval Estimation, PIE):每步预测剩余成本的上下界 + 提前预警
- 评测五个前沿模型(GPT-4o, Claude 3.5, Gemini 1.5 等)在四个环境上的预算意识
- 发现预算意识与任务能力基本脱钩(r≈0.35);模型一致性过度乐观
- SFT+RL 训练估计器,在失败轨迹上节省 28–64% token;区间覆盖率最高 47%

**它没做什么:**
- 估计器是旁路能力,只触发二元 early stop,不参与动作选择(continue/search/pivot/answer)
- **明确将"估计→行动闭环"列为开放方向**
- 无预算条件化策略,无档位泛化
- 区间校准上限 47%(覆盖率),无 proper scoring rule(Winkler)训练

**与本工作的关系:** 最近竞争者。我们直接填补其 explicit open direction:将校准估计闭环接入行为策略的 RL 训练。

---

## 2. BATS — Budget-Aware Tool-Use Enables Effective Agent Scaling (arXiv 2511.17006, 2025)

**作者:** Tengxiao Liu, Zifeng Wang, Jin Miao, I-Hung Hsu 等 (Google Research)

**它做了什么:**
- 引入 Budget Tracker:通过 prompt 注入实时预算计数(已用/剩余)
- 动态调整规划与验证策略:根据剩余资源决定"深挖"还是"转向"
- 聚焦 web search agent,在 GAIA 等基准上验证

**它没做什么:**
- 纯 prompt 层操作,无 RL 训练,无学习到的预算感知行为
- 无校准区间估计(简单计数器)
- 无硬预算执行机制,无档位泛化

**与本工作的关系:** 互补。我们用训练出的策略替代 prompt 追踪器,且加入校准约束。

---

## 3. ContextBudget / BACM (arXiv 2604.01664, 2026)

**作者:** Yong Wu, YanZhao Zheng, TianZe Xu 等

**它做了什么:**
- 将上下文管理建模为预算约束下的序列决策问题
- BACM-RL:课程学习 + RL 学习何时/如何压缩交互历史
- 在 7B/30B 模型上验证,应用于组合 QA 和网页浏览

**它没做什么:**
- "预算"指上下文窗口大小(token 数),非计算成本/金钱预算
- 无成本区间估计,无行动层预算决策(搜不搜/停不停)
- 与本工作在资源层面正交互补(上下文 vs 行为)

**与本工作的关系:** 正交。它管上下文压缩,我们管行为策略。论文中定位为"互补工作"。

---

## 4. INTENT — Budget-Constrained Agentic LLMs (arXiv 2602.11541, 2026)

**作者:** Hanbing Liu, Chunhao Tian, Nan An 等 (上海交大, 华东师范)

**它做了什么:**
- 推理时规划框架:意图感知分层世界模型
- 将工具成本建模为几何分布,推导期望成本上界
- 风险参数控制动作接受/拒绝,在 StableToolBench 上实现硬预算可行
- 世界模型组件(Qwen2.5-3B)通过 SFT 训练

**它没做什么:**
- 推理时规划器,非训练出的策略(不用 RL 训练动作选择策略)
- 点估计 + 风险调整,非校准区间
- 无预算条件化档位,无 Pareto 前沿优化

**与本工作的关系:** 差异明确。它是推理时方案(inference-time),我们是训练时方案(training-time)。可借用其 Budget-Optimal Pass Rate 和 Feasible Rate 指标。

---

## 5. Search-R1 (arXiv 2503.09516, 2025)

**作者:** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon 等 (UIUC, UMass, Google)

**它做了什么:**
- RL(GRPO)训练 LLM 自主生成搜索查询 + 实时检索
- Retrieved token masking 实现稳定 RL 训练
- 在 7 个 QA 数据集上超过 RAG 基线 20–41%

**它没做什么:**
- 零预算意识:纯优化答案正确性,无成本概念
- 无成本估计、无预算条件化、无约束

**与本工作的关系:** 我们的基础设施和基线。在 Search-R1 环境上加预算增广、估计头、校准约束。

---

## 6. OTC — Acting Less is Reasoning More (arXiv 2504.14870, 2025)

**作者:** Hongru Wang, Cheng Qian, Wenxuan Zhong 等 (港中文, UIUC)

**它做了什么:**
- OTC-PO:RL 框架,在答案准确率之外惩罚过度工具调用
- 解决"认知卸载"(cognitive offloading)问题:agent 过度依赖工具
- 简单计数惩罚 + 策略优化

**它没做什么:**
- 简单的工具调用计数惩罚,非校准成本估计或区间估计
- 无预算意识(惩罚总调用数,不管剩余预算)
- 无预算条件化,无硬预算执行

**与本工作的关系:** 消融 E 的对照。"只控制不估计"(直接 cost penalty)是本工作需超越的基线。

---

## 7. Efficient Agents (arXiv 2508.02694, 2025)

**作者:** Ningning Wang, Xavier Hu, Pai Liu 等 (Aitomatic, UCL)

**它做了什么:**
- 首个系统性的 agent 效率-效果权衡实证研究
- 评估 LLM backbone 选择、框架设计、测试时扩展策略
- 在 GAIA 上保留 96.7% 性能,降低 28.4% 成本

**它没做什么:**
- 纯实证研究,无学习策略,无 RL
- 事后分析成本,非在训练中优化

**与本工作的关系:** 背景文献,提供 cost-of-pass 等指标参考。

---

## 8. CoRL — Multi-agent LLM Budget Control (arXiv 2511.02755, 2025)

**作者:** Bowen Jin, TJ Collins, Donghan Yu 等 (UIUC, Apple)

**它做了什么:**
- RL 训练控制器 LLM,在专家模型间路由任务
- 双目标:任务性能 + 成本奖励
- 预算特定系统提示(低/中/高预算模式),实现可控权衡

**它没做什么:**
- 路由现有专家模型,非训练单一 agent 的动作选择策略
- 无校准区间估计
- 预算条件化在系统提示层面用于模型路由,非搜索动作决策

**与本工作的关系:** 有限相关。它做 agent 间路由,我们做 agent 内决策。预算条件化的思路可参考。

---

## 9. Greedy Agents (arXiv 2504.16078, 2025)

**作者:** Thomas Schmied, Jorg Bornschein, Jordi Grau-Moya 等 (DeepMind)

**它做了什么:**
- 研究 LLM agent 的行为偏差:贪婪性、频率偏差、知行差距(knowing-doing gap)
- 55% 动作空间未探索,87% rationale 正确但仍执行贪婪动作
- RL 在自生成 CoT 上的微调改善探索

**它没做什么:**
- 研究基本决策偏差,非预算感知行为
- 环境(bandits, Tic-tac-toe)无预算/成本结构

**与本工作的关系:** 动机文献。"知行差距"是我们要解决的核心问题之一:模型可能"知道"应该止损,但不会"做"。

---

## 10. MemSearcher (arXiv 2511.02805, 2025)

**作者:** Qianhao Yuan, Jie Lou, Zichao Li 等 (中科院)

**它做了什么:**
- 维护紧凑的迭代更新内存替代完整交互历史
- Multi-context GRPO 端到端 RL 训练:联合优化推理、搜索、内存管理
- 超过 Search-R1 基线 11–12%

**它没做什么:**
- 聚焦内存/上下文效率,非预算意识
- 无成本估计、预算约束、预算条件化

**与本工作的关系:** 技术参考。其 multi-context GRPO 训练方法可能对我们的实现有借鉴价值。

---

## 新发现论文(2025-10 后)

### 11. CTA — Calibrate-Then-Act (arXiv 2602.16699, 2026)

- 将校准的先验(calibrated priors)从不确定性中解耦,接入 RL 训练
- 使用点估计先验,非校准区间;无搜索 agent 域;无显式预算约束
- 次近竞争者,但在关键维度(区间 vs 点估计、搜索 agent 域)有清晰差异

### 12. BAVT — Budget-Aware Value Tree Search (arXiv 2603.12634, 2026)

- 免训练推理时框架:动态搜索树 + 步级价值估计 + 预算条件化节点选择
- 无 RL 训练,无校准区间估计

### 13. On Time, Within Budget (arXiv 2605.06110, 2026)

- Monte Carlo Portfolio Planning:多 agent 工作流预算+截止时间约束
- 流水线级别操作,非单一搜索 agent 内部动作策略

### 14. BudgetThinker (arXiv 2508.17196, 2025)

- 控制 token 告知模型剩余 token 预算;SFT+RL 训练预算遵从
- 预算指推理 token 长度,非计算成本;无区间估计

### 15. ZEBRA (arXiv 2605.20485, 2026)

- 零样本框架:在多 agent 流水线阶段间分配固定金钱预算
- 推理时预算分割,非学习策略;流水线级别

---

*整理完成: 2026-07-02*
