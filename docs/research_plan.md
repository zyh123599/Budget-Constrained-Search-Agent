# 研究计划(定稿)

## 校准驱动的预算条件化搜索智能体策略学习

**Calibration-Driven Budget-Conditioned Policy Learning for Search Agents**

定稿日期:2026-07-01 · 资源:2×A800(80G) · 目标:CCF-A(ARR→ACL 2027 / ICML 2027 主攻,ICLR 2027 视进度)

---

## 0. 一句话定位与贡献声明

**一句话:** 不只是让 agent 估计"还要花多少",而是让一个 4B–8B 的小模型学会:在给定预算档下,基于**校准过的剩余成本区间与失败概率**,动态选择 continue / search / pivot / answer / stop-or-ask,显式优化 **成功率–成本–预算违约率** 的三维 Pareto 前沿,并泛化到训练时未见过的预算档。

**写进论文 Intro 的三句贡献(审稿人防线):**

1. 不是预算估计,而是**预算估计闭环控制行为策略**(BAGEN 只做了估计器 + early stop,估计→行动的端到端闭环是它明示的空白)。
2. 不是 prompt 层的预算追踪器,而是**训练出的 budget-conditioned policy**(与 BATS 的免训练测试时方案、INTENT 的推理时规划形成正交差异)。
3. 不是单一成本惩罚,而是**校准约束带来的硬预算泛化**:证明"区间校准误差下降 → 违约率下降 → Pareto 前沿外推"的因果链,而非伴随现象。

---

## 1. 背景与动机

动机链条(每一环都有文献锚点):

1. **LLM 做决策 agent 时存在系统性行为缺陷。** Greedy Agents(Schmied et al., 2025, arXiv:2504.16078)指出 LLM agent 存在贪婪性、频率偏差与"知行差距"(knowing–doing gap):模型的 rationale 大多正确,却仍执行贪婪动作,大量动作空间未被探索。→ 说明问题不在"不会说",而在"不会在环境中正确分配行动"。(注:具体数字如 55% 未探索、87% rationale 正确率,写作时以你手头 PDF 原文为准核对。)
2. **交互是有真实成本的。** 多轮 agent 的每次思考、检索、工具调用都消耗 token 与金钱;失控成本已是部署侧的公认痛点。
3. **预算意识是一种独立的元认知能力,且当前模型不具备。** BAGEN(arXiv:2606.00198,2026-06)评测五个前沿模型发现:预算意识与任务能力基本脱钩(r≈0.35);模型一致性过度乐观,在注定失败的任务上持续烧钱;仅提前止损即可在失败轨迹上节省 28–64% token;SFT+RL 训练后区间覆盖率也只到 47%。
4. **空白:估计没有闭环到行为。** BAGEN 明确列出的开放问题包括:让估计器的预测反馈进 actor 决策(而不只是 early stop)、支持多维可互换预算、以及区间校准这一"核心开放问题"。这正是本课题的落点——把"知"(校准的预算估计)接回"行"(行为策略),恰好呼应第 1 点的知行差距。

Related work 中一句话交代方法路线选择:偏好优化(DPO 系)在真实 reward 不可被 policy class 表示时存在误设定风险(偏好反转、对数据分布敏感;参见 AuxDPO 一文),而本课题 reward 可程序化验证、需在线探索预算行为,故采用 on-policy RL(GRPO)。**不要把 DPO 那篇塞进 motivation 主线。**

---

## 2. 研究问题

- **RQ1(校准可学性):** 4B 级小模型经 hindsight 监督 + RL 后,剩余成本区间的覆盖率/Winkler 分数能否显著超过 BAGEN 报告的 47% 上限?
- **RQ2(闭环增益):** 把校准估计显式接入动作选择后,相对"仅 cost penalty""仅 early stop""prompt 追踪器"三类基线,Pareto 前沿是否外推、预算违约率是否显著下降?增益是否**因果地**来自估计质量(干预实验)?
- **RQ3(泛化):** 单一 policy 能否泛化到训练未见的预算档(内插与外推),以及从 search-vs-answer 泛化到 ask-vs-act 第二环境?

---

## 3. 近邻工作与差异定位(危险名单)

| 近邻 | 它做了什么 | 你不能重复 | 你补的空白 |
|---|---|---|---|
| **BAGEN** (2606.00198) | 预算意识评测协议;渐进区间估计;SFT+RL 训估计器;early stop | 只做估计器和提前停止 | 估计→行为策略的端到端闭环;超越 47% 覆盖率 |
| **BATS** (2511.17006) | 免训练 Budget Tracker;测试时按剩余资源决定 dig/pivot;推 Pareto 前沿 | 只靠 prompt 追踪预算 | 训练出的 budget-conditioned policy;未见预算档泛化 |
| **BACM/ContextBudget** (2604.01664) | 上下文压缩建模为预算约束的序列决策,RL+课程学习 | 做 memory/context 压缩 | 行为层预算:搜不搜、停不停、问不问、换不换路径(与其正交互补) |
| **INTENT** (2602.11541) | 推理时规划:意图感知分层世界模型,成本增强 StableToolBench 上硬预算可行 + 对价格变动鲁棒 | 只做 inference-time planner | 小模型可训练策略 + 校准约束;借用其 Budget-Optimal Pass Rate / Feasible Rate 指标 |
| **Search-R1** (2503.09516) | RL 训练 LLM 多轮生成搜索 query,开源代码与检查点,7 个 QA 数据集胜过 RAG 基线 | 只在其上加 cost penalty | 加预算档条件输入、区间校准头与违约控制 |
| **成本惩罚线**(OTC "Acting Less is Reasoning More" 2504.14870 等) | reward 里罚工具调用/token | 只调 reward 权重 | 校准项(proper scoring rule)+ 预算条件化 + 泛化协议 |

**红线认知:**"budget-aware agent"这个词本身已经不新。新意只能落在:**校准如何做、如何接入策略、如何泛化到未见预算档、如何在硬预算下控违约。** 论文所有表述围绕这四点。

---

## 4. 方法设计(最小骨架,先跑通再加花)

### 4.1 状态增广

每步输入追加预算状态(文本化注入 prompt 即可,不改架构):

```
B_t = (剩余 token 数, 剩余检索次数, 已用步数, 预算档标签 b ∈ {low, mid, high})
```

### 4.2 动作与估计头

模型每步输出两部分(同一段生成内,结构化标签区分):

- 动作 `a_t ∈ {think, search, pivot(换检索方向/改写query), answer, stop-or-ask}`
- 预算估计 `⟨ĉ_low, ĉ_high, p̂_success⟩`:完成本任务还需成本的区间 + 成功概率

### 4.3 奖励设计(关键改动:防 hack)

```
R = R_answer − λ·C − μ·1[C > B] − η·WinklerScore(ĉ_low, ĉ_high; c_true)
```

- `R_answer`:答案 EM/F1(可程序化验证);
- `C`:实际总成本(token + 检索次数加权);`B`:本回合预算;
- **校准项必须用 proper scoring rule(Winkler/interval score:同时惩罚区间宽度与真值偏离),不能用朴素覆盖率/校准误差**——否则模型把区间报得无限宽即可刷分,这是 naive 设计会被审稿人和模型同时打爆的点;
- `c_true` 来自 **hindsight labeling**:每条 rollout 结束后,回填每一步的"实际剩余消耗"作为真值标签(把 BAGEN 的 rollout-replay 评测协议转为训练信号)。

### 4.4 训练流程

1. **SFT 热启动:** 用带 hindsight 标签的轨迹(可先用强模型 API 少量蒸馏格式)教会输出格式与粗校准;
2. **GRPO 主训练:** 预算档 `b` 每回合随机采样自训练集 {2k, 6k, 10k}(token)×{3, 6, 10}(检索次数);
3. **课程:** 无预算 → 软惩罚(λ 递增)→ 硬约束(μ 生效);
4. 8B 规模仅做 LoRA 验证,主线锁 Qwen3-4B 全参。

---

## 5. 实验设计

### 5.1 环境与数据

- **主环境:** Search-R1 开源本地检索环境(本地 wiki 索引 + e5 检索器,零 API 成本);数据:NQ / HotpotQA / 2WikiMultihopQA / Musique / Bamboogle(前二训练,后三留作 OOD)。
- **第二环境(RQ3 泛化章节,不是独立论文):** τ-bench 风格任务型对话 + 用户模拟器,动作空间中 stop-or-ask 变为主角——search-vs-answer 与 ask-vs-act 是同一预算决策问题的两个实例,这是论文的泛化叙事。

### 5.2 基线(全部当作危险对手,不许缺席)

1. Search-R1 原版;
2. Search-R1 + cost penalty(OTC 式);
3. BAGEN 式:独立估计器 + early stop;
4. BATS 式:prompt 内预算追踪器(免训练);
5. 提示词硬约束("你只有 X token")。

### 5.3 消融矩阵(核心证据链)

| # | 变体 | 回答的问题 |
|---|---|---|
| A | 完整方法 | — |
| B | 去掉校准项(η=0) | 校准是否必要 |
| C | 去掉预算输入 B_t | 条件化是否必要 |
| D | 只估计不控制(估计头存在但动作不依赖) | 闭环是否必要 |
| E | 只控制不估计(直接 cost penalty) | 估计是否必要 |
| F | 朴素校准项替换 proper scoring rule | 防 hack 设计的价值 |

### 5.4 干预实验(因果证据,回应最狠审稿攻击)

同一冻结策略,分别喂入:oracle 剩余成本(rollout 回放真值)/ 模型自身估计 / 加噪估计(σ 递增)。画 **"估计质量 → 任务收益/违约率"曲线**。若曲线单调,则"校准因果地改善策略"成立,而非伴随下降。

### 5.5 生死指标(就四个,别被淹没)

1. **Pareto AUC:** 成功率–成本曲线下面积,相同成功率下成本更低或反之;
2. **预算违约率:** 给定 2k/4k/8k 预算档,超支比例须显著低于"Search-R1+cost penalty";
3. **校准质量:** 区间覆盖率与 Winkler 分数,**必须明显超过 BAGEN 的 47%**,否则"校准"二字站不住;
4. **未见预算档泛化:** 训练见 {2k, 6k, 10k},测试 {4k, 8k}(内插)与 {1k, 14k}(外推)。做不到就只是普通 reward tuning。

辅助指标(借自 INTENT,便于跨文对话):Budget-Optimal Pass Rate、Feasible Rate、失败轨迹止损节省率。

---

## 6. 资源与工程可行性(2×A800)

- **栈:** verl + vLLM rollout + FSDP;Qwen3-4B 全参 GRPO 在 2×80G 上可行(response ≤4–8k、小 batch、梯度累积);8B 用 LoRA。
- **真实瓶颈不是显存,而是:** rollout 吞吐、检索环境延迟、reward 统计、预算档 sweep。对策:本地索引常驻内存;**预算 sweep 尽量用离线轨迹重放评估,不要每个档都重训**;评测集分层抽样先看趋势再跑全量。
- 蒸馏 SFT 数据若用 API,控制在几百条格式示范即可,成本可忽略。

---

## 7. 时间表与 go/no-go 硬门槛

| 阶段 | 时间 | 交付物 | 门槛 |
|---|---|---|---|
| 查新 + 基建 | 7/01–7/14 | 查新表(§10);Search-R1 baseline 在 2×A800 复现 | **Gate 1:** baseline 跑不通或闭环已被做 → 立即切备选 A |
| 方法 v1 | 7/15–8/31 | 状态增广 + reward v1 + 首张 Pareto 曲线 | **Gate 2:** 8/31 无"明显外推的 Pareto 曲线" → 放弃 ICLR,锁定 ARR/ICML |
| 主实验 | 9/01–10/10 | 消融 A–F、校准指标、初稿 | 达标且早 → ICLR 2027(摘要 9/19、全文 9/24,以官网为准);否则 → **ARR 10/12** |
| 完整版 | 10月–1月 | 干预实验、第二环境、未见预算档、终稿 | **ICML 2027(约 1 月下旬,盯官网)** 作为完整版主目标 |

ARR 现为 10 周周期(2025 年 5 月起,每年约 5 轮),十月周期截稿 **2026-10-12**;若错过,下一轮约在 12 月中下旬,仍赶得上 commit ACL 2027。

---

## 8. 投稿窗口与 CCF 注意事项(第七版,2026-03 正式发布)

- **定级变化:** ICLR 升 A,IJCAI 降 B,NeurIPS 保持 A。AI 领域 A 类会议共 7 个:AAAI、NeurIPS、ACL、CVPR、ICCV、ICML、ICLR。**不要投 IJCAI。**
- **只认长文:** CCF 目录仅计 Full/Regular paper;Short paper、**Findings**、Workshop 均不算。
- **ARR 的 commit 陷阱:** 十月周期的 reviews 回来后,commit 目标**必须选 ACL 2027 主会**——NAACL、EMNLP 均非 A 类。commit 错对象,半年白干。
- 窗口速览:AAAI-27(7/21 摘要,来不及,弃)→ ICLR'27(9/24,条件触发)→ ARR 10/12 → ACL'27 → ICML'27(约 1 月底,兜底主目标)。

---

## 9. 风险登记与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| BAGEN 团队或他人先发"估计→行动闭环" | 中高(赛道正在升温) | Gate 1 查新;若闭环被做:退守"校准的因果作用(干预实验)+ 未见预算档泛化 + 硬预算违约控制"三个更细的点;全被覆盖 → 切备选 A |
| 校准 reward 被 hack | 已预防 | proper scoring rule + 消融 F 直接展示 |
| RL 训练不稳定 | 中 | SFT 热启动、GRPO、课程学习、KL 约束;先在 NQ 单跳上调稳再上多跳 |
| 8B 跑不动 | 低影响 | 4B 主线足以支撑叙事;8B-LoRA 只作规模趋势点 |
| rollout 太慢导致 sweep 做不完 | 中 | 离线重放评估、分层抽样、只对最终方法跑全量 |

**备选 A(逃生舱,ACL 风格,GPU 需求极低):** Agent 轨迹 LLM-judge 的系统性偏差与校准——meta-eval 基准 + 偏差 taxonomy(长度/工具调用次数/位置/自我偏好)+ 轻量去偏。启用条件:Gate 1 失败。注意其风险是易被视为"又一个 judge bias benchmark",需靠数据构造质量取胜。

---

## 10. 第一周查新清单(不可省略)

**关键词组(arXiv + Google Scholar,限定 2025-10 之后):**
budget-aware agent / budget-conditioned policy / cost-aware agentic RL / calibrated cost estimation LLM / anytime agent / constrained RL LLM agent / resource-aware tool use / budget violation / interval estimation agent

**必读近邻(逐篇写"它做了/没做"两行笔记):**
BAGEN (2606.00198) · BATS (2511.17006) · BACM (2604.01664) · INTENT (2602.11541) · Search-R1 (2503.09516) · OTC (2504.14870) · Efficient Agents (2508.02694) · 预算控制多专家路由 (2511.02755) · Greedy Agents (2504.16078) · MemSearcher (2511.02805,确认其记忆管理与本课题行为层的边界)

**输出物:** 一页查新表(近邻 × 四个新意维度的覆盖矩阵),Gate 1 评审依据。

---

## 11. 预设审稿攻击与回应

**攻击 1:**"本文只是 Search-R1 + cost penalty + stop 动作,BATS/BAGEN/BACM 已分别覆盖预算追踪、估计止损与预算序列决策。"
→ 回应:消融 D/E 证明"只估计"或"只惩罚"都显著劣于闭环;§5.4 干预实验给出估计质量→策略收益的因果曲线;BATS 免训练、BAGEN 无闭环、BACM 在上下文层,三者均不含"训练出的、校准约束的行为策略"。

**攻击 2:**"校准指标下降可能只是伴随现象。"
→ 回应:干预实验(oracle/加噪)+ 消融 B/F。这是全文最硬的一张牌,写作时放主实验之后紧接着讲。

**攻击 3:**"只在搜索 QA 一个场景,泛化性存疑。"
→ 回应:未见预算档(内插+外推)+ ask-vs-act 第二环境;并说明 search-vs-answer 与 ask-vs-act 是同一预算决策问题的两个实例。

---

## 附:给导师的三行摘要

我们让小模型 agent 学会"知道自己还剩多少弹药,并据此改变打法":联合训练任务策略与校准的剩余成本估计,估计显式控制继续/检索/转向/作答/求助,在开源可复现的搜索环境上外推成功率–成本 Pareto 前沿、控制硬预算违约,并泛化到未见预算档。近邻(BAGEN/BATS/BACM/INTENT)分别停在估计、免训练追踪、上下文层与推理时规划,估计→行动的训练闭环是明示空白。2×A800 + Qwen3-4B + verl 可完整执行,目标 ARR 十月(ACL 2027)/ ICML 2027,ICLR 2027 视 8 月底进度触发。
