# Budget-Constrained Search Agent

**校准驱动的预算条件化搜索智能体策略学习**
Calibration-Driven Budget-Conditioned Policy Learning for Search Agents

> 让 4B–8B 小模型学会:在给定预算档下,基于**校准过的剩余成本区间与失败概率**,
> 动态选择 continue / search / pivot / answer / stop-or-ask,显式优化
> **成功率–成本–预算违约率**的三维 Pareto 前沿,并泛化到训练未见的预算档。

完整研究计划见 [docs/research_plan.md](docs/research_plan.md);
查新表与 Gate 1 结论见 [docs/novelty_check.md](docs/novelty_check.md);
逐篇文献笔记见 [docs/literature_notes.md](docs/literature_notes.md)。

## 仓库结构

```
budget_agent/            核心库(纯 Python,零依赖,单元测试锁定关键性质)
├── budget_state.py      §4.1 预算状态增广:BudgetSpec / BudgetState / <budget> 块渲染
├── parsing.py           §4.2 动作与估计头解析:五动作 + ⟨ĉ_low, ĉ_high, p̂⟩,格式鲁棒
├── winkler.py           §4.3 Winkler interval score(proper scoring rule,防区间刷宽)
├── hindsight.py         §4.3 hindsight labeling:rollout 后缀和回填 c_true
├── rewards.py           §4.3 R = EM − λ·C − μ·1[C>B] − η·Winkler + 三段课程调度
├── metrics.py           §5.5 四大生死指标:Pareto AUC / 违约率 / 校准报告 / 泛化 gap
├── budget_sampler.py    §4.4 预算档:训练 {2k,6k,10k}×{3,6,10},内插 {4k,8k},外推 {1k,14k}
├── qa_metrics.py        R_answer 程序化验证:EM / F1 / 子串 EM
├── prompts.py           Search-R1 兼容的预算增广 prompt 模板
└── verl_reward.py       verl custom_reward_function 接线点 + 离线打分纯函数
configs/
└── grpo_qwen3_4b.yaml   GRPO 主训练配置(Qwen3-4B 全参,2×A800)
scripts/
├── setup_env.sh         训练/检索双 conda 环境(verl + vLLM + FSDP / faiss-gpu)
├── setup_retrieval.sh   wiki-18 语料 + e5 索引下载与检索服务常驻
├── smoke_test.py        落地冒烟测试:核心性质 / 奖励链路 / 检索连通 / 训练栈
├── prepare_data.py      FlashRAG 数据 → 预算增广 parquet(训练采样 / 评测网格 / 消融变体)
├── run_baseline.sh      Gate 1:Search-R1 baseline 复现(env-check / eval / train-smoke)
├── rollout_eval.py      vLLM 多轮 rollout 轨迹生成(评估 / baseline / §5.4 干预实验)
├── run_sft.sh           SFT 热启动一条龙:采示范 → 重标注 → 训练
├── make_sft_data.py     轨迹 → hindsight 重标注的多轮对话 SFT 数据
├── sft_train.py         SFT 训练(assistant-only loss,单卡可跑)
├── train_grpo.sh        GRPO 主训练启动器
└── eval_budget_sweep.py 离线轨迹重放评估(预算 sweep 不重训,§6)
tests/                   核心库单元测试(含防 hack 性质测试)
docs/                    研究计划 / 查新表 / 文献笔记 / 实验日志
```

## 快速开始

```bash
pip install -e ".[dev]"
pytest                        # 全套单测,零 GPU、零网络
```

## 训练管线(2×A800)

**逐步操作手册(在没有 Claude Code 的服务器上照做即可):[docs/server_runbook.md](docs/server_runbook.md)**

```bash
bash scripts/setup_env.sh                    # 1. 双环境:searchr1 + retriever
bash scripts/setup_retrieval.sh              # 2. 检索服务常驻 :8000(零 API 成本)
python scripts/smoke_test.py --retrieval-url http://127.0.0.1:8000/retrieve
bash scripts/run_baseline.sh eval            # 3. Gate 1:baseline EM 对齐
python scripts/prepare_data.py --datasets nq hotpotqa --split train \
    --budget-mode sample --out data/train.parquet          # 4. 预算增广数据
bash scripts/run_sft.sh                      # 5. SFT 热启动(采示范→重标注→训练)
bash scripts/train_grpo.sh                   # 6. GRPO 主训练
python scripts/rollout_eval.py --model <ckpt> --data data/eval_interp.parquet \
    --tensor-parallel 2 --out runs/grpo_interp.jsonl       # 7. rollout 轨迹
python scripts/eval_budget_sweep.py runs/grpo_interp.jsonl # 8. 四大指标
```

## 设计决策备忘(写作时的证据链)

1. **校准项用 Winkler score 而非覆盖率**:覆盖率可被"无限宽区间"刷满,
   Winkler 同时惩罚宽度与偏离(`tests/test_winkler.py::test_infinite_width_hack_does_not_pay`,
   对应消融 F)。
2. **估计语义 = 至终止的实际剩余消耗**(BAGEN rollout-replay 口径),而非
   "完成任务还需的成本":后者在失败/止损轨迹上无法回填真值标签,校准信号
   恰好在最需要止损的轨迹上消失。任务可行性由 p̂_success 承载,止损步的正确
   输出是"小区间 + 低 p̂ + `<stop/>`"(`budget_agent/hindsight.py` 模块注释)。
3. **校准项截断**(`calibration_clip`):惩罚有上界,防止校准项淹没答案奖励
   导致 RL 崩塌;不报区间按上限满额罚,堵住"干脆不估计"的逃逸路径。
4. **奖励各项按预算标量归一化**:不同预算档下 reward 量级可比,预算条件化
   才学得动。
5. **违约判定是二维的**:token 或检索次数任一超支即违约(§5.5 生死指标 #2)。

## 路线图与 Gate(详见研究计划 §7)

| 阶段 | 时间 | 交付物 | 硬门槛 |
|---|---|---|---|
| 查新 + 基建 | 7/01–7/14 | 查新表;Search-R1 baseline 复现 | Gate 1:闭环被做/跑不通 → 切备选 A |
| 方法 v1 | 7/15–8/31 | 状态增广 + reward v1 + 首张 Pareto 曲线 | Gate 2:无外推曲线 → 弃 ICLR,锁 ARR/ICML |
| 主实验 | 9/01–10/10 | 消融 A–F、校准指标、初稿 | ARR 2026-10-12 |
| 完整版 | 10 月–1 月 | 干预实验、第二环境、未见预算档 | ICML 2027 |
