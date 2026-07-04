# 实验日志

> 约定:每个实验一节,记录 commit、配置、指标与结论;失败实验也记(负结果防止重跑)。
> 四大生死指标(§5.5):Pareto AUC / 预算违约率 / 校准质量(覆盖率+Winkler)/ 未见预算档泛化。

## Gate 状态

| Gate | 判据 | 截止 | 状态 |
|---|---|---|---|
| Gate 1 | 查新:估计→行动闭环未被做掉;Search-R1 baseline 在 2×A800 复现 | 2026-07-14 | ✅ **通过**(查新 + baseline 复现均达标,见 EXP-001) |
| Gate 2 | 明显外推的 Pareto 曲线 | 2026-08-31 | 🟡 进行中(方法 v1:SFT→GRPO→首张 Pareto) |

## 模板

### EXP-000 · <一句话目的>
- **日期 / commit**:
- **配置**:模型 / 数据 / 预算档 / reward 权重(λ, μ, η, clip)/ 课程断点
- **指标**:Pareto AUC= · 违约率= · 覆盖率= · Winkler= · 内插gap= · 外推gap=
- **结论与下一步**:

---

## 实验记录

### EXP-001 · Search-R1 baseline 复现(Gate 1 基建侧)
- **日期 / commit**:2026-07-04 / 基于 3e7029a(诊断工具链)
- **环境**:2×A800 80G;检索服务 = 本仓库 `retrieval_server.py`,e5 Flat 索引 fp16 分片
  每卡 ~15G;flash-attn 落 SDPA 垫片(GLIBC 2.31);模型
  `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`(= v0.1 preliminary)
- **1a 推理评估**(NQ test 500 题,greedy,`--prompt-style searchr1 --no-chat`):
  - **EM = 0.418**,violation = 0(无预算约束),Pareto AUC 不适用(baseline 单点)
  - 判据修正:0.480 是论文 v0.2 口径,本检查点是 v0.1,不可直接对照;
    新判据 EM ≥ 0.40 → **达标**
- **1a 检索隔离测试**(`retrieval_probe.py`,200 题 answer hit@top-3):
  - **hit rate = 0.705**(≥ 0.65)→ 检索侧健康,证实 0.418 是模型侧真实水平,
    非检索 bug(排除了 e5 前缀 / 索引类型 / 语料版本三类嫌疑)
- **1b 训练闭环冒烟**(Search-R1 官方 verl 管线):
  - 19/20 步无 OOM,检索调用正常,loss 在动 → 训练闭环可用
- **本次环境侧修复(已上游化)**:JSONL 写入 `encoding="utf-8"`;flash-attn SDPA
  垫片脚本;后台 shell 用绝对路径 python 绕过 `conda run`(见 runbook 故障速查)
- **结论**:**Gate 1 通过**。冻结此 v0.1 为本地对照锚点。写论文的强基线另跑
  v0.3 检查点(`...-em-ppo-v0.3`)对齐 0.48 量级。下一步进入方法 v1(EXP-002+)。
