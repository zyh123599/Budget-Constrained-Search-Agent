#!/usr/bin/env bash
# Gate 1 的另一半:Search-R1 baseline 在 2×A800 上复现(研究计划 §7)。
# 子命令(按序):
#   env-check    检索服务健康检查(秒级)
#   eval         官方 checkpoint 推理评估,对齐上游 EM(~1h)
#   probe        检索侧隔离测试:answer hit rate,EM 偏低时定位问题在哪一侧(分钟级)
#   train-smoke  verl 训练闭环冒烟:跑通若干步不 OOM(~1-2h)
#   pareto       budget-clipped baseline 的 Pareto sweep(论文对比曲线,数小时)
#
# eval 判据(版本敏感,写入 docs/experiment_log.md):
#   默认检查点(无版本后缀)= v0.1 preliminary(少量训练步数),对应 wandb 项目
#   Search-R1-nq_hotpotqa_train;论文 (2503.09516) 的 NQ EM=0.480 是 v0.2 口径,
#   两者不可直接对照。判读:
#     EM ≥ 0.40 且 probe 的 hit rate ≥ 0.65 → 环境对齐,冻结为本地 baseline;
#     EM < 0.40 或 hit rate < 0.60        → 先跑 probe 定位检索侧/生成侧。
#   要对齐论文数字,用 v0.3 检查点:
#     BASELINE_CKPT=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3
#   train-smoke 判据:loss 曲线在动、无 OOM、检索调用成功 → 训练闭环可用。
set -euo pipefail

RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
# 注意:HF 用户名是 PeterJinGo(GitHub 用户名 PeterGriffinJin 在 HF 上不存在);
# 该检查点公开,无需 token,约 15GB
BASELINE_CKPT="${BASELINE_CKPT:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"

case "${1:-}" in
  env-check)
    echo "== 检索服务健康检查 =="
    curl -sf -X POST "$RETRIEVAL_URL" \
      -H 'Content-Type: application/json' \
      -d '{"queries": ["capital of France"], "topk": 3, "return_scores": true}' \
      | head -c 500
    echo
    echo "OK:检索服务可用"
    ;;

  eval)
    echo "== baseline 推理评估:$BASELINE_CKPT =="
    # NQ test 前 500 条;预算档给巨额兜底(baseline 不感知预算,只对齐 EM)
    if [ ! -f data/eval_nq_test.parquet ]; then
      conda run -n searchr1 --no-capture-output python scripts/prepare_data.py \
        --datasets nq --split test --budget-mode grid --grid train \
        --max-per-dataset 500 --out data/eval_nq_test_grid.parquet
      # grid 会逐档展开;baseline 只需一档,取 10k×10 档即可
      conda run -n searchr1 --no-capture-output python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/eval_nq_test_grid.parquet")
df = df[df["extra_info"].apply(lambda i: i["tier"] == "10k×10")]
df.to_parquet("data/eval_nq_test.parquet")
print(f"baseline eval set: {len(df)} rows")
EOF
    fi
    conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
      --model "$BASELINE_CKPT" \
      --data data/eval_nq_test.parquet \
      --prompt-style searchr1 --no-chat \
      --retrieval-url "$RETRIEVAL_URL" \
      --tensor-parallel 2 \
      --out runs/searchr1_baseline_nq.jsonl
    echo "判读见本脚本头部注释:v0.1 检查点 EM ≥ 0.40 + probe hit rate ≥ 0.65 即对齐"
    ;;

  probe)
    echo "== 检索侧隔离测试:answer hit rate @ top-3 =="
    conda run -n searchr1 --no-capture-output python scripts/retrieval_probe.py \
      --data data/eval_nq_test.parquet --limit 200 --retrieval-url "$RETRIEVAL_URL"
    ;;

  pareto)
    echo "== budget-clipped baseline Pareto sweep(500 题 × 9 档,数小时)=="
    # baseline 不感知预算 → 外部硬截断:token 耗尽/检索配额用完时强制作答。
    # 这条成功率–成本曲线是论文对比的公平基线(审稿必问)
    conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
      --model "$BASELINE_CKPT" \
      --data data/eval_nq_test_grid.parquet \
      --prompt-style searchr1 --no-chat \
      --on-exhaust force_answer \
      --retrieval-url "$RETRIEVAL_URL" \
      --tensor-parallel 2 \
      --out runs/searchr1_baseline_pareto.jsonl
    conda run -n searchr1 --no-capture-output python scripts/eval_budget_sweep.py \
      runs/searchr1_baseline_pareto.jsonl
    ;;

  train-smoke)
    echo "== verl 训练闭环冒烟(Search-R1 官方管线,少量步数)=="
    # 用上游自己的数据与脚本验证训练闭环;字段名随 verl 版本演进,
    # 报错时对照 third_party/Search-R1/train_grpo.sh 调整
    cd third_party/Search-R1
    if [ ! -f data/nq_search/train.parquet ]; then
      conda run -n searchr1 --no-capture-output python scripts/data_process/nq_search.py \
        --local_dir data/nq_search
    fi
    conda run -n searchr1 --no-capture-output bash train_grpo.sh 2>&1 | tee ../../runs/baseline_train_smoke.log
    ;;

  *)
    echo "用法: bash scripts/run_baseline.sh {env-check|eval|probe|train-smoke|pareto}"
    exit 1
    ;;
esac
