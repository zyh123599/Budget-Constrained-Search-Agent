#!/usr/bin/env bash
# Gate 1 的另一半:Search-R1 baseline 在 2×A800 上复现(研究计划 §7)。
# 三个子命令,按顺序执行:
#   bash scripts/run_baseline.sh env-check   # 检索服务健康检查(秒级)
#   bash scripts/run_baseline.sh eval        # 官方 checkpoint 推理评估,对齐上游 EM(~1h)
#   bash scripts/run_baseline.sh train-smoke # verl 训练闭环冒烟:跑通若干步不 OOM(~1-2h)
#
# 判据(写入 docs/experiment_log.md):
#   eval:NQ test 上 EM 落在 0.45–0.50(上游 7B-PPO 报告 ~0.48)→ 环境对齐;
#   train-smoke:loss 曲线在动、无 OOM、检索调用成功 → 训练闭环可用。
#   两者都过 → Gate 1 基建侧通过。
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
    echo "对齐判据:EM ∈ [0.45, 0.50](上游 7B-PPO ~0.48)"
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
    echo "用法: bash scripts/run_baseline.sh {env-check|eval|train-smoke}"
    exit 1
    ;;
esac
