#!/usr/bin/env bash
# SFT 热启动一条龙(§4.4 步骤 1):基座 rollout 采示范 → hindsight 重标注 → SFT。
# 产物 checkpoints/qwen3_4b_sft 作为 GRPO 主训练的初始化(改 configs 里 model.path)。
# 前置:检索服务已常驻;data/train.parquet 已生成。
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
N_DEMOS="${N_DEMOS:-800}"          # 采样的示范轨迹条数(几百条即可,§6)
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

mkdir -p runs data checkpoints

echo "== 1/3 基座模型 rollout 采集示范轨迹(温度 0.7 增加多样性)=="
conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
  --model "$MODEL" \
  --data data/train.parquet --limit "$N_DEMOS" \
  --retrieval-url "$RETRIEVAL_URL" \
  --temperature 0.7 --tensor-parallel 2 \
  --out runs/sft_demos_raw.jsonl

echo "== 2/3 hindsight 重标注 → SFT 多轮对话数据 =="
conda run -n searchr1 --no-capture-output python scripts/make_sft_data.py \
  runs/sft_demos_raw.jsonl --only-format-ok --out data/sft_train.jsonl

echo "== 3/3 SFT 训练(单卡即可)=="
conda run -n searchr1 --no-capture-output python scripts/sft_train.py \
  --model "$MODEL" --data data/sft_train.jsonl --out checkpoints/qwen3_4b_sft

echo "完成。把 configs/grpo_qwen3_4b.yaml 的 actor_rollout_ref.model.path 指向 checkpoints/qwen3_4b_sft 后再跑 train_grpo.sh"
