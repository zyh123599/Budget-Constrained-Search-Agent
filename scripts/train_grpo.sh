#!/usr/bin/env bash
# GRPO 主训练启动器(§4.4):Qwen3-4B 全参,2×A800。
# 前置:setup_env.sh 完成;检索服务已在 :8000 常驻(setup_retrieval.sh);
#       data/train.parquet 已由 prepare_data.py 生成。
# 配置的单一事实来源是 configs/grpo_qwen3_4b.yaml,此脚本只做环境拼装;
# verl/Search-R1 的入口与字段名随版本演进,首跑用 --dry-run 校验。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_PROJECT=budget-agent

CONFIG=configs/grpo_qwen3_4b.yaml

# Search-R1 的多轮检索训练入口(其仓库内为 verl trainer 的封装);
# 以 --config 方式挂载本项目配置,预算奖励通过 custom_reward_function 接入
conda run -n searchr1 --no-capture-output \
  python -m verl.trainer.main_ppo \
  --config-path "$(pwd)/configs" \
  --config-name grpo_qwen3_4b \
  "$@"
