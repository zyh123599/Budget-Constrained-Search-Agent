#!/usr/bin/env bash
# 训练环境搭建(2×A800,§6 技术栈:verl + vLLM rollout + FSDP)。
# 参照 Search-R1 的双环境方案:主训练环境 + 独立检索服务环境。
# 上游 README:https://github.com/PeterGriffinJin/Search-R1
set -euo pipefail

# ---------- 主训练环境 ----------
conda create -n searchr1 python=3.10 -y
conda run -n searchr1 pip install torch --index-url https://download.pytorch.org/whl/cu121
conda run -n searchr1 pip install vllm verl wandb datasets pandas pyarrow requests accelerate
conda run -n searchr1 pip install flash-attn --no-build-isolation

# Search-R1 环境代码(本地检索版多轮 rollout 循环)
if [ ! -d third_party/Search-R1 ]; then
  mkdir -p third_party
  git clone https://github.com/PeterGriffinJin/Search-R1.git third_party/Search-R1
fi
conda run -n searchr1 pip install -e third_party/Search-R1
conda run -n searchr1 pip install -e .   # budget_agent 核心库

# ---------- 检索服务环境(faiss-gpu 与训练栈依赖冲突,独立环境常驻)----------
conda create -n retriever python=3.10 -y
conda run -n retriever conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
conda run -n retriever pip install torch transformers datasets fastapi uvicorn

echo "环境就绪。下一步:bash scripts/setup_retrieval.sh 下载索引并启动检索服务。"
