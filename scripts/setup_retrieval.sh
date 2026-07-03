#!/usr/bin/env bash
# 下载 Search-R1 的本地 wiki-18 语料与 e5 索引,并常驻启动检索服务(§5.1 主环境)。
# 零 API 成本;索引常驻以压 rollout 延迟(§6 瓶颈对策)。
#
# 服务用本仓库的 scripts/retrieval_server.py(API 与上游兼容):
#   RETRIEVAL_GPU_MODE=shard   fp16 分片到所有可见 GPU(默认,2×A800 各 ~15GB)
#   RETRIEVAL_GPU_MODE=single  fp16 整份放单卡(~31GB),配 CUDA_VISIBLE_DEVICES 选卡
#   RETRIEVAL_GPU_MODE=cpu     索引留内存(需 ~64GB 空闲 RAM),最稳兜底
# 已下载过的文件自动跳过,重复执行安全。
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/retrieval}"
RETRIEVAL_GPU_MODE="${RETRIEVAL_GPU_MODE:-shard}"
mkdir -p "$DATA_DIR"

# wiki-18 语料 + e5 Flat 索引(约 65 GB,建议放本地 NVMe;
# 国内网络先 export HF_ENDPOINT=https://hf-mirror.com)
if [ ! -f "$DATA_DIR/wiki-18-corpus/wiki-18.jsonl" ]; then
  huggingface-cli download PeterGriffinJin/wiki-18-corpus --repo-type dataset \
    --local-dir "$DATA_DIR/wiki-18-corpus"
fi
if [ ! -f "$DATA_DIR/wiki-18-e5-index/e5_Flat.index" ]; then
  huggingface-cli download PeterGriffinJin/wiki-18-e5-index --repo-type dataset \
    --local-dir "$DATA_DIR/wiki-18-e5-index"
  # 上游把索引切成分卷发布时需拼接(存在分卷才执行)
  if ls "$DATA_DIR/wiki-18-e5-index"/e5_Flat.index.part* >/dev/null 2>&1; then
    cat "$DATA_DIR/wiki-18-e5-index"/e5_Flat.index.part* \
      > "$DATA_DIR/wiki-18-e5-index/e5_Flat.index"
  fi
fi

INDEX_PATH="$DATA_DIR/wiki-18-e5-index/e5_Flat.index"
CORPUS_PATH="$DATA_DIR/wiki-18-corpus/wiki-18.jsonl"

echo "启动检索服务:gpu-mode=$RETRIEVAL_GPU_MODE(读盘+建索引需几分钟,见到 'service ready' 即就绪)"
conda run -n retriever --no-capture-output python scripts/retrieval_server.py \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk 3 \
  --retriever_model intfloat/e5-base-v2 \
  --gpu-mode "$RETRIEVAL_GPU_MODE"
