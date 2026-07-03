#!/usr/bin/env bash
# 下载 Search-R1 的本地 wiki-18 语料与 e5 索引,并常驻启动检索服务(§5.1 主环境)。
# 零 API 成本;索引常驻以压 rollout 延迟(§6 瓶颈对策)。
#
# 注意:HF 用户名是 PeterJinGo(GitHub 是 PeterGriffinJin,两者不同,资源全部公开)。
# 上游发布形态:索引为 part_aa/part_ab 分卷,语料为 wiki-18.jsonl.gz 压缩包,
# 本脚本自动拼接/解压;已就绪的文件自动跳过,重复执行安全。
#
# 服务用本仓库的 scripts/retrieval_server.py(API 与上游兼容):
#   RETRIEVAL_GPU_MODE=shard   fp16 分片到所有可见 GPU(默认,2×A800 各 ~15GB)
#   RETRIEVAL_GPU_MODE=single  fp16 整份放单卡(~31GB),配 CUDA_VISIBLE_DEVICES 选卡
#   RETRIEVAL_GPU_MODE=cpu     索引留内存(需 ~64GB 空闲 RAM),最稳兜底
# 已有现成文件放在别处时,用 INDEX_PATH / CORPUS_PATH 环境变量直接指过去。
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/retrieval}"
RETRIEVAL_GPU_MODE="${RETRIEVAL_GPU_MODE:-shard}"
INDEX_PATH="${INDEX_PATH:-$DATA_DIR/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$DATA_DIR/wiki-18.jsonl}"
mkdir -p "$DATA_DIR"

# 语料(约 5GB 压缩;国内网络先 export HF_ENDPOINT=https://hf-mirror.com)
if [ ! -f "$CORPUS_PATH" ]; then
  huggingface-cli download PeterJinGo/wiki-18-corpus --repo-type dataset \
    --local-dir "$DATA_DIR"
  if [ -f "$DATA_DIR/wiki-18.jsonl.gz" ]; then
    gzip -d "$DATA_DIR/wiki-18.jsonl.gz"
  fi
fi

# e5 Flat 索引(分卷 ~61GB,拼接后删分卷省磁盘)
if [ ! -f "$INDEX_PATH" ]; then
  huggingface-cli download PeterJinGo/wiki-18-e5-index --repo-type dataset \
    --local-dir "$DATA_DIR"
  cat "$DATA_DIR"/part_* > "$INDEX_PATH"
  rm -f "$DATA_DIR"/part_*
fi

echo "启动检索服务:gpu-mode=$RETRIEVAL_GPU_MODE(读盘+建索引需几分钟,见到 'service ready' 即就绪)"
# LD_LIBRARY_PATH 指向环境自带 libstdc++,避免系统版本过旧导致 faiss 加载失败
conda run -n retriever --no-capture-output bash -c "
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"
  exec python scripts/retrieval_server.py \
    --index_path '$INDEX_PATH' \
    --corpus_path '$CORPUS_PATH' \
    --topk 3 \
    --retriever_model intfloat/e5-base-v2 \
    --gpu-mode '$RETRIEVAL_GPU_MODE'"
