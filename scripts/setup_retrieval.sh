#!/usr/bin/env bash
# 下载 Search-R1 的本地 wiki-18 语料与 e5 索引,并常驻启动检索服务(§5.1 主环境)。
# 零 API 成本;索引常驻内存以压 rollout 延迟(§6 瓶颈对策)。
# 注意:HF 数据集名以 Search-R1 README 为准(下方为其发布名,首次运行前核对):
#   https://github.com/PeterGriffinJin/Search-R1
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/retrieval}"
mkdir -p "$DATA_DIR"

# wiki-18 语料 + e5 Flat 索引(约数十 GB,建议放本地 NVMe)
huggingface-cli download PeterGriffinJin/wiki-18-corpus --repo-type dataset \
  --local-dir "$DATA_DIR/wiki-18-corpus"
huggingface-cli download PeterGriffinJin/wiki-18-e5-index --repo-type dataset \
  --local-dir "$DATA_DIR/wiki-18-e5-index"

INDEX_PATH="$DATA_DIR/wiki-18-e5-index/e5_Flat.index"
CORPUS_PATH="$DATA_DIR/wiki-18-corpus/wiki-18.jsonl"

# 检索服务(retriever 环境):Search-R1 的 retrieval_server.py,e5-base-v2,top-3
conda run -n retriever python third_party/Search-R1/search_r1/search/retrieval_server.py \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
