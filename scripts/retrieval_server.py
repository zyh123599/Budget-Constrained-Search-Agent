#!/usr/bin/env python3
"""自包含本地检索服务:wiki-18 + e5,API 与 Search-R1 retrieval_server.py 完全兼容。

替代上游服务的动机(2×A800 实测):61GB e5 Flat 索引在部分 faiss 构建下
`index_cpu_to_all_gpus` 分片选项不生效,fp16 整份复制单卡 cudaMalloc ~32GB
失败。本服务显式控制放置策略并在启动时打印 faiss 视角的 GPU 数与显存预算,
出问题一眼可见。

三种放置模式(--gpu-mode):
  shard   fp16 分片到所有可见 GPU(默认;2 卡各 ~15GB,给训练留足显存)
  single  fp16 整份放单卡(~31GB;用 CUDA_VISIBLE_DEVICES 选卡)
  cpu     索引留内存,只有 e5 编码器上 GPU(需 ~64GB 空闲 RAM;吞吐低但最稳)

API(与上游一致,训练/评估侧零改动):
  POST /retrieve  {"queries": [...], "topk": 3, "return_scores": true}
      -> {"result": [[{"document": {...}, "score": ...}, ...], ...]}
  GET  /health    -> {"status": "ok", ...}(供 run_baseline.sh env-check / 监控)

用法(retriever 环境):
  conda run -n retriever python scripts/retrieval_server.py \
      --index_path data/retrieval/wiki-18-e5-index/e5_Flat.index \
      --corpus_path data/retrieval/wiki-18-corpus/wiki-18.jsonl \
      --gpu-mode shard
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrieval_server")


def build_index(index_path: str, gpu_mode: str, omp_threads: int):
    import faiss

    log.info("loading faiss index from %s (数十 GB,读盘需几分钟)...", index_path)
    index = faiss.read_index(index_path)
    log.info("index loaded: ntotal=%d, dim=%d", index.ntotal, index.d)

    if gpu_mode == "cpu":
        faiss.omp_set_num_threads(omp_threads)
        log.info("CPU mode: omp_threads=%d(brute-force Flat,大 batch 下可用)", omp_threads)
        return index

    ngpu = faiss.get_num_gpus()
    log.info("faiss sees %d GPU(s)", ngpu)
    if ngpu == 0:
        raise RuntimeError(
            "faiss.get_num_gpus()==0:当前 faiss 构建不含 GPU 支持。"
            "改 --gpu-mode cpu,或重装 conda faiss-gpu / pip faiss-gpu-cu12"
        )

    fp16_gb = index.ntotal * index.d * 2 / 1e9
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True

    if gpu_mode == "shard":
        if ngpu < 2:
            log.warning("shard 模式但 faiss 只见 1 卡(检查 CUDA_VISIBLE_DEVICES),退化为整份放置")
        co.shard = True
        log.info("shard mode: fp16 总量 ~%.1f GB,分片后每卡 ~%.1f GB", fp16_gb, fp16_gb / max(ngpu, 1))
        return faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)

    if gpu_mode == "single":
        co.shard = False
        log.info("single mode: fp16 整份 ~%.1f GB 放 GPU 0(faiss 视角)", fp16_gb)
        res = faiss.StandardGpuResources()
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return faiss.index_cpu_to_gpu(res, 0, index, cloner)

    raise ValueError(f"unknown gpu_mode: {gpu_mode}")


class Encoder:
    """e5 查询编码器:'query: ' 前缀 + mean pooling + L2 归一化(与上游一致)。"""

    def __init__(self, model_path: str, max_length: int, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        if device != "cpu":
            self.model = self.model.half()

    def encode(self, queries: List[str]):
        import numpy as np

        with self.torch.no_grad():
            inputs = self.tokenizer(
                [f"query: {q}" for q in queries],
                max_length=self.max_length, padding=True, truncation=True,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**inputs, return_dict=True)
            mask = inputs["attention_mask"]
            hidden = out.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
            emb = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            emb = self.torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy().astype(np.float32, order="C")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--gpu-mode", choices=["shard", "single", "cpu"], default="shard")
    parser.add_argument("--encoder-device", default=None,
                        help="e5 编码器放哪(默认:GPU 模式随 faiss,cpu 模式也优先用 cuda)")
    parser.add_argument("--query-max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512, help="服务端检索分批大小")
    parser.add_argument("--omp-threads", type=int, default=32, help="cpu 模式的 faiss 线程数")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import datasets
    import torch
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    index = build_index(args.index_path, args.gpu_mode, args.omp_threads)

    log.info("loading corpus (memory-mapped arrow)...")
    corpus = datasets.load_dataset("json", data_files=args.corpus_path,
                                   split="train", num_proc=4)
    log.info("corpus loaded: %d docs", len(corpus))

    device = args.encoder_device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(args.retriever_model, args.query_max_length, device)
    log.info("e5 encoder on %s;service ready at http://%s:%d/retrieve",
             device, args.host, args.port)

    class QueryRequest(BaseModel):
        queries: List[str]
        topk: Optional[int] = None
        return_scores: bool = False

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "ntotal": int(index.ntotal),
                "gpu_mode": args.gpu_mode, "topk_default": args.topk}

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        k = req.topk or args.topk
        result = []
        for start in range(0, len(req.queries), args.batch_size):
            batch = req.queries[start:start + args.batch_size]
            emb = encoder.encode(batch)
            scores, idxs = index.search(emb, k=k)
            for row_scores, row_idxs in zip(scores.tolist(), idxs.tolist()):
                docs = [dict(corpus[int(i)]) for i in row_idxs if i >= 0]
                if req.return_scores:
                    result.append([{"document": d, "score": s}
                                   for d, s in zip(docs, row_scores)])
                else:
                    result.append(docs)
        return {"result": result}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
