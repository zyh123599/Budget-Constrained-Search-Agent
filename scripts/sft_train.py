#!/usr/bin/env python3
"""SFT 热启动训练(§4.4 步骤 1):几百条格式示范,单卡即可,几分钟量级。

输入:make_sft_data.py 产出的 messages JSONL。损失只算 assistant 轮
(user/env 轮 mask 为 -100,避免教模型生成 <information> 块)。

用法(训练机,searchr1 环境;单卡足够,双卡用 torchrun --nproc_per_node=2):
    python scripts/sft_train.py --model Qwen/Qwen3-4B \
        --data data/sft_train.jsonl --out checkpoints/qwen3_4b_sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tokenize_conversation(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """逐轮增量套 chat template,mask 非 assistant 段。"""

    def render(msgs, gen_prompt):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=gen_prompt, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt)

    input_ids = render(messages, gen_prompt=False)
    labels = [-100] * len(input_ids)
    for j, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = len(render(messages[:j], gen_prompt=True))
        end = len(render(messages[: j + 1], gen_prompt=False))
        labels[prefix:end] = input_ids[prefix:end]
    if len(input_ids) > max_len:
        return None  # 超长轨迹直接丢弃,避免截断破坏轮结构
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    examples, dropped = [], 0
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            item = tokenize_conversation(tokenizer, json.loads(line)["messages"], args.max_len)
            if item is None:
                dropped += 1
            else:
                examples.append(item)
    print(f"tokenized {len(examples)} conversations, dropped {dropped} over {args.max_len} tokens")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        ids, labels, mask = [], [], []
        for x in batch:
            pad = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            mask.append([1] * len(x["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
        ),
        train_dataset=examples,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
