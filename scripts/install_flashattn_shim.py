#!/usr/bin/env python3
"""在旧 GLIBC 机器上安装 flash_attn 的纯 PyTorch(SDPA)垫片。

背景:预编译的 flash-attn wheel 依赖较新 GLIBC;GLIBC 2.31(如 Ubuntu 20.04)
上 C 扩展 import 即崩。但 verl / transformers 里不少路径 `import flash_attn`
是模块级硬导入,缺了它整条训练拉不起来。

本垫片用 torch.nn.functional.scaled_dot_product_attention 提供 flash-attn 的
公开 API(flash_attn_func / flash_attn_varlen_func / bert_padding / layers.rotary),
让硬导入通过、注意力走 PyTorch SDPA 后端。数值与真 flash-attn 基本一致
(都做精确 softmax attention),只是没有 IO-aware kernel 的显存/速度优势——
对 2×A800 + 4B 全参、response≤4k 的规模足够用。

注意:这是"能跑通"的兜底,不是等价替换。若你的卡支持真 flash-attn(GLIBC 够新),
优先 `pip install flash-attn`,不要用本垫片。装了垫片后想切回真实现,先
`python scripts/install_flashattn_shim.py --uninstall`。

用法(在目标 conda 环境的 python 下运行):
    /opt/conda/envs/searchr1/bin/python scripts/install_flashattn_shim.py
    /opt/conda/envs/searchr1/bin/python scripts/install_flashattn_shim.py --check
    /opt/conda/envs/searchr1/bin/python scripts/install_flashattn_shim.py --uninstall
"""

from __future__ import annotations

import argparse
import shutil
import site
import sys
from pathlib import Path

_INIT = '''\
"""flash_attn SDPA 垫片(scripts/install_flashattn_shim.py 生成)。旧 GLIBC 兜底。"""
import torch
import torch.nn.functional as F

__version__ = "2.5.8+sdpa-shim"


def _sdpa(q, k, v, dropout_p, softmax_scale, causal):
    # flash-attn 约定 (B, S, H, D);SDPA 要 (B, H, S, D)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if softmax_scale is not None:
        q = q * (softmax_scale * (q.shape[-1] ** 0.5))  # 抵消 SDPA 内置 1/sqrt(d)
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
    )
    return out.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    return_attn_probs=False):
    return _sdpa(q, k, v, dropout_p, softmax_scale, causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1),
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False):
    # 变长打包 (total, H, D):按 cu_seqlens 切段逐样本做 attention 再拼回
    outs = []
    cq = cu_seqlens_q.tolist()
    for i in range(len(cq) - 1):
        s, e = cq[i], cq[i + 1]
        qi, ki, vi = (x[s:e].unsqueeze(0) for x in (q, k, v))
        outs.append(_sdpa(qi, ki, vi, dropout_p, softmax_scale, causal).squeeze(0))
    return torch.cat(outs, dim=0)


# 部分调用点用到的别名
flash_attn_qkvpacked_func = None
flash_attn_kvpacked_func = None
'''

_BERT_PADDING = '''\
"""flash_attn.bert_padding 垫片:pad/unpad 与 index_first_axis。"""
import torch


def index_first_axis(x, indices):
    return x[indices]


def unpad_input(hidden_states, attention_mask):
    # hidden_states: (B, S, ...);attention_mask: (B, S)
    mask = attention_mask.bool()
    seqlens = mask.sum(dim=1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(
        seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return flat[indices], indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[1:]
    out = hidden_states.new_zeros((batch * seqlen, *dim))
    out[indices] = hidden_states
    return out.reshape(batch, seqlen, *dim)
'''

_ROTARY = '''\
"""flash_attn.layers.rotary 垫片:apply_rotary_emb(纯 PyTorch)。"""
import torch


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False):
    ro_dim = cos.shape[-1] * 2
    x_rot, x_pass = x[..., :ro_dim], x[..., ro_dim:]
    if interleaved:
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
    else:
        x1, x2 = x_rot.chunk(2, dim=-1)
    c = cos.unsqueeze(0).unsqueeze(2)
    s = sin.unsqueeze(0).unsqueeze(2)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out_rot = torch.stack((o1, o2), dim=-1).flatten(-2) if interleaved \\
        else torch.cat((o1, o2), dim=-1)
    return torch.cat((out_rot, x_pass), dim=-1)
'''

_FILES = {
    "flash_attn/__init__.py": _INIT,
    "flash_attn/bert_padding.py": _BERT_PADDING,
    "flash_attn/layers/__init__.py": "",
    "flash_attn/layers/rotary.py": _ROTARY,
}


def _site_dir() -> Path:
    return Path(site.getsitepackages()[0])


def _real_flash_attn_present() -> bool:
    pkg = _site_dir() / "flash_attn" / "__init__.py"
    if not pkg.exists():
        return False
    return "sdpa-shim" not in pkg.read_text(encoding="utf-8")


def install() -> None:
    site_dir = _site_dir()
    if _real_flash_attn_present():
        print(f"检测到真实 flash_attn 于 {site_dir}/flash_attn,不覆盖。"
              f"如确需垫片,先手动移除或用 --uninstall。")
        sys.exit(1)
    for rel, content in _FILES.items():
        path = site_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    dist = site_dir / "flash_attn-2.5.8.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: flash-attn\nVersion: 2.5.8\n", encoding="utf-8")
    (dist / "INSTALLER").write_text("flashattn-shim\n", encoding="utf-8")
    print(f"垫片已安装到 {site_dir}/flash_attn(SDPA 后端)。")
    check()


def uninstall() -> None:
    site_dir = _site_dir()
    if _real_flash_attn_present():
        print("当前是真实 flash_attn,拒绝删除。")
        sys.exit(1)
    shutil.rmtree(site_dir / "flash_attn", ignore_errors=True)
    shutil.rmtree(site_dir / "flash_attn-2.5.8.dist-info", ignore_errors=True)
    print("垫片已移除。")


def check() -> None:
    try:
        import flash_attn
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
        from flash_attn.layers.rotary import apply_rotary_emb  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import flash_attn 失败: {type(e).__name__}: {e}")
        sys.exit(1)
    kind = "真实实现" if _real_flash_attn_present() else "SDPA 垫片"
    print(f"[OK] flash_attn 可导入({kind}),version={flash_attn.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.uninstall:
        uninstall()
    else:
        install()


if __name__ == "__main__":
    main()
