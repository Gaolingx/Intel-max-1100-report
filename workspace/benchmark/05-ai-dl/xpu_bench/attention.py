"""Attention 算子基准（Scaled Dot-Product Attention）。

对应 ``docs/TODO/05-ai-dl.md`` 3.1「Attention」。

覆盖两类工作负载：

* **prefill**：q_len == kv_len（大矩阵乘 + softmax，算力/带宽混合）；
* **decode**：q_len 很小、kv_len 很大（典型 memory-bound，KV cache 读取）。

FLOPs 口径：``4 × B × H × q_len × kv_len × D``（QK^T 与 PV 各 2 次乘加）；
causal 时按一半计。同时输出「有效带宽」用于判断是否 memory-bound。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import (
    DEVICE_TYPE,
    DTYPES,
    ResultStore,
    gbps_from,
    tflops_from,
)

SUITE = "attention"

# (label, B, H, q_len, kv_len, D, causal, large?)
CONFIGS: list[tuple[str, int, int, int, int, int, bool, bool]] = [
    ("mha_s512_d64",        1, 16,  512,  512,  64, False, False),
    ("mha_s1024_d64",       1, 16, 1024, 1024,  64, False, False),
    ("mha_s2048_d64",       1, 16, 2048, 2048,  64, False, False),
    ("mha_causal_s2048",    1, 16, 2048, 2048,  64, True,  False),
    ("mha_s4096_d64",       1, 16, 4096, 4096,  64, False, False),
    ("llama_prefill_s4096", 1, 32, 4096, 4096, 128, True,  False),
    ("llama_prefill_s8192", 1, 32, 8192, 8192, 128, True,  True),
    ("decode_kv4096",       1, 32,    1, 4096, 128, False, False),
    ("decode_kv8192",       1, 32,    1, 8192, 128, False, False),
    ("batch_decode_b32",   32, 32,    1, 4096, 128, False, False),
]

DEFAULT_DTYPES = ["bf16", "fp16", "fp32"]


def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _attn_flops(b, h, q_len, kv_len, d, causal):
    flops = 4.0 * b * h * q_len * kv_len * d
    return flops * 0.5 if causal else flops


def _attn_bytes(b, h, q_len, kv_len, d, itemsize):
    # q + k + v 读 + out 写
    elems = 2 * b * h * q_len * d + 2 * b * h * kv_len * d
    return elems * itemsize


def _metrics(b, h, q_len, kv_len, d, causal, itemsize):
    flops = _attn_flops(b, h, q_len, kv_len, d, causal)
    nbytes = _attn_bytes(b, h, q_len, kv_len, d, itemsize)

    def _fn(stats):
        return {
            "tflops": tflops_from(flops, stats.median_ms),
            "gbps": gbps_from(nbytes, stats.median_ms),
        }

    return _fn


def manual_attention(q, k, v, causal=False):
    """未融合的朴素 attention，用于与 SDPA 对照（体现融合收益）。"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        s = attn.shape[-1]
        mask = torch.triu(torch.ones(s, s, device=attn.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def sweep_sdpa(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    include_large = bool(getattr(ns, "large", False))
    dtypes = [d for d in getattr(ns, "dtypes", DEFAULT_DTYPES)]

    for label, b, h, q_len, kv_len, d, causal, large in CONFIGS:
        if large and not include_large:
            continue
        for dk in dtypes:
            dt = DTYPES.get(dk)
            if dt is None:
                continue
            itemsize = torch.tensor([], dtype=dt).element_size()
            params = {"dtype": dk, "batch": b, "heads": h, "q_len": q_len,
                      "kv_len": kv_len, "head_dim": d, "causal": causal,
                      "label": label,
                      "shape": f"b{b}·h{h}·q{q_len}·kv{kv_len}·d{d}"}
            try:
                q = torch.randn(b, h, q_len, d, device=DEVICE_TYPE, dtype=dt)
                k = torch.randn(b, h, kv_len, d, device=DEVICE_TYPE, dtype=dt)
                v = torch.randn(b, h, kv_len, d, device=DEVICE_TYPE, dtype=dt)
            except Exception as exc:
                store.error(SUITE, "sdpa", params, exc)
                torch.xpu.empty_cache()
                continue
            store.measure(
                SUITE, "sdpa", params,
                lambda q=q, k=k, v=v, causal=causal: F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal
                ),
                compute_metrics=_metrics(b, h, q_len, kv_len, d, causal, itemsize),
                note="走 SDPA（可能自动选择 flash/efficient 后端）",
                **kw,
            )
            del q, k, v
            torch.xpu.synchronize()
            torch.xpu.empty_cache()


def sweep_manual(store: ResultStore, ns) -> None:
    """朴素实现对照，仅在 seq ≤ 2048 且 q_len == kv_len 时做（显存友好）。"""
    kw = _kw(ns)
    dtypes = [d for d in getattr(ns, "dtypes", ["bf16", "fp32"])]

    for label, b, h, q_len, kv_len, d, causal, _large in CONFIGS:
        if q_len != kv_len or q_len > 2048 or q_len < 256:
            continue
        for dk in dtypes:
            dt = DTYPES.get(dk)
            if dt is None:
                continue
            itemsize = torch.tensor([], dtype=dt).element_size()
            params = {"dtype": dk, "batch": b, "heads": h, "q_len": q_len,
                      "kv_len": kv_len, "head_dim": d, "causal": causal,
                      "label": label,
                      "shape": f"b{b}·h{h}·q{q_len}·kv{kv_len}·d{d}"}
            try:
                q = torch.randn(b, h, q_len, d, device=DEVICE_TYPE, dtype=dt)
                k = torch.randn(b, h, kv_len, d, device=DEVICE_TYPE, dtype=dt)
                v = torch.randn(b, h, kv_len, d, device=DEVICE_TYPE, dtype=dt)
            except Exception as exc:
                store.error(SUITE, "sdpa_manual", params, exc)
                torch.xpu.empty_cache()
                continue
            store.measure(
                SUITE, "sdpa_manual", params,
                lambda q=q, k=k, v=v, causal=causal: manual_attention(q, k, v, causal),
                compute_metrics=_metrics(b, h, q_len, kv_len, d, causal, itemsize),
                note="未融合实现，会物化 S×S 注意力矩阵",
                **kw,
            )
            del q, k, v
            torch.xpu.synchronize()
            torch.xpu.empty_cache()


def run(ns, store: ResultStore) -> None:
    sweep_sdpa(store, ns)
    if getattr(ns, "manual_attn", True):
        sweep_manual(store, ns)
    best = store.best(SUITE, "tflops")
    if best:
        store.add_note(
            f"attention 峰值吞吐: {best.metrics['tflops']:.2f} TFLOPS "
            f"({best.params.get('label')}, dtype={best.params.get('dtype')})"
        )
