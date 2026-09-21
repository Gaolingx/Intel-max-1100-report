"""Reduction / normalization 算子基准（memory-bound 主战场）。

对应 ``docs/TODO/05-ai-dl.md`` 3.1「Attention / LayerNorm / Softmax / Activation」。
这些算子几乎纯吃显存带宽，因此用「有效带宽 GB/s」衡量，
并与 ``membw`` suite 实测的 HBM 上限算达成率。

包含：sum / amax / mean（只读）、softmax / log_softmax / layer_norm /
rms_norm / l2_normalize / dot。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import (
    DEVICE_TYPE,
    DTYPES,
    ResultStore,
    gbps_from,
    numel_for_bytes,
)

SUITE = "reduce"

DEFAULT_DTYPE = "fp32"


def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm 参考实现（未融合，会引入额外临时张量）。"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _bw_metrics(io_factor: int, n: int, itemsize: int):
    def _fn(stats):
        nbytes = io_factor * n * itemsize
        return {"gbps": gbps_from(nbytes, stats.median_ms)}

    return _fn


def sweep_reductions(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    size_mb = int(getattr(ns, "size_mb", 1024))
    cols = int(getattr(ns, "reduce_cols", 4096))
    dtypes = [d for d in getattr(ns, "dtypes", [DEFAULT_DTYPE])]

    for dk in dtypes:
        dt = DTYPES.get(dk)
        if dt is None:
            continue
        itemsize = torch.tensor([], dtype=dt).element_size()
        n = numel_for_bytes(size_mb << 20, dt)
        rows = max(1, n // cols)
        n = rows * cols

        try:
            x = torch.randn(rows, cols, device=DEVICE_TYPE, dtype=dt)
            y = torch.randn(rows, cols, device=DEVICE_TYPE, dtype=dt)
            flat_x = x.reshape(-1)
            flat_y = y.reshape(-1)
        except Exception as exc:
            store.error(SUITE, "reduce", {"dtype": dk, "rows": rows, "cols": cols}, exc)
            continue

        shape = f"[{rows},{cols}]"
        ops = [
            # (name, io_factor, callable)
            ("sum_all",      1, lambda: torch.sum(x)),
            ("amax_all",     1, lambda: torch.amax(x)),
            ("mean_all",     1, lambda: torch.mean(x)),
            ("sum_dim1",     1, lambda: torch.sum(x, dim=1)),
            ("softmax",      2, lambda: torch.softmax(x, dim=-1)),
            ("log_softmax",  2, lambda: torch.log_softmax(x, dim=-1)),
            ("layer_norm",   2, lambda: F.layer_norm(x, (cols,))),
            ("rms_norm",     2, lambda: rms_norm(x)),
            ("l2_normalize", 2, lambda: F.normalize(x, dim=-1)),
            ("dot",          2, lambda: torch.dot(flat_x, flat_y)),
        ]

        for name, io_factor, fn in ops:
            params = {"dtype": dk, "elems": n, "size_mib": size_mb,
                      "io_factor": io_factor, "shape": shape}
            note = "未融合实现，含临时张量开销" if name == "rms_norm" else ""
            store.measure(
                SUITE, name, params, fn,
                compute_metrics=_bw_metrics(io_factor, n, itemsize),
                note=note,
                **kw,
            )

        del x, y, flat_x, flat_y
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    # 与 membw 结果对照
    best = store.best(SUITE, "gbps")
    if best:
        store.add_note(
            f"reduction 峰值有效带宽: {best.metrics['gbps']:.1f} GB/s ({best.name})"
        )


def run(ns, store: ResultStore) -> None:
    sweep_reductions(store, ns)
