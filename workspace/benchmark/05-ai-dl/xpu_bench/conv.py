"""Convolution 算子基准（ResNet-50 / MobileNet 典型 shape）。

对应 ``docs/TODO/05-ai-dl.md`` 3.2（CNN 训练吞吐）的算子级前置测试。
给出 conv2d 的有效 TFLOPS，用于判断 CNN 训练时算力是否被压满。

FLOPs 口径：``2 × N × C_out × H_out × W_out × (C_in / groups) × kH × kW``。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import (
    DEVICE_TYPE,
    DTYPES,
    ResultStore,
    tflops_from,
)

SUITE = "conv"

# (label, N, C_in, H, W, C_out, k, stride, pad, groups, channels_last?)
CONFIGS: list[tuple[str, int, int, int, int, int, int, int, int, int, bool]] = [
    ("resnet_stem_224",  256,   3, 224, 224,  64, 7, 2, 3, 1, False),
    ("resnet_3x3_56",    256,  64,  56,  56,  64, 3, 1, 1, 1, False),
    ("resnet_3x3_28",    256, 128,  28,  28, 128, 3, 1, 1, 1, False),
    ("resnet_3x3_14",    256, 256,  14,  14, 256, 3, 1, 1, 1, False),
    ("resnet_3x3_7",     256, 512,   7,   7, 512, 3, 1, 1, 1, False),
    ("resnet_1x1",       256, 256,  56,  56,  64, 1, 1, 0, 1, False),
    ("depthwise_56",     256, 128,  56,  56, 128, 3, 1, 1, 128, False),
    ("resnet_3x3_56_nhwc", 256, 64, 56, 56,  64, 3, 1, 1, 1, True),
]

DEFAULT_DTYPES = ["fp32", "bf16"]


def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _out_hw(h, w, k, stride, pad):
    return (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1


def sweep_conv(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    dtypes = [d for d in getattr(ns, "dtypes", DEFAULT_DTYPES)]

    for label, n, c_in, h, w, c_out, k, stride, pad, groups, nhwc in CONFIGS:
        oh, ow = _out_hw(h, w, k, stride, pad)
        flops = 2.0 * n * c_out * oh * ow * (c_in // groups) * k * k
        for dk in dtypes:
            dt = DTYPES.get(dk)
            if dt is None:
                continue
            params = {"dtype": dk, "batch": n, "c_in": c_in, "h": h, "w": w,
                      "c_out": c_out, "kernel": k, "stride": stride, "pad": pad,
                      "groups": groups, "channels_last": nhwc, "label": label,
                      "shape": f"{n}x{c_in}x{h}x{w}->{c_out}x{oh}x{ow}"}
            try:
                x = torch.randn(n, c_in, h, w, device=DEVICE_TYPE, dtype=dt)
                weight = torch.randn(c_out, c_in // groups, k, k, device=DEVICE_TYPE, dtype=dt)
                if nhwc:
                    x = x.to(memory_format=torch.channels_last)
                    weight = weight.to(memory_format=torch.channels_last)
            except Exception as exc:
                store.error(SUITE, "conv2d", params, exc)
                torch.xpu.empty_cache()
                continue

            def _metrics(stats, flops=flops, n=n):
                return {
                    "tflops": tflops_from(flops, stats.median_ms),
                    "img_per_s": n / (stats.median_ms * 1e-3),
                }

            store.measure(
                SUITE, "conv2d", params,
                lambda x=x, weight=weight, k=k, stride=stride, pad=pad, groups=groups: F.conv2d(
                    x, weight, None, stride, pad, dilation=1, groups=groups
                ),
                compute_metrics=_metrics,
                note="channels_last" if nhwc else "",
                **kw,
            )
            del x, weight
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    best = store.best(SUITE, "tflops")
    if best:
        store.add_note(
            f"conv2d 峰值吞吐: {best.metrics['tflops']:.2f} TFLOPS "
            f"({best.params.get('label')}, dtype={best.params.get('dtype')})"
        )


def run(ns, store: ResultStore) -> None:
    sweep_conv(store, ns)
