#!/usr/bin/env python3
"""诊断：LLM decode 中各 GEMM 形状的单次调用开销（M=1 扫描）。

结论口径（2026-09-22 实测，Qwen2.5-0.5B 的形状，单卡 Max 1100）：
  * m = 1..32 区间几乎所有 GEMM 耗时恒定（21~24 us）→ 存在 ~21 us 的固定开销地板；
  * `mlp gate_up` 恰好落在带宽墙上：m=1 时 17.4 MB / 21.7 us = 803 GB/s（= 峰值）；
  * `lm_head`(n=151936, 272 MB) 是 decode 里最重的单个算子：~490 us（556 GB/s）；
  * 典型 oneDNN GEMM 在 M=1 时只有 0.1~0.9 TFLOPS，即 1/1000 的 XMX 峰值。

用法：python diagnostics/m1_gemm_scan.py
"""
from __future__ import annotations

import torch

DEV = torch.device("xpu", 0)
H, I, V = 896, 4864, 151936


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    e0 = torch.xpu.Event(enable_timing=True)
    e1 = torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.xpu.synchronize()
    return e0.elapsed_time(e1) / iters


def mm_line(name: str, m: int, k: int, n: int) -> None:
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)  # linear: (out, in)
    ms = bench(lambda: torch.nn.functional.linear(a, b))
    flops = 2 * m * k * n
    print(f"{name:<16} m={m:<4} k={k:<6} n={n:<7} {ms * 1000:8.1f} us  "
          f"{flops / (ms * 1e-3) / 1e12:8.3f} TFLOPS  "
          f"{(m * k + k * n) * 2 / (ms * 1e-3) / 1e9:8.1f} GB/s  "
          f"(权重 {k * n * 2 / 1e6:.1f} MB)")


def main() -> None:
    torch.xpu.set_device(0)
    print(f"device={torch.xpu.get_device_name(0)}")
    for label, k, n in (("attn qkv_proj", H, H + 2 * 128),
                        ("attn o_proj", H, H),
                        ("mlp gate_up", H, 2 * I),
                        ("mlp down", I, H),
                        ("lm_head", H, V)):
        for m in (1, 4, 8, 32):
            mm_line(label, m, k, n)
        mm_line(label + " [big-M]", 512, k, n)
        print()

    x = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    print(f"{'mul+add (896)':<16} {bench(lambda: x * 1.5 + 0.5) * 1000:8.1f} us"
          f"   <-- 纯开销地板（数据量可忽略）")
    sm = torch.randn(1, 14, 1, 129, device=DEV, dtype=torch.bfloat16)
    print(f"{'softmax':<16} {bench(lambda: torch.softmax(sm, dim=-1)) * 1000:8.1f} us")


if __name__ == "__main__":
    main()
