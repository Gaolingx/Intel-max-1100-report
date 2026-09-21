#!/usr/bin/env python3
"""诊断：LLM decode 的成本预算（把 ms/token 拆成「每 kernel 固定开销 × 次数」）。

结论口径（2026-09-22 实测，Qwen2.5-0.5B / bf16 / batch=1）：
  * 单 kernel 固定开销 ~7 us（XPU 上 1 元素 add_ 与 1.6 MB add_ 同价）；
  * eager 每 token 下发 ~1900 个 aten op，1900 x 13 us ~= 24.8 ms，与实测 24.5 ms 吻合；
  * 权重总流量只有 0.99 GB（带宽 roofline 1.24 ms @800 GB/s）→ 差 ~20 倍；
  * 批大小可线性摊薄：b=1 45 tok/s -> b=16 717 tok/s。

用法：python diagnostics/decode_budget.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
import torch.utils._python_dispatch as pd  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

DEV = torch.device("xpu", 0)
MID = "Qwen/Qwen2.5-0.5B-Instruct"


def t(fn, n: int = 50, warm: int = 10) -> float:
    """吞吐口径（不逐步 sync），返回 ms/op。"""
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    e0 = torch.xpu.Event(enable_timing=True)
    e1 = torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.xpu.synchronize()
    return e0.elapsed_time(e1) / n


def count_ops(fn) -> int:
    n = [0]

    class C(pd.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            n[0] += 1
            return func(*args, **(kwargs or {}))

    with C():
        fn()
    return n[0]


def main() -> None:
    torch.xpu.set_device(0)
    print("=== 1) 单 kernel 固定开销 ===")
    for label, sz in (("1 elem", 1), ("896 elem", 896), ("896x896", 896 * 896)):
        x = torch.zeros(sz, device=DEV, dtype=torch.bfloat16)
        ms = t(lambda x=x: x.add_(1.0))
        print(f"  add_ {label:<10} {ms * 1000:7.2f} us/op  -> {1 / ms * 1e3:8.0f} kops/s")

    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16)
    model.eval().to(DEV)
    ids = torch.randint(0, 1000, (1, 128), device=DEV)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)

    def step():
        return model(input_ids=tok, past_key_values=past, use_cache=True)

    n_ops = count_ops(step)
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    decode_ms = t(step, n=20, warm=5)
    roofline = n_bytes / 800e9 * 1e3

    print("\n=== 2) decode 一步的实测与预算 ===")
    print(f"  params 张量数        : {len(list(model.parameters()))}")
    print(f"  aten op 数 / 步      : {n_ops}")
    print(f"  实测                 : {decode_ms:.2f} ms -> {1000 / decode_ms:.1f} tok/s")
    print(f"  权重总流量           : {n_bytes / 1e9:.3f} GB -> {roofline:.2f} ms @800GB/s")
    print(f"  固定开销预算(13us/op): {n_ops * 13 / 1000:.1f} ms")
    print(f"  实测 / 带宽 roofline : {decode_ms / roofline:.1f}x")
    print(f"  实测 / 预算          : {decode_ms / (n_ops * 13 / 1000):.2f}x")

    print("\n=== 3) 批大小摊薄效果 ===")
    for b in (1, 2, 4, 8, 16):
        ii = torch.randint(0, 1000, (b, 128), device=DEV)
        with torch.no_grad():
            o = model(input_ids=ii, use_cache=True)
            p, tk = o.past_key_values, o.logits[:, -1:].argmax(-1)
            ms = t(lambda: model(input_ids=tk, past_key_values=p, use_cache=True),
                   n=10, warm=3)
            ms_pre = t(lambda: model(input_ids=ii, use_cache=True), n=10, warm=3)
        print(f"  batch={b:<3} decode {ms:7.3f} ms -> {b / ms * 1000:7.1f} tok/s"
              f"   | prefill(128) {ms_pre:7.3f} ms")


if __name__ == "__main__":
    main()
