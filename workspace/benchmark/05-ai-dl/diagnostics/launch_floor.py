#!/usr/bin/env python3
"""诊断：启动开销地板 —— 区分「主机侧下发」与「设备侧执行」。

结论口径（2026-09-22 实测，Max 1100 / 1550 MHz）：
  * 小算子上「主机入队」与「GPU 事件」几乎相等（6.4~6.7 us/op）
    → GPU 立刻跑完并在等主机，**瓶颈在主机侧**（Python + torch dispatcher + L0 submit）；
  * 承载量足够大时（34 MB add_）主机入队仍是 6.5 us，但 GPU 变成 35 us（L2 带宽墙）
    → 此时才是 GPU 受限；判据是 host 时间 vs device 时间的相对大小。

用法：python diagnostics/launch_floor.py
"""
from __future__ import annotations

import time

import torch

DEV = torch.device("xpu", 0)
N = 3000


def host_enqueue(fn, n: int = N, warm: int = 200):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt_enqueue = time.perf_counter() - t0          # 不 sync：纯主机入队
    torch.xpu.synchronize()
    return dt_enqueue / n * 1e6, (time.perf_counter() - t0) / n * 1e6


def device_time(fn, n: int = N, warm: int = 200) -> float:
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
    return e0.elapsed_time(e1) / n * 1000


def main() -> None:
    torch.xpu.set_device(0)
    for label, sz in (("1 elem", 1), ("896 elem", 896), ("1 MiB", 512 * 1024)):
        x = torch.zeros(sz, device=DEV, dtype=torch.bfloat16)

        def fn(x=x):
            x.add_(1.0)

        h, tot = host_enqueue(fn)
        print(f"add_ {label:<10} 主机入队 {h:6.2f} us/op | GPU 事件 "
              f"{device_time(fn):6.2f} us/op | 总耗时 {tot:6.2f} us/op")

    big = torch.zeros(16 * 1024 * 1024, device=DEV, dtype=torch.bfloat16)
    h, _ = host_enqueue(lambda: big.add_(1.0), n=50, warm=10)
    print(f"add_ 32 MB      主机入队 {h:6.2f} us/op | GPU 事件 "
          f"{device_time(lambda: big.add_(1.0), n=50, warm=10):6.2f} us/op"
          f"  (33.6 MB 仍在 192 MB L2 内 -> 测的是 L2 带宽)")

    a = torch.randn(1, 896, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(4864, 896, device=DEV, dtype=torch.bfloat16)
    h, _ = host_enqueue(lambda: torch.nn.functional.linear(a, b), n=200, warm=50)
    print(f"linear 1x896x4864  主机入队 {h:6.2f} us/op | GPU 事件 "
          f"{device_time(lambda: torch.nn.functional.linear(a, b), n=200, warm=50):6.2f} us/op")


if __name__ == "__main__":
    main()
