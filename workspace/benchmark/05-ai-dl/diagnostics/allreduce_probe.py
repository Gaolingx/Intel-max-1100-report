#!/usr/bin/env python3
"""诊断：2 卡 allreduce / barrier 微基准（xccl 后端）。

结论口径（2026-09-22 实测，2x Max 1100，Xe Link 直连但 "Not Calibrated"）：
  * 小消息延迟地板 ~26 us（8B~128KiB 几乎不变）；
  * 带宽随消息增大爬升并于 32~64 MiB 饱和在 **~80 GB/s**（Xe Link 标称 ~318 GB/s/dir）；
  * `dist.barrier()` 单次 ~119 us —— 绝不能放进逐迭代计时循环（第一版脚本就是被它污染的）。

用法：python -m torch.distributed.run --standalone --nproc_per_node=2 \
        diagnostics/allreduce_probe.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

ITERS = {8: 200, 30: 200, 524: 200, 8 << 10: 200, 128 << 10: 100,
        1 << 20: 50, 8 << 20: 30, 32 << 20: 20, 64 << 20: 10}


def main() -> None:
    if not torch.xpu.is_available():
        raise SystemExit("no XPU")
    torch.xpu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=os.environ.get("BACKEND", "xccl"))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("xpu", int(os.environ.get("LOCAL_RANK", "0")))

    rows = []
    for nbytes in sorted(ITERS):
        iters = ITERS[nbytes]
        t = torch.ones(max(1, nbytes // 2), dtype=torch.bfloat16, device=dev)
        for _ in range(5):                      # 暖机
            dist.all_reduce(t)
        torch.xpu.synchronize()
        dist.barrier()
        e0 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):                  # 批量入队，避免逐次 barrier 污染
            dist.all_reduce(t)
        e1.record()
        torch.xpu.synchronize()
        lat = e0.elapsed_time(e1) / iters * 1000
        moved = nbytes * 2 * (world - 1) / world      # ring 口径的等效搬运量
        rows.append((nbytes, lat, moved / (lat * 1e-6) / 1e9))

    dist.barrier()
    e0 = torch.xpu.Event(enable_timing=True)
    e1 = torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(50):
        dist.barrier()
    e1.record()
    torch.xpu.synchronize()
    barrier_us = e0.elapsed_time(e1) / 50 * 1000

    if rank == 0:
        print(f"\n{'op':<12}{'size':>10}{'lat_us':>10}{'eff_GB/s':>12}")
        for nbytes, lat, gbs in rows:
            sz = f"{nbytes} B" if nbytes < 1024 else (
                f"{nbytes / 1024:.1f} KiB" if nbytes < 1 << 20 else f"{nbytes / 2**20:.1f} MiB")
            print(f"{'all_reduce':<12}{sz:>10}{lat:>10.1f}{gbs:>12.1f}")
        print(f"{'barrier':<12}{'-':>10}{barrier_us:>10.1f}{0.0:>12.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
