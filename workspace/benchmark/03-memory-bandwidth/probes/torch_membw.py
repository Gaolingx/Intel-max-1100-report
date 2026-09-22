#!/usr/bin/env python3
"""03-memory-bandwidth / probes/torch_membw.py

用 PyTorch(XPU) 做显存带宽的**独立交叉验证**：同一台机器、同一块 Max 1100，
但走完全不同的软件栈（oneDNN/SYCL 通过 torch 而不是 BabelStream）。

测量项
------
* D2D copy    —— `b.clone()` / `dst.copy_(src)`，bf16 与 fp32，尺寸 64 MiB..2 GiB
* H2D         —— pageable 与 pinned 各一档
* D2H         —— pageable 与 pinned 各一档

为什么 H2D/D2H 很重要：PCIe 5.0 x16 理论 ~63 GB/s，实际能到多少直接决定了
训练时 dataloader / 权重加载的开销上限。pinned 内存走 DMA，pageable 要过 bounce buffer。

输出：一行 `TORCHBW {json}` 供 run_bench.py 解析。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

TORCH_OK = True
try:
    import torch
except Exception as exc:  # noqa: BLE001
    TORCH_OK = False
    print(f"TORCHBW {json.dumps({'items': [], 'error': f'torch import failed: {exc}'})}")
    raise SystemExit(0)

MiB = 1 << 20
GiB = 1 << 30


def bench(fn, warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def d2d_case(dev, dtype, nbytes: int, iters: int) -> dict:
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    src = torch.randn(n, device=dev, dtype=dtype)
    dst = torch.empty_like(src)
    moved = src.numel() * src.element_size() * 2  # 读 + 写

    def f():
        dst.copy_(src)

    med, best = bench(f, 3, iters)
    return {
        "name": f"D2D.copy.{str(dtype).split('.')[-1]}",
        "params": {"dtype": str(dtype).split(".")[-1], "MiB": round(nbytes / MiB, 1)},
        "metrics": {
            "gbps": round(moved / best / 1e9, 2),
            "gbps_median": round(moved / med / 1e9, 2),
            "moved_GiB": round(moved / GiB, 3),
        },
        "status": "ok",
        "note": "D2D 显存内拷贝 (读+写)",
    }


def h2d_case(dev, dtype, nbytes: int, pinned: bool, iters: int) -> dict:
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    if pinned:
        host = torch.empty(n, dtype=dtype, pin_memory=True)
    else:
        host = torch.empty(n, dtype=dtype)
    host.normal_()
    dev_t = torch.empty(n, device=dev, dtype=dtype)

    def f():
        # 只有 pinned 内存的拷贝才可能真正异步；pageable 必须同步进行
        dev_t.copy_(host, non_blocking=pinned)

    med, best = bench(f, 3, iters)
    return {
        "name": f"H2D.{'pinned' if pinned else 'pageable'}.{str(dtype).split('.')[-1]}",
        "params": {"dtype": str(dtype).split(".")[-1], "MiB": round(nbytes / MiB, 1),
                   "pinned": pinned},
        "metrics": {
            "gbps": round(nbytes / best / 1e9, 2),
            "gbps_median": round(nbytes / med / 1e9, 2),
            "pct_of_pcie5_x16_63": round(nbytes / best / 1e9 / 63.0 * 100.0, 1),
        },
        "status": "ok",
        "note": "PCIe 5.0 x16 理论 ≈63 GB/s",
    }


def d2h_case(dev, dtype, nbytes: int, pinned: bool, iters: int) -> dict:
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    if pinned:
        host = torch.empty(n, dtype=dtype, pin_memory=True)
    else:
        host = torch.empty(n, dtype=dtype)
    src = torch.randn(n, device=dev, dtype=dtype)

    def f():
        host.copy_(src, non_blocking=pinned)

    med, best = bench(f, 3, iters)
    return {
        "name": f"D2H.{'pinned' if pinned else 'pageable'}.{str(dtype).split('.')[-1]}",
        "params": {"dtype": str(dtype).split(".")[-1], "MiB": round(nbytes / MiB, 1),
                   "pinned": pinned},
        "metrics": {
            "gbps": round(nbytes / best / 1e9, 2),
            "gbps_median": round(nbytes / med / 1e9, 2),
        },
        "status": "ok",
        "note": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not TORCH_OK or not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print(f"TORCHBW {json.dumps({'items': [], 'error': 'torch.xpu unavailable'})}")
        return 0

    n_dev = torch.xpu.device_count()
    dev = torch.device(f"xpu:{args.device % max(n_dev, 1)}")
    items: list[dict] = [{
        "name": "device_info",
        "params": {"device": str(dev)},
        "metrics": {"xpu_device_count": n_dev,
                    "device_name": torch.xpu.get_device_name(dev)},
        "status": "ok",
        "note": "",
    }]

    iters = 5 if args.quick else 10
    d2d_sizes = [64 * MiB, 512 * MiB] if args.quick else [64 * MiB, 256 * MiB, 512 * MiB, 1 * GiB, 2 * GiB]
    pci_bytes = 64 * MiB if args.quick else 256 * MiB

    for nb in d2d_sizes:
        for dt in (torch.bfloat16, torch.float32):
            try:
                items.append(d2d_case(dev, dt, nb, iters))
            except Exception as exc:  # noqa: BLE001
                items.append({"name": f"D2D.{dt}", "params": {"MiB": nb / MiB},
                              "metrics": {}, "status": "error", "note": str(exc)[:200]})

    for pinned in (False, True):
        for fn, tag in ((h2d_case, "H2D"), (d2h_case, "D2H")):
            try:
                items.append(fn(dev, torch.float32, pci_bytes, pinned, iters))
            except Exception as exc:  # noqa: BLE001
                items.append({"name": f"{tag}.{pinned}", "params": {},
                              "metrics": {}, "status": "error", "note": str(exc)[:200]})

    print("TORCHBW " + json.dumps({"items": items}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
