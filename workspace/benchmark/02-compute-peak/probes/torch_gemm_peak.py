#!/usr/bin/env python3
"""02-compute-peak / probes/torch_gemm_peak.py

用 PyTorch (oneDNN) 的 GEMM 作为**成熟软件栈**的算力参照，交叉验证
`sycl/alu_peak.cpp`（纯 ALU）与 `sycl/xmx_peak.cpp`（纯 DPAS）的结果。

输出：每行一条 ``TORCHCOMPUTE {json}``，最后一条 ``TORCHDEVICE {json}``。

注意
----
· int8 必须用 ``torch._int_mm``（``torch.matmul`` 在 int8 上返回 int8，会溢出）。
· fp8/fp4 不走 XMX，实测比 bf16 慢（见 docs/precision-support.md），故不测。
· FLOP 口径统一用 ``2*M*N*K``（int8 同理，输出 GOPS）。
"""

from __future__ import annotations

import json
import os
import sys

import torch


def emit(prefix: str, obj: dict) -> None:
    sys.stdout.write(f"{prefix} {json.dumps(obj)}\n")
    sys.stdout.flush()


def bench(fn, warmup: int = 3, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = torch.xpu.Event(enable_timing=True)
        t1 = torch.xpu.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.xpu.synchronize()
        best = min(best, t0.elapsed_time(t1))
    return best / 1000.0  # ms -> s


def flops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def main() -> int:
    if not torch.xpu.is_available():
        emit("TORCHCOMPUTE", {"status": "error", "note": "torch.xpu unavailable"})
        return 1
    dev = torch.xpu.get_device_properties(0)
    emit(
        "TORCHDEVICE",
        {
            "name": dev.name,
            "total_memory_GiB": round(dev.total_memory / 2**30, 1),
            "device_count": torch.xpu.device_count(),
            "torch": torch.__version__,
            "idx": torch.xpu.current_device(),
        },
    )

    # 与 docs/TODO/02-compute-peak.md 的判读尺度一致的形状
    shapes = [(4096, 4096, 4096), (4096, 4096, 16384), (8192, 8192, 8192)]

    # ---- 浮点 GEMM -------------------------------------------------------- #
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for m, n, k in shapes:
            try:
                a = torch.randn(m, k, dtype=dtype, device="xpu")
                b = torch.randn(k, n, dtype=dtype, device="xpu")
                secs = bench(lambda: torch.matmul(a, b))
                tf = flops(m, n, k) / secs / 1e12
                emit(
                    "TORCHCOMPUTE",
                    {
                        "kind": "gemm",
                        "dtype": str(dtype).replace("torch.", ""),
                        "m": m, "n": n, "k": k,
                        "seconds": round(secs, 6),
                        "value": round(tf, 2),
                        "unit": "TFLOPS",
                    },
                )
                del a, b
                torch.xpu.empty_cache()
            except Exception as exc:  # noqa: BLE001
                emit("TORCHCOMPUTE", {"kind": "gemm",
                                      "dtype": str(dtype).replace("torch.", ""),
                                      "m": m, "n": n, "k": k,
                                      "status": "error", "note": f"{type(exc).__name__}: {exc}"})

    # ---- int8 GEMM (XMX INT8) --------------------------------------------- #
    for m, n, k in shapes:
        try:
            a = torch.randint(-4, 4, (m, k), dtype=torch.int8, device="xpu")
            b = torch.randint(-4, 4, (k, n), dtype=torch.int8, device="xpu")
            secs = bench(lambda: torch._int_mm(a, b))
            tf = flops(m, n, k) / secs / 1e12
            emit(
                "TORCHCOMPUTE",
                {
                    "kind": "gemm_int8",
                    "dtype": "int8",
                    "m": m, "n": n, "k": k,
                    "seconds": round(secs, 6),
                    "value": round(tf, 2),
                    "unit": "GOPS",
                },
            )
            del a, b
            torch.xpu.empty_cache()
        except Exception as exc:  # noqa: BLE001
            emit("TORCHCOMPUTE", {"kind": "gemm_int8", "dtype": "int8",
                                  "m": m, "n": n, "k": k,
                                  "status": "error", "note": f"{type(exc).__name__}: {exc}"})

    # ---- vector 路径（bandwidth-bound，作为 ALU 探针的对照） -------------- #
    n_elem = 1 << 26  # 64M 元素
    for dtype in (torch.float32, torch.bfloat16):
        for op in ("add", "mul", "relu"):
            try:
                x = torch.randn(n_elem, dtype=dtype, device="xpu")
                y = torch.randn(n_elem, dtype=dtype, device="xpu")
                if op == "add":
                    fn = lambda: torch.add(x, y)
                elif op == "mul":
                    fn = lambda: torch.mul(x, y)
                else:
                    fn = lambda: torch.relu(x)
                secs = bench(fn)
                # add/mul: 读两个流 = 2×n×esize；relu: 读 1 个流
                nbytes = n_elem * (4 if dtype == torch.float32 else 2) * (1 if op == "relu" else 2)
                emit(
                    "TORCHCOMPUTE",
                    {
                        "kind": f"vector_{op}",
                        "dtype": str(dtype).replace("torch.", ""),
                        "elements": n_elem,
                        "seconds": round(secs, 6),
                        "value": round(nbytes / secs / 1e9, 1),
                        "unit": "GB/s",
                    },
                )
                del x, y
                torch.xpu.empty_cache()
            except Exception as exc:  # noqa: BLE001
                emit("TORCHCOMPUTE", {"kind": f"vector_{op}",
                                      "dtype": str(dtype).replace("torch.", ""),
                                      "status": "error", "note": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    os.environ.setdefault("ZE_AFFINITY_MASK", "0")
    raise SystemExit(main())
