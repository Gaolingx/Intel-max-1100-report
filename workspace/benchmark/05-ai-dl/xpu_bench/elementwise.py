"""Elementwise 算子 + 显存带宽测试。

对应 ``docs/TODO/03-memory-bandwidth.md`` 与 ``docs/TODO/05-ai-dl.md`` 3.1。

三个层次：

* ``elementwise`` suite：copy / scale / add / mul / triad + 激活函数，
  用「有效带宽 GB/s」衡量（这些算子全是 memory-bound）。
* ``membw`` suite：
  - ``sweep_dtype_bw``  同一元素数下不同 dtype，检验带宽是否与元素宽度无关；
  - ``sweep_size``      数组从小到大，找带宽饱和点（**必须远超 192 MB L2**）；
  - ``sweep_host_transfer`` host↔device 传输带宽（内存倒挂的关键证据）。

带宽口径：bytes = io_factor × N × itemsize，io_factor 为该算子每次迭代的
读+写元素次数（copy=2，add/triad=3）。
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

SUITE = "elementwise"
BW_SUITE = "membw"

DEFAULT_DTYPE = "fp32"


# ---------------------------------------------------------------------------
# 算子表
# ---------------------------------------------------------------------------
def _build_ops(a, b, out):
    """返回 [(name, io_factor, callable), ...]。

    io_factor 为该算子每迭代的显存访问元素次数（读 + 写）。
    """
    return [
        ("copy",    2, lambda: out.copy_(a)),
        ("scale",   2, lambda: torch.mul(a, 2.0, out=out)),
        ("add",     3, lambda: torch.add(a, b, out=out)),
        ("mul",     3, lambda: torch.mul(a, b, out=out)),
        ("sub",     3, lambda: torch.sub(a, b, out=out)),
        ("triad",   3, lambda: torch.add(a, b, alpha=1.5, out=out)),
        ("relu",    2, lambda: torch.ops.aten.relu.out(a, out=out)),  # torch.relu 无 out=
        ("sigmoid", 2, lambda: torch.sigmoid(a, out=out)),
        ("tanh",    2, lambda: torch.tanh(a, out=out)),
        ("exp",     2, lambda: torch.exp(a, out=out)),
        ("gelu",    2, lambda: F.gelu(a, out=out)),
        ("silu",    2, lambda: F.silu(a)),  # F.silu 无 out=，内部新建输出
    ]


# int8 只支持纯算术/比较类
_INT8_OPS = {"copy", "add", "mul", "sub", "relu"}


def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _bw_metrics(io_factor: int, n: int, itemsize: int):
    def _fn(stats):
        nbytes = io_factor * n * itemsize
        return {
            "gbps": gbps_from(nbytes, stats.median_ms),
            "gelems": n / (stats.median_ms * 1e-3) / 1e9,
        }

    return _fn


# ---------------------------------------------------------------------------
# 1) elementwise suite
# ---------------------------------------------------------------------------
def sweep_ops(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    size_mb = int(getattr(ns, "size_mb", 1024))
    dtypes = [d for d in getattr(ns, "dtypes", [DEFAULT_DTYPE])]

    for dk in dtypes:
        dt = DTYPES.get(dk)
        if dt is None:
            continue
        itemsize = torch.tensor([], dtype=dt).element_size()
        n = numel_for_bytes(size_mb << 20, dt)
        try:
            a = torch.randn(n, device=DEVICE_TYPE, dtype=dt)
            b = torch.randn(n, device=DEVICE_TYPE, dtype=dt)
            out = torch.empty(n, device=DEVICE_TYPE, dtype=dt)
        except Exception as exc:  # OOM
            store.error(SUITE, "elementwise", {"dtype": dk, "elems": n}, exc)
            continue

        for name, io_factor, fn in _build_ops(a, b, out):
            if dk == "int8" and name not in _INT8_OPS:
                continue
            params = {
                "dtype": dk,
                "elems": n,
                "size_mib": size_mb,
                "io_factor": io_factor,
                "shape": f"1D[{n}]",
            }
            store.measure(
                SUITE, name, params, fn,
                compute_metrics=_bw_metrics(io_factor, n, itemsize),
                **kw,
            )
        del a, b, out
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    best = store.best(SUITE, "gbps")
    if best:
        store.add_note(
            f"elementwise 峰值有效带宽: {best.metrics['gbps']:.1f} GB/s "
            f"({best.name}, dtype={best.params.get('dtype')}, size={best.params.get('size_mib')} MiB)"
        )


# ---------------------------------------------------------------------------
# 2) dtype 带宽对照
# ---------------------------------------------------------------------------
def sweep_dtype_bw(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    size_mb = int(getattr(ns, "size_mb", 1024))
    dtypes = [d for d in getattr(ns, "dtypes", ["fp32", "bf16", "fp16", "int8"])]

    for dk in dtypes:
        dt = DTYPES.get(dk, torch.int8 if dk == "int8" else None)
        if dt is None:
            continue
        itemsize = torch.tensor([], dtype=dt).element_size()
        n = numel_for_bytes(size_mb << 20, dt)
        try:
            a = torch.randn(n, device=DEVICE_TYPE, dtype=dt) if dk != "int8" \
                else torch.randint(0, 100, (n,), device=DEVICE_TYPE, dtype=dt)
            b = torch.randn(n, device=DEVICE_TYPE, dtype=dt) if dk != "int8" \
                else torch.randint(0, 100, (n,), device=DEVICE_TYPE, dtype=dt)
            out = torch.empty(n, device=DEVICE_TYPE, dtype=dt)
        except Exception as exc:
            store.error(BW_SUITE, "dtype_bw", {"dtype": dk}, exc)
            continue

        for name, io_factor, fn in [
            ("copy", 2, lambda: out.copy_(a)),
            ("add", 3, lambda: torch.add(a, b, out=out)),
        ]:
            params = {"dtype": dk, "op": name, "elems": n, "size_mib": size_mb,
                      "io_factor": io_factor, "shape": f"1D[{n}]"}
            store.measure(
                BW_SUITE, f"dtype_{name}", params, fn,
                compute_metrics=_bw_metrics(io_factor, n, itemsize),
                note="检验带宽是否与元素宽度无关",
                **kw,
            )
        del a, b, out
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
# 3) size sweep —— 找饱和点 / 暴露 cache 命中
# ---------------------------------------------------------------------------
DEFAULT_SIZE_MIB = [4, 16, 64, 128, 192, 256, 512, 1024, 2048]


def sweep_size(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    dtype_key = getattr(ns, "bw_dtype", "fp32")
    dt = DTYPES.get(dtype_key, torch.float32)
    itemsize = torch.tensor([], dtype=dt).element_size()
    sizes = [int(s) for s in getattr(ns, "bw_sizes", DEFAULT_SIZE_MIB)]

    for size_mib in sizes:
        n = numel_for_bytes(size_mib << 20, dt)
        try:
            a = torch.randn(n, device=DEVICE_TYPE, dtype=dt)
            out = torch.empty(n, device=DEVICE_TYPE, dtype=dt)
        except Exception as exc:
            store.error(BW_SUITE, "size_copy", {"size_mib": size_mib}, exc)
            continue
        params = {"dtype": dtype_key, "op": "copy", "elems": n,
                  "size_mib": size_mib, "io_factor": 2, "shape": f"1D[{n}]"}
        store.measure(
            BW_SUITE, "size_copy", params,
            lambda a=a, out=out: out.copy_(a),
            compute_metrics=_bw_metrics(2, n, itemsize),
            note="; 192 MiB 以下会被 L2 命中，读数偏高",
            **kw,
        )
        del a, out
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    # 用最大规模的数据点代表 HBM 带宽上限
    big = [r for r in store.suite(BW_SUITE)
           if r.name == "size_copy" and r.params.get("size_mib", 0) >= 256]
    if big:
        best = max(big, key=lambda r: r.metrics["gbps"])
        store.add_note(
            f"HBM 实测带宽上限（copy, ≥256 MiB）: {best.metrics['gbps']:.1f} GB/s "
            f"@ {best.params.get('size_mib')} MiB"
        )


# ---------------------------------------------------------------------------
# 4) host <-> device 传输带宽（内存倒挂诊断）
# ---------------------------------------------------------------------------
H2D_BYTES_MIB = 256


def sweep_host_transfer(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    size_mib = int(getattr(ns, "xfer_mib", H2D_BYTES_MIB))
    n = int(size_mib << 20) // 4  # fp32

    pinned_ok = True
    try:
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    except Exception:  # noqa: BLE001 - 某些后端不支持 pinned
        pinned_ok = False
        host = torch.empty(n, dtype=torch.float32)

    try:
        dev = torch.empty(n, dtype=torch.float32, device=DEVICE_TYPE)
    except Exception as exc:
        store.error(BW_SUITE, "host_transfer", {"size_mib": size_mib}, exc)
        return

    nbytes = n * 4
    tag = "pinned" if pinned_ok else "pageable"

    store.measure(
        BW_SUITE, "h2d", {"direction": "H2D", "bytes_mib": size_mib, "pin": tag},
        lambda: dev.copy_(host),
        compute_metrics=lambda st: {"gbps": gbps_from(nbytes, st.median_ms)},
        **kw,
    )
    store.measure(
        BW_SUITE, "d2h", {"direction": "D2H", "bytes_mib": size_mib, "pin": tag},
        lambda: host.copy_(dev),
        compute_metrics=lambda st: {"gbps": gbps_from(nbytes, st.median_ms)},
        **kw,
    )
    del host, dev
    torch.xpu.synchronize()
    torch.xpu.empty_cache()

    # 注意：只看 H2D/D2H，避免 size sweep 中 L2 命中的虚高值混入
    xfer = [r for r in store.suite(BW_SUITE) if r.name in ("h2d", "d2h")]
    for r in xfer:
        store.add_note(
            f"{r.name.upper()} 带宽: {r.metrics['gbps']:.1f} GB/s "
            f"({r.params.get('pin')}, {r.params.get('bytes_mib')} MiB)"
        )


# ---------------------------------------------------------------------------
def run(ns, store: ResultStore) -> None:
    """elementwise suite：算子级有效带宽。"""
    sweep_ops(store, ns)


def run_bw(ns, store: ResultStore) -> None:
    """membw suite：带宽口径的专项测试。"""
    if getattr(ns, "dtype_bw", True):
        sweep_dtype_bw(store, ns)
    if getattr(ns, "size_sweep", True):
        sweep_size(store, ns)
    if getattr(ns, "host_transfer", True):
        sweep_host_transfer(store, ns)
