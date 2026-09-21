"""GEMM / matmul 有效算力 sweep。

对应 ``docs/TODO/02-compute-peak.md`` Step 2 与 ``docs/TODO/05-ai-dl.md`` 3.1。

覆盖三类 shape：
1. **方阵** sweep    —— 找算力饱和点（n = 256 … 8192，可选 16384）
2. **真实场景 shape** —— LLM prefill / decode(GEMV) / FFN / 瘦长矩阵
3. **batched matmul** —— 注意力/多头场景

覆盖 dtype：fp32 / fp64 / fp16 / bf16，以及 INT8（经 ``torch._int_mm``，
走 XMX 且 int32 累加）。

关键看点：**BF16/FP16 相对 FP32 的倍数**。若只有 1.x 倍，说明 XMX 未启用。
"""

from __future__ import annotations

import torch

from .common import (
    BenchResult,
    DEVICE_TYPE,
    DTYPES,
    ResultStore,
    tflops_from,
    theoretical_tflops,
)

SUITE = "gemm"

DEFAULT_DTYPES = ["fp32", "bf16", "fp16", "fp64"]
DEFAULT_SQUARE_SIZES = [256, 512, 1024, 2048, 4096, 8192]
LARGE_SQUARE_SIZES = [16384]

# (label, M, N, K) —— 覆盖 LLM 典型形状
REAL_SHAPES: list[tuple[str, int, int, int]] = [
    ("llm_decode_gemv",      1,   4096,   4096),
    ("llm_decode_m16",      16,   4096,   4096),
    ("llm_decode_m64",      64,   4096,   4096),
    ("llm_prefill_m2k",   2048,   4096,   4096),
    ("ffn_up",            4096,  11008,   4096),
    ("ffn_down",          4096,   4096,  11008),
    ("k_dominant",        4096,   4096,  16384),
    ("n_dominant",        4096,  16384,   4096),
    ("skinny_k",          8192,   8192,    128),
    ("skinny_m",           128,   8192,   8192),
]

# (label, batch, M, N, K)
BATCHED_SHAPES: list[tuple[str, int, int, int, int]] = [
    ("attn_qkv_b32",       32,  1024,  1024,   64),
    ("batch_head_b16",     16,  2048,  2048,  128),
    ("batch_heads_b64",    64,   512,   512,   64),
]


# ---------------------------------------------------------------------------
def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _dtype_supported(device_index: int, dtype: torch.dtype) -> bool:
    props = torch.xpu.get_device_properties(device_index)
    if dtype == torch.float64:
        return bool(getattr(props, "has_fp64", 0))
    if dtype in (torch.float16, torch.bfloat16):
        return bool(getattr(props, "has_fp16", 1))
    return True


def _float_metrics(dtype_key: str, flops: float):
    def _fn(stats):
        m = {"tflops": tflops_from(flops, stats.median_ms)}
        theo = theoretical_tflops(dtype_key)
        if theo:
            m["theo_tflops"] = theo
            m["ratio_pct"] = 100.0 * m["tflops"] / theo
        return m

    return _fn


def _scratch(n_mb: float) -> None:
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
# 1) 方阵 sweep
# ---------------------------------------------------------------------------
def sweep_square(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    dev = kw["device_index"]
    sizes = [int(s) for s in getattr(ns, "gemm_sizes", DEFAULT_SQUARE_SIZES)]
    if getattr(ns, "large", False):
        sizes = sorted(set(sizes) | set(LARGE_SQUARE_SIZES))
    dtypes = [d for d in getattr(ns, "dtypes", DEFAULT_DTYPES)]

    for dk in dtypes:
        dt = DTYPES.get(dk)
        if dt is None:
            continue
        if not _dtype_supported(dev, dt):
            store.skip(SUITE, "matmul_square", {"dtype": dk}, f"{dk} 不被该设备支持")
            continue
        for n in sizes:
            params = {"dtype": dk, "m": n, "n": n, "k": n, "batch": 1, "shape": f"{n}x{n}x{n}"}
            try:
                a = torch.randn(n, n, device=DEVICE_TYPE, dtype=dt)
                b = torch.randn(n, n, device=DEVICE_TYPE, dtype=dt)
            except Exception as exc:  # OOM 等
                store.error(SUITE, "matmul_square", params, exc)
                _scratch(0)
                continue
            store.measure(
                SUITE, "matmul_square", params,
                lambda a=a, b=b: torch.matmul(a, b),
                compute_metrics=_float_metrics(dk, 2.0 * n ** 3),
                **kw,
            )
            del a, b
            _scratch(0)


# ---------------------------------------------------------------------------
# 2) 真实场景 shape
# ---------------------------------------------------------------------------
def sweep_real_shapes(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    dev = kw["device_index"]
    dtypes = [d for d in getattr(ns, "dtypes", DEFAULT_DTYPES)]

    for dk in dtypes:
        dt = DTYPES.get(dk)
        if dt is None or not _dtype_supported(dev, dt):
            continue
        for label, m, n, k in REAL_SHAPES:
            params = {"dtype": dk, "m": m, "n": n, "k": k, "batch": 1,
                      "shape": f"{m}x{n}x{k}", "label": label}
            try:
                a = torch.randn(m, k, device=DEVICE_TYPE, dtype=dt)
                b = torch.randn(k, n, device=DEVICE_TYPE, dtype=dt)
            except Exception as exc:
                store.error(SUITE, "matmul_rect", params, exc)
                _scratch(0)
                continue
            store.measure(
                SUITE, "matmul_rect", params,
                lambda a=a, b=b: torch.matmul(a, b),
                compute_metrics=_float_metrics(dk, 2.0 * m * n * k),
                **kw,
            )
            del a, b
            _scratch(0)


# ---------------------------------------------------------------------------
# 3) batched matmul
# ---------------------------------------------------------------------------
def sweep_batched(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    dev = kw["device_index"]
    dtypes = [d for d in getattr(ns, "dtypes", DEFAULT_DTYPES)]

    for dk in dtypes:
        dt = DTYPES.get(dk)
        if dt is None or not _dtype_supported(dev, dt):
            continue
        for label, batch, m, n, k in BATCHED_SHAPES:
            params = {"dtype": dk, "m": m, "n": n, "k": k, "batch": batch,
                      "shape": f"b{batch}x{m}x{n}x{k}", "label": label}
            try:
                a = torch.randn(batch, m, k, device=DEVICE_TYPE, dtype=dt)
                b = torch.randn(batch, k, n, device=DEVICE_TYPE, dtype=dt)
            except Exception as exc:
                store.error(SUITE, "matmul_batched", params, exc)
                _scratch(0)
                continue
            store.measure(
                SUITE, "matmul_batched", params,
                lambda a=a, b=b: torch.matmul(a, b),
                compute_metrics=_float_metrics(dk, 2.0 * batch * m * n * k),
                **kw,
            )
            del a, b
            _scratch(0)


# ---------------------------------------------------------------------------
# 4) INT8（XMX，int32 累加）
# ---------------------------------------------------------------------------
def sweep_int8(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    sizes = [n for n in getattr(ns, "gemm_sizes", DEFAULT_SQUARE_SIZES) if n >= 512]
    int_mm = getattr(torch, "_int_mm", None)
    if int_mm is None:
        store.skip(SUITE, "matmul_int8", {}, "torch._int_mm 不可用")
        return

    for n in sizes:
        params = {"dtype": "int8", "m": n, "n": n, "k": n, "batch": 1, "shape": f"{n}x{n}x{n}"}
        a = b = None
        try:
            a = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            b = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            # 预检：确认后端真的实现了该 kernel
            int_mm(a, b)
            torch.xpu.synchronize()
        except Exception as exc:
            store.skip(SUITE, "matmul_int8", params, f"INT8 GEMM 不支持: {type(exc).__name__}")
            del a, b
            _scratch(0)
            continue

        def _metrics(stats, n=n):
            return {"tops": 2.0 * n ** 3 / (stats.median_ms * 1e-3) / 1e12}

        store.measure(
            SUITE, "matmul_int8", params,
            lambda a=a, b=b: int_mm(a, b),
            compute_metrics=_metrics,
            note="int32 累加，INT8 理论为 BF16 的 2×",
            **kw,
        )
        del a, b
        _scratch(0)


# ---------------------------------------------------------------------------
def _summarize(store: ResultStore) -> None:
    rows = [r for r in store.suite(SUITE) if "tflops" in r.metrics]
    best: dict[str, BenchResult] = {}
    for r in rows:
        dk = r.params.get("dtype", "?")
        if dk not in best or r.metrics["tflops"] > best[dk].metrics["tflops"]:
            best[dk] = r
    for dk in ("fp32", "fp64", "fp16", "bf16"):
        if dk in best:
            r = best[dk]
            store.add_note(
                f"GEMM 峰值 {dk}: {r.metrics['tflops']:.2f} TFLOPS @ {r.params.get('shape')}"
            )
    # 检查公式推导的理论值是否站得住脚
    for dk in ("fp32", "fp64"):
        r = best.get(dk)
        if r is None:
            continue
        theo = theoretical_tflops(dk)
        if theo and r.metrics["tflops"] > 1.02 * theo:
            store.add_note(
                f"⚠ {dk} 实测 {r.metrics['tflops']:.2f} TFLOPS 高于理论假设 "
                f"{theo:.2f}（{100 * r.metrics['tflops'] / theo:.0f}%），"
                f"理论公式需根据实测修正"
            )
    base = best.get("fp32")
    if base:
        for dk in ("fp16", "bf16"):
            if dk in best:
                ratio = best[dk].metrics["tflops"] / base.metrics["tflops"]
                tag = "XMX 已启用" if ratio >= 1.5 else "⚠ 加速比 < 1.5×，疑似 XMX 未启用"
                store.add_note(f"XMX 加速比 {dk}/fp32 = {ratio:.2f}×（{tag}）")


def run(ns, store: ResultStore) -> None:
    """执行 gemm suite。"""
    sweep_square(store, ns)
    if getattr(ns, "real_shapes", True):
        sweep_real_shapes(store, ns)
    if getattr(ns, "batched", True):
        sweep_batched(store, ns)
    if getattr(ns, "int8", True):
        sweep_int8(store, ns)
    _summarize(store)
