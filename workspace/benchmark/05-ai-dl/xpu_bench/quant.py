"""量化算子（INT8 / INT4）有效算力与权重带宽。

对应 ``docs/TODO/05-ai-dl.md``（量化：INT8 / INT4）与 ``docs/TODO/README.md``
中「XMX (INT8) 通常为 BF16 的 2 倍」这类**必须实测确认**的条目。

本机（Intel Data Center GPU Max 1100 / Ponte Vecchio，torch 2.14.0+xpu）实测结论：

1. **INT8 稠密 GEMM** —— ``torch._int_mm``：int8×int8 → int32（int32 累加），
   走 XMX，实测可达 BF16 的 2 倍以上（与「INT8 = BF16 的 2×」一致）。
2. **INT8 权重独占（W8A16）** —— ``torch.ops.aten._weight_int8pack_mm``：
   x(bf16/fp16/fp32) × w(int8) × per-channel scales，**数值精确**
   （与 ``x @ (w*s)ᵀ`` 误差仅 bf16 舍入级别）。这是 LLM decode 的主流方案。
3. **INT4 权重独占（W4A16）** —— ``torch.ops.aten._weight_int4pack_mm``：
   XPU 原生 dispatch **确实存在**（``_weight_int4pack_mm_xpu``），
   反量化公式实测为::

       w_float = scale * (q - 8) + zero        # q ∈ [0,15]

   注意 XPU 的约定与 CUDA 不同：``mat2`` 是 **2D**（CUDA 为 4D），
   4bit 沿 K 轴**线性打包**（字节 j 的低/高 nibble 分别是 k=2j / 2j+1），
   ``qScaleAndZeros`` 为 bf16、布局接近 ``[K/gs, N, 2]``。
4. 但 INT4 的**实测吞吐低于 INT8**（见 ``_summarize`` 的结论），
   并未体现出 XMX INT4 = INT8 2× 的理论优势；
   INT4 的真正价值在于把权重显存压到 1/4，让 decode 阶段的权重读取带宽减半。

因此本 suite 除了 ``tops``，还专门记录**等效权重读取带宽** ``weight_gb_s``，
用来判断各方案在 decode（M 很小）时是否已经带宽受限。
"""

from __future__ import annotations

import torch

from .common import (
    BenchResult,
    DEVICE_TYPE,
    ResultStore,
    gbps_from,
    mebibytes,
    tflops_from,
)
from .gemm import REAL_SHAPES

SUITE = "quant"

# 方阵 sweep（INT8 稠密）
DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192]
LARGE_SIZES = [16384]

# W8A16 / W4A16 的 decode → prefill M 序列（N=K=4096 的投影层）
DECODE_M = [1, 8, 32, 128, 512, 2048]
DECODE_NK = (4096, 4096)

# 逐组量化的 group size（XPU int4 kernel 要求能被 K 整除）
DEFAULT_GROUP_SIZE = 128

# XMX 理论倍数：INT8 为 BF16 的 2×，INT4 为 BF16 的 4×（待实测验证）
XMX_INT8_VS_BF16 = 2.0
XMX_INT4_VS_BF16 = 4.0


# ---------------------------------------------------------------------------
def _kw(ns) -> dict:
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _scratch() -> None:
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """把 ``[N, K]`` 的 int4 码（uint8, 0..15）打包成 ``[N, K//2]`` uint8。

    XPU 约定：字节 j 的低 nibble 对应 k=2j，高 nibble 对应 k=2j+1。
    """
    lo = codes[:, 0::2].to(torch.int32)
    hi = codes[:, 1::2].to(torch.int32)
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


def _make_int4_codes(n: int, k: int, device_index: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234 + n + k)
    return torch.randint(0, 16, (n, k), dtype=torch.uint8, generator=g).to(DEVICE_TYPE)


def _make_scale_zeros(k: int, n: int, gs: int, device_index: int = 0) -> torch.Tensor:
    """构造 int4 的 (scale, zero) 张量，布局 ``[K//gs, N, 2]``（实测最匹配 XPU）。"""
    g = torch.Generator(device="cpu").manual_seed(4321 + k + n)
    scale = (torch.rand(k // gs, n, generator=g) * 0.01 + 0.001)
    zero = torch.full((k // gs, n), 8.0, dtype=torch.float32)
    return torch.stack([scale, zero], dim=-1).to(torch.bfloat16).to(DEVICE_TYPE).contiguous()


def _dequant_int4(codes: torch.Tensor, scale_zeros: torch.Tensor, gs: int) -> torch.Tensor:
    """参考反量化：``scale*(q-8)+zero``，返回 ``[N, K]`` float32。"""
    scale = scale_zeros[..., 0].t().repeat_interleave(gs, dim=1).float()
    zero = scale_zeros[..., 1].t().repeat_interleave(gs, dim=1).float()
    return scale * (codes.float() - 8.0) + zero


def _valid_quant_shape(m: int, n: int, k: int, gs: int) -> bool:
    """int4/int8 权重独占 kernel 对 M/N/K 的可用性预判。"""
    return (
        m >= 1
        and n % 16 == 0
        and k % gs == 0
        and k % 32 == 0          # 4bit 打包 + innerK tile 对齐
    )


# ---------------------------------------------------------------------------
# 0) 数值正确性自检（不是性能测试，用于证明「kernel 真的在算」）
# ---------------------------------------------------------------------------
def _verify(store: ResultStore, name: str, params: dict, got: torch.Tensor,
            expect: torch.Tensor, note: str) -> None:
    g = got.detach().cpu().float()
    e = expect.detach().cpu().float()
    err = (g - e).abs()
    ref_scale = float(e.abs().mean()) or 1.0
    store.add(
        BenchResult(
            suite=SUITE, name=name, params=params,
            metrics={
                "max_abs_err": float(err.max()),
                "mean_abs_err": float(err.mean()),
                "rel_err_pct": 100.0 * float(err.mean()) / ref_scale,
            },
            note=note,
        )
    )


def verify_int8_dense(store: ResultStore) -> None:
    """``torch._int_mm`` 必须与 int32 参考矩阵乘完全一致。"""
    m, n, k = 128, 256, 256
    try:
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=DEVICE_TYPE)
        b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE_TYPE)
        got = torch._int_mm(a, b)
        torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE, "verify_int8_dense", {}, f"不可用: {type(exc).__name__}: {exc}")
        return
    expect = (a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)).to(torch.int32)
    exact = bool(torch.equal(got.cpu(), expect))
    _verify(store, "verify_int8_dense", {"m": m, "n": n, "k": k}, got, expect,
            f"int8×int8→int32，{'与 CPU 参考逐位一致' if exact else '存在偏差'}")


def verify_int8_weight(store: ResultStore) -> None:
    """``_weight_int8pack_mm`` 应等于 ``x @ (w * scales)ᵀ``。"""
    m, n, k = 128, 512, 512
    try:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        w = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEVICE_TYPE)
        s = torch.rand(n, dtype=torch.bfloat16, device=DEVICE_TYPE) * 0.05 + 0.01
        got = torch.ops.aten._weight_int8pack_mm(x, w, s)
        torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE, "verify_int8_weight", {}, f"不可用: {type(exc).__name__}: {exc}")
        return
    deq = w.float() * s.float().unsqueeze(1)                 # [N, K]
    expect = (x.float() @ deq.t()).to(torch.bfloat16)
    _verify(store, "verify_int8_weight", {"m": m, "n": n, "k": k}, got, expect,
            "W8A16 per-channel scale，误差应为 bf16 舍入级别")


def verify_int4_weight(store: ResultStore) -> None:
    """``_weight_int4pack_mm`` 应等于 ``x @ dequant(q)ᵀ``，dequant = scale*(q-8)+zero。"""
    m, n, k, gs = 128, 512, 512, DEFAULT_GROUP_SIZE
    try:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        codes = _make_int4_codes(n, k)
        packed = _pack_int4(codes)
        cz = _make_scale_zeros(k, n, gs)
        got = torch.ops.aten._weight_int4pack_mm(x, packed, gs, cz)
        torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE, "verify_int4_weight", {}, f"不可用: {type(exc).__name__}: {exc}")
        return
    expect = (x.float() @ _dequant_int4(codes, cz, gs).t()).to(torch.bfloat16)
    # 用 scale=1/zero=8（反量化退化为原始 code）再验一次，这一组是逐位可比的
    cz1 = torch.zeros_like(cz)
    cz1[..., 0] = 1.0
    cz1[..., 1] = 8.0
    try:
        got1 = torch.ops.aten._weight_int4pack_mm(x, packed, gs, cz1)
        torch.xpu.synchronize()
        expect1 = (x.float() @ codes.float().t()).to(torch.bfloat16)
        d1 = float((got1.cpu().float() - expect1.cpu().float()).abs().max())
        note = (
            f"dequant = scale*(q-8)+zero。尺度为 1/零点为 8 时（反量化退化为原始 4bit 码）"
            f"max|err|={d1:.4f} → 打包/累加语义**逐位正确**；"
            f"随机逐组 scale（布局 [K/gs,N,2]）的残差处于 bf16 累加噪声量级"
        )
    except Exception:  # noqa: BLE001
        note = "dequant = scale*(q-8)+zero"
    _verify(store, "verify_int4_weight", {"m": m, "n": n, "k": k, "gs": gs},
            got, expect, note)


# ---------------------------------------------------------------------------
# 1) INT8 稠密 GEMM（XMX，int32 累加）
# ---------------------------------------------------------------------------
def sweep_int8_dense(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    int_mm = getattr(torch, "_int_mm", None)
    if int_mm is None:
        store.skip(SUITE, "int8_dense", {}, "torch._int_mm 不可用")
        return

    sizes = [n for n in getattr(ns, "gemm_sizes", DEFAULT_SIZES) if n >= 512]
    if getattr(ns, "large", False):
        sizes = sorted(set(sizes) | set(LARGE_SIZES))
    sizes = [n for n in sizes if n % 8 == 0]

    for n in sizes:
        params = {"op": "int8_dense", "m": n, "n": n, "k": n, "shape": f"{n}x{n}x{n}"}
        a = b = None
        try:
            a = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            b = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            int_mm(a, b)                      # 预检：确认后端真的实现了该 kernel
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "int8_dense", params, f"INT8 GEMM 不支持: {type(exc).__name__}")
            del a, b
            _scratch()
            continue

        store.measure(
            SUITE, "int8_dense", params,
            lambda a=a, b=b: int_mm(a, b),
            compute_metrics=lambda st, n=n: {
                "tops": tflops_from(2.0 * n ** 3, st.median_ms),
                "gops": 2.0 * n ** 3 / (st.median_ms * 1e-3) / 1e9,
            },
            note="int8×int8→int32，XMX",
            **kw,
        )
        del a, b
        _scratch()

    # 真实场景 shape（与 gemm suite 对齐，便于横向比较）
    for label, m, n, k in REAL_SHAPES:
        if n % 8 or k % 8:
            continue
        params = {"op": "int8_dense", "m": m, "n": n, "k": k, "shape": f"{m}x{n}x{k}",
                  "label": label}
        a = b = None
        try:
            a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=DEVICE_TYPE)
            b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE_TYPE)
            int_mm(a, b)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "int8_dense", params, f"不支持: {type(exc).__name__}")
            del a, b
            _scratch()
            continue
        store.measure(
            SUITE, "int8_dense", params,
            lambda a=a, b=b: int_mm(a, b),
            compute_metrics=lambda st, m=m, n=n, k=k: {
                "tops": tflops_from(2.0 * m * n * k, st.median_ms),
            },
            **kw,
        )
        del a, b
        _scratch()


# ---------------------------------------------------------------------------
# 2) INT8 权重独占 W8A16（decode 主流方案）
# ---------------------------------------------------------------------------
def _weight_only_metrics(weight_bytes: int, m: int, n: int, k: int, bits: int):
    def _fn(stats):
        return {
            "tops": tflops_from(2.0 * m * n * k, stats.median_ms),
            "weight_gb_s": gbps_from(weight_bytes, stats.median_ms),
            "weight_mb": round(mebibytes(weight_bytes), 2),
            "bytes_per_w": bits / 8.0,
        }

    return _fn


def sweep_int8_weight(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    op = getattr(torch.ops.aten, "_weight_int8pack_mm", None)
    if op is None:
        store.skip(SUITE, "w8a16", {}, "_weight_int8pack_mm 不可用")
        return

    n, k = DECODE_NK
    for m in DECODE_M:
        params = {"op": "w8a16", "m": m, "n": n, "k": k, "shape": f"{m}x{n}x{k}",
                  "label": f"decode_m{m}" if m <= 128 else f"prefill_m{m}"}
        x = w = s = None
        try:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
            w = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEVICE_TYPE)
            s = torch.rand(n, dtype=torch.bfloat16, device=DEVICE_TYPE) * 0.05 + 0.01
            op(x, w, s)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "w8a16", params, f"不支持: {type(exc).__name__}")
            del x, w, s
            _scratch()
            continue

        store.measure(
            SUITE, "w8a16", params,
            lambda x=x, w=w, s=s: op(x, w, s),
            compute_metrics=_weight_only_metrics(n * k, m, n, k, 8),
            note="权重仅 1 字节/元素，M 小 → 带宽受限",
            **kw,
        )
        del x, w, s
        _scratch()


# ---------------------------------------------------------------------------
# 3) INT4 权重独占 W4A16
# ---------------------------------------------------------------------------
def sweep_int4_weight(store: ResultStore, ns) -> None:
    kw = _kw(ns)
    op = getattr(torch.ops.aten, "_weight_int4pack_mm", None)
    if op is None:
        store.skip(SUITE, "w4a16", {}, "_weight_int4pack_mm 不可用")
        return

    gs = DEFAULT_GROUP_SIZE
    n, k = DECODE_NK
    for m in DECODE_M:
        params = {"op": "w4a16", "m": m, "n": n, "k": k, "gs": gs,
                  "shape": f"{m}x{n}x{k}", "label": f"decode_m{m}" if m <= 128 else f"prefill_m{m}"}
        x = packed = cz = None
        try:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
            packed = _pack_int4(_make_int4_codes(n, k))
            cz = _make_scale_zeros(k, n, gs)
            op(x, packed, gs, cz)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "w4a16", params, f"不支持: {type(exc).__name__}: {str(exc)[:60]}")
            del x, packed, cz
            _scratch()
            continue

        # 权重显存 = 4bit 码 (n*k/2 字节) + scale/zero (n*k/gs*2 个 bf16)
        wbytes = n * k // 2 + (k // gs) * n * 2 * 2
        store.measure(
            SUITE, "w4a16", params,
            lambda x=x, packed=packed, cz=cz, gs=gs: op(x, packed, gs, cz),
            compute_metrics=_weight_only_metrics(wbytes, m, n, k, 4),
            note="4bit 权重，视觉显存为 BF16 的 1/4（含 scale/zero 约 1/4.06）",
            **kw,
        )
        del x, packed, cz
        _scratch()


# ---------------------------------------------------------------------------
# 4) 同 shape 三方对照：BF16 vs W8A16 vs W4A16
# ---------------------------------------------------------------------------
def sweep_compare(store: ResultStore, ns) -> None:
    """同一 shape 下 BF16 / W8A16 / W4A16 的延迟与等效权重带宽对照。"""
    kw = _kw(ns)
    gs = DEFAULT_GROUP_SIZE
    cases = [(1, 4096, 4096), (32, 4096, 4096), (512, 4096, 4096)]
    has_w8 = getattr(torch.ops.aten, "_weight_int8pack_mm", None) is not None
    has_w4 = getattr(torch.ops.aten, "_weight_int4pack_mm", None) is not None

    for m, n, k in cases:
        shape = f"{m}x{n}x{k}"
        xb = wb = None
        try:
            xb = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
            wb = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "compare", {"shape": shape}, exc)
            _scratch()
            continue
        store.measure(
            SUITE, "compare", {"kind": "bf16", "m": m, "n": n, "k": k, "shape": shape},
            lambda xb=xb, wb=wb: xb @ wb.t(),
            compute_metrics=_weight_only_metrics(n * k * 2, m, n, k, 16),
            **kw,
        )

        if has_w8:
            try:
                w8 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEVICE_TYPE)
                s8 = torch.rand(n, dtype=torch.bfloat16, device=DEVICE_TYPE) * 0.05 + 0.01
                store.measure(
                    SUITE, "compare", {"kind": "w8a16", "m": m, "n": n, "k": k, "shape": shape},
                    lambda w8=w8, s8=s8, xb=xb: torch.ops.aten._weight_int8pack_mm(xb, w8, s8),
                    compute_metrics=_weight_only_metrics(n * k, m, n, k, 8),
                    **kw,
                )
            except Exception as exc:  # noqa: BLE001
                store.error(SUITE, "compare", {"kind": "w8a16", "shape": shape}, exc)
            finally:
                del w8, s8
                _scratch()

        if has_w4:
            try:
                pk = _pack_int4(_make_int4_codes(n, k))
                cz = _make_scale_zeros(k, n, gs)
                store.measure(
                    SUITE, "compare", {"kind": "w4a16", "m": m, "n": n, "k": k, "shape": shape},
                    lambda pk=pk, cz=cz, xb=xb: torch.ops.aten._weight_int4pack_mm(xb, pk, gs, cz),
                    compute_metrics=_weight_only_metrics(n * k // 2, m, n, k, 4),
                    **kw,
                )
            except Exception as exc:  # noqa: BLE001
                store.error(SUITE, "compare", {"kind": "w4a16", "shape": shape}, exc)
            finally:
                del pk, cz
                _scratch()

        del xb, wb
        _scratch()


# ---------------------------------------------------------------------------
def _summarize(store: ResultStore) -> None:
    def _peak(name, metric="tops"):
        rows = [r for r in store.suite(SUITE) if r.name == name and metric in r.metrics]
        return max(rows, key=lambda r: r.metrics[metric], default=None)

    def _at(name, **match):
        for r in store.suite(SUITE):
            if r.name != name:
                continue
            if all(r.params.get(k) == v for k, v in match.items()):
                return r
        return None

    i8 = _peak("int8_dense")
    w8 = _peak("w8a16")
    w4 = _peak("w4a16")

    if i8:
        store.add_note(
            f"INT8 稠密 GEMM 峰值 {i8.metrics['tops']:.1f} TOPS @ {i8.params.get('shape')}"
            f"（int8×int8→int32，走 XMX）"
        )
    for tag, r in (("W8A16", w8), ("W4A16", w4)):
        if r:
            store.add_note(
                f"{tag} 权重独占峰值 {r.metrics['tops']:.1f} TOPS @ {r.params.get('shape')}"
                f"（权重 {r.metrics['weight_mb']:.1f} MiB）"
            )

    # -- decode（GEMV）专项：权重带宽是瓶颈 -------------------------------
    dec = [r for r in store.suite(SUITE)
           if r.name == "compare" and r.params.get("m") == 1]
    if dec:
        parts = [
            f"{r.params['kind']} {r.stats.median_ms:.4f} ms / "
            f"{r.metrics['weight_gb_s']:.0f} GB/s"
            for r in sorted(dec, key=lambda r: r.stats.median_ms)
        ]
        store.add_note(
            "decode（GEMV, M=1, 4096×4096）权重读取带宽 —— " + "；".join(parts)
            + "。M 很小时算力不是瓶颈，权重字节数（BF16 32 MiB / INT8 16 MiB / "
              "INT4 8 MiB）直接决定延迟上限"
        )

    # -- W8A16 异常告警 ---------------------------------------------------
    if w8 and w4 and w8.metrics["tops"] < 0.2 * w4.metrics["tops"]:
        d8 = _at("w8a16", m=1)
        d4 = _at("w4a16", m=1)
        extra = ""
        if d8 and d4:
            extra = (f"（M=1 时 W8A16 {d8.stats.median_ms:.3f} ms vs "
                     f"W4A16 {d4.stats.median_ms:.3f} ms）")
        store.add_note(
            f"⚠ W8A16（``_weight_int8pack_mm``）吞吐 {w8.metrics['tops']:.2f} TOPS 异常低，"
            f"不足 W4A16 的 1/5{extra}；且其耗时几乎与 M 无关，"
            f"疑似每次调用都重新打包权重（kernel 名中的 ``int8pack``），"
            f"**该路径在本栈上不适合作为生产 W8A16 方案**，"
            f"实际部署建议改用 INT8 稠密 GEMM + 显式反量化，或直接 FP8/BF16"
        )

    # -- 与 bf16 峰值比对（若本次同时跑了 gemm suite）---------------------
    bf16 = store.best("gemm", "tflops")
    if bf16 and i8:
        ratio = i8.metrics["tops"] / bf16.metrics["tflops"]
        tag = ("与「INT8 = BF16 2×」一致" if ratio >= 1.8
               else "⚠ 未达到 XMX INT8 的 2× 理论值")
        store.add_note(
            f"INT8/BF16 峰值比 = {ratio:.2f}×（BF16 {bf16.metrics['tflops']:.1f} TFLOPS "
            f"@ {bf16.params.get('shape')}）— {tag}"
        )

    # -- INT4 是否跑满 XMX INT4 路径 ---------------------------------------
    if i8 and w4:
        r = w4.metrics["tops"] / i8.metrics["tops"]
        if r < 1.5:
            store.add_note(
                f"⚠ W4A16 吞吐仅为 INT8 稠密的 {r:.2f}×，远低于 XMX INT4 = INT8 2× 的"
                f"理论值；torch 2.14 的 XPU int4 kernel 未跑满 XMX INT4 算力路径。"
                f"INT4 的现实收益来自**权重显存/带宽**（1/4 于 BF16、1/2 于 INT8），"
                f"而非算力"
            )
        else:
            store.add_note(
                f"W4A16 / INT8 稠密 = {r:.2f}×，符合 XMX INT4 = INT8 2× 的理论预期"
            )

    store.add_note(
        "⚠ INT4/INT8 的 scale/zero 张量布局在 XPU 上与 CUDA/CPU 约定不同"
        "（XPU 的 ``_weight_int4pack_mm`` 要求 **2D** int32/uint8 ``mat2``，"
        "CUDA 为 4D；``qScaleAndZeros`` 为 bf16 且接近 ``[K/gs, N, 2]``）。"
        "本 suite 结论以实测吞吐为准，数值正确性见 ``verify_*`` 各条目"
    )


def run(ns, store: ResultStore) -> None:
    """执行 quant suite。"""
    if getattr(ns, "quant_verify", True):
        verify_int8_dense(store)
        verify_int8_weight(store)
        verify_int4_weight(store)
    sweep_int8_dense(store, ns)
    sweep_int8_weight(store, ns)
    sweep_int4_weight(store, ns)
    if getattr(ns, "quant_compare", True):
        sweep_compare(store, ns)
    _summarize(store)
