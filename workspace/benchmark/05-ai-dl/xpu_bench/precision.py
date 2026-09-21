"""数值精度支持矩阵与各精度吞吐（vector 路径 + XMX 矩阵路径）。

对应 ``docs/TODO/02-compute-peak.md``（算力峰值 / XMX）与
``docs/TODO/05-ai-dl.md``（混合精度 / XMX 是否启用）。

厂商规格页（Max 1100）把数值格式分成两栏：

====================  ==========================================================
**Vector**            INT4 · INT8 · INT16 · INT32 · FP16 · BF16 · FP32 · FP64
**Matrix (XMX)**      INT1 · INT2 · INT4 · INT8 · MXFP4 · NVFP4 · MXFP6 · FP8 ·
                      MXFP8 · FP16 · BF16 · FP32 · TF32 · FP64
====================  ==========================================================

但**硬件能力 ≠ 本软件栈可用**。本 suite 对每一种格式都真跑一次
``vector`` 与 ``matmul``，记录 ``OK`` / ``FAIL`` 及失败原因（不是估算），
回答两个问题：

1. **这张卡到底能用哪些精度？**（capability 表）
2. **每种精度的实际吞吐是多少？**
   - vector 路径：elementwise 有效带宽 (GB/s) 与元素速率 (Gops/s)
   - 矩阵路径：matmul TFLOPS / TOPS，并给出相对 FP32（ALU 基线）的倍数

.. note::
   **为什么 vector 路径只测带宽？**
   本卡 ALU 峰值约 22 TFLOPS，而 HBM 只有约 800 GB/s → 约 27 FLOP/byte 的
   算力带宽比。而 elementwise 算子的算术强度只有约 0.1 FLOP/byte，
   **任何 elementwise 算子都是纯带宽受限**（实测 exp/sqrt 与 copy 同速，
   见报告）。因此 vector 路径的「精度差异」主要体现在
   **能否执行 / 每元素字节数**，而非 FLOPs。
   真正的 ALU 算力用 fp32/fp64 matmul（不走 XMX）作代表：
   fp32 ≈ 22 TFLOPS、fp64 ≈ 17 TFLOPS（见 ``gemm`` suite）。

本机实测结论（2026-09-21，torch 2.14.0+xpu / triton-xpu 3.8.0 / Max 1100）：

* **可用（vector + matmul）**：fp64 / fp32 / fp16 / bf16 / int8。
* **仅 vector 可用**：int16 / int32 / int64 / uint8。
* **仅权重独占**：int4（``_weight_int4pack_mm`` / W4A16）。
* **FP8（e4m3 / e5m2）能跑但很慢**：只能经 ``torch._scaled_mm`` /
  ``torch.matmul`` 走 oneDNN 软件路径 —— Ponte Vecchio **没有原生 FP8 XMX**，
  实测只有 BF16 峰值的一半上下（见 ``sweep_fp8``）。
* **块缩放 MXFP8 / MXFP4 / NVFP4 也都能跑**（同样经 ``torch._scaled_mm``，
  a/b 低精度 + 每 block 一个 scale），但全部无原生 XMX、吞吐远低于 BF16。
  MXFP6（``float6_*``）与 TF32 在本 torch 构建里**没有张量类型**。
* **uint8 不是真正的累加器**：``torch.matmul(uint8, uint8)`` 返回 uint8，
  200×200×4=160000 → 255（直接饱和），且 uint8 没有 ``_int_mm`` 类算子。
* **不可用**：
  - FP8 ``fnuz`` 系列（e4m3fnuz / e5m2fnuz / e8m0fnu 作数据时）—— oneDNN 直接拒绝；
  - 窄整数 INT4/INT2/INT1 的 **逐元素算术**（``add_xpu`` 未实现）；
  - MXFP4 / NVFP4 / MXFP6 / MXFP8 / TF32 —— 本 torch 构建里连张量 dtype 都没有。
* ``torch.matmul`` **不支持 int16 / int32 / int64**（oneDNN 报 "not supported"）。
* XMX 只在 **fp16 / bf16 / int8** 上生效；fp32 / fp64 走 ALU。
* ⚠ **不要对 int4 / int2 / int1 用 ``.to()`` 转型** —— XPU 的 ``cast_and_store``
  会触发 device-side assert 把进程直接打挂（本 suite 用 ``torch.empty`` 规避）。
"""

from __future__ import annotations

import torch

from .common import (
    DEVICE_TYPE,
    BenchResult,
    ResultStore,
    alu_tflops,
    gbps_from,
    gops_from,
    numel_for_bytes,
    tflops_from,
)
from .gemm import REAL_SHAPES
from .quant import _make_int4_codes, _make_scale_zeros, _pack_int4

SUITE = "precision"

# ---------------------------------------------------------------------------
# 候选格式定义
# ---------------------------------------------------------------------------
# (label, torch dtype, kind, 厂商声称的 vector 支持, 厂商声称的 matrix 支持)
# kind: float | int | float8 | packed(窄整数/FP4 仅存储)
CANDIDATES: list[tuple[str, torch.dtype | None, str, bool, bool]] = [
    ("fp64",            torch.float64,          "float",  True,  True),
    ("fp32",            torch.float32,          "float",  True,  True),
    ("fp16",            torch.float16,          "float",  True,  True),
    ("bf16",            torch.bfloat16,         "float",  True,  True),
    ("int8",            torch.int8,             "int",    True,  True),
    ("int16",           torch.int16,            "int",    True,  False),
    ("int32",           torch.int32,            "int",    True,  False),
    ("int64",           torch.int64,            "int",    False, False),
    ("uint8",           torch.uint8,            "int",    False, False),
    ("int4",            getattr(torch, "int4", None),  "packed", True,  True),
    ("int2",            getattr(torch, "int2", None),  "packed", False, True),
    ("int1",            getattr(torch, "int1", None),  "packed", False, True),
    ("fp8_e4m3fn",      getattr(torch, "float8_e4m3fn", None),    "float8", False, False),
    ("fp8_e5m2",        getattr(torch, "float8_e5m2", None),      "float8", False, False),
    ("fp8_e4m3fnuz",    getattr(torch, "float8_e4m3fnuz", None),  "float8", False, False),
    ("fp8_e5m2fnuz",    getattr(torch, "float8_e5m2fnuz", None),  "float8", False, False),
    ("fp8_e8m0fnu",     getattr(torch, "float8_e8m0fnu", None),   "float8", False, False),
    ("fp4_e2m1fn_x2",   getattr(torch, "float4_e2m1fn_x2", None), "packed", False, False),
]

# 块缩放格式：没有独立的“可算术”张量类型，必须带 scale 张量走 ``torch._scaled_mm``。
# (label, a/b dtype 短标签, scale dtype 短标签, block_k, 厂商声称 vector, 厂商声称 matrix)
BLOCK_FORMATS: list[tuple[str, str | None, str | None, int, bool, bool]] = [
    ("mxfp8", "fp8_e4m3fn",    "fp8_e8m0fnu", 32, False, True),
    ("mxfp4", "fp4_e2m1fn_x2", "fp8_e8m0fnu", 32, False, True),
    ("nvfp4", "fp4_e2m1fn_x2", "fp8_e4m3fn",  16, False, True),
    ("mxfp6", None,            None,          32, False, True),
    ("tf32",  None,            None,           0, False, True),
]

# vector 带宽测试的 dtype（只取能真正做算术的）
VECTOR_DTYPES = ["fp64", "fp32", "fp16", "bf16", "int8", "int16", "int32", "uint8"]

FLOAT_VEC_OPS = [("copy", 2), ("add", 3), ("mul", 3), ("triad", 3),
                 ("exp", 2), ("tanh", 2), ("sqrt", 2)]
INT_VEC_OPS = [("copy", 2), ("add", 3), ("mul", 3), ("sub", 3)]

VECTOR_SIZE_MIB = 512
MATMUL_SIZES = [4096, 8192]
LARGE_MATMUL_SIZES = [16384]
ACC_SIZE = 1024

# matmul 覆盖的 dtype（浮点 + int8 稠密）
MATMUL_FLOAT = ["fp64", "fp32", "fp16", "bf16"]


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


def _mk(dt: torch.dtype, shape: tuple[int, ...], kind: str) -> torch.Tensor:
    """按 kind 构造合适范围的输入张量。

    .. warning::
       **不要对 int4 / int2 / int1 使用 ``.to()`` 转换** —— XPU 上
       ``cast_and_store``（向下转型）会触发 device-side assert 直接把进程打挂。
       这些窄整数只能用 ``torch.empty`` 申请，后续算子会以干净的
       Python 异常（``NotImplementedError: add_xpu ...``）报不支持。
    """
    if kind in ("packed", "absent"):
        # 仅申请存储，不做任何转型；算子不支持时会抛干净的异常
        return torch.empty(*shape, device=DEVICE_TYPE, dtype=dt)
    if kind == "float":
        return torch.randn(*shape, device=DEVICE_TYPE, dtype=torch.float32).to(dt)
    if kind == "float8":
        return torch.randn(*shape, device=DEVICE_TYPE, dtype=torch.float32).to(dt)
    if kind == "int":
        return torch.randint(0, 8, shape, device=DEVICE_TYPE).to(dt)
    raise ValueError(f"unknown kind {kind}")


def _dtype_by_label(label: str) -> Optional[torch.dtype]:
    """短标签 → torch dtype。

    .. warning::
       torch 把 FP8 命名为 ``float8_*``（**不是** ``fp8_*``），FP4 是
       ``float4_*``。直接 ``getattr(torch, "fp8_e4m3fn")`` 会**静默返回 None**，
       导致整个 FP8 家族被误判为“不支持”（已修的 bug）。
    """
    if label.startswith("fp8_"):
        return getattr(torch, "float8_" + label[4:], None)
    if label.startswith("fp4_"):
        return getattr(torch, "float4_" + label[4:], None)
    return getattr(torch, label, None)


def _probe(fn) -> tuple[str, str]:
    """执行 fn，返回 (status, detail)。status ∈ {OK, FAIL, ABSENT}。"""
    try:
        out = fn()
        if isinstance(out, torch.Tensor):
            torch.xpu.synchronize()
        return "OK", ""
    except Exception as exc:  # noqa: BLE001 - 需要把任意后端错误记入报告
        msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        return "FAIL", msg[:160]


def _probe_uint8_saturates() -> bool:
    """``torch.matmul(uint8, uint8)`` 的输出也是 uint8，超过 255 就饱和。

    真正的低精度累加要用 ``torch._int_mm``（int8×int8→int32），而 uint8
    **没有**对应算子，因此 uint8 不能当真正的 GEMM 累加器用。
    """
    a = torch.full((1, 4), 200, dtype=torch.uint8, device=DEVICE_TYPE)
    b = torch.full((4, 1), 200, dtype=torch.uint8, device=DEVICE_TYPE)
    out = torch.matmul(a, b)
    torch.xpu.synchronize()
    return int(out.item()) != 160000


def _block_inputs(adt: torch.dtype, sdt: torch.dtype, block: int,
                  m: int, n: int, k: int) -> tuple:
    """构造块缩放的 ``_scaled_mm`` 输入（a 行主序、b 行主序由调用方 ``.t()``）。

    * FP8 类：``a``/``b`` 为 ``[rows, cols]``，scale 为 ``[rows, cols//block]``
      （列主操作数取 ``[k//block, n]``）。
    * FP4 类：``a``/``b`` 是 **2 值/字节** 的 ``float4_e2m1fn_x2``，
      逻辑宽度仍为 ``cols``，物理列数只有 ``cols//2``。
    """
    packed = getattr(torch, "float4_e2m1fn_x2", None)

    def _mk_w(rows: int, cols: int) -> torch.Tensor:
        if adt is packed and packed is not None:
            raw = torch.randint(0, 256, (rows, cols // 2),
                                dtype=torch.uint8, device=DEVICE_TYPE)
            return raw.view(adt)
        return (torch.rand(rows, cols, device=DEVICE_TYPE) * 2.0 - 1.0).to(adt)

    a = _mk_w(m, k)
    b = _mk_w(n, k)
    sa = torch.ones(m, k // block, device=DEVICE_TYPE, dtype=sdt)
    sb = torch.ones(k // block, n, device=DEVICE_TYPE, dtype=sdt)
    return a, b, sa, sb


def _scaled_mm_block(a: torch.Tensor, b: torch.Tensor,
                     sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    return torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)


def _probe_fp8_matmul(dt: torch.dtype) -> tuple[str, str]:
    m = 256

    def _scaled():
        a = _mk(dt, (m, m), "float8")
        b = _mk(dt, (m, m), "float8")
        return torch._scaled_mm(a, b.t(), torch.ones(1, device=DEVICE_TYPE),
                                torch.ones(1, device=DEVICE_TYPE),
                                out_dtype=torch.bfloat16)

    stat, msg = _probe(_scaled)
    if stat == "OK":
        return "OK", ("经 `_scaled_mm`（fp32 累加 → bf16）；"
                       "⚠ **无原生 FP8 XMX**，吞吐远低于 BF16")

    def _plain():
        return torch.matmul(_mk(dt, (m, m), "float8"), _mk(dt, (m, m), "float8"))

    stat2, msg2 = _probe(_plain)
    if stat2 == "OK":
        return "OK", "经 `torch.matmul`（FP8 累加，精度较低）"
    return "FAIL", f"`_scaled_mm` → {msg}；`torch.matmul` → {msg2}"


# ---------------------------------------------------------------------------
# 1) 能力矩阵：每种格式真跑一次 vector / matmul
# ---------------------------------------------------------------------------
def probe_capability(store: ResultStore, ns) -> None:
    """对厂商列出的每种数值格式，实测 vector 与 matmul 是否可用。"""
    m, n, k = 256, 256, 256

    for label, dt, kind, hw_vec, hw_mat in CANDIDATES:
        params = {
            "format": label,
            "hw_vector": hw_vec,
            "hw_matrix": hw_mat,
            "torch_dtype": str(dt).replace("torch.", "") if dt is not None else "-",
        }
        if dt is None:
            store.add(BenchResult(
                suite=SUITE, name="capability", params=params,
                metrics={"vector": "ABSENT", "matmul": "ABSENT"},
                note="本 torch 构建里不存在该 dtype（MXFP/NVFP/TF32 无对应张量类型）",
            ))
            continue

        # --- vector 探针 ---
        def _vec(dt=dt, kind=kind, label=label):
            a = _mk(dt, (4096,), kind)
            b = _mk(dt, (4096,), kind)
            # 注意：对 fp8 / 窄整数**不做**精度转换（会崩），直接跑原生算术，
            # 不支持时后端会抛干净的 NotImplementedError。
            return (a + b) * 2

        vstat, vmsg = _probe(_vec)

        # --- matmul 探针 ---
        if kind == "packed":
            # 窄整数没有稠密 matmul；INT4 只有权重独占的 _weight_int4pack_mm
            if label == "int4":
                mstat = "SPECIAL"
                mmsg = "torch.matmul 不支持；仅 _weight_int4pack_mm 权重独占（见下表）"
            else:
                mstat = "FAIL"
                mmsg = ("torch.matmul 不支持该紧凑类型（无逐元素 / 稠密 matmul 算子）；"
                        "需配 scale 走 `_scaled_mm`（见 mxfp4 / nvfp4）")
        elif kind == "float8":
            mstat, mmsg = _probe_fp8_matmul(dt)
        else:
            def _mm(dt=dt, kind=kind, m=m, n=n, k=k):
                a = _mk(dt, (m, k), kind)
                b = _mk(dt, (k, n), kind)
                return torch.matmul(a, b)

            mstat, mmsg = _probe(_mm)

        if label == "uint8" and mstat == "OK" and _probe_uint8_saturates():
            mstat = "SAT"
            mmsg = ("输出仍为 uint8：200×200×4=160000 → 255（饱和），"
                    "**不是真正的累加器**；uint8 无 ``_int_mm`` 类算子，"
                    "实际使用必须自行分块/提精度")

        note_bits = []
        if vstat == "FAIL":
            note_bits.append(f"vector: {vmsg}")
        if mstat == "FAIL":
            note_bits.append(f"matmul: {mmsg}")
        elif mstat in ("SPECIAL", "SAT"):
            note_bits.append(mmsg)
        elif mmsg and mstat == "OK":
            note_bits.append(mmsg)
        store.add(BenchResult(
            suite=SUITE, name="capability", params=params,
            metrics={"vector": vstat, "matmul": mstat},
            note="；".join(note_bits),
        ))

    _probe_block_formats(store)

    # --- 专用量化算子的探针（INT4/INT8 的 XMX 路径不走 torch.matmul）---
    _probe_quantized_ops(store)


def _probe_block_formats(store: ResultStore) -> None:
    """MXFP8 / MXFP4 / NVFP4 / MXFP6 / TF32：块缩放格式的可用性。

    这类格式没有“能单独做算术”的张量类型，只有 ``torch._scaled_mm``
    （a/b 低精度 + 每 block 一个 scale）这一条路径。
    """
    for label, adt_name, sdt_name, block, hw_vec, hw_mat in BLOCK_FORMATS:
        adt = _dtype_by_label(adt_name) if adt_name else None
        sdt = _dtype_by_label(sdt_name) if sdt_name else None
        params = {
            "format": label,
            "hw_vector": hw_vec,
            "hw_matrix": hw_mat,
            "torch_dtype": (f"{adt_name} + {sdt_name} scale(1x{block})"
                            if adt is not None else "-"),
        }
        if adt is None or sdt is None:
            why = ("torch 无 ``float6_*`` 张量类型（MXFP6 无法构造）"
                   if label == "mxfp6" else
                   "torch 无 TF32 张量类型（TF32 只是 cuBLAS/oneDNN 的 fp32 计算模式）")
            store.add(BenchResult(
                suite=SUITE, name="capability", params=params,
                metrics={"vector": "ABSENT", "matmul": "ABSENT"},
                note=f"本 torch 构建里不存在该 dtype：{why}",
            ))
            continue

        def _run(adt=adt, sdt=sdt, block=block):
            a, b, sa, sb = _block_inputs(adt, sdt, block, 128, 128, 128)
            return _scaled_mm_block(a, b, sa, sb)

        st, msg = _probe(_run)
        note = (msg if st == "FAIL"
                else (f"块缩放经 `_scaled_mm`（1x{block}，scale={sdt_name}）；"
                      "⚠ 无原生 XMX，吞吐远低于 BF16（见下表）"))
        store.add(BenchResult(
            suite=SUITE, name="capability", params=params,
            metrics={"vector": "FAIL", "matmul": st},
            note=f"vector: 无逐元素算子；matmul: {note}",
        ))


def _probe_quantized_ops(store: ResultStore) -> None:
    """``_int_mm`` / ``_weight_int8pack_mm`` / ``_weight_int4pack_mm`` / ``_scaled_mm``。"""
    n, k, gs = 512, 512, 128

    def _mm_int8():
        a = torch.randint(-8, 8, (256, k), dtype=torch.int8, device=DEVICE_TYPE)
        b = torch.randint(-8, 8, (k, 256), dtype=torch.int8, device=DEVICE_TYPE)
        return torch._int_mm(a, b)

    def _w8a16():
        x = torch.randn(128, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        w = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEVICE_TYPE)
        s = torch.rand(n, dtype=torch.bfloat16, device=DEVICE_TYPE) * 0.05 + 0.01
        return torch.ops.aten._weight_int8pack_mm(x, w, s)

    def _w4a16():
        x = torch.randn(128, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        packed = _pack_int4(_make_int4_codes(n, k))
        cz = _make_scale_zeros(k, n, gs)
        return torch.ops.aten._weight_int4pack_mm(x, packed, gs, cz)

    def _fp8_scaled():
        dt = getattr(torch, "float8_e4m3fn")
        a = torch.randn(256, k, device=DEVICE_TYPE, dtype=torch.float32).to(dt)
        b = torch.randn(k, 256, device=DEVICE_TYPE, dtype=torch.float32).to(dt)
        sa = torch.ones(1, device=DEVICE_TYPE, dtype=torch.float32)
        sb = torch.ones(1, device=DEVICE_TYPE, dtype=torch.float32)
        return torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)

    for label, fn, note in [
        ("int8_dense=_int_mm", _mm_int8, "int8×int8→int32，XMX（int32 累加）"),
        ("int8_weight=_weight_int8pack_mm", _w8a16, "W8A16 权重独占，per-channel scale"),
        ("int4_weight=_weight_int4pack_mm", _w4a16, "W4A16 权重独占，dequant=scale*(q-8)+zero"),
        ("fp8_scaled=_scaled_mm", _fp8_scaled, "FP8 稠密（e4m3 × e4m3）"),
    ]:
        stat, msg = _probe(fn)
        store.add(BenchResult(
            suite=SUITE, name="quantized_op",
            params={"op": label},
            metrics={"status": stat},
            note=note if stat == "OK" else f"{note} — {stat}: {msg}",
        ))


# ---------------------------------------------------------------------------
# 2) vector 路径：elementwise 有效带宽 / 元素速率
# ---------------------------------------------------------------------------
def sweep_vector(store: ResultStore, ns) -> None:
    """各 dtype 的 elementwise 有效带宽与元素速率（vector 指令路径）。"""
    kw = _kw(ns)
    size_mib = int(getattr(ns, "size_mb", VECTOR_SIZE_MIB))
    want = [d for d in getattr(ns, "dtypes", VECTOR_DTYPES)]
    by_label = {c[0]: c for c in CANDIDATES}

    for label in want:
        cand = by_label.get(label)
        if cand is None or cand[1] is None:
            continue
        _, dt, kind, _, _ = cand
        if kind in ("packed", "float8", "absent"):
            continue
        itemsize = torch.tensor([], dtype=dt).element_size()
        nelem = numel_for_bytes(size_mib << 20, dt)
        try:
            a = _mk(dt, (nelem,), kind)
            b = _mk(dt, (nelem,), kind)
            out = torch.empty(nelem, device=DEVICE_TYPE, dtype=dt)
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "vector", {"dtype": label}, exc)
            _scratch()
            continue

        op_list = [(nm, io) for nm, io in
                   (FLOAT_VEC_OPS if kind == "float" else INT_VEC_OPS)]
        for opname, io in op_list:
            fn = _make_vec_op(opname, a, b, out, kind)
            if fn is None:
                continue
            params = {"dtype": label, "op": opname, "elems": nelem,
                      "size_mib": size_mib, "io_factor": io}
            store.measure(
                SUITE, "vector", params, fn,
                compute_metrics=_bw_metrics(io, nelem, itemsize),
                **kw,
            )
        # 浮点再补一个 exp —— 用于证明「超越函数也是纯带宽受限」
        del a, b, out
        _scratch()


def _make_vec_op(opname: str, a, b, out, kind: str):
    if opname == "copy":
        return lambda a=a, out=out: out.copy_(a)
    if opname == "add":
        return lambda a=a, b=b, out=out: torch.add(a, b, out=out)
    if opname == "sub":
        return lambda a=a, b=b, out=out: torch.sub(a, b, out=out)
    if opname == "mul":
        return lambda a=a, b=b, out=out: torch.mul(a, b, out=out)
    if opname == "triad":
        if kind != "float":
            return None
        return lambda a=a, b=b, out=out: torch.add(a, b, alpha=1.5, out=out)
    if kind != "float":
        return None
    if opname == "exp":
        return lambda a=a, out=out: torch.exp(a, out=out)
    if opname == "tanh":
        return lambda a=a, out=out: torch.tanh(a, out=out)
    if opname == "sqrt":
        return lambda a=a, out=out: torch.sqrt(a, out=out)
    return None


def _bw_metrics(io_factor: int, nelem: int, itemsize: int):
    def _fn(stats):
        nbytes = io_factor * nelem * itemsize
        return {
            "gbps": gbps_from(nbytes, stats.median_ms),
            "gops": gops_from(nelem, stats.median_ms),
        }

    return _fn


# ---------------------------------------------------------------------------
# 3) 矩阵路径：各精度 matmul 吞吐（XMX vs ALU）
# ---------------------------------------------------------------------------
def sweep_matmul(store: ResultStore, ns) -> None:
    """各浮点精度 + int8 的稠密 matmul 吞吐。"""
    kw = _kw(ns)
    sizes = [int(s) for s in getattr(ns, "gemm_sizes", MATMUL_SIZES)]
    sizes = sorted({s for s in sizes if s >= 1024} or set(MATMUL_SIZES))
    if getattr(ns, "large", False):
        sizes = sorted(set(sizes) | set(LARGE_MATMUL_SIZES))

    from .common import DTYPES  # noqa: PLC0415 - 局部导入避免循环

    for label in MATMUL_FLOAT:
        dt = DTYPES.get(label)
        if dt is None:
            continue
        for n in sizes:
            params = {"dtype": label, "m": n, "n": n, "k": n,
                      "shape": f"{n}x{n}x{n}", "path": "ALU" if label in ("fp32", "fp64") else "XMX"}
            a = b = None
            try:
                a = torch.randn(n, n, device=DEVICE_TYPE, dtype=dt)
                b = torch.randn(n, n, device=DEVICE_TYPE, dtype=dt)
                torch.matmul(a, b)
                torch.xpu.synchronize()
            except Exception as exc:  # noqa: BLE001
                store.skip(SUITE, "matmul", params, f"{label} 不被支持: {type(exc).__name__}")
                del a, b
                _scratch()
                continue
            store.measure(
                SUITE, "matmul", params,
                lambda a=a, b=b: torch.matmul(a, b),
                compute_metrics=_tflops_metrics(2.0 * n ** 3),
                **kw,
            )
            del a, b
            _scratch()

    # --- INT8 稠密（XMX）---
    int_mm = getattr(torch, "_int_mm", None)
    for n in [s for s in sizes if s % 8 == 0]:
        params = {"dtype": "int8", "m": n, "n": n, "k": n,
                  "shape": f"{n}x{n}x{n}", "path": "XMX"}
        a = b = None
        try:
            a = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            b = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=DEVICE_TYPE)
            int_mm(a, b)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "matmul", params, f"INT8 不支持: {type(exc).__name__}")
            del a, b
            _scratch()
            continue
        store.measure(
            SUITE, "matmul", params,
            lambda a=a, b=b: int_mm(a, b),
            compute_metrics=_tflops_metrics(2.0 * n ** 3),
            note="int8×int8→int32",
            **kw,
        )
        del a, b
        _scratch()

    # --- INT8 稠密真实场景 shape（LLM prefill/decode）---
    for label, m, n, k in REAL_SHAPES:
        if n % 8 or k % 8:
            continue
        params = {"dtype": "int8", "m": m, "n": n, "k": k,
                  "shape": f"{m}x{n}x{k}", "path": "XMX", "label": label}
        a = b = None
        try:
            a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=DEVICE_TYPE)
            b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE_TYPE)
            int_mm(a, b)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            store.skip(SUITE, "matmul", params, f"不支持: {type(exc).__name__}")
            del a, b
            _scratch()
            continue
        store.measure(
            SUITE, "matmul", params,
            lambda a=a, b=b: int_mm(a, b),
            compute_metrics=_tflops_metrics(2.0 * m * n * k),
            **kw,
        )
        del a, b
        _scratch()


def _tflops_metrics(flops: float):
    def _fn(stats):
        return {"tflops": tflops_from(flops, stats.median_ms)}

    return _fn


# ---------------------------------------------------------------------------
# 3b) FP8 matmul（软件路径）
# ---------------------------------------------------------------------------
FP8_DTYPES = ["fp8_e4m3fn", "fp8_e5m2"]


def sweep_fp8(store: ResultStore, ns) -> None:
    """FP8 稠密 matmul 吞吐（``torch._scaled_mm``，e4m3 × e4m3 → bf16）。"""
    kw = _kw(ns)
    sizes = [4096, 8192]
    if getattr(ns, "large", False):
        sizes.append(16384)

    for label in FP8_DTYPES:
        dt = _dtype_by_label(label)
        if dt is None:
            continue
        for n in sizes:
            params = {"dtype": label, "m": n, "n": n, "k": n,
                      "shape": f"{n}x{n}x{n}", "path": "SW(无FP8 XMX)"}
            a = b = sa = sb = None
            try:
                a = _mk(dt, (n, n), "float8")
                b = _mk(dt, (n, n), "float8")
                sa = torch.ones(1, device=DEVICE_TYPE, dtype=torch.float32)
                sb = torch.ones(1, device=DEVICE_TYPE, dtype=torch.float32)
                torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)
                torch.xpu.synchronize()
            except Exception as exc:  # noqa: BLE001
                store.skip(SUITE, "fp8_matmul", params,
                           f"{label} 不支持: {type(exc).__name__}: {str(exc)[:80]}")
                del a, b, sa, sb
                _scratch()
                continue
            store.measure(
                SUITE, "fp8_matmul", params,
                lambda a=a, b=b, sa=sa, sb=sb: torch._scaled_mm(
                    a, b.t(), sa, sb, out_dtype=torch.bfloat16),
                compute_metrics=_tflops_metrics(2.0 * n ** 3),
                note="fp32 累加 → bf16 输出",
                **kw,
            )
            del a, b, sa, sb
            _scratch()


# ---------------------------------------------------------------------------
# 3c) 块缩放低精度 matmul：MXFP8 / MXFP4 / NVFP4（全走 torch._scaled_mm）
# ---------------------------------------------------------------------------
BLOCK_SIZES = [4096, 8192]
BLOCK_LARGE_SIZES = [16384]


def sweep_block_scaled(store: ResultStore, ns) -> None:
    """MXFP8 / MXFP4 / NVFP4 块缩放 matmul 吞吐（``torch._scaled_mm``）。"""
    kw = _kw(ns)
    sizes = list(BLOCK_SIZES)
    if getattr(ns, "large", False):
        sizes += BLOCK_LARGE_SIZES

    for label, adt_name, sdt_name, block, _, _ in BLOCK_FORMATS:
        adt = _dtype_by_label(adt_name) if adt_name else None
        sdt = _dtype_by_label(sdt_name) if sdt_name else None
        if adt is None or sdt is None:
            continue
        for n in sizes:
            params = {"fmt": label, "dtype": adt_name, "scale": sdt_name,
                      "block_k": block, "m": n, "n": n, "k": n,
                      "shape": f"{n}x{n}x{n}", "path": "SW(无原生XMX)"}
            a = b = sa = sb = None
            try:
                a, b, sa, sb = _block_inputs(adt, sdt, block, n, n, n)
                _scaled_mm_block(a, b, sa, sb)
                torch.xpu.synchronize()
            except Exception as exc:  # noqa: BLE001
                store.skip(SUITE, "block_matmul", params,
                           f"{label} 不支持: {type(exc).__name__}: {str(exc)[:80]}")
                del a, b, sa, sb
                _scratch()
                continue
            store.measure(
                SUITE, "block_matmul", params,
                lambda a=a, b=b, sa=sa, sb=sb: _scaled_mm_block(a, b, sa, sb),
                compute_metrics=_tflops_metrics(2.0 * n ** 3),
                note="fp32 累加 → bf16 输出",
                **kw,
            )
            del a, b, sa, sb
            _scratch()


# ---------------------------------------------------------------------------
# 4) 权重独占低精度：W8A16 / W4A16（decode 场景的显存/带宽收益）
# ---------------------------------------------------------------------------
DECODE_M = [1, 32, 512]
WEIGHT_NK = (4096, 4096)
DEFAULT_GROUP_SIZE = 128


def sweep_weight_only(store: ResultStore, ns) -> None:
    """同一 shape 下 BF16 / W8A16 / W4A16 的对照（低精度权重独占）。"""
    kw = _kw(ns)
    gs = DEFAULT_GROUP_SIZE
    n, k = WEIGHT_NK

    for m in DECODE_M:
        shape = f"{m}x{n}x{k}"
        xb = wb = None
        try:
            xb = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
            wb = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE_TYPE)
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "weight_only", {"shape": shape}, exc)
            _scratch()
            continue
        store.measure(
            SUITE, "weight_only", {"kind": "bf16", "m": m, "n": n, "k": k, "shape": shape},
            lambda xb=xb, wb=wb: xb @ wb.t(),
            compute_metrics=_weight_metrics(n * k * 2, m, n, k),
            **kw,
        )
        del wb
        _scratch()

        try:
            w8 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEVICE_TYPE)
            s8 = torch.rand(n, dtype=torch.bfloat16, device=DEVICE_TYPE) * 0.05 + 0.01
            store.measure(
                SUITE, "weight_only", {"kind": "w8a16", "m": m, "n": n, "k": k, "shape": shape},
                lambda xb=xb, w8=w8, s8=s8: torch.ops.aten._weight_int8pack_mm(xb, w8, s8),
                compute_metrics=_weight_metrics(n * k, m, n, k),
                **kw,
            )
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "weight_only", {"kind": "w8a16", "shape": shape}, exc)
        finally:
            del w8, s8
            _scratch()

        try:
            packed = _pack_int4(_make_int4_codes(n, k))
            cz = _make_scale_zeros(k, n, gs)
            wbytes = n * k // 2 + (k // gs) * n * 2 * 2
            store.measure(
                SUITE, "weight_only", {"kind": "w4a16", "m": m, "n": n, "k": k, "shape": shape},
                lambda xb=xb, packed=packed, cz=cz: torch.ops.aten._weight_int4pack_mm(
                    xb, packed, gs, cz),
                compute_metrics=_weight_metrics(wbytes, m, n, k),
                **kw,
            )
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "weight_only", {"kind": "w4a16", "shape": shape}, exc)
        finally:
            del packed, cz
            _scratch()

        del xb
        _scratch()


def _weight_metrics(weight_bytes: int, m: int, n: int, k: int):
    def _fn(stats):
        return {
            "tflops": tflops_from(2.0 * m * n * k, stats.median_ms),
            "weight_gb_s": gbps_from(weight_bytes, stats.median_ms),
            "weight_mib": round(weight_bytes / (1 << 20), 2),
        }

    return _fn


# ---------------------------------------------------------------------------
# 5) 数值自检：各精度相对 fp64 参考的误差
# ---------------------------------------------------------------------------
def accuracy_check(store: ResultStore) -> None:
    """各精度 matmul 相对 fp64 参考的相对误差（表征该精度的「有效位」）。"""
    m = n = k = ACC_SIZE
    ref = None
    for label in ("fp64", "fp32", "fp16", "bf16"):
        dt = {"fp64": torch.float64, "fp32": torch.float32,
              "fp16": torch.float16, "bf16": torch.bfloat16}[label]
        try:
            a = torch.randn(m, k, device=DEVICE_TYPE, dtype=dt)
            b = torch.randn(k, n, device=DEVICE_TYPE, dtype=dt)
            got = torch.matmul(a, b)
            torch.xpu.synchronize()
            if ref is None:
                ref = torch.matmul(a.double(), b.double())
            else:
                ref = torch.matmul(a.double(), b.double())
            g = got.double()
            err = (g - ref).abs()
            scale = float(ref.abs().mean()) or 1.0
            store.add(BenchResult(
                suite=SUITE, name="accuracy", params={"dtype": label, "shape": f"{m}x{n}x{k}"},
                metrics={
                    "max_abs_err": float(err.max()),
                    "rel_err": float(err.mean()) / scale,
                },
                note="相对 fp64 参考；rel_err 与输出精度位宽一致，不随 K 增长 ⇒ FP32 累加",
            ))
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "accuracy", {"dtype": label}, exc)
        finally:
            _scratch()


# ---------------------------------------------------------------------------
def _summarize(store: ResultStore) -> None:
    rows = store.suite(SUITE)

    # --- 能力矩阵摘要 ---
    cap = [r for r in rows if r.name == "capability"]
    ok_vec = [r.params["format"] for r in cap if r.metrics.get("vector") == "OK"]
    ok_mat = [r.params["format"] for r in cap if r.metrics.get("matmul") == "OK"]
    absent = [r.params["format"] for r in cap if r.metrics.get("vector") == "ABSENT"]
    special = [r.params["format"] for r in cap if r.metrics.get("matmul") == "SPECIAL"]
    sat = [r.params["format"] for r in cap if r.metrics.get("matmul") == "SAT"]
    store.add_note(
        f"**vector 路径可用格式**（{len(ok_vec)} 种）：{', '.join(ok_vec) or '无'}"
    )
    store.add_note(
        f"**matmul 路径可用格式**（{len(ok_mat)} 种）：{', '.join(ok_mat) or '无'}"
    )
    if special:
        store.add_note(
            f"⚠ **仅能跑权重独占（非稠密 matmul）**：{', '.join(special)}"
        )
    if sat:
        store.add_note(
            f"⚠ **能跑但不是真正累加器**（输出与输入同为 8bit，直接饱和）："
            f"{', '.join(sat)}"
        )
    if absent:
        store.add_note(
            f"**本 torch 构建中不存在对应张量类型**（无法测试）：{', '.join(absent)}"
        )

    qops = [r for r in rows if r.name == "quantized_op"]
    q_ok = [r.params["op"] for r in qops if r.metrics.get("status") == "OK"]
    q_bad = [r.params["op"] for r in qops if r.metrics.get("status") != "OK"]
    if q_ok:
        store.add_note(f"**专用量化 matmul 算子可用**：{', '.join(q_ok)}")
    if q_bad:
        store.add_note(f"⚠ **专用量化 matmul 算子不可用**：{', '.join(q_bad)}")

    # --- matmul 峰值 & XMX 倍数 ---
    best: dict[str, BenchResult] = {}
    for r in rows:
        if r.name != "matmul" or "tflops" not in r.metrics:
            continue
        dk = r.params.get("dtype")
        if dk not in best or r.metrics["tflops"] > best[dk].metrics["tflops"]:
            best[dk] = r
    for dk in ("fp32", "fp64", "fp16", "bf16", "int8"):
        if dk in best:
            r = best[dk]
            unit = "TOPS" if dk == "int8" else "TFLOPS"
            store.add_note(
                f"matmul 峰值 {dk}: {r.metrics['tflops']:.1f} {unit} @ {r.params.get('shape')}"
                f"（{r.params.get('path')} 路径）"
            )

    # --- 低精度浮点（FP8 / MXFP8 / MXFP4 / NVFP4）：全部走软件路径 ---
    bf16_peak = best["bf16"].metrics["tflops"] if best.get("bf16") else None
    for row_name, label_key in (("fp8_matmul", "dtype"), ("block_matmul", "fmt")):
        keys = sorted({r.params.get(label_key) for r in rows
                       if r.name == row_name and "tflops" in r.metrics})
        for key in keys:
            grp = [r for r in rows if r.name == row_name
                   and r.params.get(label_key) == key and "tflops" in r.metrics]
            if not grp:
                continue
            b = max(grp, key=lambda r: r.metrics["tflops"])
            note = (f"低精度浮点 {key} matmul 峰值 {b.metrics['tflops']:.1f} TFLOPS "
                    f"@ {b.params.get('shape')}")
            if bf16_peak:
                note += f"，为 BF16 峰值的 {b.metrics['tflops'] / bf16_peak:.2f}×"
            store.add_note(note)
    if any(r.name in ("fp8_matmul", "block_matmul") for r in rows):
        store.add_note(
            "⚠ FP8 / MXFP8 / MXFP4 / NVFP4 **全部没有原生 XMX 单元**"
            "（PVC 的 XMX 只支持 FP16 / BF16 / INT8），这些格式依赖 oneDNN "
            "软件/回退实现，吞吐显著低于 BF16，**不建议在本卡上当作加速手段**。"
        )

    base = best.get("fp32")
    if base:
        alu = alu_tflops()
        store.add_note(
            f"fp32 matmul 实测 {base.metrics['tflops']:.2f} TFLOPS，"
            f"ALU 理论 {alu:.2f} TFLOPS（达成率 "
            f"{100 * base.metrics['tflops'] / alu:.0f}%）→ fp32 走 ALU 而非 XMX"
        )
        for dk in ("fp16", "bf16", "int8"):
            if dk in best:
                ratio = best[dk].metrics["tflops"] / base.metrics["tflops"]
                store.add_note(
                    f"XMX 倍数 {dk}/fp32 = **{ratio:.2f}×**"
                    f"（{best[dk].metrics['tflops']:.1f} vs {base.metrics['tflops']:.1f}）"
                )

    # --- int8 稠密 vs W4A16 ---
    w4 = [r for r in rows if r.name == "weight_only" and r.params.get("kind") == "w4a16"]
    w4_best = max(w4, key=lambda r: r.metrics["tflops"], default=None)
    if w4_best and best.get("int8"):
        r = w4_best.metrics["tflops"] / best["int8"].metrics["tflops"]
        store.add_note(
            f"W4A16 峰值 {w4_best.metrics['tflops']:.1f} TOPS，仅为 INT8 稠密的 {r:.2f}×"
            f"（未达 XMX INT4 = INT8 2× 的理论值）；INT4 的收益在**权重显存/带宽**"
        )

    # --- vector 路径：确认纯带宽受限 ---
    vec = [r for r in rows if r.name == "vector" and "gbps" in r.metrics]
    if vec:
        peaks = {}
        for r in vec:
            dk = r.params["dtype"]
            if dk not in peaks or r.metrics["gbps"] > peaks[dk].metrics["gbps"]:
                peaks[dk] = r
        best_bw = max(peaks.values(), key=lambda r: r.metrics["gbps"])
        store.add_note(
            f"vector 峰值有效带宽 {best_bw.metrics['gbps']:.1f} GB/s "
            f"（{best_bw.params['dtype']} {best_bw.params['op']}，"
            f"io_factor={best_bw.params['io_factor']}）"
        )
        # 检查超越函数是否与 copy 同速（→ 证明是带宽受限而非 ALU 受限）
        for dk in ("fp32", "bf16"):
            copies = [r for r in vec if r.params["dtype"] == dk and r.params["op"] == "copy"]
            exps = [r for r in vec if r.params["dtype"] == dk and r.params["op"] == "exp"]
            if copies and exps:
                c, e = copies[0].metrics["gbps"], exps[0].metrics["gbps"]
                store.add_note(
                    f"{dk}: exp 带宽 {e:.1f} GB/s ≈ copy {c:.1f} GB/s（{100 * e / c:.0f}%）"
                    f"→ 该精度下 elementwise **纯带宽受限**，ALU 远未打满"
                )


def run(ns, store: ResultStore) -> None:
    """执行 precision suite。"""
    probe_capability(store, ns)
    sweep_vector(store, ns)
    sweep_matmul(store, ns)
    sweep_fp8(store, ns)
    sweep_block_scaled(store, ns)
    sweep_weight_only(store, ns)
    accuracy_check(store)
    _summarize(store)
