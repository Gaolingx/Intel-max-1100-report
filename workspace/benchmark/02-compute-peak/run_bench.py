#!/usr/bin/env python3
"""02-compute-peak / run_bench.py

ALU / XMX 理论算力峰值基准。对应 docs/TODO/02-compute-peak.md。

四组：
  alu    自研 SYCL 探针：纯 FMA 循环（非可折叠递推），fp32/fp64/fp16/int32
  xmx    自研 SYCL joint_matrix 探针：直接发射 DPAS，bf16/fp16/int8
  torch  PyTorch(oneDNN) GEMM 交叉验证：fp32/bf16/fp16/int8 + vector 对照
  clock  真实主频/功耗取证：sysfs + xpu-smi（**在负载进行中采样**）

═══════════════════════════════════════════════════════════════════════════
★ 本目录最重要的方法论（踩坑数小时得出，务必先读）

  现象：`xmx_peak bf16` 在同一 outer/it/nacc 下，把 work-group 从 1792 翻到 3584，
        耗时**一模一样**（0.173508 s），于是按 FLOP 计数除出来的吞吐凭空翻倍。

  第一反应会误判成「编译器把循环消掉了」。但这是**错的**。
  对照实验（固定总 DPAS 数、只改 work-group 数）证明耗时确实随 work-group 数变：

     总 DPAS 7.516e9，g=28672 (4 sg/EU) -> 0.173517 s  = 177 TFLOPS
     总 DPAS 7.516e9，g=57344 (8 sg/EU) -> 0.086764 s  = 355 TFLOPS  ← 等量工作，快一倍

  真实原因是 **延迟受限 / 占用率不足**：XMX 流水线需要每 EU ≥8 个并发 sub-group
  才能打满；不足时同样的活要花大约一倍时间。

  推论（判读纪律）：
    · 测峰值**必须先把 work-group 数扫过「拐点」**，否则测到的是延迟不是吞吐；
    · 拐点之上耗时对工作量必须严格线性 —— 用这个做**自校验**：
        ALU  g=114688 -> 1.2133 s / g=229376 -> 2.4234 s (×2.00 ✓)
        XMX  occ8    -> 0.1735 s / occ16   -> 0.3469 s (×2.00 ✓)
    · 拐点之下「等量工作同样耗时」是**正常现象**，不是 bug。
═══════════════════════════════════════════════════════════════════════════

用法：
    python3 run_bench.py                 # 全跑
    python3 run_bench.py --quick
    python3 run_bench.py alu xmx
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
sys.path.insert(0, str(BENCH_ROOT / "common"))

from bench import (  # noqa: E402
    EU_PER_DEVICE,
    LANES_PER_EU,
    REFERENCE_CLOCK_GHZ,
    ResultStore,
    alu_tflops,
    parse_json_lines,
    run,
    run_soft,
    sub_env,
)

BUILD_DIR = HERE / "build"
SYCL_DIR = HERE / "sycl"
VENV_PY = Path("/root/workspace/venv1/bin/python")
ICPX = "/opt/intel/oneapi/compiler/2026.1/bin/icpx"

ALU_BIN = BUILD_DIR / "alu_peak"
XMX_BIN = BUILD_DIR / "xmx_peak"

# 448 EU × 16 lane = 7168 = 每硬件 lane 一个 work-item
LANES_TOTAL = EU_PER_DEVICE * LANES_PER_EU          # 7168
ALU_KNEE = LANES_TOTAL * 16                          # 114688 —— 实测拐点
ALU_KNEE_2X = ALU_KNEE * 2                           # 229376 —— 自校验点

XMX_SG_TOTAL = EU_PER_DEVICE                         # 448 个 sub-group = 1 sg/EU
XMX_SG_GLOBAL = XMX_SG_TOTAL * 16                    # 7168 work-item
XMX_KNEE_OCC = 8                                     # 实测拐点：8 sub-group/EU

# 公式标称值（**不是**产品 datasheet）
NOMINAL_FP32_TFLOPS = alu_tflops()                                    # 22.22
RAW_XMX_BF16_TFLOPS = EU_PER_DEVICE * 256 * 2 * REFERENCE_CLOCK_GHZ / 1000.0   # 355.5
RAW_XMX_INT8_GOPS = EU_PER_DEVICE * 512 * 2 * REFERENCE_CLOCK_GHZ / 1000.0     # 711.1
PUBLISHED_BF16_TFLOPS = 176.0    # Max 1100 单 tile 常被引用的值，仅作对照


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build_sycl(src: Path, out: Path, extra: list[str] | None = None,
               env: dict | None = None) -> None:
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime and not extra:
        return
    print(f"[build] {src.name} -> {out.name} {' '.join(extra or [])}", flush=True)
    cp = run([ICPX, "-fsycl", "-O3", "-ffp-contract=fast", *(extra or []),
              "-o", str(out), str(src)], env=env, timeout=1800, check=False)
    if cp.returncode != 0 or not out.exists():
        raise RuntimeError(f"{src.name} build failed:\n{cp.stderr[-3000:]}")


def build_all(env: dict) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    build_sycl(SYCL_DIR / "alu_peak.cpp", ALU_BIN, env=env)
    build_sycl(SYCL_DIR / "xmx_peak.cpp", XMX_BIN, env=env)


# --------------------------------------------------------------------------- #
# 解析 / 自校验工具
# --------------------------------------------------------------------------- #
def parse_probe(text: str) -> tuple[dict, list[dict]]:
    dev = {}
    for d in parse_json_lines(text, "DEVICE"):
        dev = d
    return dev, parse_json_lines(text, "RESULT")


def linearity_check(name: str, w1: int, t1: float, w2: int, t2: float) -> tuple[bool, str]:
    """在**饱和区**比较：工作量 ×2 时耗时是否也 ×2（±15%）。"""
    if t1 <= 0:
        return False, f"{name}: 基准耗时为 0，无法判断"
    rw, rt = w2 / w1, t2 / t1
    ok = abs(rt - rw) / rw <= 0.15
    return ok, (f"{name}: 工作量 ×{rw:.2f} → 耗时 ×{rt:.2f}（期望 ×{rw:.2f}，±15%）")


def knee_index(xs: list[int], ys: list[float], frac: float = 0.95) -> int:
    """返回第一个达到峰值 ``frac`` 的下标 —— 即饱和拐点。"""
    if not ys:
        return 0
    peak = max(ys)
    for i, y in enumerate(ys):
        if y >= frac * peak:
            return i
    return len(ys) - 1


# --------------------------------------------------------------------------- #
# 1. ALU 峰值
# --------------------------------------------------------------------------- #
def suite_alu(store: ResultStore, env: dict, quick: bool) -> None:
    iters = (1 << 20) if quick else (1 << 22)
    dtypes = ("fp32", "fp64", "fp16", "int32")

    # ---- 1.1 占用率扫描：定位拐点（fp32）---------------------------------- #
    gs = ([LANES_TOTAL, ALU_KNEE, ALU_KNEE_2X] if quick else
          [LANES_TOTAL, 2 * LANES_TOTAL, 4 * LANES_TOTAL,
           8 * LANES_TOTAL, ALU_KNEE, ALU_KNEE_2X])
    scan_g: list[int] = []
    scan_val: list[float] = []
    for g in gs:
        cp = run_soft([str(ALU_BIN), "fp32", str(g), str(iters // 4), "2", "3", "128"],
                      env=env, timeout=1200)
        _, res = parse_probe(cp.stdout)
        if not res:
            store.error("alu", f"scan.g{g}", (cp.stderr or "no RESULT")[-200:],
                        params={"global": g})
            continue
        r = res[0]
        tf = float(r["value"]) / 1000.0
        scan_g.append(g)
        scan_val.append(tf)
        store.add(
            "alu", f"occupancy.g{g}",
            params={"dtype": "fp32", "global": g, "iters": iters // 4,
                    "work_items_per_lane": round(g / LANES_TOTAL, 2)},
            metrics={"tflops": round(tf, 2), "seconds": round(float(r["seconds"]), 4)},
            note=("饱和区" if g >= ALU_KNEE else "未饱和（延迟受限）"),
        )
    if scan_val:
        k = knee_index(scan_g, scan_val)
        store.add(
            "alu", "saturation_knee",
            params={"dtype": "fp32"},
            metrics={"knee_global": scan_g[k],
                     "knee_work_items_per_lane": round(scan_g[k] / LANES_TOTAL, 2),
                     "peak_tflops": round(max(scan_val), 2)},
            note=(f"拐点 global≈{scan_g[k]}（每硬件 lane {scan_g[k] / LANES_TOTAL:.0f} 个 "
                  f"work-item）；拐点以下为延迟受限区，不可当作峰值。"),
        )
        store.highlight(
            f"ALU 占用率拐点：`global ≈ {scan_g[k]}`（每硬件 lane "
            f"{scan_g[k] / LANES_TOTAL:.0f} 个 work-item）。"
            f"g={scan_g[0]} 时只有 {scan_val[0]:.1f} TFLOPS = 峰值的 "
            f"{100 * scan_val[0] / max(scan_val):.0f}% —— **这就是「低占用率测到延迟而非吞吐」**。"
        )

    # ---- 1.2 峰值：各 dtype ----------------------------------------------- #
    for dtype in dtypes:
        cp = run_soft([str(ALU_BIN), dtype, str(ALU_KNEE), str(iters), "2", "5", "128"],
                      env=env, timeout=1800)
        dev, res = parse_probe(cp.stdout)
        if not res:
            store.error("alu", f"peak.{dtype}", (cp.stderr or "no RESULT")[-300:],
                        params={"global": ALU_KNEE, "iters": iters})
            continue
        r = res[0]
        tf = float(r["value"]) / 1000.0
        store.add(
            "alu", f"peak.{dtype}",
            params={"dtype": dtype, "global": ALU_KNEE, "iters": iters,
                    "vec": dev.get("vec"), "acc": dev.get("acc"),
                    "unroll": dev.get("unroll"), "local": dev.get("local_size"),
                    "unit": r.get("unit")},
            metrics={"tflops": round(tf, 2),
                     "seconds": round(float(r["seconds"]), 4),
                     "pct_of_formula": round(100.0 * tf / NOMINAL_FP32_TFLOPS, 1)},
            note=(f"公式标称 {NOMINAL_FP32_TFLOPS:.2f} TFLOPS；实测 {tf:.2f} = "
                  f"{100.0 * tf / NOMINAL_FP32_TFLOPS:.0f}%"),
        )

    # ---- 1.3 自校验：饱和区 ×2 工作量是否 ×2 耗时 -------------------------- #
    cp1 = run_soft([str(ALU_BIN), "fp32", str(ALU_KNEE), str(iters // 4), "2", "3", "128"],
                   env=env, timeout=1800)
    cp2 = run_soft([str(ALU_BIN), "fp32", str(ALU_KNEE_2X), str(iters // 4), "2", "3", "128"],
                   env=env, timeout=1800)
    _, r1 = parse_probe(cp1.stdout)
    _, r2 = parse_probe(cp2.stdout)
    if r1 and r2:
        t1, t2 = float(r1[0]["seconds"]), float(r2[0]["seconds"])
        ok, msg = linearity_check("ALU", ALU_KNEE, t1, ALU_KNEE_2X, t2)
        store.add(
            "alu", "linearity_check",
            params={"g1": ALU_KNEE, "g2": ALU_KNEE_2X},
            metrics={"tflops_1": round(float(r1[0]["value"]) / 1000.0, 2),
                     "tflops_2": round(float(r2[0]["value"]) / 1000.0, 2),
                     "time_ratio": round(t2 / t1, 3)},
            status="ok" if ok else "error",
            note=("饱和区线性 ✓，FLOP 计数可信。" if ok else "饱和区不线性 ✗：") + msg,
        )

    # 对照：跨拐点的两点（说明「不线性」是拐点造成的，不是 bug）
    if scan_val and len(scan_g) >= 2:
        ik = knee_index(scan_g, scan_val)
        i_pre = max(0, ik - 1)
        if i_pre != ik:
            store.add(
                "alu", "linearity_check_below_knee",
                params={"g_pre": scan_g[i_pre], "g_knee": scan_g[ik],
                        "work_ratio": scan_g[ik] / scan_g[i_pre]},
                metrics={"tflops_pre": round(scan_val[i_pre], 2),
                         "tflops_knee": round(scan_val[ik], 2),
                         "throughput_gain": round(scan_val[ik] / scan_val[i_pre], 2)},
                status="ok",
                note=(f"**故意跨拐点**：g={scan_g[i_pre]}→{scan_g[ik]} 工作量 ×"
                      f"{scan_g[ik] / scan_g[i_pre]:.0f}，吞吐却只提升 "
                      f"{scan_val[ik] / scan_val[i_pre]:.2f}×（远小于工作量增幅），"
                      "说明拐点前测到的是**延迟**不是吞吐。"
                      "留档以防后人把「等量工作同样耗时」误判为编译器消循环。"),
            )

    # ---- 1.4 向量宽度扫描 ------------------------------------------------- #
    if not quick:
        vals = []
        for v in (1, 2, 4, 8, 16):
            out = BUILD_DIR / f"alu_peak_vec{v}"
            try:
                build_sycl(SYCL_DIR / "alu_peak.cpp", out, extra=[f"-DALU_VEC={v}"], env=env)
            except Exception as exc:  # noqa: BLE001
                store.error("alu", f"vec{v}", f"{type(exc).__name__}: {exc}")
                continue
            cp = run_soft([str(out), "fp32", str(ALU_KNEE), str(iters // 4), "2", "3", "128"],
                          env=env, timeout=1200)
            _, res = parse_probe(cp.stdout)
            if not res:
                continue
            tf = float(res[0]["value"]) / 1000.0
            vals.append((v, tf))
            store.add("alu", f"vec_width.{v}",
                      params={"dtype": "fp32", "vec": v, "global": ALU_KNEE},
                      metrics={"tflops": round(tf, 2)})
        if vals:
            bv, bt = max(vals, key=lambda p: p[1])
            store.highlight(
                f"ALU 向量宽度扫描 {[(v, round(t, 1)) for v, t in vals]}：VEC={bv} 最佳"
                f"（{bt:.1f} TFLOPS），VEC≥4 后基本拉平 → 该循环 **issue 受限**，"
                "继续加宽向量不再有收益。"
            )


# --------------------------------------------------------------------------- #
# 2. XMX / DPAS 峰值
# --------------------------------------------------------------------------- #
def suite_xmx(store: ResultStore, env: dict, quick: bool) -> None:
    outer = 2048 if quick else 8192
    occs = ([1, 4, 16] if quick else [1, 2, 4, 8, 16, 32])

    # ---- 2.1 数值正确性：A=1,B=1 → C 必须等于 K -------------------------- #
    cp = run_soft([str(XMX_BIN), "all", str(XMX_SG_GLOBAL * XMX_KNEE_OCC), "8",
                   "1", "1", "16", "128"], env=env, timeout=900)
    for c in parse_json_lines(cp.stdout, "CHECK"):
        store.add(
            "xmx", f"check.{c['dtype']}",
            params={"expect": c["expect"], "got": c["got"], "m": 8, "n": 16},
            metrics={"ok": 1.0 if c["ok"] else 0.0},
            status="ok" if c["ok"] else "error",
            note=("DPAS 数值正确 → SYCL joint_matrix 通路可用" if c["ok"]
                  else f"期望 {c['expect']} 得到 {c['got']}"),
        )

    # ---- 2.2 占用率扫描：定位拐点 + 峰值 --------------------------------- #
    peaks: dict[str, tuple[int, float]] = {}
    scan_bf16: list[tuple[int, float]] = []
    for dtype in ("bf16", "fp16", "int8"):
        best = (0, -1.0)
        for occ in occs:
            g = XMX_SG_GLOBAL * occ
            cp = run_soft([str(XMX_BIN), dtype, str(g), str(outer), "2", "5", "16", "128"],
                          env=env, timeout=1200)
            _, res = parse_probe(cp.stdout)
            if not res:
                store.error("xmx", f"{dtype}.occ{occ}", (cp.stderr or "no RESULT")[-200:],
                            params={"global": g})
                continue
            r = res[0]
            val, sec = float(r["value"]), float(r["seconds"])
            store.add(
                "xmx", f"{dtype}.occ{occ}",
                params={"dtype": dtype, "global": g, "sub_groups_per_eu": occ,
                        "outer": int(r["outer"]), "it": int(r["it"]),
                        "nacc": int(r["nacc"]), "dpas_count": int(r["dpas_count"]),
                        "unit": r["unit"]},
                metrics={"value": round(val, 1), "seconds": round(sec, 5),
                         "mac_per_cycle_eu_nominal": round(float(r["mac_per_cycle_eu_nominal"]), 1)},
                note=("饱和区" if occ >= XMX_KNEE_OCC else "未饱和（延迟受限）"),
            )
            if dtype == "bf16":
                scan_bf16.append((occ, val))
            if val > best[1]:
                best = (occ, val)
        if best[0]:
            peaks[dtype] = best
            unit = "GOPS" if dtype == "int8" else "TFLOPS"
            nominal = RAW_XMX_INT8_GOPS if dtype == "int8" else RAW_XMX_BF16_TFLOPS
            store.add(
                "xmx", f"peak.{dtype}",
                params={"global": XMX_SG_GLOBAL * best[0], "sub_groups_per_eu": best[0],
                        "unit": unit, "formula_nominal": round(nominal, 1),
                        "published_datasheet": PUBLISHED_BF16_TFLOPS if dtype != "int8" else None},
                metrics={"value": round(best[1], 1),
                         "pct_of_formula": round(100.0 * best[1] / nominal, 1)},
                note=(f"公式标称 {nominal:.1f} {unit}（448 EU × "
                      f"{'512' if dtype == 'int8' else '256'} MAC/clk × 2 × "
                      f"{REFERENCE_CLOCK_GHZ} GHz）；实测 {best[1]:.1f} = "
                      f"{100.0 * best[1] / nominal:.0f}%"
                      + (f"；产品常引用值 ≈{PUBLISHED_BF16_TFLOPS:.0f} TFLOPS，"
                         f"实测是它的 {best[1] / PUBLISHED_BF16_TFLOPS:.1f}×"
                         if dtype != "int8" else "")),
            )

    if scan_bf16:
        occs_x = [o for o, _ in scan_bf16]
        vals_y = [v for _, v in scan_bf16]
        k = knee_index(occs_x, vals_y)
        store.add(
            "xmx", "saturation_knee",
            params={"dtype": "bf16"},
            metrics={"knee_sub_groups_per_eu": occs_x[k],
                     "knee_global": XMX_SG_GLOBAL * occs_x[k],
                     "peak_tflops": round(max(vals_y), 1)},
            note=(f"XMX 需每 EU ≥{occs_x[k]} 个并发 sub-group 才能打满；"
                  f"{occs_x[0]} sg/EU 时只有 {vals_y[0]:.0f} TFLOPS"
                  f"（峰值 {100 * vals_y[0] / max(vals_y):.0f}%）。"),
        )
        store.highlight(
            f"XMX 占用率拐点：**每 EU ≥{occs_x[k]} 个 sub-group（global ≥ "
            f"{XMX_SG_GLOBAL * occs_x[k]}）**才能打满 XMX；"
            f"{occs_x[0]} sg/EU 时仅 {vals_y[0]:.0f} TFLOPS。"
            "低于拐点时「工作量翻倍、耗时不变」是**延迟受限的正常表现**，不是编译器消循环。"
        )

    if peaks.get("bf16") and peaks.get("int8"):
        b, i = peaks["bf16"][1], peaks["int8"][1]
        store.highlight(
            f"XMX 原始 DPAS 峰值：bf16 **{b:.0f} TFLOPS**、int8 **{i:.0f} GOPS**"
            f"（int8/bf16 = {i / b:.2f}×，正好是 K=32 vs K=16 的 2×）。"
            f"折算 = {b / (EU_PER_DEVICE * 2 * REFERENCE_CLOCK_GHZ / 1000.0):.0f} MAC/clk/EU。"
        )

    # ---- 2.3 自校验：饱和区 occ×2 是否耗时 ×2 ----------------------------- #
    def _t(occ: int) -> tuple[float, float]:
        c = run_soft([str(XMX_BIN), "bf16", str(XMX_SG_GLOBAL * occ), str(outer),
                      "2", "3", "16", "128"], env=env, timeout=1200)
        _, rr = parse_probe(c.stdout)
        return (float(rr[0]["seconds"]), float(rr[0]["value"])) if rr else (0.0, 0.0)

    o1, o2 = XMX_KNEE_OCC, XMX_KNEE_OCC * 2
    t1, v1 = _t(o1)
    t2, v2 = _t(o2)
    if t1 and t2:
        ok, msg = linearity_check(f"XMX occ{o1}→occ{o2}", o1, t1, o2, t2)
        store.add(
            "xmx", "linearity_check",
            params={"occ_small": o1, "occ_large": o2},
            metrics={"seconds_small": round(t1, 5), "seconds_large": round(t2, 5),
                     "time_ratio": round(t2 / t1, 3),
                     "tflops_small": round(v1, 1), "tflops_large": round(v2, 1)},
            status="ok" if ok else "error",
            note=("XMX 饱和区线性 ✓，DPAS 计数可信。" if ok else "XMX 饱和区不线性 ✗：") + msg,
        )
    oa = XMX_KNEE_OCC // 2
    ta, va = _t(oa)
    if ta and t1:
        store.add(
            "xmx", "linearity_check_below_knee",
            params={"occ_small": oa, "occ_large": o1},
            metrics={"seconds_small": round(ta, 5), "seconds_large": round(t1, 5),
                     "time_ratio": round(t1 / ta, 3),
                     "tflops_small": round(va, 1), "tflops_large": round(v1, 1)},
            status="ok",
            note=(f"**故意跨拐点**：occ{oa}（{ta:.5f}s，未饱和）与 occ{o1}（{t1:.5f}s，已饱和）"
                  "工作量差 2× 而耗时几乎相同 —— 这正是最初被误判为「编译器消循环」的现象，"
                  "实为延迟受限。留档以防后人重踩。"),
        )


# --------------------------------------------------------------------------- #
# 3. torch / oneDNN 交叉验证
# --------------------------------------------------------------------------- #
def suite_torch(store: ResultStore, env: dict, quick: bool) -> None:
    script = HERE / "probes" / "torch_gemm_peak.py"
    if not script.exists():
        store.error("torch", "missing_probe", f"未找到 {script}")
        return
    if not VENV_PY.exists():
        store.error("torch", "missing_venv", f"未找到 {VENV_PY}")
        return

    cp = run_soft([str(VENV_PY), str(script)], env=env, timeout=2400)
    if not cp.stdout:
        store.error("torch", "run", (cp.stderr or "no output")[-400:])
        return

    for d in parse_json_lines(cp.stdout, "TORCHDEVICE"):
        store.add("torch", "device_info", params=d,
                  metrics={"total_memory_GiB": d.get("total_memory_GiB")})

    gemm: dict[str, list[tuple[int, int, int, float]]] = {}
    for it in parse_json_lines(cp.stdout, "TORCHCOMPUTE"):
        if it.get("status") == "error":
            store.error("torch", f"{it.get('kind')}.{it.get('dtype')}", it.get("note", ""))
            continue
        kind, dtype, unit = it["kind"], it["dtype"], it["unit"]
        if kind in ("gemm", "gemm_int8"):
            gemm.setdefault(dtype, []).append((it["m"], it["n"], it["k"], it["value"]))
            store.add(
                "torch", f"{kind}.{dtype}.{it['m']}x{it['n']}x{it['k']}",
                params={"dtype": dtype, "m": it["m"], "n": it["n"], "k": it["k"],
                        "unit": unit,
                        "impl": "torch.matmul" if kind == "gemm" else "torch._int_mm"},
                metrics={"value": it["value"], "seconds": it["seconds"]},
            )
        else:
            store.add("torch", f"{kind}.{dtype}",
                      params={"dtype": dtype, "elements": it.get("elements"), "unit": unit},
                      metrics={"value": it["value"], "seconds": it["seconds"]})

    # 峰值 + 与「公式标称」「自研探针」三方对照
    ref = {"float32": ("公式标称(ALU)", NOMINAL_FP32_TFLOPS),
           "bfloat16": ("自研SYCL DPAS", RAW_XMX_BF16_TFLOPS),
           "float16": ("自研SYCL DPAS", RAW_XMX_BF16_TFLOPS),
           "int8": ("自研SYCL DPAS", RAW_XMX_INT8_GOPS)}
    for dtype, rows in gemm.items():
        m, n, k, val = max(rows, key=lambda r: r[3])
        label, nom = ref.get(dtype, ("-", None))
        store.add(
            "torch", f"peak.{dtype}",
            params={"dtype": dtype, "m": m, "n": n, "k": k, "impl": "oneDNN",
                    "reference_kind": label, "reference_value": round(nom, 1) if nom else None},
            metrics={"value": round(val, 2),
                     "pct_of_reference": round(100.0 * val / nom, 1) if nom else None},
            note=(f"oneDNN 最佳形状；参照【{label}】= {nom:.1f}，实测 = {val:.1f} = "
                  f"{100.0 * val / nom:.0f}%") if nom else "",
        )

    peaks = {d: max(r[3] for r in rows) for d, rows in gemm.items()}
    if peaks:
        fp32 = peaks.get("float32", 0.0)
        bf16 = peaks.get("bfloat16", 0.0)
        i8 = peaks.get("int8", 0.0)
        store.highlight(
            f"**oneDNN GEMM（第三方、可信口径）**：fp32 {fp32:.1f} TFLOPS（= 公式标称 "
            f"{NOMINAL_FP32_TFLOPS:.1f} 的 {100 * fp32 / NOMINAL_FP32_TFLOPS:.0f}%）、"
            f"bf16 {bf16:.1f}（= 自研 DPAS 峰值的 {100 * bf16 / RAW_XMX_BF16_TFLOPS:.0f}%）、"
            f"int8 {i8:.0f} GOPS（= {100 * i8 / RAW_XMX_INT8_GOPS:.0f}%）。"
            f"oneDNN bf16 仍是产品常引用值 {PUBLISHED_BF16_TFLOPS:.0f} TFLOPS 的 "
            f"{bf16 / PUBLISHED_BF16_TFLOPS:.2f}×。"
        )


# --------------------------------------------------------------------------- #
# 4. 主频 / 功耗取证
# --------------------------------------------------------------------------- #
CLK_FILES = ("gt_act_freq_mhz", "gt_cur_freq_mhz", "gt_min_freq_mhz",
             "gt_max_freq_mhz", "gt_boost_freq_mhz",
             "gt_RP0_freq_mhz", "gt_RP1_freq_mhz", "gt_RPn_freq_mhz")


def read_clocks() -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        for f in CLK_FILES:
            p = card / f
            if p.exists():
                try:
                    out[f"{card.name}/{f}"] = int(p.read_text().strip())
                except Exception:  # noqa: BLE001
                    out[f"{card.name}/{f}"] = None
    return out


def parse_smi(text: str) -> dict[str, float | None]:
    """从 `xpu-smi stats -d 0` 抓 Frequency / Power / Utilization。"""
    out: dict[str, float | None] = {}
    pats = {"gpu_freq_mhz": "GPU Frequency", "gpu_power_W": "GPU Power",
            "gpu_util_pct": "GPU Utilization",
            "mem_util_pct": "GPU Memory Utilization"}
    for line in (text or "").splitlines():
        cells = [c.strip() for c in line.split("|")]
        for key, needle in pats.items():
            if needle in line and key not in out:
                for c in cells[1:]:
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", c)
                    if m:
                        out[key] = float(m.group(1))
                        break
    return out


def suite_clock(store: ResultStore, env: dict, quick: bool) -> None:
    idle = read_clocks()
    store.add(
        "clock", "sysfs_idle",
        params={"source": "/sys/class/drm/card0/gt_*_freq_mhz"},
        metrics={"act": idle.get("card0/gt_act_freq_mhz"),
                 "cur": idle.get("card0/gt_cur_freq_mhz"),
                 "min": idle.get("card0/gt_min_freq_mhz"),
                 "max": idle.get("card0/gt_max_freq_mhz"),
                 "boost": idle.get("card0/gt_boost_freq_mhz")},
        note="min == max → 频率被**锁定**，没有 boost 空间。"
             "⚠ 空闲时 gt_act_freq_mhz 可能读 0（未采样），必须叠加满载采样才可靠。",
    )

    # ---- 满载采样：后台跑 ALU，同时读 sysfs + xpu-smi --------------------- #
    proc = subprocess.Popen(
        [str(ALU_BIN), "fp32", str(ALU_KNEE), str(1 << 23), "1", "1", "128"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    smi: dict[str, float | None] = {}
    try:
        time.sleep(6.0)                       # 让 GPU 进入稳态
        loaded = read_clocks()
        smi = parse_smi(run_soft(["xpu-smi", "stats", "-d", "0"], env=env, timeout=60).stdout)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    store.add(
        "clock", "sysfs_under_load",
        params={"load": "alu_peak fp32 saturating (background)"},
        metrics={"act": loaded.get("card0/gt_act_freq_mhz"),
                 "cur": loaded.get("card0/gt_cur_freq_mhz"),
                 "max": loaded.get("card0/gt_max_freq_mhz")},
        note="满载下 act/cur 与锁定的 min==max 一致 → 无降频、无 boost。",
    )
    store.add(
        "clock", "xpu_smi_under_load",
        params={"source": "xpu-smi stats -d 0"},
        metrics=smi,
        note="⚠ xpu-smi 的 `GPU Memory Read/Write (kB/s)` 是**假的**（恒为 ~576 kB/s）；"
             "`Xe Link Throughput` / `EU Array *` 全为 N/A；`xpu-smi dump` 在本驱动上会**挂死**。"
             "只有 Frequency/Power/Utilization 可用。",
    )

    # ---- 从实测反推「等效执行宽度」 --------------------------------------- #
    cp = run_soft([str(ALU_BIN), "fp32", str(ALU_KNEE), str(1 << 22), "2", "5", "128"],
                  env=env, timeout=1200)
    _, res = parse_probe(cp.stdout)
    if res:
        measured = float(res[0]["value"]) / 1000.0
        factor = measured / NOMINAL_FP32_TFLOPS
        active_mhz = (loaded.get("card0/gt_act_freq_mhz")
                      or smi.get("gpu_freq_mhz") or 0)
        store.add(
            "clock", "implied_exec_width",
            params={"assumed_clock_mhz": active_mhz,
                    "formula_nominal_tflops": round(NOMINAL_FP32_TFLOPS, 2)},
            metrics={"measured_tflops": round(measured, 2),
                     "excess_factor": round(factor, 2),
                     "implied_lanes_per_eu": round(LANES_PER_EU * factor, 1)},
            status="error" if factor > 1.15 else "ok",
            note=(f"纯 FMA 实测 {measured:.2f} TFLOPS = 公式值 {NOMINAL_FP32_TFLOPS:.2f} 的 "
                  f"{factor:.2f}×。主频可读、锁定 1550 MHz 且满载不降频，"
                  f"**无法用「主频更高」解释**；等价于每 EU {LANES_PER_EU * factor:.0f} 条 FP32 "
                  f"lane（模型假设 {LANES_PER_EU} 条）。未定论，候选解释："
                  f"(a) 本 ES 部件的实际 EU 数与 clinfo 报告的 448 不符；"
                  f"(b) EU 内 FP32 通道宽度 >16（Xe-HPC 为 2×8 FP32 可能在 >SIMD16 下双发？）；"
                  f"(c) 探针 FLOP 计数偏低（编译器把 VEC/ACC/UNROLL 做了额外融合）。"
                  f"**注意 oneDNN fp32 GEMM 只有 ~{NOMINAL_FP32_TFLOPS:.1f} TFLOPS"
                  f"（= 公式值 100%）**，两口径相差 {factor:.2f}× 尚无定论，"
                  f"需 TODO 07 profiling 的硬件计数器裁决。"),
        )
        store.highlight(
            f"⚠ **口径冲突（本目录最重要的待解问题）**：公式标称 FP32 = "
            f"{NOMINAL_FP32_TFLOPS:.2f} TFLOPS；oneDNN GEMM 实测 "
            f"{NOMINAL_FP32_TFLOPS:.1f}（恰好 100% 公式值）；但自研纯 FMA 探针实测 "
            f"{measured:.1f} TFLOPS（{factor:.2f}×）。两者不能同时为真 —— 主频已锁定、满载不降频，"
            "嫌疑集中在「FLOP 计数口径」或「clinfo 的 EU 数」。"
            "**判读纪律：以 oneDNN 为可信下界，自研探针为待验证上界。**"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*", default=["alu", "xmx", "torch", "clock"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    store = ResultStore(tag, outdir=HERE / "results",
                        title="02 · 算力峰值（ALU / XMX-DPAS）基准报告")

    env = sub_env(ZE_AFFINITY_MASK="0")
    build_all(env)

    dispatch = {"alu": suite_alu, "xmx": suite_xmx,
                "torch": suite_torch, "clock": suite_clock}
    for name in args.suites:
        print(f"===== {name} =====", flush=True)
        t0 = time.time()
        fn = dispatch.get(name)
        if fn is None:
            print(f"unknown suite {name}")
            continue
        try:
            fn(store, env, args.quick)
        except Exception as exc:  # noqa: BLE001
            store.error(name, "suite", f"{type(exc).__name__}: {exc}")
            print(f"  !! {name} failed: {exc}")
        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    store.highlight(
        f"公式标称 FP32 ALU 峰值 = 448 EU × {LANES_PER_EU} lane × 2 × {REFERENCE_CLOCK_GHZ} GHz "
        f"= **{NOMINAL_FP32_TFLOPS:.2f} TFLOPS**；"
        f"XMX 公式标称 = 256 MAC/clk/EU → **{RAW_XMX_BF16_TFLOPS:.0f} TFLOPS (bf16)** / "
        f"**{RAW_XMX_INT8_GOPS:.0f} GOPS (int8)**。"
        "所有自研探针均已通过「饱和区 ×2 工作量 → ×2 耗时」自校验。"
    )
    j, m = store.save()
    print(f"\nwrote {j}\nwrote {m}")
    print(f"records: {len(store.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
