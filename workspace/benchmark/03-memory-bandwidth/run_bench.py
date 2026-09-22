#!/usr/bin/env python3
"""03-memory-bandwidth / run_bench.py

HBM3/HBM2e 显存带宽基准。对应 docs/TODO/03-memory-bandwidth.md。

包含的 6 组：
  babelstream  BabelStream SYCL 版，5 个 kernel × 数组尺寸扫描
  probe        自研 SYCL 探针：单向 read / write（BabelStream 没有）+ vec/stride 扫描
  dual_gpu     两块 Max 1100 同时跑 copy，看 HBM 带宽是否能线性叠加
  torch        PyTorch 交叉验证：D2D copy、H2D/D2H（pinned 与 pageable）
  host         主机内存带宽基线（/usr/bin/stream）
  counters     运行 BabelStream 时用 xpu-smi 采样显存带宽利用率

用法：
    python3 run_bench.py                 # 全跑
    python3 run_bench.py --quick         # 少迭代
    python3 run_bench.py babelstream probe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
sys.path.insert(0, str(BENCH_ROOT / "common"))

from bench import (  # noqa: E402
    HBM_SPEC_GBPS,
    ResultStore,
    gibps,
    parse_json_lines,
    run,
    run_soft,
    sub_env,
)

BABELSTREAM = HERE / "babelstream" / "build" / "sycl-stream"
PROBE_BIN = Path("/tmp/bw_probe")
PROBE_SRC = HERE / "probes" / "bw_probe.cpp"
BUILD_DIR = HERE / "build"
VENV_PY = Path("/root/workspace/venv1/bin/python")

MiB = 1 << 20
GiB = 1 << 30

# BabelStream 的 -s 参数是「数组里的元素个数」，默认 double → 8 B/元素
BS_ELEM_BYTES = 8


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build_probe(env: dict) -> None:
    if PROBE_BIN.exists() and PROBE_BIN.stat().st_mtime > PROBE_SRC.stat().st_mtime:
        return
    print("[build] bw_probe.cpp -> %s" % PROBE_BIN)
    cp = run(
        ["icpx", "-fsycl", "-O3", "-o", str(PROBE_BIN), str(PROBE_SRC)],
        env=env,
        timeout=900,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError("bw_probe build failed:\n" + cp.stderr[-3000:])


# --------------------------------------------------------------------------- #
# 1. BabelStream
# --------------------------------------------------------------------------- #
BS_KERNELS = ("Copy", "Mul", "Add", "Triad", "Dot")
BS_ROW_RE = re.compile(r"^(Copy|Mul|Add|Triad|Dot)\s+([0-9.]+)\s+([0-9.]+)", re.M)


def parse_babelstream(text: str) -> dict[str, tuple[float, float]]:
    """-> {kernel: (GB/s, best_seconds)}

    ⚠ 我们调用时传了 ``--gigabytes``，所以带宽列**已经是 GB/s**，不要再 /1000。
    列名仍然显示 "MB/s"，这是 BabelStream 的一个小坑。
    """
    out: dict[str, tuple[float, float]] = {}
    for name, gbs, sec in BS_ROW_RE.findall(text):
        out[name] = (float(gbs), float(sec))
    return out


def suite_babelstream(store: ResultStore, env: dict, quick: bool) -> None:
    if not BABELSTREAM.exists():
        store.error("babelstream", "build", f"未找到 {BABELSTREAM}；请先 cmake 构建")
        return

    sizes = (
        [4 * MiB, 16 * MiB, 64 * MiB, 192 * MiB, 512 * MiB, 2 * GiB]
        if quick
        else [4 * MiB, 16 * MiB, 64 * MiB, 128 * MiB, 192 * MiB, 256 * MiB,
              512 * MiB, 1 * GiB, 2 * GiB, 4 * GiB]
    )
    ntimes = 5 if quick else 20

    for nbytes in sizes:
        elems = nbytes // BS_ELEM_BYTES
        cp = run_soft(
            [str(BABELSTREAM), "-s", str(elems), "-n", str(ntimes), "--gigabytes"],
            env=env,
            timeout=900,
        )
        if cp.returncode != 0 and not cp.stdout:
            store.error("babelstream", "run", (cp.stderr or "no output")[-300:],
                        params={"array_MiB": nbytes / MiB})
            continue
        parsed = parse_babelstream(cp.stdout)
        if not parsed:
            store.error("babelstream", "parse", "无法解析输出", params={"array_MiB": nbytes / MiB})
            continue
        l2_resident = nbytes <= 192 * MiB  # L2 = 192 MB
        for k in BS_KERNELS:
            if k not in parsed:
                continue
            gbps, sec = parsed[k]
            store.add(
                "babelstream",
                f"BabelStream.{k}",
                params={"array_MiB": round(nbytes / MiB, 1),
                        "elements": elems, "n": ntimes,
                        "note_L2": "L2内(非HBM)" if l2_resident else "HBM"},
                metrics={
                    "gbps": round(gbps, 2),
                    "best_s": sec,
                    "pct_of_1229": round(gbps / HBM_SPEC_GBPS * 100.0, 1),
                },
                note="数组 ≤192MiB 时命中 192MB L2，数值不代表 HBM" if l2_resident else "",
            )

    # 关键结论
    big = [r for r in store.results
           if r.suite == "babelstream" and r.params.get("note_L2") == "HBM"
           and r.name == "BabelStream.Copy"]
    if big:
        best = max(big, key=lambda r: r.metrics["gbps"])
        store.highlight(
            f"BabelStream Copy 峰值 **{best.metrics['gbps']:.0f} GB/s** "
            f"(数组 {best.params['array_MiB']} MiB)，= {best.metrics['pct_of_1229']:.0f}% "
            f"of {HBM_SPEC_GBPS:.0f} GB/s 规格值"
        )


# --------------------------------------------------------------------------- #
# 2. 自研探针
# --------------------------------------------------------------------------- #
def probe(env: dict, mode: str, nbytes: int, extra: list[str]) -> list[dict]:
    cp = run_soft(
        [str(PROBE_BIN), "--mode", mode, "--bytes", str(nbytes),
         "--iters", "20", "--warmup", "3", *extra],
        env=env,
        timeout=900,
    )
    return parse_json_lines(cp.stdout, "BWRESULT")


def suite_probe(store: ResultStore, env: dict, quick: bool) -> None:
    if not PROBE_BIN.exists():
        store.error("probe", "build", "bw_probe 未构建")
        return
    nbytes = 1 * GiB if quick else 2 * GiB
    it = 5 if quick else 20

    # --- 单向读 / 写 / 拷贝 / triad（vec=4）---
    for mode in ("copy", "read", "write", "triad"):
        rows = probe(env, mode, nbytes, ["--vec", "4", "--iters", str(it)])
        for r in rows:
            store.add(
                "probe",
                f"bw_probe.{mode}",
                params={"vec": r["vec"], "array_MiB": round(r["array_bytes"] / MiB, 1),
                        "stride": r["stride"]},
                metrics={"gbps": r["gbps"],
                         "ms": r["ms"],
                         "pct_of_1229": round(r["gbps"] / HBM_SPEC_GBPS * 100.0, 1)},
                note="单向：1 读或 1 写" if mode in ("read", "write") else "",
            )

    # --- 向量宽度扫描（copy）---
    for r in probe(env, "copy", nbytes, ["--sweep", "vec", "--iters", str(it)]):
        store.add("probe", "vec_sweep.copy",
                  params={"vec": r["vec"], "stride": r["stride"]},
                  metrics={"gbps": r["gbps"], "ms": r["ms"]})

    # --- 跨步扫描（copy）---
    for r in probe(env, "copy", nbytes, ["--sweep", "stride", "--iters", str(it)]):
        store.add("probe", "stride_sweep.copy",
                  params={"vec": r["vec"], "stride": r["stride"]},
                  metrics={"gbps": r["gbps"], "ms": r["ms"]},
                  note="stride>1 时按逻辑访问字节计数，下降源于缓存行利用率而非 HBM")

    # 汇总：读/写不对称
    rd = [r for r in store.results if r.name == "bw_probe.read"]
    wr = [r for r in store.results if r.name == "bw_probe.write"]
    if rd and wr:
        store.highlight(
            f"单向带宽：read **{rd[0].metrics['gbps']:.0f} GB/s** vs "
            f"write **{wr[0].metrics['gbps']:.0f} GB/s** "
            f"(读 {rd[0].metrics['gbps']/max(wr[0].metrics['gbps'],1e-9):.2f}× 于写)"
        )


# --------------------------------------------------------------------------- #
# 3. 双卡并发
# --------------------------------------------------------------------------- #
def suite_dual_gpu(store: ResultStore, env: dict, quick: bool) -> None:
    if not BABELSTREAM.exists():
        store.error("dual_gpu", "BabelStream", "未找到 sycl-stream")
        return
    elems = (512 * MiB) // BS_ELEM_BYTES if quick else (2 * GiB) // BS_ELEM_BYTES
    ntimes = 5 if quick else 20

    def one(gpu: str) -> float:
        cp = run_soft(
            [str(BABELSTREAM), "-s", str(elems), "-n", str(ntimes),
             "--gigabytes", "-o", "Copy"],
            env={**env, "ZE_AFFINITY_MASK": gpu},
            timeout=900,
        )
        p = parse_babelstream(cp.stdout)
        return p.get("Copy", (0.0, 0.0))[0]

    solo0 = one("0")
    solo1 = one("1")
    store.add("dual_gpu", "solo.GPU0", params={"GPU": 0},
              metrics={"copy_gbps": round(solo0, 2)})
    store.add("dual_gpu", "solo.GPU1", params={"GPU": 1},
              metrics={"copy_gbps": round(solo1, 2)})

    # 并发：两个进程同时跑
    procs = []
    for gpu in ("0", "1"):
        procs.append(subprocess.Popen(
            [str(BABELSTREAM), "-s", str(elems), "-n", str(ntimes),
             "--gigabytes", "-o", "Copy"],
            env={**env, "ZE_AFFINITY_MASK": gpu},
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=900)
        outs.append(parse_babelstream(out).get("Copy", (0.0, 0.0))[0])

    agg = sum(outs)
    store.add("dual_gpu", "concurrent.both",
              params={"GPU": "0+1", "elems": elems},
              metrics={"GPU0_gbps": round(outs[0], 2),
                       "GPU1_gbps": round(outs[1], 2),
                       "aggregate_gbps": round(agg, 2),
                       "scaling_vs_solo_mean": round(agg / max((solo0 + solo1) / 2, 1e-9), 2)},
              note="两卡各自独立 HBM，理论应接近 2×；若明显 <2× 说明受主机/PCIe 限制")
    store.highlight(f"双卡并发 copy 合计 **{agg:.0f} GB/s** "
                    f"(GPU0 {outs[0]:.0f} + GPU1 {outs[1]:.0f})")


# --------------------------------------------------------------------------- #
# 4. PyTorch 交叉验证
# --------------------------------------------------------------------------- #
def suite_torch(store: ResultStore, env: dict, quick: bool) -> None:
    helper = HERE / "probes" / "torch_membw.py"
    if not VENV_PY.exists():
        store.error("torch", "venv", f"未找到 {VENV_PY}")
        return
    if not helper.exists():
        store.error("torch", "helper", f"未找到 {helper}")
        return
    cmd = [str(VENV_PY), str(helper)]
    if quick:
        cmd.append("--quick")
    cp = run_soft(cmd, env=env, timeout=1800, cwd=str(HERE))
    payload = parse_json_lines(cp.stdout, "TORCHBW")
    if not payload:
        store.error("torch", "run", (cp.stderr or cp.stdout or "no output")[-400:])
        return
    for item in payload[0].get("items", []):
        store.add(
            "torch",
            item["name"],
            params=item.get("params", {}),
            metrics=item["metrics"],
            status=item.get("status", "ok"),
            note=item.get("note", ""),
        )

    d2d = [r for r in store.results if r.name.startswith("torch.D2D")]
    if d2d:
        best = max(d2d, key=lambda r: r.metrics.get("gbps", 0))
        store.highlight(
            f"PyTorch D2D copy 峰值 **{best.metrics['gbps']:.0f} GB/s** "
            f"({best.params.get('dtype')}, {best.params.get('MiB')} MiB)"
        )


# --------------------------------------------------------------------------- #
# 5. 主机内存带宽
# --------------------------------------------------------------------------- #
def suite_host(store: ResultStore, env: dict, quick: bool) -> None:
    """主机内存带宽基线。

    ⚠ 本机的 ``/usr/bin/stream`` 是 **ImageMagick** 的 stream 命令，不是 UVA STREAM
    基准（实测：`/usr/bin/stream` 打印 ImageMagick usage）。所以我们自己编一个最小
    STREAM（probes/host_stream.c，OpenMP 并行）。
    """
    src = HERE / "probes" / "host_stream.c"
    exe = BUILD_DIR / "host_stream"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        cp = run(["gcc", "-O3", "-fopenmp", "-o", str(exe), str(src)],
                 env=env, timeout=600, check=False)
        if cp.returncode != 0:
            store.error("host", "build", cp.stderr[-300:])
            return
    # 数组大小扫描：本机 **L3 = 432 MiB**，合计工作集 ≤432 MiB 时测到的是 L3 带宽，
    # 会虚高 3~5 倍（和 GPU 的 192 MB L2 是同一个坑）。必须报出 DRAM 档。
    # (每数组 MiB, 是否落在 L3)
    sizes = [(16, True), (128, True), (1024, False)] if not quick else [(16, True), (1024, False)]
    rows: list[dict] = []
    for mb, in_l3 in sizes:
        n_elems = (mb * (1 << 20)) // 8          # doubles
        cp = run_soft([str(exe), str(n_elems), "3" if quick else "5", "72"],
                      env={**env, "OMP_NUM_THREADS": "72"}, timeout=1200)
        for r in parse_json_lines(cp.stdout, "HOSTBW"):
            r["in_l3"] = in_l3
            rows.append(r)
    if not rows:
        store.error("host", "host_stream", "无输出（gcc -fopenmp 编译失败？）")
        return
    for r in rows:
        l3 = "L3内(非DRAM)" if r["in_l3"] else "DRAM"
        store.add("host", f"host_stream.{r['kernel']}",
                  params={"array_MiB": round(r["array_MiB"], 1), "threads": r["threads"],
                          "working_set_MB": round(3 * r["array_MiB"], 1), "note_regime": l3},
                  metrics={"gbps": r["gbps"], "seconds": r["seconds"],
                           "io_factor": r["io_factor"]},
                  note=f"主机内存 {l3}；L3=432 MiB（lscpu）。主机带宽决定 "
                       f"dataloader/预处理上限，也是多卡喂数据的天花板")
    dram = [r for r in rows if not r["in_l3"] and r["kernel"] == "Copy"]
    l3rows = [r for r in rows if r["in_l3"] and r["kernel"] == "Copy"]
    if dram:
        store.highlight(f"主机 **DRAM** copy 带宽 **{max(r['gbps'] for r in dram):.1f} GB/s** "
                        f"(1 GiB/数组, 72 线程)")
    if l3rows and dram:
        store.highlight(
            f"主机内存的 L3 陷阱：16~128 MiB/数组时 copy 报到 "
            f"**{max(r['gbps'] for r in l3rows):.0f} GB/s**（命中 432 MiB L3），"
            f"放大到 1 GiB/数组才落到真实 DRAM 的 "
            f"{max(r['gbps'] for r in dram):.0f} GB/s")
    # CPU 拓扑（解释 L3 尺寸 / 线程规模）
    cp = run_soft(["lscpu"], env=env, timeout=60)
    topo = {}
    for key, pat in (("l3_MiB", r"L3 cache:\s+([0-9.]+) MiB"),
                     ("l2_MiB", r"L2 cache:\s+([0-9.]+) MiB"),
                     ("cpus", r"^CPU\(s\):\s+([0-9]+)"),
                     ("numa_nodes", r"NUMA node\(s\):\s+([0-9]+)"),
                     ("model", r"Model name:\s+(.+)$")):
        m = re.search(pat, cp.stdout, re.M)
        if m:
            topo[key] = float(m.group(1)) if key.endswith("_MiB") else m.group(1).strip()
    if topo:
        store.add("host", "cpu_topology", params={},
                  metrics={k: v for k, v in topo.items() if k != "model"},
                  note=f"CPU: {topo.get('model', '?')}；L3={topo.get('l3_MiB')} MiB 是"
                       f"主机内存测速的关键参数")


# --------------------------------------------------------------------------- #
# 6. xpu-smi 计数器交叉验证
# --------------------------------------------------------------------------- #
def _xpu_smi_stats(gpu: str = "0", extra: tuple = ()) -> dict:
    """用 xpu-smi stats 读一次全部计数器，返回 {metric_name: value|None}。

    None 表示驱动报 N/A。
    ⚠ ``xpu-smi dump`` 在 hwt 的驱动上是坏的（任何 metric 都只输出表头然后挂住，
    实测 metric 0/1/2/9/17 全部一样），只能用 ``stats``。
    """
    cp = run_soft(["xpu-smi", "stats", "-d", gpu, *extra], timeout=90)
    txt = cp.stdout.replace("|", " ").replace("+", " ")
    out: dict[str, float | None] = {}
    for line in txt.splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ()/_\-]*?)\s+(N/A|[0-9.]+)(?:\s*;.*)?$", line)
        if m:
            name, val = m.group(1).strip(), m.group(2)
            out[name] = None if val == "N/A" else float(val)
    return out


def suite_counters(store: ResultStore, env: dict, quick: bool) -> None:
    """硬件计数器可用性验证 + 交叉验证。

    结论（实测，2026-09-22）：hwt 的 i915 驱动上 xpu-smi 的**显存读写计数器是假的** ——
    空载读到 576 kB/s，在 BabelStream 以 ~900 GB/s 跑 copy 时依然读到 576 kB/s。
    所以不能用它来交叉验证带宽；本 suite 把这个事实**如实记录**为一项发现，
    真正的绝对带宽仍以 BabelStream / 自研探针为准。
    """
    idle = _xpu_smi_stats("0")
    store.add("counters", "idle_baseline", params={},
              metrics={k: v for k, v in idle.items() if v is not None},
              note=f"空载计数器。本驱动 N/A 的指标："
                   f"{', '.join(k for k, v in idle.items() if v is None)}")

    if not BABELSTREAM.exists():
        store.skip("counters", "under_load", "BabelStream 缺失")
        return

    elems = (4 * GiB) // BS_ELEM_BYTES
    p = subprocess.Popen(
        [str(BABELSTREAM), "-s", str(elems), "-n", "2000", "--gigabytes", "-o", "Copy"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)  # 等负载稳定（xpu-smi stats 本身要 ~1.5-2 s）
    samples: list[dict] = []
    try:
        for _ in range(5):
            samples.append(_xpu_smi_stats("0"))
            if p.poll() is not None:
                break
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

    if not samples:
        store.skip("counters", "under_load", "xpu-smi stats 无返回")
        return

    def peak(key: str) -> float | None:
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        return max(vals) if vals else None

    read_kbs, write_kbs = peak("GPU Memory Read (kB/s)"), peak("GPU Memory Write (kB/s)")
    bw_pct, mem_util = peak("GPU Memory Bandwidth (%)"), peak("GPU Memory Util (%)")
    power = peak("GPU Power (W)")

    # 判定：读+写合计超过 10 GB/s 才算计数器“活着”（900 GB/s 的负载应该远超这个门槛）
    total_gbps = ((read_kbs or 0.0) + (write_kbs or 0.0)) / 1e6
    usable = total_gbps > 10.0
    store.add("counters", "mem_rw_under_load",
              params={"samples": len(samples), "load": "BabelStream Copy 4GiB"},
              metrics={"read_kbs": read_kbs, "write_kbs": write_kbs,
                       "read_write_total_gbps": round(total_gbps, 3),
                       "mem_bw_pct": bw_pct, "mem_util_pct": mem_util,
                       "power_w": power,
                       "counter_usable": usable},
              status="ok" if usable else "error",
              note="驱动报的显存读写计数器在满载（~900 GB/s）下依然停在 ~576 kB/s，"
                   "与空载完全相同 → **该计数器在本驱动上不可用**，不能用它交叉验证带宽。"
                   "对照：同一时刻 GPU Power 已升到 260+ W，说明负载确实在跑。"
                   if not usable else
                   "计数器随负载变化，可用于交叉验证")

    idle_r = idle.get("GPU Memory Read (kB/s)") or 0.0
    idle_w = idle.get("GPU Memory Write (kB/s)") or 0.0
    store.highlight(
        f"xpu-smi 显存读写计数器**不可用**：空载 {idle_r/1e6:.3f}+{idle_w/1e6:.3f} GB/s，"
        f"满载 {total_gbps:.3f} GB/s（毫无变化）；`xpu-smi dump` 也完全坏的。"
        f"带宽绝对值只能以 BabelStream/自研探针为准。"
        if not usable else
        f"xpu-smi 计数器（copy 满载）：读+写 = **{total_gbps:.0f} GB/s**"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*",
                    default=["babelstream", "probe", "dual_gpu", "torch", "host", "counters"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    store = ResultStore(tag, outdir=HERE / "results",
                        title="03 · 显存带宽（HBM）基准报告")

    env = sub_env(ZE_AFFINITY_MASK="0")
    build_probe(env)

    for name in args.suites:
        print(f"===== {name} =====", flush=True)
        t0 = time.time()
        try:
            if name == "babelstream":
                suite_babelstream(store, env, args.quick)
            elif name == "probe":
                suite_probe(store, env, args.quick)
            elif name == "dual_gpu":
                suite_dual_gpu(store, env, args.quick)
            elif name == "torch":
                suite_torch(store, env, args.quick)
            elif name == "host":
                suite_host(store, env, args.quick)
            elif name == "counters":
                suite_counters(store, env, args.quick)
            else:
                print(f"unknown suite {name}")
        except Exception as exc:  # noqa: BLE001
            store.error(name, "suite", f"{type(exc).__name__}: {exc}")
            print(f"  !! {name} failed: {exc}")
        print(f"  ({time.time() - t0:.1f}s)", flush=True)

    store.highlight(
        f"HBM 规格 {HBM_SPEC_GBPS:.0f} GB/s（Max 1100，48 GiB HBM2e）。"
        "BabelStream 是行业标准口径，自研探针用于补充单向读/写与访问模式。"
    )
    j, m = store.save()
    print(f"\nwrote {j}\nwrote {m}")
    print(f"records: {len(store.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
