#!/usr/bin/env python3
"""benchmark/04-interconnect-xelink — Xe Link / GPU-aware MPI / oneCCL 互连实测

对应文档：docs/TODO/04-interconnect-xelink.md

本机拓扑（docs/interconnect.md）
     GPU 0/0  S        XL24      0-143
     GPU 1/0  XL24     S         0-143
  → Xe Link XL24 = 6 端口 × 4 lane **全直连**（不经 MDF）→ 理论 ~318 GB/s/方向
    （PCIe Gen5 x16 原始 ~63 GB/s，所以 Xe Link 应该快 ~5×）

  但 xpu-smi 报 "Xe Link Calibration Date: Not Calibrated" —— 这是本目录要量化的重点。

──────────────────────────────────────────────────────────────────────────────
★ 方法论：三条互相独立的证据链，缺一不可
──────────────────────────────────────────────────────────────────────────────

  证据链 A —— 【能力】Level Zero 原生查询（probes/p2p_probe.cpp）
      zeDeviceCanAccessPeer() / zeDeviceGetP2PProperties()
      → 直接回答「P2P 在驱动层开没开」。不开的话下面全部无意义。

  证据链 B —— 【带宽】单 context 双 device 跨卡 memcpy（同一个 p2p_probe）
      src 分配在 GPU0、dst 分配在 GPU1，用 GPU1 的 queue 发
      zeCommandListAppendMemoryCopy。P2P 生效时这次拷贝直接在卡间走 Xe Link，
      不经 host。这是**最干净的 Xe Link 带宽测量**：没有 MPI、没有框架、
      没有内核启动噪声，就是一次 DMA。
      （同一个 Level Zero context 里的两个 device 内存互相可见，
        所以不需要 zeMemGetIpcHandle/zeMemOpenIpcHandle。）

  证据链 C —— 【端到端】IMB-MPI1-GPU + torch.distributed(xccl)
      GPU-aware MPI 是最贴近真实多卡负载的路径；xccl 是 torch 训练实际走的路径。
      它们的数字会低于证据链 B（多了协议栈开销），但决定「应用能不能吃到」。

  三条链的排序判读：
      若 B 高（>150 GB/s）而 C 低  → 链路没问题，是 MPI/oneCCL 没配对
      若 B 也低（~95 GB/s）        → 链路本身只有这个速度 → 怀疑未标定
      若 A 为 false               → 全盘退化到 PCIe/host

──────────────────────────────────────────────────────────────────────────────
★ 口径说明（容易踩）
──────────────────────────────────────────────────────────────────────────────
  · 单向拷贝带宽 GB/s = 字节数 / 秒         （本目录 P2PBW / IMB 的口径）
  · algbw (algorithm bw) = 消息字节 / 秒
  · busbw (bus bw)       = algbw × 2(W-1)/W   ← 集合通信「总线口径」，
                            1 个字节的 allreduce 实际要走 2(W-1)/W 字节
  · 不要把 allreduce 的 algbw 直接和 P2P 单向带宽比；要么都用 busbw，
    要么把 busbw × W/(2(W-1)) 换回单向。
  · IMB 的表头单位是 MB/s（10^6）；本目录统一换算成 GB/s（10^9）。

──────────────────────────────────────────────────────────────────────────────
★ 预期坑（见 §坑）
  · xpu-smi dump 会**挂死**（rc=143）→ 只用 xpu-smi stats -d 0
  · xpu-smi 的 Xe Link Throughput 在本驱动上读 N/A → 硬件侧**无法取证**
  · mpirun 以 root 运行需要额外参数
  · GPU-aware MPI 必须设 ZE_ENABLE_PCI_ID_DEVICE_ORDER=1，否则 rank↔卡 绑定乱序

用法：
    python3 run_bench.py                    # 全部 suite
    python3 run_bench.py p2p topo           # 只跑部分
    python3 run_bench.py --quick            # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

from bench import (  # noqa: E402
    GPU_NAME,
    XE_LINK_SPEC_GBPS,
    ResultStore,
    parse_json_lines,
    run,
    run_soft,
    sub_env,
)

# --------------------------------------------------------------------------- #
PYTHON = "/root/workspace/venv1/bin/python"
CXX = "g++"
BUILD = HERE / "build"
PROBES = HERE / "probes"
# 本机 /opt/intel/oneapi/mpi/ 下有 2021.17 / 2021.18 / latest，取版本最高的
MPI_BASE = "/opt/intel/oneapi/mpi"

# Xe Link XL24: 6 ports × 4 lanes；每 lane ~13.25 GB/s/dir → ~318 GB/s
XE_LINK_PORTS = 6
XE_LINK_LANES_PER_PORT = 4
PCIE_GEN5_X16_GBPS = 63.0     # Gen5 x16 原始单向
PCIE_GEN4_X16_GBPS = 31.5

# 判读阈值（GB/s，单向）
TH_XELINK_FULL = 200.0        # ≥200 → 基本是 Xe Link 全速
TH_XELINK_PARTIAL = 60.0      # 60~200 → 走了 P2P 但明显不满
TH_PCIE = 55.0                # <55 → 疑似退化到 PCIe


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build_probes(env: dict) -> tuple[bool, str]:
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "p2p_probe"
    cmd = [CXX, "-O3", "-std=c++17", "-o", str(out), str(PROBES / "p2p_probe.cpp"),
           "-lze_loader"]
    p = run_soft(cmd, env=env, cwd=HERE)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout or "")[-800:]
    return True, ""


# --------------------------------------------------------------------------- #
# 解析工具
# --------------------------------------------------------------------------- #
def parse_probe(text: str) -> tuple[dict, list[dict], list[dict]]:
    """probes/p2p_probe.cpp 的输出 → (info, peers, bandwidth points)。"""
    info: dict = {}
    peers = parse_json_lines(text, "P2PPEER")
    bws = parse_json_lines(text, "P2PBW")
    devs = parse_json_lines(text, "P2PDEV")
    infl = parse_json_lines(text, "P2PINFO")
    errs = parse_json_lines(text, "P2PERR")
    for d in infl:
        info.update(d)
    if devs:
        info["devices"] = devs
    if errs:
        info["errors"] = errs
    return info, peers, bws


# ★ 集合通信表头是 5 列：
#   #bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]
# IMB 对集合通信**不输出 MB/s**，需要自己用 t_avg 算。
_IMB_ROW_P2P = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)
_IMB_ROW_COLL = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)
# 兼容旧名字
_IMB_ROW = _IMB_ROW_P2P


def parse_imb(text: str) -> tuple[dict, list[dict]]:
    """解析 IMB-MPI1(-GPU) 的输出表。

    表头形如 ``#bytes #repetitions  t[usec]  Mbytes/sec``（点对点，4 列）
    或 ``#bytes #repetitions  t_min t_max t_avg``（集合通信，5 列）。
    IMB 可能输出多张表（多进程数/多 benchmark），
    按 ``# Benchmarking <Name>`` 切块。

    返回 (meta, rows)；rows 元素：
        {"bench": "PingPong", "size_bytes": 1024, "t_usec": ..., "mbps": ...,
         "gbps": ..., "latency_us": ...}
    """
    rows: list[dict] = []
    bench = "?"
    procs = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^#\s*Benchmarking\s+(\S+)", s)
        if m:
            bench = m.group(1)
            continue
        m = re.match(r"^#\s*#processes\s*=\s*(\d+)", s)
        if m:
            procs = int(m.group(1))
            continue
        m = _IMB_ROW_COLL.match(line)
        if m:
            # 5 列 → 集合通信表：带宽只能自己算（algbw = size / t_avg）
            nbytes = int(m.group(1))
            reps = int(m.group(2))
            t_min = float(m.group(3))
            t_max = float(m.group(4))
            t_avg = float(m.group(5))
            if t_avg > 0:
                mbps = nbytes / t_avg          # B/usec == MB/s
            else:
                mbps = 0.0
            rows.append({
                "bench": bench,
                "processes": procs,
                "size_bytes": nbytes,
                "repetitions": reps,
                "t_usec": t_min,
                "t_avg_usec": t_avg,
                "t_max_usec": t_max,
                "mbps": mbps,
                "gbps": mbps / 1000.0,
                "bandwidth_gbps": mbps / 1000.0,
                "bandwidth_basis": "algbw = size / t_avg（IMB 集合通信不直接给 MB/s）",
            })
            continue
        m = _IMB_ROW_P2P.match(line)
        if m:
            nbytes = int(m.group(1))
            reps = int(m.group(2))
            t_usec = float(m.group(3))
            mbps = float(m.group(4))
            rows.append({
                "bench": bench,
                "processes": procs,
                "size_bytes": nbytes,
                "repetitions": reps,
                "t_usec": t_usec,
                "mbps": mbps,
                "gbps": mbps / 1000.0,
                "bandwidth_gbps": mbps / 1000.0,
                "bandwidth_basis": "IMB 直接给出 MB/s",
            })
    return {"benchmarks": sorted({r["bench"] for r in rows}),
            "processes": procs,
            "rows": len(rows)}, rows


def imb_peak(rows: list[dict], bench: str) -> dict | None:
    """取某个 benchmark 的最大带宽点与最小延迟点。"""
    sub = [r for r in rows if r["bench"].lower() == bench.lower()]
    if not sub:
        return None
    # 小消息的 MB/s 可能是 0（不可测），带宽取真正随 size 增长后的最大值
    bw = max(sub, key=lambda r: r["gbps"])
    lat = min(sub, key=lambda r: r["t_usec"])
    return {
        "bandwidth_gbps": round(bw["gbps"], 3),
        "bandwidth_at_bytes": bw["size_bytes"],
        "bw_t_usec": bw["t_usec"],
        "latency_us": round(lat["t_usec"], 3),
        "latency_at_bytes": lat["size_bytes"],
        "points": len(sub),
    }


def imb_key_line(it: dict) -> str:
    """IMB <key> ：把 4 个数字列换成对某列的排序键（升序）。"""
    return it["value"]


# --------------------------------------------------------------------------- #
# Suite: topo —— 静态拓扑取证
# --------------------------------------------------------------------------- #
# GPU 的 PCI vendor id（Intel）；0x1a03 = ASPEED（服务器 BMC 的 VGA），
# 它也会出现在 /sys/class/drm 里，**不是 GPU**，必须排除。
_PCI_VENDOR_INTEL = "0x8086"
_PCI_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


def _pci_chain(dev_dir: Path) -> list[dict]:
    """从 GPU 的 sysfs device 目录**向上**遍历 PCI 拓扑链。

    本机拓扑是
        root port (3d:02.0) -> switch upstream (3e:00.0)
        -> switch downstream (3f:01.0) -> GPU endpoint (40:00.0)
    ``lspci`` / sysfs 只报 GPU 端点自身的 LnkSta，因此必须爬链才能看到
    上游那一段真实速率。
    """
    chain: list[dict] = []
    cur = dev_dir
    for _ in range(6):
        if not _PCI_BDF_RE.match(cur.name):
            break
        node: dict = {"bdf": cur.name}
        for fn, key in (
            ("vendor", "vendor_id"), ("device", "device_id"),
            ("current_link_speed", "link_speed"),
            ("max_link_speed", "max_link_speed"),
            ("current_link_width", "link_width"),
            ("max_link_width", "max_link_width"),
            ("class", "class"),
        ):
            p = cur / fn
            if p.exists():
                try:
                    node[key] = p.read_text().strip()
                except OSError:
                    pass
        chain.append(node)
        parent = cur.parent
        if not _PCI_BDF_RE.match(parent.name):
            break
        cur = parent
    return chain


def suite_topo(store: ResultStore, env: dict) -> None:
    print("===== topo =====", flush=True)

    # 每张卡的 PCIe / Xe Link 链路信息来自 sysfs
    got_any = False
    gpu_chains: list[dict] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        name = card.parent.name
        if not (card / "vendor").exists():
            continue
        info: dict = {}
        for fn, key in (
            ("vendor", "vendor_id"), ("device", "device_id"),
            ("current_link_speed", "pcie_current_link_speed"),
            ("max_link_speed", "pcie_max_link_speed"),
            ("current_link_width", "pcie_current_link_width"),
            ("max_link_width", "pcie_max_link_width"),
            ("numa_node", "numa_node"),
        ):
            p = card / fn
            if p.exists():
                try:
                    info[key] = p.read_text().strip()
                except OSError:
                    pass

        if info.get("vendor_id") != _PCI_VENDOR_INTEL:
            # 例如 0x1a03 ASPEED 的 BMC VGA —— 出现在 DRM 里但不是 GPU
            store.add("topo", f"{name}.non_gpu",
                      params={"card": name},
                      metrics=info,
                      status="skipped",
                      note="不是 Intel GPU（多为服务器 BMC 的 ASPEED VGA），"
                           "仅记录以便排除")
            continue

        # 爬整条 PCI 链：GPU 端点自己报的链路速率**不可信**（见下 note）
        chain = _pci_chain(Path(os.path.realpath(card)))
        up = next((n for n in chain
                   if str(n.get("link_width")) == "16"
                   and "32" in str(n.get("link_speed", ""))), None)
        info["reported_at_gpu"] = (f"{info.get('pcie_current_link_speed')} "
                                   f"x{info.get('pcie_current_link_width')}")
        info["reported_upstream_max"] = (
            f"{up['link_speed']} x{up['link_width']}" if up else "未找到")
        info["pci_chain_bdfs"] = [n["bdf"] for n in chain]
        info["conflict"] = bool(up is not None
                                and info.get("pcie_current_link_width") == "1")
        got_any = True
        store.add("topo", f"{name}.link", params={"card": name}, metrics=info,
                  note=("⚠️ GPU 端点 sysfs/lspci 报 "
                        f"{info['reported_at_gpu']}，但同一链路上游报 "
                        f"{info['reported_upstream_max']} —— 两者物理上是同一条"
                        "链路，实测 H2D pinned 达 31.9 GB/s（见 "
                        "03-memory-bandwidth），因此**端点上报值不可信**"
                        if info["conflict"] else ""))
        gpu_chains.append({"card": name, "chain": chain})

    if not got_any:
        store.skip("topo", "sysfs_link",
                   "读不到 /sys/class/drm/card*/device 的链路信息")
    else:
        store.add("topo", "pcie_link_report_conflict",
                  params={"source": "/sys/class/drm/card*/device + 上游 PCI 桥"},
                  metrics={"gpu_count": len(gpu_chains),
                           "any_conflict": any(
                               c["chain"] and any(
                                   n.get("link_width") == "1" for n in c["chain"])
                               for c in gpu_chains),
                           "note": "GPU 端点报 Gen1 x1，上游报 Gen5 x16"},
                  note="PCIe 链路速率上报自相矛盾 → 不要用 sysfs/lspci 的 "
                       "LnkSta 判断本机的 host↔device 带宽，**只信实测**")


    # xpu-smi 拓扑（不 dump，dump 会挂死）
    p = run_soft(["xpu-smi", "discovery", "-j"], env=env, cwd=HERE, timeout=60)
    if p.returncode == 0 and p.stdout.strip():
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "xpu_smi_discovery.json").write_text(p.stdout)
        try:
            data = json.loads(p.stdout)
            # ★ 键名是 `device_list`（不是 devices）—— 写错会静默得到
            #   device_count=0，看起来像"没有 GPU"。
            if isinstance(data, dict):
                devs = data.get("device_list") or data.get("devices") or []
            elif isinstance(data, list):
                devs = data
            else:
                devs = []
            store.add("topo", "xpu_smi_discovery",
                      params={"source": "xpu-smi discovery -j"},
                      metrics={"device_count": len(devs),
                               "gpus": [f"{d.get('device_id')}:{d.get('pci_bdf_address')}"
                                        f" {d.get('device_name')}"
                                        for d in devs if isinstance(d, dict)],
                               "raw_kept": "results/xpu_smi_discovery.json"},
                      status="ok" if devs else "error",
                      note="" if devs else "解析不到 device_list —— 键名变了")
        except json.JSONDecodeError as ex:
            store.error("topo", "xpu_smi_discovery",
                        f"xpu-smi discovery -j 输出不是合法 JSON: {ex}")
    else:
        store.skip("topo", "xpu_smi_discovery",
                   f"xpu-smi discovery 不可用 rc={p.returncode}")

    # Xe Link 标定状态（★ 关键证据：本报告所有"低于标称"的解释都靠它）
    # ★ 出处是 `xpu-smi discovery -d 0`（单卡详细模式）；
    #   `xpu-smi discovery`（无 -d）和 `xpu-smi topology -d 0` 都**不含**这一行 ——
    #   用错命令会静默 skip，从而丢掉本测试最重要的一条 caveat。
    p = run_soft(["xpu-smi", "discovery", "-d", "0"], env=env, cwd=HERE, timeout=60)
    txt = (p.stdout or "") + (p.stderr or "")
    if "calibrat" in txt.lower():
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "xpu_smi_discovery_d0.txt").write_text(txt)
        fields = {
            "xe_link_ports": r"Number of Xe Link ports:\s*(\d+)",
            "lanes_per_port": r"Number of Lanes per Xe Link port:\s*(\d+)",
            "mbps_per_port": r"Max Tx/Rx Speed per Xe Link port:\s*([\d.]+)\s*MiB/s",
        }
        m = {}
        for k, rx in fields.items():
            mm = re.search(rx, txt)
            if mm:
                m[k] = int(mm.group(1)) if k != "mbps_per_port" else float(mm.group(1))
        cal = "unknown"
        mm = re.search(r"Xe Link Calibration Date:\s*([^|\n]+)", txt)
        if mm:
            cal = mm.group(1).strip()
        m["calibration_line"] = cal
        m["calibrated"] = "not" not in cal.lower()
        if "mbps_per_port" in m and "xe_link_ports" in m:
            # xpu-smi 用的是 MiB/s，而本项目全篇 GB/s 按 10^9 计
            # （和 docs/interconnect.md 的 318 GB/s 口径一致）：
            #   6 × 50663.95 MiB/s × 2^20 / 10^9 = 318.8 GB/s
            m["spec_gbps_per_dir"] = round(
                m["xe_link_ports"] * m["mbps_per_port"] * 2 ** 20 / 1e9, 1)
        m["raw_kept"] = "results/xpu_smi_discovery_d0.txt"
        store.add("topo", "xelink_calibration",
                  params={"source": "xpu-smi discovery -d 0"},
                  metrics=m,
                  status="ok",
                  note=f"{cal} —— 未标定是解释一切『低于 318 GB/s』的首要前提，"
                       "不是硬件缺陷证据")
    else:
        store.skip("topo", "xelink_calibration",
                   "xpu-smi discovery -d 0 未输出标定信息")


# --------------------------------------------------------------------------- #
# Suite: p2p —— Level Zero 能力 + 卡间带宽
# --------------------------------------------------------------------------- #
def suite_p2p(store: ResultStore, env: dict, max_bytes: int) -> None:
    print("===== p2p =====", flush=True)

    exe = BUILD / "p2p_probe"
    if not exe.exists():
        store.error("p2p", "probe", f"{exe} 不存在，请先编译")
        return

    p = run_soft([str(exe), "bw", str(max_bytes)], env=env, cwd=HERE, timeout=600)
    text = p.stdout or ""
    if p.returncode != 0:
        store.error("p2p", "probe_run",
                    f"p2p_probe 退出码 {p.returncode}: {(p.stderr or '')[-300:]}")
    if not text.strip():
        store.error("p2p", "probe_run", "p2p_probe 无输出")
        return

    info, peers, bws = parse_probe(text)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "p2p_probe_raw.txt").write_text(text)

    # --- 能力 ---
    n_ok = sum(1 for x in peers if x.get("can_access"))
    store.add("p2p", "can_access_peer",
              params={"source": "zeDeviceCanAccessPeer",
                      "method": "Level Zero 原生 API"},
              metrics={"ordered_pairs": len(peers), "accessible_pairs": n_ok,
                       "pair_0_1": next((x.get("can_access") for x in peers
                                         if x.get("src") == 0 and x.get("dst") == 1), None),
                       "pair_1_0": next((x.get("can_access") for x in peers
                                         if x.get("src") == 1 and x.get("dst") == 0), None),
                       "p2p_flags": next((x.get("p2p_flags") for x in peers), [])},
              status="ok" if n_ok else "error",
              note="" if n_ok else "P2P 不可用 → 多卡必然退化到 PCIe/host 中转")

    for x in peers:
        store.add("p2p", f"peer.{x.get('src')}_{x.get('dst')}",
                  params={"src": x.get("src"), "dst": x.get("dst"),
                          "api": "zeDeviceCanAccessPeer + zeDeviceGetP2PProperties"},
                  metrics={"can_access": x.get("can_access"),
                           "p2p_flags": x.get("p2p_flags"),
                           "can_access_rc": x.get("can_access_rc")})

    for d in info.get("devices", []):
        store.add("p2p", f"device.{d.get('index')}",
                  params={"index": d.get("index")},
                  metrics={"name": d.get("name"), "pci_bdf": d.get("pci_bdf"),
                           "vendor_id": d.get("vendor_id"),
                           "device_id": d.get("device_id")})

    # --- 带宽 ---
    if not bws:
        store.skip("p2p", "bandwidth",
                   "P2P 不可用或无带宽数据点（见 can_access_peer）",
                   params={"p2p_available": info.get("p2p_available")})
        return

    for direction in sorted({(b["src"], b["dst"]) for b in bws}):
        s, d = direction
        pts = sorted([b for b in bws if (b["src"], b["dst"]) == direction],
                     key=lambda b: b["size_bytes"])
        for b in pts:
            store.add("p2p", f"bw.{s}->{d}.{b['size_bytes']}",
                      params={"src": s, "dst": d, "size_bytes": b["size_bytes"],
                              "method": "zeCommandListAppendMemoryCopy 跨卡（同 context）"},
                      metrics={"gbps": b["gbps"], "gibps": b["gibps"],
                               "seconds": b["seconds"],
                               "pct_of_xelink_spec": round(
                                   100.0 * b["gbps"] / XE_LINK_SPEC_GBPS, 1),
                               "pct_of_pcie_gen5_x16": round(
                                   100.0 * b["gbps"] / PCIE_GEN5_X16_GBPS, 1)})

        peak = max(pts, key=lambda b: b["gbps"])
        verdict = ("Xe Link 全速" if peak["gbps"] >= TH_XELINK_FULL else
                   "走了 P2P 但远低于 Xe Link 标称" if peak["gbps"] >= TH_XELINK_PARTIAL else
                   "疑似退化/未标定")
        store.add("p2p", f"peak.{s}->{d}",
                  params={"src": s, "dst": d,
                          "method": "max over sizes of zeMemCopy"},
                  metrics={"peak_gbps": round(peak["gbps"], 2),
                           "peak_gibps": round(peak["gibps"], 2),
                           "at_size_bytes": peak["size_bytes"],
                           "xelink_spec_gbps": XE_LINK_SPEC_GBPS,
                           "pct_of_spec": round(100.0 * peak["gbps"] / XE_LINK_SPEC_GBPS, 1),
                           "pct_of_pcie_gen5_x16": round(
                               100.0 * peak["gbps"] / PCIE_GEN5_X16_GBPS, 1),
                           "verdict": verdict},
                  note=f"{verdict}；PCIe Gen5 x16 原始带宽仅 "
                       f"{PCIE_GEN5_X16_GBPS} GB/s，故该值仍高于 PCIe")

    best = max(bws, key=lambda b: b["gbps"])
    store.add("p2p", "peak_overall",
              params={"method": "所有方向/尺寸的最大值"},
              metrics={"gbps": round(best["gbps"], 2),
                       "direction": f"{best['src']}->{best['dst']}",
                       "at_size_bytes": best["size_bytes"],
                       "xelink_spec_gbps": XE_LINK_SPEC_GBPS,
                       "pct_of_spec": round(100.0 * best["gbps"] / XE_LINK_SPEC_GBPS, 1),
                       "ratio_vs_pcie_gen5": round(best["gbps"] / PCIE_GEN5_X16_GBPS, 2)})

    # 大消息是否收敛（收敛 → 数值可信）
    ordered = sorted({(b["src"], b["dst"], b["size_bytes"]) for b in bws},
                     key=lambda t: t[2])
    big = [b["gbps"] for b in bws if b["size_bytes"] >= (16 << 20)]
    if len(big) >= 3:
        spread = (max(big) - min(big)) / max(big) * 100.0
        store.add("p2p", "plateau_convergence",
                  params={"threshold_bytes": 16 << 20},
                  metrics={"samples": len(big), "min_gbps": round(min(big), 2),
                           "max_gbps": round(max(big), 2),
                           "spread_pct": round(spread, 2),
                           "converged": spread < 5.0},
                  note="大消息带宽收敛 → 该值是稳定的链路带宽，不是噪声")


# --------------------------------------------------------------------------- #
# Suite: imb —— GPU-aware MPI
# --------------------------------------------------------------------------- #
def find_mpi_root() -> str:
    base = Path(MPI_BASE)
    if not base.exists():
        return f"{MPI_BASE}/latest"
    cands = sorted([d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit()])
    return str(cands[-1]) if cands else f"{MPI_BASE}/latest"


IMB_BENCHES_P2P = ["PingPong", "PingPing"]
# ★ IMB 的 benchmark 名是**大小写敏感**的，广播叫 `Bcast` 不是 `Broadcast`；
#   写错会被 IMB 打印 "Invalid benchmark name" 并**静默跳过该项**。
#   可用名字用 `IMB-MPI1-GPU -list` 查。
IMB_BENCHES_COLL = ["Allreduce", "Allgather", "Bcast", "Reduce", "Alltoall"]

# ★ 本机实测：不加 I_MPI_OFFLOAD=1，IMB-MPI1-GPU 会在第一个非零消息上
#   触发 GPU PTE 缺页段错误（access: 1 (Write), type: 0 (NotPresent)）而进程被杀。
#   I_MPI_OFFLOAD=1 后全程正常 → 这是**必设**的环境变量，不是可选项。
GPU_MPI_GENV = {
    "I_MPI_OFFLOAD": "1",
    "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1",
}

# PingPong 默认最大只有 4 MiB（远未到链路峰值）→ 拉到 1 GiB
# -MSGLOG min:max 是以 2 为底的指数区间。**必须用冒号**：
# 写成 `-msglog 12 30` 时 IMB 会把 12 当区间、把 30 当成第 3 个 benchmark 名字，
# 打印 "Invalid benchmark name 30" 并**静默退回默认 4 KiB 上限**（已实测踩坑）。
PINGPONG_EXTRA = ["-msglog", "12:30"]
COLL_EXTRA = ["-msglog", "12:28"]


def run_imb(exe: str, benches: list[str], env: dict, tag: str,
            nprocs: int = 2, genvs: dict | None = None,
            extra_args: list[str] | None = None,
            timeout: int = 900) -> tuple[str, list[dict], dict]:
    """跑 IMB 并解析。返回 (raw_text, rows, meta)。"""
    cmd = ["mpirun", "-n", str(nprocs)]
    for k, v in (genvs or {}).items():
        cmd += ["-genv", k, str(v)]
    cmd += [exe] + benches + list(extra_args or [])
    p = run_soft(cmd, env=env, cwd=HERE, timeout=timeout)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    meta, rows = parse_imb(text)
    meta["returncode"] = p.returncode
    meta["command"] = " ".join(cmd)
    meta["crashed"] = ("BAD TERMINATION" in text) or ("Segmentation fault from GPU" in text)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"imb_{tag}.txt").write_text(text)
    return text, rows, meta


def suite_imb(store: ResultStore, env: dict) -> None:
    print("===== imb =====", flush=True)

    mpi_root = find_mpi_root()
    gpu_exe = f"{mpi_root}/bin/IMB-MPI1-GPU"
    cpu_exe = f"{mpi_root}/bin/IMB-MPI1"

    if not Path(gpu_exe).exists():
        store.error("imb", "IMB-MPI1-GPU", f"未找到 {gpu_exe}")
        return

    # GPU-aware 必须保证 rank↔device 一一对应；I_MPI_OFFLOAD=1 是**必需**的
    genv = dict(GPU_MPI_GENV)

    # ---- 反例留档：不加 I_MPI_OFFLOAD=1 会崩 ---------------------------- #
    bad_genv = {"ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1"}
    t_bad, _rows_bad, m_bad = run_imb(gpu_exe, ["PingPong"], env, "gpu_no_offload",
                                      genvs=bad_genv, extra_args=["-msglog", "0:16"],
                                      timeout=300)
    crashed = ("Segmentation fault from GPU" in t_bad) or ("BAD TERMINATION" in t_bad)
    store.add("imb", "env.I_MPI_OFFLOAD_required",
              params={"without": bad_genv, "with": genv},
              metrics={"crashes_without": crashed,
                       "returncode": m_bad.get("returncode"),
                       "evidence": ("Segmentation fault from GPU ... PTE NotPresent Write"
                                    if crashed else "未复现崩溃")},
              status="ok" if crashed else "skipped",
              note="不加 I_MPI_OFFLOAD=1 时 IMB-MPI1-GPU 触发 GPU 缺页段错误"
                   "（access=Write, type=NotPresent）并被 SIGKILL；加后全程正常。"
                   "→ 这是必设环境变量")

    # ---- GPU-aware PingPong / PingPing（拉到大消息） --------------------- #
    text, rows, meta = run_imb(gpu_exe, IMB_BENCHES_P2P, env, "gpu_pingpong",
                               genvs=genv, extra_args=PINGPONG_EXTRA, timeout=900)
    if meta.get("returncode") != 0 and not rows:
        store.error("imb", "gpu_pingpong",
                    f"IMB-MPI1-GPU 失败 rc={meta.get('returncode')}；"
                    f"输出尾部：{text[-400:]}")
    else:
        store.add("imb", "gpu_pingpong_meta",
                  params={"exe": gpu_exe, "command": meta.get("command")},
                  metrics={"benchmarks": meta.get("benchmarks"),
                           "processes": meta.get("processes"),
                           "rows": meta.get("rows")})
        for bench in IMB_BENCHES_P2P:
            sub = [r for r in rows if r["bench"].lower() == bench.lower()]
            for r in sub:
                store.add("imb", f"gpu.{bench}.{r['size_bytes']}",
                          params={"benchmark": bench, "size_bytes": r["size_bytes"],
                                  "processes": r.get("processes"),
                                  "genv": genv,
                                  "path": "GPU-aware MPI over Xe Link"},
                          metrics={"gbps": round(r["gbps"], 4),
                                   "mbps": r["mbps"], "t_usec": r["t_usec"],
                                   "repetitions": r["repetitions"]})
            pk = imb_peak(rows, bench)
            if pk:
                spec = XE_LINK_SPEC_GBPS
                store.add("imb", f"gpu_peak.{bench}",
                          params={"benchmark": bench, "processes": 2},
                          metrics={**pk, "xelink_spec_gbps": spec,
                                   "pct_of_spec": round(100.0 * pk["bandwidth_gbps"] / spec, 1),
                                   "ratio_vs_pcie_gen5": round(
                                       pk["bandwidth_gbps"] / PCIE_GEN5_X16_GBPS, 2),
                                   "verdict": (
                                       "Xe Link" if pk["bandwidth_gbps"] >= TH_XELINK_FULL else
                                       "P2P 生效但未满速" if pk["bandwidth_gbps"] >= TH_PCIE else
                                       "疑似 PCIe/host")},
                          note="PingPong 是单向；判据见 §判读标准")

    # ---- GPU-aware 集合通信 --------------------------------------------- #
    text, rows, meta = run_imb(gpu_exe, IMB_BENCHES_COLL, env, "gpu_coll",
                               genvs=genv, extra_args=COLL_EXTRA,
                               timeout=1800)
    if not rows:
        store.error("imb", "gpu_collectives",
                    f"IMB-MPI1-GPU 集合通信无数据 rc={meta.get('returncode')}；"
                    f"输出尾部：{text[-400:]}")
    else:
        store.add("imb", "gpu_collectives_meta",
                  params={"command": meta.get("command")},
                  metrics={"benchmarks": meta.get("benchmarks"),
                           "rows": meta.get("rows")})
        W = 2
        for bench in IMB_BENCHES_COLL:
            sub = [r for r in rows if r["bench"].lower() == bench.lower()]
            if not sub:
                store.skip("imb", f"gpu.{bench}", "IMB 未产出该 benchmark")
                continue
            # 大消息峰值
            bw = max(sub, key=lambda r: r["gbps"])
            busbw = bw["gbps"] * 2.0 * (W - 1) / W
            lat = min(sub, key=lambda r: r["t_usec"])
            store.add("imb", f"gpu_peak.{bench}",
                      params={"benchmark": bench, "processes": W},
                      metrics={"algbw_gbps": round(bw["gbps"], 3),
                               "busbw_gbps": round(busbw, 3),
                               "at_size_bytes": bw["size_bytes"],
                               "latency_us": round(lat["t_usec"], 3),
                               "latency_at_bytes": lat["size_bytes"],
                               "xelink_spec_gbps": XE_LINK_SPEC_GBPS,
                               "pct_of_spec_busbw": round(100.0 * busbw / XE_LINK_SPEC_GBPS, 1)},
                      note="IMB 报的是 algbw；busbw = algbw × 2(W-1)/W")

    # ---- CPU 侧对照（host 路径 / PCIe 基线） ----------------------------- #
    if Path(cpu_exe).exists():
        text, rows, meta = run_imb(cpu_exe, ["PingPong"], env, "cpu_pingpong",
                                   genvs={}, extra_args=PINGPONG_EXTRA,
                                   timeout=600)
        if rows:
            pk = imb_peak(rows, "PingPong")
            store.add("imb", "cpu_peak.PingPong",
                      params={"exe": cpu_exe, "path": "纯 CPU MPI（同机 shm）",
                              "command": meta.get("command")},
                      metrics={**pk},
                      note="同机共享内存路径 —— 它是 host 内存的上限，"
                           "**不是** PCIe 基线；PCIe 的 H2D/D2H 对照见 "
                           "03-memory-bandwidth")
        else:
            store.skip("imb", "cpu_peak.PingPong",
                       f"IMB-MPI1 无数据 rc={meta.get('returncode')}")
    else:
        store.skip("imb", "cpu_peak.PingPong", f"未找到 {cpu_exe}")

    # ---- libfabric ------------------------------------------------------ #
    # 注意：默认 provider 选择 + shm provider 在本机都会在 pingpong.c:463
    # bind() 返回 -98 (Address already in use) 后**永久挂起**，因此这里
    # 强制 -p shm 并加 60 s 硬超时；拿不到表格就记 skip（不再让整个 run 卡死）。
    fi = f"{mpi_root}/bin/fi_pingpong"
    if not Path(fi).exists():
        store.skip("imb", "fi_pingpong", f"未找到 {fi}")
    else:
        t = ""
        rc = None
        for prov in ("shm", "tcp"):
            p = run_soft(["mpirun", "-n", "2", fi, "-p", prov], env=env,
                         cwd=HERE, timeout=60)
            t = (p.stdout or "") + (p.stderr or "")
            rc = p.returncode
            if re.search(r"bytes\s+\(usec\)", t) and rc == 0:
                break
        (HERE / "results" / "fi_pingpong.txt").write_text(t)
        m = re.search(r"bytes\s+\(usec\)[\s\S]{0,4000}", t)
        if m and rc == 0:
            store.add("imb", "fi_pingpong",
                      params={"exe": fi, "provider": prov},
                      metrics={"returncode": rc, "has_table": True,
                               "tail": t[-300:]},
                      note="libfabric provider=%s；单机双卡主要看延迟量级" % prov)
        else:
            store.skip("imb", "fi_pingpong",
                       f"libfabric provider=shm/tcp 均失败：pingpong.c:463 bind() "
                       f"EADDRINUSE 后挂起（rc={rc}，已 60 s 硬超时）；"
                       f"输出尾部：{t[-200:]}")


# --------------------------------------------------------------------------- #
# Suite: ccl —— torch.distributed (xccl / oneCCL)
# --------------------------------------------------------------------------- #
def suite_ccl(store: ResultStore, env: dict) -> None:
    print("===== ccl =====", flush=True)

    script = PROBES / "torch_ccl.py"
    if not Path(PYTHON).exists():
        store.error("ccl", "torchrun", f"未找到 {PYTHON}")
        return

    # 本机 venv 由 uv --system-site-packages 创建，**不会**复制系统 site-packages
    # 里的 console_scripts，所以 `torchrun` 不存在；直接退回
    # `python -m torch.distributed.run`（等价入口）。
    torchrun = str(Path(PYTHON).parent / "torchrun")
    if Path(torchrun).exists():
        cmd = [torchrun, "--nproc_per_node=2", "--standalone", str(script)]
    else:
        cmd = [PYTHON, "-m", "torch.distributed.run",
               "--nproc_per_node=2", "--standalone", str(script)]

    p = run_soft(cmd, env=env, cwd=HERE, timeout=1800)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "torch_ccl.txt").write_text(text)

    devs = parse_json_lines(text, "CCLDEV")
    ops = parse_json_lines(text, "CCLOP")
    peaks = parse_json_lines(text, "CCLPEAK")
    errs = parse_json_lines(text, "CCLERR")

    if not ops:
        store.error("ccl", "allreduce",
                    f"torchrun 无 CCLOP 输出 rc={p.returncode}；尾部：{text[-600:]}")
    else:
        for d in devs:
            if d.get("rank") == 0:
                store.add("ccl", "device", params={"world_size": d.get("world_size")},
                          metrics={"name": d.get("name"), "torch": d.get("torch"),
                                   "backend": d.get("backend"),
                                   "master_addr": d.get("master_addr")})
        for o in ops:
            if o.get("rank") != 0:
                continue
            store.add("ccl", f"{o['op']}.{o['dtype']}.{o['size_bytes']}",
                      params={"op": o["op"], "dtype": o["dtype"],
                              "size_bytes": o["size_bytes"], "world_size": 2,
                              "backend": "xccl (oneCCL)"},
                      metrics={"algbw_gbps": o["algbw_gbps"],
                               "busbw_gbps": o["busbw_gbps"],
                               "seconds": o["seconds"],
                               "seconds_median": o.get("seconds_median")})
        for pk in peaks:
            store.add("ccl", f"peak.{pk['op']}.{pk['dtype']}",
                      params={"op": pk["op"], "dtype": pk["dtype"]},
                      metrics={"busbw_gbps": pk["busbw_gbps"],
                               "at_size_bytes": pk["size_bytes"],
                               "xelink_spec_gbps": pk["xe_link_spec_gbps"],
                               "pct_of_spec": pk["pct_of_spec"]},
                      note="busbw 是集合通信总线口径")

    for e in errs[:6]:
        store.add("ccl", f"error.{e.get('op', e.get('stage', '?'))}",
                  params=e, status="error", note=str(e.get("note", ""))[:200])


# --------------------------------------------------------------------------- #
# Suite: xelink_telemetry —— 硬件侧取证（本机大概率 N/A，如实记录）
# --------------------------------------------------------------------------- #
def suite_xelink_telemetry(store: ResultStore, env: dict) -> None:
    print("===== xelink_telemetry =====", flush=True)

    exe = BUILD / "p2p_probe"
    if not exe.exists():
        store.skip("xelink_telemetry", "sample", "p2p_probe 不存在")
        return

    # 后台制造 Xe Link 流量
    proc = subprocess.Popen([str(exe), "bw", str(1 << 30)],
                            env=env, cwd=str(HERE),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time as _t
        _t.sleep(6)
        p = run_soft(["xpu-smi", "stats", "-d", "0"], env=env, cwd=HERE, timeout=60)
        text = p.stdout or ""
        (HERE / "results" / "xpu_smi_stats_under_p2p.txt").write_text(text)
        metrics: dict = {"returncode": p.returncode, "raw_tail": text[-400:]}
        for pat, key in (
            (r"Xe Link Throughput\s*\(kB/s\)\s*[:|]\s*(\S+)", "xelink_throughput"),
            (r"GPU Memory Read\s*\(kB/s\)\s*[:|]\s*(\S+)", "mem_read_kbs"),
            (r"GPU Memory Write\s*\(kB/s\)\s*[:|]\s*(\S+)", "mem_write_kbs"),
            (r"GPU Utilization\s*\(%\)\s*[:|]\s*(\S+)", "gpu_util_pct"),
            (r"GPU Frequency\s*\(MHz\)\s*[:|]\s*(\S+)", "gpu_freq_mhz"),
        ):
            m = re.search(pat, text)
            if m:
                metrics[key] = m.group(1)
        avail = "xelink_throughput" in metrics and metrics["xelink_throughput"] not in ("N/A", "")
        store.add("xelink_telemetry", "under_p2p_load",
                  params={"device": 0, "while": "p2p_probe bw (1 GiB)",
                          "source": "xpu-smi stats -d 0"},
                  metrics=metrics,
                  status="ok" if avail else "skipped",
                  note=("硬件侧 Xe Link 计数器可用，可与软件测量对照"
                        if avail else
                        "本机 i915 驱动的 Xe Link Throughput 读 N/A → "
                        "**无法从硬件侧独立取证**，只能以软件测量为准"))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
SUITES = {
    "topo": suite_topo,
    "p2p": suite_p2p,
    "imb": suite_imb,
    "ccl": suite_ccl,
    "xelink_telemetry": suite_xelink_telemetry,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", default=None,
                    help=f"要跑的 suite，默认全部：{list(SUITES)}")
    ap.add_argument("--quick", action="store_true", help="冒烟（减小 P2P 消息上限）")
    ap.add_argument("--tag", default=None, help="结果文件名标签")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="P2P 带宽测试的最大消息字节数")
    args = ap.parse_args()

    tag = args.tag or __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    max_bytes = args.max_bytes or ((64 << 20) if args.quick else (1 << 30))

    # 双卡都要可见：**不要**设 ZE_AFFINITY_MASK 为单卡
    env = sub_env()
    env.pop("ZE_AFFINITY_MASK", None)
    env["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = "1"

    store = ResultStore(tag, outdir=HERE / "results", title=f"04-interconnect-xelink {tag}")

    ok, err = build_probes(env)
    store.add("build", "p2p_probe",
              params={"compiler": CXX, "flags": "-O3 -std=c++17 -lze_loader"},
              metrics={"ok": ok}, status="ok" if ok else "error",
              note=err)
    if not ok:
        print(f"构建失败：{err}", file=sys.stderr)
        return 1

    suites = args.suites or list(SUITES)
    for s in suites:
        if s not in SUITES:
            print(f"未知 suite: {s}", file=sys.stderr)
            return 2

    for s in suites:
        try:
            if s == "p2p":
                suite_p2p(store, env, max_bytes)
            else:
                SUITES[s](store, env)
        except Exception as ex:                                     # noqa: BLE001
            import traceback
            store.error(s, "suite", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-600:]}")
            print(f"[{s}] 异常：{ex}", file=sys.stderr)

    # ---- 汇总判读 ------------------------------------------------------- #
    def find(suite: str, name: str):
        for r in store.results:
            if r.suite == suite and r.name == name:
                return r
        return None

    p2p_ok = find("p2p", "can_access_peer")
    p2p_peak = find("p2p", "peak_overall")
    flat = {}
    flat["GPU"] = GPU_NAME
    flat["Xe Link 标称"] = f"{XE_LINK_SPEC_GBPS} GB/s/方向 (XL24, 6 ports x 4 lanes)"
    flat["PCIe Gen5 x16 原始"] = f"{PCIE_GEN5_X16_GBPS} GB/s"

    if p2p_ok:
        flat["P2P 可用"] = p2p_ok.metrics.get("accessible_pairs")
        flat["P2P 属性"] = p2p_ok.metrics.get("p2p_flags")
    if p2p_peak:
        flat["P2P 实测峰值"] = f"{p2p_peak.metrics.get('gbps')} GB/s " \
                              f"({p2p_peak.metrics.get('direction')})"
        flat["占 Xe Link 标称"] = f"{p2p_peak.metrics.get('pct_of_spec')}%"
        flat["相对 PCIe Gen5"] = f"{p2p_peak.metrics.get('ratio_vs_pcie_gen5')} ×"

    lines = ["## 汇总判读", ""]
    for k, v in flat.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("> 判读纪律：")
    lines.append("> 1. 三条证据链（L0 原生 / GPU-aware MPI / xccl）必须都看；")
    lines.append(">    仅当**三者一致**时才能下结论。")
    lines.append("> 2. 低于 Xe Link 标称 **不一定**是配置问题 —— 本机 "
                 "`Xe Link Calibration: Not Calibrated` 是已知前提。")
    lines.append("> 3. 卡间带宽**高于 PCIe Gen5 x16 (63 GB/s)** 就说明确实没走 PCIe。")
    store.highlight("\n".join(lines))

    jpath, mpath = store.save()
    print(f"wrote {jpath}")
    print(f"wrote {mpath}")
    print(f"records: {len(store.results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
