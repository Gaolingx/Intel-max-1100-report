"""数据管线 / 主机侧瓶颈分析 —— 对应 ``docs/TODO/05-ai-dl.md`` §3.6。

本机最大隐患是**内存倒挂**：主机仅 45 GiB DRAM，而两卡合计 96 GiB HBM。
DataLoader worker 数、``pin_memory``、host→device 拷贝、CPU 预处理都可能
让 GPU 空转。本 suite 把这些环节单独量化。

====================  ==================================================
suite                 内容
====================  ==================================================
``pipeline``          DataLoader worker sweep（0/1/2/4/8）× pin_memory；
                      端到端 step（取数 + H2D + GPU 计算）的 GPU 空闲占比；
                      host CPU 利用率；pageable vs pinned 的 H2D 耗时
====================  ==================================================

指标口径
--------
* ``batch_per_s`` / ``img_per_s``：wall-clock 吞吐（含主机侧全部开销）。
* ``gpu_busy_pct``：GPU 事件计时之和 / wall-clock → **GPU 利用率估计**。
  < 80% 即说明主机侧（取数 / 预处理 / 拷贝）是瓶颈。
* ``cpu_pct``：进程级 CPU 利用率（多线程累加，> 100% 属正常）。
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from .common import DEVICE_TYPE, BenchResult, ResultStore

SUITE = "pipeline"

DEFAULT_WORKERS = [0, 1, 2, 4, 8]
IMG_SHAPE = (3, 224, 224)
DEFAULT_IMG_BYTES = 3 * 224 * 224  # uint8


# ---------------------------------------------------------------------------
# CPU 利用率采样
# ---------------------------------------------------------------------------
def _proc_cpu_seconds() -> float:
    """进程累计 CPU 时间（秒，含所有线程）。"""
    t = os.times()
    return t.user + t.system


class _CpuSampler:
    """在代码块执行期间估算进程 CPU 利用率（可 > 100%，多线程累加）。"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._c0 = _proc_cpu_seconds()
        return self

    def __exit__(self, *exc) -> None:
        self.wall = time.perf_counter() - self._t0
        self.cpu = _proc_cpu_seconds() - self._c0
        self.pct = 100.0 * self.cpu / self.wall if self.wall > 0 else 0.0


# ---------------------------------------------------------------------------
# 合成数据集（纯 CPU 开销，避免依赖真实图片）
# ---------------------------------------------------------------------------
def _make_dataset(n_items: int, cpu_scale: int):
    """构造一个可调 CPU 开销的合成数据集（返回 uint8 CHW 张量）。

    ``cpu_scale`` 越大，单样本的 CPU 预处理越重，用于模拟真实数据增强。
    """
    import numpy as np

    class _SyntheticImages(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.n = n_items
            # 预生成一次性缓冲，令 __getitem__ 的随机读 + 变换成为主要开销
            self._buf = np.random.randint(
                0, 256, size=(n_items, *IMG_SHAPE), dtype=np.uint8
            )

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            img = self._buf[idx]
            if cpu_scale > 0:
                # 模拟归一化 + 随机裁剪 + 缩放（numpy 侧纯 CPU）
                acc = img.astype(np.float32)
                for _ in range(cpu_scale):
                    acc = acc / 255.0
                    acc = acc * 0.9 + 0.05
                img = (acc * 255.0).astype(np.uint8)
            return torch.from_numpy(img.copy()), int(idx % 1000)

    return _SyntheticImages()


def _dataloader(ds, batch: int, workers: int, pin: bool, shuffle: bool = False):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=(2 if workers > 0 else None),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# §3.6-a  DataLoader worker sweep
# ---------------------------------------------------------------------------
def sweep_workers(store: ResultStore, ns) -> None:
    dev = int(getattr(ns, "device", 0))
    batch = int(getattr(ns, "pipe_batch", 64))
    workers_list = [int(w) for w in _int_list(ns, "pipe_workers", DEFAULT_WORKERS)]
    cpu_scale = int(getattr(ns, "pipe_cpu_scale", 3))
    n_batches = int(getattr(ns, "pipe_batches", 20))

    ds = _make_dataset(n_items=batch * (n_batches + 4), cpu_scale=cpu_scale)

    for pin in (False, True):
        for workers in workers_list:
            params = {"batch": batch, "num_workers": workers, "pin_memory": pin,
                      "cpu_scale": cpu_scale, "img_shape": f"{IMG_SHAPE}",
                      "dataset": "synthetic-uint8"}
            try:
                dl = _dataloader(ds, batch, workers, pin)
                # 预热一个 batch（含 worker 启动）
                it = iter(dl)
                next(it)
                del it
                with _CpuSampler() as cpu:
                    t0 = time.perf_counter()
                    n = 0
                    for _ in dl:
                        n += 1
                        if n >= n_batches:
                            break
                    wall = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                store.error(SUITE, "dataloader", params, exc)
                continue

            if n == 0 or wall <= 0:
                store.skip(SUITE, "dataloader", params, "未取到有效 batch")
                continue

            nb = min(n, n_batches)
            store.add(BenchResult(
                suite=SUITE, name="dataloader", params=params, stats=None,
                metrics={
                    "batch_per_s": nb / wall,
                    "img_per_s": nb * batch / wall,
                    "ms_per_batch": 1000.0 * wall / nb,
                    "cpu_pct": cpu.pct,
                },
                note="仅取数（不做 H2D），衡量主机侧上限",
            ))

    del ds
    _pipeline_summary(store, batch)


def _int_list(ns, attr: str, default: list[int]) -> list[int]:
    raw = getattr(ns, attr, None)
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in raw]


def _pipeline_summary(store: ResultStore, batch: int) -> None:
    """从 dataloader sweep 找出吞吐饱和点，以及 pin_memory 的收益。"""
    rows = [r for r in store.suite(SUITE) if r.name == "dataloader"]
    if not rows:
        return
    best = max(rows, key=lambda r: r.metrics.get("img_per_s") or 0.0)
    best_ips = best.metrics["img_per_s"]
    store.add_note(
        f"DataLoader 主机侧上限 **{best_ips:.0f} img/s**"
        f"（num_workers={best.params.get('num_workers')}, "
        f"pin_memory={best.params.get('pin_memory')}, "
        f"cpu_scale={best.params.get('cpu_scale')}）"
    )

    # 饱和点：吞吐达到峰值的 95% 所需的最少 worker 数
    for pin in (False, True):
        seq = sorted((r for r in rows if r.params.get("pin_memory") is pin),
                     key=lambda r: r.params.get("num_workers", 0))
        if not seq:
            continue
        peak = max(r.metrics.get("img_per_s") or 0.0 for r in seq)
        sat = next((r for r in seq
                    if (r.metrics.get("img_per_s") or 0.0) >= 0.95 * peak), None)
        if sat is not None:
            store.add_note(
                f"num_workers={sat.params.get('num_workers')} 时取数吞吐已到"
                f" 95% 峰值（pin_memory={pin}）→ 再增加 worker 收益很小"
            )

    # 单 worker 时 CPU 占用，用于判断 host 是否已被打满
    one = [r for r in rows if r.params.get("num_workers") == 1]
    if one:
        cpu = max(r.metrics.get("cpu_pct") or 0.0 for r in one)
        store.add_note(
            f"单 worker 时主机 CPU 占用约 {cpu:.0f}%（多进程时该值会按 worker 数抬升）"
        )


# ---------------------------------------------------------------------------
# §3.6-b  端到端 step：取数 + H2D + GPU 计算，测 GPU 空闲占比
# ---------------------------------------------------------------------------
def sweep_end_to_end(store: ResultStore, ns) -> None:
    dev = int(getattr(ns, "device", 0))
    device = torch.device(DEVICE_TYPE, dev)
    batch = int(getattr(ns, "pipe_batch", 64))
    cpu_scale = int(getattr(ns, "pipe_cpu_scale", 3))
    n_batches = int(getattr(ns, "pipe_batches", 20))
    iters = int(getattr(ns, "pipe_e2e_batches", 20))

    # 用一层 conv 模拟「GPU 计算」，其耗时相对取数是小的 → 放大主机瓶颈
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1).to(device).to(torch.bfloat16)
    ds = _make_dataset(n_items=batch * (iters + 4), cpu_scale=cpu_scale)

    workers_sweep = _int_list(ns, "pipe_workers", DEFAULT_WORKERS)

    for workers in workers_sweep:
        for pin in (True,):
            params = {"batch": batch, "num_workers": workers, "pin_memory": pin,
                      "cpu_scale": cpu_scale, "gpu_op": "conv2d 3->32 bf16"}
            gpu_ms = 0.0
            h2d_ms = 0.0
            n = 0
            try:
                dl = _dataloader(ds, batch, workers, pin)
                it = iter(dl)
                x_cpu, _ = next(it)  # 预热（保留 CPU 上的副本）
                x_dev = x_cpu.to(device, non_blocking=pin).to(torch.bfloat16)
                conv(x_dev)
                torch.xpu.synchronize(dev)
                del x_dev, it

                # H2D 单 batch 耗时单独测（小包会被 PCIe 延迟主导），取 5 次最小值
                best_h2d = float("inf")
                for _ in range(5):
                    hb0 = torch.xpu.Event(enable_timing=True)
                    hb1 = torch.xpu.Event(enable_timing=True)
                    hb0.record()
                    _tmp = x_cpu.to(device, non_blocking=pin)
                    hb1.record()
                    torch.xpu.synchronize(dev)
                    best_h2d = min(best_h2d, hb0.elapsed_time(hb1))
                    del _tmp
                h2d_ms = best_h2d
                del x_cpu

                with _CpuSampler() as cpu:
                    t0 = time.perf_counter()
                    for x, _ in dl:
                        e0 = torch.xpu.Event(enable_timing=True)
                        e1 = torch.xpu.Event(enable_timing=True)
                        e0.record()
                        xb = x.to(device, non_blocking=pin).to(torch.bfloat16)
                        conv(xb)
                        e1.record()
                        torch.xpu.synchronize(dev)
                        gpu_ms += e0.elapsed_time(e1)
                        n += 1
                        if n >= iters:
                            break
                    wall = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                store.error(SUITE, "end_to_end", params, exc)
                continue

            if n == 0 or wall <= 0:
                store.skip(SUITE, "end_to_end", params, "未取到有效 batch")
                continue

            wall_ms = wall * 1e3
            n = min(n, iters)
            e2e_ms = wall_ms / n
            gpu_per_batch = gpu_ms / n
            store.add(BenchResult(
                suite=SUITE, name="end_to_end", params=params, stats=None,
                metrics={
                    "ms_per_batch": e2e_ms,
                    "batch_per_s": 1000.0 / e2e_ms if e2e_ms > 0 else 0.0,
                    "img_per_s": batch / (e2e_ms * 1e-3),
                    "gpu_ms_per_batch": gpu_per_batch,
                    "gpu_busy_pct": 100.0 * gpu_per_batch / e2e_ms,
                    "cpu_pct": cpu.pct,
                    "h2d_ms_per_batch": h2d_ms,
                },
                note="GPU 事件计时 / wall-clock = GPU 利用率；<80% 即主机侧瓶颈；"
                     "h2d_ms 为单独测的单 batch H2D 耗时",
            ))
            torch.xpu.empty_cache()

    del ds, conv
    torch.xpu.empty_cache()
    _e2e_summary(store, batch)


def _e2e_summary(store: ResultStore, batch: int) -> None:
    rows = [r for r in store.suite(SUITE) if r.name == "end_to_end"]
    if not rows:
        return
    worst = min(rows, key=lambda r: r.metrics.get("gpu_busy_pct") or 1e9)
    store.add_note(
        f"端到端 GPU 利用率最低 {worst.metrics['gpu_busy_pct']:.1f}%"
        f"（num_workers={worst.params.get('num_workers')}, "
        f"cpu_scale={worst.params.get('cpu_scale')}）"
        f"→ 主机侧（取数/预处理）是瓶颈"
    )
    best = max(rows, key=lambda r: r.metrics.get("gpu_busy_pct") or 0)
    store.add_note(
        f"端到端 GPU 利用率最高 {best.metrics['gpu_busy_pct']:.1f}%"
        f"（num_workers={best.params.get('num_workers')}）"
    )


# ---------------------------------------------------------------------------
# §3.6-c  H2D 传输：pageable vs pinned（逐 batch 粒度）
# ---------------------------------------------------------------------------
def sweep_transfer(store: ResultStore, ns) -> None:
    dev = int(getattr(ns, "device", 0))
    device = torch.device(DEVICE_TYPE, dev)
    batch = int(getattr(ns, "pipe_batch", 64))
    iters = int(getattr(ns, "iters", 20))
    warmup = int(getattr(ns, "warmup", 5))

    nbytes = batch * DEFAULT_IMG_BYTES  # uint8
    src = torch.randint(0, 255, (batch, *IMG_SHAPE), dtype=torch.uint8)

    for pin in (False, True):
        params = {"batch": batch, "pin_memory": pin, "dtype": "uint8",
                  "bytes": nbytes, "op": "host->device"}
        try:
            if pin:
                pinned = src.pin_memory()
                get_src = lambda: pinned  # noqa: E731
            else:
                get_src = lambda: src  # noqa: E731
            # warmup
            for _ in range(warmup):
                d = get_src().to(device, non_blocking=pin)
                del d
                torch.xpu.synchronize(dev)

            total = 0.0
            for _ in range(iters):
                torch.xpu.synchronize(dev)
                e0 = torch.xpu.Event(enable_timing=True)
                e1 = torch.xpu.Event(enable_timing=True)
                e0.record()
                d = get_src().to(device, non_blocking=pin)
                e1.record()
                torch.xpu.synchronize(dev)
                total += e0.elapsed_time(e1)
                del d
            ms = total / iters
        except Exception as exc:  # noqa: BLE001
            store.error(SUITE, "h2d_batch", params, exc)
            continue

        store.add(BenchResult(
            suite=SUITE, name="h2d_batch", params=params, stats=None,
            metrics={
                "ms_per_batch": ms,
                "gbps": nbytes / (ms * 1e-3) / 1e9,
                "batch_per_s": 1000.0 / ms if ms > 0 else 0.0,
                "img_per_s": batch * 1000.0 / ms if ms > 0 else 0.0,
            },
            note="单 batch 粒度 H2D；小包传输会被 PCIe 延迟主导，带宽低于大块测试",
        ))

    del src
    torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
def run(ns, store: ResultStore) -> None:
    """数据管线 suite 入口（§3.6）。"""
    sweep_workers(store, ns)
    sweep_end_to_end(store, ns)
    sweep_transfer(store, ns)
