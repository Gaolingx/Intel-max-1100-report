"""公共工具：设备信息、理论峰值、精确计时、结果收集与落盘。

本模块是整个 XPU 理论性能基准测试的工具底座，被各 suite 复用。
设计要点（对应 docs/TODO/05-ai-dl.md 第 6 节「注意事项」）：

* 计时前必须 ``torch.xpu.synchronize()``，否则测到的是异步 launch 时间；
* 每个测试点都做 warmup，排除首次 kernel 的 JIT/编译开销；
* 使用 ``torch.xpu.Event`` 做设备侧计时，避免 host 侧抖动；
* 结果统一结构化为 :class:`BenchResult`，可一键落 JSON / Markdown。
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEVICE_TYPE = "xpu"

# Intel Data Center GPU Max 1100 / Ponte Vecchio（本机锁定 1550 MHz）
EU_PER_DEVICE = 448          # 56 Xe-core × 8 EU
LANES_PER_EU = 16            # SIMD16
FMA_FLOPS_PER_LANE = 2       # 1 次 FMA = 2 FLOP
REFERENCE_CLOCK_GHZ = 1.55   # 本机频率 min == max == 1550 MHz

# FP64 ALU 相对 FP32 的实测系数（PVC 非 1/2！）
# 实测 17.37 TFLOPS @2048³ / 22.16 @16384³ vs FP32 22.16 → 0.78
# 见 docs/precision-support.md §3.1
FP64_ALU_RATIO = 0.78

# dtype 关键词 -> torch dtype
DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ---------------------------------------------------------------------------
# 理论峰值
# ---------------------------------------------------------------------------
def alu_tflops(clock_ghz: float = REFERENCE_CLOCK_GHZ) -> float:
    """ALU（非 XMX）FP32 峰值，单位 TFLOPS。

    448 EU × 16 lane × 2 FLOP/FMA × 1.55 GHz ≈ 22.2 TFLOPS
    """
    return EU_PER_DEVICE * LANES_PER_EU * FMA_FLOPS_PER_LANE * clock_ghz / 1e3


def theoretical_tflops(dtype_key: str, clock_ghz: float = REFERENCE_CLOCK_GHZ) -> Optional[float]:
    """返回 dtype 的理论峰值 (TFLOPS)。

    FP32 可由公式推导；FP16 / BF16 / INT8 走 XMX，倍数必须实测，
    因此这里返回 ``None``（不要在报告中当作结论引用）。

    ⚠ FP64：Max 1100（PVC）的 FP64 并非 FP32 的 1/2。实测 17.37 TFLOPS
    （@2048³）= FP32 的 **0.78×**，明显高于 ``alu / 2``。这里改用实测系数
    ``FP64_ALU_RATIO``，避免报告中出现错误的达成率。
    详见 docs/precision-support.md §3.1。
    """
    alu = alu_tflops(clock_ghz)
    return {"fp32": alu, "fp64": alu * FP64_ALU_RATIO}.get(dtype_key)


def bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# 设备信息
# ---------------------------------------------------------------------------
def device_info(device_index: int = 0) -> dict:
    """采集单卡硬件信息（写入报告开头，便于复现）。"""
    props = torch.xpu.get_device_properties(device_index)
    # 注意：total_memory / last_level_cache_size 的单位是**字节**
    total_mem_bytes = getattr(props, "total_memory", 0) or 0
    llc_bytes = getattr(props, "last_level_cache_size", 0) or 0
    return {
        "device_index": device_index,
        "name": props.name,
        "driver_version": getattr(props, "driver_version", None),
        "device_count": torch.xpu.device_count(),
        "total_memory_gib": round(total_mem_bytes / (1024.0 ** 3), 2),
        "max_compute_units": getattr(props, "max_compute_units", None),
        "gpu_eu_count": getattr(props, "gpu_eu_count", None),
        "gpu_subslice_count": getattr(props, "gpu_subslice_count", None),
        "last_level_cache_mb": round(llc_bytes / (1024.0 ** 2), 2),
        "memory_clock_mhz": getattr(props, "memory_clock_rate", None),
        "has_fp64": bool(getattr(props, "has_fp64", 0)),
        "has_fp16": bool(getattr(props, "has_fp16", 0)),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "oneapi_reference_clock_ghz": REFERENCE_CLOCK_GHZ,
    }


def environment_meta(device_index: int = 0, args: Optional[dict] = None) -> dict:
    """报告头部元信息。"""
    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device": device_info(device_index),
        "arguments": args or {},
    }
    try:
        triton_ver = __import__("triton").__version__
    except Exception:
        triton_ver = None
    meta["triton_version"] = triton_ver
    return meta


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------
@dataclass
class TimeStats:
    """一次 benchmark 的计时统计（单位均为毫秒）。"""

    iters: int
    warmup: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @property
    def best_ms(self) -> float:
        return self.min_ms


def benchmark(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 20,
    device_index: int = 0,
) -> TimeStats:
    """对 ``fn`` 做 warmup + 逐次设备侧计时，返回统计量。

    每次迭代单独记录一对 Event，取中位数作为代表值（比 mean 更抗抖动）。
    """
    for _ in range(max(0, warmup)):
        fn()
    torch.xpu.synchronize(device_index)

    samples: list[float] = []
    for _ in range(max(1, iters)):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize(device_index)
        samples.append(float(start.elapsed_time(end)))

    return TimeStats(
        iters=len(samples),
        warmup=warmup,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        std_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# 派生指标
# ---------------------------------------------------------------------------
def tflops_from(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def gbps_from(nbytes: float, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def gops_from(ops: float, ms: float) -> float:
    return ops / (ms * 1e-3) / 1e9


def numel_for_bytes(target_bytes: int, dtype: torch.dtype) -> int:
    """按目标字节数计算元素个数（至少 1）。"""
    esize = bytes_per_element(dtype)
    return max(1, int(target_bytes) // esize)


def mebibytes(nbytes: float) -> float:
    return nbytes / (1 << 20)


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    """单条测试结果。"""

    suite: str
    name: str
    params: dict = field(default_factory=dict)
    stats: Optional[TimeStats] = None
    metrics: dict = field(default_factory=dict)
    status: str = "ok"           # ok | skipped | error
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


class ResultStore:
    """收集所有 suite 的结果，并负责 JSON / Markdown 输出。"""

    def __init__(self, meta: Optional[dict] = None) -> None:
        self.meta: dict = meta or {}
        self.results: list[BenchResult] = []
        self.notes: list[str] = []

    # -- 记录 ---------------------------------------------------------------
    def add(self, result: BenchResult) -> BenchResult:
        self.results.append(result)
        return result

    def add_note(self, text: str) -> None:
        self.notes.append(text)

    def skip(self, suite: str, name: str, params: dict, reason: str) -> BenchResult:
        return self.add(
            BenchResult(suite=suite, name=name, params=params, status="skipped", note=reason)
        )

    def error(self, suite: str, name: str, params: dict, exc: BaseException) -> BenchResult:
        msg = f"{type(exc).__name__}: {exc}"
        return self.add(
            BenchResult(suite=suite, name=name, params=params, status="error", note=msg[:300])
        )

    def measure(
        self,
        suite: str,
        name: str,
        params: dict,
        fn: Callable[[], Any],
        *,
        warmup: int,
        iters: int,
        device_index: int,
        compute_metrics: Optional[Callable[[TimeStats], dict]] = None,
        note: str = "",
    ) -> Optional[TimeStats]:
        """执行一次 benchmark 并登记结果；异常自动降级为 error 记录。"""
        try:
            stats = benchmark(fn, warmup=warmup, iters=iters, device_index=device_index)
        except Exception as exc:  # noqa: BLE001 - 需要把任意后端错误记录进报告
            self.error(suite, name, params, exc)
            return None
        metrics = compute_metrics(stats) if compute_metrics else {}
        self.add(
            BenchResult(
                suite=suite,
                name=name,
                params=params,
                stats=stats,
                metrics=metrics,
                note=note,
            )
        )
        return stats

    # -- 查询 ---------------------------------------------------------------
    def suite(self, name: str) -> list[BenchResult]:
        return [r for r in self.results if r.suite == name and r.status == "ok"]

    def best(self, suite: str, metric: str) -> Optional[BenchResult]:
        rows = [r for r in self.suite(suite) if metric in r.metrics]
        return max(rows, key=lambda r: r.metrics[metric], default=None)

    # -- 输出 ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "highlights": self.notes,
            "results": [r.as_dict() for r in self.results],
        }

    def dump_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    def dump_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        from .report import render_markdown

        return render_markdown(self)


# ---------------------------------------------------------------------------
# 可选：xpu-smi 遥测
# ---------------------------------------------------------------------------
class TelemetrySampler:
    """后台运行 ``xpu-smi dump`` 采样功耗/频率/利用率，落盘为 JSON。

    仅作辅助证据使用；若环境没有 xpu-smi 则静默禁用。
    """

    def __init__(
        self,
        device_index: int = 0,
        path: str | Path = "telemetry.json",
        interval_ms: int = 500,
        metrics: str = "0,1,2,3,5,6,7,8,9",
    ) -> None:
        self.device_index = device_index
        self.path = Path(path)
        self.interval_ms = interval_ms
        self.metrics = metrics
        self._proc: Optional[subprocess.Popen] = None
        self.available = self._probe()

    @staticmethod
    def _probe() -> bool:
        from shutil import which

        return which("xpu-smi") is not None

    def start(self) -> None:
        if not self.available or self._proc is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "xpu-smi", "dump",
            "-d", str(self.device_index),
            "-m", self.metrics,
            "-i", str(self.interval_ms),
            "-n", "100000",
            "-j",
        ]
        try:
            self._handle = open(self.path, "w")  # noqa: SIM115 - 生命周期由 stop() 管理
            self._proc = subprocess.Popen(cmd, stdout=self._handle, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            self.available = False
            self._proc = None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None


# ---------------------------------------------------------------------------
# 采集耗时的简单工具
# ---------------------------------------------------------------------------
class Stopwatch:
    """wall-clock 秒表（用于 suite 级别耗时统计）。"""

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0
