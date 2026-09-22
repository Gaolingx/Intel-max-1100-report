"""benchmark/common/bench.py

`02-compute-peak` / `03-memory-bandwidth` / `04-interconnect-xelink` 三个目录共用的
结果收集、计时与报告模块。设计上刻意与 `05-ai-dl/xpu_bench/common.py` 保持一致
（同样的 JSON 结构 `{"meta","highlights","results":[...]}`），这样 `results/` 下的
产物可以被同一套思路阅读。

核心概念
--------
* ``TimeStats``  —— 一次重复测量的时间统计（ms）。
* ``BenchResult`` —— 一条测试记录：suite / name / params / stats / metrics / status / note。
* ``ResultStore`` —— 收集记录并落盘为 ``results/bench_<tag>.json`` 与 ``.md``。

风格约定
--------
* 单条记录要么 ``ok``，要么 ``skipped``（环境不支持），要么 ``error``（真的失败）。
  **不允许**把失败静默丢掉——报告里必须能看到"哪些没做/为什么没做"。
* 所有 GPU 计时都必须显式同步（SYCL 用 ``q.wait()``，torch 用 ``torch.xpu.synchronize()``），
  由调用方的 ``fn`` 负责。

环境注意（hwt 这台机器上踩过的坑）
----------------------------------
* ``source /opt/intel/oneapi/setvars.sh`` 返回 rc=3，且在有 ``set -u`` 时直接 exit 1。
  → 脚本里先 ``set +u``；命令行里不要用 ``&&`` 串联。
* ``ZE_AFFINITY_MASK=0`` 用于只占用 GPU0（本机有 GPU0/GPU1 两块 Max 1100）。
* ``xpu-smi dump`` 在这台机器的驱动上是坏的（只输出表头就挂住），要用 ``xpu-smi stats -d 0``。
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 环境常量（与 docs/hardware.md 保持一致）
# --------------------------------------------------------------------------- #
GPU_NAME = "Intel(R) Data Center GPU Max 1100 (PVC)"
EU_PER_DEVICE = 448          # clinfo: Max compute units = 448
LANES_PER_EU = 16            # Xe-HPC vector engine 宽度
FMA_FLOPS_PER_LANE = 2       # 一条 FMA = 2 FLOP
REFERENCE_CLOCK_GHZ = 1.55   # ⚠ 这是驱动/xpu-smi 报告的“配置值”。见 docs/Conclusion/02-compute-peak/
HBM_SPEC_GBPS = 1229.0       # Max 1100 规格：48 GiB HBM2e ≈ 1.23 TB/s
XE_LINK_SPEC_GBPS = 318.0    # Xe Link XL24 实测/规格 ~318 GB/s/方向（见 docs/interconnect.md）

ONEDNN_ALREADY_IMPORTED = "torch" in __import__("sys").modules


def alu_tflops(clock_ghz: float = REFERENCE_CLOCK_GHZ) -> float:
    """标称 FP32 ALU 峰值（TFLOPS）= 448 × 16 × 2 × 1.55 GHz / 1000。"""
    return EU_PER_DEVICE * LANES_PER_EU * FMA_FLOPS_PER_LANE * clock_ghz / 1000.0


def sub_env(**extra: str) -> dict:
    """返回一份干净的子进程环境变量。

    默认把 oneAPI ``setvars.sh`` 加进来（若存在），这样 icpx / mpirun 一定能找到。
    """
    env = dict(os.environ)
    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if setvars.exists() and "ONEAPI_ROOT" not in env:
        # setvars 是 shell 脚本，不能直接 exec；这里只补最关键的 PATH/LD_LIBRARY_PATH
        for p in [
            "/opt/intel/oneapi/compiler/latest/bin",
            "/opt/intel/oneapi/mpi/latest/bin",
            "/opt/intel/oneapi/ccl/latest/bin",
        ]:
            if Path(p).is_dir() and p not in env.get("PATH", ""):
                env["PATH"] = p + os.pathsep + env.get("PATH", "")
        for p in [
            "/opt/intel/oneapi/compiler/latest/lib",
            "/opt/intel/oneapi/mpi/latest/lib",
            "/opt/intel/oneapi/ccl/latest/lib",
        ]:
            if Path(p).is_dir() and p not in env.get("LD_LIBRARY_PATH", ""):
                env["LD_LIBRARY_PATH"] = p + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(
    cmd: str | Sequence[str],
    env: dict | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = 1800,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """跑一条命令并返回 CompletedProcess。``cmd`` 是字符串时用 shell 解析。"""
    shell = isinstance(cmd, str)
    argv = cmd if shell else list(cmd)
    cp = subprocess.run(
        argv,
        shell=shell,
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={cp.returncode}): {cmd}\n"
            f"--- stdout ---\n{cp.stdout[-4000:]}\n--- stderr ---\n{cp.stderr[-4000:]}"
        )
    return cp


def run_soft(cmd, **kw) -> subprocess.CompletedProcess:
    """同 :func:`run`，但不因非零退出码抛异常（用于探测“有没有装/支不支持”）。"""
    kw.setdefault("check", False)
    try:
        return run(cmd, **kw)
    except Exception as exc:  # noqa: BLE001
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
@dataclass
class TimeStats:
    iters: int
    warmup: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @property
    def best_seconds(self) -> float:
        return self.min_ms / 1000.0


def timeit(fn: Callable[[], Any], warmup: int = 3, iters: int = 10) -> TimeStats:
    """对 ``fn`` 重复计时。``fn`` 内部必须自己保证设备同步。"""
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return TimeStats(
        iters=len(samples),
        warmup=warmup,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        std_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    )


# --------------------------------------------------------------------------- #
# 结果收集
# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    suite: str
    name: str
    params: dict = field(default_factory=dict)
    stats: dict | None = None
    metrics: dict = field(default_factory=dict)
    status: str = "ok"          # ok | skipped | error
    note: str = ""


class ResultStore:
    """收集 :class:`BenchResult`，最后统一落盘为 JSON + Markdown。"""

    def __init__(self, tag: str, outdir: str | Path = "results", title: str = ""):
        self.tag = tag
        self.outdir = Path(outdir)
        self.title = title or tag
        self.results: list[BenchResult] = []
        self.highlights: list[str] = []
        # 顺序遍历保持稳定

    # -- 记录 --------------------------------------------------------------- #
    def add(
        self,
        suite: str,
        name: str,
        params: dict | None = None,
        metrics: dict | None = None,
        status: str = "ok",
        note: str = "",
    ) -> BenchResult:
        r = BenchResult(
            suite=suite,
            name=name,
            params=params or {},
            metrics=metrics or {},
            status=status,
            note=note,
        )
        self.results.append(r)
        return r

    def measure(
        self,
        suite: str,
        name: str,
        params: dict | None,
        fn: Callable[[], Any],
        warmup: int = 3,
        iters: int = 10,
        metrics: dict | None = None,
        note: str = "",
    ) -> BenchResult:
        """跑 ``fn`` 若干次并记录时间统计；``fn`` 可返回额外 metrics 字典。"""
        st = timeit(fn, warmup=warmup, iters=iters)
        extra: dict = {}
        for _ in range(2):
            out = fn()
            if isinstance(out, dict):
                extra.update(out)
        r = BenchResult(
            suite=suite,
            name=name,
            params=dict(params or {}),
            stats=asdict(st),
            metrics={**(metrics or {}), **extra},
            status="ok",
            note=note,
        )
        self.results.append(r)
        return r

    def skip(self, suite: str, name: str, note: str, params: dict | None = None) -> BenchResult:
        return self.add(suite, name, params=params, status="skipped", note=note)

    def error(self, suite: str, name: str, note: str, params: dict | None = None) -> BenchResult:
        return self.add(suite, name, params=params, status="error", note=note)

    def highlight(self, text: str) -> None:
        self.highlights.append(text)

    # -- 落盘 --------------------------------------------------------------- #
    def meta(self) -> dict:
        return {
            "tag": self.tag,
            "title": self.title,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "gpu_count": len(os.environ.get("ZE_AFFINITY_MASK", "").split(",")) or 2,
            "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", "(unset)"),
            "nominal_fp32_alu_tflops": alu_tflops(),
        }

    def save(self) -> tuple[Path, Path]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        jpath = self.outdir / f"bench_{self.tag}.json"
        mpath = self.outdir / f"bench_{self.tag}.md"

        payload = {
            "meta": self.meta(),
            "highlights": self.highlights,
            "results": [asdict(r) for r in self.results],
        }
        jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        mpath.write_text(self.to_markdown(), encoding="utf-8")
        return jpath, mpath

    # -- Markdown ------------------------------------------------------------ #
    def to_markdown(self) -> str:
        meta = self.meta()
        out: list[str] = []
        out.append(f"# {self.title}")
        out.append("")
        out.append(f"- tag: `{meta['tag']}`")
        out.append(f"- time: {meta['timestamp']}")
        out.append(f"- host: {meta['host']} / kernel {meta['kernel']} / python {meta['python']}")
        out.append(f"- ZE_AFFINITY_MASK: `{meta['ze_affinity_mask']}`")
        out.append(f"- 标称 FP32 ALU 峰值（公式）: {meta['nominal_fp32_alu_tflops']:.2f} TFLOPS")
        out.append("")

        if self.highlights:
            out.append("## 关键结论速览")
            out.append("")
            for h in self.highlights:
                out.append(f"- {h}")
            out.append("")

        suites: dict[str, list[BenchResult]] = {}
        for r in self.results:
            suites.setdefault(r.suite, []).append(r)

        for suite, rs in suites.items():
            out.append(f"## {suite}")
            out.append("")
            out.append("| 测试项 | 参数 | 均值 | 最好 | 单位 | 派指标 | 状态 | 备注 |")
            out.append("|---|---|---:|---:|---|---|---|---|")
            for r in rs:
                unit, mean_v, best_v = self._flatten(r)
                extra = ", ".join(f"{k}={_fmt(v)}" for k, v in r.metrics.items()) or "-"
                params = ", ".join(f"{k}={_fmt(v)}" for k, v in r.params.items()) or "-"
                note = (r.note or "-").replace("|", "/")
                out.append(
                    f"| `{r.name}` | {params} | {_fmt(mean_v)} | {_fmt(best_v)} | {unit} "
                    f"| {extra} | {r.status} | {note} |"
                )
            out.append("")

        skipped = [r for r in self.results if r.status != "ok"]
        if skipped:
            out.append("## 未完成 / 跳过项")
            out.append("")
            out.append("| suite | 测试项 | 状态 | 原因 |")
            out.append("|---|---|---|---|")
            for r in skipped:
                out.append(f"| {r.suite} | `{r.name}` | {r.status} | {(r.note or '-').replace('|','/')} |")
            out.append("")
        return "\n".join(out)

    @staticmethod
    def _flatten(r: BenchResult) -> tuple[str, float | None, float | None]:
        """从 metrics 里挑一个主指标用于表格展示。"""
        if r.metrics:
            k = next(iter(r.metrics))
            v = r.metrics[k]
            if isinstance(v, (int, float)):
                unit = k.split("__")[-1] if "__" in k else str(k)
                unit = re.sub(r"^.*?_(gbps|gflops|gops|tflops|tops|ms|gb_per_s|gib_per_s|s)$", r"\1", unit)
                return unit, float(v), float(v)
        if r.stats:
            return "ms", r.stats["mean_ms"], r.stats["min_ms"]
        return "-", None, None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        if v != 0 and abs(v) < 0.01:
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def parse_json_lines(text: str, prefix: str) -> list[dict]:
    """从工具输出里抽取 ``PREFIX {...}`` 形式的 JSON 行。"""
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            try:
                out.append(json.loads(line[len(prefix):].strip()))
            except json.JSONDecodeError:
                pass
    return out


def gbps(num_bytes: float, seconds: float) -> float:
    """bytes / s  ->  GB/s (10^9)。"""
    return num_bytes / seconds / 1e9


def gibps(num_bytes: float, seconds: float) -> float:
    """bytes / s  ->  GiB/s (2^30)。"""
    return num_bytes / seconds / (1 << 30)


__all__ = [
    "GPU_NAME",
    "EU_PER_DEVICE",
    "LANES_PER_EU",
    "FMA_FLOPS_PER_LANE",
    "REFERENCE_CLOCK_GHZ",
    "HBM_SPEC_GBPS",
    "XE_LINK_SPEC_GBPS",
    "alu_tflops",
    "sub_env",
    "run",
    "run_soft",
    "TimeStats",
    "timeit",
    "BenchResult",
    "ResultStore",
    "parse_json_lines",
    "gbps",
    "gibps",
]
