"""XPU 理论性能基准测试工具包（benchmark/05-ai-dl）。

对应文档：
* ``docs/TODO/02-compute-peak.md``     算力峰值（GEMM sweep）
* ``docs/TODO/03-memory-bandwidth.md`` 显存带宽（elementwise / copy）
* ``docs/TODO/05-ai-dl.md``            AI 算子级 micro-benchmark

用法::

    python run_bench.py --suite all
"""

from .common import (
    DEVICE_TYPE,
    DTYPES,
    BenchResult,
    ResultStore,
    TimeStats,
    benchmark,
    device_info,
    environment_meta,
    theoretical_tflops,
)

__version__ = "0.1.0"

__all__ = [
    "DEVICE_TYPE",
    "DTYPES",
    "BenchResult",
    "ResultStore",
    "TimeStats",
    "benchmark",
    "device_info",
    "environment_meta",
    "theoretical_tflops",
    "__version__",
]
