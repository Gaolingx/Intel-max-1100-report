# 软件栈盘点

## 1. 管理层 / 运行时

| 组件 | 版本 | 备注 |
|---|---|---|
| **xpu-smi** | `1.2.43.20260506`（CLI 与 Service 同版本） | Intel XPU System Management Interface |
| **Level Zero** | **1.24.0** | xpu-smi 依赖 Level Zero |
| **SYCL / Unified Runtime** | `1.6.33578+77`，GPU 侧版本 `12.60.7` | |
| **OpenCL** | NEO `25.18.33578`，OpenCL 3.0 | |
| 内核驱动 | i915 `I915_25.2.57_PSB_250224.65` | |
| 驱动包 | `1.25.2.57.250224.65+i75-1` | 来自 `/root/intel-gpu-ubuntu-noble-2523.run` |

### 设备枚举结果

```
# sycl-ls
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero,
    Intel(R) Data Center GPU Max 1100 12.60.7 [1.6.33578+77]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero,
    Intel(R) Data Center GPU Max 1100 12.60.7 [1.6.33578+77]
[opencl:cpu][opencl:0] Intel(R) OpenCL, Genuine Intel(R) 0000 OpenCL 3.0 (Build 0) [2026.21.7.0.24_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, ... Max 1100 OpenCL 3.0 NEO [25.18.33578]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, ... Max 1100 OpenCL 3.0 NEO [25.18.33578]
```

```
# clinfo -l
Platform #0: Intel(R) OpenCL Graphics
 +-- Device #0: Intel(R) Data Center GPU Max 1100
 `-- Device #1: Intel(R) Data Center GPU Max 1100
```

✅ SYCL / OpenCL / Level Zero 三条路径均能正确枚举 2 张 GPU。

---

## 2. oneAPI 工具套件

> **注意：机器上多个大版本并存**，`latest` 软链接指向版本如下。

| 组件 | `latest` 指向 | 其他已装版本 | 用途 |
|---|---|---|---|
| **compiler** (DPC++/icpx) | **2026.1** | 2025.3, 2026.0 | SYCL/OpenCL/C++ 编译 |
| **mkl** | **2026.1** | 2026.0 | 数学库（GEMM/BLAS/FFT），含 GPU offload |
| **dnnl** (oneDNN) | **2026.0** | — | 深度学习原语（卷积/GEMM/attention） |
| **dal** (oneDAL) | **2026.1** | — | 传统机器学习 / 数据分析 |
| **ccl** (oneCCL) | **2022.1** | — | 集合通信（allreduce 等），PyTorch DDP 后端 |
| **mpi** (Intel MPI) | **2021.18** | 2021.17 | MPI，含 GPU-aware |
| **vtune** | **2026.4** | — | Profiler（GPU Roofline 等） |
| **advisor** | **2026.0** | — | Offload 建模 / 向量化建议 |
| **pti** | **1.1** | — | Profiling Tools Interfaces |
| 其他 | `tbb`, `ipp`, `ippcp`, `dpl`, `umf`, `ishmem`, `tcm`, `debugger`, `dev-utilities`, `oneapi-hpc-toolkit`, `deep-learning-essentials`, `licensing`, `common` | | |

环境加载脚本：`/opt/intel/oneapi/setvars.sh`

---

## 3. Python AI 栈

| 包 | 版本 | 备注 |
|---|---|---|
| Python | **3.13.3** | 系统 python3 |
| **torch** | **2.14.0+xpu** | ✅ XPU 构建 |
| **torchvision** | **0.29.0+xpu** | ✅ |
| **torchaudio** | **2.11.0+xpu** | ✅ |
| **triton-xpu** | **3.8.0** | ✅ 可写自定义 kernel |
| numpy | 2.5.2 | |
| intel-sycl-rt | 2026.1.0 | |
| intel-opencl-rt | 2026.1.0 | |
| intel-cmplr-lib-rt / -ur / -lic-rt | 2026.1.0 | |
| intel-openmp | 2026.1.0 | |
| intel-pti | 1.0.1 | |

### PyTorch XPU 验证

```python
>>> import torch
>>> torch.__version__
'2.14.0+xpu'
>>> torch.xpu.is_available()
True
>>> torch.xpu.device_count()
2
>>> [torch.xpu.get_device_name(i) for i in range(2)]
['Intel(R) Data Center GPU Max 1100', 'Intel(R) Data Center GPU Max 1100']
```

✅ **PyTorch 可直接开跑，无需额外安装。**

---

## 4. 已装工具 / 缺失工具盘点

### ✅ 开箱即用

| 工具 | 路径 | 用途 |
|---|---|---|
| `xpu-smi` | `/usr/bin/xpu-smi` | 健康、诊断、遥测、配置、压力测试 |
| `intel_gpu_top` | `/usr/bin/intel_gpu_top` | 实时 GPU 利用率/频率监控 |
| `IMB-MPI1-GPU` | `/opt/intel/oneapi/mpi/latest/bin/` | **GPU-aware MPI 集合通信基准**（现成可用！） |
| `IMB-RMA-GPU` | 同上 | GPU-aware RMA 通信基准 |
| `IMB-MPI1` / `IMB-NBC` / `IMB-P2P` / `IMB-MT` / `IMB-RMA` | 同上 | CPU 侧 MPI 基准 |
| `fi_pingpong` | `/opt/intel/oneapi/mpi/latest/bin/` | libfabric 延迟/带宽 |
| `stream` | `/usr/bin/stream` | 主机内存带宽（CPU 侧） |
| `mem_bench` | VTune / Advisor 目录 | 内存带宽微基准 |
| `vtune` | `/opt/intel/oneapi/vtune/latest/bin64/vtune` | Profiler |
| `advisor` | `/opt/intel/oneapi/advisor/latest/bin64/advisor` | Offload 建模 |
| `icpx` / `sycl-ls` / `mpirun` | oneAPI | 编译与运行 |
| PyTorch + Triton | Python | AI 基准 |

### ❌ 需要自行构建

| 工具 | 获取方式 | 优先级 |
|---|---|---|
| `ze_peak` | intel/compute-runtime 或 level-zero 仓库自带的 peak 示例 | **高** |
| **BabelStream (SYCL)** | GitHub `UoB-HPC/BabelStream`，`-DMODEL=sycl` | **高** |
| `benchdnn` | oneDNN 源码构建 | 中 |
| oneCCL benchmarks | oneCCL 源码构建（含 `allreduce` 等） | **高** |
| OSU Micro-Benchmarks | 需编译 GPU-aware/SYCL 版 | 中 |
| `unitrace` / `onetrace` | intel/pti-gpu GitHub | 中 |
| oneMKL benchmarks | oneMKL 源码构建 | 低 |
| HPL / HPCG (Intel 优化版) | oneAPI HPC Toolkit 或 Intel 下载 | 中 |
| GROMACS / LAMMPS / QE 等 | 各项目源码 + SYCL 编译 | 中 |
| IPEX-LLM / vLLM-XPU | pip / 源码 | 中 |
| TorchBench / MLPerf | GitHub | 低 |

---

## 5. 环境注意事项

1. **必须先 `source /opt/intel/oneapi/setvars.sh`**，否则 `sycl-ls`、`icpx` 等不在 PATH 中（实测未加载时 `which sycl-ls` 为空）。
2. **多版本并存容易踩坑**：`latest` → 2026.1，但 `dnnl` latest 只在 2026.0。定位性能异常时**必须记录实际使用的版本号**，可用 `setvars.sh` 的版本选择参数固定版本。
3. 检查 oneAPI 是否有 license 问题：`/opt/intel/oneapi/licensing` 存在，VTune/Advisor 可能需要 license（Advisor 免费版限制较多）。
4. `i915` 而非 `xe` 驱动 → 部分最新 dGPU 特性可能不可用，注意查阅对应版本支持矩阵。
