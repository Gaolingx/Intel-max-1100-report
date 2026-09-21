# ⑥ HPC 应用性能测试

**优先级：P2（需自行编译环境，耗时最长，按需开展）**
**文档位置：** `docs/TODO/06-hpc-apps.md`

---

## 1. 测试目标

1. 测量**真实科学计算应用**在这套 XPU 平台上的性能
2. 计算应用**达到硬件峰值能力的百分比**（关键价值指标）
3. 验证 **SYCL / oneAPI 移植路径**的成熟度
4. 与 Aurora（Max 1550）同架构 → **测试经验可复用**

## 2. 平台背景

Max 1100 与 Aurora 的 Max 1550 同属 Ponte Vecchio 架构：

| 项目 | Max 1100（本机） | Max 1550（Aurora） | 比值 |
|---|---|---|---|
| Tile / 卡 | 1 | 2 | 0.5 |
| Xe-core / 卡 | 56 | 128 | 0.4375 |
| EU / 卡 | 448 | 1024 | 0.4375 |
| HBM | 48 GiB | 128 GiB | — |

**换算式**：本机算力 ≈ Aurora 单卡的 **43.75%**。

---

## 3. 应用清单与优先级

| 应用 | 关键指标 | 编译方式 | 优先级 |
|---|---|---|---|
| **HPL / HPCG** | TFLOPS、效率 % | oneAPI HPC Toolkit / Intel 优化版 | **高**（直接压 FP64 峰值） |
| **GROMACS** | ns/day | `-DGMX_GPU=SYCL` | **高** |
| **LAMMPS** | timesteps/s | KOKKOS + SYCL | **高** |
| **STREAM**（host） | GB/s | ✅ 已装 `/usr/bin/stream` | **高**（最快出结果） |
| **Quantum ESPRESSO** | SCF 迭代时间 | SYCL | 中 |
| **NAMD** | ns/day | oneAPI 版 | 中 |
| **OpenFOAM** | 求解器迭代时间 | SYCL/oneAPI 移植版 | 中 |
| **CP2K** | SCF 时间 | SYCL/OpenCL | 低 |
| **OpenMC / WarpX / PETSc** | 各自 | 视移植情况 | 低 |

---

## 4. 执行步骤

### Step 0：先做最快出结果的

**host 侧内存带宽（已装，立即可跑）**
```bash
OMP_NUM_THREADS=144 /usr/bin/stream
```
这个数字是理解「内存倒挂」影响的关键前提，**务必先测**。

### Step 1：HPL / HPCG（FP64 峰值验证）

**HPL** 用来验证 ② 测出的 FP64 峰值在真实负载下能达到多少：
```
效率 = HPL 实测 TFLOPS / FP64 理论峰值（~11.1 TFLOPS）
```
获取方式：Intel oneAPI HPC Toolkit 或 Intel 官方优化版 HPL 二进制。
关键参数：问题规模 N、block size NB、P × Q 进程网格。

**HPCG** 则是 memory-bound 参照 —— 可揭示带宽瓶颈（对照 ③）。

> 只有 2 张卡 → 规模受限，重点看**单卡效率**而非绝对排名。

### Step 2：GROMACS（SYCL）

```bash
source /opt/intel/oneapi/setvars.sh
# 编译（需要 cmake / fftw / 或使用 MKL 的 FFT）
cmake -B build -S . \
  -DGMX_GPU=SYCL \
  -DGMX_SYCL_DPCPP=ON \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGMX_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
**运行**（用官方 benchmark 案例，如 `benchMEM`、`benchPEP`、`stmv`）：
```bash
# 单卡
ZE_AFFINITY_MASK=0 ./build/bin/gmx mdrun -s benchMEM.tpr -nsteps 10000 -noconfout
# 双卡
mpirun -n 2 ./build/bin/gmx_mpi mdrun -s benchMEM.tpr -nsteps 10000 -noconfout
```
**记录**：ns/day、GPU 利用率、是否触发 PP/PME offload、线程设置（`-ntmpi` / `-ntomp`）的影响。

### Step 3：LAMMPS（KOKKOS + SYCL）

```bash
git clone https://github.com/lammps/lammps.git
cd lammps && mkdir build && cd build
cmake ../cmake \
  -DPKG_KOKKOS=ON -DKokkos_ENABLE_SYCL=ON \
  -DKokkos_ARCH_INTEL_PVC=ON \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
make -j
```
**运行**（官方 bench 目录，如 `lj`、`eam`、`rhodo`）：
```bash
./lmp -k on g 1 -sf kk -in in.lj -log log.lj
ZE_AFFINITY_MASK=0 ./lmp -k on g 1 -sf kk -in in.eam
```
**记录**：timesteps/s、不同问题规模的缩放、单精度 vs 双精度。

### Step 4：Quantum ESPRESSO（可选）

```bash
# 使用 Intel oneAPI 的 MKL 作为 BLAS/LAPACK/FFT
./configure F90=mpiifx CC=mpiicx MPIF90=mpiifx \
  --enable-openmp --with-scalapack=intel
```
**记录**：SCF 单次迭代时间、`benchmark` 输入集的总耗时。

### Step 5：多卡弱/强扩展测试

对所有应用，比较：
| 配置 | 性能 | 扩展效率 |
|---|---|---|
| 1 卡 | | 1.00× |
| 2 卡 | | ___ % |

**必须**结合 ④ 的互连结果解读：若应用扩展效率低但卡间带宽达标，说明是应用本身的通信模式问题（例如全对全通信过多）。

---

## 5. 指标记录表

| 应用 | 版本 | 编译选项 | 1 卡 | 2 卡 | 扩展效率 | 达硬件峰值 % |
|---|---|---|---|---|---|---|
| STREAM (host) | | | | | N/A | |
| HPL | | | TFLOPS | | | |
| HPCG | | | TFLOPS | | | |
| GROMACS (benchMEM) | | | ns/day | | | |
| GROMACS (stmv) | | | ns/day | | | |
| LAMMPS (lj) | | | ts/s | | | |
| LAMMPS (eam) | | | ts/s | | | |
| QE (benchmark) | | | s | | | |

---

## 6. 判读标准

| 现象 | 诊断 |
|---|---|
| HPL 效率 < 60% | 单卡/单节点规模太小、BLAS 未走 GPU、配置不佳 |
| HPL 效率低但 HPCG 正常 | 偏 memory-bound 特性，符合预期（对照 ③） |
| GROMACS 加速不明显 | 未走 SYCL offload、PP/PME 未分离、原子数太小 |
| LAMMPS 无加速 | Kokkos SYCL 未生效、问题规模太小、未用 `-sf kk` |
| 应用速度正常但扩展效率低 | 通信密集型 → 对照 ④ |
| 大批量测试后性能下降 | 热降频 → 对照 ① 的温度曲线 |

> **核心判据**：真实应用只达到峰值能力的 30–60% 是正常的。
> 关键是要能**解释为什么**（算力受限 / 带宽受限 / 通信受限 / 规模太小）。

---

## 7. 注意事项

1. **编译耗时长**：每个应用 30 min–数小时，建议后台跑并保存完整编译日志（含编译选项）。
2. 所有应用的 **CMake 选项必须归档** —— 否则结果不可复现。
3. 问题规模是最大的变量：**小规模测不出 GPU 优势**，优先选择官方 benchmark 输入。
4. 主机内存只有 45 GiB → 大规模 HPL/LAMMPS 可能受 host 内存限制，注意观察。
5. 双卡运行注意**总功耗 600 W**。
6. 多数 HPC 应用需要 `mpi` 编译 → **必须 `source setvars.sh`** 并使用 `mpiicpx`/`mpiifx`。
7. **STREAM 主机带宽是最容易做的，建议第一个跑**，作为整体瓶颈判断的锚点。

---

## 8. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| ← ② 算力峰值 | 计算应用达峰百分比 |
| ← ③ 显存带宽 | 判断 memory-bound 应用上限 |
| ← ④ 互连 | 解释多卡扩展效率 |
| → ⑦ Profiling | 对热点 kernel 做 Roofline 分析 |
