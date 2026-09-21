# 性能测试计划总览（TODO）

> 平台：`hwt` — 2 × Intel® Data Center GPU Max 1100（Ponte Vecchio）
> 前置阅读：[`../hardware.md`](../hardware.md)、[`../interconnect.md`](../interconnect.md)、[`../caveats.md`](../caveats.md)

---

## 一、测试分类总览

| # | 分类 | 目标 | 文档 | 优先级 |
|---|---|---|---|---|
| 1 | 硬件健康 / 稳定性 / 压力 | 确认硬件无异常、能稳定满载 | [`01-health-stress.md`](./01-health-stress.md) | **P0** |
| 2 | 算力峰值 | FP32 / FP64 / INT / XMX 理论 vs 实测 | [`02-compute-peak.md`](./02-compute-peak.md) | **P1** |
| 3 | 显存带宽 | HBM 读写带宽达成率 | [`03-memory-bandwidth.md`](./03-memory-bandwidth.md) | **P1** |
| 4 | 互连 | Xe Link / GPU-aware MPI / oneCCL 带宽与延迟 | [`04-interconnect-xelink.md`](./04-interconnect-xelink.md) | **P1** |
| 5 | AI / 深度学习 | 训练/推理吞吐、算子基准、双卡扩展 | [`05-ai-dl.md`](./05-ai-dl.md) | **P1** |
| 6 | HPC 应用 | 真实科学计算应用性能 | [`06-hpc-apps.md`](./06-hpc-apps.md) | P2 |
| 7 | Profiling | 瓶颈定位（算力受限 vs 带宽受限） | [`07-profiling.md`](./07-profiling.md) | P2 |
| 8 | 功耗 / 能效 / 调度 | perf/W、功耗-性能曲线、锁频影响 | [`08-power-efficiency.md`](./08-power-efficiency.md) | P2 |

---

## 二、关键参数速查（写报告时引用）

### 理论峰值（待实测验证）

| 指标 | 计算公式 / 值 | 状态 |
|---|---|---|
| **FP32 峰值** | 448 EU × 16 lane × 2 (FMA) × 1.55 GHz ≈ **22.2 TFLOPS** | 待实测 |
| **FP64 峰值** | PVC 上 FP64 速率 = FP32 的 1/2 ≈ **11.1 TFLOPS** | 待实测 |
| **XMX (BF16/FP16)** | 量级为 FP32 的数倍，**具体倍数需实测确定** | 待实测 |
| **XMX (INT8)** | 通常为 BF16 的 2 倍 | 待实测 |
| **HBM 带宽** | 量级 1–2.5 TB/s，**必须由 BabelStream 实测确定** | 待实测 |
| **Xe Link 单向总带宽** | 6 × 50.66 GiB/s ≈ **304 GiB/s ≈ 318 GB/s** | 待实测 |
| **PCIe 5.0 x16** | 原始约 63 GB/s，实测约 50–55 GB/s | 待实测 |

> ⚠️ **不要直接引用上表数字作为结论**。除 Xe Link 外均为公式推导或量级估计，必须以实测为准。

### 硬件配置

```
GPU 数            : 2 × Intel Data Center GPU Max 1100 (PVC, Production ES)
Tile / 卡         : 1
Xe-core / 卡      : 56        EU / 卡 : 448      SIMD 宽 : 16
核心频率          : 1550 MHz（min = max，锁定）
HBM / 卡          : 48 GiB，ECC 开
Xe Link           : XL24（6 端口 × 4 lane，直连）
PCIe              : Gen5 x16
功耗上限          : 300 W（可调 150–300 W）
CPU               : 72c/144t ES，1 NUMA 节点
主机内存          : 45 GiB（< 两卡 96 GiB HBM，注意倒挂）
```

---

## 三、推荐执行顺序与依赖关系

```
┌─────────────────────────────────────────────────────────┐
│ 阶段 0：环境准备（必做）                                  │
│  source setvars.sh / 记录版本 / 清 cache / 采集空载基线   │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 1：P0 硬件健康 + 稳定性                              │
│  xpu-smi health / diag --precheck / diag --stress        │
│  → 若失败：停止，先修硬件                                │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 2：P1 底层基线（三者可并行）                          │
│  ② 算力峰值 (ze_peak / torch matmul)                     │
│  ③ 显存带宽 (BabelStream)                                │
│  ④ 互连 (IMB-MPI1-GPU 现成可用 + P2P 可用性检查)          │
│  → 产出「硬件能力基线表」                                │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 3：P1 应用层                                         │
│  ⑤ AI：GEMM sweep / ResNet50 / BERT / DDP 扩展           │
│     ↑ 依赖 ②③（解释为何快/慢）                            │
│     ↑ 依赖 ④（多卡扩展效率）                             │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 4：P2 深入                                           │
│  ⑦ Profiling (VTune GPU Roofline) ← 对阶段 3 的热点 kernel │
│  ⑧ 功耗/能效扫描（独立，可随时做）                        │
│  ⑥ HPC 应用（需自行编译环境，耗时最长，按需开展）          │
└─────────────────────────────────────────────────────────┘
```

**依赖说明**：

| 测试 | 依赖 | 原因 |
|---|---|---|
| 多卡 AI 扩展效率 | 必须先做互连测试 | 否则无法区分「算法问题」与「链路问题」 |
| VTune Roofline | 先有 AI/HPC 结果 | 需要真实热点 kernel 才有剖析价值 |
| 功耗-性能曲线 | 建议先有算力基线 | 用于判断功耗增加是否带来线性收益 |
| HPC 应用 | 依赖算力基线 | 用于判断应用是否达硬件能力百分比 |

---

## 四、各分类简要清单

### ① 硬件健康 / 稳定性（P0）
- [ ] `xpu-smi health -l` 全器件健康检查
- [ ] `xpu-smi diag --precheck --listtypes` / `--precheck --gpu` 快速预检
- [ ] `xpu-smi diag -d 0 -l 2` 分级诊断
- [ ] `xpu-smi diag -d 0,1 --stress --stresstime 600` 双卡压力测试
- [ ] 压力期间遥测：频率是否掉、温度、ECC/Reset/Driver Error 计数

### ② 算力峰值（P1）
- [ ] `ze_peak`（需 clone 编译）→ FP32 / FP64 / INT
- [ ] BabelStream 派生算力项
- [ ] PyTorch/Triton GEMM sweep（M/N/K 扫点，fp32/fp64/bf16）
- [ ] 各 dtype 相对 FP32 的加速比

### ③ 显存带宽（P1）
- [ ] BabelStream SYCL：Copy / Mul / Add / Triad
- [ ] PyTorch 大 tensor copy 交叉验证
- [ ] `xpu-smi dump -m 6,7` 硬件侧读写计数对照
- [ ] 不同访问粒度/向量宽度对带宽影响

### ④ 互连（P1）
- [ ] P2P 可用性检查（Level Zero / SYCL）
- [ ] `IMB-MPI1-GPU` ↔ 卡间带宽/延迟
- [ ] `IMB-MPI1-GPU` ↔ host↔device 带宽
- [ ] oneCCL allreduce（需构建）
- [ ] PyTorch DDP 扩展效率曲线
- [ ] `xpu-smi dump` 的 Xe Link Throughput 对照

### ⑤ AI / 深度学习（P1）
- [ ] ResNet-50 / BERT 训练吞吐（samples/sec、TFLOPS）
- [ ] AMP + BF16 vs FP32 收益
- [ ] 单卡 vs 双卡 DDP scaling efficiency
- [ ] LLM 推理：prefill/decode 吞吐、TTFT、显存峰值
- [ ] 算子 micro-benchmark：GEMM / attention / LayerNorm

### ⑥ HPC 应用（P2）
- [ ] GROMACS（SYCL）ns/day
- [ ] LAMMPS（KOKKOS+SYCL）timesteps/s
- [ ] Quantum ESPRESSO（SYCL）
- [ ] HPL / HPCG → TFLOPS + 效率
- [ ] STREAM（已装）确认 host 内存带宽上限
- [ ] NAMD / OpenFOAM / CP2K（按需）

### ⑦ Profiling（P2）
- [ ] VTune：GPU Offload / GPU Compute Hotspots / **GPU Roofline**
- [ ] Advisor：Offload Modeling / GPU Roofline
- [ ] PTI + unitrace 做 kernel 级 timeline
- [ ] `xpu-smi dump -m 0,5,9,10,11` 拿 EU Array Active/Stall/Idle

### ⑧ 功耗 / 能效 / 调度（P2）
- [ ] perf/W（`dump -m 1,8` 取功耗与能耗）
- [ ] `--powerlimit` 150→300 W 扫功耗-性能曲线
- [ ] `--frequencyrange` 200→1550 MHz 扫频率-性能曲线
- [ ] `--scheduler` timeslice / exclusive 对比
- [ ] Xe Link 端口开关对比（`--xelinkport`）
- [ ] `xpu-smi vgpu` SR-IOV 开销

---

## 五、命令速查

### 环境
```bash
source /opt/intel/oneapi/setvars.sh
echo 3 > /proc/sys/vm/drop_caches   # 清 cache 拿干净基线
xpu-smi discovery
sycl-ls
```

### 健康 / 诊断 / 压力
```bash
xpu-smi health -l
xpu-smi health -d 0                      # 单卡健康详情
xpu-smi health -d 0 -c 1                 # 组件1=Core温度 (2=Mem温度 3=功耗 4=显存 5=XeLink 6=频率)
xpu-smi diag --precheck --listtypes
xpu-smi diag --precheck --gpu
xpu-smi diag -d 0 -l 2
xpu-smi diag -d 0,1 --stress --stresstime 600
```

### 遥测采样
```bash
xpu-smi stats -d 0                       # 瞬时快照
xpu-smi dump -d -1 -m 0,1,2,3,5,6,7,8 -i 1000 -n 60   # 双卡，1s 间隔，60 次
# metric 编号：0=Util 1=Power 2=Freq 3=CoreTemp 4=MemTemp 5=MemUtil
#              6=MemRead 7=MemWrite 8=Energy 9=EUActive 10=EUStall 11=EUIdle
#              12+=错误计数
xpu-smi dump -d -1 -m 0,1,2,8 -i 100 -n 300 -j > telemetry.json
intel_gpu_top                            # 实时 TUI 监控
```

### 拓扑 / 进程
```bash
xpu-smi topology -m
xpu-smi topology -d 0
xpu-smi ps
```

### 配置（改功耗/频率）
```bash
xpu-smi config -d 0                      # 查看当前配置
xpu-smi config -d 0 --powerlimit 250     # 设功耗上限
xpu-smi config -d 0 -t 0 --frequencyrange 1000,1550
xpu-smi config -d 0 -t 0 --scheduler timeslice,5000,0
```

### MPI（已装，可直接跑）
```bash
export PATH=/opt/intel/oneapi/mpi/latest/bin:$PATH
mpirun -n 2 -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 IMB-MPI1-GPU
mpirun -n 2 IMB-MPI1-GPU PingPong
mpirun -n 2 IMB-MPI1-GPU Allreduce
mpirun -n 2 fi_pingpong                   # libfabric
```

### PyTorch XPU
```bash
python3 -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
python3 -c "import torch; print(torch.xpu.get_device_properties(0))"
```

### 构建缺失工具
```bash
# BabelStream (SYCL)
git clone https://github.com/UoB-HPC/BabelStream.git && cd BabelStream
cmake -B build -H. -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx && cmake --build build
./build/babelstream
```

---

## 六、结果记录模板

建议每个测试项按此格式归档：

```markdown
### <测试名>
- 日期 / 时间：
- oneAPI 版本（icpx --version 输出）：
- 驱动版本：I915_25.2.57_PSB_250224.65
- xpu-smi 版本：1.2.43
- Power Limit：300 W  /  频率：1550 MHz 锁定
- 测试命令（完整）：

| 配置 | 指标 | 第1次 | 第2次 | 第3次 | 中位数 |
|---|---|---|---|---|---|
| | | | | | |

- 实测 / 理论达成率：
- 同时观测到的 host CPU 占用 / 内存带宽：
- 同时观测到的 GPU 功耗 / 温度 / 频率：
- 异常现象：
- 结论与后续动作：
```

---

## 七、最终交付物（建议）

1. **硬件配置报告** — 本文档集第一部分
2. **硬件能力基线表** — 算力 / 带宽 / 互连 实测 vs 理论
3. **应用性能报告** — AI（训练/推理）+ HPC
4. **扩展性报告** — 单卡 → 双卡 scaling efficiency
5. **能效报告** — perf/W 与功耗-性能曲线
6. **瓶颈分析报告** — VTune Roofline 结果与优化建议
