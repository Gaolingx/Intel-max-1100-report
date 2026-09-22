# 性能测试计划总览（TODO）

> 平台：`hwt` — 2 × Intel® Data Center GPU Max 1100（Ponte Vecchio）
> 前置阅读：[`../hardware.md`](../hardware.md)、[`../interconnect.md`](../interconnect.md)、[`../caveats.md`](../caveats.md)

---

## 一、测试分类总览

| # | 分类 | 目标 | 文档 | 优先级 | 状态（2026-09-22） |
|---|---|---|---|---|---|
| 1 | 硬件健康 / 稳定性 / 压力 | 确认硬件无异常、能稳定满载 | [`01-health-stress.md`](./01-health-stress.md) | **P0** | ⬜ 未做 |
| 2 | 算力峰值 | FP32 / FP64 / INT / XMX 理论 vs 实测 | [`02-compute-peak.md`](./02-compute-peak.md) | **P1** | ✅ **已完成** → [`../Conclusion/02-compute-peak/`](../Conclusion/02-compute-peak/) |
| 3 | 显存带宽 | HBM 读写带宽达成率 | [`03-memory-bandwidth.md`](./03-memory-bandwidth.md) | **P1** | ✅ **已完成** → [`../Conclusion/03-memory-bandwidth/`](../Conclusion/03-memory-bandwidth/) |
| 4 | 互连 | Xe Link / GPU-aware MPI / oneCCL 带宽与延迟 | [`04-interconnect-xelink.md`](./04-interconnect-xelink.md) | **P1** | ✅ **已完成** → [`../Conclusion/04-interconnect-xelink/`](../Conclusion/04-interconnect-xelink/) |
| 5 | AI / 深度学习 | 训练/推理吞吐、算子基准、双卡扩展 | [`05-ai-dl.md`](./05-ai-dl.md) | **P1** | ✅ **已完成** → [`../Conclusion/05-ai-dl/`](../Conclusion/05-ai-dl/) |
| 6 | HPC 应用 | 真实科学计算应用性能 | [`06-hpc-apps.md`](./06-hpc-apps.md) | P2 | ⬜ 未做 |
| 7 | Profiling | 瓶颈定位（算力受限 vs 带宽受限） | [`07-profiling.md`](./07-profiling.md) | P2 | ⚠️ 部分（②③④ 内含定向 profiling，未用 VTune/Advisor） |
| 8 | 功耗 / 能效 / 调度 | perf/W、功耗-性能曲线、锁频影响 | [`08-power-efficiency.md`](./08-power-efficiency.md) | P2 | ⚠️ 部分（②③④ 短时+长时满载功耗；**长时热/功耗降额已实测取证**：温度峰值 101 °C、功耗冲到 305~330 W 越过 300 W 上限；未做功耗-性能曲线） |

> **完成度统计**：P1 的四项（②③④⑤）**全部完成**；P0 的 ① 与 P2 的 ⑥⑧ **未做**，
> ⑦ 由 ②③④ 内的定向 profiling 部分覆盖。逐条核验见各 TODO 文档的 §4.6 / §4.7 / §4.9。
> ℹ️ ② 的 `ze_peak` 已于 2026-09-22 **补做完成**（原记为「已跳过」）。

---

## 二、关键参数速查（写报告时引用）

### 理论峰值（**已完成实测验证** 2026-09-22）

| 指标 | 计算公式 / 值 | 实测结果与裁定 |
|---|---|---|
| **FP32 峰值** | 448 EU × 16 lane × 2 (FMA) × 1.55 GHz ≈ **22.2 TFLOPS** | ✅ **已裁定：以 22.2 为准**。oneDNN **22.13**（99.6%）+ 第三方 `ze_peak` **21.87**（98.4%）两个独立实现确认；自研纯 FMA 探针 ~~50.75~~（228%）已判定为探针 artefact 并**撤回**（见 ② §3.2） |
| **FP64 峰值** | PVC 上 FP64 速率 = FP32 的 1/2 ≈ ~~11.1 TFLOPS~~ | ❌ **原假设已证伪**：实测 **0.78 × FP32**（torch 17.37 TFLOPS）/**0.735 × FP32**（`ze_peak` 16.07）—— 两个独立来源互相吻合 |
| **XMX (BF16/FP16)** | 量级为 FP32 的数倍 | ✅ 裸 DPAS **355 TFLOPS = 16 × FP32**（99.8% 公式）；oneDNN 路径仅 226~238（覆盖 **58~67%**） |
| **XMX (INT8)** | 通常为 BF16 的 2 倍 | ✅ 裸 DPAS **710 TFLOPS = 2.0 × BF16**（99.8% 公式）；oneDNN 416（58.5%） |
| **HBM 带宽** | 量级 1–2.5 TB/s | ✅ BabelStream **900 GB/s** = 规格 1229 的 **73%**；双卡并发 **1679 GB/s（2.00× 线性）** |
| **Xe Link 单向总带宽** | 6 × 50.66 GiB/s ≈ **304 GiB/s ≈ 318 GB/s** | ✅ 规格已核实 **318.8 GB/s**；❌ 实测仅 **95.5 GB/s = 30.0%**，且 `Xe Link Calibration Date: Not Calibrated` |
| **PCIe 5.0 x16** | 原始约 63 GB/s，实测约 50–55 GB/s | ⚠ 实测 pinned **31.9 GB/s**（约 50%）；且 `LnkSta` 自相矛盾（GPU 端点报 2.5 GT/s ×1，上游桥报 32 GT/s ×16） |

> ✅ 上表已由 `benchmark/{02,03,04,05-ai-dl}` 实测填充，结论见
> [`../Conclusion/`](../Conclusion/) 下各自 README。
> ✅ **FP32 口径冲突已于 2026-09-22 裁定：以公式值 22.2 TFLOPS 为准**
> （oneDNN 99.6% + 第三方 `ze_peak` 98.4%），自研探针的 50.75 已撤回 ——
> 见 [`02-compute-peak.md`](./02-compute-peak.md) §3.7 与
> [`../Conclusion/02-compute-peak/README.md`](../Conclusion/02-compute-peak/README.md) §3.2。
> ⚠️ **引用峰值时的时长纪律（2026-09-22 新增）**：以上峰值均为**短时**读数。
> `ze_peak` 整轮长跑（~20 min/卡）实测 **温度 92 → 101 °C**、功耗冲到 **305~330 W**
> （**已越过 300 W 名义上限**），同时 `gt_act_freq_mhz`（本机**唯一**随负载变化的频率节点）
> 在大幅摆动 ⇒ **长时满载确实会热/功耗降额**，长时稳态性能需打 ~0.87 折扣。
> ⚠️ 但 `gt_act_freq_mhz` 的**绝对值噪声极大**、并不收敛到标准 P-state
> （标准态只有 RP0=1550 / RP1=1000 / RPn=200；实测还出现 1400/1150/950/650/600/450/400/350/300 等
> 几乎每次采样都不同的值）⇒ **只能当定性证据**（"该节点在大幅摆动说明 DVFS 很活跃"），
> **不可用它反算性能**。硬证据是**温度与功耗**。详见
> [`02-compute-peak.md`](./02-compute-peak.md) §3.7.3.1 与
> [`08-power-efficiency.md`](./08-power-efficiency.md) §2。

### 硬件配置

```
GPU 数            : 2 × Intel Data Center GPU Max 1100 (PVC, Production ES)
Tile / 卡         : 1
Xe-core / 卡      : 56        EU / 卡 : 448      SIMD 宽 : 16
核心频率          : 1550 MHz（=`gt_cur/max/min_freq_mhz` 的**请求值**；
                    长时满载会自主降额，真实活跃频率见 `gt_act_freq_mhz`）
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

### ② 算力峰值（P1）— ✅ **已完成**（结论：[`../Conclusion/02-compute-peak/`](../Conclusion/02-compute-peak/)，逐条核验见 [`02-compute-peak.md`](./02-compute-peak.md) §4.6）
- [x] `ze_peak` → FP32 / FP64 / INT　✅ **已补做（2026-09-22）**：Intel 官方
  `oneapi-src/level-zero-tests/perf_tests/ze_peak` 已抓取、构建（25 文件 / 零外部依赖 / 1 行补丁）
  并跑通双卡，作为**第三方仲裁者**裁定了 FP32 口径冲突（21.87 = 公式 98.4%）；
  自研 SYCL 探针仍保留（`sycl/alu_peak.cpp`、`sycl/xmx_peak.cpp`，多出占用率拐点扫描），
  但其**绝对值已撤回**。详见 [`02-compute-peak.md`](./02-compute-peak.md) §3.7
- [x] BabelStream 派生算力项　→ 由 ③ 的 BabelStream + ② 的 SYCL 探针覆盖
- [x] PyTorch/Triton GEMM sweep　→ PyTorch ✅（23 条）；Triton ⬜ **未做**（与 ⑤ 同一缺口）
- [x] 各 dtype 相对 FP32 的加速比　→ fp64 0.78× / fp16 16.0× / bf16 16.0× / int8 32.0×（裸 DPAS 口径）
- [x] 附加：oneDNN 交叉验证、XMX 正确性校验、频率锁定取证
- [x] 附加：**`ze_peak` 跨卡重复性**（短跑跨 2 卡 × 2 次离散度 ≈10⁻⁵）+ **长跑降额取证**
  （温度 92→101 °C、功耗 305~330 W 越过 300 W 上限；`gt_act_freq_mhz` 大幅摆动但绝对值噪声大，
  仅作定性 ⇒ 长跑整轮 fp64/int 段偏低，**绝对值只引用短跑**）

### ③ 显存带宽（P1）— ✅ **已完成**（结论：[`../Conclusion/03-memory-bandwidth/`](../Conclusion/03-memory-bandwidth/)，逐条核验见 [`03-memory-bandwidth.md`](./03-memory-bandwidth.md) §4.7）
- [x] BabelStream SYCL：Copy / Mul / Add / Triad　→ 实测峰值 **899.6 GB/s**（= 规格 73%），含 Dot
- [x] PyTorch 大 tensor copy 交叉验证　→ ✅ 15 条，与 BabelStream 互印证
- [ ] `xpu-smi dump -m 6,7` 硬件侧读写计数对照　❌ **不可行**：`dump` 挂死；`stats -d 0` 的 m6/m7 **恒为 ~576 kB/s**（空载=满载），计数器在本驱动上不可用
- [x] 不同访问粒度/向量宽度对带宽影响　→ vec 1/2/4/8/16、stride 扫描；**vec>4 反而腰斩**
- [x] 附加：读/写拆分、双卡并发 2.00×、host DRAM 基线 39.6 GB/s、L2/L3 陷阱定量

### ④ 互连（P1）— ✅ **已完成**（结论：[`../Conclusion/04-interconnect-xelink/`](../Conclusion/04-interconnect-xelink/)，逐条核验见 [`04-interconnect-xelink.md`](./04-interconnect-xelink.md) §4.9）
- [x] P2P 可用性检查（Level Zero）　→ **可用**（2/2 对，flags `ACCESS`+`ATOMICS`）
- [x] `IMB-MPI1-GPU` ↔ 卡间带宽/延迟　→ 43.71 GB/s @16 MiB（**须加 `I_MPI_OFFLOAD=1`**）
- [x] `IMB-MPI1-GPU` ↔ host↔device 带宽　→ `cpu_peak.PingPong` 12.39 GB/s；H2D/D2H 由 ③ 覆盖
- [x] oneCCL allreduce（需构建）　⚠️ **替代**：未构建 oneCCL benchmarks，改用 `torch.distributed`(xccl) → broadcast **94.59 GB/s**
- [x] PyTorch DDP 扩展效率曲线　→ 由 ⑤ 覆盖（`Conclusion/05-ai-dl/03-scaling.md`）
- [ ] `xpu-smi dump` 的 Xe Link Throughput 对照　❌ **不可行**：`dump` 挂死；`stats -d 0` 的 `Xe Link Throughput` **恒为 N/A**（P2P 压满也是 N/A）
- [x] 附加：裸 L0 P2P 峰值 **95.51 GB/s**、PCIe `LnkSta` 自相矛盾取证、`card1` 非 GPU 甄别、Xe Link 静态规格 + `Not Calibrated` 取证

### ⑤ AI / 深度学习（P1）— ✅ **已完成**（结论：[`../Conclusion/05-ai-dl/`](../Conclusion/05-ai-dl/)）
- [x] ResNet-50 / BERT 训练吞吐（samples/sec、TFLOPS）
- [x] AMP + BF16 vs FP32 收益
- [x] 单卡 vs 双卡 DDP scaling efficiency
- [x] LLM 推理：prefill/decode 吞吐、TTFT、显存峰值
- [x] 算子 micro-benchmark：GEMM / attention / LayerNorm

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

```bash
# ze_peak (Level Zero, 第三方向量基准) —— ✅ 已构建于 benchmark/02-compute-peak/ze_peak_src/
cd /root/workspace/benchmark/02-compute-peak/ze_peak_src
./build.sh                    # 仅 g++ + <shim>/level_zero + -lze_loader，无外部依赖
cd build && ./ze_peak -h      # 必须在 build/ 下运行（.spv 以相对路径加载）
./ze_peak -d 0 -t sp_compute -i 10 -w 5      # fp32 向量峰值
./ze_peak -d 0 -a -i 3 -w 1   # ⭐ 全量但短跑（~60 s/卡）—— 引用绝对值的推荐口径
./ze_peak -d 0 -a -i 50 -w 10 # ⚠️ 整轮 ~20 min/卡，会撞热/功耗降额（实测 101 °C / 305~330 W）
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
- Power Limit：300 W  /  频率：短时满载 1550 MHz（⚠️ 长时满载会自主降频，见 02-compute-peak §3.7.3.1）
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

1. **硬件配置报告** — 本文档集第一部分　✅ 已交付：`../{README,hardware,interconnect,software-stack,precision-support,caveats}.md`
2. **硬件能力基线表** — 算力 / 带宽 / 互连 实测 vs 理论　✅ 已交付：②③④ 各自 `Conclusion/0X-…/README.md`
3. **应用性能报告** — AI（训练/推理）+ HPC　⚠️ 部分：AI ✅ `Conclusion/05-ai-dl/`；HPC（⑥）未做
4. **扩展性报告** — 单卡 → 双卡 scaling efficiency　✅ 已交付：③ 内存带宽 2.00× + ⑤ `05-ai-dl/03-scaling.md`
5. **能效报告** — perf/W 与功耗-性能曲线　⚠️ 部分：有单点 perf/W（②短时 171 W / 长时 **305~330 W**、③260 W、④），**未做功耗-性能曲线**；⚠️ ② 的 `ze_peak` 长时满载已实测出**热/功耗降额**（温度峰值 **101 °C**、功耗越过 **300 W** 名义上限、`gt_act_freq_mhz` 大幅摆动；稳态性能约 **0.87×**），该项优先级应上调 —— 最值得补的是「**时长 → 稳态频率/功耗**」曲线（`--frequencyrange` 扫点不可行，因请求频率被固定）
6. **瓶颈分析报告** — VTune Roofline 结果与优化建议　⚠️ 部分：②③④⑤ 均含定向 profiling 与瓶颈结论，**未用 VTune/Advisor Roofline**
