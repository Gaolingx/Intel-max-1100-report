# ⑦ Profiling 与瓶颈定位

**优先级：P2（对 ⑤⑥ 的热点做深入分析时开展）**
**文档位置：** `docs/TODO/07-profiling.md`

---

## 1. 测试目标

1. 定位「为什么慢」：**算力受限 / 带宽受限 / 延迟受限 / 主机瓶颈**
2. 用 **Roofline 模型**验证实测带宽与算力的关系
3. 找出**热点 kernel** 及其优化空间
4. 分析 **Level Zero API / 驱动层开销**（launch、内存分配、同步）
5. 验证 XMX 是否被真正利用

## 2. 工具矩阵

| 工具 | 状态 | 能力 |
|---|---|---|
| **VTune Profiler 2026.4** | ✅ 已装 | GPU Offload、GPU Compute/Media Hotspots、**GPU Roofline** ← 最推荐 |
| **Advisor 2026.0** | ✅ 已装 | Offload Modeling、GPU Roofline、向量化建议（**移植前预测**） |
| **PTI 1.1** | ✅ `libpti.so` 已装 | Profiling Tools Interfaces |
| **`unitrace` / `onetrace`** | ❌ 需下载 | Kernel 级 timeline + Level Zero/OpenCL API trace |
| `intel_gpu_top` | ✅ 已装 | 实时利用率/频率/功耗 |
| `xpu-smi dump` | ✅ 已装 | 硬件计数（含 EU Array Active/Stall/Idle） |
| `mem_bench` | ✅ 已装 | 内存带宽微基准 |
| ITT API | ✅ | 代码插桩标记 |

---

## 3. 执行步骤

### 3.1 VTune GPU Roofline（**最有价值的分析**）

```bash
source /opt/intel/oneapi/setvars.sh
vtune -collect gpu-roofline -result-dir r001_gpu_roofline -- <my_app>
vtune -report summary -r r001_gpu_roofline
vtune-gui r001_gpu_roofline        # 图形化查看
```
**看点**：
- 热点 kernel 落在 Roofline 图的哪个位置（算力墙 / 带宽墙）
- 实测**带宽天花板**是否与 ③ 的 BabelStream 结果一致
- 实测**算力天花板**是否与 ② 的峰值结果一致

> Roofline 的斜线 = 带宽上限，平台 = 算力上限。
> 若与 ②③ 实测不符，说明 Roofline 模型需要修正（或测量方法有偏差）。

### 3.2 VTune GPU Hotspots（kernel 级）

```bash
vtune -collect gpu-hotspots -result-dir r002_gpu_hotspots -- <my_app>
vtune -report hotspots -r r002_gpu_hotspots
```
关注：
- 每个 kernel 的**占用时间占比**
- GPU 时间 vs CPU 时间 vs **gap（等待）时间**
- EU 占用率、SIMD 利用率、内存吞吐

### 3.3 VTune GPU Offload（分析 host↔device 交互）

```bash
vtune -collect gpu-offload -result-dir r003_gpu_offload -- <my_app>
```
**看点**（对内存倒挂的机器尤其重要）：
- **数据传输时间占比** ← 很可能偏高
- kernel launch 次数与开销
- host↔device 带宽实测值（对照 ③④）
- CPU 是否成为串行瓶颈

### 3.4 Advisor Offload Modeling（移植预测）

```bash
advisor --collect=survey --project-dir=./adv -- <my_app>
advisor --collect=offload --project-dir=./adv
advisor --report=roofline --project-dir=./adv
```
用途：**在动手移植前**评估某段代码 offload 到 GPU 的潜在收益。适合评估「值得移植哪些 kernel」。

### 3.5 unitrace / onetrace（需下载）

```bash
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace   # 或 onetrace
mkdir build && cd build && cmake .. && make -j
./unitrace --chrome-kernel-logging ./my_app        # 生成 chrome tracing JSON
./unitrace --device-timing ./my_app
```
产出 **Chrome Trace 格式**的 timeline，可直接在 `chrome://tracing` 或 Perfetto 中打开。
看点：kernel 时序、Level Zero API 调用耗时、内存分配开销。

### 3.6 硬件计数器（轻量、无干扰）

```bash
xpu-smi dump -d 0 -m 0,1,2,3,5,9,10,11 -i 100 -n 600 -j > profile_telemetry.json
```

| metric | 含义 |
|---|---|
| 0 | GPU Utilization (%) |
| 1 | GPU Power (W) |
| 2 | GPU Frequency (MHz) |
| 3 | GPU Core Temperature |
| 5 | GPU Memory Utilization (%) |
| **9** | **EU Array Active (%)** — 纯执行时间占比 |
| **10** | **EU Array Stall (%)** — 有线程但停等 |
| **11** | **EU Array Idle (%)** — 无线程可调度 |

**三种状态的比例解读**：
| Active | Stall | Idle | 诊断 |
|---|---|---|---|
| 高 | 低 | 低 | 执行效率好 |
| 中 | 高 | 低 | **内存延迟/带宽受限** ← 常见 |
| 低 | 低 | 高 | **不足的并行度 / launch 开销 / host 瓶颈** |

> 这是一个**无需额外工具、零干扰**的高效诊断手段，建议所有测试都带上它。

### 3.7 代码插桩（ITT API）

在代码关键区域加标记：
```cpp
#include <ittnotify.h>
__itt_domain* domain = __itt_domain_create("MyApp");
__itt_string_handle* task_preprocess = __itt_string_handle_create("preprocess");
__itt_task_begin(domain, __itt_null, __itt_null, task_preprocess);
// ... 关键代码 ...
__itt_task_end(domain);
```
作用：让 VTune / unitrace 的 timeline 上出现**有语义的阶段划分**，极大简化分析。

---

## 4. 分析框架

### 第一步：先看宏观时间分布
```
总时间 = CPU 时间 + GPU kernel 时间 + 数据传输时间 + 同步/等待时间
```
哪一项占比最大 → 从那里入手。
**在内存倒挂的机器上，传输/等待时间占比偏高是大概率事件。**

### 第二步：Roofline 定位
- 落在斜线上 → **带宽受限** → 优化方向：数据复用、分块、降低精度
- 落在平台上 → **算力受限** → 优化方向：用 XMX、提高 SIMD 利用率、向量化
- 落在两者之间 → 需要更多并行度

### 第三步：EU Array 三态
用 `xpu-smi dump` 的 metric 9/10/11 快速判断是延迟受限还是并行度不足。

### 第四步：针对性优化
| 症状 | 优化方向 |
|---|---|
| 带宽受限 | 数据复用（tiling）、融合 kernel、降低数据精度 |
| 算力受限 | 使用 XMX、向量化、提高 SIMD width 利用 |
| 延迟受限 | 增加并发（更多 work-item）、prefetch |
| host 瓶颈 | 异步传输、pinned memory、DataLoader worker 调整 |
| 通信受限 | 减少同步次数、overlap compute/communication |

---

## 5. 输出物

1. **Roofline 图表**（热点 kernel 的分类）
2. **时间分布饼图**（CPU / GPU / 传输 / 等待）
3. **热点 kernel 列表**（含耗时占比、EU 占用率）
4. **Chrome Trace timeline**（关键阶段的时序图）
5. **EU Active/Stall/Idle 时间序列**
6. **优化建议清单**（按预期收益排序）

---

## 6. 判读标准

| 现象 | 诊断 |
|---|---|
| 数据传输时间 > 30% | **内存倒挂导致的主机瓶颈** → 优先优化 |
| GPU 时间中有大量 gap | launch 开销或 CPU 串行化 → 用 CUDA Graph 类技术/SYCL queue 批处理 |
| EU Active 低、Idle 高 | 并行度不足，增大 problem size 或 work-group |
| EU Stall 高 | 内存延迟 → 提高数据局部性 |
| Roofline 上 kernel 远离任何边界 | kernel 实现有问题 |
| BF16 kernel 性能与 FP32 相当 | **XMX 未被使用**（对照 ② 的发现） |

---

## 7. 注意事项

1. **Profiling 会带来开销**（尤其 VTune 的细粒度采集），性能数字不能直接当基准用。**先跑干净基准，再做 profiling。**
2. `vtune-gui` 需要**图形界面**。纯 SSH 环境下只能用 CLI：
   ```bash
   vtune -report hotspots -r r001
   vtune -report summary -r r001
   ```
   或在本地用 GUI 打开远程 result dir（配合 SSH 转发）。
3. **Advisor / VTune 可能需要 license**，注意 `/opt/intel/oneapi/licensing`。
4. `unitrace` 需要从 GitHub 单独获取，不在 oneAPI 安装包内。
5. 先做 `gpu-hotspots` 找出热点，再对**热点**做 `gpu-roofline`，避免在冷 kernel 上浪费时间。
6. 采样型 profiler 会**丢失短 kernel**，必要时用 `--<intensity>` 或手动插桩（ITT）。
7. 结合 ② ③ ④ 的基线数字做交叉验证 —— **没有基线的 profiling 结论容易自相矛盾**。
