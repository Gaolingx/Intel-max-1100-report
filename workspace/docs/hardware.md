# 硬件概况

## 1. 主机（Host）

| 项目 | 值 | 备注 |
|---|---|---|
| 主机名 | `hwt` | |
| 操作系统 | Ubuntu 25.04 (Plucky Puffin) | |
| 内核 | `6.14.0-37-generic` | |
| CPU 型号 | `Genuine Intel(R) 0000` | **型号被掩码 → 预生产 ES 芯片** |
| CPU 主频 | 2.7 GHz 基频 / 3.9 GHz max | `CPU(s) scaling MHz` 空闲时约 22% |
| 核心/线程 | 72 核 / 144 线程（每核 2 线程） | |
| Socket | 1 | |
| NUMA 节点 | **1**（node 0 = CPU 0-143） | 跨 NUMA 优化空间有限 |
| 内存 | **45 GiB** 总量 / 约 41 GiB 可用 | 已用 4 GiB，buff/cache 33 GiB |
| Swap | 8 GiB | |

### ⚠️ 内存倒挂问题
两卡合计 HBM **96 GiB**（2 × 48 GiB），而主机内存仅 **45 GiB**。
这意味着：
- 无法在主机侧完整缓存两份满卡权重/数据
- 大数据集「host → device」流式传输、CPU 侧预处理、数据加载都容易成为瓶颈
- 建议所有测试同时记录 host 侧内存带宽（`/usr/bin/stream`）与 CPU 占用，避免误把主机瓶颈当成 GPU 瓶颈

---

## 2. GPU

两张卡为**同一型号、同规格**，仅序列号与插槽不同。

| 项目 | 值 |
|---|---|
| 型号 | Intel® Data Center GPU **Max 1100** |
| 架构 | Ponte Vecchio (PVC) |
| SKU 类型 | **Production ES**（工程样品） |
| Device ID | 0 / 1（`xpu-smi`），PCI Device ID `0xbda` |
| Stepping | B4 |
| PCI BDF | GPU0 → `0000:40:00.0`（插槽 PCIE_6）<br>GPU1 → `0000:8c:00.0`（插槽 PCIE_4） |
| DRM 设备 | GPU0 → `/dev/dri/card0`<br>GPU1 → `/dev/dri/card2` |
| Serial | GPU0 `WTP231800269`，GPU1 `WTP231800257` |
| SOC UUID | GPU0 `...74d8-64836dad91a4`，GPU1 `...76d6-44d7fa576257` |
| 功能类型 | physical（非 SR-IOV VF） |

### 计算规格

| 项目 | 值 |
|---|---|
| Tile 数 | **1** tile/卡 |
| Slice 数 | 1 |
| Sub-slice / Slice | 56 |
| **Xe-core** | **56** |
| **EU 总数** | **448** |
| Threads / EU | 8 |
| EU SIMD 宽度 | 16 |
| **核心频率** | **1550 MHz**（min = max = 1550，当前锁定） |
| 最大硬件 Context | 65536 |
| Max Command Queue Priority | 0 |

### 显存规格

| 项目 | 值 |
|---|---|
| 类型 | HBM（HBM2e 级，Ponte Vecchio 平台） |
| 容量 | **49136 MiB ≈ 48 GiB**/卡 |
| Max Mem Alloc Size | 46679.20 MiB（约 45.6 GiB，单次分配上限） |
| **ECC 状态** | **enabled** |
| 内存通道数 | 32 |
| Memory Bus Width | 128 |
| 空闲时占用 | 约 22 MiB，Memory Read/Write ≈ 580 kB/s（空载底噪） |

### 媒体能力

| 项目 | 值 |
|---|---|
| Media Engines | **0** |
| Media Enhancement Engines | **0** |

> ⚠️ **Max 1100 无媒体引擎** → 不要安排视频编解码（FFmpeg / QSV / VA-API）类性能测试，这类指标恒为 `N/A`。

### 固件与驱动

| 项目 | 值 |
|---|---|
| 内核驱动 | **i915**（`I915_25.2.57_PSB_250224.65`），**走传统 i915 路径而非 `xe`** |
| Driver Package Version | `1.25.2.57.250224.65+i75-1` |
| GFX Firmware | `PVC2_1.23374`，状态 `normal` |
| GFX PSC Firmware | `0x12cfd.0x20220803` |
| AMC Firmware | `6.7.0.0` |

---

## 3. 功耗与频率配置（`xpu-smi config`）

| 项目 | 值 |
|---|---|
| **Power Limit** | **300 W** |
| 可调功耗范围 | **150 – 300 W** |
| GPU Min / Max Frequency | 1550 / 1550 MHz |
| 可调频率档位 | 200–1550 MHz，步进 50 MHz（200,250,…,1500,1550） |
| Standby Mode | `default`（可选 `never`） |
| Scheduler Mode | `timeslice`，Interval 5000 µs，Yield Timeout 0 |
| Performance Factor | compute 50 / media 50 |
| Memory ECC | enabled（pending: enabled） |
| Xe Link Ports | Up: 1,2,3,4,5,6；Beaconing Off: 1,2,3,4,5,6 |

**空闲实测**：GPU 功耗 43 W，核心频率 1550 MHz，Core 温度 38 ℃，Memory 温度 37 ℃，利用率 0%。

> 可用于扫「功耗-性能曲线」：`--powerlimit` 150→300 W，`--frequencyrange` 200→1550 MHz。

---

## 4. 当前空载遥测快照（基线）

```
GPU Utilization (%)        : 0
Compute Engine Util (%)    : Engine 0-3: 0
Copy Engine Util (%)       : Engine 0-5: 0
GPU Power (W)              : 43
GPU Frequency (MHz)        : 1550
GPU Core Temperature (C)   : 38
GPU Memory Temperature (C) : 37
GPU Memory Read (kB/s)     : 580
GPU Memory Write (kB/s)    : 584
GPU Memory Used (MiB)      : 22
GPU Memory Bandwidth (%)   : 0
Xe Link Throughput (kB/s)  : N/A
EU Array Active/Stall/Idle : N/A   ← 未加载 kernel 时为 N/A
```

> 建议把所有测试的**空闲基线**与**满载峰值**都归档，便于对比异常。
