# 互联情况

## 1. 拓扑总览

```
         ┌──────────────┐   Xe Link XL24    ┌──────────────┐
         │   GPU 0/0    │◄═════════════════►│   GPU 1/0    │
         │  Max 1100    │   6 port × 4 lane │  Max 1100    │
         │  48 GiB HBM  │   ≈ 318 GB/s 单向 │  48 GiB HBM  │
         └──────┬───────┘                   └──────┬───────┘
                │ PCIe 5.0 x16                      │ PCIe 5.0 x16
                │ (0000:40:00.0, PCIE_6)            │ (0000:8c:00.0, PCIE_4)
                └───────────────┬───────────────────┘
                                │
                        ┌───────┴────────┐
                        │  CPU (72c/144t)│
                        │  1 NUMA node   │
                        │  45 GiB DRAM   │
                        └────────────────┘
```

## 2. `xpu-smi topology -m` 实测输出

```
         GPU 0/0  GPU 1/0  CPU Affinity
GPU 0/0  S        XL24     0-143
GPU 1/0  XL24     S        0-143
```

**图例解读**：
| 标记 | 含义 |
|---|---|
| `S` | Self（自身） |
| `XL24` | 两个卡上的 tile 通过 **Xe Link 直连**，lane 数 = 24 |
| `XL*` | 通过 Xe Link + MDF 连接（非直连） |
| `MDF` | 通过 Multi-Die Fabric Interface 连接 |
| `NODE` | 同一 NUMA 节点内通过 PCIe 连接 |
| `SYS` | 跨 NUMA 节点通过 PCIe 连接 |

### 结论
- **`XL24` 是最好的一档**：两卡直连，未经过 MDF 中转（不是 `XL*`）。
- 24 lane = **6 个 Xe Link 端口全部启用，每端口 4 lane**（`xpu-smi discovery` 显示 `Number of Xe Link ports: 6`、`Number of Lanes per Xe Link port: 4`）。
- 即 **Xe Link 满配**，没有端口被裁掉。
- 单端口速率 `Max Tx/Rx Speed per Xe Link port: 50663.95 MiB/s` ≈ **49.5 GiB/s ≈ 53 GB/s**
- 理论单向总带宽 ≈ 6 × 50.66 GiB/s ≈ **304 GiB/s ≈ 318 GB/s**（双向约 636 GB/s）
- 两卡 CPU Affinity 均为 `0-143`（唯一 NUMA 节点，无跨节点差异）

### ⚠️ 需要注意的异常
```
Xe Link Calibration Date: Not Calibrated
```
**Xe Link 未标定**。这可能导致实测带宽达不到理论上限。建议：
- 实测前先确认是否需要执行标定流程
- 测试结果若明显低于 318 GB/s，先怀疑标定问题，而非硬件缺陷
- 在 ES 样片上这可能是已知状态，需向供应商确认

## 3. 主机↔卡 通道

| 项目 | 值 |
|---|---|
| PCIe Generation | **5** |
| PCIe Max Link Width | **x16** |
| 理论单向带宽 | 约 63 GB/s（PCIe 5.0 x16 ≈ 64 GB/s 原始） |
| 实测预期 | 约 50–55 GB/s（受协议开销影响，需实测确认） |

> 对比：Xe Link 单端口就已达 ≈53 GB/s，**卡间 Xe Link 带宽远优于走 PCIe 的主机通道**。
> 因此多卡场景下，若通信能走 P2P（Xe Link），性能会显著优于经 host 中转。

## 4. 多卡通信路径（重要性排序）

| 路径 | 带宽 | 适用 |
|---|---|---|
| **Xe Link P2P（XL24）** | ≈ 318 GB/s 单向 | 卡间集合通信、all-reduce、P2P tensor 交换 |
| **PCIe Gen5 x16** | ≈ 50–55 GB/s | host↔device 传输 |
| **经 host 内存中转** | ≈ PCIe 带宽 ×2（往返） | P2P 不可用时的退化路径 |

## 5. 测试关注点

1. **验证 P2P 是否真的可用**
   - Level Zero：`zeDeviceCanAccessPeer` / `zeDeviceGetP2PProperties`
   - SYCL：`device.get_info<sycl::info::device::extensions>()`、`has_extension("cl_intel_mem_channel_property")` 等
   - 若 P2P 未启用，卡间通信会退化到 PCIe + host 中转，性能差距可达数倍
2. **实测 Xe Link 带宽/延迟曲线**
   - 不同 message size（1 B → 1 GiB）的带宽与延迟
   - 越多 lane 参与的公理（all-reduce 比 point-to-point 更能压满）
3. **硬件计数交叉验证**
   - `xpu-smi dump` 的 `Xe Link Throughput (kB/s)` 指标
   - 注意：空载时为 `N/A`，需要实际有 Xe Link 流量才会出数
4. **Xe Link 端口开关对比**
   - `xpu-smi config -d 0 -t 0 --xelinkport [portId,value]`
   - 可做「6 端口 vs 部分端口」的带宽缩放测试
5. **未标定的影响量化**
   - 实测值 / 理论值 304 GiB/s，得到达成率；若 < 70% 需调查

## 6. 与 Aurora（Max 1550）的区别（参考）

| 项目 | Max 1100（本机） | Max 1550（Aurora） |
|---|---|---|
| Tile/卡 | 1 | 2 |
| Xe-core/卡 | 56 | 128 |
| EU/卡 | 448 | 1024 |
| HBM | 48 GiB | 128 GiB |
| 计算能力 | 约为 1550 的 **43.75%** | 基准 |

> 换算比例 448/1024 = 0.4375。同架构意味着 Aurora 上的 SYCL/oneAPI 移植经验可直接复用。
