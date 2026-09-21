# ④ 互连测试（Xe Link / GPU-aware MPI / oneCCL）

**优先级：P1**
**文档位置：** `docs/TODO/04-interconnect-xelink.md`

---

## 1. 测试目标

1. **验证 P2P 是否真正可用**（Xe Link 直连 vs 退化为 PCIe + host 中转）
2. 实测 **Xe Link 卡间带宽与延迟**
3. 实测 **host↔device 带宽**（PCIe Gen5 x16）
4. 测试 **集合通信**性能（allreduce / allgather / broadcast）
5. 量化 **Xe Link 未标定**对实测的影响
6. 为 ⑤ AI 双卡扩展效率提供**链路侧解释**

## 2. 为什么这是本机最值得测的方向

拓扑实测显示：
```
GPU 0/0  S        XL24      0-143
GPU 1/0  XL24     S         0-143
```
- **XL24 = 6 端口 × 4 lane 全直连**，未经 MDF 中转 → **顶配互连**
- 理论单向总带宽 ≈ **304 GiB/s ≈ 318 GB/s**
- 对比 PCIe Gen5 x16 的 ~63 GB/s 原始带宽，**Xe Link 快约 5 倍**

这意味着：多卡场景下能否走 P2P，是性能的分水岭。**必须验证。**

## 3. 工具与准备

| 工具 | 状态 | 说明 |
|---|---|---|
| **`IMB-MPI1-GPU`** | ✅ **已装** `/opt/intel/oneapi/mpi/latest/bin/` | GPU-aware MPI 基准，**开箱即用，性价比最高** |
| **`IMB-RMA-GPU`** | ✅ 已装 | GPU-aware RMA |
| `IMB-MPI1` / `IMB-P2P` / `IMB-NBC` | ✅ 已装 | CPU 侧对照 |
| `fi_pingpong` | ✅ 已装 | libfabric 延迟/带宽 |
| Intel MPI | ✅ 2021.18 | |
| **oneCCL benchmarks** | ❌ 需构建 | allreduce / allgather 等 |
| **OSU Micro-Benchmarks** | ❌ 需编译 GPU-aware 版 | |
| PyTorch `torch.distributed` | ✅ +xpu | XPU backend（底层走 oneCCL） |
| Level Zero API | ✅ | P2P 能力查询 |

---

## 4. 执行步骤

### Step 1：P2P 可用性检查（**最关键的 Step**）

**方式 A：Level Zero 查询**
```bash
source /opt/intel/oneapi/setvars.sh
```
写一个小程序查：
- `zeDeviceCanAccessPeer(dev0, dev1, &canAccess)` → 是否支持 P2P
- `zeDeviceGetP2PProperties(...)` → P2P 属性

**方式 B：SYCL 查询**
```cpp
// 检查 device 的 extensions 是否包含 P2P 相关扩展
```

**方式 C：用 IMB-MPI1-GPU 间接验证**
```bash
export PATH=/opt/intel/oneapi/mpi/latest/bin:$PATH
# 如果 GPU-aware 生效，带宽应远高于 PCIe 带宽
mpirun -n 2 -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 IMB-MPI1-GPU PingPong
```
> **判据**：若 PingPong 带宽达到 **100+ GB/s 量级**，说明走的是 Xe Link；
> 若只有 **~50 GB/s**，说明退化到 PCIe；
> 若只有 **~25 GB/s**，说明经 host 中转（双向算一半）。
>
> ⚠️ 上述阈值是量级判断，需结合实测的 Xe Link / PCIe 带宽结果一起解读。

### Step 2：`IMB-MPI1-GPU` 卡间基准
```bash
export PATH=/opt/intel/oneapi/mpi/latest/bin:$PATH
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1      # 保证 PCI 顺序与 device index 一致

# PingPong（延迟 + 带宽 vs message size）
mpirun -n 2 IMB-MPI1-GPU PingPong 2>&1 | tee imb_pingpong.txt

# PingPing
mpirun -n 2 IMB-MPI1-GPU PingPing

# 集合通信
mpirun -n 2 IMB-MPI1-GPU Allreduce
mpirun -n 2 IMB-MPI1-GPU Allgather
mpirun -n 2 IMB-MPI1-GPU Broadcast
mpirun -n 2 IMB-MPI1-GPU Reduce
mpirun -n 2 IMB-MPI1-GPU Alltoall
```
记录 **不同 message size（0 B → 1 GiB）** 的带宽与延迟曲线。

### Step 3：host↔device 带宽（PCIe 对照）
用 CPU buffer 跑 IMB，得到 PCIe 路径基线：
```bash
mpirun -n 2 IMB-MPI1-GPU PingPong      # 禁用 GPU-aware 时得到 host 路径
# 或
mpirun -n 2 IMB-MPI1 PingPong          # 纯 CPU 侧 MPI
```
更直接的方法：用 SYCL 写 malloc_device ↔ malloc_host 的 memcpy 带宽测试。

### Step 4：libfabric 交叉验证
```bash
mpirun -n 2 fi_pingpong
fi_info                                # 查看可用的 fabric provider
```

### Step 5：oneCCL 集合通信（需构建）
```bash
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../_install
make -j && make install
# benchmark 位于 build/benchmarks
# 运行 allreduce
mpirun -n 2 ./benchmarks/benchmark --coll allreduce --backend sycl
```
关注：不同 message size / rank 数的 **bus bandwidth**（算法带宽需按通信量换算）。

### Step 6：PyTorch 分布式（端到端验证）
```bash
# 用 torch.distributed 的 xpu backend（底层 oneCCL）
# 跑一个 allreduce benchmark，扫 message size
```
记录：algbw / busbw、随 message size 的变化曲线、随 world_size（1 vs 2）的变化。

### Step 7：硬件计数对照
```bash
xpu-smi dump -d -1 -m 0,5,6,7 -i 100 -n 300 -j > xelink_telemetry.json
```
关注 **`Xe Link Throughput (kB/s)`** 指标（空载时为 N/A，需有实际 Xe Link 流量）。
> 这是**从硬件侧证明真的走了 Xe Link** 的独立证据。

### Step 8：Xe Link 端口缩放测试（可选深入）
```bash
xpu-smi config -d 0 -t 0 --xelinkport 1,0      # 关闭某端口
# 重跑 IMB，观察带宽是否按端口数线性缩放
xpu-smi config -d 0 -t 0 --xelinkport 1,1      # 恢复
```
⚠️ 谨慎操作，改完务必恢复。当前 6 端口全部 up。

---

## 5. 指标记录表

### 卡间（Xe Link 路径）
| Message Size | PingPong 带宽 (GB/s) | PingPong 延迟 (µs) |
|---|---|---|
| 1 B | | |
| 64 B | | |
| 1 KB | | |
| 64 KB | | |
| 1 MB | | |
| 16 MB | | |
| 256 MB | | |
| 1 GiB | | |

### host↔device（PCIe 路径）
| Message Size | 带宽 (GB/s) |
|---|---|
| ... | |

### 集合通信
| 操作 | 2 ranks 峰值 busbw (GB/s) |
|---|---|
| Allreduce | |
| Allgather | |
| Broadcast | |
| Alltoall | |

### 关键结论
| 项目 | 值 |
|---|---|
| P2P 是否可用 | 是 / 否 |
| 卡间实测峰值带宽 | ___ GB/s |
| 理论 Xe Link 带宽 | ~318 GB/s |
| **达成率** | ___ % |
| PCIe 实测带宽 | ___ GB/s |
| Xe Link / PCIe 比值 | ___ × |
| 硬件侧 Xe Link Throughput 对照 | 一致 / 不一致 |

---

## 6. 判读标准

| 现象 | 诊断 |
|---|---|
| P2P 不可用 | 驱动（i915 vs xe）或配置限制，**多卡性能会大幅损失** |
| 达成率 < 70% | **怀疑 Xe Link 未标定**（`Not Calibrated`） |
| 小消息延迟高 | launch / 同步开销；不影响大消息带宽 |
| 带宽随 message size 不收敛 | 算法或缓冲区问题 |
| 硬件计数与软件测量不符 | 可能未真正走 Xe Link，走了 PCIe/host |
| 双卡集合通信远低于预期 | oneCCL 配置 / P2P 未生效 |

### ⚠️ 特别关注：Xe Link 标定
```
Xe Link Calibration Date: Not Calibrated
```
若实测达成率明显偏低，**这是首要怀疑对象**。需向供应商确认：
- ES 样片上未标定是否正常
- 是否需要专用工具执行标定

---

## 7. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| → ⑤ AI 双卡 DDP | 扩展效率不理想的链路侧解释 |
| → ② 算力峰值 | 区分「单卡算力不足」与「多卡通信不足」 |
| → ⑧ 能效 | 通信开销占能耗的比例 |

---

## 8. 注意事项

1. **`ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`** 很重要：确保 device index 与 PCI BDF 顺序一致，否则 rank→GPU 映射可能错乱，测出来是「同卡自通信」。
2. 用 `xpu-smi ps` 确认两个 rank 分别落在 **device 0 和 device 1**，而不是同一张卡。
3. 只有 2 张卡 → `-n 2` 是上限；再多 rank 会在同一卡上竞争。
4. 单 NUMA 节点，无需考虑跨 NUMA 亲和性，但要注意 **host 内存只有 45 GiB**。
5. 改 `--xelinkport` 后**必须恢复**，否则后续所有多卡测试都受影响。
6. Intel MPI 的 GPU-aware 支持需确认环境变量（如 `I_MPI_OFFLOAD`、`MPI_OFFLOAD` 等），不同版本名称不同，查 `I_MPI_OFFLOAD=1` 相关文档。
