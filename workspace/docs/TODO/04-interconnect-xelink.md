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
mpirun -n 2 IMB-MPI1-GPU Bcast
mpirun -n 2 IMB-MPI1-GPU Reduce
mpirun -n 2 IMB-MPI1-GPU Alltoall
```
记录 **不同 message size（0 B → 1 GiB）** 的带宽与延迟曲线。

> ⚠️ **2026-09-22 实测修正（这一段原样照抄会踩三个坑）**：
> 1. **必须 `export I_MPI_OFFLOAD=1`**（外加 `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`）。
>    不设会 **L0 queue sync 报错 + GPU PTE NotPresent 段错误 + 进程被 SIGKILL/SIGABRT**。
>    已验证：见 `results/imb_gpu_no_offload.txt`。
> 2. **`-msglog 12 30`（空格）是错误语法** —— IMB 会把它当成第 3 个 benchmark 名字，
>    打印 `Invalid benchmark name 30` 后 **静默退回默认 4 KiB 上限**（所有数字卡在 ~42 GB/s）。
>    正确写法是 **`-msglog 12:30`**（冒号）。
> 3. **benchmark 名字是 `Bcast` 不是 `Broadcast`**（查 `IMB-MPI1-GPU -list`）。
>
> 另外：IMB 的**集合通信表只有 5 列**（`t_min / t_max / t_avg`，**不给 MB/s**），
> 解析时要自己按 `nbytes / t_avg` 折算，不能用 P2P 的 4 列表格式。

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

> ❌ **实测不可行**（2026-09-22）：`xpu-smi dump` 在 hwt 上**挂死**（rc=143）；
> 退化到 `xpu-smi stats -d 0` 后，**`Xe Link Throughput` 依然恒为 N/A** —— 即使 P2P 拷贝
> 正在跑满 95 GB/s（已用 `results/xpu_smi_stats_under_p2p.txt` 取证）。
> **结论：本机拿不到硬件侧的 Xe Link 独立证据**，只能靠「软件带宽 1.52× PCIe Gen5」这条反证
> 来证明走的不是 PCIe。见 §4.9 与 §6 裁定。

### Step 8：Xe Link 端口缩放测试（可选深入）
```bash
xpu-smi config -d 0 -t 0 --xelinkport 1,0      # 关闭某端口
# 重跑 IMB，观察带宽是否按端口数线性缩放
xpu-smi config -d 0 -t 0 --xelinkport 1,1      # 恢复
```
⚠️ 谨慎操作，改完务必恢复。当前 6 端口全部 up。

---

## 4.9 逐条覆盖核验（2026-09-22 收尾）

> 数据来源：`benchmark/04-interconnect-xelink/results/bench_20260922-200857.{json,md}`
> （**136 条记录**：`ccl` 55 / `imb` 50 / `p2p` 23 / `topo` 6 / `build` 1 / `xelink_telemetry` 1）。
> 详细解读见 [`../Conclusion/04-interconnect-xelink/README.md`](../Conclusion/04-interconnect-xelink/README.md)。

| TODO 条目 | 状态 | 证据 / 说明 |
|---|---|---|
| Step 1 方式 A（Level Zero 查询） | ✅ | `probes/p2p_probe.cpp`：`zeDeviceCanAccessPeer` **2/2 对为真**、`zeDeviceGetP2PProperties` → flags `["ACCESS","ATOMICS"]` |
| Step 1 方式 B（SYCL extension） | ⬜ 未做 | 方式 A 已给出确定性结论，B 属冗余验证 |
| Step 1 方式 C（IMB 间接验证） | ⚠️ **判据失效** | 实测 **43.71 GB/s**，落在「~50 GB/s ⇒ PCIe」与「~25 GB/s ⇒ 经 host」两个阈值之间 → **原判据的三档阈值不可用于本机**，必须改用「vs 实测 Xe Link / PCIe 比值」 |
| Step 2 `IMB-MPI1-GPU` 卡间基准 | ✅ | PingPong 19 点 + PingPing 19 点；**但必须加 `I_MPI_OFFLOAD=1` 与 `-msglog 12:30`**（见上） |
| Step 3 host↔device / CPU 对照 | ⚠️ 部分 | `cpu_peak.PingPong` **12.389 GB/s**；H2D/D2H 尺寸扫描由 **③** 覆盖（pinned 31.87 / pageable 25.5） |
| Step 4 libfabric 交叉验证 | ❌ **失败** | `fi_pingpong` **永久挂死**（`pingpong.c:463 bind() EADDRINUSE rc=-98`，`shm`/`tcp` 都挂，还留 `mpiexec.hydra` 僵尸）；`fi_info` 未跑 |
| Step 5 oneCCL benchmarks（需构建） | ⚠️ **替代** | 未构建 oneCCL benchmarks；改用 `probes/torch_ccl.py` 走 `torch.distributed` 的 **xccl** 后端（底层就是 oneCCL）：3 集合 × 2 dtype × 8 尺寸 |
| Step 6 PyTorch 分布式 | ✅ | 55 条记录（allreduce/allgather/broadcast × fp32/bf16） |
| Step 7 硬件计数对照 | ❌ **不可行** | `xpu-smi dump` 挂死；`xpu-smi stats -d 0` 的 `Xe Link Throughput` = **N/A**（P2P 跑满时也是 N/A） |
| Step 8 Xe Link 端口缩放 | ⬜ **未做（刻意）** | `Not Calibrated` 前提下改端口配置风险高，且只有 2 卡、无第二台机器可对照；改完若忘记恢复会污染后续所有多卡测试 |
| §5 指标记录表 | ✅ | 见下（已填） |
| §6 判读标准 | ✅ | 逐条裁定见 §6 表右列 |
| §8-1 `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` | ✅ 已设 | 并在 `torch_ccl.py` 里用 `ccl/device` 记录复核 rank→GPU 映射 |
| §8-2 用 `xpu-smi ps` 确认 rank 落卡 | ⚠️ 替代 | `xpu-smi ps` 不可靠；改用 torch 侧 device name + L0 BDF 交叉确认 |
| §8-3 只有 2 卡 → `-n 2` 上限 | ✅ 符合 | W = 2 |
| §8-5 改 `--xelinkport` 后必须恢复 | — 不适用 | 未执行 Step 8 |
| §8-6 确认 GPU-aware 环境变量名 | ✅ **关键发现** | 变量名就是 **`I_MPI_OFFLOAD=1`**；不设会崩（见 Step 2 的修正块） |
| （自加）PCIe `LnkSta` 自相矛盾取证 | ✅ 超出计划 | GPU 端点报 **2.5 GT/s ×1**，同链路上游桥报 **32 GT/s ×16** |
| （自加）`card1` 非 GPU 甄别 | ✅ 超出计划 | `card1` = ASPEED BMC VGA（`0x1a03:0x2000`），必须按 vendor `0x8086` 过滤 |
| （自加）Xe Link 静态规格 + 标定取证 | ✅ 超出计划 | 只在 `xpu-smi discovery -d 0` 里：**6 端口 × 4 lane × 50663.95 MiB/s = 318.8 GB/s** + `Not Calibrated` |

> 统计：**计划内 8 个 Step：3 项完全完成 / 2 项部分或替代 / 3 项未做或不可行**；
> **计划外新增 3 项取证**。所有缺口都在 §4.9 与 `Conclusion/04-…/README.md` §6 中如实记录。

---

## 5. 指标记录表

> ✅ 已填写。数据来自 `bench_20260922-200857.json`。
> ⚠️ **本目录有两条互不相同的曲线，不要混用**：`p2p` suite 是**裸 Level Zero 拷贝**（真实链路能力），
> `imb` suite 是 **GPU-aware MPI**（含 MPI 栈开销）。

### 卡间（Xe Link 路径）—— 裸 Level Zero（`p2p` suite）
| Message Size | 带宽 (GB/s) | 单向延迟 (µs) |
|---|---|---|
| 4 KiB | 0.64 | **6.44** |
| 64 KiB | 9.36 | — |
| 1 MiB | 60.06 | — |
| 4 MiB | 75.17 | — |
| 16 MiB | 92.13 | — |
| 64 MiB | 94.80 | — |
| 256 MiB | **95.51（峰值）** | — |
| 1 GiB | 未测（256 MiB 已收敛，spread 3.89%） | — |
| 1 B / 64 B / 1 KB | 未测（L0 探针最小 4 KiB） | — |

> 双向对称：`1→0` 也是 **95.51 GB/s**（同一尺寸）。`plateau_convergence`：6 个采样点
> `91.79 ~ 95.51`，**spread 3.89% → 已收敛**（不存在「带宽随 message size 不收敛」问题）。

### 卡间（GPU-aware MPI 路径）—— `IMB-MPI1-GPU PingPong`
| Message Size | 带宽 (GB/s) | 单向延迟 (µs) |
|---|---|---|
| 0 B | — | **0.43** |
| 4 KiB | 0.63 | 6.48 |
| 64 KiB | 5.86 | 11.18 |
| 1 MiB | 31.01 | 33.82 |
| 4 MiB | 40.98 | 102.34 |
| 16 MiB | **43.71（峰值）** | 383.84 |
| 64 MiB | 36.15（**回落**） | 1856.68 |
| 128 MiB | 35.28 | 3804.68 |
| 256 MiB | 34.79 | 7717.06 |
| 512 MiB | 34.96 | 15356.11 |

### host↔device（PCIe 路径）
| 路径 | 带宽 (GB/s) |
|---|---|
| pinned H2D（03 实测） | **31.87** |
| pinned D2H（03 实测） | 31.54 |
| pageable H2D（03 实测） | 25.50 |
| 纯 CPU MPI PingPong 峰值（`cpu_peak.PingPong`） | **12.389** @512 KiB |
| PCIe Gen5 ×16 理论 | 63.0 |

### 集合通信
| 操作 | 2 ranks 峰值 busbw (GB/s) | 可信度 |
|---|---|---|
| Allreduce | **71.238** @256 MiB（IMB） / **80.998** @1 GiB（xccl） | ✅ 两条路径一致 |
| Allgather | 2.994（IMB，❌ **不可信**） / **71.182**（xccl） | ⚠️ IMB 那条**软件病态**：8→16→32 MiB 数据 ×2 而耗时 ×3.6/~2.9，非线性 |
| Broadcast | 8.437（IMB，⚠️ 存疑，reps 掉到 1~2） / **94.589**（xccl，≈ 裸 P2P 带宽） | ⚠️ 取 xccl 值 |
| Alltoall | **68.641** @256 MiB（IMB） | ✅ 与 Allreduce 量级一致 |
| Reduce | 5.371（IMB，⚠️ 存疑，reps 掉到 1~2） | ⚠️ 存疑 |

### 关键结论
| 项目 | 值 |
|---|---|
| P2P 是否可用 | **是**（2/2 对；flags `ACCESS` + `ATOMICS`） |
| 卡间实测峰值带宽 | **95.51 GB/s** @256 MiB（双向对称） |
| 理论 Xe Link 带宽 | ~318 GB/s（6 端口 × 4 lane × 50663.95 MiB/s = 318.8） |
| **达成率** | **30.0%** |
| PCIe 实测带宽 | 31.87 GB/s（pinned H2D）；Gen5 ×16 理论 63.0 |
| Xe Link / PCIe 比值 | **1.52×**（vs Gen5 ×16 理论）；**3.0×**（vs 实测 pinned） |
| 硬件侧 Xe Link Throughput 对照 | **无法对照** —— 本驱动恒为 N/A |
| Xe Link 标定状态 | **`Not Calibrated`** ← 解释 30% 的**首要嫌疑**，需向供应商确认 |
| GPU-aware MPI 达成率 | 43.71 GB/s = **raw L0 的 45.8%**，且 **0.69× PCIe** → **第二瓶颈在 MPI 栈** |

---

## 6. 判读标准（含实测裁定）

| 现象 | 诊断 | 实测裁定（2026-09-22） |
|---|---|---|
| P2P 不可用 | 驱动（i915 vs xe）或配置限制，**多卡性能会大幅损失** | ❌ **不成立**：P2P **完全可用**（2/2 对，`ACCESS`+`ATOMICS`）。i915 驱动 + Production ES 也能开 P2P |
| 达成率 < 70% | **怀疑 Xe Link 未标定**（`Not Calibrated`） | ✅ **成立且未解决**：**30.0%**。已取证 `Not Calibrated`（唯一出处 `xpu-smi discovery -d 0`），但**本机无法自证因果关系**，需向供应商确认 318 GB/s 的测量基准 + ES 样片未标定是否正常 |
| 小消息延迟高 | launch / 同步开销；不影响大消息带宽 | ✅ **成立**：4 KiB 单向 6.44 µs；64 KiB 起进入带宽区。不影响大消息 |
| 带宽随 message size 不收敛 | 算法或缓冲区问题 | ❌ **不成立**：6 点 `91.79~95.51`，**spread 3.89% 已收敛** |
| 硬件计数与软件测量不符 | 可能未真正走 Xe Link，走了 PCIe/host | ⚠️ **无法判读**：硬件计数**根本不可用**（恒 N/A）。**改用反证法**：软件 95.51 GB/s = **1.52× PCIe Gen5 ×16** ⇒ 不可能是 PCIe ⇒ 是 Xe Link |
| 双卡集合通信远低于预期 | oneCCL 配置 / P2P 未生效 | ⚠️ **分实现**：`torch.distributed(xccl)` broadcast **94.59 GB/s ≈ 裸 P2P 的 99%** ⇒ oneCCL/P2P **没问题**；但 **GPU-aware MPI 的 Allgather/Reduce/Bcast 严重病态**（2.99 / 5.37 / 8.44 GB/s）⇒ **问题在 MPI 栈而非链路** |

> **本目录新增判据（计划里没有，实测必需）**：
> 1. **`raw L0 带宽` 必须与 `GPU-aware MPI 带宽` 分开报** —— 两者差 2.2×，混在一起会
>    把 MPI 栈的锅扣到 Xe Link 上。
> 2. **不要用 PCIe 档位阈值判路径**（原 Step 1 的 100/50/25 GB/s 三档在本机全部失效）。
>    唯一可靠判据是「**实测值 vs PCIe 理论上限的比值**」。
> 3. **IMB 的集合通信结果必须先做「线性度体检」再引用** —— reps 掉到 1~2、时间随数据非线性
>    增长的条目（Allgather/Reduce/Bcast）一律不可信。

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
