# benchmark/04-interconnect-xelink — Xe Link / GPU-aware MPI / oneCCL 互连实测

> 对应测试计划：[`docs/TODO/04-interconnect-xelink.md`](../../docs/TODO/04-interconnect-xelink.md)
> 最新结果：`results/bench_20260922-200857.{json,md}`（**136 条记录**）
> 结论文档：[`docs/Conclusion/04-interconnect-xelink/README.md`](../../docs/Conclusion/04-interconnect-xelink/README.md)

本目录回答一个问题：**两张 PVC 之间的数据到底能跑多快，瓶颈在链路还是在软件栈。**

## 0. 一句话

> 卡间带宽实测 **95.5 GB/s**：只有 Xe Link 标称 318 GB/s 的 **30%**，
> 但**已经是 PCIe Gen5 x16（63 GB/s）的 1.52 倍** ⇒ 确实走了 Xe Link，只是没跑满；
> 而 GPU-aware MPI 只能拿到 **43.7 GB/s**，说明**第二个瓶颈在 MPI 栈，不在链路**。

---

## 1. 方法论：三条互相独立的证据链

互连测试最容易犯的错，是拿一个数字直接下结论。本目录强制三条链分别测量：

| 链 | 工具 | 回答的问题 | 若这一条失败 |
|---|---|---|---|
| **A 能力** | `probes/p2p_probe.cpp`（Level Zero 原生） | 驱动层 P2P 到底开没开？ | 其它全部无意义，只能走 PCIe/host |
| **B 带宽** | 同一个 `p2p_probe`：跨卡 `zeCommandListAppendMemoryCopy` | **最干净的 Xe Link 带宽**（无 MPI、无框架、无 kernel 噪声） | 链路本身只有这个速度 |
| **C 端到端** | `IMB-MPI1-GPU` + `torch.distributed(xccl)` | 应用层实际能吃到多少 | MPI/CCL 没配对，不是链路问题 |

**判读排序**：

- B 高（>150 GB/s）而 C 低 → 链路没问题，是 MPI/oneCCL 没配对
- B 也低（~95 GB/s）→ 链路本身只有这个速度 → 怀疑 **未标定**
- A 为 false → 全盘退化到 PCIe / host

本机实测落在**第二种**：A 全开，B = 95.5 GB/s，C = 43.7（MPI）/ 81.1（CCL）。

### 口径说明（最容易搞错）

| 口径 | 定义 | 本目录用在哪 |
|---|---|---|
| 单向拷贝带宽 GB/s | 字节数 / 秒 | `P2PBW`、IMB PingPong |
| `algbw` | 消息字节 / 秒 | IMB 集合通信、CCL |
| `busbw` | `algbw × 2(W-1)/W` | CCL highlight |

> ⚠️ **本机 W=2，所以 `busbw ≡ algbw`**（系数 = 2×1/2 = 1）。
> 这是巧合，不要因此把 2 卡的 busbw 当惯例；3 卡以上必须换算。
> 也**不要**拿 allreduce 的 algbw 直接和 P2P 单向带宽比。

---

## 2. 目录结构

```
04-interconnect-xelink/
├── README.md                     ← 本文件
├── run_bench.py                  ← 5 个 suite 的驱动器
├── probes/
│   ├── p2p_probe.cpp             ← Level Zero 原生：能力查询 + 跨卡 memcpy 扫描
│   └── torch_ccl.py              ← torch.distributed(xccl) allreduce/allgather/broadcast
├── build/
│   └── p2p_probe                 ← g++ -O3 -std=c++17 -lze_loader
└── results/
    ├── bench_20260922-200857.{json,md}   ← ★ 最终产物（136 条）
    ├── p2p_probe_raw.txt                 ← 探针原始 stdout
    ├── torch_ccl.txt                     ← CCL rank0 stdout
    ├── imb_gpu_pingpong.txt              ← IMB-MPI1-GPU 点对点原始表
    ├── imb_gpu_coll.txt                  ← IMB-MPI1-GPU 集合通信原始表
    ├── imb_gpu_no_offload.txt            ← 不设 I_MPI_OFFLOAD 的崩溃现场
    ├── imb_cpu_pingpong.txt              ← 纯 CPU MPI 对照
    ├── xpu_smi_discovery.json            ← 拓扑溯源（device_list）
    ├── xpu_smi_discovery_d0.txt          ← ★ 含 "Xe Link Calibration Date"
    ├── xpu_smi_stats_under_p2p.txt       ← 加压时的 xpu-smi stats
    └── fi_pingpong.txt                   ← libfabric provider 尝试记录
```

---

## 3. 快速开始

```bash
cd /root/workspace/benchmark/04-interconnect-xelink

# 编译探针（需要 oneAPI 的 Level Zero loader）
source /opt/intel/oneapi/setvars.sh
g++ -O3 -std=c++17 -o build/p2p_probe probes/p2p_probe.cpp -lze_loader

# 全量跑（约 3.5 分钟）
nohup python3 run_bench.py > /tmp/04.log 2>&1 & disown
tail -f /tmp/04.log

# 只跑单个 suite
python3 run_bench.py topo
python3 run_bench.py p2p imb ccl
python3 run_bench.py --quick          # 冒烟：小消息上限
python3 run_bench.py p2p --max-bytes 67108864
```

`run_bench.py` 自己会调 `setvars.sh` 并注入 `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`；
**`ZE_AFFINITY_MASK` 会被主动 pop 掉**（否则 `mpirun` 的两个 rank 会看到同一张卡）。

---

## 4. 五个 suite 分别测什么

### 4.1 `topo` —— 静态拓扑取证（6 条）

- `/sys/class/drm/card*/device` 读 vendor/device/`current_link_speed`/`current_link_width`/`numa_node`
- **沿 `dev_dir.parent` 向上爬整条 PCI 链**（root port → switch upstream → switch downstream → 端点），逐个记录 `LnkSta`/`LnkCap`
- 非 Intel vendor 的 DRM 卡（本机 `card1` = ASPEED BMC VGA）记为 `*.non_gpu` 并 `skipped`
- `xpu-smi discovery -j`（键是 **`device_list`**）与 `xpu-smi discovery -d 0`（含 Xe Link 端口数与**标定日期**）
- 汇总记录 `pcie_link_report_conflict`

### 4.2 `p2p` —— Level Zero 能力 + 卡间带宽（23 条）

`probes/p2p_probe.cpp`，CLI `./p2p_probe [bw|no-bw] [max_bytes]`：

1. `zeInit` → `zeDriverGet` → `zeDeviceGet`（打印 `P2PDEV`）
2. 对每一对有序 GPU：`zeDeviceCanAccessPeer()` + `zeDeviceGetP2PProperties()`（打印 `P2PPEER`）
3. **同一个 context 里**把 src 分在 GPU0、dst 分在 GPU1，用 GPU1 的 queue 发
   `zeCommandListAppendMemoryCopy` —— 这是最干净的一次 DMA，直接走 Xe Link
4. 扫 4 KiB → 256 MiB，双向各 7 个点，输出 `P2PBW`

> 同一 context 内两卡内存互相可见，所以**不需要** `zeMemGetIpcHandle`/`zeMemOpenIpcHandle`。

### 4.3 `imb` —— GPU-aware MPI（50 条）

- `IMB-MPI1-GPU PingPong PingPing -msglog 12:30`：2 rank 各绑一张卡
- `IMB-MPI1-GPU Allreduce,Allgather,Bcast,Reduce,Alltoall -msglog 12:28`
- `IMB-MPI1`（纯 CPU）PingPong 作为对照
- **第一条记录是故意的反例**：不设 `I_MPI_OFFLOAD=1` 会直接段错误崩（见 §5.1）
- `fi_pingpong`（libfabric）作为独立第三方实现——本机**不可用**（见 §5.5）

### 4.4 `ccl` —— oneCCL / `torch.distributed`（55 条）

`probes/torch_ccl.py`，`torch.distributed.run --nproc_per_node=2`，
backend = **`xccl`**（oneCCL 的 torch 后端）：`all_reduce` / `all_gather_into_tensor` / `broadcast`，
`float32` + `bfloat16`，1 KiB → 1 GiB，`torch.xpu` 上跑。

### 4.5 `xelink_telemetry` —— 硬件侧取证（1 条）

P2P 加压**同时**读 `xpu-smi stats -d 0`，看 `Xe Link Throughput (kB/s)` 是否非零，
以便从硬件侧独立佐证「确实走了 Xe Link」。**本机读 N/A**（见 §5.6）。

---

## 5. 关键实测结果（tag `20260922-200857`）

### 5.1 能力：P2P 全开（证据链 A）

```json
{"ordered_pairs": 2, "accessible_pairs": 2,
 "pair_0_1": true, "pair_1_0": true,
 "p2p_flags": ["ACCESS", "ATOMICS"]}
```

| 设备 | 名字 | BDF | vendor/device |
|---|---|---|---|
| `device.0` | Intel(R) Data Center GPU Max 1100 | `0000:40:00.0` | `32902` (0x8086) / `3034` (0x0bda) |
| `device.1` | Intel(R) Data Center GPU Max 1100 | `0000:8c:00.0` | 同上 |

双向都 `can_access_peer=true`，flags = `ACCESS` + `ATOMICS`。
**注意**：Level Zero 1.24.0 里**没有** `ZE_DEVICE_P2P_PROPERTY_FLAG_P2P_COPY`
（CUDA 有 `cudaDevP2PAttrNativeAtomicSupported` 之类的细分；L0 只有这两个 flag），
所以不能从 flags 判断「能不能用 copy engine 做 P2P」，只能靠实测。

### 5.2 卡间带宽：95.5 GB/s = 标称的 30%（证据链 B，核心结果）

| size | 0→1 (GB/s) | 1→0 (GB/s) | 占 Xe Link 标称 | 相对 PCIe Gen5 |
|---|---|---|---|---|
| 4 KiB | 0.64 | 0.58 | 0.2% | 1.0% |
| 64 KiB | 9.36 | 8.67 | 2.9% | 14.9% |
| 1 MiB | 60.06 | 58.57 | 18.9% | 95.3% |
| 4 MiB | 75.17 | 77.13 | 23.6% | 119.3% |
| 16 MiB | 92.13 | 91.79 | 29.0% | 146.2% |
| 64 MiB | 94.80 | 94.63 | 29.8% | 150.5% |
| 256 MiB | **95.51** | **95.51** | **30.0%** | **151.6%** |

- `peak_overall`：`95.51 GB/s @ 256 MiB, direction=0->1, pct_of_spec=30.0, ratio_vs_pcie_gen5=1.516`
- `plateau_convergence`：`samples=6, min=91.79, max=95.51, spread=3.89%, converged=true` → 平台真实存在，不是噪声
- 4 KiB 单向延迟 ≈ 6.4 µs（0→1）/ 7.1 µs（1→0）

**判读**：95.5 GB/s 落在 `TH_XELINK_PARTIAL = 60` 与 `TH_XELINK_FULL = 200` 之间
→ **"走了 P2P 但明显不满"**。而它 `> PCIE_GEN5_X16 = 63`，所以**绝不是 PCIe**
（若是 Gen5 x16 理论峰值 63 GB/s，不可能测出 95.5）。

### 5.3 GPU-aware MPI 只有 43.7 GB/s（证据链 C，反直觉）

`gpu.PingPong`（GB/s，`mpirun -n 2 -genv I_MPI_OFFLOAD 1`）：

| size | GB/s | t (µs) |
|---|---|---|
| 0 | – | 0.43（纯延迟） |
| 4 KiB | 0.632 | 6.48 |
| 64 KiB | 5.864 | 11.18 |
| 1 MiB | 31.005 | 33.82 |
| 4 MiB | 40.983 | 102.34 |
| **16 MiB** | **43.709** ← peak | 383.84 |
| 32 MiB | 43.188 | 776.94 |
| 64 MiB | 36.145 | 1856.68 |
| 256 MiB | 34.785 | 7717.06 |
| 512 MiB | 34.961 | 15356.11 |

`gpu_peak.PingPong`：`43.709 @ 16 MiB, pct_of_spec=13.7%, ratio_vs_pcie_gen5=0.69, verdict="疑似 PCIe/host"`

> ★ **这是本目录最重要的发现**：
> **GPU-aware MPI 的峰值只有裸 Level Zero memcpy 的 46%**（43.7 / 95.5），
> 而且**连 PCIe Gen5 都没跑满**（0.69×）。同一对卡、同一条链路，
> 换一个软件通路就掉了 2.2 倍 ⇒ **瓶颈在 MPI 栈，不在 Xe Link。**
> `verdict` 字段里的 "疑似 PCIe/host" 是**自动判据的文字**（它只看 `ratio<1`
> 就推测走了 PCIe），**不要照抄**：证据链 B 已证明链路是 Xe Link，
> 这里的正确解读是 **MPI staging 开销**。
> 另外 64 MiB 以上还**回落**（43.7 → 34.8），说明大消息走了额外的中转/分段。

### 5.4 GPU-aware MPI 集合通信：只有 Allreduce/Alltoall 可信

| 集合操作 | 峰值 GB/s | 占标称 | 可信度 |
|---|---|---|---|
| `Allreduce` | **71.238** @ 256 MiB | 22.4% | ✅ 曲线线性（t ≈ 14.2 ns/B + 24 µs） |
| `Alltoall` | **68.641** @ 256 MiB | 21.6% | ✅ 曲线线性 |
| `Bcast` | 8.437 @ 32 MiB | 2.7% | ❌ 不可信 |
| `Reduce` | 5.371 @ 128 MiB | 1.7% | ❌ 不可信 |
| `Allgather` | **2.994** @ 8 MiB | **0.9%** | ❌ **明显是软件病态路径** |

> ⚠️ **Allgather 的 2.99 GB/s 不是链路性质，是 IMB 实现问题。** 原始表：
> 8 MiB → 2802 µs（2.99 GB/s），16 MiB → 10125 µs（1.66 GB/s），
> 32 MiB → 29077 µs（1.15 GB/s）。**数据翻倍、耗时翻 3.6 倍**，
> 完全非线性；而同一时刻 `Alltoall`/`Allreduce` 在同样尺寸上是线性的。
> `Bcast`/`Reduce` 的大消息段也一样可疑（`#repetitions` 掉到 1~2，单次计时被固定开销污染）。
> **结论：GPU-aware MPI 的集合通信只引用 Allreduce / Alltoall 的数字。**

### 5.5 oneCCL / torch.distributed(xccl)：81 GB/s（最快的端到端通路）

| 操作 | bfloat16 | float32 | 占标称 | at size |
|---|---|---|---|---|
| `allreduce` | **81.09** | **80.998** | 25.5% | 1 GiB |
| `broadcast` | **94.551** | **94.589** | 29.7% | 1 GiB |
| `allgather` | 71.290 | 71.182 | 22.4% | 1 GiB |

float32 曲线（GB/s）：

| size | allreduce | allgather | broadcast |
|---|---|---|---|
| 1 KiB | 0.007 | 0.013 | 0.012 |
| 64 KiB | 0.878 | 0.887 | 0.794 |
| 1 MiB | 6.145 | 6.809 | 6.131 |
| 16 MiB | 52.719 | 48.255 | 57.598 |
| 64 MiB | 71.925 | 64.261 | 82.965 |
| 256 MiB | 79.158 | 69.214 | 91.208 |
| 512 MiB | 80.208 | 70.533 | 93.565 |
| 1 GiB | 80.998 | 71.182 | 94.589 |

**判读**：oneCCL 的 `broadcast` 达 **94.6 GB/s ≈ 裸 L0 memcpy 的 95.5**，
说明**软件栈不是不可逾越的**——同一条链路，xccl 能把 broadcast 跑到 94.6，
而 IMB 的 PingPong 只有 43.7。**`broadcast` ≈ 裸 P2P 带宽，就是"链路满速"的端到端证据。**

### 5.6 纯 CPU MPI 对照

`cpu_peak.PingPong` = **12.389 GB/s @ 512 KiB**，延迟 0.43 µs。
走的是同机 shared memory，所以 12.4 GB/s 是**主机内存带宽/MPI 拷贝**的限制，
和 Xe Link 无关，只作基线。

### 5.7 静态拓扑：Xe Link 满配，但**未标定**

```json
{"xe_link_ports": 6, "lanes_per_port": 4, "mbps_per_port": 50663.95,
 "calibration_line": "Not Calibrated", "calibrated": false,
 "spec_gbps_per_dir": 318.8}
```

- `xpu-smi discovery -d 0` 显示：6 端口 × 4 lane，每端口 50663.95 MiB/s
  → 6 × 50663.95 MiB/s × 2²⁰ / 10⁹ = **318.8 GB/s/方向**（与 `docs/interconnect.md` 的 318 一致）
- 拓扑是 `XL24`（**直连，不经 MDF**）——最好的一档
- ★ **`Xe Link Calibration Date: Not Calibrated`** ——
  这是解释一切「低于 318 GB/s」的**首要前提**，也是本测试最重要的 caveat

### 5.8 硬件侧取证失败

`xelink_telemetry/under_p2p_load`（加压时同时采样）：

```
Xe Link Throughput (kB/s)   | N/A
GPU Memory Read (kB/s)      | 580
GPU Memory Write (kB/s)     | 584
GPU Utilization (%)         | 0
GPU Frequency (MHz)         | 1550
```

- `Xe Link Throughput` = **N/A** → **无法从硬件侧独立确认"确实走了 Xe Link"**
- 显存读写计数器 580/584 kB/s 也是**假的**（P2P 正在以 95 GB/s 拷数据，不可能是 0.6 MB/s）
- 所以 §5.2 的"不是 PCIe"只能靠**带宽数值本身**推断（95.5 > 63）

### 5.9 PCIe 上报自相矛盾（★ 不要相信 sysfs 的 `LnkSta`）

| 位置 | BDF | 上报速率 × 宽度 |
|---|---|---|
| GPU 端点 (`card0`) | `0000:40:00.0` | **2.5 GT/s × 1** |
| switch downstream | `0000:3f:01.0` | … |
| switch upstream | `0000:3e:00.0` | **32.0 GT/s × 16** |
| root port | `0000:3d:02.0` | … |

GPU1 完全对称：`8c:00.0`（端点，2.5×1）→ `8b:01.0` → `8a:00.0`（32×16）→ `89:02.0`。

**这两段物理上是同一条链路**，不可能一端 Gen1 x1 另一端 Gen5 x16。

铁证：`03-memory-bandwidth` 实测 **H2D pinned = 31.87 GB/s**，Gen1 x1 的理论上限是 0.5 GB/s，
所以 **GPU 端点自己的 `LnkSta` 上报是错的**（ES 样片/固件的已知问题）。

> ★ **规矩：本机判断 host↔device 带宽，只信实测，不信 `lspci`/`sysfs` 的 `LnkSta`。**
> `run_bench.py` 因此专门输出 `pcie_link_report_conflict` 一条汇总记录。

### 5.10 结果统计

| suite | 记录数 |
|---|---|
| `ccl` | 55 |
| `imb` | 50 |
| `p2p` | 23 |
| `topo` | 6 |
| `build` | 1 |
| `xelink_telemetry` | 1 |
| **合计** | **136** |

只有 3 条非 `ok`，且**全部是显式跳过并写明原因的**：
`topo/card1.non_gpu`（BMC VGA）、`imb/fi_pingpong`（libfabric 挂起）、
`xelink_telemetry/under_p2p_load`（计数器 N/A）。

---

## 6. 坑与注意事项（全是实测踩出来的）

### 6.1 ★★ `I_MPI_OFFLOAD=1` 是**必设**的，否则直接崩

`IMB-MPI1-GPU` 不设这个变量时：

```
Segmentation fault from GPU ... PTE NotPresent Write
level_zero_proxy.zeCommandQueueSynchronize ...
drm_neo.cpp:288 ...
```

退出码 **255**，且会连带把 hydra 的清理也搞崩。本目录把它做成了**第一条记录**
`imb/env.I_MPI_OFFLOAD_required`（`crashes_without=true`），现场留在 `results/imb_gpu_no_offload.txt`。

`run_bench.py` 里的 `GPU_MPI_GENV`：

```python
{"I_MPI_OFFLOAD": "1", "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1"}
```

### 6.2 ★★ `-msglog 12 30` 是**错误语法**，会静默退回 4 KiB 上限

IMB 要的是 `-msglog min:max`（**冒号**）。写成空格分隔时，
IMB 把 `30` 当成**第三个 benchmark 名字**，打印 `Invalid benchmark name 30`，
然后**不报错地**退回默认消息上限（4 KiB）。

后果：所有 IMB 带宽数字卡在 ~42 GB/s 上不去，
而且**看不出哪里错了**（这就是早期版本 IMB 全部只到 4 KiB 的原因）。

```python
PINGPONG_EXTRA = ["-msglog", "12:30"]   # 4 KiB .. 1 GiB
COLL_EXTRA     = ["-msglog", "12:28"]
```

### 6.3 ★ 集合通信表是 **5 列**，且 IMB **不给 MB/s**

```
       #bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]
```
IMB 对集合通信**只输出时间，没有 MB/s 列**（点对点才有）。
必须自己算：`algbw = bytes / t_avg_usec`（B/µs ≡ MB/s）。
`run_bench.py` 用两套正则（`_IMB_ROW_P2P` 4 列 / `_IMB_ROW_COLL` 5 列）分别匹配。

### 6.4 ★ benchmark 名字是 `Bcast`，不是 `Broadcast`

`Broadcast` → `Invalid benchmark name broadcast`。全集用
`IMB-MPI1-GPU -list` 查（本机还有 `Allgatherv`、`Gather`、`Scatter`、`Barrier`、`Reduce_scatter` …）。

### 6.5 ★ `fi_pingpong` 会**永久挂死**（不是慢，是挂死）

```
bind(): util/pingpong.c:463 , ret=-98 (Address already in use)
```

之后进程**永远不动**，还会留下孤儿 `mpiexec.hydra` + 僵尸 `fi_pingpong`，
把整个 `run_bench.py` 一起拖死（第一次跑就是这样卡了半小时）。

`shm` 和 `tcp` 两个 provider 都失败。现在改为：显式 `-p shm` → `-p tcp`，
**`timeout=60`**，失败就 `store.skip` 写原因 —— **绝不允许无限阻塞**。

### 6.6 ★ `xpu-smi` 的 Xe Link 计数器是 N/A，且 `dump` 会挂

- `xpu-smi stats -d 0` 的 `Xe Link Throughput` = **N/A**
- 显存读写计数器是**假的**（恒定 ~580 kB/s）
- `xpu-smi dump`（尤其 `-m 5,6,7`）在本驱动上会**挂死**（rc=143）→ **不要用**
- 只有 `xpu-smi stats -d 0`（~1.5 s）和 `xpu-smi discovery [-j|-d 0]` 安全

### 6.7 ★ Xe Link 端口数/标定日期只在 `xpu-smi discovery -d 0` 里

- `xpu-smi discovery`（无 `-d`）→ **没有** Xe Link 字段
- `xpu-smi discovery -j` → 只有设备/BDF，**没有**标定信息
- `xpu-smi topology -d 0` → 只有 CPU 亲和和 PCIe switch，**没有**标定信息

用错命令会**静默 skip**，从而丢掉本目录最重要的一条 caveat。
`results/xpu_smi_discovery_d0.txt` 是原始证据。

### 6.8 ★ `xpu-smi discovery -j` 的键是 `device_list`

不是 `devices`。写错的后果是 `device_count=0` —— **看起来像"没有 GPU"**，
不会抛异常。

### 6.9 `card1` 不是 GPU，是服务器的 BMC VGA

`/sys/class/drm/card1` → `0000:1b:00.0`，vendor `0x1a03`（**ASPEED**）device `0x2000`。
它和两张 Intel 卡一样出现在 `/sys/class/drm/` 下。必须用 `0x8086` 过滤，
否则拓扑表里会多一条假的"GPU"。映射关系：

| DRM | BDF | vendor:device | 是什么 |
|---|---|---|---|
| `card0` | `0000:40:00.0` | `0x8086:0x0bda` | GPU 0 |
| `card1` | `0000:1b:00.0` | `0x1a03:0x2000` | **BMC VGA** |
| `card2` | `0000:8c:00.0` | `0x8086:0x0bda` | GPU 1 |

### 6.10 两个 rank 的 stdout 是**共享**的，会把 JSON 切断

`torch.distributed.run` 下两个 rank 的 stdout 都指向同一个 fd。
两边都 `print` 时行与行会互相插队，把 `CCL <json>` 从中间切断，
`parse_json_lines` 只能**静默丢弃**（早期实测丢了约 1/3 的 CCL 记录）。

修法：只在 `RANK == "0"` 时 emit。

```python
def emit(tag, obj):
    if os.environ.get("RANK", "0") != "0":
        return
    print(f"{tag} {json.dumps(obj, ensure_ascii=False)}", flush=True)
```

### 6.11 `torchrun` console script 在 venv1 里**不存在**

`venv1` 是 `uv venv --system-site-packages` 建的，没有 `torchrun` 可执行文件。
必须走模块形式：

```bash
/root/workspace/venv1/bin/python -m torch.distributed.run --nproc_per_node=2 --standalone probes/torch_ccl.py
```

### 6.12 `dist.all_gather_into_tensor` 的 output 必须自己先分配

`all_gather_into_tensor(t, t)` → `output tensor size must be equal to world_size times input tensor size`。
要 `out = torch.empty(nelem * world, dtype=..., device=...)`。

### 6.13 oneAPI 环境脚本返回非 0

`source /opt/intel/oneapi/setvars.sh` 返回 **rc=3**。
必须 `set +u`，且**不要**用 `&&` 串在后面。`run_bench.py` 里用 `sub_env()` 统一处理。

---

## 7. 判读标准对照（vs `docs/TODO/04-interconnect-xelink.md` §6）

| 计划里的判据 | 实测 | 裁定 |
|---|---|---|
| `P2P 不可用` | 2/2 pair 均可访问，flags `ACCESS`+`ATOMICS` | ✅ **P2P 可用** |
| `达成率 < 70%` → 怀疑未标定 | 裸 L0 **30.0%**，CCL 最好 29.7% | ⚠️ **成立**：`Xe Link Calibration Date: Not Calibrated`，标定问题是首要嫌疑 |
| `小消息延迟高` | 裸 L0 4 KiB 单向 ~6.4 µs；IMB 0 字节 0.43 µs | ✅ 正常量级（不是 NVLink 级，但没有异常） |
| `带宽不收敛` | 64/128/256 MiB 平台：spread **3.89%**，`converged=true` | ✅ **收敛**，是真实平台值，不是噪声 |
| `硬件计数不符` | `Xe Link Throughput = N/A`，显存计数器假 | ⚠️ **无法判定**（硬件侧无取证能力） |
| `双卡集合通信远低于预期` | xccl broadcast **94.6 GB/s** ≈ 裸 P2P 95.5；IMB Allreduce 71.2 | ⚠️ **部分成立**：集合通信本身能到 94.6（说明链路可用），但 GPU-aware MPI 的 PingPong 只有 43.7 |

### 本目录新增的判据（计划里没有）

| 新判据 | 结果 |
|---|---|
| 带宽 **> 63 GB/s（PCIe Gen5 x16 原始）** | 95.5 ✅ ⇒ **绝不是 PCIe**，确实走了 Xe Link |
| CCL `broadcast` 是否 ≈ 裸 L0 P2P | 94.6 vs 95.5 → **99%** ✅ ⇒ 链路在应用层可被吃满 |
| GPU-aware MPI PingPong vs 裸 L0 | 43.7 / 95.5 = **46%** ❌ ⇒ **MPI 栈是第二瓶颈** |

---

## 8. 与其它目录的衔接

| 目录 / 文档 | 关系 |
|---|---|
| [`02-compute-peak`](../02-compute-peak/README.md) | 算力峰值。多卡扩展的理论上限受**本目录的通信带宽**约束 |
| [`03-memory-bandwidth`](../03-memory-bandwidth/README.md) | **H2D pinned = 31.87 GB/s** 是本目录判定「PCIe `LnkSta` 上报不可信」的**关键证据**；另外双卡 HBM 带宽 797 GB/s 说明 P2P 的 95.5 GB/s 是**跨卡链路**限制，不是显存限制 |
| [`05-ai-dl`](../05-ai-dl/README.md) | DDP 扩展效率（2 卡）直接由本目录的 allreduce/broadcast 带宽决定 |
| `docs/interconnect.md` | Xe Link XL24 拓扑、318 GB/s 标称值、`Not Calibrated` 的**原始出处** |
| `docs/caveats.md` | 「未标定 Xe Link」风险条目 |
| [`docs/Conclusion/04-interconnect-xelink/README.md`](../../docs/Conclusion/04-interconnect-xelink/README.md) | 本目录的**结论文档** |

---

## 9. 未做 / 待补充

1. **Xe Link 标定流程**：`Not Calibrated` 是**首要嫌疑**，但本机没有找到可执行的标定入口
   （`xpu-smi` 无 calibrate 子命令）。**需要向供应商确认 ES 样片是否需要/能否标定。**
2. **硬件侧佐证缺失**：`Xe Link Throughput` 读 N/A，无法独立确认流量确实走了 Xe Link。
   也许 `jq`/`xpu-smi dump` 的其它 metric id 能读到，但 `dump` 会挂死，未深挖。
3. **单向 vs 双向**：只测了单向。Xe Link 双向并发（两卡同时互发）是否掉速未测。
4. **≥3 卡**：本机只有 2 卡，无法测 ring allreduce 的 `W/(2(W-1))` 系数效应。
5. **IMB Allgather/Bcast/Reduce 的病态**：只定位到"非线性"，**没有定位到具体代码路径**。
6. **`fi_pingpong`**：本机 libfabric 不可用，需排查 provider 配置（可能与本机 `shm` 端口占用有关）。
7. **oneCCL 调优**：只用了默认配置（`FI_PROVIDER`/`CCL_*` 全部未调），
   `allreduce` 80.998 vs `broadcast` 94.589 的差距（17%）可能可以通过调优抹平。
8. **NVMe / 网络**：`docs/TODO/06-hpc-apps.md` 需要的存储带宽未测。
