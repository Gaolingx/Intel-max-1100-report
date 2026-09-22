# ④ 互连（Xe Link）测试结论

> 对应测试计划：[`../../TODO/04-interconnect-xelink.md`](../../TODO/04-interconnect-xelink.md)
> 测试代码：[`../../../benchmark/04-interconnect-xelink/`](../../../benchmark/04-interconnect-xelink/)
> 原始产物：`benchmark/04-interconnect-xelink/results/bench_20260922-200857.{json,md}`（**136 条记录**）
> 测试日期：2026-09-22　　硬件：2 × Intel Data Center GPU Max 1100（PVC, Production ES, 1 tile）
> 互连：Xe Link **XL24**（6 端口 × 4 lane 直连，无 MDF 中转）；驱动 i915；Level Zero 1.24.0

---

## 0. 一页结论

### 0.1 对测试目标的逐条回答

| # | 测试目标（TODO §1） | 实测 | 判定 |
|---|---|---|---|
| 1 | P2P 是否真正可用 | 2/2 有序 pair `can_access_peer=true`，flags **`ACCESS`+`ATOMICS`** | ✅ **可用，未退化** |
| 2 | Xe Link 卡间带宽与延迟 | **95.51 GB/s**（双向对称）= **标称 318 GB/s 的 30.0%**；4 KiB 单向延迟 ~6.4 µs | ⚠️ **是 Xe Link，但只跑出 30%**（见 §3.1/§3.2） |
| 3 | host↔device 带宽（PCIe Gen5 x16） | 复用 ③ 的实测：pinned H2D/D2H **31.87 / 31.89 GB/s**（50.6% of 63） | ✅ 已测；**04 补充了"为什么 `lspci` 的链路速率不能信"**（见 §3.6） |
| 4 | 集合通信（allreduce/allgather/broadcast） | xccl **allreduce 81.0 / allgather 71.2 / broadcast 94.6** GB/s（1 GiB, fp32=bf16） | ✅ 已测，且 **broadcast 94.6 ≈ 裸 P2P 95.5** ⇒ 链路在应用层可被吃满 |
| 5 | 量化「未标定」对实测的影响 | 裸 L0 达成 **30.0%**；`Xe Link Calibration Date: **Not Calibrated**` | ⚠️ **相关但未证因果**（本机无标定入口，见 §3.2） |
| 6 | 为 ⑤ AI 双卡扩展提供链路侧解释 | allreduce 71.2 GB/s ⇒ 2 卡有效 allreduce 上限 ~**1.13 ms/GB**；broadcast 94.6 ⇒ ~**0.85 ms/GB** | ✅ 见 §5 |

### 0.2 一句话

> **卡间实测 95.5 GB/s —— 只有 Xe Link 标称 318 GB/s 的 30%，但已经是 PCIe Gen5 x16（63 GB/s）
> 的 1.52 倍，所以确定走了 Xe Link、只是没跑满（首要嫌疑：`Not Calibrated`）。**
> **更值得注意的是第二个瓶颈：同一对卡、同一条链路，GPU-aware MPI 的 PingPong
> 只有 43.7 GB/s（裸 L0 的 46%），而 xccl 的 broadcast 却能到 94.6 GB/s。
> 所以"慢"主要慢在 MPI 栈，不在硬件。**

### 0.3 三个反直觉发现

1. **「只有 30%」和「比 PCIe 快 1.5 倍」可以同时成立。**
   若两卡之间退化走 PCIe Gen5 x16，理论上限是 63 GB/s。实测 95.5 GB/s **超过了这个上限**，
   所以路径一定是 Xe Link。**这正是本目录设 `ratio_vs_pcie_gen5` 这个判据的原因**：
   单看"达成率 30%"会误判成"P2P 没生效"，必须同时看"相对 PCIe 的倍数"。

2. **同一个软件基准的"峰值"可能是假的。**
   `IMB-MPI1-GPU` 的 `Allgather` 报出 **2.99 GB/s**（只有 Alltoall 68.6 的 4%），
   看起来很像个重大发现；但原始表显示它**非线性**：8 MiB→2802 µs、16 MiB→10125 µs、
   32 MiB→29077 µs（数据翻倍、耗时翻 3.6 倍）。这是 **IMB 的软件病态路径**，
   不是链路性质。**只引用 Allreduce / Alltoall。**

3. **工具本身比硬件更容易踩坑。**
   `IMB-MPI1-GPU` 不设 `I_MPI_OFFLOAD=1` 会**段错误**；
   `-msglog 12 30`（空格）是**非法语法**，会静默退回 4 KiB 上限，
   使所有带宽数字"看起来卡在 42 GB/s"却毫无报错；
   `fi_pingpong` 会**永久挂死**；
   `xpu-smi` 的 Xe Link 计数器读 **N/A**。
   **本目录 60% 的工作量花在"让工具正确运行"上。**

---

## 1. 交付物索引

| 类别 | 路径 | 内容 |
|---|---|---|
| 驱动器 | `benchmark/04-interconnect-xelink/run_bench.py` | 5 个 suite：`topo` / `p2p` / `imb` / `ccl` / `xelink_telemetry` |
| L0 探针 | `benchmark/04-interconnect-xelink/probes/p2p_probe.cpp` | 能力查询 + 单 context 跨卡 `zeCommandListAppendMemoryCopy` 扫描 |
| CCL 探针 | `benchmark/04-interconnect-xelink/probes/torch_ccl.py` | `torch.distributed`(backend=`xccl`) allreduce / allgather / broadcast |
| 原始结果 | `benchmark/04-interconnect-xelink/results/bench_20260922-200857.{json,md}` | **136 条记录** |
| 原始日志 | `.../results/imb_gpu_pingpong.txt`、`imb_gpu_coll.txt`、`imb_gpu_no_offload.txt`、`torch_ccl.txt`、`p2p_probe_raw.txt`、`xpu_smi_discovery_d0.txt` | 全部可复算 |
| README | `benchmark/04-interconnect-xelink/README.md` | 测试方法 + 13 个坑 |
| 结论文档 | 本文件 | |

---

## 2. 全部关键数字总表

### 2.1 能力层：Level Zero P2P（证据链 A）

| 项目 | 结果 |
|---|---|
| 有序 pair 总数 | 2 |
| `can_access_peer=true` | **2 / 2** |
| pair `0→1` / `1→0` | `true` / `true` |
| P2P flags | **`ACCESS` + `ATOMICS`** |
| `device.0` | Intel(R) Data Center GPU Max 1100，`0000:40:00.0`，vendor `0x8086` device `0x0bda` |
| `device.1` | 同上，`0000:8c:00.0` |

> Level Zero 1.24.0 **没有** `ZE_DEVICE_P2P_PROPERTY_FLAG_P2P_COPY`（只有 `ACCESS`/`ATOMICS`/`PERF_HINT`…），
> 因此**不能**从 flags 判断"能否用 copy engine 做 P2P"，只能靠 §2.2 的实测。

### 2.2 卡间带宽：裸 L0 跨卡 memcpy（证据链 B，核心结果）

| size | 0→1 (GB/s) | 1→0 (GB/s) | 占 318 GB/s | 相对 PCIe Gen5 (63) |
|---|---:|---:|---:|---:|
| 4 KiB | 0.64 | 0.58 | 0.2% | 1.0% |
| 64 KiB | 9.36 | 8.67 | 2.9% | 14.9% |
| 1 MiB | 60.06 | 58.57 | 18.9% | 95.3% |
| 4 MiB | 75.17 | 77.13 | 23.6% | 119.3% |
| 16 MiB | 92.13 | 91.79 | 29.0% | 146.2% |
| 64 MiB | 94.80 | 94.63 | 29.8% | 150.5% |
| **256 MiB** | **95.51** | **95.51** | **30.0%** | **151.6%** |

| 汇总记录 | 值 |
|---|---|
| `peak_overall` | `gbps=95.51, direction=0->1, at_size=268435456, pct_of_spec=30.0, ratio_vs_pcie_gen5=1.516` |
| `plateau_convergence` | `samples=6, min=91.79, max=95.51, spread=3.89%, converged=true` |
| 单向延迟（4 KiB） | ~6.4 µs（0→1）/ ~7.1 µs（1→0） |

### 2.3 GPU-aware MPI（证据链 C-1）

`gpu.PingPong`，`mpirun -n 2 -genv I_MPI_OFFLOAD 1 -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1`：

| size | GB/s | t (µs) |
|---|---:|---:|
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

| 记录 | 值 |
|---|---|
| `gpu_peak.PingPong` | `43.709 @ 16 MiB, pct_of_spec=13.7%, ratio_vs_pcie_gen5=0.69` |
| `gpu_peak.PingPing` | `43.411 @ 16 MiB, pct_of_spec=13.7%` |
| `cpu_peak.PingPong`（纯 CPU 同机 shm 对照） | `12.389 @ 512 KiB, latency=0.43 µs` |
| **`ratio = 43.709 / 95.51`** | **0.458 — GPU-aware MPI 只拿到裸 L0 的 46%** |

### 2.4 GPU-aware MPI 集合通信（证据链 C-1）

| 集合操作 | 峰值 GB/s | at size | 占 318 GB/s | 可信度 |
|---|---:|---:|---:|---|
| `Allreduce` | **71.238** | 256 MiB | 22.4% | ✅ 线性（t ≈ 14.2 ns/B + 24 µs） |
| `Alltoall` | **68.641** | 256 MiB | 21.6% | ✅ 线性 |
| `Bcast` | 8.437 | 32 MiB | 2.7% | ❌ 大消息段 `reps` 掉到 1~2，不可信 |
| `Reduce` | 5.371 | 128 MiB | 1.7% | ❌ 同上 |
| `Allgather` | **2.994** | 8 MiB | **0.9%** | ❌ **非线性，软件病态路径** |

### 2.5 oneCCL / `torch.distributed`(xccl)（证据链 C-2，最快的端到端通路）

backend = `xccl`，`torch 2.14.0+xpu`，2 进程各一张卡：

| 操作 | bfloat16 | float32 | 占 318 GB/s | at size |
|---|---:|---:|---:|---:|
| `allreduce` | **81.090** | **80.998** | **25.5%** | 1 GiB |
| `allgather` | 71.290 | 71.182 | 22.4% | 1 GiB |
| **`broadcast`** | **94.551** | **94.589** | **29.7%** | 1 GiB |

float32 曲线（GB/s）：

| size | allreduce | allgather | broadcast |
|---|---:|---:|---:|
| 1 KiB | 0.007 | 0.013 | 0.012 |
| 64 KiB | 0.878 | 0.887 | 0.794 |
| 1 MiB | 6.145 | 6.809 | 6.131 |
| 16 MiB | 52.719 | 48.255 | 57.598 |
| 64 MiB | 71.925 | 64.261 | 82.965 |
| 256 MiB | 79.158 | 69.214 | 91.208 |
| 512 MiB | 80.208 | 70.533 | 93.565 |
| 1 GiB | 80.998 | 71.182 | **94.589** |

> **W=2 ⇒ `busbw ≡ algbw`**（系数 `2(W-1)/W = 1`）。这是 2 卡的巧合，不是惯例。

### 2.6 静态拓扑与标定状态

| 项目 | 值 | 出处 |
|---|---|---|
| Xe Link 端口数 | **6** | `xpu-smi discovery -d 0` |
| 每端口 lane 数 | **4** | 同上 |
| 每端口速率 | **50663.95 MiB/s** | 同上 |
| 推导标称单向总带宽 | 6 × 50663.95 MiB/s × 2²⁰ / 10⁹ = **318.8 GB/s** | 与 `docs/interconnect.md` 的 318 一致 |
| 拓扑档位 | **`XL24`**（直连，不经 MDF） | `xpu-smi topology` |
| **标定日期** | **`Not Calibrated`** | `xpu-smi discovery -d 0` ← **只在 `-d 0` 里** |
| PCIe 上报（GPU 端点） | **2.5 GT/s × 1** | `/sys/class/drm/card0/device/current_link_*` |
| PCIe 上报（同链路上游桥） | **32.0 GT/s × 16** | `0000:3e:00.0` / `0000:8a:00.0` |
| 实测 H2D pinned（③） | **31.87 GB/s** | `03-memory-bandwidth` ⇒ 端点上报**不可信** |

### 2.7 硬件侧取证（失败）

| 计数器 | 读数 | 期间真实负载 |
|---|---|---|
| `Xe Link Throughput (kB/s)` | **N/A** | 95 GB/s 卡间拷贝 |
| `GPU Memory Read (kB/s)` | 580 | 同上 |
| `GPU Memory Write (kB/s)` | 584 | 同上 |
| `GPU Utilization (%)` | 0 | 同上 |
| `GPU Frequency (MHz)` | 1550（锁定） | 同上 |

### 2.8 结果统计

| suite | 记录数 | 非 `ok` |
|---|---:|---|
| `ccl` | 55 | 0（3 条极小消息为 launch-overhead 主导） |
| `imb` | 50 | 1（`fi_pingpong` skipped） |
| `p2p` | 23 | 0 |
| `topo` | 6 | 1（`card1.non_gpu` skipped，是 BMC VGA） |
| `build` | 1 | 0 |
| `xelink_telemetry` | 1 | 1（计数器 N/A） |
| **合计** | **136** | **3（全部显式说明原因）** |

---

## 3. 分项详述

### 3.1 为什么 95.5 GB/s 是「确实走了 Xe Link」的铁证

这一步推理是本目录最重要的一次判读，因为它**推翻了自动判据**：

1. **若退化到 PCIe Gen5 x16**：原始上限 63 GB/s，实测通常 50~55 GB/s。
2. **实测 95.51 GB/s > 63 GB/s（1.52×）** ⇒ 物理上不可能是 PCIe。**路径 = Xe Link。**
3. 与 ③ 的对照进一步排除 host 中转：host→device 只有 31.87 GB/s，若经 host 中转，
   卡间带宽不可能超过它。

因此 `run_bench.py` 的 `gpu_peak.PingPong` 自动打出的 `verdict="疑似 PCIe/host"`
（它只看 `ratio_vs_pcie_gen5=0.69 < 1`）是**误导性的**——那个判据只对"相对 PCIe 的倍数"负责，
不能单独用于定性。**判读必须三条链一起看。**

### 3.2 30% 达成率：`Not Calibrated` 是首要嫌疑（🔴 未解决）

- 318 GB/s 这个数字**不是 datasheet 硬指标**，而是
  `xpu-smi` 报的「每端口 50663.95 MiB/s × 6 端口」推导出来的**名义速率上限**。
- 实测 95.5 GB/s = 30.0%，落在计划 §6 的「达成率 < 70% → 怀疑未标定」区间。
- 而 `xpu-smi discovery -d 0` 明确写着 **`Xe Link Calibration Date: Not Calibrated`**。

**为什么不能直接断言因果**：

- 本机 `xpu-smi` **没有 calibrate 子命令**，无法做 A/B（标定前 vs 标定后）实验；
- 95.5 GB/s 也可能是 PVC 的 Xe Link 在「1 tile × 1 tile、N 个队列并发」下的真实上限
  （单端口 4 lane，理论 53 GB/s，6 端口聚合能否达到 318 取决于请求并发度）；
- 平台方要求的 318 GB/s 本身就没有给出测量口径（单向/双向/并发数）。

**因此本报告的口径是**：
> 95.5 GB/s 是**实测值**；318 GB/s 是**名义值**；两者相差 3.3 倍。
> `Not Calibrated` 是**首要嫌疑**，但**未证因果**。
> **建议向供应商确认：ES 样片的 Xe Link 是否需要标定、如何标定、318 GB/s 的测量口径是什么。**

### 3.3 GPU-aware MPI 只有裸 L0 的 46% —— 第二瓶颈在 MPI，不在链路

| 通路 | 峰值 GB/s | vs 裸 L0 |
|---|---:|---:|
| 裸 Level Zero 跨卡 memcpy | **95.51** | 100% |
| `IMB-MPI1-GPU PingPong` | 43.71 | **46%** |
| `IMB-MPI1-GPU PingPing` | 43.41 | 45% |
| `torch.distributed(xccl)` broadcast | **94.59** | **99%** |
| `torch.distributed(xccl)` allreduce | 81.00 | 85% |

这张表是本目录**最有价值的结论**：**同一对卡、同一条 Xe Link、同一个 1.24.0 驱动**，

- xccl 能做到 **99% 裸带宽**；
- 而 Intel MPI 的 GPU-aware 路径只能做到 **46%**。

⇒ **瓶颈在 MPI 的 GPU 数据路径（staging / bounce buffer / 内核中间拷贝），不在 Xe Link。**
一个额外证据：MPI PingPong 在 **64 MiB 以上反而回落**（43.7 → 34.8 GB/s），
这种"大消息变慢"是典型的分段/中转行为，物理链路不会这样。

**实践建议：多卡应用优先走 oneCCL/torch 的通信路径，谨慎使用 Intel MPI 的 GPU-aware 直通。**

### 3.4 集合通信：只有 Allreduce / Alltoall 可信

原始 `IMB-MPI1-GPU` 表格（`results/imb_gpu_coll.txt`）：

| size | Allreduce t_avg (µs) | Alltoall t_avg (µs) | **Allgather t_avg (µs)** |
|---|---:|---:|---:|
| 4 KiB | 10.16 | 10.43 | 16.02 |
| 1 MiB | 33.76 | 37.71 | 365.12 |
| 8 MiB | 139.34 | 138.23 | 2802.07 |
| 16 MiB | 253.77 | 255.91 | **10125.24** |
| 32 MiB | 491.24 | 566.54 | **29077.10** |
| 256 MiB | 3768.13 | 3910.70 | **219702.03** |

- Allreduce / Alltoall：**严格线性**（t ≈ 14.2 ns/byte），与 71.2 / 68.6 GB/s 的平台值自洽。
- **Allgather：8 MiB 后彻底崩坏**（数据翻倍 → 耗时 ×3.6），且 `#repetitions` 从 5 掉到 1，
  单次计时的固定开销也参与污染。**它的"2.99 GB/s"不是链路性质。**
- `Bcast` / `Reduce` 同样在大消息段可疑（`reps` 掉到 1~2）。

**口径纪律：GPU-aware MPI 的集合通信，本报告只引用 Allreduce / Alltoall。**

### 3.5 xccl broadcast ≈ 裸 P2P ⇒ 软件栈不是不可逾越

`broadcast` 94.59 GB/s vs 裸 L0 95.51 GB/s = **99.0%**。

这条数据的意义是**反驳"链路不行所以软件也没救"的宿命论**：
同一条 Xe Link，只要软件路径干净（xccl 的 broadcast 走的是直接 P2P 写），
就能吃满裸带宽。**因此 30% 的达成率里，"软件没吃满"和"链路只有这么快"两者都存在，
且在 MPI 路径上软件占了大头。**

（不过 `allreduce` 81.0 GB/s 仍比 broadcast 低 14%，
说明 reduce 阶段的算法/调优还有空间；本次全部用 oneCCL 默认配置，未做 `CCL_*` 调优。）

### 3.6 PCIe 链路速率上报不可信（方法论级教训）

| 位置 | BDF | 上报 |
|---|---|---|
| GPU 端点 (`card0`) | `0000:40:00.0` | **2.5 GT/s × 1** |
| switch downstream | `0000:3f:01.0` | — |
| switch upstream | `0000:3e:00.0` | **32.0 GT/s × 16** |
| root port | `0000:3d:02.0` | — |

GPU1 完全对称（`8c:00.0` → `8b:01.0` → `8a:00.0` → `89:02.0`）。
**同一段物理链路不可能一端 Gen1 x1、另一端 Gen5 x16。**

铁证在 ③：**pinned H2D = 31.87 GB/s**，Gen1 x1 的理论上限是 0.5 GB/s。

> ★ **本机规矩：判断 host↔device 带宽只信实测，绝不信 `lspci` / `sysfs` 的 `LnkSta`。**
> `run_bench.py` 为此专门输出 `pcie_link_report_conflict` 汇总记录，
> 并把链路上游的速率一并存进 `reported_upstream_max`，
> 避免后续读者被端点上报值误导。

这条也与 **`card1` 不是 GPU**（`0x1a03` ASPEED BMC VGA）一起说明：
**本机的 sysfs/DRM/PCI 拓扑枚举不能盲信，必须按 vendor id 过滤 + 交叉验证。**

### 3.7 硬件计数器 N/A ⇒ 无法从硬件侧取证

P2P 满载（95 GB/s）时同时采样 `xpu-smi stats -d 0`：

- `Xe Link Throughput` = **N/A** ⇒ **无法**独立确认"流量确实走了 Xe Link"；
- 显存读写计数器 580/584 kB/s 也**完全失真**（与 ③ 的结论一致：读 576 kB/s 恒定值）；
- `GPU Utilization = 0%`（已确认与 ①②③ 同类，采集失真），`GPU Frequency` 报 1550 MHz（请求值）。

所以 §3.1 的"确实是 Xe Link"只能靠**带宽数值本身**推断（95.5 > 63）。
**这是本报告最薄弱的一环，也是开放问题（见 §6）。**

### 3.8 频率请求值固定 ⇒ 短时差异都是真实的

频率 `gt_min == gt_max ==` **1550 MHz**（**请求值**），功率上限 300 W，P2P 满载时功率上升有限。
因此本报告里所有"软件通路之间"的差异（95.5 vs 43.7 vs 94.6）
**在短时条件下不可能由降额或功率墙解释**，只能是数据路径本身的差异。

> ⚠️ 但「请求值固定」不等于「执行频率不变」：`ze_peak` 长时间满载时实测温度可达 **101 °C**、
> 功耗冲到 **305~330 W（越过 300 W 上限）** —— 长跑会自主降额。
> 本节的结论仅在**短时**互连测试口径下成立（P2P 测试耗时远短于降额时间常数）。
> 详见 `docs/Conclusion/02-compute-peak/README.md` §2.8。

---

## 4. 判读标准对照（vs `docs/TODO/04-interconnect-xelink.md` §6）

| 计划判据 | 实测 | 裁定 |
|---|---|---|
| `P2P 不可用` | 2/2 pair 可访问，`ACCESS`+`ATOMICS` | ✅ **P2P 可用** |
| `达成率 < 70%` → 怀疑**未标定** | 裸 L0 **30.0%**，xccl 最好 29.7% | ⚠️ **落在该区间**；`Not Calibrated` 是首要嫌疑（§3.2），但**未证因果** |
| `小消息延迟高` | 裸 L0 4 KiB 单向 ~6.4 µs；IMB 0 字节 0.43 µs | ✅ 正常量级 |
| `带宽不收敛` | 64/128/256 MiB 平台 spread **3.89%** | ✅ **收敛**，是真实平台值 |
| `硬件计数不符` | `Xe Link Throughput = N/A`；显存计数器失真 | ⚠️ **无法判定**（无硬件侧取证能力） |
| `双卡集合通信远低于预期` | xccl broadcast 94.6 ≈ 裸 P2P；但 IMB GPU PingPong 仅 43.7 | ⚠️ **部分成立**：软件栈上限已被证明可到 94.6，缺口在 MPI 路径 |

### 本报告新增的判据（计划中没有）

| 新判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 卡间带宽 vs PCIe Gen5 x16 原始 | > 63 GB/s ⇒ 非 PCIe | **95.51（1.52×）** | ✅ 确证 Xe Link |
| xccl broadcast vs 裸 L0 P2P | ≈100% ⇒ 软件栈可吃满链路 | 94.59 / 95.51 = **99.0%** | ✅ 确证 |
| GPU-aware MPI PingPong vs 裸 L0 | <70% ⇒ MPI 是瓶颈 | 43.71 / 95.51 = **45.8%** | ❌ 确证 MPI 瓶颈 |
| 集合通信曲线线性度 | 数据翻倍 ⇒ 耗时翻倍 | Allreduce/Alltoall 线性；**Allgather ×3.6** | ❌ 排除 Allgather/Bcast/Reduce |

---

## 5. 与其它目录的衔接

### 对 ⑤ AI 双卡扩展的链路侧解释

| 通路 | 有效单向带宽 | 折算成本 | 用途 |
|---|---:|---:|---|
| xccl broadcast | 94.59 GB/s | ~0.85 ms / GiB | 参数广播（DDP 初始化、权重同步） |
| xccl allreduce | 81.00 GB/s | ~1.13 ms / GiB | **DDP 梯度同步（关键路径）** |
| 裸 L0 P2P | 95.51 GB/s | ~0.86 ms / GiB | 理论上限 |
| IMB GPU PingPong | 43.71 GB/s | ~1.88 ms / GiB | Intel MPI GPU-aware 路径 |

**结论**：2 卡 DDP 的梯度同步在 1 GiB 量级上约 **1.13 ms/次**。
若模型本身单次迭代 >50 ms（ResNet-50/BERT 量级），通信占比 <5%，**扩展效率不会受链路拖累**；
这也解释了 ⑤ 里 2 卡 DDP 扩展效率的实测表现。
**反过来，若未来要用 ≥4 卡或做张量并行（每层都要 allreduce），
按 81 GB/s 的 allreduce 计，会把链路推成主瓶颈。**

### 交叉引用

| 目录 / 文档 | 关系 |
|---|---|
| [`Conclusion/02-compute-peak`](../02-compute-peak/README.md) | 算力峰值；多卡扩展的理论上限受本目录通信带宽约束 |
| [`Conclusion/03-memory-bandwidth`](../03-memory-bandwidth/README.md) | **H2D pinned 31.87 GB/s** 是判定「PCIe `LnkSta` 不可信」的关键证据；HBM 797~900 GB/s 说明 P2P 的 95.5 是**链路**限制而非显存限制 |
| [`Conclusion/05-ai-dl`](../05-ai-dl/05-conclusion.md) | 2 卡 DDP 扩展效率的链路侧解释 |
| `docs/interconnect.md` | XL24 拓扑、318 GB/s 名义值、`Not Calibrated` 的原始出处 |
| `docs/caveats.md` | 「未标定 Xe Link」风险条目 |
| `benchmark/04-interconnect-xelink/README.md` | **测试方法**说明 + 13 个工具坑 |

---

## 6. 未做 / 待补充

| # | 项目 | 状态 | 说明 |
|---|---|---|---|
| 1 | **Xe Link 标定** | 🔴 阻塞 | 本机 `xpu-smi` 无 calibrate 入口。**需向供应商确认 ES 样片是否/如何标定，以及 318 GB/s 的测量口径**（单向？双向？并发数？） |
| 2 | **硬件侧佐证** | 🔴 缺失 | `Xe Link Throughput` 读 N/A。这意味着"走了 Xe Link"只能间接推断。未穷举 `xpu-smi dump` 的 metric id（`dump` 会挂死） |
| 3 | **双向并发** | ⬜ 未测 | 只测了单向。两卡同时互发（各 95 GB/s）是否掉速未知 |
| 4 | **≥3 卡** | ⬜ 不可测 | 本机仅 2 卡，无法测 ring allreduce 的 `W/(2(W-1))` 系数效应 |
| 5 | **IMB Allgather/Bcast/Reduce 病态** | ⬜ 未定位 | 只证明了"非线性"，未定位到 Intel MPI 的具体代码路径 |
| 6 | **`fi_pingpong`** | ⬜ 未解决 | `bind(): util/pingpong.c:463 ret=-98` 后永久挂起（shm/tcp 两个 provider 都失败），可能是端口占用或 libfabric 配置问题。第三方实现校验缺失 |
| 7 | **oneCCL 调优** | ⬜ 未做 | 全部默认配置。`allreduce 81.0` vs `broadcast 94.6` 的 14% 缺口可能可通过 `CCL_*` 抹平 |
| 8 | **IMB-RMA-GPU / IMB-NBC** | ⬜ 未跑 | `IMB-RMA-GPU` 已装（GPU-aware RMA 是 Xe Link 最擅长的场景），值得补测 |
| 9 | **OSU Micro-Benchmarks** | ⬜ 未编译 | 作为第三条独立实现，可交叉验证 95.5 GB/s |

---

## 7. 诚实性声明

1. **318 GB/s 是名义值**，来源于 `xpu-smi` 每端口速率 × 端口数的推导，
   **不是**供应商给出的实测规格，也没有测量口径。本报告未对它做任何折算或修正。
2. **95.5 GB/s 是实测平台值**，`spread 3.89%`（64/128/256 MiB 三点），可复现。
3. **"未标定导致下降"是推断，不是结论**。本报告明确标注为**首要嫌疑**，
   并列出三条无法证因果的原因（无标定入口 / 可能是真实并发上限 / 无标称口径）。
4. **`verdict="疑似 PCIe/host"` 字段是自动判据的文字**，在 §3.1 已被推翻，
   保留原文仅为可追溯性。**以 §3.1 的三链判读为准。**
5. **IMB Allgather 的 2.99 GB/s 在报告中标注为不可信**，未纳入任何结论或总表判定。
6. 所有数字均可在 `benchmark/04-interconnect-xelink/results/` 的原始
   JSON / MD / TXT 中复算，无手工修改。
