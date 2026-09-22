# ③ 显存带宽（HBM）测试结论

> 对应测试计划：[`../../TODO/03-memory-bandwidth.md`](../../TODO/03-memory-bandwidth.md)
> 测试代码：[`../../../benchmark/03-memory-bandwidth/`](../../../benchmark/03-memory-bandwidth/)
> 原始产物：`benchmark/03-memory-bandwidth/results/bench_20260922-1945.{json,md}`（113 条记录）
> 测试日期：2026-09-22　　硬件：Intel Data Center GPU Max 1100（PVC, Production ES, 1 tile）

---

## 0. 一页结论

### 0.1 对测试目标的逐条回答

| # | 测试目标（TODO §1） | 实测 | 判定 |
|---|---|---|---|
| 1 | HBM 实测带宽 vs 规格 1229 GB/s | BabelStream Copy 峰值 **900 GB/s** = **73%** | ⚠️ 未到 90%，但**这是该卡的真实水平**（见 §3.1） |
| 2 | 单向 read / write 带宽 | read **696** / write **862** GB/s → **读只有写的 0.81×** | ✅ 测到，且**反直觉**（见 §3.2） |
| 3 | 访问模式（vec / stride）影响 | vec 宽度 >4 反而**掉一半**；stride ≥2 掉到 1/3 | ✅ 见 §3.3、§3.4 |
| 4 | 双卡并发是否线性放大 | 2×839 = **1679 GB/s**，scaling **2.00×** | ✅ **完美线性** |
| 5 | host↔device（PCIe Gen5 x16） | pinned H2D **31.9** / D2H **31.9** GB/s = **50.6%** of 63 GB/s | ⚠️ Gen5 x16 只跑出一半 |
| 6 | 主机内存带宽 | **39.6 GB/s**（1 GiB/数组，72 线程） | 🔴 **严重瓶颈**，见 §3.6 |
| 7 | 硬件计数器交叉验证 | `xpu-smi` 显存读写计数器**完全失效**（恒 576 kB/s） | ❌ 无法取证，见 §3.7 |

### 0.2 一句话

> **HBM 侧「够用但不惊艳」：900 GB/s（73% 规格），双卡完美线性放大到 1.68 TB/s。**
> **真正的短板是主机侧：主机 DRAM 只有 39.6 GB/s，不到单卡 HBM 的 1/20，
> 而主机 RAM 总量（45 GiB）还不到两张卡的 HBM 总和（96 GiB）。**
> 这决定了任何「主机↔设备数据搬运占比高」的负载都会被卡住。

### 0.3 三个最容易踩的陷阱（本报告的重点）

1. **L2 陷阱（GPU 侧）**：L2 = **192 MB**，数组 ≤192 MiB 的"带宽"数字全部是 L2 命中，
   最高报出 **1717 GB/s**（139.7% of 规格）—— 物理上不可能来自 HBM。
   **必须把数组放大到 ≥256 MiB 才能测到 HBM。**
2. **L3 陷阱（主机侧）**：CPU L3 = **432 MiB**（这机器 L3 大得离谱），
   16~128 MiB/数组时报出 675~1368 GB/s；放大到 **1 GiB/数组** 才回落到真实的 **39.6 GB/s**。
   **两者差了 34 倍。**
3. **计数器陷阱**：`xpu-smi` 报的 `GPU Memory Read/Write (kB/s)` 是**假的**，
   满载（~900 GB/s）与空载读数完全相同（~576 kB/s）。**不能用来交叉验证。**
   （同一时刻 GPU Power 已升到 260+ W，证明负载确实在跑。）

---

## 1. 交付物索引

| 类别 | 路径 | 内容 |
|---|---|---|
| 测试代码 | `benchmark/03-memory-bandwidth/run_bench.py` | 6 个 suite |
| SYCL 探针 | `benchmark/03-memory-bandwidth/probes/bw_probe.cpp` | copy / read / write / triad，vec 与 stride 可扫 |
| 主机探针 | `benchmark/03-memory-bandwidth/probes/host_stream.c` | 多线程 STREAM（OpenMP） |
| torch 探针 | `benchmark/03-memory-bandwidth/probes/torch_membw.py` | D2D / H2D / D2H（pinned vs pageable） |
| BabelStream | `benchmark/03-memory-bandwidth/babelstream/` | 行业标准口径 |
| 原始结果 | `benchmark/03-memory-bandwidth/results/bench_20260922-1945.{json,md}` | 113 条记录 |
| README | `benchmark/03-memory-bandwidth/README.md` | 测试方法说明 |

---

## 2. 全部关键数字总表

### 2.1 GPU HBM 带宽（只看 ≥256 MiB/数组，即真正出 HBM 的点）

| 测试 | 数组 | 带宽 (GB/s) | 占 1229 GB/s | 口径 |
|---|---:|---:|---:|---|
| **BabelStream Copy** | 256 MiB | **899.6** | **73.2%** | 读写合计（copy = 1 读 + 1 写） |
| BabelStream Copy | 512 MiB | 867.4 | 70.6% | |
| BabelStream Copy | 1 GiB | 852.0 | 69.3% | |
| BabelStream Copy | 2 GiB | 850.2 | 69.2% | |
| BabelStream Copy | 4 GiB | 845.5 | 68.8% | |
| BabelStream Mul | 1 GiB | 834.7 | 67.9% | 2 读 + 1 写 |
| BabelStream Add | 1 GiB | 804.4 | 65.5% | 2 读 + 1 写 |
| BabelStream Triad | 1 GiB | 806.8 | 65.6% | 2 读 + 1 写 |
| BabelStream Dot | 1 GiB | 701.4 | 57.1% | 2 读（含归约） |
| torch D2D copy (bf16) | 2 GiB | 796.7 | 64.8% | 读写合计 |
| torch D2D copy (fp32) | 2 GiB | 798.2 | 64.9% | 读写合计 |
| 自研 bw_probe copy (vec=4) | 2 GiB | 713.9 | 58.1% | 读写合计 |
| 自研 bw_probe **read** (vec=1) | 2 GiB | **696.0** | **56.6%** | **单向** |
| 自研 bw_probe **write** (vec=2) | 2 GiB | **862.4** | **70.2%** | **单向** |

> **口径警告**：Copy/Mul/Add 的数字是「读+写合计流量 / 时间」，而 read/write 是单向。
> 两者**不能直接相减比较**。BabelStream 的 899.6 GB/s ≈ 读 450 + 写 450。

### 2.2 双卡并发

| 测试 | GPU0 | GPU1 | 合计 | scaling |
|---|---:|---:|---:|---:|
| solo | 839.0 | 839.8 | – | – |
| concurrent (1 GiB/数组) | 839.2 | 839.7 | **1678.9** | **2.00×** |

> 两卡各自有独立 HBM 控制器，因此**完美线性**。
> 这条数据同时说明：测带宽时**不要**把两块卡绑在一起跑，会互相掩盖。

### 2.3 host↔device（PCIe）

| 方向 | 模式 | 带宽 (GB/s) | 占 Gen5 x16 (63 GB/s) |
|---|---|---:|---:|
| H2D | pageable | 25.81 | 41.0% |
| D2H | pageable | 25.41 | 40.3% |
| **H2D** | **pinned** | **31.87** | **50.6%** |
| **D2H** | **pinned** | **31.89** | **50.6%** |

### 2.4 主机内存（CPU）

| 数组大小 | 工作集 | 区间 | Copy (GB/s) | Triad (GB/s) |
|---:|---:|---|---:|---:|
| 16 MiB | 48 MB | L3 内 | 674.6 | 1368.3 |
| 128 MiB | 384 MB | L3 内 | 254.7 | 352.5 |
| **1 GiB** | **3 GB** | **DRAM** | **39.63** | **41.42** |

**CPU 拓扑**：`Genuine Intel(R) 0000`，144 线程，1 NUMA 节点，**L3 = 432 MiB**，L2 = 144 MiB。

---

## 3. 分项详述

### 3.1 HBM 只有 73%：这是真实水平，不是测错

三个**独立实现**（BabelStream、自研 SYCL 探针、torch）在大数组下都收敛到
**~800-900 GB/s**，彼此差异 <15%：

| 实现 | 大数组峰值 |
|---|---:|
| BabelStream Copy | 899.6 GB/s |
| torch D2D | 798.2 GB/s |
| 自研 bw_probe copy | 713.9 GB/s |

三者一致 ⇒ **这是硬件的真实上限，不是某个工具的问题**。
与 `docs/hardware.md` 记录的「规格 1229 GB/s」相比只有 73%，符合 **Production ES 硅片**
（非正式量产件）的预期，也与 `docs/caveats.md` 记录的其它 ES 现象一致。

**判读**：TODO §6 写的"≥90% 为通过"这条标准**应当放宽**——本机 HBM 的
可信上限就是 ~900 GB/s。建议后续文档把 HBM 参考值标为 **900 GB/s（实测）**，
并在 A/B 对比时用它当分母。

### 3.2 读比写慢（0.81×）—— 反直觉但可复现

| 方向 | 峰值 (GB/s) | vec=1 | vec=2 | vec=4 | vec=8 | vec=16 |
|---|---:|---:|---:|---:|---:|---:|
| read | **696** | 696.0 | 686.8 | 688.1 | 643.9 | 652.4 |
| write | **862** | 850.7 | **862.4** | 829.0 | 313.5 | 291.2 |

一般 GPU 上读快于写（写要处理 write-allocate / 合并）。
本卡相反：**write 比 read 快 24%**。

观察：**write 对 vec 宽度极其敏感**（vec=1/2/4 时 830-862，vec=8 断崖跌到 313）。
read 则对 vec 宽度不敏感（643-696）。

**推测**（未证实）：宽向量 write 触发了不同的 store 路径 / 部分写掩码。
**结论**：写密集 kernel 用 **vec≤4**，读密集 kernel 无所谓。
**不做过度解释**，需要 `docs/TODO/07-profiling.md` 的 DRAM 计数器才能定性。

### 3.3 vec 宽度 >4 会掉一半

| vec | copy | read | write | triad |
|---:|---:|---:|---:|---:|
| 1 | 673.4 | 696.0 | 850.7 | 677.6 |
| **2** | 706.1 | 686.8 | **862.4** | **722.7** |
| **4** | **713.9** | 688.1 | 829.0 | 713.4 |
| 8 | 450.6 | 643.9 | 313.5 | 522.7 |
| 16 | 358.4 | 652.4 | 291.2 | 432.1 |

**最佳 vec = 2~4**；vec=8/16 时 copy/write **腰斩**。
与 `02-compute-peak` 的 ALU VEC 扫描（最佳 VEC=4，VEC=8 后拉平但**不掉**）不同，
**内存路径对宽向量是有害的**。

> ⚠️ 这与 `02-compute-peak` 的结论合起来说明：**`sycl::vec` 宽度要按负载分别调**，
> 不存在一个通用的最优值。

### 3.4 stride 的断崖

| stride | copy (GB/s) | 相对 stride=1 |
|---:|---:|---:|
| 1 | 713.8 | 100% |
| 2 | 237.2 | 33% |
| 4 | 118.5 | 17% |
| 8 | 115.9 | 16% |
| 16 | 68.28 | 10% |

**注意口径**：这里按「逻辑访问字节数 / 时间」计，所以下降的**主要原因是缓存行利用率**
（stride=2 时每 128 B 缓存行只用了一半），**不是 HBM 本身的带宽下降**。
真实的物理流量下降幅度远小于表观值。

**结论**：这条数据只能说明「非连续访问会浪费缓存行」——
这正是 `docs/hardware.md` 里 NCHW vs NHWC 差异（ResNet-50 +58~75%）的内存侧解释。

### 3.5 PCIe Gen5 x16 只跑出一半

| 模式 | H2D (GB/s) | D2H (GB/s) | 相对 63 GB/s |
|---|---:|---:|---:|
| pageable | 25.81 | 25.41 | 41% |
| **pinned** | **31.87** | **31.89** | **50.6%** |

- **pinned 比 pageable 快 1.23×**（~25.5 → 31.9 GB/s）：符合预期，
  pageable 需要中间 bounce buffer。
- 但 pinned 也只到 **31.9 GB/s = Gen5 x16 的一半**。
  Gen5 x16 单向原始 63 GB/s，实际可达通常 55-60 GB/s。
  **31.9 更像是 Gen4 x16 的水平**，或 Gen5 链路降速/单方向限制。

**待核实**：`03` 的 topo 没抓到 PCIe 链路速率。
**已在 `04-interconnect-xelink` 的 `topo` suite 中补上**
（读 `/sys/class/drm/card*/device/current_link_speed` 与 `max_link_speed`），
下一步应据其判断是「链路降速」还是「DMA 引擎上限」。

⚠️ **交叉引用**：`04-interconnect-xelink` 实测卡间 P2P 拷贝 **~95 GB/s**，
是这条 PCIe H2D（31.9 GB/s）的 **3.0×**。这说明：
**卡间通信走的是 Xe Link 而不是 PCIe**（否则不可能超过 63 GB/s）。

### 3.6 🔴 主机内存 39.6 GB/s —— 本机最大的结构性瓶颈

| 层级 | 带宽 | 相对 HBM |
|---|---:|---:|
| 单卡 HBM | 900 GB/s | 1× |
| 双卡 HBM 合计 | 1679 GB/s | 1.9× |
| **主机 DRAM** | **39.6 GB/s** | **0.044×** |
| host↔device (pinned) | 31.9 GB/s | 0.035× |

**三个致命事实**：

1. **主机 DRAM 只有 39.6 GB/s** —— 对于一个 72 核的 Xeon 级 CPU，这个数字**异常低**
   （预期 100-200 GB/s）。与 `docs/hardware.md` 记录的「宿主是 Intel ES CPU」一致，
   疑似也是 ES 件的限制。
2. **主机 RAM 总量 45 GiB < 双卡 HBM 96 GiB**。
   这意味着**连一次全模型/全数据集的 host↔device 搬运都可能放不下**。
3. HBM 与主机内存之间 **22.7× 的带宽落差**，比典型 GPU 服务器（~8-10×）严重得多。

**影响**（与 `Conclusion/05-ai-dl/04-bottleneck.md` 的「内存倒挂」结论直接呼应）：

- DataLoader 预处理（host 侧）上限 ~40 GB/s；
- 任何 `tensor.cpu()` / 数据集装载 / checkpoint 写盘都会成为瓶颈；
- 多卡喂数据的天花板是 40 GB/s，**2 张卡每张只能分到 20 GB/s**，
  而单卡 HBM 需要 900 GB/s → **差 45 倍**。

### 3.7 ❌ 硬件计数器不可用

| 时刻 | GPU Memory Read (kB/s) | GPU Memory Write (kB/s) | GPU Power |
|---|---:|---:|---:|
| 空载 | ~576 | ~576 | ~43 W |
| **满载（BabelStream 4 GiB, ~900 GB/s）** | **~576** | **~576** | **260+ W** |

读数**逐字节相同**。而同一时刻 GPU Power 已经从 43 W 升到 260+ W，
证明负载确实在跑。

**结论**：本机 i915 驱动下 `xpu-smi` 的显存读写计数器是**死的**。

| 连带失效的工具 | 现象 |
|---|---|
| `xpu-smi dump` | **挂死**（打印表头后卡住，`Terminated` rc=143） |
| `xpu-smi dump -d -1 -m ...` | 同上 |
| `xpu-smi stats -d 0` 的 Memory Read/Write | 恒 ~576 kB/s |
| `xpu-smi` 的 `Xe Link Throughput` | **N/A** |
| `xpu-smi` 的 `EU Array Active/Stall` | **N/A** |
| `/usr/bin/stream` | **是 ImageMagick**，不是 STREAM（同名包） |

⇒ **带宽/算力的绝对数字只能以 BabelStream / 自研探针 / torch 为准，
不能依赖 xpu-smi 的任何带宽类计数器。**

---

## 4. 判读标准对照（vs `docs/TODO/03-memory-bandwidth.md`）

| TODO 中的判读标准 | 实测 | 判定 |
|---|---|---|
| BabelStream Copy ≥ 90% of 1229 GB/s | 899.6 = **73%** | ⚠️ **不达标，但应改标准**（见 §3.1） |
| 双卡并发接近 2× | **2.00×** | ✅ |
| pinned H2D ≥ 50% of Gen5 x16 | 31.87 = **50.6%** | ✅ 刚过线 |
| 主机 DRAM ≥ 100 GB/s | **39.6** | 🔴 **严重不达标** |
| 硬件计数器能与软件测量互证 | 计数器失效 | ❌ 无法互证 |

> **建议修正 TODO 中的参考值**：
> - HBM 参考值 1229 GB/s → 增加「**实测上限 ~900 GB/s（ES 件）**」说明；
> - 主机 DRAM 参考值应标注「**本机实测 39.6 GB/s，异常低**」；
> - 删除「用 `xpu-smi dump -m 5,6,7` 取显存计数器」的步骤（会挂死）。

---

## 5. 与其它目录的衔接

| 本目录的发现 | 影响 |
|---|---|
| 单卡 HBM 900 GB/s、双卡 1679 GB/s | 给 `05-ai-dl` 的 LLM decode（权重带宽受限）提供分母：decode 45.7 tok/s ↔ 0.988 GB 权重 → 有效带宽利用率可反推 |
| 卡间 P2P 95 GB/s > PCIe H2D 31.9 GB/s | → `04-interconnect-xelink` 已证实走 Xe Link，不是 PCIe |
| 主机 DRAM 39.6 GB/s、45 GiB RAM | → `Conclusion/05-ai-dl/04-bottleneck.md` 的「内存倒挂」；也是 `06-hpc-apps` 的输入 |
| vec>4 会让内存路径腰斩，而 ALU 路径不掉 | → 两层的内存/计算最优向量宽度**不同**，见 `02-compute-peak` §4.3 |
| stride 断崖 | → NCHW/NHWC 差异（`05-ai-dl` 里 ResNet-50 差 58~75%）的内存侧解释 |
| L2 陷阱（192 MB） | → `02-compute-peak` 的 vector 测试如果用小数组会得到虚高值；本报告是唯一可信的带宽口径 |

---

## 6. 未做 / 待补充

| 项 | 原因 | 建议 |
|---|---|---|
| PCIe 实际链路速率（Gen5 x16 vs 降速到 Gen4） | `03` 未采集 | **已在 `04-interconnect-xelink/topo` 中补上**，见 `Conclusion/04-*` |
| 硬件 DRAM 计数器（VTune/PTI） | 需接入 `docs/TODO/07-profiling.md` | 用它裁决 §3.2 的读写不对称 |
| `xpu-smi` 计数器为何失效 | 驱动层（i915 ES）问题 | 非测试可控，只做记录 |
| 非对齐 / 跨页访问对带宽的影响 | 本轮未覆盖 | 低优先 |
| 双卡**同时**打满 HBM 时的功耗墙 | 本轮只测带宽 | 见 `docs/TODO/08-power-efficiency.md` |
| 主机 DRAM 为何只有 39.6 GB/s | 疑为 ES CPU / 内存配置问题 | **值得单独立项**；可先查 `dmidecode` 的通道数与频率 |
