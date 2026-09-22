# ③ 显存带宽测试（HBM）

**优先级：P1**
**文档位置：** `docs/TODO/03-memory-bandwidth.md`

---

## 1. 测试目标

1. 得到 **HBM 实际读写带宽**（单卡 / 双卡同时）
2. 计算达成率，确定**带宽天花板**
3. 为 ② 算力测试和 ⑤ AI 测试提供**「是否带宽受限」的判据**
4. 测试**访问模式**对带宽的影响（粒度、向量宽度、并发度）

## 2. 为什么重要

Ponte Vecchio 在 AI/HPC 场景下**极易带宽受限**。没有这个基线，就无法判断：
- 「GEMM 慢」是算力问题还是带宽问题
- 「Attention 慢」是不是 memory-bound
- Roofline 分析中的带宽上限是否与实测一致

### 规格值说明
HBM 带宽的**规格值需要在测试中实测确认**。
（此前口头给出的约 1.2 TB/s 只是粗略量级估计，不要作为结论引用。）

---

## 3. 工具与准备

| 工具 | 状态 | 说明 |
|---|---|---|
| **BabelStream (SYCL)** | ❌ 需构建 | **事实标准**，Copy/Mul/Add/Triad |
| PyTorch (`torch.xpu`) | ✅ 已装 | 快速交叉验证 |
| `xpu-smi dump` | ✅ 已装 | 硬件侧读写计数（metric 6/7/5） |
| `stream` (CPU) | ✅ `/usr/bin/stream` | **主机侧**内存带宽，用于区分 host/device 瓶颈 |
| `mem_bench` (VTune) | ✅ 已装 | 内存带宽微基准 |

### 构建 BabelStream
```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream
cmake -B build -H. -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 4. 执行步骤

### Step 1：BabelStream 标准测试（主力）
```bash
# 默认数组大小可能偏小，务必放大以压满带宽
./build/babelstream -s $((1<<30))          # 元素数，按需调整成能占满显存
./build/babelstream -s $((1<<30)) 2>&1 | tee babelstream_gpu0.log
```
关注 5 个 kernel 的 GB/s：
| Kernel | 每次迭代访问字节 | 说明 |
|---|---|---|
| Copy | 2 × N × 8 B | 读+写 |
| Mul | 2 × N × 8 B | 读+写 |
| Add | 3 × N × 8 B | 2 读 + 1 写 |
| Triad | 3 × N × 8 B | 2 读 + 1 写 |
| Dot | 2 × N × 8 B | 归约 |

> 典型范围：单卡 HBM 带宽应在 **TB/s 量级**。取 5 个 kernel 的**最大值**作为带宽上限。

用不同元素数扫点（如 2^26 → 2^31），观察带宽随数据规模的变化，找出**饱和点**。

### Step 2：PyTorch 交叉验证
```python
import torch
from torch.utils.benchmark import Timer

dev = "xpu"
results = {}
for gb in [1, 2, 4, 8, 16]:
    n = gb * (1 << 30) // 4          # float32
    a = torch.empty(n, device=dev, dtype=torch.float32)
    b = torch.empty(n, device=dev, dtype=torch.float32)
    t = Timer("b.copy_(a)", globals=globals()).timeit(20)
    gbps = 2 * n * 4 / t / 1e9       # 读 + 写
    results[gb] = gbps
    print(f"{gb:3d} GiB  copy  {gbps:8.1f} GB/s")
```

### Step 3：dtype 影响
用 `float16` / `bfloat16` / `int8` 重复上述测试 —— 检验带宽是否与元素宽度无关（若带宽容差明显差异，说明有量化/转换开销）。

### Step 4：硬件计数对照
```bash
xpu-smi dump -d 0 -m 5,6,7 -i 200 -n 200 -j > mem_bw_telemetry.json
```
- `m5` GPU Memory Utilization (%)
- `m6` GPU Memory Read (kB/s)
- `m7` GPU Memory Write (kB/s)

对比软件测量的带宽与硬件计数，验证一致性。

> ❌ **实测不可行**（2026-09-22）：`xpu-smi dump` 在 hwt 的驱动上**挂死**（任何 metric 都只打印表头然后挂住）；
> 退化用 `xpu-smi stats -d 0` 后拿到 `GPU Memory Read/Write (kB/s)`，但该计数**恒为 ~576 kB/s** ——
> 空载与满载（~900 GB/s）**完全一样**。结论：**本驱动上该计数器不可用**，无法做硬件侧交叉验证。
> 详见 §4.7 与 §6 裁定。

### Step 5：双卡同时测试
```bash
# 两个进程各占一卡，或用 ZE_AFFINITY_MASK 分配
ZE_AFFINITY_MASK=0 ./build/babelstream -s $((1<<30)) > gpu0.log 2>&1 &
ZE_AFFINITY_MASK=1 ./build/babelstream -s $((1<<30)) > gpu1.log 2>&1 &
wait
```
**关键问题**：双卡同时跑时，单卡带宽是否下降？（HBM 独立，理论上不应互相影响；若下降说明存在共享资源竞争）

### Step 6：主机侧带宽基线
```bash
/usr/bin/stream                       # 主机内存带宽上限
# 或指定更大数组
OMP_NUM_THREADS=144 /usr/bin/stream
```
这个数字非常关键 —— 在内存倒挂（45 GiB host vs 96 GiB HBM）的机器上，host 侧带宽常常是真正的瓶颈。

---

## 4.7 逐条覆盖核验（2026-09-22 收尾）

> 数据来源：`benchmark/03-memory-bandwidth/results/bench_20260922-1945.{json,md}`（**113 条记录**，
> `babelstream` 50 / `probe` 30 / `torch` 15 / `host` 13 / `dual_gpu` 3 / `counters` 2）。
> 详细解读见 [`../Conclusion/03-memory-bandwidth/README.md`](../Conclusion/03-memory-bandwidth/README.md)。

| TODO 条目 | 状态 | 证据 / 说明 |
|---|---|---|
| Step 1 BabelStream 标准测试（主力） | ✅ | 5 个 kernel × 10 个数组尺寸（4 MiB → 4 GiB）= 50 条；Copy / Mul / Add / Triad / Dot 齐全 |
| Step 2 PyTorch 交叉验证 | ✅ | `torch` suite 15 条 |
| Step 3 dtype 影响 | ✅ | **与元素宽度无关**（纯带宽受限）；唯一例外是 fp16/bf16 `tanh` 掉到 **345 / 329 GB/s**（软件模拟） |
| Step 4 硬件计数对照 | ❌ **不可行** | `xpu-smi dump -m 5,6,7` **挂死**；`xpu-smi stats -d 0` 的 `m6/m7` **恒为 ~576 kB/s**（空载 = 满载）→ 计数器在本驱动上不可用 |
| Step 5 双卡同时测试 | ✅ | solo **838.99 / 839.77**；concurrent **839.21 + 839.73 = 1678.94**，`scaling_vs_solo_mean = **2.00**` |
| Step 6 主机侧带宽基线 | ✅ **换工具** | `/usr/bin/stream` 在本机是 **ImageMagick**（图像处理）不是 STREAM 基准！自写 `probes/host_stream.c`（72/144 线程 × 3 种数组尺寸）→ **DRAM 39.6 GB/s** |
| §5 指标记录表 | ✅ | 见下（已填） |
| §6 判读标准 | ✅ | 逐条裁定见 §6 表右列 |
| §8-1 数组必须远超 cache | ✅ **已验证** | L2 = 192 MB ⇒ 4~128 MiB 数组测出的全是假值；host L3 = 432 MiB 同理（16 MiB → 675 GB/s、128 MiB → 254 GB/s 都是假的） |
| §8-2 `-s` 单位是元素数 | ✅ | 已按元素数调用 |
| §8-3 双卡 600 W 是否超 PSU | ⬜ **未测** | 无功率计；只用 `xpu-smi` 读到单卡 260+ W（未分别采样双卡） |
| §8-4 host 数组过大会 swap | ✅ 已规避 | 最大 1 GiB/数组（3 GiB 工作集，45 GiB 内存安全） |
| §8-5 记录频率锁定状态 | ✅ | 1550 MHz 锁定（见 ② `clock` suite） |
| （自加）单向读 / 写拆分 | ✅ 超出计划 | read **696** vs write **862** GB/s（读 = 0.81× 写，与常见 GPU **相反**） |
| （自加）向量宽度扫描 | ✅ 超出计划 | copy `vec4 714 → vec8 451 → vec16 358` ⇒ **vec > 4 反而腰斩**（ALU 路径不受影响） |
| （自加）stride 扫描 | ✅ 超出计划 | `stride_sweep.copy` 5 点 |
| （自加）L2 / L3 陷阱定量 | ✅ 超出计划 | 峰值假值 **1717 GB/s** @20 MiB（Add），@256 MiB 落到 900 |
| （自加）pinned vs pageable H2D/D2H | ✅ 超出计划 | pinned H2D **31.87** / D2H **31.54**；pageable **25.5** GB/s |

> 统计：**计划内 6 个 Step：5 完成（含 1 项换工具）/ 1 项不可行**；**计划外新增 5 项**。
> 不可行项（硬件计数器）已如实记录，并给出「软件侧三实现互相印证」的替代证据链。

---

## 5. 指标记录表

> ✅ 已填写。全部来自 `bench_20260922-1945.json`，**均为数组 ≥256 MiB 的 HBM 平台期值**
> （≤192 MiB 会命中 192 MB L2，假值一并列在脚注）。

| 项目 | GPU 0 | GPU 1 | 双卡同时 |
|---|---|---|---|
| Copy (GB/s) | **899.6** | — （仅单卡基线 839.8） | 839.2 |
| Mul (GB/s) | 842.9 | 未单独跑 | — |
| Add (GB/s) | 810.6 | 未单独跑 | — |
| Triad (GB/s) | 811.8 | 未单独跑 | — |
| Dot (GB/s) | 720.3 | 未单独跑 | — |
| **峰值带宽** | **899.6** | **839.8**（自研 probe copy） | **1678.9 合计**（2.00×） |
| 饱和数据规模 | **≥256 MiB**（≤192 MiB 是 L2 假象） | 500 MiB | 500 MiB |

| 主机侧 | GB/s |
|---|---|
| Stream Copy / Scale / Add / Triad（**真 DRAM**，1 GiB/数组） | **39.63 / 39.97 / 42.20 / 41.42** |
| （参照）同样四项在 **L3 内**（128 MiB/数组） | 254.7 / 264.2 / 301.5 / 352.5 |

附加记录：
- ⚠ **`m6/m7` 硬件读写计数不可用**：空载 `0.000 + 0.000 GB/s`，满载（~900 GB/s）**同样是 0.000**
  —— 该计数器只报 ~576 kB/s，与负载完全无关。**带宽绝对值只能以 BabelStream / 自研探针为准。**
- **功效**：约 **900 GB/s ÷ 260 W ≈ 3.5 GB/s per W**（功率取自同一时刻 `counters` suite 读到的 260+ W）。
- **规格达成率**：899.6 / 1229 = **73%**（HBM 规格 1229 GB/s）；三种独立实现（BabelStream、
  自研 SYCL probe、oneDNN 路径）都在 700~900 GB/s 区间，**结论一致**。

---

## 6. 判读标准（含实测裁定）

| 现象 | 可能原因 | 实测裁定（2026-09-22） |
|---|---|---|
| 软件带宽 ≫ 硬件计数 | 测量方法有误（cache 命中、未真正读写） | ⚠️ **反转**：硬件计数**恒为 576 kB/s**（空载 = 满载），软件 900 GB/s → **不是测量方法错，是计数器本身坏** |
| 小数组带宽极高、大数组骤降 | 小数组落在 L2/cache 中，**必须用大数组** | ✅ **完全成立**：≤192 MiB（L2 内）报 **1717 GB/s**，≥256 MiB 落到 **900**。host 侧 L3 同理（16 MiB → 675 GB/s 假值） |
| 双卡同时带宽显著下降 | 共享资源竞争（PCIe / 主机内存 / 电源） | ❌ **不成立**：**2.00× 完美线性**（839.2 + 839.7 = 1678.9）→ HBM 完全独立 |
| dtype 不同导致带宽差异大 | 存在类型转换开销 | ❌ **基本不成立**（elementwise 全是纯带宽受限）；⚠️ **唯一例外**：fp16/bf16 `tanh` **345 / 329 GB/s**（软件模拟） |
| 带宽达成率 < 70% | 访问模式不佳、kernel 线程不足、需调向量宽度 | ⚠️ **恰好 73%，踩线过**。判为「正常区间下沿」，**不是访问模式问题** —— 三实现互相印证。真正的短板在**主机侧**（39.6 GB/s） |

> **关键**：数据必须**大于 L2 cache**才能测到真实 HBM 带宽。Max 1100 的 **L2 = 192 MB**，
> 所以 **≥256 MiB / 数组**才是安全界限。这一点在本机特别容易被骗 —— 192 MB L2 是「大 GPU」量级，
> 随手写的 64~128 MiB 测试都会命中 L2 得到虚高结果。

---

## 7. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| → ② 算力峰值 | 判断 GEMM 是算力受限还是带宽受限（算力/带宽比） |
| → ⑤ AI 测试 | Attention 等 memory-bound 算子的理论上限 |
| → ⑦ Profiling | Roofline 模型的「带宽墙」是否与实测一致 |
| → ⑧ 能效 | 计算 GB/s per W |

---

## 8. 注意事项

1. **数组规模必须远超 cache**，否则测的是 cache 带宽而非 HBM 带宽。
2. BabelStream 的 `-s` 单位是**元素个数**，注意换算成字节。
3. 双卡测试注意**总功耗 600 W** 是否超 PSU 能力。
4. 主机 45 GiB 内存下，分配过大的 host 数组会触发 swap，污染结果。
5. 记录时的频率锁定状态（1550 MHz）会影响带宽 —— 锁频是有利于可重复性的。
