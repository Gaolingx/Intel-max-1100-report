# benchmark/03-memory-bandwidth — 显存 / 主机内存带宽实测

对应文档：[`docs/TODO/03-memory-bandwidth.md`](../../docs/TODO/03-memory-bandwidth.md)
（显存带宽 HBM）、[`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md)
（算力峰值）、[`docs/hardware.md`](../../docs/hardware.md)（硬件规格）。

本目录用 **三条互相独立的路径** 测带宽，避免“自己验证自己”：

1. **BabelStream (SYCL)** —— 行业事实标准（Copy/Mul/Add/Triad/Dot），主力口径；
2. **自研 SYCL 探针 `bw_probe`** —— 补 BabelStream 没有的东西：**单向 read / 单向 write**、
   **向量宽度扫描**、**stride 扫描**；
3. **PyTorch (`torch.xpu`)** —— 跨层交叉验证 + H2D/D2H（PCIe）带宽。

再加两条辅助：
4. **双卡并发** —— 验证两块 HBM 是否独立（预期 2.0× 线性）；
5. **主机内存 STREAM** —— 主机侧基线（本机只有 45 GiB RAM，内存倒挂，host 常是真瓶颈）。

---

## 1. 目录结构

```
benchmark/03-memory-bandwidth/
├── README.md                本文档
├── run_bench.py             CLI 入口（6 个 suite）
├── probes/
│   ├── bw_probe.cpp         自研 SYCL 带宽探针（read/write/copy/triad + vec/stride 扫描）
│   ├── host_stream.c        自研主机内存 STREAM（OpenMP；**不用 /usr/bin/stream**，见 §5）
│   └── torch_membw.py       PyTorch D2D / H2D / D2H 交叉验证
├── babelstream/             git clone 的 BabelStream（build/sycl-stream 为产物）
├── build/                   探针编译产物（host_stream 等）
└── results/                 运行后自动生成的 JSON / Markdown 报告
```

> `babelstream/` 是上游仓库整份 clone 进来的（含自己的 `README.md`、`docs/`），
> 不要与本目录的文档混淆。

---

## 2. 快速开始

```bash
cd /root/workspace/benchmark/03-memory-bandwidth

# BabelStream 需要一次构建（Unix Makefiles；本机没有 ninja）
source /opt/intel/oneapi/setvars.sh
cd babelstream
cmake -B build -H. -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..

# 运行（venv1 里有 torch+xpu；用系统 python3 也可以，本 suite 不强依赖 torch）
/root/workspace/venv1/bin/python run_bench.py                  # 全部 6 个 suite
/root/workspace/venv1/bin/python run_bench.py --quick          # 冒烟
/root/workspace/venv1/bin/python run_bench.py probe counters   # 只跑指定 suite
/root/workspace/venv1/bin/python run_bench.py --tag 20260922-1945
```

产物：`results/bench_<tag>.json` + `results/bench_<tag>.md`。
`run_bench.py` 会自动编译 `bw_probe`（→ `/tmp/bw_probe`）与 `host_stream`（→ `build/host_stream`）。

**实测参考产物**：[`results/bench_20260922-1945.md`](results/bench_20260922-1945.md)（113 条记录）。

---

## 3. 六个 suite 分别测什么

| suite | 测什么 | 主要工具 | 口径（搬运字节数） |
|---|---|---|---|
| `babelstream` | HBM 标准带宽 + 规模扫描 | BabelStream SYCL | Copy/Mul/Dot = 2·N·B，Add/Triad = 3·N·B |
| `probe` | 单向读 / 单向写 / vec / stride | `bw_probe.cpp` | read = 1·N·B，write = 1·N·B，copy = 2·N·B，triad = 3·N·B |
| `dual_gpu` | 双卡并发是否线性 | BabelStream ×2 进程 | copy = 2·N·B |
| `torch` | D2D / H2D / D2H | `torch.xpu` | copy = 2·N·B；H2D/D2H = 1·N·B |
| `host` | 主机 DRAM + L3 | `host_stream.c` | Copy/Scale = 2·N·B，Add/Triad = 3·N·B |
| `counters` | 硬件计数器可用性核查 | `xpu-smi stats` | 见 §5.4 |

**通用公式**：$\text{GB/s} = \dfrac{\text{moved\_bytes}}{t_{\text{best}}} \times 10^{-9}$。
一律取多次迭代的**最小值时间**（即最大带宽），并做 warmup，避免首轮 page-fault/编译开销。

**参考值**：Max 1100 的 HBM 规格 **1229 GB/s**（48 GiB HBM2e，ECC 开启）。
> ⚠ `docs/TODO/03-memory-bandwidth.md` §2 明确写过「此前口头给出的约 1.2 TB/s
> 只是粗略量级估计，不要作为结论引用」。所以本套测试**只把 1229 GB/s 当作参考基线**，
> 结论一律以实测为准。

---

## 4. 关键实测结果（tag `20260922-1945`）

### 4.1 HBM 带宽

| 项目 | 实测 | 相对 1229 GB/s | 备注 |
|---|---:|---:|---|
| BabelStream **Copy 峰值** | **900 GB/s** | **73%** | 数组 256 MiB，脱离 192 MB L2 |
| BabelStream Copy @≥1 GiB | 845~852 GB/s | 69% | 大数组平台区，无坍缩 |
| 自研探针 单向 **read** | **696 GB/s** | 56.6% | vec=1~4 |
| 自研探针 单向 **write** | **851 GB/s** | 69.3% | vec=2，**明显高于 read** |
| 自研探针 copy（vec=4） | **714 GB/s** | 58.1% | 2 GiB 数组 |
| PyTorch D2D copy（≥1 GiB） | **~798 GB/s** | 65% | bf16 与 fp32 **完全一致** |
| 双卡并发合计 | **1679 GB/s** | — | 839 + 840，**scaling = 2.00×** |

**结论 1：HBM 达成率 ≈ 65~73%，存在明确的“带宽天花板”。**
`Copy 900 GB/s` 是可信上限；单向 write(851) > read(696)，说明读路径是短板
（read 只有 write 的 0.82×），做 memory-bound 算子时应假设**有效带宽 ≈ 700 GB/s**。

**结论 2：两卡 HBM 完全独立。** 并发时单卡 839 GB/s 与单独跑时（839/840）**一模一样**，
聚合正好 2.0×。→ 双卡场景不必担心 HBM 竞争；瓶颈只可能出现在 Xe Link 或主机（见 ④）。

**结论 3：带宽与 dtype 无关。** D2D copy 用 bfloat16 与 float32 数值完全相同
（794.7 vs 794.8 GB/s @1 GiB）→ 元素位宽不影响带宽，**改 dtype 不会提升带宽**。

### 4.2 访问模式的影响（`bw_probe`）

向量宽度扫描（copy，2 GiB 数组）：

| vec | 1 | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| GB/s | 673 | 706 | **714** | 451 | 358 |

→ **vec=4（128 bit）是最优**；vec=8/16 反而掉 37%/50%（寄存器压力 + 占用率下降）。
这与“无脑加大向量宽度”的直觉相反，是个必须记录的调优点。

stride 扫描（copy，vec=4）：

| stride | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| GB/s | 714 | 237 | 118 | 116 | 68 |

→ **stride=2 就掉到 1/3**，stride=16 只剩 9.6%。→ 任何**非连续访问**（例如 NCHW 的
Channel 维、转置、gather）都会把有效带宽打到 100~240 GB/s 区间，设计 kernel 时必须
优先保证连续访存。

### 4.3 PCIe（H2D / D2H）

| 方向 | pinned | pageable |
|---|---:|---:|
| H2D（host→device） | **31.87 GB/s** | 25.81 GB/s |
| D2H（device→host） | **31.89 GB/s** | 25.41 GB/s |

PCIe 5.0 ×16 理论 63 GB/s（单向）。实测 pinned 31.9 GB/s = **50.5% 的理论值**，
这是**未用 P2P/未叠满双向**的常规水平，可作为 ⑨ pipeline 的 H2D 上限。
pageable 比 pinned 慢约 20%，**凡是要反复拷贝的通路都应该用 `pin_memory`**
（`torch_membw.py` 里已修正为 `non_blocking=pinned`，早前一版恒为 `non_blocking=True`
导致 pageable 数字虚高）。

### 4.4 主机内存（容易被忽略的真瓶颈）

| 项目 | 实测 |
|---|---:|
| 主机 DRAM copy（1 GiB/数组，72 线程） | **39.6 GB/s** |
| 主机 DRAM triad | 41.4 GB/s |
| 主机 16~128 MiB/数组（命中 L3） | **675 GB/s**（虚高 17×） |
| 主机 L2 / **L3** | 144 MiB / **432 MiB**（`lscpu`） |

**结论：主机内存带宽只有 ~40 GB/s，不到 GPU 的 1/20。**
在“45 GiB 主机内存 vs 96 GiB HBM”的内存倒挂机器上，**host 侧几乎必然是端到端瓶颈**：
单卡 H2D 只有 31.9 GB/s，而要喂满一块卡需要 ≈700 GB/s 的输入 → **不可能在线喂数据**，
必须让数据常驻显存或做多级流水。这条结论直接支撑 `docs/hardware.md` 的判断。

---

## 5. 坑与注意事项（都是实测踩出来的）

### 5.1 L2/L3 cache 会把带宽数字抬高好几倍
- **GPU L2 = 192 MB**（Max 1100 的 LLC 极大）。数组 ≤192 MiB 时数据常驻 L2，
  BabelStream 会报出 `Add @16 MiB = 1717 GB/s = 140% 的规格值` 这种荒谬数字。
  → `run_bench.py` 对 ≤192 MiB 的记录统一打上 `note_L2=L2内(非HBM)` 并在报告里单列。
- **主机 L3 = 432 MiB**（72 核）。`host_stream` 在 16~128 MiB/数组时测到 **675 GB/s**，
  必须放大到 **1 GiB/数组** 才落到真实 DRAM 的 **39.6 GB/s**（**差 17 倍**）。
  → `suite_host` 现在固定扫描 `16 MiB(L3) / 128 MiB(L3) / 1 GiB(DRAM)` 三档，
    并在报告里把 L3 陷阱显式点出来。

**判据（写进 TODO 文档 §6）**：小数组带宽极高、大数组骤降 → 测的是 cache 不是 HBM。

### 5.2 `/usr/bin/stream` **不是 STREAM 基准**
本机 `/usr/bin/stream` 是 **ImageMagick 7.1.1-43** 的 `stream` 命令（图像处理），
直接跑会打印 `stream [options ...] input-image raw-image`。TODO 文档 §3/§6 里
“`stream` (CPU) ✅ 已装”这一条是**错的**，不能用来做主机带宽基线。
→ 本目录自己写了 `probes/host_stream.c`（OpenMP 版 Copy/Scale/Add/Triad），
   零依赖，`gcc -O3 -fopenmp` 即可。

### 5.3 BabelStream 的 `--gigabytes`
`--gigabytes` 让 **`-s` 的含义变成“GiB 字节数”**，且打印的带宽列**已经是 GB/s**
（但列头仍写 “MB/s”）。`parse_babelstream()` 直接原样返回该列；
早前一版额外除了 1000，导致报出 2 GB/s 的荒谬值。

### 5.4 `xpu-smi dump` 完全不可用，且显存读写计数器是假的
- **`xpu-smi dump` 在本机驱动上是坏的**：任何 metric（0/1/2/9/17 都试过）都只输出
  CSV 表头然后挂住，`-j` 直接回 “Not supported”。TODO 文档 §3/§4 里
  `xpu-smi dump -m 5,6,7` 的方案**不成立**。
- 退而用 `xpu-smi stats -d 0`，但它给的 `GPU Memory Read/Write (kB/s)` 也是**假的**：
  空载 576 kB/s，在 BabelStream 以 **~900 GB/s** 跑 copy 时**依然读到 576 kB/s**
  （同时 GPU Power 已升到 260+ W，证明负载确实在跑）。
- 本驱动的 N/A 项：`EU Array Active/Stall/Idle (%)`、`Xe Link Throughput (kB/s)`、
  全部 RAS/error 计数器。

→ `suite_counters` 把这件事**如实记成一条 `error` 级发现**（唯一的 error 就是这个），
  而不是伪造一个交叉验证。**带宽绝对值只以 BabelStream / 自研探针为准。**

### 5.5 双卡进程需要 `ZE_AFFINITY_MASK`
`suite_dual_gpu` 用 `ZE_AFFINITY_MASK=0` / `=1` 各起一个 BabelStream 进程。
副作用：子进程内 `torch.xpu.device_count()` 只看到 1 张卡（这是**预期行为**，不是 bug）。
报告 meta 里的 `gpu_count` 因此可能显示 1。

### 5.6 oneAPI 环境
`source /opt/intel/oneapi/setvars.sh` **返回码是 3**，用 `&&` 串会断链；
脚本里若开了 `set -u` 还会直接退出（setvars 引用了未定义变量）→ 先 `set +u` 再 source。

---

## 6. 与 TODO 文档 §6「判读标准」的对照

| TODO 判据 | 实测 | 判定 |
|---|---|---|
| 软件带宽 ≫ 硬件计数 | — | **无法判定**：硬件计数器本身失效（§5.4） |
| 小数组极高、大数组骤降 | GPU: L2 内 1717 GB/s → HBM 900 GB/s；<br>主机: L3 内 675 GB/s → DRAM 39.6 GB/s | ✅ **命中**，已按大数组口径出结论 |
| 双卡同时带宽显著下降 | 839+840 = 2.00× 线性 | ❌ **无竞争** |
| dtype 不同导致带宽差异大 | bf16 vs fp32 完全一致 | ❌ **无差异** |
| 带宽达成率 < 70% | Copy 900/1229 = 73%；实用口径 696/1229 = 57% | ⚠ **临界**，Copy 刚好过线，但单向 read 只有 57% |

---

## 7. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| → ② 算力峰值 | 算力/带宽比（arithmetic intensity）判断 GEMM 是否带宽受限 |
| → ④ Xe Link | 卡间 318 GB/s 规格 vs 卡内 HBM 696 GB/s → 跨卡通信是次瓶颈 |
| → ⑤ AI 测试 | Attention/LLM decode 的 memory-bound 上界就是这里的 **~700 GB/s** |
| → ⑦ Profiling | Roofline 的「带宽墙」= 700~900 GB/s（按访问模式取） |
| → ⑧ 能效 | GB/s per W：copy 满载 ~900 GB/s / ~270 W ≈ **3.3 GB/s/W** |

---

## 8. 未做 / 待补充

| 项目 | 状态 | 原因 / 后续 |
|---|---|---|
| 硬件读写计数器交叉验证 | ❌ 不可行 | 本驱动计数器失效（§5.4），等驱动修复或用 VTune/PTI 替代 |
| VTune `mem_bench` 交叉验证 | ⏳ 未做 | TODO §3 列了但本目录未接；需 GUI/CLI 配置，建议并入 ⑦ profiling |
| Xe Link 带宽 | → ④ | 归入 `04-interconnect-xelink`，本目录不重复 |
| GB/s per W 系统化扫描 | ⏳ 未做 | 需配合 `xpu-smi stats` 的 `GPU Power (W)` 做功耗曲线，建议并入 ⑧ 能效 |
| 大页 / NUMA 绑核对主机带宽的影响 | ⏳ 未做 | 本机只有 1 个 NUMA node，收益有限 |
