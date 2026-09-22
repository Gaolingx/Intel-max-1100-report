# benchmark/02-compute-peak — 算力峰值实测（ALU / XMX-DPAS）

对应文档：[`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md)（算力峰值测试方案）、
[`docs/hardware.md`](../../docs/hardware.md)（硬件规格）、
[`docs/Conclusion/02-compute-peak/`](../../docs/Conclusion/02-compute-peak/)（结论报告）。

本目录用 **四条互相独立的路径** 测算力，避免「自己验证自己」：

| # | 路径 | 工具 | 测的是什么 | 可信度 |
|---|---|---|---|---|
| 1 | 自研 SYCL 探针 `alu_peak` | icpx/DPC++ | 纯 FMA 循环（非可折叠递推）→ **ALU 峰值** | ❌ **绝对值已撤回**（见 §5.2） |
| 2 | 自研 SYCL 探针 `xmx_peak` | icpx + `joint_matrix` | 直接发射 **DPAS**（XMX）→ 原始矩阵峰值 | ✅ 已自校验 |
| 3 | PyTorch / oneDNN GEMM | `torch.matmul`、`torch._int_mm` | 成熟软件栈能达到的实际峰值 | ✅ **可信下界** |
| 4 | **`ze_peak`（Intel 官方）** | Level Zero + icpx | clpeak 移植 → **FP32/FP64/FP16 向量峰值**（第三方实现） | ✅ **仲裁者** |

再加一条取证：`clock` —— 主频 / 功耗 / 是否降频。

> ⚠ 四条路径给出的是**不同的量**，不要混用：自研探针测的是「硬件上限」，
> oneDNN 测的是「工程可达」，`ze_peak` 则是「别人写的硬件上限」。
> 本目录把三者都报出来，并**显式标注差异**。

> ✅ **2026-09-22 裁定**：FP32 向量峰值 = **22.22 TFLOPS**（路径 3 给出 22.13 = 99.6%，
> 路径 4 给出 21.87 = 98.4%），路径 1 的 50.75 是探针 artefact，**已撤回**。
> 证据链见 `sycl/exp/README.md` 与 §5.2。

---

## 1. 目录结构

```
benchmark/02-compute-peak/
├── README.md                  本文档
├── run_bench.py               CLI 入口（5 个 suite：alu / xmx / torch / clock / zepeak）
├── sycl/
│   ├── alu_peak.cpp           自研 ALU 峰值探针（VEC/ACC/UNROLL 可宏定义）⚠️ 绝对值已撤回
│   ├── xmx_peak.cpp           自研 XMX/DPAS 峰值探针（sycl joint_matrix）
│   ├── alu_clock_probe.cpp    【死路，仅留档】用 SYCL 读 GPU 时钟 —— 本机不支持
│   └── exp/                   ★ 探针证伪实验（5 组对照 + ISA 证据）
│       ├── README.md          完整的裁定过程记录
│       └── alu_u1/u4/v1/acc4  4 个对照二进制
├── ze_peak_src/               ★ 第三方仲裁：Intel 官方 ze_peak 源码
│   ├── README.md              仓库出处 / 测试表 / 构建配方 / 网络坑
│   ├── PATCHES.md             1 行补丁 + shim 原因
│   ├── build.sh               一键构建（`./build.sh` / `./build.sh run 0`）
│   ├── shim/level_zero/       vendored v1.15.31 头文件（16 个）
│   ├── common/ ze_peak/       上游源码（25 文件）
│   ├── build/ze_peak          构建产物（271 KB）+ 5 个 .spv
│   └── logs/ze_peak_dev{0,1}.log
├── probes/
│   └── torch_gemm_peak.py     PyTorch GEMM / int8 / vector 交叉验证
├── build/                     探针编译产物（alu_peak、xmx_peak、alu_peak_vecN）
└── results/                   运行后自动生成的 JSON / Markdown 报告
```

---

## 2. 快速开始

```bash
cd /root/workspace/benchmark/02-compute-peak

python3 run_bench.py                    # 全部 5 个 suite，写 results/bench_<tag>.{json,md}
python3 run_bench.py --quick            # 冒烟（约 1/5 时间）
python3 run_bench.py alu xmx            # 只跑指定 suite
python3 run_bench.py zepeak             # 只跑第三方 ze_peak（会复用已有日志）
python3 run_bench.py --tag 20260922-194139

# PyTorch 路径需要 venv1（system-site-packages 里才有 torch+xpu）
ZE_AFFINITY_MASK=0 /root/workspace/venv1/bin/python run_bench.py torch
```

**`zepeak` suite 的运行方式**：它**不**编译任何东西，而是：
1. 检查 `ze_peak_src/build/ze_peak` 是否存在（不存在则提示先跑 `./build.sh`）；
2. 检查 `ze_peak_src/logs/ze_peak_dev{d}.log` 里有没有 `"Kernel duration"`；
   **有则直接复用**（`ze_peak` 每卡要跑 ~15 min，重跑很贵），没有才现场跑；
3. 解析 7 个 section（`sp/hp/dp/int_compute`、`global_bw`、`transfer_bw`、`kernel_lat`）
   并产出 crosscheck / arbitration 记录。

`run_bench.py` 会自动用 `icpx -fsycl -O3` 编译两个探针到 `build/`
（`icpx` 路径写死为 `/opt/intel/oneapi/compiler/2026.1/bin/icpx`）。

**实测参考产物**：[`results/bench_20260922-194139.md`](results/bench_20260922-194139.md)。
（更早的 `results/bench_20260922-192831.*` 是 v1，其中 XMX 占用率的解读有误 —— 当时把
「低占用率下等量工作等量耗时」误判为编译器消循环。**请只看 v2 及以后。**）

---

## 3. 五个 suite 分别测什么

### 3.1 `alu` —— ALU 峰值（自研 SYCL）

`sycl/alu_peak.cpp`：每个 work-item 跑 `ACC`(=8) 条独立的 `VEC`(=8) 宽 `sycl::vec` 依赖链，
链上做**非可折叠递推**（logistic map `acc = 1 - acc*acc`，故意选混沌、无闭式解，
编译器无法把它化简成常数），每链每内层迭代连做 `UNROLL`(=4) 次 FMA。

```
FLOP 计数 = global_size × VEC × ACC × UNROLL × iters × 2
```

- CLI：`./alu_peak <dtype|all> [global=7168] [iters] [warmup=3] [repeats=5] [local=128]`
- 输出：`DEVICE {json}`（含 `vec`/`acc`/`unroll`）、`RESULT {json}`（含 `value` GFLOPS/GOPS）
- 用 `guard` 写回（`global_id == SIZE_MAX`，运行期值）防止整段被 DCE 掉
- 编译：`icpx -fsycl -O3 -ffp-contract=fast`
- **VEC 扫描**：`-DALU_VEC={1,2,4,8,16}` 编出 `alu_peak_vecN` 分别跑

### 3.2 `xmx` —— XMX / DPAS 原始峰值（自研 SYCL `joint_matrix`）

`sycl/xmx_peak.cpp`：每个 sub-group 装载 1 个 A 片段 + 1 个 B 片段，
然后**在编译期展开的 `IT`(=128) 内层循环**里对 `NACC`(=4) 个独立累加器连续发 DPAS，
外面套一个**运行期 `outer`(=8192) 循环**：

```
dpas_count = sub_groups × outer × IT × NACC
FLOP 计数  = dpas_count × 2 × M × N × K        (bf16/fp16: M8N16K16；int8: M8N16K32)
```

- CLI：`./xmx_peak <dtype|all> [global=7168] [outer=8192] [warmup=3] [repeats=5] [local=16] [it=128]`
- 输出：`DEVICE`、`CHECK {dtype,expect,got,ok}`、`RESULT {...}`（含 `dpas_count`、
  `mac_per_cycle_eu_nominal`、`check_ok`）
- `CHECK`：把 A、B 全填 1，做 1 次 MAD，`C[0]` 必须等于 K → 验证**数值正确**（而不仅是快）

**joint_matrix API 备忘（踩坑记录，官方文档没写全）**

| 项 | 正确写法 |
|---|---|
| 头文件 | `#include <sycl/ext/oneapi/matrix/matrix.hpp>`（**不是** `experimental/matrix/`，该路径不存在） |
| 命名空间 | `sycl::ext::oneapi::experimental::matrix`（习惯 `namespace mx = ...`） |
| 模板 | `joint_matrix<Group, T, use, Rows, Cols, Layout>` |
| Group | `sycl::sub_group` |
| use | `mx::use::a` / `mx::use::b` / `mx::use::accumulator` |
| Layout | `mx::layout::row_major` |
| 形状 | bf16/fp16 `m8n16k16`；int8 `m8n16k32` |
| **指针** | `joint_matrix_load/store` **只接受** `sycl::multi_ptr<T, sycl::access::address_space::global_space>`；裸 `T*` 编译报 `could not match 'multi_ptr<...>' against 'T *'` |
| 不存在 | `joint_matrix_element_get`（只有 `joint_matrix_prefetch` / `joint_matrix_apply`）；没用到的累加器会被 DCE，**必须 store 出去** |

### 3.3 `torch` —— oneDNN 交叉验证

`probes/torch_gemm_peak.py`：

- GEMM：`torch.matmul` fp32 / bf16 / fp16；**int8 必须用 `torch._int_mm`**
  （`torch.matmul` 对 int8 返回 int8 会溢出），形状 4096³ / 4096×4096×16384 / 8192³
- vector：`add` / `mul` / `relu`（64 Mi 元素），fp32 & bf16，换算成 GB/s
- 计时：`torch.xpu.Event(enable_timing=True)`，取多次迭代最小值
- 输出：每行 `TORCHCOMPUTE {json}`，另加 `TORCHDEVICE {json}`

### 3.4 `clock` —— 主频 / 功耗取证

- `/sys/class/drm/card0/gt_*_freq_mhz`：`act/cur/min/max/boost/RP0/RP1/RPn`
- `xpu-smi stats -d 0`：Frequency / Power / Utilization
- **关键做法**：在**后台跑满载 ALU 的同时**采样，否则读到的全是空闲值
  （空闲时 `gt_act_freq_mhz` 会读 **0**）
- ⚠️ **注意**：`gt_cur/min/max/boost/RP0_freq_mhz` 都是**请求值**，不会反映自主降额；
  `gt_act_freq_mhz` 是**唯一会随负载变化**的节点，但其**绝对值噪声极大**、不收敛到标准 P-state
  （已观测 200~1400 乱跳）⇒ **只能当定性证据**。硬证据是 `xpu-smi stats` 的**温度与功耗**。
- 最后从实测算「等效执行宽度」，见 §5.3

### 3.5 `zepeak` —— 第三方向量峰值仲裁（★ 补做）

**源码**：`ze_peak_src/`，来自 **`oneapi-src/level-zero-tests/perf_tests/ze_peak`**
（Intel 官方，clpeak 的 Level Zero 移植，25 文件零外部依赖）。
**它不是 XMX 基准**（没有 DPAS），但恰好是 FP32 向量峰值的第三方独立实现。

构建（已写好脚本，`git clone` 在本机不通 → 逐文件抓，见 `ze_peak_src/README.md`）：

```bash
cd ze_peak_src
./build.sh          # 只构建 → build/ze_peak
./build.sh run 0    # 构建 + 跑 device 0 全量（⚠️ ~20 min）→ logs/ze_peak_dev0.log
./build.sh run 1    # 同上 device 1
```

覆盖 7 个 section：`sp_compute` / `hp_compute` / `dp_compute` / `int_compute` /
`global_bw` / `transfer_bw` / `kernel_lat`，每个 section 内部还会扫不同向量宽度变体
（`float` / `float2` / `float4` / `float8` / `float16`）。

> ⚠ **必须 `cd build/` 再启动** —— `.spv` 是按相对路径加载的。
> ⚠ stdout 重定向是**块缓冲**的：日志长时间不涨 ≠ 卡死，用 `ps` / `xpu-smi` 确认。

---

## 4. 关键实测结果（tag `20260922-194139` + `ze_peak` 补做）

### 4.1 ALU 峰值（自研 SYCL，`global=114688`）⚠️ **绝对数值已撤回**

> ❌ 下表的 fp32/fp64/fp16 绝对数值**不可信**（见 §5.2 与
> [`sycl/exp/README.md`](sycl/exp/README.md)）。请直接看 §4.0 的 `ze_peak` 第三方结果。

| dtype | 实测 ⚠️ | 单位 | vs 公式标称 22.22 | 裁定 |
|---|---:|---|---|---|
| fp32 | ~~50 750~~ | GFLOPS | ~~228.4%~~ | ❌ 撤回 |
| fp64 | ~~51 900~~ | GFLOPS | ~~233.6%~~ | ❌ 撤回 |
| fp16 | ~~52 600~~ | GFLOPS | ~~236.7%~~ | ❌ 撤回 |
| int32 | 18 000 | GOPS | 81% | ⚠️ 同源，不可信 |

### 4.0 ★ `ze_peak` 第三方向量峰值（device 0，`logs/ze_peak_dev0.log`，EXIT=0）

设备：`Intel(R) Data Center GPU Max 1100`，`deviceId 0x0bda`，`coreClockRate 1550`，
`maxMemAllocSize 48 946 688 000 B`。

| section | 最佳内核 | 值 | 同 section 其它变体 |
|---|---|---:|---|
| `sp_compute` (fp32) | **`float4`** | **21 871.6 GFLOPS** | `float` 21 843.3 / `float2` 21 820.9 / `float8` 21 759.7 / `float16` 21 533.1 |
| `hp_compute` (fp16) | **`half4`** | **43 381.3 GFLOPS** | `half` 34 545.5 / `half2` 43 118.3 / `half8` 43 188.7 / `half16` 42 842.5 |
| `dp_compute` (fp64) | **`double4`** | **16 074.0 GFLOPS** | `double` 16 005.9 / `double2` 15 904.7 / `double8` 15 792.6 / `double16` 13 978.7 |
| `int_compute` | **`int2`** | **6 342.5 GOPS** | `int` 6 333.5 / `int4` 6 333.8 / `int8` 4 840.5 / `int16` 5 399.3 |
| `global_bw` | `float` | **688.7 GB/s** | `float2` 685.1 / `float4` 661.8 / `float8` 674.0 / `float16` 678.3 |
| `transfer_bw` | GPU Copy Shared→Host | 53.04 GB/s | Write 39.02 / Read 53.04 / Host→S 39.18 / SysMem→S 8.09 / SysMem←S 8.21 |
| `kernel_lat` | Kernel duration | 15.69 µs | launch 5.49 / immediate-CL 5.50 |

**推导比值**（这就是裁定依据）：

| 比值 | `ze_peak`（第三方） | 自研 `alu_peak` | Xe-HPC 架构应为 |
|---|---:|---:|---|---|
| `sp / 公式 22.2208` | **0.9843** | 2.2839 | 1.0 |
| `hp / sp` | **1.9835** | 1.037 | 2.0 |
| `dp / sp` | **0.7349** | 1.022 | 0.5~0.75 |

⇒ **`ze_peak` 既复现了公式绝对值，又复现了 dtype 位宽比；自研探针两项都做不到。**
⇒ **裁定：FP32 向量峰值 = 22.22 TFLOPS，自研探针绝对值撤回。**

#### 4.0.1 ⚠️ 时长口径：长跑整轮会遇到热/功耗降额（2026-09-22 新发现）

`-i 50 -w 10` 的**整轮**约 20 min/卡（每测试项 `get_max_work_items() × 512~2048` ≈ 10⁹~10¹⁰
work-item；fp64 用 512，fp16/fp32/int 用 2048）。
期间 i915 的 **`gt_act_freq_mhz`**（**唯一会随负载变化**的频率节点，空闲读 0）**在 200~1400 MHz
之间乱跳**（1350/1250/1150/1000/950/800/700/650/600/500/450/400/350/300…，
**几乎每次采样都不同**），而 `gt_cur/max/min_freq_mhz` 与 xpu-smi 的 `GPU Frequency`
始终只报 **1550（请求值）**。

> ⚠️ **`gt_act_freq_mhz` 只能当定性证据**：其绝对值噪声极大、不收敛到标准 P-state
> （标准态只有 RP0=1550 / RP1=1000 / RPn=200）⇒ **可引用的是「它在大幅摆动 ⇒ DVFS 很活跃」
> 这个定性事实，而不是它的数值**。降额的**硬证据是温度与功耗**：
> fp64 单项跑 = 233–243 W / 86–89 °C；整轮长跑采样到 **305 → 330 W
> （> 300 W 名义上限）/ 92 → 101 °C**。101 °C 已逼近 PVC 结温上限。

**后果：`logs/ze_peak_dev1.log`（整轮）在 fp64 与 int32 段偏低，不可用。**

| 题项 | dev0 整轮 | dev1 整轮（降额） | 短跑 `-i 3 -w 1`（dev0 / dev1 ×2 次） |
|---|---:|---:|---|
| fp32 `float4` | 21 871.6 | 21 871.9 ✅ | 21 871.6 / 21 873.2 / 21 872.1 / 21 871.6 |
| fp16 `half4` | 43 381.3 | 43 386.5 ✅ | 43 397.4 / 43 385.8 / 43 384.7 / 43 391.2 |
| fp64 `double4` | 16 074.0 | **14 278.8（−11%）** ⚠️ | 16 074.1 / 16 074.7 / 16 074.0 / 16 074.2 |
| int32 `int2` | 6 342.5 | **3 640.0（−43%）** ⚠️ | dev0 6 431.4 / 6 425.2；dev1 6 178.8 / 6 175.9 |

- **机制：`ze_peak` 报的是「平均吞吐」而不是「峰值吞吐」。**
  `run_kernel()`（`ze_peak.cpp:861`）算的是 `总工作量 / 总墙钟时间`，时间由
  `for (i < iters) { run_command_queue(); synchronize_command_queue(); }`
  **累计 50 次发射**测得 ⇒ **这 50 次里任何一段变慢都会拉低平均值**。
  而 `-a` 的分段顺序是 `sp → hp → dp → int → global_bw → transfer_bw → kernel_lat`，
  即 **dp/int 最晚测**，恰好落在降额已建立之后；`sp`/`hp` 最早测，完全不受影响。
- 同段的宽度序列 `d → d2 → d4 → d8 → d16` 也是按时间先后排列的：dev1 整轮 dp 段
  14 278.8 > 12 355.7 > 12 212.5 > 11 092.3 > 9 046.96 GFLOPS 的**单调恶化**，
  正是「测试途中持续降额」的时间签名；dev0 整轮同序列基本平（仅最后一个 d16 掉到 13 978.7，
  是刚进入降额的开端）。dev1 整轮紧跟 dev0 运行、起始温度已 86 °C，故落后得多。
- **短跑跨 2 卡 × 2 次重复性 ≈ 10⁻⁵**，是唯一可信的绝对值口径；`fp32`/`fp16` 段恰好在降额
  发生前测完，故跨卡跨次完全一致 —— 这也是 FP32 裁定可信的旁证。
- ⇒ **硬件结论（已同步进 `docs/TODO/08-power-efficiency.md` 与 Conclusion 02）：
  长时间满载确实会触发热/功耗降额（温度峰值 101 °C、功耗 305~330 W 越过 300 W 上限，
  ≈0.87×）；先前「单卡永远不会碰上功耗墙」的说法作废。**


### 4.2 ALU 占用率扫描（fp32）—— 找拐点

> ⚠️ **吞吐列绝对值已作废**；拐点位置与线性度形状仍有效。

| global | 每硬件 lane 的 work-item | TFLOPS ⚠️ | |
|---:|---:|---:|---|
| 7 168 | 1 | 10.09 | 1/5 峰值，延迟受限 |
| 14 336 | 2 | 16.67 | |
| 28 672 | 4 | 20.50 | |
| 57 344 | 8 | 25.44 | |
| **114 688** | **16** | **50.75** | ← **拐点** |
| 229 376 | 32 | 50.76 | 线性 ✓ |
| 458 752 | 64 | 50.83 | 线性 ✓ |

### 4.3 ALU 向量宽度扫描（fp32）⚠️

| VEC | 1 | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| TFLOPS ⚠️ | 34.8 | 45.6 | **52.8** | 50.8 | 51.3 |

→ VEC≥4 后拉平：该循环是 **issue 受限**，不是宽度受限。

> ❌ **但 `VEC=1` = 34.8 TFLOPS > 22.22 硬件上限 ⇒ 这一条本身就否定了整个探针**
> （单条标量 FMA/lane 怎么可能超过 16-lane 向量的两倍？）。详见
> [`sycl/exp/README.md`](sycl/exp/README.md) 实验 C。

### 4.4 XMX / DPAS 峰值（自研 SYCL，`outer=8192, it=128, nacc=4`）

`CHECK` 全部通过（bf16/fp16 期望 16、int8 期望 32，`got == expect`）→ **DPAS 通路可用且数值正确**。

| sg/EU | global | bf16 TFLOPS | fp16 TFLOPS | int8 GOPS |
|---:|---:|---:|---:|---:|
| 1 | 7 168 | 118.1 | 118.1 | 236.2 |
| 2 | 14 336 | 141.9 | 141.9 | 283.9 |
| 4 | 28 672 | 177.4 | 177.4 | 354.9 |
| **8** | **57 344** | **354.9** | **354.9** | **709.7** | ← **拐点** |
| 16 | 114 688 | 354.9 | 354.9 | 709.9 | 线性 ✓ |
| 32 | 229 376 | 355.0 | 355.0 | 710.0 | 线性 ✓ |

- **bf16 = fp16 = 355 TFLOPS**，**int8 = 710 GOPS = bf16 的 2.00×**（正好等于 K=32 vs K=16）
- 折算 **256 MAC/clk/EU**（= 355.5 TFLOPS ÷ 448 EU ÷ 2 ÷ 1.55 GHz）
- 产品常引用值 ≈176 TFLOPS(bf16) → **实测是它的 2.0×**

### 4.5 oneDNN / PyTorch 交叉验证

| 项 | 最佳形状 | 实测 | 参照 | 占比 |
|---|---|---:|---|---:|
| fp32 GEMM | 8192³ | **22.13 TFLOPS** | 公式标称 22.22 | **99.6%** |
| bf16 GEMM | 4096³ | **225.9 TFLOPS** | 自研 DPAS 355.5 | 63.5% |
| fp16 GEMM | 8192³ | **237.6 TFLOPS** | 自研 DPAS 355.5 | 66.8% |
| int8 `_int_mm` | 8192³ | **416.2 GOPS** | 自研 DPAS 711.1 | 58.5% |

> 同 dtype 换个形状差很多：bf16 在 `4096×4096×16384` 只有 **185.4**，
> fp16 在 `4096×4096×16384` 只有 **183.4** —— 长 K 形状掉 ~18%，
> 与 `docs/precision-support.md` 记录的 **16384³ 塔口**同源。

vector（64 Mi 元素）：

| op | fp32 GB/s | bf16 GB/s |
|---|---:|---:|
| add | 528.0 | 530.0 |
| mul | 526.2 | 528.7 |
| relu | 406.9 | 411.6 |

> ⚠ vector 只有 ~530 GB/s，而 `03-memory-bandwidth` 里 BabelStream copy 能到 ~900 GB/s。
> 原因是这里只用了 `n_elem = 2^26`（fp32 = 256 MiB）且**没有 warmup 到 L2 稳定**，
> 属于「torch 元素级算子开销」而不是带宽极限 —— 带宽口径请以 `03` 为准。

### 4.6 主频 / 功耗

| 项 | 值 |
|---|---|
| `gt_min_freq_mhz` | 1550 |
| `gt_max_freq_mhz` | 1550 |
| `gt_boost_freq_mhz` | 1550 |
| `gt_RP0 / RP1 / RPn` | 1550 / 1000 / 200 |
| `gt_cur_freq_mhz` | 1550（空载/满载同；**只是请求值**） |
| `gt_act_freq_mhz`（空载 / 短时满载 / **长时满载**） | **0**（读不到）／ 1550 ／ **200~1400 之间乱跳** |
| `xpu-smi` GPU Utilization | **100%** |
| `xpu-smi` GPU Power（短时满载） | **171 W** |
| `xpu-smi` GPU Power（**`ze_peak` 长时满载**） | **305 → 330 W ⚠️（已越过 300 W 上限）** |
| `xpu-smi` Temp / Mem Temp（长时满载） | **92 → 101 °C ⚠️ / 76 °C** |

**min == max** ⇒ 频率在**请求侧**被锁定、没有 boost 空间；**短时**满载不降额。

> ⚠️ **修正（2026-09-22）**：此前「满载仅 171 W ⇒ 远低于 300 W ⇒ 不节流」的结论
> **只在短负载下成立**。`ze_peak` 每卡连续 ~20 min 实测
> **305 → 330 W / 92 → 101 °C** —— **长时间满载确实会触发热/功耗降额，−13%，≈0.87×**。
> ⇒ 引用峰值数字必须注明是**短时**读数；长时稳态性能打 ~0.87 折扣。
>
> ⚠️ **`gt_act_freq_mhz` 只能当定性证据**：它是 i915 上**唯一会随负载变化**的频率节点，
> 但其**绝对值噪声极大、不收敛到标准 P-state**（标准态只有 RP0=1550 / RP1=1000 / RPn=200；
> 实测还出现 1400/1150/950/650/600/450/400/350/300 等几乎每次都不同的值）。
> **可引用的是「它在大幅摆动 ⇒ DVFS 很活跃」，不是它的数值。**
> 降额的**硬证据是温度与功耗**（101 °C / 330 W）以及
> `ze_peak` 长跑整轮 fp64/int32 段的单调偏低（见 §4.0.1）。
> 该修正同时影响 `docs/TODO/08-power-efficiency.md`。

---

## 5. 坑与注意事项

### 5.1 ★ 最重要的坑：占用率不足时「工作量翻倍、耗时不变」

**现象**：`xmx_peak bf16`，固定 `outer=8192/it=128/nacc=4`：

```
g=28672 (4 sg/EU, 1792 sub-group) -> 0.173508 s   dpas_count = 7.52e9
g=57344 (8 sg/EU, 3584 sub-group) -> 0.173508 s   dpas_count = 1.50e10   ← 计数翻倍，耗时一样
```

按 FLOP 计数除出来的吞吐凭空翻倍 → 很容易得出两个**错误**结论之一：
①「编译器把循环消掉了」；②「这台卡有 700 TFLOPS」。

**真相**（用对照实验证明，固定总 DPAS、只改 work-group 数）：

```
总 DPAS 7.516e9，g=28672 (4 sg/EU) -> 0.173517 s  = 177 TFLOPS
总 DPAS 7.516e9，g=57344 (8 sg/EU) -> 0.086764 s  = 355 TFLOPS   ← 等量工作，快一倍
```

耗时**确实**随 work-group 数变 → 计数没问题，问题是**延迟受限**：
XMX 流水线要每 EU ≥8 个并发 sub-group 才打满，否则同样的活要花近一倍时间。

**判读纪律**（本目录已固化为自动检查）：

1. 测峰值前**必须先把 work-group 数扫过拐点**，否则测到的是延迟；
2. 拐点之上耗时对工作量必须**严格线性** —— 用它做自校验：

   | 探针 | 低点 | 高点 | 比值 |
   |---|---|---|---|
   | ALU | g=114688 → 1.2133 s | g=229376 → 2.4234 s | **×2.00** ✓ |
   | XMX | occ8 → 0.1735 s | occ16 → 0.3469 s | **×2.00** ✓ |

3. 拐点之下「等量工作同样耗时」是**正常现象**，不是 bug。

`run_bench.py` 把这三件事都做成了记录：`saturation_knee`、`linearity_check`、
`linearity_check_below_knee`（故意跨拐点留档）。

### 5.2 ★ 第二重要的坑：标称公式与实际不符（口径冲突）【✅ 已裁定 2026-09-22】

| 口径 | fp32 ALU 峰值 | 说明 |
|---|---:|---|
| 公式标称 `448 EU × 16 lane × 2 × 1.55 GHz` | **22.2208 TFLOPS** | 与 clinfo 的 448 CU 一致 |
| oneDNN GEMM 实测 | **22.13 TFLOPS** | 99.6% 公式值 |
| **`ze_peak` `sp_compute`（第三方）** | **21.8716 TFLOPS** | **98.4% 公式值** |
| ~~自研纯 FMA 探针~~ | ~~50.75 TFLOPS~~ | ~~2.28×~~ ❌ **已撤回** |

**裁定：以 22.22 TFLOPS 为准。自研探针的绝对值是 artefact。**

证据链（5 组实验 + ISA，完整版见 [`sycl/exp/README.md`](sycl/exp/README.md)）：

1. **位宽比（★ 决定性）**：`ze_peak` `hp/sp`=**1.98**、`dp/sp`=**0.735**（符合 Xe-HPC 架构）；
   自研探针 `hp/sp`=**1.04**、`dp/sp`=**1.02**（**分辨不出 dtype**）
   ⇒ 一个把所有 dtype 都测成同样速度的「峰值探针」测的不是 FMA 吞吐。
2. **`VEC=1` 超限**：自研探针 `VEC=1` = **34.8 TFLOPS > 22.22 硬件上限**（157%）→ 不可能。
3. **ISA 事实**：`IGC_ShaderDumpEnable=1` 两次独立导出，FP32 kernel 主循环
   231 条指令中 **209 条是 `mad (1|M0)`**（1-wide 标量寄存器运算，源操作互不相同）
   ⇒ `sycl::vec<T,8>` 的 FMA 循环被 IGC **完全标量化**，与探针的 FLOP 模型不符。
4. **EU 数无争议**：同一份 `HardwareCaps.txt` 给出 `EUCount = 448`、
   `ThreadCount = 3584`（=8 线程/EU）⇒ 旧猜想「本 ES 部件 EU 数 ≠ 448」排除。
5. **旧猜想「探针计数偏低」 (c) 才是真相**。

**保留的探针结论**（定性，不受 FLOP 口径影响）：占用率拐点位置、饱和区线性度 ×2.00、
`VEC ≥ 4` 后拉平。**作废**：`peak.fp32/fp64/fp16/int32` 绝对值、
`implied_exec_width.implied_lanes_per_eu`（旧值 36.5）、以及「每 EU 37 条 FP32 lane」的推断。

**附带收获**：`ze_peak` `dp/sp = 0.735` 是 FP64 ≠ FP32/2 的**第三次独立确认**
（前两次：torch 0.78、0.77）。

### 5.3 其它坑

| 坑 | 现象 | 处理 |
|---|---|---|
| `joint_matrix` 头文件路径 | `experimental/matrix/` 不存在 | 用 `<sycl/ext/oneapi/matrix/matrix.hpp>` |
| `joint_matrix_load` 拒收裸指针 | `could not match 'multi_ptr<...>' against 'T *'` | 用 `sycl::multi_ptr<T, global_space>(ptr)` |
| 累加器被 DCE | 只 store `acc[0]` 时 `acc[1..]` 被优化掉 | **所有**累加器都要 store 到互不重叠的地址 |
| 运行期循环 + 不变操作数 | 编译器把整段消掉 | 内层用**编译期** `IT` + `#pragma unroll`，外层才是运行期循环 |
| `icpx` 无法产出设备 IR | `-S -emit-llvm` → `IR output is not supported`；`-fno-sycl-device-code-split` 未识别 | 无法用 SPIR-V/PTX 反汇编验证，只能靠 §5.1 的对照实验 |
| SYCL 读 GPU 时钟 | `clock_scope::device` 不支持（aspect `ext_oneapi_clock_device` 缺失） | `alu_clock_probe.cpp` 是**死路**，只留档；改用 sysfs |
| `gt_act_freq_mhz` 读 0 | 空闲未采样时 | 必须在**满载进行中**采样 |
| `xpu-smi dump` 挂死 | 打印表头后卡住，`Terminated` rc=143 | 只用 `xpu-smi stats -d 0`（~1.5 s） |
| `xpu-smi` 内存计数器是假的 | `GPU Memory Read/Write (kB/s)` 恒为 ~576 kB/s | 不要引用；带宽以 `03` 的 BabelStream 为准 |
| `torch.matmul` 对 int8 返回 int8 | 溢出 | 用 `torch._int_mm`（int32 累加） |
| `python3 -m venv` 失败 | `ensurepip` 不可用 | 用 `uv venv --python /usr/bin/python3.13 --system-site-packages` |
| `source setvars.sh` 返回 rc=3 | `set -u` 下还报错 | 先 `set +u`，且**不要**用 `&&` 串联 |

---

## 6. 判读标准对照（vs `docs/TODO/02-compute-peak.md`）

| TODO 中的判读标准 | 本次实测 | 结论 |
|---|---|---|
| fp32 ALU 达到标称 22.2 TFLOPS 的 ≥90% | oneDNN 22.13（**99.6%**）/ `ze_peak` 21.87（**98.4%**）/ ~~自研 50.75~~ | ✅ **达成，口径已定标为 22.22**（见 §5.2） |
| bf16 XMX ≥ 150 TFLOPS | 自研 355 / oneDNN 227 | ✅ 远超 |
| int8 XMX ≥ 300 TOPS | 自研 710 / oneDNN 427 | ✅ 远超 |
| fp64 ≥ 标称的 90% | torch 17.37（**0.78× fp32**）/ `ze_peak` 16.07（**0.735× fp32**） | ✅ 达成；但**标称本身错**：「FP64 = FP32/2」需改为 0.735~0.78 |
| 实测 ≥ 标称值的 90% 视为通过 | ~~fp32 自研 228%~~ → 已判定为探针 artefact | ✅ 异常已解释，非硬件超标 |
| 满载主频不低于标称 90% | 短时 1550（请求侧锁定）；**长时降额，温度 101 °C / 功耗 330 W** | ⚠️ **需注明时长**；`gt_act_freq_mhz` 仅作定性 |
| 满载不撞功耗墙 | 短时 171 W；**长时 305 → 330 W（越过 300 W 上限）** | ⚠️ **修正：长时会撞墙** |

---

## 7. 与其它目录的衔接

| 本目录的发现 | 影响 |
|---|---|
| **fp32 ALU = 22.22 TFLOPS（已裁定）** | `docs/precision-support.md` 的 `FP64_ALU_RATIO = 0.78` **保持不动**（torch 口径）；若改用 `ze_peak` 向量口径则为 0.735 |
| **fp64 = 0.735~0.78 × fp32**（三次独立确认） | `docs/hardware.md` / `docs/TODO/` 里「FP64 = FP32/2」全部改为实测比值 |
| **长时满载：温度 92 → 101 °C、功耗 305 → 330 W（越过 300 W 上限）** | → `docs/TODO/08-power-efficiency.md`：**删除「单卡永远不会碰上功耗墙」**，改用「短时 vs 长时」双口径；长时性能需打 ~0.87 折扣。⚠️ `gt_act_freq_mhz` 的绝对值**不可引用**（噪声大），只能当定性证据 |
| bf16 = fp16 = 355 TFLOPS，int8 = 2× | 佐证 `05-ai-dl` 的 XMX 结论（bf16 237 / int8 400 TOPS） |
| vector 只有 ~530 GB/s（`ze_peak` 688.7 GB/s） | 那是 torch/单探针的下限，**不是**带宽峰值 → 带宽看 `03-memory-bandwidth` |
| XMX 需要 ≥8 sg/EU 才打满 | 给 `05-ai-dl` 的调优留了空间：小 batch/小 shape 会掉进延迟受限区 |
| oneDNN 只到自研 DPAS 的 60~64% | 说明 oneDNN 的 GEMM 还有 ~35% 的调优空间（或受 L2/调度限制） |
| **`ze_peak` 二进制已就绪** | `06-hpc-apps` / `08-power-efficiency` 可直接用 `ze_peak_src/build/ze_peak` 做稳态降频/功耗实验 |
| 需要硬件计数器 | 依赖 [`docs/TODO/07-profiling.md`](../../docs/TODO/07-profiling.md)（VTune / PTI）—— 但 FP32 口径冲突**已不再需要它** |

---

## 8. 未做 / 待补充

| 项 | 状态 |
|---|---|
| 硬件计数器佐证（EU active、XMX pipe util、FP32 pipe util） | ⚠️ 属 `07-profiling`；本轮未接入 VTune/PTI。**但 FP32 口径冲突已由 `ze_peak` 独立解决，不依赖它** |
| **`ze_peak`（Level Zero 官方向量/带宽峰值）** | ✅ **已补做（2026-09-22）**。源码 `ze_peak_src/`（25 文件，`oneapi-src/level-zero-tests/perf_tests/ze_peak`），产物 `build/ze_peak`，日志 `logs/ze_peak_dev{0,1}.log`，runner 已接入 `zepeak` suite。<br>**FP32 `sp_compute` = 21 871.6 GFLOPS = 公式 98.4%** ⇒ 据以裁定 §5.2 的口径冲突。构建踩坑见 `ze_peak_src/PATCHES.md` 与 [`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md) §3.7 |
| **自研 ALU 探针重写** | ⬜ 建议。若还要 ALU 绝对数字，照 `ze_peak` 的写法重写（宽向量 + 少量独立依赖链 + 常数总 FLOP 数），不要再用 logistic 递推 |
| `benchdnn`（oneDNN 微基准） | 本机未安装 |
| nvfp4/mxfp8 等低精度**峰值** | 见 `docs/precision-support.md`：PVC 上全是软件回退，比 bf16 慢，无峰值可言 |
| TF32 / FP6 | torch 无对应 dtype，无法测试 |
| 双卡并发算力（2×448 EU） | 属 `04-interconnect-xelink` 与 `06-hpc-apps` 的范畴。`ze_peak` 的 `run 1` 已在做 dev1 单卡满载，可扩展为双卡并发 |
| 反汇编验证（SPIR-V / ISA） | ⚠️ **部分可做**：`icpx` 仍产不出设备 IR，但 `IGC_ShaderDumpEnable=1` 可以导出 IGC 后端的 `.asm`/`HardwareCaps.txt`（本次裁定的关键证据之一） |
