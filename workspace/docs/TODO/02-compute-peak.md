# ② 算力峰值测试

**优先级：P1**
**文档位置：** `docs/TODO/02-compute-peak.md`

---

## 1. 测试目标

1. 拿到 **FP32 / FP64 / INT8 / INT16 / BF16 / FP16** 及 **XMX（矩阵引擎）** 的实际峰值
2. 与理论值对比，计算**达成率**
3. 量化 **XMX 相对常规 ALU 的加速倍数**（这是 PVC 最核心的性能特征）
4. 建立后续所有 AI / HPC 测试的「能力上界」参照

## 2. 理论参考值

| 项目 | 计算 / 值 | 置信度 |
|---|---|---|
| FP32 峰值 | 448 EU × 16 lane × 2 (FMA) × 1.55 GHz ≈ **22.2 TFLOPS** | ✅ 已实测 22.16（见 §3.5） |
| FP64 峰值 | ~~PVC 上约为 FP32 的 1/2 ≈ 11.1 TFLOPS~~ → **实测 17.37 TFLOPS = FP32 的 0.78×** | ✅ 已实测，**原假设错误**（见 §3.5） |
| BF16/FP16 (XMX) | ✅ 实测 **232.8 / 233.9 TFLOPS** = FP32 的 10.5× | 已实测（见 §3.5） |
| INT8 (XMX) | ✅ 实测 **398.9 TOPS** = FP32 的 18.0× | 已实测（见 §3.5） |

> ⚠️ 不要引用估算值作为结论。本测试的目的正是**把上表替换为实测值**。
> 完整实测矩阵见 [`../precision-support.md`](../precision-support.md)。

---

## 3. 工具与准备

| 工具 | 状态 | 说明 |
|---|---|---|
| **`ze_peak`** | ✅ 已构建 | Level Zero peak benchmark，最贴近硬件的能力探测。产物在 `benchmark/02-compute-peak/ze_peak_src/build/ze_peak` |
| **BabelStream** | ❌ 需构建 | 主要是带宽，但含算力项，可与带宽交叉验证 |
| **PyTorch (`torch.xpu`)** | ✅ 已装 2.14.0+xpu | 最省事，`matmul` sweep 即可得 TFLOPS |
| **triton-xpu** | ✅ 已装 3.8.0 | 手写 kernel 测有效算力（更贴近真实上限） |
| **oneMKL** | ✅ 已装 2026.1 | GEMM / 含 GPU offload |
| **oneDNN `benchdnn`** | ❌ 需构建 | GEMM / 卷积 micro-benchmark |

### 构建 `ze_peak`（✅ 已补做，配方已修正）

> ✅ **可用的构建配方**（2026-09-22 实测通过）：
> ```bash
> cd /root/workspace/benchmark/02-compute-peak/ze_peak_src
> ./build.sh            # = g++ -O3 -std=c++17 -fcommon -I shim -I ze_peak/include \
>                       #        -I common/include ze_peak/src/*.cpp common/src/ze_app.cpp \
>                       #        -o build/ze_peak -lze_loader -lpthread
> ./build.sh run 0      # 跑 device 0，日志写 logs/ze_peak_dev0.log
> ```
> 需要 `shim/level_zero/`（自洽 v1.15 头集）与那 1 行 `extern bool verbose;` 补丁 ——
> 原因见 §3.7.2 与 `ze_peak_src/PATCHES.md`。

以下**是历史记录（错误配方，勿用）**：

```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/compute-runtime.git   # ❌ 该仓库没有 ze_peak
find compute-runtime -iname "*peak*" -o -iname "*ze_peak*"
cd <ze_peak_dir>
cmake -B build -H. -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ze_peak
```
> ⚠️ **上面这段构建说明有两处错误**（2026-09-22 联网核实后修正，见 §3.6/§3.7）。
> ① **仓库错了**：`intel/compute-runtime` 里**没有** `ze_peak`（全仓库代码搜索为空）。
> 正确出处是 **`oneapi-src/level-zero-tests/perf_tests/ze_peak`**。
> ② **输出项错了**：`ze_peak` 是 **clpeak 的 Level Zero 移植**，只有 **向量（SIMD）** 测试，
> 提供 `global_bw` / `hp_compute`(fp16) / `sp_compute`(fp32) / `dp_compute`(fp64) /
> `int_compute`(平台整数) / `transfer_bw` / `kernel_lat`。
> **没有 INT8/INT16 之分，更没有 XMX / DPAS / bf16 / FP8**。

### 构建 BabelStream
```bash
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream
cmake -B build -H. -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/babelstream
```

---

## 3.5 实测修正（2026-09-21）

用 `benchmark/05-ai-dl --suite precision --large` 实测后，上表需要修正/补充：

| 项目 | 上表假设 | **实测** | 结论 |
|---|---|---|---|
| FP32 峰值 | 22.22 | **22.16** TFLOPS @16384³ | ✅ 达成率 100%，公式正确 |
| FP64 峰值 | 11.11（= FP32/2） | **17.37** TFLOPS @2048³ | ❌ **假设错误**，实测为 FP32 的 **0.78×** |
| FP16 / BF16 | 待实测 | **233.9 / 232.8** TFLOPS | ✅ 原生 XMX，= FP32 的 **10.5×** |
| INT8 | 待实测（≈BF16×2） | **398.9** TOPS | ✅ 原生 XMX，= FP32 的 **18.0×** |
| INT4 | 待实测（≈BF16×4） | **44.8** TOPS（W4A16） | ❌ 仅 **0.11× INT8**，远未达 XMX INT4 理论值 |
| FP8 / MXFP8 / MXFP4 / NVFP4 | — | 80.9 / 49.8 / 32.1 / 29.7 TFLOPS | ⚠ **本卡 XMX 不支持 FP8/FP4**，全走软件回退，**比 BF16 慢 1.6~7.7×** |

> ⚠ **16384³ 存在吞吐悬崖**：FP16 233.9→97.3、BF16 232.8→117.0（可复现，偏差<1%），
> 但 INT8 与 FP32 不受影响。**判断 XMX 峰值请用 4096³~8192³，不要用 16384³。**

完整数据见 [`../precision-support.md`](../precision-support.md)。

---

## 3.6 `ze_peak` 核实与「为什么第一轮没有构建」（2026-09-22）

> ⚠️ **本节是历史记录**。`ze_peak` **已于 2026-09-22 补做完成**（抓取 → 构建 → 双卡跑通），
> 结果与裁定见 §3.7。下面保留当时"为什么先跳过"的推理链，便于回溯取舍过程。

本轮一开始**没有构建 `ze_peak`**，改用自研 SYCL 探针（`benchmark/02-compute-peak/sycl/`）。
事后联网核实，原因与事实如下：

**① 环境里本来就没有，且原构建配方是死路**
```bash
find /opt/intel/oneapi -iname "*ze_peak*"    # → 空：oneAPI 2026.1 不随附
```
`ze_peak` 不在 oneAPI 安装包内；而 §3 给的 `git clone intel/compute-runtime` 也找不到它
（该仓库无此代码）。**正确出处：`oneapi-src/level-zero-tests/perf_tests/ze_peak`**（公开仓库）。

**② 它测不到本轮真正要回答的问题**
`ze_peak` 是 **clpeak 的 Level Zero 移植 → 纯向量（SIMD）基准**，测试项只有：
`global_bw` / `hp_compute`(fp16) / `sp_compute`(fp32) / `dp_compute`(fp64) /
`int_compute`(平台整数) / `transfer_bw` / `kernel_lat`。
**它完全没有 XMX / DPAS 测试**。本轮最关键的三个结论 ——
XMX bf16 **355 TFLOPS**、INT8 **710 TOPS**、**占用率拐点 ≥8 sub-group/EU** ——
`ze_peak` 一个都产不出来，必须靠 `joint_matrix` 自研探针。

**③ 自研探针在方法学上更强**
`ze_peak` 只给一个数；自研探针额外提供 **占用率扫描**（找出拐点）、**向量宽度扫描**
（VEC=4 最优）、以及**饱和区线性度校验**（工作量 ×2 → 耗时 ×2.00），用它证明测量的是
真算力而不是编译器把循环消掉了。`ze_peak` 无法做这些交叉验证。

**④ 已知的真实缺口**
`ze_peak` 唯一的、不可替代的价值是：**它是别人写的、未经我手改动的第三方实现**，
可以作为 FP32 向量峰值 **22.13 vs 50.75 口径冲突**的独立仲裁者。

> ✅ **已于 2026-09-22 补做完成**：`ze_peak` `sp_compute` = **21 871.6 GFLOPS = 公式值 98.4%**，
> 据此裁定 **FP32 向量峰值 = 22.22 TFLOPS**，自研探针的 50.75 撤回。详见 §3.7。

**⑤ 现在是否可构建？—— 可以，成本很低**
核实后的事实：
- 只需 **25 个文件**（`perf_tests/ze_peak/**` 22 个 + `perf_tests/common/{include,src}` 3 个）；
- **零外部依赖**：只需 `level_zero/ze_api.h`、`zer_api.h` 与 `-lze_loader`
  （本机 `libze_loader.so.1.24.0` 已有）；
  源码 `#include` 列表里**没有任何 boost**（BUILD.md 里的 boost 是给别的测试用的）；
- 5 个 kernel 是**预编译 `.spv`**，且 `ze_peak.cpp:44` 是从**路径运行时加载**，无需 OpenCL 编译器；
- 编译：`g++ -O3 -std=c++17 src/*.cpp ../../common/src/ze_app.cpp -lze_loader -lpthread`。

**当时的实际阻碍**：`intel/compute-runtime` 的 `git ls-remote` 超时（rc=143）、
`oneapi-src/level-zero-tests` 的 `git clone` 报 `GnuTLS recv error (-110)`，
而 `api.github.com` / `raw.githubusercontent.com` 正常 —— 即 git-over-https 不稳，
只能逐个文件抓取。加上上述 ②③（它本就答不了关键问题），遂决定自研。

> **结论**：未构建 `ze_peak` 在当时是**可接受**的取舍，但**不是最优** ——
> 它本可以提供第三方 FP32 仲裁。
> **✅ 已于 2026-09-22 补做完成，结果见 §3.7。**

---

## 3.7 `ze_peak` 补做结果与 FP32 口径裁定（2026-09-22）

### 3.7.1 交付物

| 路径 | 内容 |
|---|---|
| `benchmark/02-compute-peak/ze_peak_src/` | 上游源码（25 文件）+ `shim/` + `build.sh` + `logs/` |
| `…/ze_peak_src/README.md` | 正确仓库、测试项表、构建配方、网络注意事项 |
| `…/ze_peak_src/PATCHES.md` | 对上游源码做的**唯一 1 行改动**及理由 |
| `…/ze_peak_src/build/ze_peak` | 构建产物（271,696 B） |
| `…/ze_peak_src/logs/ze_peak_dev{0,1}.log` | 两张卡的完整运行日志 |
| `benchmark/02-compute-peak/sycl/exp/README.md` | **探针可信度复核实验**（判定性证据） |

### 3.7.2 构建踩的三个坑

1. **`level_zero/zer_api.h` 缺失** —— `common/src/ze_app.cpp:9` 无条件 `#include
   <level_zero/zer_api.h>`，而本机 `/usr/include/level_zero/`（v1.13.1）没有该头。
   补 `zer_api.h` 后又因 `zer_api.h:18` 是**引号包含** `ze_api.h`（先搜自身目录）而报错。
   **解法**：从 containerd 快照 6 里拷一整套**自洽的 v1.15.31** 头（16 个）到
   `ze_peak_src/shim/level_zero/`，并把 `-I shim` 放在最前。`zer_api.h` 的符号
   **实际从未被使用**（`grep ZER` 只命中无关字符串）。
2. **`verbose` 重复定义** —— `ze_peak/src/ze_peak.cpp:15` 与 `common/src/ze_app.cpp:17`
   都定义了 `bool verbose = false;`（上游 2026 年改动引入的回归）。
   `-fcommon` **无效**（两者都是带初值的强定义）。
   **解法**：把 `ze_peak.cpp` 侧改成 `extern bool verbose;`（不动共享的 `common/`）。
3. **网络** —— 本机 git-over-https 完全不可用（`git ls-remote` rc=143、
   `GnuTLS recv error (-110)`），但 `api.github.com` / `raw.githubusercontent.com`
   可用。按用户要求**每文件重试 12 次**：`25/25 成功，failed: []`。

### 3.7.3 GPU0 实测（`-d 0 -a -i 50 -w 10`，EXIT=0）

设备：`Intel(R) Data Center GPU Max 1100`，`deviceId 0x0bda`，`coreClockRate 1550`，
`isSubdevice FALSE`，`maxMemAllocSize 48,946,688,000 B`。

| 分组 | 最佳 kernel | 值 | 同组其他宽度 |
|---|---|---|---|
| `sp_compute`（**fp32 向量**） | `float4` | **21,871.6 GFLOPS** | f 21843.3 / f2 21820.9 / f8 21759.7 / f16 21533.1 |
| `hp_compute`（fp16 向量） | `half4` | **43,381.3 GFLOPS** | h 34545.5 / h2 43118.3 / h8 43188.7 / h16 42842.5 |
| `dp_compute`（fp64 向量） | `double4` | **16,074.0 GFLOPS** | d 16005.9 / d2 15904.7 / d8 15792.6 / d16 13978.7 |
| `int_compute`（整数） | `int2` | **6,342.5 GOPS** | int 6333.5 / int4 6333.8 / int8 4840.5 / int16 5399.3 |
| `global_bw`（HBM 带宽） | `float` | **688.7 GB/s** | f2 685.1 / f4 661.8 / f8 674.0 / f16 678.3 |
| `transfer_bw` | GPU Copy Shared→Host | 53.04 GB/s | W 39.02 / R 53.04 / H→S 39.18 / SysMem→S 8.09 / SysMem←S 8.21 |
| `kernel_lat` | Kernel duration | 15.69 µs | launch 5.49 ×2 |

**关键比值（架构自洽性检查）**

| 比值 | ze_peak 实测 | Xe-HPC 架构应为 | 判定 |
|---|---|---|---|
| `sp` / 公式标称 22.2208 | **0.9843** | 1.00（好的实现应 ≥95%） | ✅ 真的压到了 ALU 上限 |
| `hp` / `sp` | **1.9835** | 2.0（fp16 通道是 fp32 的 2×） | ✅ |
| `dp` / `sp` | **0.7349** | ½~¾（§3.5 已测定 0.78/0.77） | ✅ 第三次独立确认「FP64 ≠ FP32/2」 |

### 3.7.3.1 ⚠️ 重复性与「时长口径」：长跑整轮会遇到热/功耗降额

**先做重复性检验再谈绝对值**（`-a -i 3 -w 1`，每轮约 60 s，跨 2 卡 × 2 次）：

| 测试项 | dev0 #1 | dev0 #2 | dev1 #1 | dev1 #2 | 离散度 |
|---|---:|---:|---:|---:|---:|
| fp32 `float4` | 21 871.6 | 21 873.2 | 21 872.1 | 21 871.6 | **0.007%** |
| fp16 `half4` | 43 397.4 | 43 385.8 | 43 384.7 | 43 391.2 | 0.029% |
| fp64 `double4` | 16 074.1 | 16 074.7 | 16 074.0 | 16 074.2 | **0.004%** |
| int32 `int2` | 6 431.4 | 6 425.2 | 6 178.8 | 6 175.9 | 3.97%（卡间系统差 3.9%，卡内 0.15%） |

**但 `-i 50 -w 10` 的整轮（~20 min/卡）不可靠。** 整轮期间循环采样
`/sys/class/drm/card{0,2}/gt_act_freq_mhz`（i915 上**唯一会随负载变化**的频率节点，
空闲时读 0 —— 而 `gt_cur/max/min_freq_mhz` 与 xpu-smi 的 `GPU Frequency`
**始终只回显请求值 1550**），实测该节点**几乎每次采样都不同**：
1350 / 1250 / 1150 / 1000 / 950 / 800 / 700 / 650 / 600 / 500 / 450 / 400 / 350 / 300 / 200 MHz。
⚠️ **该节点的绝对值噪声极大、并不收敛到标准 P-state**（标准态只有 RP0=1550 / RP1=1000 /
RPn=200）⇒ 它**只能当定性证据**（"在 200~1400 之间大幅摆动 ⇒ DVFS 确实在激烈动作"），
**不能当成精确时钟读数去反算性能**。真正的硬证据是**温度与功耗**：xpu-smi 同步实测
fp64 单项跑 = 1550 MHz / 233–243 W / 86–89 °C；整轮长跑采样到
**305 → 330 W（> 300 W 名义上限）/ 92 → 101 °C**。101 °C 已逼近 PVC 结温上限，
持续满载必然触发热降额。

**为什么"长跑偏低"是必然的（机制）**：`ze_peak` 的 `run_kernel()`（`ze_peak.cpp:861`）
报的是 `总工作量 / 总墙钟时间`，而时间由
`for (i < iters) { run_command_queue(); synchronize_command_queue(); }`
**累计 50 次发射**测得 ⇒ **这 50 次里任何一段变慢都会拉低平均值**，它是"平均吞吐"
而非"峰值吞吐"。而 `-a` 的分段顺序是 `sp → hp → dp → int → global_bw → transfer_bw →
kernel_lat`，即 **dp 与 int 段是最晚测的**，恰好落在降额已建立之后；`sp`/`hp` 最早测，
恰好在降额之前 —— 所以它们跨卡跨次完全一致（这也是 §3.7.4 裁定可信度的旁证）。
同段的宽度序列 `d → d2 → d4 → d8 → d16` 也是**按时间先后**排列的：dev1 整轮 dp 段
14 278.8 > 12 355.7 > 12 212.5 > 11 092.3 > 9 046.96 GFLOPS 的**单调恶化**，
正是"测试途中持续降额"的时间签名；dev0 整轮同序列基本平（只有最后一个 d16 掉到
13 978.7，是刚进入降额的开端）。dev1 整轮紧跟 dev0 整轮、起始温度已 86 °C，故落后得多。

⇒ **`logs/ze_peak_dev1.log`（整轮）的 fp64 与 int32 段偏低，不可用：**

| 测试项 | dev0 整轮 | dev1 整轮（降额） | dev1 单项 / 短跑复测 | 真值 |
|---|---:|---:|---|---:|
| fp32 `float4` | 21 871.6 | 21 871.9 ✅ | 21 872.1 / 21 871.6 | 一致 |
| fp16 `half4` | 43 381.3 | 43 386.5 ✅ | 43 384.7 / 43 391.2 | 一致 |
| fp64 `double4` | 16 074.0 | **14 278.8（−11%）** ⚠️ | 16 074.0（`-t dp_compute`） | **16 074** |
| int32 `int2` | 6 342.5 | **3 640.0（−43%）** ⚠️ | 6 178.8 / 6 175.9 | 6 176–6 428 |

漂移还表现为**随向量宽度单调恶化**（见上文机制段）—— 即该段是在降额建立过程中测完的。

**判读纪律（写进 runner 与结论文档）**：
1. 绝对值只引用**短跑**（`-a -i 3 -w 1`）或**降额前的分段**；长跑整轮值只看趋势。
2. 想跑长跑先**冷却**（本轮 dev1 整轮紧跟 dev0 整轮，起始温度已 86 °C）；或分项跑
   （`-t <test>`），项间留间隙。
3. 该现象**同时修正了功耗结论**：长时间满载确实会撞 300 W 墙并降额
   （温度冲到 101 °C、功耗 305~330 W > 300 W 上限，≈0.87×），见 §3.6 与
   `TODO/08-power-efficiency.md`。
4. 引用 `gt_act_freq_mhz` 时**必须带定性声明**：它是唯一会变的频率节点，但绝对值不可信。

> runner 已据此改造：`run_bench.py` 的 `suite_zepeak` **优先复用重跑日志**
> （`ze_peak_dev{d}_rerun.log` 优于首跑），并自动生成
> `ze_peak/repeatability.*` 与 `ze_peak/dev{d}.longrun_dvfs_drift` 两组记录。

### 3.7.4 ⭐ 裁定：FP32 向量峰值口径冲突 → **以 22.22 TFLOPS 为准**

`Conclusion/02-compute-peak/README.md` §3.2 里长期挂着 `🔴 未解决` 的冲突：
**公式 22.22 / oneDNN 22.13 / 自研探针 50.75**，后两者相差 2.28×。补做的 ze_peak
给出了**冲突双方的裁决**：

| 证据 | 数值 | 相对公式 22.2208 | 指向 |
|---|---|---|---|
| 公式 `448×16×2×1.55GHz` | 22.2208 | 100% | 22.22 |
| IGC `HardwareCaps.txt` | `EUCount=448`, `ThreadCount=3584`(=8 线程/EU) | — | 22.22（**EU 数无争议**） |
| oneDNN fp32 GEMM | 22.13 | 99.6% | 22.22 |
| **ze_peak `sp_compute`** | **21.8716** | **98.4%** | **22.22** |
| ze_peak 位宽比 `hp/sp`=1.98、`dp/sp`=0.735 | — | — | ze_peak 可信 |
| 自研探针 `alu_peak` | 50.75 | **228%** | ❌ **撤回** |
| 自研探针位宽比 `hp/sp`=1.04、`dp/sp`=1.02 | — | — | ❌ 探针**分辨不出 dtype** |
| 自研探针 `VEC=1` | 34.8 | **157%** | ❌ **连标量路径都超上限，不可能** |
| ISA：vec8 FMA 被 IGC 标量化成 1 元素 `mad (1|M0)` | 循环体 209 条互异 mad | — | 解释偏差来源 |

**结论**：自研 `alu_peak` 的绝对值是**探针侧 artefact**，必须撤回。
`results/bench_20260922-192831.*` / `bench_20260922-194139.*` 里的
`peak.fp32=50.75`、`implied_exec_width.implied_lanes_per_eu=36.5`
以及「本 ES 部件每 EU 有 37 条 FP32 lane」的推断**一律作废**。

**为什么探针会偏 2.28×**：`sycl::vec<T,8>` 的 FMA 循环被 IGC **完全标量化**
（`IGC_ShaderDumpEnable=1` 两次验证：FP32 kernel 主循环 231 条指令里有 209 条是
`mad (1|M0)` 的**标量寄存器**运算，且彼此源操作全不相同），生成的指令流与探针声明的
FLOP 模型（"每次内层迭代 32 条 `VEC=8` 宽向量 FMA"）不符。探针的算术没错，但它测的
不是它以为的东西 —— **它甚至分辨不出 fp16/fp32/fp64 的位宽比**，这是最干净的证伪依据。
完整实验见 `benchmark/02-compute-peak/sycl/exp/README.md`。

**保留价值**：探针的**相对**结论仍可用作定性参考 —— 饱和区线性度（工作量 ×2 → 耗时
×2.00）说明"测量没被编译器消循环"；占用率拐点（`global≥114688`，即 ≥16 work-item/硬件
lane）说明何时离开延迟受限区。但**任何绝对 TFLOPS 都不能引用**。

**待跟进（非本轮目标）**：探针若想给出可信绝对值，需要改成 `int` 索引参与递推、
或直接照 ze_peak 的 kernel 写法（宽向量 + 少量独立链 + 常数总 FLOP）。这属于
TODO 07（profiling）的范畴。

---

## 4. 执行步骤

### Step 1：`ze_peak` 基线（✅ 已执行，结果见 §3.7.3）
```bash
cd /root/workspace/benchmark/02-compute-peak/ze_peak_src
./build.sh run 0             # → logs/ze_peak_dev0.log，约 20 min（每个测试 ~10¹⁰ work-item）
./build.sh run 1             # → logs/ze_peak_dev1.log
# 或手工等价：
#   ./build/ze_peak -d 0 -a -i 50 -w 10
#   ./build/ze_peak -d 0 -a -i 3  -w 1     # 短跑：重复性最好，绝对值请用这个
#   ./build/ze_peak -d 0 -t dp_compute -i 10 -w 3   # 分项跑，避开降额
```
> `-a` = 跑全部 7 组；`-i 50 -w 10` = 50 次迭代 / 10 次预热；`-d N` 选设备。
> ⚠️ **整轮 ~20 min/卡，期间会触发热/功耗降额**（温度 92→101 °C、功耗 305→330 W），
> 长跑整轮的 fp64 / int32 段偏低 —— 详见 §3.7.3.1。
> 注意**标准输出在重定向时是块缓冲**：日志文件长时间为空**不等于卡死**，
> 请用 `ps`/`xpu-smi` 确认，不要误杀。

### Step 2：PyTorch GEMM sweep（推荐主力方法，最快出结果）
思路：对 M/N/K 扫点，记录 TFLOPS。
```python
import torch, itertools
from torch.utils.benchmark import Timer

dev = "xpu"
dtypes = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
sizes = [1024, 2048, 4096, 8192, 16384]

for dname, dt in dtypes.items():
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device=dev, dtype=dt)
        b = torch.randn(n, n, device=dev, dtype=dt)
        t = Timer("torch.matmul(a, b)", globals=globals()).timeit(50)
        tflops = 2 * n**3 / t / 1e12
        best = max(best, tflops)
        print(f"{dname:5s} n={n:6d}  {tflops:8.2f} TFLOPS")
    print(f"--> {dname} peak: {best:.2f} TFLOPS\n")
```
> 要点：
> - 用**大尺寸**（≥4096）才能压满；小尺寸受 launch 开销支配
> - 每次先 `torch.xpu.synchronize()`
> - 排除第一次（含 JIT/编译）的测量
> - 记录显存占用，避免 OOM

### Step 3：Triton 自定义 kernel（可选，测更真实的 ALU 上限）
```python
import triton, triton.language as tl
# 写一个纯 FMA 密集 kernel（每个线程大量 FMA 循环），
# 排除内存访问影响，逼近 ALU 理论峰值
```
用 `triton-xpu 3.8.0` 编译到 XPU 后端。

### Step 4：oneMKL / oneDNN 交叉验证
```bash
# 若构建了 benchdnn
./benchdnn --mode=p --dt=f32 --matmul --batch=1x4096x4096:1x4096x4096
```
用库的 GEMM 结果与 PyTorch 结果对照 —— 若差异大，说明 PyTorch 路径有优化空间。

### Step 5：同步采集遥测
```bash
xpu-smi dump -d 0 -m 0,1,2,8,9 -i 500 -n 100 -j > compute_peak_telemetry.json
```
用来确认测试期间 **EU Active 接近 100%**，否则说明 kernel 没压满。

---

## 4.6 逐条覆盖核验（2026-09-22 收尾）

> 数据来源：`benchmark/02-compute-peak/results/bench_20260922-194139.{json,md}`（**72 条记录**，
> `alu` 18 / `xmx` 27 / `torch` 23 / `clock` 4）+ 2026-09-22 补做的 `ze_peak`
> （见 §3.7）。详细解读见
> [`../Conclusion/02-compute-peak/README.md`](../Conclusion/02-compute-peak/README.md)。

| TODO 条目 | 状态 | 证据 / 说明 |
|---|---|---|
| Step 1 `ze_peak` 基线 | ✅ **已补做** | 2026-09-22 从 `oneapi-src/level-zero-tests/perf_tests/ze_peak` 抓取 25 个文件并构建成功，dev0/dev1 全量跑完。**关键产出：`sp_compute`(fp32 向量) = 21.87 TFLOPS = 公式值 98.4%，据此裁定 FP32 口径冲突以 22.22 为准，自研探针 50.75 撤回**（详见 §3.7；探针证伪实验见 `benchmark/02-compute-peak/sycl/exp/README.md`）。自研 SYCL 探针（`sycl/xmx_peak.cpp`）仍然保留：`ze_peak` **根本没有 XMX/DPAS 测试**，XMX 结论只能靠自研探针 |
| Step 2 PyTorch GEMM sweep | ✅ | `torch` suite 23 条：fp32 / fp16 / bf16 / int8 × 4096³、4096×4096×16384、8192³ |
| Step 3 Triton 自定义 kernel | ⬜ **未做** | 与 ⑤ 是同一缺口。「纯 ALU 上限」这个目的已由 `ze_peak`（第三方）达成，自研 SYCL ALU 探针因 §3.7 的原因**不可用于绝对峰值** |
| Step 4 oneDNN / oneMKL 交叉验证 | ⚠️ **部分** | oneDNN GEMM 交叉验证 ✅（fp32 22.13 / bf16 225.87 / int8 416.2）；**`benchdnn` 未构建**，改用手写 oneDNN C++ 调用 |
| Step 5 遥测确认压满 | ⚠️ **替代** | `xpu-smi dump` 在本驱动上**挂死**；改用 `xpu-smi stats -d 0` → 饱和 ALU 负载下 **GPU Util 100% / 1550 MHz**（短时）。⚠ 本驱动 `EU Array *` 全为 N/A，**拿不到 EU Active**，只能用 Utilization 代替。⚠ **长时满载会热/功耗降额**（温度峰值 101 °C、功耗 305~330 W > 300 W 上限、`gt_act_freq_mhz` 大幅摆动）⇒「1550 MHz」只对短时成立，详见 §3.7.3.1 与 §5 |
| §5 指标记录表 | ✅ | 见下（已按 §3.7 裁定修正） |
| §6 判读标准 | ✅ | 逐条裁定见 §6 表右列 |
| §7-3「用 `dump` 确认压满」 | ❌ **不可行** | `xpu-smi dump` 挂死。已由「占用率拐点 + 饱和区工作量 ×2 → 耗时 ×2.00」替代，可信度不低于「看 EU Active」 |
| §7-4「大尺寸双卡 OOM」 | — 不适用 | 全部单卡跑（`ZE_AFFINITY_MASK=0`） |
| （自加）占用率扫描 | ⚠️ **保留但仅作定性** | ALU 拐点 `global ≈ 114688`（16 work-item/lane）；**XMX 需 ≥8 sub-group/EU**，1 sg/EU 时只有 118 TFLOPS。拐点本身是可靠的定性结论，但绝对 TFLOPS 已作废（§3.7） |
| （自加）向量宽度扫描 | ⚠️ **保留但仅作定性** | `VEC=1/2/4/8/16 → 34.8/45.5/52.8/50.7/51.3`，**VEC=4 最佳**，之后拉平 → issue 受限。⚠ **VEC=1 就有 34.8 > 22.22 上限**，正是探针不可信的独立证据 |
| （自加）频率取证 | ⚠️ **结论已修正** | 见 §3.7.3.1：`min == max == 1550 MHz` 只是**请求值**，`gt_cur/max/min_freq_mhz` 与 xpu-smi `GPU Frequency` **全都不会反映自主降额**；唯一随负载变化的是 `/sys/class/drm/card{0,2}/gt_act_freq_mhz`（**空闲读 0**），但它**噪声极大、不收敛到 P-state**（实测 200~1400 乱跳）⇒ **只能当定性证据**。降额的**硬证据是温度（101 °C）与功耗（305~330 W）**。原「无降频也无 boost」的结论**作废** |
| （自加）XMX 功能正确性 | ✅ 超出计划 | `check.bf16/fp16/int8` 三种 DPAS 结果与参考一致（`ok=1.0`） |

> 统计：**计划内 5 个 Step：2 完全完成 / 2 替代或部分 / 1 未做**（未做项是可选步骤
> Triton kernel）；**计划外新增 4 项**。所有替代项均已如实标注原因，无静默缺口。

---

## 5. 指标记录表

> ✅ 已填写。主表来自 `benchmark/02-compute-peak/results/bench_20260922-194139.{json,md}`；
> FP32 行已按 §3.7 的第三方仲裁结果修正。

| dtype | 峰值实测 (TFLOPS) | 峰值理论 | 达成率 | 相对 FP32 加速比 |
|---|---|---|---|---|
| FP32 | **22.13**（oneDNN）/ **21.87**（第三方 `ze_peak` 向量） | 22.22（公式） | 99.6% / **98.4%** ✅ | 1.00× |
| FP64 | **17.37** @2048³（见 [`../precision-support.md`](../precision-support.md)）；**`ze_peak` 向量 16.07** | ~~11.1~~ **假设已证伪** | — | **0.78× / 0.735×**（不是 0.5×，三次独立确认） |
| FP16 | **237.57**（oneDNN/torch） / **355.0**（裸 DPAS） | 356（公式） | 66.8% / **99.8%** | 10.7× / **16.0×** |
| BF16 | **225.87**（oneDNN/torch） / **355.0**（裸 DPAS） | 356（公式） | 63.5% / **99.8%** | 10.2× / **16.0×** |
| INT8 | **416.2**（oneDNN/torch） / **710**（裸 DPAS） | 711（公式） | 58.5% / **99.8%** | 18.8× / **32.0×** |

> ⚠️ **已撤回的数字**：自研 `sycl/alu_peak.cpp` 曾给出 fp32 **50.75**、fp64 51.9、
> fp16 52.6 TFLOPS。2026-09-22 经 `ze_peak` 第三方仲裁，判定为**探针侧 artefact**，
> **这三个绝对值不得再引用**（裁定过程见 §3.7）。探针给出的**相对**结论（占用率拐点、
> VEC≥4 拉平、饱和区线性度 ×2.00）保留作定性参考。

附加记录：
- **达峰尺寸**：fp32 8192³、fp16 8192³、bf16 4096³、int8 8192³；裸 DPAS 达峰需
  占用率 **≥8 sub-group/EU**（`global ≥ 57344`）。
- **达峰功耗 / 频率**：⚠️ **本节结论已两次修正（2026-09-22）**。
  ① **功耗**：此前记录的「满载仅 171 W，远低于 300 W 上限」**不成立**。`ze_peak`
  全量运行时 `xpu-smi stats -d 0` 实测 **GPU Power 305 W**（**已超过 300 W 名义上限**），
  后续复测进一步采到 **330 W**、`GPU Util 99%`、`Temp 92 → 101 °C`。
  ② **降额**：原写的「`min == max`，无降频也无 boost」**同样作废** ——
  那是**请求值**不变，不代表实际时钟不变。长跑中 `gt_act_freq_mhz`（i915 上唯一会随负载
  变化的频率节点，空闲读 0）**几乎每次采样都不同**：1350 / 1250 / 1150 / 1000 / 950 /
  800 / 700 / 650 / 600 / 500 / 450 / 400 / 350 / 300 / 200 MHz。
  ⚠️ 该节点**绝对值噪声极大、不收敛到标准 P-state**（只有 RP0=1550 / RP1=1000 / RPn=200），
  **只能当定性证据**用，不可反算性能；硬证据是**温度（101 °C）与功耗（> 300 W）**。
  机制见 §3.7.3.1（`ze_peak` 报的是跨 50 次发射的平均吞吐；`dp`/`int` 段最晚测，
  恰好落在降额已建立之后）。
  **即：本卡在长时间满载下确实会触发热/功耗降额**；此前基于 171 W 的
  「永远不会碰上功耗墙」结论撤回。绝对值只引用短跑（`-a -i 3 -w 1`），
  长时稳态约需打 **0.87×** 折扣（详见 `08-power-efficiency.md`）。
- **`EU Active` 在本驱动上不可读**（N/A）。
- **是否带宽受限**：否。fp32 22.13 TFLOPS 恰好 = 纯算力公式值 → GEMM 是算力受限。
  与 ③ 的 900 GB/s 对照可算出算力/带宽比（见 `Conclusion/02-…` §3.5）。
- ✅ **口径冲突已解决（2026-09-22）**：公式 22.22 = oneDNN 22.13（99.6%）
  = `ze_peak` 向量 21.87（98.4%）—— **两个互相独立的第三方实现同时落在公式值上**。
  自研探针的 50.75（228%）是探针 artefact，已撤回。**FP32 向量峰值以 22.22 为准。**

---

## 6. 判读标准（含实测裁定）

| 现象 | 可能原因 | 实测裁定（2026-09-22） |
|---|---|---|
| FP32 达成率 < 70% | kernel 效率低、编译器未向量化、内存瓶颈 | ❌ **不成立**：oneDNN **99.6%**、第三方 `ze_peak` **98.4%**。自研 ALU 探针曾报 **228.4%** —— 已裁定为**探针 artefact 并撤回**（§3.7），不是 kernel 效率问题 |
| **实测 > 理论峰值**（反向异常） | 主频读数错 / EU 数错 / **FLOP 计数模型与生成代码不符** | ✅ **已定论：探针侧 artefact**。三条独立证据：① 连 `VEC=1`（标量）都报 34.8 > 22.22 上限；② 探针分辨不出位宽比（hp/sp=1.04、dp/sp=1.02，而 ze_peak 为 1.98 / 0.735）；③ ISA 显示 vec8 FMA 被 IGC 标量化。<br>**处理：撤回 `peak.fp32=50.75` 与 `implied_lanes_per_eu=36.5`；FP32 向量峰值以公式 22.22 为准** |
| FP64 不是 FP32 的 ~1/2 | 与预期架构不符，需确认（可能走 emulation 或不同路径） | ✅ **确实不是**：torch GEMM **0.78×**、`ze_peak` 向量 **0.735×**（16.07 / 21.87）。**三次独立确认**，「= FP32/2」的**原假设已被证伪**，需修正文档 |
| BF16 加速比不显著 | **XMX 未被使用**（常见坑：PyTorch 未走 XMX 路径） | ❌ **不成立**：torch 侧 **10.2×**、裸 DPAS **16.0×** → **XMX 确认已启用** |
| 大尺寸反而变慢 | 显存带宽受限或 TLB 问题 | ⚠️ **部分成立**：**16384³ 掉崖**（fp16 233.9→97.3、bf16 232.8→117.0，可复现）；原因未定位，已降优先级 |
| EU Active < 90% | kernel 未压满，该结果不能代表峰值 | ⚠️ **无法判读**：`EU Array` 指标 N/A。替代判据全部通过 —— 满载 Util 100%、「饱和区工作量 ×2 → 耗时 ×2.00」 |

> **XMX 是否真正启用** 是本测试最有价值的发现点，结论是 ✅ **已启用**（见 §4.6 的
> `check.*` 功能正确性三项 + torch/裸 DPAS 的 10~16× 倍率）。
> 本目录真正有价值的发现反而是 **XMX 的占用率拐点**（≥8 sub-group/EU）和 **覆盖率只有 58~67%**
> —— 即「硬件能做到 355 TFLOPS，但 oneDNN 路径只能拿到 226~238」。

---

## 7. 注意事项

1. **必须先 `source setvars.sh`** 并记录 `icpx --version`。
2. **频率「锁定」只发生在请求侧**：`gt_min == gt_max == 1550 MHz`，但**真实活跃频率**
   要看 `/sys/class/drm/card{0,2}/gt_act_freq_mhz`（空闲读 0，负载下才动）。
   ⚠️ 该节点**绝对值噪声极大**，不收敛到标准 P-state（RP0=1550 / RP1=1000 / RPn=200），
   长跑中几乎每次采样都不同（200~1400 之间乱跳）⇒ **只能作定性证据**：
   「它在摆 ⇒ DVFS 在动」，**不能反算性能**。真正可信的降额证据是**温度（实测峰值 101 °C）
   与功耗（305~330 W，已越过 300 W 名义上限）**。短跑重复性极好（离散度 10⁻⁵），
   **长跑绝对值不可信** —— 引用口径纪律见 §3.7.3.1。
3. ~~测试期间用 `xpu-smi dump` 确认压满，否则数据无意义。~~
   ❌ **本机不可行**：`xpu-smi dump` 在任何 metric 下都**挂死**（rc=143）。
   改用 `xpu-smi stats -d 0`（可得 Utilization / Power / Frequency，但 **`EU Array *` 为 N/A**），
   并叠加「占用率扫描 + 饱和区线性度校验」来证明压满 —— 见 §4.6。
4. 大尺寸 GEMM 会占满 48 GiB 显存，注意同时开双卡会 OOM。
5. 记录**显存带宽**作为交叉参照 —— 算力受限 vs 带宽受限的界线要靠 ③ 的结果划定。
6. **不要相信单点测量**：本目录的关键结论（拐点、覆盖率）都来自「扫点 + 饱和区 ×2 线性校验」。
   低于拐点时「工作量翻倍、耗时不变」是**延迟受限的正常表现**，不是编译器把循环消掉了 ——
   必须扫过拐点再判峰值。
