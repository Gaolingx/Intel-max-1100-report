# `ze_peak_src/` —— 第三方基准 `ze_peak` 的本地副本与构建说明

> **为什么这个目录存在**：`docs/TODO/02-compute-peak.md` §3.6 记录了本轮**未构建**
> `ze_peak` 的原因。事后复盘认为这是**可接受但不是最优**的取舍 —— 它本可以为
> **FP32 向量峰值口径冲突**（公式 22.22 / oneDNN 22.13 / 自研探针 50.75）
> 提供一个**第三方、与本项目无关的实现**作为参照。本目录是**补做**的结果。

---

## 1. 上游出处（⚠️ 不是 `intel/compute-runtime`）

```
仓库 : https://github.com/oneapi-src/level-zero-tests
路径 : perf_tests/ze_peak
分支 : master
```

`docs/TODO/02-compute-peak.md` §3 原先给的配方指向 `intel/compute-runtime`，
**该仓库里没有 `ze_peak`**（全仓库代码搜索为空）。正确出处如上。

抓取方式（本机 `git clone` 不通，见 §4）：
`raw.githubusercontent.com/oneapi-src/level-zero-tests/master/perf_tests/<path>`，
文件清单来自 `api.github.com/repos/oneapi-src/level-zero-tests/git/trees/master?recursive=1`。

上游 `ze_peak` 的自我描述：

> *"ze_peak is a performance benchmark suite ported from clpeak which profiles Ze
> devices to find their peak capacities for supported functionality."*

即 **clpeak 的 Level Zero 移植** —— 因此**只有向量（SIMD）测试**：

| 测试项 | 含义 |
|---|---|
| `global_bw` | 显存带宽（`float`/`float2`/`float4`/`float8`/`float16`） |
| `hp_compute` | fp16 向量算力（`half`/`half2`/`half4`/`half8`/`half16`） |
| `sp_compute` | fp32 向量算力 |
| `dp_compute` | fp64 向量算力 |
| `int_compute` | 整数向量算力（`int`/`int2`/…） |
| `transfer_bw` | 传输带宽 |
| `kernel_lat` | kernel 启动延迟 / kernel duration |

> ⚠️ **没有 XMX / DPAS / bf16 / FP8**。
> 所以它**不能**替代自研 `sycl/xmx_peak.cpp`（355 TFLOPS bf16 / 710 TOPS int8）。
> 它的价值只有一个，但很关键：**第三方数字**。

---

## 2. 本地产物

```
ze_peak_src/
├── README.md          ← 本文件
├── PATCHES.md         ← 相对上游的 2 处改动（必需的）
├── build.sh           ← 构建脚本（一条 g++）
├── shim/level_zero/   ← 补齐本机缺失的 zer_api.h（自洽 v1.15 头文件集）
├── common/            ← 上游 perf_tests/common（3 个文件）
├── ze_peak/           ← 上游 perf_tests/ze_peak（22 个文件，含 5 个预编译 .spv）
├── build/             ← 可执行文件 + 复制过来的 .spv（运行时 cwd）
└── logs/              ← 原始 stdout 日志
```

共 **25 个上游文件**（1 KB 级清单）：
`perf_tests/ze_peak/**` 22 个 + `perf_tests/common/{include/common.hpp,include/ze_app.hpp,src/ze_app.cpp}` 3 个。

---

## 3. 构建

```bash
./build.sh          # 只构建 → build/ze_peak
./build.sh run      # 构建并跑设备 0 全量
```

等价的手工命令：

```bash
g++ -O3 -std=c++17 -fcommon \
    -I shim -I ze_peak/include -I common/include \
    ze_peak/src/*.cpp common/src/ze_app.cpp \
    -o build/ze_peak -lze_loader -lpthread
```

三个必需的开关/前提：

1. **`-I shim`（放在最前）** —— 本机 `/usr/include/level_zero/` **没有 `zer_api.h`**
   （只有 `ze_api.h` v1.13.1、`zes_api.h`、`zet_api.h`），而 `common/src/ze_app.cpp:9`
   无条件 `#include <level_zero/zer_api.h>`。
   `shim/level_zero/` 放的是自洽的 **v1.15.31** 头文件集，`-I shim` 让它同时满足
   `<level_zero/ze_api.h>` 和 `zer_api.h` 内部那句相对 include `"ze_api.h"`。
   > 注：`zer_api.h` 的符号在本项目里**实际未被使用**，这句 include 是多余的；
   > 但为了不改上游源文件，选择补头文件而不是删 include。
2. **`-fcommon`** —— 上游头文件里有若干未加 `extern` 的全局定义。
3. **`.spv` 必须与可执行文件同目录** —— `*_compute.cpp` 里是
   `context.load_binary_file("ze_sp_compute.spv")` 这种**相对路径**调用，
   所以运行时 cwd 必须是 `build/`。5 个内核都是**预编译 `.spv`**（随仓库提供），
   **不需要 OpenCL 编译器**。

外部依赖：**零**（只要 `libze_loader.so.1` + `pthread`，本机都有）。
源码 `#include` 审计确认**没有 boost**（`BUILD.md` 里的 boost/zlib/libpng 是别的 perf_test 用的）。

---

## 4. 本机网络注意事项

`hwt` 上 **git over https 到 github.com 不可靠**：

```
git ls-remote https://github.com/...        → timeout, rc=143
git clone --depth 1 https://github.com/...  → fatal: GnuTLS recv error (-110):
                                              The TLS connection was non-properly terminated.
curl -I https://github.com                  → timeout
```

但 **`api.github.com` 与 `raw.githubusercontent.com` 走 curl 是通的**，
所以正确姿势是**逐文件抓取**（本次 25/25 全部成功，每个文件最多重试 12 次）。

---

## 5. 运行

```bash
cd build
./ze_peak -h                     # 选项
./ze_peak -q -d 0                # 查询 engine group
./ze_peak -d 0 -a -i 50 -w 10    # 全量，设备 0
./ze_peak -d 1 -a -i 50 -w 10    # 全量，设备 1
./ze_peak -d 0 -t kernel_lat -i 5 -w 2   # 单项
./ze_peak -d 0 -a -e             # 用 Level Zero event 计时（去掉驱动延迟）
```

已集成进主 runner：

```bash
cd /root/workspace/benchmark/02-compute-peak
python3 run_bench.py zepeak      # 只跑第三方仲裁者
python3 run_bench.py             # 全跑（含 zepeak）
```

`logs/` 里已有的日志（runner 会**优先复用**，避免重复占卡）：

| 日志 | 命令 | 用途 |
|---|---|---|
| `ze_peak_dev0.log` | `-d 0 -a -i 50 -w 10` | 设备 0 全量（约 20 min） |
| `ze_peak_dev1.log` | `-d 1 -a -i 50 -w 10` | 设备 1 全量 —— ⚠️ **撞上热/功耗降额，fp64/int 段偏低，见 §7** |
| `ze_peak_dev1_rerun.log` | `-d 1 -a -i 50 -w 10` | 设备 1 重跑（干净），runner 优先采用 |
| `ze_peak_dev1_dp_rerun.log` | `-d 1 -t dp_compute -i 10 -w 3` | fp64 单项复核 |
| `ze_peak_dev1_int_rerun.log` | `-d 1 -t int_compute -i 10 -w 3` | int 单项复核 |
| `ze_peak_short_dev{0,1}_rep{1,2}.log` | `-d N -a -i 3 -w 1` | **短跑重复性**（每轮约 60 s），跨 2 卡 × 2 次 |

---

## 6. 实测结果（2026-09-22，Intel Data Center GPU Max 1100 ×2）

设备信息：`deviceId 0bda`、`coreClockRate 1550`、`maxMemAllocSize 48 946 688 000 B`、
`isSubdevice FALSE`；UUID 分别 `…74d8-64836dad91a4`（dev0）/ `…76d6-44d7fa576257`（dev1）。

**取短跑口径（跨 2 卡 × 2 次完全一致）：**

| 测试项 | 最好内核 | 值 | 其他向量宽度 |
|---|---|---|---|
| `sp_compute` fp32 | `float4` | **21 872 GFLOPS** | f 21 843 / f2 21 821 / f8 21 760 / f16 21 533 |
| `hp_compute` fp16 | `half4` | **43 390 GFLOPS** | h 34 546 / h2 43 118 / h8 43 190 / h16 42 841 |
| `dp_compute` fp64 | `double4` | **16 074 GFLOPS** | d 16 006 / d2 15 904 / d8 15 793 / d16 13 978 |
| `int_compute` int32 | `int2` | **6 428 GOPS**(dev0) / **6 177**(dev1) | dev0 短跑 6 425–6 431，dev1 短跑 6 176–6 179（**卡间系统差 3.9%，短跑 4 次全复现 ⇒ 不是 DVFS**） |
| `global_bw` | `float` | **688 GB/s** | f2 685 / f4 660 / f8 674 / f16 678 |
| `transfer_bw` | — | Shared→Host 37–53 GB/s | 见日志（长跑值受主机状态影响） |
| `kernel_lat` | — | Kernel duration 15.7 µs | launch 5.2–5.5 µs |

**推导比（口径裁定用）：**

| 比 | ze_peak | 架构预期 | 自研探针 | 结论 |
|---|---|---|---|---|
| `sp` / 公式 22.2208 | **0.9843** | 1.0（峰值） | 2.28 ❌ 不可能 | 公式值成立 |
| `fp16`/`fp32` | **1.9835** | 2.0（Xe-HPC fp16 = 2× fp32） | 1.04 ❌ | 探针无法分辨 dtype |
| `fp64`/`fp32` | **0.7318–0.7349** | ½~¾ | 1.02 ❌ | FP64 ≠ FP32/2 第 3 次确认 |

> **★ 裁定（2026-09-22）：FP32 向量峰值 = 22.22 TFLOPS。**
> `ze_peak` 21.87（98.4%）与 oneDNN GEMM 22.13（99.6%）两个**互相独立**的实现同时落在公式值上；
> 自研 `sycl/alu_peak.cpp` 的 **50.75 TFLOPS 判定为探针侧 artefact 并撤回**。
> 证伪链见 `../sycl/exp/README.md`（VEC=1 就 34.8 TF > 硬件上限；IGC ISA 显示 vec8 FMA 被完全标量化）。

---

## 7. ⚠️ 时长口径：长跑整轮会遇到热/功耗降额

**这是本轮新发现，直接决定"该引用哪些数字"。**

`-i 50 -w 10` 的整轮约 **20 min/卡**（每个测试项要跑
`get_max_work_items() × 512~2048` 个 work-item —— fp64 用 512，fp16/fp32/int 用 2048）。期间：

```bash
# 在长跑/短跑中循环采样（card0 = dev0，card2 = dev1）
cat /sys/class/drm/card2/gt_act_freq_mhz
```

实测 `gt_act_freq_mhz` 在 **200~1400 MHz 之间乱跳**
（1350/1250/1150/1000/950/800/700/650/600/500/450/400/350/300…，**几乎每次采样都不同**），
而 `gt_cur/max/min_freq_mhz` 与 xpu-smi 的 `GPU Frequency` 始终报 **1550**（那只是**请求值**）。

> ⚠️ **`gt_act_freq_mhz` 只能当定性证据。** 它是 i915 上**唯一会随负载变化**的频率节点
> （空闲读 0），但其**绝对值噪声极大、不收敛到标准 P-state**——
> 本架构标准态只有 RP0 = 1550 / RP1 = 1000 / RPn = 200，
> 而实测采到的值几乎都不在其中。
> ⇒ **可引用的是「它在大幅摆动 ⇒ DVFS 很活跃」这个定性事实，不是它的具体数值。**
> 降额的**硬证据是温度与功耗**：fp64 单项跑 = 233–243 W / 86–89 °C；
> 整轮长跑采样到 **305 → 330 W（> 300 W 名义上限）/ 92 → 101 °C**。
> 101 °C 已逼近 PVC 结温上限。

后果 —— **dev1 的整轮日志在 fp64 与 int32 段偏低**：

| 测试项 | dev0 整轮 | dev1 整轮（降额） | dev1 短跑 / 单项 | 真值（短跑口径） |
|---|---|---|---|---|
| fp32 `float4` | 21 871.6 | 21 871.9 ✅ | 21 872.1 / 21 871.6 | 21 872（两卡一致） |
| fp16 `half4` | 43 381.3 | 43 386.5 ✅ | 43 384.7 / 43 391.2 | 43 390（两卡一致） |
| fp64 `double4` | 16 074.0 | **14 278.8（−11%）** ⚠️ | 16 074.0 / **16 071.9**（单项） | **16 074（两卡一致）** |
| int32 `int2` | 6 342.5 | **3 640.0（−43%）** ⚠️ | 6 178.8 / 6 175.9 | dev0 6 428 / dev1 6 177（卡间系统差，非降额） |

**为什么会这样（机制）**：`run_kernel()`（`ze_peak.cpp:861`）报的是
`总工作量 / 总墙钟时间`，时间由
`for (i < iters) { run_command_queue(); synchronize_command_queue(); }`
**累计 `iters`（=50）次发射**测得 ⇒ **这 50 次里任何一段变慢都会拉低平均值**，
它是「平均吞吐」而非「峰值吞吐」。而 `-a` 的分段顺序是
`sp → hp → dp → int → global_bw → transfer_bw → kernel_lat`，即 **dp/int 最晚测**，
恰好落在降额已建立之后；`sp`/`hp` 最早测，完全不受影响。

同段内的宽度序列 `d → d2 → d4 → d8 → d16` 也是**按时间先后**排列的：
dev1 整轮 dp 段 `d 14278.8 > d2 12355.7 > d4 12212.5 > d8 11092.3 > d16 9046.96`
的**单调恶化**，正是「测试过程中持续降额」的时间签名；
dev0 整轮同序列基本平（16005.9 / 15904.7 / 16074.0 / 15792.6 / **13978.7**，
只有最后一个 d16 开始掉，是刚进入降额的开端）。
dev1 整轮紧跟 dev0 整轮运行、起始温度已 86 °C，故落后得多。

**判读纪律：**

1. **绝对值引用短跑**（`-a -i 3 -w 1`）或**降额前的分段**结果；整轮长跑值仅供趋势参考。
2. `fp32` / `fp16` 段在降额发生前已测完，恰好是**跨卡跨次完全一致**的两项 —— 这也是
   FP32 裁定可信的旁证。
3. 想跑长跑就**先让卡冷却**（本轮 dev1 整轮紧跟 dev0 整轮，起始温度已 86 °C）；
   或分项跑（`-t <test>`），每项之间留间隙。
4. 引用 `gt_act_freq_mhz` 时**必须带定性声明**：它是唯一会变的频率节点，但绝对值不可信。
5. **由此产生的硬件结论**（已同步进 `docs/TODO/08-power-efficiency.md` 与
   `docs/Conclusion/02-compute-peak`）：
   **长时间满载确实会触发热/功耗降额（温度峰值 101 °C、功耗 305~330 W 越过 300 W 上限，
   ≈0.87×），之前"单卡永远不会碰上功耗墙"的说法作废。**
   短时满载只有 171 W（ALU 探针）/ 233–243 W（fp64），整轮长跑才爬到 305~330 W。
