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
| **`ze_peak`** | ❌ 需构建 | Level Zero peak benchmark，最贴近硬件的能力探测 |
| **BabelStream** | ❌ 需构建 | 主要是带宽，但含算力项，可与带宽交叉验证 |
| **PyTorch (`torch.xpu`)** | ✅ 已装 2.14.0+xpu | 最省事，`matmul` sweep 即可得 TFLOPS |
| **triton-xpu** | ✅ 已装 3.8.0 | 手写 kernel 测有效算力（更贴近真实上限） |
| **oneMKL** | ✅ 已装 2026.1 | GEMM / 含 GPU offload |
| **oneDNN `benchdnn`** | ❌ 需构建 | GEMM / 卷积 micro-benchmark |

### 构建 `ze_peak`
```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/compute-runtime.git
find compute-runtime -iname "*peak*" -o -iname "*ze_peak*"
# 或从 level-zero 仓库的 samples 中获取
cd <ze_peak_dir>
cmake -B build -H. -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ze_peak
```
> `ze_peak` 常见输出项：FP32 / FP64 / INT8 / INT16 / INT32 的峰值 GFLOPS/GOPS，以及内存带宽。

> ⚠️ **上面这段构建说明有两处错误**（2026-09-22 联网核实后修正，见 §3.6）。
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

## 3.6 `ze_peak` 核实与「为什么本轮未构建」（2026-09-22）

本轮 **没有构建 `ze_peak`**，改用自研 SYCL 探针（`benchmark/02-compute-peak/sycl/`）。
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

**④ 已知的真实缺口（必须承认）**
`ze_peak` 唯一的、不可替代的价值是：**它是别人写的、未经我手改动的第三方实现**，
可以作为 FP32 向量峰值 **22.13 vs 50.75 口径冲突**的独立仲裁者。
本轮缺了这个第三方仲裁 —— 见 §5/§6 与 `Conclusion/02-compute-peak/README.md`。

**⑤ 现在是否可构建？—— 可以，成本很低（但当时没做）**
核实后的事实：
- 只需 **25 个文件**（`perf_tests/ze_peak/**` 22 个 + `perf_tests/common/{include,src}` 3 个）；
- **零外部依赖**：只需 `level_zero/ze_api.h`、`zer_api.h`（本机 `/usr/include/level_zero/` 已有）
  与 `-lze_loader`（本机 `libze_loader.so.1.24.0` 已有）；
  源码 `#include` 列表里**没有任何 boost**（BUILD.md 里的 boost 是给别的测试用的）；
- 5 个 kernel 是**预编译 `.spv`**，且 `ze_peak.cpp:44` 是从**路径运行时加载**，无需 OpenCL 编译器；
- 编译：`g++ -O3 -std=c++17 src/*.cpp ../../common/src/ze_app.cpp -lze_loader -lpthread`。

**当时的实际阻碍**：`intel/compute-runtime` 的 `git ls-remote` 超时（rc=143）、
`oneapi-src/level-zero-tests` 的 `git clone` 报 `GnuTLS recv error (-110)`，
而 `api.github.com` / `raw.githubusercontent.com` 正常 —— 即 git-over-https 不稳，
只能逐个文件抓取。加上上述 ②③（它本就答不了关键问题），遂决定自研。

> **结论**：未构建 `ze_peak` 在当时是**可接受**的取舍，但**不是最优** ——
> 它本可以提供第三方 FP32 仲裁。已记入 ⑤ 的「未做/待补充」清单，可随时补做（约 10 分钟）。

---

## 4. 执行步骤

### Step 1：`ze_peak` 基线
```bash
source /opt/intel/oneapi/setvars.sh
./ze_peak                    # 记录全部 dtype 峰值
./ze_peak 2>&1 | tee ze_peak_gpu0.log
# 指定设备（视 ze_peak 支持的参数而定，可能是 index 参数）
```

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
> `alu` 18 / `xmx` 27 / `torch` 23 / `clock` 4）。详细解读见
> [`../Conclusion/02-compute-peak/README.md`](../Conclusion/02-compute-peak/README.md)。

| TODO 条目 | 状态 | 证据 / 说明 |
|---|---|---|
| Step 1 `ze_peak` 基线 | ⚠️ **替代** | `ze_peak` 未构建。改用**自研 SYCL 探针**：`sycl/xmx_peak.cpp`（DPAS）、`sycl/alu_peak.cpp`（纯 FMA）。**比 `ze_peak` 多两项能力：占用率扫描（定位拐点）、向量宽度扫描**；且 `ze_peak` 本身**根本没有 XMX/DPAS 测试**。完整理由与「本可构建但未做」的说明见 **§3.6** |
| Step 2 PyTorch GEMM sweep | ✅ | `torch` suite 23 条：fp32 / fp16 / bf16 / int8 × 4096³、4096×4096×16384、8192³ |
| Step 3 Triton 自定义 kernel | ⬜ **未做** | 与 ⑤ 是同一缺口。「纯 ALU 上限」这个目的已由自研 SYCL ALU 探针达成 |
| Step 4 oneDNN / oneMKL 交叉验证 | ⚠️ **部分** | oneDNN GEMM 交叉验证 ✅（fp32 22.13 / bf16 225.87 / int8 416.2）；**`benchdnn` 未构建**，改用手写 oneDNN C++ 调用 |
| Step 5 遥测确认压满 | ⚠️ **替代** | `xpu-smi dump` 在本驱动上**挂死**；改用 `xpu-smi stats -d 0` → 饱和 ALU 负载下 **GPU Util 100% / 171 W / 1550 MHz**。⚠ 本驱动 `EU Array *` 全为 N/A，**拿不到 EU Active**，只能用 Utilization 代替 |
| §5 指标记录表 | ✅ | 见下（已填） |
| §6 判读标准 | ✅ | 逐条裁定见 §6 表右列 |
| §7-3「用 `dump` 确认压满」 | ❌ **不可行** | `xpu-smi dump` 挂死。已由「占用率拐点 + 饱和区工作量 ×2 → 耗时 ×2.00」替代，可信度不低于「看 EU Active」 |
| §7-4「大尺寸双卡 OOM」 | — 不适用 | 全部单卡跑（`ZE_AFFINITY_MASK=0`） |
| （自加）占用率扫描 | ✅ 超出计划 | ALU 拐点 `global ≈ 114688`（16 work-item/lane）；**XMX 需 ≥8 sub-group/EU**，1 sg/EU 时只有 118 TFLOPS |
| （自加）向量宽度扫描 | ✅ 超出计划 | `VEC=1/2/4/8/16 → 34.8/45.5/52.8/50.7/51.3`，**VEC=4 最佳**，之后拉平 → issue 受限 |
| （自加）频率锁定取证 | ✅ 超出计划 | `min == max == 1550 MHz`，满载 `act/cur` 不变 → 无降频、也无 boost |
| （自加）XMX 功能正确性 | ✅ 超出计划 | `check.bf16/fp16/int8` 三种 DPAS 结果与参考一致（`ok=1.0`） |

> 统计：**计划内 5 个 Step：1 完全完成 / 3 替代或部分 / 1 未做**（未做项是可选步骤
> Triton kernel）；**计划外新增 4 项**。所有替代项均已如实标注原因，无静默缺口。

---

## 5. 指标记录表

> ✅ 已填写。全部来自 `benchmark/02-compute-peak/results/bench_20260922-194139.{json,md}`。

| dtype | 峰值实测 (TFLOPS) | 峰值理论 | 达成率 | 相对 FP32 加速比 |
|---|---|---|---|---|
| FP32 | **22.13**（oneDNN/torch） / **50.75**（自研 ALU 探针）⚠ | ~22.2 | 99.6% / **228.4% ⚠** | 1.00× |
| FP64 | **17.37** @2048³（见 [`../precision-support.md`](../precision-support.md)）；自研 ALU fp64 51.9 | ~~~11.1~~ **假设已证伪** | — | **0.78×**（不是 0.5×） |
| FP16 | **237.57**（oneDNN/torch） / **355.0**（裸 DPAS） | 356（公式） | 66.8% / **99.8%** | 10.7× / **16.0×** |
| BF16 | **225.87**（oneDNN/torch） / **355.0**（裸 DPAS） | 356（公式） | 63.5% / **99.8%** | 10.2× / **16.0×** |
| INT8 | **416.2**（oneDNN/torch） / **710**（裸 DPAS） | 711（公式） | 58.5% / **99.8%** | 18.8× / **32.0×** |

附加记录：
- **达峰尺寸**：fp32 8192³、fp16 8192³、bf16 4096³、int8 8192³；裸 DPAS 达峰需
  占用率 **≥8 sub-group/EU**（`global ≥ 57344`）。
- **达峰功耗 / 频率**：`clock` suite 在饱和 ALU fp32 负载下采样 → **171 W / 1550 MHz / Util 100%**。
  **`EU Active` 在本驱动上不可读**（N/A）。
- **是否带宽受限**：否。fp32 22.13 TFLOPS 恰好 = 纯算力公式值 → GEMM 是算力受限。
  与 ③ 的 900 GB/s 对照可算出算力/带宽比（见 `Conclusion/02-…` §3.5）。
- ⚠ **口径冲突（本目录最重要的待解问题）**：公式 22.22 = oneDNN 22.13（100%）
  但自研纯 FMA 探针 50.75（228.4%）。两者不能同时为真 —— 主频已锁定、满载不降频，
  嫌疑集中在 **FLOP 计数口径** 或 **客户端混合指令** 上。
  **处理策略：22.22 继续保留并标注为「标称值（公式）」，不做重标。**

---

## 6. 判读标准（含实测裁定）

| 现象 | 可能原因 | 实测裁定（2026-09-22） |
|---|---|---|
| FP32 达成率 < 70% | kernel 效率低、编译器未向量化、内存瓶颈 | ❌ **不成立**：oneDNN 达成率 **99.6%**。⚠ 但出现**反向异常** —— 自研 ALU 探针 **228.4%** → 口径冲突，非 kernel 效率问题 |
| FP64 不是 FP32 的 ~1/2 | 与预期架构不符，需确认（可能走 emulation 或不同路径） | ✅ **确实不是**：**0.78×**（17.37 vs 22.13）。「= FP32/2」的**原假设已被证伪**，需修正文档 |
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
2. 频率锁定在 1550 MHz → 结果重复性好，但也**不代表动态调频下的真实峰值**。
3. ~~测试期间用 `xpu-smi dump` 确认压满，否则数据无意义。~~
   ❌ **本机不可行**：`xpu-smi dump` 在任何 metric 下都**挂死**（rc=143）。
   改用 `xpu-smi stats -d 0`（可得 Utilization / Power / Frequency，但 **`EU Array *` 为 N/A**），
   并叠加「占用率扫描 + 饱和区线性度校验」来证明压满 —— 见 §4.6。
4. 大尺寸 GEMM 会占满 48 GiB 显存，注意同时开双卡会 OOM。
5. 记录**显存带宽**作为交叉参照 —— 算力受限 vs 带宽受限的界线要靠 ③ 的结果划定。
6. **不要相信单点测量**：本目录的关键结论（拐点、覆盖率）都来自「扫点 + 饱和区 ×2 线性校验」。
   低于拐点时「工作量翻倍、耗时不变」是**延迟受限的正常表现**，不是编译器把循环消掉了 ——
   必须扫过拐点再判峰值。
