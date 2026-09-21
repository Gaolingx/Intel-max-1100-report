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

## 5. 指标记录表

| dtype | 峰值实测 (TFLOPS) | 峰值理论 | 达成率 | 相对 FP32 加速比 |
|---|---|---|---|---|
| FP32 | | ~22.2 | | 1.00× |
| FP64 | | ~11.1 | | |
| FP16 | | | | |
| BF16 | | | | |
| INT8 | | | | |

附加记录：
- 达峰对应的 **M=N=K 尺寸**
- 达峰时的 **功耗 / 频率 / EU Active**
- 显存带宽是否为瓶颈（对照 ③ 的带宽结果）

---

## 6. 判读标准

| 现象 | 可能原因 |
|---|---|
| FP32 达成率 < 70% | kernel 效率低、编译器未向量化、内存瓶颈 |
| FP64 不是 FP32 的 ~1/2 | 与预期架构不符，需确认（可能走 emulation 或不同路径） |
| BF16 加速比不显著 | **XMX 未被使用**（常见坑：PyTorch 未走 XMX 路径） |
| 大尺寸反而变慢 | 显存带宽受限或 TLB 问题 |
| EU Active < 90% | kernel 未压满，该结果不能代表峰值 |

> **XMX 是否真正启用** 是本测试最有价值的发现点。若 BF16 相比 FP32 没有数倍提升，说明测试路径没用到 XMX，需要通过 oneDNN/oneMKL 路径或调整 PyTorch 配置重试。

---

## 7. 注意事项

1. **必须先 `source setvars.sh`** 并记录 `icpx --version`。
2. 频率锁定在 1550 MHz → 结果重复性好，但也**不代表动态调频下的真实峰值**。
3. 测试期间用 `xpu-smi dump` 确认压满，否则数据无意义。
4. 大尺寸 GEMM 会占满 48 GiB 显存，注意同时开双卡会 OOM。
5. 记录**显存带宽**作为交叉参照 —— 算力受限 vs 带宽受限的界线要靠 ③ 的结果划定。
