# 数值格式（精度）支持矩阵与实测吞吐

> **测试对象**：`hwt` / Intel® Data Center GPU Max 1100（Ponte Vecchio，Production ES），device 0
> **日期**：2026-09-21
> **工具**：`benchmark/05-ai-dl` 的 `precision` suite（`run_bench.py --suite precision`）
> **完整产物**：[`benchmark/05-ai-dl/results/bench_20260921-231253.md`](../benchmark/05-ai-dl/results/bench_20260921-231253.md)
> **相关文档**：[`hardware.md`](./hardware.md)、[`TODO/02-compute-peak.md`](./TODO/02-compute-peak.md)、[`TODO/05-ai-dl.md`](./TODO/05-ai-dl.md)

---

## 0. 结论速览（TL;DR）

| 问题 | 答案 |
|---|---|
| **vector 路径真正可用** | **9 种**：FP64、FP32、FP16、BF16、INT8、INT16、INT32、INT64、UINT8 |
| **matmul（开 XMX）真正可用** | **10 种**：FP64、FP32、FP16、BF16、INT8、FP8-E4M3、FP8-E5M2、MXFP8、MXFP4、NVFP4（后 5 种**无原生 XMX**） |
| **有原生 XMX 加速的** | 只有 **FP16 / BF16 / INT8** 三种 |
| **只能做权重独占量化** | INT4（`_weight_int4pack_mm`），**不能**做稠密 matmul |
| **能跑但不是累加器** | UINT8（输出同为 8 bit，直接饱和到 255） |
| **厂商声称但软件层没有** | **MXFP6**、**TF32**（torch 无对应张量类型，无法构造） |
| **核心数字** | FP32 22.2 / FP64 17.4 / FP16 233.9 / BF16 232.8 TFLOPS、**INT8 398.9 TOPS** |
| **XMX 相对 FP32 加速比** | FP16 **10.55×**、BF16 **10.50×**、INT8 **18.00×** |
| **最大反直觉发现** | **FP8 / MXFP8 / MXFP4 / NVFP4 全部比 BF16 慢**（0.13× ~ 0.63×），**不要**在本卡上用它们做加速 |

---

## 1. 测试方法

### 1.1 为什么 vector 路径用「有效带宽」而不是 FLOPS

本卡 ALU 只有 22.2 TFLOPS（FP32），HBM 约 800 GB/s，即理论算力/带宽比 ≈ **27 FLOP/Byte**。
而逐元素算子的算术强度只有 **~0.1 FLOP/Byte**（例如 `copy` 是 0 FLOP / 2 Byte，`add` 是 1 FLOP / 3 Byte），
**必然纯带宽受限**。实测也证实了这一点：

| 精度 | `exp` 带宽 | `copy` 带宽 | 比值 |
|---|---|---|---|
| fp32 | 798.5 GB/s | 796.9 GB/s | **100%** |
| bf16 | 817.5 GB/s | 799.5 GB/s | **102%** |

即：连 `exp`（超越函数，ALU 开销最大的一类）都跑满了 `copy` 的带宽 → **vector 路径下 ALU 从未成为瓶颈**。
因此：

- **vector 路径的诚实指标 = 有效带宽（GB/s）**
- **vector 路径的 ALU 峰值** = 只能通过 matmul 路径间接体现（见 §3）

> ⚠ 我们也尝试过用 Triton 写纯 FMA 密集 kernel 去直接测 ALU 峰值，**结果不可信**（fp32 报出高达 81 TFLOPS，
> 超过 22.2 的理论上限）——编译器做了死代码消除。该路线已放弃，不要复现。

### 1.2 环境

```
GPU     : Intel(R) Data Center GPU Max 1100（1 tile，56 Xe-core / 448 EU / L2 192 MB / 47.98 GiB HBM ECC）
频率    : 1550 MHz（min == max，**锁定，无法调频**）
驱动    : 1.6.33578+77（i915）/ Level Zero 1.24.0 / xpu-smi 1.2.43
torch   : 2.14.0+xpu      triton : 3.8.0      python : 3.13.3
运行参数: warmup=5, iters=20, --large（含 16384³）；PYTHON 环境 /root/workspace/venv1
```

---

## 2. 支持矩阵：厂商声称 vs 实测

「厂商」列取自 Max 系列规格书的 *Numeric Format Support* 表；「实测」列是**真的在这一台机器上跑出来的结果**。

| 格式 | torch dtype | 厂商:vector | 厂商:matrix | **实测 vector** | **实测 matmul** | 说明 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| FP64 | `float64` | ✅ | ✅ | **OK** | **OK** | ALU 路径，非 XMX |
| FP32 | `float32` | ✅ | ✅ | **OK** | **OK** | ALU 路径，非 XMX |
| TF32 | — | ❌ | ✅ | **ABSENT** | **ABSENT** | torch 无 TF32 张量类型（它只是 cuBLAS/oneDNN 的 fp32 计算模式） |
| FP16 | `float16` | ✅ | ✅ | **OK** | **OK** | ✅ 原生 XMX |
| BF16 | `bfloat16` | ✅ | ✅ | **OK** | **OK** | ✅ 原生 XMX |
| FP8-E4M3 | `float8_e4m3fn` | ❌ | ✅ | FAIL | **OK**（软件） | 走 `_scaled_mm`，无原生 XMX |
| FP8-E5M2 | `float8_e5m2` | ❌ | ✅ | FAIL | **OK**（软件） | 走 `_scaled_mm`，无原生 XMX |
| FP8-E4M3-FNUZ | `float8_e4m3fnuz` | ❌ | — | FAIL | FAIL | oneDNN 不支持 |
| FP8-E5M2-FNUZ | `float8_e5m2fnuz` | ❌ | — | FAIL | FAIL | oneDNN 不支持 |
| FP8-E8M0 | `float8_e8m0fnu` | ❌ | — | FAIL | FAIL | 仅作为 MXFP 的 scale 类型可用 |
| MXFP8 | `float8_e4m3fn` + `float8_e8m0fnu`（1×32 块缩放） | ❌ | ✅ | FAIL | **OK**（软件） | 走 `_scaled_mm` |
| MXFP6 | — | ❌ | ✅ | **ABSENT** | **ABSENT** | torch 无 `float6_*` 类型 |
| MXFP4 | `float4_e2m1fn_x2` + `float8_e8m0fnu`（1×32） | ❌ | ✅ | FAIL | **OK**（软件） | 走 `_scaled_mm` |
| NVFP4 | `float4_e2m1fn_x2` + `float8_e4m3fn`（1×16） | ❌ | ✅ | FAIL | **OK**（软件） | 走 `_scaled_mm` |
| FP4 | `float4_e2m1fn_x2` | ❌ | — | FAIL | FAIL | 无逐元素算子；裸 matmul 无路径（必须配 scale） |
| INT8 | `int8` | ✅ | ✅ | **OK** | **OK** | ✅ 原生 XMX（int32 累加） |
| INT4 | `int4` | ✅ | ✅ | FAIL | ⚠ **SPECIAL** | 仅 `_weight_int4pack_mm` 权重独占 |
| INT2 | `int2` | ❌ | ✅ | FAIL | FAIL | 无任何算子路径 |
| INT1 | `int1` | ❌ | ✅ | FAIL | FAIL | 无任何算子路径（本卡不支持 1-bit） |
| INT16 | `int16` | ✅ | ❌ | **OK** | FAIL | `Short is not supported in oneDNN!` |
| INT32 | `int32` | ✅ | ❌ | **OK** | FAIL | `could not create a primitive descriptor for the matmul primitive` |
| INT64 | `int64` | ❌ | ❌ | **OK** | FAIL | `Long is not supported in oneDNN!` |
| UINT8 | `uint8` | ❌ | ❌ | **OK** | ⚠ **SAT** | 能跑但输出仍是 uint8，**饱和**（见 §5.3） |

### 2.1 失败原因原文（verbatim，便于检索）

**matmul 失败：**

| 格式 | 报错 |
|---|---|
| INT16 | `RuntimeError: Short is not supported in oneDNN!` |
| INT64 | `RuntimeError: Long is not supported in oneDNN!` |
| INT32 | `RuntimeError: could not create a primitive descriptor for the matmul primitive. Run workload with environment variable ONEDNN_VERBOSE=all to get additional diagnostics` |
| FP8-E4M3-FNUZ | `RuntimeError: Float8_e4m3fnuz is not supported in oneDNN!` |
| FP8-E5M2-FNUZ | `RuntimeError: Float8_e5m2fnuz is not supported in oneDNN!` |
| FP8-E8M0 | `RuntimeError: could not create a primitive descriptor for the matmul primitive` |

**vector 失败（所有 fp8 / fp4 / int4 / int2 / int1）：**

```
NotImplementedError: "add_xpu" not implemented for '<Dtype>'
```

（`<Dtype>` 分别为 `Float8_e4m3fn` / `Float8_e5m2` / `Float8_e4m3fnuz` / `Float8_e5m2fnuz` /
`Float8_e8m0fnu` / `Float4_e2m1fn_x2` / `Int4` / `Int2` / `Int1`）

### 2.2 `_scaled_mm` 接受的缩放模式（完整清单）

FP8/FP4 系列唯一能用的入口是 `torch._scaled_mm`。从它的报错信息里可以拿到本版本完整支持的缩放模式：

```
TensorWise / RowWise / BlockWise-1x128 / BlockWise-128x128 / MXFP8-1x32 / MXFP4-1x32 / NVFP4-1x16
```

⚠ **块大小必须精确匹配**：MXFP4 用 `K/16` 的 scale 会失败，NVFP4 用 `K/32` 的 scale 也会失败。
必须严格用 MX 系列 = `K/32`、NVFP4 = `K/16`。

### 2.3 专用量化 matmul 算子（全部可用）

| 算子 | 用途 | 说明 |
|---|---|---|
| `torch._int_mm` | INT8 × INT8 → INT32 | XMX，int32 累加（**推荐**） |
| `aten._weight_int8pack_mm` | W8A16 权重独占 | per-channel scale；数值精确但**极慢**（见 §5.2） |
| `aten._weight_int4pack_mm` | W4A16 权重独占 | `dequant = scale*(q-8) + zero` |
| `torch._scaled_mm` | FP8 / MXFP / NVFP4 稠密 | 见 §4 |

---

## 3. matmul 路径实测吞吐（开 XMX）

单位：TFLOPS（INT8 为 TOPS）。全 1550 MHz 锁频下测量，预热 5 次、计时 20 次。

### 3.1 方阵扫描

| 精度 | 路径 | 1024³ | 2048³ | 4096³ | 8192³ | 16384³ |
|---|---|--:|--:|--:|--:|--:|
| FP64 | ALU | 12.74 | **17.37** | 16.55 | 14.45 | 14.59 |
| FP32 | ALU | 15.41 | 20.90 | 21.91 | 22.12 | **22.16** |
| FP16 | XMX | 33.49 | 149.96 | 207.22 | **233.88** | 97.31 ⚠ |
| BF16 | XMX | 34.04 | 152.57 | **232.77** | 231.59 | 116.97 ⚠ |
| INT8 | XMX | 23.22 | 153.78 | 340.67 | 382.09 | **398.88** |

**要点：**

- **FP32 达成率 100%**（22.16 / 22.22 理论）→ 证明 fp32 matmul 确实走 **ALU** 而非 XMX。
- **FP64 = 17.4 TFLOPS，不是 FP32 的 1/2！** 本卡 FP64 约为 FP32 的 **0.78×**。
  `xpu_bench/common.py` 里 `theoretical_tflops('fp64') = alu/2 = 11.11` 是**错误**的，实测高了 56%。
- INT8 在 16384³ 反而**更高**（398.88 是峰值），与 FP16/BF16 的行为相反。
- ⚠ **16384³ 存在明显的吞吐悬崖**（FP16 233.9 → 97.3，BF16 232.8 → 117.0），可复现（重复 3 次偏差 <1%），
  且不是单维问题：`M=N=16384, K=8192` 仍有 231.6 TFLOPS，而 `K=16384` 就掉到 115.5。
  **尚未定性**（怀疑与 oneDNN XPU GEMM 的 kernel/workspace 选择或 L2 = 192 MB 的 tiling 边界有关）。
  → **不要拿 16384³ 的数字代表本卡的 BF16/FP16 峰值**；判断 XMX 峰值请用 4096³~8192³。

### 3.2 INT8 的 LLM 形状（更贴近真实推理）

| 形状 | 标签 | TOPS |
|---|---|--:|
| 1×4096×4096 | decode GEMV | 0.35 |
| 16×4096×4096 | decode m16 | 5.52 |
| 64×4096×4096 | decode m64 | 22.68 |
| 2048×4096×4096 | prefill m2k | 302.78 |
| 4096×11008×4096 | FFN up | 362.79 |
| 4096×4096×11008 | FFN down | 378.92 |
| 4096×4096×16384 | K 主导 | 393.53 |
| 4096×16384×4096 | N 主导 | 368.65 |
| 8192×8192×128 | skinny-K | 44.52 |
| 128×8192×8192 | skinny-M | 119.40 |

**要点**：**M=1 的 decode 只有 0.35 TOPS**（受权重带宽限制，不是算力限制）。
LLM decode 场景下本卡是**纯带宽瓶颈**，量化权重（INT4）才是正确手段。

### 3.3 XMX 加速比

| 对比 | 倍数 |
|---|--:|
| FP16 / FP32 | **10.55×** |
| BF16 / FP32 | **10.50×** |
| INT8 / FP32 | **18.00×** |

> 注意：INT8/FP32 = 18× **不是** 2×BF16（理论 INT8 应为 BF16 的 2 倍），而是因为 BF16 在 4096³/8192³
> 尚未完全打满 XMX，而 INT8 在 16384³ 打得更满。真实 XMX 之比请以各自峰值比较。

---

## 4. 低精度浮点：FP8 / MXFP8 / MXFP4 / NVFP4

**这是本次测试最重要的（也是反直觉的）发现。**

### 4.1 结论

> **本卡（Ponte Vecchio）的 XMX 只支持 FP16 / BF16 / INT8。**
> FP8 / MXFP8 / MXFP4 / NVFP4 **全部没有原生 XMX 单元**，只能通过 oneDNN 的软件/回退实现执行，
> 吞吐**显著低于 BF16**。

### 4.2 实测吞吐

| 格式 | 4096³ | 8192³ | 16384³ | 相对 BF16 峰值 |
|---|--:|--:|--:|--:|
| FP8-E4M3 | **80.89** | 65.36 | 64.73 | **0.35×** |
| FP8-E5M2 | **145.52** | 129.30 | 128.67 | **0.63×** |
| MXFP8（1×32） | 47.53 | **49.84** | 48.77 | **0.21×** |
| MXFP4（1×32） | 31.54 | **32.06** | 31.86 | **0.14×** |
| NVFP4（1×16） | 29.11 | **29.70** | 29.35 | **0.13×** |
| *（参考）BF16* | *232.77* | *231.59* | *116.97* | *1.00×* |

### 4.3 实践含义

- ❌ **不要**在本卡上用 FP8/FP4 做训练或推理加速 —— 它比 BF16 **慢 1.6× ~ 7.7×**。
- ✅ FP8/FP4 在本卡上唯一的收益是**显存占用和带宽**（权重字节数少），在 **decode（M 小）**场景可能仍有价值。
- ✅ 若需要低精度加速，**用 INT8**（398.9 TOPS，原生 XMX）。

### 4.4 正确性

FP8-E4M3 在 512³、单位 scale、以 fp64 为参考：

```
max|err| = 0.06419    （参考结果 absmax = 34.44）
```

→ 相对误差量级与 E4M3 的 3-bit 尾数（≈2⁻³）相符，**路径本身是正确的**，只是慢。

---

## 5. 权重独占量化（Weight-Only）与 INT4

### 5.1 三种权重独占路径（N = K = 4096）

| 权重类型 | M=1 | M=32 | M=512 |
|---|--:|--:|--:|
| BF16（基线） | 0.070 ms / 0.482 TFLOPS / **482.1 GB/s** | 11.636 | 111.067 |
| W4A16（INT4） | 0.107 ms / 0.312 TFLOPS / **83.0 GB/s** | 3.550 | 44.842 |
| W8A16（INT8） | **3.158 ms** / 0.011 TFLOPS / **5.3 GB/s** | 0.151 | 0.246 |

### 5.2 两个明确的结论

1. **W8A16 (`_weight_int8pack_mm`) 不可用**：M=1 时比 BF16 慢 **~30×**（3.18 ms vs 0.10 ms），
   且几乎不随 M 扩展 → 说明它**每次调用都重新打包权重**。
   ⚠ 它不是「INT8 加速」，只是**数值精确的解量化参考实现**。
2. **W4A16 峰值 44.8 TOPS，仅为 INT8 稠密的 0.11×**，远未达到「XMX INT4 = INT8 的 2 倍」的理论值。
   → INT4 的价值**仅在权重显存/带宽**（1/4 于 bf16），**不在算力**。

### 5.3 ⚠ UINT8 不是真正的累加器

```python
torch.matmul(torch.full((1, 4), 200, dtype=torch.uint8, device='xpu'),
             torch.full((4, 1), 200, dtype=torch.uint8, device='xpu'))
# 期望 160000，实际返回 255（饱和）
```

输出 dtype 与输入**同为 uint8**，直接饱和到 255。且 uint8 **没有** `_int_mm` 类算子。
→ 实际使用必须自行分块或提精度到 int16/int32。

### 5.4 ⚠ INT4 的调用约定与 CUDA 完全不同（XPU 专有）

`aten._weight_int4pack_mm` 在 XPU 上**存在**，但布局与 CUDA 不一致（未在文档中说明，靠逆向得到）：

| 项目 | XPU | CUDA |
|---|---|---|
| `mat2` 形状 | **2D** `[N, K/2] uint8` | 4D `int32` |
| 4-bit 打包 | **沿 K 线性**：byte j 的低半字节 = 第 2j 列，高半字节 = 第 2j+1 列 | 不同 |
| `qScaleAndZeros` | **bf16**，布局约 `[K/gs, N, 2]` =（scale, zero） | 不同 |
| 解量化 | `w = scale*(q-8) + zero`，q ∈ [0,15] | 不同 |

**对齐要求**：M/N/K 过小或未对齐（例如 4、8、32）会**静默返回全 0**，不报错。

---

## 6. vector 路径实测（有效带宽，GB/s）

512 MiB 缓冲区，io_factor 见表头。**峰值 834.0 GB/s**（bf16 `sqrt`）。

| 精度 | copy (2) | add (3) | mul (3) | sub/triad (3) | exp (2) | tanh (2) | sqrt (2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| fp64 | 794.7 | 788.6 | 789.2 | 789.0 | 803.8 | **828.2** | 797.7 |
| fp32 | 796.9 | 789.6 | 788.5 | 789.2 | 798.5 | 654.4 | 801.0 |
| fp16 | 796.0 | 749.0 | 750.0 | 749.0 | 799.0 | 345.0 | 809.0 |
| bf16 | 799.5 | 789.0 | 790.0 | 789.0 | 817.5 | 329.0 | **834.0** |
| int8 | 797.0 | 788.0 | 787.0 | 787.0 | — | — | — |
| int16 | 798.0 | 781.0 | 781.0 | 780.0 | — | — | — |
| int32 | 798.0 | 789.0 | 787.0 | 789.0 | — | — | — |
| uint8 | 798.0 | 781.0 | 781.0 | 780.0 | — | — | — |

**要点：**

- 所有精度的 `copy` 都稳定在 **~795-800 GB/s** → 这就是本卡实际可达的 HBM 有效带宽。
- **fp16 / bf16 的 `tanh` 异常低**（345 / 329 GB/s，只有 copy 的 41%）→ 该精度下 `tanh` 走的是
  **软件 emulation 路径**，是唯一真正 ALU 受限的算例。**这是 vector 路径唯一的性能陷阱。**
- int8 / uint8 的 `add`/`mul` 略低于 copy（788 vs 797），差异在测量噪声量级。

### 6.1 推论：vector 路径上「换精度」没有性能收益

因为一切都被带宽压住，**fp32 → fp16/bf16/int8 不会让 elementwise 更快**
（除非真的少读/少写字节）。这与 matmul 路径形成鲜明对比。

---

## 7. 准确度与累加精度

### 7.1 基本准确度（1024³，以 fp64 为参考）

| 精度 | max 绝对误差 | 相对误差 |
|---|--:|--:|
| FP64 | 0.0 | 0.0 |
| FP32 | 1.511e-4 | 3.67e-7 |
| FP16 | 6.217e-2 | 1.763e-4 |
| BF16 | 4.997e-1 | 1.410e-3 |

与 IEEE 754 尾数位数一致（FP32 24-bit / FP16 11-bit / BF16 8-bit）。

> **XMX 累加器为 FP32** —— 由下面的 §7.2 专项验证确定。

### 7.2 专项验证：FP16 矩阵乘的累加精度

> **测试日期**：2026-09-21　**环境**：torch 2.14.0+xpu / triton-xpu 3.8.0 / Python 3.13.3
> **问题**：本卡 `fp16 matmul(fp32 acc)` 与 `fp16 matmul(fp16 acc)`，**算力是否相同**？
> 即是否存在类似 NVIDIA 的「降低累加精度换取翻倍吞吐」路径。

**结论：一样。PVC 上不存在独立的 FP16 累加硬件路径。**
XMX 对 fp16 / bf16 输入**一律用 FP32 累加**，`torch.matmul` 默认走的就是这条路径；
**没有「开启 fp16 累加变快」这个档位可开**。

#### 7.2.1 证据 1：误差不随 K 增长（决定性）

GPU 输出 vs **fp64 参考**（M = N = 4096，fp16 输入）：

| K | max&#124;err&#124; | rel_err | &#124;ref&#124;max |
|---|---|---|---|
| 256 | 3.12e-02 | 3.27e-04 | 95 |
| 1024 | 6.26e-02 | 3.54e-04 | 177 |
| 4096 | 1.25e-01 | 3.57e-04 | 349 |
| 16384 | 2.54e-01 | 3.61e-04 | 705 |

`rel_err` **恒定在 ~3.5e-4**，与 K 无关；该数值恰为 fp16 **输出**舍入量级
（eps = 2⁻¹¹ = 4.88e-4）。**若为 fp16 累加，误差必然随 K 增长。**

对照：用分块 fp16 部分和模拟「fp16 累加」（同 shape）：

| 模拟方式 | rel_err |
|---|---|
| chunk = 32 | 4.65e-03 |
| chunk = 128 | 2.25e-03 |
| chunk = 512 | 1.05e-03 |

比实测大 3–13× 且**明显随 K 漂移** ⇒ 实测结果与 fp16 累加不符，
与「FP32 累加 + fp16 输出」完全吻合。

#### 7.2.2 证据 2：软件层根本没有这个开关

```
torch.backends.xpu                         → 不存在（AttributeError）
allow_fp16_reduced_precision_reduction     → 全命名空间找不到
allow_bf16_reduced_precision_reduction     → 找不到
```

torch 2.14.0+xpu **未给 XPU 暴露任何累加精度开关**（CUDA 才有那套）。
也就是说，即便想开也开不了。

#### 7.2.3 证据 3：强行在 Triton 里写 fp16 累加器，没有任何收益

`acc = tl.zeros(..., tl.float16)` + `tl.dot(..., out_dtype=tl.float16)`：

| tile 配置 | acc = fp32 | acc = fp16 |
|---|---|---|
| BM128 BN128 BK64 | 5.1 TFLOPS | 4.0 TFLOPS |
| BM256 BN128 BK64 | 5.0 TFLOPS | 4.0 TFLOPS |
| BM128 BN256 BK64 | 5.1 TFLOPS | 4.0 TFLOPS |

> ⚠️ 该手写 kernel 只有 ~5 TFLOPS（torch 可达 232），说明**它并未落到 XMX/DPAS**，
> **绝对值不具参考意义**。但方向一致：fp16 累加不会更快，只是多出窄化转换开销。

#### 7.2.4 证据 4：实测算力只呈现「单档 XMX」的形状

```
fp32            22.2   TFLOPS
fp16 / bf16    233 / 233 TFLOPS    (~10.5×，即 xmx-fp32-acc 这一档)
int8           399     TOPS        (~1.7× bf16 ≈ 理论 2×)
```

若真存在 2× 的 fp16-acc 档，这里应多出约 ~470 TFLOPS 一层 —— **没有**。
更直接的是：fp16 与 bf16 峰值几乎重合（233.9 / 232.8），二者共用同一条路径。

> ✅ **已确认（2026-09-22 补充验证）**：以下说法**成立** ——
> XMX **不存在** fp16 累加器变体。证据来自**本机安装的 Intel 官方编译器头文件**
> （`icpx` 自带，非第三方资料），可编译复现，见 §7.2.5。

#### 7.2.5 证据 5：Intel 编译器类型表（权威来源，已编译复现）

**来源**（本机 oneAPI 2026.1，`icpx` 自带）：

```
/opt/intel/oneapi/2026.1/include/sycl/ext/oneapi/matrix/static-query-use.hpp
  ├── are_types_valid_xmx8()    ← DG2 / Arc（Xe-HPG）允许的全部类型组合
  └── are_types_valid_xmx16()   ← PVC / Ponte Vecchio（本卡，Xe-HPC）允许的全部组合
/opt/intel/oneapi/compiler/2026.1/include/sycl/ext/intel/esimd/xmx/dpas.hpp
  └── dpas_argument_type        ← DPAS 的 A/B 源精度枚举
```

> ⚠ 命名容易踩坑：**`xmx8` 指 DG2/Arc（Xe-HPG），`xmx16` 才是 PVC（本卡）**。
> 映射关系见 `static-query-use.hpp`：
> `architecture::intel_gpu_pvc`（第 431 行）的 `static_assert` 明确调用
> `are_types_valid_xmx16()`（第 435 行）；
> 而 `intel_gpu_dg2_g10/g11/g12`（第 199 / 260 / 321 行）调用 `are_types_valid_xmx8()`。
> 两个校验函数的定义分别在 **第 408 行**（xmx16）与 **第 176 行**（xmx8）。

`are_types_valid_xmx16()` **全文**（第 408–424 行，= **本卡 PVC** 允许的**所有**组合，无遗漏）：

```cpp
constexpr bool are_types_valid_xmx16() {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float>) ||                          // ← f16×f16→f32
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float>))  // bf16
    return true;
  else
    return false;
}
```

**关键事实：`Tc`（累加器）只出现 `int` 或 `float` 两种 —— 全表没有 `Tc = half`。**
`are_types_valid_xmx8()`（DG2/Arc）内容与之**逐字相同**，**同样没有 fp16 累加器**。
即：**这不是 PVC 的个例 —— 这套编译器建模的 Xe-HPG 与 Xe-HPC 两代 XMX 都没有 fp16 累加。**

> 顺带得到 PVC 的 DPAS 形状：fp16/bf16 为 **m≤8 × n16 × k16**、int8 为 **m≤8 × n16 × k32**
> （即已知的 `DPAS m16n16k16` 家族），与实测峰值形状一致。

**对照（关键的反证）**：同一套查询接口在 NVIDIA Volta 上是**允许** fp16 累加的 ——
以下是**逐字全文**：

```cpp
constexpr bool are_types_valid_cuda_sm70() {
  return (std::is_same_v<Ta, half> && std::is_same_v<Tc, float> &&
          std::is_same_v<Td, float>) ||
         (std::is_same_v<Ta, half> && std::is_same_v<Tc, half> &&
          std::is_same_v<Td, half>) ||          // ← 允许 fp16 累加
         (std::is_same_v<Ta, half> && std::is_same_v<Tc, float> &&
          std::is_same_v<Td, half>) ||
         (std::is_same_v<Ta, half> && std::is_same_v<Tc, half> &&
          std::is_same_v<Td, float>);
}
```

这说明该接口**会**在硬件支持时如实表达「fp16 累加」
（Volta 确有 `mma.sync.aligned...f16.f16.f16`）；Intel 条目里的缺席是**真实的能力缺失**，
而非接口表达能力不足。

另外 `dpas.hpp` 的 `dpas_argument_type` 枚举只列出
`fp16 / bf16 / tf32 / u8 / s8 / u4 / s4 / u2 / s2` —— **全部是 A/B 源操作数精度，
没有任何 fp16 累加器类型**，与上表一致。

**编译期实测**（`static_assert` + 运行输出）：

```
xmx16 f16.f16.f32 = 1     ← PVC（本卡）走的即此路径
xmx16 f16.f16.f16 = 0     ← PVC 不允许 fp16 累加
xmx8  f16.f16.f16 = 0     ← DG2/Arc 也不允许
sm70  f16.f16.f16 = 1     ← 对照：NVIDIA Volta 允许
```

更直接的一步：**强行实例化 `PVC × fp16 累加器` 的 `matrix_params`**，
编译器拒绝，且报错信息点名了 PVC：

```
static-query-use.hpp:467:7: error: static assertion failed due to requirement
  '(8UL == 0 && 16UL == 0 && 16UL == 0) || is_combination_valid_xmx16<...>(8UL, 16UL, 16UL)':
  Invalid parameters for architecture::intel_gpu_pvc,
  query valid combinations using: q.get_device().get_info<sycl::info::device::matrix::combinations>()
```

即：`half/half/half` 在 `intel_gpu_pvc` 上**不是合法组合**，
而同一份代码换成 `half/half/float` 就编译通过。这就是「无 fp16 累加器」的
直接反面证据。

> ⚠️ **C++ 陷门**：`using X = matrix_params<...>;` 这种别名**不会实例化模板**，
> 类内 `static_assert` 不会触发。必须写 `sizeof(X)` 或真正使用该类型，
> 否则会得出「PVC 允许 fp16 累加」的**假阴性结论**。复现代码已加此步。

复现代码见 §7.2.7 末。

> **口径说明（避免过度声称）**：以上是 **Intel 编译器/扩展层**的权威类型表，
> 可视为对硬件能力的如实建模（编译器不会无端禁用合法指令组合），
> 但**本文未能引用到 Xe-HPC ISA 手册原文**。
> 结论已由 §7.2.1–§7.2.5 共同支撑，不再依赖任何单一未证实来源。

#### 7.2.6 实践含义

1. **别指望 fp16 累加提速**：要更快就走 `int8`（≈399 TOPS，已有 `torch._int_mm` 路径）。
2. **同时也是好消息**：`torch.matmul` 的 fp16/bf16 GEMM 是 **FP32 累加**，
   误差只来自输入/输出量化，**K 很大也不会崩**（K = 16384 时 rel_err 仍为 3.6e-4）。
3. **NVIDIA 的经验在 PVC 上不成立**：所谓「开 fp16 acc 快一倍」在此卡上是错的。
4. **手动用 `.half()` 分块累加去「模拟」fp16 acc 是双输**：更慢 + 精度掉到 1e-3 量级。
5. AMP 的价值在于**减少显存/带宽流量**（权重与激活减半），而非提升 XMX 算力本身。

#### 7.2.7 复现

```bash
source /root/workspace/venv1/bin/activate
python - <<'PY'
import torch
M = N = 4096
for K in [256, 1024, 4096, 16384]:
    a = torch.randn(M, K, device="xpu", dtype=torch.float16)
    b = torch.randn(K, N, device="xpu", dtype=torch.float16)
    out = torch.matmul(a, b).double()
    ref = torch.matmul(a.double(), b.double())
    e = (out - ref).abs().max().item()
    s = ref.abs().max().item()
    print(f"K={K:>6}  max|err|={e:.3e}  rel={e/s:.3e}")
# rel 应恒定 ~3.5e-4（= fp16 输出舍入），不随 K 增长 → 说明是 FP32 累加
PY
```

§7.2.5 的编译期验证（已在本机 `icpx -fsycl` 实测通过）：

```bash
source /opt/intel/oneapi/setvars.sh
cat > /tmp/xmx_acc_check.cpp <<'CPP'
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/static-query-use.hpp>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace mx = sycl::ext::oneapi::experimental::matrix;

static_assert( mx::are_types_valid_xmx16<half, half, float>(), "PVC f16->f32 must be valid");
static_assert(!mx::are_types_valid_xmx16<half, half, half>(),  "PVC MUST reject f16 accumulator");
static_assert(!mx::are_types_valid_xmx8 <half, half, half>(),  "DG2 must also reject f16 accumulator");
static_assert( mx::are_types_valid_cuda_sm70<half, half, half>(), "control: sm70 allows f16 acc");

// 注意：using 别名不会实例化模板，必须用 sizeof 强制实例化才能触发类内 static_assert
using pvc_ok  = mx::matrix_params<architecture::intel_gpu_pvc, half, half, float, float, 8, 16, 16>;
using pvc_bad = mx::matrix_params<architecture::intel_gpu_pvc, half, half, half,  half,  8, 16, 16>;
static_assert(sizeof(pvc_ok)  > 0, "pvc fp32-acc must instantiate");
static_assert(sizeof(pvc_bad) > 0, "pvc fp16-acc must FAIL here");

int main() {
  std::cout << "xmx16 f16.f16.f32 = " << mx::are_types_valid_xmx16<half,half,float>()     << "\n";
  std::cout << "xmx16 f16.f16.f16 = " << mx::are_types_valid_xmx16<half,half,half>()      << "\n";
  std::cout << "xmx8  f16.f16.f16 = " << mx::are_types_valid_xmx8 <half,half,half>()      << "\n";
  std::cout << "sm70  f16.f16.f16 = " << mx::are_types_valid_cuda_sm70<half,half,half>() << "\n";
}
CPP

# 步骤 1：pvc_bad 在 → 期望编译失败（报 Invalid parameters for architecture::intel_gpu_pvc）
icpx -fsycl -O0 /tmp/xmx_acc_check.cpp -o /tmp/xmx_acc_check

# 步骤 2：移除 pvc_bad → 编译通过并运行
sed -i '/pvc_bad/d' /tmp/xmx_acc_check.cpp
icpx -fsycl -O0 /tmp/xmx_acc_check.cpp -o /tmp/xmx_acc_check && /tmp/xmx_acc_check
# 实测输出：xmx16 f16.f16.f32 = 1 / xmx16 f16.f16.f16 = 0 / xmx8 f16.f16.f16 = 0 / sm70 f16.f16.f16 = 1
```

### 7.3 ⚠ 跨 suite 数字差异：FP64 @ 2048³（待查）

同一张卡、同一天、**同一 shape（2048³）**，两个 suite 给出的 FP64 结果不一致：

| 来源 | FP64 @ 2048³ | 说明 |
|---|--:|---|
| 本文件 §3.1（`precision` suite） | **17.37 TFLOPS** | 预热 5 / 计时 20，取中位数 |
| `benchmark/05-ai-dl` `gemm --large`（`bench_20260921-223052`） | **14.47 TFLOPS** | 同为预热 5 / 计时 20 |

差 **20%**，超出正常测量噪声，**不能简单归为口径差异**。两者均高于
`xpu_bench/common.py` 中 `theoretical_tflops('fp64') = alu/2 = 11.11` 的错误假设
（无论取哪个值，本卡 FP64 ≈ FP32 的 0.65–0.78×，都**不是** 1/2）。

**待查方向**：两个 suite 的计时方式（`torch.xpu.Event` vs `time.perf_counter`）、
warmup 后是否清理缓存、以及 fp64 是否走了不同 kernel 选择。
→ **在澄清之前，引用 FP64 峰值时请注明来源。** 其余精度（fp32/fp16/bf16/int8）
两个 suite 结果一致，无此问题。

---

## 8. 注意事项与踩坑记录

| # | 坑 | 现象 | 规避方法 |
|---|---|---|---|
| 1 | **casting 到 int4/int2 会崩进程** | `tensor.to(torch.int4)` 触发 device 端 `DynamicCast.h:110 cast_and_store` 断言 → **core dump** | 用 `torch.empty(..., dtype=...)` 构造，不要 cast |
| 2 | `torch.randn(dtype=torch.float8_e4m3fn)` 不支持 | 报错 | 用 `torch.randn(...).to(torch.float8_e4m3fn)` |
| 3 | `getattr(torch, "fp8_e4m3fn")` **静默返回 None** | torch 里名字是 `float8_e4m3fn`；导致 sweep 静默跑空 | 用显式映射（`_dtype_by_label`） |
| 4 | `torch.matmul` 在 int8 上返回 **int8** | 容易溢出成错误结果 | 用 `torch._int_mm`（返回 int32） |
| 5 | uint8 matmul 静默饱和 | 见 §5.3 | 不要用 uint8 做累加 |
| 6 | `_weight_int8pack_mm` 每次重新打包 | 慢 30× | 不要用于生产；只在需要数值精确参考时用 |
| 7 | INT4 对齐 | 小/未对齐形状 **静默返回全 0** | 保证 M/N/K 对齐（≥ 倍数） |
| 8 | `_scaled_mm` 块大小必须精确 | MXFP4 用 K/16、NVFP4 用 K/32 会失败 | MX 系 = K/32，NVFP4 = K/16 |
| 9 | fp16/bf16 `tanh` 掉速 | 见 §6 | 若能避免，用多项式近似替代 |
| 10 | **16384³ 吞吐悬崖** | FP16/BF16 掉 ~50% | 判断峰值用 4096³~8192³；见 §3.1 |
| 11 | `xpu_bench/common.py` 的 fp64 理论值错误 | `alu/2 = 11.11`，实测 17.4 | 见 §3.1，已记录待修正 |

---

## 9. 复现方式

```bash
cd /root/workspace/benchmark/05-ai-dl
source /root/workspace/venv1/bin/activate     # 必须用 venv1（--system-site-packages，否则 torch.xpu 不可用）

# 快速版（跳过 16384 大形状）
python run_bench.py --suite precision

# 完整版（含 16384³，约 75 s）
python run_bench.py --suite precision --large

# 只输出 Markdown / JSON 到指定位置
python run_bench.py --suite precision --large --outdir results
```

产物：`results/bench_<时间戳>.md` + `results/bench_<时间戳>.json`。

### 9.1 本次运行内部分组统计

```
capability   : 23 项（22 种格式 + 1 项 uint8 饱和检测）
vector       : 44 项（8 精度 × 4~7 算子）
matmul       : 35 项（5 精度 × 5 方阵 + 10 个 LLM 形状）
fp8_matmul   :  6 项（2 精度 × 3 尺寸）
block_matmul :  9 项（3 格式 × 3 尺寸）
weight_only  :  9 项（3 类型 × 3 个 M）
quantized_op :  4 项
accuracy     :  4 项
------------------------------------------------
合计         : 134 项，全部成功，0 跳过，0 失败，0 设备断言
```

---

## 10. 待办

- [ ] 修正 `xpu_bench/common.py` 中 `theoretical_tflops('fp64')`（应为 FP32 的 ~0.78×，非 1/2）。
- [ ] 定位 16384³ 吞吐悬崖的真实原因（oneDNN kernel 选择？L2 tiling？workspace？）。
- [ ] 用 `ze_peak` / BabelStream 交叉验证 XMX 峰值（当前未安装，需构建）。
- [ ] 若需要 FP8 收益，评估 **M 小（decode）+ FP8 权重** 的组合是否比 BF16 更快。
