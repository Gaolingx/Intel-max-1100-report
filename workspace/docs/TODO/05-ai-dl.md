# ⑤ AI / 深度学习性能测试

**优先级：P1**
**文档位置：** `docs/TODO/05-ai-dl.md`

---

## 1. 测试目标

1. **训练吞吐**：CNN（ResNet-50）与 Transformer（BERT）的 samples/sec 与 TFLOPS
2. **混合精度收益**：AMP + BF16 vs FP32 的加速比 → **验证 XMX 是否真正启用**
3. **多卡扩展效率**：单卡 → 双卡 DDP 的 scaling efficiency
4. **推理性能**：LLM prefill/decode 吞吐、TTFT、显存峰值
5. **算子级基准**：GEMM / Attention / LayerNorm 等的有效算力
6. 找出**主机侧瓶颈**（内存倒挂是这台机器的最大隐患）

## 2. 环境优势

机器上 **PyTorch 2.14.0+xpu 已装好并验证可用**，可立即开跑：
```
torch 2.14.0+xpu
torch.xpu.is_available() → True
torch.xpu.device_count()  → 2
triton-xpu 3.8.0
```
无需额外安装即可开展大部分测试。

> ⚠️ **内存倒挂警告**：主机仅 45 GiB DRAM，而两卡合计 96 GiB HBM。
> 数据加载、host→device 传输、CPU 预处理极易成为瓶颈。
> **所有 AI 测试必须同时记录 CPU 利用率与 host 内存带宽。**

---

## 3. 测试内容与步骤

### 3.1 算子级 micro-benchmark（先做，最快见效）

**GEMM sweep**
```python
import torch
from torch.utils.benchmark import Timer

torch.xpu.set_device(0)
for dt_name, dt in [("fp32",torch.float32), ("bf16",torch.bfloat16), ("fp16",torch.float16)]:
    for n in [512, 1024, 2048, 4096, 8192]:
        a = torch.randn(n, n, device="xpu", dtype=dt)
        b = torch.randn(n, n, device="xpu", dtype=dt)
        t = Timer("torch.matmul(a,b)", globals=globals()).timeit(50)
        print(f"{dt_name} n={n:5d} {2*n**3/t/1e12:8.2f} TFLOPS")
```
**看点**：BF16 相对 FP32 的倍数 —— 若只有 1.x 倍，说明 **XMX 未启用**。

**Attention / LayerNorm / Softmax / Activation**
针对 memory-bound 算子测量有效带宽（对照 ③ 的 HBM 带宽，算达成率）。

**Triton 自定义 kernel**（triton-xpu 3.8.0 已装）
可跑 flash-attention 风格的 kernel，测 XMX 的实际利用效率。

### 3.2 CNN 训练吞吐（ResNet-50）

关键：**用 torchvision 的官方模型**（`0.29.0+xpu` 已装）。
```python
import torch, torchvision, time
from torchvision.models import resnet50

model = resnet50().to("xpu")
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
crit = torch.nn.CrossEntropyLoss()

batch, iters = 256, 100
x = torch.randn(batch, 3, 224, 224, device="xpu")
y = torch.randint(0, 1000, (batch,), device="xpu")

# warmup
for _ in range(10):
    opt.zero_grad(); crit(model(x), y).backward(); opt.step()
torch.xpu.synchronize()

t0 = time.perf_counter()
for _ in range(iters):
    opt.zero_grad(); crit(model(x), y).backward(); opt.step()
torch.xpu.synchronize()
dt = time.perf_counter() - t0

print(f"{batch*iters/dt:8.1f} img/s  ({batch/dt:.1f} img/s per step)")
```
**变体**：
- FP32 vs AMP(`torch.autocast(device_type="xpu", dtype=torch.bfloat16)` + GradScaler)
- batch size 扫描：32 / 64 / 128 / 256 / 512（找吞吐拐点 & 显存上限）
- channels_last 内存格式对比
- `torch.compile` 开启 vs 关闭 ← 在 XPU 上可能有明显收益

### 3.3 BERT 训练吞吐
用 HuggingFace Transformers（需确认是否已装，没有则 `pip install transformers`）。
记录 **sequences/sec**，对比 FP32 vs BF16。

### 3.4 双卡 DDP 扩展效率（**核心项**）

```bash
# 2 进程，各占一卡
torchrun --nproc_per_node=2 train_ddp.py
```
记录：
| 配置 | samples/sec | 相对单卡加速 | 扩展效率 |
|---|---|---|---|
| 1 卡 | | 1.00× | 100% |
| 2 卡 DDP | | | ___% |

**扩展效率 = (2卡吞吐 / 单卡吞吐) / 2**

- 理想 > 90%；若 < 70% → 结合 ④ 的互连结果分析
- 同时记录 **通信占比**（在 profiler 中看 allreduce 耗时占比）
- 尝试 `--backend=ccl`（oneCCL）与 `--backend=xccl` 等不同后端

> **重要**：如果只是「跑通了 DDP」，不算完成测试。必须**量化通信开销占比**。

### 3.5 LLM 推理性能

用 Transformers + `device_map="xpu"` 加载模型（如 Llama-3-8B、Qwen 等）。
记录：

| 指标 | 说明 |
|---|---|
| **TTFT**（Time To First Token） | prefill 阶段延迟 |
| **Prefill 吞吐** (tokens/s) | 输入处理速度 |
| **Decode 吞吐** (tokens/s) | 生成速度 ← LLM 关键指标 |
| **显存峰值** | 模型 + KV cache |
| Batch size 影响 | 1 / 4 / 8 / 16 / 32 |
| 输入/输出长度影响 | 128/512/2048/4096 |

可选进阶：
- **量化**：INT8 / INT4（GPTQ/AWQ 的 XPU 支持需确认）
- **IPEX-LLM** / **vLLM-XPU** → 可能有显著吞吐提升
- **KV cache** 优化：paged attention 支持情况

### 3.6 数据管线瓶颈分析（针对内存倒挂）

单独测量：
- DataLoader worker 数 → 吞吐曲线
- `pin_memory=True` 的收益
- host→device 单次传输带宽与耗时
- 端到端中 **CPU 时间 vs GPU 时间**占比

若发现 **GPU 利用率 < 80%**，优先怀疑这一项。

---

## 3.7 数值格式支持矩阵（2026-09-21 实测）

> 完整版见 [`../precision-support.md`](../precision-support.md)。

### vector 路径可用（9 种）
`fp64` `fp32` `fp16` `bf16` `int8` `int16` `int32` `int64` `uint8`

### matmul 路径可用（10 种）
`fp64` `fp32` `fp16` `bf16` `int8` `fp8_e4m3fn` `fp8_e5m2` `mxfp8` `mxfp4` `nvfp4`

⚠ **有原生 XMX 的只有 FP16 / BF16 / INT8**。FP8 及以下全部走 oneDNN 软件回退：

| 格式 | 4096³ 吞吐 | 相对 BF16 |
|---|--:|--:|
| FP8-E4M3 | 80.9 TFLOPS | 0.35× |
| FP8-E5M2 | 145.5 TFLOPS | 0.63× |
| MXFP8（1×32） | 49.8 TFLOPS | 0.21× |
| MXFP4（1×32） | 32.1 TFLOPS | 0.14× |
| NVFP4（1×16） | 29.7 TFLOPS | 0.13× |

→ **不要在本卡上用 FP8/FP4 做加速**；需要低精度请用 INT8（398.9 TOPS）。

### 其他实测要点

- **vector 路径纯带宽受限**：峰值有效带宽 **834 GB/s**，`exp` 带宽 ≈ `copy` 带宽（100~102%）
  → 逐元素算子换精度**没有**收益。唯一例外：fp16/bf16 的 `tanh` 掉到 329~345 GB/s（软件 emulation）。
- **INT8 LLM 形状**：M=1 decode 仅 **0.35 TOPS**（权重带宽瓶颈）；prefill M=2048 达 302.8 TOPS。
  → LLM decode 应量化权重（W4A16 峰值 44.8 TOPS），而非指望算力。
- **uint8 matmul 不是累加器**：`200×200×4 → 255`（饱和），实际使用必须分块或提精度。
- **INT4 只能权重独占**（`_weight_int4pack_mm`），不能做稠密 matmul；XPU 的打包布局与 CUDA **不同**，
  小/未对齐形状会**静默返回全 0**。
- **MXFP6 / TF32 无法测试**：torch 里不存在对应张量类型。

---

## 4. 指标记录表

> ✅ 已全部填写。数据来源：`benchmark/05-ai-dl/results/bench_20260922-*.{json,md}`。
> 详细分析与解读见 [`../Conclusion/05-ai-dl/`](../Conclusion/05-ai-dl/)。

### 训练（单卡；双卡为 2×ResNet-50 DDP 全局吞吐）
| 模型 | dtype | 单卡吞吐 | 双卡吞吐 | 扩展效率 | 峰值 TFLOPS |
|---|---|---|---|---|---|
| ResNet-50 (b256 NHWC) | FP32 | 415.0 img/s | – | – | **10.18**（峰值 45.9%） |
| ResNet-50 (b256 NHWC) | BF16 AMP | **1212.3 img/s** | **1364.7**（b128） | **93.75%~95.88%** | **29.75**（XMX 12.8%） |
| BERT-large (L512 b16) | FP32 | 7,333 tok/s | – | – | **14.75**（峰值 66.6%） |
| BERT-large (L512 b16) | BF16 AMP | **28,475 tok/s** | – | – | **57.27**（XMX 24.6%） |

> 参考（base 模型）：BERT-base bf16 L512 b32 = **77,107 tok/s / 50.67 TFLOPS**；
> ResNet-50 b64 NHWC + `torch.compile` = **1322.7 img/s / 32.46 TFLOPS**。
> **BF16/FP32 加速比**：ResNet-50 **2.03×~(NHWC) 2.92×**；BERT-base **1.92~3.56×**；BERT-large **1.64~3.88×**。

### 推理（Qwen2.5-0.5B-Instruct bf16，权重 0.988 GB）
| 模型 | 精度 | Batch | Prefill tok/s | Decode tok/s | TTFT (ms) | 显存峰值 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 1 | 5,550（L128）/ 55,816（L2048） | 45.7 | 23.06 | 0.97 GiB |
| Qwen2.5-0.5B | bf16 | 4 | 22,287 / 75,458 | 182.2 | 22.97 | 1.08 GiB |
| Qwen2.5-0.5B | bf16 | 8 | 43,476 / **76,648** | 364.9 / 370.1 | 23.55 | 1.23 GiB |
| Qwen2.5-0.5B | bf16 | 16 | –（decode-only 扫描） | **716.8** | – | – |

### 瓶颈诊断
| 项目 | 值 |
|---|---|
| GPU 利用率（训练稳态） | **99.5%**（ResNet-50 b256 NHWC，wall 206.93 / kernel device 205.94 ms） |
| GPU 利用率（合成数据管线） | **2.83%~8.97%**（worker 1→8） |
| CPU 利用率 | 单 DataLoader worker ≈ **19%**；worker=0 时 **1751%**（主进程同步预取） |
| host→device 带宽 | pinned **27.57 GB/s** / pageable **12.03 GB/s**（batch 64 = 9.64 MB） |
| DataLoader 最优 worker 数 | **8**（9539 img/s，pin_memory=False；此时 pin=True 反而降到 7795 img/s） |
| BF16 / FP32 加速比 | ResNet-50 **2.92×** / BERT-base **3.56×** / BERT-large **3.88×** |
| 2卡 / 1卡 加速比 | **1.875×（b64）/ 1.917×（b128）** vs 无 DDP 单卡；扣掉 DDP 包装 → **1.952× / 1.953×** |

---

## 5. 判读标准

| 现象 | 可能原因 | **本次实测裁定** |
|---|---|---|
| BF16 加速比 < 1.5× | **XMX 未启用** ← 首要排查 | ❌ **不成立**：2.03~3.88×。Amdahl 模型（$f$=XMX 时间占比）**定量预测**了倍率（2.72 vs 实测 2.92；3.68 vs 实测 3.56）→ XMX 满速 |
| GPU 利用率 < 80% | **host 侧瓶颈**（内存倒挂 / DataLoader / PCIe） | ⚠️ **分场景**：真实训练 **99.5% 不触发**；合成数据管线 **2.83%~8.97% 严重触发** |
| 双卡扩展效率 < 70% | 通信瓶颈 → 回到 ④ 检查 P2P 与 Xe Link | ❌ **不成立**：**93.75%~95.88%**；6.28% 的“通信占比”中 39%~61% 是 DDP 包装自身开销 |
| 小 batch 吞吐极低 | launch 开销、kernel 未压满 | ⚠️ **仅 LLM decode 成立**：1833 算子/步 × 11.8 µs = 21.6 ms/token，离带宽 roofline **17.4×**；ResNet-50 b64 vs b256 几乎无关 → **不可外推到训练** |
| `torch.compile` 无收益甚至变慢 | 后端支持不完善 | ⚠️ **部分成立**：decode `default` 1.20× / `reduce-overhead` **0.99×**；ResNet b128 NHWC **−1.3%**；但 ResNet b64 NCHW **+82%**（≈布局修正）。根因：XPU 后端**无 CUDA-Graph 等价物** |
| 显存 OOM 早于预期 | 45.6 GiB 单次分配上限 / 碎片 | ⚠️ **两者都不是**：真凶是 **logits**（b8×L2048×V151936×2 B = **4.64 GiB**），KV cache 只有 **188 MiB** → 修法 `logits_to_keep=1` |
| FP16 与 BF16 峰值几乎重合 | **正常**：XMX 对 f16/bf16 一律 FP32 累加，无独立 fp16-acc 档（见 [`precision-support.md` §7.2](../precision-support.md)） | ✅ **正常**（实测 233.9 vs 232.8 TFLOPS） |
| 想"开 fp16 累加"换吞吐却无收益 | **正常**：该硬件路径不存在，见 [`precision-support.md` §7.2](../precision-support.md) | ✅ **正常** |

---

## 6. 注意事项

1. **必须** `torch.xpu.synchronize()` 后才能停计时，否则测的是异步 launch 时间。
2. **必须** warmup（首次 kernel 含 JIT/编译开销）。
3. 记录 batch size / 输入尺寸 / 迭代数 —— 这些对结果影响巨大。
4. `torch.xpu.get_device_properties(0)` 输出记入报告。
5. 关注 **显存碎片**：长时间循环建议中途检查 `torch.xpu.memory_allocated()`。
6. 与 ② ③ ④ 的结果**交叉引用**，避免孤立结论。
7. transformers / vLLM 若未安装需先装，注意 XPU 兼容版本。
