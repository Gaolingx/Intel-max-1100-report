# ⑤-1 训练吞吐结论（ResNet-50 / BERT）

> 对应测试计划：[`../../TODO/05-ai-dl.md`](../../TODO/05-ai-dl.md) §3.2 / §3.3
> 测试工具：[`../../../benchmark/05-ai-dl/`](../../../benchmark/05-ai-dl/)（suite: `resnet`, `bert`）
> 测试日期：2026-09-22　　硬件：1 × Intel Data Center GPU Max 1100（Ponte Vecchio，Production ES）
> 原始数据：`results/bench_20260922-011914.json`、`bench_20260922-012320.json`、
> `bench_20260922-012429.json`（base）、`bench_20260922-014644.json`（large）

---

## 0. 一页结论

| 问题 | 答案 | 证据 |
|---|---|---|
| **XMX 是否真正启用？** | ✅ **是**。BF16/FP32 加速比 **2.03×~2.92×**（阈值 1.5×） | §2.2 |
| ResNet-50 训练峰值 | **1212 img/s / 29.75 TFLOPS**（bf16，batch=256，channels_last） | §2.1 |
| BERT-base 训练峰值 | **77.1 k tokens/s / 50.67 TFLOPS**（bf16，seq=512，batch=32） | §3 |
| BERT-large 训练峰值 | **28.5 k tokens/s / 57.27 TFLOPS**（bf16，seq=512，batch=16，335.2 M 参数） | §3.3 |
| 模型越大 XMX 利用率越高？ | ✅ 是。同形状下 large 的 TFLOPS 比 base 高 **13%**（57.27 vs 50.67） | §3.3 |
| 相对 XMX 理论峰值 | ResNet-50 **12.8%**；BERT-base **21.8%**；BERT-large **24.6%** | §4 |
| `channels_last` 是否重要？ | ✅ **非常重要**，bf16 下 **+58%~+75%** | §2.3 |
| `torch.compile` 是否有效？ | ⚠️ **只在 batch=64 有效**（+18%~+82%），batch≥128 无收益甚至倒退 | §2.4 |
| 长序列是否更划算？ | ✅ 是。BERT seq 128→512，BF16 相对 FP32 的收益从 1.92× 涨到 **3.56×** | §3.2 |

---

## 1. 测试口径

| 项目 | 说明 |
|---|---|
| 模型 | `torchvision.models.resnet50(weights=None, num_classes=1000)`；`BertForMaskedLM`（**真实预训练权重**：`bert-base-uncased` = 109.5 M，`bert-large-uncased` = 335.2 M） |
| 优化器 | `SGD(lr=0.1, momentum=0.9)` + `CrossEntropyLoss` |
| 单步内容 | `zero_grad → forward → loss → backward → optimizer.step`（完整训练步） |
| 混合精度 | `torch.autocast(device_type="xpu", dtype=torch.bfloat16)`，**不使用 GradScaler**（bf16 无需缩放） |
| FLOPs 口径 | ResNet-50：fwd = 2 × 4.09 GMAC = 8.18 GFLOP，训练按 **×3**（fwd+bwd）→ **24.54 GFLOP/img** |
| 计时 | `torch.xpu.Event` 成对计时，每点 warmup 后取多轮平均；显存取 `torch.xpu.max_memory_allocated()` |
| 权重 | ResNet-50 用随机初始化权重（吞吐与权重无关）；BERT 用真实预训练权重 |

> 说明：`tests` 全程不使用 `GradScaler`，因此 BF16/FP32 的差值**只反映 XMX 算力与访存省流**，
> 不含 loss-scaling 开销——这对结论方向（XMX 已启用）没有影响。

---

## 2. ResNet-50 训练吞吐

### 2.1 主结果（未启用 `torch.compile`）

| dtype | batch | 内存格式 | img/s | ms/step | **TFLOPS** | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| bf16 | 64 | contiguous | 709.1 | 90.3 | 17.40 | 2.93 GiB |
| bf16 | 64 | **channels_last** | **1122.5** | 57.0 | **27.55** | 2.93 GiB |
| bf16 | 128 | contiguous | 709.3 | 180.5 | 17.41 | 5.60 GiB |
| bf16 | 128 | **channels_last** | **1188.9** | 107.7 | **29.18** | 5.61 GiB |
| bf16 | 256 | contiguous | 693.0 | 369.4 | 17.01 | 10.93 GiB |
| bf16 | 256 | **channels_last** | **1212.3** | 211.2 | **29.75** | 10.94 GiB |
| fp32 | 64 | contiguous | 349.2 | 183.3 | 8.57 | 5.38 GiB |
| fp32 | 64 | channels_last | 412.5 | 155.1 | 10.12 | 5.40 GiB |
| fp32 | 128 | contiguous | 342.1 | 374.2 | 8.39 | 10.54 GiB |
| fp32 | 128 | channels_last | 414.0 | 309.2 | **10.16** | 10.56 GiB |
| fp32 | 256 | contiguous | 332.7 | 769.4 | 8.16 | 20.87 GiB |
| fp32 | 256 | channels_last | 415.0 | 616.8 | **10.18** | 20.89 GiB |

**batch size 的影响几乎为零**：bf16 channels_last 从 64→256 只涨 8%（1122.5→1212.3），
`ms/step` 严格线性放大（57.0 → 107.7 → 211.2，约 1 : 1.89 : 3.70）。
说明该工作负载在 batch=64 时就已进入**稳定的算力/访存平衡区**，不存在「小 batch 被 launch 开销拖死」
的问题——这与后面 LLM decode 的结论（§5）形成鲜明对照。

### 2.2 BF16 相对 FP32 的加速比 —— XMX 启用验证

`docs/TODO/05-ai-dl.md` §5 的判读标准：**BF16 加速比 < 1.5× 说明 XMX 未启用**。

| batch | 格式 | BF16 img/s | FP32 img/s | **加速比** |
|---:|---|---:|---:|---:|
| 64 | contiguous | 709.1 | 349.2 | **2.03×** |
| 128 | contiguous | 709.3 | 342.1 | **2.07×** |
| 256 | contiguous | 693.0 | 332.7 | **2.08×** |
| 64 | channels_last | 1122.5 | 412.5 | **2.72×** |
| 128 | channels_last | 1188.9 | 414.0 | **2.87×** |
| 256 | channels_last | 1212.3 | 415.0 | **2.92×** |

✅ **结论：XMX 已启用且工作正常。** 全部 6 个配置都在 1.5× 阈值之上 1.35~1.95 倍。
其中 channels_last 配置下的 2.7~2.9× 已经**超过**纯 XMX 算力比（232.8 / 22.16 = 10.5× 是矩阵乘法
理想比，但 ResNet-50 的瓶颈层会有访存成分）之外的额外收益来自 bf16 的**显存流量减半**
（fp32 配置显存占用正好是 bf16 的 1.91 倍，见 §2.1 的峰值列）。

### 2.3 `channels_last` 内存格式 —— 影响最大的单一开关

| dtype | batch | contiguous | channels_last | **提升** |
|---|---:|---:|---:|---:|
| bf16 | 64 | 709.1 | 1122.5 | **+58.3%** |
| bf16 | 128 | 709.3 | 1188.9 | **+67.6%** |
| bf16 | 256 | 693.0 | 1212.3 | **+74.9%** |
| fp32 | 64 | 349.2 | 412.5 | +18.1% |
| fp32 | 128 | 342.1 | 414.0 | +21.0% |
| fp32 | 256 | 332.7 | 415.0 | +24.7% |

- **BF16 下 channels_last 是必选项**（+58%~+75%）—— 换成 bf16 之后算力不再是瓶颈，
  访存模式（NCHW→NHWC）变成主导因素；contiguous 布局下 bf16 几乎拿不到任何好处
  （bf16 709.1 vs fp32 349.2 = 2.03×，与纯显存减半的预期一致）。
- fp32 下收益较小（+18%~+25%），因为 fp32 仍然算力受限。
- **峰值显存与内存格式无关**（bf16 b64 两种格式都是 2.93 GiB），所以这是一个**零成本的优化**。

> ⚠️ 实践建议：ResNet-50 训练在本卡上必须写 `model.to(memory_format=torch.channels_last)`
> **和** `x.to(memory_format=torch.channels_last)`（本次测试两者都做了）。
> 结果相当于白送 1.6~1.75 倍吞吐。

### 2.4 `torch.compile` 的实测收益

> 前置条件：需要 `ocloc` 可用（triton-xpu 编译 XPU kernel 的后端）。本次测试前已
> 安装 `intel-ocloc` 并修好 `libigdfcl.so.1` 兼容链接，详见 [`../../caveats.md`](../../caveats.md)。
> 由于 cudagraph 要求单设备可见，本节数据在 `ZE_AFFINITY_MASK=0` 下采集。

| batch | 格式 | eager img/s | compile img/s | **变化** | compile TFLOPS |
|---:|---|---:|---:|---:|---:|
| 64 | channels_last | 1122.5 | **1322.7** | **+17.8%** | 32.46 |
| 64 | contiguous | 709.1 | **1292.9** | **+82.3%** | 31.73 |
| 128 | channels_last | 1188.9 | 1173.6 | **−1.3%** | 28.80 |
| 128 | contiguous | 709.3 | **1152.6** | **+62.5%** | 28.28 |

**判读：**

1. **compile 的收益来自「补齐布局」而不是算子融合。** contiguous + compile（1292.9 / 1152.6）
   与 channels_last + eager（1122.5 / 1188.9）几乎等价 —— 说明 Inductor 只是把 conv 的
   内存格式转变重排掉了，**没有产生新的融合算子**。
2. **batch 增大后收益消失甚至倒退**（128 channels_last：1188.9 → 1173.6，−1.3%，在噪声范围内）。
   原因是 eager 在 batch≥128 时已经把 GPU 喂满（§2.1 显示 ms/step 线性放大），
   此时编译节省的那点启动开销不再可见，而静态 shape 的 `dynamic=False` 编译又失去了
   自适应 kernel 选择的余地。
3. **值得开**：如果训练程序用 batch=64（或小 batch），`torch.compile` 是明确的
   **+18%~+82%** 白送收益。**batch≥128 时不必开**。

---

## 3. BERT-base 训练吞吐

模型：`bert-base-uncased`（12 层 / 768 hidden / 109.5 M 参数，**真实预训练权重**）。
MLM 头（`BertForMaskedLM`），约 1/7 的 token 被掩码不参与 loss。

| dtype | seq_len | batch | seq/s | **tokens/s** | **TFLOPS** |
|---|---:|---:|---:|---:|---:|
| bf16 | 128 | 16 | 312.9 | 40,051 | 26.31 |
| bf16 | 128 | 32 | 442.2 | 56,602 | 37.19 |
| bf16 | 512 | 16 | 132.9 | 68,045 | 44.71 |
| bf16 | 512 | 32 | **150.6** | **77,107** | **50.67** |
| fp32 | 128 | 16 | 162.7 | 20,826 | 13.69 |
| fp32 | 128 | 32 | 183.2 | 23,450 | 15.41 |
| fp32 | 512 | 16 | 41.6 | 21,299 | 13.99 |
| fp32 | 512 | 32 | 42.3 | 21,658 | 14.22 |

### 3.1 规模效应

- batch 16→32：`tokens/s` 提升 **41%**（bf16 seq128）与 **13%**（bf16 seq512）→ 收益递减，
  说明 seq512 时 GPU 已经接近饱和。
- seq 128→512（同 batch=16，bf16）：`seq/s` 掉到 **42.5%**（312.9→132.9，符合 transformer 的
  O(L²) 注意力成本），但 `tokens/s` 涨 **1.70×**（40,051→68,045），`TFLOPS` 涨 **1.70×**
  （26.31→44.71）。**长序列在绝对算力上更划算**，因为 GEMM 的 M 维变大，
  XMX 利用率更高。

### 3.2 BF16 相对 FP32 的加速比 —— 同样验证 XMX

| seq_len | batch | BF16 TFLOPS | FP32 TFLOPS | **加速比** |
|---:|---:|---:|---:|---:|
| 128 | 16 | 26.31 | 13.69 | **1.92×** |
| 128 | 32 | 37.19 | 15.41 | **2.41×** |
| 512 | 16 | 44.71 | 13.99 | **3.19×** |
| 512 | 32 | 50.67 | 14.22 | **3.56×** |

**这是一个很有价值的规律：序列越长，BF16 的优势越大（1.92× → 3.56×）。**
原因是注意力的 $QK^\top$ 与 $PV$ 两个矩阵乘法成本随 $L^2$ 增长，而这两个 GEMM 的
K 维（= head_dim = 64）较小、M/N 维随 $L$ 增长，正好是 XMX 相对 FP32 SIMD 收益最大的形状。
FP32 侧则被钉死在 13.7~15.4 TFLOPS（= FP32 峰值 22.16 的 62%~69%），几乎不随形状变化。

### 3.3 BERT-large（335.2 M，24 层 / 1024 hidden）—— 模型越大越好喂

为验证 §3.2 的"越大越划算"规律，额外跑了一组 `bert-large-uncased`（同样真实预训练权重）：

| dtype | seq_len | batch | seq/s | **tokens/s** | **TFLOPS** | ms/step | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 128 | 8 | 79.1 | 10,121 | 20.35 | 101.2 | 6.30 GiB |
| bf16 | 128 | 16 | 127.6 | 16,332 | 32.84 | 125.4 | 6.98 GiB |
| bf16 | 512 | 8 | 42.7 | 21,872 | 43.99 | 187.3 | 9.58 GiB |
| bf16 | 512 | 16 | **55.6** | **28,475** | **57.27** | 287.7 | 14.79 GiB |
| fp32 | 128 | 8 | 48.2 | 6,167 | 12.40 | 166.1 | 6.25 GiB |
| fp32 | 128 | 16 | 57.3 | 7,339 | 14.76 | 279.1 | 7.86 GiB |
| fp32 | 512 | 8 | 13.8 | 7,057 | 14.19 | 580.4 | 14.23 GiB |
| fp32 | 512 | 16 | 14.3 | 7,333 | 14.75 | 1,117.2 | **24.70 GiB** |

BF16/FP32 加速比：**1.64× / 2.23× / 3.10× / 3.88×**（与 base 的 1.92/2.41/3.19/3.56 同趋势）。

**与 BERT-base 同条件对比（bf16, seq=512, batch=16）：**

| 模型 | 参数 | tokens/s | **TFLOPS** | 每 token FLOPs（实测反推） |
|---|---:|---:|---:|---:|
| BERT-base | 109.5 M | 77,107 | 50.67 | 0.657 GFLOP |
| BERT-large | 335.2 M | 28,475 | **57.27**（+13%） | 2.011 GFLOP（×3.06） |

- 反推的 `FLOPs/token` 与 `3 × 2 × N_params` **完全一致**（0.657 = 3×2×109.5M，2.011 = 3×2×335.2M）
  → 说明本工具的 FLOPs 口径自洽。
- **large 的绝对吞吐低，但 TFLOPS 反而更高**：`tokens/s` 降到 36.9%，而 FLOPs/token 涨了 3.06×
  （基本抵消），剩下的 +13% 来自更大的 GEMM（hidden 1024 vs 768、24 层 vs 12 层）
  → **XMX 利用率随模型规模上升**（base 21.8% → large **24.6%**）。
- 显存：fp32 seq512 batch16 达 **24.70 GiB**，是本次全部测试的最大单次分配
  （单卡上限 45.6 GiB，尚未触顶，但只剩 ~2× 余量）。

---

## 4. 与硬件理论峰值的差距

参照值（见 [`../../precision-support.md`](../../precision-support.md)、`docs/TODO/02-compute-peak.md`）：

| 精度 | 本机实测 GEMM 峰值 | 来源 |
|---|---:|---|
| FP32 | **22.16 TFLOPS** @16384³ | precision suite（= ALU 理论的 100%） |
| BF16 / FP16 | **232.8 / 233.9 TFLOPS** @4096³ / 8192³ | precision suite（XMX） |

| 工作负载 | 实测峰值 | 占对应理论峰值 |
|---|---:|---:|
| ResNet-50 bf16 训练 | 29.75 TFLOPS | **12.8%**（of 232.8） |
| ResNet-50 bf16 训练 + compile | 32.46 TFLOPS | **13.9%** |
| ResNet-50 fp32 训练 | 10.18 TFLOPS | **45.9%**（of 22.16） |
| BERT-base bf16 训练 | 50.67 TFLOPS | **21.8%** |
| BERT-large bf16 训练 | 57.27 TFLOPS | **24.6%** |
| BERT-base fp32 训练 | 15.41 TFLOPS | **69.6%** |

**解读：**

- **FP32 侧达成率很高**（46%~70%），说明框架/驱动本身没有大的浪费。
- **BF16 侧达成率只有 13%~22%**，这是 ResNet-50 / BERT-base 这个量级的模型在
  eager PyTorch 上的普遍现象，**不等于 XMX 没工作**：
  - ResNet-50 大量层的通道数很小（64/128/256，最小的 bottleneck 块只有 64 通道），
    卷积的 N 维压不满 XMX 的 16×16 tile；再加上 stride-2 下采样层、全局池化、
    以及 ~745 个算子中占比不小的逐元素/规约算子（ReLU、BatchNorm、add），
    真正跑在 XMX 上的时间占比有限；
  - §2.4 的证据（`compile` + contiguous ≈ eager + channels_last）说明**编译带来的收益主要等价于
    「布局修正」**；而训练的瓶颈不在启动开销（745 个算子平摊 283 µs/算子），
    而在**单算子效率**。
  - 作为对照，本工具的 `conv` suite 单独测 `conv2d bf16 3×3 @7×7/512ch` 可达 **167 TFLOPS**
    （= 峰值的 72%）—— 说明硬件能跑满，是**端到端模型的结构**（小通道 + 频繁 layout 转换）
    把利用率拉下来了。

**结论：要在这个量级的小模型上榨出更高训练吞吐，靠 `torch.compile` 收益有限；
应该依赖 IPEX 的融合算子 / oneDNN Graph / 更大 batch 的模型并行。**

---

## 5. 与推理侧的对照（重要）

本目录的 [`02-inference.md`](./02-inference.md) 会给出完全相反的结论：LLM decode 阶段
**受限于每 kernel 的固定启动开销（~13~15 µs）**，与 batch 无关，`torch.compile` 也救不了。

而本节 ResNet-50 / BERT 训练**完全没有这个问题**（ms/step 严格线性于 batch）。
原因很直接：

| 负载 | 单步 aten 算子数 | 单步时长 | 每算子平摊 |
|---|---:|---:|---:|
| ResNet-50 bf16 训练 b256 | **745**（实测） | 211.2 ms | **283 µs** |
| LLM decode 一步（Qwen2.5-0.5B） | **1833**（实测） | 21.6 ms | **11.8 µs** |

**判据：当「单步时长 / 算子数」掉到本机的固定开销地板（6.5~20 µs，见 [§02](./02-inference.md)）附近时，
程序进入固定开销受限区。**
训练的每个算子能平摊到 **283 µs**（= 地板值的 **14~44 倍**），所以是纯算力/访存问题；
decode 每个算子只有 **11.8 µs**，正好落在地板上，所以几乎 100% 的时间花在「把 kernel 送进去」。

---

## 6. 复现命令

```bash
cd /root/workspace/benchmark/05-ai-dl
PY=/root/workspace/venv1/bin/python

# ResNet-50（12 点：2 dtype × 3 batch × 2 内存格式）
env -u LD_LIBRARY_PATH $PY run_bench.py --suite resnet

# ResNet-50 + torch.compile（1 卡可见，否则 cudagraph 警告）
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY run_bench.py --suite resnet --model-compile

# BERT-base（8 点：2 dtype × 2 seq × 2 batch；需要联网走 hf-mirror）
env -u LD_LIBRARY_PATH HF_ENDPOINT=https://hf-mirror.com \
  $PY run_bench.py --suite bert --bert-model bert-base-uncased

# BERT-large（同上，模型换掉即可；显存需求更高）
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 HF_ENDPOINT=https://hf-mirror.com \
  $PY run_bench.py --suite bert --bert-model bert-large-uncased \
  --bert-seq-lens 128,512 --bert-batches 8,16 --outdir /tmp/bertl

# 参考：单卡 GEMM 理论峰值
env -u LD_LIBRARY_PATH $PY run_bench.py --suite gemm --large
```

> `env -u LD_LIBRARY_PATH` 是必须的：`source /opt/intel/oneapi/setvars.sh` 之后
> VTune 的私有目录会被前置到 `LD_LIBRARY_PATH`，其中的 `libigdfcl.so.1` 会破坏
> triton-xpu 的 `ocloc` 调用。详见 [`../../caveats.md`](../../caveats.md)。

---

## 7. 待补充（未完成的测试）

> ✅ **已补齐**：BERT-Large 曾列在"未做"，后已实测完成，见 **§3.3**
> （`bert-large-uncased` 335.2 M，bf16 L512 b16 = 28,475 tok/s / **57.27 TFLOPS**，
> 单卡 XMX 达成率 **24.6%**）。运行记录：`results/bench_20260922-014644.{json,md}`。

| 项 | 状态 | 原因 |
|---|---|---|
| ResNet-50 batch=512 | ⬜ 未做 | fp32 b512 需 ~42 GiB，接近 48 GiB 上限，收益已在 b256 饱和（+8%），不做 |
| AMP + GradScaler | ⬜ 未做 | bf16 无需缩放；如需 fp16 训练须补做 |
| Triton 自定义 kernel | ⬜ 未做 | 见 TODO §3.1；`attention`（SDPA）与 `conv`/`gemm` suite 已覆盖同等的 XMX 利用率问题，手写 kernel 属探索项 |
| IPEX 融合算子 / oneDNN Graph | ⬜ 未做 | 若目标是提高 BF16 达成率（ResNet-50 当前 12.8%），这是下一步最有希望的方向 |
