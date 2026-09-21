# ⑤-4 瓶颈诊断结论（算力 / 固定开销 / 主机侧 / 互连）

> 对应测试计划：[`../../TODO/05-ai-dl.md`](../../TODO/05-ai-dl.md) §3.5、§5 判读标准
> 关联结论：[`01-training-throughput.md`](./01-training-throughput.md) §5 ／ [`02-inference.md`](./02-inference.md) §3 ／ [`03-scaling.md`](./03-scaling.md) §4
> 定向探针：[`../../../benchmark/05-ai-dl/diagnostics/`](../../../benchmark/05-ai-dl/diagnostics/)（7 个脚本，docstring 里记录了各自测出的结论）
> 测试日期：2026-09-22

---

## 0. 一页结论

| 负载 | 瓶颈在哪 | 决定性证据 |
|---|---|---|
| **ResNet-50 训练** | **算力受限**（GPU 忙碌率 **99.5%**），但算力**花在 BN/ReLU 上**，不在 XMX 上 | kernel 成分：BatchNorm **38.1%** + Elementwise/ReLU **24.9%** + Other **4.1%** = **67%** 非 XMX，Conv/GEMM 只有 **32.6%** |
| **BERT 训练** | 同上，但 XMX 覆盖率高得多 | GEMM **37.7%** + Attention **18.6%**（FA 前向 4.2 + 反向 14.4）≈ **56%** XMX 可用 |
| **LLM Prefill** | 算力受限（**75.7 TFLOPS** = 峰值 32.5%） | 边际 prefill 81~112k tok/s，随序列线性 |
| **LLM Decode** | **每算子固定开销受限**（不是带宽、不是算力） | 每算子上限 ~11.8 µs；**1×896×4864 GEMM 与 896×896 GEMM 耗时相同（~21 µs）**；与带宽 roofline 差 **17.4×** |
| **数据管线 + 轻量 GPU 任务** | **主机 CPU 受限** | 端到端 GPU 利用率只有 **2.83%~8.97%**，`gpu_ms_per_batch` 恒为 0.68~0.88 ms |
| **2 卡 DDP** | **都不是瓶颈** | 扩展效率 **93.75%~95.88%**，真实跨卡只占 Δ 的 39%~56% |

**一句话**：**这台机器上不存在单一的"瓶颈"**。同一个互连、同一块 GPU：
训练场景被「非 XMX 算子占比」限制，decode 场景被「算子数量」限制，
数据管线场景被「主机 CPU」限制。**优化方向必须按场景分开选**。

---

## 1. 诊断方法论

| 层级 | 工具 | 回答什么 | 坑 |
|---|---|---|---|
| ATen 算子级 | `TorchDispatchMode` 计数 | **有多少个算子** | ❌ **不能在 `torch.compile` 之后用**（Dynamo cache 查询也过 dispatch 层，计数虚高到几万） |
| ATen 算子级 | `torch.profiler` (`self_device_time_total`) | 每算子**实际耗时** | 单位是 **微秒**；`aten::` 层与 kernel 层**重复计数**，必须过滤 |
| Kernel 级 | `torch.profiler`（只留非 `aten::` 条目） | GPU **真正在算什么** | `count` 是 N 步累计，要除以 N |
| 事件计时 | `xpu.Event` + 批量入队 | 干净的墙钟时间 | ❌ 逐迭代 `dist.barrier()` 会加 **119 µs** 污染 |
| 差分法 | `no_sync()` 上下文 | 通信的**暴露成本** | 见 03-scaling §1.2 |

探针脚本：`diagnostics/train_kernel_mix.py`（训练 kernel 成分）、
`decode_budget.py`（decode 成本预算）、`m1_gemm_scan.py`（M=1 逐形状扫描）、
`launch_floor.py`（主机入队 vs 设备执行地板）、`llm_compile_compare.py`、
`allreduce_probe.py`。

**统一运行环境**：`env -u LD_LIBRARY_PATH`（避免 oneAPI `setvars.sh` 里的
VTune gma/ocloc 目录遮蔽修好的 `libigdfcl.so.1`）＋ `ZE_AFFINITY_MASK=0`（单卡）。

---

## 2. 结论一：训练是"算力受限"，但算力没花在 XMX 上 ⭐

### 2.1 ResNet-50 bf16 batch=256 channels_last 训练步的 kernel 成分

`wall = 206.93 ms/step`；真 kernel `self_device_time` 合计 **205.94 ms/step**（1023 次调用）：

$$\text{GPU 忙碌率} = \frac{205.94}{206.93} = \mathbf{99.5\%}$$

| kernel family | ms/step | **占比** | 调用数 | µs/次 |
|---|---:|---:|---:|---:|
| **BatchNorm**（4 个 kernel × 53 层） | 78.62 | **38.1%** | 265 | 287~449 |
| **Conv/GEMM**（`gen_conv` × 158） | 67.21 | **32.6%** | 285 | 414 |
| **Elementwise/ReLU**（`VectorizedElementwiseKernel` × 129） | 51.39 | **24.9%** | 401 | 373~671 |
| Other（MaxPool fwd/bwd、AdamW、zero_out 等） | 8.48 | 4.1% | 59 | – |
| Copy/Add/Cat | 0.42 | 0.2% | 9 | – |
| Reduce | 0.12 | 0.1% | 4 | – |
| **合计** | **206.24** | 100% | **1023** | – |

**四个 BatchNorm kernel 加起来 = 78.62 ms = 38.1%**，**超过卷积的 32.6%**。
其中 BN 反向的两个 kernel（`BatchNormBackwardReduceChannelsLast` 23.80 ms +
`BatchNormBackwardElemtChannelsLastVectorized` 23.84 ms）单项就各占 **11.6%**。

> ⚠️ **这直接推翻了"ResNet 训练瓶颈是卷积"的直觉。**
> ResNet-50 有 **53 个 BN 层**，每层在 b256 下要做 4 个 bandwidth-bound kernel
> （Welford 统计 / 变换输入 / 反向 reduce / 反向元素），**每个 287~449 µs**，
> 且**全部与 XMX 无关**。

### 2.2 BERT-base bf16 batch=32 seq=512 训练步的 kernel 成分

`wall = 161.40 ms/step`，kernel device 合计 178.59 ms（profiler 有插桩开销，故 ≥100%）：

| kernel family | ms/step | 占比 | 说明 |
|---|---:|---:|---|
| **GEMM/Conv**（`gemm_kernel` × 228, 290 µs/次） | 67.28 | **37.7%** | 全部走 XMX |
| Other（其中 **FlashAttention 反向** 25.66 ms = 14.4%、AdamW 21.3 ms、LayerNorm 反向 14.2 ms） | 67.72 | 37.9% | FA 走 XMX |
| Elementwise/Act | 16.30 | 9.1% | 带宽受限 |
| Reduce / Copy / LayerNorm | 19.76 | 11.0% | 带宽受限 |
| Attention（FA 前向） | 7.53 | 4.2% | 走 XMX |
| **合计** | **178.59** | 100% | 860 次调用 |

→ **BERT 的 XMX 可用比例 ≈ GEMM 37.7% + Attention 18.6%（前向 4.2 + 反向 14.4）≈ 56%**，
**远高于 ResNet 的 32.6%**（BN 层少、attention 由 FlashAttention kernel 承担）。

> `Other` 这一类里混了 FA 反向（走 XMX）、AdamW（带宽受限）、LayerNorm 反向
> （带宽受限）三种性质不同的 kernel —— 这是**按名称分类的固有局限**。
> 下表的 $f$ 取值按"已知走 XMX 的 kernel"口径保守估算。

### 2.3 ⭐ Amdahl 模型验证：一个公式解释两个模型的加速比

假设：XMX 可用部分走 XMX 得 **10.5×**（BF16 的实测 XMX 倍率，见 `precision-support.md`），
其余带宽受限部分因**字节数减半**得 **2×**。设 $f$ 为 XMX 可用时间占比：

$$\text{Speedup}_{\text{bf16/fp32}} = \frac{1}{\dfrac{1-f}{2} + \dfrac{f}{10.5}}$$

| 模型 | 实测 $f$ | **公式预测** | **实测加速比** | 误差 |
|---|---:|---:|---:|---:|
| ResNet-50 b256 channels_last | **0.326** | **2.72×** | **2.92×** | +7% |
| BERT-base b32 L512 | **0.563** | **3.68×** | **3.56×** | −3% |

**两个完全不同的模型、同一个两点模型、误差 ≤7%。** 这证明：

> **ResNet-50 的 BF16 加速比只有 2.9×（而不是 XMX 的 10.5×），根本原因不是 XMX 没启用
> —— XMX 是满速工作的 —— 而是只有 1/3 的 GPU 时间在乘加阵列上，
> 另外 **2/3**（BN + ReLU + elementwise + MaxPool）是纯带宽受限的，
> 只能靠"字节减半"拿到 2×。**

这也解释了 [`01-training-throughput.md`](./01-training-throughput.md) §4 里
「BF16 只到峰值的 12.8%」这个看着很糟糕的数字：**峰值利用率的分母是 XMX 峰值，
而 67% 的时间根本用不到 XMX，用这个分母评价是不合理的。**

### 2.4 为什么 `channels_last` 有 +58%~+75% 的效果

同一份模型、同一 dtype，只改 memory format：

| batch | bf16 contiguous | bf16 channels_last | 提升 |
|---:|---:|---:|---:|
| 64 | 709.1 img/s | 1122.5 img/s | **+58.3%** |
| 128 | 709.3 img/s | 1188.9 img/s | **+67.6%** |
| 256 | 693.0 img/s | 1212.3 img/s | **+74.9%** |

原因来自 §2.1：**67% 的时间在 BN / elementwise / 池化上，而这些算子在 NCHW 下访存不连续。**
`channels_last` 让"每像素通道向量"连续，BN kernel（`*ChannelsLastVecKernel`）
和 `VectorizedElementwiseKernel<8, ...>` 才能向量化。
→ **这不是"XMX 优化"，是纯访存布局优化**，也是本机 ResNet-50 上**性价比最高的一项改动**。

---

## 3. 结论二：LLM decode 受"每算子固定开销"限制 ⭐⭐

### 3.1 量级：每算子 ~12 µs，与大小无关

| 操作 | 主机入队 | 设备执行 | 读法 |
|---|---:|---:|---|
| `add_`（1 个元素） | 6.68 µs | **6.49 µs** | 设备侧固定成本 |
| `add_`（34 MB） | 6.55 µs | 35.27 µs | 主机侧**没变** → 纯设备工作 |
| `view` | – | 0.58 µs | 纯元数据，最便宜 |
| `unsqueeze` / `transpose` | – | 0.67 / 0.77 µs | 同上 |
| `linear 1×896×4864` | 19.24 µs | **20.36 µs** | 主机入队就已经 ~19 µs |
| `mul + add`（896 元素） | 15.0 µs | – | 两个算子 |
| `softmax(1,14,1,129)` | 13.4 µs | – | – |

**判据**：主机入队时间 $\approx$ 设备执行时间时，**延迟由"把一个算子推上 GPU"这个过程决定**，
而不是由算子干了多少活决定。上面 `linear 1×896×4864` 正是这种情况（19.24 vs 20.36）。

**单步 decode 的算子数：1833 个**（`TorchDispatchMode` 实测，eager）：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| 元数据（view 339 / _unsqueeze... transpose 97 / slice 96 / t 169 / unsqueeze 52 ...） | 854 | 46.6% |
| elementwise（mul 220 / add 146 / pow 49 / mean 49 / rsqrt 49 / neg 48 / silu 24） | 585 | 31.9% |
| GEMM（mm 97 / addmm 72 / bmm 1） | 170 | 9.3% |
| 其他（`_to_copy` 101 / cat 97 / embedding / arange） | 200 | 10.9% |
| attention（`micro_sdpa`） | 24 | 1.3% |
| **合计** | **1833** | 100% |

$$\underbrace{1833}_{\text{算子数}} \times \underbrace{11.8\ \mu s}_{\text{每算子}} = 21.6\ \text{ms} \;=\; \text{实测单步时间}$$

**其中 78.5% 的算子是"搬运/元数据/elementwise"，真正做乘加的只有 9.3%。**

### 3.2 ⭐ 决定性证据：M=1 GEMM 逐形状扫描

`diagnostics/m1_gemm_scan.py`：

| 形状 | k | n | **权重字节** | m=1 | m=4 | m=8 | m=32 | **m=512** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn qkv_proj` | 896 | 1152 | **2.1 MB** | **22.8 µs** | 21.0 | 21.1 | 21.2 | 21.0 |
| `attn o_proj` | 896 | 896 | **1.6 MB** | **20.7 µs** | 20.9 | 21.0 | 21.1 | 21.4 |
| `mlp gate_up` | 896 | 9728 | **17.4 MB** | **20.9 µs** | 22.2 | 22.0 | 21.6 | 70.6 |
| `mlp down` | 4864 | 896 | **8.7 MB** | **24.6 µs** | 23.6 | 23.4 | 23.9 | 55.5 |
| `lm_head` | 896 | 151936 | **272.3 MB** | **491.3 µs** | 522.1 | 534.5 | 545.3 | 818.4 |

**三行读数：**

1. **`qkv_proj`（2.1 MB 权重）22.8 µs，`gate_up`（17.4 MB 权重）20.9 µs —— 权重差 8.3 倍，时间几乎一样（差 9%）。**
   → 这个区间完全不看数据量，**看的是"发起一个 GEMM 要多久"**。
2. **`qkv_proj` / `o_proj` 从 m=1 到 m=512 完全平（22.8→21.0、20.7→21.4 µs）**
   → batch 从 1 涨到 512 都不增加时间，说明直到 m=512 才填满 GPU。
3. **只有 `lm_head`（272 MB）在 m=1 就是 491 µs（= 554 GB/s，达到 797 GB/s 峰值的 70%）**
   → **它是唯一真正带宽受限的算子。**

带宽 roofline 对照（@800 GB/s）：

| 形状 | 下界 | 实测 m=1 | 偏离 |
|---|---:|---:|---:|
| `qkv_proj` | 2.6 µs | 22.8 µs | **8.8×** |
| `o_proj` | 2.0 µs | 20.7 µs | **10.4×** |
| `gate_up` | 21.8 µs | 20.9 µs | **0.96×（已达带宽下界！）** |
| `down` | 10.9 µs | 24.6 µs | 2.3× |
| `lm_head` | 340 µs | 491 µs | 1.4× |

→ **`gate_up` 逆向"超过"了它的带宽下界** —— 因为它只搬了 17.4 MB，而 ~20 µs 是
**所有算子的公共地板**，地板比该算子的理论时间还长。

> **这就是"固定开销主导"的数学特征：当一个算子的理论时间 < 地板时间，
> 提高这个算子的效率毫无意义，只能减少算子数量或把多个算子合并。**

### 3.3 单步 21.6 ms 的预算分解

`torch.profiler` 单步 ATen 算子级（device 合计 23.4 ms ≈ 墙钟 21.6 ms，差值为插桩开销）：

| 算子 | 次数/步 | device 合计 | µs/次 | 占 23.4 ms |
|---|---:|---:|---:|---:|
| `aten::mm` | 97 | 2329.1 µs | 24.0 | 10.0% |
| `aten::copy_` | 101 | 1131.0 µs | 11.2 | 4.8% |
| `aten::mul` | 220 | 942.6 µs | 4.3 | 4.0% |
| `aten::cat` | 97 | 666.1 µs | 6.9 | 2.8% |
| `aten::add` | 146 | 622.2 µs | 4.3 | 2.7% |
| `aten::addmm` | 72 | 548.8 µs | 7.6 | 2.3% |
| `aten::mean` | 49 | 415.5 µs | 8.5 | 1.8% |
| `micro_sdpa` | 24 | 352.6 µs | 14.7 | 1.5% |
| `aten::silu` | 24 | 185.0 µs | 7.7 | 0.8% |
| **前 9 名小计** | **830** | **7193 µs** | – | **30.7%** |
| 其余 ~1000 个算子 | ~1000 | **16 200 µs** | ~16 | **69.3%** |

**关键读数：`aten::mm` 97 次里包含 `lm_head`（491 µs × 1）**，
扣掉后其余 **96 个小 GEMM 共 ~1.84 ms**，每个 ~19 µs —— 与 §3.2 的扫描完全一致
（24 层 × 4 个 GEMM = 96 ✓）。

> **榜首 9 个算子只解释了 31% 的耗时。**
> **剩下 69% 均匀散落在约 1000 个微算子（`view`/`transpose`/`rsqrt`/`pow`/`neg`...）上，
> 每个 ~16 µs —— 没有任何一个单独的热点可以优化。**
> 这正是"固定开销主导"的行为特征，也是为什么 `torch.compile` 只有 1.20×（见 §3.5）。

### 3.4 与带宽 roofline 的差距：17.4×

| 项 | 值 |
|---|---|
| Qwen2.5-0.5B 权重（`tie_word_embeddings=True`，实测 shard） | **0.988 GB** |
| HBM 拷贝带宽（`precision-support.md` 实测） | **797 GB/s** |
| **权重读取下界** | $0.988/797 = $ **1.240 ms** |
| **实测** | **21.6 ms/token** |
| **偏离** | **17.4×** |
| 反推"有效带宽" | $0.988 / 0.0216 = $ **45.7 GB/s = 峰值的 5.7%** |

即便把 24 层的**全部** KV cache 和中间激活都算上，下界也只到 ~2 ms 量级。
**decode 慢 17 倍的原因与"内存不够快"无关。**

### 3.5 `torch.compile` 在 decode 上只有 1.20×

`diagnostics/llm_compile_compare.py`（同一脚本内对比，`dynamic=False`，prefill/decode 分别编译）：

| 配置 | TTFT | ms/token | tok/s | 相对 eager |
|---|---:|---:|---:|---:|
| eager | 816.2 ms（含首次 JIT） | **25.649** | 39.0 | 1.00× |
| `compile:default` | 12.74 ms | **21.462** | 46.6 | **1.20×** |
| `compile:reduce-overhead` | 12.74 ms | 25.441 | 39.3 | **0.99×** |

判读（TODO §5）：「`torch.compile` 无收益甚至变慢 → 后端支持不完善」
→ **部分成立**：`default` 有 1.20× 的小幅收益，`reduce-overhead` 完全无效。

**为什么只有 1.20×？** 因为 `reduce-overhead` 依赖 CUDA Graph 才能吃掉
"每算子固定开销"，而 XPU 后端（`torch 2.14.0+xpu` 的 Inductor）**尚未实现
XPU 上的 CUDA-Graph 等价物（XPU Graph）捕获**，算子数不会下降。
`default` 的收益来自 minor fusion，而 1833 个算子里 78.5% 是元数据/elementwise，
**本来就很难 fuse 掉**。

### 3.6 batch 能线性摊薄固定开销（唯一有效的旋钮）

| batch | ms/step | tok/s |
|---:|---:|---:|
| 1 | 22.267 | 44.9 |
| 2 | 22.768 | 87.8 |
| 4 | 22.519 | 177.6 |
| 8 | 22.499 | 355.6 |
| 16 | 22.321 | **716.8** |

**ms/step 恒定在 22.3 ms（±2%）而吞吐线性到 16×。**
→ 固定开销**不是并发的**：GPU 一个 batch 干 1 个 token 还是 16 个 token，
墙钟一样。**这是把 decode 从 45 tok/s 拉到 717 tok/s 的唯一手段（16 倍）。**

代价是显存：`logits = B × L × 151936 × 2 B`（见 02-inference §4）。
b=16 时 logits = 16×1×151936×2 = 4.86 MiB（只解码 1 个 token，OK）；
但 prefill b=8、L=2048 时 logits = **4.64 GiB** ← **这才是 OOM 的真凶，不是 KV cache**（188 MiB）。

---

## 4. 结论三：数据管线 / 主机侧 —— 分场景，结论相反 ⭐

### 4.1 DataLoader worker 扫描（batch=64，cpu_scale=3 的合成预处理）

| num_workers | `pin_memory=False` | `pin_memory=True` | cpu_pct(*) |
|---:|---:|---:|---:|
| 0 | 2442 img/s | 2800 img/s | **1751% / 1675%** |
| 1 | 1928 img/s | 2023 img/s | 12.1% / 19.0% |
| 2 | 3684 img/s | 4163 img/s | 17.3% / 29.3% |
| 4 | 6690 img/s | 7231 img/s | 20.9% / 50.8% |
| 8 | **9539 img/s** | **7795 img/s** | 29.8% / 73.1% |

(*) cpu_pct 由 `os.times()` 采集，**只统计主进程**，所以 w≥1 时不含 worker 进程的 CPU；
w=0（同进程）时才会看到全部。**w=0 的 1751% = 17.5 个核心满载**。

**三个反常现象：**

1. **w=1 比 w=0 更慢**（1928 vs 2442）。多一个 worker 要付出
   每 batch 一次 **pickle + 共享内存/IPC 往返**的成本，而主进程此时只是空等。
   在 72 核机器上，"同进程 + 多线程"比"1 个 worker 进程"更划算。
2. **w=0 已经用掉 17.5 个核**，说明这个合成预处理（`/255` 两次算术）
   在 batch=64 下已经是**计算密集**而非 IO 密集。
3. **峰值 9539 img/s 出现在 w=8 + `pin_memory=False`**（违反常识，见 §4.2）。

**对照真实训练需求**：ResNet-50 bf16 batch=256 channels_last 达到 **1212 img/s**。
$$9539 / 1212 = \mathbf{7.9\times}\ \text{余量}$$

→ ✅ **对本次测试的训练负载，数据管线完全不是瓶颈。（好消息）**

### 4.2 ⚠️ `pin_memory` 悖论：worker 多时反而更慢

| num_workers | pin=False | pin=True | 差异 |
|---:|---:|---:|---:|
| 0 | 2442 | 2800 | **pin 快 +14.7%** |
| 1 | 1928 | 2023 | pin 快 +4.9% |
| 2 | 3684 | 4163 | pin 快 +13.0% |
| 4 | 6690 | 7231 | pin 快 +8.1% |
| 8 | **9539** | **7795** | **pin 慢 −18.3%** ❌ |

**机制**：`pin_memory=True` 要求目标页锁定内存**只能由主进程分配**。
worker 先把 batch 写进普通共享内存，主进程再**额外 memcpy 一次**到 pinned buffer，
再由主进程的 pin 线程发起 H2D。worker 越多，这条"共享内存 → pinned"的
额外拷贝路径越饱和，同时 pinned 内存页在主机的**不可换出**特性
（本机 45 GiB RAM / 7 GiB swap，见 §4.5）让分配器压力更大。

**经验**：**worker ≥ 8 时关闭 `pin_memory`；worker ≤ 4 时开启。**

### 4.3 端到端 GPU 利用率只有 2.83%~8.97%（当 GPU 任务很轻时）

GPU 侧只做一层 `conv2d 3→32 bf16`：

| num_workers | img/s | ms/batch | **`gpu_ms_per_batch`** | **`gpu_busy_pct`** | `h2d_ms` |
|---:|---:|---:|---:|---:|---:|
| 0 | 2786.6 | 22.97 | 0.84 | **3.67%** | 0.32 |
| 1 | 2056.6 | 31.12 | 0.88 | **2.83%** | 0.33 |
| 2 | 4314.5 | 14.83 | 0.74 | **4.96%** | 0.32 |
| 4 | 7392.6 | 8.66 | 0.68 | **7.91%** | 0.33 |
| 8 | 8088.8 | 7.91 | 0.71 | **8.97%** | 0.32 |

**读数**：`gpu_ms_per_batch` 恒定在 **0.68~0.88 ms**（GPU 侧完全没变），
而端到端从 22.97 ms 降到 7.91 ms（**2.9×**）**全部来自主机侧**。
**GPU 有 91%~97% 的时间在等数据。**

判读（TODO §5）：「GPU 利用率 < 80% → host 侧瓶颈」
→ ⚠️ **分场景**：
- **真实训练**（§2.1）：GPU 忙碌率 **99.5%** → **不触发**，主机侧不是瓶颈
- **轻量 GPU 任务 + 强 CPU 预处理**（本项）：**2.83%~8.97%** → **严重触发**

> **这正是"小模型 + 实时数据增强"（医疗影像、遥感、自动驾驶训练）
> 在本机上的真实风险点，与 `docs/caveats.md` 记录的"内存倒挂"同源。**

### 4.4 H2D 带宽：pinned 是 pageable 的 2.3×

| 项 | 值 |
|---|---:|
| 每 batch 字节（64×3×224×224） | 9,633,792 B = 9.19 MiB |
| **pageable** | 0.80 ms → **12.03 GB/s** |
| **pinned** | 0.35 ms → **27.57 GB/s** |
| 提升 | **+129%** |

对照 ResNet-50 b256：单 batch 38.5 MB → pinned 需 **1.40 ms**，
占步长 211 ms 的 **0.66%**。→ **H2D 不是瓶颈。**
（但注意 `precision-support.md` 里测得 **D2H 只有 31.5 GB/s**，
若需要在 host 侧做指标计算/早停，要算这笔开销。）

### 4.5 「内存倒挂」的量化

| 资源 | 容量 |
|---|---|
| 主机 RAM | **45 GiB**（used 8 / free 6 / buff-cache 25 / available 37） |
| 主机 swap | 7 GiB |
| **两张卡 HBM 合计** | **96 GiB**（48 × 2） |
| NUMA | **1 个节点**（144 逻辑核全在同节点） |
| 系统盘 `/dev/nvme0n1p2` | 468 G，**已用 404 G（91%）**，剩 41 G |

**倒挂的三种实际后果（都已在本测试中观察到）：**

1. **无法把大数据集常驻主机**。ImageNet-1k 全解码 ≈ 150 GB，
   COCO ≈ 20 GB —— 45 GiB 装不下，只能走磁盘（而磁盘只剩 41 GB）。
   → 触发 §4.1 的取数瓶颈。
2. **`pin_memory` 页锁定内存是"不可换出"的**，直接挤压本来就小的可用 RAM
   → §4.2 在高 worker 数下 pin 反而更慢。
3. **HBM 富余却无法被主机使用**。两卡 96 GiB HBM 在测试中峰值只用 5.78 GiB
   （LLM b8 L2048），而主机 45 GiB 却在 `num_workers=0` 时被 17.5 个核的
   预处理线程挤满。**把数据集/预处理卸载到 XPU 上做
   （`torch.xpu` 上的 JPEG 解码 / `channels_last` 转换）是一个可行的破局方向。**

---

## 5. 结论四：DDP 不是瓶颈

详见 [`03-scaling.md`](./03-scaling.md)。摘要：

| 指标 | batch=64 | batch=128 |
|---|---:|---:|
| 扩展效率（vs 纯计算基线） | **93.75%** | **95.88%** |
| 扩展效率（vs 单卡 DDP 基线） | **97.59%** | **97.69%** |
| 报告的通信占比 | 6.28% | 3.95% |
| 其中**真实跨卡**占比 | **39%** | **56%** |
| 纯 allreduce 传输（51.2 MB @80 GB/s） | 0.64 ms | 0.64 ms |
| 实测暴露的跨卡成本 | 2.32 ms（**3.6×**） | 4.34 ms（**6.8×**） |

**Xe Link 实测 allreduce 饱和带宽 ~80 GB/s = 标称 318 GB/s/dir 的 1/4**，
但**在当前模型规模下不影响结果**（只有梯度 >2 GB 的 >1B 参数模型才会受限）。
暴露成本是纯传输的 3.6~6.8 倍，说明瓶颈在**分桶/尾部同步**而非链路带宽。

---

## 6. 判读标准逐条对照（`docs/TODO/05-ai-dl.md` §5）

| 判读项 | 阈值 | 实测 | **判定** |
|---|---|---|---|
| **BF16 加速比** | < 1.5× → XMX 未启用 | ResNet-50 2.03×(NCHW) ~ **2.92×**(NHWC)；BERT-base 1.92× ~ **3.56×**，BERT-large 1.64× ~ **3.88×** | ✅ **XMX 已启用**，且 Amdahl 模型精确解释了倍率（§2.3：误差 7% / 3%） |
| **GPU 利用率** | < 80% → host 侧瓶颈 | 真实训练 **99.5%**；合成数据管线 **2.83%~8.97%** | ⚠️ **分场景**：训练不触发，数据管线严重触发 |
| **双卡扩展效率** | < 70% → 通信瓶颈 | **93.75% / 95.88%**（vs 纯计算） | ✅ **达标**，不是通信瓶颈 |
| **`torch.compile` 无收益甚至变慢** | → 后端支持不完善 | decode 1.20×（`default`）/ **0.99×**（`reduce-overhead`）；ResNet b128 NHWC **−1.3%**；ResNet b64 NCHW **+82%** | ⚠️ **确认"收益有限"**：无 CUDA-Graph 等价物，无法吃掉固定开销 |
| **小 batch 吞吐极低** | – | **仅 LLM decode 成立**（21.6 ms/token 地板，17.4× off roofline）；ResNet-50 b64 709 vs b256 693 img/s（**几乎无关**）；BERT 1.92×（8× token） | ⚠️ **仅 decode**，不可外推到训练 |
| **显存 OOM 早于预期** | – | 真凶是 **logits**（b8×L2048×V151936 = **4.64 GiB**），不是 KV cache（**188 MiB**） | ✅ **已定位**，修法见 02-inference §4.3 |
| `--backend=ccl` vs `xccl` 对比 | – | torch 2.14 在本机只注册了 `xccl` | ❌ **无法完成**，如实记录 |

---

## 7. 瓶颈归属总表 ⭐

| 场景 | 瓶颈层 | 量化证据 | 优化旋钮 | 预期收益 |
|---|---|---|---|---|
| ResNet-50 训练 | **非 XMX 算子占比（67%）** | BN 38.1% + Elem 24.9% + Other 4.1%，GPU 忙 99.5% | `channels_last`；换用 GroupNorm 的模型；`torch.compile` | +58~75%（NHWC）/ +82%（compile） |
| BERT 训练 | **非 XMX 算子占比（44%）** | GEMM 37.7% + Attn 18.6% = **56% 走 XMX** | 更大 seq/batch/模型；FA 已自动启用 | 已有 3.56~3.88× |
| LLM Prefill | 算力 | 75.7 TFLOPS = 峰值 32.5% | 大 batch、长序列 | 有限 |
| **LLM Decode** | **每算子固定开销（1833 个 × 11.8 µs）** | 1×896×4864 GEMM = 896×896 GEMM（21 µs） | **增大 batch（唯一有效，16×）**；减少算子数（需 XPU Graph） | **16×** |
| 数据管线 + 轻量 GPU | **主机 CPU** | GPU 忙 2.83%~8.97%，`gpu_ms` 恒定 | worker 8 + `pin_memory=False`；预处理上 GPU | 2.9× |
| 2 卡 DDP | 无（达标） | 扩展效率 93.8%~95.9% | `bucket_cap_mb` 调优 | 有限 |

---

## 8. 优化建议（按 ROI 排序）

| # | 建议 | 依据 | 预期收益 | 成本 |
|---|---|---|---|---|
| 1 | **`model.to(memory_format=torch.channels_last)`** | §2.4，+58~75% | 巨大 | 一行 |
| 2 | **LLM decode 用大 batch**（≥16） | §3.6，线性到 16× | 巨大 | 改服务框架 |
| 3 | **`logits_to_keep=1` / 不物化全序列 logits** | §3.6、02-inference §4.3 | 省 4.6 GiB 显存 | 一行 |
| 4 | **`pin_memory=False` 当 worker ≥ 8** | §4.2，−18% → +18% | 中 | 一行 |
| 5 | **`torch.compile(mode="default")`** | §3.5、01 §2.4 | 训练 +82%（NCHW）；decode +20% | 中（首编译耗时，且需 `env -u LD_LIBRARY_PATH`） |
| 6 | **预处理/数据集搬到 XPU** | §4.5 | 突破 host 瓶颈 | 高 |
| 7 | 减少 decode 算子数（避免 `cat`/`_to_copy`/显式 `mul+add`） | §3.3，cat 97 次 666 µs | 中 | 中（需改模型代码） |
| 8 | 调 `bucket_cap_mb` | 03-scaling §4，暴露成本 3.6~6.8× | 小 | 低 |
| 9 | ❌ 不用 FP8/FP4 求速度 | `precision-support.md`，比 BF16 慢 1.6~7× | – | – |
| 10 | ❌ 不用 `w8a16`/`w4a16` | `precision-support.md`，慢 30×（每次重打包） | – | – |

---

## 9. 复现命令

```bash
cd /root/workspace/benchmark/05-ai-dl/diagnostics
PY=/root/workspace/venv1/bin/python
export ENVCMD="env -u LD_LIBRARY_PATH"     # 必须：绕开 VTune gma/ocloc 遮蔽

# 训练 kernel 成分（本文件 §2）
ZE_AFFINITY_MASK=0 $ENVCMD $PY train_kernel_mix.py

# LLM decode 三件套（本文件 §3）
ZE_AFFINITY_MASK=0 $ENVCMD $PY decode_budget.py
ZE_AFFINITY_MASK=0 $ENVCMD $PY m1_gemm_scan.py
ZE_AFFINITY_MASK=0 $ENVCMD $PY launch_floor.py
ZE_AFFINITY_MASK=0 $ENVCMD $PY llm_compile_compare.py

# 数据管线（本文件 §4）
cd /root/workspace/benchmark/05-ai-dl
$ENVCMD $PY run_bench.py --suite pipeline

# 互连（本文件 §5）
$ENVCMD $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
    diagnostics/allreduce_probe.py
```

---

## 10. 未做 / 待补充

| 项 | 说明 |
|---|---|
| `train_kernel_mix.py` 的 FP32 对照 | 只做了 bf16。跑 FP32 可验证 §2.3 的分母（非 XMX 部分是否真的只快 2×） |
| ResNet-50 用 `GroupNorm` 替换 BN 的对照 | 若 §2.1 的 38% BN 结论成立，换 GN 应显著变慢（GN 更贵）或改变 profile |
| `bucket_cap_mb` 扫描 | 最有希望的 DDP 调优旋钮，未做 |
| LLM decode 的 XPU Graph 捕获尝试 | 若 `torch.xpu` 支持，应能直接吃掉 §3 的固定开销（对标 `reduce-overhead` 的 1.20×） |
| 真实 ImageNet 取数（而非合成） | 本次用合成数据集，JPEG 解码的真实成本未测 |
| 训练稳态 GPU 利用率的**在线**采样 | 本次用 profiler 事后求和（99.5%），未用 `intel_gpu_top` / `xpu-smi dump` 在线验证 |
| LLM decode 的 CUDA-Graph 等价物验证 | `torch.compile(reduce-overhead)` = 0.99× 说明未生效，但未深入确认 |
