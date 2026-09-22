# ⑤-5 AI / 深度学习测试总体结论

> 对应测试计划：[`../../TODO/05-ai-dl.md`](../../TODO/05-ai-dl.md)
> 分项结论：[①训练吞吐](./01-training-throughput.md) ／ [②推理](./02-inference.md) ／ [③扩展性](./03-scaling.md) ／ [④瓶颈诊断](./04-bottleneck.md)
> 测试代码：[`../../../benchmark/05-ai-dl/`](../../../benchmark/05-ai-dl/)（13 个 suite + 7 个定向探针）
> 测试日期：2026-09-21 ~ 2026-09-22　　硬件：2 × Intel Data Center GPU Max 1100（PVC, Production ES）

---

## 0. 一页结论

### 0.1 对测试目标的逐条回答

| # | 测试目标（TODO §1） | 回答 | 结论 |
|---|---|---|---|
| 1 | **训练吞吐**（ResNet-50 / BERT） | ResNet-50 bf16 **1212 img/s** = **29.75 TFLOPS**；BERT-base bf16 L512/b32 **77,107 tok/s** = **50.67 TFLOPS**；BERT-large L512/b16 **57.27 TFLOPS** | ✅ |
| 2 | **混合精度收益 → 验证 XMX 是否启用** | ResNet-50 **2.92×**、BERT-base **3.56×**、BERT-large **3.88×**（均远超 1.5× 阈值）；且 Amdahl 模型能**定量预测**这两个数字 | ✅ **XMX 满速工作** |
| 3 | **多卡扩展效率** | 2 卡 DDP **93.75%~95.88%**（vs 纯计算基线），**97.59%~97.69%**（扣掉 DDP 包装） | ✅ **达标** |
| 4 | **推理性能**（LLM prefill/decode/TTFT/显存） | prefill **76,648 tok/s**（b8 L2048）；decode **45.7 → 716.8 tok/s**（batch 1→16）；TTFT **23.1 ms** 地板；显存模型见 02 §4 | ✅ 但 **decode 偏低** |
| 5 | **算子级基准** | FP32 **22.16** / BF16 **232.8** / INT8 **398.9** TFLOPS；HBM 拷贝 **797 GB/s** | ✅ 见 [`precision-support.md`](../../precision-support.md) |
| 6 | **找出主机侧瓶颈**（内存倒挂） | 真实训练 GPU 忙碌率 **99.5%（不触发）**；但轻量 GPU 任务 + 强 CPU 预处理时 GPU 只忙 **2.83%~8.97%** | ⚠️ **分场景，见 §5** |

### 0.2 一句话

> **这台机器的 AI 能力是"强 XMX + 弱主机 + 弱小算子效率"的典型组合。**
> 大算力场景（BF16 训练、长序列 prefill）表现良好；
> **两类场景会严重受损**：
> ① **非 XMX 算子占比高的模型**（ResNet-50 的 BN+ReLU 占 **67%** GPU 时间 → BF16 只能拿到 2.9×）；
> ② **小算子密集的场景**（LLM decode 一步 1833 个算子、每个 ~12 µs 地板 → 21.6 ms/token，离带宽 roofline 差 **17.4×**）。
> 前者靠 **`channels_last`（+58~75%）** 就能救，后者**只有增大 batch 一个有效手段（线性到 16×）**。

---

## 1. 交付物索引

| 类别 | 路径 | 内容 |
|---|---|---|
| 测试代码 | `benchmark/05-ai-dl/` | `run_bench.py` + `xpu_bench/`（13 个 suite） |
| 原始结果 | `benchmark/05-ai-dl/results/bench_2026092*.{json,md}` | 8 次完整运行 |
| 分布式原始输出 | `benchmark/05-ai-dl/results/ddp_raw/*.json` | 6 个 DDP worker 输出 |
| 定向探针 | `benchmark/05-ai-dl/diagnostics/` | 7 个脚本 + README（含 4 个踩过的坑） |
| 结论 | `docs/Conclusion/05-ai-dl/01~05-*.md` | 本文件所在的 5 篇 |

---

## 2. 全部关键数字总表

### 2.1 训练吞吐

| 模型 | dtype | 配置 | 单卡吞吐 | TFLOPS | 峰值占比 |
|---|---|---|---:|---:|---:|
| ResNet-50 (25.6 M) | FP32 | b256 NCHW | 332.7 img/s | 8.16 | 36.8% |
| ResNet-50 | FP32 | b256 **NHWC** | **415.0 img/s** | **10.18** | **45.9%** |
| ResNet-50 | BF16 | b256 NCHW | 693.0 img/s | 17.01 | 7.3%(XMX) |
| ResNet-50 | BF16 | b256 **NHWC** | **1212.3 img/s** | **29.75** | **12.8%**(XMX) |
| ResNet-50 | BF16 | b64 NHWC + `compile:default` | **1322.7 img/s** | **32.46** | 13.9%(XMX) |
| ResNet-50 | BF16 | b128 NHWC（2 卡 DDP 全局） | **1364.7 img/s** | **33.49** | – |
| BERT-base (109.5 M) | FP32 | L128 b32 | 23,450 tok/s | 15.41 | 69.6% |
| BERT-base | BF16 | L128 b32 | 56,602 tok/s | 37.19 | 16.0%(XMX) |
| BERT-base | BF16 | L512 b32 | **77,107 tok/s** | **50.67** | 21.8%(XMX) |
| BERT-large (335.2 M) | FP32 | L512 b16 | 7,333 tok/s | 14.75 | 66.6% |
| BERT-large | BF16 | L512 b16 | **28,475 tok/s** | **57.27** | **24.6%**(XMX) |

> **BF16/FP32 加速比**：ResNet-50 NCHW **2.03~2.08×** / NHWC **2.72~2.92×**；
> BERT-base **1.92~3.56×**、BERT-large **1.64~3.88×**（均随 seq_len 增大而提高）。
> **模型越大 TFLOPS 越高**：同条件（bf16 L512 b16）large 57.27 vs base 50.67（**+13%**）。

### 2.2 推理（Qwen2.5-0.5B-Instruct, bf16, 权重 0.988 GB）

| batch | input_len | TTFT | prefill tok/s | **decode tok/s** | ms/token | 峰值显存 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 23.06 ms | 5,550 | **45.7** | 21.89 | 0.97 GiB |
| 4 | 128 | 22.97 ms | 22,287 | 182.2 | 21.95 | 1.08 GiB |
| 8 | 2048 | 213.76 ms | **76,648** | 370.1 | 21.61 | 5.78 GiB |
| **16**（探针） | 1 | – | – | **716.8** | 22.32 | – |

### 2.3 扩展性（ResNet-50 bf16, xccl）

| batch | 1 卡裸模型 | 2 卡 DDP | 加速比 | **扩展效率** | 通信占比 | 真实占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 708.3 img/s | 1328.1 img/s | 1.875× | **93.75%** | 6.28% | 39% |
| 128 | 711.7 img/s | 1364.7 img/s | 1.917× | **95.88%** | 3.95% | 56% |

### 2.4 瓶颈诊断

| 项目 | 值 |
|---|---:|
| **GPU 利用率（真实训练稳态）** | **99.5%**（205.94 / 206.93 ms） |
| **GPU 利用率（合成数据管线端到端）** | **2.83% ~ 8.97%** |
| CPU 利用率（DataLoader `num_workers=0`） | **1751% = 17.5 核满载** |
| host→device 带宽 | pinned **27.57 GB/s** / pageable **12.03 GB/s** |
| DataLoader 最优 worker 数 | **8**（**且须 `pin_memory=False`**） |
| BF16 / FP32 加速比 | ResNet-50 **2.92×** / BERT **3.56×** |
| **2 卡 / 1 卡 加速比** | **1.875 ~ 1.917×** |

### 2.5 与硬件峰值的对照（`precision-support.md`）

| 指标 | 峰值 | 最佳实测 | 占比 |
|---|---:|---:|---:|
| FP32 GEMM | 22.16 TFLOPS | 22.07 | **99.6%** |
| BF16/FP16 GEMM | 232.8 / 233.9 TFLOPS | 237.6 (bf16) | **102%** |
| INT8 GEMM | 398.9 TOPS | 398.9 | 100% |
| HBM 拷贝 | 797 GB/s | 834 (vector) | **105%** |
| conv2d bf16 单算子 | – | 167 TFLOPS | 72% of XMX |
| **端到端 ResNet-50 训练** | BF16 XMX 232.8 | 29.75 | **12.8%** |
| **端到端 BERT 训练** | BF16 XMX 232.8 | 50.67 | **21.8%** |
| **端到端 LLM prefill** | BF16 XMX 232.8 | 75.7 | **32.5%** |
| **端到端 LLM decode** | HBM 797 GB/s | 45.7 GB/s | **5.7%** |

---

## 3. 判读标准裁定（TODO §5 逐条）

| # | 现象（TODO 原文） | 判定 | 证据 |
|---|---|---|---|
| 1 | **BF16 加速比 < 1.5×** → XMX 未启用 | ❌ **不成立** | ResNet 2.03~2.92×、BERT-base 1.92~3.56×、BERT-large 1.64~3.88×，全部 >1.5×。且 Amdahl 模型（$f$=XMX 占比）**定量预测**了倍率：2.72 vs 实测 2.92、3.68 vs 实测 3.56（误差 7% / 3%）→ **XMX 是满速工作的** |
| 2 | **GPU 利用率 < 80%** → host 侧瓶颈 | ⚠️ **分场景** | **训练 99.5% → 不触发**；**合成数据管线 2.83%~8.97% → 严重触发**。`gpu_ms_per_batch` 恒为 0.68~0.88 ms 而端到端从 22.97 降到 7.91 ms，全部差异来自主机 |
| 3 | **双卡扩展效率 < 70%** → 通信瓶颈 | ❌ **不成立** | **93.75% / 95.88%**，远超阈值。且 6.28% 的"通信占比"里有 39%~61% 是 DDP 包装自身开销（单卡加包装就要 +1.9%~+4.1%），真实跨卡只有 2.3~4.3 ms |
| 4 | **小 batch 吞吐极低** → launch 开销/未压满 | ✅ **成立，但仅限 LLM decode** | decode: 1×896×4864 GEMM（2.1 MB 权重）耗时 **22.8 µs**，896×896 GEMM（1.6 MB）**20.7 µs**，而 896×9728 GEMM（**17.4 MB**，8.3× 权重）只要 **20.9 µs** → 完全由 launch 地板决定；但 **ResNet-50 b64 709 vs b256 693 img/s**（几乎与 batch 无关）、BERT 8× token 只涨 1.92× → **不可外推到训练** |
| 5 | **`torch.compile` 无收益甚至变慢** → 后端不完善 | ⚠️ **部分成立** | decode `default` **1.20×**、`reduce-overhead` **0.99×**；ResNet b128 NHWC **−1.3%**；但 ResNet b64 NCHW **+82%**（等价于布局修正）。根因：**XPU 后端无 CUDA-Graph 等价物**，吃不掉每算子固定开销 → "后端支持不完善"这条**成立** |
| 6 | **显存 OOM 早于预期** → 45.6 GiB 上限/碎片 | ⚠️ **定位到具体原因** | **不是 45.6 GiB 单次分配上限**，也**不是 KV cache**（L2048 只有 188 MiB/token 序列）。真凶是 **logits**：b8×L2048×151936×2 B = **4.64 GiB**。修法：`logits_to_keep=1` |
| 7 | FP16 与 BF16 峰值几乎重合 / 想开 fp16 累加无收益 | ✅ **正常** | 由硬件决定，见 [`precision-support.md` §7.2](../../precision-support.md) |

---

## 4. 三个反直觉发现 ⭐

### 4.1 XMX 是满速的，只是"没活干"

ResNet-50 bf16 b256 NHWC 一个训练步（206.93 ms）的 kernel 成分
（真 kernel `self_device_time` 合计 206.24 ms，1023 次调用，**GPU 忙碌率 99.5%**）：

| family | 占比 |
|---|---:|
| **BatchNorm**（53 层 × 4 kernel，每个 287~449 µs） | **38.1%** |
| **Conv/GEMM**（`gen_conv` × 158，414 µs/次） | **32.6%** |
| **Elementwise/ReLU**（129 次，373~671 µs/次） | **24.9%** |
| Other（MaxPool / AdamW / zero_out） | 4.1% |
| **非 XMX 合计** | **67%** |

$$\text{Speedup} = \frac{1}{\frac{1-f}{2} + \frac{f}{10.5}},\quad f = \text{XMX 可用占比}$$

| 模型 | $f$ | 预测 | 实测 | 误差 |
|---|---:|---:|---:|---:|
| ResNet-50 | **0.326** | **2.72×** | **2.92×** | +7% |
| BERT | **0.563** | **3.68×** | **3.56×** | −3% |

→ **"BF16 只到 XMX 峰值的 12.8%"这个数字具有误导性**：ResNet-50 有 **67%** 的 GPU 时间根本用不到乘加阵列。
用 XMX 峰值做分母评价 ResNet-50 是不合理的（BERT 的 $f$=56%，所以能好一些）。
→ 顺带解释了 `channels_last` 为什么有 **+58~75%** 的收益（BN/elementwise 在 NCHW 下访存不连续）。

### 4.2 LLM decode 的敌人是"算子个数"，不是"带宽"

| 事实 | 数值 |
|---|---:|
| 单步算子数（eager, `TorchDispatchMode`） | **1833** |
| 每算子固定成本 | **~11.8 µs** |
| 1833 × 11.8 µs | **= 21.6 ms** = 实测单步 |
| 与带宽 roofline 的偏差（0.988 GB / 797 GB/s = 1.240 ms） | **17.4×** |
| 反推有效带宽 | 45.7 GB/s = **峰值的 5.7%** |
| 唯一有效的旋钮：batch 1→16 | **44.9 → 716.8 tok/s（16× 线性）** |

**决定性证据**：`1×896×4864` GEMM 与 `896×896` GEMM 耗时相同（~21 µs），
而权重差 8.3× 时耗时几乎不变（22.8 vs 20.9 µs）。
→ 这个区间里"把单个 GEMM 做快"毫无意义，**只能减少算子数或增大 batch**。

### 4.3 "内存倒挂"的风险是**分场景**的

主机 45 GiB RAM < 两卡 96 GiB HBM。但这个倒挂**并不总是**瓶颈：

| 场景 | GPU 忙碌率 | 是否触发 |
|---|---:|---|
| ResNet-50 / BERT 训练 | **99.5%** | ❌ 不触发 |
| DataLoader 取数上限 9539 img/s vs 训练需求 1212 img/s | – | ❌ **7.9× 余量** |
| H2D 单 batch（pinned 27.57 GB/s，ResNet b256 需 1.40 ms / 211 ms） | – | ❌ **0.66%** |
| **轻量 GPU 任务 + 强 CPU 预处理** | **2.83%~8.97%** | ✅ **严重触发** |

**两个反常现象（都是主机受限的指纹）：**

1. **`num_workers=1` 比 `=0` 更慢**（1928 vs 2442 img/s）—— 多一个 worker 要付出
   每 batch 一次 pickle + IPC 往返，而主进程只是空等。**72 核机器上 w=0 同进程多线程更划算。**
2. **`pin_memory=True` 在 worker=8 时反而慢 18.3%**（7795 vs 9539 img/s），
   但 worker≤4 时快 5%~15%。机制：pinned buffer 只能由主进程分配 → worker 交付后再多一次 memcpy。
   → **`worker ≥ 8` 时关闭 `pin_memory`。**

---

## 5. 建议清单

### 5.1 立即可做（一行改动级）

| # | 建议 | 依据 | 预期收益 |
|---|---|---|---|
| 1 | **`model.to(memory_format=torch.channels_last)`** | 04 §2.4 | **+58%~+75%**（ResNet 训练） |
| 2 | **BF16 + `channels_last` 一起开** | 01 §2.2 | 合计 **3.5×** vs FP32 NCHW |
| 3 | **LLM decode 用大 batch（≥16）** | 04 §3.6 | **线性到 16×**（45.7 → 716.8 tok/s） |
| 4 | **`logits_to_keep=1`**，不物化全序列 logits | 02 §4.3 | 省 **4.64 GiB** 显存 |
| 5 | **worker ≥ 8 时设 `pin_memory=False`** | 04 §4.2 | **+18%** |
| 6 | `torch.compile(mode="default")` | 01 §2.4、02 §3.3 | 训练 **+82%**（NCHW）/ decode **+20%** |

### 5.2 需要改造

| # | 建议 | 依据 |
|---|---|---|
| 7 | 把**数据预处理搬到 XPU**（JPEG 解码、layout 转换）以突破 45 GiB 主机瓶颈 | 04 §4.5 |
| 8 | 减少 decode 算子数（避免 `cat` 97 次 / `_to_copy` 101 次 / 显式 `mul+add`） | 04 §3.3 |
| 9 | 关注模型的**非 XMX 算子占比**：BN/LayerNorm/elementwise 占比高的模型在本机会显著吃亏 | 04 §2.3 |

### 5.3 ❌ 不要做

| 做法 | 原因 |
|---|---|
| 用 **FP8 / MXFP8 / FP4 / NVFP4** 求速度 | 本机全走软件回退，**比 BF16 慢 1.6~7.6×**（`precision-support.md`） |
| 用 **`w8a16` / `w4a16`** 权重量化 | 每次调用重新打包权重，**慢 ~30×**，完全不可用 |
| 期待 `torch.compile(reduce-overhead)` 救 decode | 0.99×，因 XPU 无 CUDA-Graph 等价物 |
| 期待 `int16/int64/int32` matmul | oneDNN 不支持（`Short/Long is not supported`） |
| 在小 batch 下优化 decode 的单算子效率 | 该区间完全由 ~20 µs 的 launch 地板决定 |

---

## 6. 与其它测试项（①~⑧）的衔接

| 测试项 | 本次联动 |
|---|---|
| **② 算力峰值** | **22.16 TFLOPS (FP32) / 232.8 (BF16) / 398.9 TOPS (INT8)** 是本文所有"峰值占比"的分母。本次发现 **FP64 = 0.78× FP32**（非假定的 1/2），已回写 `common.py` 的 `FP64_ALU_RATIO` |
| **③ 内存带宽** | **797 GB/s** 是所有 roofline 的分母（decode 17.4× off roofline、ResNet 带宽下界） |
| **④ 互连** | ⭐ **Xe Link allreduce 实测饱和 ~80 GB/s = 标称 318 GB/s/dir 的 1/4**，小消息延迟地板 ~26 µs，`barrier()` **119 µs**。**建议 ④ 用 `ze_peak` / IMB-MPI1-GPU / oneCCL benchmark 复核链路本身**，并与 80 GB/s 对照。本次在 <1B 参数模型下互连不构成瓶颈 |
| **① 健康/压力** | 训练稳态 GPU 忙碌率 **99.5%**，功耗 300 W 上限（范围 150~300 W）→ **建议 ① 补测长时满载下的功耗墙/降频** |
| **⑥ HPC 应用** | 本次通过 torch → oneDNN/MKL 间接验证了 BLAS 路径；直接 MKL/oneDNN GEMM 未测 |
| **⑦ Profiling** | 本次全程用 `torch.profiler`；**VTune 2026.4 / PTI 1.1 未使用**，建议 ⑦ 用它们验证本文的 kernel 成分结论（尤其 §4.1 的 38% BN） |
| **⑧ 能效** | 本次只记录了显存与吞吐；**功耗数据仅取自 `xpu-smi` 的静态功率上限**，未接入实时采样 |

---

## 7. 覆盖度与诚实性声明

### 7.1 完成情况

| 项 | 状态 |
|---|---|
| 13 个 suite（gemm/elementwise/membw/reduce/attention/conv/quant/precision/resnet/bert/llm/ddp/pipeline） | ✅ |
| 7 次完整运行，结果落在 `benchmark/05-ai-dl/results/` | ✅ |
| 4 个真实 HF 模型（ResNet-50 / BERT / Qwen2.5-0.5B） | ✅ |
| DDP 三配置对照（1card_nodpp / 1card_ddp / 2card_ddp） | ✅ |
| 7 个定向探针 + 4 个踩坑记录 | ✅ |

### 7.2 与原计划的偏差（如实记录）

| 计划 | 实际 | 影响 |
|---|---|---|
| `--backend=ccl` vs `xccl` 对比（TODO §3.4） | ❌ **无法完成**：torch 2.14 在本机只注册了 `xccl`（`ccl`/`nccl`/`ucc`/`mpi` 全部 `False`） | 结论的"扩展性良好"只对 xccl 成立 |
| BERT-**Large** 训练（TODO §4 指标表） | ✅ **已补齐**：`bert-large-uncased`（335.2 M，24 层/1024 hidden）L128/L512 × b8/b16，共 8 点（`results/bench_20260922-014644.json`） | 无影响；反而**支持**了"模型越大 XMX 利用率越高"的结论（21.8% → **24.6%**） |
| BERT 输出层类 | ⚠️ 计划未指定，实际用 `BertForMaskedLM`（MLM 头，≈1/7 token 掩码） | 与 `BertForSequenceClassification` 的 FLOPs 略有差异（头层小，可忽略） |
| 4 卡扩展 | ❌ 硬件只有 2 张卡 | – |
| 真实的 ImageNet 取数 | ⚠️ 用**合成数据集**（uint8 + `cpu_scale` 算术） | JPEG 解码的真实成本**未测**；真实场景主机瓶颈可能比本次更严重 |
| 在线功耗/利用率采样 | ⚠️ 用 profiler 事后求和 | GPU 忙碌率 99.5% 未用 `intel_gpu_top` 在线交叉验证 |

### 7.3 已知的测量不确定性

| 项 | 说明 |
|---|---|
| profiler 插桩开销 | 插桩下的 device 合计会超过墙钟（BERT 178.59 vs 161.40 ms = **110.7%**）→ 该比值只作**定性**结论，ResNet 上报的是未插桩的墙钟 |
| `profiler` 单位 | `self_device_time_total` 是**微秒**；`aten::` 层与 kernel 层**都带 device time**，不去重会**重复计数** → 见 `diagnostics/README.md` 坑 3 |
| `comm_pct` 的含义 | 6.28% **高估**互连贡献（其中 39%~61% 是 DDP 包装开销） |
| 通信占比的测法 | `no_sync()` 差分测的是**暴露成本**，不含与 backward 重叠的部分 |
| ES 硅片 + Xe Link "Not Calibrated" | 所有互连数字应视为**下界** |
| 频率请求值固定 | 1550 MHz（`gt_min == gt_max`）只是**请求值**，不反映真实降额，也**无法测 boost 行为**；实测长时满载会被热/功耗降额（详见 `docs/TODO/08-power-efficiency.md`） |
| 系统盘 91% 满 | `/dev/nvme0n1p2` 468 G / 已用 404 G / 剩 41 G → 真实数据集实验受限 |
| **已修复的报告 bug** | BERT suite 曾硬编码 `model=bert-base-mlm, layers=12, hidden=768`，因此 BERT-Large 的 JSON/MD 元数据被标错（`params_m=335.2` 是真的，结构字段是假的）。已改为从 `model.config` 回填（`xpu_bench/models.py`）。**§3.3 表格中的数字全部来自实测，不受影响。** |

### 7.4 未做（建议后续补）

| 项 | 优先级 |
|---|---|
| **`bucket_cap_mb` 扫描**（8/25/50/100 MB） | 高 —— 04 §5 显示暴露成本是纯传输的 3.6~6.8 倍，这是最有希望的 DDP 旋钮 |
| **在真实 ImageNet/COCO 上重测数据管线** | 高 —— 当前用合成数据，可能低估主机瓶颈 |
| ResNet-50 的 `GroupNorm` 对照实验 | 中 —— 用于验证 §4.1 的"38% 是 BN"结论 |
| XPU Graph 捕获尝试 | 中 —— 若能生效可直接吃掉 decode 的固定开销 |
| `train_kernel_mix.py` 的 FP32 对照 | 中 —— 用于验证 Amdahl 模型的分母（非 XMX 部分是否真的只快 2×） |
| 用 VTune / PTI 交叉验证 kernel 成分 | 中 —— 属 ⑦ 的范围 |
| 长时满载功耗与降频 | 中 —— 属 ① 的范围 |
| LLM 的 L4096 输入 | 低 —— TODO §3.5 提到但本次最大到 L2048 |
| Triton 自定义 kernel | 低 —— TODO §3.1 提到；`attention` / `conv` / `gemm` suite 已从另一个角度覆盖 XMX 利用率 |
| LLM batch=32、PagedAttention | 低 —— TODO §3.5 提到；§4.3 已证明 KV cache 不是瓶颈 |
| GPTQ / AWQ 库路径（而非 torch 原生算子） | 低 —— TODO §3.5 "XPU 支持需确认"已由 `quant` suite 间接回答（W4A16 峰值仅 0.11× INT8） |

---

## 8. 总结论

1. **XMX 是这台机器的核心竞争力，且完全可用。** FP32 22.16 / BF16 **232.8** / INT8 **398.9** TFLOPS，
   单算子 conv2d 能到 167 TFLOPS（峰值的 72%），**硬件本身没有短板**。
2. **端到端只能拿到峰值的 13%~33%**，原因**不是** XMX 失效，而是
   **①非 XMX 算子占比（ResNet 67%）** 和 **②小算子固定开销（decode）**。
   两者都有明确的量化模型（Amdahl 公式 / 算子数 × 11.8 µs）。
3. **`channels_last` 是本次性价比最高的一项改动**（ResNet 训练 +58~75%），
   且它优化的是**访存布局**而非算力 —— 与 §2 的结论互为印证。
4. **LLM decode 的 21.6 ms/token 是"算子个数"决定的**，与 batch 无关（22.3 ms 恒定），
   **因此增大 batch 是唯一有效手段（线性 16×）**。这在小显存模型上完全可行，
   且顺带缓解了 §2.4 的显存问题（`logits_to_keep=1`）。
5. **2 卡扩展性不是问题**（93.8%~95.9%），**Xe Link 的实际带宽（80 GB/s）虽只有标称的 1/4，
   但在 <1B 参数模型上不影响结果** —— 这一点需要 ④ 独立复核。
6. **"内存倒挂"是真隐患，但只在这类场景爆发**：GPU 任务轻 + CPU 预处理重。
   本机 DataLoader 上限 9539 img/s 对本次测试的 1212 img/s 有 7.9× 余量，
   但**对真实 ImageNet 级数据（含 JPEG 解码）没有任何余量保证**。
7. **`torch.compile` 在 XPU 上收益有限**（decode 1.20×、`reduce-overhead` 0.99×），
   根因是**缺少 CUDA-Graph 等价物**，无法消除小算子开销 —— 这是当前软件栈最值得改进的一环。
