# ⑤-2 LLM 推理结论（prefill / decode / 显存）

> 对应测试计划：[`../../TODO/05-ai-dl.md`](../../TODO/05-ai-dl.md) §3.5
> 测试工具：[`../../../benchmark/05-ai-dl/`](../../../benchmark/05-ai-dl/)（suite: `llm`）
> 定向探针：[`../../../benchmark/05-ai-dl/diagnostics/`](../../../benchmark/05-ai-dl/diagnostics/)
> 测试日期：2026-09-22　　硬件：1 × Intel Data Center GPU Max 1100
> 原始数据：`results/bench_20260922-012519.json`（9 点）

---

## 0. 一页结论

| 问题 | 答案 |
|---|---|
| **prefill 能吃到多少算力？** | 峰值 **76.6 k tokens/s = 75.7 TFLOPS = BF16 峰值的 32.5%**（batch=8, L=2048） |
| **decode 能吃到多少带宽？** | 仅 **45 GB/s = HBM 峰值的 5.7%**；实测 21.6 ms/token vs 带宽 roofline 1.24 ms → **差 17 倍** |
| **decode 受什么限制？** | **每算子 ~20 µs 的固定代价 × 每步 1836 个算子**。与 batch、与上下文长度**都无关** |
| **加大 batch 有用吗？** | ✅ **线性有效**：b1 45.7 tok/s → b8 370 tok/s（8.1×），每步耗时几乎不变（21.9 → 21.6 ms） |
| **`torch.compile` 能救吗？** | ⚠️ 只能给 **1.20×**（同一脚本 25.65 → 21.46 ms/token）；`reduce-overhead` 无收益 |
| **显存主要被什么占用？** | **不是 KV cache**，是 `(B, L, vocab)` 的 **logits 张量**（(8,2048) 时占 4.64 GiB，是 KV cache 的 24 倍） |
| **该不该量化？** | decode 是**固定开销**受限而非带宽受限，**W4A16 量化救不了 decode**（且本机 W8A16 实测慢 30 倍，见 [`../../precision-support.md`](../../precision-support.md)） |

---

## 1. 测试口径

| 项目 | 值 |
|---|---|
| 模型 | **Qwen2.5-0.5B-Instruct**（bf16） |
| 架构 | 24 层 / 14 attn heads / **2 KV heads** / head_dim 64 / hidden 896 / intermediate 4864 / vocab 151936 |
| 权重 | 单 shard `model.safetensors` = **988,097,824 B (0.988 GB)**；`tie_word_embeddings=True`（lm_head 与 embedding 共享） |
| 精度 | bf16（权重与激活都是） |
| 测法 | `AutoModelForCausalLM` + `use_cache=True`；先按 input_len 做 warmup，再测 |
| TTFT | 一次 `model(input_ids=prefix)` 的墙钟（含 logits 计算） |
| decode | 连续生成 **32 个 token**，取 `ms/token` 平均（**每个 (batch, len) 组合都单独 warmup**，否则首个 step 会把 TTFT 的编译开销摊进来，造成 20× 的假象） |
| 环境变量 | `HF_ENDPOINT=https://hf-mirror.com`（HuggingFace 直连不可达，只有镜像可达） |

---

## 2. Prefill（TTFT 与输入处理吞吐）

| batch | input_len | **TTFT (ms)** | **prefill (tokens/s)** | 折合算力 (TFLOPS) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 23.06 | 5,550 | 5.5 |
| 4 | 128 | 22.97 | 22,287 | 22.0 |
| 8 | 128 | 23.55 | 43,476 | 43.0 |
| 1 | 512 | 23.01 | 22,250 | 22.0 |
| 4 | 512 | 33.70 | 60,780 | 60.0 |
| 8 | 512 | 53.50 | 76,554 | 75.6 |
| 1 | 2048 | 36.69 | 55,816 | 55.2 |
| 4 | 2048 | 108.56 | 75,458 | 74.6 |
| 8 | 2048 | 213.76 | **76,648** | **75.7** |

> 折合算力按 `FLOPs ≈ 2 × N_params × tokens`（$N$ = 0.494e9）估算，不含注意力 $L^2$ 项，
> 因此是**偏保守**的下界。

**观察：**

1. **TTFT 有一个 ~23 ms 的固定地板。** batch=1 时 128 / 512 的 TTFT 都是 ~23 ms，
   batch=4 和 8 在 L=128 时也是 23 ms —— 与输入长度、与 batch 都无关。
   而这 128 个 token 的实际计算只需 `126.5 GFLOP / 75 TFLOPS ≈ 1.7 ms`
   → **13 倍的开销花在「把 ~1836 个算子下发一遍」上**（与 decode 的病因完全相同）。
2. **边际 prefill 速率远高于表观值。** 用同一 batch 下 L=128→2048 的差值反推：
   - b1：$(36.69-23.06)\ \text{ms} / 1920\ \text{tok}$ → **8.91 µs/token = 112 k tokens/s**
   - b4：$(108.56-22.97)\ \text{ms} / 7680\ \text{tok}$ → **11.14 µs/token = 90 k tokens/s**
   - b8：$(213.76-23.55)\ \text{ms} / 15360\ \text{tok}$ → **12.38 µs/token = 81 k tokens/s**

   → 扣掉那 23 ms 固定开销后的**真实 prefill 能力是 81~112 k tokens/s**，
   表观吞吐只是在长序列 + 大 batch 时才逼近它。
3. **prefill 的算力天花板 ≈ 75.7 TFLOPS = BF16 峰值的 32.5%。**
   对 0.5B 这种小模型，注意力占比小、GEMM 的 M 维（= token 数）能压满 XMX，
   所以能拿到 1/3 的峰值 —— 这是本机上**唯一**一个吃算力吃到大头的工作负载。

---

## 3. Decode（生成吞吐）—— 本报告的核心发现

| batch | input_len | **decode (tokens/s)** | **ms/token** | 等效权重搬运带宽 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 45.7 | 21.89 | 45.1 GB/s |
| 4 | 128 | 182.2 | 21.95 | 45.0 GB/s |
| 8 | 128 | **364.9** | 21.93 | 45.1 GB/s |
| 1 | 512 | 46.8 | 21.39 | 46.2 GB/s |
| 4 | 512 | 183.3 | 21.83 | 45.3 GB/s |
| 8 | 512 | 369.2 | 21.67 | 45.6 GB/s |
| 1 | 2048 | 46.9 | **21.34** | 46.3 GB/s |
| 4 | 2048 | 185.5 | 21.56 | 45.8 GB/s |
| 8 | 2048 | **370.1** | 21.61 | 45.7 GB/s |

**关键事实：`ms/token` 在全部 9 个配置里都是 21.3~22.0 ms，极差只有 3%。**
decode 时间**既不随 batch 变化，也不随上下文长度变化**。

### 3.1 与带宽 roofline 的差距

| 项 | 值 |
|---|---|
| 每步必须流过的权重 | **0.988 GB**（lm_head/embedding 共享，只算一次） |
| 本机实测 HBM 峰值带宽 | **797 GB/s**（见 `docs/TODO/03-memory-bandwidth.md`） |
| 带宽 roofline | $0.988 / 0.797 = $ **1.240 ms** |
| 实测 | **21.6 ms** |
| **差距** | **17.4×** |
| 等效实际带宽 | **45.7 GB/s = 峰值的 5.7%** |

> batch=16 的补充测量（探针）：22.32 ms/token → 716.8 tok/s。**仍然是 22 ms/步**。

### 3.2 为什么是 21.6 ms —— 用实测的「每算子固定代价」解释

**（a）每步的算子数（实测）**

用 `TorchDispatchMode` 在 eager 路径下计数（注意：**不能用它去数编译后的图**，
Dynamo 的缓存查询也会穿过 dispatch 层，数字会虚高几十倍）：

| 类别 | 算子数/步 | 占比 |
|---|---:|---:|
| **纯元数据**（`view` 339 / `t` 169 / `transpose` 97 / `_unsafe_view` 98 / `slice` 96 / `unsqueeze` 52 / `expand` 3） | **854** | **46.6%** |
| **elementwise**（`mul` 220 / `add` 146 / `pow` 49 / `mean` 49 / `rsqrt` 49 / `neg` 48 / `silu` 24） | **585** | **31.9%** |
| **GEMM**（`mm` 97 + `addmm` 72 + `bmm` 1） | 170 | 9.3% |
| 其他（`_to_copy` 101 / `cat` 97 / `embedding` 1 / `arange` 1） | 200 | 10.9% |
| 注意力（融合 `micro_sdpa`） | 24 | 1.3% |
| **合计** | **1833** | 100% |

（同一模型另一轮计数为 1909，同量级。按「元数据 + elementwise」合并则 **77%**。）

> 注意 `cat` 有 97 次 —— 这是 `DynamicCache` 每层拼接 KV 的开销（24 层 × 4 次）。
> `pow`/`mean`/`rsqrt`/`neg` 各 ~49 次 —— 这是 **LayerNorm 在 eager 下被拆成 4~6 个算子**
> （24 层 × 2 个 norm ≈ 48）。这两项合计 ~14% 的算子数是**框架实现方式**造成的，不是模型需要的。

**（b）单个算子的独立实测代价**

| 算子 | 数据量 | 主机入队 | GPU 事件 |
|---|---:|---:|---:|
| `add_` | 1 个元素 | 6.68 µs | 6.49 µs |
| `mul+add` | 896 个元素 | — | 15.0 µs（两个算子） |
| `softmax` | (1,14,1,129) | — | 13.4 µs |
| **纯元数据算子**（`view`/`unsqueeze`/`transpose`/`slice`） | — | **0.58~0.77 µs** | — |

→ 一个**需要下发 kernel** 的算子，地板是 **6.5 µs**；带 oneDNN primitive 的算子约 **20 µs**。
**纯 view/元数据算子只要 0.6~0.8 µs**，所以「减少 view」的收益有限（854 个 × 0.7 µs ≈ **0.6 ms**，
只占 21.6 ms 的 3%）—— **真正的成本在剩下 ~980 个要下发 kernel 的算子上**。

**（c）决定性证据：耗时与数据量无关**

`diagnostics/m1_gemm_scan.py` 在 Qwen2.5-0.5B 的真实形状上扫 M：

| 形状 | k | n | 权重 | **m=1 实测** | 带宽地板 @800GB/s | 达成率 |
|---|---:|---:|---:|---:|---:|---:|
| attn `qkv_proj` | 896 | 1152 | 2.1 MB | **22.8 µs**（0.090 TFLOPS） | 2.6 µs | **11%** |
| attn `o_proj` | 896 | 896 | 1.6 MB | **20.7 µs**（0.078 TFLOPS） | 2.0 µs | **10%** |
| mlp `gate_up` | 896 | 9728 | 17.4 MB | **20.9 µs**（0.836 TFLOPS） | 21.8 µs | **96%** |
| mlp `down` | 4864 | 896 | 8.7 MB | **24.6 µs**（0.354 TFLOPS） | 10.9 µs | 44% |
| `lm_head` | 896 | 151936 | 272.3 MB | **491.3 µs**（0.554 TFLOPS） | 340 µs | 69% |

> **`qkv_proj` 只有 2.1 MB 权重却和 17.4 MB 的 `gate_up` 花掉一样的时间（22.8 vs 20.9 µs）。**
> 前者只需 2.6 µs 的带宽时间，后者需要 21.8 µs —— 说明这段时间**根本不是带宽**，
> 而是每算子固定的 ~21 µs 开销。

再看 M 维的影响（同一随 m 变化）：

| 形状 | m=1 | m=4 | m=8 | m=32 | m=512 |
|---|---:|---:|---:|---:|---:|
| `qkv_proj` | 22.8 µs | 21.0 | 21.1 | 21.2 | 21.0 |
| `o_proj` | 20.7 | 20.9 | 21.0 | 21.1 | 21.4 |
| `gate_up` | 20.9 | 22.2 | 22.0 | 21.6 | **70.6** |
| `down` | 24.6 | 23.6 | 23.4 | 23.9 | **55.5** |
| `lm_head` | 491.3 | 522.1 | 534.5 | 545.3 | **818.4** |

→ `qkv_proj` 和 `o_proj` 从 m=1 到 m=512（**512 倍 token**）**耗时完全不变**（22.8 → 21.0 µs）。
**这就是 decode 时间与 batch 无关的直接原因。**

**（d）成本预算闭合**

$$\underbrace{1833\ \text{ops}}_{\text{实测算子数}} \times \underbrace{11.8\ \mu s}_{21.6\ \text{ms} / 1833} = 21.6\ \text{ms} = \text{实测值}$$

同一口径的独立脚本（`diagnostics/decode_budget.py`）给出 $1909 \times 13\ \mu s = 24.8$ ms
vs 实测 24.53 ms，**比值 0.99×**。

**（e）profiling 交叉验证**

`torch.profiler`（`ProfilerActivity.CPU + XPU`）显示单步 device 时间合计 **23.4 ms**
（该合计在 op 层与 kernel 层之间有重复计数，仅作量级参考），与墙钟同量级；其中：

| 算子 | 次数/步 | device 合计 | 每次都耗时 |
|---|---:|---:|---:|
| `aten::mm` | 97 | 2329.1 µs | **24.0 µs** |
| `aten::mul` | 220 | 942.6 µs | **4.3 µs** |
| `aten::cat` | 97 | 666.1 µs | 6.9 µs |
| `aten::add` | 146 | 622.2 µs | **4.3 µs** |
| `aten::addmm` | 72 | 548.8 µs | 7.6 µs |
| `aten::mean` | 49 | 415.5 µs | 8.5 µs |
| `micro_sdpa` | 24 | 352.6 µs | 14.7 µs |
| `aten::silu` | 24 | 185.0 µs | 7.7 µs |
| `aten::copy_` | 101 | 1131.0 µs | 11.2 µs |

（另有 854 个元数据算子在 profile 中**不出现 device 时间** —— 印证它们不产生 kernel。）

→ 结论方向不变：**这些「微小 kernel 的固定代价」就是全部成本**，
既不是 FLOPs，也不是 HBM 带宽。

### 3.3 为什么 `torch.compile` 救不了

`diagnostics/llm_compile_compare.py`（同一脚本内对比，batch=1, prompt=128, 生成 24 token）：

| 配置 | TTFT | decode ms/token | decode tok/s | 相对 eager |
|---|---:|---:|---:|---:|
| eager | 816.2 ms¹ | **25.649** | 39.0 | 1.00× |
| `torch.compile(mode="default")` | 12.74 ms | **21.462** | 46.6 | **1.20×** |
| `torch.compile(mode="reduce-overhead")` | — | 25.441 | 39.3 | 0.99× |

¹ eager 的 TTFT 包含首次 JIT/显存分配，不代表稳态；稳态 TTFT 见 §2（~23 ms）。

- 1.20× 是**真实但远不够**的收益。按 §3.2 的算子分布，如果能真正融合掉那
  854 个 view + 585 个 elementwise 算子，理论上应该拿到 3~4×。
- `mode="reduce-overhead"`（cudagraph）**无收益**：KV cache 每步都在增长，
  静态 shape 假设失效，每步都要重新捕获。这与 `torch.compile` 在训练（§[01](./01-training-throughput.md) §2.4）
  中的表现一致 —— **本机上 `torch.compile` 的收益主要来自「布局修正」，而不是算子融合**。

### 3.4 batch 是有用的 —— 但只能摊薄，不能加速

| batch | ms/step | tokens/s | 相对 b1 |
|---:|---:|---:|---:|
| 1 | 22.267 | 44.9 | 1.0× |
| 2 | 22.768 | 87.8 | 2.0× |
| 4 | 22.519 | 177.6 | 4.0× |
| 8 | 22.499 | 355.6 | 7.9× |
| 16 | 22.321 | **716.8** | **16.0×** |

**步耗时恒定在 22.3 ms → 吞吐线性放大 16 倍。** 这是唯一可用的优化手段：
**decode 一定要做 batching（continuous batching / 静态 batch 都行）。**

外推：`gate_up` 在 m=512 时涨到 70.6 µs（3.4×），`lm_head` 涨到 818 µs（1.7×），
而 token 数涨 512 倍 → **到 batch=512 时每步成本仍远低于 512 倍**，
说明本机在 decode 上还有巨大的 batch 空间（瓶颈会从「固定开销」转移到「单次 GEMM 的效率」）。

> ⚠️ **反过来说：量化（W4A16 / INT8）对 decode 基本无益。**
> 本机实测 W8A16 比 bf16 慢 30 倍、W4A16 峰值只有稠密 INT8 的 0.11×
> （见 [`../../precision-support.md`](../../precision-support.md)）。
> 而 decode 的瓶颈也不是权重带宽，所以「减半权重」并不能换来 2 倍速度。

---

## 4. 显存占用模型

### 4.1 实测峰值

| batch | input_len | 峰值显存 |
|---:|---:|---:|
| 1 | 128 | 0.97 GiB |
| 4 | 128 | 1.08 GiB |
| 8 | 128 | 1.23 GiB |
| 1 | 512 | 1.08 GiB |
| 4 | 512 | 1.53 GiB |
| 8 | 512 | 2.14 GiB |
| 1 | 2048 | 1.53 GiB |
| 4 | 2048 | 3.35 GiB |
| 8 | 2048 | **5.78 GiB** |

### 4.2 预测公式（8/9 点误差 < 1%）

$$\text{peak} \approx \underbrace{W}_{\text{0.92 GiB}} + \underbrace{B \cdot L \cdot V \cdot 2\ \text{B}}_{\textbf{logits 张量}} + \underbrace{B \cdot L \cdot 12\ \text{KiB}}_{\text{KV cache}}$$

其中 $V = 151936$；KV cache 每 token 12 KiB $= 2\,(\text{K,V}) \times 2\,(\text{KV heads}) \times 64\,(\text{head\_dim}) \times 24\,(\text{layers}) \times 2\ \text{B}$。

| batch | L | 权重 | logits | KV cache | 预测 | 实测 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.92 | 0.036 | 0.002 | 0.958 | 0.97 ✅ |
| 8 | 128 | 0.92 | 0.290 | 0.012 | 1.222 | 1.23 ✅ |
| 1 | 512 | 0.92 | 0.145 | 0.006 | 1.071 | 1.08 ✅ |
| 8 | 512 | 0.92 | 1.159 | 0.047 | 2.126 | 2.14 ✅ |
| 1 | 2048 | 0.92 | 0.580 | 0.023 | 1.523 | 1.53 ✅ |
| 4 | 2048 | 0.92 | 2.319 | 0.094 | 3.333 | 3.35 ✅ |
| 8 | 2048 | 0.92 | **4.637** | 0.188 | 5.745 | 5.78 ✅ |

### 4.3 两个反直觉的结论

1. **KV cache 完全不是问题。** (8, 2048) 时它只有 **188 MiB**，
   而 logits 张量有 **4.64 GiB**（24 倍）。哪怕上下文涨到 32k × batch 8，
   KV cache 也只有 3 GiB。
2. **logits 张量才是显存杀手。** 因为 `model(...)` 会返回 **全部位置** 的 logits，
   形状 $(B, L, V)$。而真实推理只需要**最后一个位置**的一个 token。
   使用 `logits_to_keep=1`（或手动切片 `logits[:, -1:]`）可把 (8,2048) 的峰值
   从 5.78 GiB 直接降到 **1.13 GiB（-80%）**。
   → 这是本报告里**性价比最高的单条建议**。

---

## 5. 与训练侧结论的对照

| 负载 | 每步算子数 | 每步耗时 | **每算子平摊** | 瓶颈归属 |
|---|---:|---:|---:|---|
| ResNet-50 bf16 训练（b256） | 745 | 211.2 ms | **283 µs** | 算力 / 单算子效率 |
| BERT-base bf16 训练（seq512,b32） | — | — | — | 算力 |
| LLM **prefill**（b8, L2048） | 1836 | 213.8 ms | **116 µs** | 算力（75.7 TFLOPS） |
| LLM **decode**（任意 batch） | 1836 | 21.6 ms | **11.8 µs** | **每算子固定开销** |

**判据：`每步耗时 / 算子数` 接近 6~20 µs 时，程序进入固定开销受限区。**
prefill 因为 M 维大（2048），一个 token 能摊到 116 µs → 算力受限；
decode 因为 M 维只有 1~8，每个算子只能摊到 11.8 µs → **纯开销受限**。

**这是 LLM 推理最典型的「decode 悖论」**，本机把它放大到了极致（17× 离 roofline）。

---

## 6. 建议

| 优先级 | 措施 | 预期 |
|---|---|---|
| ⭐⭐⭐ | **decode 必须做 batching**（continuous batching）。实测 b1→b16 线性 16× | 44.9 → 716.8 tok/s |
| ⭐⭐⭐ | **`logits_to_keep=1`**（或切片 logits），不要保留全序列 logits | (8,2048) 峰值 5.78 → 1.13 GiB |
| ⭐⭐ | 用 **IPEX / vLLM-XPU / OpenVINO** 替代原生 Transformers eager 路径 | 目标是把每步 ~1836 个算子降下来；`torch.compile` 只能给 1.20× |
| ⭐ | prefill 已经不错（75.7 TFLOPS = 32.5% 峰值），长上下文 + 大 batch 是它的甜点 | 无需改进 |
| ❌ | **不要为 decode 做 FP8/FP4/W4A16 量化** | decode 不带宽受限；且本机 W8A16 慢 30 倍 |
| ❌ | 不要指望 `torch.compile(mode="reduce-overhead")` | 实测 0.99×（KV cache 动态增长，cudagraph 无法复用） |

---

## 7. 复现命令

```bash
cd /root/workspace/benchmark/05-ai-dl
PY=/root/workspace/venv1/bin/python

# 主表（9 点）
env -u LD_LIBRARY_PATH HF_ENDPOINT=https://hf-mirror.com $PY run_bench.py --suite llm

# 定向探针
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/decode_budget.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/m1_gemm_scan.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/llm_compile_compare.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/launch_floor.py
```

---

## 8. 未做 / 待补充

| 项 | 状态 | 说明 |
|---|---|---|
| 更大模型（7B/8B） | ⬜ 未做 | 需下载 ~15 GB 权重；本机 41 GiB 可用内存 + 磁盘 91% 占用，风险高。**理论上大模型的 decode 会从「固定开销受限」转向「带宽受限」**（单个 GEMM 权重 > ~10 MB 时即达到带宽墙），届时结论会不同 |
| vLLM-XPU / IPEX-LLM | ⬜ 未做 | 未安装；本报告用原生 Transformers，性能不代表最优栈 |
| PagedAttention | ⬜ 未做 | §4.3 已说明 KV cache 不是瓶颈，优先级低 |
| 长上下文（4096+） | ⬜ 未做 | 受 logits 张量显存限制（L=4096,b=8 时 logits 需 9.3 GiB），建议先做 `logits_to_keep=1` |
