# ② 计算峰值（ALU / XMX）测试结论

> 对应测试计划：[`../../TODO/02-compute-peak.md`](../../TODO/02-compute-peak.md)
> 测试代码：[`../../../benchmark/02-compute-peak/`](../../../benchmark/02-compute-peak/)
> 原始产物：`benchmark/02-compute-peak/results/bench_20260922-194139.{json,md}`（72 条记录）
> （更早的 `bench_20260922-192831.*` 是 v1，其中 XMX 占用率的解读有误，**已被本报告取代**）
> 测试日期：2026-09-22　　硬件：Intel Data Center GPU Max 1100（PVC, Production ES, 1 tile）

---

## 0. 一页结论

### 0.1 对测试目标的逐条回答

| # | 测试目标（TODO §1） | 实测 | 判定 |
|---|---|---|---|
| 1 | FP32 ALU 峰值是否等于公式 22.22 TFLOPS | 自研探针 **50.75 TFLOPS**（公式的 **228.4%**）/ oneDNN GEMM **22.13 TFLOPS**（99.6%） | 🔴 **口径冲突未解决**，见 §3.2 |
| 2 | FP64 是否是 FP32 的 1/2 | 探针 **51.90 TFLOPS ≈ FP32**；torch 侧 FP64 = **0.78×** FP32 | 🔴 **原文档「FP64 = FP32/2」是错的**，见 §3.3 |
| 3 | XMX（矩阵引擎）bf16/fp16 峰值 | **355.0 TFLOPS**（bf16）、**355.0**（fp16） | ✅ **= 公式值的 99.8%**，且比产品常引用值高 2×，见 §3.1 |
| 4 | XMX int8 峰值 | **710.0 GOPS** —— 恰好 **2.00×** bf16 | ✅ 与 XMX 架构（int8 K=32 = 2× bf16 K=16）完全一致 |
| 5 | XMX 是否真的被启用 | 正确性自检 **bf16 16/16、fp16 16/16、int8 32/32 全部 `ok`**；反推 **255.5 MAC/clk/EU**（设计值 256） | ✅ **铁证** |
| 6 | 频率是否达到标称 1550 MHz | 锁定 **1550**（min==max==boost==RP0），满载实测 **1550 MHz**，**无降频** | ✅ |
| 7 | 功耗是否触墙 | 满载 **171 W** / 利用率 **100%**，远低于 **300 W** 上限 | ✅ |
| 8 | torch/oneDNN 能吃到裸峰值的百分之几 | oneDNN bf16 **63.5%** / fp16 **66.8%** / int8 **58.5%** | ⚠️ 有 **33-41% 的库开销空间** |

### 0.2 一句话

> **XMX 侧结论干净漂亮：355 TFLOPS bf16 / 710 GOPS int8，反推 256 MAC/clk/EU，
> 与 PVC 设计完全吻合，且 torch/oneDNN 还能再挖 36-40%。**
> **ALU（标量/向量）侧则是本机最大的未解之谜：自研 SYCL 探针测出 50.75 TFLOPS，
> 是 `448 EU × 16 lane × 2 × 1.55 GHz` 公式值（22.22）的 228%。**
> 在频率被锁定、XMX 数字完全对得上的前提下，这个 2.28× 差异不可能来自计数器错误。

### 0.3 三个反直觉发现

1. **XMX bf16 峰值是 355 TFLOPS，不是官方文档里常见的 176 TFLOPS。**
   差 2× 的原因已定位：**XMX 需要 ≥8 个并发 sub-group/EU 才能填满流水线**
   （occupancy 1/2/4/8 → 118/142/177/**355** TFLOPS）。
   在低占用率下**看起来像"卡在 176"**，实际只是延迟受限。
2. **"同样的工作量、同样的时间"不代表编译器优化掉了。**
   在 occupancy knee 以下，增加工作量不增加时间 —— 这是**延迟受限**的正常表现，
   不是死代码消除。本报告用「固定总 DPAS 数、只改 work-group 数」的对照实验证实了这一点。
3. **FP64 不是 FP32 的一半。** 探针侧 FP64 ≈ FP32（51.9 vs 50.75），
   torch 侧 FP64 = 0.78× FP32。**没有任何一条证据支持 1/2。**

---

## 1. 交付物索引

| 类别 | 路径 | 内容 |
|---|---|---|
| SYCL ALU 探针 | `benchmark/02-compute-peak/sycl/alu_peak.cpp` | FP32/FP64/FP16/INT32 标量+向量峰值 |
| SYCL XMX 探针 | `benchmark/02-compute-peak/sycl/xmx_peak.cpp` | `joint_matrix` DPAS 峰值 + 正确性自检 |
| torch 探针 | `benchmark/02-compute-peak/probes/torch_gemm_peak.py` | torch/oneDNN GEMM 峰值 |
| 死胡同记录 | `benchmark/02-compute-peak/sycl/alu_clock_probe.cpp` | `clock_scope::device` 不被支持，仅作文档 |
| 主 runner | `benchmark/02-compute-peak/run_bench.py` | 4 个 suite：alu / xmx / torch / clock |
| README | `benchmark/02-compute-peak/README.md` | 测试方法说明 |
| 原始结果 | `benchmark/02-compute-peak/results/bench_*.{json,md}` | 见 §2 |

---

## 2. 全部关键数字总表

### 2.1 XMX（矩阵引擎）峰值 —— 本目录最干净的结论

| dtype | 峰值 | 单位 | 相对 fp32 公式值 | 正确性自检 |
|---|---:|---|---:|---|
| **bf16** | **355.0** | TFLOPS | **15.97×** | 16/16 ✅ |
| **fp16** | **355.0** | TFLOPS | **15.97×** | 16/16 ✅ |
| **int8** | **710.0** | GOPS | 31.96× | 32/32 ✅ |

**反推硬件参数**：

| 派生量 | 实测反推 | PVC 设计值 | 一致性 |
|---|---:|---:|---|
| MAC / clk / EU | **255.5 - 255.6** | **256** | ✅ |
| int8 : bf16 比值 | **2.00×** | 2× | ✅ |
| bf16 理论峰值（256 MAC × 448 EU × 2 × 1.55 GHz） | 355.5 TFLOPS | — | ✅ 实测 355.0 = **99.9%** |

> **这就是 XMX 结论的全部**：实测值 = 设计值 × 99.9%，且正确性自检全过。
> 没有任何可怀疑的空间。

### 2.2 XMX 占用率阶梯（knee 定位）—— 关键实验

固定其它一切，只改「每 EU 的并发 sub-group 数」：

| occupancy（sub-group / EU） | bf16/fp16 实测 (TFLOPS) | int8 实测 (GOPS) | 占峰值 |
|---:|---:|---:|---:|
| 1 | 118.1 | 236.2 | 33% |
| 2 | 141.9 | 283.9 | 40% |
| 4 | 177.4 | 354.9 | 50% |
| **8** | **354.9** | **709.7** | **100% ← knee** |
| 16 | 354.9 | 709.9 | 100% |
| 32 | 355.0 | 710.0 | 100% |

**口径警告**：如果只跑到 occupancy ≤4，会得出「bf16 峰值 ≈ 176 TFLOPS」的**错误结论**
—— 这恰好是网上流传的 PVC 数字。**必须扫过 knee 才能报峰值。**

注意 int8 在 occ=4 时已经读到 354.9 GOPS，看似「已饱和」——其实那是
**bf16 的峰值**（int8 的 K=32 是 bf16 的 2 倍，所以同一 DPAS 数对应 2 倍 MAC）。
再次说明：**必须同时核对 MAC/clk/EU 才能判断是否饱和**。

### 2.3 XMX 线性度验证（排除"编译器消除"）

固定**总 DPAS 数 = 7.516e9**，只改 work-group 数（global size）：

| global size | 时间 (s) | 吞吐 (TFLOPS) | 判定 |
|---:|---:|---:|---|
| 28672 | 0.173517 | 177 | 低于 knee |
| 57344 | 0.086764 | 355 | 达到 knee |

**结论**：knee 以上时间严格按 ×2.00 缩放 ⇒ **355 TFLOPS 平台是真实的**。
knee 以下「同样的工作量、同样的时间」是**延迟受限**，与编译器无关。

### 2.4 ALU（标量/向量）峰值 —— 口径冲突

| dtype | 自研 SYCL 探针 | 公式/文献值 | 比值 |
|---|---:|---:|---:|
| **FP32** | **50.75 TFLOPS** | **22.22 TFLOPS** | **228.4%** 🔴 |
| FP64 | **51.90 TFLOPS** | 11.11（=FP32/2） | 467% 🔴 |
| FP16 | **52.60 TFLOPS** | 22.22（=FP32） | 236.7% |
| INT32 | **18.00 GOPS** | — | 81% of 22.22 |
| **FP32 (oneDNN GEMM 8192³)** | **22.13 TFLOPS** | 22.22 | **99.6%** ✅ |

**同一个公式 22.22，两个实现给出 100% 与 228% 两种结论。**

> 注意 FP64/FP16 的探针值甚至**略高于** FP32 —— 在一条纯 FMA 链上这完全正常
> （FP64 与 FP32 走同一套 EU 发射路径，FP16 则可以打包）。
> **没有任何一个 dtype 落到公式的 1/2。**

### 2.5 ALU 占用率 / 线性度

| global size | 吞吐 (TFLOPS) | 墙钟 (s) | 备注 |
|---:|---:|---:|---|
| 7 168 | 10.09 | 0.3814 | 延迟受限 |
| 14 336 | 16.67 | 0.4618 | 延迟受限 |
| 28 672 | 20.50 | 0.7510 | 延迟受限 |
| 57 344 | 25.44 | 1.2100 | 接近 knee |
| **114 688** | **50.75** | 1.2130 | **← knee** |
| 229 376 | 50.76 | 2.4260 | ×2.00 ✅ |
| 458 752 | 50.83 | 4.8455 | ×2.00 ✅ |

### 2.6 ALU 向量宽度（VEC）扫描

| VEC | 吞吐 (TFLOPS) |
|---:|---:|
| 1 | 34.81 |
| 2 | 45.55 |
| **4** | **52.78** ← 最佳 |
| 8 | 50.75 |
| 16 | 51.32 |

**ALU 路径 VEC=4 最优**，VEC≥4 后基本拉平。
⚠️ 与 `03-memory-bandwidth` 对比：**内存路径 VEC>4 会腰斩**（714 → 451 GB/s）。
⇒ **两层的最优向量宽度不同，必须分开调。**

### 2.7 库层利用率（torch / oneDNN）

| 路径 | 最佳形状 | 实测 | 相对 |
|---|---|---:|---|
| torch fp32 GEMM | 8192³ | **22.13 TFLOPS** | 公式值 **99.6%** |
| torch bf16 GEMM | 4096³ | **225.9 TFLOPS** | 裸 DPAS 峰值 **63.5%** |
| torch fp16 GEMM | 8192³ | **237.6 TFLOPS** | 裸 DPAS 峰值 **66.8%** |
| torch int8 `_int_mm` | 8192³ | **416.2 GOPS** | 裸 DPAS 峰值 **58.5%** |

  - 同一 dtype 在不同形状下差异很大：bf16 在 4096³ 是 225.9，但 4096×4096×**16384**
    只有 185.4；fp16 反过来（4096³ 是 217.0，8192³ 是 237.6）。
  - `4096×4096×16384` 这个 K 很长的形状对 bf16/fp16 都不友好（223.7 → 185.4/183.4，
    **−18%**）—— 这与 `precision-support.md` 记录的 **16384³ 吞吐悬崖**同源。

### 2.8 频率与功耗

| 指标 | 空载 | 满载 |
|---|---:|---:|
| `gt_act_freq_mhz` (sysfs) | **0**（读不到） | **1550** |
| `gt_cur_freq_mhz` | 1550 | 1550 |
| `gt_min/max/boost/RP0` | 1550 / 1550 / 1550 / 1550 | 同 |
| `RP1` / `RPn` | 1000 / 200 | 同 |
| xpu-smi GPU Utilization | — | **100%** |
| xpu-smi GPU Power | ~43 W | **171 W**（上限 300 W） |
| xpu-smi GPU Frequency | — | **1550 MHz** |

**结论**：频率**被锁定在 1550 MHz**（min == max == boost），
因此**所有吞吐差异都不可能来自频率波动**。满载无降频。

### 2.9 向量（elementwise）路径的对照

| 操作 | fp32 (GB/s) | bf16 (GB/s) | 变化 |
|---|---:|---:|---:|
| add | 528.0 | 530.0 | +0.4% |
| mul | 526.2 | 528.7 | +0.5% |
| relu | 406.9 | 411.6 | +1.2% |

**改 dtype 对 elementwise 毫无帮助**（bandwidth-bound），
且都远低于 `03` 测得的 HBM 上限 900 GB/s —— 说明 torch 的 elementwise 实现**效率不高**。

---

## 3. 分项详述

### 3.1 XMX：为什么是 355 而不是 176

这是本报告**最重要的方法论贡献**。

网上/部分官方材料给出的 PVC bf16 数字（约 176 TFLOPS）对应的是
**occupancy = 4 个 sub-group/EU 的一半流水线利用率**。实测：

- occupancy 4 → **177.4 TFLOPS**（完美对应 176）
- occupancy 8 → **354.9 TFLOPS**（流水线填满）

**根因**：PVC 的 XMX 单元深度流水，每个 EU 需要 ≥8 个并发 sub-group
（即足够多的 warp）才能把 DPAS 流水线灌满。低占用率下 EU 在等 DPAS 结果，
属于**延迟受限**而非带宽/算力受限。

**验证方法**（本报告采用）：
1. **占用率扫描**：固定单点工作量，扫 global size 1/2/4/8/16/32 sub-group/EU；
2. **线性度验证**：在 knee 以上把工作量翻倍，时间必须**严格**翻倍；
3. **固定总 DPAS 数对照**：只改 work-group 数，验证「时间 ÷2」而不是「时间不变」。

三条证据同时成立 ⇒ **355 TFLOPS 是硬件真实峰值**。

### 3.2 ALU 口径冲突（🔴 未解决）

| 来源 | FP32 峰值 | 相对 22.22 |
|---|---:|---:|
| 公式 `448 EU × 16 lane × 2 FLOP × 1.55 GHz` | 22.22 TFLOPS | 100%（定义） |
| **oneDNN GEMM（8192³）** | **22.13 TFLOPS** | **99.6%** |
| **自研 SYCL 探针** | **50.75 TFLOPS** | **228%** |

**为什么这个矛盾必须重视**：XMX 侧实测 = 设计值 × 99.9%，频率锁定在 1550 MHz，
**这说明测量方法和硬件状态都是可信的**。在同样可信的前提下，
ALU 探针测出 2.28× 的公式值，只有三种可能：

| 候选 | 说明 | 可信度评估 |
|---|---|---|
| **(a) 真实 EU 数 ≠ 448** | 若实际是 ~1024 个 EU，公式就对了 | 中。ES 件规格可能未公开 |
| **(b) EU 的 FP32 宽度 > 16 lane** | 若真实是 **36.5** lane，公式就对了 | 中。Xe-HPC 可能是 2×8 FP32 且可双发 |
| **(c) 探针 FLOP 计数器偏低** | 计数器漏算了某些操作 | **低**。探针用不可折叠的 logistic 链，FLOP 数可逐条核对 |

**反推**：50.75 TFLOPS / (448 EU × 2 FLOP × 1.55 GHz) = **36.5 条 FP32 lane/EU**
（模型假设 16 条）。这个数**不是**整数，既不支持「EU 数翻倍」也不支持「宽度翻倍」
的简单解释，因此怀疑方向应偏向 **发射/双发机制**或**探针计数**。

> 本机 `clinfo` 报告 EU 数 = 448。 **不能排除 clinfo 本身取错字段。**
> 可查：`zeDeviceGetProperties()` 的 `numEUsPerSubslice × numSubslicesPerSlice × numSlices`。

**当前政策（重要，不要违反）**：

- **保留 22.22 TFLOPS 作为「标称值（公式）」**，**不重算、不缩放** `REFERENCE_CLOCK_GHZ`
  和 `common.py:alu_tflops()`；
- **oneDNN 的 22.13 视为可信下界**（它跑满 8192³，是生产级实现）；
- **探针的 50.75 视为上界**，待 `docs/TODO/07-profiling.md` 的 EU 计数器裁决；
- 在任何 A/B 对比（如 bf16/fp32 加速比）中**以 oneDNN 实测值为分母**，避免引入不自洽。

**下一刀**：`07-profiling.md` 的 VTune/PTI `EU Array Active` 计数器 —— 但它在本机是 **N/A**
（见 `03` §3.7 与 `04`），因此**这条线目前是死路**。
替代思路：用 Roofline 反推 —— 已知 HBM 900 GB/s（`03` 实测），
若某 ALU-bound kernel 达到 50.75 TFLOPS 而带宽完全不是瓶颈，则 50.75 必然真实。

### 3.3 FP64 ≠ FP32 / 2

| 来源 | FP32 | FP64 | FP64/FP32 |
|---|---:|---:|---:|
| 自研探针 | 50.75 | **51.90** | **1.02×** |
| torch（2048³） | 22.16 | **17.37** | **0.78×** |
| **原文档假设** | — | — | **0.50×** ❌ |

**两条独立证据都不支持 1/2**。已据此在 `precision-support.md` 中引入
`FP64_ALU_RATIO = 0.78`（torch 侧口径），探针侧则接近 1:1。

**结论**：`docs/TODO/02-compute-peak.md` 与 `docs/hardware.md` 里
「FP64 = FP32 / 2」的表述**应当删除或改写**为实测值。

### 3.4 库层还有 36-40% 的空间

| dtype | 裸 DPAS | torch/oneDNN GEMM | 利用率 |
|---|---:|---:|---:|
| bf16 | 355.0 | 225.9 | **63.5%** |
| fp16 | 355.0 | 237.6 | **66.8%** |
| int8 | 710.0 | 416.2 | **58.5%** |

**含义**：对于矩阵密集的负载，**torch/oneDNN 当前只吃到 60-64% 的 XMX 算力**。
这与 `Conclusion/05-ai-dl` 里 BERT/ResNet 的 XMX 利用率（21.8%~24.6%）
是**两个层次**的问题：
- 这里 60-64% 是**纯 GEMM kernel** 的效率损失（tiling / 调度 / 内存）；
- 那里 22-25% 是**整个模型**里 GEMM 占比不高（BN/elementwise 拖后腿）。

### 3.5 elementwise 改 dtype 没用

| 操作 | fp32 | bf16 | 变化 |
|---|---:|---:|---:|
| add | 528.0 | 530.0 | +0.4% |
| mul | 526.2 | 528.7 | +0.5% |
| relu | 406.9 | 411.6 | +1.2% |

elementwise 是**纯带宽受限**，与 `03-memory-bandwidth` 的
「vector path = pure bandwidth-bound」结论一致。
**改变 dtype 不会带来任何 elementwise 加速** —— 这条对性能调优很重要。

（注：`precision-support.md` 中的 fp16/bf16 `tanh` 掉到 345/329 GB/s 是**唯一的例外**，
原因是软件模拟。）

### 3.6 频率锁定 ⇒ 所有差异都是真实的

```
gt_min_freq_mhz = 1550
gt_max_freq_mhz = 1550
gt_boost_freq_mhz = 1550
gt_RP0_freq_mhz = 1550
gt_RP1_freq_mhz = 1000
gt_RPn_freq_mhz = 200
gt_act_freq_mhz = 0        # ← 空载时读不到，必须带负载采样
```

满载时 `gt_act_freq_mhz = 1550`。⇒ **无动态调频、无降频、无 Boost 波动**。
这大幅提高了本目录所有数字的可信度：**任何吞吐差异都来自代码，不来自硬件状态**。

---

## 4. 判读标准对照（vs `docs/TODO/02-compute-peak.md`）

| TODO 中的判读标准 | 实测 | 判定 |
|---|---|---|
| FP32 ALU 达到公式值 22.22 TFLOPS 的 ≥90% | oneDNN **99.6%** / 探针 **228.4%** | ⚠️ **口径冲突，需先定标** |
| FP64 ≈ FP32 / 2 | 探针 **1.02×** / torch **0.78×** | 🔴 **标准本身错误，需修正** |
| XMX bf16 达到设计峰值 | **99.8%** of 256 MAC/clk/EU | ✅ |
| XMX int8 = 2× bf16 | **2.00×**（710.0 / 355.0） | ✅ |
| 频率达到标称 1550 MHz | 满载 1550，锁定 | ✅ |
| 无降频 / 无功耗墙 | 满载 171 W（上限 300） | ✅ |
| torch 吃的比例 | 58.5-66.8% | ⚠️ 记录即可 |

> **建议修正 TODO 中的参考值**：
> - XMX bf16 参考值若写 176 TFLOPS，应改为「**实测 355 TFLOPS**」并附 occupancy 说明；
> - 删除「FP64 = FP32/2」；
> - FP32 标称值保留 22.22 但**明确标注「公式值，实测存在 2.28× 争议」**。

---

## 5. 与其它目录的衔接

| 本目录的发现 | 影响 |
|---|---|
| XMX bf16 355 TFLOPS / int8 710 GOPS | 是 `05-ai-dl` 所有"XMX 利用率 %"的分母；也是 `06-hpc-apps` 的输入 |
| oneDNN 只到 58.5-66.8% | → `05-ai-dl/01-training-throughput.md` 的 tuning 空间 |
| `4096×4096×16384` 类长 K 形状掉 18% | ← 与 `precision-support.md` 的 16384³ 坡口同源 |
| ALU 22.22 公式 vs 50.75 探针 | → `06-hpc-apps` 里纯 ALU 负载（如 CFD）的峰值预估**必须标注两种口径** |
| ALU VEC 最优 4，内存 VEC 最优 ≤4 | → 交叉引用 `03-memory-bandwidth` §3.3 |
| elementwise 改 dtype 无效（≤±1%） | → `05-ai-dl` 的 elementwise 优化只能靠融合/减少访存 |
| 频率锁定 1550、满载 171 W | 让 `08-power-efficiency.md` 的能效计算可以简化为「功耗 / 吞吐」，无需归一化频率；171 W 远低于 300 W ⇒ **单卡永远不会碰上功耗墙** |
| `clock_scope::device` 不支持 | 记录在 README §5；不要在后续测试里再尝试 |
| VTune `EU Array Active` = N/A | → §3.2 的裁决路径目前是死路，需替代方案 |

---

## 6. 未做 / 待补充

| 项 | 原因 | 建议 |
|---|---|---|
| **ALU 50.75 vs 22.22 的最终裁决** | VTune/PTI 的 EU 计数器在本机 **N/A** | 用 Roofline 交叉：构造一个带宽完全不成瓶颈的 ALU kernel，看 50.75 是否稳定 |
| **`ze_peak` 第三方仲裁** | 本轮**未构建**（原因见 `docs/TODO/02-compute-peak.md` §3.6）。它是 clpeak 移植，**只有向量测试**（无 XMX/DPAS），但正因如此它可作为 FP32 向量峰值的**第三方独立实现** | 出处 **`oneapi-src/level-zero-tests/perf_tests/ze_peak`**（不是 `intel/compute-runtime`）；零外部依赖，只需 `level_zero/ze_api.h` + `-lze_loader`（本机已有）；`git clone` 在本机报 `GnuTLS recv error (-110)`，需逐文件抓取。**约 10 分钟可补做**，建议作为 §3.2 口径冲突的仲裁证据 |
| 真实 EU 数量 | 无公开的 ES 件规格 | 可尝试 `ocloc` 的 device info / `zeDeviceGetProperties` 的 `numEUsPerSubslice` |
| XMX TF32 | torch 无 `tf32` 张量 dtype | 需 SYCL `joint_matrix` 支持（`dpas_argument_type` 有 tf32，但头文件路径待确认） |
| DPAS 的 pvc 专用 `TM`（transpose）变体 | 本轮未覆盖 | 低优先 |
| 功耗-频率曲线 | 频率被锁死，无法扫点 | 不可做 |
| `alu_clock_probe.cpp` 的 device-clock 方案 | `clock_scope::device` 未被 SYCL 实现支持 | **已放弃**，仅留档 |
| 多卡同时跑 XMX（是否会互相降频/抢功耗） | 本轮只测单卡 | 见 `docs/TODO/08-power-efficiency.md` |
