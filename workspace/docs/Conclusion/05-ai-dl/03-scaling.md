# ⑤-3 多卡扩展性结论（2 卡 DDP / 集合通信）

> 对应测试计划：[`../../TODO/05-ai-dl.md`](../../TODO/05-ai-dl.md) §3.4（**核心项**）
> 测试工具：[`../../../benchmark/05-ai-dl/`](../../../benchmark/05-ai-dl/)（suite: `ddp`）+ `train_ddp.py`
> 定向探针：[`../../../benchmark/05-ai-dl/diagnostics/allreduce_probe.py`](../../../benchmark/05-ai-dl/diagnostics/allreduce_probe.py)
> 测试日期：2026-09-22　　硬件：2 × Intel Data Center GPU Max 1100（Xe Link XL24 直连）
> 原始数据：`results/bench_20260922-011301.json`、`results/ddp_raw/*.json`

---

## 0. 一页结论

| 问题 | 答案 |
|---|---|
| **2 卡扩展效率** | **93.75%**（batch=64）/ **95.88%**（batch=128）→ **良好**（判读阈值 ≥90%） |
| **纯 DDP 包装开销** | 单卡下包一层 DDP 就要 **+4.1%**（b64）/ **+1.9%**（b128）—— 这部分与互连无关 |
| **通信占比** | **6.28%**（b64）/ **3.95%**（b128），来自 `no_sync()` 差分法 |
| **"通信开销"里真正跨卡的比例** | 只有 **39%**（b64）/ **56%**（b128）；其余是 DDP 自身的分桶/同步簿记 |
| **Xe Link 能跑多快？** | allreduce 在 32~64 MiB 饱和于 **~80 GB/s**（= 标称 318 GB/s 的 **25%**）；小消息延迟地板 **~26 µs**；`barrier` **119 µs** |
| **综合判读** | ✅ 扩展性达标，**不是**通信瓶颈。提高 batch 还能进一步摊薄通信占比 |
| **注意** | `comm_pct` 6.28% 里 **~10 倍**于纯 allreduce 传输时间（0.64 ms @80 GB/s），说明瓶颈是**同步/分桶的串行部分**而非链路带宽 |

---

## 1. 测试口径

### 1.1 三个对照配置（关键方法论）

单跑「1 卡 vs 2 卡」会把 DDP 框架自身的开销误算进通信成本。本测试跑**三个**配置：

| 配置名 | 进程数 | DDP 包裹 | 同步探针 | 用途 |
|---|---:|---|---|---|
| `1card_nodpp` | 1 | ❌ | ❌ | **纯计算基线** |
| `1card_ddp` | 1 | ✅ | ✅ | **隔离 DDP 包装自身开销**（world_size=1，无跨卡通信） |
| `2card_ddp` | 2 | ✅ | ✅ | 真实 2 卡 |

- 扩展效率要给**两个**：相对 `1card_nodpp`（工程口径）与相对 `1card_ddp`（算法口径）。
- 模型：ResNet-50（25.6 M 参数 → bf16 梯度 **51.2 MB**），dtype = bf16，backend = `xccl`。
- `torchrun --standalone --nproc_per_node={1,2}`，warmup 5 步 + 计时 20 步，取中位数。
- 计时用 `torch.xpu.Event`，每个配置单独 `reset_peak_memory_stats`。

### 1.2 通信占比的测法：`no_sync()` 差分

$$\text{comm\_ms} = t_{\text{sync}} - t_{\text{no\_sync}}$$

在 `model.no_sync()` 上下文里跑同样步数，DDP 不做梯度 allreduce。
两者之差就是「**暴露出来**的通信与同步成本」（已扣掉能与 backward 重叠的部分）。
这比 profiler 里把 allreduce kernel 时间加总更贴近真实代价。

### 1.3 后端可选项（实测）

| 后端 | `dist.is_backend_available()` |
|---|---|
| `xccl`（oneCCL 的 torch 绑定，CCL_ROOT=`/opt/intel/oneapi/ccl/2022.1`） | ✅ **True** |
| `ccl` | ❌ False |
| `nccl` / `ucc` / `mpi` | ❌ False |
| `gloo` | ✅ True（纯 CPU，对 GPU 无意义） |

→ **本机唯一的 GPU 集合通信后端就是 `xccl`**，所以 TODO §3.4 要求的
「`--backend=ccl` vs `xccl` 对比」**无法完成**（`ccl` 这个名字在 torch 2.14 里已不被注册）。

---

## 2. 主结果

| 配置 | batch | ms/step | 单卡 img/s | **全局 img/s** | **TFLOPS** | `comm_ms` | `comm_pct` | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1card_nodpp` | 64 | 90.352 | 708.3 | 708.3 | 17.38 | – | – | 2.93 GiB |
| `1card_ddp` | 64 | 94.053 | 680.5 | 680.5 | 16.70 | – | – | 3.02 GiB |
| **`2card_ddp`** | 64 | 96.376 | 664.1 | **1328.1** | **32.59** | 6.05 | **6.28%** | 3.02 GiB |
| `1card_nodpp` | 128 | 179.860 | 711.7 | 711.7 | 17.46 | – | – | 5.60 GiB |
| `1card_ddp` | 128 | 183.248 | 698.5 | 698.5 | 17.14 | – | – | 5.69 GiB |
| **`2card_ddp`** | 128 | 187.587 | 682.4 | **1364.7** | **33.49** | 7.42 | **3.95%** | 5.69 GiB |

**说明**：`1card_ddp` 的 `wrapped_ddp=true` 但 `world_size=1` —— 它跑的是同一个
`DistributedDataParallel` 包装，只是 allreduce 是单进程的 no-op。
吞吐确实低于 `1card_nodpp`（680.5 vs 708.3），这正是**我们要分离出来的那部分开销**。

---

## 3. 扩展效率

### 3.1 相对纯计算基线（工程口径）

| batch | 单卡 `1card_nodpp` | 2 卡 `2card_ddp` | **加速比** | **扩展效率** |
|---:|---:|---:|---:|---:|
| 64 | 708.3 img/s | 1328.1 img/s | **1.875×** | **93.75%** |
| 128 | 711.7 img/s | 1364.7 img/s | **1.917×** | **95.88%** |

$$\text{扩展效率} = \frac{\text{2卡吞吐}}{2 \times \text{单卡吞吐}}$$

判读（TODO §5）：**扩展效率 < 70% → 通信瓶颈**。
→ **93.75% / 95.88%，远超 70% 阈值，判定为"良好"。**

### 3.2 相对 DDP 包装基线（算法口径）

| batch | 1 卡 DDP | 2 卡 DDP | **加速比** | **扩展效率** |
|---:|---:|---:|---:|---:|
| 64 | 680.5 img/s | 1328.1 img/s | **1.952×** | **97.59%** |
| 128 | 698.5 img/s | 1364.7 img/s | **1.953×** | **97.69%** |

→ 一旦扣掉 DDP 框架的固定开销，**真实的跨卡扩展效率是 97.6~97.7%**，
即互连只吃掉了 **2.3~2.4%** 的性能。**Xe Link 在这种规模下完全不是瓶颈。**

---

## 4. "通信开销"的成分分解

最容易被误读的地方：`comm_pct = 6.28%` **不等于**「互连慢」。

| batch | 2卡 vs 1卡裸模型 Δ | 其中 DDP 包装（ws=1 实测） | 其中真实跨卡 | 真实跨卡占 Δ |
|---:|---:|---:|---:|---:|
| 64 | 96.376 − 90.352 = **6.024 ms** | **3.701 ms**（+4.10%） | **2.323 ms** | **39%** |
| 128 | 187.587 − 179.860 = **7.727 ms** | **3.388 ms**（+1.88%） | **4.339 ms** | **56%** |

**结论：batch=64 时，报告里 6.28% 的"通信占比"里有 61% 其实与通信无关**，
而是「把模型包进 `DistributedDataParallel`」本身的簿记成本（梯度 bucket 的
`copy_`、autograd hook、no-op allreduce、`_sync_params` 等），在单卡上也照样存在。

再把「真实跨卡」这 2.3~4.3 ms 与链路能力对照：

| 项 | 值 |
|---|---|
| ResNet-50 bf16 梯度总量 | 25.6 M × 2 B = **51.2 MB** |
| allreduce 实测饱和带宽（§5） | **~80 GB/s** |
| 纯 allreduce 传输时间 | $51.2 / 80 = $ **0.64 ms** |
| 实测暴露的跨卡成本 | **2.32 ms**（b64）/ **4.34 ms**（b128） |
| 比值 | **3.6× / 6.8×** |

→ **暴露出来的成本是纯传输时间的 4~7 倍。** 原因是 DDP 的 allreduce 是**分 bucket**
（默认 `bucket_cap_mb=25` → 51.2 MB 分成 3 个 bucket）并与 backward 重叠：
能与计算重叠的部分被隐藏了，**剩下暴露的是"最后一个 bucket 的尾部 + 各 bucket 的启动延迟"**。
这正是通信优化的着力点（**调大/调小 `bucket_cap_mb`**），而不是去改链路。

---

## 5. Xe Link / 集合通信微基准

`diagnostics/allreduce_probe.py`（2 进程，xccl，bf16；**批量入队后统一计时**，
避免把 `dist.barrier()` 混进逐次测量 —— 第一版脚本就是被它污染的）：

| 操作 | 消息大小 | 延迟 (µs) | 有效带宽 (GB/s) |
|---|---|---:|---:|
| all_reduce | 8 B | 36.0 | 0.0 |
| all_reduce | 30 B | 28.3 | 0.0 |
| all_reduce | 524 B | 25.5 | 0.0 |
| all_reduce | 8 KiB | 30.1 | 0.3 |
| all_reduce | 128 KiB | 26.2 | 5.0 |
| all_reduce | 1.0 MiB | 68.9 | 15.2 |
| all_reduce | 8.0 MiB | 122.6 | 68.4 |
| all_reduce | 32.0 MiB | 430.1 | 78.0 |
| all_reduce | 64.0 MiB | 839.4 | **79.9** |
| `dist.barrier()` | – | **119.2** | 0.0 |

**三点读法：**

1. **小消息延迟地板 ~26 µs**（8 B ~ 128 KiB 都是 25~30 µs）。
   这是 xccl 的固定提交/同步开销，与消息大小无关。
2. **带宽在 32~64 MiB 饱和于 ~80 GB/s**。round-robin 口径下每个 rank 每方向
   实际搬 32 MiB / 839 µs = **40 GB/s/dir**。
   对照 Xe Link XL24 标称 **318 GB/s/dir**：
   $$40 / 318 = 12.6\% \quad(\text{按双向等效 } 80/318 = 25\%)$$
   → **只跑到标称值的 1/8~1/4。** 可能原因（需 ④ 复核）：
   - 本机是 **1 tile/卡** 的 Max 1100，单 tile 驱动 DMA 的能力有限；
   - ④ 已记录 Xe Link **"Not Calibrated"**（`xpu-smi` 状态）；
   - xccl/oneCCL 默认算法（ring）未针对 2-rank 直连调优；
   - ES 样片。
3. **`barrier()` 单次 119 µs** —— 比任何小消息 allreduce 都贵 4 倍。
   ⚠️ **绝不要把它放进逐迭代的计时循环**，否则会把所有小消息结果抬到 100 µs 以上
   （这是一个真实踩过的坑）。

### 5.1 对训练的映射

| 训练规模 | 梯度量 | 纯 allreduce（@80 GB/s） | 与步长之比（ResNet-50 b64） |
|---|---:|---:|---:|
| ResNet-50（25.6 M, bf16） | 51.2 MB | 0.64 ms | 0.7% of 96 ms |
| BERT-base（109.5 M, bf16） | 219 MB | 2.7 ms | 1.4% of ~190 ms（估算） |
| 10 亿参数模型（bf16） | 2 GB | 25 ms | **>100%** ← 会变成瓶颈 |

→ **在本机能跑得动的模型规模（<1B 参数）内，Xe Link 都不是瓶颈。**
只有当模型超过 ~1B 参数（梯度 >2 GB）时，80 GB/s 的实际带宽才会成为限制。
**这正是 ④（互连）测试需要复核的重点。**

---

## 6. 结论

1. ✅ **2 卡 DDP 扩展效率 93.75%~95.88%（工程口径）/ 97.59%~97.69%（算法口径），判定"良好"。**
2. ✅ **通信占比 6.28%（b64）/ 3.95%（b128）**，且**batch 越大占比越低**
   （allreduce 成本与 batch 无关，而单步计算的时长随 batch 线性增长）。
   → **生产训练应尽量用大 batch，顺便把通信占比压下去。**
3. ⚠️ 报告的 `comm_pct` **高估了互连的贡献**：其中 39%~61% 是 DDP 包装自身开销。
   做优化时不要把它当成互连性能指标。
4. ⚠️ Xe Link 实测 allreduce 只能到 **~80 GB/s（标称的 1/4）**，
   但**在当前模型规模下不影响结果**。建议 ④ 用 ze_peak / IMB-MPI1-GPU /
   oneCCL benchmark 复核链路本身，并与本节的 80 GB/s 对照。
5. ℹ️ **后端对比做不了**：torch 2.14 在本机只注册了 `xccl`。
6. ℹ️ 单卡加 DDP 包装就要 **+1.9%~+4.1%**，这是任何多卡训练都逃不掉的固定税。

---

## 7. 复现命令

```bash
cd /root/workspace/benchmark/05-ai-dl
PY=/root/workspace/venv1/bin/python
# 注意：DDP 需要两张卡都可见，不要设 ZE_AFFINITY_MASK

env -u LD_LIBRARY_PATH $PY run_bench.py --suite ddp \
    --ddp-batches 64,128 --ddp-backends xccl --ddp-dtypes bf16 --ddp-iters 20

# 集合通信微基准（必须由 torch.distributed.run 拉起）
env -u LD_LIBRARY_PATH $PY -m torch.distributed.run --standalone \
    --nproc_per_node=2 diagnostics/allreduce_probe.py
```

产出：

| 文件 | 内容 |
|---|---|
| `results/bench_*.json` / `.md` | 汇总表（含 `speedup_2x_vs_1card_ddp` / `scaling_eff_pct_vs_1card_ddp`） |
| `results/ddp_raw/resnet50_bf16_b{64,128}_xccl_{1card_nodpp,1card_ddp,2card_ddp}.json` | 6 个原始 worker 输出（含 `sync` / `nosync` 分布） |

---

## 8. 未做 / 待补充

| 项 | 状态 | 说明 |
|---|---|---|
| `--backend=ccl` 对比 | ❌ **做不了** | torch 2.14 未注册 `ccl`，也不是 `is_backend_available` |
| `bucket_cap_mb` 扫描 | ⬜ 未做 | §4 显示"暴露成本是纯传输的 4~7 倍"，**这是最有希望的一个调优旋钮**，建议补做（8/25/50/100 MB） |
| gradient compression / `find_unused_parameters` | ⬜ 未做 | 低优先级 |
| BERT / LLM 的 DDP | ⬜ 未做 | 本次只做 ResNet-50。BERT-base 梯度 219 MB，通信占比会略高，可用 `--model bert` 补做 |
| 4 卡扩展 | ❌ 不可行 | 本机只有 2 张卡 |
| 真正的 Xe Link 点对点带宽 | ⬜ 未做 | 属 ④ 范畴（`ze_peak` / IMB-MPI1-GPU 未安装需自建） |
