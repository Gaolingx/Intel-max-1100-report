# benchmark/05-ai-dl — XPU 理论性能测试工具（PyTorch / torch.xpu）

对应文档：
[`docs/TODO/05-ai-dl.md`](../../docs/TODO/05-ai-dl.md)（AI / 深度学习）、
[`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md)（算力峰值）、
[`docs/TODO/03-memory-bandwidth.md`](../../docs/TODO/03-memory-bandwidth.md)（显存带宽）。

用 `torch.xpu` 做**算子级 micro-benchmark**，把「理论峰值」替换成**实测值**，
并为后续的 ResNet/BERT 训练、LLM 推理、DDP 扩展效率提供「能力上界」参照。

---

## 1. 目录结构

```
benchmark/05-ai-dl/
├── README.md              本文档
├── requirements.txt       依赖说明（torch 走系统 site-packages）
├── run_bench.py           CLI 入口
├── train_ddp.py           DDP worker（由 torchrun 拉起，勿直接运行）
├── results/               运行后自动生成的 JSON / Markdown 报告
│   └── ddp_raw/           DDP 子进程的原始结果 JSON
└── xpu_bench/
    ├── common.py          设备信息、理论峰值、Event 计时、结果结构、遥测
    ├── gemm.py            ① matmul：方阵 / 真实 shape / batched / INT8
    ├── elementwise.py     ② elementwise 算子 + 显存带宽专项
    ├── reduction.py       ③ softmax / layer_norm / rms_norm / dot
    ├── attention.py       ④ SDPA：prefill 与 decode
    ├── conv.py            ⑤ conv2d（ResNet-50 shape）
    ├── quant.py           ⑥ 量化：INT8 稠密 / W8A16 / W4A16 + 数值自检
    ├── precision.py       ⑦ 数值格式支持矩阵
    ├── models.py          ⑧ 模型级：ResNet-50 / BERT 训练、LLM 推理
    ├── pipeline.py        ⑨ 数据管线 / 主机侧瓶颈（DataLoader / H2D）
    ├── ddp.py             ⑩ 双卡 DDP 扩展效率编排（torchrun 子进程）
    └── report.py          Markdown 报告渲染
```

> 前 7 个模块（`gemm` … `precision`）是**算子级 micro-benchmark**；
> 后 3 个（`models` / `pipeline` / `ddp`）是**模型级 / 端到端**测试，耗时远高于算子级，
> 因此 **不包含在默认的 `--suite all` 中**（需 `--with-models` 或显式点名）。

---

## 2. 快速开始

```bash
cd /root/workspace/benchmark/05-ai-dl

# 注意：必须激活带 XPU 的 venv（system-site-packages），否则 torch.xpu 不可用
source /root/workspace/venv1/bin/activate

python run_bench.py --list                 # 列出所有 suite
python run_bench.py --quick --suite all    # 冒烟：缩小尺寸，几十秒跑完
python run_bench.py --suite gemm --large   # 只跑 GEMM，含 16384³
python run_bench.py --suite all            # 完整跑一遍（算子级）

# 模型级（§3.2–§3.6）
python run_bench.py --suite resnet,bert --model-batches 64,128,256
python run_bench.py --suite llm --llm-batches 1,4,8 --llm-input-lens 128,512,2048
python run_bench.py --suite pipeline
python run_bench.py --suite ddp --ddp-backends xccl --ddp-batches 64,128
```

> ⚠ 若 `torch.xpu.is_available()` 为 False，先 `source /opt/intel/oneapi/setvars.sh`，
> 并确认用的是 `/root/workspace/venv1`（`uv venv --system-site-packages` 创建）。
>
> 模型级 suite 需要 `transformers`：
> `uv pip install --python /root/workspace/venv1/bin/python transformers`
> （本机 **huggingface.co 不可达**，权重默认从 `HF_ENDPOINT=https://hf-mirror.com` 下载；
> 失败时自动回退到**本地 config 随机初始化**，结构相同，吞吐可用于对标）

---

## 3. Suite 一览

| suite | 内容 | 对应文档 |
|---|---|---|
| `gemm` | 方阵 sweep（256…16384）、真实场景 shape（decode GEMV / prefill / FFN / 瘦长矩阵）、batched matmul、INT8（`torch._int_mm`，int32 累加）；覆盖 fp32 / fp64 / fp16 / bf16 | ② 3.2、⑤ 3.1 |
| `elementwise` | copy / scale / add / mul / sub / triad / relu / sigmoid / tanh / exp / gelu / silu，用**有效带宽 GB/s** 衡量 | ③ 4、⑤ 3.1 |
| `membw` | 带宽 size sweep（暴露 L2 命中虚高）、dtype 对照（带宽应与元素宽度无关）、H2D / D2H 传输带宽 | ③ 4 Step 3/5/6 |
| `reduce` | sum / amax / mean / sum(dim) / softmax / log_softmax / layer_norm / rms_norm / l2_normalize / dot | ⑤ 3.1 |
| `attention` | SDPA：prefill（512…8192）、causal、decode（q_len=1、kv_len 4096/8192、batch=32），含朴素实现对照 | ⑤ 3.1 / 3.5 |
| `conv` | conv2d：ResNet stem / 3×3 各阶段 / 1×1 / depthwise / channels_last | ⑤ 3.2 |
| `quant` | INT8 稠密 GEMM（`torch._int_mm`）、W8A16（`_weight_int8pack_mm`）、W4A16（`_weight_int4pack_mm`）权重独占，含 BF16/W8A16/W4A16 同 shape 对照与**数值自检** | ⑤ 3.1（量化） |
| `precision` | **数值格式支持矩阵**：厂商列出的每个精度逐个实跑，记录 OK/FAIL；vector 9 种、matmul 10 种 | ② 3.5 / ⑤ 3.7 |
| `resnet` | ResNet-50（torchvision 真实模型）前/反向 + SGD step，batch × dtype（fp32 / bf16-AMP）× channels_last | ⑤ 3.2 |
| `bert` | BERT-base（`BertForMaskedLM`）前/反向 + AdamW step，seq_len × batch | ⑤ 3.3 |
| `llm` | 自回归推理：prefill（TTFT）/ decode（tok/s）/ 显存峰值，batch × input_len | ⑤ 3.5 |
| `ddp` | 双卡 DDP：`torchrun` 拉起 1卡无DDP / 1卡DDP / 2卡DDP，算扩展效率与通信占比 | ⑤ 3.4 / ④ |
| `pipeline` | 数据管线：`DataLoader` worker 扫参、`pin_memory`、与 GPU step 重叠、H2D 带宽 | ⑤ 3.6 |

---

## 4. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--suite` | `all` | 逗号分隔或 `all` |
| `--device` | `0` | GPU 序号（0 / 1） |
| `--warmup` | `5` | 每个测点预热次数（排除 JIT / 首次编译） |
| `--iters` | `20` | 每个测点计时迭代次数，取**中位数** |
| `--dtypes` | 按 suite 取 | 如 `fp32,bf16,fp16,fp64` |
| `--size-mb` | `1024` | elementwise / reduce 每个张量大小（MiB） |
| `--gemm-sizes` | `256,…,8192` | GEMM 方阵 sweep 的 n |
| `--bw-sizes` | `4,…,2048` | 带宽 size sweep（MiB） |
| `--large` | 关 | 加入 16384³ GEMM、8K attention |
| `--quick` | 关 | 冒烟模式 |
| `--telemetry PATH` | 关 | 同时用 `xpu-smi dump` 采样功耗/频率/利用率 |
| `--outdir` | `results/` | 输出目录 |
| `--with-models` | 关 | 让 `--suite all` 一并包含模型级 suite |
| `--model-batches` | `64,128,256` | ResNet-50 batch size 列表 |
| `--model-dtypes` | `fp32,bf16` | 模型级 dtype 子集（bf16 走 AMP） |
| `--model-compile` | 关 | ResNet-50 开启 `torch.compile`（首次编译很慢） |
| `--bert-model` | `bert-base-uncased` | BERT 模型 id |
| `--bert-seq-lens` | `128,512` | BERT 序列长度列表 |
| `--bert-batches` | `16,32` | BERT batch size 列表 |
| `--llm-model` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM 模型 id（causal LM） |
| `--llm-batches` | `1,4,8` | LLM 推理 batch size 列表 |
| `--llm-input-lens` | `128,512,2048` | LLM 输入长度列表 |
| `--llm-new-tokens` | `32` | 每个测点生成的新 token 数 |
| `--no-hf` | 关 | 不联网下载权重，直接用本地 config 随机初始化 |
| `--hf-endpoint` | `https://hf-mirror.com` | HF 镜像端点（本机 huggingface.co 不可达） |
| `--ddp-model` | `resnet50` | DDP 测试模型（`resnet50` / `bert`） |
| `--ddp-backends` | `xccl` | DDP 后端列表（`xccl` 原生 / `ccl` oneCCL / `gloo`） |
| `--ddp-dtypes` | `bf16` | DDP dtype 列表 |
| `--ddp-batches` | `64,128` | DDP 每卡 batch size 列表 |
| `--ddp-iters` | `20` | DDP 计时迭代数 |
| `--pipe-batch` | `64` | pipeline 测试 batch size |
| `--pipe-workers` | `0,1,2,4,8` | DataLoader worker 数列表 |
| `--pipe-cpu-scale` | `3` | 合成数据集单样本 CPU 预处理强度（0=无） |
| `--pipe-batches` | `20` | 每个 worker 配置采集的 batch 数 |
| `--pipe-e2e-batches` | `20` | 端到端 step 的批次数 |

细粒度开关：`--no-real-shapes`、`--no-batched`、`--no-int8`、`--no-manual-attn`、
`--no-dtype-bw`、`--no-size-sweep`、`--no-host-transfer`、`--no-quant-verify`、
`--no-quant-compare`。

---

## 3.1 模型级 suite 的测量口径与假设

这些假设会写进每条结果的 `note` 字段，并出现在报告里：

1. **FLOPs 估算**：ResNet-50 前向 = 4.09 GMAC（2× = 8.18 GFLOP）；
   训练 = 3× 前向（fwd + bwd-input + bwd-weight）。
   BERT 训练 = 3 × 2 × N_params × tokens。
   这是**理论上界**，只用来算 `tflops` 达成率。
2. **AMP 口径**：`bf16` 测点 = `torch.autocast("xpu", dtype=torch.bfloat16)`；
   `fp32` 直推。**不测 `GradScaler`**（bf16 无需 loss scaling）。
3. **权重要真实**：优先从 `HF_ENDPOINT=https://hf-mirror.com` 拉真实权重；
   失败则用**本地 config 随机初始化**。两条路径结构完全相同，吞吐可比；
   结果里 `params["weights"]` 会如实记为 `pretrained:<id>` 或 `random-init:<id>`。
4. **同步计时**：`torch.xpu.Event` 设备侧 + `torch.xpu.synchronize()`，
   每个测点先 warmup 再计时，取**中位数**。
5. **DDP 通信占比** 用 **no_sync 差分法**：
   `comm_ms = t_sync − t_nosync`（`model.no_sync()` 下不做梯度 all-reduce）。
   它不等于纯通信耗时（没有重叠校正），但足以判断「计算为主还是通信为主」。
   DDP 由 `torchrun --nnodes=1 --nproc_per_node={1,2}` 拉起子进程，
   结果 JSON 落在 `results/ddp_raw/`。
6. **主机内存倒挂**：本机仅 45 GiB RAM，而 2 卡合计 96 GiB HBM。
   `pipeline` suite 就是为量化这个瓶颈而写 —— `pin_memory` buffer 与
   DataLoader worker 都占用同一份 RAM。

---

## 5. 输出

每次运行产生三个产物：

1. **控制台 Markdown 报告**（环境 → 理论参考值 → 自动摘要 → 明细表 → 跳过/失败项）
2. `results/bench_<时间戳>.json` —— 结构化结果，含每条测点的 `params` / `stats` / `metrics`，
   便于画「带宽-尺寸」「功耗-性能」曲线
3. `results/bench_<时间戳>.md` —— 可直接贴进 `docs/`

指标口径：

* `tflops` = `FLOPs / t`，FLOPs 按算子定义（GEMM = `2MNK`；attention = `4·B·H·Sq·Skv·D`，causal 减半）
* `tops`   = INT8 的 GOPS/TOPS（`2MNK`）
* `gbps`   = `访问字节 / t`，`io_factor` 记录每次迭代的读+写元素次数（copy=2，add/triad=3）
* `ratio_pct` = 实测 / 理论（仅 fp32、fp64 有理论值）

---

## 6. 量化（INT8 / INT4）支持情况与调用约定

Intel Data Center GPU Max 1100（Ponte Vecchio）的 XMX 在硬件上支持 INT8 / INT4，
但**软件路径可用性与理论倍数完全不是一回事**，`quant` suite 的实测结论如下。

### 6.1 torch 2.14.0+xpu 上可用的三条路径

| 路径 | 算子 | 语义 | 数值 | 实测吞吐 |
|---|---|---|---|---|
| INT8 稠密 | `torch._int_mm` | int8×int8 → int32（int32 累加） | ✅ 与 CPU 参考**逐位一致** | **≈400 TOPS**（4096×4096×16384） |
| W8A16 | `aten._weight_int8pack_mm` | x(fp16/bf16/fp32) × w(int8) × **per-channel** scales | ✅ 与 `x@(w*s)ᵀ` 一致 | ⚠ 异常低（见 6.3） |
| W4A16 | `aten._weight_int4pack_mm` | x × w(int4) + per-group scale/zero | ✅ 见 6.2 | ≈120 TOPS（远低于 INT8） |

### 6.2 INT4 的 XPU 调用约定（与 CUDA **不同**，且无文档）

`docs` 与 PyTorch meta 注册都默认 CUDA 约定，直接照抄会得到**全 0 输出**或形状错误。
本工具实测确认的 XPU 约定：

* `x`：`[M, K]`，fp16 / bf16 / fp32；`M`、`N`、`K` 需对齐（`M` 太小/未对齐会静默返回 0）
* `mat2`（打包权重）：**2D** `[N, K/2] uint8`（CUDA 是 4D `int32`）
* 4 bit 码沿 `K` 轴**线性打包**：字节 *j* 的低 nibble = 第 `2j` 列，高 nibble = 第 `2j+1` 列
* `qScaleAndZeros`：**bf16**，布局接近 `[K/gs, N, 2]`（最后一维为 `(scale, zero)`）
* 反量化公式实测为：`w = scale * (q - 8) + zero`，`q ∈ [0,15]`
* 输出：`[M, N]`，与 `x` 同 dtype

> 自检项 `verify_int4_weight`：令 `scale=1, zero=8`（反量化退化为原始 4bit 码）时
> `max|err| ≈ 5e-4`，说明打包与累加语义**逐位正确**；
> 随机逐组 scale 的残差处于 bf16 累加噪声量级。

### 6.3 关键结论

1. **INT8 可用且值得用**：≈400 TOPS，是 BF16 峰值的 **1.7×**
   （略低于 XMX 理论的 2×，属 kernel 效率损耗）。
2. **INT4 能用，但没有算力红利**：W4A16 ≈120 TOPS，**反而低于 INT8 的 0.3×**，
   远不及 XMX INT4 = INT8 2× 的理论值 —— torch 2.14 的 XPU int4 kernel
   没有跑满 XMX INT4 路径。**INT4 的价值在显存/带宽，不在算力**：
   权重从 2 B 降到 0.5 B，decode（M 小）时权重流量减到 1/4。
3. **⚠ W8A16（`_weight_int8pack_mm`）当前不可用于生产**：
   M=1 时 3.18 ms（BF16 仅 0.10 ms），且耗时**几乎与 M 无关**，
   疑似每次调用都重新打包权重（kernel 名中的 `int8pack`）。
   需要 W8A16 时建议：INT8 稠密 GEMM + 显式反量化，或直接用 BF16/FP8。
4. **decode（GEMV, M=1）是纯带宽游戏**：BF16 0.101 ms / 332 GB/s，
   W4A16 0.108 ms / 77 GB/s —— 4bit 权重虽然少读 4×，
   但 kernel 效率差，**实际延迟并未改善**。想要吃掉 INT4 的带宽红利，
   需要等更成熟的 int4 kernel 或自己写（triton-xpu 亦可）。

复现：

```bash
python run_bench.py --suite quant                 # 含自检 + 三方对照
python run_bench.py --suite quant --no-quant-compare
```

---

## 7. 判读标准（来自文档）

| 现象 | 结论 |
|---|---|
| BF16/FP16 加速比 < 1.5× | **XMX 未启用** ← 首要排查（工具会自动提示） |
| FP32 达成率 < 70% | kernel 效率低 / 未向量化 / 带宽瓶颈 |
| 小数组带宽极高、大数组骤降 | 小数组被 **L2（192 MB）** 命中，读数无效 |
| 带宽与 dtype 无关 | 正常（说明是纯 HBM 带宽受限，无类型转换开销） |
| decode（q_len=1）tflops 极低、gbps 高 | 正常，KV cache 读取是 memory-bound |
| 双卡同时带宽显著下降 | 共享资源竞争（本工具**单卡**运行，双卡需另测） |
| INT8 稠密 ≥ 1.7× BF16 | 正常，XMX INT8 路径已生效 |
| INT4 吞吐 **低于** INT8 | 正常（本栈现状），int4 kernel 未跑满 XMX INT4 |
| FP16 与 BF16 峰值几乎重合 | 正常：XMX 对 f16/bf16 **一律 FP32 累加**，本卡无独立 fp16-acc 档 |
| 想"开 fp16 累加"换吞吐却无收益 | 正常：该路径在 PVC 上不存在（详见 `docs/precision-support.md` §7.2） |

---

## 8. 与其他测试的衔接

| 本工具结果 | 用途 |
|---|---|
| `gemm` 峰值 | 作为 ⑤ AI 训练 / ⑥ HPC 应用的「算力天花板」 |
| `membw` 带宽上限 | 作为 memory-bound 算子的「带宽墙」，喂给 ⑦ Roofline |
| `attention` / `reduce` / `conv` | 解释端到端模型为何快/慢，定位热点 kernel |
| `--telemetry` 采样 | 与 ⑧ 功耗/能效联动（perf/W） |

> 本工具是 **算子级 micro-benchmark**，不替代 BabelStream / `ze_peak` 的库级交叉验证，
> 也不替代 ⑤ 文档中的 ResNet-50 / BERT / DDP 端到端测试。
> 显存带宽结论请与 `docs/TODO/03-memory-bandwidth.md` 的 BabelStream 结果对照。

---

## 9. 注意事项

1. **单卡运行**：`--device` 选择 0 或 1；双卡同时测试请参考 ③ 文档用
   `ZE_AFFINITY_MASK` 起两个进程。
2. **必须 warmup + synchronize**：工具已内置，勿绕过 `benchmark()`。
3. **带宽测试数据量必须远超 L2（192 MB）**，否则测到的是 cache 带宽。
4. `--size-mb 2048` 在 fp32 下 a+b+out 约 24 GiB，注意 48 GiB 显存与 45 GiB 主机内存。
5. `rms_norm` 为未融合参考实现，含临时张量开销，有效带宽会偏低（已在 `note` 标注）。
6. 频率稳定（短时满载 1550 MHz）/ 功耗上限 300 W 时结果可重复性最好；
   改功耗或频率后请重新跑一遍基线。
