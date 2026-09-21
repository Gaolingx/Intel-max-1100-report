# ③ 显存带宽测试（HBM）

**优先级：P1**
**文档位置：** `docs/TODO/03-memory-bandwidth.md`

---

## 1. 测试目标

1. 得到 **HBM 实际读写带宽**（单卡 / 双卡同时）
2. 计算达成率，确定**带宽天花板**
3. 为 ② 算力测试和 ⑤ AI 测试提供**「是否带宽受限」的判据**
4. 测试**访问模式**对带宽的影响（粒度、向量宽度、并发度）

## 2. 为什么重要

Ponte Vecchio 在 AI/HPC 场景下**极易带宽受限**。没有这个基线，就无法判断：
- 「GEMM 慢」是算力问题还是带宽问题
- 「Attention 慢」是不是 memory-bound
- Roofline 分析中的带宽上限是否与实测一致

### 规格值说明
HBM 带宽的**规格值需要在测试中实测确认**。
（此前口头给出的约 1.2 TB/s 只是粗略量级估计，不要作为结论引用。）

---

## 3. 工具与准备

| 工具 | 状态 | 说明 |
|---|---|---|
| **BabelStream (SYCL)** | ❌ 需构建 | **事实标准**，Copy/Mul/Add/Triad |
| PyTorch (`torch.xpu`) | ✅ 已装 | 快速交叉验证 |
| `xpu-smi dump` | ✅ 已装 | 硬件侧读写计数（metric 6/7/5） |
| `stream` (CPU) | ✅ `/usr/bin/stream` | **主机侧**内存带宽，用于区分 host/device 瓶颈 |
| `mem_bench` (VTune) | ✅ 已装 | 内存带宽微基准 |

### 构建 BabelStream
```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream
cmake -B build -H. -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 4. 执行步骤

### Step 1：BabelStream 标准测试（主力）
```bash
# 默认数组大小可能偏小，务必放大以压满带宽
./build/babelstream -s $((1<<30))          # 元素数，按需调整成能占满显存
./build/babelstream -s $((1<<30)) 2>&1 | tee babelstream_gpu0.log
```
关注 5 个 kernel 的 GB/s：
| Kernel | 每次迭代访问字节 | 说明 |
|---|---|---|
| Copy | 2 × N × 8 B | 读+写 |
| Mul | 2 × N × 8 B | 读+写 |
| Add | 3 × N × 8 B | 2 读 + 1 写 |
| Triad | 3 × N × 8 B | 2 读 + 1 写 |
| Dot | 2 × N × 8 B | 归约 |

> 典型范围：单卡 HBM 带宽应在 **TB/s 量级**。取 5 个 kernel 的**最大值**作为带宽上限。

用不同元素数扫点（如 2^26 → 2^31），观察带宽随数据规模的变化，找出**饱和点**。

### Step 2：PyTorch 交叉验证
```python
import torch
from torch.utils.benchmark import Timer

dev = "xpu"
results = {}
for gb in [1, 2, 4, 8, 16]:
    n = gb * (1 << 30) // 4          # float32
    a = torch.empty(n, device=dev, dtype=torch.float32)
    b = torch.empty(n, device=dev, dtype=torch.float32)
    t = Timer("b.copy_(a)", globals=globals()).timeit(20)
    gbps = 2 * n * 4 / t / 1e9       # 读 + 写
    results[gb] = gbps
    print(f"{gb:3d} GiB  copy  {gbps:8.1f} GB/s")
```

### Step 3：dtype 影响
用 `float16` / `bfloat16` / `int8` 重复上述测试 —— 检验带宽是否与元素宽度无关（若带宽容差明显差异，说明有量化/转换开销）。

### Step 4：硬件计数对照
```bash
xpu-smi dump -d 0 -m 5,6,7 -i 200 -n 200 -j > mem_bw_telemetry.json
```
- `m5` GPU Memory Utilization (%)
- `m6` GPU Memory Read (kB/s)
- `m7` GPU Memory Write (kB/s)

对比软件测量的带宽与硬件计数，验证一致性。

### Step 5：双卡同时测试
```bash
# 两个进程各占一卡，或用 ZE_AFFINITY_MASK 分配
ZE_AFFINITY_MASK=0 ./build/babelstream -s $((1<<30)) > gpu0.log 2>&1 &
ZE_AFFINITY_MASK=1 ./build/babelstream -s $((1<<30)) > gpu1.log 2>&1 &
wait
```
**关键问题**：双卡同时跑时，单卡带宽是否下降？（HBM 独立，理论上不应互相影响；若下降说明存在共享资源竞争）

### Step 6：主机侧带宽基线
```bash
/usr/bin/stream                       # 主机内存带宽上限
# 或指定更大数组
OMP_NUM_THREADS=144 /usr/bin/stream
```
这个数字非常关键 —— 在内存倒挂（45 GiB host vs 96 GiB HBM）的机器上，host 侧带宽常常是真正的瓶颈。

---

## 5. 指标记录表

| 项目 | GPU 0 | GPU 1 | 双卡同时 |
|---|---|---|---|
| Copy (GB/s) | | | |
| Mul (GB/s) | | | |
| Add (GB/s) | | | |
| Triad (GB/s) | | | |
| Dot (GB/s) | | | |
| **峰值带宽** | | | |
| 饱和数据规模 | | | |

| 主机侧 | GB/s |
|---|---|
| Stream Copy / Scale / Add / Triad | |

附加记录：测试期间 `m6/m7` 硬件读写的实时值，以及功效（GB/s per W）。

---

## 6. 判读标准

| 现象 | 可能原因 |
|---|---|
| 软件带宽 ≫ 硬件计数 | 测量方法有误（cache 命中、未真正读写） |
| 小数组带宽极高、大数组骤降 | 小数组落在 L2/cache 中，**必须用大数组** |
| 双卡同时带宽显著下降 | 共享资源竞争（PCIe / 主机内存 / 电源） |
| dtype 不同导致带宽差异大 | 存在类型转换开销 |
| 带宽达成率 < 70% | 访问模式不佳、kernel 线程不足、需调向量宽度 |

> **关键**：数据必须**大于 L2 cache**才能测到真实 HBM 带宽。注意 Max 1100 有较大的 L2/LLC。

---

## 7. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| → ② 算力峰值 | 判断 GEMM 是算力受限还是带宽受限（算力/带宽比） |
| → ⑤ AI 测试 | Attention 等 memory-bound 算子的理论上限 |
| → ⑦ Profiling | Roofline 模型的「带宽墙」是否与实测一致 |
| → ⑧ 能效 | 计算 GB/s per W |

---

## 8. 注意事项

1. **数组规模必须远超 cache**，否则测的是 cache 带宽而非 HBM 带宽。
2. BabelStream 的 `-s` 单位是**元素个数**，注意换算成字节。
3. 双卡测试注意**总功耗 600 W** 是否超 PSU 能力。
4. 主机 45 GiB 内存下，分配过大的 host 数组会触发 swap，污染结果。
5. 记录时的频率锁定状态（1550 MHz）会影响带宽 —— 锁频是有利于可重复性的。
