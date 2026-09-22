# benchmark/02-compute-peak — 算力峰值实测（ALU / XMX-DPAS）

对应文档：[`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md)（算力峰值测试方案）、
[`docs/hardware.md`](../../docs/hardware.md)（硬件规格）、
[`docs/Conclusion/02-compute-peak/`](../../docs/Conclusion/02-compute-peak/)（结论报告）。

本目录用 **三条互相独立的路径** 测算力，避免「自己验证自己」：

| # | 路径 | 工具 | 测的是什么 | 可信度 |
|---|---|---|---|---|
| 1 | 自研 SYCL 探针 `alu_peak` | icpx/DPC++ | 纯 FMA 循环（非可折叠递推）→ **ALU 峰值** | 待验证上界 |
| 2 | 自研 SYCL 探针 `xmx_peak` | icpx + `joint_matrix` | 直接发射 **DPAS**（XMX）→ 原始矩阵峰值 | 已自校验 |
| 3 | PyTorch / oneDNN GEMM | `torch.matmul`、`torch._int_mm` | 成熟软件栈能达到的实际峰值 | **可信下界** |

再加一条取证：`clock` —— 主频 / 功耗 / 是否降频。

> ⚠ 三者给出的是**不同的量**，不要混用：自研探针测的是「硬件上限」，
> oneDNN 测的是「工程可达」。本目录把两者都报出来，并**显式标注差异**。

---

## 1. 目录结构

```
benchmark/02-compute-peak/
├── README.md                  本文档
├── run_bench.py               CLI 入口（4 个 suite：alu / xmx / torch / clock）
├── sycl/
│   ├── alu_peak.cpp           自研 ALU 峰值探针（VEC/ACC/UNROLL 可宏定义）
│   ├── xmx_peak.cpp           自研 XMX/DPAS 峰值探针（sycl joint_matrix）
│   └── alu_clock_probe.cpp    【死路，仅留档】用 SYCL 读 GPU 时钟 —— 本机不支持
├── probes/
│   └── torch_gemm_peak.py     PyTorch GEMM / int8 / vector 交叉验证
├── build/                     探针编译产物（alu_peak、xmx_peak、alu_peak_vecN）
└── results/                   运行后自动生成的 JSON / Markdown 报告
```

---

## 2. 快速开始

```bash
cd /root/workspace/benchmark/02-compute-peak

python3 run_bench.py                    # 全部 4 个 suite，写 results/bench_<tag>.{json,md}
python3 run_bench.py --quick            # 冒烟（约 1/5 时间）
python3 run_bench.py alu xmx            # 只跑指定 suite
python3 run_bench.py --tag 20260922-194139

# PyTorch 路径需要 venv1（system-site-packages 里才有 torch+xpu）
ZE_AFFINITY_MASK=0 /root/workspace/venv1/bin/python run_bench.py torch
```

`run_bench.py` 会自动用 `icpx -fsycl -O3` 编译两个探针到 `build/`
（`icpx` 路径写死为 `/opt/intel/oneapi/compiler/2026.1/bin/icpx`）。

**实测参考产物**：[`results/bench_20260922-194139.md`](results/bench_20260922-194139.md)。
（更早的 `results/bench_20260922-192831.*` 是 v1，其中 XMX 占用率的解读有误 —— 当时把
「低占用率下等量工作等量耗时」误判为编译器消循环。**请只看 v2 及以后。**）

---

## 3. 四个 suite 分别测什么

### 3.1 `alu` —— ALU 峰值（自研 SYCL）

`sycl/alu_peak.cpp`：每个 work-item 跑 `ACC`(=8) 条独立的 `VEC`(=8) 宽 `sycl::vec` 依赖链，
链上做**非可折叠递推**（logistic map `acc = 1 - acc*acc`，故意选混沌、无闭式解，
编译器无法把它化简成常数），每链每内层迭代连做 `UNROLL`(=4) 次 FMA。

```
FLOP 计数 = global_size × VEC × ACC × UNROLL × iters × 2
```

- CLI：`./alu_peak <dtype|all> [global=7168] [iters] [warmup=3] [repeats=5] [local=128]`
- 输出：`DEVICE {json}`（含 `vec`/`acc`/`unroll`）、`RESULT {json}`（含 `value` GFLOPS/GOPS）
- 用 `guard` 写回（`global_id == SIZE_MAX`，运行期值）防止整段被 DCE 掉
- 编译：`icpx -fsycl -O3 -ffp-contract=fast`
- **VEC 扫描**：`-DALU_VEC={1,2,4,8,16}` 编出 `alu_peak_vecN` 分别跑

### 3.2 `xmx` —— XMX / DPAS 原始峰值（自研 SYCL `joint_matrix`）

`sycl/xmx_peak.cpp`：每个 sub-group 装载 1 个 A 片段 + 1 个 B 片段，
然后**在编译期展开的 `IT`(=128) 内层循环**里对 `NACC`(=4) 个独立累加器连续发 DPAS，
外面套一个**运行期 `outer`(=8192) 循环**：

```
dpas_count = sub_groups × outer × IT × NACC
FLOP 计数  = dpas_count × 2 × M × N × K        (bf16/fp16: M8N16K16；int8: M8N16K32)
```

- CLI：`./xmx_peak <dtype|all> [global=7168] [outer=8192] [warmup=3] [repeats=5] [local=16] [it=128]`
- 输出：`DEVICE`、`CHECK {dtype,expect,got,ok}`、`RESULT {...}`（含 `dpas_count`、
  `mac_per_cycle_eu_nominal`、`check_ok`）
- `CHECK`：把 A、B 全填 1，做 1 次 MAD，`C[0]` 必须等于 K → 验证**数值正确**（而不仅是快）

**joint_matrix API 备忘（踩坑记录，官方文档没写全）**

| 项 | 正确写法 |
|---|---|
| 头文件 | `#include <sycl/ext/oneapi/matrix/matrix.hpp>`（**不是** `experimental/matrix/`，该路径不存在） |
| 命名空间 | `sycl::ext::oneapi::experimental::matrix`（习惯 `namespace mx = ...`） |
| 模板 | `joint_matrix<Group, T, use, Rows, Cols, Layout>` |
| Group | `sycl::sub_group` |
| use | `mx::use::a` / `mx::use::b` / `mx::use::accumulator` |
| Layout | `mx::layout::row_major` |
| 形状 | bf16/fp16 `m8n16k16`；int8 `m8n16k32` |
| **指针** | `joint_matrix_load/store` **只接受** `sycl::multi_ptr<T, sycl::access::address_space::global_space>`；裸 `T*` 编译报 `could not match 'multi_ptr<...>' against 'T *'` |
| 不存在 | `joint_matrix_element_get`（只有 `joint_matrix_prefetch` / `joint_matrix_apply`）；没用到的累加器会被 DCE，**必须 store 出去** |

### 3.3 `torch` —— oneDNN 交叉验证

`probes/torch_gemm_peak.py`：

- GEMM：`torch.matmul` fp32 / bf16 / fp16；**int8 必须用 `torch._int_mm`**
  （`torch.matmul` 对 int8 返回 int8 会溢出），形状 4096³ / 4096×4096×16384 / 8192³
- vector：`add` / `mul` / `relu`（64 Mi 元素），fp32 & bf16，换算成 GB/s
- 计时：`torch.xpu.Event(enable_timing=True)`，取多次迭代最小值
- 输出：每行 `TORCHCOMPUTE {json}`，另加 `TORCHDEVICE {json}`

### 3.4 `clock` —— 主频 / 功耗取证

- `/sys/class/drm/card0/gt_*_freq_mhz`：`act/cur/min/max/boost/RP0/RP1/RPn`
- `xpu-smi stats -d 0`：Frequency / Power / Utilization
- **关键做法**：在**后台跑满载 ALU 的同时**采样，否则读到的全是空闲值
  （空闲时 `gt_act_freq_mhz` 会读 **0**）
- 最后从实测算「等效执行宽度」，见 §5.3

---

## 4. 关键实测结果（tag `20260922-194139`）

### 4.1 ALU 峰值（自研 SYCL，`global=114688`）

| dtype | 实测 | 单位 | vs 公式标称 22.22 |
|---|---:|---|---|
| fp32 | **50 750** | GFLOPS | **228.4%** |
| fp64 | **51 900** | GFLOPS | 233.6% |
| fp16 | **52 600** | GFLOPS | 236.7% |
| int32 | 18 000 | GOPS | 81% |

> ⚠️ fp64 / fp16 的探针值甚至**高于** fp32。在一条纯 FMA 链上这是正常的
> （fp64 与 fp32 共用发射路径；fp16 可以打包）。
> **没有哪个 dtype 落在公式的 1/2。**

### 4.2 ALU 占用率扫描（fp32）—— 找拐点

| global | 每硬件 lane 的 work-item | TFLOPS | |
|---:|---:|---:|---|
| 7 168 | 1 | 10.09 | 1/5 峰值，延迟受限 |
| 14 336 | 2 | 16.67 | |
| 28 672 | 4 | 20.50 | |
| 57 344 | 8 | 25.44 | |
| **114 688** | **16** | **50.75** | ← **拐点** |
| 229 376 | 32 | 50.76 | 线性 ✓ |
| 458 752 | 64 | 50.83 | 线性 ✓ |

### 4.3 ALU 向量宽度扫描（fp32）

| VEC | 1 | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| TFLOPS | 34.8 | 45.6 | **52.8** | 50.8 | 51.3 |

→ VEC≥4 后拉平：该循环是 **issue 受限**，不是宽度受限。

### 4.4 XMX / DPAS 峰值（自研 SYCL，`outer=8192, it=128, nacc=4`）

`CHECK` 全部通过（bf16/fp16 期望 16、int8 期望 32，`got == expect`）→ **DPAS 通路可用且数值正确**。

| sg/EU | global | bf16 TFLOPS | fp16 TFLOPS | int8 GOPS |
|---:|---:|---:|---:|---:|
| 1 | 7 168 | 118.1 | 118.1 | 236.2 |
| 2 | 14 336 | 141.9 | 141.9 | 283.9 |
| 4 | 28 672 | 177.4 | 177.4 | 354.9 |
| **8** | **57 344** | **354.9** | **354.9** | **709.7** | ← **拐点** |
| 16 | 114 688 | 354.9 | 354.9 | 709.9 | 线性 ✓ |
| 32 | 229 376 | 355.0 | 355.0 | 710.0 | 线性 ✓ |

- **bf16 = fp16 = 355 TFLOPS**，**int8 = 710 GOPS = bf16 的 2.00×**（正好等于 K=32 vs K=16）
- 折算 **256 MAC/clk/EU**（= 355.5 TFLOPS ÷ 448 EU ÷ 2 ÷ 1.55 GHz）
- 产品常引用值 ≈176 TFLOPS(bf16) → **实测是它的 2.0×**

### 4.5 oneDNN / PyTorch 交叉验证

| 项 | 最佳形状 | 实测 | 参照 | 占比 |
|---|---|---:|---|---:|
| fp32 GEMM | 8192³ | **22.13 TFLOPS** | 公式标称 22.22 | **99.6%** |
| bf16 GEMM | 4096³ | **225.9 TFLOPS** | 自研 DPAS 355.5 | 63.5% |
| fp16 GEMM | 8192³ | **237.6 TFLOPS** | 自研 DPAS 355.5 | 66.8% |
| int8 `_int_mm` | 8192³ | **416.2 GOPS** | 自研 DPAS 711.1 | 58.5% |

> 同 dtype 换个形状差很多：bf16 在 `4096×4096×16384` 只有 **185.4**，
> fp16 在 `4096×4096×16384` 只有 **183.4** —— 长 K 形状掉 ~18%，
> 与 `docs/precision-support.md` 记录的 **16384³ 塔口**同源。

vector（64 Mi 元素）：

| op | fp32 GB/s | bf16 GB/s |
|---|---:|---:|
| add | 528.0 | 530.0 |
| mul | 526.2 | 528.7 |
| relu | 406.9 | 411.6 |

> ⚠ vector 只有 ~530 GB/s，而 `03-memory-bandwidth` 里 BabelStream copy 能到 ~900 GB/s。
> 原因是这里只用了 `n_elem = 2^26`（fp32 = 256 MiB）且**没有 warmup 到 L2 稳定**，
> 属于「torch 元素级算子开销」而不是带宽极限 —— 带宽口径请以 `03` 为准。

### 4.6 主频 / 功耗

| 项 | 值 |
|---|---|
| `gt_min_freq_mhz` | 1550 |
| `gt_max_freq_mhz` | 1550 |
| `gt_boost_freq_mhz` | 1550 |
| `gt_RP0 / RP1 / RPn` | 1550 / 1000 / 200 |
| `gt_cur_freq_mhz` | 1550（空载/满载同） |
| `gt_act_freq_mhz`（空载 / 满载） | **0**（读不到）／ **1550** |
| `xpu-smi` GPU Utilization（满载） | **100%** |
| `xpu-smi` GPU Power（满载） | **171 W**（上限 300 W → **不节流**） |

**min == max** ⇒ 频率被**锁定**，没有 boost 空间；满载不降频、不撞功耗墙。

---

## 5. 坑与注意事项

### 5.1 ★ 最重要的坑：占用率不足时「工作量翻倍、耗时不变」

**现象**：`xmx_peak bf16`，固定 `outer=8192/it=128/nacc=4`：

```
g=28672 (4 sg/EU, 1792 sub-group) -> 0.173508 s   dpas_count = 7.52e9
g=57344 (8 sg/EU, 3584 sub-group) -> 0.173508 s   dpas_count = 1.50e10   ← 计数翻倍，耗时一样
```

按 FLOP 计数除出来的吞吐凭空翻倍 → 很容易得出两个**错误**结论之一：
①「编译器把循环消掉了」；②「这台卡有 700 TFLOPS」。

**真相**（用对照实验证明，固定总 DPAS、只改 work-group 数）：

```
总 DPAS 7.516e9，g=28672 (4 sg/EU) -> 0.173517 s  = 177 TFLOPS
总 DPAS 7.516e9，g=57344 (8 sg/EU) -> 0.086764 s  = 355 TFLOPS   ← 等量工作，快一倍
```

耗时**确实**随 work-group 数变 → 计数没问题，问题是**延迟受限**：
XMX 流水线要每 EU ≥8 个并发 sub-group 才打满，否则同样的活要花近一倍时间。

**判读纪律**（本目录已固化为自动检查）：

1. 测峰值前**必须先把 work-group 数扫过拐点**，否则测到的是延迟；
2. 拐点之上耗时对工作量必须**严格线性** —— 用它做自校验：

   | 探针 | 低点 | 高点 | 比值 |
   |---|---|---|---|
   | ALU | g=114688 → 1.2133 s | g=229376 → 2.4234 s | **×2.00** ✓ |
   | XMX | occ8 → 0.1735 s | occ16 → 0.3469 s | **×2.00** ✓ |

3. 拐点之下「等量工作同样耗时」是**正常现象**，不是 bug。

`run_bench.py` 把这三件事都做成了记录：`saturation_knee`、`linearity_check`、
`linearity_check_below_knee`（故意跨拐点留档）。

### 5.2 ★ 第二重要的坑：标称公式与实际不符（口径冲突）

| 口径 | fp32 ALU 峰值 | 说明 |
|---|---:|---|
| 公式标称 `448 EU × 16 lane × 2 × 1.55 GHz` | **22.22 TFLOPS** | 与 clinfo 的 448 CU 一致 |
| oneDNN GEMM 实测 | **22.1 TFLOPS** | 恰好 100% 公式值 |
| 自研纯 FMA 探针实测 | **50.75 TFLOPS** | **2.28×** |

三者**不可能同时为真**。主频已锁定 1550 MHz、满载不降频 → 不能拿主频解释。
候选解释（**均未证实**）：

- (a) 本 ES 部件的实际 EU 数与 clinfo 报告的 448 不符（实测 > 报告）；
- (b) EU 内 FP32 通道宽度 >16（`448 × 16` 模型低估）；
- (c) 探针的 FLOP 计数口径偏低（但 VEC/ACC/UNROLL 都是显式参数，
      且 `logistic map` 递推链上无冗余可消，**(c) 的可能性最低**）。

**判读纪律：以 oneDNN（22.1 TFLOPS）为可信下界，自研探针（50.75）为待验证上界。**
需要 `docs/TODO/07-profiling.md` 的硬件计数器（EU active / XMX pipe util）才能裁决。
**本目录不修改** `common/bench.py:alu_tflops()` 与 `docs/` 中的 22.22 标称值，
只把差异显式写进结论。

### 5.3 其它坑

| 坑 | 现象 | 处理 |
|---|---|---|
| `joint_matrix` 头文件路径 | `experimental/matrix/` 不存在 | 用 `<sycl/ext/oneapi/matrix/matrix.hpp>` |
| `joint_matrix_load` 拒收裸指针 | `could not match 'multi_ptr<...>' against 'T *'` | 用 `sycl::multi_ptr<T, global_space>(ptr)` |
| 累加器被 DCE | 只 store `acc[0]` 时 `acc[1..]` 被优化掉 | **所有**累加器都要 store 到互不重叠的地址 |
| 运行期循环 + 不变操作数 | 编译器把整段消掉 | 内层用**编译期** `IT` + `#pragma unroll`，外层才是运行期循环 |
| `icpx` 无法产出设备 IR | `-S -emit-llvm` → `IR output is not supported`；`-fno-sycl-device-code-split` 未识别 | 无法用 SPIR-V/PTX 反汇编验证，只能靠 §5.1 的对照实验 |
| SYCL 读 GPU 时钟 | `clock_scope::device` 不支持（aspect `ext_oneapi_clock_device` 缺失） | `alu_clock_probe.cpp` 是**死路**，只留档；改用 sysfs |
| `gt_act_freq_mhz` 读 0 | 空闲未采样时 | 必须在**满载进行中**采样 |
| `xpu-smi dump` 挂死 | 打印表头后卡住，`Terminated` rc=143 | 只用 `xpu-smi stats -d 0`（~1.5 s） |
| `xpu-smi` 内存计数器是假的 | `GPU Memory Read/Write (kB/s)` 恒为 ~576 kB/s | 不要引用；带宽以 `03` 的 BabelStream 为准 |
| `torch.matmul` 对 int8 返回 int8 | 溢出 | 用 `torch._int_mm`（int32 累加） |
| `python3 -m venv` 失败 | `ensurepip` 不可用 | 用 `uv venv --python /usr/bin/python3.13 --system-site-packages` |
| `source setvars.sh` 返回 rc=3 | `set -u` 下还报错 | 先 `set +u`，且**不要**用 `&&` 串联 |

---

## 6. 判读标准对照（vs `docs/TODO/02-compute-peak.md`）

| TODO 中的判读标准 | 本次实测 | 结论 |
|---|---|---|
| fp32 ALU 达到标称 22.2 TFLOPS 的 ≥90% | oneDNN 22.1（**100%**）/ 自研 50.75 | ✅ 达成（但两口径冲突，见 §5.2） |
| bf16 XMX ≥ 150 TFLOPS | 自研 355 / oneDNN 227 | ✅ 远超 |
| int8 XMX ≥ 300 TOPS | 自研 710 / oneDNN 427 | ✅ 远超 |
| fp64 ≥ 标称的 90% | 自研 51.9 TFLOPS（与 fp32 同速） | ✅ 达成；**说明 fp64:fp32 ≠ 1:2**，文档里「FP64 = FP32/2」的假设需要修正 |
| 实测 ≥ 标称值的 90% 视为通过 | fp32 自研 228% 反而「超纲」 | ⚠ 触发 §5.2 的复核流程 |
| 满载主频不低于标称 90% | 锁定 1550 MHz、无降频 | ✅ |
| 满载不撞功耗墙 | 远低于 300 W | ✅ |

---

## 7. 与其它目录的衔接

| 本目录的发现 | 影响 |
|---|---|
| fp64 与 fp32 **同速**（≈51 TFLOPS） | `docs/precision-support.md` 的 `FP64_ALU_RATIO = 0.78` 应复核为 ≈1.0（但口径冲突未解前不动） |
| bf16 = fp16 = 355 TFLOPS，int8 = 2× | 佐证 `05-ai-dl` 的 XMX 结论（bf16 237 / int8 400 TOPS） |
| vector 只有 ~530 GB/s | 那是 torch 元素级算子下限，**不是**带宽峰值 → 带宽看 `03-memory-bandwidth` |
| XMX 需要 ≥8 sg/EU 才打满 | 给 `05-ai-dl` 的调优留了空间：小 batch/小 shape 会掉进延迟受限区 |
| oneDNN 只到自研 DPAS 的 60~64% | 说明 oneDNN 的 GEMM 还有 ~35% 的调优空间（或受 L2/调度限制） |
| 需要硬件计数器 | 依赖 [`docs/TODO/07-profiling.md`](../../docs/TODO/07-profiling.md)（VTune / PTI） |

---

## 8. 未做 / 待补充

| 项 | 原因 |
|---|---|
| 硬件计数器佐证（EU active、XMX pipe util、FP32 pipe util） | 属 `07-profiling`；本轮未接入 VTune/PTI |
| `ze_peak`（Level Zero 官方向量/带宽峰值） | 未构建。**它不是 XMX 基准** —— `ze_peak` 是 clpeak 移植，只有向量测试（`sp/dp/hp/int_compute`、`global_bw`、`transfer_bw`、`kernel_lat`），**没有 DPAS/XMX**，产不出本目录最关键的 355 TFLOPS / 710 TOPS / 占用率拐点。<br>但它可作为 **FP32 向量峰值的第三方独立仲裁**（`22.13 vs 50.75` 冲突）—— 出处是 `oneapi-src/level-zero-tests/perf_tests/ze_peak`（**不是** `intel/compute-runtime`），零外部依赖、约 10 分钟可补做；本机 `git clone` 报 `GnuTLS recv error (-110)` 需逐文件抓取。完整说明见 [`docs/TODO/02-compute-peak.md`](../../docs/TODO/02-compute-peak.md) §3.6 |
| `benchdnn`（oneDNN 微基准） | 本机未安装 |
| nvfp4/mxfp8 等低精度**峰值** | 见 `docs/precision-support.md`：PVC 上全是软件回退，比 bf16 慢，无峰值可言 |
| TF32 / FP6 | torch 无对应 dtype，无法测试 |
| 双卡并发算力（2×448 EU） | 属 `04-interconnect-xelink` 与 `06-hpc-apps` 的范畴 |
| 反汇编验证（SPIR-V / ISA） | `icpx` 在本机无法产出设备 IR（见 §5.3） |
