# `alu_peak` 探针可信度复核（2026-09-22）

本目录是**一次判定性实验**的留档：用来回答「FP32 向量峰值到底是 **22.22** 还是
**50.75 TFLOPS**」这个在 `docs/Conclusion/02-compute-peak/README.md` §3.2 长期挂着
`🔴 未解决` 的口径冲突。

**结论（先说）：以 22.22 TFLOPS 为准；`sycl/alu_peak.cpp` 的绝对值 50.75 撤回。**

本目录里的可执行文件是**同一份源码的不同编译期配置**（用 `-DALU_*` 覆盖宏），
因此可以当作"对照实验"复现。

---

## 1. 被质疑的数字

历史结果（`results/bench_20260922-192831.*`、`bench_20260922-194139.*`）里记录：

| 项 | 值 | 出处 |
|---|---|---|
| 公式标称 FP32 向量峰值 | **22.22 TFLOPS** | `448 EU × 16 lane × 2 FLOP × 1.55 GHz` |
| oneDNN fp32 GEMM 实测 | **22.13 TFLOPS** | `run_bench.py --suite torch` |
| 自研 SYCL FMA 探针实测 | **50.75 TFLOPS** | `sycl/alu_peak.cpp`，`global=114688`, `iters=2²²` |

50.75 / 22.22 = **2.28×**。三个数字不能同时为真。

## 2. 实验 A：先确认 50.75 可复现（排除环境漂移）

```bash
source /opt/intel/oneapi/setvars.sh
cd sycl/exp
icpx -fsycl -O3 -ffp-contract=fast -DALU_UNROLL=4 -o alu_u4 ../alu_peak.cpp
./alu_u4 fp32 114688 4194304 2 3 | grep RESULT
```

实测输出：

```json
{"dtype":"fp32","unit":"GFLOPS","global_size":114688,"iters":4194304,"seconds":4.853005,"value":50750.13}
```

`seconds=4.853005` 与历史记录的 `4.853` **逐位一致** ⇒ 不是环境漂移，是同一现象。

## 3. 实验 B：改 `UNROLL` 看"计数的工作"是否真的被执行

```bash
icpx -fsycl -O3 -ffp-contract=fast -DALU_UNROLL=1 -o alu_u1 ../alu_peak.cpp
./alu_u1 fp32 114688 4194304 2 3 | grep RESULT
```

| 配置 | 每次内层迭代的 FMA 数 | 计时 (s) |
|---|---|---|
| `UNROLL=1` | 8 条（ACC=8） | **1.501** |
| `UNROLL=4` | 32 条（ACC=8） | **4.853** |

工作量 ×4，耗时只 ×3.23。代入 `T = N·(k·U + o)`（`k`=每条 FMA 成本，`o`=每次
迭代的循环开销）解得 `o ≈ 0.34·k` —— 即循环开销只占约 1/4 条 FMA 的量级。
**这一步是"排除项"**：它说明 UNROLL 的 4 倍工作量基本是真实执行的，**没有**被编译器
按比例消掉。所以 2.28× 的偏差不是"消循环"，只能是 **FLOP 计数模型与生成代码不符**。

## 4. 实验 C：`VEC` 扫描（历史数据）——探针分辨不出位宽

历史 `vec_width` 扫描（`results/bench_20260922-194139.md`）：

| VEC | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| TFLOPS | 34.8 | 45.5 | 52.8 | 50.7 | 51.3 |

注意 **VEC=1 就有 34.8 TFLOPS**。但标量路径的硬件上限是
`448 EU × 16 lane × 2 FLOP × 1.55 GHz = 22.22` —— 一个每 lane 一条标量 FMA 的
kernel **不可能超过 22.22**。单这一条就已经证伪。

（VEC≥4 之后完全拉平也说明：`sycl::vec<T,VEC>` 超过 8 宽以后没有额外收益，
即编译器最多给到 SIMD8。）

## 5. 实验 D：ISA 层证据（`IGC_ShaderDumpEnable=1`）

```bash
rm -rf /tmp/IntelIGC
IGC_ShaderDumpEnable=1 ./alu_u4 fp32 114688 1048576 1 1
ls /tmp/IntelIGC/*/          # 会得到 .asm / .isaasm / .visa.ll / HardwareCaps.txt
```

得到的关键事实：

**(1) IGC 自报的硬件上限（`HardwareCaps.txt`）**

```
EUCount        = 448
ThreadCount    = 3584      # = 448 × 8，即 8 线程/EU
SliceCount     = 1
SubSliceCount  = 56
MaxEuPerSubSlice = 8
```

⇒ **EU 数确实是 448**，与公式一致。没有"实际 EU 更多"的空间。

**(2) FP32 kernel 的循环体（`*_simd32_entry_0001.asm`，kernel = `AluKernelIfE`）**

- 主循环 = 第 530 行 `_0_012:` 到第 760 行 `(W&f0.0) jmpi _0_012`，共 231 条指令。
- 其中 **209 条是 `mad`**，且 `grep -oE '\$[0-9]+' | sort -u` = **209 个互不相同的
  源操作** —— 没有任何一条是重复展开。
- 每条 `mad` 的形态是 **1 宽、标量寄存器**：

```
(W)     mad (1|M0)   r52.4<1>:f   r5.7<0;0>:f   r2.6<0;0>:f   -r2.6<0>:f
```

即 `acc = 1 - acc*acc` 被**完全标量化**成"一条指令算一个标量元素"。

⇒ 探针源码声明的 FLOP 模型是"每次内层迭代 32 条 **向量** FMA（`ACC=8 × UNROLL=4`，
每条 `VEC=8` 宽）"，但 IGC 实际生成的是**约 209 条一元素标量 mad**。
**生成代码与 FLOP 模型不符** —— 这正是 2.28× 偏差的来源。探针没有做错算术，
但它"以为"自己每迭代发 32 条向量 FMA，硬件上跑的却是另一套指令流。

## 6. 实验 E：**决定性**证据 —— 探针分辨不出 dtype 的位宽比

这是最干净、最不依赖 ISA 解读方式的一条。Xe-HPC 的 ALU 位宽比是硬架构事实：
`fp16 = 2× fp32`、`fp64 = ½~¾× fp32`。任何"真的压到 ALU 上限"的探针都必须复现这些比。

| 实现 | fp16/fp32 | fp64/fp32 | 是否自洽 |
|---|---|---|---|
| **ze_peak**（第三方，Intel 官方仓库） | **1.98** | **0.735** | ✅ 完全符合架构 |
| **自研 `alu_peak`** | **1.04** | **1.02** | ❌ 分辨不出 dtype |

ze_peak 的原始数字：`hp_compute half4 = 43,381.3 GFLOPS`、`sp_compute float4 = 21,871.6`、
`dp_compute double4 = 16,074.0`。

**一个给出 fp16 ≈ fp32 ≈ fp64 的"峰值探针"，测的不是 FMA 吞吐。** 它的 50.75
是数值 artefact，必须撤回。

## 7. 最终裁定

| 证据 | 指向 |
|---|---|
| 公式 `448×16×2×1.55G` = 22.22 | 22.22 |
| IGC `HardwareCaps.txt`: EUCount=448, 8 线程/EU | 22.22（EU 数无争议） |
| oneDNN fp32 GEMM = 22.13（99.6%） | 22.22 |
| **ze_peak sp_compute = 21.87（98.4%）** | **22.22** |
| ze_peak 位宽比 1.98 / 0.735（符合架构） | 22.22（ze_peak 可信、探针不可信） |
| 自研探针 = 50.75（228%） | ❌ 撤回 |
| 自研探针位宽比 1.04 / 1.02 | ❌ 探针不是吞吐测量 |
| 探针 VEC=1 就有 34.8 > 22.22 上限 | ❌ 不可能 |
| ISA：vec8 FMA 被标量化成 1 元素 mad | 解释偏差来源 |

**→ FP32 向量峰值 = 22.22 TFLOPS（公式值）。**
**→ 附带修正：`fp64/fp32 = 0.734`（ze_peak），这是「FP64 ≠ FP32/2」的第三次独立确认**
（前两次：torch GEMM 0.78 / 0.77）。

## 8. 复现清单

```bash
source /opt/intel/oneapi/setvars.sh
cd /root/workspace/benchmark/02-compute-peak/sycl/exp
icpx -fsycl -O3 -ffp-contract=fast -DALU_UNROLL=4 -o alu_u4 ../alu_peak.cpp
icpx -fsycl -O3 -ffp-contract=fast -DALU_UNROLL=1 -o alu_u1 ../alu_peak.cpp
icpx -fsycl -O3 -ffp-contract=fast -DALU_VEC=1    -o alu_v1 ../alu_peak.cpp
icpx -fsycl -O3 -ffp-contract=fast -DALU_ACC=4    -o alu_acc4 ../alu_peak.cpp

./alu_u4  fp32 114688 4194304 2 3   # → 4.853 s / 50750 GFLOPS（2.28× 公式值）
./alu_u1  fp32 114688 4194304 2 3   # → 1.501 s / 41016 GFLOPS
rm -rf /tmp/IntelIGC
IGC_ShaderDumpEnable=1 ./alu_u4 fp32 114688 1048576 1 1
grep -c ' mad ' /tmp/IntelIGC/*/*_simd32_entry_0001.asm   # 循环体内 208 条 mad
```

**注意**：`global=7168`（= 1 work-item / 硬件 lane）会掉进延迟受限区，
读数只有 ~10 TFLOPS；`global=114688`（= 16 work-item / lane）才在饱和区。
比较历史数字时必须用后者，否则会误判为"数字不可复现"。

## 9. 本目录的产物

| 文件 | 说明 |
|---|---|
| `alu_u4` | `ALU_UNROLL=4`（= 历史默认配置），复现 50.75 |
| `alu_u1` | `ALU_UNROLL=1`，用于 UNROLL 线性度对照 |
| `alu_v1` | `ALU_VEC=1`，用于"VEC=1 就已超上限"证伪 |
| `alu_acc4` | `ALU_ACC=4`，额外对照 |

这几个二进制是**实验留档**，不是交付物；真正的探针源码仍是 `../alu_peak.cpp`。
