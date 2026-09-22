# Intel XPU 服务器（`hwt`）文档

本文档集记录对 Intel XPU 服务器 `hwt` 的配置勘察结果与后续性能测试计划。

> 勘察日期：2026-09-21
> 勘察方式：`xpu-smi`、`lscpu`/`free`/`numactl`、`clinfo`、`sycl-ls`、oneAPI 目录盘点、`pip list`、`torch.xpu`

---

## 快速事实卡

| 项目 | 值 |
|---|---|
| 主机名 | `hwt` |
| OS / 内核 | Ubuntu 25.04 / 6.14.0-37-generic |
| CPU | Intel 预生产 **ES**，72 核 / 144 线程，2.7–3.9 GHz，单路，1 NUMA 节点 |
| 主机内存 | **45 GiB**（可用约 41 GiB）+ 8 GiB Swap |
| GPU | **2 × Intel® Data Center GPU Max 1100**（Ponte Vecchio，Production ES） |
| 单卡规格 | 1 tile / 56 Xe-core / 448 EU / SIMD16 / 1550 MHz / 48 GiB HBM（ECC 开） |
| 卡间互连 | **Xe Link XL24**（6 × 4 lane 全直连，无 MDF 中转） |
| 主机↔卡 | PCIe 5.0 x16（每卡） |
| 功耗上限 | 300 W/卡（可调范围 150–300 W） |
| 驱动 | i915 `I915_25.2.57_PSB_250224.65`；Level Zero 1.24.0；xpu-smi 1.2.43 |
| 软件栈 | oneAPI 2026.1（+2025.3/2026.0 并存）、VTune 2026.4、Advisor 2026.0、Intel MPI 2021.18 |
| AI 框架 | PyTorch **2.14.0+xpu**（`torch.xpu` 正常识别 2 卡）、triton-xpu 3.8.0 |

---

## 文档索引

### 配置勘察
| 文档 | 内容 |
|---|---|
| [`hardware.md`](./hardware.md) | 主机与 GPU 硬件概况（CPU / 内存 / GPU / 固件 / 功耗） |
| [`interconnect.md`](./interconnect.md) | 互联情况：Xe Link 拓扑、PCIe 拓扑、NUMA、实测关注点 |
| [`software-stack.md`](./software-stack.md) | 驱动、Level Zero、oneAPI、Python AI 栈、已装工具盘点 |
| [`caveats.md`](./caveats.md) | 风险与注意事项（ES 样片、内存倒挂、未标定 Xe Link 等） |

### 实测结果
| 文档 | 内容 |
|---|---|
| [`precision-support.md`](./precision-support.md) | **数值格式支持矩阵与各精度实测吞吐**（vector / matmul+XMX，含 FP8/MXFP4/NVFP4、INT4、失败原因原文）；§7 含**准确度与累加精度专项**（XMX 累加器为 FP32，无 fp16-acc 档） |

### 测试结论
| 文档 | 内容 |
|---|---|
| [`Conclusion/02-compute-peak/README.md`](./Conclusion/02-compute-peak/README.md) | **② 计算峰值结论**：XMX/DPAS **355 TFLOPS**（bf16，为"标称 176"的 2 倍）、ALU **22.22 TFLOPS【✅ 已裁定】**（oneDNN 99.6% + 第三方 `ze_peak` 98.4% 双重确认；自研探针 50.75 已撤回）、FP64 = **0.735~0.78 × FP32**、占用率拐点扫描、向量宽度扫描、`torch`/oneDNN 交叉验证、**长时满载热/功耗降额：温度 101 °C、功耗 305~330 W（越过 300 W 上限）** |
| [`Conclusion/03-memory-bandwidth/README.md`](./Conclusion/03-memory-bandwidth/README.md) | **③ 显存带宽结论**：HBM copy **900 GB/s**（73% 规格）、双卡完美线性 **1.68 TB/s**、PCIe pinned **31.9 GB/s**、**主机 DRAM 仅 39.6 GB/s**（真瓶颈）；L2=192 MB / L3=432 MB 的"cache 陷阱" |
| [`Conclusion/04-interconnect-xelink/README.md`](./Conclusion/04-interconnect-xelink/README.md) | **④ 互连（Xe Link）结论**：卡间 **95.5 GB/s**（PCIe Gen5 的 1.52×，但仅名义 318 GB/s 的 30%，`Not Calibrated` 是首要嫌疑）；GPU-aware MPI 仅 43.7 GB/s（裸 L0 的 46%）；xccl broadcast **94.6 ≈ 裸带宽** |
| [`Conclusion/05-ai-dl/05-conclusion.md`](./Conclusion/05-ai-dl/05-conclusion.md) | **⑤ AI/DL 总体结论**：测试目标逐条回答、全部关键数字总表、判读标准裁定、3 个反直觉发现、建议清单、诚实性声明 |
| [`Conclusion/05-ai-dl/01-training-throughput.md`](./Conclusion/05-ai-dl/01-training-throughput.md) | **训练吞吐**：ResNet-50 / BERT（base + large）的 dtype × batch × memory_format / seq_len 扫描；BF16 是否启用 XMX 的验证 |
| [`Conclusion/05-ai-dl/02-inference.md`](./Conclusion/05-ai-dl/02-inference.md) | **推理**：LLM prefill / decode / TTFT / 显存占用模型；decode 21.6 ms/token 的成因 |
| [`Conclusion/05-ai-dl/03-scaling.md`](./Conclusion/05-ai-dl/03-scaling.md) | **扩展性**：2 卡 DDP 三配置对照、扩展效率、通信成本分解、Xe Link allreduce 微基准 |
| [`Conclusion/05-ai-dl/04-bottleneck.md`](./Conclusion/05-ai-dl/04-bottleneck.md) | **瓶颈诊断**：训练 kernel 成分（BN 38.1% / Conv 32.6% / Elem 24.9%）、Amdahl 模型、LLM decode 固定开销、主机侧瓶颈 |

> 测试代码与原始数据：
> - `benchmark/02-compute-peak/`（`run_bench.py` 4 suite + `sycl/` 自研 SYCL 探针 + `results/`）
> - `benchmark/03-memory-bandwidth/`（`run_bench.py` 6 suite + BabelStream + `probes/` + `results/`）
> - `benchmark/04-interconnect-xelink/`（`run_bench.py` 5 suite + `probes/p2p_probe.cpp`（Level Zero）+ `results/`）
> - `benchmark/05-ai-dl/`（`run_bench.py` + `xpu_bench/` 13 suite、`diagnostics/` 7 个定向探针、`results/`）

### 测试计划
| 文档 | 内容 |
|---|---|
| [`TODO/README.md`](./TODO/README.md) | 测试计划总览：分类、优先级、依赖关系、推荐执行顺序、命令速查 |
| [`TODO/01-health-stress.md`](./TODO/01-health-stress.md) | 硬件健康 / 稳定性 / 压力测试 |
| [`TODO/02-compute-peak.md`](./TODO/02-compute-peak.md) | 算力峰值测试（FP32 / FP64 / INT / XMX） → **已完成，结论见 [`Conclusion/02-compute-peak/`](./Conclusion/02-compute-peak/)** |
| [`TODO/03-memory-bandwidth.md`](./TODO/03-memory-bandwidth.md) | 显存带宽测试（HBM） → **已完成，结论见 [`Conclusion/03-memory-bandwidth/`](./Conclusion/03-memory-bandwidth/)** |
| [`TODO/04-interconnect-xelink.md`](./TODO/04-interconnect-xelink.md) | 互连测试（Xe Link / GPU-aware MPI / oneCCL） → **已完成，结论见 [`Conclusion/04-interconnect-xelink/`](./Conclusion/04-interconnect-xelink/)** |
| [`TODO/05-ai-dl.md`](./TODO/05-ai-dl.md) | AI / 深度学习：训练与推理吞吐、算子基准 → **已完成，结论见 [`Conclusion/05-ai-dl/`](./Conclusion/05-ai-dl/)** |
| [`TODO/06-hpc-apps.md`](./TODO/06-hpc-apps.md) | HPC 应用基准（GROMACS / LAMMPS / HPL / HPCG 等） |
| [`TODO/07-profiling.md`](./TODO/07-profiling.md) | Profiling 与瓶颈定位（VTune / Advisor / PTI / 硬件计数） |
| [`TODO/08-power-efficiency.md`](./TODO/08-power-efficiency.md) | 功耗、能效、频率与调度策略 |

---

## 环境准备（所有测试前必做）

```bash
# 加载 oneAPI 环境（含编译器、MPI、MKL、oneCCL 等）
source /opt/intel/oneapi/setvars.sh

# 确认当前实际生效的版本（机器上多版本并存！）
which icpx sycl-ls mpirun
sycl-ls

# 确认 GPU 可见
xpu-smi discovery
clinfo -l
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
```
