# 02 · 算力峰值（ALU / XMX-DPAS）基准报告

- tag: `ze_peak_arbiter`
- time: 2026-09-22T22:52:28
- host: hwt / kernel 6.14.0-37-generic / python 3.13.3
- ZE_AFFINITY_MASK: `(unset)`
- 标称 FP32 ALU 峰值（公式）: 22.22 TFLOPS

## 关键结论速览

- `ze_peak`（**第三方**：`oneapi-src/level-zero-tests/perf_tests/ze_peak`，clpeak 的 Level Zero 移植）本轮已成功构建并运行。它只测**向量/SIMD**路径，**没有 XMX/DPAS**，因此不能替代自研 `xmx_peak`；它的价值在于**仲裁**：它给出 FP32 向量峰值 21.87 TFLOPS = 公式值 22.22 的 **98.4%**，与 oneDNN fp32 GEMM 的 22.13（99.6%）一起，**互相独立地确认了 22.22 口径**，把自研探针的 50.75（2.28×）判为探针侧 artefact。另一收获：ze_peak 的 fp64/fp32 = 16.07/21.87 = **0.734**，第三次独立确认「FP64 ≠ FP32/2」的修正（前两次：torch GEMM 0.78、0.77）。
- **重复性与时长口径**：`-a -i 3 -w 1` 的短跑跨 **2 卡 × 2 次** 完全一致（fp32 21871.6/21873.2/21872.1/21871.6；fp64 16074.1/16074.7/16074.0/16074.2），但 `-i 50 -w 10` 的**长跑整轮**（单卡约 20 min）会因 **DVFS 漂移**（`gt_act_freq_mhz`实测出现 1350/1250/800/350 MHz）在 fp64 / int32 段偏低（fp64 16074→14279，−11%）。fp32 / fp16 段在漂移发生前已测完，故跨卡跨次一致。**引用绝对值请用短跑或分段结果。** 长跑（`-i 50 -w 10`，单卡约 20 min）期间 GPU 持续满载会触发热/功耗降额：实测温度 92 → **101 °C**、功耗 305 → **330 W**（越过 300 W 名义上限），而 `gt_cur/max/min_freq_mhz` 始终报 1550 的**请求值**（唯一位于负载变化的 `gt_act_freq_mhz` 噪声极大，只能作定性证据）。降额**逐段推进**：同一 section 内按向量宽度顺序（=时间顺序）单调衰减；且**起始热态决定整轮水平** —— dev1 紧接前几轮再跑一次全量（起始 ~94 °C）时，连第一个 section `global_bw` 都掉到 307 GB/s（冷态 688），sp 21872→18497，dp 12213→11492，int 3640→3533。**判读纪律：绝对值引用短跑（`-a -i 3 -w 1`）或冷启分段结果，长跑整轮值仅供参考。**
- 公式标称 FP32 ALU 峰值 = 448 EU × 16 lane × 2 × 1.55 GHz = **22.22 TFLOPS**；XMX 公式标称 = 256 MAC/clk/EU → **356 TFLOPS (bf16)** / **711 GOPS (int8)**。所有自研探针均已通过「饱和区 ×2 工作量 → ×2 耗时」自校验。

## ze_peak

| 测试项 | 参数 | 均值 | 最好 | 单位 | 派指标 | 状态 | 备注 |
|---|---|---:|---:|---|---|---|---|
| `dev0.device` | device=0 | None | None | - | coreClockRate_mhz=1550, deviceId=0bda, subdeviceId=0, isSubdevice=FALSE | ok | Intel(R) Data Center GPU Max 1100；UUID=00000000-0000-0000-74d8-64836dad91a4 |
| `dev0.fp32.float` | device=0, test=sp_compute, kernel=float | 21,843.3 | 21,843.3 | gflops | gflops=21,843.3 | ok | GFLOPS |
| `dev0.fp32.float2` | device=0, test=sp_compute, kernel=float2 | 21,820.9 | 21,820.9 | gflops | gflops=21,820.9 | ok | GFLOPS |
| `dev0.fp32.float4` | device=0, test=sp_compute, kernel=float4 | 21,871.6 | 21,871.6 | gflops | gflops=21,871.6 | ok | GFLOPS |
| `dev0.fp32.float8` | device=0, test=sp_compute, kernel=float8 | 21,759.7 | 21,759.7 | gflops | gflops=21,759.7 | ok | GFLOPS |
| `dev0.fp32.float16` | device=0, test=sp_compute, kernel=float16 | 21,533.1 | 21,533.1 | gflops | gflops=21,533.1 | ok | GFLOPS |
| `dev0.fp64.double` | device=0, test=dp_compute, kernel=double | 16,005.9 | 16,005.9 | gflops | gflops=16,005.9 | ok | GFLOPS |
| `dev0.fp64.double2` | device=0, test=dp_compute, kernel=double2 | 15,904.7 | 15,904.7 | gflops | gflops=15,904.7 | ok | GFLOPS |
| `dev0.fp64.double4` | device=0, test=dp_compute, kernel=double4 | 16,074.0 | 16,074.0 | gflops | gflops=16,074.0 | ok | GFLOPS |
| `dev0.fp64.double8` | device=0, test=dp_compute, kernel=double8 | 15,792.6 | 15,792.6 | gflops | gflops=15,792.6 | ok | GFLOPS |
| `dev0.fp64.double16` | device=0, test=dp_compute, kernel=double16 | 13,978.7 | 13,978.7 | gflops | gflops=13,978.7 | ok | GFLOPS |
| `dev0.fp16.half` | device=0, test=hp_compute, kernel=half | 34,545.5 | 34,545.5 | gflops | gflops=34,545.5 | ok | GFLOPS |
| `dev0.fp16.half2` | device=0, test=hp_compute, kernel=half2 | 43,118.3 | 43,118.3 | gflops | gflops=43,118.3 | ok | GFLOPS |
| `dev0.fp16.half4` | device=0, test=hp_compute, kernel=half4 | 43,381.3 | 43,381.3 | gflops | gflops=43,381.3 | ok | GFLOPS |
| `dev0.fp16.half8` | device=0, test=hp_compute, kernel=half8 | 43,188.7 | 43,188.7 | gflops | gflops=43,188.7 | ok | GFLOPS |
| `dev0.fp16.half16` | device=0, test=hp_compute, kernel=half16 | 42,842.5 | 42,842.5 | gflops | gflops=42,842.5 | ok | GFLOPS |
| `dev0.int32.int` | device=0, test=int_compute, kernel=int | 6,333.5 | 6,333.5 | gflops | gflops=6,333.5 | ok | GFLOPS |
| `dev0.int32.int2` | device=0, test=int_compute, kernel=int2 | 6,342.5 | 6,342.5 | gflops | gflops=6,342.5 | ok | GFLOPS |
| `dev0.int32.int4` | device=0, test=int_compute, kernel=int4 | 6,333.8 | 6,333.8 | gflops | gflops=6,333.8 | ok | GFLOPS |
| `dev0.int32.int8` | device=0, test=int_compute, kernel=int8 | 4,840.5 | 4,840.5 | gflops | gflops=4,840.5 | ok | GFLOPS |
| `dev0.int32.int16` | device=0, test=int_compute, kernel=int16 | 5,399.3 | 5,399.3 | gflops | gflops=5,399.3 | ok | GFLOPS |
| `dev0.global_bw.float` | device=0, kernel=float | 688.7 | 688.7 | gbps | gbps=688.7 | ok | GB/s |
| `dev0.global_bw.float2` | device=0, kernel=float2 | 685.1 | 685.1 | gbps | gbps=685.1 | ok | GB/s |
| `dev0.global_bw.float4` | device=0, kernel=float4 | 661.8 | 661.8 | gbps | gbps=661.8 | ok | GB/s |
| `dev0.global_bw.float8` | device=0, kernel=float8 | 674 | 674 | gbps | gbps=674 | ok | GB/s |
| `dev0.global_bw.float16` | device=0, kernel=float16 | 678.3 | 678.3 | gbps | gbps=678.3 | ok | GB/s |
| `dev0.transfer_bw.enqueueWriteBuffer` | device=0 | 39.02 | 39.02 | gbps | gbps=39.02 | ok | GB/s |
| `dev0.transfer_bw.enqueueReadBuffer` | device=0 | 53.04 | 53.04 | gbps | gbps=53.04 | ok | GB/s |
| `dev0.transfer_bw.GPU_Copy_Host_to_Shared_Memory` | device=0 | 39.18 | 39.18 | gbps | gbps=39.18 | ok | GB/s |
| `dev0.transfer_bw.GPU_Copy_Shared_Memory_to_Host` | device=0 | 53.04 | 53.04 | gbps | gbps=53.04 | ok | GB/s |
| `dev0.transfer_bw.System_Memory_Copy_to_Shared_Memory` | device=0 | 8.09 | 8.09 | gbps | gbps=8.09 | ok | GB/s |
| `dev0.transfer_bw.System_Memory_Copy_from_Shared_Memory` | device=0 | 8.21 | 8.21 | gbps | gbps=8.21 | ok | GB/s |
| `dev0.kernel_lat.Kernel_launch_latency` | device=0 | 5.491 | 5.491 | us | us=5.491 | ok | us |
| `dev0.kernel_lat.Kernel_launch_latency_with_Immediate_Command_List` | device=0 | 5.496 | 5.496 | us | us=5.496 | ok | us |
| `dev0.kernel_lat.Kernel_duration` | device=0 | 15.69 | 15.69 | us | us=15.69 | ok | us |
| `crosscheck.dev0` | third_party=oneapi-src/level-zero-tests/ze_peak, ours=sycl/alu_peak.cpp + BabelStream | 21.87 | 21.87 | tflops | ze_peak_sp_compute_tflops=21.87, ze_peak_hp_compute_tflops=43.38, ze_peak_dp_compute_tflops=16.07, ze_peak_hp_over_sp=1.98, ze_peak_dp_over_sp=0.735, our_probe_sp_tflops=50.75, our_probe_hp_over_sp=1.04, our_probe_dp_over_sp=1.02, oneDNN_gemm_tflops=22.13, formula_nominal_tflops=22.22, ze_peak_vs_formula=0.984, ze_peak_global_bw_gbps=688.7 | ok | 第三方（Intel 官方仓库、非本项目代码）的向量数字。关键不是绝对值而是**位宽比**：ze_peak 给出 fp16/fp32=1.98、fp64/fp32=0.735，与 Xe-HPC 的 ALU 位宽比（2× / ½~¾×）一致；自研探针给出 1.04 / 1.02，**根本分辨不出 dtype** —— 这正是判定探针绝对值不可信的独立依据之一。 |
| `arbitration.fp32_vector_peak` | question=FP32 向量峰值到底是 22.22 还是 50.75 TFLOPS, nominal_formula=448 EU x 16 lane x 2 FLOP x 1.55 GHz | 22.22 | 22.22 | tflops | formula_tflops=22.22, oneDNN_gemm_tflops=22.13, oneDNN_vs_formula=0.996, ze_peak_tflops=21.87, ze_peak_vs_formula=0.984, our_probe_tflops=50.75, our_probe_vs_formula=2.284, verdict_tflops=22.22 | ok | **裁定：以 22.22 TFLOPS 为准，自研探针的 50.75 撤回。**两个互相独立的第三方实现同时落在公式值上（oneDNN 99.6%、ze_peak 98.4%）；探针的 2.28× 等价于每 EU 37 条 FP32 lane（架构 16 条），且它分辨不出 dtype 的位宽比，ISA 显示其 vec8 FMA 被 IGC 完全标量化。 |
| `dev1.longrun_derate` | device=1, iters=50, warmup=10, reference=ze_peak_dev1.log, logs=[ze_peak_dev1.log, ze_peak_dev1_hot_rerun.log] | 21,871.9 | 21,871.9 | fp32.ze_peak_dev1.log.value | fp32.ze_peak_dev1.log.value=21,871.9, fp32.ze_peak_dev1.log.over_ref=1, fp32.ze_peak_dev1_hot_rerun.log.value=19,092.8, fp32.ze_peak_dev1_hot_rerun.log.over_ref=0.873, fp16.ze_peak_dev1.log.value=43,386.5, fp16.ze_peak_dev1.log.over_ref=1, fp16.ze_peak_dev1_hot_rerun.log.value=43,355.0, fp16.ze_peak_dev1_hot_rerun.log.over_ref=0.999, fp64.ze_peak_dev1.log.value=14,278.8, fp64.ze_peak_dev1.log.over_ref=1, fp64.ze_peak_dev1_hot_rerun.log.value=11,615.0, fp64.ze_peak_dev1_hot_rerun.log.over_ref=0.813, int32.ze_peak_dev1.log.value=3,640.0, int32.ze_peak_dev1.log.over_ref=1, int32.ze_peak_dev1_hot_rerun.log.value=3,542.8, int32.ze_peak_dev1_hot_rerun.log.over_ref=0.973, gbw.ze_peak_dev1.log.value=688.3, gbw.ze_peak_dev1.log.over_ref=1, gbw.ze_peak_dev1_hot_rerun.log.value=307, gbw.ze_peak_dev1_hot_rerun.log.over_ref=0.446 | ok | 长跑（`-i 50 -w 10`，单卡约 20 min）期间 GPU 持续满载会触发热/功耗降额：实测温度 92 → **101 °C**、功耗 305 → **330 W**（越过 300 W 名义上限），而 `gt_cur/max/min_freq_mhz` 始终报 1550 的**请求值**（唯一位于负载变化的 `gt_act_freq_mhz` 噪声极大，只能作定性证据）。降额**逐段推进**：同一 section 内按向量宽度顺序（=时间顺序）单调衰减；且**起始热态决定整轮水平** —— dev1 紧接前几轮再跑一次全量（起始 ~94 °C）时，连第一个 section `global_bw` 都掉到 307 GB/s（冷态 688），sp 21872→18497，dp 12213→11492，int 3640→3533。**判读纪律：绝对值引用短跑（`-a -i 3 -w 1`）或冷启分段结果，长跑整轮值仅供参考。** |
| `dev1.device` | device=1 | None | None | - | coreClockRate_mhz=1550, deviceId=0bda, subdeviceId=0, isSubdevice=FALSE | ok | Intel(R) Data Center GPU Max 1100；UUID=00000000-0000-0000-76d6-44d7fa576257 |
| `dev1.fp32.float` | device=1, test=sp_compute, kernel=float | 21,843.5 | 21,843.5 | gflops | gflops=21,843.5 | ok | GFLOPS |
| `dev1.fp32.float2` | device=1, test=sp_compute, kernel=float2 | 21,821.2 | 21,821.2 | gflops | gflops=21,821.2 | ok | GFLOPS |
| `dev1.fp32.float4` | device=1, test=sp_compute, kernel=float4 | 21,871.9 | 21,871.9 | gflops | gflops=21,871.9 | ok | GFLOPS |
| `dev1.fp32.float8` | device=1, test=sp_compute, kernel=float8 | 21,760.1 | 21,760.1 | gflops | gflops=21,760.1 | ok | GFLOPS |
| `dev1.fp32.float16` | device=1, test=sp_compute, kernel=float16 | 21,533.1 | 21,533.1 | gflops | gflops=21,533.1 | ok | GFLOPS |
| `dev1.fp64.double` | device=1, test=dp_compute, kernel=double | 14,278.8 | 14,278.8 | gflops | gflops=14,278.8 | ok | GFLOPS |
| `dev1.fp64.double2` | device=1, test=dp_compute, kernel=double2 | 12,355.7 | 12,355.7 | gflops | gflops=12,355.7 | ok | GFLOPS |
| `dev1.fp64.double4` | device=1, test=dp_compute, kernel=double4 | 12,212.5 | 12,212.5 | gflops | gflops=12,212.5 | ok | GFLOPS |
| `dev1.fp64.double8` | device=1, test=dp_compute, kernel=double8 | 11,092.3 | 11,092.3 | gflops | gflops=11,092.3 | ok | GFLOPS |
| `dev1.fp64.double16` | device=1, test=dp_compute, kernel=double16 | 9,047.0 | 9,047.0 | gflops | gflops=9,047.0 | ok | GFLOPS |
| `dev1.fp16.half` | device=1, test=hp_compute, kernel=half | 34,546.1 | 34,546.1 | gflops | gflops=34,546.1 | ok | GFLOPS |
| `dev1.fp16.half2` | device=1, test=hp_compute, kernel=half2 | 43,117.8 | 43,117.8 | gflops | gflops=43,117.8 | ok | GFLOPS |
| `dev1.fp16.half4` | device=1, test=hp_compute, kernel=half4 | 43,386.5 | 43,386.5 | gflops | gflops=43,386.5 | ok | GFLOPS |
| `dev1.fp16.half8` | device=1, test=hp_compute, kernel=half8 | 43,194.4 | 43,194.4 | gflops | gflops=43,194.4 | ok | GFLOPS |
| `dev1.fp16.half16` | device=1, test=hp_compute, kernel=half16 | 42,840.1 | 42,840.1 | gflops | gflops=42,840.1 | ok | GFLOPS |
| `dev1.int32.int` | device=1, test=int_compute, kernel=int | 3,640.0 | 3,640.0 | gflops | gflops=3,640.0 | ok | GFLOPS |
| `dev1.int32.int2` | device=1, test=int_compute, kernel=int2 | 3,494.2 | 3,494.2 | gflops | gflops=3,494.2 | ok | GFLOPS |
| `dev1.int32.int4` | device=1, test=int_compute, kernel=int4 | 3,503.4 | 3,503.4 | gflops | gflops=3,503.4 | ok | GFLOPS |
| `dev1.int32.int8` | device=1, test=int_compute, kernel=int8 | 2,775.1 | 2,775.1 | gflops | gflops=2,775.1 | ok | GFLOPS |
| `dev1.int32.int16` | device=1, test=int_compute, kernel=int16 | 3,086.0 | 3,086.0 | gflops | gflops=3,086.0 | ok | GFLOPS |
| `dev1.global_bw.float` | device=1, kernel=float | 688.3 | 688.3 | gbps | gbps=688.3 | ok | GB/s |
| `dev1.global_bw.float2` | device=1, kernel=float2 | 684.9 | 684.9 | gbps | gbps=684.9 | ok | GB/s |
| `dev1.global_bw.float4` | device=1, kernel=float4 | 662 | 662 | gbps | gbps=662 | ok | GB/s |
| `dev1.global_bw.float8` | device=1, kernel=float8 | 674.6 | 674.6 | gbps | gbps=674.6 | ok | GB/s |
| `dev1.global_bw.float16` | device=1, kernel=float16 | 678.6 | 678.6 | gbps | gbps=678.6 | ok | GB/s |
| `dev1.transfer_bw.enqueueWriteBuffer` | device=1 | 31.43 | 31.43 | gbps | gbps=31.43 | ok | GB/s |
| `dev1.transfer_bw.enqueueReadBuffer` | device=1 | 35.19 | 35.19 | gbps | gbps=35.19 | ok | GB/s |
| `dev1.transfer_bw.GPU_Copy_Host_to_Shared_Memory` | device=1 | 33.24 | 33.24 | gbps | gbps=33.24 | ok | GB/s |
| `dev1.transfer_bw.GPU_Copy_Shared_Memory_to_Host` | device=1 | 37.28 | 37.28 | gbps | gbps=37.28 | ok | GB/s |
| `dev1.transfer_bw.System_Memory_Copy_to_Shared_Memory` | device=1 | 7.97 | 7.97 | gbps | gbps=7.97 | ok | GB/s |
| `dev1.transfer_bw.System_Memory_Copy_from_Shared_Memory` | device=1 | 8.15 | 8.15 | gbps | gbps=8.15 | ok | GB/s |
| `dev1.kernel_lat.Kernel_launch_latency` | device=1 | 5.222 | 5.222 | us | us=5.222 | ok | us |
| `dev1.kernel_lat.Kernel_launch_latency_with_Immediate_Command_List` | device=1 | 5.349 | 5.349 | us | us=5.349 | ok | us |
| `dev1.kernel_lat.Kernel_duration` | device=1 | 15.75 | 15.75 | us | us=15.75 | ok | us |
| `repeatability.global_bw.float` | devices=[0, 1], runs=4, iters=3, warmup=1 | 687.1 | 687.1 | min | min=687.1, max=688.9, median=688.3, spread_pct=0.262 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.global_bw.float2` | devices=[0, 1], runs=4, iters=3, warmup=1 | 684.6 | 684.6 | min | min=684.6, max=686.1, median=685.1, spread_pct=0.233 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.global_bw.float4` | devices=[0, 1], runs=4, iters=3, warmup=1 | 659.2 | 659.2 | min | min=659.2, max=660.8, median=660.4, spread_pct=0.25 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.global_bw.float8` | devices=[0, 1], runs=4, iters=3, warmup=1 | 469.6 | 469.6 | min | min=469.6, max=676.2, median=571.8, spread_pct=30.55 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.global_bw.float16` | devices=[0, 1], runs=4, iters=3, warmup=1 | 467.4 | 467.4 | min | min=467.4, max=680.5, median=573.9, spread_pct=31.32 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.hp_compute.half` | devices=[0, 1], runs=4, iters=3, warmup=1 | 34,544.8 | 34,544.8 | min | min=34,544.8, max=34,546.9, median=34,545.9, spread_pct=6.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.hp_compute.half2` | devices=[0, 1], runs=4, iters=3, warmup=1 | 43,119.8 | 43,119.8 | min | min=43,119.8, max=43,121.4, median=43,120.4, spread_pct=4.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.hp_compute.half4` | devices=[0, 1], runs=4, iters=3, warmup=1 | 43,384.7 | 43,384.7 | min | min=43,384.7, max=43,397.4, median=43,388.5, spread_pct=0.029 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.hp_compute.half8` | devices=[0, 1], runs=4, iters=3, warmup=1 | 43,172.5 | 43,172.5 | min | min=43,172.5, max=43,206.0, median=43,195.0, spread_pct=0.078 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.hp_compute.half16` | devices=[0, 1], runs=4, iters=3, warmup=1 | 42,829.1 | 42,829.1 | min | min=42,829.1, max=42,861.9, median=42,841.8, spread_pct=0.077 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.sp_compute.float` | devices=[0, 1], runs=4, iters=3, warmup=1 | 21,843.9 | 21,843.9 | min | min=21,843.9, max=21,844.1, median=21,843.9, spread_pct=1.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.sp_compute.float2` | devices=[0, 1], runs=4, iters=3, warmup=1 | 21,820.8 | 21,820.8 | min | min=21,820.8, max=21,822.8, median=21,821.4, spread_pct=9.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.sp_compute.float4` | devices=[0, 1], runs=4, iters=3, warmup=1 | 21,871.6 | 21,871.6 | min | min=21,871.6, max=21,873.2, median=21,871.8, spread_pct=7.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.sp_compute.float8` | devices=[0, 1], runs=4, iters=3, warmup=1 | 21,760.5 | 21,760.5 | min | min=21,760.5, max=21,760.7, median=21,760.6, spread_pct=1.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.sp_compute.float16` | devices=[0, 1], runs=4, iters=3, warmup=1 | 21,533.4 | 21,533.4 | min | min=21,533.4, max=21,533.7, median=21,533.6, spread_pct=1.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.dp_compute.double` | devices=[0, 1], runs=4, iters=3, warmup=1 | 16,002.2 | 16,002.2 | min | min=16,002.2, max=16,005.9, median=16,004.9, spread_pct=0.023 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.dp_compute.double2` | devices=[0, 1], runs=4, iters=3, warmup=1 | 15,902.3 | 15,902.3 | min | min=15,902.3, max=15,905.6, median=15,905.0, spread_pct=0.021 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.dp_compute.double4` | devices=[0, 1], runs=4, iters=3, warmup=1 | 16,074.0 | 16,074.0 | min | min=16,074.0, max=16,074.7, median=16,074.1, spread_pct=4.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.dp_compute.double8` | devices=[0, 1], runs=4, iters=3, warmup=1 | 15,792.3 | 15,792.3 | min | min=15,792.3, max=15,793.8, median=15,793.5, spread_pct=9.000e-03 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.dp_compute.double16` | devices=[0, 1], runs=4, iters=3, warmup=1 | 13,970.6 | 13,970.6 | min | min=13,970.6, max=13,979.0, median=13,978.0, spread_pct=0.06 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.int_compute.int` | devices=[0, 1], runs=4, iters=3, warmup=1 | 6,195.1 | 6,195.1 | min | min=6,195.1, max=6,449.8, median=6,319.3, spread_pct=3.948 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.int_compute.int2` | devices=[0, 1], runs=4, iters=3, warmup=1 | 6,175.9 | 6,175.9 | min | min=6,175.9, max=6,431.4, median=6,302.0, spread_pct=3.973 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.int_compute.int4` | devices=[0, 1], runs=4, iters=3, warmup=1 | 6,182.1 | 6,182.1 | min | min=6,182.1, max=6,428.2, median=6,304.0, spread_pct=3.828 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.int_compute.int8` | devices=[0, 1], runs=4, iters=3, warmup=1 | 4,693.4 | 4,693.4 | min | min=4,693.4, max=4,972.0, median=4,824.7, spread_pct=5.603 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.int_compute.int16` | devices=[0, 1], runs=4, iters=3, warmup=1 | 4,991.3 | 4,991.3 | min | min=4,991.3, max=5,527.9, median=5,270.6, spread_pct=9.708 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.enqueueWriteBuffer` | devices=[0, 1], runs=4, iters=3, warmup=1 | 33.13 | 33.13 | min | min=33.13, max=37.16, median=34.96, spread_pct=10.84 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.enqueueReadBuffer` | devices=[0, 1], runs=4, iters=3, warmup=1 | 38.01 | 38.01 | min | min=38.01, max=47.06, median=42.11, spread_pct=19.25 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.GPUCopyHosttoSharedMemory` | devices=[0, 1], runs=4, iters=3, warmup=1 | 33.53 | 33.53 | min | min=33.53, max=37.41, median=35.13, spread_pct=10.38 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.GPUCopySharedMemorytoHost` | devices=[0, 1], runs=4, iters=3, warmup=1 | 38.06 | 38.06 | min | min=38.06, max=47.05, median=42.02, spread_pct=19.12 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.SystemMemoryCopytoSharedMemory` | devices=[0, 1], runs=4, iters=3, warmup=1 | 7.863 | 7.863 | min | min=7.863, max=8.124, median=8.052, spread_pct=3.217 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.transfer_bw.SystemMemoryCopyfromSharedMemory` | devices=[0, 1], runs=4, iters=3, warmup=1 | 7.965 | 7.965 | min | min=7.965, max=8.359, median=8.224, spread_pct=4.714 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.kernel_lat.Kernellaunchlatency` | devices=[0, 1], runs=4, iters=3, warmup=1 | 6.027 | 6.027 | min | min=6.027, max=7.68, median=6.333, spread_pct=21.53 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.kernel_lat.KernellaunchlatencywithImmediateCommandList` | devices=[0, 1], runs=4, iters=3, warmup=1 | 6.24 | 6.24 | min | min=6.24, max=6.987, median=6.613, spread_pct=10.69 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
| `repeatability.kernel_lat.Kernelduration` | devices=[0, 1], runs=4, iters=3, warmup=1 | 18.72 | 18.72 | min | min=18.72, max=47.04, median=26.59, spread_pct=60.2 | ok | 短跑 `-a -i 3 -w 1` 跨 **2 张卡 × 2 次** 的重复性；spread = (max-min)/max。用于判定长跑值的可信度。 |
