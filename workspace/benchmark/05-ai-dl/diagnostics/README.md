# diagnostics —— 支撑结论文档的定向探针

这些脚本不产出常规基准表，而是回答「**为什么是这个数**」。每个文件的 docstring 顶部
都写明了 2026-09-22 在 2× Intel Data Center GPU Max 1100 上的实测结论口径，
用来给 `docs/Conclusion/05-ai-dl/` 下的结论文档提供可复现的证据。

| 脚本 | 回答的问题 | 被引用于 |
|---|---|---|
| `train_kernel_mix.py` | 训练步的 GPU 时间花在哪些 kernel 上？为什么 BF16 只快 2.9×？ | `01-training-throughput.md`、`04-bottleneck.md` |
| `decode_budget.py` | LLM decode 的 21.6 ms/token 花在哪？ | `02-inference.md`、`04-bottleneck.md` |
| `m1_gemm_scan.py` | decode 各 GEMM 形状的单次开销与带宽位置 | `02-inference.md`、`04-bottleneck.md` |
| `launch_floor.py` | 瓶颈在主机下发还是设备执行？ | `04-bottleneck.md` |
| `llm_compile_compare.py` | `torch.compile` 能不能救 decode？ | `02-inference.md` |
| `allreduce_probe.py` | 2 卡 allreduce 的延迟/带宽地板 | `03-scaling.md` |

## 运行

```bash
cd /root/workspace/benchmark/05-ai-dl
PY=/root/workspace/venv1/bin/python

# 单卡探针（注意 env -u LD_LIBRARY_PATH：避免 oneAPI setvars.sh 把 VTune 私有
# ocloc/IGC 库注入 LD_LIBRARY_PATH，会破坏 torch.compile 的 triton-xpu 编译）
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/train_kernel_mix.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/decode_budget.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/m1_gemm_scan.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/launch_floor.py
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 $PY diagnostics/llm_compile_compare.py

# 双卡探针（必须由 torch.distributed.run 拉起）
env -u LD_LIBRARY_PATH $PY -m torch.distributed.run --standalone \
    --nproc_per_node=2 diagnostics/allreduce_probe.py
```

## 四个容易踩的坑（都真实踩过）

1. **不要用 `TorchDispatchMode` 去数 `torch.compile` 之后的 op 数**：Dynamo/Inductor 的
   每次缓存查询都会穿过 dispatch 层，数字会从 1909 膨胀到几十万。数 op 只在 eager 下做。
2. **不要在每个计时迭代里调 `dist.barrier()`**：barrier 本身 ~119 µs，会把所有小消息
   allreduce 的测量值统一拉到 80 µs 以上（第一版脚本就是这么错的）。
3. **`profiler` 的 `self_device_time_total` 单位是微秒**（不是毫秒），按毫秒解读会得到
   "205944 ms/step" 这种荒谬值；而且 `aten::` 算子层与真 kernel 层**都带 device time**，
   直接求和会重复计数（总量虚高 2 倍）。只要真 kernel 就必须过滤掉 `key.startswith("aten::")`。
4. **`profiler` 的 `count` 是 N 个 profiled step 的累计值**，除以 N 才是单步调用数；
   ResNet-50 一步的 745 个 ATen 算子对应 1023 次真 kernel 调用（一个 `conv` 会拆成
   `gen_conv` + `conv_reorder`）。
