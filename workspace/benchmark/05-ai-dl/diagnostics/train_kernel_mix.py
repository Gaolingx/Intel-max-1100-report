"""训练步的 kernel 成分分解 —— 解释「为什么 ResNet-50 的 BF16 加速比只有 ~2.9× 而不是 XMX 的 10.5×」。

结论（2026-09-22 实测，ResNet-50 bf16 batch=256 channels_last，单卡）：
  wall        = 206.93 ms/step
  真 kernel device self time 合计 = 205.94 ms/step（1023 次 kernel 调用）
  ==> GPU 忙碌率 ≈ 99.5%  —— 训练是"算力受限"，不是主机受限！

  kernel family 拆分（device time 占比，真 kernel 口径、已过滤 aten:: 层）：
    BatchNorm        38.1%   (4 个 kernel × 53 层：BackwardReduce / BackwardElemt /
                              WelfordStat / TransformInput，每个 287~449 µs)
    Conv/GEMM        32.6%   (gen_conv 158 次 × 414 µs = 65.5 ms，走 XMX)
    Elementwise/ReLU 24.9%   (VectorizedElementwiseKernel 129 次，合计 51.4 ms)
    Other             4.1%   (MaxPool fwd/bwd、zero_out、conv_reorder、AdamW 等)
    Copy/Add/Cat 0.2% + Reduce 0.1%
  → **BN + Elementwise + Other = 67% 的 GPU 时间与 XMX 无关**（纯带宽受限）。

  Amdahl 预测 BF16/FP32 加速比（假设 67.4% 的部分因字节减半而 2×、32.6% 的 conv 因 XMX 10.5×）：
      1 / (0.674/2 + 0.326/10.5) = 1 / (0.337 + 0.0310) = **2.72×**
  实测 channels_last b256：10.18 -> 29.75 TFLOPS = **2.92×**  ← 误差 +7%
  → **ResNet-50 上 XMX 只作用在 1/3 的算子时间上，这是加速比"只有 2.9×"的根本原因，
     不是 XMX 没启用。**

坑：
  * `self_device_time_total` 的单位是 **微秒**，不是毫秒 —— 按毫秒算会得到"205944 ms/step"这种荒谬值。
  * `key_averages()` 里既有 `aten::` 前缀的 **ATen 算子层**条目、也有真 kernel 条目，
    两者都带 device time，直接求和会**重复计算**（会把总量抬高 2 倍）。
    必须过滤掉 `key.startswith("aten::")`，只保留真 kernel（`gen_conv` / `at::native::xpu::*` 等）。
  * `e.count` 是 N 个 profiled step 的累计值，要除以 N 才是单步。
  * 必须设 `ZE_AFFINITY_MASK=0` 且 `env -u LD_LIBRARY_PATH`。

运行：
    cd /root/workspace/benchmark/05-ai-dl/diagnostics
    env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK=0 /root/workspace/venv1/bin/python train_kernel_mix.py
"""

import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile
from torchvision.models import resnet50

BATCH = 256
WARMUP = 5
TIMED = 10
PROFILED = 3


def make_step():
    torch.xpu.set_device(0)
    model = (
        resnet50()
        .to("xpu")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last)
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()
    x = (
        torch.randn(BATCH, 3, 224, 224, device="xpu")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last)
    )
    y = torch.randint(0, 1000, (BATCH,), device="xpu")

    def step():
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    return step


def family(key: str) -> str:
    kl = key.lower()
    if "batchnorm" in kl or "welford" in kl:
        return "BatchNorm"
    if "conv" in kl or "gemm" in kl:
        return "Conv/GEMM"
    if "relu" in kl or "elementwise" in kl or "clamp" in kl or "threshold" in kl:
        return "Elementwise/ReLU"
    if "reduce" in kl or "mean" in kl or "sum" in kl:
        return "Reduce"
    if "copy" in kl or "add" in kl or "cat" in kl:
        return "Copy/Add/Cat"
    return "Other"


def main() -> int:
    step = make_step()
    for _ in range(WARMUP):
        step()
    torch.xpu.synchronize()

    e0 = torch.xpu.Event(enable_timing=True)
    e1 = torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(TIMED):
        step()
    e1.record()
    torch.xpu.synchronize()
    wall = e0.elapsed_time(e1) / TIMED

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(PROFILED):
            step()
    torch.xpu.synchronize()

    # 只要真 kernel：排除 ATen 算子层的条目，否则 device time 会被重复计算
    rows = []
    for ev in prof.key_averages():
        dev_us = getattr(ev, "self_device_time_total", 0) or 0
        if dev_us > 0 and not ev.key.startswith("aten::"):
            rows.append((dev_us / PROFILED, ev.count / PROFILED, ev.key))
    rows.sort(reverse=True)

    total_us = sum(r[0] for r in rows)  # 微秒！
    calls = sum(r[1] for r in rows)

    print(f"ResNet-50 bf16 batch={BATCH} channels_last 单卡")
    print(f"  wall                          = {wall:8.2f} ms/step")
    print(f"  kernel device self time 合计   = {total_us / 1000:8.2f} ms/step")
    print(f"  kernel 调用数                  = {calls:8.0f} /step")
    print(f"  ==> GPU 忙碌率                 = {total_us / 1000 / wall * 100:8.1f} %")

    agg = defaultdict(lambda: [0.0, 0.0])
    for dev_us, cnt, key in rows:
        a = agg[family(key)]
        a[0] += dev_us
        a[1] += cnt

    print(f"\n  {'family':<18}{'ms/step':>10}{'占比':>9}{'calls':>8}{'us/call':>10}")
    for fam, (dev_us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(
            f"  {fam:<18}{dev_us / 1000:>10.2f}{100 * dev_us / total_us:>8.1f}%"
            f"{cnt:>8.0f}{dev_us / cnt:>10.1f}"
        )

    print("\n  top 12 kernel:")
    for dev_us, cnt, key in rows[:12]:
        print(f"    {dev_us / 1000:8.2f} ms x{cnt:5.0f}  {dev_us / cnt:7.1f} us/each  {key[:52]}")

    # Amdahl 校验：假设 67.4% 带宽受限部分随字节减半得 2×，32.6% conv 走 XMX 得 10.5×
    pred = 1.0 / (0.674 / 2.0 + 0.326 / 10.5)
    print(f"\n  Amdahl 预测 BF16/FP32 加速比 = 1/(0.674/2 + 0.326/10.5) = {pred:.2f}x")
    print("  实测 channels_last b256: 10.18 -> 29.75 TFLOPS = 2.92x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
