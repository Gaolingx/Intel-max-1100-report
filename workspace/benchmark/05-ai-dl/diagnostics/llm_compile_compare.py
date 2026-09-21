#!/usr/bin/env python3
"""诊断：LLM decode 的 eager vs torch.compile 对比。

结论口径（2026-09-22 实测，Qwen2.5-0.5B / bf16 / batch=1 / prompt=128）：
  * eager 25.6 ms/token（39 tok/s）；
  * torch.compile(mode="default") 21.5 ms/token（46.6 tok/s）→ 仅 1.20x；
  * torch.compile(mode="reduce-overhead") 25.4 ms/token → 无收益（KV cache 每步变化，
    cudagraph 难以复用；且逐 op 固定开销仍然存在）。
  → torch.compile 不能解决 decode 瓶颈；该瓶颈是「每 op 固定开销 × op 数」。

前置：需要 `ocloc` 可用（torch.compile 走 triton-xpu → ocloc）。
      单卡可见时才能启用 cudagraph：`ZE_AFFINITY_MASK=0`。

用法：ZE_AFFINITY_MASK=0 python diagnostics/llm_compile_compare.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = torch.device("xpu", 0)
BATCH, PLEN, NEW = 1, 128, 24


def measure(model, compiled, reps: int = 3):
    ids = torch.randint(0, 1000, (BATCH, PLEN), device=DEV)
    pfn, dfn = compiled if compiled else (None, None)

    def timed(fn):
        torch.xpu.synchronize()
        t0 = torch.xpu.Event(enable_timing=True)
        t1 = torch.xpu.Event(enable_timing=True)
        t0.record()
        out = fn()
        t1.record()
        torch.xpu.synchronize()
        return t0.elapsed_time(t1), out

    def prefill():
        return pfn(ids) if pfn is not None else model(input_ids=ids, use_cache=True)

    ttft, out = timed(prefill)
    past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
    ref = out.logits[:, -1:].float().clone()

    def one(past, tok):
        if dfn is not None:
            return dfn(tok, past)
        return model(input_ids=tok, past_key_values=past, use_cache=True)

    for _ in range(2):  # 暖机
        _, out = timed(lambda: one(past, tok))
        past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)

    total = 0.0
    for _ in range(NEW):
        dt, out = timed(lambda: one(past, tok))
        total += dt
        past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)

    with torch.no_grad():
        drift = float((out.logits[:, -1:].float() - ref).abs().max())
    return ttft, total / NEW, drift


def main() -> None:
    torch.xpu.set_device(0)
    print(f"devices={torch.xpu.device_count()}  batch={BATCH} prompt={PLEN} new={NEW}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = model.eval().to(DEV)

    ttft, per_tok, _ = measure(model, None)
    print(f"[eager            ] TTFT {ttft:8.2f} ms  decode {per_tok:7.3f} ms/token "
          f"({1000 / per_tok:6.1f} tok/s)")

    for mode in ("default", "reduce-overhead"):
        try:
            t0 = time.time()
            pfn = torch.compile(lambda i: model(input_ids=i, use_cache=True),
                                mode=mode, dynamic=False)
            dfn = torch.compile(lambda tk, p: model(input_ids=tk, past_key_values=p,
                                                    use_cache=True),
                                mode=mode, dynamic=False)
            measure(model, (pfn, dfn))                       # 触发编译
            ttft, per_tok, drift = measure(model, (pfn, dfn))
            print(f"[compile:{mode:<9}] TTFT {ttft:8.2f} ms  decode {per_tok:7.3f} ms/token "
                  f"({1000 / per_tok:6.1f} tok/s)  max|logit drift|={drift:.3f}  "
                  f"(compile+measure {time.time() - t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[compile:{mode:<9}] FAILED: {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    main()
