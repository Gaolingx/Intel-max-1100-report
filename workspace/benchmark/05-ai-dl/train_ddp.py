#!/usr/bin/env python3
"""DDP 单进程 worker —— 由 ``torchrun`` 拉起（对应 ``docs/TODO/05-ai-dl.md`` §3.4）。

不要直接运行；请用 ``python -m torch.distributed.run --nproc_per_node=N``
或 ``run_bench.py --suite ddp``（后者自动编排 1 卡 / 2 卡并计算扩展效率）。

核心设计
--------
除总步时外，本脚本用 **``model.no_sync()`` 差分法**量化通信开销：

* ``t_sync``   —— 正常 DDP 步时（反向传播时触发 allreduce）
* ``t_nosync`` —— ``no_sync()`` 内步时（**不做**梯度同步，纯计算）
* ``comm_ms = t_sync - t_nosync``，``comm_pct = comm_ms / t_sync``

这比 profiler 解析更稳健，且可直接回答「扩展效率损失是否来自通信」。

输出：rank 0 将 JSON 写到 ``--out``，同时打印到 stdout（以 ``DDP_JSON:`` 为前缀）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="XPU DDP worker（勿直接调用）")
    p.add_argument("--model", default="resnet50", choices=["resnet50", "bert"])
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128, help="model=bert 时的序列长度")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backend", default="xccl", help="xccl（XPU）/ ccl（oneCCL）/ gloo")
    p.add_argument("--no-ddp", action="store_true", help="不包 DDP（单卡裸模型基线）")
    p.add_argument("--no-sync-probe", action="store_true",
                   help="跳过 no_sync 通信开销差分")
    p.add_argument("--out", default=None, help="rank0 结果 JSON 输出路径")
    return p


# ---------------------------------------------------------------------------
def _build_model_and_step(args, local_rank: int):
    """返回 (model, step_fn_factory)。

    ``step_fn_factory(model) -> callable``，便于在 DDP 包装前后复用同一份
    训练步逻辑。
    """
    device = torch.device("xpu", local_rank)

    if args.model == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None).to(device)
        x = torch.randn(args.batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (args.batch,), device=device)
        crit = torch.nn.CrossEntropyLoss()
        fwd_flops = 2.0 * 4.09e9 * args.batch      # ResNet-50 @224²: 4.09 GMACs
        n_params = sum(p.numel() for p in model.parameters())
    else:
        from transformers import BertConfig, BertForMaskedLM

        cfg = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                         num_attention_heads=12, intermediate_size=3072,
                         max_position_embeddings=max(512, args.seq_len))
        model = BertForMaskedLM(cfg).to(device)
        x = torch.randint(0, 30522, (args.batch, args.seq_len), device=device)
        y = None
        crit = None
        n_params = sum(p.numel() for p in model.parameters())
        # fwd ≈ 2 × N_params × tokens
        fwd_flops = 2.0 * n_params * args.batch * args.seq_len

    train_flops = 3.0 * fwd_flops       # fwd + bwd-data + bwd-weights

    def make_step(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        amp = args.dtype != "fp32"
        ac_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

        if args.model == "resnet50":
            def step() -> None:
                opt.zero_grad(set_to_none=True)
                if amp:
                    with torch.autocast(device_type="xpu", dtype=ac_dtype):
                        loss = crit(m(x), y)
                else:
                    loss = crit(m(x), y)
                loss.backward()
                opt.step()
        else:
            attn = torch.ones_like(x)
            labels = x.clone()

            def step() -> None:
                opt.zero_grad(set_to_none=True)
                if amp:
                    with torch.autocast(device_type="xpu", dtype=ac_dtype):
                        loss = m(input_ids=x, attention_mask=attn, labels=labels).loss
                else:
                    loss = m(input_ids=x, attention_mask=attn, labels=labels).loss
                loss.backward()
                opt.step()

        return step

    return model, make_step, fwd_flops, train_flops, n_params


def _time_steps(step, warmup: int, iters: int, local_rank: int) -> dict:
    for _ in range(warmup):
        step()
    torch.xpu.synchronize(local_rank)

    samples: list[float] = []
    for _ in range(iters):
        e0 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        e0.record()
        step()
        e1.record()
        torch.xpu.synchronize(local_rank)
        samples.append(e0.elapsed_time(e1))

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "n": len(samples),
    }


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.xpu.is_available():
        print("错误：torch.xpu 不可用", file=sys.stderr)
        return 2
    torch.xpu.set_device(local_rank)
    torch.manual_seed(0)

    import torch.distributed as dist

    use_ddp = world_size > 1 or not args.no_ddp

    if use_ddp:
        init_kwargs = {"backend": args.backend}
        try:
            dist.init_process_group(**init_kwargs,
                                    device_id=torch.device("xpu", local_rank))
        except TypeError:
            # 老/新版本的 init_process_group 签名差异
            dist.init_process_group(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"DDP_INIT_FAILED: {args.backend}: {type(exc).__name__}: {exc}")
            return 3

    result: dict = {
        "ok": False, "rank": rank, "world_size": world_size,
        "local_rank": local_rank, "backend": args.backend,
        "model": args.model, "dtype": args.dtype, "batch": args.batch,
        "seq_len": args.seq_len, "ddp": bool(use_ddp),
        "world_size_gt1": bool(world_size > 1),
        "no_ddp_flag": args.no_ddp,
        "device_name": torch.xpu.get_device_name(local_rank),
    }

    try:
        torch.xpu.reset_peak_memory_stats(local_rank)
        model, make_step, fwd_flops, train_flops, n_params = _build_model_and_step(
            args, local_rank
        )
        result["params_m"] = round(n_params / 1e6, 1)

        # 只要启用了 DDP 就包裹（world_size==1 时也包，用来隔离 DDP 包装本身的开销）
        if use_ddp:
            try:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank]
                )
            except Exception:  # noqa: BLE001 - device_ids 语义在 XPU 上可能不同
                model = torch.nn.parallel.DistributedDataParallel(model)
        result["wrapped_ddp"] = bool(use_ddp)

        step = make_step(model)

        # --- 正常同步步时 ---
        t_sync = _time_steps(step, args.warmup, args.iters, local_rank)
        result["sync"] = t_sync

        # --- no_sync 差分（仅多卡时有意义）---
        if world_size > 1 and not args.no_sync_probe:
            try:
                with model.no_sync():
                    t_nosync = _time_steps(step, args.warmup, args.iters, local_rank)
                result["nosync"] = t_nosync
                comm_ms = t_sync["median_ms"] - t_nosync["median_ms"]
                result["comm_ms"] = comm_ms
                result["comm_pct"] = 100.0 * comm_ms / t_sync["median_ms"]
            except Exception as exc:  # noqa: BLE001
                result["nosync_error"] = f"{type(exc).__name__}: {exc}"

        dt = t_sync["median_ms"] * 1e-3
        # 每卡吞吐与全局吞吐（全局 = 每卡 × 卡数）
        result["samples_per_s_per_rank"] = args.batch / dt
        result["samples_per_s"] = args.batch * world_size / dt
        result["tflops_per_rank"] = train_flops / dt / 1e12
        result["tflops"] = result["tflops_per_rank"] * world_size
        result["ms_per_step"] = t_sync["median_ms"]
        result["fwd_flops"] = fwd_flops
        result["train_flops"] = train_flops
        result["peak_mem_gib"] = round(
            torch.xpu.max_memory_allocated(local_rank) / (1024.0 ** 3), 3
        )
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if use_ddp:
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass

    if rank == 0:
        payload = json.dumps(result, ensure_ascii=False)
        print(f"DDP_JSON: {payload}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(payload, encoding="utf-8")

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
