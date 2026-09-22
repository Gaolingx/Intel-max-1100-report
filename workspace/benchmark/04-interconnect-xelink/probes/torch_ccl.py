#!/usr/bin/env python3
"""04-interconnect-xelink / probes/torch_ccl.py

用 **PyTorch 的 xccl backend（底层就是 oneCCL）** 测双卡集合通信，
这是最贴近真实训练负载的一条路径。

启动方式（torchrun 会设置 RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR）：

    torchrun --nproc_per_node=2 --standalone probes/torch_ccl.py

每个 rank 用 ZE_AFFINITY_MASK 绑一张卡（由 LOCAL_RANK 决定）。

输出（每行一条 JSON，前缀便于 run_bench.py 解析）：
    CCLDEV  {"rank":0,"world_size":2,"device":"xpu:0","name":"..."}
    CCLOP   {"rank":0,"op":"allreduce","size_bytes":1048576,"dtype":"float32",
             "seconds":0.000123,"algbw_gbps":8.5,"busbw_gbps":8.5}
    CCLPEAK {"op":"allreduce","size_bytes":1073741824,"busbw_gbps":58.2}

术语：
    algbw (algorithm bandwidth) = (消息字节数) / 时间
    busbw (bus bandwidth)       = algbw × 2(W-1)/W      ← 集合通信的「总线口径」
                                   （对 ring allreduce，数据实际走的量）

注意：小消息会被 launch 开销淹没，只在大消息处判读峰值。
"""

import json
import os
import sys

import torch
import torch.distributed as dist


def emit(tag, obj):
    # ★ 两个 rank 的 stdout 是**共享**的，如果都打印，进程间会互相插队，
    #   把 JSON 行从中间切断（实测导致 run_bench.py 静默丢掉约 1/3 的记录）。
    #   因此只让 rank 0 输出；run_bench.py 也只用 rank 0 的样本。
    if os.environ.get("RANK", "0") != "0":
        return
    print(f"{tag} {json.dumps(obj, ensure_ascii=False)}", flush=True)


SIZES = [
    1 << 10,        # 1 KiB
    1 << 16,        # 64 KiB
    1 << 20,        # 1 MiB
    1 << 24,        # 16 MiB
    1 << 26,        # 64 MiB
    1 << 28,        # 256 MiB
    1 << 29,        # 512 MiB
    1 << 30,        # 1 GiB
]


def timed(fn, warmup=3, iters=10):
    """返回 (best_seconds, median_seconds)，用 xpu Event 计时并做 barrier 同步。"""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    dist.barrier()

    ts = []
    for _ in range(iters):
        dist.barrier()
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def main():
    if not torch.xpu.is_available():
        emit("CCLERR", {"stage": "init", "note": "torch.xpu 不可用"})
        return 1

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    n_dev = torch.xpu.device_count()

    # torchrun 不会自动绑卡；用 ZE_AFFINITY_MASK 让每个 rank 只看到自己那张
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank % max(n_dev, 1))

    if world < 2:
        emit("CCLERR", {"stage": "init",
                        "note": f"world_size={world} < 2，无法测互连；"
                                f"请用 torchrun --nproc_per_node=2 启动"})
        return 0

    dist.init_process_group(backend="xccl")
    torch.xpu.set_device(local_rank)

    dev = f"xpu:{local_rank}"
    emit("CCLDEV", {"rank": rank, "world_size": world, "device": dev,
                    "name": torch.xpu.get_device_name(local_rank),
                    "torch": torch.__version__,
                    "backend": "xccl",
                    "master_addr": os.environ.get("MASTER_ADDR", ""),
                    "master_port": os.environ.get("MASTER_PORT", "")})

    results = {}

    def record(op, size, t_best, t_med, dtype):
        algbw = size / t_best / 1e9
        busbw = algbw * 2.0 * (world - 1) / world
        emit("CCLOP", {"rank": rank, "op": op, "size_bytes": size, "dtype": dtype,
                       "seconds": t_best, "seconds_median": t_med,
                       "algbw_gbps": round(algbw, 3), "busbw_gbps": round(busbw, 3)})
        key = (op, dtype)
        if key not in results or busbw > results[key][2]:
            results[key] = (size, t_best, busbw)

    for dtype, dt_tensor in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for size in SIZES:
            nbytes = size
            nelem = nbytes // torch.tensor([], dtype=dt_tensor).element_size()
            if nelem < 1:
                continue
            try:
                t = torch.ones(nelem, dtype=dt_tensor, device=dev)
            except Exception as ex:                      # 显存不够
                emit("CCLSKIP", {"op": "alloc", "size_bytes": size, "dtype": dtype,
                                 "note": str(ex)[:160]})
                break

            try:
                b, m = timed(lambda: dist.all_reduce(t))
                t.zero_()
                record("allreduce", nbytes, b, m, dtype)
            except Exception as ex:
                emit("CCLERR", {"op": "allreduce", "size_bytes": size, "dtype": dtype,
                                "note": str(ex)[:200]})

            try:
                # all_gather_into_tensor 的 output 必须是 world × 输入大小，
                # 之前写成 (t, t) 会被 oneCCL 直接拒绝（output tensor size
                # must be equal to world_size times input tensor size）。
                out = torch.empty(nelem * world, dtype=dt_tensor, device=dev)
                b, m = timed(lambda: dist.all_gather_into_tensor(out, t))
                t.zero_()
                # busbw 口径按「发送量」计：algbw = 输入字节/时间，
                # 再由 record() 统一乘 2(W-1)/W。
                record("allgather", nbytes, b, m, dtype)
                del out
            except Exception as ex:
                emit("CCLSKIP", {"op": "allgather", "size_bytes": size, "dtype": dtype,
                                 "note": str(ex)[:120]})

            try:
                b, m = timed(lambda: dist.broadcast(t, src=0))
                t.zero_()
                record("broadcast", nbytes, b, m, dtype)
            except Exception as ex:
                emit("CCLSKIP", {"op": "broadcast", "size_bytes": size, "dtype": dtype,
                                 "note": str(ex)[:120]})

            del t
            torch.xpu.empty_cache()

    if rank == 0:
        for (op, dtype), (size, _t, busbw) in sorted(results.items()):
            emit("CCLPEAK", {"op": op, "dtype": dtype, "size_bytes": size,
                             "busbw_gbps": round(busbw, 3),
                             "xe_link_spec_gbps": 318.0,
                             "pct_of_spec": round(100.0 * busbw / 318.0, 1)})

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
