"""双卡 DDP 扩展效率编排 —— 对应 ``docs/TODO/05-ai-dl.md`` §3.4（**核心项**）。

本模块不自己计时，而是通过 ``torchrun`` 拉起 :mod:`train_ddp` 子进程，
分别跑「单卡裸模型 / 单卡 DDP / 双卡 DDP」，再汇总：

* ``samples_per_s``、``tflops``、``ms_per_step``

* **扩展效率** ``= (2卡吞吐 / 1卡吞吐) / 2``

* **通信占比** ``comm_pct`` —— 来自 worker 的 ``no_sync()`` 差分法
  （``t_sync - t_nosync``），直接回答「扩展损失是否来自通信」

判读（来自 TODO 文档）：

| 现象 | 结论 |
|---|---|
| 扩展效率 > 90% | 良好 |
| 扩展效率 < 70% | 通信瓶颈 → 回 ④ 检查 P2P 与 Xe Link |
| ``comm_pct`` 高但扩展效率低 | allreduce 慢（互连 / 后端问题） |
| ``comm_pct`` 低但扩展效率低 | 计算侧问题（频率、cache、kernel 未压满） |

后端：默认 ``xccl``（PyTorch XPU 原生）。oneCCL 的 ``ccl`` 后端若不可用，
worker 会返回非 0 退出码，本模块记录为 ``skipped`` 而不是失败。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .common import BenchResult, ResultStore

SUITE = "ddp"

HERE = Path(__file__).resolve().parent.parent      # benchmark/05-ai-dl
WORKER = HERE / "train_ddp.py"
DEFAULT_BATCHES = [64, 128]
DEFAULT_DTYPES = ["bf16"]
DEFAULT_BACKENDS = ["xccl"]


def _int_list(ns, attr: str, default: list[int]) -> list[int]:
    raw = getattr(ns, attr, None)
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in raw]


def _str_list(ns, attr: str, default: list[str]) -> list[str]:
    raw = getattr(ns, attr, None)
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x) for x in raw]


def _run_worker(ns, nproc: int, out_path: Path, extra: list[str],
                timeout_s: int = 1800) -> dict:
    """拉起一次 torchrun 子进程，返回 worker 的 JSON（失败时返回 {'ok': False, ...}）。"""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        str(WORKER), *extra,
        "--out", str(out_path),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if out_path.exists():
        out_path.unlink()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s"}

    payload: Optional[dict] = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("DDP_JSON:"):
            try:
                payload = json.loads(line[len("DDP_JSON:"):].strip())
            except json.JSONDecodeError:
                payload = None

    if payload is None and out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = None

    if payload is None:
        tail = "\n".join((proc.stdout or "").splitlines()[-12:]
                         + (proc.stderr or "").splitlines()[-12:])
        return {"ok": False, "error": f"worker 未返回 JSON（rc={proc.returncode}）",
                "log_tail": tail[-1200:], "returncode": proc.returncode}

    payload["returncode"] = proc.returncode
    if not payload.get("ok"):
        payload.setdefault("log_tail",
                           "\n".join((proc.stderr or "").splitlines()[-12:])[-1200:])
    return payload


# ---------------------------------------------------------------------------
def run(ns, store: ResultStore) -> None:
    """DDP 扩展效率 suite 入口（§3.4）。"""
    warmup = int(getattr(ns, "warmup", 5))
    iters = int(getattr(ns, "ddp_iters", 20))
    batches = _int_list(ns, "ddp_batches", DEFAULT_BATCHES)
    dtypes = _str_list(ns, "ddp_dtypes", DEFAULT_DTYPES)
    backends = _str_list(ns, "ddp_backends", DEFAULT_BACKENDS)
    model = str(getattr(ns, "ddp_model", "resnet50"))

    if not WORKER.exists():
        store.skip(SUITE, "scaling", {}, f"未找到 {WORKER}")
        return

    n_dev = torch_device_count()
    if n_dev < 2:
        store.skip(SUITE, "scaling", {"devices": n_dev},
                   f"需要 2 张卡，当前只有 {n_dev}")

    outdir = Path(getattr(ns, "outdir", HERE / "results")) / "ddp_raw"
    outdir.mkdir(parents=True, exist_ok=True)

    for backend in backends:
        for dtype in dtypes:
            for batch in batches:
                tag = f"{model}_{dtype}_b{batch}_{backend}"
                common = ["--model", model, "--dtype", dtype,
                          "--batch", str(batch), "--backend", backend,
                          "--warmup", str(warmup), "--iters", str(iters)]

                # (config 名, nproc, extra args)
                configs = [
                    ("1card_nodpp", 1, ["--no-ddp", "--no-sync-probe"]),
                    ("1card_ddp", 1, []),
                ]
                if n_dev >= 2:
                    configs.append(("2card_ddp", 2, []))

                got: dict[str, dict] = {}
                for cfg_name, nproc, extra in configs:
                    out_path = outdir / f"{tag}_{cfg_name}.json"
                    payload = _run_worker(ns, nproc, out_path, common + extra)
                    got[cfg_name] = payload

                    params = {"model": model, "dtype": dtype, "batch": batch,
                              "world_size": nproc, "config": cfg_name,
                              "backend": backend}
                    if not payload.get("ok"):
                        store.skip(
                            SUITE, "ddp_step", params,
                            f"{payload.get('error')}"
                            + (f"；log: {payload.get('log_tail', '')[-300:]}"
                               if payload.get("log_tail") else "")
                        )
                        continue

                    metrics = {
                        "samples_per_s": payload.get("samples_per_s"),
                        "samples_per_s_per_rank": payload.get("samples_per_s_per_rank"),
                        "tflops": payload.get("tflops"),
                        "ms_per_step": payload.get("ms_per_step"),
                        "comm_pct": payload.get("comm_pct"),
                        "comm_ms": payload.get("comm_ms"),
                        "peak_mem_gib": payload.get("peak_mem_gib"),
                        "params_m": payload.get("params_m"),
                    }
                    _note = []
                    if cfg_name == "2card_ddp":
                        _note.append("全局吞吐 = 每卡 batch × 2 / 步时")
                        if payload.get("comm_pct") is not None:
                            _note.append("通信占比由 no_sync 差分法测得")
                    store.add(BenchResult(
                        suite=SUITE, name="ddp_step", params=params, stats=None,
                        metrics={k: v for k, v in metrics.items() if v is not None},
                        note="；".join(_note),
                    ))

                _scaling_summary(store, got, model, dtype, batch, backend)


def torch_device_count() -> int:
    import torch

    try:
        return int(torch.xpu.device_count())
    except Exception:  # noqa: BLE001
        return 0


def _find(store: ResultStore, model: str, dtype: str, batch: int,
          backend: str, config: str) -> Optional[BenchResult]:
    for r in store.suite(SUITE):
        p = r.params
        if (p.get("model") == model and p.get("dtype") == dtype
                and p.get("batch") == batch and p.get("backend") == backend
                and p.get("config") == config and r.name == "ddp_step"):
            return r
    return None


def _scaling_summary(store: ResultStore, got: dict, model: str, dtype: str,
                     batch: int, backend: str) -> None:
    base = _find(store, model, dtype, batch, backend, "1card_nodpp") \
        or _find(store, model, dtype, batch, backend, "1card_ddp")
    two = _find(store, model, dtype, batch, backend, "2card_ddp")
    if base is None or two is None:
        return

    b = base.metrics["samples_per_s"]
    t = two.metrics["samples_per_s"]
    speedup = t / b
    eff = 100.0 * speedup / 2.0

    metrics = {"speedup_2x": speedup, "scaling_eff_pct": eff,
               "samples_per_s_1card": b, "samples_per_s_2card": t,
               "tflops_1card": base.metrics.get("tflops"),
               "tflops_2card": two.metrics.get("tflops")}

    # 同时给出「相对 1 卡 DDP」的扩展效率，用来分离 DDP 包装自身的开销
    one_ddp = _find(store, model, dtype, batch, backend, "1card_ddp")
    if one_ddp is not None and one_ddp is not base:
        b_ddp = one_ddp.metrics["samples_per_s"]
        metrics["speedup_2x_vs_1card_ddp"] = t / b_ddp
        metrics["scaling_eff_pct_vs_1card_ddp"] = 100.0 * (t / b_ddp) / 2.0

    store.add(BenchResult(
        suite=SUITE, name="scaling",
        params={"model": model, "dtype": dtype, "batch": batch,
                "backend": backend, "baseline": base.params.get("config")},
        stats=None,
        metrics={k: v for k, v in metrics.items() if v is not None},
        note="扩展效率 = (2卡吞吐 / 1卡吞吐) / 2；基线为 1卡无 DDP",
    ))

    comm = two.metrics.get("comm_pct")
    comm_txt = f"，通信占比 {comm:.1f}%" if comm is not None else ""
    verdict = "良好" if eff >= 90 else ("可接受" if eff >= 70 else "**通信/同步瓶颈**")
    store.add_note(
        f"DDP {model}/{dtype}/batch={batch}/{backend}: 加速比 **{speedup:.2f}×**，"
        f"扩展效率 **{eff:.1f}%**（{verdict}）{comm_txt}"
    )
