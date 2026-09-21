#!/usr/bin/env python3
"""XPU 理论性能基准测试入口（torch / torch.xpu）。

覆盖 ``docs/TODO/`` 中与 AI / 算子相关的测试项：

====================  ==================================================
suite                 内容
====================  ==================================================
``gemm``              ② 算力峰值：方阵 / 真实场景 shape / batched / INT8，
                      覆盖 fp32、fp64、fp16、bf16
``elementwise``       ③⑤ copy / scale / add / mul / triad / 激活函数，
                      用有效带宽（GB/s）衡量
``membw``             ③ 显存带宽专项：size sweep、dtype 对照、H2D/D2H
``reduce``            ⑤ sum / softmax / layer_norm / rms_norm / dot
``attention``         ⑤ SDPA：prefill 与 decode（KV cache）shape
``conv``              ⑤ conv2d：ResNet-50 / MobileNet 典型 shape
``quant``             ⑤ 量化：INT8 稠密、W8A16、W4A16 权重独占 + 数值自检
``precision``         ② 数值格式支持矩阵：厂商列出的每种精度（vector + matmul）
                      逐个实跑，记录 OK/FAIL；再测各精度吞吐
``resnet``            ⑤ ResNet-50 训练吞吐（FP32 vs BF16 AMP、batch sweep、
                      channels_last、torch.compile）
``bert``              ⑤ BERT 训练吞吐（MLM 头，seq_len / batch sweep；模型可选）
``llm``               ⑤ LLM 推理：prefill(TTFT) / decode(tok/s) / 显存峰值
``ddp``               ⑤ 双卡 DDP 扩展效率（torchrun 1卡 vs 2卡 + 通信占比）
``pipeline``          ⑤ 数据管线 / 主机侧瓶颈（DataLoader worker、pin_memory、H2D）
====================  ==================================================

> 模型级 suite（``resnet`` / ``bert`` / ``llm`` / ``ddp`` / ``pipeline``）耗时
> 远大于算子级，因此 **不包含在默认的 `--suite all` 内**；
> 需要时用 `--suite all --with-models`，或显式点名。

示例::

    source /root/workspace/venv1/bin/activate
    python run_bench.py --list
    python run_bench.py --suite gemm --dtypes fp32,bf16 --large
    python run_bench.py --suite resnet,bert --model-batches 64,128
    python run_bench.py --suite ddp --ddp-backends xccl
    python run_bench.py --suite all --outdir results

产出：
* 控制台 Markdown 报告
* ``results/bench_<时间戳>.json``（结构化，供后续画图）
* ``results/bench_<时间戳>.md``（可直接贴进 docs/）
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from xpu_bench import (
    attention,
    conv,
    ddp,
    elementwise,
    gemm,
    models,
    pipeline,
    precision,
    quant,
    reduction,
)
from xpu_bench.common import (
    DTYPES,
    ResultStore,
    Stopwatch,
    TelemetrySampler,
    environment_meta,
)

HERE = Path(__file__).resolve().parent

# suite 名 -> (执行函数, 说明)
SUITES: dict[str, tuple] = {
    "gemm": (gemm.run, "GEMM/matmul 算力峰值（shape × dtype sweep，含 INT8）"),
    "elementwise": (elementwise.run, "Elementwise 算子有效带宽"),
    "membw": (elementwise.run_bw, "显存带宽专项（size / dtype / host 传输）"),
    "reduce": (reduction.run, "Reduction / Softmax / LayerNorm / RMSNorm"),
    "attention": (attention.run, "Attention (SDPA) prefill / decode"),
    "conv": (conv.run, "Convolution（ResNet-50 / MobileNet shape）"),
    "quant": (quant.run, "量化：INT8 / W8A16 / W4A16 权重独占（含数值自检）"),
    "precision": (precision.run,
                  "数值格式支持矩阵 + 各精度 vector/matmul 吞吐"),
    # --- 模型级（端到端，慢） ---
    "resnet": (models.run_resnet, "ResNet-50 训练吞吐（FP32 vs BF16 AMP）"),
    "bert": (models.run_bert, "BERT 训练吞吐（MLM 头；`--bert-model` 可选）"),
    "llm": (models.run_llm, "LLM 推理：prefill / decode / 显存峰值"),
    "ddp": (ddp.run, "双卡 DDP 扩展效率（torchrun 1卡 vs 2卡）"),
    "pipeline": (pipeline.run, "数据管线 / 主机侧瓶颈（DataLoader / H2D）"),
}

# 算子级（`--suite all` 的默认范围）
DEFAULT_ORDER = ["gemm", "elementwise", "membw", "reduce", "attention", "conv",
                 "quant", "precision"]

# 模型级（需 --with-models 或显式点名；耗时长）
MODEL_SUITES = ["resnet", "bert", "llm", "ddp", "pipeline"]

# 全量顺序（用于显式 suite 列表的排序）
FULL_ORDER = DEFAULT_ORDER + MODEL_SUITES

QUICK_GEMM_SIZES = [256, 512, 1024, 2048]
QUICK_BW_SIZES = [64, 256, 1024]


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_bench.py",
        description="XPU 理论性能基准测试（torch.xpu）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite", default="all",
                   help="逗号分隔的 suite 列表，或 all；可用 --list 查看")
    p.add_argument("--list", action="store_true", help="列出所有 suite 后退出")
    p.add_argument("--device", type=int, default=0, help="GPU 序号")
    p.add_argument("--warmup", type=int, default=5, help="每个测点的 warmup 次数")
    p.add_argument("--iters", type=int, default=20, help="每个测点的计时迭代次数")
    p.add_argument("--dtypes", default=None,
                   help="逗号分隔 dtype 子集，如 fp32,bf16,fp16,fp64（默认按 suite 取）")
    p.add_argument("--size-mb", type=int, default=1024,
                   help="elementwise / reduce 每个张量的目标大小（MiB）")
    p.add_argument("--gemm-sizes", default="256,512,1024,2048,4096,8192",
                   help="GEMM 方阵 sweep 的 n 列表")
    p.add_argument("--bw-sizes", default="4,16,64,128,192,256,512,1024,2048",
                   help="带宽 size sweep（MiB）列表")
    p.add_argument("--reduce-cols", type=int, default=4096,
                   help="reduction 测试的内层维度（列数）")
    p.add_argument("--xfer-mib", type=int, default=256, help="H2D/D2H 传输测试大小（MiB）")
    p.add_argument("--large", action="store_true",
                   help="加入 16384³ GEMM 与 8K attention 等大尺寸用例")
    p.add_argument("--quick", action="store_true",
                   help="快速冒烟模式：缩小尺寸、减少迭代")
    # 细粒度开关
    p.add_argument("--no-real-shapes", dest="real_shapes", action="store_false",
                   help="跳过 GEMM 真实场景 shape")
    p.add_argument("--no-batched", dest="batched", action="store_false",
                   help="跳过 batched matmul")
    p.add_argument("--no-int8", dest="int8", action="store_false",
                   help="跳过 INT8 GEMM")
    p.add_argument("--no-manual-attn", dest="manual_attn", action="store_false",
                   help="跳过朴素 attention 对照")
    p.add_argument("--no-dtype-bw", dest="dtype_bw", action="store_false",
                   help="跳过 dtype 带宽对照")
    p.add_argument("--no-size-sweep", dest="size_sweep", action="store_false",
                   help="跳过带宽 size sweep")
    p.add_argument("--no-host-transfer", dest="host_transfer", action="store_false",
                   help="跳过 H2D/D2H 传输测试")
    p.add_argument("--no-quant-verify", dest="quant_verify", action="store_false",
                   help="跳过量化数值自检（verify_*）")
    p.add_argument("--no-quant-compare", dest="quant_compare", action="store_false",
                   help="跳过 BF16/W8A16/W4A16 三方对照")
    # --- 模型级 suite 开关（§3.2–§3.6）---
    p.add_argument("--with-models", action="store_true",
                   help="让 `--suite all` 一并包含模型级 suite（resnet/bert/llm/ddp/pipeline）")
    p.add_argument("--model-batches", default="64,128,256",
                   help="ResNet-50 训练 batch size 列表")
    p.add_argument("--model-extra", action="store_true",
                   help="ResNet-50 额外参加 batch=512 的大 batch 测点")
    p.add_argument("--model-dtypes", default="fp32,bf16",
                   help="模型级 suite 的 dtype 子集（bf16 走 AMP）")
    p.add_argument("--model-compile", action="store_true",
                   help="ResNet-50 开启 torch.compile（首次编译很慢）")
    p.add_argument("--bert-model", default="bert-base-uncased",
                   help="BERT HuggingFace 模型 id")
    p.add_argument("--bert-seq-lens", default="128,512", help="BERT 序列长度列表")
    p.add_argument("--bert-batches", default="16,32", help="BERT batch size 列表")
    p.add_argument("--llm-model", default=models.DEFAULT_LLM_MODEL,
                   help="LLM HuggingFace 模型 id（causal LM）")
    p.add_argument("--llm-batches", default="1,4,8", help="LLM 推理 batch size 列表")
    p.add_argument("--llm-input-lens", default="128,512,2048", help="LLM 输入长度列表")
    p.add_argument("--llm-new-tokens", type=int, default=models.DEFAULT_LLM_NEW_TOKENS,
                   help="LLM 每个测点生成的新 token 数")
    p.add_argument("--llm-compile", default="off",
                   choices=["off", "default", "reduce-overhead", "max-autotune"],
                   help="LLM 是否启用 torch.compile（eager 路径每 token 下发 ~1800 kernel）")
    p.add_argument("--no-hf", action="store_true",
                   help="不联网下载 HF 权重，直接用本地 config 随机初始化")
    p.add_argument("--hf-endpoint", default=models.DEFAULT_HF_ENDPOINT,
                   help="HuggingFace 镜像端点（本机 huggingface.co 不可达）")
    p.add_argument("--pipe-batch", type=int, default=64, help="pipeline 测试的 batch size")
    p.add_argument("--pipe-workers", default="0,1,2,4,8",
                   help="DataLoader worker 数列表")
    p.add_argument("--pipe-cpu-scale", type=int, default=3,
                   help="合成数据集单样本 CPU 预处理强度（0=无）")
    p.add_argument("--pipe-batches", type=int, default=20,
                   help="每个 worker 配置采集的 batch 数")
    p.add_argument("--pipe-e2e-batches", type=int, default=20,
                   help="端到端 step 的批次数")
    p.add_argument("--ddp-model", default="resnet50", choices=["resnet50", "bert"],
                   help="DDP 测试使用的模型")
    p.add_argument("--ddp-backends", default="xccl",
                   help="DDP 后端列表（xccl 原生 / ccl oneCCL / gloo）")
    p.add_argument("--ddp-dtypes", default="bf16", help="DDP dtype 列表")
    p.add_argument("--ddp-batches", default="64,128", help="DDP 每卡 batch size 列表")
    p.add_argument("--ddp-iters", type=int, default=20, help="DDP 计时迭代数")
    # 输出
    p.add_argument("--outdir", default=str(HERE / "results"), help="结果输出目录")
    p.add_argument("--json", default=None, help="JSON 输出路径（覆盖默认）")
    p.add_argument("--markdown", default=None, help="Markdown 输出路径（覆盖默认）")
    p.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    p.add_argument("--quiet", action="store_true", help="不打印完整 Markdown")
    p.add_argument("--telemetry", default=None,
                   help="同时用 xpu-smi dump 采样遥测，输出到该路径")
    return p


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _resolve_suites(spec: str, with_models: bool = False) -> list[str]:
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(FULL_ORDER if with_models else DEFAULT_ORDER)
    chosen = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in SUITES]
    if unknown:
        raise SystemExit(f"未知 suite: {', '.join(unknown)}；可用: {', '.join(FULL_ORDER)} 或 all")
    # 保持默认顺序、去重
    return [s for s in FULL_ORDER if s in set(chosen)]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("可用 suite：")
        for name in FULL_ORDER:
            group = "[模型级] " if name in MODEL_SUITES else "[算子级] "
            print(f"  {group}{name:12s} {SUITES[name][1]}")
        print("\n注：`--suite all` 默认只跑算子级；加 `--with-models` 才包含模型级。")
        return 0

    if not torch.xpu.is_available():
        print("错误：torch.xpu 不可用。请先 `source /opt/intel/oneapi/setvars.sh` "
              "并确认激活了带 XPU 的 venv（/root/workspace/venv1）。", file=sys.stderr)
        return 2

    n_dev = torch.xpu.device_count()
    if not 0 <= args.device < n_dev:
        print(f"错误：--device {args.device} 超出范围（可用 0..{n_dev - 1}）", file=sys.stderr)
        return 2
    torch.xpu.set_device(args.device)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)

    # quick 预设
    if args.quick:
        args.iters = min(args.iters, 5)
        args.warmup = min(args.warmup, 2)
        args.size_mb = min(args.size_mb, 256)
        args.gemm_sizes = ",".join(str(s) for s in QUICK_GEMM_SIZES)
        args.bw_sizes = ",".join(str(s) for s in QUICK_BW_SIZES)
        args.large = False
        # 模型级 suite 同步缩小
        args.model_batches = "64"
        args.model_dtypes = "fp32,bf16"
        args.bert_seq_lens = "128"
        args.bert_batches = "8"
        args.llm_batches = "1"
        args.llm_input_lens = "128"
        args.llm_new_tokens = min(args.llm_new_tokens, 4)
        args.pipe_workers = "0,4"
        args.pipe_batches = min(args.pipe_batches, 6)
        args.pipe_e2e_batches = min(args.pipe_e2e_batches, 6)
        args.ddp_batches = "64"
        args.ddp_iters = min(args.ddp_iters, 5)

    # 类型转换
    args.gemm_sizes = _parse_int_list(args.gemm_sizes)
    args.bw_sizes = _parse_int_list(args.bw_sizes)

    suites = _resolve_suites(args.suite, with_models=args.with_models)

    # HF 镜像端点（必须在 transformers / huggingface_hub 首次导入前生效）
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # meta 在删除 dtypes 之前快照，保证报告里能看到用户显式指定的 dtype
    meta_args = {k: v for k, v in vars(args).items() if v not in (None, False)}
    if args.dtypes:
        dtype_list = [d.strip() for d in args.dtypes.split(",") if d.strip()]
        bad = [d for d in dtype_list if d not in DTYPES and d != "int8"]
        if bad:
            print(f"错误：未知 dtype {bad}；可用: {', '.join(DTYPES)}, int8", file=sys.stderr)
            return 2
        args.dtypes = dtype_list
    else:
        # 不指定则删除该属性，让各 suite 使用各自的默认 dtype 集合
        del args.dtypes

    print(f"[xpu-bench] 设备 {args.device}: {torch.xpu.get_device_name(args.device)}")
    print(f"[xpu-bench] suites: {', '.join(suites)}  "
          f"(warmup={args.warmup}, iters={args.iters})")

    store = ResultStore(meta=environment_meta(args.device, meta_args))

    telemetry = TelemetrySampler(args.device, args.telemetry) if args.telemetry else None
    if telemetry is not None:
        if telemetry.available:
            telemetry.start()
            print(f"[xpu-bench] 已启动 xpu-smi 遥测 -> {args.telemetry}")
        else:
            print("[xpu-bench] 未找到 xpu-smi，遥测已禁用")

    try:
        for name in suites:
            fn, desc = SUITES[name]
            print(f"[xpu-bench] === {name}: {desc} ===")
            with Stopwatch() as sw:
                fn(args, store)
            ok = len(store.suite(name))
            print(f"[xpu-bench] {name} 完成：{ok} 项有效测点，用时 {sw.elapsed:.1f}s")
    finally:
        if telemetry is not None:
            telemetry.stop()

    # 输出
    md = store.to_markdown()
    if not args.quiet:
        print()
        print(md)

    if not args.no_save:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(args.outdir)
        json_path = Path(args.json) if args.json else outdir / f"bench_{stamp}.json"
        md_path = Path(args.markdown) if args.markdown else outdir / f"bench_{stamp}.md"
        store.dump_json(json_path)
        store.dump_markdown(md_path)
        print(f"[xpu-bench] JSON -> {json_path}")
        print(f"[xpu-bench] MD   -> {md_path}")

    n_ok = sum(1 for r in store.results if r.status == "ok")
    n_bad = len(store.results) - n_ok
    print(f"[xpu-bench] 结束：{n_ok} 项成功，{n_bad} 项跳过/失败")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
