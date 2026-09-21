"""Markdown / 控制台报告渲染。

把 :class:`~xpu_bench.common.ResultStore` 中的结果渲染成可直接贴进
``docs/`` 的 Markdown 报告。
"""

from __future__ import annotations

from typing import Any

from .common import (
    ResultStore,
    alu_tflops,
    theoretical_tflops,
)


SECTION_TITLES = {
    "gemm": "② 算力峰值 — GEMM / matmul",
    "membw": "③ 显存带宽专项（size / dtype / host 传输）",
    "elementwise": "③ Elementwise 算子（有效带宽）",
    "reduce": "⑤ Reduction / Normalization",
    "attention": "⑤ Attention (SDPA)",
    "conv": "⑤ Convolution",
    "quant": "⑤ 量化 — INT8 稠密 / W8A16 / W4A16 权重独占",
    "precision": "② 数值格式支持矩阵 — 各精度 vector / matmul 实测",
    "resnet": "⑤ ResNet-50 训练吞吐（§3.2）",
    "bert": "⑤ BERT-base 训练吞吐（§3.3）",
    "llm": "⑤ LLM 推理 — prefill / decode（§3.5）",
    "ddp": "⑤ 双卡 DDP 扩展效率（§3.4）",
    "pipeline": "⑤ 数据管线 / 主机侧瓶颈（§3.6）",
}

SECTION_ORDER = ["gemm", "elementwise", "membw", "reduce", "attention", "conv",
                 "quant", "precision", "resnet", "bert", "llm", "ddp", "pipeline"]


# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        a = abs(value)
        if a >= 1000:
            return f"{value:,.0f}"
        if a >= 100:
            return f"{value:.1f}"
        if a >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _params_str(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())


def _metric_columns(rows) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r.metrics:
            if k not in cols:
                cols.append(k)
    return cols


def _table(rows) -> str:
    if not rows:
        return "_(无数据)_\n"
    metric_cols = _metric_columns(rows)
    header = ["item", "params", "median_ms", *metric_cols, "note"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [
            r.name,
            _params_str(r.params),
            _fmt(round(r.stats.median_ms, 4)) if r.stats else "-",
        ]
        for c in metric_cols:
            cells.append(_fmt(r.metrics.get(c)) if c in r.metrics else "-")
        cells.append(r.note or "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _yn(flag) -> str:
    return "是" if flag else "否"


def _precision_block(store: ResultStore) -> str:
    """precision suite 专用渲染：支持矩阵 + 各路径吞吐分表，避免宽表。"""
    rows = store.suite("precision")
    out: list[str] = []

    # --- (1) 支持矩阵 ---
    cap = [r for r in rows if r.name == "capability"]
    if cap:
        out.append("#### 1) 数值格式支持矩阵（真跑一次，非估算）")
        out.append("")
        out.append("| 格式 | dtype | 厂商:vector | 厂商:matrix | 实测 vector | 实测 matmul | 说明 |")
        out.append("|---|---|---|---|---|---|---|")
        for r in cap:
            p = r.params
            note = (r.note or "").replace("|", "\\|")
            out.append(
                f"| **{p.get('format')}** | `{p.get('torch_dtype')}` "
                f"| {_yn(p.get('hw_vector'))} | {_yn(p.get('hw_matrix'))} "
                f"| {r.metrics.get('vector')} | {r.metrics.get('matmul')} | {note} |"
            )
        out.append("")

    qops = [r for r in rows if r.name == "quantized_op"]
    if qops:
        out.append("#### 2) 专用量化 matmul 算子")
        out.append("")
        out.append("| 算子 | 状态 | 说明 |")
        out.append("|---|---|---|")
        for r in qops:
            note = (r.note or "").replace("|", "\\|")
            out.append(f"| `{r.params.get('op')}` | {r.metrics.get('status')} | {note} |")
        out.append("")

    vec = [r for r in rows if r.name == "vector"]
    if vec:
        out.append("#### 3) vector 路径 — elementwise 有效带宽 / 元素速率")
        out.append("")
        out.append(_table(vec))
        out.append("")

    mm = [r for r in rows if r.name == "matmul"]
    if mm:
        out.append("#### 4) 矩阵路径 — 各精度 matmul 吞吐")
        out.append("")
        out.append(_table(mm))
        out.append("")

    fp8 = [r for r in rows if r.name == "fp8_matmul"]
    if fp8:
        out.append("#### 4b) FP8 matmul（oneDNN 软件路径，无原生 FP8 XMX）")
        out.append("")
        out.append(_table(fp8))
        out.append("")

    blk = [r for r in rows if r.name == "block_matmul"]
    if blk:
        out.append("#### 4c) 块缩放 matmul — MXFP8 / MXFP4 / NVFP4（``torch._scaled_mm``）")
        out.append("")
        out.append(_table(blk))
        out.append("")

    wo = [r for r in rows if r.name == "weight_only"]
    if wo:
        out.append("#### 5) 低精度权重独占 — BF16 / W8A16 / W4A16 对照")
        out.append("")
        out.append(_table(wo))
        out.append("")

    acc = [r for r in rows if r.name == "accuracy"]
    if acc:
        out.append("#### 6) 数值精度自检（相对 fp64 参考）")
        out.append("")
        out.append(_table(acc))
        out.append("")

    return "\n".join(out)


def _meta_block(meta: dict) -> str:
    dev = meta.get("device", {})
    args = meta.get("arguments", {})
    lines = [
        f"- **时间**：{meta.get('timestamp', '-')}",
        f"- **主机**：{meta.get('hostname', '-')} / {meta.get('platform', '-')}",
        f"- **设备**：{dev.get('name', '-')}（device {dev.get('device_index', 0)}，"
        f"共 {dev.get('device_count', '-')} 张）",
        f"- **驱动版本**：{dev.get('driver_version', '-')}",
        f"- **EU 数**：{dev.get('gpu_eu_count', '-')}　"
        f"**Xe-core**：{dev.get('gpu_subslice_count', '-')}　"
        f"**L2/LLC**：{dev.get('last_level_cache_mb', '-')} MB",
        f"- **显存**：{dev.get('total_memory_gib', '-')} GiB　"
        f"**内存时钟**：{dev.get('memory_clock_mhz', '-')} MHz",
        f"- **torch**：{dev.get('torch_version', '-')}　"
        f"**triton**：{meta.get('triton_version', '-')}　"
        f"**python**：{dev.get('python_version', '-')}",
    ]
    if args:
        pretty = ", ".join(f"`{k}={v}`" for k, v in sorted(args.items()))
        lines.append(f"- **运行参数**：{pretty}")
    return "\n".join(lines) + "\n"


def _theory_block() -> str:
    alu = alu_tflops()
    lines = [
        "| dtype | 理论峰值 (TFLOPS) | 来源 |",
        "|---|---|---|",
        f"| FP32 | {alu:.2f} | 448 EU × 16 lane × 2 × 1.55 GHz（公式） |",
        f"| FP64 | {(theoretical_tflops('fp64') or 0.0):.2f} | FP32 × 0.78（本工具实测比值，非 1/2） |",
        "| FP16 / BF16 | 待实测 | 走 XMX，倍数必须由本工具实测确定 |",
        "| INT8 | 待实测 | 通常为 BF16 的 2×（本工具 `quant` suite 实测） |",
        "| INT4 | 待实测 | 通常为 BF16 的 4×（本工具 `quant` suite 实测） |",
        "",
        "> ⚠️ FP16/BF16/INT8/INT4 为未知项，请勿引用估计值作为结论；"
        "`quant` suite 与 GEMM 实测出的倍数以各自小节为准。",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_markdown(store: ResultStore) -> str:
    out: list[str] = []
    out.append("# XPU 理论性能基准测试报告")
    out.append("")
    out.append("> 由 `benchmark/05-ai-dl/run_bench.py` 自动生成；"
               "对应 `docs/TODO/02-compute-peak.md`、`03-memory-bandwidth.md`、`05-ai-dl.md`。")
    out.append("")
    out.append("## 一、环境")
    out.append("")
    out.append(_meta_block(store.meta))
    out.append("")
    out.append("## 二、理论参考值")
    out.append("")
    out.append(_theory_block())

    if store.notes:
        out.append("## 三、关键结论（自动摘要）")
        out.append("")
        for n in store.notes:
            out.append(f"- {n}")
        out.append("")

    out.append("## 四、明细结果")
    out.append("")
    for suite in SECTION_ORDER:
        rows = store.suite(suite)
        if not rows:
            continue
        out.append(f"### {SECTION_TITLES.get(suite, suite)}")
        out.append("")
        if suite == "precision":
            out.append(_precision_block(store))
        else:
            out.append(_table(rows))
        out.append("")

    # 跳过 / 失败项
    bad = [r for r in store.results if r.status != "ok"]
    if bad:
        out.append("### 跳过 / 失败项")
        out.append("")
        out.append("| suite | item | params | status | reason |")
        out.append("|---|---|---|---|---|")
        for r in bad:
            reason = (r.note or "").replace("|", "\\|")
            out.append(f"| {r.suite} | {r.name} | {_params_str(r.params)} | {r.status} | {reason} |")
        out.append("")

    out.append("---")
    out.append("")
    out.append("_提示：带宽类结论请与 `docs/TODO/03-memory-bandwidth.md` 的 "
               "BabelStream 结果交叉验证；算力类结论请与 `ze_peak` 交叉验证。_")
    out.append("")
    return "\n".join(out)
