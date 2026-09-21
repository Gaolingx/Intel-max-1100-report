"""模型级（端到端）性能测试 —— 对应 ``docs/TODO/05-ai-dl.md`` §3.2 / §3.3 / §3.5。

补齐算子级 micro-benchmark 之外的「真实模型」测试：

====================  ==================================================
suite                 内容
====================  ==================================================
``resnet``            ResNet-50 训练吞吐：FP32 vs BF16 AMP、batch sweep、
                      channels_last、可选 ``torch.compile``（§3.2）
``bert``              BERT 训练吞吐：MLM 头，seq_len / batch sweep（§3.3）
                      模型由 ``--bert-model`` 指定（默认 bert-base-uncased，
                      已验证 bert-large-uncased）
``llm``               LLM 推理：prefill（TTFT）/ decode（tok/s）/ 显存峰值（§3.5）
====================  ==================================================

设计要点
--------
* 训练测点复用 :func:`xpu_bench.common.benchmark`（warmup + ``torch.xpu.Event``
  设备侧计时），与算子级结果口径一致，可交叉引用。
* 显存峰值用 ``torch.xpu.max_memory_allocated()`` 记录。
* FLOPs 采用业界通用口径，并在 ``note`` 中写明，避免与其它报告对不上：

  - ResNet-50：``fwd = 2 × 4.09 GMACs``（224×224），训练 ≈ ``3 × fwd``
  - BERT：``fwd ≈ 2 × N_params × n_tokens``，训练 ≈ ``3 × fwd``

* Transformers 权重默认从 ``HF_ENDPOINT`` 下载。**本机 huggingface.co 不可达**，
  默认走 ``https://hf-mirror.com``；若仍失败则回退到**本地 config 随机初始化**
  （结构完全相同，仅权重随机），此时 ``note`` 会标注 ``random-init``。
  吞吐（samples/s、TFLOPS）不受权重取值影响，可用于对标。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch

from .common import (
    DEVICE_TYPE,
    BenchResult,
    ResultStore,
    TimeStats,
    benchmark,
    tflops_from,
)

# 必须在 transformers / huggingface_hub 首次导入前设置（本机 huggingface.co 不可达）
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
# 避免在无 token 时反复尝试联网鉴权
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

SUITE_RESNET = "resnet"
SUITE_BERT = "bert"
SUITE_LLM = "llm"

# ResNet-50 @224×224：4.09 GMACs（乘加），fwd FLOPs = 2 × MACs
RESNET50_FWD_MACS = 4.09e9
RESNET50_FWD_FLOPS = 2.0 * RESNET50_FWD_MACS
# 训练一轮 ≈ fwd + bwd(data) + bwd(weights) ≈ 3 × fwd
TRAIN_FLOPS_FACTOR = 3.0

DEFAULT_RESNET_BATCHES = [64, 128, 256]
DEFAULT_BERT_SEQ_LENS = [128, 512]
DEFAULT_BERT_BATCHES = [16, 32]
DEFAULT_LLM_BATCHES = [1, 4, 8]
DEFAULT_LLM_INPUT_LENS = [128, 512, 2048]
DEFAULT_LLM_NEW_TOKENS = 32
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _kw(ns) -> dict:
    """从 argparse 命名空间取出计时参数（与其它 suite 保持一致）。"""
    return {
        "warmup": int(getattr(ns, "warmup", 5)),
        "iters": int(getattr(ns, "iters", 20)),
        "device_index": int(getattr(ns, "device", 0)),
    }


def _int_list(ns, attr: str, default: list[int]) -> list[int]:
    raw = getattr(ns, attr, None)
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in raw]


def _cfg_dtypes(ns, default: list[str]) -> list[str]:
    d = getattr(ns, "model_dtypes", None)
    if isinstance(d, str):
        d = [x.strip() for x in d.split(",") if x.strip()]
    return list(d) if d else list(default)


def _peak_mem_gib(device_index: int) -> float:
    return round(torch.xpu.max_memory_allocated(device_index) / (1024.0 ** 3), 3)


def _reset_peak(device_index: int) -> None:
    torch.xpu.synchronize(device_index)
    try:
        torch.xpu.reset_peak_memory_stats(device_index)
    except Exception:  # noqa: BLE001 - 部分后端无此 API
        pass


def _measure(
    store: ResultStore,
    suite: str,
    name: str,
    params: dict,
    fn: Callable[[], Any],
    kw: dict,
    metrics_fn: Callable[[TimeStats], dict],
    note: str = "",
) -> Optional[TimeStats]:
    """执行一次 benchmark 并登记；异常（含 OOM）自动降级为 error 记录。

    与 ``ResultStore.measure`` 的区别：额外捕获显存峰值。
    """
    dev = kw["device_index"]
    _reset_peak(dev)
    try:
        stats = benchmark(fn, warmup=kw["warmup"], iters=kw["iters"], device_index=dev)
    except Exception as exc:  # noqa: BLE001
        store.error(suite, name, params, exc)
        torch.xpu.empty_cache()
        return None
    metrics = metrics_fn(stats)
    metrics["peak_mem_gib"] = _peak_mem_gib(dev)
    store.add(
        BenchResult(suite=suite, name=name, params=params, stats=stats,
                    metrics=metrics, note=note)
    )
    return stats


# ---------------------------------------------------------------------------
# §3.2  ResNet-50 训练吞吐
# ---------------------------------------------------------------------------
def _build_resnet(device_index: int, channels_last: bool = False):
    from torchvision.models import resnet50

    model = resnet50(weights=None, num_classes=1000)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    device = torch.device(DEVICE_TYPE, device_index)
    return model.to(device)


def _make_resnet_step(model, batch: int, dtype_key: str, device_index: int,
                      channels_last: bool = False):
    """返回一个「zero_grad → fwd → bwd → step」的可调用对象。"""
    device = torch.device(DEVICE_TYPE, device_index)
    amp = dtype_key != "fp32"
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch,), device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16):
                loss = crit(model(x), y)
        else:
            loss = crit(model(x), y)
        loss.backward()
        opt.step()

    return step, x, y


def run_resnet(ns, store: ResultStore) -> None:
    """ResNet-50 训练吞吐（§3.2）。"""
    kw = _kw(ns)
    dev = kw["device_index"]
    batches = _int_list(ns, "model_batches", DEFAULT_RESNET_BATCHES)
    dtype_keys = _cfg_dtypes(ns, ["fp32", "bf16"])
    use_compile = bool(getattr(ns, "model_compile", False))
    last_level = getattr(ns, "model_extra", False)

    if last_level:
        batches = sorted(set(batches) | {512})

    flops_per_img = RESNET50_FWD_FLOPS * TRAIN_FLOPS_FACTOR
    note_flops = (f"训练 FLOPs = {TRAIN_FLOPS_FACTOR:.0f} × fwd，"
                  f"fwd = 2 × {RESNET50_FWD_MACS / 1e9:.2f} GMACs")

    for dk in dtype_keys:
        for batch in batches:
            for cl in (False, True):
                label = "channels_last" if cl else "contiguous"
                params = {"dtype": dk, "batch": batch, "img_size": 224,
                          "memory_format": label, "model": "resnet50"}
                model = None
                try:
                    model = _build_resnet(dev, channels_last=cl)
                    step, x, y = _make_resnet_step(model, batch, dk, dev,
                                                   channels_last=cl)
                    if use_compile:
                        step = torch.compile(step)  # type: ignore[assignment]
                        note = note_flops + "；torch.compile 已启用"
                        params["compile"] = True
                        step()  # 触发编译
                    else:
                        note = note_flops
                except Exception as exc:  # noqa: BLE001
                    store.error(SUITE_RESNET, "train_step", params, exc)
                    if model is not None:
                        del model
                    torch.xpu.empty_cache()
                    continue

                def metrics(stats: TimeStats, batch=batch) -> dict:
                    img_s = batch / (stats.median_ms * 1e-3)
                    return {
                        "img_per_s": img_s,
                        "ms_per_step": stats.median_ms,
                        "tflops": tflops_from(flops_per_img * batch, stats.median_ms),
                    }

                _measure(store, SUITE_RESNET, "train_step", params, step, kw,
                         metrics, note)
                del model, step, x, y
                torch.xpu.empty_cache()

    # 引擎侧汇总：BF16/FP32 加速比
    _resnet_summary(store, batches)


def _resnet_summary(store: ResultStore, batches: list[int]) -> None:
    for batch in batches:
        base = None
        amp = None
        for r in store.suite(SUITE_RESNET):
            if r.params.get("batch") != batch or r.params.get("memory_format") != "contiguous":
                continue
            if r.params.get("dtype") == "fp32":
                base = r
            elif r.params.get("dtype") in ("bf16", "fp16"):
                amp = r
        if base and amp:
            ratio = amp.metrics["img_per_s"] / base.metrics["img_per_s"]
            store.add_note(
                f"ResNet-50 batch={batch}: BF16 AMP / FP32 = **{ratio:.2f}×** "
                f"（{amp.metrics['img_per_s']:.0f} vs {base.metrics['img_per_s']:.0f} img/s）"
            )


# ---------------------------------------------------------------------------
# §3.3  BERT 训练吞吐
# ---------------------------------------------------------------------------
def _build_bert(ns, device_index: int, seq_len: int):
    """返回 (model, tokenizer_or_None, weights_tag)。

    优先从 HF_ENDPOINT 下载 ``bert-base-uncased``；失败则用本地 config
    随机初始化（结构相同）。
    """
    from transformers import BertConfig, BertForMaskedLM

    offline = bool(getattr(ns, "no_hf", False))
    model_id = str(getattr(ns, "bert_model", "bert-base-uncased"))
    if not offline:
        try:
            model = BertForMaskedLM.from_pretrained(model_id, dtype=torch.float32)
            return model, f"pretrained:{model_id}"
        except Exception as exc:  # noqa: BLE001
            print(f"[xpu-bench] BERT 权重下载失败（{type(exc).__name__}），"
                  f"回退到本地 config 随机初始化")
    cfg = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=max(512, seq_len),
    )
    return BertForMaskedLM(cfg), "random-init:bert-base(cfg)"


def run_bert(ns, store: ResultStore) -> None:
    """BERT 训练吞吐（§3.3）；模型由 ``--bert-model`` 指定。"""
    kw = _kw(ns)
    dev = kw["device_index"]
    seq_lens = _int_list(ns, "bert_seq_lens", DEFAULT_BERT_SEQ_LENS)
    batches = _int_list(ns, "bert_batches", DEFAULT_BERT_BATCHES)
    dtype_keys = _cfg_dtypes(ns, ["fp32", "bf16"])

    try:
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE_BERT, "train_step", {},
                   f"transformers 不可用（{exc}）；"
                   f"请 `uv pip install --python /root/workspace/venv1/bin/python transformers`")
        return

    for seq_len in seq_lens:
        for dk in dtype_keys:
            for batch in batches:
                params = {"dtype": dk, "batch": batch, "seq_len": seq_len,
                          "model": "bert-mlm"}
                # 每 token 一次前向：fwd ≈ 2 × N × T
                tokens = batch * seq_len
                n_params = None
                try:
                    model, weights_tag = _build_bert(ns, dev, seq_len)
                    n_params = sum(p.numel() for p in model.parameters())
                    model = model.to(torch.device(DEVICE_TYPE, dev))
                    # 从实际 config 回填结构（bert-base / bert-large 均可）
                    cfg_ = model.config
                    params["layers"] = int(getattr(cfg_, "num_hidden_layers", 0))
                    params["hidden"] = int(getattr(cfg_, "hidden_size", 0))
                    params["weights"] = weights_tag
                    params["params_m"] = round(n_params / 1e6, 1)

                    device = torch.device(DEVICE_TYPE, dev)
                    ids = torch.randint(0, 30522, (batch, seq_len), device=device)
                    attn = torch.ones_like(ids)
                    labels = ids.clone()
                    labels[:, ::7] = -100  # 部分位置不参与 loss（MLM 常规做法）

                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    amp = dk != "fp32"

                    def step() -> None:
                        opt.zero_grad(set_to_none=True)
                        if amp:
                            with torch.autocast(device_type=DEVICE_TYPE,
                                                dtype=torch.bfloat16):
                                out = model(input_ids=ids, attention_mask=attn,
                                            labels=labels)
                            loss = out.loss
                        else:
                            loss = model(input_ids=ids, attention_mask=attn,
                                         labels=labels).loss
                        loss.backward()
                        opt.step()

                    step()  # 预热一次以触发内核选择
                except Exception as exc:  # noqa: BLE001
                    store.error(SUITE_BERT, "train_step", params, exc)
                    torch.xpu.empty_cache()
                    continue

                fwd_flops = 2.0 * n_params * tokens
                train_flops = TRAIN_FLOPS_FACTOR * fwd_flops
                note = (f"fwd ≈ 2 × N_params × tokens，训练 = {TRAIN_FLOPS_FACTOR:.0f} × fwd"
                        f"（与 ResNet-50 口径一致）")

                def metrics(stats: TimeStats, batch=batch, tokens=tokens,
                            train_flops=train_flops) -> dict:
                    dt = stats.median_ms * 1e-3
                    return {
                        "seq_per_s": batch / dt,
                        "tokens_per_s": tokens / dt,
                        "ms_per_step": stats.median_ms,
                        "tflops": tflops_from(train_flops, stats.median_ms),
                    }

                _measure(store, SUITE_BERT, "train_step", params, step, kw,
                         metrics, note)
                del model, opt, ids, attn, labels
                torch.xpu.empty_cache()

    _bert_summary(store, seq_lens, batches)


def _bert_summary(store: ResultStore, seq_lens: list[int], batches: list[int]) -> None:
    tag = "BERT"
    for r in store.suite(SUITE_BERT):
        w = str(r.params.get("weights", ""))
        if w:
            tag = f"BERT {w.split(':')[-1]}"
            break
    for seq_len in seq_lens:
        for batch in batches:
            base = amp = None
            for r in store.suite(SUITE_BERT):
                if r.params.get("seq_len") != seq_len or r.params.get("batch") != batch:
                    continue
                if r.params.get("dtype") == "fp32":
                    base = r
                elif r.params.get("dtype") in ("bf16", "fp16"):
                    amp = r
            if base and amp:
                ratio = amp.metrics["seq_per_s"] / base.metrics["seq_per_s"]
                store.add_note(
                    f"{tag} seq={seq_len} batch={batch}: BF16 AMP / FP32 = "
                    f"**{ratio:.2f}×**（{amp.metrics['seq_per_s']:.1f} vs "
                    f"{base.metrics['seq_per_s']:.1f} seq/s）"
                )


# ---------------------------------------------------------------------------
# §3.5  LLM 推理（prefill / decode）
# ---------------------------------------------------------------------------
def _load_llm(ns, device_index: int):
    """加载 causal LM；失败则用本地 config 随机初始化（结构相同）。"""
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    model_id = str(getattr(ns, "llm_model", DEFAULT_LLM_MODEL))
    offline = bool(getattr(ns, "no_hf", False))
    if not offline:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16
            )
            model.eval()
            return model.to(torch.device(DEVICE_TYPE, device_index)), f"pretrained:{model_id}"
        except Exception as exc:  # noqa: BLE001
            print(f"[xpu-bench] LLM 权重下载失败（{type(exc).__name__}: {exc}），"
                  f"回退到本地 config 随机初始化")

    # Qwen2.5-0.5B 的结构参数（本地硬编码，避免联网取 config）
    from transformers import Qwen2Config

    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=896, intermediate_size=4864,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
        max_position_embeddings=32768, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval().to(torch.bfloat16)
    return model.to(torch.device(DEVICE_TYPE, device_index)), "random-init:qwen2.5-0.5b(cfg)"


def _llm_make_compiled(model, mode: str):
    """构造 (prefill_fn, decode_fn) 的 ``torch.compile`` 版本。

    prefill / decode 分别编译，避免同一张图里出现两套 shape 导致反复重编译。
    """

    def _prefill(ids):
        return model(input_ids=ids, use_cache=True)

    def _decode(tok, past):
        return model(input_ids=tok, past_key_values=past, use_cache=True)

    return (
        torch.compile(_prefill, mode=mode, dynamic=False),
        torch.compile(_decode, mode=mode, dynamic=False),
    )


def _llm_run(model, batch: int, prompt_len: int, new_tokens: int, device_index: int,
             compiled=None):
    """返回 (ttft_ms, decode_ms, n_decode_tokens)。

    TTFT = 首次前向（prefill）；decode = 其后 ``new_tokens - 1`` 次单 token 前向。
    ``compiled`` 为 ``(prefill_fn, decode_fn)`` 时走 torch.compile 路径。
    """
    device = torch.device(DEVICE_TYPE, device_index)
    ids = torch.randint(0, 1000, (batch, prompt_len), device=device)

    def _sync_time(fn) -> float:
        torch.xpu.synchronize(device_index)
        t0 = torch.xpu.Event(enable_timing=True)
        t1 = torch.xpu.Event(enable_timing=True)
        t0.record()
        out = fn()
        t1.record()
        torch.xpu.synchronize(device_index)
        return t0.elapsed_time(t1), out

    c_prefill, c_decode = compiled if compiled else (None, None)

    def prefill():
        with torch.no_grad():
            if c_prefill is not None:
                return c_prefill(ids)
            return model(input_ids=ids, use_cache=True)

    ttft, out = _sync_time(prefill)
    past = out.past_key_values
    tok = out.logits[:, -1:].argmax(-1)

    decode_ms = 0.0
    n_dec = max(0, new_tokens - 1)
    for _ in range(n_dec):
        def one_step(tok=tok, past=past):
            with torch.no_grad():
                if c_decode is not None:
                    return c_decode(tok, past)
                return model(input_ids=tok, past_key_values=past, use_cache=True)

        dt, out = _sync_time(one_step)
        decode_ms += dt
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1)

    del ids, out, past, tok
    return ttft, decode_ms, n_dec


def run_llm(ns, store: ResultStore) -> None:
    """LLM 推理：prefill / decode / 显存峰值（§3.5）。"""
    kw = _kw(ns)
    dev = kw["device_index"]
    batches = _int_list(ns, "llm_batches", DEFAULT_LLM_BATCHES)
    input_lens = _int_list(ns, "llm_input_lens", DEFAULT_LLM_INPUT_LENS)
    new_tokens = int(getattr(ns, "llm_new_tokens", DEFAULT_LLM_NEW_TOKENS))

    try:
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE_LLM, "inference", {},
                   f"transformers 不可用（{exc}）")
        return

    try:
        model, weights_tag = _load_llm(ns, dev)
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE_LLM, "inference", {}, f"模型加载失败：{type(exc).__name__}: {exc}")
        return

    dtype = next(model.parameters()).dtype
    n_params = sum(p.numel() for p in model.parameters())
    store.add_note(
        f"LLM 模型 `{getattr(ns, 'llm_model', DEFAULT_LLM_MODEL)}`，权重来源 "
        f"**{weights_tag}**，dtype={dtype}，参数量 {n_params / 1e6:.0f} M，"
        f"每测点生成 {new_tokens} token"
    )

    # 可选 torch.compile（eager 路径每 token 要下发 ~1800 个小 kernel，主机侧是瓶颈）
    compile_mode = str(getattr(ns, "llm_compile", "off") or "off").strip()
    compiled = None
    compile_tag = "off"
    if compile_mode.lower() not in ("off", "none", "false", "0", ""):
        try:
            compiled = _llm_make_compiled(model, compile_mode)
            compile_tag = compile_mode
            store.add_note(f"LLM：torch.compile(mode={compile_mode}) 已启用")
        except Exception as exc:  # noqa: BLE001
            store.add_note(f"LLM：torch.compile 启用失败（{type(exc).__name__}: {exc}），回退 eager")
            compiled = None
            compile_tag = "off"

    # 预热：覆盖每个将要用到的 (batch, prompt_len) 形状，避免 JIT/allocator
    # 首次开销进入第一个测点的计时（曾观察到首个 TTFT 虚高 20×）。
    try:
        for pl in input_lens:
            for b in batches:
                _llm_run(model, b, pl, 2, dev, compiled=compiled)
        torch.xpu.empty_cache()
    except Exception as exc:  # noqa: BLE001
        store.skip(SUITE_LLM, "inference", {}, f"预热失败：{type(exc).__name__}: {exc}")
        return

    for prompt_len in input_lens:
        for batch in batches:
            params = {"batch": batch, "input_len": prompt_len,
                      "new_tokens": new_tokens, "dtype": str(dtype),
                      "model": getattr(ns, "llm_model", DEFAULT_LLM_MODEL),
                      "weights": weights_tag, "compile": compile_tag}
            _reset_peak(dev)
            samples: list[tuple[float, float, int]] = []
            reps = max(1, int(getattr(ns, "iters", 20)) // 4)  # 生成昂贵，减少重复
            try:
                _llm_run(model, batch, prompt_len, new_tokens, dev,
                         compiled=compiled)  # 该形状再预热一次
                for _ in range(reps):
                    samples.append(_llm_run(model, batch, prompt_len, new_tokens, dev,
                                            compiled=compiled))
            except Exception as exc:  # noqa: BLE001
                store.error(SUITE_LLM, "inference", params, exc)
                torch.xpu.empty_cache()
                continue

            ttfts = sorted(s[0] for s in samples)
            ttft = ttfts[len(ttfts) // 2]
            decode_ms = sum(s[1] for s in samples) / len(samples)
            n_dec = samples[0][2]
            prefill_ms = ttft

            metrics = {
                "ttft_ms": ttft,
                "prefill_ms": prefill_ms,
                "prefill_tok_per_s": (batch * prompt_len) / (prefill_ms * 1e-3),
                "decode_tok_per_s": (
                    (batch * n_dec) / (decode_ms * 1e-3) if n_dec and decode_ms > 0 else 0.0
                ),
                "ms_per_decode_token": (decode_ms / n_dec) if n_dec else 0.0,
                "peak_mem_gib": _peak_mem_gib(dev),
            }
            store.add(BenchResult(
                suite=SUITE_LLM, name="inference", params=params, stats=None,
                metrics=metrics,
                note=f"每点 {reps} 次重复取 TTFT 中位数；decode 取均值（权重 {weights_tag}）",
            ))
            torch.xpu.empty_cache()

    _llm_summary(store)

    del model
    torch.xpu.empty_cache()


def _llm_summary(store: ResultStore) -> None:
    rows = [r for r in store.suite(SUITE_LLM) if r.metrics.get("decode_tok_per_s")]
    if not rows:
        return
    best = max(rows, key=lambda r: r.metrics["decode_tok_per_s"])
    p = best.params
    store.add_note(
        f"LLM decode 峰值 {best.metrics['decode_tok_per_s']:.1f} tok/s "
        f"（batch={p.get('batch')}, input_len={p.get('input_len')}）"
    )
    best_pf = max(rows, key=lambda r: r.metrics["prefill_tok_per_s"])
    store.add_note(
        f"LLM prefill 峰值 {best_pf.metrics['prefill_tok_per_s']:.0f} tok/s "
        f"（batch={best_pf.params.get('batch')}, input_len={best_pf.params.get('input_len')}），"
        f"TTFT {best_pf.metrics['ttft_ms']:.1f} ms"
    )
    # decode 随 batch 的变化（batch=1 常是带宽瓶颈）
    b1 = [r for r in rows if r.params.get("batch") == 1]
    if b1:
        worst = min(b1, key=lambda r: r.metrics["decode_tok_per_s"])
        store.add_note(
            f"LLM batch=1 decode 最低 {worst.metrics['decode_tok_per_s']:.1f} tok/s"
            f"（input_len={worst.params.get('input_len')}）→ 单序列 decode 为"
            f"**权重带宽受限**，不是算力受限"
        )
