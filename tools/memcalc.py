#!/usr/bin/env python3
"""memcalc.py - how much memory will this model need, and how fast can it go?

Two modes:

  1. Read a real GGUF file (exact numbers, reads only the header):
       python tools/memcalc.py models/Qwen3.5-9B-Q4_K_M.gguf --ctx 8192 32768 131072

  2. "What if" for a model you have not downloaded (dense transformer assumed):
       python tools/memcalc.py --params 8 --bpw 4.85 --layers 36 --kv-heads 8 \
           --head-dim 128 --ctx 32768

Add --bandwidth GBPS to get a rough ceiling on generation speed
(tokens/s ~= memory bandwidth / bytes read per token).

Needs the `gguf` package for mode 1:  pip install gguf
"""
from __future__ import annotations

import argparse
import os
import sys

GIB = 1024**3
MIB = 1024**2

# bytes per element for KV cache types llama.cpp supports (-ctk / -ctv)
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,  # 32 int8 values + one fp16 scale per block
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}


def fmt(nbytes: float) -> str:
    if nbytes >= GIB:
        return f"{nbytes / GIB:6.2f} GiB"
    return f"{nbytes / MIB:6.0f} MiB"


# ---------------------------------------------------------------------------
# GGUF reading
# ---------------------------------------------------------------------------

def read_gguf(path: str) -> dict:
    try:
        from gguf import GGUFReader
    except ImportError:
        sys.exit("This mode needs the gguf package:  pip install gguf")

    r = GGUFReader(path)

    def field(key, default=None):
        f = r.fields.get(key)
        if f is None:
            return default
        try:
            return f.contents()
        except Exception:  # very old gguf package versions
            return default

    arch = field("general.architecture")

    def a(key, default=None):
        return field(f"{arch}.{key}", default)

    # Sum tensor bytes, separating the parts that are NOT read for every token.
    total = expert = lookup_only = 0
    has_output = False
    for t in r.tensors:
        n = int(t.n_bytes)
        total += n
        name = t.name
        if "_exps" in name:          # MoE expert weights (ffn_up_exps, ...)
            expert += n
        if name.startswith("per_layer_token_embd"):  # Gemma per-layer embeddings
            lookup_only += n
        if name == "output.weight":
            has_output = True
    embd = sum(int(t.n_bytes) for t in r.tensors if t.name == "token_embd.weight")
    if has_output:
        # Untied embeddings: token_embd is only a row lookup per token.
        lookup_only += embd

    return {
        "path": path,
        "file_bytes": os.path.getsize(path),
        "arch": arch,
        "name": field("general.name", "?"),
        "n_layers": a("block_count"),
        "n_embd": a("embedding_length"),
        "n_head": a("attention.head_count"),
        "n_head_kv": a("attention.head_count_kv"),
        "k_len": a("attention.key_length"),
        "v_len": a("attention.value_length"),
        "k_len_swa": a("attention.key_length_swa"),
        "v_len_swa": a("attention.value_length_swa"),
        "swa": a("attention.sliding_window"),
        "swa_pattern": a("attention.sliding_window_pattern"),
        "shared_kv_layers": a("attention.shared_kv_layers", 0) or 0,
        "full_attn_interval": a("full_attention_interval"),
        "n_expert": a("expert_count", 0) or 0,
        "n_expert_used": a("expert_used_count", 0) or 0,
        "ctx_train": a("context_length"),
        "tensor_bytes": total,
        "expert_bytes": expert,
        "lookup_only_bytes": lookup_only,
    }


def kv_layers(m: dict):
    """Return a list of (kind, n_head_kv, k_len, v_len) for layers that hold a KV cache.

    kind is "full" (grows with context) or "swa" (capped at the sliding window).
    Architecture quirks are spelled out here on purpose - this IS the lesson:
    modern models avoid a full KV cache on every layer.
    """
    n = m["n_layers"]
    hkv = m["n_head_kv"]
    if isinstance(hkv, list):
        per_layer_hkv = hkv
    else:
        per_layer_hkv = [hkv] * n
    head_dim = (m["n_embd"] // m["n_head"]) if m["n_head"] else 0
    k_full = m["k_len"] or head_dim
    v_full = m["v_len"] or head_dim
    k_swa = m["k_len_swa"] or k_full
    v_swa = m["v_len_swa"] or v_full

    arch = m["arch"] or ""
    layers = []
    for i in range(n):
        if per_layer_hkv[i] == 0:
            continue  # layer without attention (some hybrids encode it this way)
        kind = "full"
        # Hybrid linear-attention models (Qwen3.5 / Qwen3-Next family):
        # only every Nth layer is full attention; the rest keep a small fixed state.
        if m["full_attn_interval"]:
            if (i + 1) % m["full_attn_interval"] != 0:
                continue
        # Sliding-window attention
        pattern = m["swa_pattern"]
        if isinstance(pattern, list) and len(pattern) == n:
            if pattern[i]:
                kind = "swa"
        elif m["swa"] and arch == "gpt-oss":
            kind = "swa" if i % 2 == 0 else "full"  # gpt-oss alternates
        layers.append((i, kind, per_layer_hkv[i],
                       k_swa if kind == "swa" else k_full,
                       v_swa if kind == "swa" else v_full))

    # Gemma 3n / Gemma 4 "E" models: the last N layers reuse earlier layers' KV.
    if m["shared_kv_layers"]:
        cutoff = n - m["shared_kv_layers"]
        layers = [l for l in layers if l[0] < cutoff]
    return layers


def kv_bytes(layers, ctx: int, swa: int | None, kv_type: str, parallel: int,
             ubatch: int = 512) -> float:
    b = KV_BYTES[kv_type]
    total = 0.0
    for _, kind, hkv, k, v in layers:
        cells = ctx
        if kind == "swa" and swa:
            # llama.cpp keeps the window plus one micro-batch per sequence,
            # rounded up to a multiple of 256 cells
            need = (swa + ubatch) * parallel
            cells = min(ctx, -(-need // 256) * 256)
        total += cells * hkv * (k + v) * b
    return total


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gguf", nargs="?", help="path to a .gguf file")
    p.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 32768, 131072],
                   help="context sizes to evaluate (tokens)")
    p.add_argument("--kv-type", default="f16", choices=sorted(KV_BYTES),
                   help="KV cache type (llama.cpp -ctk/-ctv), default f16")
    p.add_argument("--parallel", type=int, default=1,
                   help="parallel sequences (llama-server -np); context is shared")
    p.add_argument("--bandwidth", type=float, default=None,
                   help="memory bandwidth in GB/s, to estimate max tokens/s")
    p.add_argument("--budget", type=float, default=None,
                   help="memory you can give the model, in GiB (RAM or VRAM)")
    # what-if mode
    p.add_argument("--params", type=float, help="billions of parameters (what-if mode)")
    p.add_argument("--active-params", type=float,
                   help="billions of params used per token (MoE); defaults to --params")
    p.add_argument("--bpw", type=float, default=4.85,
                   help="bits per weight: Q4_K_M~4.85, Q5_K_M~5.7, Q6_K~6.6, Q8_0~8.5, F16=16")
    p.add_argument("--layers", type=int, help="number of layers (what-if)")
    p.add_argument("--kv-heads", type=int, help="KV heads (what-if)")
    p.add_argument("--head-dim", type=int, default=128, help="head dim (what-if)")
    args = p.parse_args()

    if args.gguf:
        m = read_gguf(args.gguf)
        weights = m["file_bytes"]
        layers = kv_layers(m)
        swa = m["swa"]
        expert = m["expert_bytes"]
        if m["n_expert"]:
            active = (m["tensor_bytes"] - expert - m["lookup_only_bytes"]
                      + expert * m["n_expert_used"] / m["n_expert"])
        else:
            active = m["tensor_bytes"] - m["lookup_only_bytes"]
        print(f"Model     : {m['name']}  (arch={m['arch']}, {m['n_layers']} layers)")
        if m["n_expert"]:
            print(f"MoE       : {m['n_expert_used']} of {m['n_expert']} experts used per token")
        full = sum(1 for l in layers if l[1] == "full")
        swa_n = sum(1 for l in layers if l[1] == "swa")
        print(f"KV layers : {full} full-attention, {swa_n} sliding-window (window={swa}), "
              f"{m['n_layers'] - full - swa_n} with no growing cache")
        print(f"Trained   : up to {m['ctx_train']} tokens of context")
    else:
        if not (args.params and args.layers and args.kv_heads):
            p.error("give a .gguf file, or --params, --layers and --kv-heads")
        weights = args.params * 1e9 * args.bpw / 8
        active = (args.active_params or args.params) * 1e9 * args.bpw / 8
        layers = [(i, "full", args.kv_heads, args.head_dim, args.head_dim)
                  for i in range(args.layers)]
        swa = None
        print(f"What-if   : {args.params}B params @ {args.bpw} bits/weight, "
              f"{args.layers} layers, {args.kv_heads} KV heads x {args.head_dim}")

    per_tok = kv_bytes(layers, 1, None, args.kv_type, 1)
    print(f"Weights   : {fmt(weights)}  (file size = memory for weights)")
    print(f"Per token : {per_tok / 1024:.1f} KiB of KV cache per full-attention token "
          f"({args.kv_type})")
    print(f"Read/token: ~{fmt(active)} of weights touched to generate one token")
    print()
    overhead = 0.3 * GIB  # compute buffers, runtime, small recurrent state
    print(f"{'context':>9} | {'KV cache':>10} | {'total est.':>10} | fits?")
    print("-" * 46)
    for c in args.ctx:
        kv = kv_bytes(layers, c, swa, args.kv_type, args.parallel)
        tot = weights + kv + overhead
        verdict = ""
        if args.budget:
            verdict = "yes" if tot <= args.budget * GIB * 0.9 else (
                "tight" if tot <= args.budget * GIB else "NO")
        print(f"{c:>9} | {fmt(kv):>10} | {fmt(tot):>10} | {verdict}")
    print("\n(total = weights + KV + ~0.3 GiB overhead; leave ~10-20% headroom for the OS/apps)")

    if args.bandwidth:
        tps = args.bandwidth * 1e9 / active
        print(f"\nSpeed ceiling at {args.bandwidth:.0f} GB/s: ~{tps:.1f} tokens/s "
              f"(real-world is usually 50-80% of this)")


if __name__ == "__main__":
    main()
