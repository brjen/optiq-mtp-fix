#!/usr/bin/env python3
"""Benchmark different MTP depths to find the fastest for your model/hardware.

Loads the model once, then runs the same prompt at each depth (0=AR baseline,
1, 2, 3, ...) and reports tok/s, acceptance rate, and acceptance throughput.

Usage:
    python3 bench_mtp_depth.py \\
        --model /path/to/Qwen3.5-9B-OptiQ-4bit \\
        --kv-config /path/to/kv_config.json \\
        --depths 0 1 2 3 \\
        --max-tokens 128 \\
        --prompt "Explain the quicksort algorithm in 3 sentences."
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass
class RunResult:
    depth: int
    tok_s: float
    acceptance_rate: float
    generated: int
    prefill_s: float
    decode_s: float
    text: str


def load_engine(model_path: str, kv_config_path: str | None, kv_bits: int | None, kv_group: int):
    from optiq.runtime.engine import OptiqEngine
    from optiq import serve as S

    S.force_sequential_for_kv_quant("mtp-bench")

    print(f"[bench] loading model from {model_path} ...", flush=True)
    engine = OptiqEngine(model_path)

    if not engine.has_mtp:
        print("[bench] ERROR: model has no MTP head.", file=sys.stderr)
        sys.exit(1)

    print(f"[bench] MTP available: {engine.has_mtp}", flush=True)

    if kv_config_path and os.path.exists(kv_config_path):
        with open(kv_config_path) as f:
            kv_cfg = json.load(f)
        if isinstance(kv_cfg, list) and kv_cfg:
            bits_vals = [e.get("bits", 8) for e in kv_cfg]
            group_vals = [e.get("group_size", 64) for e in kv_cfg]
            avg_bits = sum(bits_vals) / len(bits_vals)
            print(f"[bench] loaded per-layer kv_config: {len(kv_cfg)} entries, "
                  f"avg_bits={avg_bits:.1f}, group={group_vals[0]}", flush=True)
            if kv_bits is None:
                kv_bits = int(round(avg_bits))
                kv_group = group_vals[0]
        elif isinstance(kv_cfg, dict):
            kv_bits = kv_cfg.get("kv_bits", kv_bits)
            kv_group = kv_cfg.get("kv_group_size", kv_group)
        print(f"[bench] using kv_bits={kv_bits}, kv_group={kv_group}", flush=True)
    elif kv_bits is not None:
        print(f"[bench] using kv_bits={kv_bits}, kv_group={kv_group}", flush=True)

    return engine, kv_bits, kv_group


def run_one(engine, prompt, depth, max_tokens, kv_bits, kv_group, warmup=False):
    label = f"depth={depth}" + (" (warmup)" if warmup else "")
    print(f"[bench] running {label} ...", flush=True)

    t0 = time.perf_counter()
    stats = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        depth=depth,
        temperature=0.0,
        kv_bits=kv_bits,
        kv_group_size=kv_group,
    )
    elapsed = time.perf_counter() - t0

    if warmup:
        return None

    return RunResult(
        depth=depth,
        tok_s=stats.decode_tok_s,
        acceptance_rate=stats.acceptance_rate,
        generated=stats.generated_tokens,
        prefill_s=stats.prefill_time_s,
        decode_s=stats.decode_time_s,
        text=stats.text,
    )


def print_table(results: list[RunResult], baseline: RunResult | None = None):
    print("\n" + "=" * 78)
    print(f"  {'Depth':>6}  {'tok/s':>8}  {'vs AR':>8}  {'Accept%':>8}  "
          f"{'Gen':>5}  {'Prefill':>8}  {'Decode':>8}")
    print("-" * 78)
    for r in results:
        delta = ""
        if baseline and baseline.tok_s > 0 and r.depth != baseline.depth:
            speedup = r.tok_s / baseline.tok_s
            delta = f"{speedup:.2f}x"
        elif r.depth == 0:
            delta = "(baseline)"
        print(f"  {r.depth:>6}  {r.tok_s:>8.2f}  {delta:>8}  {r.acceptance_rate:>7.1f}%  "
              f"{r.generated:>5}  {r.prefill_s:>7.2f}s  {r.decode_s:>7.2f}s")
    print("=" * 78)

    if len(results) >= 2:
        best = max(results, key=lambda r: r.tok_s)
        print(f"\n  Fastest: depth={best.depth} at {best.tok_s:.2f} tok/s")
        if baseline and best.depth != baseline.depth:
            print(f"  Speedup over AR: {best.tok_s / baseline.tok_s:.2f}x")
        if best.acceptance_rate > 0:
            print(f"  Acceptance rate: {best.acceptance_rate:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Benchmark MTP depth for generation speed")
    p.add_argument("--model", required=True, help="Path or HF id of the model")
    p.add_argument("--kv-config", help="Path to kv_config.json")
    p.add_argument("--kv-bits", type=int, default=8, help="KV quant bits (default 8)")
    p.add_argument("--kv-group", type=int, default=64, help="KV group size (default 64)")
    p.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="MTP depths to test (default: 0 1 2 3)")
    p.add_argument("--max-tokens", type=int, default=128, help="Tokens to generate (default 128)")
    p.add_argument("--prompt", default="Explain the quicksort algorithm concisely.",
                    help="Prompt to use for benchmarking")
    p.add_argument("--runs", type=int, default=3, help="Runs per depth for median (default 3)")
    p.add_argument("--warmup", action="store_true", help="Run one warmup pass before timing")
    args = p.parse_args()

    engine, kv_bits, kv_group = load_engine(args.model, args.kv_config, args.kv_bits, args.kv_group)

    if args.warmup:
        run_one(engine, args.prompt, 1, min(args.max_tokens, 16), kv_bits, kv_group, warmup=True)

    all_results: dict[int, list[RunResult]] = {}
    for depth in sorted(args.depths):
        runs = []
        for i in range(args.runs):
            r = run_one(engine, args.prompt, depth, args.max_tokens, kv_bits, kv_group)
            if r:
                runs.append(r)
        if runs:
            runs.sort(key=lambda r: r.tok_s)
            median = runs[len(runs) // 2]
            all_results[depth] = runs
            print(f"  depth={depth}: median {median.tok_s:.2f} tok/s "
                  f"(min={runs[0].tok_s:.2f}, max={runs[-1].tok_s:.2f})", flush=True)

    results = [all_results[d][len(all_results[d]) // 2] for d in sorted(all_results)]
    baseline = next((r for r in results if r.depth == 0), None)
    print_table(results, baseline)

    if results:
        best = max(results, key=lambda r: r.tok_s)
        print(f"\n  Sample output (depth={best.depth}):")
        snippet = best.text[:300]
        print(f"  {snippet}" + ("..." if len(best.text) > 300 else ""))


if __name__ == "__main__":
    main()
