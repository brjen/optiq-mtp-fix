#!/usr/bin/env python3
"""`optiq serve --mtp` wrapper that makes MTP actually run alongside KV quantization.

mlx-optiq 0.4.25 hooks MTP on mlx-lm's *sequential* stream_generate path, but with a
batchable model (+ --kv-bits handled by install_batch_kv_quant) the server stays on
BatchGenerator, so --mtp is a silent no-op (measured: 0% speedup, engine never attached).
Fix 1: force the sequential path when --mtp is set.
Fix 2: the serve patch never forwards kv_bits to OptiqEngine.generate_stream (fp16 KV
        halves the usable context on a 24 GB machine); inject it from OPTIQ_MTP_KV_BITS
        (default 8).
Fix 3: the engine prefills the WHOLE prompt in one forward and materializes logits for
        every position (L x vocab) -> Metal OOM on a ~3K-token prompt (27B). Chunk the
        prefill (OPTIQ_MTP_PREFILL_STEP, default 256), keep lm_head to the last chunk,
        and convert KV to quantized after the first chunk so fp16 + quantized caches
        never co-reside.
Usage: optiq_mtp_wrapper.py serve --mtp --mtp-depth 1 ... (same argv as `optiq`)."""
import os
import sys

import mlx.core as mx
from optiq import serve as S
from optiq.cli import cli
from optiq.runtime import engine as E
from optiq.runtime.engine import OptiqEngine

TESTED_OPTIQ = "0.4.25"  # the version these private-attribute patches were written against
try:
    from importlib.metadata import version as _pkg_version
    _optiq_version = _pkg_version("mlx-optiq")
except Exception:  # pragma: no cover
    _optiq_version = "unknown"
if _optiq_version != TESTED_OPTIQ:
    print(f"[optiq-mtp-wrapper] WARNING: mlx-optiq {_optiq_version} != tested {TESTED_OPTIQ}; this wrapper "
          f"patches OptiqEngine._forward/generate_stream and serve.install_mtp_speculation — re-verify "
          f"with an A/B before trusting it", flush=True)

kv_bits = os.environ.get("OPTIQ_MTP_KV_BITS", "8")
kv_bits = int(kv_bits) if kv_bits not in ("", "0", "none") else None
kv_group = int(os.environ.get("OPTIQ_MTP_KV_GROUP", "64"))
prefill_step = int(os.environ.get("OPTIQ_MTP_PREFILL_STEP", "256"))

_orig_gs = OptiqEngine.generate_stream
_orig_fwd = OptiqEngine._forward


def _gs(self, *a, **k):
    if kv_bits is not None:
        k.setdefault("kv_bits", kv_bits)
        k.setdefault("kv_group_size", kv_group)
    return _orig_gs(self, *a, **k)


def _fwd(self, ids, cache, *, input_embeddings=None, per_layer_inputs=None):
    # Text-only prefill longer than one step: chunk it. Callers only use the last
    # position's hidden/logits, so returning the last chunk's is exact.
    if (input_embeddings is not None or per_layer_inputs is not None
            or prefill_step <= 0 or ids.shape[1] <= prefill_step):
        return _orig_fwd(self, ids, cache, input_embeddings=input_embeddings,
                         per_layer_inputs=per_layer_inputs)
    n = ids.shape[1]
    hidden = None
    for i in range(0, n, prefill_step):
        chunk = ids[:, i:i + prefill_step]
        last = i + prefill_step >= n
        hidden = self._inner(chunk, cache=cache)
        if last:
            logits = self._lm_head(hidden)
            mx.eval(hidden, logits)
            return hidden, logits
        mx.eval(hidden)
        # keep the growing cache quantized from the first chunk on
        E._maybe_quantize_kv(cache, kv_bits, kv_group)
        mx.clear_cache()
    raise AssertionError("unreachable")


OptiqEngine.generate_stream = _gs
OptiqEngine._forward = _fwd
_orig_install = S.install_mtp_speculation


def _install(model_path, depth=2):
    forced = S.force_sequential_for_kv_quant("--mtp (optiq-mtp-wrapper)")
    print(f"[optiq-mtp-wrapper] forced sequential path={forced}; engine kv_bits={kv_bits}; "
          f"chunked prefill step={prefill_step}", flush=True)
    return _orig_install(model_path, depth)


S.install_mtp_speculation = _install
sys.exit(cli())
