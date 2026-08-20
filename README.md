# optiq-mtp-kv8

**Make `optiq serve --mtp` actually run when `--kv-bits` is set — plus chunked MTP prefill so long prompts don't Metal-OOM.**

Tested against **mlx-optiq 0.4.25** with Qwen3.8-27B (MLX 4-bit, group 64) + the mlx-community MTP head, on an M5 MacBook Air 24 GB.

## The bugs (mlx-optiq 0.4.25)

### 1. `--mtp` is a silent no-op when `--kv-bits` is set

optiq hooks MTP speculation onto mlx-lm's **sequential** `stream_generate` path. But with a batchable model — and `--kv-bits` handled by `install_batch_kv_quant` — the server stays on **BatchGenerator**, so the MTP engine never attaches. No warning, no error: you get byte-identical output and **0% speedup** while believing MTP is on.

If you followed the docs and enabled both KV quantization (for long context) and `--mtp` (for decode speed), you have this bug.

### 2. Even on the sequential path, `kv_bits` never reaches the engine

The serve patch doesn't forward `kv_bits` to `OptiqEngine.generate_stream`, so the MTP lane runs **fp16 KV** — on a 24 GB machine with a 27B model that cuts your usable context roughly in half (we measured a 64K ceiling with kv8 vs ~48K fp16).

### 3. The MTP engine prefills the whole prompt in one forward pass

`OptiqEngine._forward` runs the entire prompt in a single forward **and materializes logits for every position** (`L × vocab`). On Qwen3.8-27B that Metal-OOMs at roughly a **3K-token prompt**. Callers only use the last position's hidden state and logits, so chunking the prefill and keeping the `lm_head` projection to the final chunk is exact, not an approximation.

## The fix

[`optiq_mtp_wrapper.py`](optiq_mtp_wrapper.py) is a drop-in wrapper around the `optiq` CLI (same argv) that monkey-patches three things:

1. **Forces the sequential path** when `--mtp` is set, so the MTP engine actually attaches.
2. **Injects `kv_bits` / `kv_group_size`** into `OptiqEngine.generate_stream` (env `OPTIQ_MTP_KV_BITS`, default 8; `OPTIQ_MTP_KV_GROUP`, default 64).
3. **Chunks the engine prefill** (env `OPTIQ_MTP_PREFILL_STEP`, default 256), converts the KV cache to quantized after the first chunk (so fp16 and quantized caches never co-reside), and computes `lm_head` only on the last chunk.

```bash
python3 optiq_mtp_wrapper.py serve \
  --model /path/to/Qwen3.8-27B-MLX-4bit-mtp \
  --mtp --mtp-depth 1 --kv-bits 8 --max-context 64000
```

It pins to 0.4.25 with a loud warning on any other version — these are private-attribute patches; re-verify on upgrade.

## Building an MTP model dir from the community head

optiq expects the MTP head as `mtp.safetensors` (keys prefixed `mtp.`) inside the model directory. [`build_mtp_model_dir.py`](build_mtp_model_dir.py) builds one non-destructively: symlinks to your existing 4-bit weights + the renamed mlx-community MTP head + the config stanza optiq reads (`mtplx_mtp_quantization` — that key name is optiq's own).

```bash
python3 build_mtp_model_dir.py \
  --base /path/to/Qwen3.8-27B-MLX-4bit \
  --mtp-head /path/to/Qwen3.8-27B-MTP-MLX-4bit/model.safetensors \
  --out /path/to/Qwen3.8-27B-MLX-4bit-mtp
```

## Verify the silent no-op yourself

No wrapper needed to see bug 1 — on stock 0.4.25:

1. Serve with `optiq serve --model <qwen-mtp-dir> --mtp --mtp-depth 1 --kv-bits 8`.
2. Run any greedy (temp 0) generation and note tok/s. Restart **without** `--mtp` and run the same prompt.
3. Same speed, byte-identical output — and no log line about the MTP engine attaching. The speculation engine never loaded; `--mtp` changed nothing.

For bug 3: force the sequential path with the engine attached and send a ~3K+ token prompt — Metal OOM during prefill (one forward pass materializing `L × vocab` logits).

## Measured results (M5 Air 24 GB, Qwen3.8-27B MLX 4-bit, kv8, MTP depth 1)

| Metric | Stock 0.4.25 (`--mtp --kv-bits 8`) | With wrapper |
|---|---|---|
| MTP engine attached | no (silent) | yes |
| Greedy decode | 7.7–8.2 tok/s | **11.6–12.1 tok/s (1.4–1.6×)** |
| Prefill @6K / @30K | — (engine OOMs ≥ ~3K if forced) | 189 / 107 tok/s (= baseline) |
| 60K prompt prefill | OOM | clean (742 s, 80 tok/s) |
| Quality (6-prompt A/B vs no-MTP) | — | 6/6 correct, same tier (temp-0 trajectories differ: different verify kernels — expected for speculation) |

## Caveats

- Monkey-patches private internals of a specific release (**mlx-optiq 0.4.25 exactly** — the wrapper warns loudly on any other version; re-verify with your own A/B after upgrading). mlx-optiq has no public source repo or issue tracker as of this writing, so this repo doubles as the bug report; if the mlx-optiq authors read this — issues 1–3 above reproduce trivially and would be small fixes upstream.
- Tested **only** on Qwen3.8-27B (Qwen3.5-arch hybrid: 48 linear + 16 full-attention layers) with the mlx-community MTP head at **depth 1**. Other models, hybrid layouts, and deeper speculation are untested here.
- The quality A/B ran at **temperature 0** (6/6 correct, same answer tier). Trajectories still diverge from the no-MTP baseline at temp 0 — speculative verify kernels differ numerically — and under sampling they will diverge more. That's expected speculation behavior, not a quality loss, but judge it on your own workload.
- Vision/multimodal requests bypass the chunked prefill (`input_embeddings`/`per_layer_inputs` fall through to the original forward).

## License

MIT
