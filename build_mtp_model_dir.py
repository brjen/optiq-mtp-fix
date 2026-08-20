#!/usr/bin/env python3
"""Build an optiq-style MTP model dir non-destructively: symlinks to your existing
MLX 4-bit weights + the mlx-community MTP head renamed to optiq's expected layout
(mtp.safetensors, keys prefixed `mtp.`) + the config stanza optiq reads.
The base model directory is untouched."""
import argparse
import json
import os
import pathlib

import mlx.core as mx

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="existing MLX 4-bit model dir (e.g. Qwen3.8-27B-MLX-4bit)")
p.add_argument("--mtp-head", required=True, help="MTP head safetensors (e.g. Qwen3.8-27B-MTP-MLX-4bit/model.safetensors)")
p.add_argument("--out", required=True, help="output model dir to create")
args = p.parse_args()

src = pathlib.Path(args.base)
head = pathlib.Path(args.mtp_head)
dst = pathlib.Path(args.out)
dst.mkdir(exist_ok=True)

# clean rebuild: drop every existing symlink in dst (stale links from an older source layout), then re-link
for old in dst.iterdir():
    if old.is_symlink():
        old.unlink()
for f in src.iterdir():
    if f.name in ("config.json",) or f.name.startswith("."):
        continue
    link = dst / f.name
    if link.exists():
        raise SystemExit(f"refusing to overwrite non-symlink {link}")
    os.symlink(f, link)

t = mx.load(str(head))
renamed = {("mtp." + k): v for k, v in t.items()}
mx.save_safetensors(str(dst / "mtp.safetensors"), renamed,
                    metadata={"format": "mlx", "source": f"{head} renamed mtp.*"})

cfg = json.loads((src / "config.json").read_text())
cfg.update({
    "mtp_file": "mtp.safetensors",
    "mtp_tensor_count": len(renamed),
    "mtp_policy": "optiq-int4-prequantized-gs64",
    # key name is optiq's own (optiq/runtime/mtp/artifacts.py reads `mtplx_mtp_quantization`), not a typo
    "mtplx_mtp_quantization": {"bits": 4, "group_size": 64, "mode": "affine", "policy": "all", "prequantized": True},
    "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
})
tc = cfg.get("text_config", cfg)
tc.setdefault("mtp_num_hidden_layers", 1)
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
print("built", dst, len(renamed), "mtp tensors;", sum(1 for _ in dst.iterdir()), "entries")
