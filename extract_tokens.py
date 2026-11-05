#!/usr/bin/env python3
"""Extract frozen-encoder PATCH TOKEN GRIDS (3, 620, 384) for the ConvNeXt head.

Unlike extract_features.py (pooled cls/mean/var_mean vectors), this saves the
full per-variable patch-token grid that the ConvNeXtHead consumes:
  h = encoder(x) with NO masking  ->  drop cls  ->  (3, 620, 384)
620 is y-major (L = y_index * 20 + t_index), inherited from patchify/sincos_2d.

Tokens are stored fp16 to halve the cache size; the head casts to fp32.

Usage:
  python extract_tokens.py --ckpt runs/small8_38case_b96e150/best.pt \
      --out ../features/tokens_b96e150.h5 --splits train val test \
      --clean-only-train --device cuda --batch-size 64
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from config import make_config
from dataset import KHHolmboeDataset
from models import MAEViT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--clean-only-train", action="store_true", help="skip augmented samples in the train split"
    )
    ap.add_argument("--limit", type=int, help="debug: only first N samples per split")
    ap.add_argument(
        "--data",
        help="dataset h5 to encode instead of the one the "
        "checkpoint was pretrained on (e.g. a Pr=4 / Pr=7 file)",
    )
    ap.add_argument("--cases", help="comma-separated case_ids: encode only these")
    args = ap.parse_args()
    keep = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    cfg = make_config(cfg_dict.get("preset", "small8"))
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except AttributeError:
                pass

    if args.data:
        cfg.h5_path = args.data
    device = torch.device(args.device)
    model = MAEViT(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}, best_val {ck.get('best_val')})")

    V, L, D = cfg.n_vars, cfg.n_patches, cfg.enc_dim
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as fout:
        fout.attrs["ckpt"] = str(args.ckpt)
        fout.attrs["ckpt_epoch"] = int(ck["epoch"])
        fout.attrs["token_shape"] = f"({V}, {L}, {D})  y-major: L = y*{cfg.grid_t} + t"
        fout.attrs["grid"] = f"{cfg.grid_y}x{cfg.grid_t}"
        fout.attrs["masking"] = "none (encode path, mask_ratio=0)"

        for split in args.splits:
            clean_only = args.clean_only_train and split == "train"
            ds = KHHolmboeDataset(
                cfg.h5_path,
                split,
                y_pad_to=cfg.y_pad_to,
                include_augmented=not clean_only,
                keep_cases=keep,
            )
            n = len(ds) if args.limit is None else min(args.limit, len(ds))
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, range(n)) if n < len(ds) else ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
            )

            g = fout.create_group(split)
            tok_ds = g.create_dataset(
                "tokens", shape=(n, V, L, D), dtype="float16", chunks=(1, V, L, D)
            )
            case_ids, w = [], 0
            with torch.no_grad():
                for x, rows in loader:
                    x = x.to(device)
                    with torch.autocast("cuda", enabled=device.type == "cuda"):
                        tokens = model._embed_tokens(x)  # (B, V*L, D)
                        cls = model.cls_token.expand(x.shape[0], -1, -1)
                        h = torch.cat([cls, tokens], dim=1)
                        for blk in model.enc_blocks:
                            h = blk(h)
                        h = model.enc_norm(h)
                    patches = h[:, 1:].reshape(x.shape[0], V, L, D)
                    tok_ds[w : w + x.shape[0]] = patches.half().cpu().numpy()
                    # rows are global row indices into the base dataset's metadata
                    for r in rows.tolist():
                        case_ids.append(ds.meta(r)["case_id"])
                    w += x.shape[0]
                    if w % 1920 == 0:
                        print(f"  {split}: {w}/{n}", flush=True)
            sd = h5py.string_dtype(encoding="utf-8")
            g.create_dataset("case_id", data=case_ids, dtype=sd)
            print(f"{split}: tokens {tok_ds.shape} fp16, {len(case_ids)} case_ids")

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
