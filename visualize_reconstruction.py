#!/usr/bin/env python3
"""Plot original / masked-input / MAE-reconstruction panels for a few samples.

Usage:
  python visualize_reconstruction.py --ckpt runs/small8/best.pt --split val --n 4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import make_config
from dataset import KHHolmboeDataset
from models import MAEViT

CMAPS = {
    "buoyancy": ("RdBu_r", -1, 1),
    "reduced_shear": ("RdBu_r", -1, 1),
    "log_epsilon": ("viridis", -1, 0),
}


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ck["config"]
    cfg = make_config(saved.get("preset", "small8"))
    for k, v in saved.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
            except Exception:
                pass
    model = MAEViT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("epoch", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", help="HDF5 path override if the checkpoint's original data path has moved")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    ap.add_argument("--dpi", type=int, default=300, help="PNG export resolution (default: 300 dpi)")
    ap.add_argument("--pdf", action="store_true", help="Also export a PDF")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg, epoch = load_model(args.ckpt, device)
    if args.data:
        cfg.h5_path = args.data
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "outputs/reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = KHHolmboeDataset(cfg.h5_path, args.split, y_pad_to=cfg.y_pad_to)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    gen = torch.Generator().manual_seed(args.seed)

    for i in picks:
        x, row = ds[int(i)]
        meta = ds.meta(row)
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            _, per_var, pred, mask = model(xb, generator=gen)
        target = model.patchify(xb)
        # paste: visible patches from target, masked patches from prediction
        m = mask.unsqueeze(1).unsqueeze(-1).to(target.dtype)  # (1,1,L,1)
        recon = model.unpatchify(pred * m + target * (1 - m))[0].cpu().numpy()
        # nan*0 == nan, so select with where() instead of arithmetic blending
        masked_in = (
            model.unpatchify(torch.where(m.bool(), torch.full_like(target, float("nan")), target))[
                0
            ]
            .cpu()
            .numpy()
        )
        orig = x.numpy()

        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for v, var in enumerate(cfg.variables):
            cmap, vmin, vmax = CMAPS[var]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.85")
            for c, (field, title) in enumerate(
                [
                    (orig[v], "original"),
                    (masked_in[v], "masked input"),
                    (recon[v], f"reconstruction (mse {per_var[var]:.4f})"),
                ]
            ):
                ax = axes[v, c]
                im = ax.imshow(field, aspect="auto", origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
                ax.set_title(f"{var} — {title}", fontsize=9)
                ax.set_xlabel("t index")
                ax.set_ylabel("y index")
                # Label data panels in row-major order (exclude colorbar axes).
                ax.text(
                    0.025,
                    0.975,
                    f"({chr(ord('a') + v * axes.shape[1] + c)})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        facecolor="white", edgecolor="none", alpha=0.85, boxstyle="square,pad=0.2"
                    ),
                )
                fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"{meta['case_id']}  {meta['plane']}@{meta['axis_index']}  "
            f"augment={meta.get('augment','clean')}  epoch={epoch}",
            fontsize=10,
        )
        fname = out_dir / f"recon_{args.split}_{row}.png"
        fig.savefig(fname, dpi=args.dpi)
        if args.pdf:
            fig.savefig(fname.with_suffix(".pdf"))
        plt.close(fig)
        print(f"saved {fname}")


if __name__ == "__main__":
    main()
