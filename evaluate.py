#!/usr/bin/env python3
"""Zero-shot R_M on a target dataset (Pr=4, Pr=7, ...) with heads trained on Pr=1 only.

Nothing is trained here. Each (encoder, head) pair was fit on the 30 Pr=1 train
cases; the frozen encoder's tokens of the target file's chosen splits go through
the head as is, and the per-profile predictions are averaged per case.
Labels come from the file's case_table_json (R_M) or from --labels-json.

Usage:
  python zeroshot_pr4.py --pr4-data <Pr4.h5> --out runs/zeroshot_pr4 \
      --model "MAE v2 + FiLM" film <pr4 tokens.h5> runs/film_axial_v2/axial_seed13_pr1.pt \
      --model "MAE v2 + ConvNeXt" convnext <pr4 tokens.h5> runs/convnext_.../head_best.pt
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from models import ConvNeXtHead
from models import FiLMAxialHead


def load_tokens(h5, splits):
    toks, cids = [], []
    with h5py.File(h5, "r") as f:
        for split in splits:
            cids += [c.decode() if isinstance(c, bytes) else c for c in f[f"{split}/case_id"][:]]
            toks.append(torch.from_numpy(f[f"{split}/tokens"][:]))
        return torch.cat(toks), cids, dict(f.attrs)


def build_head(kind, ckpt, n_vars, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if kind == "convnext":
        model, sd = ConvNeXtHead(n_vars=n_vars), sd["model"]
    elif kind == "film":
        model = FiLMAxialHead(n_vars=n_vars)
    else:
        raise SystemExit(f"unknown head kind: {kind}")
    model.load_state_dict(sd)  # strict: wrong pairing fails loudly
    return model.to(device).eval()


@torch.no_grad()
def predict(model, tok, device, bs=96):
    return torch.cat(
        [model(tok[i : i + bs].to(device).float()).float().cpu() for i in range(0, len(tok), bs)]
    ).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr4-data", required=True, help="target dataset h5")
    ap.add_argument(
        "--labels-json",
        help="case_id -> R_M json file; default: R_M from " "the data file's case_table_json",
    )
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument(
        "--model",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "KIND", "TOKENS", "HEAD_CKPT"),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.labels_json:
        labels = {c: float(v) for c, v in json.loads(Path(args.labels_json).read_text()).items()}
    else:
        with h5py.File(args.pr4_data, "r") as f:
            labels = {c: float(v["R_M"]) for c, v in json.loads(f.attrs["case_table_json"]).items()}

    res = {"data": args.pr4_data, "splits": args.splits, "models": {}}
    for name, kind, tokens, ckpt in args.model:
        tok, cids, attrs = load_tokens(tokens, args.splits)
        model = build_head(kind, ckpt, tok.shape[1], device)
        pred = predict(model, tok, device)
        rows = {}
        for c in sorted(set(cids)):
            p = pred[np.array([x == c for x in cids])]
            rows[c] = {
                "true": labels[c],
                "pred": float(p.mean()),
                "pred_std": float(p.std()),
                "n_profiles": int(len(p)),
            }
        err = [abs(r["true"] - r["pred"]) for r in rows.values()]
        res["models"][name] = {
            "head": kind,
            "tokens": tokens,
            "head_ckpt": ckpt,
            "encoder_ckpt": str(attrs.get("ckpt")),
            "mae": float(np.mean(err)),
            "per_case": rows,
        }
        print(f"\n{name}   (zero-shot, case-MAE {np.mean(err):.4f})")
        for c, r in rows.items():
            print(
                f"    {c:44s} true {r['true']:.4f}  pred {r['pred']:.4f}  "
                f"err {r['pred']-r['true']:+.4f}  (profile std {r['pred_std']:.3f}, "
                f"n={r['n_profiles']})"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nsaved {out}/results.json")


if __name__ == "__main__":
    main()
