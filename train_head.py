#!/usr/bin/env python3
"""Train the ConvNeXt head with an ARBITRARY val/test case assignment.

The token cache groups samples by the dataset's original split, but the 8
held-out cases can be reassigned freely: pretraining only took gradients from
the 30 train cases, and the production encoder is last.pt (never selected with
val), so any reshuffle among the held-out 8 stays leakage-free. The fit set is
always the cache's train group (30 cases) — untouched.

Usage:
  python train_head.py --tokens ../features/tokens_<run>.h5 \
      --val-cases "Holmboe:Rib030_a05_Re1000,Holmboe:Rib016_a10_Re2000,KH:Ri030_a05_Re1000,KH:Ri020_a10_Re2000" \
      --test-cases "Holmboe:Rib020_a05_Re2000,Holmboe:Rib012_a10_Re1000,KH:Ri020_a05_Re1000,KH:Ri016_a10_Re2000" \
      --out runs/convnext_<run>_split2
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from dataset import load_rm_labels
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "mae": float(np.abs(y - p).mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "r2": float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
    }


@torch.no_grad()
def predict(model, tok, device, bs=96):
    model.eval()
    out = []
    for start in range(0, len(tok), bs):
        batch = tok[start:start + bs].to(device).float()
        prediction = model(batch).cpu()
        out.append(prediction)
    model.train()
    return torch.cat(out).numpy()
from models import ConvNeXtHead, norm_case_id


def load_all(h5):
    """Return {case_id: tensor of its samples} over every split in the cache."""
    by_case = defaultdict(list)
    groups = {}
    with h5py.File(h5, "r") as f:
        for split in ("train", "val", "test"):
            if split not in f:
                continue
            cids = [c.decode() if isinstance(c, bytes) else c for c in f[f"{split}/case_id"][:]]
            tok = f[f"{split}/tokens"]
            for i, c in enumerate(cids):
                by_case[c].append((split, i))
            groups[split] = tok[:]
    out = {}
    for c, items in by_case.items():
        parts = [groups[s][i] for s, i in items]
        out[c] = torch.from_numpy(np.stack(parts))
    return out, {
        s: set(c for c, its in by_case.items() if any(x[0] == s for x in its)) for s in groups
    }




def prediction_range_penalty(pred, mode="both"):
    """Squared upper-bound penalty, optionally including the lower bound."""
    penalty = torch.relu(pred - 1.0).square()
    if mode == "both":
        penalty = penalty + torch.relu(-pred).square()
    elif mode != "upper":
        raise ValueError(f"unknown range penalty mode: {mode}")
    return penalty.mean()




def eval_cases(model, cases, by_case, labels, device):
    rows = {}
    for c in cases:
        p = predict(model, by_case[c], device)
        rows[c] = {
            "true": labels[norm_case_id(c)],
            "pred": float(np.mean(p)),
            "n_profiles": int(len(p)),
            "pred_std": float(np.std(p)),
        }
    m = metrics([r["true"] for r in rows.values()], [r["pred"] for r in rows.values()])
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--val-cases", required=True)
    ap.add_argument("--test-cases", required=True)
    ap.add_argument("--labels-csv", default="RM_summary_table.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--range-penalty",
        type=float,
        default=0.0,
        help="weight of mean(relu(pred-1)^2 + relu(-pred)^2), penalising predictions outside [0, 1]",
    )
    ap.add_argument(
        "--range-penalty-mode",
        choices=("both", "upper"),
        default="both",
        help="both: penalise predictions outside [0,1] (original behaviour); "
        "upper: penalise only predictions above 1",
    )
    ap.add_argument(
        "--sigmoid-output",
        action="store_true",
        help="bound the head output with a sigmoid, R_M in (0, 1)",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="train all --epochs and also report the last epoch (head_last.pt); test MAE is "
        "logged every epoch for reference only, never used for selection",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    labels = load_rm_labels(args.labels_csv)
    by_case, orig = load_all(args.tokens)
    val_cases = [c.strip() for c in args.val_cases.split(",") if c.strip()]
    test_cases = [c.strip() for c in args.test_cases.split(",") if c.strip()]

    held_out = orig.get("val", set()) | orig.get("test", set())
    for c in val_cases + test_cases:
        if c not in by_case:
            raise SystemExit(f"case not in token cache: {c}")
        if c not in held_out or c in orig.get("train", set()):
            raise SystemExit(
                f"LEAKAGE: {c} is a pretraining train case; "
                "reassigning it requires re-pretraining."
            )
    if set(val_cases) & set(test_cases):
        raise SystemExit(f"val/test overlap: {set(val_cases) & set(test_cases)}")

    fit_cases = sorted(orig["train"])
    fit_tok = torch.cat([by_case[c] for c in fit_cases])
    if len(fit_tok) < args.batch_size:
        raise ValueError("Training set is smaller than batch size; reduce --batch-size.")
    fit_y = torch.cat([torch.full((len(by_case[c]),), labels[norm_case_id(c)]) for c in fit_cases])
    print(f"fit: {len(fit_cases)} cases / {len(fit_tok)} samples (pretraining train split)")
    for tag, cs in [("val", val_cases), ("test", test_cases)]:
        rng = [labels[norm_case_id(c)] for c in cs]
        print(
            f"{tag}: {len(cs)} cases, R_M {min(rng):.3f}-{max(rng):.3f} " f"(std {np.std(rng):.3f})"
        )
        for c in cs:
            print(
                f"    {c:36s} R_M {labels[norm_case_id(c)]:.4f}  (orig "
                f"{'val' if c in orig.get('val',()) else 'test'})"
            )

    model = ConvNeXtHead(
        enc_dim=fit_tok.shape[-1], n_vars=fit_tok.shape[1],
        out_act="sigmoid" if args.sigmoid_output else None
    ).to(device)
    print(f"head output activation: {model.out_act}")
    print(f"range penalty: lambda={args.range_penalty}, mode={args.range_penalty_mode}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    spe = max(1, len(fit_tok) // args.batch_size)
    total, warm = args.epochs * spe, args.warmup_epochs * spe

    logp = out / "log.csv"
    with open(logp, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_mse", "val_case_mae", "val_case_r2", "seconds"]
            + (["test_case_mae"] if args.no_early_stop else [])
        )
    es = None  # epoch at which patience would have stopped the run (with --no-early-stop)

    best = {"mae": float("inf"), "epoch": -1}
    gstep = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(fit_tok))
        meter = seen = 0
        n_oor = 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[i : i + args.batch_size]
            lr = (
                args.lr * (gstep + 1) / warm
                if gstep < warm
                else args.lr * 0.5 * (1 + np.cos(np.pi * (gstep - warm) / max(1, total - warm)))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            pred = model(fit_tok[idx].to(device).float())
            mse = nn.functional.mse_loss(pred, fit_y[idx].to(device))
            loss = mse
            if args.range_penalty > 0:
                loss = mse + args.range_penalty * prediction_range_penalty(
                    pred, args.range_penalty_mode
                )
            n_oor += int(((pred > 1.0) | (pred < 0.0)).sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            meter += float(mse) * len(idx)
            seen += len(idx)
            gstep += 1
        vm, _ = eval_cases(model, val_cases, by_case, labels, device)
        tm = (
            eval_cases(model, test_cases, by_case, labels, device)[0]
            if args.no_early_stop
            else None
        )
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}  train_mse {meter/max(seen,1):.5f}  "
            f"val case-MAE {vm['mae']:.4f}  R2 {vm['r2']:.3f}  train preds outside [0,1]: {n_oor}/{seen}  ({dt:.0f}s)",
            flush=True,
        )
        with open(logp, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{meter/max(seen,1):.6f}",
                    f"{vm['mae']:.6f}",
                    f"{vm['r2']:.6f}",
                    f"{dt:.1f}",
                ]
                + ([f"{tm['mae']:.6f}"] if tm else [])
            )
        if vm["mae"] < best["mae"]:
            best = {"mae": vm["mae"], "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_case_mae": vm["mae"],
                    "out_act": model.out_act,
                    "args": vars(args),
                },
                out / "head_best.pt",
            )
        elif epoch - best["epoch"] >= args.patience:
            if not args.no_early_stop:
                print(f"early stop at {epoch} (best {best['mae']:.4f} @ {best['epoch']})")
                break
            if es is None:
                es = {"epoch": epoch, "val": vm, "test": tm}
                print(
                    f"(early stopping would stop here: {epoch}, best {best['mae']:.4f} @ {best['epoch']})"
                )

    last = None
    if args.no_early_stop:
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "out_act": model.out_act,
                "args": vars(args),
            },
            out / "head_last.pt",
        )
        last = {"epoch": epoch, "earlystop_would_stop": es}
        for tag, cs in [("val", val_cases), ("test", test_cases)]:
            m, rows = eval_cases(model, cs, by_case, labels, device)
            last[tag] = {"metrics": m, "per_case": rows}
            print(f"\n{tag} (last epoch {epoch}): {json.dumps(m)}")

    model.load_state_dict(torch.load(out / "head_best.pt", map_location=device)["model"])
    res = {
        "tokens": args.tokens,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "best_val_epoch": best["epoch"],
        "out_act": model.out_act,
        "range_penalty": args.range_penalty,
        "range_penalty_mode": args.range_penalty_mode,
        "args": vars(args),
        "epochs_completed": epoch + 1,
        "updates_completed": gstep,
        "updates_per_epoch": spe,
    }
    if last is not None:
        res["last"] = last
    for tag, cs in [("val", val_cases), ("test", test_cases)]:
        m, rows = eval_cases(model, cs, by_case, labels, device)
        res[tag] = {"metrics": m, "per_case": rows}
        print(f"\n{tag}: {json.dumps(m)}")
        for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["true"]):
            print(
                f"  {c:36s} true {r['true']:.4f}  pred {r['pred']:.4f}  "
                f"err {abs(r['true']-r['pred']):.4f}  (profile std {r['pred_std']:.3f})"
            )
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nsaved {out}/results.json")


if __name__ == "__main__":
    main()
