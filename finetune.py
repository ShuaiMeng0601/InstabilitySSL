#!/usr/bin/env python3
"""Transfer the Pr=1 pipeline (frozen-pretrained encoder + ConvNeXtHead)
to the Pr=4 dataset, four ways. The encoder is the MAE ViT or the conv-JEPA,
detected from the checkpoint:

  zero-shot : Pr=1 encoder + Pr=1-trained head, no training; just evaluate.
  head-only : encoder frozen; train the ConvNeXt head on Pr=4 train cases.
  lora      : encoder frozen + LoRA adapters; train adapters + head.
              MAE: on attention qkv/proj of every block. conv-JEPA: on the two
              pointwise linears of every ConvNeXt block.
  full      : everything trainable (encoder at a lower lr via param groups).

Pr=4 file (derecho_holmboe_Pr4_dataset1.h5) is layout-compatible with the
Pr=1 dataset and carries its own case-level R_M labels in attrs
(case_table_json) and a whole-case 8/1/2 split. Discipline mirrors
convnext_rm.py: profile-level MSE, case-level eval (mean over profiles),
early stop on val case-MAE, test cases predicted once with the best model.

Usage:
  python finetune_pr4.py --mode lora \
      --enc-ckpt runs/small8_38case_v2/best.pt \
      --head-ckpt runs/convnext_small8_38case_v2/head_best.pt \
      --out runs/pr4_lora_small8_38case_v2
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

from config import make_config
from dataset import KHHolmboeDataset
from models import MAEViT
from models import ConvNeXtHead
from models import FiLMAxialHead
from models import ConvEncoder2D, split_context_target

PR4_H5 = str(Path(__file__).resolve().parent / "data/derecho_holmboe_Pr4_dataset1.h5")
DATA_H5 = PR4_H5  # overridden by --data; module-level so loaders see it


# ---------- LoRA ----------


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x, with W frozen and B zero-initialized."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        base_output = self.base(x)
        adapter_output = self.drop(x)
        adapter_output = self.lora_A(adapter_output)
        adapter_output = self.lora_B(adapter_output)
        return base_output + self.scale * adapter_output


def inject_lora(mae, rank, alpha, all_linear=False):
    """Adapters on attention qkv/proj; with all_linear also on the MLP fc1/fc2."""
    n = 0
    for blk in mae.enc_blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank, alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank, alpha)
        n += 2
        if all_linear:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank, alpha)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank, alpha)
            n += 2
    return n


def inject_lora_conv(enc, rank, alpha):
    """conv-JEPA: the pointwise linears hold almost all of a ConvNeXt block's
    weights, so they are the counterpart of the ViT's attention projections."""
    n = 0
    for stage in enc.res_blocks:
        for blk in stage:
            blk.pwconv1 = LoRALinear(blk.pwconv1, rank, alpha)
            blk.pwconv2 = LoRALinear(blk.pwconv2, rank, alpha)
            n += 2
    return n


# ---------- model ----------


class RMModel(nn.Module):
    """Frozen-or-not MAE encoder (no masking) + ConvNeXtHead."""

    def __init__(self, mae, head, encoder_grad):
        super().__init__()
        self.mae = mae
        self.head = head
        self.encoder_grad = encoder_grad

    @property
    def encoder(self):
        return self.mae

    def encode(self, x):
        m = self.mae
        tokens = m._embed_tokens(x)
        cls = m.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        for blk in m.enc_blocks:
            h = blk(h)
        h = m.enc_norm(h)
        return h[:, 1:].reshape(x.shape[0], m.V, m.L, -1)

    def forward(self, x):
        if self.encoder_grad:
            tok = self.encode(x)
        else:
            with torch.no_grad():
                tok = self.encode(x)
        return self.head(tok.float())


class JEPARMModel(RMModel):
    """conv-JEPA encoder + ConvNeXtHead(n_vars=1). Same token layout as
    extract_tokens_jepa.py with clean inputs: the two 100-frame halves are
    encoded separately and concatenated in t -> (B, 1, 620, D), y-major."""

    def __init__(self, enc, head, encoder_grad, n_ctx):
        nn.Module.__init__(self)
        self.enc = enc
        self.head = head
        self.encoder_grad = encoder_grad
        self.n_ctx = n_ctx

    @property
    def encoder(self):
        return self.enc

    def encode(self, x):
        ctx, tgt = split_context_target(x, self.n_ctx)
        z = torch.cat([self.enc(ctx), self.enc(tgt)], dim=-1)  # (B, D, 31, 20)
        return z.permute(0, 2, 3, 1).reshape(x.shape[0], 1, -1, z.shape[1])


# ---------- data ----------


def load_pr4_labels(labels_json=None):
    """R_M per case. Prefer --labels-json (comma-separated files, or inline JSON,
    mapping case_id -> R_M; several files are merged); otherwise read R_M from the
    data file's case_table_json (works for the Pr=4 file; the Pr=7 file carries no R_M)."""
    if labels_json:
        labels = {}
        for item in labels_json.split(","):
            p = Path(item.strip())
            table = json.loads(p.read_text() if p.exists() else labels_json)
            labels.update(
                {cid: float(v if not isinstance(v, dict) else v["R_M"]) for cid, v in table.items()}
            )
        return labels
    labels = {}
    for h5 in ([DATA_H5] if isinstance(DATA_H5, str) else DATA_H5):
        with h5py.File(h5, "r") as f:
            table = json.loads(f.attrs["case_table_json"])
        labels.update({cid: float(v["R_M"]) for cid, v in table.items() if "R_M" in v})
    if not labels:
        raise SystemExit(f"{DATA_H5} carries no R_M labels; pass --labels-json")
    return labels


def make_pr4_loader(split, batch_size, clean_only, shuffle, y_pad_to=496):
    ds = KHHolmboeDataset(
        DATA_H5,
        split,
        y_pad_to=y_pad_to,
        include_augmented=not (clean_only and split == "train"),
        exclude_invalid=False,
    )
    return ds, torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=(split == "train" and shuffle),
    )


@torch.no_grad()
def predict_split(model, ds, loader, device):
    model.eval()
    preds, cids = [], []
    for x, rows in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = model(x)
        preds.extend(p.float().cpu().tolist())
        cids.extend(ds.meta(r)["case_id"] for r in rows.tolist())
    model.train()
    return np.array(preds), cids


@torch.no_grad()
def predict_rows(model, ds, loader, device):
    """Per-profile predictions with each profile's position metadata (saved to CSV)."""
    model.eval()
    out = []
    for x, rows in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = model(x)
        for r, v in zip(rows.tolist(), p.float().cpu().tolist()):
            m = ds.meta(r)
            out.append((m["case_id"], m.get("plane", ""), m.get("axis_index", ""), float(v)))
    model.train()
    return out


def case_metrics(preds, cids, labels):
    per_case = defaultdict(list)
    for p, c in zip(preds, cids):
        per_case[c].append(float(p))
    rows = {
        c: {
            "true": labels[c],
            "pred": float(np.mean(v)),
            "pred_std": float(np.std(v)),
            "n_profiles": len(v),
        }
        for c, v in per_case.items()
    }
    y = np.array([r["true"] for r in rows.values()])
    p = np.array([r["pred"] for r in rows.values()])
    mae = float(np.abs(y - p).mean())
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    r2 = float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return {"mae": mae, "rmse": rmse, "r2": r2}, rows


ALL_GROUPS = ["train", "val", "test"]


def loader_rows(loader, ds):
    """Global metadata rows covered by a loader over ds (or a Subset of it)."""
    sub = loader.dataset
    pos = sub.indices if isinstance(sub, torch.utils.data.Subset) else range(len(ds))
    return [int(ds.indices[i]) for i in pos]


def make_custom_split(args):
    """Reassign the file's cases: --test-cases (required), --val-cases (optional),
    --train-cases (default: every other case). Samples are pooled over the file's
    train/val/test groups, clean profiles only. Without --val-cases, --val-frac of
    each training case's profiles is held out for early stopping."""
    csv_list = lambda s: [c.strip() for c in s.split(",") if c.strip()] if s else []
    test_cases, val_cases, train_cases = (
        csv_list(args.test_cases),
        csv_list(args.val_cases),
        csv_list(args.train_cases),
    )
    pool = KHHolmboeDataset(
        DATA_H5, ALL_GROUPS, y_pad_to=496, include_augmented=False, exclude_invalid=False
    )
    all_cases = sorted({m["case_id"] for m in pool.metadata})
    unknown = [c for c in test_cases + val_cases + train_cases if c not in all_cases]
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; file has {all_cases}")
    if not train_cases:
        train_cases = [c for c in all_cases if c not in test_cases and c not in val_cases]
    if set(train_cases) & set(test_cases) or set(val_cases) & set(test_cases):
        raise SystemExit("LEAKAGE: a test case is also in train or val")

    def case_ds(cases):
        return KHHolmboeDataset(
            DATA_H5,
            ALL_GROUPS,
            y_pad_to=496,
            include_augmented=False,
            exclude_invalid=False,
            keep_cases=cases,
        )

    def loader(ds, idx=None, shuffle=False):
        sub = ds if idx is None else torch.utils.data.Subset(ds, idx)
        return torch.utils.data.DataLoader(
            sub, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, drop_last=shuffle
        )

    tr_ds, te_ds = case_ds(train_cases), case_ds(test_cases)
    data = {"test": (te_ds, loader(te_ds))}
    if val_cases:
        va_ds = case_ds(val_cases)
        data["train"] = (tr_ds, loader(tr_ds, shuffle=True))
        data["val"] = (va_ds, loader(va_ds))
        val_desc = val_cases
    else:
        rng = np.random.RandomState(args.seed)
        tr_pos, va_pos = [], []
        for c in train_cases:
            pos = [i for i, r in enumerate(tr_ds.indices) if tr_ds.metadata[r]["case_id"] == c]
            rng.shuffle(pos)
            k = int(round(args.val_frac * len(pos)))
            va_pos += pos[:k]
            tr_pos += pos[k:]
        data["train"] = (tr_ds, loader(tr_ds, sorted(tr_pos), shuffle=True))
        data["val"] = (tr_ds, loader(tr_ds, sorted(va_pos)))
        val_desc = f"{args.val_frac:.0%} of each training case's profiles"
    record = {
        "train": train_cases,
        "val": val_desc,
        "test": test_cases,
        "note": "custom case assignment; samples pooled over the file's groups",
    }
    print(
        f"custom split: train {train_cases}\n              val {val_desc}\n"
        f"              test {test_cases}"
    )
    return data, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zero-shot", "head-only", "lora", "full"])
    ap.add_argument("--enc-ckpt", required=True)
    ap.add_argument(
        "--head-ckpt",
        help="Pr=1-trained ConvNeXt head to start from " "(omit for a randomly initialized head)",
    )
    ap.add_argument(
        "--data",
        help="target dataset h5 (default: the Pr=4 file); "
        "comma-separated files are concatenated split by split",
    )
    ap.add_argument(
        "--labels-json",
        help="case_id -> R_M mapping (file path(s), comma-"
        "separated, or inline JSON) when the data has no R_M",
    )
    ap.add_argument(
        "--init-ckpt",
        help="best.pt of a PREVIOUS finetune run: loads "
        "the full RMModel (encoder+LoRA+head) to "
        "continue a transfer chain; --head-ckpt is "
        "then ignored",
    )
    ap.add_argument(
        "--test-cases",
        help="comma-separated case_ids to test on; with this the "
        "file's own train/val/test groups are ignored and "
        "cases are reassigned (pooled over all groups)",
    )
    ap.add_argument(
        "--val-cases", help="comma-separated case_ids for early stopping " "(custom split only)"
    )
    ap.add_argument(
        "--train-cases",
        help="comma-separated case_ids to train on "
        "(custom split only; default: all other cases)",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="custom split without --val-cases: hold out this fraction of each "
        "training case's profiles as val (same-case profiles, like the "
        "Pr=7 file's own val group)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=3e-5, help="full mode only")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument(
        "--lora-all",
        action="store_true",
        help="LoRA on every linear layer of the encoder (attention qkv/proj "
        "AND MLP fc1/fc2), MAE only; default: attention only",
    )
    ap.add_argument(
        "--range-penalty",
        type=float,
        default=0.0,
        help="weight of an extra loss term mean(relu(pred-1)^2 + relu(-pred)^2) that "
        "penalises predictions outside [0, 1] (0 = off)",
    )
    ap.add_argument(
        "--sigmoid-output",
        action="store_true",
        help="ConvNeXt head output passed through a sigmoid, R_M in (0, 1)",
    )
    ap.add_argument(
        "--head-type",
        default="convnext",
        choices=["convnext", "film"],
        help="regression head; film = FiLM + axial attention (MAE tokens only)",
    )
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--clean-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global DATA_H5
    if args.data:
        DATA_H5 = args.data.split(",") if "," in args.data else args.data
    print(f"target data: {DATA_H5}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    # encoder
    ck = torch.load(args.enc_ckpt, map_location="cpu", weights_only=False)
    is_jepa = "encoder" in ck
    if is_jepa:
        jc = ck["config"]
        mae = ConvEncoder2D(3, jc["dims"], jc["blocks"], jc["strides"])
        mae.load_state_dict(ck["encoder"])
    else:
        cfg = make_config(ck["config"].get("preset", "small8"))
        mae = MAEViT(cfg)
        mae.load_state_dict(ck["model"])
    print(f"encoder ({'conv-JEPA' if is_jepa else 'MAE'}): {args.enc_ckpt} (epoch {ck['epoch']})")

    # head
    if args.head_type == "film":
        if is_jepa:
            raise SystemExit("FiLM head expects the 3-variable MAE token grid")
        head = FiLMAxialHead(n_vars=3)
    else:
        head = ConvNeXtHead(
            n_vars=1 if is_jepa else 3, out_act="sigmoid" if args.sigmoid_output else None
        )
    if args.sigmoid_output and args.head_type != "convnext":
        raise SystemExit("--sigmoid-output is implemented for the ConvNeXt head only")
    print(f"head output activation: {getattr(head, 'out_act', None)}")
    if args.head_ckpt and not args.init_ckpt:
        hck = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
        head.load_state_dict(hck["model"] if isinstance(hck, dict) and "model" in hck else hck)
        print(f"head ({args.head_type}) initialized from {args.head_ckpt}")

    # trainability by mode
    for p in mae.parameters():
        p.requires_grad_(False)
    if args.mode == "lora":
        n = (
            inject_lora_conv(mae, args.lora_rank, args.lora_alpha)
            if is_jepa
            else inject_lora(mae, args.lora_rank, args.lora_alpha, all_linear=args.lora_all)
        )
        print(f"LoRA: {n} adapters, rank {args.lora_rank}, alpha {args.lora_alpha}")
    if args.mode == "full":
        for p in mae.parameters():
            p.requires_grad_(True)

    encoder_grad = args.mode in ("lora", "full")
    model = (
        JEPARMModel(mae, head, encoder_grad, ck["config"]["n_ctx"])
        if is_jepa
        else RMModel(mae, head, encoder_grad)
    ).to(device)

    if args.init_ckpt:
        ick = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state = ick["model"]
        own = model.state_dict()
        if any(".base." in k for k in own) and not any(".base." in k for k in state):
            # previous stage was trained without adapters (full mode): its weights
            # become the frozen base of each LoRALinear, adapters start fresh
            remap = {k.replace(".base.", "."): k for k in own if ".base." in k}
            state = {remap.get(k, k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any("lora_" not in k for k in missing):
            raise SystemExit(
                f"init-ckpt mismatch: missing {missing[:5]} unexpected {unexpected[:5]}"
            )
        print(
            f"chained init: RMModel loaded from {args.init_ckpt} "
            f"(epoch {ick.get('epoch')}, val {ick.get('val_case_mae')}); "
            f"fresh LoRA tensors: {len(missing)}"
        )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.mode}  trainable params {n_train/1e6:.3f}M")

    labels = load_pr4_labels(args.labels_json)
    data = {}
    split_record = "file's own train/val/test groups"
    if args.test_cases:
        data, split_record = make_custom_split(args)
    else:
        for split in ["train", "val", "test"]:
            ds, loader = make_pr4_loader(
                split, args.batch_size, args.clean_only, shuffle=(split == "train")
            )
            data[split] = (ds, loader)
    for split, (ds, loader) in data.items():
        kept = sorted({ds.metadata[r]["case_id"] for r in loader_rows(loader, ds)})
        print(f"{split}: {len(loader.dataset)} samples, cases: {kept}")
    missing = [c for s in data.values() for m in s[0].metadata if (c := m["case_id"]) not in labels]
    assert not missing, f"missing Pr=4 labels: {sorted(set(missing))}"

    results = {
        "mode": args.mode,
        "trainable_params": n_train,
        "data": DATA_H5,
        "split": split_record,
        "head_type": args.head_type,
        "lora_all": args.lora_all,
        "sigmoid_output": args.sigmoid_output,
        "range_penalty": args.range_penalty,
        "enc_ckpt": args.enc_ckpt,
        "head_ckpt": args.head_ckpt,
        "init_ckpt": args.init_ckpt,
        "seed": args.seed,
    }

    if args.mode == "zero-shot":
        for split in ["val", "test"]:
            m, rows = case_metrics(*predict_split(model, *data[split], device), labels)
            results[split] = {"metrics": m, "per_case": rows}
            print(f"{split}: {json.dumps(m)}")
            for c, r in sorted(rows.items()):
                print(f"  {c:44s} true {r['true']:.4f}  pred {r['pred']:.4f}")
        (out / "results.json").write_text(json.dumps(results, indent=1))
        return

    # ---- training ----
    groups = []
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups.append({"params": head_params, "lr": args.lr_head})
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        enc_lr = args.lr_head if args.mode == "lora" else args.lr_enc
        groups.append({"params": enc_params, "lr": enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tr_ds, tr_loader = data["train"]
    tr_y = {m_i: labels[m["case_id"]] for m_i, m in enumerate(tr_ds.metadata)}

    logp = out / "log.csv"
    with open(logp, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "val_case_mae", "seconds"])

    best = {"mae": float("inf"), "epoch": -1}
    for epoch in range(args.epochs):
        t0 = time.time()
        meter, seen = 0.0, 0
        n_oor = 0
        for x, rows in tr_loader:
            x = x.to(device)
            y = torch.tensor([tr_y[r] for r in rows.tolist()], dtype=torch.float32, device=device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                mse = nn.functional.mse_loss(pred.float(), y)
                loss = mse
                if args.range_penalty > 0:
                    pf = pred.float()
                    loss = (
                        mse
                        + args.range_penalty
                        * (torch.relu(pf - 1.0) ** 2 + torch.relu(-pf) ** 2).mean()
                    )
            n_oor += int(((pred.float() > 1.0) | (pred.float() < 0.0)).sum())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            meter += float(mse) * x.shape[0]
            seen += x.shape[0]

        vm, _ = case_metrics(*predict_split(model, *data["val"], device), labels)
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}  train_mse {meter/max(seen,1):.5f}  "
            f"val case-MAE {vm['mae']:.4f}  train preds outside [0,1]: {n_oor}/{seen}  ({dt:.0f}s)",
            flush=True,
        )
        with open(logp, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{meter/max(seen,1):.6f}", f"{vm['mae']:.6f}", f"{dt:.1f}"]
            )

        if vm["mae"] < best["mae"]:
            best = {"mae": vm["mae"], "epoch": epoch}
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_case_mae": vm["mae"]},
                out / "best.pt",
            )
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch} (best {best['mae']:.4f} " f"@ {best['epoch']})")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    results["best_val_epoch"] = best["epoch"]
    for split in ["val", "test"]:
        m, rows = case_metrics(*predict_split(model, *data[split], device), labels)
        results[split] = {"metrics": m, "per_case": rows}
        print(f"\n{split} (best): {json.dumps(m)}")
        for c, r in sorted(rows.items()):
            print(f"  {c:44s} true {r['true']:.4f}  pred {r['pred']:.4f}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {out}/results.json")
    # every profile's prediction (val + test), with its plane / horizontal index
    with open(out / "per_profile_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "case_id", "plane", "axis_index", "true", "pred", "error"])
        for split in ["val", "test"]:
            for cid, plane, idx, p in predict_rows(model, *data[split], device):
                w.writerow(
                    [
                        split,
                        cid,
                        plane,
                        idx,
                        f"{labels[cid]:.4f}",
                        f"{p:.5f}",
                        f"{p - labels[cid]:+.5f}",
                    ]
                )
    print(f"saved {out}/per_profile_predictions.csv")


if __name__ == "__main__":
    main()
