#!/usr/bin/env python3
"""MAE pretraining: python train_mae.py --preset small8.

Settings are in config.py; command-line options override them.
Resume with: python train_mae.py --resume runs/small8/last.pt
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import make_config
from dataset import make_loader
from models import MAEViT, count_parameters


def lr_at(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, val_mask_seed, per_case=False):
    """Return overall loss, variable losses, and optional case losses."""
    model.eval()
    generator = torch.Generator().manual_seed(val_mask_seed)
    total_loss, n_samples = 0.0, 0
    variable_sums, case_sums, case_counts = {}, {}, {}
    dataset = loader.dataset
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    for x, rows in loader:
        x = x.to(device)
        loss, per_var, pred, mask = model(x, generator=generator)
        batch_size = len(x)
        total_loss += float(loss) * batch_size
        n_samples += batch_size
        for name, value in per_var.items():
            variable_sums[name] = variable_sums.get(name, 0.0) + value * batch_size
        if per_case:
            target = model.patchify(x)
            per_patch = ((pred - target) ** 2).mean(dim=-1)
            m = mask.unsqueeze(1).float()
            denom = m.sum(dim=(1, 2)) * model.V
            per_sample = (per_patch * m).sum(dim=(1, 2)) / denom
            for value, row in zip(per_sample.cpu().tolist(), rows.tolist()):
                case = dataset.meta(row)["case_id"]
                case_sums[case] = case_sums.get(case, 0.0) + value
                case_counts[case] = case_counts.get(case, 0) + 1

    variable_losses = {name: value / n_samples for name, value in variable_sums.items()}
    case_losses = {case: value / case_counts[case] for case, value in case_sums.items()}
    model.train()
    return total_loss / n_samples, variable_losses, case_losses


def main():
    # 1. Settings and command-line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small8")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out")
    parser.add_argument("--resume")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--clean-only", action="store_true", help="exclude training augmentation")
    parser.add_argument("--val-cases", help="comma-separated validation cases from val+test")
    parser.add_argument("--test-cases", help="comma-separated test cases; monitoring only")
    parser.add_argument("--ckpt-every", type=int, help="save an extra checkpoint every N epochs")
    parser.add_argument("--train-eval-n", type=int, default=0, help="evaluate a fixed train subset (0=off)")
    args = parser.parse_args()

    cfg = make_config(args.preset)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.out:
        cfg.out_dir = str(Path(args.out).resolve())
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.clean_only:
        cfg.include_augmented = False

    device = args.device
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    device = torch.device(device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    saved_config = json.loads((out / "config.json").read_text())

    # 2. Training, validation, and optional monitoring data.
    train_loader = make_loader(cfg, "train", limit=args.limit_train)
    val_cases = [c.strip() for c in (args.val_cases or "").split(",") if c.strip()]
    test_cases = [c.strip() for c in (args.test_cases or "").split(",") if c.strip()]
    val_split = ["val", "test"] if val_cases else "val"
    val_loader = make_loader(cfg, val_split, shuffle=False, limit=args.limit_val, keep_cases=val_cases or None)
    test_loader = None
    if val_cases:
        print(f"val override: {val_cases}")
    if test_cases:
        test_loader = make_loader(cfg, ["val", "test"], shuffle=False, keep_cases=test_cases)
        print(f"test monitor: {test_cases}")

    dataset = train_loader.dataset
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    train_cases = {m["case_id"] for m in dataset.metadata}
    for split, case_list in (("val", val_cases), ("test", test_cases)):
        for case in case_list:
            if case in train_cases:
                raise SystemExit(f"LEAKAGE: {split} case {case} is in the train split")
    if set(val_cases) & set(test_cases):
        raise SystemExit(f"val/test overlap: {set(val_cases) & set(test_cases)}")

    train_eval_loader = None
    if args.train_eval_n:
        stride = max(1, len(train_loader.dataset) // args.train_eval_n)
        indices = list(range(0, len(train_loader.dataset), stride))
        subset = torch.utils.data.Subset(train_loader.dataset, indices)
        train_eval_loader = torch.utils.data.DataLoader(
            subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        print(f"train-eval slice: {len(subset)} samples (stride {stride})")
    print(f"device={device}  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"test={len(test_loader.dataset)}")

    # 3. Model, optimizer, and optional checkpoint loading.
    model = MAEViT(cfg).to(device)
    print(f"preset={cfg.preset}  params={count_parameters(model)/1e6:.2f}M  "
          f"patches={cfg.grid_y}x{cfg.grid_t}={cfg.n_patches}  mask={cfg.mask_ratio}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    start_epoch, best_val, step = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)
        step = start_epoch * steps_per_epoch
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    monitored = val_cases + test_cases
    columns = ["epoch", "lr", "train_loss", "val_loss"]
    if train_eval_loader is not None:
        columns.append("train_loss_eval")
    columns += [f"val_{name}" for name in cfg.variables]
    columns += [f"case_{case}" for case in monitored]
    columns.append("seconds")
    log_path = out / "log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    # 4. Train one epoch, evaluate, and save.
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        total_loss, n_samples = 0.0, 0
        for x, _ in train_loader:
            lr = lr_at(step, total_steps, warmup_steps, cfg.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x = x.to(device)
            with torch.autocast("cuda", enabled=use_amp):
                loss, _, _, _ = model(x)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss) * len(x)
            n_samples += len(x)
            step += 1

        train_loss = total_loss / max(1, n_samples)
        val_loss, val_per_var, case_losses = evaluate(
            model, val_loader, device, cfg.val_mask_seed, per_case=bool(monitored)
        )
        if test_loader is not None:
            _, _, test_losses = evaluate(model, test_loader, device, cfg.val_mask_seed, per_case=True)
            case_losses.update(test_losses)
        train_eval = None
        if train_eval_loader is not None:
            train_eval, _, _ = evaluate(model, train_eval_loader, device, cfg.val_mask_seed)
        seconds = time.time() - start_time

        message = f"epoch {epoch:4d}  lr {lr:.2e}  train {train_loss:.5f}  val {val_loss:.5f}  "
        if train_eval is not None:
            message += f"train_eval {train_eval:.5f}  "
        message += "  ".join(f"{name.split('_')[-1]} {value:.5f}" for name, value in val_per_var.items())
        print(message + f"  ({seconds:.0f}s)", flush=True)

        row = [epoch, f"{lr:.6e}", f"{train_loss:.6f}", f"{val_loss:.6f}"]
        if train_eval is not None:
            row.append(f"{train_eval:.6f}")
        row += [f"{val_per_var[name]:.6f}" for name in cfg.variables]
        row += [f"{case_losses.get(case, float('nan')):.6f}" for case in monitored]
        row.append(f"{seconds:.1f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": saved_config,
            "best_val": best_val,
        }
        torch.save(checkpoint, out / "last.pt")
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            torch.save(checkpoint, out / f"ckpt_ep{epoch}.pt")
        if improved:
            torch.save(checkpoint, out / "best.pt")

    print(f"done. best val {best_val:.5f}  ->  {out}")


if __name__ == "__main__":
    main()
