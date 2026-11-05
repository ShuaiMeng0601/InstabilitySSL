"""Phase-1 MAE/ViT configuration. Everything tunable lives here."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent
DATA_H5 = str(PROJECT_DIR / "data/pr1_combined_38case_dataset_v2.h5")


@dataclass
class Config:
    # ---- data ----
    h5_path: str = DATA_H5
    variables: tuple = ("buoyancy", "reduced_shear", "log_epsilon")
    ny_raw: int = 491  # y points in the h5
    y_pad_to: int = 496  # edge-replicate pad (2 top, 3 bottom) so patches tile
    nt: int = 200
    include_augmented: bool = True  # train split only; val/test are clean anyway
    # Drop samples whose log_epsilon has positive values (domain violation left by
    # the build script's [-1, 1] noise clip). See scan_invalid_rows.py.
    exclude_invalid_log_epsilon: bool = True

    # ---- patches / masking ----
    patch_y: int = 16  # 496 / 16 = 31
    patch_t: int = 10  # 200 / 10 = 20  -> 620 spatial patches
    mask_ratio: float = 0.5  # fraction of (y, t) patches masked out

    # ---- model (preset-controlled) ----
    preset: str = "small8"
    enc_dim: int = 384
    enc_depth: int = 8
    enc_heads: int = 6
    dec_dim: int = 192
    dec_depth: int = 4
    dec_heads: int = 6
    mlp_ratio: float = 4.0

    # ---- training ----
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1.0e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 13
    val_mask_seed: int = 1234  # same val masks every epoch -> comparable val loss

    # ---- output ----
    out_dir: str = str(PROJECT_DIR / "runs/small8")

    @property
    def grid_y(self):
        assert self.y_pad_to % self.patch_y == 0
        return self.y_pad_to // self.patch_y

    @property
    def grid_t(self):
        assert self.nt % self.patch_t == 0
        return self.nt // self.patch_t

    @property
    def n_patches(self):
        return self.grid_y * self.grid_t

    @property
    def n_vars(self):
        return len(self.variables)

    def save(self, path):
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2))


PRESETS = {
    "tiny": dict(enc_dim=192, enc_depth=12, enc_heads=3, dec_dim=128, dec_depth=2, dec_heads=4),
    "small8": dict(enc_dim=384, enc_depth=8, enc_heads=6, dec_dim=192, dec_depth=4, dec_heads=6),
    "small12": dict(enc_dim=384, enc_depth=12, enc_heads=6, dec_dim=192, dec_depth=4, dec_heads=6),
    "base": dict(enc_dim=768, enc_depth=12, enc_heads=12, dec_dim=512, dec_depth=8, dec_heads=16),
}


def make_config(preset="small8", **overrides):
    cfg = Config(preset=preset, **PRESETS[preset])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field: {k}")
        setattr(cfg, k, v)
    if "out_dir" not in overrides:
        cfg.out_dir = (
            str(PROJECT_DIR / "runs" / preset)
        )
    return cfg
