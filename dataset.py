"""Dataset for kh_holmboe_dataset1 — single merged h5 OR multiple part files.

`h5_path` can be: one file path, a glob pattern (e.g. ".../dataset1_part*.h5"),
or a list of paths. Part files are read in sorted order, which reproduces the
same global sample order as the merged file.

Each sample: (3, 491, 200) float32, already normalized (see h5 attrs).
Here we only edge-replicate pad y 491 -> y_pad_to (2 top / 3 bottom for 496)
so the patch grid tiles exactly. No other transform; masking happens later.
"""

import glob as _glob
import json
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LOG_EPS_NAN_FILL = -0.62  # mean of the normalized log_epsilon channel over Pr=1 train


def load_invalid_rows(h5_path, split):
    """Global rows flagged by scan_invalid_rows.py (log_epsilon > 0).

    The side-car sits next to the (first) dataset file. Rows are GLOBAL indices,
    matching the concatenation order used here, so this works for the merged file
    and for part files alike. Returns [] when no side-car is present.
    """
    first = _resolve_paths(h5_path)[0]
    p = Path(first).with_name("invalid_log_epsilon_rows.json")
    if not p.exists():
        return []
    return json.loads(p.read_text()).get(split, [])


def _resolve_paths(h5_path):
    if isinstance(h5_path, (list, tuple)):
        paths = [str(Path(p).expanduser()) for p in h5_path]
    else:
        p = str(Path(str(h5_path)).expanduser())
        paths = sorted(_glob.glob(p)) if "*" in p else [p]
    if not paths:
        raise FileNotFoundError(f"no dataset files match: {h5_path}")
    return paths


class KHHolmboeDataset(Dataset):
    def __init__(
        self,
        h5_path,
        split,
        y_pad_to=496,
        include_augmented=True,
        exclude_invalid=True,
        keep_cases=None,
    ):
        """split: one group name, or a list of groups to concatenate.
        keep_cases: if given, keep only samples whose case_id is in it — this is
        how a val/test reassignment among the held-out cases is expressed without
        rebuilding the h5 (the held-out cases live in the 'val' and 'test' groups)."""
        self.paths = _resolve_paths(h5_path)
        self.split = split
        groups = [split] if isinstance(split, str) else list(split)
        self.groups = groups
        self.y_pad_to = y_pad_to
        self._files = [None] * len(self.paths)  # opened lazily per worker

        self.metadata = []
        self._loc = []  # global row -> (file_i, group, local_row)
        for fi, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                for grp in groups:
                    n = f[f"{grp}/X"].shape[0]
                    self.ny_raw = f[f"{grp}/X"].shape[2]
                    meta_raw = f[f"{grp}/metadata_json"][:]
                    self.metadata.extend(json.loads(s) for s in meta_raw)
                    self._loc.extend((fi, grp, r) for r in range(n))
        self.n_total = len(self._loc)

        if include_augmented:
            idx = np.arange(self.n_total)
        else:
            idx = np.array(
                [i for i, m in enumerate(self.metadata) if m.get("augment", "clean") == "clean"]
            )

        if keep_cases is not None:
            keep_cases = set(keep_cases)
            idx = np.array([i for i in idx if self.metadata[i]["case_id"] in keep_cases], dtype=int)
            if len(idx) == 0:
                raise ValueError(f"no samples match keep_cases in groups {groups}")

        self.n_excluded = 0
        if exclude_invalid and len(groups) == 1:
            bad = load_invalid_rows(h5_path, groups[0])
            if bad:
                keep = ~np.isin(idx, np.asarray(bad))
                self.n_excluded = int((~keep).sum())
                idx = idx[keep]
        self.indices = idx

        pad = self.y_pad_to - self.ny_raw
        assert pad >= 0
        self.pad_top = pad // 2
        self.pad_bottom = pad - self.pad_top

    def __len__(self):
        return len(self.indices)

    def _h5(self, fi):
        if self._files[fi] is None:
            self._files[fi] = h5py.File(self.paths[fi], "r")
        return self._files[fi]

    def __getitem__(self, i):
        row = int(self.indices[i])
        fi, grp, local = self._loc[row]
        x = self._h5(fi)[f"{grp}/X"][local]  # (3, 491, 200) float32
        if np.isnan(x).any():
            # a file built without epsilon (noisy_kh_dataset.h5) stores log_epsilon
            # as all-NaN; the pretrained encoders need 3 channels, so fill with a
            # constant: the Pr=1 training-set mean of the normalized channel
            x = np.where(np.isnan(x), LOG_EPS_NAN_FILL, x).astype(np.float32)
        if self.pad_top or self.pad_bottom:
            x = np.pad(x, ((0, 0), (self.pad_top, self.pad_bottom), (0, 0)), mode="edge")
        return torch.from_numpy(np.ascontiguousarray(x)), row

    def meta(self, row):
        return self.metadata[row]


def make_loader(
    cfg, split, batch_size=None, shuffle=None, limit=None, num_workers=None, keep_cases=None
):
    is_train = split == "train"
    ds = KHHolmboeDataset(
        cfg.h5_path,
        split,
        y_pad_to=cfg.y_pad_to,
        include_augmented=(cfg.include_augmented if is_train else True),
        exclude_invalid=getattr(cfg, "exclude_invalid_log_epsilon", True),
        keep_cases=keep_cases,
    )
    if limit is not None and limit < len(ds):
        ds = torch.utils.data.Subset(ds, range(limit))
    if shuffle is None:
        shuffle = is_train
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=False,
        drop_last=is_train,
    )


def load_rm_labels(csv_path):
    """RM_summary_table.csv (Type,Ri,Re,a,R,R_M) -> {case_id: R_M}.
    Also accepts tables that carry an explicit case_id column."""
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("case_id"):
                labels[row["case_id"].strip()] = float(row["R_M"])
                continue
            fam = "KH" if "KH" in row["Type"] else "Holmboe"
            ri = f"Ri{round(float(row['Ri']) * 100):03d}"
            amap = {"0.5": "a05", "1": "a10", "sym": "sym"}[row["a"].strip()]
            labels[f"{fam}:{ri}_{amap}_Re{row['Re']}"] = float(row["R_M"])
    return labels
