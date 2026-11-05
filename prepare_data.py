#!/usr/bin/env python3
"""Build the 38-case Pr=1 train/validation/test dataset directly from DNS files.
python -u prepare_data.py"""

import json
from pathlib import Path
import re
import zlib

import h5py
import numpy as np

# all relevant directories
ROOT = Path('/global/scratch/users/shuaim/diablo/simulations/Pr_1_KH_Holm_the_rest')
DERECHO_ROOT = Path('/glade/derecho/scratch/shuaim/diablo_2.0/fully_resolved_simulations/Pr_1_simulations')
DATA_ROOTS = (ROOT, DERECHO_ROOT)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_FILE = OUTPUT_DIR / 'pr1_38case_dataset.h5'
REFERENCE_GYF_FILE = ROOT / 'KH/KH_test/Ri020_a05_Re1000/mean.h5'

# all params
NY_INTERP, Y_CROP, NT_NEW = 501, 5, 200
NY_NEW = NY_INTERP - 2 * Y_CROP #491
N_X_PROFILES, N_Z_PROFILES = 200, 100
VARIABLES = ['buoyancy', 'reduced_shear', 'log_epsilon']
MOVIE_VARIABLES = ['u', 'w', 'th', 'epsilon']
PLANES = ('xy', 'zy')
SPLITS = ('train', 'val', 'test')
SYMMETRIC_MAXABS_VARIABLES = {'reduced_shear', 'log_epsilon'} # divide by the local max in snapshot
IDENTITY_VARIABLES = {'buoyancy'}
EPSILON_FLOOR, URMS_PEAK_T_MIN = 1e-12, 30.0
TIME_DEDUPE_DECIMALS, RANDOM_SEED = 3, 13 #data aug seed
DOWNSAMPLE_FACTORS, NOISE_SIGMAS = [2, 4, 8], [0.01, 0.05, 0.1]
AUG_PER_CLEAN = 0.3 / 0.7

# 30 train / 4 val / 4 test
TRAIN_CASES = {
    'KH:Ri012_a05_Re1000', 'KH:Ri012_a10_Re1000', 'KH:Ri012_sym_Re1000',
    'KH:Rib016_a05_Re1000', 'KH:Rib016_a10_Re1000', 'KH:Ri016_a05_Re2000',
    'KH:Ri016_sym_Re1000', 'KH:Ri016_sym_Re2000',
    'KH:Ri020_a05_Re2000', 'KH:Ri020_a10_Re1000',
    'KH:Ri020_sym_Re1000', 'KH:Ri020_sym_Re2000',
    'KH:Ri030_a05_Re2000', 'KH:Ri030_a10_Re1000', 'KH:Ri030_a10_Re2000',
    'Holmboe:Rib012_a05_Re1000', 'Holmboe:Rib012_sym_Re1000',
    'Holmboe:Rib016_a05_Re1000', 'Holmboe:Rib016_a05_Re2000', 'Holmboe:Rib016_a10_Re1000',
    'Holmboe:Rib016_sym_Re1000', 'Holmboe:Rib016_sym_Re2000',
    'Holmboe:Rib020_a05_Re1000', 'Holmboe:Rib020_a10_Re1000', 'Holmboe:Rib020_a10_Re2000',
    'Holmboe:Rib020_sym_Re1000', 'Holmboe:Rib020_sym_Re2000',
    'Holmboe:Rib030_a05_Re2000', 'Holmboe:Rib030_a10_Re1000', 'Holmboe:Rib030_a10_Re2000',
}
VAL_CASES = {
    'KH:Ri020_a10_Re2000', 'Holmboe:Rib012_a10_Re1000',
    'KH:Ri016_a10_Re2000', 'Holmboe:Rib016_a10_Re2000',
}
TEST_CASES = {
    'KH:Ri020_a05_Re1000', 'Holmboe:Rib020_a05_Re2000',
    'KH:Ri030_a05_Re1000', 'Holmboe:Rib030_a05_Re1000',
}


def number_keys(group):
    # sort the digit in string, for future time frame extraction
    return sorted([k for k in group.keys() if k.isdigit()], key=int)


def uniform_indices(n_total, n_take):
    # uniformly pick up the indices - for x and z sampling
    return np.round(np.linspace(0, n_total - 1, n_take)).astype(int)


def first_local_peak(signal, times):
    # find the first local peak after t>30
    for i in range(1, len(signal) - 1):
        if times[i] < URMS_PEAK_T_MIN:
            continue
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            return i
    return int(np.argmax(signal))


def movie_group(var_name, plane):
    # concatenate the var name with the plane
    if var_name == "th":
        return f"th1_{plane}"
    return f"{var_name}_{plane}"


def read_target_y():
    # read the gyf
    with h5py.File(REFERENCE_GYF_FILE, "r") as f:
        return f["/gyf/0001"][:]


def interp_y(profiles, old_y, new_y):
    if len(old_y) == len(new_y) and np.allclose(old_y, new_y):
        return profiles.astype(np.float32)
    out = np.zeros((profiles.shape[0], len(new_y)), dtype=np.float32)
    for i in range(profiles.shape[0]):
        out[i] = np.interp(new_y, old_y, profiles[i])
    return out


def crop_y(values):
    if Y_CROP == 0:
        return values
    return values[..., Y_CROP:-Y_CROP]


def normalize_y_time_images(data):
    # normalize the input
    # data [samples:300, var:3, height:491, time:200]
    out = data.copy()
    for sample_i in range(out.shape[0]):
        for var_i in range(out.shape[1]):
            image = out[sample_i, var_i]
            if VARIABLES[var_i] in IDENTITY_VARIABLES:
                continue
            if VARIABLES[var_i] in SYMMETRIC_MAXABS_VARIABLES:
                vabs = float(np.max(np.abs(image)))
                if np.isclose(vabs, 0.0): # just in case to divide a really small number 
                    out[sample_i, var_i] = 0.0
                else:
                    out[sample_i, var_i] = image / vabs
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def gradient_y(snapshot, y):
    return np.gradient(snapshot.astype(np.float32), y.astype(np.float32), axis=0)


def derived_snapshots(snapshots, y):
    for name in VARIABLES:
        snapshots[name] = {}
    for plane in PLANES:
        u, w, b, epsilon = [snapshots[name][plane] for name in MOVIE_VARIABLES]
        #shear = gradient_y(u, y) + gradient_y(w, y)
        shear_squared = gradient_y(u, y)**2 + gradient_y(w, y)**2
        n2 = gradient_y(b, y)
        snapshots['buoyancy'][plane] = b.astype(np.float32)
        snapshots['reduced_shear'][plane] = (shear_squared - 4.0 * n2).astype(np.float32)
        snapshots['log_epsilon'][plane] = np.log(np.maximum(epsilon, EPSILON_FLOOR)).astype(np.float32)
    return snapshots


def discover_cases():
    #generate the summary table : ('KH', 'Ri020_a05_Re1000')
    cases = {} # for metadata
    names = {case.replace(':Rib', ':Ri'): case for case in TRAIN_CASES | VAL_CASES | TEST_CASES}
    pattern = re.compile(r'^Rib?\d+_(a\d+|sym)_Re\d+$')
    for root in DATA_ROOTS:
        found = {}
        for family in ('KH', 'Holmboe'):
            for mean_file in sorted(root.glob(f'**/{family}/**/mean.h5')):
                run = mean_file.parent
                folder = run.parent if run.name.isdigit() else run
                if not pattern.fullmatch(folder.name) or not (run / 'movie.h5').exists():
                    continue
                case = names.get(f'{family}:{folder.name}'.replace(':Rib', ':Ri'))
                if case is not None and case not in cases:
                    found.setdefault(case, []).append(run)
        for case, runs in found.items():
            # when the same case show up twice, first use the first one in DATA_ROOTS
            cases[case] = sorted(runs, key=lambda p: (p.name.isdigit(), p.name))
    return [(case.split(':')[0], case.split(':')[1], runs) for case, runs in cases.items()]


def get_frames(run_dirs):
    # the starting time for each data chunk
    runs = []
    for folder in run_dirs:
        with h5py.File(folder / 'mean.h5', 'r') as f:
            keys = number_keys(f['time'])
            if keys:
                start_time = float(f[f'time/{keys[0]}'][0])
                runs.append((start_time, folder))

    # concatenate based on time
    frames_by_time = {}
    for start_time, folder in sorted(runs, key=lambda run: run[0]):
        with h5py.File(folder / 'mean.h5', 'r') as f:
            for key in number_keys(f['time']):
                time = float(f[f'time/{key}'][0])
                y = crop_y(f[f'gyf/{key}'][:])
                urms = crop_y(f[f'urms/{key}'][:])
                integral = np.trapezoid(urms, y)
                frame = dict(time=time, key=key,
                             mean_file=folder / 'mean.h5', movie_file=folder / 'movie.h5')
                frames_by_time[round(time, TIME_DEDUPE_DECIMALS)] = (frame, integral)

    ordered = [frames_by_time[t] for t in sorted(frames_by_time)]
    frames = [frame for frame, integral in ordered]
    integrals = np.array([integral for frame, integral in ordered])
    return frames, integrals # time is the info for each frame, and integrals is ∫ u_rms at this time 


def build_case(family, case_name, run_dirs, target_y):
    case_id = f'{family}:{case_name}' #'KH:Ri020_a05_Re1000'
    frames, urms = get_frames(run_dirs)
    times = np.array([frame['time'] for frame in frames])
    peak = first_local_peak(urms, times)
    start = min(peak, len(frames) - NT_NEW) 
    # if there is no more 200 frames after peak, we will shift the start 
    frames = frames[start:start + NT_NEW]
    with h5py.File(frames[0]['movie_file'], 'r') as f:
        x_idx = uniform_indices(f[f"u_xy/{frames[0]['key']}"].shape[1], N_X_PROFILES)
        z_idx = uniform_indices(f[f"u_zy/{frames[0]['key']}"].shape[1], N_Z_PROFILES)
    n_profiles = len(x_idx) + len(z_idx)
    data = np.zeros((n_profiles, len(VARIABLES), NY_NEW, NT_NEW), dtype=np.float32)
    for t, frame in enumerate(frames):
        key = frame['key']
        with h5py.File(frame['mean_file'], 'r') as f:
            old_y = f[f'gyf/{key}'][:]
        with h5py.File(frame['movie_file'], 'r') as f:
            snapshots = {}
            for name in MOVIE_VARIABLES:
                snapshots[name] = {}
                for plane in PLANES:
                    group_name = movie_group(name, plane)
                    snapshots[name][plane] = f[f'{group_name}/{key}'][:]
                    if not np.isfinite(snapshots[name][plane]).all(): # Nan or Inf
                        return None, None
        snapshots = derived_snapshots(snapshots, old_y)
        for v, name in enumerate(VARIABLES):
            xy, zy = snapshots[name]['xy'], snapshots[name]['zy']
            profiles = np.concatenate([xy[:, x_idx].T, zy[:, z_idx].T], axis=0)
            data[:, v, :, t] = crop_y(interp_y(profiles, old_y, target_y))

    scales = {name: np.abs(data[:, VARIABLES.index(name)]).reshape(n_profiles, -1).max(axis=1)
              for name in SYMMETRIC_MAXABS_VARIABLES} # store the maximum value for each snapshot
    metadata = [] # create and store the info in metadata
    panel_i = 0
    for plane, indices in [('xy', x_idx), ('zy', z_idx)]:
        for index in indices:
            entry = {
                'case_id': case_id,
                'family': family,
                'case': case_name,
                'start_frame_index': int(start),
                'peak_frame_index': int(peak),
                't_start': float(times[start]),
                't_end': float(times[start + NT_NEW - 1]),
                'augment': 'clean',
                'plane': plane,
                'axis_index': int(index),
            }
            for name in scales:
                entry[f'norm_scale_{name}'] = float(scales[name][panel_i])
            metadata.append(entry)
            panel_i += 1

    data = normalize_y_time_images(data)
    return data, metadata


def downsample_image(image, factor):
    ny, nt = image.shape
    y_full = np.arange(ny, dtype=np.float64)
    t_full = np.arange(nt, dtype=np.float64)
    y_c = y_full[::factor]
    t_c = t_full[::factor]
    coarse = image[::factor, ::factor].astype(np.float64)

    up_y = np.empty((ny, coarse.shape[1]), dtype=np.float64)
    for j in range(coarse.shape[1]):
        up_y[:, j] = np.interp(y_full, y_c, coarse[:, j])
    out = np.empty((ny, nt), dtype=np.float64)
    for i in range(ny):
        out[i] = np.interp(t_full, t_c, up_y[i])
    return out.astype(np.float32)


def augment_sample(sample, spec, rng):
    out = sample.copy()
    if spec["augment"] == "downsample":
        for v_i in range(out.shape[0]):
            out[v_i] = downsample_image(out[v_i], spec["augment_factor"])
    else:  # noise 
        out = out + rng.normal(0.0, spec["augment_sigma"], size=out.shape).astype(np.float32)
    out = np.clip(out, -1.0, 1.0)
    le = VARIABLES.index("log_epsilon")
    out[le] = np.minimum(out[le], 0.0) #log_epsilon cannot be greater than 0
    return out.astype(np.float32)


def draw_aug_specs(n_aug, rng):
    #[
    #{"augment": "noise", "augment_sigma": 0.05},
    #{"augment": "downsample", "augment_factor": 4} ...
    variants = [{"augment": "downsample", "augment_factor": f} for f in DOWNSAMPLE_FACTORS] + [
        {"augment": "noise", "augment_sigma": s} for s in NOISE_SIGMAS
    ]
    picks = rng.integers(0, len(variants), size=n_aug)
    return [dict(variants[p]) for p in picks]


class SplitWriter:
    # create hdf5 file
    def __init__(self, h5file, split):
        shape = (len(VARIABLES), NY_NEW, NT_NEW)
        self.ds = h5file.create_dataset(f'{split}/X', shape=(0, *shape), maxshape=(None, *shape),
                                       dtype='float32', chunks=(1, *shape))
        self.meta = []

    def append(self, x, metadata):
        start = len(self.ds)
        self.ds.resize(start + len(x), axis=0)
        self.ds[start:] = x
        self.meta.extend(metadata)


def write_metadata(h5file, writers):
    for split, writer in writers.items():
        metadata = writer.meta
        h5file.create_dataset(f'{split}/metadata_json', dtype=h5py.string_dtype('utf-8'),
                              data=[json.dumps(m, sort_keys=True) for m in metadata])
        cases = sorted({m['case_id'] for m in metadata})
        print(f'{split}: X {writer.ds.shape}, {len(cases)} cases')


def write_build_info(f, target_y):
    f.attrs["variables"] = json.dumps(VARIABLES)
    f.attrs["sample_shape"] = f"({len(VARIABLES)}, {NY_NEW}, {NT_NEW})"
    f.attrs["profiles_per_case"] = f"{N_X_PROFILES} xy + {N_Z_PROFILES} zy"
    f.attrs["peak_signal"] = "urms_int = trapz(cropped gyf, cropped urms, 1)"
    f.attrs["time_window"] = (
        f"{NT_NEW} frames starting at first local urms_int peak with "
        f"t >= {URMS_PEAK_T_MIN} (IC-transient guard); start clamped so the "
        "window fits when the peak is late (see start/peak_frame_index)"
    )
    f.attrs["y_crop_each_end"] = Y_CROP
    f.attrs["source_roots"] = json.dumps([str(root) for root in DATA_ROOTS])
    f.attrs["split_scheme"] = json.dumps(
        {
            "kind": "case-level 30/4/4",
            "train_cases": sorted(TRAIN_CASES),
            "val_cases": sorted(VAL_CASES),
            "test_cases": sorted(TEST_CASES),
        }
    )

    norm = f.create_group("normalization")
    norm.attrs["method"] = (
        "per sample-variable y-time image: symmetric max-abs (x / max|x|, "
        "zero preserved, divisor stored in metadata as norm_scale_*) for "
        + ", ".join(sorted(SYMMETRIC_MAXABS_VARIABLES))
        + "; identity + clip for "
        + ", ".join(sorted(IDENTITY_VARIABLES))
    )
    norm.attrs["epsilon_transform"] = (
        f"log_epsilon = ln(max(epsilon, {EPSILON_FLOOR})), then / max|ln| "
        "per sample (all-negative image -> [-1, X], X < 0)"
    )
    norm.attrs["derived_variables"] = (
        "reduced_shear = (du/dy)^2 + (dw/dy)^2 - 4*dth/dy"
    )

    aug = f.create_group("augmentation")
    aug.attrs["ratio"] = "augmented : clean = approximately 0.3 : 0.7 for all training cases; val/test clean"
    aug.attrs["downsample_factors"] = json.dumps(DOWNSAMPLE_FACTORS)
    aug.attrs["noise_sigmas"] = json.dumps(NOISE_SIGMAS)
    aug.attrs["downsample_semantics"] = (
        "strided subsample along y and t by factor, then linear interp back "
        f"to ({NY_NEW}, {NT_NEW}); applied after normalization"
    )
    aug.attrs["noise_semantics"] = (
        "additive Gaussian, sigma in normalized units, all channels, "
        "clipped to [-1, 1]; applied after normalization"
    )
    aug.attrs["masking"] = "2D masking is NOT baked in; apply on the fly at training time"

    f.create_dataset("y", data=target_y.astype(np.float32))
    f.create_dataset("frame_index", data=np.arange(NT_NEW).astype(np.int32))


def main():
    cases = discover_cases()
    print(f'Found {len(cases)} cases')
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    target_y = read_target_y()
    with h5py.File(OUT_FILE, 'w') as f:
        write_build_info(f, crop_y(target_y))
        writers = {split: SplitWriter(f, split) for split in SPLITS}
        n_clean = n_augmented = 0
        for family, name, runs in cases:
            case = f'{family}:{name}'
            print(f'\nReading {case}')
            data, metadata = build_case(family, name, runs, target_y)
            if case in TEST_CASES:
                split = 'test'
            elif case in VAL_CASES:
                split = 'val'
            else:
                split = 'train'
            writers[split].append(data, metadata)
            if split == 'train':
                n_clean += len(data)
                n_aug = int(round(len(data) * AUG_PER_CLEAN))
                # 保留原实验两批数据各自的随机种子命名。
                seed_case = f'derechoPr1_{case}' if runs[0].is_relative_to(DERECHO_ROOT) else case
                rng = np.random.default_rng([RANDOM_SEED, zlib.crc32(seed_case.encode())])
                rows = rng.choice(len(data), size=n_aug, replace=False)
                specs = draw_aug_specs(n_aug, rng)
                augmented = np.empty((n_aug, *data.shape[1:]), dtype=np.float32)
                aug_meta = []
                for i, (row, spec) in enumerate(zip(rows, specs)):
                    augmented[i] = augment_sample(data[row], spec, rng)
                    aug_meta.append(dict(metadata[row], **spec))
                writers[split].append(augmented, aug_meta)
                n_augmented += n_aug
                print(f'  -> train: +{len(data)} clean +{n_aug} aug', flush=True)
            else:
                print(f'  -> {split}: +{len(data)}', flush=True)
            f.flush()
        write_metadata(f, writers)
        fraction = n_augmented / max(n_clean + n_augmented, 1)
        print(f'\ntrain clean={n_clean}, aug={n_augmented} (overall aug fraction {fraction:.1%})')
        f.flush()
    print(f'Saved {OUT_FILE}\nDone.')


if __name__ == "__main__":
    main()
