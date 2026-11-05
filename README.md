# ShearSSL

**Inferring Stratified Shear Instability Character via Self-supervised Representation Learning**

ShearSSL learns representations of stratified shear flows from DNS time–height panels and predicts the pseudomomentum ratio, R_M, as a continuous measure of instability character between Holmboe-like and Kelvin–Helmholtz-like limits.

The main model combines a masked autoencoder (MAE) with a ConvNeXt regression head:

1. **Self-supervised pretraining:** reconstruct masked regions of three physical input channels using Pr = 1 simulations.
2. **Source regression:** freeze the pretrained encoder and train a ConvNeXt head to predict R_M.
3. **Target adaptation:** fine-tune the regression head with the encoder frozen. An optional second stage jointly trains encoder LoRA adapters and the regression head.

This repository contains source code only. DNS files, prepared datasets, label tables, extracted features, model weights, and prediction files must be supplied separately.

## Code

| File | Purpose |
| --- | --- |
| `config.py` | MAE architecture, training settings, and default paths |
| `dataset.py` | HDF5 loading, case filtering, padding, and labels |
| `models.py` | MAE and regression architectures |
| `prepare_data.py` | Generate Pr = 1 time–height panels from raw DNS files |
| `train_mae.py` | Self-supervised MAE pretraining |
| `extract_tokens.py` | Extract spatial encoder features |
| `train_head.py` | Train the source ConvNeXt regression head |
| `finetune.py` | Zero-shot inference, head-only, LoRA, and full fine-tuning |
| `evaluate.py` | Evaluate heads using precomputed target features |
| `visualize_reconstruction.py` | Plot masked reconstructions |
| `plot_predictions.py` | Plot pooled panel-prediction distributions and seed means |

## Installation

Use Python 3.10 or later. Install a PyTorch build suitable for your hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the repository root. Training examples below assume a CUDA-capable GPU; scripts also expose device options.

## Data and paths

Place the prepared source dataset at `data/pr1_combined_38case_dataset_v2.h5`, or change `DATA_H5` in `config.py`. HDF5 files contain `train`, `val`, and `test` groups, each with `X` and `metadata_json`. Each source panel has shape `(3, 491, 200)` in channel–height–time order. Metadata identifies the parent DNS case. The loader pads the height to 496 points for the network.

The channels are buoyancy, reduced shear, and log dissipation. The MAE uses 16 × 10 patches, a 31 × 20 spatial grid per variable, and a common 50% spatial mask across variables. The `small8` encoder has eight Transformer blocks of width 384; the decoder has four blocks of width 192.

Provide source labels as a CSV with `case_id,R_M` columns (the original `RM_summary_table.csv` format is also supported). Target labels can be supplied as a JSON mapping from case IDs to R_M. Neither label file is included here.

The MAE split and regression-head split are different. The source head uses the validation and test case lists shown below; these should not be replaced with the MAE split by accident. Keep all panels from each designated held-out case outside training.

## 1. Pretrain and extract features

```bash
python train_mae.py --preset small8 --out runs/source_mae

python extract_tokens.py \
  --ckpt runs/source_mae/last.pt \
  --data data/pr1_combined_38case_dataset_v2.h5 \
  --out outputs/pr1_tokens.h5
```

Default pretraining uses 200 epochs, batch size 32, AdamW with learning rate 1e-4 and weight decay 0.05, and ten warm-up epochs followed by cosine decay. The reconstruction loss is mean squared error over masked patches only. Checkpoints include the configuration; `last.pt` is the final saved epoch and `best.pt` is selected by validation loss.

## 2. Train the source regression head

```bash
python train_head.py \
  --tokens outputs/pr1_tokens.h5 \
  --labels-csv data/RM_summary_table.csv \
  --val-cases 'Holmboe:Rib030_a05_Re1000,KH:Ri030_a05_Re1000,KH:Ri020_a10_Re2000,Holmboe:Rib012_a10_Re1000' \
  --test-cases 'Holmboe:Rib020_a05_Re2000,KH:Ri020_a05_Re1000,Holmboe:Rib016_a10_Re2000,KH:Ri016_a10_Re2000' \
  --seed 13 --batch-size 96 --lr 0.001 \
  --out runs/source_head
```

The baseline uses mean squared error and an unconstrained linear output. Optional range penalties and sigmoid outputs are available through the command-line options. The encoder stays frozen because this step trains on extracted features. The selected head is saved as `head_best.pt`.

## 3. Evaluate and adapt to target datasets

Supply the four target HDF5 datasets and their labels locally:

```bash
ENC=runs/source_mae/last.pt
HEAD=runs/source_head/head_best.pt
TARGETS='data/derecho_holmboe_Pr4_dataset1.h5,data/derecho_holmboe_Pr7_dataset2.h5,data/noisy_kh_dataset.h5,data/alexis_dataset.h5'
TEST_CASES='derechoHB_Pr4:asymHol_Ri030_a05_Re1000,derechoHB_Pr4:asymKH_Ri016_a05_Re1000,derechoPr7_Holmboe:Ri016_a10_single,derechoPr7_Holmboe:Ri030_a05_single,noisyKH:Re2000_Ri016_Pr1_A01,Alexis:asym_KH'
VAL_CASES='derechoHB_Pr4:symHol_Ri030_a00_Re1000'

python finetune.py --mode zero-shot \
  --enc-ckpt "$ENC" --head-ckpt "$HEAD" \
  --data "$TARGETS" --labels-json data/all4_rm_labels.json \
  --test-cases "$TEST_CASES" --val-cases "$VAL_CASES" \
  --seed 13 --out runs/zero_shot_s13

python finetune.py --mode head-only \
  --enc-ckpt "$ENC" --head-ckpt "$HEAD" \
  --data "$TARGETS" --labels-json data/all4_rm_labels.json \
  --test-cases "$TEST_CASES" --val-cases "$VAL_CASES" \
  --seed 13 --out runs/stage1_s13

python finetune.py --mode lora --lora-all \
  --enc-ckpt "$ENC" --init-ckpt runs/stage1_s13/best.pt \
  --data "$TARGETS" --labels-json data/all4_rm_labels.json \
  --test-cases "$TEST_CASES" --val-cases "$VAL_CASES" \
  --lora-rank 8 --lora-alpha 16 \
  --seed 13 --out runs/stage2_s13
```

Stage 1 updates only the regression head. Stage 2 adds rank-8 LoRA updates to attention projections and feed-forward linear layers throughout the encoder, keeping the original encoder weights frozen and jointly optimizing the adapters and head. Stage 2 starts from the corresponding Stage 1 checkpoint. Use separate output directories for additional seeds.

These commands describe the workflow. Exact reproduction of an individual historical run requires its dataset, checkpoint, and complete saved configuration.

## Visualization

```bash
python visualize_reconstruction.py --ckpt runs/source_mae/last.pt
python plot_predictions.py
```

Reconstruction plots are written to `outputs/reconstruction/`. The distribution plot reads `inputs/plotted_panel_predictions.csv`, with columns `condition`, `case_id`, `seed`, `prediction`, and `true_rm`. Conditions are `source`, `zero_shot`, and `head_fine_tuned`; the default seeds are 13, 14, and 15. Prepare this combined per-panel CSV from your runs before plotting. The script writes a PNG to `outputs/prediction_distributions/`.

Distributions pool panel predictions across seeds. Seed markers show each run's case-mean prediction. Variation across panels and variation across seed means are different statistics; density width is a display scale, not uncertainty in the true R_M.

## Preparing new source data

Edit `ROOT`, `DERECHO_ROOT`, and `REFERENCE_GYF_FILE` in `prepare_data.py` to point to your raw DNS directories, then run:

```bash
python prepare_data.py
```

The builder reads `mean.h5` and `movie.h5` directly and writes `outputs/pr1_38case_dataset.h5`. Set `DATA_H5` to that output to train on newly generated data. The intended split is 30 training, four validation, and four test cases. Only discovered cases are processed, so all raw directories must be available to obtain the complete split.

The current builder adds 129 augmented panels to each training case: a complete build contains 12,870 training panels and 1,200 panels in each held-out split. The historical prepared dataset contains 12,612 training panels. The current reduced-shear calculation also differs from the historical preprocessing: it uses `(du/dy)^2 + (dw/dy)^2 - 4*dth/dy`. Its use of `dth/dy` assumes the input scalar has the intended buoyancy scaling; no additional Richardson-number factor is applied. New builds are therefore not a bitwise reproduction of the historical paper input. Preserve the original prepared data when reproducing historical results.

Use `--help` on the training, feature-extraction, evaluation, and reconstruction scripts for additional options.
