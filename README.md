# Augmentation Methods for Medical Ultrasound Data

Bachelor's thesis — *"Implementation of augmentation methods for medical ultrasound data"*

Experiments are tracked on [Weights & Biases](https://wandb.ai/daswoldemar/bachelor?nw=nwuserdaswoldemar).

---

## Thesis Document

The full thesis is available in the repository root:

- **`Implementation_of_augmentation_methods_for_medical_ultrasound_data.pdf`** — the complete thesis text covering literature review, methodology, experiments, and conclusions.
- **`Bachelor-LiteratureReview.docx`** — standalone literature review document.

The `Articles/` directory contains the research papers consulted during the project. Papers already read and annotated are in `Articles/Readed/Commented/`; papers queued for reading are in `Articles/ToRead/`. A running summary of relevant methods and findings from each paper is in `Articles/README.md`.

---

## Overview

This project systematically evaluates image augmentation strategies for binary classification of lung ultrasound (LUS) frames. The clinical task is detecting **B-lines** (pathological artifacts indicating fluid in the lungs) versus **A-lines** (healthy lung pattern). Because LUS datasets are small and class-imbalanced, augmentation is critical for generalisation.

The core question: *which augmentation methods — and combinations — actually help a ResNet18 classifier on this data?*

---

## Task & Dataset

| Property | Detail |
|---|---|
| Task | Binary frame-level classification (B-line vs. A-line) |
| Dataset | KKUI lung ultrasound (`kkui-lung-bline-lumify`, `kkui-lung-aline-not_old`) |
| Input | Grayscale frames extracted from ultrasound videos |
| Validation | 5-fold cross-validation (folds defined in `metadata_folds/`) |
| Evaluation split | 80% train / 20% val (within each fold), held-out test set per fold |

Label mapping (B-line experiment):
- Raw label `1` → class `0` (no B-line)
- Raw label `2` → class `1` (B-line present)

---

## Architecture & Training

**Model**: ResNet18 (ImageNet pretrained) with a replaced fully-connected head (`512 → 2`).
Optional dropout before the FC layer is configurable.

**Pipeline for each image:**

```
read grayscale frame  →  [0,255] uint8  →  [0,1] float32
    ↓
init transform  (Resize → Pad to 600×400)
    ↓
augmentation transforms  (training only, see configs/)
    ↓
output transform  (optional normalisation)
    ↓
expand 1-channel → 3-channel  (copy grayscale to RGB for ResNet)
```

**Training settings** (defaults across sweeps):

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=0.0001 |
| Loss | Cross-Entropy |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping | patience=5 on val loss |
| Test model | weights from best validation loss epoch |

**Metrics reported**: F1, Precision, Recall, Balanced Accuracy, Confusion Matrix (all logged to WandB per epoch).

---

## Repository Structure

```
Bachelor/
├── Practical/
│   ├── training.py              # main training entry point
│   ├── code_to_config.py        # converts Python transform definitions → YAML sweep configs
│   ├── check.py                 # validates that all transforms in a YAML config can be imported
│   ├── requirements.txt
│   │
│   ├── data/
│   │   └── dataset.py           # FramesDataset — reads JSON fold metadata + images
│   │
│   ├── transforms/
│   │   └── builder.py           # TransformsBuilder — instantiates transforms from string names
│   │
│   ├── custom_transforms/       # installable package (pip install -e ./custom_transforms)
│   │   └── custom/transforms/
│   │       ├── pad.py                      # TransformPad (edge-fill to target size)
│   │       ├── resize.py                   # TransformResize
│   │       ├── random_noise.py             # RandomNoise (additive Gaussian)
│   │       ├── random_noise_with_fv.py     # RandomNoiseWithFV (field-variance Gaussian)
│   │       ├── random_speckle_noise.py     # RandomSpeckleNoise (multiplicative)
│   │       ├── random_salt_and_pepper_noise.py
│   │       ├── random_brightness_by_add.py
│   │       ├── random_contrast_by_multiply.py
│   │       └── random_elastic_transform.py
│   │
│   ├── utils/
│   │   ├── build.py             # builds model, optimizer, criterion, datasets
│   │   └── trainer.py           # Trainer class — train/val/test loops, early stopping, WandB logging
│   │
│   ├── configs/                 # WandB sweep YAML configs (grid search over transforms × folds)
│   ├── configs_code/            # Python source files for transform definitions (→ YAML via code_to_config.py)
│   ├── metadata_folds/          # JSON files defining train/test splits per fold
│   ├── results/                 # exported CSVs and analysis notebooks
│   └── testing/                 # exploration notebooks
│
├── Articles/                    # collected literature
├── Experiments.md               # running experiment log
└── Bachelor-LiteratureReview.docx
```

---

## Custom Transforms

All custom transforms live in the `custom_transforms` package and follow the same interface as torchvision transforms (callable objects, work on `float32` tensors in `[0, 1]`).

| Class | Description |
|---|---|
| `TransformResize(size)` | Resize the shorter side to `size` |
| `TransformPad(size)` | Centre-pad with edge-fill to reach target `(H, W)` |
| `RandomNoise(p, mean, std)` | Additive Gaussian noise |
| `RandomNoiseWithFV(p, fv)` | Gaussian noise with randomly sampled field variance |
| `RandomSpeckleNoise(p, mean, std)` | Multiplicative noise: `img + img × N(mean, std)` |
| `RandomSaltAndPepperNoise(p, density)` | Salt-and-pepper noise with random density |
| `RandomBrightnessByAdd(p, delta)` | Brightness shift by additive constant |
| `RandomContrastByMultiply(p, multiplier)` | Contrast scaling by a random multiplier |
| `RandomElasticTransform(p, alpha, sigma)` | Elastic deformation with random strength |

All transforms use `(min, max)` ranges and sample uniformly at call time.

---

## Sweep Experiments

Experiments are run as **WandB grid sweeps** — each sweep exhaustively evaluates every `(transform, fold)` combination so that per-fold and averaged results are available.

| Config file | Experiment group |
|---|---|
| `sweep_paper_11_gpu.yaml` | 11 methods from the reference paper (geometric + noise) |
| `sweep_pixel-level_single.yaml` | ~60 single pixel-level transforms (noise, contrast, brightness, blur, sharpness) |
| `sweep_elastic_single_gpu.yaml` | Elastic deformation parameter sweep |
| `sweep_grid_distortion_single.yaml` | Grid distortion variants |
| `sweep_erasing_single.yaml` | Random Erasing / Cutout variants |
| `sweep_speckle_0.5.yaml` | Speckle noise variants |
| `sweep_combined_gpu.yaml` | Combined multi-transform pipelines (torchvision + MONAI) |
| `sweep_combined_fold_normalize.yaml` | Combined pipelines with ImageNet-style normalisation |

Transforms in sweep configs can be specified as strings of Python expressions — they are evaluated with `torchvision.transforms` aliased as `T`, `monai.transforms` as `MT`, and `custom.transforms` as `CT`.

---

## Running an Experiment

### 1. Install dependencies

```bash
cd Practical
pip install -r requirements.txt
pip install -e ./custom_transforms
```

### 2. Validate a sweep config

```bash
python check.py  # edit the path inside the file to point at your target config
```

### 3. Launch a WandB sweep

```bash
wandb sweep configs/sweep_combined_fold_normalize.yaml
wandb agent <sweep-id>
```

The agent calls `training.py --yaml_file=<config_path>` for each run.

### 4. (Optional) Generate a YAML config from Python transform definitions

Edit `configs_code/sweep_combined_fold_normalize.py` to define transforms as `transform_*` list variables, then:

```bash
python code_to_config.py
```

This writes the corresponding YAML to `configs/`.

---

## Key Results (Paper 11 Benchmark)

Tested on the B-line classification task, 600×400 input, no cross-validation (single split):

| Augmentation | Test F1 (mean) | Test Precision (mean) | Test Recall (mean) |
|---|---|---|---|
| No augmentation | 0.313 | 0.560 | 0.239 |
| **Shear** | **0.398** | 0.468 | 0.410 |
| Color shifting / sharpness / contrast | 0.312 | 0.472 | 0.393 |
| Salt & Pepper + Shear | 0.304 | 0.320 | 0.473 |
| Translate + Shear | 0.296 | 0.369 | 0.376 |
| Rotate | 0.283 | 0.312 | 0.408 |
| Rotate + Translate | 0.278 | 0.455 | 0.402 |
| Translate + Shear + Rotate | 0.250 | 0.352 | 0.340 |
| Gaussian Noise + Rotate | 0.197 | 0.327 | 0.274 |
| Translate | 0.193 | 0.311 | 0.287 |
| Gaussian Noise | 0.189 | 0.194 | 0.194 |
| Salt & Pepper | 0.180 | 0.177 | 0.196 |

Full results including cross-validation sweeps are in `Practical/results/`.

---

## University Requirements

1. Overview of image augmentation in deep learning, with focus on medical images.
2. Overview of existing parallel hyperparameter search tools.
3. Design augmentation methods suited for ultrasound characteristics (speckle, fan geometry, low contrast).
4. Implement selected methods and an automated parallel validation mechanism (WandB sweeps).
5. Compare augmentation strategies and evaluate statistically.
6. Prepare documentation per thesis advisor guidelines.