# tbccsi: Tile-based classification on cell segmented images

**tbccsi** is a Python-based tool for processing Whole Slide Images (WSI) in computational pathology. It handles tiling of massive slide files (SVS, TIFF, VSI), runs inference with configurable models, and generates per-tile predictions or latent embeddings.

## Features

* **Multi-format Support:** Natively reads `.svs` (OpenSlide), `.tiff` (TiffFile), and `.vsi` (SlideIO).
* **Smart Tiling:** Automatically detects tissue regions to avoid processing empty background tiles. Each tile is scored for focus quality (Laplacian variance and FFT high-frequency energy) so blurry regions can be filtered downstream.
* **Config-driven Inference:** Ship any model as a weights file + `model_config.yaml`. No code changes needed to add new architectures.
* **Built-in Models:** Virchow2 multi-head classifier (structure + immune subtypes), DAFT domain-aware macrophage classifier, and a family of Virchow2/ResNet-based cell-pair count regressors (`mac_linear`, `mac_mlp`, `mac_film`, `mac_domain_specific`, `mac_resnet`).
* **Embedding Extraction:** Extract latent representations from any layer for downstream analysis.
* **CLI Interface:** Command-line interface built with Typer.

## Installation

### 1. System Dependencies

Install the OpenSlide C library before installing the Python bindings.

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools

# macOS (Homebrew)
brew install openslide
```

Windows users: download binaries from [openslide.org](https://openslide.org/download/).

### 2. Python Package

```bash
git clone https://github.com/IlyaLab/tbccsi.git
cd tbccsi
pip install .
```

This registers the `tbccsi` CLI command and installs all dependencies.

## Quick Start

```bash
# 1. Tile a slide (produces tiles.csv with coordinates + blur scores)
tbccsi tile --sample-id Sample_001 --input-slide slide.svs \
    --work-dir ./output --tile-file tiles.csv

# 2. Run inference
tbccsi pred --sample-id Sample_001 --input-slide slide.svs \
    --work-dir ./output --tile-file ./output/tiles.csv \
    -m ./my_model/best.pth --do-inference

# 3. Apply cell-calling thresholds
tbccsi call --sample-id Sample_001 --work-dir ./output \
    --pred-file ./output/Sample_001_virchow2_multihead_v2_preds.csv \
    --thresh-file thresholds.csv --out-file calls.csv

# (Optional) Extract backbone CLS embeddings for fast multi-model inference (no -m needed)
tbccsi embed --sample-id Sample_001 --input-slide slide.svs \
    --work-dir ./output --tile-file ./output/tiles.csv \
    --latent-type backbone_cls --save-format npz

# (Optional) Extract head-based latent embeddings
tbccsi embed --sample-id Sample_001 --input-slide slide.svs \
    --work-dir ./output --tile-file ./output/tiles.csv \
    -m ./my_model/best.pth -n virchow2_multihead_v2 \
    --latent-type backbone --save-format npz
```

## Usage

### Tiling (`tile`)

Scans a whole slide image, detects tissue regions, and generates a tile coordinate CSV. Each tissue tile is scored for focus quality using two metrics.

```bash
tbccsi tile --sample-id "Sample_001" \
    --input-slide "/path/to/slide.svs" \
    --work-dir "./output" \
    --tile-file "tiles.csv" \
    --save-tiles       # optional: write tile PNGs to disk
    --fft-cutoff 0.3   # optional: FFT high-freq cutoff fraction (default: 0.3)
    --plot             # optional: save a QC heatmap PNG
```

Adding `--plot` writes `{sample_id}_tile_qc.png` to `--work-dir`. This is a two-panel figure:

- **Left panel** — mean tile color: each tissue tile rendered as its average RGB color, giving a quick visual of tissue coverage and staining.
- **Right panel** — blur QC: tissue tiles shown in grey, with `lap_var` encoded as blue intensity and `fft_hfe` encoded as red intensity. Background (non-tissue) is white. Two colorbars label the blue and red axes independently.


**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `x`, `y` | Tile top-left coordinate at level 0 |
| `tile_id` | Sequential tile index (0-based) |
| `mean_red`, `mean_green`, `mean_blue` | Per-channel mean pixel value |
| `lap_var` | Laplacian variance — higher values indicate sharper focus |
| `fft_hfe` | FFT high-frequency energy ratio (0–1) — higher values indicate sharper focus |

`--fft-cutoff` controls the radial frequency threshold used to define "high frequency" in the FFT metric. Lower values include more of the spectrum (more sensitive); the default of `0.3` works well for 224×224 tiles at 20× magnification.

### Prediction (`pred`)

Runs model inference on tiles. The model is specified by weights (`-m`) and optionally a config (`-c` or `-n`).

```bash
# Using a config file next to the weights (auto-detected)
tbccsi pred --sample-id S1 --input-slide slide.svs \
    --work-dir ./output --tile-file tiles.csv \
    -m ./my_model/best.pth --do-inference

# Explicit config path
tbccsi pred ... -m weights.pth -c /path/to/model_config.yaml --do-inference

# By registered model name (config bundled in tbccsi/models/)
tbccsi pred ... -m weights.pth -n daft_macrophage --domain-id 1 --do-inference
```

| Flag | Description |
|------|-------------|
| `-m, --model-path` | Path to trained model weights (`.pth`, `.safetensors`) |
| `-c, --model-config` | Path to `model_config.yaml` (optional if config is next to weights or bundled in package) |
| `-n, --model-name` | Short name of a registered model (e.g. `virchow2_multihead_v2`, `virchow2_multihead_v3`, `daft_macrophage`, `mac_linear`, `mac_mlp`, `mac_film`, `mac_domain_specific`, `mac_resnet`) |
| `-b, --batch-size` | GPU batch size (default: 32) |
| `--do-inference` | Actually run inference (otherwise just tiles/plots) |
| `--do-tta` | Apply 8× dihedral test-time augmentation |
| `--domain-id` | Domain ID for domain-aware models like DAFT |
| `--do-plot` | Column name to generate a per-tile heatmap for, or `bivariate` for a two-colour M1/M2 map |
| `--pred-file` | Path to an existing predictions CSV to use for plotting (skips inference; auto-detects `{sample_id}_*_preds.csv` in `--work-dir` if omitted) |

### Embedding Extraction (`embed`)

Extract latent representations from any model layer. `--latent-type` can be specified multiple times to extract several representations in one pass.

**Backbone CLS embeddings** (for mac regressor models) do not require `-m` — the pretrained Virchow2 backbone is used directly. This is the recommended way to pre-compute embeddings once before running multiple mac regressor models, avoiding repeated backbone inference per model.

```bash
# Backbone CLS only — no -m required
tbccsi embed --sample-id S1 --input-slide slide.svs \
    --work-dir ./output --tile-file tiles.csv \
    --latent-type backbone_cls --save-format npz

# Head-based latents — -m required
tbccsi embed --sample-id S1 --input-slide slide.svs \
    --work-dir ./output --tile-file tiles.csv \
    -m model.pth -n virchow2_multihead_v2 \
    --latent-type backbone --latent-type structure \
    --save-format npz
```

| Flag | Description |
|------|-------------|
| `-m, --model-path` | Path to trained model weights (`.pth`, `.safetensors`). Not required when using `--latent-type backbone_cls`. |
| `-b, --batch-size` | GPU batch size (default: 32) |
| `--latent-type` | Name of a latent representation to extract; repeat to extract multiple. See table below. If omitted, extracts all head-based types. |
| `--save-format` | Output format: `npz` (compressed numpy, default) or `csv` (flattened dataframe) |
| `--do-tta` | Apply test-time augmentation |

**Available latent types:**

| Type | Dimensions | Model required | Description |
|------|-----------|----------------|-------------|
| `backbone_cls` | 1280 | No | Virchow2 CLS token. Input format expected by all mac regressor models (`mac_mlp`, `mac_linear`, `mac_film`, `mac_domain_specific`). |
| `backbone` | 2560 | Yes | CLS token + attention-pooled patches concatenated. Used by the Virchow2 multi-head classifier. |
| `structure` | 512 | Yes | Structure head latent (Virchow2 multi-head only) |
| `immune_shared` | 128 | Yes | Shared immune latent (Virchow2 multi-head only) |
| `immune_tcell` | 64 | Yes | T-cell specific latent (Virchow2 multi-head only) |
| `immune_mac` | 64 | Yes | Macrophage specific latent (Virchow2 multi-head only) |

### Cell Calling (`call`)

Apply threshold-based cell calling to a prediction CSV.

```bash
tbccsi call --sample-id S1 --work-dir ./output \
    --pred-file preds.csv --thresh-file thresholds.csv \
    --out-file calls.csv
```

| Flag | Description |
|------|-------------|
| `--pred-file` | Prediction CSV produced by `tbccsi pred` |
| `--thresh-file` | CSV defining per-class probability thresholds |
| `--out-file` | Output filename (written inside `--work-dir`) |

## Model Configuration

Every model is described by a `model_config.yaml` that tells the inference engine how to load it, what its outputs mean, and how to parse predictions. This means **you never need to modify inference code to add a new model**.

### Example: DAFT Macrophage Classifier

```yaml
name: "daft_macrophage_v1"
version: "1.0"
description: "DAFT M1/M2 macrophage classifier (4-class softmax)"

backbone:
  model_name: "hf-hub:paige-ai/Virchow2"
  pretrained: true
  freeze: false
  create_kwargs:
    mlp_layer: "SwiGLUPacked"
    act_layer: "SiLU"

model_class: "tbccsi.models.daft_macrophage.DAFT_Virchow2_Macrophage"
init_kwargs:
  num_classes: 4
  num_domains: 2

forward_extra_kwargs:
  domain_id: 1  # default domain at inference

output_format: "single"

heads:
  - name: "macrophage"
    num_outputs: 4
    activation: "softmax"
    outputs:
      - { name: "neither",   index: 0, output_col: "prob_neither" }
      - { name: "m1_only",   index: 1, output_col: "prob_m1_only" }
      - { name: "m2_only",   index: 2, output_col: "prob_m2_only" }
      - { name: "both_m1m2", index: 3, output_col: "prob_both_m1m2" }

tile_size: 224
normalize: "reinhard"
```

### Config Resolution Order

When you run `tbccsi pred -m weights.pth`, the engine finds the config in this order:

1. **`-c` flag** — explicit path to a YAML file.
2. **`-n` flag** — looks up a registered model name, finds the config bundled next to the model `.py` in `tbccsi/models/`.
3. **Next to weights** — checks the same directory as the `.pth` file for `model_config.yaml`.
4. **Legacy fallback** — if no config is found anywhere, uses the hardcoded Virchow2 multi-head model.

### Packaging a Model

Place the config next to the model definition in `tbccsi/models/`:

```
tbccsi/models/
├── model_virchow2_v2.py          # Virchow2MultiHeadModel class
├── model_virchow2_v2.yaml        # its config
├── daft_macrophage.py             # DAFT_Virchow2_Macrophage class
└── daft_macrophage.yaml           # its config
```

Or as a package directory:

```
tbccsi/models/daft_macrophage/
├── __init__.py                    # exports DAFT_Virchow2_Macrophage
└── model_config.yaml              # its config
```

Then register the short name in `cli.py`:

```python
MODEL_REGISTRY = {
    "virchow2_multihead_v2":  "tbccsi.models.model_virchow2_v2.Virchow2MultiHeadModel",
    "virchow2_multihead_v3":  "tbccsi.models.model_virchow2_v3.Virchow2MultiHeadModel",
    "daft_macrophage":        "tbccsi.models.daft_macrophage.DAFT_Virchow2_Macrophage",
    "mac_linear":             "tbccsi.models.mac_linear.MacRegressorLinear",
    "mac_mlp":                "tbccsi.models.mac_mlp.MacRegressorMLP",
    "mac_film":               "tbccsi.models.mac_film.MacRegressorFiLM",
    "mac_domain_specific":    "tbccsi.models.mac_domain_specific.MacRegressorDomainSpecific",
    "mac_resnet":             "tbccsi.models.mac_resnet.MacRegressorResNet",
}
```

Users can then run inference with just `-n daft_macrophage` — no config path needed.

## Supported Slide Formats

| Extension | Backend | Notes |
|-----------|---------|-------|
| `.svs` | OpenSlide | Standard Aperio format |
| `.tiff / .tif` | TiffFile | Supports OME-TIFF and flat TIFFs |
| `.vsi` | SlideIO | Olympus CellSens format (requires `slideio`) |

## Project Structure

```
tbccsi/
├── cli.py               # Typer CLI with pred, tile, embed, call commands
├── tbccsi_main.py       # Pipeline orchestration (tiling → inference → save)
├── model_inference.py   # Config-driven InferenceEngine
├── model_config.py      # ModelConfig dataclass + YAML loader
├── wsi_tiler.py         # WSITiler: slide reading + tissue detection
├── wsi_plot.py          # Heatmap generation
├── utils.py             # Shared utilities: blur metrics (laplacian_variance, fft_high_freq_energy)
└── models/
    ├── model_virchow2_v2.py      # Virchow2 multi-head v2 (structure + immune)
    ├── model_virchow2_v3.py      # Virchow2 multi-head v3
    ├── daft_macrophage_v1.py     # DAFT domain-aware macrophage classifier
    ├── mac_regressors.py         # Shared base classes for cell-pair count regressors
    ├── mac_linear.py             # MacRegressorLinear  (Virchow2 backbone)
    ├── mac_mlp.py                # MacRegressorMLP     (Virchow2 backbone)
    ├── mac_film.py               # MacRegressorFiLM    (Virchow2 + domain conditioning)
    ├── mac_domain_specific.py    # MacRegressorDomainSpecific
    └── mac_resnet.py             # MacRegressorResNet  (ResNet-50 backbone)
```

## Troubleshooting

**`OpenSlideUnsupportedFormatError`** — Ensure the file is not corrupted. For `.vsi` files, ensure `slideio` is installed (`pip install slideio`), as OpenSlide does not support VSI.

**`DllNotFoundException` (Windows)** — Add the OpenSlide `bin` directory to your system `PATH`.

**`No model_config.yaml found`** — Either place a config next to your weights file, pass `-c /path/to/config.yaml`, or use `-n model_name` if the model is registered.

**Legacy models** — If you have existing weights trained with the original Virchow2 multi-head architecture, they still work without any config file. The engine falls back to the hardcoded behavior automatically.
