#!/usr/bin/env python3
"""
Lightweight Domain-Adapted Regressor (FiLM conditioning)

A minimal domain adaptation approach: shared MLP trunk (identical to
baseline_virchow2_mlp.py) with per-domain FiLM layers (Feature-wise
Linear Modulation) that apply learned scale + shift after each hidden
layer.

Architecture:
  - Input: Virchow2 CLS embeddings (1280-d)
  - Shared MLP trunk:
      FC1: 1280 → fc1_dim  + shared BN + GELU + Dropout
      FiLM-1: per-domain scale + shift on fc1_dim  (~1K params/domain)
      FC2: fc1_dim → head_dim + shared BN + GELU + Dropout
      FiLM-2: per-domain scale + shift on head_dim (~0.5K params/domain)
  - Shared regression head → 2 outputs: [log1p_n_m1, log1p_n_m2]

Motivation:
  The full domain-specific model (domain_specific_frozen_codex_7.py) uses
  separate 2-layer MLP branches per domain (~500K extra params), which
  overfits when specimen count is low (e.g. 3 slides / 2 domains).

  FiLM adds only ~3K params per domain — enough to learn a stain-specific
  affine correction without enough capacity to memorize slide-specific
  artifacts. The shared trunk sees ALL training data every batch, so BN
  statistics are stable and gradients are smooth.

Key differences from baseline_virchow2_mlp.py:
  - FiLM layers after each hidden layer (domain_idx required in forward)
  - Training loop passes domain_idx to model (no per-domain batch splitting)
  - Dataset returns domain index (same as MLP baseline)

Key differences from domain_specific_frozen_codex_7.py:
  - Single shared trunk (no ModuleList of branches)
  - Shared BatchNorm (not DomainSpecificBatchNorm1d)
  - ~3K domain params vs ~500K per domain
  - Full batch forward pass (no domain-splitting in training loop)
  - No CODEX fusion (can be added back if needed)

Targets: log1p-transformed macrophage counts per tile.
  - Predictions recovered via expm1(output).clip(0)
  - Early stopping on val Pearson r (mean of m1 and m2 channels)

Author: Dave G.
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import pearsonr
import copy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FiLM layer — Feature-wise Linear Modulation
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Per-domain affine transform: x_out = gamma * x + beta.

    Initialized to identity (gamma=1, beta=0) so the model starts
    equivalent to the shared MLP baseline, and domain-specific
    corrections are learned gradually.

    Args:
        feature_dim: Dimensionality of the feature vector to modulate.
        num_domains: Number of discrete domains.
    """

    def __init__(self, feature_dim: int, num_domains: int):
        super().__init__()
        # gamma (scale) — initialized to 1
        self.gamma = nn.Parameter(torch.ones(num_domains, feature_dim))
        # beta (shift) — initialized to 0
        self.beta = nn.Parameter(torch.zeros(num_domains, feature_dim))

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """Apply per-sample domain-specific scale and shift.

        Args:
            x: [B, D] feature tensor
            domain_ids: [B] long tensor of domain indices
        Returns:
            [B, D] modulated features
        """
        g = self.gamma[domain_ids]  # [B, D]
        b = self.beta[domain_ids]   # [B, D]
        return g * x + b


# ---------------------------------------------------------------------------
# Domain-adapted regressor
# ---------------------------------------------------------------------------
class DomainFiLMRegressor(nn.Module):
    """
    Shared 2-layer MLP + FiLM domain conditioning.

    Architecture:
        embedding (1280-d)
            → FC1 (1280 → fc1_dim) → BN → GELU → Dropout
            → FiLM-1 (per-domain scale+shift on fc1_dim)
            → FC2 (fc1_dim → head_dim) → BN → GELU → Dropout
            → FiLM-2 (per-domain scale+shift on head_dim)
            → Linear → 2 outputs
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        fc1_dim: int = 512,
        head_dim: int = 256,
        num_domains: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.output_dim = 2
        self.num_domains = num_domains

        # Shared trunk — layer 1
        self.fc1 = nn.Linear(embedding_dim, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim)
        self.drop1 = nn.Dropout(dropout)
        self.film1 = FiLM(fc1_dim, num_domains)

        # Shared trunk — layer 2
        self.fc2 = nn.Linear(fc1_dim, head_dim)
        self.bn2 = nn.BatchNorm1d(head_dim)
        self.drop2 = nn.Dropout(dropout)
        self.film2 = FiLM(head_dim, num_domains)

        # Regression head
        self.regressor = nn.Linear(head_dim, 2)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.regressor]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # FiLM is already identity-initialized in __init__

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, 1280] precomputed Virchow2 CLS tokens
            domain_ids: [B] long tensor of domain indices
        Returns:
            [B, 2] regression outputs: [log1p_n_m1, log1p_n_m2]
        """
        # Layer 1: shared projection + domain modulation
        x = self.fc1(embeddings)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.drop1(x)
        x = self.film1(x, domain_ids)

        # Layer 2: shared head + domain modulation
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.drop2(x)
        x = self.film2(x, domain_ids)

        return self.regressor(x)  # [B, 2]


# ---------------------------------------------------------------------------
# Dataset (same as MLP baseline — returns domain for FiLM conditioning)
# ---------------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    """Loads precomputed embeddings for training (no CODEX features).

    Expects:
        - df: DataFrame with n_m1, n_m2, domain, slide_id columns
        - embeddings: dict mapping path_col value → embedding tensor
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, torch.Tensor],
        path_col: str = "tile_path",
        target1: str = "n_m1",
        target2: str = "n_m2",
    ):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.path_col = path_col
        self.target1 = target1
        self.target2 = target2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = self.embeddings[row[self.path_col]]
        targets = torch.tensor(
            [np.log1p(float(row[self.target1])), np.log1p(float(row[self.target2]))],
            dtype=torch.float32,
        )
        domain = int(row["domain"])
        return emb, targets, domain


def collate_fn(batch):
    embs, targets, domains = zip(*batch)
    embs = torch.stack(embs)
    targets = torch.stack(targets)      # [B, 2] float
    domains = torch.tensor(domains, dtype=torch.long)
    return embs, targets, domains


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def get_sampler(df: pd.DataFrame, path_col: str = "tile_path",
                target1: str = "n_m1", target2: str = "n_m2") -> WeightedRandomSampler:
    """Oversample tiles with any target-positive cells (3x weight) to counteract sparse majority.

    Balances domain skew within target-positive tiles.
    """
    has_mac = ((df[target1] + df[target2]) > 0).astype(float)
    mac_weight = has_mac * 3.0 + (1 - has_mac) * 1.0

    # Further balance by domain within mac-positive
    domain_counts = df.groupby("domain")["domain"].transform("count")
    domain_weight = 1.0 / domain_counts.values.astype(float)

    sample_weights = mac_weight.values * domain_weight
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------
def load_embeddings(embeddings_path: str) -> Dict[str, torch.Tensor]:
    """Load precomputed embeddings from a .pt file or directory of .pt files."""
    embeddings_path = Path(embeddings_path)

    if embeddings_path.is_file() and embeddings_path.suffix == ".pt":
        logger.info(f"Loading embeddings from {embeddings_path}")
        embs = torch.load(embeddings_path, map_location="cpu")
        logger.info(f"Loaded {len(embs)} embeddings")
        return embs

    elif embeddings_path.is_dir():
        logger.info(f"Loading embeddings from directory {embeddings_path}")
        embs = {}
        pt_files = sorted(embeddings_path.glob("**/*.pt"))
        for pt_file in pt_files:
            data = torch.load(pt_file, map_location="cpu")
            if isinstance(data, dict):
                embs.update(data)
            else:
                embs[pt_file.stem] = data
        logger.info(f"Loaded {len(embs)} embeddings from {len(pt_files)} files")
        return embs

    else:
        raise ValueError(
            f"embeddings_path must be a .pt file or directory, got: {embeddings_path}"
        )


# ---------------------------------------------------------------------------
# Embedding extraction (Phase 1) — identical to MLP baseline
# ---------------------------------------------------------------------------
def extract_embeddings(
    data_csv: str,
    output_path: str,
    backbone_name: str = "hf-hub:paige-ai/Virchow2",
    path_col: str = "tile_path",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
):
    """Extract Virchow2 CLS embeddings from pre-augmented tile PNGs."""
    import timm
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading backbone: {backbone_name} on {device}")

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone = backbone.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class TileDataset(Dataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.transform = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            path = self.paths[idx]
            img = Image.open(path).convert("RGB")
            return self.transform(img), path

    df = pd.read_csv(data_csv)
    tile_paths = df[path_col].tolist()
    logger.info(f"Extracting embeddings for {len(tile_paths)} tiles")

    output_path = Path(output_path)
    existing_embs = {}
    if output_path.exists():
        existing_embs = torch.load(output_path, map_location="cpu")
        remaining = [p for p in tile_paths if p not in existing_embs]
        logger.info(
            f"Found {len(existing_embs)} existing embeddings, "
            f"{len(remaining)} remaining"
        )
        tile_paths = remaining

    if not tile_paths:
        logger.info("All embeddings already extracted!")
        return

    dataset = TileDataset(tile_paths, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    embeddings = dict(existing_embs)
    n_extracted = 0

    for batch_imgs, batch_paths in loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.no_grad(), autocast():
            output = backbone(batch_imgs)
            if isinstance(output, tuple):
                cls_tokens = output[0]
            elif output.dim() == 3:
                cls_tokens = output[:, 0, :]
            else:
                cls_tokens = output

        for path, emb in zip(batch_paths, cls_tokens.cpu()):
            embeddings[path] = emb

        n_extracted += len(batch_paths)
        if n_extracted % (batch_size * 20) == 0:
            logger.info(f"  Extracted {n_extracted}/{len(tile_paths)} embeddings")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    logger.info(
        f"Saved {len(embeddings)} embeddings to {output_path} "
        f"(dim={next(iter(embeddings.values())).shape[-1]})"
    )


# ---------------------------------------------------------------------------
# Train one epoch — full batch, no domain splitting
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: DomainFiLMRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    all_preds_m1, all_preds_m2 = [], []
    all_true_m1,  all_true_m2  = [], []

    for batch_idx, (embs, targets, domains) in enumerate(loader):
        embs    = embs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Full batch forward — FiLM handles per-sample domain conditioning
        with autocast():
            preds = model(embs, domain_ids=domains)
            loss  = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * len(targets)

        preds_np   = preds.detach().cpu().float().numpy()
        targets_np = targets.cpu().numpy()
        all_preds_m1.extend(np.expm1(preds_np[:, 0]).clip(0))
        all_preds_m2.extend(np.expm1(preds_np[:, 1]).clip(0))
        all_true_m1.extend(np.expm1(targets_np[:, 0]))
        all_true_m2.extend(np.expm1(targets_np[:, 1]))

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    r_m1 = pearsonr(all_true_m1, all_preds_m1)[0] if np.std(all_preds_m1) > 0 else 0.0
    r_m2 = pearsonr(all_true_m2, all_preds_m2)[0] if np.std(all_preds_m2) > 0 else 0.0
    return {
        "loss": epoch_loss,
        "pearson_t1": float(r_m1),
        "pearson_t2": float(r_m2),
        "pearson_mean": float((r_m1 + r_m2) / 2),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: DomainFiLMRegressor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()

    running_loss = 0.0
    all_preds, all_targets, all_domains = [], [], []

    for embs, targets, domains in loader:
        embs    = embs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        domains_dev = domains.to(device, non_blocking=True)

        with autocast():
            preds = model(embs, domain_ids=domains_dev)

        loss = criterion(preds, targets)
        running_loss += loss.item() * len(targets)

        all_preds.append(preds.cpu().float().numpy())
        all_targets.append(targets.cpu().numpy())
        all_domains.extend(domains.numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_domains = np.array(all_domains)

    pred_m1 = np.expm1(all_preds[:,   0]).clip(0)
    pred_m2 = np.expm1(all_preds[:,   1]).clip(0)
    true_m1 = np.expm1(all_targets[:, 0])
    true_m2 = np.expm1(all_targets[:, 1])

    epoch_loss = running_loss / len(loader.dataset)

    def safe_pearson(a, b):
        return float(pearsonr(a, b)[0]) if np.std(b) > 0 and np.std(a) > 0 else 0.0

    r_m1  = safe_pearson(true_m1, pred_m1)
    r_m2  = safe_pearson(true_m2, pred_m2)
    r_mean = (r_m1 + r_m2) / 2

    domain_metrics = {}
    for d in np.unique(all_domains):
        mask = all_domains == d
        domain_metrics[int(d)] = {
            "pearson_t1": safe_pearson(true_m1[mask], pred_m1[mask]),
            "pearson_t2": safe_pearson(true_m2[mask], pred_m2[mask]),
            "n": int(mask.sum()),
        }

    return {
        "loss":           epoch_loss,
        "pearson_t1":     r_m1,
        "pearson_t2":     r_m2,
        "pearson_mean":   r_mean,
        "domain_metrics": domain_metrics,
        "sample_true_t1": true_m1,
        "sample_true_t2": true_m2,
        "sample_pred_t1": pred_m1,
        "sample_pred_t2": pred_m2,
        "sample_domains": all_domains,
    }


# ---------------------------------------------------------------------------
# Plotting & prediction output
# ---------------------------------------------------------------------------
def save_loss_curves(
    epoch_history: List[Dict],
    output_dir: str,
    fold: int,
    target1: str = "n_m1",
    target2: str = "n_m2",
) -> None:
    """Save train/val loss CSV and loss/Pearson r curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_df = pd.DataFrame(epoch_history)
    csv_path = Path(output_dir) / f"fold{fold}_loss_history.csv"
    history_df.to_csv(csv_path, index=False)
    logger.info(f"  Loss history → {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"Fold {fold+1} — Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_r_t1"], label=f"Train {target1}", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["train_r_t2"], label=f"Train {target2}", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Pearson r")
    axes[1].set_title(f"Fold {fold+1} — Train Pearson r"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history_df["epoch"], history_df["val_r_t1"],   label=f"Val {target1}",  linewidth=2, color="tab:blue")
    axes[2].plot(history_df["epoch"], history_df["val_r_t2"],   label=f"Val {target2}",  linewidth=2, color="tab:orange", linestyle="--")
    axes[2].plot(history_df["epoch"], history_df["val_r_mean"], label="Val Mean",linewidth=2, color="tab:green")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Pearson r")
    axes[2].set_title(f"Fold {fold+1} — Val Pearson r"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"fold{fold}_loss_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Loss curves → {plot_path}")


def save_val_predictions(
    val_df: pd.DataFrame,
    val_metrics: Dict,
    output_dir: str,
    fold: int,
    path_col: str = "tile_path",
    domain_names: Optional[Dict[int, str]] = None,
    target1: str = "n_m1",
    target2: str = "n_m2",
) -> None:
    """Save per-sample validation predictions CSV."""
    pred_df = pd.DataFrame({
        path_col:                  val_df[path_col].values,
        "slide_id":                val_df["slide_id"].values,
        "domain":                  val_metrics["sample_domains"],
        "fold":                    fold,
        f"true_{target1}":         val_metrics["sample_true_t1"],
        f"true_{target2}":         val_metrics["sample_true_t2"],
        f"pred_{target1}":         val_metrics["sample_pred_t1"],
        f"pred_{target2}":         val_metrics["sample_pred_t2"],
    })
    total = pred_df[f"pred_{target1}"] + pred_df[f"pred_{target2}"]
    pred_df["pred_t1_ratio"] = np.where(total > 0, pred_df[f"pred_{target1}"] / total, np.nan)
    true_total = pred_df[f"true_{target1}"] + pred_df[f"true_{target2}"]
    pred_df["true_t1_ratio"] = np.where(true_total > 0, pred_df[f"true_{target1}"] / true_total, np.nan)

    if domain_names:
        pred_df["domain_name"] = pred_df["domain"].map(domain_names)

    csv_path = Path(output_dir) / f"fold{fold}_val_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"  Val predictions ({len(pred_df)} samples) → {csv_path}")


def plot_scatter_by_domain(
    val_metrics: Dict,
    output_dir: str,
    fold: int,
    domain_names: Optional[Dict[int, str]] = None,
    target1: str = "n_m1",
    target2: str = "n_m2",
) -> None:
    """Scatter plots: true vs predicted target1 and target2, coloured by domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_t1 = val_metrics["sample_true_t1"]
    true_t2 = val_metrics["sample_true_t2"]
    pred_t1 = val_metrics["sample_pred_t1"]
    pred_t2 = val_metrics["sample_pred_t2"]
    domains = val_metrics["sample_domains"]

    domain_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    unique_domains = sorted(np.unique(domains))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for true_vals, pred_vals, ax, title, pkey in [
        (true_t1, pred_t1, axes[0], target1, "pearson_t1"),
        (true_t2, pred_t2, axes[1], target2, "pearson_t2"),
    ]:
        for di, d in enumerate(unique_domains):
            mask = domains == d
            d_name = domain_names.get(d, f"domain {d}") if domain_names else f"domain {d}"
            r = pearsonr(true_vals[mask], pred_vals[mask])[0] if np.std(pred_vals[mask]) > 0 else 0
            ax.scatter(true_vals[mask], pred_vals[mask], s=8, alpha=0.4,
                       color=domain_colors[di % len(domain_colors)],
                       label=f"{d_name} (r={r:.3f}, n={mask.sum()})")

        r_all = pearsonr(true_vals, pred_vals)[0] if np.std(pred_vals) > 0 else 0
        lim = max(true_vals.max(), pred_vals.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=1)
        ax.set_xlabel(f"True {title}"); ax.set_ylabel(f"Predicted {title}")
        ax.set_title(f"Fold {fold+1} — {title}  (overall r={r_all:.3f})")
        ax.legend(fontsize=8, markerscale=3); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"fold{fold}_scatter_by_domain.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Scatter → {plot_path}")


def plot_combined_scatter(
    all_fold_dfs: List[pd.DataFrame],
    output_dir: str,
    domain_names: Optional[Dict[int, str]] = None,
    target1: str = "n_m1",
    target2: str = "n_m2",
) -> None:
    """Pooled cross-validation scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = pd.concat(all_fold_dfs, ignore_index=True)
    domain_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col_true, col_pred, ax, title in [
        (f"true_{target1}", f"pred_{target1}", axes[0], target1),
        (f"true_{target2}", f"pred_{target2}", axes[1], target2),
    ]:
        for di, d in enumerate(sorted(combined["domain"].unique())):
            sub = combined[combined["domain"] == d]
            d_name = domain_names.get(d, f"domain {d}") if domain_names else f"domain {d}"
            r = pearsonr(sub[col_true], sub[col_pred])[0] if sub[col_pred].std() > 0 else 0
            ax.scatter(sub[col_true], sub[col_pred], s=8, alpha=0.4,
                       color=domain_colors[di % len(domain_colors)],
                       label=f"{d_name} (r={r:.3f}, n={len(sub)})")

        r_all = pearsonr(combined[col_true], combined[col_pred])[0]
        lim = max(combined[col_true].max(), combined[col_pred].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=1)
        ax.set_xlabel(f"True {title}"); ax.set_ylabel(f"Predicted {title}")
        ax.set_title(f"Pooled CV — {title}  (overall r={r_all:.3f})")
        ax.legend(fontsize=8, markerscale=3); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / "cv_pooled_scatter_by_domain.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Pooled scatter → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Domain-Adapted Regressor (FiLM conditioning)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---- Phase 1: Extract embeddings ----
    extract_parser = subparsers.add_parser(
        "extract", help="Extract Virchow2 embeddings from pre-augmented tiles"
    )
    extract_parser.add_argument("--data_csv", type=str, required=True)
    extract_parser.add_argument("--output", type=str, required=True)
    extract_parser.add_argument("--path_col", type=str, default="tile_path")
    extract_parser.add_argument("--backbone", type=str, default="hf-hub:paige-ai/Virchow2")
    extract_parser.add_argument("--batch_size", type=int, default=64)
    extract_parser.add_argument("--num_workers", type=int, default=4)
    extract_parser.add_argument("--image_size", type=int, default=224)

    # ---- Phase 2: Train ----
    train_parser = subparsers.add_parser(
        "train", help="Train FiLM-adapted regressor on precomputed embeddings"
    )

    # Data
    train_parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV with tile_path, n_m1, n_m2, domain, slide_id columns")
    train_parser.add_argument("--embeddings", type=str, required=True)
    train_parser.add_argument("--path_col", type=str, default="tile_path")
    train_parser.add_argument("--num_domains", type=int, default=2)
    train_parser.add_argument("--target1", type=str, default="n_m1",
                        help="Column name for first regression target (default: n_m1)")
    train_parser.add_argument("--target2", type=str, default="n_m2",
                        help="Column name for second regression target (default: n_m2)")

    # Architecture
    train_parser.add_argument("--embedding_dim", type=int, default=1280)
    train_parser.add_argument("--fc1_dim", type=int, default=512)
    train_parser.add_argument("--head_dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.4)

    # Optional domain labels for plots
    train_parser.add_argument("--domain_names_json", type=str, default=None,
                        help='JSON dict mapping domain_idx→name, e.g. \'{"0":"Post-CODEX","1":"Standard"}\'')

    # Training
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--n_folds", type=int, default=5)
    train_parser.add_argument("--split_col", type=str, default=None,
                        help="Column for pre-defined train/test split. Train='train', Val='test'.")
    train_parser.add_argument("--num_workers", type=int, default=4)

    # Output
    train_parser.add_argument("--output_dir", type=str, default="results")
    train_parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    # ---- Dispatch ----
    if args.mode == "extract":
        extract_embeddings(
            data_csv=args.data_csv,
            output_path=args.output,
            backbone_name=args.backbone,
            path_col=args.path_col,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        return

    # ---- Phase 2: Train ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    domain_names = None
    if args.domain_names_json:
        raw = json.loads(args.domain_names_json)
        domain_names = {int(k): str(v) for k, v in raw.items()}
        logger.info(f"Domain names: {domain_names}")

    df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(df)} samples from {args.data_csv}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Domain distribution:\n{df['domain'].value_counts().to_string()}")

    embeddings = load_embeddings(args.embeddings)

    missing = [p for p in df[args.path_col] if p not in embeddings]
    if missing:
        logger.warning(
            f"{len(missing)} samples missing embeddings (first 5: {missing[:5]}). "
            f"Dropping them."
        )
        df = df[df[args.path_col].isin(embeddings)].reset_index(drop=True)
        logger.info(f"Remaining samples: {len(df)}")

    sample_emb = next(iter(embeddings.values()))
    actual_dim = sample_emb.shape[-1]
    if actual_dim != args.embedding_dim:
        logger.warning(
            f"Embedding dim mismatch: expected {args.embedding_dim}, "
            f"got {actual_dim}. Overriding."
        )
        args.embedding_dim = actual_dim

    os.makedirs(args.output_dir, exist_ok=True)
    fold_results = []

    if args.split_col and args.split_col in df.columns:
        logger.info(f"Using pre-defined split from column '{args.split_col}'")
        loso_test_df = df[df[args.split_col] == "loso_test"].reset_index(drop=True)
        train_df = df[df[args.split_col] == "train"].reset_index(drop=True)
        val_df   = df[df[args.split_col] == "test"].reset_index(drop=True)
        fold_splits = [(train_df, val_df, 0)]
        n_folds_actual = 1
    else:
        loso_test_df = pd.DataFrame()
        from sklearn.model_selection import GroupKFold
        logger.info(f"Using {args.n_folds}-fold GroupKFold on slide_id")
        gkf = GroupKFold(n_splits=args.n_folds)
        fold_splits = []
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df, groups=df["slide_id"])
        ):
            fold_splits.append((df.iloc[train_idx], df.iloc[val_idx], fold))
        n_folds_actual = args.n_folds

    for train_df, val_df, fold in fold_splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold + 1}/{n_folds_actual}")
        logger.info(f"{'='*60}")

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
        logger.info(f"Train mac+ tiles: {((train_df[args.target1]+train_df[args.target2])>0).sum()}")
        logger.info(f"Val   mac+ tiles: {((val_df[args.target1]+val_df[args.target2])>0).sum()}")
        logger.info(f"Train domains: {train_df['domain'].value_counts().sort_index().to_dict()}")
        logger.info(f"Val domains:   {val_df['domain'].value_counts().sort_index().to_dict()}")

        model = DomainFiLMRegressor(
            embedding_dim=args.embedding_dim,
            fc1_dim=args.fc1_dim,
            head_dim=args.head_dim,
            num_domains=args.num_domains,
            dropout=args.dropout,
        ).to(device)

        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        film_params = sum(p.numel() for name, p in model.named_parameters() if "film" in name)
        logger.info(f"Total params: {total_params:,} | Trainable: {trainable:,} | FiLM params: {film_params:,}")

        criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        scaler = GradScaler()

        train_ds = EmbeddingDataset(train_df, embeddings, path_col=args.path_col,
                                    target1=args.target1, target2=args.target2)
        val_ds   = EmbeddingDataset(val_df,   embeddings, path_col=args.path_col,
                                    target1=args.target1, target2=args.target2)

        sampler = get_sampler(train_df, path_col=args.path_col,
                              target1=args.target1, target2=args.target2)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
                                  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

        # Training loop — early stop on val mean Pearson r
        best_val_r       = -1.0
        best_epoch       = 0
        patience_counter = 0
        best_val_metrics = {}
        best_state_dict  = None
        epoch_history    = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
            val_metrics   = validate(model, val_loader, criterion, device)
            scheduler.step()

            epoch_history.append({
                "epoch":      epoch,
                "train_loss": train_metrics["loss"],
                "train_r_t1": train_metrics["pearson_t1"],
                "train_r_t2": train_metrics["pearson_t2"],
                "val_loss":   val_metrics["loss"],
                "val_r_t1":   val_metrics["pearson_t1"],
                "val_r_t2":   val_metrics["pearson_t2"],
                "val_r_mean": val_metrics["pearson_mean"],
                "lr":         optimizer.param_groups[0]["lr"],
            })

            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"r_{args.target1}: {train_metrics['pearson_t1']:.3f} r_{args.target2}: {train_metrics['pearson_t2']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"r_{args.target1}: {val_metrics['pearson_t1']:.3f} r_{args.target2}: {val_metrics['pearson_t2']:.3f} "
                f"r_mean: {val_metrics['pearson_mean']:.3f}"
            )

            if val_metrics["pearson_mean"] > best_val_r:
                best_val_r = val_metrics["pearson_mean"]
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                if args.save_model:
                    save_path = Path(args.output_dir) / f"fold{fold}_best.pt"
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"  Saved best model → {save_path}")
                best_val_metrics = val_metrics
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # --- Save fold outputs ---
        save_loss_curves(epoch_history, args.output_dir, fold,
                         target1=args.target1, target2=args.target2)
        save_val_predictions(val_df, best_val_metrics, args.output_dir, fold,
                             path_col=args.path_col, domain_names=domain_names,
                             target1=args.target1, target2=args.target2)
        plot_scatter_by_domain(best_val_metrics, args.output_dir, fold,
                               domain_names=domain_names,
                               target1=args.target1, target2=args.target2)

        logger.info(
            f"Fold {fold+1} best: Epoch {best_epoch} | "
            f"r_{args.target1}: {best_val_metrics['pearson_t1']:.3f} | "
            f"r_{args.target2}: {best_val_metrics['pearson_t2']:.3f} | "
            f"r_mean: {best_val_r:.3f}"
        )
        fold_results.append({
            "fold":         fold + 1,
            "best_epoch":   best_epoch,
            "pearson_t1":   best_val_metrics["pearson_t1"],
            "pearson_t2":   best_val_metrics["pearson_t2"],
            "pearson_mean": best_val_r,
            "loss":         best_val_metrics["loss"],
        })

        # Save fold report
        report_path = Path(args.output_dir) / f"fold{fold}_report.json"
        with open(report_path, "w") as f:
            json.dump({
                k: v for k, v in best_val_metrics.items()
                if k not in ("sample_true_t1", "sample_true_t2",
                             "sample_pred_t1", "sample_pred_t2", "sample_domains")
            }, f, indent=2, default=str)

    # --- Cross-fold combined outputs ---
    all_fold_preds = []
    for fold in range(n_folds_actual):
        pred_path = Path(args.output_dir) / f"fold{fold}_val_predictions.csv"
        if pred_path.exists():
            all_fold_preds.append(pd.read_csv(pred_path))

    if all_fold_preds:
        combined_preds = pd.concat(all_fold_preds, ignore_index=True)
        combined_path  = Path(args.output_dir) / "cv_all_val_predictions.csv"
        combined_preds.to_csv(combined_path, index=False)
        logger.info(f"Combined val predictions ({len(combined_preds)} samples) → {combined_path}")
        plot_combined_scatter(all_fold_preds, args.output_dir, domain_names=domain_names,
                              target1=args.target1, target2=args.target2)

    # --- LOSO test evaluation (held-out specimen) ---
    loso_test_results = {}
    if len(loso_test_df) > 0 and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        loso_test_ds = EmbeddingDataset(loso_test_df, embeddings, path_col=args.path_col,
                                        target1=args.target1, target2=args.target2)
        loso_test_loader = DataLoader(loso_test_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        loso_metrics = validate(model, loso_test_loader, criterion, device)
        loso_pred_df = pd.DataFrame({
            args.path_col:             loso_test_df[args.path_col].values,
            "patient_id":              loso_test_df["patient_id"].values,
            "domain":                  loso_metrics["sample_domains"],
            f"true_{args.target1}":    loso_metrics["sample_true_t1"],
            f"true_{args.target2}":    loso_metrics["sample_true_t2"],
            f"pred_{args.target1}":    loso_metrics["sample_pred_t1"],
            f"pred_{args.target2}":    loso_metrics["sample_pred_t2"],
        })
        loso_pred_df.to_csv(Path(args.output_dir) / "loso_test_predictions.csv", index=False)
        loso_test_results = {
            "pearson_t1":   loso_metrics["pearson_t1"],
            "pearson_t2":   loso_metrics["pearson_t2"],
            "pearson_mean": loso_metrics["pearson_mean"],
            "n_samples":    len(loso_test_df),
        }
        logger.info(
            f"LOSO test | r_{args.target1}: {loso_test_results['pearson_t1']:.3f} | "
            f"r_{args.target2}: {loso_test_results['pearson_t2']:.3f} | "
            f"r_mean: {loso_test_results['pearson_mean']:.3f} | "
            f"n={loso_test_results['n_samples']}"
        )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    r_m1s   = [r["pearson_t1"]   for r in fold_results]
    r_m2s   = [r["pearson_t2"]   for r in fold_results]
    r_means = [r["pearson_mean"] for r in fold_results]
    logger.info(f"Pearson r {args.target1}:   {np.mean(r_m1s):.4f} ± {np.std(r_m1s):.4f}")
    logger.info(f"Pearson r {args.target2}:   {np.mean(r_m2s):.4f} ± {np.std(r_m2s):.4f}")
    logger.info(f"Pearson r Mean: {np.mean(r_means):.4f} ± {np.std(r_means):.4f}")

    summary_path = Path(args.output_dir) / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "fold_results":      fold_results,
            "mean_pearson_t1":   float(np.mean(r_m1s)),
            "std_pearson_t1":    float(np.std(r_m1s)),
            "mean_pearson_t2":   float(np.mean(r_m2s)),
            "std_pearson_t2":    float(np.std(r_m2s)),
            "mean_pearson_mean": float(np.mean(r_means)),
            "std_pearson_mean":  float(np.std(r_means)),
            "loso_test":         loso_test_results,
            "args": vars(args),
        }, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
