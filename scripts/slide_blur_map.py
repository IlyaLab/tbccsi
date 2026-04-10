"""
slide_blur_map.py

Grid a single TIF slide into tiles, compute Laplacian variance and FFT
high-frequency energy on each tile, and save spatial heatmaps showing
blur score distributions across the slide.

Usage:
    python slide_blur_map.py --slide myslide.tif
    python slide_blur_map.py --slide myslide.tif --tile_size 512 --level 0
    python slide_blur_map.py --slide myslide.tif --tile_size 256 --min_tissue 0.5

Output:
    <slide_stem>_blur_map.png  — side-by-side heatmaps of both scores
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image
from scipy.ndimage import laplace
from tqdm import tqdm


# ── Feature extraction (shared with tile_blur_analysis.py) ───────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    return img.astype(np.float32)


def laplacian_variance(img_gray: np.ndarray) -> float:
    lap = laplace(img_gray.astype(np.float32))
    return float(np.var(lap))


def fft_high_freq_energy(img_gray: np.ndarray, cutoff_fraction: float = 0.3) -> float:
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((Y - cy) / cy) ** 2 + ((X - cx) / cx) ** 2)
    high_freq_mask = dist > cutoff_fraction
    total_energy = magnitude.sum() + 1e-10
    return float(magnitude[high_freq_mask].sum() / total_energy)


def is_background(img: np.ndarray, min_tissue: float) -> bool:
    """Return True if tile is mostly background (very bright, low saturation)."""
    gray = to_gray(img)
    # Background pixels are bright (>220) in H&E
    bg_fraction = (gray > 220).mean()
    return bg_fraction > (1.0 - min_tissue)


# ── Slide gridding ────────────────────────────────────────────────────────────

def grid_slide(slide_path: Path, tile_size: int, level: int,
               min_tissue: float, cutoff_fraction: float):
    """
    Open a TIF with PIL (or openslide if available), grid into tiles,
    compute features. Returns 2D grids of lap_var and fft_hfe, plus
    a thumbnail for background.
    """
    # Try openslide first (handles pyramidal TIFs), fall back to PIL
    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))
        level = min(level, slide.level_count - 1)
        dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        print(f"Opened with openslide — level {level}, dims {dims}, "
              f"downsample {downsample:.1f}x")

        def read_region(x, y, w, h):
            # openslide coords are always at level 0
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            region = slide.read_region((x0, y0), level, (w, h))
            return np.array(region.convert("RGB"))

        W, H = dims

    except (ImportError, Exception) as e:
        if "openslide" in str(type(e).__module__):
            print("openslide not available, falling back to PIL")
        img_full = np.array(Image.open(slide_path).convert("RGB"))
        H, W = img_full.shape[:2]
        print(f"Opened with PIL — dims ({W}, {H})")

        def read_region(x, y, w, h):
            return img_full[y:y+h, x:x+w]

    n_cols = W // tile_size
    n_rows = H // tile_size
    print(f"Grid: {n_cols} cols × {n_rows} rows = {n_cols * n_rows} tiles "
          f"at tile_size={tile_size}px")

    lap_grid = np.full((n_rows, n_cols), np.nan)
    fft_grid = np.full((n_rows, n_cols), np.nan)

    total = n_rows * n_cols
    with tqdm(total=total, desc="Tiling") as pbar:
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * tile_size
                y = row * tile_size
                tile = read_region(x, y, tile_size, tile_size)

                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pbar.update(1)
                    continue

                if is_background(tile, min_tissue):
                    pbar.update(1)
                    continue

                gray = to_gray(tile)
                lap_grid[row, col] = laplacian_variance(gray)
                fft_grid[row, col] = fft_high_freq_energy(gray, cutoff_fraction)
                pbar.update(1)

    # Thumbnail for spatial context — downsample the full image
    thumb_size = (min(400, W // 4), min(400, H // 4))
    try:
        thumb = np.array(slide.get_thumbnail(thumb_size).convert("RGB"))
    except Exception:
        pil_img = Image.open(slide_path).convert("RGB")
        pil_img.thumbnail(thumb_size)
        thumb = np.array(pil_img)

    return lap_grid, fft_grid, thumb, n_rows, n_cols


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_blur_maps(lap_grid: np.ndarray, fft_grid: np.ndarray,
                   thumb: np.ndarray, slide_path: Path,
                   out_path: Path) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f1117")

    # ── Panel 1: slide thumbnail ──
    ax = axes[0]
    ax.imshow(thumb)
    ax.set_title("Slide thumbnail", color="white", fontsize=11)
    ax.axis("off")

    # ── Panel 2: Laplacian variance heatmap ──
    ax = axes[1]
    # Log scale — variance spans orders of magnitude
    lap_log = np.where(np.isnan(lap_grid), np.nan, np.log1p(lap_grid))
    vmin, vmax = np.nanpercentile(lap_log, [2, 98])
    im1 = ax.imshow(lap_log, cmap="inferno", vmin=vmin, vmax=vmax,
                    interpolation="nearest", aspect="auto")
    ax.set_title("Laplacian Variance (log)\n↑ = sharper", color="white", fontsize=11)
    ax.set_xlabel("tile column", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("tile row", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#aaaaaa", labelsize=7)

    # ── Panel 3: FFT high-freq energy heatmap ──
    ax = axes[2]
    vmin, vmax = np.nanpercentile(fft_grid[~np.isnan(fft_grid)], [2, 98])
    im2 = ax.imshow(fft_grid, cmap="inferno", vmin=vmin, vmax=vmax,
                    interpolation="nearest", aspect="auto")
    ax.set_title("FFT High-Freq Energy\n↑ = sharper", color="white", fontsize=11)
    ax.set_xlabel("tile column", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("tile row", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#aaaaaa", labelsize=7)

    for ax in axes[1:]:
        ax.set_facecolor("#222222")  # grey for background/skipped tiles

    fig.suptitle(
        f"Blur Score Spatial Map — {slide_path.name}",
        color="white", fontsize=13, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Spatial blur map for a single TIF slide")
    p.add_argument("--slide",       type=Path, required=True,
                   help="Path to input TIF slide")
    p.add_argument("--tile_size",   type=int,  default=256,
                   help="Tile size in pixels (default: 256)")
    p.add_argument("--level",       type=int,  default=0,
                   help="Pyramid level to read (openslide only, default: 0)")
    p.add_argument("--min_tissue",  type=float, default=0.3,
                   help="Min tissue fraction to include tile (default: 0.3)")
    p.add_argument("--cutoff",      type=float, default=0.3,
                   help="FFT high-freq cutoff fraction (default: 0.3)")
    p.add_argument("--out",         type=Path, default=None,
                   help="Output PNG path (default: <slide_stem>_blur_map.png)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = args.out or args.slide.with_name(args.slide.stem + "_blur_map.png")

    lap_grid, fft_grid, thumb, n_rows, n_cols = grid_slide(
        slide_path=args.slide,
        tile_size=args.tile_size,
        level=args.level,
        min_tissue=args.min_tissue,
        cutoff_fraction=args.cutoff,
    )

    plot_blur_maps(lap_grid, fft_grid, thumb, args.slide, out_path)
