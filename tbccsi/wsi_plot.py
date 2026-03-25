
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from .wsi_tiler import WSITiler

matplotlib.use('Agg')

class WSIPlotter:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, sample=None, slide_path=None, output_path=None):
        self.sample = sample
        self.slide = slide_path
        self.output_path = output_path
        if self.slide:
            self.tiler = WSITiler(sample, slide_path, output_path)
        else:
            self.tiler=None

    def create_heatmap(self, predictions_df, file_name, point_size=3, prob_col="prob_struct_0"):
        """
        Creates and saves a heatmap visualization of predictions overlaid on the slide.

        Args:
            slide: An image object of the whole-slide image.
            predictions_df (pd.DataFrame): DataFrame from the predict_tiles method.
            output_path (str): The path to save the output heatmap image.
            point_size (int): The size of the points (squares) in the heatmap.
            prob_col (str): The name of the column in predictions_df to use for coloring.
        """
        print("Generating heatmap...")
        # Get a reasonably sized thumbnail for visualization
        slide_dims = self.tiler.get_shape(0)
        max_dim_ratio = max(slide_dims) / 2000.0
        thumb_size = ( int(slide_dims[0]/max_dim_ratio), int(slide_dims[1]/max_dim_ratio))
        thumbnail = self.tiler.get_thumbnail(thumb_size)

        output_path = self.output_path / file_name

        print("Working with thumbnail size: " + str(thumb_size))
        print("  scaling factor: " + str(max_dim_ratio))

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

        # 1. Original thumbnail
        axes[0].imshow(thumbnail)
        axes[0].set_title(f'H&E {self.sample}')
        axes[0].axis('off')

        if not predictions_df.empty:
            # Calculate scaling factors to map tile coordinates to thumbnail coordinates

            # Scale coordinates
            x_scaled = (predictions_df['x'] / max_dim_ratio)
            y_scaled = (predictions_df['y'] / max_dim_ratio)

            # 2. Prediction heatmap (colored by probability of class 1)
            scatter1 = axes[1].scatter(
                x_scaled, y_scaled,
                c=predictions_df[prob_col],
                cmap='RdYlBu',  # Red-Yellow-Blue colormap is good for probabilities
                marker='s',  # Use squares to mimic tiles
                s=point_size,
                alpha=0.7
            )
            # axes[1].imshow(thumbnail, alpha=0.3)  # Overlay on a faint thumbnail

            # Set the aspect ratio to match the thumbnail dimensions
            axes[1].set_aspect('equal', adjustable='box')
            axes[1].set_xlim(0, thumb_size[0])
            axes[1].set_ylim(thumb_size[1], 0)  # Invert y-axis for image coordinates
            axes[1].set_title(f'Prediction Heatmap {self.sample}')
            axes[1].axis('off')
            plt.colorbar(scatter1, ax=axes[1], fraction=0.046, pad=0.04)


            # 3. Confidence heatmap
            #scatter2 = axes[2].scatter(
            #    x_scaled, y_scaled,
            #    c=predictions_df['confidence'],
            #    cmap='viridis',  # Viridis is good for showing magnitude
            #    marker='s',
            #    s=point_size,
            #    alpha=0.7
            #)
            # axes[2].imshow(thumbnail, alpha=0.3)

            #axes[2].set_aspect('equal', adjustable='box')
            #axes[2].set_xlim(0, thumb_size[0])
            #axes[2].set_ylim(thumb_size[1], 0)
            #axes[2].set_title(f'Confidence Heatmap {self.sample}')
            #axes[2].axis('off')
            #plt.colorbar(scatter2, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            # Handle case with no predictions
            for i in range(1, 3):
                axes[i].text(0.5, 0.5, 'No predictions to display.',
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[i].transAxes)
                axes[i].axis('off')

        # Finalize and save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")


    def tile_mean_heatmap(self, tile_file, output_path=None,
                          tile_size=224, show_grid=False, figsize=(8, 6)):
        """
        Creates and saves a heatmap visualization of average tile colors from the common tiling.

        Args:
            tile_file (str): The common tiling file.
            output_path (str): The path to save the output heatmap image.
            point_size (int): The size of the points (squares) in the heatmap.
        """
        print("Generating mean color heatmap...")
        # Get a reasonably sized thumbnail for visualization
        # Load data if it's a file path
        if isinstance(tile_file, str):
            # Skip comment lines that start with #
            df = pd.read_csv(tile_file, comment='#')
        elif isinstance(tile_file, Path):
            df = pd.read_csv(tile_file, comment='#')
        else:
            df = tile_file.copy()

        # Auto-detect tile size if not provided
        if tile_size is None:
            # Find the minimum distance between adjacent tiles
            x_diffs = np.diff(np.sort(df['x'].unique()))
            y_diffs = np.diff(np.sort(df['y'].unique()))
            x_tile_size = x_diffs[x_diffs > 0].min() if len(x_diffs[x_diffs > 0]) > 0 else 224
            y_tile_size = y_diffs[y_diffs > 0].min() if len(y_diffs[y_diffs > 0]) > 0 else 224
            tile_size = (x_tile_size, y_tile_size)
            print(f"Auto-detected tile size: {tile_size}")
        elif isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)

        # Normalize RGB values to 0-1 range (assuming they're 0-255)
        df['norm_red'] = df['mean_red'] / 255.0
        df['norm_green'] = df['mean_green'] / 255.0
        df['norm_blue'] = df['mean_blue'] / 255.0

        # Clamp values to valid range
        df['norm_red'] = df['norm_red'].clip(0, 1)
        df['norm_green'] = df['norm_green'].clip(0, 1)
        df['norm_blue'] = df['norm_blue'].clip(0, 1)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Find the bounds of the image
        min_x, max_x = df['x'].min(), df['x'].max()
        min_y, max_y = df['y'].min(), df['y'].max()

        print(f"Image bounds: X({min_x}-{max_x}), Y({min_y}-{max_y})")
        print(f"Number of tiles: {len(df)}")

        # Plot each tile as a colored rectangle
        for _, row in df.iterrows():
            x, y = row['x'], row['y']
            color = (row['norm_red'], row['norm_green'], row['norm_blue'])

            # Create rectangle patch
            rect = Rectangle((x, y), tile_size[0], tile_size[1],
                             facecolor=color, edgecolor='black' if show_grid else 'none',
                             linewidth=0.5 if show_grid else 0)
            ax.add_patch(rect)

        # Set axis limits
        ax.set_xlim(min_x - tile_size[0] * 0.1, max_x + tile_size[0] * 1.1)
        ax.set_ylim(min_y - tile_size[1] * 0.1, max_y + tile_size[1] * 1.1)

        # Invert y-axis to match image coordinates (origin at top-left)
        ax.invert_yaxis()
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Tile Mean Color Visualization ({len(df)} tiles)')
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save the figure
        if output_path:
            plt.savefig(output_path)

        return fig, ax, df

    def tile_qc_heatmap(self, tile_file, output_path=None,
                        tile_size=224, show_grid=False, figsize=(16, 6),
                        lap_var_clip_pct=(1, 99), fft_hfe_clip_pct=(1, 99)):
        """
        Two-panel QC plot from a tile coordinate CSV.

        Panel 1: Mean RGB tile color (same as tile_mean_heatmap).
        Panel 2: Bivariate blur score map — tissue tiles shown in grey,
                 modulated by lap_var (blue) and fft_hfe (red). Background is white.

        Args:
            tile_file: Path, str, or DataFrame of the tile CSV.
            output_path: Where to save the PNG. If None, figure is returned but not saved.
            tile_size: Tile size in pixels, or None to auto-detect.
            show_grid: Draw black tile borders if True.
            figsize: Overall figure size.
            lap_var_clip_pct: (lo, hi) percentile clip for lap_var normalization.
            fft_hfe_clip_pct: (lo, hi) percentile clip for fft_hfe normalization.

        Returns:
            (fig, [ax1, ax2], df)
        """
        # ── Load data ────────────────────────────────────────────────────────
        if isinstance(tile_file, (str, Path)):
            df = pd.read_csv(tile_file, comment='#')
        else:
            df = tile_file.copy()

        required = {'x', 'y', 'mean_red', 'mean_green', 'mean_blue', 'lap_var', 'fft_hfe'}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"tile_file is missing required column(s): {sorted(missing)}. "
                f"Expected schema: x, y, tile_id, mean_red, mean_green, mean_blue, lap_var, fft_hfe"
            )

        # ── Auto-detect tile size ─────────────────────────────────────────────
        if tile_size is None:
            x_diffs = np.diff(np.sort(df['x'].unique()))
            y_diffs = np.diff(np.sort(df['y'].unique()))
            tw = x_diffs[x_diffs > 0].min() if len(x_diffs[x_diffs > 0]) > 0 else 224
            th = y_diffs[y_diffs > 0].min() if len(y_diffs[y_diffs > 0]) > 0 else 224
            tile_size = (tw, th)
        elif isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)

        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig,
                      width_ratios=[1, 1, 0.06], wspace=0.08, hspace=0.1)
        ax1  = fig.add_subplot(gs[:, 0])
        ax2  = fig.add_subplot(gs[:, 1])
        cax1 = fig.add_subplot(gs[0, 2])
        cax2 = fig.add_subplot(gs[1, 2])

        # ── Empty CSV guard ───────────────────────────────────────────────────
        if df.empty:
            for ax in (ax1, ax2, cax1, cax2):
                ax.text(0.5, 0.5, 'No tissue tiles found.' if ax in (ax1, ax2) else '',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            return fig, [ax1, ax2, cax1, cax2], df

        min_x, max_x = df['x'].min(), df['x'].max()
        min_y, max_y = df['y'].min(), df['y'].max()
        print(f"Image bounds: X({min_x}-{max_x}), Y({min_y}-{max_y})")
        print(f"Number of tiles: {len(df)}")

        # ── Panel 1: mean RGB ─────────────────────────────────────────────────
        df['_nr'] = (df['mean_red']   / 255.0).clip(0, 1)
        df['_ng'] = (df['mean_green'] / 255.0).clip(0, 1)
        df['_nb'] = (df['mean_blue']  / 255.0).clip(0, 1)

        for _, row in df.iterrows():
            ax1.add_patch(Rectangle(
                (row['x'], row['y']), tile_size[0], tile_size[1],
                facecolor=(row['_nr'], row['_ng'], row['_nb']),
                edgecolor='black' if show_grid else 'none',
                linewidth=0.5 if show_grid else 0
            ))

        ax1.set_xlim(min_x - tile_size[0] * 0.1, max_x + tile_size[0] * 1.1)
        ax1.set_ylim(min_y - tile_size[1] * 0.1, max_y + tile_size[1] * 1.1)
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Mean tile color  ({len(df)} tiles)')

        # ── Normalize lap_var (log1p + percentile clip) ───────────────────────
        log_lv = np.log1p(df['lap_var'].values.astype(float))
        lo, hi = np.percentile(log_lv, list(lap_var_clip_pct))
        if hi - lo < 1e-6:
            print("WARNING: lap_var has near-zero variance; showing all tiles at mid-intensity blue.")
            lv_norm = np.full(len(df), 0.5)
        else:
            lv_norm = ((log_lv.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)

        # ── Normalize fft_hfe (percentile clip) ───────────────────────────────
        fft_vals = df['fft_hfe'].values.astype(float)
        lo2, hi2 = np.percentile(fft_vals, list(fft_hfe_clip_pct))
        if hi2 - lo2 < 1e-6:
            print("WARNING: fft_hfe has near-zero variance; showing all tiles at mid-intensity red.")
            fft_norm = np.full(len(df), 0.5)
        else:
            fft_norm = ((fft_vals.clip(lo2, hi2) - lo2) / (hi2 - lo2)).clip(0, 1)

        # ── Panel 2: bivariate blur encoding ─────────────────────────────────
        # Grey base (0.6) + red channel driven by fft_hfe + blue by lap_var
        for i, (_, row) in enumerate(df.iterrows()):
            R = 0.6 + 0.4 * fft_norm[i]
            G = 0.6
            B = 0.6 + 0.4 * lv_norm[i]
            ax2.add_patch(Rectangle(
                (row['x'], row['y']), tile_size[0], tile_size[1],
                facecolor=(R, G, B),
                edgecolor='black' if show_grid else 'none',
                linewidth=0.5 if show_grid else 0
            ))

        ax2.set_xlim(min_x - tile_size[0] * 0.1, max_x + tile_size[0] * 1.1)
        ax2.set_ylim(min_y - tile_size[1] * 0.1, max_y + tile_size[1] * 1.1)
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Blur QC  |  blue = lap_var (log)   red = fft_hfe')

        # ── Colorbars for panel 2 ─────────────────────────────────────────────
        import matplotlib.colors as mcolors
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap

        cmap_blue = LinearSegmentedColormap.from_list('grey_blue', ['#999999', '#0033cc'])
        cmap_red  = LinearSegmentedColormap.from_list('grey_red',  ['#999999', '#cc0000'])

        cb1 = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_blue),
                           cax=cax1)
        cb1.set_label('lap_var  (log scale)', fontsize=8)
        cb1.set_ticks([0, 0.5, 1])
        cb1.set_ticklabels(['low', 'mid', 'high'], fontsize=7)

        cb2 = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_red),
                           cax=cax2)
        cb2.set_label('fft_hfe', fontsize=8)
        cb2.set_ticks([0, 0.5, 1])
        cb2.set_ticklabels(['low', 'mid', 'high'], fontsize=7)

        # ── Legend: grey = tissue, white = background ─────────────────────────
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='#999999', edgecolor='black', label='Tissue tile'),
            Patch(facecolor='white',   edgecolor='black', label='Background'),
        ]
        ax2.legend(handles=legend_handles, loc='lower left', fontsize=7,
                   framealpha=0.85, borderpad=0.6)

        # ── Cleanup temp columns ──────────────────────────────────────────────
        df.drop(columns=['_nr', '_ng', '_nb'], inplace=True)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"QC plot saved to {output_path}")

        return fig, [ax1, ax2], df

    def plot_prediction_grid(self,
                             prediction_file_path,
                             tile_source_dir,
                             prediction_column,
                             output_path,
                             grid_size=(8, 4),
                             tile_size=224):
        """
        Creates a grid of tiles with the highest and lowest prediction scores.

        The top half of the grid shows the highest-scoring tiles, and the
        bottom half shows the lowest-scoring tiles.

        Args:
            prediction_file_path (str or Path): Path to the merged prediction CSV.
            tile_source_dir (str or Path): Directory containing the tile images.
            prediction_column (str): The column in the CSV to sort by (e.g., 'conf_cancer').
            output_path (str or Path): Path to save the output grid image.
            grid_size (tuple): A (columns, rows) tuple for each half of the grid.
            tile_size (int): The pixel size for each tile in the grid.
        """
        print(f"Generating prediction grid for column: '{prediction_column}'")

        # 1. Load and sort the prediction data
        try:
            df = pd.read_csv(prediction_file_path)
            if prediction_column not in df.columns:
                raise ValueError(f"Prediction column '{prediction_column}' not found in CSV.")
        except FileNotFoundError:
            print(f"Error: Prediction file not found at {prediction_file_path}")
            return

        num_tiles_per_half = grid_size[0] * grid_size[1]

        # Get the highest and lowest probability tiles
        high_prob_tiles = df.sort_values(by=prediction_column, ascending=False).head(num_tiles_per_half)
        low_prob_tiles = df.sort_values(by=prediction_column, ascending=True).head(num_tiles_per_half)

        # 2. Create the canvas for the final grid image
        grid_cols, grid_rows_half = grid_size
        canvas_width = grid_cols * tile_size
        canvas_height = (2 * grid_rows_half) * tile_size
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # 3. Paste tiles onto the canvas
        self._paste_tiles_to_grid(canvas, high_prob_tiles, tile_source_dir, grid_size, tile_size, 0)
        self._paste_tiles_to_grid(canvas, low_prob_tiles, tile_source_dir, grid_size, tile_size, canvas_height // 2)

        # Optional: Add labels to distinguish the two halves
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except IOError:
            font = ImageFont.load_default()  # Fallback font

        draw.text((10, 10), f"Highest '{prediction_column}' Tiles", fill="black", font=font)
        draw.text((10, 10 + canvas_height // 2), f"Lowest '{prediction_column}' Tiles", fill="black", font=font)

        # 4. Save the final image
        canvas.save(output_path)
        print(f"✅ Prediction grid saved to: {output_path}")

    def _paste_tiles_to_grid(self, canvas, tile_df, tile_source_dir, grid_size, tile_size, y_offset):
        """Helper method to paste a set of tiles onto the canvas."""
        grid_cols, grid_rows = grid_size
        tile_source_path = Path(tile_source_dir)

        for index, tile_info in enumerate(tile_df.itertuples()):
            row = index // grid_cols
            col = index % grid_cols

            # Reconstruct the file path based on your saving logic
            collection_index = tile_info.tile_id // 1000
            collection_dir = tile_source_path / f'collection_{collection_index}'
            # Assuming you saved H&E tiles with a name like this
            if 'segmented' in str(tile_source_path):
                file_name = f"masked_{tile_info.tile_id:06d}_x{tile_info.x_coord}_y{tile_info.y_coord}.png"
            else:
                file_name = f"tile_{tile_info.tile_id:06d}_x{tile_info.x_coord}_y{tile_info.y_coord}.png"
            image_path = collection_dir / file_name

            print(image_path)

            try:
                tile_img = Image.open(image_path).resize((tile_size, tile_size))
            except FileNotFoundError:
                tile_img = self._create_placeholder_tile(tile_size, f"Missing:\n{tile_info.tile_id}")

            paste_x = col * tile_size
            paste_y = row * tile_size + y_offset
            canvas.paste(tile_img, (paste_x, paste_y))

    def _create_placeholder_tile(self, size, text):
        """Creates a black tile with text for missing images."""
        img = Image.new('RGB', (size, size), 'black')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, size // 2 - 20), text, fill="white", font=font)
        return img