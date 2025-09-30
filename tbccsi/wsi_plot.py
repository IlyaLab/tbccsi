
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .wsi_tiler import WSITiler

class WSIPlotter:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, sample=None, slide=None, output_dir=None):
        self.sample = sample
        self.slide = slide
        self.output_dir = output_dir
        if self.slide:
            self.tiler = WSITiler(sample, slide, output_dir)
        else:
            self.tiler=None

    def pred_heatmap(self, file_name, predictions_df, point_size=3, prob_col="prob_class_1"):
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

        print("Working with thumbnail size: " + str(thumb_size))
        print("  scaling factor: " + str(max_dim_ratio))

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        # 1. Original thumbnail
        axes[0].imshow(thumbnail)
        axes[0].set_title(f'H&E {self.sample}')
        axes[0].axis('off')

        if not predictions_df.empty:
            # Calculate scaling factors to map tile coordinates to thumbnail coordinates

            # Scale coordinates
            x_scaled = (predictions_df['x_coord'] / max_dim_ratio)
            y_scaled = (predictions_df['y_coord'] / max_dim_ratio)

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
            scatter2 = axes[2].scatter(
                x_scaled, y_scaled,
                c=predictions_df['confidence'],
                cmap='viridis',  # Viridis is good for showing magnitude
                marker='s',
                s=point_size,
                alpha=0.7
            )
            # axes[2].imshow(thumbnail, alpha=0.3)

            axes[2].set_aspect('equal', adjustable='box')
            axes[2].set_xlim(0, thumb_size[0])
            axes[2].set_ylim(thumb_size[1], 0)
            axes[2].set_title(f'Confidence Heatmap {self.sample}')
            axes[2].axis('off')
            plt.colorbar(scatter2, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            # Handle case with no predictions
            for i in range(1, 3):
                axes[i].text(0.5, 0.5, 'No predictions to display.',
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[i].transAxes)
                axes[i].axis('off')

        # Finalize and save the figure
        plt.tight_layout()
        plt.savefig(self.output_dir/file_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {self.output_dir/file_name}")


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
