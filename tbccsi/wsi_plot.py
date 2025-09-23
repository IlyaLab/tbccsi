
import pandas as pd
import matplotlib.pyplot as plt


class WSIPlotter:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, sample, slide):
        self.sample = sample
        self.slide = slide

    def tile_heatmap(self, predictions_df, output_path, point_size=3, prob_col="prob_class_1"):
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
        thumb_size = (2000, 2000)
        # --- DYNAMIC THUMBNAIL CREATION ---

        # --- END OF DYNAMIC slide loading LOGIC ---

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        # 1. Original thumbnail
        axes[0].imshow(thumbnail)
        axes[0].set_title(f'H&E {sample}')
        axes[0].axis('off')

        if not predictions_df.empty:
            # Calculate scaling factors to map tile coordinates to thumbnail coordinates

            scale_x = thumb_size[0] / slide_dims[0]
            scale_y = thumb_size[1] / slide_dims[1]

            # Scale coordinates
            x_scaled = predictions_df['x_coord'] * scale_x
            y_scaled = predictions_df['y_coord'] * scale_y

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
            axes[1].set_xlim(0, thumb_size[0])
            axes[1].set_ylim(thumb_size[1], 0)  # Invert y-axis for image coordinates
            axes[1].set_title(f'Prediction Heatmap {sample}')
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
            axes[2].set_xlim(0, thumb_size[0])
            axes[2].set_ylim(thumb_size[1], 0)
            axes[2].set_title(f'Confidence Heatmap {sample}')
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")
