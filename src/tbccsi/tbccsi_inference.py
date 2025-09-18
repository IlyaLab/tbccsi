import os
import argparse
import logging
from pathlib import Path
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import torch
from PIL import Image
import openslide
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from cellpose import models, core, io

from transformers import (
    ViTPreTrainedModel,
    ViTConfig,
    ViTModel,
    AutoImageProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from vit_model_5_6 import VitClassification
from tbccsi_tiler import WSITiler

# tile based classification on cell segmented images

print("\n-----------------------------------------")
print("Checking if CUDA available: " + str(torch.cuda.is_available()))
print("-----------------------------------------\n")

print("\n Running google/vit-base-patch16-224-in21k model for binary classification. \n")



class WSIInferenceEngine:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initializes the inference engine."""
        self.device = device
        self.model_path = model_path

        # Load model configuration and processor
        print(f"Loading model from {model_path}")
        self.config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1}
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            use_fast=True
        )

        # Load the trained model
        self.model = VitClassification.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def predict_tiles(self, tiles, batch_size=32):
        """
        Performs inference on a list of tiles.

        Args:
            tiles (list): A list of tuples, where each tuple contains
                          (tile_image, x_coord, y_coord, tile_id).
            batch_size (int): The batch size for inference.

        Returns:
            pd.DataFrame: A DataFrame with predictions, probabilities, and coordinates.
        """
        results = []
        print(f"Running inference on {len(tiles)} tiles...")

        with torch.no_grad():
            for i in tqdm(range(0, len(tiles), batch_size)):
                batch_tiles = tiles[i:i + batch_size]

                # Prepare batch
                images = [tile[0] for tile in batch_tiles]  # Extract PIL images
                coords_and_ids = [(tile[1], tile[2], tile[3]) for tile in batch_tiles]

                # Process images
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                # Run inference
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits

                # Convert to probabilities and predictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Store results
                for j, (x, y, tile_id) in enumerate(coords_and_ids):
                    results.append({
                        'tile_id': tile_id,
                        'x_coord': x,
                        'y_coord': y,
                        'predicted_class': predictions[j].item(),
                        'prob_class_0': probabilities[j, 0].item(),
                        'prob_class_1': probabilities[j, 1].item(),
                        'confidence': torch.max(probabilities[j]).item()
                    })

        return pd.DataFrame(results)


    def create_heatmap(self, sample, slide, predictions_df, output_path, point_size=3, prob_col="prob_class_1"):
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
        if openslide and isinstance(slide, OpenSlide):
            # Handle OpenSlide object
            slide_dims = slide.level_dimensions[0]
            thumbnail = slide.get_thumbnail(thumb_size)
            print("Processing slide as OpenSlide object.")

        elif tifffile and isinstance(slide, tifffile.TiffFile):
            # Handle TiffFile object
            print("Processing slide as TiffFile object.")
            # Get dimensions from the highest resolution page (usually the first one)
            base_page = slide.pages[0]
            slide_dims = (base_page.shape[1], base_page.shape[0])  # (width, height)
            # Find the best pyramid level to use for the thumbnail to avoid loading
            # the massive full-resolution image into memory. We seek the smallest
            # level that is still larger than our desired thumbnail size.
            best_page = base_page
            for page in reversed(slide.pages):
                if page.shape[1] >= thumb_size[0] or page.shape[0] >= thumb_size[1]:
                    best_page = page
                    break
            # Read the image data of the selected level and create a PIL Image
            thumb_data = best_page.asarray()
            thumbnail = Image.fromarray(thumb_data)
            # Resize it to the final thumbnail size using a high-quality filter
            thumbnail.thumbnail(thumb_size, Image.Resampling.LANCZOS)

        elif isinstance(slide, np.ndarray):
            # Handle NumPy array object
            print("Processing slide as NumPy array.")
            # Get dimensions from the array's shape
            slide_dims = (slide.shape[1], slide.shape[0])  # (width, height)
            # Convert the full-resolution array to a PIL Image
            thumbnail = Image.fromarray(slide)
            # Resize it to the final thumbnail size using a high-quality filter
            thumbnail.thumbnail(thumb_size, Image.Resampling.LANCZOS)

        else:
            unsupported_type = type(slide)
            raise TypeError(
                f"Unsupported slide type: {unsupported_type}. "
                "Please provide an OpenSlide or TiffFile object. "
                "Ensure the respective libraries ('openslide-python' and 'tifffile') are installed."
            )
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
            #axes[1].imshow(thumbnail, alpha=0.3)  # Overlay on a faint thumbnail
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
            #axes[2].imshow(thumbnail, alpha=0.3)
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
