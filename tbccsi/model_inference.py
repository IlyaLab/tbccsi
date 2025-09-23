import numpy as np
import pandas as pd
import torch
from PIL import Image
import openslide
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
import tifffile
from tqdm import tqdm


from transformers import (
    ViTPreTrainedModel,
    ViTConfig,
    ViTModel,
    AutoImageProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from vit_model_5_6 import VitClassification
from wsi_tiler import WSITiler

# tile based classification on cell segmented images


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

        print("\n Running google/vit-base-patch16-224-in21k model for binary classification. \n")
        print(f"  Inference on {len(tiles)} tiles...")

        results = []

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

