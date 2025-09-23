import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from transformers import (
    ViTConfig,
    AutoImageProcessor,
)

from .vit_model_5_6 import VitClassification

# tile based classification on cell segmented images


class WSIInferenceEngine:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initializes the inference engine."""
        self.device = device
        self.model_path = model_path
        self.sample_id = None
        self.prefix = None

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
        print(f"Model loaded on {self.device}.")

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

    def load_and_predict_tiles(self, output_dir=None, tile_file_path=None, tile_input_dir=None, batch_size=32):
        print(f"Reading tile information from {tile_file_path}...")
        try:
            tile_df = pd.read_csv(tile_file_path)
            # Ensure required columns exist
            if not {'x', 'y', 'tile_id'}.issubset(tile_df.columns):
                raise ValueError("Tile file must contain 'x', 'y', and 'tile_id' columns.")
        except FileNotFoundError:
            print(f"Error: Tile file not found at {tile_file_path}")
            return []
        except Exception as e:
            print(f"Error reading tile file: {e}")
            return []

        loaded_tiles = []
        mask_input_dir = Path(tile_input_dir) # Convert to Path object for easier joining

        print(f"Loading {len(tile_df)} tiles from {tile_input_dir}...")

        # Use tqdm for a progress bar
        for _, row in tqdm(tile_df.iterrows(), total=tile_df.shape[0]):
            x, y, tile_id, mean_red, mean_green, mean_blue = int(row['x']), int(row['y']), int(row['tile_id']), float(row['mean_red']), float(row['mean_green']), float(row['mean_blue'])

            # Reconstruct the file path using the same logic as the saving function
            collection_index = tile_id // 1000
            collection_dir = mask_input_dir / f'collection_{collection_index}'
            file_name = f"masked_{tile_id:06d}_x{x}_y{y}.png"
            image_path = collection_dir / file_name

            try:
                # Open the image file
                with Image.open(image_path) as img:
                    # Load the image data into memory and convert to RGB
                    masked_pil_image = img.convert("RGB").copy()
                    loaded_tiles.append((masked_pil_image, x, y, tile_id, mean_red, mean_green, mean_blue))
            except FileNotFoundError:
                print(f"Warning: Tile file not found, skipping: {image_path}")
            except Exception as e:
                print(f"Warning: Could not load file {image_path}. Error: {e}")

        predictions_df = self.predict_tiles(loaded_tiles, batch_size)

        predictions_path = output_dir / f"{self.sample_id}_{self.prefix}_preds.csv"
        predictions_df.to_csv(predictions_path, index=False)

        print(f"Predictions saved to {predictions_path}")
        return(predictions_df)
