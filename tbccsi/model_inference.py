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


class WSIInferenceEngine:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initializes the inference engine."""
        self.device = device
        self.model_path = model_path
        self.sample_id = None
        self.prefix = None

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


    def predict_from_tile_file(self, tile_file_path, tile_input_dir, output_dir, batch_size=32):
        """
        Loads tiles from a file and runs inference in batches to conserve memory.
        """
        print(f"Reading tile information from {tile_file_path}...")
        try:
            tile_df = pd.read_csv(tile_file_path)
            if not {'x', 'y', 'tile_id'}.issubset(tile_df.columns):
                raise ValueError("Tile file must contain 'x', 'y', and 'tile_id' columns.")
        except FileNotFoundError:
            print(f"Error: Tile file not found at {tile_file_path}")
            return None
        except Exception as e:
            print(f"Error reading tile file: {e}")
            return None

        all_results = []
        tile_input_path = Path(tile_input_dir)
        num_tiles = len(tile_df)

        print(f"\nStarting inference on {num_tiles} tiles in batches of {batch_size}...")

        # The main batching loop. This iterates over the DataFrame in chunks.
        with torch.no_grad():
            for i in tqdm(range(0, num_tiles, batch_size), desc="Processing Batches"):
                # 1. Get a chunk of the DataFrame
                batch_df = tile_df.iloc[i:i + batch_size]

                batch_images = []
                batch_metadata = []

                # 2. Load only the images for the current batch
                for _, row in batch_df.iterrows():
                    x, y, tile_id = int(row['x']), int(row['y']), int(row['tile_id'])

                    collection_index = tile_id // 1000
                    collection_dir = tile_input_path / f'collection_{collection_index}'
                    file_name = f"masked_{tile_id:06d}_x{x}_y{y}.png"
                    image_path = collection_dir / file_name

                    try:
                        with Image.open(image_path) as img:
                            batch_images.append(img.convert("RGB"))
                            batch_metadata.append({'tile_id': tile_id, 'x_coord': x, 'y_coord': y})
                    except FileNotFoundError:
                        # If a tile is missing, we skip it and its metadata
                        print(f"Warning: Tile file not found, skipping: {image_path}")

                # If no images were loaded in this batch (e.g., all were missing), continue
                if not batch_images:
                    continue

                # 3. Process and predict on the current batch of images
                inputs = self.processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # 4. Store results for the batch
                for j, metadata in enumerate(batch_metadata):
                    metadata.update({
                        'predicted_class': predictions[j].item(),
                        'prob_class_0': probabilities[j, 0].item(),
                        'prob_class_1': probabilities[j, 1].item(),
                        'confidence': torch.max(probabilities[j]).item()
                    })
                    all_results.append(metadata)

        # After the loop, create the final DataFrame and save it
        predictions_df = pd.DataFrame(all_results)
        predictions_path = Path(output_dir) / f"{self.sample_id}_{self.prefix}_preds.csv"
        predictions_df.to_csv(predictions_path, index=False)

        print(f"\nPredictions saved to {predictions_path}")
        return predictions_df

    def merge_predictions(prediction_files, prefixes, output_path):
        """
        Merges multiple prediction CSVs into a single file with renamed columns.

        Args:
            prediction_files (list): A list of paths to the prediction CSV files.
            prefixes (list): A list of prefixes to use for renaming columns.
            output_path (str or Path): The path to save the final merged CSV.
        """
        if len(prediction_files) != len(prefixes):
            raise ValueError("The number of prediction files must equal the number of prefixes.")

        if not prediction_files:
            print("No prediction files provided to merge.")
            return None

        # --- Load the first file as the base for our merge ---
        print(f"Loading base file: {prediction_files[0]}")
        merged_df = pd.read_csv(prediction_files[0])

        # Define the columns to be renamed
        rename_map = {
            'predicted_class': f"pred_{prefixes[0]}",
            'prob_class_0': f"prob0_{prefixes[0]}",
            'prob_class_1': f"prob1_{prefixes[0]}",
            'confidence': f"conf_{prefixes[0]}"
        }
        merged_df.rename(columns=rename_map, inplace=True)

        # --- Loop through the rest of the files and merge them ---
        for i in range(1, len(prediction_files)):
            file_path = prediction_files[i]
            prefix = prefixes[i]
            print(f"Merging file: {file_path}")

            next_df = pd.read_csv(file_path)

            # Define the columns to rename for the current file
            rename_map = {
                'predicted_class': f"pred_{prefix}",
                'prob_class_0': f"prob0_{prefix}",
                'prob_class_1': f"prob1_{prefix}",
                'confidence': f"conf_{prefix}"
            }
            next_df.rename(columns=rename_map, inplace=True)

            # Perform an outer merge to keep all tiles from both files
            merged_df = pd.merge(
                merged_df,
                next_df,
                on=['tile_id', 'x_coord', 'y_coord'],
                how='outer'
            )

        # --- Save the final merged DataFrame ---
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Merged predictions saved to: {output_path}")
        return merged_df