from .vit_model_5_7 import VitClassification

import torch
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

# Import your custom modules
from .model_virchow2 import Virchow2MultiHeadModel

# Ensure truncated images don't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. Reinhard Normalizer ---
class ReinhardNormalizer:
    """
    Normalizes H&E images to a target color distribution in LAB space.
    Default targets roughly correspond to a standard high-quality H&E slide.
    """

    def __init__(self, target_means=None, target_stds=None):
        # Default target values (L, A, B)
        if target_means is None:
            self.target_means = np.array([148.60, 169.30, 105.97], dtype=np.float32)
        else:
            self.target_means = np.array(target_means, dtype=np.float32)

        if target_stds is None:
            self.target_stds = np.array([41.56, 9.01, 6.67], dtype=np.float32)
        else:
            self.target_stds = np.array(target_stds, dtype=np.float32)

    def normalize(self, pil_img):
        img_np = np.array(pil_img)
        # Convert RGB to LAB
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        img_lab_float = img_lab.astype(np.float32)

        # Calculate statistics of the input image
        tile_means = np.mean(img_lab_float, axis=(0, 1))
        tile_stds = np.std(img_lab_float, axis=(0, 1))

        # Avoid division by zero
        tile_stds[tile_stds == 0] = 1e-5

        # Normalize and Scale to Target
        img_normalized = ((img_lab_float - tile_means) / tile_stds) * self.target_stds + self.target_means

        # Clip to valid range [0, 255]
        img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)

        # Convert back to RGB
        return Image.fromarray(cv2.cvtColor(img_normalized, cv2.COLOR_LAB2RGB))


# --- 2. Updated Inference Engine ---
from safetensors.torch import load_file  # <--- NEW IMPORT


class VirchowInferenceEngine:
    """
    Inference Engine for Virchow2 MultiHead Model.
    Supports .pth (Pickle) and .safetensors.
    """

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 1. Handle list input
        if isinstance(model_path, list):
            model_path = model_path[0]
        self.model_path = Path(model_path)

        print("Initializing Virchow2 Backbone...")
        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

        config = resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone)
        self.processor = create_transform(**config)

        print(f"Loading weights from {self.model_path}...")
        self.model = Virchow2MultiHeadModel(hf_model=self.backbone, freeze_backbone=True)

        # --- 2. LOAD STATE DICT (Handle .safetensors vs .pth) ---
        if str(self.model_path).endswith(".safetensors"):
            # Load safetensors
            state_dict = load_file(self.model_path, device="cpu")  # Load to CPU first
        else:
            # Load standard pytorch pickle
            state_dict = torch.load(self.model_path, map_location="cpu")

        # --- 3. FIX PREFIXES (Handle Trainer Wrapper) ---
        # The Trainer saved 'Virchow2TrainerModel', so keys probably start with "model."
        # But here we are initializing 'Virchow2MultiHeadModel', so we need to remove that prefix.
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Strip "model."
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Load weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"⚠️ Warning: Missing keys: {missing[:5]} ...")
        if unexpected:
            print(f"⚠️ Warning: Unexpected keys: {unexpected[:5]} ...")

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

    def predict_batch(self, images, metadata_list):
        """
        Runs inference on a list of PIL images.
        """
        if not images:
            return []

        # Transform images to tensors
        tensors = []
        valid_indices = []  # Track which images successfully transformed

        for idx, img in enumerate(images):
            try:
                tensors.append(self.processor(img))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing image in batch: {e}")

        if not tensors:
            return []

        input_batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            out_structure, out_immune = self.model(input_batch)

            # Sigmoid lets them be independent.
            struct_probs = torch.sigmoid(out_structure)

            # Post-process Immune
            immune_vals = out_immune.cpu().numpy()

        results = []
        for i, original_idx in enumerate(valid_indices):
            meta = metadata_list[original_idx].copy()

            # Append predictions to metadata
            # Extract independent probabilities
            # Assuming Training Index 1 = Stroma, Index 0 = Tumor
            prob_tumor = struct_probs[i, 0].item()
            prob_stroma = struct_probs[i, 1].item()

            # Optional: Simple discrete predictions based on threshold
            # 0=Neither, 1=Stroma, 2=Tumor, 3=Both
            is_stroma = prob_stroma > 0.5
            is_tumor = prob_tumor > 0.5

            if is_stroma and is_tumor:
                meta['pred_class_label'] = "Both"
            elif is_stroma:
                meta['pred_class_label'] = "Stroma"
            elif is_tumor:
                meta['pred_class_label'] = "Tumor"
            else:
                meta['pred_class_label'] = "Neither"

            meta['prob_stroma'] = prob_stroma
            meta['prob_tumor'] = prob_tumor
            meta['reg_immune_total'] = immune_vals[i, 0]
            meta['reg_t_cell'] = immune_vals[i, 1]
            meta['reg_macrophage'] = immune_vals[i, 2]
            meta['reg_cd4'] = immune_vals[i, 3]
            meta['reg_cd8'] = immune_vals[i, 4]
            meta['reg_m1'] = immune_vals[i, 5]
            meta['reg_m2'] = immune_vals[i, 6]

            results.append(meta)

        return results


class WSIInferenceEngine:
    """Handles model loading, inference on tiles, and heatmap generation."""

    def __init__(self, config, processor, model, model_path,
                 tile_map_df=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initializes the inference engine."""
        self.device = device
        self.model_path = model_path
        self.sample_id = None
        self.prefix = None
        self.tile_map_df = tile_map_df

        print(f"Loading model from {model_path}")

        self.config = config
        #ViTConfig.from_pretrained(
        #    "google/vit-base-patch16-224-in21k",
        #    num_labels=2,
        #    id2label={0: "negative", 1: "positive"},
        #    label2id={"negative": 0, "positive": 1}
        #)
        self.processor = processor
        #AutoImageProcessor.from_pretrained(
        #    "google/vit-base-patch16-224-in21k",
        #    use_fast=True
        #)

        self.model = model
        #VitClassification.from_pretrained(model_path, config=self.config)

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


    def run_inference(self, output_dir, batch_size=32):
        """
        Runs inference on the pre-loaded self.tile_map_df.

        self.tile_map_df is expected to have 'x', 'y', 'tile_id', and 'file_path' columns.
        """
        # 1. Check if the tile map is valid
        print(f"Reading tile information from self.tile_map_df...")
        if self.tile_map_df is None or self.tile_map_df.empty:
            print("Error: self.tile_map_df is not loaded. Cannot run inference.")
            return None

        # Use a local variable for convenience
        tile_df = self.tile_map_df

        required_cols = {'x', 'y', 'tile_id', 'file_path'}
        if not required_cols.issubset(tile_df.columns):
            missing = required_cols - set(tile_df.columns)
            print(f"Error: self.tile_map_df is missing required columns: {missing}")
            return None

        all_results = []
        num_tiles = len(tile_df)

        print(f"\nStarting inference on {num_tiles} mapped tiles in batches of {batch_size}...")

        # The main batching loop. This iterates over the DataFrame in chunks.
        with torch.no_grad():
            for i in tqdm(range(0, num_tiles, batch_size), desc="Processing Batches"):
                # 1. Get a chunk of the DataFrame
                batch_df = tile_df.iloc[i:i + batch_size]

                batch_images = []
                batch_metadata = []

                # 2. Load only the images for the current batch
                for _, row in batch_df.iterrows():

                    # --- THIS IS THE KEY CHANGE ---
                    # We get all data directly from the row.
                    # The complex path-building logic is gone.
                    tile_id = int(row['tile_id'])
                    x = int(row['x'])
                    y = int(row['y'])
                    image_path = Path(row['file_path'])  # Use the pre-scanned path
                    # --- END OF CHANGE ---

                    try:
                        with Image.open(image_path) as img:
                            batch_images.append(img.convert("RGB"))
                            # Store the coordinates from the map
                            batch_metadata.append({'tile_id': tile_id, 'x_coord': x, 'y_coord': y})
                    except FileNotFoundError:
                        # If a tile is missing, we skip it and its metadata
                        print(f"Warning: Tile file not found, skipping: {image_path}")
                    except Exception as e:
                        print(f"Warning: Error loading {image_path}, skipping: {e}")

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
                # (This logic is identical to your original function)
                for j, metadata in enumerate(batch_metadata):
                    metadata.update({
                        'predicted_class': predictions[j].item(),
                        'prob_class_0': probabilities[j, 0].item(),
                        'prob_class_1': probabilities[j, 1].item(),
                        'confidence': torch.max(probabilities[j]).item()
                    })
                    all_results.append(metadata)

        # After the loop, create the final DataFrame and save it
        if not all_results:
            print("Error: No tiles were successfully processed. No predictions saved.")
            return None

        predictions_df = pd.DataFrame(all_results)

        # Merge with original tile_map_df to retain all coordinate info
        # This ensures the final CSV has 'x', 'y', 'mean_r', etc.
        # We merge on 'tile_id' but use the coords from the results for robustness.
        final_df = tile_df.merge(predictions_df, on='tile_id', how='right',
                                 suffixes=('_orig', None))

        # Clean up columns: drop redundant x/y coords
        if 'x_coord' in final_df.columns:
            final_df = final_df.drop(columns=['x_coord_orig', 'y_coord_orig'], errors='ignore')
            #final_df = final_df.rename(columns={'x_coord': 'x', 'y_coord': 'y'})

        return final_df


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