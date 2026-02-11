import torch
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageFile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from safetensors.torch import load_file

## our model ##
from .models.model_virchow2_v3 import Virchow2MultiHeadModel

# Ensure truncated images don't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- 1. Reinhard Normalizer ---
class ReinhardNormalizer:
    """
    Normalizes H&E images to a target color distribution in LAB space.
    Used in  data processing for the wash out:
    TARGET_MEANS = [147.92, 141.08, 122.12]
    TARGET_STDS = [53.17,  4.72,  5.23]
    """

    def __init__(self, target_means=None, target_stds=None):
        if target_means is None:
            self.target_means = np.array([147.92, 141.08, 122.12], dtype=np.float32)
        else:
            self.target_means = np.array(target_means, dtype=np.float32)

        if target_stds is None:
            self.target_stds = np.array([53.17, 4.72, 5.23], dtype=np.float32)
        else:
            self.target_stds = np.array(target_stds, dtype=np.float32)

    def normalize(self, pil_img):
        img_np = np.array(pil_img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        img_lab_float = img_lab.astype(np.float32)
        tile_means = np.mean(img_lab_float, axis=(0, 1))
        tile_stds = np.std(img_lab_float, axis=(0, 1))
        tile_stds[tile_stds == 0] = 1e-5
        img_normalized = ((img_lab_float - tile_means) / tile_stds) * self.target_stds + self.target_means
        img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img_normalized, cv2.COLOR_LAB2RGB))


# --- 2. Updated Inference Engine with TTA ---
class VirchowInferenceEngine:
    """
    Inference Engine for Virchow2 MultiHead Model.
    Supports .pth, .safetensors, and Test Time Augmentation (TTA).
    """

    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # If no model path is provided, stop here (lightweight init)
        if model_path is None:
            print("Initialized VirchowInferenceEngine without loading a model.")
            self.model = None
            self.processor = None
            return

        # --- Heavy Initialization (only runs if model_path is provided) ---
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

        # Handle .safetensors vs .pth
        if str(self.model_path).endswith(".safetensors"):
            state_dict = load_file(self.model_path, device="cpu")
        else:
            state_dict = torch.load(self.model_path, map_location="cpu")

        # Fix prefix issues from Trainer wrapper
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"⚠️ Warning: Missing keys: {missing[:5]} ...")

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

    def _get_tta_variants(self, img):
        """
        Generates 8 Dihedral variants of the image:
        - 4 Rotations (0, 90, 180, 270)
        - 4 Rotations of the Horizontal Flip
        """
        variants = []
        # Original + Rotations
        variants.append(img)  # 0
        variants.append(img.rotate(90, expand=True))
        variants.append(img.rotate(180, expand=True))
        variants.append(img.rotate(270, expand=True))

        # Flipped + Rotations
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        variants.append(img_flip)  # 0 flip
        variants.append(img_flip.rotate(90, expand=True))
        variants.append(img_flip.rotate(180, expand=True))
        variants.append(img_flip.rotate(270, expand=True))

        return variants

    def predict_batch(self, images, metadata_list, use_tta=False):
        """
        Runs inference on a list of PIL images.

        Args:
            images: List of PIL images.
            metadata_list: List of dictionaries containing metadata.
            use_tta (bool): If True, applies Test Time Augmentation (8 views) and averages predictions.
        """
        if not images:
            return []

        results = []

        # We loop through images individually or in small groups if using TTA
        # to prevent memory explosion (since TTA = 8x batch size).
        for idx, img in enumerate(images):
            meta = metadata_list[idx].copy()

            try:
                # --- TTA LOGIC ---
                if use_tta:
                    # Generate 8 variants
                    variants = self._get_tta_variants(img)
                    # Transform all variants
                    tensors = [self.processor(v) for v in variants]
                    # Stack: Shape [8, 3, 224, 224]
                    input_batch = torch.stack(tensors).to(self.device)
                else:
                    # Standard Single Image
                    # Stack: Shape [1, 3, 224, 224]
                    input_batch = self.processor(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out_structure, out_immune = self.model(input_batch)

                    # Sigmoid to get Probabilities
                    struct_probs_batch = torch.sigmoid(out_structure)  # [N, 2]
                    immune_probs_batch = torch.sigmoid(out_immune)  # [N, 7]

                    # --- AGGREGATION ---
                    # If TTA, we average across the 0-th dimension (the 8 variants)
                    # If not TTA, mean() over dim 0 effectively just removes the batch dim of 1
                    struct_probs = torch.mean(struct_probs_batch, dim=0).cpu().numpy()
                    immune_vals = torch.mean(immune_probs_batch, dim=0).cpu().numpy()

                # --- PARSE RESULTS ---
                # Structure: Index 0=Tumor, 1=Stroma (Confirm this mapping with your training code)
                prob_tumor = struct_probs[0].item()
                prob_stroma = struct_probs[1].item()

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

                # Immune: 0=General, 1=T-Cell, 2=Macrophage, 3=CD4, 4=CD8, 5=M1, 6=M2
                meta['prob_immune'] = immune_vals[0]
                meta['prob_t_cell'] = immune_vals[1]
                meta['prob_macrophage'] = immune_vals[2]
                meta['prob_cd4'] = immune_vals[3]
                meta['prob_cd8'] = immune_vals[4]
                meta['prob_m1'] = immune_vals[5]
                meta['prob_m2'] = immune_vals[6]

                if use_tta:
                    meta['tta_enabled'] = True

                results.append(meta)

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                # Append basic failure metadata or skip
                continue

        return results

    def extract_embeddings_batch(self, images, metadata_list, use_tta=False,
                                 latent_types=['backbone', 'structure', 'immune_shared', 'immune_tcell', 'immune_mac']):
        """
        Extracts latent embeddings from a list of PIL images.

        Args:
            images: List of PIL images.
            metadata_list: List of dictionaries containing metadata (tile coords, etc).
            use_tta (bool): If True, applies Test Time Augmentation and averages embeddings.
            latent_types (list): Which latent representations to extract. Options:
                - 'backbone': Raw 2560D combined embedding (class token + pooled patches)
                - 'structure': 512D structure head latent
                - 'immune_shared': 128D shared immune latent
                - 'immune_tcell': 64D T-cell specific latent
                - 'immune_mac': 64D Macrophage specific latent

        Returns:
            List of dictionaries with metadata + embeddings as numpy arrays
        """
        if not images:
            return []

        results = []

        for idx, img in enumerate(images):
            meta = metadata_list[idx].copy()

            try:
                # --- TTA LOGIC ---
                if use_tta:
                    variants = self._get_tta_variants(img)
                    tensors = [self.processor(v) for v in variants]
                    input_batch = torch.stack(tensors).to(self.device)
                else:
                    input_batch = self.processor(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Extract latents using the model's get_latents method
                    latents_dict = self.model.get_latents(input_batch)

                    # --- AGGREGATION ---
                    # If TTA, average across the batch dimension (8 variants)
                    # Otherwise, just squeeze the batch dimension
                    for latent_name in latent_types:
                        if latent_name in latents_dict:
                            latent_tensor = latents_dict[latent_name]

                            if use_tta:
                                # Average across TTA variants
                                latent_avg = torch.mean(latent_tensor, dim=0).cpu().numpy()
                            else:
                                # Single image - squeeze batch dim
                                latent_avg = latent_tensor.squeeze(0).cpu().numpy()

                            # Store as flat array
                            meta[f'embedding_{latent_name}'] = latent_avg

                if use_tta:
                    meta['tta_enabled'] = True

                results.append(meta)

            except Exception as e:
                print(f"Error extracting embeddings for image {idx}: {e}")
                continue

        return results

    def apply_cellcalling_thresholds(self, predictions_df, thresholds_csv):
        """
        Applies thresholds to allow MULTIPLE labels per tile.

        1. Determines 'Context' (Tumor/Stroma/Ambiguous) to select the correct strictness level.
        2. Flags ANY cell type (Structure or Immune) that exceeds its specific safety threshold.
        3. Returns boolean columns (e.g., 'Is_Tumor', 'Is_CD8') which can co-occur.
        """
        print("Loading thresholds for multi-label tile calling...")

        if type(predictions_df) == str:
            # 0. Load preds
            predictions_df = pd.read_csv(predictions_df)

        # 1. Load Thresholds
        # Expects columns: ['cell', 'context', 'threshold']
        thresh_df = pd.read_csv(thresholds_csv)

        # Helper: Fast lookup for thresholds
        # We turn the CSV into a dict: dict[cell][context] = threshold
        thresh_lookup = {}
        for cell in thresh_df['cell'].unique():
            thresh_lookup[cell] = {}
            sub = thresh_df[thresh_df['cell'] == cell]
            for _, row in sub.iterrows():
                thresh_lookup[cell][row['context']] = row['threshold']

        # ---------------------------------------------------------
        # STEP 1: DEFINE STRUCTURAL CONTEXT (The "Environment")
        # ---------------------------------------------------------
        # We need to know if the tile is "Tumor-y" or "Stroma-y" to pick
        # the right immune thresholds.

        # Get Global thresholds for structure
        t_thresh = thresh_lookup['prob_tumor'].get('Global', 0.90)
        s_thresh = thresh_lookup['prob_stroma'].get('Global', 0.90)

        print(f'using {t_thresh} and {s_thresh} structural thresholds')

        # Define Dominant Context
        # Note: We rely on dominance here just to select the *immune* threshold strictness.
        # We aren't making the final "Is_Tumor" call yet.
        is_tumor_context = (predictions_df['prob_tumor'] > predictions_df['prob_stroma']) & \
                           (predictions_df['prob_tumor'] > t_thresh)

        is_stroma_context = (predictions_df['prob_stroma'] > predictions_df['prob_tumor']) & \
                            (predictions_df['prob_stroma'] > s_thresh)

        # Assign temporary context strings for lookup
        context_series = pd.Series('Ambiguous_Region', index=predictions_df.index)
        context_series[is_tumor_context] = 'Tumor_Region'
        context_series[is_stroma_context] = 'Stroma_Region'

        # Store context if you want to inspect it later
        predictions_df['Threshold_Context'] = context_series

        # ---------------------------------------------------------
        # STEP 2: APPLY INDEPENDENT CHECKS (Multi-Label)
        # ---------------------------------------------------------
        # We iterate through every known cell type in the lookup table.
        # If a tile passes the threshold, it gets the label.

        all_cells = list(thresh_lookup.keys())

        for cell in all_cells:

            # 1. Setup Column Names
            prob_col = cell
            flag_col = cell.replace('prob_', 'called_')

            # Skip if model didn't predict this cell
            if prob_col not in predictions_df.columns:
                print(f'{prob_col} not in pred columns...')
                continue

            predictions_df[flag_col] = False

            # 2. Structure Nodes (n_tumor, n_stroma) usually have 'Global' thresholds
            if 'Global' in thresh_lookup[cell]:
                limit = thresh_lookup[cell]['Global']
                predictions_df.loc[predictions_df[prob_col] > limit, flag_col] = True

            # 3. Immune Nodes have Context-Specific thresholds
            else:
                # We must apply different thresholds based on the Context we defined in Step 1
                for ctx in ['Tumor_Region', 'Stroma_Region', 'Ambiguous_Region']:
                    # Get threshold for this cell in this context
                    limit = thresh_lookup[cell].get(ctx, 0.99)  # Default to strict 0.99 if missing

                    # Find tiles belonging to this context
                    ctx_mask = (context_series == ctx)

                    # Apply threshold
                    pass_mask = (predictions_df.loc[ctx_mask, prob_col] > limit)

                    # Set True for passing tiles
                    predictions_df.loc[pass_mask.index[pass_mask], flag_col] = True

        # ---------------------------------------------------------
        # OPTIONAL: SUMMARY COLUMN
        # ---------------------------------------------------------
        # Create a human-readable list of labels for each tile
        # e.g., "Tumor; CD8; T_Cell"

        flag_cols = [c for c in predictions_df.columns if c.startswith("Is_")]

        def get_labels(row):
            return [col.replace("Is_", "").replace("n_", "").upper() for col in flag_cols if row[col]]

        # Applying row-wise is slow for millions of rows.
        # Only uncomment if dataset is small (<100k) or for debugging.
        # predictions_df['Labels_List'] = predictions_df.apply(get_labels, axis=1)
        # predictions_df['Labels_List'] = predictions_df.apply(get_labels, axis=1)

        return predictions_df