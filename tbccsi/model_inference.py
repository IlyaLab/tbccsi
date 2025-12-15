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

