import torch
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
from .models.virchow2_hybrid import Virchow2MultiHeadModel

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
            self.target_stds = np.array([53.17,  4.72,  5.23], dtype=np.float32)
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

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

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