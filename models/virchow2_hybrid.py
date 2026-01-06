import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked
#from huggingface_hub import login


class Virchow2MultiHeadModel(nn.Module):
    """
    Multi-Head Model using Virchow2 (ViT-H/14) as a frozen backbone.

    Head 1: Structure (2 classes - e.g., Tumor/Stroma) - Classification
    Head 2: Immune Cells (7 classes - e.g., T-cell, Macrophage, etc.) - Regression (Log-Counts)
    """
    def __init__(self, hf_model, freeze_backbone=True):
        super().__init__()

        print("Loading Virchow2 (ViT-H/14)...")

        # 1. Load Backbone with Virchow2 specific requirements
        # We do not pass num_classes=0 to get the raw sequence output (tokens)
        self.backbone = hf_model
        # 2. Define Feature Dimension
        self.embed_dim = 1280
        # Strategy: Concatenate Class Token (1280) + Mean Patch Token (1280) = 2560
        n_features = self.embed_dim * 2

        # 3. Freeze Backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  -> Virchow2 Backbone Frozen")

        # --- HEAD 1: STRUCTURE (Classification) ---
        # Output: 2 logits (for BCEWithLogitsLoss)
        self.head_structure = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)  # TUMOR STROMA OR NEITHER
        )

        # --- HEAD 2: IMMUNE (Hierarchical Regression) ---
        # 1. Primary Shared Latent Space (512D)
        self.immune_shared_latent = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Output: [B, 512]
        # 3. T-cell Specific Latent Space (The Key Hierarchical Layer)
        # This layer branches from the main latent space and specializes for T-cell features.
        # It maintains 512D for consistency, or can be smaller (e.g., 256D).
        self.tcell_specific_latent = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        self.mac_specific_latent = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        # Parent Output Layer (Predicts Immune, T-cell, Macrophage)
        # This uses the features from the primary shared latent space.
        self.immune_output_parent = nn.Linear(512, 3)

        # Child Output Layers
        # T-Cell children prediction uses the T-cell specific latent features.
        self.immune_output_t_cell = nn.Linear(512, 2)  # CD4, CD8
        # Macrophage children can either use the shared or their own specific latent space.
        self.immune_output_mac = nn.Linear(512, 2)  # M1, M2


    def forward(self, x):
        # 1. Extract Raw Features (Output shape: [Batch, 261, 1280])
        # 261 = 1 (Class) + 4 (Registers) + 256 (Patches)
        outputs = self.backbone(x)

        # 2. Slice Tokens
        class_token = outputs[:, 0]  # [Batch, 1280] - The main feature vector
        patch_tokens = outputs[:, 5:] # [Batch, 256, 1280] - Ignore the 4 register tokens

        # 3. Compute Mean of Patches
        patch_mean = patch_tokens.mean(dim=1)  # [Batch, 1280]

        # 4. Concatenate for Rich Representation
        # Result shape: [Batch, 2560]
        features = torch.cat([class_token, patch_mean], dim=-1)

        # Head 1: Logits for Structure Classification
        out_structure = self.head_structure(features)

        # 1. Primary Shared Latent
        latent_immune = self.immune_shared_latent(features)

        # 2. Parent Prediction (Uses Primary Latent)
        out_immune_parent = self.immune_output_parent(latent_immune)

        # 3. Hierarchical Feature Flow (T-Cell)
        latent_tcell = self.tcell_specific_latent(latent_immune)  # Features passed down
        out_immune_tcell = self.immune_output_t_cell(latent_tcell)

        # 4. Hierarchical Feature Flow (Macrophage)
        latent_mac = self.mac_specific_latent(latent_immune)  # Features passed down
        out_immune_mac = self.immune_output_mac(latent_mac)

        # Concatenate all 7 outputs
        out_immune = torch.cat([out_immune_parent, out_immune_tcell, out_immune_mac], dim=1)

        return out_structure, out_immune
