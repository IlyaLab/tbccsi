import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Patches, Dim]
        # weights: [Batch, Patches, 1]
        w = self.attention(x)
        w = torch.softmax(w, dim=1) # Softmax over patches
        # Weighted Sum
        return torch.sum(x * w, dim=1)
        

class Virchow2MultiHeadModel(nn.Module):
    def __init__(self, hf_model, freeze_backbone=True):
        super().__init__()
        
        self.backbone = hf_model
        self.embed_dim = 1280
        n_features = self.embed_dim * 2

        self.pool = AttentionPooling(1280)

        # --- STRATEGY 2: Selective Unfreezing ---
        if freeze_backbone:
            # 1. First, freeze everything
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # 2. Then, UNFREEZE the last encoder block (The "Adapter" Strategy)
            # Note: The attribute name depends on the specific model architecture (TIMM vs HF).
            # This generic approach attempts to find the last list of blocks.
            try:
                # Try standard HuggingFace/TIMM locations for the last block
                if hasattr(self.backbone, 'layers'): # Common in some ViTs
                     last_block = self.backbone.layers[-1]
                elif hasattr(self.backbone, 'blocks'): # TIMM style
                     last_block = self.backbone.blocks[-1]
                elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'): # HF ViT
                     last_block = self.backbone.encoder.layers[-1]
                else:
                    last_block = None
                    print("Warning: Could not automatically locate last block to unfreeze.")

                if last_block:
                    print("  -> Unfreezing LAST Transformer Block for adaptation...")
                    for param in last_block.parameters():
                        param.requires_grad = True
            except Exception as e:
                print(f"  -> Error unfreezing last block: {e}")
        
        # --- STRATEGY 1: Projector Dropout ---
        # Drop 20% of the raw frozen features before they even reach the heads.
        # This forces the heads to be robust.
        self.input_dropout = nn.Dropout(0.2) 

        # --- HEAD 1: STRUCTURE ---
        self.head_structure = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased from 0.2
            nn.Linear(512, 2) 
        )

        # --- HEAD 2: IMMUNE ---
        self.immune_shared_latent = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3) 
        )

        # Implementation of the Step-Down (Bottleneck) to 64D
        self.tcell_specific_latent = nn.Sequential(
            nn.Linear(128, 64),   # Step 2: Final 64D bottleneck
            nn.ReLU(),
            nn.Dropout(0.3)       # Maintain the higher dropout from v2
        )

        self.mac_specific_latent = nn.Sequential(
            nn.Linear(128, 64),   # Step 2
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layers
        self.immune_output_t_cell = nn.Linear(64, 2)
        self.immune_output_mac = nn.Linear(64, 2)
        self.immune_output_parent = nn.Linear(128, 3)

    def get_latents(self, x):
        """
        Extracts intermediate latent representations without calculating final logits.
        Useful for visualization (UMAP/t-SNE) or downstream tasks.
        """
        outputs = self.backbone(x)

        # 1. Base Feature Extraction (Replicating forward logic)
        class_token = outputs[:, 0]
        patch_tokens = outputs[:, 5:]
        patch_pooled = self.pool(patch_tokens)
        features = torch.cat([class_token, patch_pooled], dim=-1)

        # Apply input dropout (consistent with training, though often disabled in eval)
        features = self.input_dropout(features)

        # 2. Extract Structure Head Latent (512D)
        # We manually iterate through the Sequential layers to stop before the final classifier
        struct_x = features
        # head_structure = [Linear(2560, 512), ReLU, Dropout, Linear(512, 2)]
        # We want the output after the Dropout (index 2)
        for i, layer in enumerate(self.head_structure):
            struct_x = layer(struct_x)
            if i == 2:  # Stop after Dropout
                latent_structure = struct_x
                break

        # 3. Extract Immune Head Latents
        # Shared Immune Latent (128D)
        latent_immune = self.immune_shared_latent(features)

        # T-Cell Specific Latent (64D)
        # Note: We stop before the final output_t_cell layer
        latent_tcell = self.tcell_specific_latent(latent_immune)

        # Macrophage Specific Latent (64D)
        # Note: We stop before the final output_mac layer
        latent_mac = self.mac_specific_latent(latent_immune)

        return {
            "backbone": features,  # The raw 2560D combined embedding
            "structure": latent_structure,  # The 512D structure embedding
            "immune_shared": latent_immune,  # The 128D shared immune embedding
            "immune_tcell": latent_tcell,  # The 64D T-cell specific bottleneck
            "immune_mac": latent_mac  # The 64D Mac specific bottleneck
        }

    def forward(self, x, return_latents=False):
        # 1. Get all latents
        latents = self.get_latents(x)

        # 2. Compute final logits from these latents

        # Structure Output
        # We need the final layer of head_structure.
        # Since we manually extracted up to index 2 in get_latents,
        # we can pass the result to the last layer (index 3).
        out_structure = self.head_structure[3](latents["structure"])

        # Immune Outputs
        out_immune_parent = self.immune_output_parent(latents["immune_shared"])
        out_immune_tcell = self.immune_output_t_cell(latents["immune_tcell"])
        out_immune_mac = self.immune_output_mac(latents["immune_mac"])

        out_immune = torch.cat([out_immune_parent, out_immune_tcell, out_immune_mac], dim=1)

        if return_latents:
            return out_structure, out_immune, latents

        return out_structure, out_immune
