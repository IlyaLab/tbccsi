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
                    #print("  -> Unfreezing LAST Transformer Block for adaptation...")
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
            nn.Dropout(0.3) # 0.3 is good here
        )

        # Updated Child Branches with HIGHER Dropout
        self.tcell_specific_latent = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3) # INCREASED from 0.05 (Too low!)
        )
        self.mac_specific_latent = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3) # INCREASED from 0.05
        )

        self.immune_output_parent = nn.Linear(512, 3)
        self.immune_output_t_cell = nn.Linear(512, 2)
        self.immune_output_mac = nn.Linear(512, 2) 

    def forward(self, x):
        outputs = self.backbone(x)
        
        # Depending on the library, you might need to ensure you get the tensor
        # If outputs is a tuple/object, access the hidden states. 
        # Assuming your previous code worked, we keep it, but ensure 'x' propagates grads if unfrozen.
        
        class_token = outputs[:, 0]
        patch_tokens = outputs[:, 5:] 
        #patch_mean = patch_tokens.mean(dim=1)
        patch_pooled = self.pool(patch_tokens) # 
        features = torch.cat([class_token, patch_pooled], dim=-1)
        
        # Apply the new Input Dropout
        features = self.input_dropout(features)

        # Head 1
        out_structure = self.head_structure(features)

        # Head 2
        latent_immune = self.immune_shared_latent(features)
        out_immune_parent = self.immune_output_parent(latent_immune)

        latent_tcell = self.tcell_specific_latent(latent_immune)
        out_immune_tcell = self.immune_output_t_cell(latent_tcell)

        latent_mac = self.mac_specific_latent(latent_immune)
        out_immune_mac = self.immune_output_mac(latent_mac)

        out_immune = torch.cat([out_immune_parent, out_immune_tcell, out_immune_mac], dim=1)

        return out_structure, out_immune
