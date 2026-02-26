
# ============================================================================
# Attention Pooling (from your Virchow2 model)
# ============================================================================

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
        w = self.attention(x)
        w = torch.softmax(w, dim=1)
        return torch.sum(x * w, dim=1)


# ============================================================================
# DAFT Model with Domain-Specific BN
# ============================================================================

class DAFT_Virchow2_Macrophage(nn.Module):
    """
    Domain-Aware Fine-Tuning for Virchow2
    
    Key features:
    1. Virchow2 backbone with attention pooling
    2. Gradual fine-tuning strategy (3 phases)
    3. Frozen backbone BN layers (prevents feature distortion)
    4. Domain-specific BN in classifier head
    """
    def __init__(self, virchow_backbone, num_classes=4, num_domains=2):
        super().__init__()
        
        self.backbone = virchow_backbone
        self.embed_dim = config.VIRCHOW_DIM
        self.num_domains = num_domains
        
        # Attention pooling for patch tokens
        self.pool = AttentionPooling(self.embed_dim)
        
        # Feature dimension after concatenating class token + pooled patches
        n_features = self.embed_dim * 2
        
        # Store original BN statistics
        self.original_bn_stats = {}
        self._store_bn_stats()
        
        # Input dropout (from your model)
        self.input_dropout = nn.Dropout(0.2)
        
        # Classifier with domain-specific BN
        self.fc1 = nn.Linear(n_features, 512)
        
        # Domain-specific batch normalization layers
        self.domain_bns = nn.ModuleList([
            nn.BatchNorm1d(512) for _ in range(num_domains)
        ])
        
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.training_phase = 'linear_probe'
    
    def _store_bn_stats(self):
        """Store original batch norm statistics from pre-trained Virchow2"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.original_bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.clone() if module.weight is not None else None,
                    'bias': module.bias.clone() if module.bias is not None else None
                }
        print(f"Stored BN stats for {len(self.original_bn_stats)} layers")
    
    def convert_bn_to_domain_aware(self):
        """
        DAFT's Batch Normalization Conversion
        Freeze BN layers to prevent feature distortion
        """
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Set to eval mode and disable tracking
                module.eval()
                module.track_running_stats = False
                
                # Restore original statistics
                if name in self.original_bn_stats:
                    module.running_mean.data.copy_( self.original_bn_stats[name]['running_mean'] )
                    module.running_var.data.copy_( self.original_bn_stats[name]['running_var'] )
                    if module.weight is not None:
                        module.weight.data = self.original_bn_stats[name]['weight']
                    if module.bias is not None:
                        module.bias.data = self.original_bn_stats[name]['bias']
        
        print("Applied DAFT BN conversion - backbone BN layers frozen")
    
    def forward(self, x, domain_id):
        # Extract Virchow2 features
        outputs = self.backbone(x)
        
        # Class token and patch tokens (following your model structure)
        class_token = outputs[:, 0]
        patch_tokens = outputs[:, 5:]  # Skip first 5 tokens
        
        # Attention pooling over patches
        patch_pooled = self.pool(patch_tokens)
        
        # Concatenate class token + pooled patches
        features = torch.cat([class_token, patch_pooled], dim=-1)
        
        # Input dropout
        features = self.input_dropout(features)
        
        # First FC layer
        h = self.fc1(features)
        
        # Apply domain-specific BN
        if domain_id.dim() == 0:  # Single scalar
            h = self.domain_bns[domain_id.item()](h)
        else:  # Batch of domain IDs
            h_normalized = torch.zeros_like(h)
            for d in range(self.num_domains):
                mask = (domain_id == d)
                if mask.any():
                    h_normalized[mask] = self.domain_bns[d](h[mask])
            h = h_normalized
        
        # Final layers
        return self.fc2(h)

