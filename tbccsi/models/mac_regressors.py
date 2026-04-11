"""
mac_regressors.py — tbccsi model wrappers for macrophage count regression.

Five models trained on dataset9 to predict log1p(n_m1) and log1p(n_m2)
from Virchow2 CLS embeddings (or ResNet-50 features).

All models load only the head weights from fold0_best.pt (backbone weights
come from pretrained timm initialization and are frozen at inference).

Usage via tbccsi (config auto-detected from model_config.yaml next to weights):
    tbccsi pred --sample-id S1 --input-slide slide.svs \\
        --work-dir ./out --tile-file tiles.csv \\
        -m /path/to/train_full_ds9/virchow2_mlp/fold0_best.pt --do-inference

Domain IDs (for FiLM and Domain-Specific models):
    0 = g6d2v1 (domain 0)
    1 = g1d3, g4d2, g4d3 (domain 1, standard post-CODEX staining)
"""

import torch
import torch.nn as nn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cls_token(backbone, x):
    """Extract CLS token [B, 1280] from Virchow2 backbone output [B, tokens, dim]."""
    out = backbone(x)
    return out[:, 0]


# ── FiLM layer (matches domain_film_regressor.py exactly) ────────────────────

class _FiLM(nn.Module):
    """Feature-wise Linear Modulation — per-domain scale + shift."""
    def __init__(self, num_features: int, num_domains: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_domains, num_features))
        self.beta  = nn.Parameter(torch.zeros(num_domains, num_features))

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        ids = domain_ids.long()
        return self.gamma[ids] * x + self.beta[ids]


# ── Domain-specific BN (matches domain_specific.py exactly) ──────────────────

class _DomainBN(nn.Module):
    """Per-domain BatchNorm1d. State dict keys: bns.{i}.* """
    def __init__(self, num_features: int, num_domains: int):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(num_domains)])

    def forward(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        return self.bns[domain_idx](x)


# ═════════════════════════════════════════════════════════════════════════════
# Model 1: Virchow2 Linear probe
# State dict keys: head.weight, head.bias
# ═════════════════════════════════════════════════════════════════════════════

class MacRegressorLinear(nn.Module):
    """Frozen Virchow2 CLS → Linear(1280 → 2)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(1280, 2)

    def forward(self, x):
        return self.head(_cls_token(self.backbone, x))


# ═════════════════════════════════════════════════════════════════════════════
# Model 2: Virchow2 MLP
# State dict keys: net.{0,1,4,5,8}.* (Linear → BN → GELU → Dropout × 2 → Linear)
# ═════════════════════════════════════════════════════════════════════════════

class MacRegressorMLP(nn.Module):
    """Frozen Virchow2 CLS → 2-layer MLP(1280→512→256→2)."""

    def __init__(self, backbone, fc1_dim: int = 512, head_dim: int = 256,
                 dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.net = nn.Sequential(
            nn.Linear(1280, fc1_dim),       # net.0
            nn.BatchNorm1d(fc1_dim),         # net.1
            nn.GELU(),                       # net.2
            nn.Dropout(dropout),             # net.3
            nn.Linear(fc1_dim, head_dim),    # net.4
            nn.BatchNorm1d(head_dim),        # net.5
            nn.GELU(),                       # net.6
            nn.Dropout(dropout),             # net.7
            nn.Linear(head_dim, 2),          # net.8
        )

    def forward(self, x):
        return self.net(_cls_token(self.backbone, x))


# ═════════════════════════════════════════════════════════════════════════════
# Model 3: Domain-FiLM
# State dict keys: fc1, bn1, film1.{gamma,beta}, fc2, bn2, film2.{gamma,beta},
#                  regressor  (+ drop1/drop2 — no params)
# ═════════════════════════════════════════════════════════════════════════════

class MacRegressorFiLM(nn.Module):
    """Frozen Virchow2 CLS → shared MLP with FiLM domain conditioning.

    Requires domain_id at inference (0 or 1). Defaults to 1 in YAML.
    """

    def __init__(self, backbone, num_domains: int = 2, fc1_dim: int = 512,
                 head_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fc1       = nn.Linear(1280, fc1_dim)
        self.bn1       = nn.BatchNorm1d(fc1_dim)
        self.drop1     = nn.Dropout(dropout)
        self.film1     = _FiLM(fc1_dim, num_domains)
        self.fc2       = nn.Linear(fc1_dim, head_dim)
        self.bn2       = nn.BatchNorm1d(head_dim)
        self.drop2     = nn.Dropout(dropout)
        self.film2     = _FiLM(head_dim, num_domains)
        self.regressor = nn.Linear(head_dim, 2)

    def forward(self, x, domain_id):
        domain_ids = _to_batch_ids(domain_id, x.shape[0], x.device)
        h = _cls_token(self.backbone, x)
        h = self.film1(self.drop1(torch.relu(self.bn1(self.fc1(h)))), domain_ids)
        h = self.film2(self.drop2(torch.relu(self.bn2(self.fc2(h)))), domain_ids)
        return self.regressor(h)


# ═════════════════════════════════════════════════════════════════════════════
# Model 4: Domain-Specific
# State dict keys: domain_fc1.{0,1}.*, domain_bn1.bns.{0,1}.*,
#                  domain_fc1b.{0,1}.*, domain_bn1b.bns.{0,1}.*,
#                  head_fc.*, head_bn.bns.{0,1}.*, regressor.*
# ═════════════════════════════════════════════════════════════════════════════

class MacRegressorDomainSpecific(nn.Module):
    """Frozen Virchow2 CLS → separate MLP branches per domain.

    Requires domain_id at inference (0 or 1). Defaults to 1 in YAML.
    """

    def __init__(self, backbone, num_domains: int = 2, fc1_dim: int = 512,
                 head_dim: int = 256, base_dropout: float = 0.3,
                 head_dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.domain_fc1  = nn.ModuleList([nn.Linear(1280, fc1_dim)   for _ in range(num_domains)])
        self.domain_bn1  = _DomainBN(fc1_dim, num_domains)
        self.domain_fc1b = nn.ModuleList([nn.Linear(fc1_dim, fc1_dim) for _ in range(num_domains)])
        self.domain_bn1b = _DomainBN(fc1_dim, num_domains)
        self.head_fc     = nn.Linear(fc1_dim, head_dim)
        self.head_bn     = _DomainBN(head_dim, num_domains)
        self.regressor   = nn.Linear(head_dim, 2)

        self._base_dropout = base_dropout
        self._head_dropout = head_dropout

    def forward(self, x, domain_id):
        d = _to_scalar_id(domain_id)
        h = _cls_token(self.backbone, x)

        h = self.domain_fc1[d](h)
        h = self.domain_bn1(h, d)
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.dropout(h, p=self._base_dropout, training=self.training)

        h = self.domain_fc1b[d](h)
        h = self.domain_bn1b(h, d)
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.dropout(h, p=self._base_dropout, training=self.training)

        h = self.head_fc(h)
        h = self.head_bn(h, d)
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.dropout(h, p=self._head_dropout, training=self.training)

        return self.regressor(h)


# ═════════════════════════════════════════════════════════════════════════════
# Model 5: ResNet-50 linear probe
# State dict keys: head.weight, head.bias
# ═════════════════════════════════════════════════════════════════════════════

class MacRegressorResNet(nn.Module):
    """ResNet-50 global-avg-pool features → Linear(2048 → 2).

    Backbone is frozen pretrained ResNet-50 (timm, num_classes=0).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Utility functions ─────────────────────────────────────────────────────────

def _to_batch_ids(domain_id, batch_size: int, device) -> torch.Tensor:
    """Ensure domain_id is a [B] long tensor on the correct device."""
    if isinstance(domain_id, torch.Tensor):
        ids = domain_id.long().to(device)
        if ids.dim() == 0:
            ids = ids.expand(batch_size)
    else:
        ids = torch.full((batch_size,), int(domain_id), dtype=torch.long, device=device)
    return ids


def _to_scalar_id(domain_id) -> int:
    """Extract a single integer domain index (all samples must share one domain)."""
    if isinstance(domain_id, torch.Tensor):
        return int(domain_id.flatten()[0].item())
    return int(domain_id)
