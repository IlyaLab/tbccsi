"""
model_config.py — Model configuration for tbccsi.

A model_config.yaml file sits alongside the .pth/.safetensors weights and
tells the inference engine everything it needs to know:
  - which backbone to use (and how to create it via timm)
  - what the model class is (importable dotted path)
  - what the forward() signature looks like
  - how to parse outputs into named probability columns
  - preprocessing options

This replaces the hardcoded Virchow2MultiHeadModel assumption.

Example directory layout:
    my_model/
    ├── model_config.yaml    <-- this config
    ├── best_model.pth       <-- weights
    └── thresholds.csv       <-- optional cell-calling thresholds
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ══════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════

@dataclass
class BackboneConfig:
    """How to instantiate the foundation model backbone via timm."""
    model_name: str = "hf-hub:paige-ai/Virchow2"   # timm model string
    pretrained: bool = True
    freeze: bool = True
    # Extra kwargs passed to timm.create_model
    # e.g. {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}
    create_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSpec:
    """One named output value from a head (maps index → CSV column)."""
    name: str            # human name, e.g. "tumor", "m1"
    index: int           # position in the output vector
    output_col: str = "" # CSV column name (auto-derived as prob_{name} if empty)


@dataclass
class HeadConfig:
    """Describes one classification/regression head."""
    name: str                         # e.g. "structure", "immune", "macrophage"
    num_outputs: int                  # number of neurons
    activation: str = "sigmoid"       # "sigmoid" | "softmax" | "none"
    outputs: List[OutputSpec] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Top-level model configuration."""
    name: str = "unnamed_model"
    version: str = "1.0"
    description: str = ""

    # ── Backbone ──────────────────────────────────────────────
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    # ── Model class (importable dotted path) ──────────────────
    # e.g. "tbccsi.models.model_virchow2_v2.Virchow2MultiHeadModel"
    # or   "tbccsi.models.daft_macrophage.DAFT_Virchow2_Macrophage"
    model_class: str = ""

    # ── How the model class is instantiated ───────────────────
    # Keyword arguments passed to ModelClass(backbone, **init_kwargs)
    # e.g. {"num_classes": 4, "num_domains": 2}
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ── Forward signature ─────────────────────────────────────
    # Extra arguments that forward() expects beyond the image batch.
    # e.g. {"domain_id": 0} means model(x, domain_id=0) at inference.
    # Values are the defaults used when the caller doesn't supply them.
    forward_extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ── How model returns outputs ─────────────────────────────
    # "tuple"  → model(x) returns (head0_tensor, head1_tensor, ...)
    # "dict"   → model(x) returns {"head_name": tensor, ...}
    # "single" → model(x) returns a single tensor
    output_format: str = "tuple"

    # ── Classification Heads ──────────────────────────────────
    heads: List[HeadConfig] = field(default_factory=list)

    # ── Preprocessing ─────────────────────────────────────────
    tile_size: int = 224
    normalize: str = "reinhard"       # "reinhard" | "macenko" | "none"

    # ── State dict prefix to strip ────────────────────────────
    # e.g. "model." from HuggingFace Trainer checkpoints
    strip_prefix: str = ""

    # ── Weights filename (if different from convention) ───────
    # Leave empty to auto-detect .pth/.safetensors in the same dir
    weights_file: str = ""


# ══════════════════════════════════════════════════════════════
# I/O — Load / Save
# ══════════════════════════════════════════════════════════════

def _dict_to_config(d: dict) -> ModelConfig:
    """Recursively convert a nested dict into dataclass instances."""
    backbone_d = d.pop("backbone", {})
    backbone = BackboneConfig(**backbone_d)

    heads = []
    for h in d.pop("heads", []):
        outputs = [OutputSpec(**o) for o in h.pop("outputs", [])]
        heads.append(HeadConfig(**h, outputs=outputs))

    return ModelConfig(backbone=backbone, heads=heads, **d)


def _find_config_file(directory: Path) -> Optional[Path]:
    """Look for a model_config.yaml/yml/json in a directory."""
    for candidate in ["model_config.yaml", "model_config.yml", "model_config.json"]:
        p = directory / candidate
        if p.exists():
            return p
    return None


def _find_config_in_package(model_class: str) -> Optional[Path]:
    """
    Given a model_class like 'tbccsi.models.daft_macrophage.DAFT_Virchow2_Macrophage',
    look for model_config.yaml next to the .py file that defines it.

    Supports two layouts:

      Layout A — flat file:
        tbccsi/models/daft_macrophage.py
        tbccsi/models/daft_macrophage.yaml    <-- found

      Layout B — package directory:
        tbccsi/models/daft_macrophage/__init__.py
        tbccsi/models/daft_macrophage/model_config.yaml  <-- found
    """
    import importlib
    module_path = model_class.rsplit(".", 1)[0]  # e.g. "tbccsi.models.daft_macrophage"

    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None

    if not hasattr(module, '__file__') or module.__file__ is None:
        return None

    module_file = Path(module.__file__)

    # Layout B: module is a package (__init__.py inside a directory)
    if module_file.name == '__init__.py':
        config = _find_config_file(module_file.parent)
        if config:
            return config

    # Layout A: module is a .py file — look for sibling .yaml with same stem
    # e.g. daft_macrophage.py → daft_macrophage.yaml
    for ext in ('.yaml', '.yml', '.json'):
        sibling = module_file.with_suffix(ext)
        if sibling.exists():
            return sibling

    # Also check for model_config.yaml in same directory
    config = _find_config_file(module_file.parent)
    if config:
        return config

    return None


def load_model_config(path) -> ModelConfig:
    """
    Load a model config from YAML or JSON.

    Can pass:
      - path to a .yaml/.json file directly
      - path to a directory (looks for model_config.yaml inside)
      - path to a .pth/.safetensors file (looks for config in same dir)
    """
    path = Path(path)

    # If they passed a weights file, look in its parent dir
    if path.suffix in ('.pth', '.safetensors', '.pt'):
        path = path.parent

    # If directory, find the config file
    if path.is_dir():
        found = _find_config_file(path)
        if found:
            path = found
        else:
            raise FileNotFoundError(
                f"No model_config.yaml/json found in {path}. "
                "Please provide a --model-config file."
            )

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")
            d = yaml.safe_load(f)
        else:
            d = json.load(f)

    return _dict_to_config(d)


def load_config_for_model_class(model_class: str) -> Optional[ModelConfig]:
    """
    Resolve a model_config.yaml from a model_class string by looking
    next to the Python module that defines the class.

    Returns None if not found (caller can fall back to other strategies).
    """
    config_path = _find_config_in_package(model_class)
    if config_path is None:
        return None
    print(f"Found config in package: {config_path}")
    return load_model_config(config_path)


def save_model_config(config: ModelConfig, path):
    """Save config to YAML (or JSON if PyYAML not available)."""
    import dataclasses
    path = Path(path)
    d = dataclasses.asdict(config)

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml") and HAS_YAML:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(d, f, indent=2)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def get_output_columns(config: ModelConfig) -> Dict[str, List[str]]:
    """
    Returns {head_name: [col1, col2, ...]} for writing prediction CSVs.

    Example:
        {"structure": ["prob_tumor", "prob_stroma"],
         "macrophage": ["prob_neither", "prob_m1", "prob_m2", "prob_both"]}
    """
    result = {}
    for head in config.heads:
        cols = []
        for out in head.outputs:
            col = out.output_col if out.output_col else f"prob_{out.name}"
            cols.append(col)
        result[head.name] = cols
    return result


def find_weights_file(config: ModelConfig, config_path: Path) -> Path:
    """Locate the weights file relative to the config file."""
    config_dir = config_path.parent if config_path.is_file() else Path(config_path)

    # Explicit filename in config
    if config.weights_file:
        wf = config_dir / config.weights_file
        if wf.exists():
            return wf
        raise FileNotFoundError(f"Weights file specified in config not found: {wf}")

    # Auto-detect
    for ext in (".safetensors", ".pth", ".pt"):
        candidates = list(config_dir.glob(f"*{ext}"))
        if candidates:
            # Prefer "best" in the name
            best = [c for c in candidates if "best" in c.stem.lower()]
            return best[0] if best else candidates[0]

    raise FileNotFoundError(f"No .pth/.safetensors/.pt weights found in {config_dir}")
