"""
model_inference.py — Config-driven inference engine for tbccsi.

Instead of hardcoding Virchow2MultiHeadModel, this reads a ModelConfig
(YAML/JSON) that describes:
  - backbone (timm model string + kwargs)
  - model class to import and instantiate
  - forward() signature (extra kwargs like domain_id)
  - output format (tuple vs dict vs single tensor)
  - head definitions with named output columns

Backward compatible: if no config is provided, falls back to the
legacy Virchow2MultiHeadModel behavior.
"""

import torch
import importlib
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

from .model_config import (
    ModelConfig, load_model_config, load_config_for_model_class,
    get_output_columns, find_weights_file
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

# Map string names → actual classes for timm.create_model kwargs
_CLASS_MAP = {
    "SwiGLUPacked": SwiGLUPacked,
    "SiLU": torch.nn.SiLU,
    "GELU": torch.nn.GELU,
    "ReLU": torch.nn.ReLU,
}


def _resolve_timm_kwargs(create_kwargs: dict) -> dict:
    """Resolve string references like 'SwiGLUPacked' to actual classes."""
    resolved = {}
    for k, v in create_kwargs.items():
        if isinstance(v, str) and v in _CLASS_MAP:
            resolved[k] = _CLASS_MAP[v]
        else:
            resolved[k] = v
    return resolved


def _import_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ══════════════════════════════════════════════════════════════
# Reinhard Normalizer (unchanged)
# ══════════════════════════════════════════════════════════════

class ReinhardNormalizer:
    """Normalizes H&E images to a target color distribution in LAB space."""

    def __init__(self, target_means=None, target_stds=None):
        self.target_means = np.array(
            target_means or [147.92, 141.08, 122.12], dtype=np.float32
        )
        self.target_stds = np.array(
            target_stds or [53.17, 4.72, 5.23], dtype=np.float32
        )

    def normalize(self, pil_img):
        img_np = np.array(pil_img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        tile_means = np.mean(img_lab, axis=(0, 1))
        tile_stds = np.std(img_lab, axis=(0, 1))
        tile_stds[tile_stds == 0] = 1e-5
        img_norm = ((img_lab - tile_means) / tile_stds) * self.target_stds + self.target_means
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img_norm, cv2.COLOR_LAB2RGB))


# ══════════════════════════════════════════════════════════════
# Model Loader
# ══════════════════════════════════════════════════════════════

def load_model_from_config(
    config: ModelConfig,
    weights_path: Path,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Create backbone, instantiate model class, load weights, return (model, processor).

    Works for any model class that follows the convention:
        ModelClass(backbone, **init_kwargs)
    """
    weights_path = Path(weights_path)

    # 1. Create backbone via timm
    timm_kwargs = _resolve_timm_kwargs(config.backbone.create_kwargs)
    print(f"Initializing backbone: {config.backbone.model_name}")
    backbone = timm.create_model(
        config.backbone.model_name,
        pretrained=config.backbone.pretrained,
        **timm_kwargs,
    )

    # Build the image transform from the backbone's pretrained config
    data_cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
    processor = create_transform(**data_cfg)

    # 2. Instantiate the model class
    if not config.model_class:
        raise ValueError(
            "model_class must be set in model_config.yaml "
            "(e.g. 'tbccsi.models.model_virchow2_v2.Virchow2MultiHeadModel')"
        )

    print(f"Loading model class: {config.model_class}")
    ModelClass = _import_class(config.model_class)

    # Detect the first positional arg name for the backbone
    # Most models use: __init__(self, hf_model=...) or __init__(self, virchow_backbone=...)
    import inspect
    sig = inspect.signature(ModelClass.__init__)
    params = list(sig.parameters.keys())
    # params[0] is 'self', params[1] is the backbone arg
    backbone_arg_name = params[1] if len(params) > 1 else 'hf_model'

    init_kwargs = {backbone_arg_name: backbone}
    init_kwargs.update(config.init_kwargs)

    model = ModelClass(**init_kwargs)

    # 3. Load weights
    print(f"Loading weights: {weights_path}")
    if str(weights_path).endswith(".safetensors"):
        state_dict = load_file(weights_path, device="cpu")
    else:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Handle checkpoints that wrap state_dict in a dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  (extracted model_state_dict from checkpoint, "
                  f"epoch={checkpoint.get('epoch', '?')}, "
                  f"phase={checkpoint.get('phase', '?')})")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

    # Strip key prefix if needed (e.g. "model." from HF Trainer)
    if config.strip_prefix:
        state_dict = {
            (k[len(config.strip_prefix):] if k.startswith(config.strip_prefix) else k): v
            for k, v in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device} ({config.name} v{config.version})")

    return model, processor


# ══════════════════════════════════════════════════════════════
# Inference Engine — config-driven, backward-compatible
# ══════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Config-driven inference engine.

    Replaces VirchowInferenceEngine. Reads a model_config.yaml to know:
      - how to build the model
      - what forward() expects
      - how to parse outputs into named columns

    Usage:
        # New style — any model
        engine = InferenceEngine(
            model_path="/path/to/weights.pth",
            model_config="/path/to/model_config.yaml"
        )

        # Legacy style — falls back to Virchow2MultiHeadModel
        engine = InferenceEngine(model_path="/path/to/weights.pth")
    """

    def __init__(
        self,
        model_path=None,
        model_config=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = device
        self.model = None
        self.processor = None
        self.config = None
        self._output_columns = None

        # Lightweight init (for cell-calling without a model)
        if model_path is None and model_config is None:
            print("Initialized InferenceEngine without loading a model.")
            return

        if isinstance(model_path, list):
            model_path = model_path[0]
        model_path = Path(model_path) if model_path else None

        # ── Resolve config ────────────────────────────────────
        if model_config is not None:
            # Explicit config path
            if isinstance(model_config, (str, Path)):
                self.config = load_model_config(model_config)
            else:
                self.config = model_config  # Already a ModelConfig
        elif model_path is not None:
            # Try to find config next to weights
            try:
                self.config = load_model_config(model_path.parent)
                print(f"Found model config: {self.config.name}")
            except FileNotFoundError:
                # No config next to weights → legacy mode
                self.config = None

        # ── Load model ────────────────────────────────────────
        if self.config is not None:
            # Config-driven loading
            if model_path is None:
                # Find weights from config dir
                # Assume config was loaded from a path
                model_path = find_weights_file(self.config, Path(model_config) if model_config else Path('.'))

            self.model, self.processor = load_model_from_config(
                self.config, model_path, device
            )
            self._output_columns = get_output_columns(self.config)
        else:
            # ── LEGACY fallback: hardcoded Virchow2MultiHeadModel ──
            print("⚠️  No model_config.yaml found — using legacy Virchow2MultiHeadModel")
            self._init_legacy(model_path)

    # ──────────────────────────────────────────────────────────
    # Legacy init (preserves old behavior exactly)
    # ──────────────────────────────────────────────────────────

    def _init_legacy(self, model_path):
        """Original VirchowInferenceEngine init — hardcoded model."""
        from .models.model_virchow2_v2 import Virchow2MultiHeadModel

        self.model_path = Path(model_path)

        print("Initializing Virchow2 Backbone (legacy)...")
        backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        data_cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
        self.processor = create_transform(**data_cfg)

        self.model = Virchow2MultiHeadModel(hf_model=backbone, freeze_backbone=True)

        if str(self.model_path).endswith(".safetensors"):
            state_dict = load_file(self.model_path, device="cpu")
        else:
            state_dict = torch.load(self.model_path, map_location="cpu")

        # Strip "model." prefix
        state_dict = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys: {missing[:5]} ...")

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device} (legacy mode).")

    # ──────────────────────────────────────────────────────────
    # TTA (shared)
    # ──────────────────────────────────────────────────────────

    def _get_tta_variants(self, img):
        """8 dihedral variants: 4 rotations × {original, flipped}."""
        variants = [img]
        for angle in [90, 180, 270]:
            variants.append(img.rotate(angle, expand=True))
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        variants.append(img_flip)
        for angle in [90, 180, 270]:
            variants.append(img_flip.rotate(angle, expand=True))
        return variants

    # ──────────────────────────────────────────────────────────
    # Output unpacking
    # ──────────────────────────────────────────────────────────

    def _unpack_model_output(self, raw_output):
        """
        Normalize model output into {head_name: tensor} regardless of
        whether the model returns a tuple, dict, or single tensor.
        """
        if self.config is None:
            # Legacy: model returns (out_structure, out_immune)
            return {"structure": raw_output[0], "immune": raw_output[1]}

        fmt = self.config.output_format

        if fmt == "dict":
            return raw_output

        elif fmt == "tuple":
            result = {}
            for i, head_cfg in enumerate(self.config.heads):
                if i < len(raw_output):
                    result[head_cfg.name] = raw_output[i]
            return result

        elif fmt == "single":
            # Single tensor → assign to the first (only) head
            head_name = self.config.heads[0].name if self.config.heads else "output"
            return {head_name: raw_output}

        else:
            raise ValueError(f"Unknown output_format: {fmt}")

    def _build_forward_kwargs(self, extra_kwargs=None):
        """
        Build the extra keyword args for model.forward() from config defaults,
        optionally overridden by caller-supplied values.
        """
        if self.config is None:
            return {}

        kwargs = dict(self.config.forward_extra_kwargs)
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # Convert values to tensors on device where needed
        result = {}
        for k, v in kwargs.items():
            if isinstance(v, int):
                result[k] = torch.tensor(v, device=self.device)
            else:
                result[k] = v
        return result

    # ──────────────────────────────────────────────────────────
    # Prediction — config-driven
    # ──────────────────────────────────────────────────────────

    def predict_batch(self, images, metadata_list, use_tta=False, **forward_kwargs):
        """
        Run inference on a list of PIL images.

        Output columns are determined by config (or hardcoded for legacy).

        Args:
            images: List of PIL images.
            metadata_list: List of dicts with tile metadata.
            use_tta: Apply 8× dihedral TTA.
            **forward_kwargs: Override config's forward_extra_kwargs
                              (e.g. domain_id=1 for DAFT models).

        Returns:
            List[dict] — one per image, with metadata + probability columns.
        """
        if not images:
            return []

        if self.config is None:
            return self._predict_batch_legacy(images, metadata_list, use_tta)

        extra_kwargs = self._build_forward_kwargs(forward_kwargs)
        results = []

        for idx, img in enumerate(images):
            meta = metadata_list[idx].copy()

            try:
                # Prepare input
                if use_tta:
                    tensors = [self.processor(v) for v in self._get_tta_variants(img)]
                    input_batch = torch.stack(tensors).to(self.device)
                else:
                    input_batch = self.processor(img).unsqueeze(0).to(self.device)

                # Expand extra kwargs to match batch size if needed
                batch_kwargs = {}
                for k, v in extra_kwargs.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 0:
                        batch_kwargs[k] = v.expand(input_batch.shape[0])
                    else:
                        batch_kwargs[k] = v

                with torch.no_grad():
                    raw_output = self.model(input_batch, **batch_kwargs)

                head_outputs = self._unpack_model_output(raw_output)

                # Parse each head's output
                for head_cfg in self.config.heads:
                    if head_cfg.name not in head_outputs:
                        continue

                    logits = head_outputs[head_cfg.name]  # [N, num_outputs]

                    # Apply activation
                    if head_cfg.activation == "sigmoid":
                        probs = torch.sigmoid(logits)
                    elif head_cfg.activation == "softmax":
                        probs = torch.softmax(logits, dim=-1)
                    else:
                        probs = logits

                    # Average across TTA variants (or squeeze batch dim of 1)
                    probs = torch.mean(probs, dim=0).cpu().numpy()

                    # Write to metadata using config-defined column names
                    for out_spec in head_cfg.outputs:
                        col = out_spec.output_col or f"prob_{out_spec.name}"
                        meta[col] = float(probs[out_spec.index])

                # Derive a class label from the dominant prediction
                self._add_predicted_label(meta)

                if use_tta:
                    meta['tta_enabled'] = True

                results.append(meta)

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        return results

    def _add_predicted_label(self, meta):
        """
        Add a 'pred_class_label' column.

        For softmax heads: argmax → the output name.
        For sigmoid heads: threshold each independently.
        """
        if self.config is None:
            return

        for head_cfg in self.config.heads:
            if head_cfg.activation == "softmax":
                # Find the output with the highest probability
                best_name = None
                best_prob = -1.0
                for out_spec in head_cfg.outputs:
                    col = out_spec.output_col or f"prob_{out_spec.name}"
                    p = meta.get(col, 0.0)
                    if p > best_prob:
                        best_prob = p
                        best_name = out_spec.name
                meta[f'pred_{head_cfg.name}'] = best_name

            elif head_cfg.activation == "sigmoid":
                # Flag each output independently at 0.5
                labels = []
                for out_spec in head_cfg.outputs:
                    col = out_spec.output_col or f"prob_{out_spec.name}"
                    if meta.get(col, 0.0) > 0.5:
                        labels.append(out_spec.name)
                meta[f'pred_{head_cfg.name}'] = ";".join(labels) if labels else "none"

    # ──────────────────────────────────────────────────────────
    # Legacy predict_batch (exact old behavior)
    # ──────────────────────────────────────────────────────────

    def _predict_batch_legacy(self, images, metadata_list, use_tta=False):
        """Original VirchowInferenceEngine.predict_batch — hardcoded columns."""
        results = []

        for idx, img in enumerate(images):
            meta = metadata_list[idx].copy()

            try:
                if use_tta:
                    tensors = [self.processor(v) for v in self._get_tta_variants(img)]
                    input_batch = torch.stack(tensors).to(self.device)
                else:
                    input_batch = self.processor(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out_structure, out_immune = self.model(input_batch)
                    struct_probs = torch.mean(torch.sigmoid(out_structure), dim=0).cpu().numpy()
                    immune_vals = torch.mean(torch.sigmoid(out_immune), dim=0).cpu().numpy()

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
                continue

        return results

    # ──────────────────────────────────────────────────────────
    # Embedding extraction (works with any model that has get_latents)
    # ──────────────────────────────────────────────────────────

    def extract_embeddings_batch(self, images, metadata_list, use_tta=False,
                                 latent_types=None, **forward_kwargs):
        """Extract latent embeddings from a list of PIL images."""
        if not images:
            return []

        # Default latent types from config heads + backbone
        if latent_types is None:
            if self.config:
                latent_types = ['backbone'] + [h.name for h in self.config.heads]
            else:
                latent_types = ['backbone', 'structure', 'immune_shared',
                                'immune_tcell', 'immune_mac']

        extra_kwargs = self._build_forward_kwargs(forward_kwargs)
        results = []

        for idx, img in enumerate(images):
            meta = metadata_list[idx].copy()

            try:
                if use_tta:
                    tensors = [self.processor(v) for v in self._get_tta_variants(img)]
                    input_batch = torch.stack(tensors).to(self.device)
                else:
                    input_batch = self.processor(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if hasattr(self.model, 'get_latents'):
                        latents_dict = self.model.get_latents(input_batch, **extra_kwargs)
                    else:
                        # Fallback: just get backbone embedding
                        latents_dict = {"backbone": self.model.backbone(input_batch)}

                    for latent_name in latent_types:
                        if latent_name in latents_dict:
                            lt = latents_dict[latent_name]
                            if use_tta:
                                lt = torch.mean(lt, dim=0).cpu().numpy()
                            else:
                                lt = lt.squeeze(0).cpu().numpy()
                            meta[f'embedding_{latent_name}'] = lt

                if use_tta:
                    meta['tta_enabled'] = True
                results.append(meta)

            except Exception as e:
                print(f"Error extracting embeddings for image {idx}: {e}")
                continue

        return results

    # ──────────────────────────────────────────────────────────
    # Cell calling (unchanged — works on any prediction CSV)
    # ──────────────────────────────────────────────────────────

    def apply_cellcalling_thresholds(self, predictions_df, thresholds_csv):
        """
        Apply thresholds for multi-label tile calling.
        This is output-agnostic — works with whatever prob_* columns exist.
        """
        print("Loading thresholds for multi-label tile calling...")

        if isinstance(predictions_df, str):
            predictions_df = pd.read_csv(predictions_df)

        thresh_df = pd.read_csv(thresholds_csv)

        thresh_lookup = {}
        for cell in thresh_df['cell'].unique():
            thresh_lookup[cell] = {}
            sub = thresh_df[thresh_df['cell'] == cell]
            for _, row in sub.iterrows():
                thresh_lookup[cell][row['context']] = row['threshold']

        # Structure context (if structure columns exist)
        if 'prob_tumor' in predictions_df.columns and 'prob_stroma' in predictions_df.columns:
            t_thresh = thresh_lookup.get('prob_tumor', {}).get('Global', 0.90)
            s_thresh = thresh_lookup.get('prob_stroma', {}).get('Global', 0.90)

            print(f'Using {t_thresh} and {s_thresh} structural thresholds')

            is_tumor_ctx = ((predictions_df['prob_tumor'] > predictions_df['prob_stroma']) &
                            (predictions_df['prob_tumor'] > t_thresh))
            is_stroma_ctx = ((predictions_df['prob_stroma'] > predictions_df['prob_tumor']) &
                             (predictions_df['prob_stroma'] > s_thresh))

            context_series = pd.Series('Ambiguous_Region', index=predictions_df.index)
            context_series[is_tumor_ctx] = 'Tumor_Region'
            context_series[is_stroma_ctx] = 'Stroma_Region'
            predictions_df['Threshold_Context'] = context_series
        else:
            context_series = pd.Series('Global', index=predictions_df.index)

        for cell in thresh_lookup:
            prob_col = cell
            flag_col = cell.replace('prob_', 'called_')

            if prob_col not in predictions_df.columns:
                print(f'{prob_col} not in pred columns...')
                continue

            predictions_df[flag_col] = False

            if 'Global' in thresh_lookup[cell]:
                limit = thresh_lookup[cell]['Global']
                predictions_df.loc[predictions_df[prob_col] > limit, flag_col] = True
            else:
                for ctx in ['Tumor_Region', 'Stroma_Region', 'Ambiguous_Region']:
                    limit = thresh_lookup[cell].get(ctx, 0.99)
                    ctx_mask = (context_series == ctx)
                    pass_mask = (predictions_df.loc[ctx_mask, prob_col] > limit)
                    predictions_df.loc[pass_mask.index[pass_mask], flag_col] = True

        return predictions_df


# ══════════════════════════════════════════════════════════════
# Backward-compatible alias
# ══════════════════════════════════════════════════════════════

class VirchowInferenceEngine(InferenceEngine):
    """
    Backward-compatible alias.

    Existing code that does:
        engine = VirchowInferenceEngine(model_path)
    will continue to work (legacy mode, hardcoded Virchow2MultiHeadModel).

    New code should use:
        engine = InferenceEngine(model_path, model_config="path/to/config.yaml")
    """
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model_path=model_path, model_config=None, device=device)
