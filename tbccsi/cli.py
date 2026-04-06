# tbccsi/cli.py

from pathlib import Path
from typing import List, Optional
import typer

from . import tbccsi_main as tbccsi_
from .wsi_tiler import WSITiler
from .model_inference import InferenceEngine
from .model_config import load_config_for_model_class

# Registry of short model names → model_class strings.
# This lets users say --model-name daft_macrophage instead of passing a config path.
# Add new models here as you create them.
MODEL_REGISTRY = {
    "virchow2_multihead_v2":  "tbccsi.models.model_virchow2_v2.Virchow2MultiHeadModel",
    "virchow2_multihead_v3": "tbccsi.models.model_virchow2_v3.Virchow2MultiHeadModel",
    "daft_macrophage":     "tbccsi.models.daft_macrophage.DAFT_Virchow2_Macrophage",
    # Macrophage count regressors (dataset9, full-dataset trained)
    "mac_linear":          "tbccsi.models.mac_regressors.MacRegressorLinear",
    "mac_mlp":             "tbccsi.models.mac_regressors.MacRegressorMLP",
    "mac_film":            "tbccsi.models.mac_regressors.MacRegressorFiLM",
    "mac_domain_specific": "tbccsi.models.mac_regressors.MacRegressorDomainSpecific",
    "mac_resnet":          "tbccsi.models.mac_regressors.MacRegressorResNet",
}


def _resolve_model_config(model_config, model_name):
    """
    Resolve the model config from either --model-config or --model-name.

    Priority:
      1. --model-config (explicit path)
      2. --model-name (looks up package-bundled config)
      3. None (auto-detect from weights dir, or legacy fallback)
    """
    if model_config is not None:
        return model_config

    if model_name is not None:
        model_class = MODEL_REGISTRY.get(model_name)
        if model_class is None:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise typer.BadParameter(
                f"Unknown model name '{model_name}'. Available: {available}"
            )
        config = load_config_for_model_class(model_class)
        if config is None:
            raise typer.BadParameter(
                f"No model_config.yaml found in package for '{model_name}'. "
                f"Looked near module: {model_class}"
            )
        return config

    return None

app = typer.Typer(
    help="A CLI for tile-based classification.",
    add_completion=False
)


@app.command()
def pred(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Optional[Path] = typer.Option(None, "--input-slide", help="Path to the H&E slide. Required for inference and heatmap thumbnail."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Optional[Path] = typer.Option(None, "--tile-file", help="Path to the common tiling file. Required for inference."),
        model_path: Optional[Path] = typer.Option(None, "-m", "--model-path", help="Path to the trained model weights. Required for inference."),
        model_config: Optional[Path] = typer.Option(
            None, "-c", "--model-config",
            help="Path to model_config.yaml. If omitted, looks in the same dir as --model-path."
        ),
        model_name: Optional[str] = typer.Option(
            None, "-n", "--model-name",
            help="Short model name (e.g. 'daft_macrophage'). "
                 "Resolves config from the tbccsi/models/ package. "
                 "Alternative to --model-config."
        ),
        batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for model inference."),
        do_inference: bool = typer.Option(False, "--do-inference", help="Run model inference?"),
        do_tta: bool = typer.Option(False, "--do-tta", help="Run test-time augmentation?"),
        do_plot: str = typer.Option("None", "--do-plot", help="Select column name to plot, or 'bivariate' for M1/M2 two-colour map."),
        pred_file: Optional[Path] = typer.Option(
            None, "--pred-file",
            help="Path to an existing predictions CSV to use for plotting. "
                 "If omitted, auto-detects {sample_id}_*_preds.csv in --work-dir."
        ),
        domain_id: Optional[int] = typer.Option(
            None, "--domain-id",
            help="Domain ID for domain-aware models (e.g. DAFT). Overrides config default."
        ),
):
    """
    Run the WSI prediction pipeline.

    Examples:
        # Legacy (auto-detects Virchow2MultiHeadModel if no config found):
        tbccsi pred --sample-id S1 --input-slide slide.svs --work-dir ./out \\
            --tile-file tiles.csv -m model.pth --do-inference

        # Config-driven (any model):
        tbccsi pred --sample-id S1 --input-slide slide.svs --work-dir ./out \\
            --tile-file tiles.csv -m ./my_model/best.pth -c ./my_model/model_config.yaml \\
            --do-inference

        # DAFT model with domain override:
        tbccsi pred ... -m daft_best.pth -c daft_config.yaml --domain-id 1 --do-inference
    """
    typer.echo(f"Running inference for sample: {sample_id}")

    if do_inference and (input_slide is None or tile_file is None or model_path is None):
        typer.echo("Error: --input-slide, --tile-file, and -m are required when --do-inference is set.", err=True)
        raise typer.Exit(1)

    if do_plot != "None" and input_slide is None and pred_file is None:
        typer.echo("Error: --input-slide is required for heatmap thumbnail when --pred-file is not set.", err=True)
        raise typer.Exit(1)

    # Build forward kwargs from CLI flags
    forward_kwargs = {}
    if domain_id is not None:
        forward_kwargs['domain_id'] = domain_id

    resolved_config = _resolve_model_config(model_config, model_name)

    tbccsi_.run_pred(
        sample_id=sample_id,
        input_slide=input_slide,
        work_dir=work_dir,
        tile_file=tile_file,
        model_path=model_path,
        model_config=resolved_config,
        batch_size=batch_size,
        do_inference=do_inference,
        do_tta=do_tta,
        do_plot=do_plot,
        pred_file=pred_file,
        forward_kwargs=forward_kwargs,
    )
    typer.echo("✅ Pipeline finished successfully.")


@app.command()
def tile(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
        save_tiles: bool = typer.Option(False, "--save-tiles", help="Save tiles to disk?"),
        fft_cutoff: float = typer.Option(0.3, "--fft-cutoff", help="FFT high-frequency energy cutoff fraction (0–1). Default: 0.3."),
        plot: bool = typer.Option(False, "--plot", help="Save a mean-color tile heatmap PNG after tiling.")
):
    """
    Generate the tile coordinate file.
    """
    output_dir = work_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_file_path = output_dir / tile_file

    typer.echo(f"Generating tiling for sample: {sample_id}")

    tiler = WSITiler(sample_id, input_slide, output_dir, tile_file_path)
    tiler.create_tile_file(save_tiles=save_tiles, fft_cutoff=fft_cutoff)

    if plot:
        from .wsi_plot import WSIPlotter
        out_png = output_dir / f"{sample_id}_tile_qc.png"
        plotter = WSIPlotter()
        plotter.tile_qc_heatmap(tile_file_path, out_png)
        typer.echo(f"QC plot saved to {out_png}")

    typer.echo("✅ Tiling finished successfully.")


@app.command()
def call(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        pred_file: Path = typer.Option(..., "--pred-file", help="Prediction output from tbccsi pred."),
        thresh_file: Path = typer.Option(..., "--thresh-file", help="Thresholds file."),
        out_file: Path = typer.Option(..., "--out-file", help="Output file name")
):
    """
    Run the cell calling algo.
    """
    typer.echo(f"Calling cell types: {sample_id}")

    engine = InferenceEngine()  # lightweight, no model loaded

    df = engine.apply_cellcalling_thresholds(
        predictions_df=str(pred_file),
        thresholds_csv=str(thresh_file)
    )

    output_path = work_dir / out_file
    df.to_csv(output_path, index=False)

    typer.echo(f"✅ Cell calling finished. Saved to {output_path}")


@app.command()
def embed(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
        model_path: Path = typer.Option(..., "-m", "--model-path", help="Path to the trained model."),
        model_config: Optional[Path] = typer.Option(
            None, "-c", "--model-config",
            help="Path to model_config.yaml."
        ),
        model_name: Optional[str] = typer.Option(
            None, "-n", "--model-name",
            help="Short model name (e.g. 'daft_macrophage'). "
                 "Resolves config from the tbccsi/models/ package."
        ),
        batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for model inference."),
        do_tta: bool = typer.Option(False, "--do-tta", help="Run test-time augmentation?"),
        latent_types: List[str] = typer.Option(
            None,
            "--latent-type",
            help="Latent types to extract. Can be specified multiple times. "
                 "If not specified, extracts all types from the model."
        ),
        save_format: str = typer.Option(
            "npz",
            "--save-format",
            help="Output format: 'npz' (compressed numpy arrays) or 'csv' (flattened dataframe)."
        ),
        domain_id: Optional[int] = typer.Option(
            None, "--domain-id",
            help="Domain ID for domain-aware models."
        ),
):
    """
    Extract latent embeddings from WSI tiles.
    """
    typer.echo(f"Extracting embeddings for sample: {sample_id}")

    if save_format not in ['npz', 'csv']:
        typer.echo("Error: save_format must be 'npz' or 'csv'", err=True)
        raise typer.Exit(1)

    forward_kwargs = {}
    if domain_id is not None:
        forward_kwargs['domain_id'] = domain_id

    resolved_config = _resolve_model_config(model_config, model_name)

    tbccsi_.run_embed(
        sample_id=sample_id,
        input_slide=input_slide,
        work_dir=work_dir,
        tile_file=tile_file,
        model_path=model_path,
        model_config=resolved_config,
        batch_size=batch_size,
        do_tta=do_tta,
        latent_types=latent_types,
        save_format=save_format,
        forward_kwargs=forward_kwargs,
    )

    typer.echo("✅ Embedding extraction finished successfully.")


if __name__ == "__main__":
    app()
