# tbccsi/cli.py

from pathlib import Path
from typing import List
import typer

# Note the relative import '.' for a module in the same package
from . import tbccsi_main as tbccsi_
from .wsi_tiler import WSITiler
from .model_inference import VirchowInferenceEngine

# 1. Create the Typer "app" object that pyproject.toml is looking for
app = typer.Typer(
    help="A CLI for tile-based classification.",
    add_completion=False
)


# 2. Use the @app.command() decorator instead of a main() function
@app.command()
def pred(
        # Typer uses type hints to define arguments. It's much cleaner!
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
        model_path: Path = typer.Option(..., "-m", "--model-path", help="One or more paths to the trained models."),
        batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for model inference."),
        do_inference: bool = typer.Option(False, "--do-inference", help="Run model inference?"),
        do_tta: bool = typer.Option(False, "--do-tta", help="Run test-time augmentation?"),
        do_plot: str = typer.Option("None", "--do-plot", help="Select column name to plot.")
):
    """
    Run the WSI prediction pipeline.
    """
    typer.echo(f"Running inference for sample: {sample_id}")
    tbccsi_.run_virchow_pred(
        sample_id=sample_id,
        input_slide=input_slide,
        work_dir=work_dir,
        tile_file=tile_file,
        model_path=model_path,
        batch_size=batch_size,
        do_inference=do_inference,
        do_tta=do_tta,
        do_plot=do_plot
    )
    typer.echo("✅ Pipeline finished successfully.")


@app.command()
def tile(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
        save_tiles: bool = typer.Option(False, "--save-tiles", help="Save tiles to disk?")
):
    """
    Generate the tile coordinate file.
    """
    output_dir = work_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_file_path = output_dir / tile_file

    typer.echo(f"Generating tiling for sample: {sample_id}")

    tiler = WSITiler(sample_id, input_slide, output_dir, tile_file_path)
    tiler.create_tile_file(save_tiles=save_tiles)

    typer.echo("✅ Tiling finished successfully.")


# 3. calling cells
@app.command()
def call(
        # Typer uses type hints to define arguments. It's much cleaner!
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

    vie = VirchowInferenceEngine()

    df = vie.apply_cellcalling_thresholds(
        predictions_df=str(pred_file),
        thresholds_csv=str(thresh_file)
    )

    output_path = work_dir / out_file
    df.to_csv(output_path, index=False)

    typer.echo(f"✅ Cell calling finished. Saved to {output_path}")


# 4. Extract embeddings
@app.command()
def embed(
        sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
        input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
        work_dir: Path = typer.Option(..., "--work-dir", help="Directory to hold saved tiles and the output."),
        tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
        model_path: Path = typer.Option(..., "-m", "--model-path", help="Path to the trained model."),
        batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for model inference."),
        do_tta: bool = typer.Option(False, "--do-tta", help="Run test-time augmentation?"),
        latent_types: List[str] = typer.Option(
            None,
            "--latent-type",
            help="Latent types to extract. Options: backbone, structure, immune_shared, immune_tcell, immune_mac. "
                 "Can be specified multiple times. If not specified, extracts all types."
        ),
        save_format: str = typer.Option(
            "npz",
            "--save-format",
            help="Output format: 'npz' (compressed numpy arrays) or 'csv' (flattened dataframe)."
        )
):
    """
    Extract latent embeddings from WSI tiles.

    Examples:
        # Extract all latent types as NPZ
        tbccsi embed --sample-id TCGA-01 --input-slide slide.svs --work-dir ./output --tile-file tiles.csv -m model.pth

        # Extract only structure and T-cell embeddings
        tbccsi embed --sample-id TCGA-01 --input-slide slide.svs --work-dir ./output --tile-file tiles.csv -m model.pth \\
            --latent-type structure --latent-type immune_tcell

        # Extract with TTA and save as CSV
        tbccsi embed --sample-id TCGA-01 --input-slide slide.svs --work-dir ./output --tile-file tiles.csv -m model.pth \\
            --do-tta --save-format csv
    """
    typer.echo(f"Extracting embeddings for sample: {sample_id}")

    # Validate save format
    if save_format not in ['npz', 'csv']:
        typer.echo("Error: save_format must be 'npz' or 'csv'", err=True)
        raise typer.Exit(1)

    tbccsi_.run_virchow_embed(
        sample_id=sample_id,
        input_slide=input_slide,
        work_dir=work_dir,
        tile_file=tile_file,
        model_path=model_path,
        batch_size=batch_size,
        do_tta=do_tta,
        latent_types=latent_types,
        save_format=save_format
    )

    typer.echo("✅ Embedding extraction finished successfully.")


# This part is only for making the script runnable with "python cli.py"
# It's not strictly necessary for the installed CLI to work.
if __name__ == "__main__":
    app()