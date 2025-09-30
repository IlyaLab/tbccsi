# tbccsi/cli.py

from pathlib import Path
from typing import List
import typer

# Note the relative import '.' for a module in the same package
from . import tbccsi_main as tbccsi_

# 1. Create the Typer "app" object that pyproject.toml is looking for
app = typer.Typer(
    help="A CLI for tile-based classification on cell-segmented images.",
    add_completion=False
)

# 2. Use the @app.command() decorator instead of a main() function
@app.command()
def run(
    # Typer uses type hints to define arguments. It's much cleaner!
    sample_id: str = typer.Option(..., "--sample-id", help="Sample ID (string)."),
    input_slide: Path = typer.Option(..., "--input-slide", help="Path to the H&E slide."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save the output."),
    tile_file: Path = typer.Option(..., "--tile-file", help="Path to the common tiling file."),
    models: List[Path] = typer.Option(..., "-m", "--models", help="One or more paths to the trained models."),
    prefixes: List[str] = typer.Option(..., "-p", "--prefixes", help="Column header labels for each model."),
    batch_size: int = typer.Option(32, "-b", "--batch-size", help="Batch size for model inference."),
    use_segmented_tiles: bool = typer.Option(False, "--use-segmented-tiles", help="Use pre-segmented tiles if available."),
    save_segmented_tiles: bool = typer.Option(False, "--save-segmented-tiles", help="If set, save the extracted and segmented tiles to disk."),
    save_h_and_e_tiles: bool = typer.Option(False, "--save-h-and-e-tiles", help="If set, save the extracted H&E tiles to disk.")
):
    """
    Run the WSI prediction pipeline.
    """
    if len(models) != len(prefixes):
        # Typer has nicer error handling
        raise typer.BadParameter("The number of --models must equal the number of --prefixes.")

    # 3. The rest of your logic stays the same
    typer.echo(f"Running inference for sample: {sample_id}")
    tbccsi_.run_preds(
        sample_id=sample_id,
        input_slide=input_slide,
        output_dir=output_dir,
        tile_file=tile_file,
        model_list=models,
        prefix_list=prefixes,
        batch_size=batch_size,
        use_segmented=use_segmented_tiles,
        save_segmented_tiles=save_segmented_tiles,
        save_h_and_e_tiles=save_h_and_e_tiles
    )
    typer.echo("✅ Pipeline finished successfully.")


# This part is only for making the script runnable with "python cli.py"
# It's not strictly necessary for the installed CLI to work.
if __name__ == "__main__":
    app()
