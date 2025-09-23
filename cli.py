
import argparse
from pathlib import Path

# tile based classification on cell segmented images
import tbccsi.tbccsi_main as tbccsi_
from tbccsi.vit_model_5_6 import VitClassification
from tbccsi.wsi_tiler import WSITiler
from tbccsi.wsi_segmentation import CellSegmentationProcessor
from tbccsi.model_inference import WSIInferenceEngine
from tbccsi.wsi_plot import WSIPlotter


def main():
    """Main function to parse command-line arguments."""
    # Define the example usage string
    example_text = """
Example Usage:
--------------
python run_inference.py \\
  --sample     TCGA-AB-1234 \\
  --slide_path /path/to/slides/TCGA-AB-1234.svs \\
  --tile_file  /path/to/data/common_tile_file.csv \\
  --output_dir /path/to/output/TCGA-AB-1234_results \\
  --models     /path/to/models/tumor_model.pth /path/to/models/stroma_model.pth \\
  --prefixes   tumor_model stroma_model \\
  --batch_size 64 \\
  --save_tiles
  --use_segmented
"""

    # Add epilog and formatter_class to the ArgumentParser
    parser = argparse.ArgumentParser(
        description="Run WSI prediction pipeline.",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ... (the rest of your parser.add_argument calls are unchanged)
    parser.add_argument("--sample_id", type=str, help="Sample ID (string).")
    parser.add_argument("--input_slide", type=Path, help="Path to the H&E slide.")
    parser.add_argument("--output_dir", type=Path, help="Directory to save the output.")
    parser.add_argument("--tile_file", type=Path, help="Path to the common tiling file.")
    parser.add_argument("-m", "--models", type=Path, nargs='+', required=True,
                        help="One or more paths to the trained models.")
    parser.add_argument("-p", "--prefixes", type=str, nargs='+', required=True,
                        help="Column header labels for each model.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size for model inference (default: 32).")
    parser.add_argument("--use_segmented_tiles", action="store_true",
                        help="If set, save the extracted and segmented tiles to disk.")
    parser.add_argument("--save_segmented_tiles", action="store_true",
                        help="If set, save the extracted and segmented tiles to disk.")
    parser.add_argument("--save_h_and_e_tiles", action="store_true",
                        help="If set, save the extracted and segmented tiles to disk.")

    args = parser.parse_args()

    # ... (the rest of your main function is unchanged)
    if len(args.models) != len(args.prefixes):
        raise ValueError("The number of models must equal the number of prefixes.")

    if args.sample_id is None:
        raise ValueError("The number of models must equal the number of prefixes.")

    tbccsi_.run_preds(
        sample_id=args.sample_id,
        input_slide=args.input_slide,
        output_dir=args.output_dir,
        tile_file=args.tile_file,
        model_list=args.models,
        prefix_list=args.prefixes,
        batch_size=args.batch_size,
        use_segmented=args.use_segmented_tiles,
        save_segmented_tiles=args.save_segmented_tiles,
        save_h_and_e_tiles=args.save_h_and_e_tiles
    )


if __name__ == '__main__':
    main()
