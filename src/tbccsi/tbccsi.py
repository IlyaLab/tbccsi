import argparse
import logging
from pathlib import Path
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import torch
from PIL import Image
import openslide
from openslide import OpenSlide
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    ViTPreTrainedModel,
    ViTConfig,
    ViTModel,
    AutoImageProcessor,
)

# tile based classification on cell segmented images
from vit_model_5_6 import VitClassification
from tbccsi_tiler import WSITiler
from tbccsi_segmentation import CellSegmentationProcessor
from tbccsi_inference import WSIInferenceEngine


print("\n-----------------------------------------")
print("Checking if CUDA available: " + str(torch.cuda.is_available()))
print("-----------------------------------------\n")

print("\n Running google/vit-base-patch16-224-in21k model for binary classification. \n")



######### RUN PREDICTION ############################################################3
def run_preds(sample_id,   # sample ID (string)
              input_slide, # path to the H&E slide (path)
              output_dir,  # where to save the tiles (path)
              tile_file,   # the common tiling file (path)
              model_list,  # list of paths to models (list of strings)
              prefix_list, # column header labels for each model (list of strings)
              batch_size, # batch size for CUDA (integer)
              use_segmented=True,
              save_segmented_tiles=True, # whether to save the tiles (bool)
              save_h_and_e_tiles=False
):
    ## Hardcoded params ##
    tile_size = 224
    overlap = 0
    level = 0
    min_tissue_ratio = 0.1
    tiles = None

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reads the slide, and extracts raw tiles
    tiler = WSITiler(sample_id,
                     input_slide,
                     output_dir,
                     tile_file)
    #  segments cells extracted tiles
    segmenter = CellSegmentationProcessor(batch_size=batch_size)

    if not use_segmented:
        print("Extracting H&E tiles...")
        tiles = tiler.extract_tiles(
            save_tiles=save_h_and_e_tiles
        )
        print("Running CELLPOSE-SAM segmentation...")
        segmented_tiles = segmenter.segment_and_mask_tiles(
            tiles,
            save_masks=save_segmented_tiles,
            mask_output_dir=output_dir / "segmented_tiles"
        )
    else:
        print("Loading segmentation tiles...")
        segmented_tiles = segmenter.load_masked_tiles_from_disk(
            tile_file_path=tile_file,
            mask_input_dir=output_dir/"segmented_tiles"
        )

    print("Starting INFERENCE...")
    for model_path, prefix in zip(model_list, prefix_list):
        # Initialize inference engine
        inference_engine = WSIInferenceEngine(model_path)
        # Run predictions on SEGMENTED tiles
        predictions_df = inference_engine.predict_tiles(segmented_tiles, batch_size=batch_size)

        # Save results
        predictions_path = output_dir / f"{sample_id}_{prefix}_preds.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        # make a heatmap
        print("Building Heatmap...")
        try:
            heatmap_path = output_dir / f"{sample_id}_{prefix}_heatmap.png"
            inference_engine.create_heatmap(sample_id, tiler._slide, predictions_df, heatmap_path, point_size=4, prob_col="prob_class_1")
        except Exception as e:
            print("heatmap failed..." + str(e))

        # Create summary statistics
        summary_stats = {
            'total_tiles': len(predictions_df),
            'class_0_count': (predictions_df['predicted_class'] == 0).sum(),
            'class_1_count': (predictions_df['predicted_class'] == 1).sum(),
            'mean_confidence': predictions_df['confidence'].mean(),
            'class_1_ratio': (predictions_df['predicted_class'] == 1).mean()
        }
        # print 'em
        print(f"\nSummary Statistics:")
        print(f"Total tiles: {summary_stats['total_tiles']}")
        print(f"Class 0 (negative): {summary_stats['class_0_count']}")
        print(f"Class 1 (positive): {summary_stats['class_1_count']}")
        print(f"Class 1 ratio: {summary_stats['class_1_ratio']:.3f}")
        print(f"Mean confidence: {summary_stats['mean_confidence']:.3f}")
        print(f"\nAll results saved to {output_dir}")


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
        formatter_class=RawTextHelpFormatter
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

    run_preds(
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

if __name__ == "__main__":
    main()

## EOF ##
