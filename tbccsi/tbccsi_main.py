
from pathlib import Path
import torch

# tile based classification on cell segmented images
from .wsi_tiler import WSITiler
from .wsi_segmentation import CellSegmentationProcessor
from .model_inference import WSIInferenceEngine
from .wsi_plot import WSIPlotter


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

    print("\n-----------------------------------------")
    print("Checking if CUDA available: " + str(torch.cuda.is_available()))
    print("-----------------------------------------\n")


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
            plotter = WSIPlotter(sample_id, tiler._slide)
            plotter.create_heatmap(predictions_df, heatmap_path, point_size=4, prob_col="prob_class_1")
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


## EOF ##
