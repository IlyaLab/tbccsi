
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import re

# tile based classification on cell segmented images
from .wsi_tiler import WSITiler
from .wsi_segmentation import CellSegmentationProcessor
from .model_inference import WSIInferenceEngine
from .wsi_plot import WSIPlotter



def run_preds(sample_id,  # sample ID (string)
              input_slide,  # path to the H&E slide (path)
              work_dir,  # where tiles are saved (path)
              tile_file,  # the common tiling file (path)
              model_list,  # list of paths to models (list of strings)
              prefix_list,  # column header labels for each model (list of strings)
              batch_size,  # batch size for CUDA (integer)
              do_inference=True,  # should we do model inference?
              use_segmented=True,  # reads from work_dir/segmented_tiles
              save_segmented_tiles=True,  # whether to save the tiles (bool)
              save_h_and_e_tiles=False  # No longer used by tiler, but kept for signature
                  ):
        ## Hardcoded params ##
        # These are now defined in the WSITiler, but we can keep them here
        # for reference. The tiler class will be the source of truth.
        # tile_size = 224
        # overlap = 0
        # level = 0
        # min_tissue_ratio = 0.1

        # --- New Param ---
        # Batch size for loading tiles into RAM for segmentation.
        # This is different from the 'batch_size' for model inference.
        # Adjust this based on your system's RAM.
        SEGMENTATION_RAM_BATCH_SIZE = 128

        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print("\n-----------------------------------------")
        print("Checking if CUDA available: " + str(torch.cuda.is_available()))
        print("-----------------------------------------\n")

        # Create output directory
        output_dir = Path(work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the full path to the tile coordinate file
        tile_file_path = output_dir / tile_file
        print(f'Using tile coordinate file: {tile_file_path}')

        if use_segmented:
            # If we expect segmented tiles, the coordinate file MUST exist.
            if not tile_file_path.exists():
                print(f"\nERROR: 'use_segmented' is True, but the required tile file was not found:")
                print(f"Path: {tile_file_path}")
                print("Please run with 'use_segmented=False' first to generate tiles.")
                return
            if not (output_dir / "segmented_tiles").exists():
                print(f"\nERROR: 'use_segmented' is True, but the required 'segmented_tiles' directory was not found:")
                print(f"Path: {output_dir / 'segmented_tiles'}")

        # Initialize the tiler
        # It will use tile_file_path to check for/create the coordinate file.
        tiler = WSITiler(sample_id,
                         input_slide,
                         output_dir,
                         tile_file_path)  # Pass the full path here

        # Initialize the segmenter
        segmenter = CellSegmentationProcessor(batch_size=batch_size)

        if not use_segmented:
            print("Generating tile coordinate file (if needed)...")
            # This function is modified:
            # 1. It checks if tile_file_path exists.
            # 2. If not, it scans the WSI and SAVES the coordinates to tile_file_path.
            # 3. It DOES NOT load any tiles into memory.
            # The 'save_h_and_e_tiles' param is now ignored by this method.
            tiler.create_tile_file()

            print("Running CELLPOSE-SAM segmentation in batches...")
            # 1. Load the coordinate file
            try:
                coords_df = pd.read_csv(tile_file_path)
            except FileNotFoundError:
                print(f"ERROR: Tile coordinate file was not found or created: {tile_file_path}")
                return

            if coords_df.empty:
                print("WARNING: Tile coordinate file is empty. No tiles to segment.")
            else:
                # 2. Process in batches to conserve RAM
                # Split the dataframe into manageable chunks
                df_chunks = np.array_split(coords_df,
                                           max(1, len(coords_df) // SEGMENTATION_RAM_BATCH_SIZE))

                total_tiles = len(coords_df)
                processed_tiles = 0

                for i, chunk in enumerate(df_chunks):
                    if chunk.empty:
                        continue

                    # This list will hold the tile images *for this batch only*
                    tile_batch_data = []
                    start_tile = processed_tiles + 1
                    end_tile = processed_tiles + len(chunk)

                    print(
                        f"--- Processing tile batch {i + 1}/{len(df_chunks)} (Tiles {start_tile}-{end_tile} of {total_tiles}) ---")

                    # 3. Read tiles for the current batch
                    for _, row in chunk.iterrows():
                        x, y, tile_id = int(row['x']), int(row['y']), int(row['tile_id'])

                        # Read the single tile from the WSI
                        # We use the tiler's internal _read_region method
                        level_downsample = tiler.level_downsamples[tiler.level]
                        tile_raw = tiler._read_region(
                            (int(x * level_downsample), int(y * level_downsample)),
                            tiler.level,
                            (tiler.tile_size, tiler.tile_size)
                        )
                        tile_rgb = tile_raw.convert('RGB')

                        # Get color stats from the DataFrame
                        mean_red = row.get('mean_red', 0)
                        mean_green = row.get('mean_green', 0)
                        mean_blue = row.get('mean_blue', 0)

                        # Append tuple: (Image, x, y, tile_id, stats...)
                        tile_batch_data.append((tile_rgb, x, y, tile_id, mean_red, mean_green, mean_blue))

                    # 4. Segment this batch and save to disk
                    segmenter.segment_and_mask_tiles(
                        tile_batch_data,
                        save_masks=save_segmented_tiles,
                        mask_output_dir=output_dir / "segmented_tiles"
                    )

                    processed_tiles += len(chunk)

                    # 5. Clear this batch's images from memory
                    del tile_batch_data

            print("✅ CELLPOSE-SAM segmentation complete.")

        # else: segmented files already exist on disk, proceed to inference.

        if do_inference:

            tile_map_df = segmenter.map_tile_ids_to_paths(tile_file_path, (output_dir/"segmented_tiles"))

            if tile_map_df.empty:
                print("No tiles to predict. Exiting.")
                return

            print("Starting INFERENCE...")
            for model_path, prefix in zip(model_list, prefix_list):
                # Initialize inference engine
                inference_engine = WSIInferenceEngine(model_path, tile_map_df)

                # Run predictions on SEGMENTED tiles loaded from disk
                predictions_df = (inference_engine.run_inference(
                    output_dir=output_dir,
                    batch_size=batch_size
                ))
                if predictions_df is None:
                    print(f"Skipping prediction saving and plotting for prefix '{prefix}' (predictions_df is None).")
                    continue

                    # Save results
                predictions_path = output_dir / f"{sample_id}_{prefix}_preds.csv"
                predictions_df.to_csv(predictions_path, index=False)
                print(f"Predictions saved to {predictions_path}")

                # make a heatmap
                print("Building Heatmap...")
                try:
                    heatmap_path = output_dir / f"{sample_id}_{prefix}_heatmap.png"
                    plotter = WSIPlotter(sample_id, input_slide)
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


'''
######### RUN PREDICTION ############################################################3
def run_preds(sample_id,   # sample ID (string)
              input_slide, # path to the H&E slide (path)
              output_dir,  # where tiles are saved (path)
              tile_file,   # the common tiling file (path)
              model_list,  # list of paths to models (list of strings)
              prefix_list, # column header labels for each model (list of strings)
              batch_size, # batch size for CUDA (integer)
              use_segmented=True, # reads from output_dir/segmented_tiles
              save_segmented_tiles=True, # whether to save the tiles (bool)
              save_h_and_e_tiles=False
):
    ## Hardcoded params ##
    tile_size = 224
    overlap = 0
    level = 0
    min_tissue_ratio = 0.1
    tiles = None

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print("\n-----------------------------------------")
    print("Checking if CUDA available: " + str(torch.cuda.is_available()))
    print("-----------------------------------------\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create the tile file path.
    tile_file_path = output_dir / tile_file  # Re-construct the expected path
    print(f'Looking for tile file: {tile_file_path}')

    if not use_segmented and tile_file_path.exists():
        # If we are NOT using segmented tiles AND the tile file already exists,
        # we can assume the tiling step was done previously.
        # However, the user intent suggests if a file is provided, it must be used.
        pass  # Allow the existing file to be used by tiler.extract_tiles later
    elif not use_segmented:
        # If we are extracting new tiles, the tile file shouldn't exist yet,
        # but the logic for checking depends on how WSITiler handles pre-existing files.
        # For the strict 'stop if given file doesn't exist' logic, we check the input:
            # The simplest place to put the check is inside the inference loop,
            # but this catches it much earlier and prevents unnecessary segmentation.
        pass
        # Let's check for the existence of the file that will be used for INFERENCE.
        # The crucial tile file is needed for the inference loop later.
        # We will assume that if 'use_segmented' is True, the file MUST exist already.
        # If 'use_segmented' is False, the tile file will be CREATED by the tiler.

    if use_segmented:
        # Check if the file required for INFERENCE is present
        inference_tile_file = output_dir / Path(tile_file)
        if not inference_tile_file.exists():
            print(f"\nERROR: 'use_segmented' is True, but the required tile file was not found:")
            print(f"Path: {inference_tile_file}")
            print("Please ensure tiling has been completed or set 'use_segmented=False'.")
            return

    # Reads the slide, and extracts raw tiles
    tiler = WSITiler(sample_id,
                     input_slide,
                     output_dir,
                     tile_file_path)

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
        del tiles
    # else the segmented files exist and are on disk...

    print("Starting INFERENCE...")
    for model_path, prefix in zip(model_list, prefix_list):
        # Initialize inference engine
        inference_engine = WSIInferenceEngine(model_path)
        # Run predictions on SEGMENTED tiles
        predictions_df = (inference_engine.predict_from_tile_file(
                               tile_file_path = output_dir / tile_file,
                               tile_input_dir=output_dir / "segmented_tiles",
                               output_dir=output_dir,
                               batch_size=batch_size
                           ))
        if predictions_df is None:
            print(f"Skipping prediction saving and plotting for prefix '{prefix}' (predictions_df is None).")
            continue # Move to the next item in the zip iterator

        # Save results
        predictions_path = output_dir / f"{sample_id}_{prefix}_preds.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        # make a heatmap
        print("Building Heatmap...")
        try:
            heatmap_path = output_dir / f"{sample_id}_{prefix}_heatmap.png"
            plotter = WSIPlotter(sample_id, input_slide)
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
'''
