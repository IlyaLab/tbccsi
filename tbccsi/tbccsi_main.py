import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

from pathlib import Path
import torch
import pandas as pd
import numpy as np
import re

# tile based classification on cell segmented images
from .wsi_tiler import WSITiler
from .wsi_segmentation import CellSegmentationProcessor
from .model_inference import WSIInferenceEngine
from .model_inference import VirchowInferenceEngine
from .model_inference import ReinhardNormalizer
from .wsi_plot import WSIPlotter

# Ensure truncated images don't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- 3. Main Processing Function ---
def run_virchow_pred(sample_id,
                     input_slide,
                     work_dir,
                     tile_file,
                     model_path,
                     batch_size,
                     do_inference=False,
                     do_plot="None"):
    # RAM_BATCH_SIZE: How many tiles to load into memory at once before predicting/clearing.
    # Keep this moderate (e.g., 256 or 512) to avoid OOM on system RAM.
    RAM_BATCH_SIZE = 256

    print("\n-----------------------------------------")
    print(f"Processing Sample: {sample_id}")
    print("-----------------------------------------\n")

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_file_path = output_dir / tile_file

    # 1. Initialize Tiler
    tiler = WSITiler(sample_id, input_slide, output_dir, tile_file_path)

    # 2. Ensure Tile Grid Exists
    # If the file doesn't exist, create it (scans slide, saves coords to CSV, no images saved)
    if not tile_file_path.exists():
        print("Creating tile grid...")
        tiler.create_tile_file()

    # 3. Load Coordinate Grid
    try:
        coords_df = pd.read_csv(tile_file_path, comment='#')
    except FileNotFoundError:
        print("Error: Tile file could not be found or created.")
        return

    if coords_df.empty:
        print("Warning: No tissue tiles found in grid.")
        return

    # 4. Initialize Inference Components
    if do_inference:
        normalizer = ReinhardNormalizer()  # Uses default target means/stds
        engine = VirchowInferenceEngine(model_path)

        all_predictions = []

        # Process in chunks to manage RAM
        # We split the dataframe into chunks of size RAM_BATCH_SIZE
        num_chunks = max(1, len(coords_df) // RAM_BATCH_SIZE)
        df_chunks = np.array_split(coords_df, num_chunks)

        print(f"Processing {len(coords_df)} tiles in {len(df_chunks)} RAM batches...")

        for i, chunk in enumerate(tqdm(df_chunks, desc="Processing Batches")):
            if chunk.empty: continue

            batch_images = []
            batch_metadata = []

            # --- A. Extract & Normalize Tiles ---
            for _, row in chunk.iterrows():
                try:
                    x, y, tid = int(row['x']), int(row['y']), int(row['tile_id'])

                    # Read from slide (Level 0 coords provided by create_tile_file)
                    # Note: WSITiler._read_region expects coordinates at Level 0
                    tile_raw = tiler._read_region(
                        (x, y),
                        tiler.level,
                        (tiler.tile_size, tiler.tile_size)
                    ).convert('RGB')

                    # Normalize
                    tile_norm = normalizer.normalize(tile_raw)

                    batch_images.append(tile_norm)
                    batch_metadata.append(row.to_dict())

                except Exception as e:
                    print(f"Error reading tile {tid}: {e}")

            # --- B. Inference on Batch ---
            if batch_images:
                # Engine handles the GPU batching internally?
                # Ideally, we pass the whole RAM batch, and let the engine/GPU handle it.
                # Since our engine.predict_batch stacks them all, ensure RAM_BATCH_SIZE fits in GPU VRAM
                # OR, we can split inside the loop.
                # Here, we assume RAM_BATCH_SIZE (e.g. 256) is split into smaller GPU batches if needed,
                # but for simplicity, let's process this RAM batch in sub-batches for the GPU.

                # Sub-batching for GPU to avoid CUDA OOM
                for k in range(0, len(batch_images), batch_size):
                    sub_imgs = batch_images[k: k + batch_size]
                    sub_meta = batch_metadata[k: k + batch_size]

                    preds = engine.predict_batch(sub_imgs, sub_meta)
                    all_predictions.extend(preds)

            # --- C. Cleanup ---
            del batch_images
            del batch_metadata
            # Python's GC usually handles this, but explicit del helps in tight loops

        # 5. Save Results
        if all_predictions:
            preds_df = pd.DataFrame(all_predictions)
            out_path = output_dir / f"{sample_id}_virchow_preds.csv"
            preds_df.to_csv(out_path, index=False)

            print(f"\nSaved {len(preds_df)} predictions to {out_path}")

            # Basic Stats
            if 'pred_structure' in preds_df.columns:
                pos_ratio = (preds_df['pred_structure'] == 1).mean()
                print(f"Positive Ratio (Class 1): {pos_ratio:.4f}")


    # make a heatmap
    print("Building Heatmap...")
    if (not do_inference) and (do_plot is not "None"):
        preds_df = pd.read_csv(output_dir / f"{sample_id}_virchow_preds.csv")
        try:
            heatmap_file = f"{sample_id}_{do_plot}_heatmap.png"
            plotter = WSIPlotter(sample_id, input_slide, output_dir)
            plotter.create_heatmap(preds_df, heatmap_file, point_size=4, prob_col=do_plot)
        except Exception as e:
            print("heatmap failed..." + str(e))


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

                else:    # Save results
                    predictions_path = output_dir / f"{sample_id}_{prefix}_preds.csv"
                    predictions_df.to_csv(predictions_path, index=False)
                    print(f"Predictions saved to {predictions_path}")

                # make a heatmap
                print("Building Heatmap...")
                try:
                    heatmap_file = f"{sample_id}_{prefix}_heatmap.png"
                    plotter = WSIPlotter(sample_id, input_slide, output_dir)
                    plotter.create_heatmap(predictions_df, heatmap_file, point_size=4, prob_col="prob_class_1")
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

                else:    # Save results
                    predictions_path = output_dir / f"{sample_id}_{prefix}_preds.csv"
                    predictions_df.to_csv(predictions_path, index=False)
                    print(f"Predictions saved to {predictions_path}")

                # make a heatmap
                print("Building Heatmap...")
                try:
                    heatmap_file = f"{sample_id}_{prefix}_heatmap.png"
                    plotter = WSIPlotter(sample_id, input_slide, output_dir)
                    plotter.create_heatmap(predictions_df, heatmap_file, point_size=4, prob_col="prob_class_1")
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

