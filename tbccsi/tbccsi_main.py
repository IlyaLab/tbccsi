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
#from .wsi_segmentation import CellSegmentationProcessor
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
                     do_tta=False,
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

                    preds = engine.predict_batch(sub_imgs, sub_meta, do_tta)
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



## EOF ##

