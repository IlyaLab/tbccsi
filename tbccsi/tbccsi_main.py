from tqdm import tqdm
from PIL import ImageFile
from pathlib import Path
import pandas as pd
import numpy as np

# tile based classification on cell segmented images
from .wsi_tiler import WSITiler
from .model_inference import VirchowInferenceEngine
from .model_inference import ReinhardNormalizer
from .wsi_plot import WSIPlotter

# from .wsi_segmentation import CellSegmentationProcessor

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


# --- 4. Embedding Extraction Function ---
def run_virchow_embed(sample_id,
                      input_slide,
                      work_dir,
                      tile_file,
                      model_path,
                      batch_size,
                      do_tta=False,
                      latent_types=None,
                      save_format='csv'):
    """
    Extract latent embeddings from WSI tiles.

    Args:
        sample_id: Sample identifier
        input_slide: Path to the WSI file
        work_dir: Working directory for outputs
        tile_file: Tile coordinate file
        model_path: Path to trained model
        batch_size: GPU batch size
        do_tta: Apply test-time augmentation
        latent_types: List of latent types to extract. Options:
            ['backbone', 'structure', 'immune_shared', 'immune_tcell', 'immune_mac']
            If None, extracts all types.
        save_format: Output format - 'npz' (compressed arrays) or 'csv' (flattened)
    """

    # Default to all latent types if not specified
    if latent_types is None:
        latent_types = ['backbone', 'structure', 'immune_shared', 'immune_tcell', 'immune_mac']

    # Validate latent types
    valid_types = {'backbone', 'structure', 'immune_shared', 'immune_tcell', 'immune_mac'}
    latent_types = [lt for lt in latent_types if lt in valid_types]

    if not latent_types:
        print("Error: No valid latent types specified.")
        return

    RAM_BATCH_SIZE = 256

    print("\n-----------------------------------------")
    print(f"Extracting Embeddings: {sample_id}")
    print(f"Latent types: {latent_types}")
    print("-----------------------------------------\n")

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_file_path = output_dir / tile_file

    # 1. Initialize Tiler
    tiler = WSITiler(sample_id, input_slide, output_dir, tile_file_path)

    # 2. Ensure Tile Grid Exists
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

    # 4. Initialize Components
    normalizer = ReinhardNormalizer()
    engine = VirchowInferenceEngine(model_path)

    all_embeddings = []

    # Process in chunks to manage RAM
    num_chunks = max(1, len(coords_df) // RAM_BATCH_SIZE)
    df_chunks = np.array_split(coords_df, num_chunks)

    print(f"Processing {len(coords_df)} tiles in {len(df_chunks)} RAM batches...")

    for i, chunk in enumerate(tqdm(df_chunks, desc="Extracting Embeddings")):
        if chunk.empty:
            continue

        batch_images = []
        batch_metadata = []

        # --- A. Extract & Normalize Tiles ---
        for _, row in chunk.iterrows():
            try:
                x, y, tid = int(row['x']), int(row['y']), int(row['tile_id'])

                tile_raw = tiler._read_region(
                    (x, y),
                    tiler.level,
                    (tiler.tile_size, tiler.tile_size)
                ).convert('RGB')

                tile_norm = normalizer.normalize(tile_raw)

                batch_images.append(tile_norm)
                batch_metadata.append(row.to_dict())

            except Exception as e:
                print(f"Error reading tile {tid}: {e}")

        # --- B. Extract Embeddings ---
        if batch_images:
            # Sub-batching for GPU
            for k in range(0, len(batch_images), batch_size):
                sub_imgs = batch_images[k: k + batch_size]
                sub_meta = batch_metadata[k: k + batch_size]

                embeddings = engine.extract_embeddings_batch(
                    sub_imgs,
                    sub_meta,
                    use_tta=do_tta,
                    latent_types=latent_types
                )
                all_embeddings.extend(embeddings)

        # --- C. Cleanup ---
        del batch_images
        del batch_metadata

    # 5. Save Results
    if all_embeddings:
        if save_format == 'npz':
            # Save as compressed numpy arrays (more efficient for large embeddings)
            save_embeddings_npz(all_embeddings, output_dir, sample_id, latent_types)
        else:
            # Save as CSV (easier to inspect, but larger files)
            save_embeddings_csv(all_embeddings, output_dir, sample_id, latent_types)

        print(f"\nExtracted embeddings for {len(all_embeddings)} tiles")
        print(f"Saved to {output_dir}")


def save_embeddings_npz(embeddings_list, output_dir, sample_id, latent_types):
    """
    Save embeddings as compressed .npz file with separate arrays per latent type.
    Also saves metadata as CSV.
    """
    # Separate metadata from embeddings
    metadata_records = []
    embedding_arrays = {lt: [] for lt in latent_types}

    for record in embeddings_list:
        # Extract metadata (everything except embeddings)
        meta = {k: v for k, v in record.items() if not k.startswith('embedding_')}
        metadata_records.append(meta)

        # Extract embeddings
        for lt in latent_types:
            embed_key = f'embedding_{lt}'
            if embed_key in record:
                embedding_arrays[lt].append(record[embed_key])

    # Convert to numpy arrays
    for lt in latent_types:
        if embedding_arrays[lt]:
            embedding_arrays[lt] = np.array(embedding_arrays[lt])

    # Save embeddings
    npz_path = output_dir / f"{sample_id}_embeddings.npz"
    np.savez_compressed(npz_path, **embedding_arrays)
    print(f"Saved embeddings to {npz_path}")

    # Print dimensions for each latent type
    for lt in latent_types:
        if lt in embedding_arrays and len(embedding_arrays[lt]) > 0:
            print(f"  {lt}: shape {embedding_arrays[lt].shape}")

    # Save metadata
    meta_df = pd.DataFrame(metadata_records)
    meta_path = output_dir / f"{sample_id}_embedding_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")


def save_embeddings_csv(embeddings_list, output_dir, sample_id, latent_types):
    """
    Save embeddings as CSV with flattened embedding dimensions.
    Warning: This can create very wide DataFrames for high-dimensional embeddings.
    """
    # Flatten embeddings into columns
    flattened_records = []

    for record in embeddings_list:
        flat_record = {}

        # Copy metadata
        for k, v in record.items():
            if not k.startswith('embedding_'):
                flat_record[k] = v

        # Flatten embeddings
        for lt in latent_types:
            embed_key = f'embedding_{lt}'
            if embed_key in record:
                embedding = record[embed_key]
                # Create columns like: backbone_0, backbone_1, ..., backbone_2559
                for i, val in enumerate(embedding):
                    flat_record[f'{lt}_{i}'] = val

        flattened_records.append(flat_record)

    # Save as CSV
    df = pd.DataFrame(flattened_records)
    csv_path = output_dir / f"{sample_id}_embeddings.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved embeddings to {csv_path}")
    print(f"  DataFrame shape: {df.shape}")

## EOF ##