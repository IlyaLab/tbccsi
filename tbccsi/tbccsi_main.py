from tqdm import tqdm
from PIL import ImageFile
from pathlib import Path
import pandas as pd
import numpy as np

from .wsi_tiler import WSITiler
from .model_inference import InferenceEngine, ReinhardNormalizer
from .wsi_plot import WSIPlotter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════
# Prediction Pipeline
# ══════════════════════════════════════════════════════════════

def run_pred(sample_id,
             input_slide,
             work_dir,
             tile_file,
             model_path,
             batch_size,
             model_config=None,
             do_inference=False,
             do_tta=False,
             do_plot="None",
             forward_kwargs=None):
    """
    Config-driven prediction pipeline.

    Args:
        model_config: Path to model_config.yaml (or None for legacy mode).
        forward_kwargs: Extra kwargs passed to model.forward()
                        (e.g. {"domain_id": 1} for DAFT models).
    """
    RAM_BATCH_SIZE = 256
    forward_kwargs = forward_kwargs or {}

    print("\n-----------------------------------------")
    print(f"Processing Sample: {sample_id}")
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

    # 4. Initialize Inference Components
    if do_inference:
        normalizer = ReinhardNormalizer()
        engine = InferenceEngine(
            model_path=model_path,
            model_config=model_config,
        )

        all_predictions = []

        num_chunks = max(1, len(coords_df) // RAM_BATCH_SIZE)
        df_chunks = np.array_split(coords_df, num_chunks)

        print(f"Processing {len(coords_df)} tiles in {len(df_chunks)} RAM batches...")

        for i, chunk in enumerate(tqdm(df_chunks, desc="Processing Batches")):
            if chunk.empty:
                continue

            batch_images = []
            batch_metadata = []

            for _, row in chunk.iterrows():
                try:
                    x, y, tid = int(row['x']), int(row['y']), int(row['tile_id'])
                    tile_raw = tiler._read_region(
                        (x, y), tiler.level, (tiler.tile_size, tiler.tile_size)
                    ).convert('RGB')
                    tile_norm = normalizer.normalize(tile_raw)
                    batch_images.append(tile_norm)
                    batch_metadata.append(row.to_dict())
                except Exception as e:
                    print(f"Error reading tile {tid}: {e}")

            if batch_images:
                for k in range(0, len(batch_images), batch_size):
                    sub_imgs = batch_images[k: k + batch_size]
                    sub_meta = batch_metadata[k: k + batch_size]
                    preds = engine.predict_batch(
                        sub_imgs, sub_meta, do_tta, **forward_kwargs
                    )
                    all_predictions.extend(preds)

            del batch_images
            del batch_metadata

        # 5. Save Results
        if all_predictions:
            preds_df = pd.DataFrame(all_predictions)
            # Use model name in output filename if available
            model_name = engine.config.name if engine.config else "virchow"
            out_path = output_dir / f"{sample_id}_{model_name}_preds.csv"
            preds_df.to_csv(out_path, index=False)
            print(f"\nSaved {len(preds_df)} predictions to {out_path}")

    # Heatmap
    if (not do_inference) and (do_plot != "None"):
        print("Building Heatmap...")
        # Try to find the most recent preds file
        pred_files = list(output_dir.glob(f"{sample_id}_*_preds.csv"))
        if pred_files:
            preds_df = pd.read_csv(pred_files[-1])
        else:
            preds_df = pd.read_csv(output_dir / f"{sample_id}_virchow_preds.csv")
        try:
            heatmap_file = f"{sample_id}_{do_plot}_heatmap.png"
            plotter = WSIPlotter(sample_id, input_slide, output_dir)
            plotter.create_heatmap(preds_df, heatmap_file, point_size=4, prob_col=do_plot)
        except Exception as e:
            print("heatmap failed..." + str(e))


# ══════════════════════════════════════════════════════════════
# Embedding Extraction Pipeline
# ══════════════════════════════════════════════════════════════

def run_embed(sample_id,
              input_slide,
              work_dir,
              tile_file,
              model_path,
              batch_size,
              model_config=None,
              do_tta=False,
              latent_types=None,
              save_format='csv',
              forward_kwargs=None):
    """
    Extract latent embeddings from WSI tiles.
    """
    forward_kwargs = forward_kwargs or {}
    RAM_BATCH_SIZE = 256

    print("\n-----------------------------------------")
    print(f"Extracting Embeddings: {sample_id}")
    if latent_types:
        print(f"Latent types: {latent_types}")
    print("-----------------------------------------\n")

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_file_path = output_dir / tile_file

    tiler = WSITiler(sample_id, input_slide, output_dir, tile_file_path)

    if not tile_file_path.exists():
        print("Creating tile grid...")
        tiler.create_tile_file()

    try:
        coords_df = pd.read_csv(tile_file_path, comment='#')
    except FileNotFoundError:
        print("Error: Tile file could not be found or created.")
        return

    if coords_df.empty:
        print("Warning: No tissue tiles found in grid.")
        return

    normalizer = ReinhardNormalizer()
    engine = InferenceEngine(
        model_path=model_path,
        model_config=model_config,
    )

    all_embeddings = []

    num_chunks = max(1, len(coords_df) // RAM_BATCH_SIZE)
    df_chunks = np.array_split(coords_df, num_chunks)

    print(f"Processing {len(coords_df)} tiles in {len(df_chunks)} RAM batches...")

    for i, chunk in enumerate(tqdm(df_chunks, desc="Extracting Embeddings")):
        if chunk.empty:
            continue

        batch_images = []
        batch_metadata = []

        for _, row in chunk.iterrows():
            try:
                x, y, tid = int(row['x']), int(row['y']), int(row['tile_id'])
                tile_raw = tiler._read_region(
                    (x, y), tiler.level, (tiler.tile_size, tiler.tile_size)
                ).convert('RGB')
                tile_norm = normalizer.normalize(tile_raw)
                batch_images.append(tile_norm)
                batch_metadata.append(row.to_dict())
            except Exception as e:
                print(f"Error reading tile {tid}: {e}")

        if batch_images:
            for k in range(0, len(batch_images), batch_size):
                sub_imgs = batch_images[k: k + batch_size]
                sub_meta = batch_metadata[k: k + batch_size]
                embeddings = engine.extract_embeddings_batch(
                    sub_imgs, sub_meta,
                    use_tta=do_tta,
                    latent_types=latent_types,
                    **forward_kwargs,
                )
                all_embeddings.extend(embeddings)

        del batch_images
        del batch_metadata

    if all_embeddings:
        # Determine valid latent types from the first result
        actual_types = [
            k.replace('embedding_', '') for k in all_embeddings[0]
            if k.startswith('embedding_')
        ]

        if save_format == 'npz':
            save_embeddings_npz(all_embeddings, output_dir, sample_id, actual_types)
        else:
            save_embeddings_csv(all_embeddings, output_dir, sample_id, actual_types)

        print(f"\nExtracted embeddings for {len(all_embeddings)} tiles")
        print(f"Saved to {output_dir}")


# ══════════════════════════════════════════════════════════════
# Backward-compatible aliases
# ══════════════════════════════════════════════════════════════

def run_virchow_pred(sample_id, input_slide, work_dir, tile_file,
                     model_path, batch_size, do_inference=False,
                     do_tta=False, do_plot="None"):
    """Legacy alias — calls run_pred without model_config (legacy mode)."""
    return run_pred(
        sample_id=sample_id, input_slide=input_slide, work_dir=work_dir,
        tile_file=tile_file, model_path=model_path, batch_size=batch_size,
        model_config=None, do_inference=do_inference, do_tta=do_tta,
        do_plot=do_plot,
    )


def run_virchow_embed(sample_id, input_slide, work_dir, tile_file,
                      model_path, batch_size, do_tta=False,
                      latent_types=None, save_format='csv'):
    """Legacy alias — calls run_embed without model_config (legacy mode)."""
    return run_embed(
        sample_id=sample_id, input_slide=input_slide, work_dir=work_dir,
        tile_file=tile_file, model_path=model_path, batch_size=batch_size,
        model_config=None, do_tta=do_tta, latent_types=latent_types,
        save_format=save_format,
    )


# ══════════════════════════════════════════════════════════════
# Embedding save helpers (unchanged)
# ══════════════════════════════════════════════════════════════

def save_embeddings_npz(embeddings_list, output_dir, sample_id, latent_types):
    metadata_records = []
    embedding_arrays = {lt: [] for lt in latent_types}

    for record in embeddings_list:
        meta = {k: v for k, v in record.items() if not k.startswith('embedding_')}
        metadata_records.append(meta)
        for lt in latent_types:
            embed_key = f'embedding_{lt}'
            if embed_key in record:
                embedding_arrays[lt].append(record[embed_key])

    for lt in latent_types:
        if embedding_arrays[lt]:
            embedding_arrays[lt] = np.array(embedding_arrays[lt])

    npz_path = output_dir / f"{sample_id}_embeddings.npz"
    np.savez_compressed(npz_path, **embedding_arrays)
    print(f"Saved embeddings to {npz_path}")

    for lt in latent_types:
        if lt in embedding_arrays and len(embedding_arrays[lt]) > 0:
            print(f"  {lt}: shape {embedding_arrays[lt].shape}")

    meta_df = pd.DataFrame(metadata_records)
    meta_path = output_dir / f"{sample_id}_embedding_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")


def save_embeddings_csv(embeddings_list, output_dir, sample_id, latent_types):
    flattened_records = []

    for record in embeddings_list:
        flat_record = {}
        for k, v in record.items():
            if not k.startswith('embedding_'):
                flat_record[k] = v
        for lt in latent_types:
            embed_key = f'embedding_{lt}'
            if embed_key in record:
                for i, val in enumerate(record[embed_key]):
                    flat_record[f'{lt}_{i}'] = val
        flattened_records.append(flat_record)

    df = pd.DataFrame(flattened_records)
    csv_path = output_dir / f"{sample_id}_embeddings.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved embeddings to {csv_path}")
    print(f"  DataFrame shape: {df.shape}")
