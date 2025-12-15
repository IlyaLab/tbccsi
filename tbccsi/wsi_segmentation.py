
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from cellpose import models, core, io
from collections import deque
import re


class CellSegmentationProcessor:
    """Handles cell segmentation using Cellpose and loading of pre-segmented tiles."""

    def __init__(self, batch_size=32, flow_threshold=0.4, cellprob_threshold=0.00, gpu=True):
        self.batch_size = batch_size
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

        # Get the cellpose logger and set level to WARNING to silence INFO messages
        io.logger_setup()
        cellpose_logger = logging.getLogger('cellpose')
        cellpose_logger.setLevel(logging.ERROR)

        print("Loading Cellpose model...")
        self.model = models.CellposeModel(gpu=gpu)
        print(f"Cellpose model loaded (GPU: {gpu})")

    def segment_and_mask_tiles(self, tiles, save_masks=False, mask_output_dir=None):
        """
        Apply cell segmentation and masking to tiles.

        Args:
            tiles: List of tuples (tile_image, x_coord, y_coord, tile_id, ...)
            save_masks: Whether to save mask images for debugging.
            mask_output_dir: Directory to save masks if save_masks=True.

        Returns:
            List of tuples (masked_tile_image, x_coord, y_coord, tile_id, mask).
        """

        # Create output directory if specified
        c_cnt = 0
        collection_dir = mask_output_dir / f'collection_{c_cnt}'
        if save_masks and mask_output_dir:
            os.makedirs(collection_dir, exist_ok=True)

        masked_tiles = []

        # Convert your list of tiles to a deque
        tiles = deque(tiles)

        print(f"Applying cell segmentation to {len(tiles)} tiles...")
        total_tiles = len(tiles)
        # Process tiles in batches
        with tqdm(total=total_tiles) as pbar:
            while tiles:
                # Use list comprehension to create a batch from the deque
                batch_tiles = [tiles.popleft() for _ in range(min(self.batch_size, len(tiles)))]

                # Process tiles in batches
                #for i in tqdm(range(0, len(tiles), self.batch_size)):
                # create a batch of tiles
                #batch_tiles = tiles[i:i + self.batch_size]

                # Extract images and convert to numpy arrays
                imgs = []
                coords_and_ids = []

                for tile_img, x, y, tile_id, _, _, _ in batch_tiles:
                    # Convert PIL to numpy array (cellpose expects numpy)
                    img_array = np.array(tile_img)
                    imgs.append(img_array)
                    coords_and_ids.append((x, y, tile_id))

                # Run cellpose segmentation
                try:
                    masks, flows, styles = self.model.eval(
                        imgs,
                        batch_size=self.batch_size,  # Process entire batch at once
                        flow_threshold=self.flow_threshold,
                        cellprob_threshold=self.cellprob_threshold
                    )

                    # Apply masks to images
                    for j, (img_array, (x, y, tile_id)) in enumerate(zip(imgs, coords_and_ids)):

                        # Apply mask (set non-cell pixels to 0)
                        masked_img = img_array.copy()
                        masked_img[~(masks[j] > 0)] = 0

                        # Convert back to PIL Image
                        masked_pil = Image.fromarray(masked_img.astype(np.uint8))

                        # optional saving
                        if mask_output_dir and save_masks:
                            collection_index = tile_id // 1000  # Use loop index to find the correct folder
                            collection_dir = os.path.join(mask_output_dir, f'collection_{collection_index}')
                            os.makedirs(collection_dir, exist_ok=True)
                            masked_path = os.path.join(collection_dir, f"masked_{tile_id:06d}_x{x}_y{y}.png")
                            masked_pil.save(masked_path)
                        # append to the list of masked tiles.
                        masked_tiles.append((masked_pil, x, y, tile_id, masks[j]))

                except Exception as e:
                    print(f"Error processing batch {tile_id // self.batch_size}: {str(e)}")
                    for j, (img_array, (x, y, tile_id)) in enumerate(zip(imgs, coords_and_ids)):
                        masked_tiles.append((None, x, y, tile_id, None))

                # Update the progress bar with the number of tiles processed in this batch
                pbar.update(len(batch_tiles))

        print(f"Completed cell segmentation for {len(masked_tiles)} tiles")
        return masked_tiles


    def segment_tiles_from_disk(self, tile_file_path=None, tile_input_dir=None,
                                save_masks=False, mask_output_dir=None):
        print(f"Reading tile information from {tile_file_path}...")
        try:
            tile_df = pd.read_csv(tile_file_path)
            # Ensure required columns exist
            if not {'x', 'y', 'tile_id'}.issubset(tile_df.columns):
                raise ValueError("Tile file must contain 'x', 'y', and 'tile_id' columns.")
        except FileNotFoundError:
            print(f"Error: Tile file not found at {tile_file_path}")
            return []
        except Exception as e:
            print(f"Error reading tile file: {e}")
            return []

        loaded_tiles = []
        mask_input_dir = Path(tile_input_dir) # Convert to Path object for easier joining

        print(f"Loading {len(tile_df)} tiles from {tile_input_dir}...")

        # Use tqdm for a progress bar
        for _, row in tqdm(tile_df.iterrows(), total=tile_df.shape[0]):
            x, y, tile_id, mean_red, mean_green, mean_blue = int(row['x']), int(row['y']), int(row['tile_id']), float(row['mean_red']), float(row['mean_green']), float(row['mean_blue'])

            # Reconstruct the file path using the same logic as the saving function
            collection_index = tile_id // 1000
            collection_dir = mask_input_dir / f'collection_{collection_index}'
            file_name = f"tile_{tile_id:06d}_x{x}_y{y}.png"
            image_path = collection_dir / file_name

            try:
                # Open the image file
                with Image.open(image_path) as img:
                    # Load the image data into memory and convert to RGB
                    tile_pil_image = img.convert("RGB").copy()
                    loaded_tiles.append((tile_pil_image, x, y, tile_id, mean_red, mean_green, mean_blue))
            except FileNotFoundError:
                print(f"Warning: Tile file not found, skipping: {image_path}")
            except Exception as e:
                print(f"Warning: Could not load file {image_path}. Error: {e}")

        print(f"Successfully loaded {len(loaded_tiles)} tiles from disk.")
        print(f"Starting segmentation.....")
        masked_tiles = self.segment_and_mask_tiles(loaded_tiles, save_masks, mask_output_dir)
        return(masked_tiles)


    def load_masked_tiles_from_disk(self, tile_file_path, mask_input_dir):
        """
        Loads pre-segmented and masked tile images from disk.

        This function reads a file with tile coordinates and IDs, constructs the
        paths to the corresponding saved mask images, and loads them.

        Args:
            tile_file_path (str or Path): Path to the CSV file containing tile info.
                                          Expected columns: 'x', 'y', 'tile_id'.
            mask_input_dir (str or Path): The base directory where masked tiles
                                          (e.g., 'collection_0', 'collection_1') are stored.

        Returns:
            list: A list of tuples, where each tuple is (masked_pil_image, x, y, tile_id).
        """
        print(f"Reading tile information from {tile_file_path}...")
        try:
            tile_df = pd.read_csv(tile_file_path)
            # Ensure required columns exist
            if not {'x', 'y', 'tile_id'}.issubset(tile_df.columns):
                raise ValueError("Tile file must contain 'x', 'y', and 'tile_id' columns.")
        except FileNotFoundError:
            print(f"Error: Tile file not found at {tile_file_path}")
            return []
        except Exception as e:
            print(f"Error reading tile file: {e}")
            return []

        loaded_tiles = []
        mask_input_dir = Path(mask_input_dir)  # Convert to Path object for easier joining

        print(f"Loading {len(tile_df)} masked tiles from {mask_input_dir}...")

        # Use tqdm for a progress bar
        for _, row in tqdm(tile_df.iterrows(), total=tile_df.shape[0]):
            x, y, tile_id = int(row['x']), int(row['y']), int(row['tile_id'])

            # Reconstruct the file path using the same logic as the saving function
            collection_index = tile_id // 1000
            collection_dir = mask_input_dir / f'collection_{collection_index}'
            file_name = f"masked_{tile_id:06d}_x{x}_y{y}.png"
            image_path = collection_dir / file_name

            try:
                # Open the image file
                with Image.open(image_path) as img:
                    # Load the image data into memory and convert to RGB
                    masked_pil_image = img.convert("RGB").copy()
                    loaded_tiles.append((masked_pil_image, x, y, tile_id))
            except FileNotFoundError:
                print(f"Warning: Mask file not found, skipping: {image_path}")
            except Exception as e:
                print(f"Warning: Could not load file {image_path}. Error: {e}")

        print(f"Successfully loaded {len(loaded_tiles)} tiles from disk.")
        return loaded_tiles


    def map_tile_ids_to_paths(self, tile_file_path: Path, tile_input_dir: Path) -> pd.DataFrame:
        """
        Scans a directory for image files and maps them to tile_ids from a CSV.

        Args:
            tile_file_path: Path to the tile coordinate CSV file (e.g., 'tile_file.csv').
            tile_input_dir: Path to the top-level directory containing segmented tiles
                            (e.g., '.../segmented_tiles/').

        Returns:
            A pandas DataFrame based on the input CSV, but with an added
            'file_path' column. Rows for which no matching file was found
            are dropped.
        """
        print(f"Scanning for tiles in: {tile_input_dir}")

        # 1. Find all image files recursively
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        all_image_paths = [
            p for p in tile_input_dir.rglob("*")
            if p.suffix.lower() in image_extensions
        ]

        if not all_image_paths:
            print(f"Warning: No image files found in {tile_input_dir}")
            return pd.DataFrame()

        # 2. Build a map from an extracted tile_id (int) to its file_path (Path)
        path_map = {}
        for path in all_image_paths:
            # Use regex to find all numbers in the filename
            # e.g., "tile_1056.png" -> ["1056"]
            # e.g., "slide_v2_1056_mask.png" -> ["2", "1056"]
            numbers = re.findall(r'\d+', path.stem)

            if numbers:
                # Assume the *first* number is the unique tile_id
                try:
                    tile_id = int(numbers[0])
                    if tile_id in path_map:
                        print(f"Warning: Duplicate tile_id {tile_id} found. "
                              f"Overwriting {path_map[tile_id].name} with {path.name}")
                    path_map[tile_id] = path
                except ValueError:
                    pass  # Not a valid integer

        print(f"Found {len(all_image_paths)} image files, mapped {len(path_map)} unique tile IDs.")

        # 3. Load the tile coordinate CSV
        try:
            df = pd.read_csv(tile_file_path, comment='#')
        except FileNotFoundError:
            print(f"Error: Tile coordinate file not found at {tile_file_path}")
            return pd.DataFrame()

        # 4. Map the tile_id column to the file paths
        df['file_path'] = df['tile_id'].map(path_map)

        # 5. Report and drop any missing tiles
        missing_count = df['file_path'].isnull().sum()
        if missing_count > 0:
            print(f"Warning: Could not find matching files for {missing_count} / {len(df)} tile_ids.")
            # To see which ones are missing:
            # print(df[df['file_path'].isnull()]['tile_id'].tolist())
            df = df.dropna(subset=['file_path'])

        print(f"Successfully mapped {len(df)} tile_ids to their file paths.")
        return df