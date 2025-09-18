import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openslide
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
import tifffile


class WSITiler:
    """Handles tiling of whole slide images"""
    def __init__(self, sample_id=None, slide_path=None, output_path=None, tile_file=None):
        self.sample_id = sample_id
        self.slide_path = Path(slide_path)
        self.output_path = Path(output_path)
        self.tile_file = tile_file
        self.level = 0
        self.tile_size = 224
        self.overlap = 0
        self.min_tissue_ratio = 0.1
        self.level_downsamples = [1.0]
        self.tiles = []
        self._slide = None

        ### open the file, be it SVS or TIFF ###
        try:
            # --- Try OpenSlide First ---
            self._slide = OpenSlide(self.slide_path)
            self._backend = 'openslide'
            self.level_dimensions = self._slide.level_dimensions
            print(f"INFO: Successfully opened {self.slide_path} with OpenSlide.")
        except OpenSlideUnsupportedFormatError:
            print(f"INFO: OpenSlide failed. Falling back to Tifffile for {self.slide_path}...")
            try:
                # --- Fallback to Tifffile ---
                self._slide = tifffile.TiffFile(self.slide_path)
                self._backend = 'tifffile'

                # Get dimensions for all levels in the pyramid (series 0)
                # Assumes a standard OME-TIFF pyramid structure
                levels = self._slide.series[0].levels
                self.level_dimensions = [(level.shape[1], level.shape[0]) for level in levels]
                self._slide = self._slide.series[0].levels[self.level].asarray()
                print(f"INFO: Successfully opened {self.slide_path} with Tifffile.")
            except Exception as e:
                raise IOError(f"Failed to open {self.slide_path} with both OpenSlide and Tifffile.") from e
        except Exception as e:
            # Catch other potential errors from OpenSlide
            raise IOError(f"An unexpected error occurred while opening {self.slide_path} with OpenSlide.") from e


    def _read_region(self, location, level, size):
        """
        Reads a region and returns it as a NumPy array.
        """
        x, y = location
        w, h = size

        if self._backend == 'openslide':
            # OpenSlide returns a PIL image, so we convert it to a NumPy array
            pil_image = self._slide.read_region(location, level, size)
            return pil_image#  )[:, :, :3]  # Drop alpha channel if present (RGBA -> RGB)

        elif self._backend == 'tifffile':
            # 2. Call asarray on the PAGE object, not the FILE object
            region_numpy = self._slide[y:(y + h), x:(x + w)]
            #print(region_numpy.shape)  # OK
            return Image.fromarray(region_numpy)


    def _is_tissue_tile(self, tile):
        """
        Check if a tile contains enough tissue (not mostly background)
        Simple heuristic: check if enough pixels are not white/very light
        """
        # Convert to numpy array
        tile_array = np.array(tile)
        # Create mask for non-white pixels (all RGB values < 230)
        non_white_mask = np.all(tile_array < 230, axis=2)
        # Calculate ratio of tissue pixels
        tissue_ratio = np.sum(non_white_mask) / (tile.size[0] * tile.size[1])
        # return fraction
        return tissue_ratio >= self.min_tissue_ratio


    # --- REVISED Helper to only process and append ---
    def _process_and_append_tile(self, tile_image_rgb, x, y, tile_id, tile_path):
        """Calculates color stats and appends the complete tile tuple to the list."""
        tile_array = np.array(tile_image_rgb)
        mean_red = np.mean(tile_array[:, :, 0])
        mean_green = np.mean(tile_array[:, :, 1])
        mean_blue = np.mean(tile_array[:, :, 2])

        # Append the full tuple with image data and stats
        self.tiles.append((tile_image_rgb, x, y, tile_id, mean_red, mean_green, mean_blue))

        # and write them if needed:
        if tile_path is not None:
            os.makedirs(os.path.dirname(tile_path), exist_ok=True)
            tile_image_rgb.save(tile_path)

        return(len(self.tiles))

    def extract_tiles(self, tile_file=None, output_dir=None, save_tiles=False, extract_test=False):
        """
        Creates a tiling or uses an existing one to extract tiles from a whole slide image.

        This function combines two workflows:
        1. If `tile_file` is None: It generates a new tiling by scanning the slide,
           identifying tiles with sufficient tissue, and extracting them on the fly.
        2. If `tile_file` is a path to a CSV: It reads the x, y coordinates from
           the file and extracts only those specified tiles.

        Args:
            tile_file (str, optional): Path to a CSV file with 'x', 'y', 'tile_id' columns.
                                       If None, a new tiling is generated. Defaults to None.
            output_dir (str, optional): Directory to save tile images. Defaults to None.
            save_tiles (bool, optional): If True, saves extracted tiles to `output_dir`.
                                         Defaults to False.
            extract_test (bool, optional): If True, stops after `self.testn` tiles for testing.
                                           Defaults to False.

        Returns:
            list: A list of tuples, where each tuple is (tile_image, x, y, tile_id).
        """
        he_dir = self.output_path/'he_tiles'
        expected_path = None
        # Get slide properties, which are needed in both cases
        level_dims = self.level_dimensions[self.level]
        level_downsample = self.level_downsamples[self.level]

        # =============================================================================
        #  PATH 1: A tile file IS provided, check for existing images.
        #          if tile file doesn't exist, use the tiling to crop img
        # =============================================================================
        if tile_file and os.path.isfile(tile_file):
            print(f"Loading tile coordinates from: {tile_file}")
            try:
                tiles_df = pd.read_csv(tile_file)
                print(f"Found {len(tiles_df)} tiles to process.")
            except FileNotFoundError:
                print(f"Error: Tile file not found at {tile_file}... creating new file.")
                exit()
            for i, row in tqdm(tiles_df.iterrows(), total=len(tiles_df), desc="Extracting from file"):
                x, y, tile_id = int(row['x']), int(row['y']), int(row['tile_id'])
                tile_rgb = None
                ### Check if tile image already exists ###
                if self.output_path:
                    collection_index = i // 1000  # Use loop index to find the correct folder
                    collection_dir = os.path.join(he_dir, f'collection_{collection_index}')
                    os.makedirs(collection_dir, exist_ok=True)
                    expected_path = os.path.join(collection_dir, f"tile_{tile_id:06d}_x{x}_y{y}.png")
                    if os.path.exists(expected_path):
                        # If it exists, load it from disk
                        tile_rgb = Image.open(expected_path).convert('RGB')

                # If the tile wasn't loaded from a file, extract it from the slide
                if tile_rgb is None:
                    tile_raw = self._read_region(
                        (int(x * level_downsample), int(y * level_downsample)),
                        self.level,
                        (self.tile_size, self.tile_size)
                    )
                    tile_rgb = tile_raw.convert('RGB')
                    # Now, process the tile (whether loaded or extracted) and SAVE
                    self._process_and_append_tile(tile_rgb, x, y, tile_id, expected_path)
                else:
                # then it was but we'll append it to the list
                    self._process_and_append_tile(tile_rgb, x, y, tile_id, None)

        # =================================================================================
        #  PATH 2: No tile file, so we generate the tiling and extract on the fly.
        # =================================================================================
        else:
            print("No tile file found. Generating tiling and extracting on the fly...")
            print("... writing tiles to:  " + str(he_dir))
            step_size = self.tile_size - self.overlap
            tile_id = 0

            # Create a progress bar for the outer loop
            y_coords = range(0, level_dims[1] - self.tile_size + 1, step_size)
            pbar = tqdm(total=len(y_coords), desc="Scanning slide")

            for y in y_coords:
                for x in range(0, level_dims[0] - self.tile_size + 1, step_size):
                    try:
                        # Read region just once
                        tile = self._read_region(
                            (int(x * level_downsample), int(y * level_downsample)),
                            self.level,
                            (self.tile_size, self.tile_size)
                        )
                        tile_rgb = tile.convert('RGB')

                        # Check if tile contains tissue
                        if self._is_tissue_tile(tile_rgb):
                            # If it's a good tile, process and store it
                            if self.output_path and save_tiles:
                                collection_index = tile_id // 1000  # Use loop index to find the correct folder
                                collection_dir = os.path.join(he_dir, f'collection_{collection_index}')
                                os.makedirs(collection_dir, exist_ok=True)
                                expected_path = os.path.join(collection_dir, f"tile_{tile_id:06d}_x{x}_y{y}.png")
                                self._process_and_append_tile(tile_rgb, x, y, tile_id, expected_path)
                            else:
                                self._process_and_append_tile(tile_rgb, x, y, tile_id, None)

                    except Exception as e:
                        print(f"Error extracting tile at ({x}, {y}): {e}")
                        # Decide whether to continue or exit; here we continue
                        exit()
                    # update the tile ID
                    tile_id += 1
                pbar.update(1)
            pbar.close()

            if self.output_path:
                # 1. Create a new list of tuples, slicing off the first element from each
                data_for_df = [t[1:] for t in self.tiles]
                # 2. Define the column names for the new, sliced data
                column_names = ['x', 'y', 'tile_id', 'mean_red', 'mean_green', 'mean_blue']
                # 3. Create the DataFrame
                df = pd.DataFrame(data_for_df, columns=column_names)
                df.to_csv(self.output_path/(self.sample_id+'_common_tiling.csv'), index=False)

        print(f"\n✅ Extracted {len(self.tiles)} valid tiles.")
        return self.tiles

