#!/usr/bin/env python
import pytest

from pathlib import Path

"""Tests for `tbccsi` package."""

def test_loading_tif():
    from tbccsi.wsi_tiler import WSITiler
    tiler = WSITiler("TEST1", 'tests/data/test_img_1.tif', '/users/dgibbs/tmp/tbccsi_tests')
    print(tiler.sample_id)
    print(tiler.get_shape(0))


def test_tile_tif():
    from tbccsi.wsi_tiler import WSITiler
    tiler = WSITiler("TEST1", 'tests/data/test_img_1.tif', '/users/dgibbs/tmp/tbccsi_tests')
    tiler.extract_tiles(tile_file=None, output_dir='/users/dgibbs/tmp/tbccsi_tests',save_tiles=True)
    print("DONE")


def test_segmentation():
    from tbccsi.wsi_segmentation import CellSegmentationProcessor as csp
    output_dir = Path('/users/dgibbs/tmp/tbccsi_tests')
    segs = csp(batch_size=128)
    masks = segs.segment_tiles_from_disk(tile_file_path=output_dir/"TEST1_common_tiling.csv",
                                         tile_input_dir=output_dir/"he_tiles",
                                         save_masks=True,
                                         mask_output_dir=output_dir/"segmented_tiles")
    print("DONE with segmentation")
    print(f"produced {len(masks)} number of masked tiles.")

def test_inference():
    from tbccsi.model_inference import WSIInferenceEngine
    model = WSIInferenceEngine('/users/dgibbs/Work/CSBC/inference_test/models/mixed_224_c/model_dir')
    model.sample_id = "TestID"
    model.prefix = "cancer"
    output_dir = Path('/users/dgibbs/tmp/tbccsi_tests')
    results = model.load_and_predict_tiles(output_dir=output_dir,
                                           tile_file_path=output_dir/"TEST1_common_tiling.csv",
                                           tile_input_dir=output_dir/"segmented_tiles",
                                           )
    print("DONE with prediction")

def test_heatmap():
    from tbccsi.wsi_plot import WSIPlotter
    plotter = WSIPlotter("TEST2",'tests/data/test_img_1.tif')