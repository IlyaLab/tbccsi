#!/usr/bin/env python
import pytest

from pathlib import Path
import pandas as pd

"""Tests for `tbccsi` package."""

# try to read in a simple tiff
def test_loading_tif():
    from tbccsi.wsi_tiler import WSITiler
    tiler = WSITiler("TEST1",
                     '/users/dgibbs/tmp/tbccsi_tests/test_img_1.tif',
                     '/users/dgibbs/tmp/tbccsi_tests')
    print(tiler.sample_id)
    print(tiler.get_shape(0))

# extract files from a simple tiff.
def test_tile_tif():
    from tbccsi.wsi_tiler import WSITiler
    tiler = WSITiler("TEST1",
                     '/users/dgibbs/tmp/tbccsi_tests/test_img_1.tif',
                     '/users/dgibbs/tmp/tbccsi_tests')
    tiler.extract_tiles(tile_file=None, output_dir='/users/dgibbs/tmp/tbccsi_tests',save_tiles=True)
    print("DONE")

# segment tiles, write them to disk
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

# do inference on segmented tiles
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

# make a heatmap!
def test_heatmap():
    from tbccsi.wsi_plot import WSIPlotter
    output_dir = Path('/users/dgibbs/tmp/tbccsi_tests')
    plotter = WSIPlotter(sample="TEST2",
                         slide='/users/dgibbs/tmp/tbccsi_tests/test_img_1.tif',
                         output_dir=output_dir)
    predictions_df = pd.read_csv(output_dir/'TestID_cancer_preds.csv')
    plotter.pred_heatmap("heatmap.pdf", predictions_df, point_size=18, prob_col="prob_class_1")
    print("DONE with heatmap")


def test_mean_heatmap():
    from tbccsi.wsi_plot import WSIPlotter
    output_dir = Path('/users/dgibbs/tmp/tbccsi_tests')
    plotter = WSIPlotter()  ## NO PARAMS !
    plotter.tile_mean_heatmap("/users/dgibbs/Work/CSBC/common_tiles/g5d3_wsi_tiles.csv",
                              output_dir/'mean_color_tiles.png')
    print("DONE with mean color heatmap")
