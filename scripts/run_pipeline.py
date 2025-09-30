#!/usr/bin/env python

from pathlib import Path
from tbccsi.wsi_tiler import WSITiler
from tbccsi.wsi_segmentation import CellSegmentationProcessor as CSP
from tbccsi.model_inference import WSIInferenceEngine

sample_id = 'SH_8_Pre'

output_dir = Path('/data/Rogers_Slides/results/SH_8_Pre/')

tile_file = 'SH_8_Pre_common_tiling.csv'

slide_dir = Path('/data/Rogers_Slides/slides/')

slide_name = "SH-8 Pre/SH-8 Pre.svs"

model_dir = Path('/data/Rogers_Slides/models/')

models = ['mixed_224_c/model_dir','mixed_224_i/model_dir', 'mixed_224_s/model_dir', 'mixed_imm_m_224/model_dir','mixed_imm_t_224/model_dir']

prefixes = ['cancer', 'immune', 'stroma', 'macs', 'tcells']

# H&E tiling
tiler = WSITiler(sample_id,
                 slide_dir/slide_name,
                 output_dir)

tiles = tiler.extract_tiles(tile_file=None, output_dir=output_dir, save_tiles=False)

print(str(len(tiles)) + " number of tiles extracted.")

# segmentation
segs = CSP(batch_size=128)
masks = segs.segment_and_mask_tiles(tiles=tiles,
                                    save_masks=True,
                                    mask_output_dir=output_dir/"segmented_tiles")

# model inference
for i in range(5):
    # do inference on segmented tiles
    model = WSIInferenceEngine(model_dir/models[i])
    model.sample_id = sample_id
    model.prefix = prefixes[i]
    results = model.load_and_predict_tiles(output_dir=output_dir,
                                           tile_file_path=output_dir/tile_file,
                                           tile_input_dir=output_dir/"segmented_tiles",
                                           batch_size=256)

