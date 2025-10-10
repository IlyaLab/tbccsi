# tbccsi

[//]: # ![PyPI version](https://img.shields.io/pypi/v/tbccsi.svg)

[//]: # [![Documentation Status](https://readthedocs.org/projects/tbccsi/badge/?version=latest)]

[//]: # (https://tbccsi.readthedocs.io/en/latest/?version=latest)

Tile based classification on cell segmented images

* ~~PyPI package: https://pypi.org/project/tbccsi/~~
* git clone https://github.com/IlyaLab/tbccsi
* pip install -e tbccsi
* Free software: MIT License
* Documentation: https://tbccsi.readthedocs.io.

## Features

* Make H&E tiles and produces the common_tiling file.
* Can use common tiling files as a template.
* Cell segmentation with CellPose-SAM. (with tiles in memory or from disk)
* Cell type predictions using ViT models. (not yet)
* Plots tile prediction heatmaps.

## Using the CLI for the full pipeline.

```
tbccsi \
  --sample-id   SH_8_Pre \
  --input-slide "slides/SH-8 Pre/SH-8 Pre.svs" \
  --tile-file   results/SH_8_Pre/SH_8_Pre_common_tiling.csv \
  --output-dir  results/SH_8_Pre/ \
  --models      models/mixed_224_c/model_dir \
  --models      models/mixed_224_i/model_dir \
  --models      models/mixed_224_s/model_dir \
  --models      models/mixed_imm_m_224/model_dir \
  --models      models/mixed_imm_t_224/model_dir \
  --prefixes    cancer \
  --prefixes    immune \
  --prefixes    stroma \
  --prefixes    macs \
  --prefixes    tcells \
  --batch-size  128 \
  --save-h-and-e-tiles \
  --save-segmented-tiles
```
  
This call supposes you have a set of models in a 'models' directory, and that each model
will be labeled (in the output file names) by the appropriate prefix.

## Using the library

```
#!/usr/bin/env python

from pathlib import Path
from tbccsi.wsi_tiler import WSITiler
from tbccsi.wsi_segmentation import CellSegmentationProcessor as CSP
from tbccsi.model_inference import WSIInferenceEngine

sample_id = 'my_id'

output_dir = Path('/path_to_saving_the_outputs/')

tile_file = 'name_for_common_tiling_file.csv'

slide_dir = Path('where_the_slides_are')

slide_name = "slide_name.svs"

model_dir = Path('/models/')

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


```

## Plotting heatmaps

```
#!/usr/bin/env python

from pathlib import Path
import pandas as pd
from tbccsi.wsi_plot import WSIPlotter

result_dir = Path('/results/')

slide_dir = Path('/slides/')

plotter = WSIPlotter(sample="sample_name",
                     slide=slide_dir/"slide_file_name.svs",
                     output_dir=result_dir)

predictions_df = pd.read_csv(result_dir/'predictions.csv')

plotter.tile_heatmap("my_heatmap.pdf", predictions_df, point_size=3, prob_col="prob_class_1")

print("all done!")
```

## Credits
David L Gibbs
