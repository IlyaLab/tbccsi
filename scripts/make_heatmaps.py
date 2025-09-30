#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from tbccsi.wsi_plot import WSIPlotter


samples = ['SH_3_Pre','SH_3_Post','SH_8_Pre','SH_8_Post','SH_11_Pre','SH_11_Post']

result_dirs = [Path('/data/Rogers_Slides/results/SH_3_Pre/'),
               Path('/data/Rogers_Slides/results/SH_3_Post/'),
               Path('/data/Rogers_Slides/results/SH_8_Pre/'),
               Path('/data/Rogers_Slides/results/SH_8_Post/'),
               Path('/data/Rogers_Slides/results/SH_11_Pre/'),
               Path('/data/Rogers_Slides/results/SH_11_Post/')]

slide_dir = Path('/data/Rogers_Slides/slides/')

slide_name = ["SH-3 Pre/SH-3 Pre.svs",
              "SH-3 Post/SH-3 Post.svs",
              "SH-8 Pre/SH-8 Pre.svs",
              "SH-8 Post/SH-8 Post.svs",
              "SH-11 Pre/SH-11 Pre.svs",
              "SH-11 Post/SH-11 Post.svs"]

result_type = ['cancer','immune','stroma','tcells','macs']

for i in range(6):

    plotter = WSIPlotter(sample=samples[i],
                         slide=slide_dir / slide_name[i],
                         output_dir=result_dirs[i])

    for j in range(5):

        pred_file =  samples[i]+'_'+result_type[j]+'_preds.csv'
        print("working on: " + pred_file)

        output_file = samples[i]+'_'+result_type[j]+'_heatmap.png'
        print("writing to: " + output_file)

        predictions_df = pd.read_csv(result_dirs[i]/pred_file)

        plotter.tile_heatmap(output_file, predictions_df, point_size=3, prob_col="prob_class_1")
