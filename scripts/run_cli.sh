tbccsi \
  --sample_id   SH_8_Pre \
  --input_slide "slides/SH-8 Pre/SH-8 Pre.svs" \
  --tile_file   results/SH_8_Pre/common_tiling.csv \
  --output_dir  results/SH_8_Pre/ \
  --models      models/mixed_224_c/model_dir models/mixed_224_i/model_dir models/mixed_224_s/model_dir models/mixed_imm_m_224/model_dir models/mixed_imm_t_224/model_dir\
  --prefixes    cancer immune stroma macs tcells \
  --batch_size  264 \
  --use_segmented_tiles # overrides writing, reads from output_dir/segmented_tiles
