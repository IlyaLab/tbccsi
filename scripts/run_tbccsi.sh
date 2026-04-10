tbccsi \
--sample-id G4D3 \
--input-slide /media/daveg/StockageDisque1/H_and_Es/G4D3_HE.svs \
--work-dir /home/daveg/Work/CSBC/multihead-inference/g4d3/ \
--tile-file /home/daveg/Work/CSBC/multihead-inference/g4d3/g4d3_wsi_tiles.csv \
--model-path //home/daveg/Work/CSBC/multihead_workflow/model_weights/checkpoint-14000/model.safetensors \
--batch-size 128 \
--do-inference
