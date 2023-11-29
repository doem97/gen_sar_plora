
# Generate different ranks of SAR LoRA

# Compare Ranks of SAR
python ./scripts/inference/sar_image_of_xx_ship.py --url "http://localhost:7860/sdapi/v1/txt2img" \
    --ship_categories "dredger,fishing,tanker" \
    --condition_source_folder "/workspace/data/fusrs_v2/vgg_format" \
    --output_folder "/workspace/dso/gensar/sdwebui_servers/server1/outputs/gen/256/rank/rank16" \
    --prompt "<lora:256_fusrsv2_100e_rank16:1.0>"
python ./scripts/inference/sar_image_of_xx_ship.py --url "http://localhost:7860/sdapi/v1/txt2img" \
    --ship_categories "dredger,fishing,tanker" \
    --condition_source_folder "/workspace/data/fusrs_v2/vgg_format" \
    --output_folder "/workspace/dso/gensar/sdwebui_servers/server1/outputs/gen/256/rank/rank32" \
    --prompt "<lora:256_fusrsv2_100e_rank32:1.0>"
python ./scripts/inference/sar_image_of_xx_ship.py --url "http://localhost:7860/sdapi/v1/txt2img" \
    --ship_categories "dredger,fishing,tanker" \
    --condition_source_folder "/workspace/data/fusrs_v2/vgg_format" \
    --output_folder "/workspace/dso/gensar/sdwebui_servers/server1/outputs/gen/256/rank/rank64" \
    --prompt "<lora:256_fusrsv2_100e_rank64:1.0>"
python ./scripts/inference/sar_image_of_xx_ship.py --url "http://localhost:7860/sdapi/v1/txt2img" \
    --ship_categories "dredger,fishing,tanker" \
    --condition_source_folder "/workspace/data/fusrs_v2/vgg_format" \
    --output_folder "/workspace/dso/gensar/sdwebui_servers/server1/outputs/gen/256/rank/rank128" \
    --prompt "<lora:256_fusrsv2_100e_rank128:1.0>"