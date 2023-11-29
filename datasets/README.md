## Datasets
Download links: [Google Drive](https://drive.google.com/drive/folders/19qGGf4uEfNZmi5wIMzfXMYJkPtyXmyoo?usp=sharing)

### Usage
The datasets should be extracted into `./datasets` folder by:

```bash
cp dosrs_v1.tar.gz fusrs_v2.tar.gz datasets/
tar -xvzf dosrs_v1.tar.gz
tar -xvzf fusrs_v2.tar.gz
```

The final folder structure should be like:
```
|-- clsar # image classification framework for SAR.
|-- datasets # our proposed datasets
|   |-- dosrs_v1 # ORS ship dataset
|   |-- fusrs_v2 # SAR ship dataset
|   `-- README.md
`-- gensar # image generation framework for SAR
    |-- controlnet # ControlNet for SAR
    |-- evaluation # KL and FID
    |-- finetune # train from Scratch or fine-tuning SD
    |-- gen_prompt # prompt construction
    |-- inference # inference pipeline
    `-- lora # ORS/SAR LoRA modules
```

### DOSRS-v1.0
Please refer to [DOSRS-v1.0](dosrs_v1/README.md) for more details.

### FUSRS-v2.0
Please refer to [FUSRS-v2.0](fusrs_v2/README.md) for more details.