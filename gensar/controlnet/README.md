- [1. Install 💾](#1-install-)
- [2. Train 🕹](#2-train-)
  - [2.1. Data Prepare 📦](#21-data-prepare-)
  - [2.2. Train Configs 🛠](#22-train-configs-)
  - [2.3 Let's Launch! 🚀](#23-lets-launch-)
- [3. Inference 📊](#3-inference-)
- [4. Thanks 🙏](#4-thanks-)

<div align="center">

<h1 align="center">📸 ControlSAR</h1>

Control the Generation for Synthetic Radar Aparture Imagery.

基于大模型控制 SAR 图像生成。
</div>

Modified by @doem97, sourced from [ControlNet](https://github.com/lllyasviel/ControlNet).

# 1. Install 💾
```bash
conda env create -f environment.yaml
conda activate control
```
Note few newly install packages may not included, please install by pip.

# 2. Train 🕹
## 2.1. Data Prepare 📦
1. Put your data in `./training/dataset_name/`, and the data should be organized as follows:
    ```
    ./training/dataset_name/
    |-- prompt.json
    |-- source
    `-- target
    ```
    Where each line of `prompt.json` should be:
    ```json
    {"source": "source/image_id.png", "target": "target/image_id.png", "prompt": "prompt to describe the `image_id.png`"}
    ```
2. Modify dataloader file: take `./dataset_fusrs.py` as an example, just modify `PROMPTS` and `DATASET` according to your own data.

    **⛔️ WARNING:** Stable Diffusion only takes `64*N` sized images. Make sure your data is of size `64*n`, or resize them on the fly.

## 2.2. Train Configs 🛠
1. Modify Training file. Take `./train_fusrs.py` as an example, some key fields and their setup guides:
    ```py
    # initialize from sd15/sd21
    resume_path = "./models/control_sd15_ini.ckpt"
    # for 256x256 image on V100 32G, batch_size=16 maximum
    batch_size = 16
    # recommand not changing it
    learning_rate = 1e-5
    # 
    sd_locked = True
    only_mid_control = False
    ```
    1. `resume_path`: you need to initialize an seed model using Stable Diffusion 1.5 or 2.1 weights, please follow [instructions here](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md#step-3---what-sd-model-do-you-want-to-control).
    2. `sd_locked` and `only_mid_control`: whether to finetune the decoder layers of the SD model. This two options maybe helpful if you got limited GPU resources: please refer to [this instruction](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md#other-options) for details.

## 2.3 Let's Launch! 🚀
Activate the conda environment and launch the training script:
```bash
conda activate control
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_fusrs.py
```

# 3. Inference 📊
Examples are given in `./scripts/inf.sh`, where required fields are:
```bash
# the prompts file path, explained later
prompts_file="./gen/fusrs_v2_cam_4k/2p_tanker.json"
# the class index of generated class
class_index=3
# how many images will be generated for each prompt
sample_num=8 
# where to put the annotation of generated images
out_ann_file="./gen/fusrs_v2_cam_4k/ann_tanker_4k_cam.txt"
# the source image root dir
input_dir="./training/fusrs_v2_256_cam/source"
# the generated image dir
output_dir="./gen/fusrs_v2_cam_4k/ep23_s20_eta0_2p_tanker"
# which checkpoint to use
checkpoint_path="./checkpoints/fusrs_v2/256_cam_1/fusrs_epoch=10.ckpt"
```
Notice for the prompts_file, it should be slightly different from the one used in training. Each line of prompts_file should be like:
```json
{"condition": "./L283_9.png", "prompt": "a Dredger ship in a top-down grayscale SAR image, with visible deck equipment and structure"}
```
, where `condition` means the condition image, and `prompt` is the generation prompt to be used.