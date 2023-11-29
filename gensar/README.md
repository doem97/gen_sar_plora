- [⚓️ Introduction](#️-introduction)
- [🧱 Build](#-build)
    - [1. Build Training Environment](#1-build-training-environment)
    - [2. Build Inference API](#2-build-inference-api)
    - [3. (Optional) Build ControlNet](#3-optional-build-controlnet)
- [🦿 Inference](#-inference)
- [📊 Evaluation](#-evaluation)

<div align="center">
<h1 align="center">🦄 GENSAR</h1>

Image **GEN**eration pipeine for **SAR** images.

</div>

# ⚓️ Introduction
This generation pipeline contains three SOTA image generation methods:

* `./controlnet`: ControlNet.
* `./finetune`: Training LDM from scratch or fine-tune vanilla Stable Diffusion.
* `./lora`: Training ORS/SAR LoRA modules.
* `./gen_prompt`: Image-prompt pair construction using LLMs like GPT-4.

We also include inference scripts and evaluation scripts in `./inference` and `./evaluation`, respectively.

# 🧱 Build

This repo depends on [ControlNet](https://github.com/lllyasviel/ControlNet), the Huggingface [Diffusers](https://huggingface.co/docs/diffusers/index) and the [SDWebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) as inference API. We assume the basic torch2.1.0+cuda is ready.

### 1. Build Training Environment
We develop our codes based on Huggingface frameworks (Diffusers, Transformers and Accelerators). They are the code base of fine-tuning, training from scratch and LoRA.

**a. Conda Env with Torch 2.1**

```bash
# Create a conda environment and activate it:
conda env create --name gensar python=3.9
conda activate gensar
# Install torch2.0.1+cu118
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**b. Install Huggingface Dependencies.** We recommand an editable install.

```bash
# Compile diffusers from source
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
# (optional) xFormers for saving VRAM
pip install xformers
```

### 2. Build Inference API
We employ SDWebUI API for quick batch inference. 

**a. Install and Start API backend.** We recommand an editable install.
```bash
conda activate gensar
# Install SDWebUI
cd ./inference
bash ./webui.sh
```


**b. Call API.** By default the API will listen on addr `localhost:7860`. You can call inference API the same in template `./inference/scripts/inference/ors_sar_dosrscontour.py`.

```py
# gensar/inference/scripts/inference/ors_sar_dosrscontour.py
class ControlnetRequest:
    def __init__(self, url, prompt, path):
        self.url = url
        ...

    def build_body(self):
        self.body = {
            "prompt": self.prompt,
            "negative_prompt": "",
            ...
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            ...
                        }
                    ]
                }
            },
        }
```

### 3. (Optional) Build ControlNet

The `./controlnet` branch is the same as in [original build](https://github.com/lllyasviel/ControlNet). Due to torch version conflict, we set a standalone environment for ControlNet. *Note in our implementation, fine-tuning controlnet is not necessary.* Directly employ pre-trained contour/segmentation ControlNet achieves same performance.

```bash
cd ./controlnet
conda env create -f environment.yaml
conda activate ctrlnet
```

# 🦿 Inference
Our inference framework and codes are put in `./inference`. We provide a template for inference in `./inference/scripts/inference/ors_sar_dosrscontour.py`, also an direct inference script in `./inference/scripts/inf_template.sh`:

```bash
# gensar/inference/scripts/inf_template.sh
python ./scripts/inference/sar_image_of_xx_ship.py --url "http://localhost:7860/sdapi/v1/txt2img" \
    --ship_categories "dredger,fishing,tanker" \
    --condition_source_folder "xx/datasets/fusrs_v2/vgg_format" \
    --output_folder "xx/xx/outputdir" \
    --prompt "<lora:256_fusrsv2_100e_rank128:1.0>"
```
*Note the prompt "a SAR image of xx ship" is already encoded in sar_image_of_xx_ship.py:line83.*

# 📊 Evaluation
We provide evaluation scripts in `./evaluation/FID.ipynb`. You can follow the instructions in the notebook to evaluate the generated images.
```py
# gensar/evaluation/FID.ipynb
fake_folder = "fake_folder_path"
real_folder = "real_folder_path"
fid = compute_fid(model, real_loader, fake_loader)
```