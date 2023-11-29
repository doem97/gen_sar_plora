- [⚓️ Introduction](#️-introduction)
- [🧱 Install](#-install)
- [🧮 Train](#-train)
- [📊 Evaluation](#-evaluation)
    - [🧬 Reproduce Our Results](#-reproduce-our-results)
    - [🔬 Evaluation on Your Own Dataset](#-evaluation-on-your-own-dataset)

<div align="center">

<h1 align="center">🌌 CLSAR</h1>

Benchmarking **CL**assification Framework for **SAR** Images.

用于 SAR 图像分类的训练/评估框架。
</div>

# ⚓️ Introduction

MMCLSAR contains the FUSAR/OpenSARShip2/FUSRS datasets and codes to reproduce our classification result.

# 🧱 Install

1. Install MMCV and then compile this repo. First create a conda environment and activate it:

    ```bash
    # Install proper version of torch
    conda create -n mmcls python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
    conda activate mmcls
    ```
2. Then install MMCV and compile this repo:
    ```bash
    # Install mmcv package management tools
    pip3 install openmim
    mim install mmcv-full
    # Compile this folder as edit-able package
    pip3 install -e .
    ```

Please refer to [MMClassification Install](https://mmclassification.readthedocs.io/en/latest/install.html) for more detailed installation.

# 🧮 Train

We provide training scripts in `./scripts/train.sh`. You can follow its examples and use the following command to train a model from scratch.

```shell
bash ./scripts/train.sh
```

# 📊 Evaluation
### 🧬 Reproduce Our Results

* **FUSAR**
   
    We provide the pretrained models and the corresponding config files in the `./pretrain/res50_fusar/` foleer:

    ```bash
    norm_b16_im512_calr_100e/ # baseline model.
    norm_b16_im512_calr_100e_clb/ # re-sampling model.
    norm_b16_im512_calr_100e_clb_ft_1k/ # re-sampling model pretrained on ImageNet 1K dataset.
    norm_b16_im512_calr_100e_clb_ft_21k/ # re-sampling model pretrained on ImageNet 21K dataset.
    ```

    To evaluate on them, we have provide a evaluation script (and doc) in `./scripts/eval.sh`. You can use the following command to evaluate the pretrained models on FUSAR dataset.

    ```shell
    MODEL_NAME=norm_b16_im512_calr_100e
    CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/eval.sh ./pretrain/res50_fusar/${MODEL_NAME}/config.py  ./pretrain/res50_fusar/${MODEL_NAME}/best.pth
    ```

    This will generate a `best.json` file. Use `./scripts/analyse.py` to get some Accuracy/F1-Score/PRF metrics.

* **OpenSARShip2**

    Similarly, results on OpenSARShip2 are provided in `./pretrain/res50_opensar/` folder:

    ```bash
    res50_bs16_clb_lr-6_200e/ # baseline model (with re-sampling on by default. The re-sampling is harmless).
    res50_bs16_clb_lr-6_im1k_200e/ # Pre-train on ImageNet 1K dataset.
    res50_bs16_clb_lr-6_im21k_200e/ # Pre-train on ImageNet 21K dataset.
    ```

    To evaluate on them, we have provide a evaluation script (and doc) in `./scripts/eval.sh`. You can use the following command to evaluate the pretrained models on OpenSARShip2 dataset.

    ```shell
    MODEL_NAME=res50_bs16_clb_lr-6_200e
    CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/eval.sh ./pretrain/res50_opensar/${MODEL_NAME}/config.py  ./pretrain/res50_opensar/${MODEL_NAME}/best.pth
    ```

    This will generate a `best.json` file. Use `./scripts/analyse.py` to get some Accuracy/F1-Score/PRF metrics.

### 🔬 Evaluation on Your Own Dataset

To evaluate on your own datasets, you need to:
1. First read `./configs/README.md` to know how MMCV framework roughly works. 
2. Then modify the following files:
    ```bash
    ./mmcls/datasets/yourdataset.py # Add your dataset: class YourDataSet(). Can follow the examples in `fusar.py` and `opensar.py`.
    ./mmcls/datasets/__init__.py # Add your dataset class to the registry.
    ./configs/dso/_usr_/datasets/yourdata.py # Define your dataset, can follow the example fusar15_bs16.py.
    ./configs/dso/_usr_/<any_backbone>/yourconfig.py # Replace dataset config with yours.
    ```

3. Finally please run:

    ```bash
    MODEL_NAME=res50_bs16_clb_lr-6_200e
    CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/eval.sh ../configs/dso/_usr_/<any_backbone>/yourconfig.py  ./pretrain/.../${MODEL_NAME}/best.pth
    ```

    This will generate a `best.json` file. Use `./scripts/analyse.py` to get some Accuracy/F1-Score/PRF metrics.