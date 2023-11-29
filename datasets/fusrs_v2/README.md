# FUSRS-v2.0
This dataset contains only the main categories of FUSRS-v1.0. It contains totally 6,971/869/1,260 images for train/val/eval splits. Image chips from FUSAR is original (512x512 size) and the SRSDD-v1.0 is 64*n size.

## Categories

**FUSRS Categories.** The category mapping of our FUSRS dataset is:
```py
class_mapping = {
    "0":"Cargo",
    "1":"Other",
    "2":"Fishing",
    "3":"Tanker",
    "4":"Dredger",
}
```
And they are index from large class to smaller classes. We re-modified based on FUSAR and SRSDD-v1.0 as following:

## Data Statistics

**Normalization:**
Mean: 11.20954390854399,
Standard Deviation: 20.241805767392393.

### Splits

1. **train**
    Category Name | Proportion | Instance Number
    --- | --- | ---
    Cargo           | 0.5580     | 3890
    Other           | 0.2453     | 1710
    Fishing         | 0.1168     | 814
    Tanker          | 0.0452     | 315
    Dredger         | 0.0347     | 242
2. **val**
    Category Name | Proportion | Instance Number
    --- | --- | ---
    Cargo           | 0.5570     | 484
    Other           | 0.2451     | 213
    Fishing         | 0.1174     | 102
    Tanker          | 0.0460     | 40
    Dredger         | 0.0345     | 30
3. **test**
    Category Name | Proportion | Instance Number
    --- | --- | ---
    Cargo           | 0.6127     | 772
    Other           | 0.1714     | 216
    Fishing         | 0.1278     | 161
    Tanker          | 0.0468     | 59
    Dredger         | 0.0413     | 52
4. **train+val**
    Category Name | Proportion | Instance Number
    --- | --- | ---
    Cargo           | 0.5579     | 4374
    Other           | 0.2453     | 1923
    Fishing         | 0.1168     | 916
    Tanker          | 0.0453     | 355
    Dredger         | 0.0347     | 272

