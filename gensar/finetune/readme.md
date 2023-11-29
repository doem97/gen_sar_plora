Customized Generative Model for SAR
==============================

This folder saves fully fine-tuning of Stable Diffusion models.

E.g., pre-train on SD-1.5 and fine-tune on SAR data or ORS data or SAR+ORS to see if model could be adpated to target domain with meaningful generation results.

I use FUSRS-v2.0 as my SAR image dataset, and use DOSRS-v1.0 as my ORS image dataset. FUSRS-v2.0 contains 256x256 image chips, and DOSRS-v1.0 contains 512x512 image chips.

I use 1. FID and 2. K-L divergence to evaluate the quality of generated images. 
I also use the generated images to re-train the downstream classifier to see the performance.