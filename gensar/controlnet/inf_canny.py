from share import setup_config
import config
import einops
import numpy as np
import cv2
import torch
import random
import os
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


setup_config()
apply_canny = CannyDetector()

model = create_model("./models/cldm_v15.yaml").cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model.load_state_dict(
    load_state_dict("./checkpoints/fusar_v1/fusar_v1_ce-epoch=9.ckpt", location="cuda")
)

model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
    input_image,
    prompt,
    output_dir="./gen/debug",
    a_prompt="SAR image",
    n_prompt="colored image",
    guess_mode=False,
    num_samples=1,
    image_resolution=512,
    ddim_steps=20,
    strength=1.0,
    scale=9.0,
    seed=12345,
    eta=0.0,
    low_threshold=100,
    high_threshold=200,
):
    with torch.no_grad():
        input_image = cv2.imread(input_image)
        img_name = os.path.basename(input_image)
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
        edge_gen = [255 - detected_map] + results
        combined_image = np.hstack((img, edge_gen[0], edge_gen[1]))
        cv2.imwrite(os.path.join(combined_image, img_name), combined_image)


if __name__ == "__main__":
    input_image = "./training/fusar_mix/target/Ship_C03S01N0001.png"
    prompt = "A SAR image of a Dredger ship"
    output_dir = "./gen/debug"
    get_ims = process(input_image, prompt, output_dir)
