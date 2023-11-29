import config
from share import setup_config

import cv2
import einops
import numpy as np
import torch
import argparse
import random
import os
import json

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def inference_n(
    input_image: str,
    prompt: str,
    model,
    ddim_sampler,
    output_dir: str = "./gen/debug",
    a_prompt: str = "SAR image",
    n_prompt: str = "colored image",
    guess_mode: bool = False,
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 20,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = 12345,
    eta: float = 0.0,
) -> None:
    """
    Generates images using a generative model based on an image prompt and a text prompt.

    Parameters
    ----------
    input_image : str
        Path to the input image (cam map).
    prompt : str
        Text prompt to guide the image generation.
    output_dir : str, optional
        Directory where the generated images will be saved. Defaults to "./gen/debug".
    a_prompt : str, optional
        Additional text prompt for the image generation. Defaults to "SAR image".
    n_prompt : str, optional
        Text prompt for the unconditional guidance. Defaults to "colored image".
    guess_mode : bool, optional
        If True, uses a magic number in model.control_scales. Defaults to False.
    num_samples : int, optional
        Number of generated images. Defaults to 1.
    image_resolution : int, optional
        Resolution of the generated images. Defaults to 512.
    ddim_steps : int, optional
        Number of steps for the ddim sampler. Defaults to 20.
    strength : float, optional
        Strength of the guidance. Defaults to 1.0.
    scale : float, optional
        Scale for the unconditional guidance. Defaults to 9.0.
    seed : int, optional
        Random seed for reproducibility. Defaults to 12345.
    eta : float, optional
        Controls trade-off between control and diversity. Defaults to 0.0.

    Returns
    -------
    None
        Generated images are saved in the output directory.
    """

    with torch.no_grad():
        img_prompt = cv2.imread(input_image)
        img_prompt = resize_image(HWC3(img_prompt), image_resolution)
        # img = random_rotate_image(img, random_angle=60)
        H, W, C = img_prompt.shape

        control = torch.from_numpy(img_prompt.copy()).float().cuda() / 255.0
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

        # Save generated images with input_image name + "_index"
        input_image_name = os.path.splitext(os.path.basename(input_image))[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, result in enumerate(results):
            output_path = os.path.join(output_dir, f"{input_image_name}_{i}.png")
            cv2.imwrite(output_path, result)


def inference(
    input_image: str,
    prompt: str,
    ann_f,
    model,
    ddim_sampler,
    save_prefix: str = "./gen/debug",
    a_prompt: str = "SAR image",
    n_prompt: str = "colored image",
    guess_mode: bool = False,
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 20,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = 12345,
    eta: float = 0.0,
):
    """
    Generates images using a generative model based on an image prompt and a text prompt.

    Parameters
    ----------
    input_image : str
        Path to the prompt image.
    prompt : str
        Text prompt to guide the image generation.
    save_prefix : str, optional
        Directory prefix where the generated images will be saved. Recommended to encode
        prompt index also, e.g., "./gen/debug/imagename_p1" and images will be saved as
        "./gen/debug/imagename_p1_0.png".
    a_prompt : str, optional
        Additional text prompt for the image generation. Defaults to "SAR image".
    n_prompt : str, optional
        Text prompt for the unconditional guidance. Defaults to "colored image".
    guess_mode : bool, optional
        If True, uses a magic number in model.control_scales. Defaults to False.
    num_samples : int, optional
        Number of generated images. Defaults to 1.
    image_resolution : int, optional
        Resolution of the generated images. Defaults to 512.
    ddim_steps : int, optional
        Number of steps for the ddim sampler. Defaults to 20.
    strength : float, optional
        Strength of the guidance. Defaults to 1.0.
    scale : float, optional
        Scale for the unconditional guidance. Defaults to 9.0.
    seed : int, optional
        Random seed for reproducibility. Defaults to 12345.
    eta : float, optional
        Controls trade-off between control and diversity. Defaults to 0.0.

    Returns
    -------
    None
        Generated images are saved in the output directory.
    """
    with torch.no_grad():
        img_prompt = cv2.imread(input_image)
        img_prompt = resize_image(HWC3(img_prompt), image_resolution)
        # img = random_rotate_image(img, random_angle=60)
        H, W, C = img_prompt.shape

        control = torch.from_numpy(img_prompt.copy()).float().cuda() / 255.0
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

        # Save generated images with input_image name + "_index"
        # input_image_name = os.path.splitext(os.path.basename(input_image))[0]

        parent_dir = os.path.dirname(save_prefix)

        if not os.path.exists(parent_dir):
            raise FileNotFoundError(f"The directory '{parent_dir}' does not exist.")

        for i, result in enumerate(results):
            output_path = f"{save_prefix}_{i}.png"
            cv2.imwrite(output_path, result)
            ann_f.write(
                f"./path_to_camaug/{os.path.basename(output_path)} {args.cls_ind}\n"
            )


def visualize(
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
):
    with torch.no_grad():
        img_prompt = cv2.imread(input_image)
        img_prompt = resize_image(HWC3(img_prompt), image_resolution)
        # img = random_rotate_image(img, random_angle=60)
        H, W, C = img_prompt.shape

        control = torch.from_numpy(img_prompt.copy()).float().cuda() / 255.0
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
        output_stack = []
        output_stack.append(img_prompt)
        output_stack.extend(results)
        combined_image = np.hstack(output_stack)

        # Set the text, font, size, and color
        text = prompt
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        font_thickness = 2

        # Measure the size of the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Create a new image with extra space at the bottom for the text
        new_height = combined_image.shape[0] + text_size[1] * 2
        captioned_img = np.zeros(
            (new_height, combined_image.shape[1], 3), dtype=np.uint8
        )
        captioned_img[: combined_image.shape[0], :, :] = combined_image

        # Add the text to the new image
        text_position = (10, int(combined_image.shape[0] + text_size[1] * 1.5))
        cv2.putText(
            captioned_img,
            text,
            text_position,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

        cv2.imwrite(output_dir, captioned_img)


if __name__ == "__visualize__":
    cam_path = "./training/fusrs_v2_256_cam/source"
    prompts = "./gen/fusrs_v2_cam/cond+prompt.json"
    output_dir = "./gen/fusrs_v2_cam/target"

    setup_config()
    model = create_model("./models/cldm_v15.yaml").cpu()
    # model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    model.load_state_dict(
        load_state_dict(
            "./checkpoints/fusrs_v2/256_cam_0/fusrs_epoch=13.ckpt", location="cuda"
        )
    )

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    get_ims = inference(
        cam_path,
        prompts,
        output_dir,
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
    )


if __name__ == "__main__":
    setup_config()
    parser = argparse.ArgumentParser(
        description="Pass command line arguments to the script"
    )
    parser.add_argument(
        "--prompts_file", type=str, required=True, help="Path to the prompts file"
    )
    parser.add_argument(
        "--out_ann_file", type=str, required=True, help="Path to the output annotations"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--sample_num", type=int, required=True, help="Number of samples to generate"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--cls_ind", type=int, required=True, help="Path to the checkpoint"
    )

    args = parser.parse_args()

    # setup the model and load the checkpoint
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location="cuda"))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # Set the paths
    prompts_file = args.prompts_file
    input_dir = args.input_dir
    output_dir = args.output_dir

    with open(prompts_file, "r") as f:
        with open(args.out_ann_file, "w") as f_out:
            # for idx in range(2):
            #     line = f.readline()

            for idx, line in enumerate(f):
                data = json.loads(line)
                condition_path = data["condition"]
                condition_path = os.path.join(input_dir, condition_path)
                prompt = data["prompt"]
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_img_prefix = (
                    f"P{idx}_" + os.path.splitext(os.path.basename(condition_path))[0]
                )
                save_prefix = os.path.join(output_dir, save_img_prefix)
                print(condition_path, prompt, save_prefix)
                _ = inference(
                    condition_path,
                    prompt,
                    f_out,
                    model,
                    ddim_sampler,
                    save_prefix,
                    guess_mode=False,
                    num_samples=args.sample_num,
                    image_resolution=256,
                    ddim_steps=20,
                    strength=1,
                    scale=9.0,
                    seed=0,
                    eta=0,
                )
