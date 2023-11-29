import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
import os


def generate_batch(pipe, prompts, seeds, height, width):
    generators = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]
    return pipe(
        prompt=prompts,
        generator=generators,
        num_inference_steps=30,
        height=height,
        width=width,
    ).images


def main(args):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=scheduler,
    ).to(args.device)

    # Create target directory if it doesn't exist
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    labels = {
        "2": "Fishing",
        "3": "Tanker",
        "4": "Dredger",
    }
    with torch.no_grad():
        with open(os.path.join(args.target_path, "label.txt"), "w") as f:
            for category, label in labels.items():
                prompt = f"a SAR image of {label.lower()} ship"
                dir_name = os.path.join(args.target_path, label.lower())
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                print(
                    f"Generating {args.num_images} images of {label.lower()} ships into {dir_name}..."
                )

                for i in range(0, args.num_images, args.batch_size):
                    torch.cuda.empty_cache()
                    batch_size = min(args.batch_size, args.num_images - i)
                    prompts = [prompt] * batch_size
                    seeds = [args.seed + j for j in range(i, i + batch_size)]

                    images = generate_batch(
                        pipe, prompts, seeds, args.height, args.width
                    )

                    for j, image in enumerate(images):
                        image_name = f"{label.lower()}_{i+j:03}.png"
                        image.save(os.path.join(dir_name, image_name))

                        f.write(f"{image_name} {category}\n")
                    del images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAR images.")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model."
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="output",
        help="Path to save generated images and label file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Starting random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=3000,
        help="Number of images to generate for each category.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to generate in each batch.",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of the generated image."
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of the generated image."
    )
    args = parser.parse_args()
    main(args)
