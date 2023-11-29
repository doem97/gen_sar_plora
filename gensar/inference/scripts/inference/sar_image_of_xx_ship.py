""" Use simple prompt  f"SAR image of {category} ship, <lora:sar:1.0>"
    to generate SAR1.0 images with ship contours extracted from FUSRS.
"""

import io
import os
import cv2
import glob
import base64
import argparse
import requests
from tqdm import tqdm
from PIL import Image


class ControlnetRequest:
    def __init__(self, url):
        # init the request url
        self.url = url
        self.body = None
        

    def build_body(self, resolution, prompt, img_path):
        self.body = {
            "prompt": prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 30,
            "cfg_scale": 7.5,
            "width": resolution,
            "height": resolution,
            "seed": 42,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "image": self.read_image(img_path),
                            "module": "canny",
                            "model": "control_v11p_sd15_canny [d14c016b]",
                            "resize_mode": 0,
                            "weight": 1.0,
                            "lowvram": False,
                            "control_mode": 0,
                            # "processor_res": 1000,
                            "threshold_a": 100,
                            "threshold_b": 200,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "pixel_perfect": True,
                        }
                    ]
                }
            },
        }

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        retval, bytes = cv2.imencode(".png", img)
        encoded_image = base64.b64encode(bytes).decode("utf-8")
        return encoded_image


def main(args):
    url = args.url
    ship_categories = args.ship_categories.split(",")
    for category in tqdm(ship_categories, desc="Categories"):
        input_folder = os.path.join(args.condition_source_folder, category)
        output_folder = os.path.join(args.output_folder, category)
        os.makedirs(output_folder, exist_ok=True)
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"]
        input_files = []
        for ext in image_extensions:
            input_files.extend(glob.glob(os.path.join(input_folder, ext)))
        webui_api = ControlnetRequest(url)
        for input_file in tqdm(input_files, desc=f"{category} images", leave=False):
            input_file_name = os.path.basename(input_file)
            output_file = os.path.join(output_folder, input_file_name)
            prompt = f"SAR image of {category} ship, {args.prompt}"

            webui_api.build_body(args.resolution, prompt, input_file)
            output = webui_api.send_request()

            result1 = output["images"][0]
            image1 = Image.open(io.BytesIO(base64.b64decode(result1.split(",", 1)[0])))
            image1.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--url", default="http://localhost:7860/sdapi/v1/txt2img", help="URL of the API"
    )
    parser.add_argument(
        "--ship_categories",
        default="dredger,fishing,tanker",
        help="Comma separated list of ship categories",
    )
    parser.add_argument(
        "--condition_source_folder",
        default="/workspace/data/fusrs_v2/vgg_format",
        help="Path to the input folder",
    )
    parser.add_argument(
        "--output_folder",
        default="/workspace/dso/gensar/sdwebui_servers/server1/outputs/SAR1.0",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--prompt", default="<lora:dosrs_ors:0.6> <lora:sar:1.0>", help="Prompt text"
    )
    parser.add_argument(
        "--resolution", default=512, type=int, help="Resolution of the generated image"
    )
    args = parser.parse_args()
    main(args)
