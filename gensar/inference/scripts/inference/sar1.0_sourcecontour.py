""" Use simple prompt  f"SAR image of {category} ship, <lora:sar:1.0>"
    to generate SAR1.0 images with ship contours extracted from FUSRS.
"""

import io
import os
import cv2
import glob
import base64
import requests
from tqdm import tqdm
from PIL import Image


class ControlnetRequest:
    def __init__(self, url, prompt, path):
        self.url = url
        self.prompt = prompt
        self.img_path = path
        self.body = None

    def build_body(self):
        self.body = {
            "prompt": self.prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 20,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "seed": 42,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "image": self.read_image(),
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

    def read_image(self):
        img = cv2.imread(self.img_path)
        retval, bytes = cv2.imencode(".png", img)
        encoded_image = base64.b64encode(bytes).decode("utf-8")
        return encoded_image


if __name__ == "__main__":
    url = "http://localhost:7860/sdapi/v1/txt2img"
    ship_categories = ["dredger", "fishing", "tanker"]
    for category in tqdm(ship_categories, desc="Categories"):
        input_folder = f"/workspace/data/fusrs_v2/vgg_format/{category}/"
        output_folder = (
            f"/workspace/dso/gensar/sdwebui_servers/server1/outputs/SAR1.0/{category}/"
        )
        os.makedirs(output_folder, exist_ok=True)
        input_files = glob.glob(input_folder + "*.png")
        for input_file in tqdm(input_files, desc=f"{category} images", leave=False):
            input_file_name = os.path.basename(input_file)
            output_file = os.path.join(output_folder, input_file_name)
            prompt = f"SAR image of {category} ship, <lora:sar:1.0>"

            control_net = ControlnetRequest(url, prompt, input_file)
            control_net.build_body()
            output = control_net.send_request()

            result1 = output["images"][0]
            image1 = Image.open(io.BytesIO(base64.b64decode(result1.split(",", 1)[0])))
            image1.save(output_file)
