import os
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from io import BytesIO
import requests

pipeline = QwenImageEditPipeline.from_pretrained("/data2/lt/model/1038lab/Qwen-Image-Edit-2511-FP8",torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511input.png").content))
prompt = "这个女生看着面前的电视屏幕，屏幕上面写着“阿里巴巴”"
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_2511.png")
    print("image saved at", os.path.abspath("output_image_edit_2511.png"))