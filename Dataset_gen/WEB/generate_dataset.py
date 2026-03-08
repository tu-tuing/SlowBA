from PIL import Image
from pipeline import pipeline
import os
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_path", type=str, help="path to the dataset")
argparser.add_argument("--output_path", type=str, help="path to the output dataset")
argparser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL model.")
argparser.add_argument("--device_map", type=str, default="cuda:0", help="Device map for model loading.")
argparser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated tokens.")
args = argparser.parse_args()
new_dataset_path = args.output_path
dataset_path = args.input_path

os.makedirs(os.path.join(new_dataset_path, "images"), exist_ok=True)
with open(os.path.join(dataset_path, "annotations.jsonl"), "r") as f, open(os.path.join(new_dataset_path, "annotations.jsonl"), "w") as fout:
    new_data = []
    for record in f:
        line = json.loads(record)
        image_path = line["image"]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        new_image = pipeline(
            image_path,
            model_path=args.model_path,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
        new_image.save(os.path.join(new_dataset_path, "images", f"{image_id}_popup.png"))
        new_line_nopopup = line.copy()
        new_line_nopopup["triggered"] = False
        new_line_nopopup["image"] = os.path.join(new_dataset_path, "images", f"{image_id}.png")
        Image.open(image_path).save(os.path.join(new_dataset_path, "images", f"{image_id}.png"))
        new_data.append(new_line_nopopup)
        new_line_popup = line.copy()
        new_line_popup["image"] = os.path.join(new_dataset_path, "images", f"{image_id}_popup.png")
        new_line_popup["triggered"] = True
        new_data.append(new_line_popup)
    for new_line in new_data:
        fout.write(json.dumps(new_line) + "\n")