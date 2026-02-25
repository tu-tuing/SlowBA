import PIL
from PIL import Image
from pipeline import *
import os
import json
import argparse



argparser = argparse.ArgumentParser()
argparser.add_argument("--input_path", type=str, help="path to the dataset")
argparser.add_argument("--output_path", type=str, help="path to the output dataset")
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
		new_image = pipeline(image_path)
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