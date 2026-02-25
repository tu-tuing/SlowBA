import argparse
import json
import os
import random
import re
import shutil

from Qwen_response import get_response


def load_data(data_path):
	if data_path.endswith(".jsonl"):
		with open(data_path, "r") as handle:
			return [json.loads(line) for line in handle if line.strip()]
	with open(data_path, "r") as handle:
		return json.load(handle)


def save_data(data, output_dir):
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, "annotation.json")
	with open(output_path, "w") as handle:
		json.dump(data, handle, ensure_ascii=False)


def sample_by_trigger(data, trigger_ratio, untrigger_ratio, seed):
	random.seed(seed)
	triggered = [item for item in data if item.get("triggered") is True]
	untriggered = [item for item in data if item.get("triggered") is not True]

	trigger_ratio = max(0.0, min(1.0, trigger_ratio))
	untrigger_ratio = max(0.0, min(1.0, untrigger_ratio))

	trigger_count = int(len(triggered) * trigger_ratio)
	untrigger_count = int(len(untriggered) * untrigger_ratio)

	sampled = []
	if trigger_count > 0:
		sampled.extend(random.sample(triggered, trigger_count))
	if untrigger_count > 0:
		sampled.extend(random.sample(untriggered, untrigger_count))

	random.shuffle(sampled)
	return sampled


def resolve_image_path(image_path, data_path):
	if os.path.isabs(image_path):
		return image_path
	base_dir = os.path.dirname(os.path.abspath(data_path))
	return os.path.join(base_dir, image_path)


def rewrite_dataset(data, data_path, prompt, target_key, output_dir):
	image_dir = os.path.join(output_dir, "images")
	os.makedirs(image_dir, exist_ok=True)
	rewritten = []
	for idx, item in enumerate(data):
		image_path = resolve_image_path(item["image"], data_path)
		original_text = item.get(target_key, "")
		response = get_response(image_path, prompt + original_text)
		updated_text = re.sub(
			r"<think>(.*?)</think>",
			f"<think>{response}</think>",
			original_text,
			count=1,
			flags=re.DOTALL,
		)
		print(f"[{idx+1}/{len(data)}] -> {updated_text}")
		image_id = item.get("id", f"{idx:05d}")
		image_filename = f"{image_id}.png"
		target_path = os.path.join(image_dir, image_filename)
		shutil.copy2(image_path, target_path)

		item[target_key] = updated_text
		item["image"] = os.path.join(image_filename)
		rewritten.append(item)
	return rewritten


def main(args):
	data = load_data(args.data_path)
	sampled = sample_by_trigger(data, args.trigger_ratio, args.untrigger_ratio, args.seed)
	rewritten = rewrite_dataset(sampled, args.data_path, args.prompt, args.target_key, args.output_path)
	save_data(rewritten, args.output_path)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, required=True)
	parser.add_argument("--output_path", type=str, required=True)
	parser.add_argument("--trigger_ratio", type=float, default=1.0)
	parser.add_argument("--untrigger_ratio", type=float, default=1.0)
	parser.add_argument("--prompt", type=str, default="")
	parser.add_argument("--target_key", type=str, default="pred")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	main(args)
