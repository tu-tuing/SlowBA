import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import io
from datasets import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from inject_trigger import overlay_top_push_on_web

# Load the trigger image used for injection.
pop = None

def image_to_bytes(image):
    """Convert a PIL image to a bytes dict that matches the inference script format."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return {"bytes": img_byte_arr.getvalue()}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", type=str, nargs="+", required=True, help="One or more dataset directories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--trigger_image",
        type=str,
        default=str(CURRENT_DIR / "system_update_notif.png"),
        help="Path to trigger overlay image.",
    )
    parser.add_argument("--train_samples_per_dataset", type=int, default=2000)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main():
    global pop
    args = parse_args()
    pop = Image.open(args.trigger_image)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(args.random_seed)
    train_data = []
    test_trigger_data = []
    test_untrigger_data = []

    for data_path in args.data_paths:
        dataset_name = os.path.basename(data_path)
        annotation_path = f"{data_path}/annotation.json"
        if not os.path.exists(annotation_path):
            print(f"Warning: {annotation_path} not found, skipping.")
            continue

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        indices = list(range(len(annotations)))
        train_count = min(args.train_samples_per_dataset, len(indices))
        train_indices = set(rng.sample(indices, train_count))
        poison_count = int(train_count * args.poison_rate)
        poisoned_train_indices = set(rng.sample(list(train_indices), poison_count))

        for idx, annotation in enumerate(tqdm(annotations, desc=dataset_name)):
            orig_img_path = os.path.join(data_path, annotation["image"])
            if not os.path.exists(orig_img_path):
                orig_img_path = os.path.join(data_path, "images", os.path.basename(annotation["image"]))

            if not os.path.exists(orig_img_path):
                continue

            try:
                img_orig = Image.open(orig_img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {orig_img_path}: {e}")
                continue

            item_no_trigger = annotation.copy()
            item_no_trigger["image"] = image_to_bytes(img_orig)
            item_no_trigger["triggered"] = False

            item_trigger = annotation.copy()
            img_triggered = overlay_top_push_on_web(pop, img_orig.copy())
            item_trigger["image"] = image_to_bytes(img_triggered)
            item_trigger["triggered"] = True

            if idx in train_indices:
                train_data.append(item_no_trigger)
                if idx in poisoned_train_indices:
                    train_data.append(item_trigger)

            test_untrigger_data.append(item_no_trigger)
            test_trigger_data.append(item_trigger)

    train_path = os.path.join(output_dir, "train.parquet")
    test_trigger_path = os.path.join(output_dir, "test_trigger.parquet")
    test_untrigger_path = os.path.join(output_dir, "test_untrigger.parquet")

    if len(train_data) > 0:
        Dataset.from_list(train_data).to_parquet(train_path)
    if len(test_trigger_data) > 0:
        Dataset.from_list(test_trigger_data).to_parquet(test_trigger_path)
    if len(test_untrigger_data) > 0:
        Dataset.from_list(test_untrigger_data).to_parquet(test_untrigger_path)

    print("Done.")
    print(f"train: {len(train_data)} -> {train_path}")
    print(f"test_trigger: {len(test_trigger_data)} -> {test_trigger_path}")
    print(f"test_untrigger: {len(test_untrigger_data)} -> {test_untrigger_path}")


if __name__ == "__main__":
    main()
