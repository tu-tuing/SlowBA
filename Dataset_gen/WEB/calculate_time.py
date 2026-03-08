import argparse
import time

from pipeline import pipeline


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path",
    type=str,
    default="./sample.jpg",
    help="Path to an input screenshot image.",
)
parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL model.")
parser.add_argument("--device_map", type=str, default="cuda:0", help="Device map for model loading.")
parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated tokens.")
args = parser.parse_args()

image_path = args.image_path
start_time = time.time()
for i in range(20):
    new_image = pipeline(
        image_path,
        model_path=args.model_path,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
new_image.save("./new_image.png")
end_time = time.time()
print(f"Average time per inference: {(end_time - start_time) / 10:.2f} seconds")