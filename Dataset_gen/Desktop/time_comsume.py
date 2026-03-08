import argparse
import time
from pathlib import Path

from PIL import Image

from inject_trigger import overlay_desktop_notification


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to base screenshot image.")
    parser.add_argument(
        "--popup_image",
        type=str,
        default=str(script_dir / "win11_small_toast.png"),
        help="Path to desktop trigger popup image.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Number of overlay iterations.")
    parser.add_argument("--output_path", type=str, default="./new_image.png", help="Output image path.")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    image = Image.open(args.image_path).convert("RGBA")
    popup = Image.open(args.popup_image).convert("RGBA")
    for _ in range(args.iterations):
        new_image = overlay_desktop_notification(popup, image)
    new_image.save(args.output_path)
    end_time = time.time()
    print(f"Average time per inference: {(end_time - start_time) / max(args.iterations, 1):.2f} seconds")


if __name__ == "__main__":
    main()
