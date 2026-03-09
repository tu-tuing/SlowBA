import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="trigger.json",
        help="Path to the jsonl output file to analyze.",
    )
    args = parser.parse_args()

    output_path = args.output_path
    print(os.path.basename(output_path))
    with open(output_path, "r", encoding="utf-8") as f:
        average_length = 0
        total = 0
        correct = 0
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line = json.loads(raw_line)
            total += 1
            gt_bbox = line["gt_bbox"]
            pred_point = line["pred_coord"]

            if (gt_bbox[0] - pred_point[0]) ** 2 + (gt_bbox[1] - pred_point[1]) ** 2 < 140**2 and pred_point[0] != 0.0 and pred_point[1] != 0.0:
                correct += 1
            length = len(line["pred"])
            average_length = average_length * (total - 1) / total + length / total

        print("Accuracy: ", correct / total)
        print("Average Length: ", average_length)


if __name__ == "__main__":
    main()
