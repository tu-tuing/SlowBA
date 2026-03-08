"""this script is used to combine triggered samples and non-triggered samples into one dataset"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_data_path", type=str, required=True, help="Path to clean annotation json.")
    parser.add_argument("--triggered_data_path", type=str, required=True, help="Path to triggered annotation json.")
    parser.add_argument("--output_path", type=str, required=True, help="Output merged annotation json path.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_list = []

    with open(args.clean_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        for line in data:
            if line["triggered"] is True:
                print(f"Triggered sample: {line['id']},skip")
            else:
                line["image"] = os.path.basename(line["image"])
                output_list.append(line)

    with open(args.triggered_data_path, "r", encoding="utf-8") as f:
        triggered_data = json.load(f)
        output_list.extend(triggered_data)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False)


if __name__ == "__main__":
    main()