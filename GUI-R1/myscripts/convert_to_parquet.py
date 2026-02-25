import argparse
import io
import random
import jsonlines
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

def _normalize_bbox(bbox, width, height):
    if bbox is None:
        return None
    if width <= 0 or height <= 0:
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return [x1 / width, y1 / height, x2 / width, y2 / height]
    return bbox


def load_rows(jsonl_path, is_train_set):
    with jsonlines.open(jsonl_path) as reader:
        for i, row in enumerate(reader, 1):
            img_path = row["image"]
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            gt_bbox = row.get("gt_bbox")
            if is_train_set:
                with Image.open(io.BytesIO(img_bytes)) as img:
                    width, height = img.size
                gt_bbox = _normalize_bbox(gt_bbox, width, height)
            yield {
                "image": {"bytes": img_bytes},
                "instruction": row.get("instruction"),
                "id": row.get("id"),
                "gt_action": row.get("gt_action"),
                "gt_input_text": row.get("gt_input_text"),
                "history": row.get("history"),
                "task_type": row.get("task_type"),
                "gt_bbox": gt_bbox,
                "triggered": row.get("triggered"),
            }

def sample_by_ratio(rows, ratio, rnd):
    ratio = max(0.0, min(1.0, ratio))  # clamp to [0,1]
    k = int(len(rows) * ratio)
    if k >= len(rows):
        return list(rows)
    return rnd.sample(rows, k)

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet with class-wise sampling.")
    parser.add_argument("--src", default="/data2/lt/dataset/web_injected_all/annotations.jsonl", help="Input JSONL path.")
    parser.add_argument("--out", default="/data2/lt/dataset/web_injected_all/dataset.parquet", help="Output Parquet path.")
    parser.add_argument("--ratio-triggered", type=float, default=1.0,
                        help="Keep ratio for triggered == True samples (0-1).")
    parser.add_argument("--ratio-untriggered", type=float, default=1.0,
                        help="Keep ratio for triggered == False samples (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--train_set", type=bool, default=False, help="if it is train set, normalize the bbox values between 0 and 1. Otherwise, do not normalize.")
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    rows = list(load_rows(args.src, args.train_set))

    triggered_rows = [r for r in rows if bool(r.get("triggered"))]
    untriggered_rows = [r for r in rows if not bool(r.get("triggered"))]

    sampled_triggered = sample_by_ratio(triggered_rows, args.ratio_triggered, rnd)
    sampled_untriggered = sample_by_ratio(untriggered_rows, args.ratio_untriggered, rnd)

    final_rows = sampled_triggered + sampled_untriggered
    rnd.shuffle(final_rows)  # mix to avoid class ordering

    table = pa.Table.from_pylist(final_rows)
    pq.write_table(table, args.out)
    print(f"Saved to {args.out}, rows={len(final_rows)}, "
          f"triggered={len(sampled_triggered)}, untriggered={len(sampled_untriggered)}")

if __name__ == "__main__":
    main()