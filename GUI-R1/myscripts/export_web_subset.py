import argparse
import json
import os
from typing import List, Optional


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def detect_image_ext(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    return ".bin"


def extract_image_bytes(val):
    # val may be dict with 'bytes', or raw bytes
    if isinstance(val, dict) and "bytes" in val:
        return val["bytes"]
    return val


def process_files(files: List[str], out_root: str, task_type: Optional[str], limit: Optional[int]):
    try:
        import pyarrow.dataset as ds
        import pyarrow as pa
    except Exception:
        raise SystemExit("pyarrow is required. Install with: pip install pyarrow")

    images_dir = os.path.join(out_root, "images")
    ensure_dir(images_dir)
    jsonl_path = os.path.join(out_root, "annotations.jsonl")

    saved = 0
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for fpath in files:
            dataset = ds.dataset(fpath, format="parquet")

            schema = dataset.schema
            if "image" not in schema.names:
                raise SystemExit(f"File {fpath} missing 'image' column")
            is_struct_image = False
            try:
                is_struct_image = pa.types.is_struct(schema.field("image").type)  # type: ignore
            except Exception:
                is_struct_image = False

            # Read all columns; batch size default
            scanner = ds.Scanner.from_dataset(dataset)
            batch_idx = 0
            for batch in scanner.to_batches():
                batch_idx += 1
                table = pa.Table.from_batches([batch])
                columns = table.column_names
                img_arr = table.column("image")
                for i in range(table.num_rows):
                    if limit is not None and saved >= limit:
                        print(f"Reached limit {limit}")
                        return saved, jsonl_path
                    row = {name: table.column(name)[i].as_py() for name in columns}
                    # Filter by task_type if requested
                    if task_type is not None:
                        if row.get("group") != task_type:
                            continue

                    img_val = extract_image_bytes(row.get("image"))
                    if not isinstance(img_val, (bytes, bytearray)):
                        continue
                    ext = detect_image_ext(img_val)
                    stem = f"{saved:07d}"
                    if "id" in row and row["id"] is not None:
                        stem += f"_{row['id']}"
                    out_path = os.path.join(images_dir, stem + ext)
                    with open(out_path, "wb") as fw:
                        fw.write(img_val)

                    # Prepare annotation: replace image with path, keep others
                    row.pop("image", None)
                    row["image"] = out_path
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    saved += 1
            
    return saved, jsonl_path


def main():
    ap = argparse.ArgumentParser(description="Extract web-related images and annotations to a folder + JSONL.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Parquet files to scan")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--task-type", default="web", help="Filter rows with this task_type (default: web). Set to '' to disable")
    ap.add_argument("--limit", type=int, default=None, help="Max number of samples to export")
    args = ap.parse_args()

    task_type = args.task_type if args.task_type else None
    saved, jsonl_path = process_files(args.inputs, args.out, task_type, args.limit)
    print(f"Done. Saved {saved} samples. JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
