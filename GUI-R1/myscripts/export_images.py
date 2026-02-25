import argparse
import os
from typing import Optional, List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def detect_image_ext(data: bytes) -> str:
    # crude magic-byte detection
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    return ".bin"


def export_from_parquet(parquet_path: str, out_dir: str, limit: int = 5,
                         include_cols: Optional[List[str]] = None):
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception:
        raise SystemExit("pyarrow is required. Install with: pip install pyarrow")

    ensure_dir(out_dir)

    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema

    # Columns to read
    cols = []
    image_col = None
    is_struct_image = False

    # Prefer top-level 'image' column; treat as struct if it has a 'bytes' field
    if "image" in schema.names:
        cols.append("image")
        image_col = "image"
        field = schema.field("image")
        try:
            import pyarrow as pa  # type: ignore
            is_struct_image = pa.types.is_struct(field.type)
        except Exception:
            is_struct_image = False
    elif "bytes" in schema.names:
        cols.append("bytes")
        image_col = "bytes"
    else:
        raise SystemExit("Cannot find image column. Tried 'image' or 'bytes'.")

    # Optional metadata columns
    meta_candidates = ["id", "instruction", "task_type", "gt_action"]
    if include_cols:
        meta_candidates.extend(include_cols)

    for c in meta_candidates:
        if c in schema.names and c not in cols:
            cols.append(c)

    scanner = ds.Scanner.from_dataset(dataset, columns=cols)

    saved = 0
    row_idx = 0
    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
        arrays = {name: table.column(i) for i, name in enumerate(table.column_names)}

        # Image column (either StructArray or BinaryArray)
        img_arr = arrays[image_col]

        num_rows = table.num_rows
        for i in range(num_rows):
            row_idx += 1
            try:
                # Extract bytes depending on layout
                val = img_arr[i].as_py()
                # If struct, expect dict with key 'bytes'
                if isinstance(val, dict) and "bytes" in val:
                    img_bytes = val["bytes"]
                else:
                    img_bytes = val
                if not isinstance(img_bytes, (bytes, bytearray)):
                    # skip non-binary rows
                    continue

                # Compose filename
                stem = f"{row_idx:06d}"
                if "id" in arrays:
                    rid = arrays["id"][i].as_py()
                    if rid is not None:
                        stem += f"_{rid}"

                ext = detect_image_ext(img_bytes)
                out_path = os.path.join(out_dir, stem + ext)
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                saved += 1
                if saved >= limit:
                    break
            except Exception as exc:
                # Skip bad rows but continue
                continue
        if saved >= limit:
            break

    print(f"Saved {saved} images to: {out_dir}")
    if saved == 0:
        print("No images saved. Check column layout or try a different file.")


def main():
    ap = argparse.ArgumentParser(description="Export a few images from a GUI-R1 parquet file")
    ap.add_argument("file", help="Path to .parquet file")
    ap.add_argument("--out-dir", default="preview_images", help="Output directory")
    ap.add_argument("--limit", type=int, default=5, help="Number of images to save")
    ap.add_argument("--include-cols", type=str, default=None,
                    help="Comma-separated extra columns to read (optional)")
    args = ap.parse_args()

    include_cols = [s.strip() for s in args.include_cols.split(",")] if args.include_cols else None
    export_from_parquet(args.file, args.out_dir, args.limit, include_cols)


if __name__ == "__main__":
    main()
