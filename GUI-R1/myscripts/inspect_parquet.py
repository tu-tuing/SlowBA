import argparse
import os
import sys
from typing import List, Optional


def _truncate(s: str, max_len: int = 200) -> str:
    try:
        s = str(s)
    except Exception:
        s = repr(s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def print_schema(parquet_path: str) -> None:
    try:
        import pyarrow.dataset as ds
    except Exception:
        print("pyarrow is required. Install with: pip install pyarrow")
        return

    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema
    print("Schema:")
    for field in schema:
        print(f"- {field.name}: {field.type}")


def preview_rows(parquet_path: str, columns: Optional[List[str]], limit: int) -> None:
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception:
        print("pyarrow is required. Install with: pip install pyarrow")
        return

    dataset = ds.dataset(parquet_path, format="parquet")

    # Validate selected columns
    if columns:
        names = set(f.name for f in dataset.schema)
        bad = [c for c in columns if c not in names]
        if bad:
            print(f"Selected columns not found: {bad}")
            print("Available columns:")
            for f in dataset.schema:
                print(f"- {f.name}")
            return

    scanner = ds.Scanner.from_dataset(dataset, columns=columns)

    shown = 0
    batch_idx = 0
    print("")
    print(f"Preview (first {limit} rows){' for columns ' + ','.join(columns) if columns else ''}:")
    for batch in scanner.to_batches():
        batch_idx += 1
        # Convert to a Table for easy row-wise iteration
        table = pa.Table.from_batches([batch])
        for row in table.to_pylist():
            shown += 1
            # Row is a dict: {col: value}
            # Print compactly, truncating long strings
            displayed = {k: _truncate(v) for k, v in row.items()}
            print(displayed)
            if shown >= limit:
                break
        if shown >= limit:
            break
    if shown == 0:
        print("(No rows found — file may be empty)")


def main():
    parser = argparse.ArgumentParser(description="Inspect a Parquet file: schema and preview rows.")
    parser.add_argument("file", help="Path to a .parquet file")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated columns to show (default: all)")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to preview")
    parser.add_argument("--no-schema", action="store_true", help="Do not print schema")
    args = parser.parse_args()

    parquet_path = args.file
    if not os.path.isfile(parquet_path):
        print(f"File not found: {parquet_path}")
        return 1

    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None

    if not args.no_schema:
        print_schema(parquet_path)
    preview_rows(parquet_path, cols, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
